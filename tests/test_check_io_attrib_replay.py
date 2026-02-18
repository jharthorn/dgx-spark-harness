import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


def _load_checker_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "check_io_attrib_replay.py"
    spec = importlib.util.spec_from_file_location("check_io_attrib_replay", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load checker module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class IOAttribReplayCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = _load_checker_module()

    def _base_summary(self, *, tier_mode: str, kv_mode: str, disk_cache_gb: float) -> dict[str, Any]:
        return {
            "tier_mode": tier_mode,
            "kv_mode": {
                "mode": kv_mode,
                "disk_cache_gb": disk_cache_gb,
            },
            "io_attribution": {"enabled": True},
        }

    def _base_report(self) -> dict[str, Any]:
        return {
            "phase_windows": {
                "populate": {
                    "start": "2026-02-12T00:00:00+00:00",
                    "end": "2026-02-12T00:05:00+00:00",
                },
                "replay": {
                    "start": "2026-02-12T00:05:00+00:00",
                    "end": "2026-02-12T00:06:00+00:00",
                },
            },
            "block_io_by_phase": {
                "replay": {
                    "read_bytes": 5242880,
                    "write_bytes": 0,
                }
            },
            "process_io_by_phase": {
                "replay": {
                    "read_bytes": 4194304,
                    "cgroup_read_bytes": 4194304,
                    "write_bytes": 0,
                    "per_pid": [{"pid": 1234, "read_bytes_delta": 4194304, "write_bytes_delta": 0}],
                }
            },
            "kvbm_disk_path": "/mnt/nvme/kvbm",
            "kvbm_file_io_by_phase": {
                "available": True,
                "phases": {
                    "replay": [
                        {
                            "pid": 1234,
                            "path": "/mnt/nvme/kvbm/block_001",
                            "observed_samples": 3,
                            "read_bytes": None,
                            "write_bytes": None,
                        }
                    ]
                },
            },
        }

    def test_b2_pass_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_json(run_dir / "summary.json", self._base_summary(tier_mode="B2", kv_mode="cpu_disk", disk_cache_gb=32))
            _write_json(run_dir / "io" / "io_attribution_report.json", self._base_report())

            verdict, rc = self.checker.evaluate(run_dir=run_dir, expect_report=True)

            self.assertEqual(rc, 0)
            self.assertTrue(verdict.get("pass"))
            self.assertEqual(verdict.get("mode"), "B2")
            self.assertGreater(int(verdict.get("replay_read_bytes_block") or 0), 0)
            self.assertGreater(int(verdict.get("replay_read_bytes_process_total") or 0), 0)
            self.assertEqual(verdict.get("process_evidence_method"), "pid")
            self.assertTrue(verdict.get("per_pid_readers_available"))
            self.assertTrue(verdict.get("per_pid_readers_nonzero"))
            self.assertTrue((run_dir / "io" / "io_attrib_verdict.json").exists())

    def test_b2_fail_fixture_zero_replay_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_json(run_dir / "summary.json", self._base_summary(tier_mode="B2", kv_mode="cpu_disk", disk_cache_gb=32))
            report = self._base_report()
            report["block_io_by_phase"]["replay"]["read_bytes"] = 0
            report["process_io_by_phase"]["replay"]["read_bytes"] = 0
            report["process_io_by_phase"]["replay"]["cgroup_read_bytes"] = 0
            report["process_io_by_phase"]["replay"]["per_pid"] = []
            report["kvbm_file_io_by_phase"]["phases"]["replay"] = []
            _write_json(run_dir / "io" / "io_attribution_report.json", report)

            verdict, rc = self.checker.evaluate(run_dir=run_dir, expect_report=True)

            self.assertEqual(rc, 2)
            self.assertFalse(verdict.get("pass"))
            checks = verdict.get("checks") or []
            failed = {item.get("name") for item in checks if item.get("status") == "FAIL"}
            self.assertIn("replay_block_reads_positive", failed)
            self.assertIn("replay_process_reads_positive", failed)
            self.assertIn("replay_process_pid_readers_present", failed)

    def test_b2_pass_via_cgroup_when_pid_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_json(run_dir / "summary.json", self._base_summary(tier_mode="B2", kv_mode="cpu_disk", disk_cache_gb=32))
            report = self._base_report()
            report["process_io_by_phase"]["replay"]["read_bytes"] = 0
            report["process_io_by_phase"]["replay"]["cgroup_read_bytes"] = 6291456
            report["process_io_by_phase"]["replay"]["per_pid"] = [{"pid": 1234, "read_bytes_delta": 0, "write_bytes_delta": 0}]
            _write_json(run_dir / "io" / "io_attribution_report.json", report)

            verdict, rc = self.checker.evaluate(run_dir=run_dir, expect_report=True)

            self.assertEqual(rc, 0)
            self.assertTrue(verdict.get("pass"))
            self.assertEqual(verdict.get("process_evidence_method"), "cgroup")
            self.assertTrue(verdict.get("per_pid_readers_available"))
            self.assertFalse(verdict.get("per_pid_readers_nonzero"))
            checks = verdict.get("checks") or []
            status_by_name = {item.get("name"): item.get("status") for item in checks}
            self.assertEqual(status_by_name.get("replay_process_reads_positive"), "PASS")
            self.assertEqual(status_by_name.get("replay_process_pid_readers_present"), "WARN")

    def test_b2_fail_when_no_process_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_json(run_dir / "summary.json", self._base_summary(tier_mode="B2", kv_mode="cpu_disk", disk_cache_gb=32))
            report = self._base_report()
            report["block_io_by_phase"]["replay"]["read_bytes"] = 6291456
            report["process_io_by_phase"]["replay"]["read_bytes"] = 0
            report["process_io_by_phase"]["replay"]["cgroup_read_bytes"] = 0
            report["process_io_by_phase"]["replay"]["per_pid"] = [{"pid": 1234, "read_bytes_delta": 0, "write_bytes_delta": 0}]
            _write_json(run_dir / "io" / "io_attribution_report.json", report)

            verdict, rc = self.checker.evaluate(run_dir=run_dir, expect_report=True)

            self.assertEqual(rc, 2)
            self.assertFalse(verdict.get("pass"))
            self.assertEqual(verdict.get("process_evidence_method"), "none")
            checks = verdict.get("checks") or []
            failed = {item.get("name") for item in checks if item.get("status") == "FAIL"}
            self.assertIn("replay_process_reads_positive", failed)
            self.assertIn("replay_process_pid_readers_present", failed)

    def test_missing_report_fails_when_expected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_json(run_dir / "summary.json", self._base_summary(tier_mode="B2", kv_mode="cpu_disk", disk_cache_gb=32))

            verdict, rc = self.checker.evaluate(run_dir=run_dir, expect_report=True)

            self.assertEqual(rc, 2)
            self.assertFalse(verdict.get("pass"))
            checks = verdict.get("checks") or []
            names = {item.get("name"): item.get("status") for item in checks}
            self.assertEqual(names.get("io_report_present"), "FAIL")

    def test_b1_non_strict_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_json(run_dir / "summary.json", self._base_summary(tier_mode="B1", kv_mode="cpu_only", disk_cache_gb=0))
            report = self._base_report()
            report["block_io_by_phase"]["replay"]["read_bytes"] = 0
            report["process_io_by_phase"]["replay"]["read_bytes"] = 0
            report["process_io_by_phase"]["replay"]["per_pid"] = []
            _write_json(run_dir / "io" / "io_attribution_report.json", report)

            verdict, rc = self.checker.evaluate(run_dir=run_dir, expect_report=True)

            self.assertEqual(rc, 0)
            self.assertTrue(verdict.get("pass"))
            self.assertFalse(verdict.get("strict_replay_gate_required"))


if __name__ == "__main__":
    unittest.main()

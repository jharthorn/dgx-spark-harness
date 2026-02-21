import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_analyzer_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "analyze_phase70_pairs.py"
    spec = importlib.util.spec_from_file_location("analyze_phase70_pairs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load analyzer module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class AnalyzePhase70PairsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_analyzer_module()

    def _make_summary(
        self,
        *,
        created_utc: str,
        replay_ttft_p95: float,
        replay_ttft_p99: float,
        replay_ttfc_p95: float,
        replay_ttfc_p99: float,
        replay_read_bytes: int,
        matched_tokens: float,
        onboard_blocks: float,
        stream: bool,
    ) -> dict:
        return {
            "created_utc": created_utc,
            "run_valid": True,
            "stream": stream,
            "stream_record_ttfb": stream,
            "overall_summary": {
                "error_rate": 0.0,
            },
            "storage": {
                "primary_nvme_model": "NVME-MODEL",
                "primary_nvme_fw": "FW1",
                "pcie_link": "Gen4 x4",
            },
            "phase_summaries": [
                {
                    "phase": "replay",
                    "ttft_ms": {"p95": replay_ttft_p95, "p99": replay_ttft_p99},
                    "ttfc_ms": {"p95": replay_ttfc_p95, "p99": replay_ttfc_p99},
                    "io_delta": {"read_bytes_delta": replay_read_bytes},
                    "kvbm_metrics_delta": {
                        "kvbm_matched_tokens_delta": matched_tokens,
                        "onboard_blocks_total_delta": onboard_blocks,
                    },
                }
            ],
        }

    def _make_verdict(self, *, passed: bool, method: str, pid_warn: bool) -> dict:
        checks = [
            {
                "name": "replay_process_pid_readers_present",
                "status": "WARN" if pid_warn else "PASS",
                "detail": "fixture",
            }
        ]
        return {
            "pass": passed,
            "process_evidence_method": method,
            "checks": checks,
        }

    def test_analyzer_outputs_per_run_and_pair_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            summary_json_path = root / "summary.json"
            summary_csv_path = root / "summary.csv"
            delta_csv_path = root / "deltas.csv"
            order_check_path = root / "order_check.json"

            entries = []

            def add_run(
                *,
                pair_id: int,
                pair_order: str,
                pair_leg: int,
                mode: str,
                run_suffix: str,
                ttft_p95: float,
                ttfc_p95: float,
                read_gib: float,
                method: str,
                pid_warn: bool,
            ) -> None:
                run_dir = root / f"bundle_{run_suffix}" / f"run_{run_suffix}"
                replay_bytes = int(read_gib * (1024**3))
                _write_json(
                    run_dir / "summary.json",
                    self._make_summary(
                        created_utc="2026-02-13T00:00:00Z",
                        replay_ttft_p95=ttft_p95,
                        replay_ttft_p99=ttft_p95 + 5.0,
                        replay_ttfc_p95=ttfc_p95,
                        replay_ttfc_p99=ttfc_p95 + 5.0,
                        replay_read_bytes=replay_bytes,
                        matched_tokens=1000.0 if mode == "B2" else 10.0,
                        onboard_blocks=300.0 if mode == "B2" else 5.0,
                        stream=True,
                    ),
                )
                _write_json(
                    run_dir / "io" / "io_attrib_verdict.json",
                    self._make_verdict(passed=True, method=method, pid_warn=pid_warn),
                )
                entries.append(
                    {
                        "pair_id": pair_id,
                        "pair_order": pair_order,
                        "pair_leg": pair_leg,
                        "mode": mode,
                        "bundle_id": f"bundle_{run_suffix}",
                        "run_id": f"run_{run_suffix}",
                        "run_dir": str(run_dir),
                        "io_attrib_checked": True,
                        "io_attrib_checker_rc": 0,
                    }
                )

            add_run(
                pair_id=1,
                pair_order="B1_B2",
                pair_leg=1,
                mode="B1",
                run_suffix="p1_b1",
                ttft_p95=120.0,
                ttfc_p95=100.0,
                read_gib=0.1,
                method="none",
                pid_warn=False,
            )
            add_run(
                pair_id=1,
                pair_order="B1_B2",
                pair_leg=2,
                mode="B2",
                run_suffix="p1_b2",
                ttft_p95=70.0,
                ttfc_p95=55.0,
                read_gib=1.2,
                method="cgroup",
                pid_warn=True,
            )
            add_run(
                pair_id=2,
                pair_order="B2_B1",
                pair_leg=1,
                mode="B2",
                run_suffix="p2_b2",
                ttft_p95=80.0,
                ttfc_p95=60.0,
                read_gib=1.0,
                method="pid",
                pid_warn=False,
            )
            add_run(
                pair_id=2,
                pair_order="B2_B1",
                pair_leg=2,
                mode="B1",
                run_suffix="p2_b1",
                ttft_p95=110.0,
                ttfc_p95=90.0,
                read_gib=0.2,
                method="none",
                pid_warn=False,
            )

            _write_json(
                manifest_path,
                {
                    "meta": {
                        "pair_count": 2,
                        "mode_a": "B1",
                        "mode_b": "B2",
                        "replay_concurrency": 4,
                    },
                    "runs": entries,
                },
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "analyze_phase70_pairs.py",
                    "--manifest",
                    str(manifest_path),
                    "--summary-json",
                    str(summary_json_path),
                    "--summary-csv",
                    str(summary_csv_path),
                    "--pair-delta-csv",
                    str(delta_csv_path),
                    "--order-check-json",
                    str(order_check_path),
                    "--mode-a",
                    "B1",
                    "--mode-b",
                    "B2",
                ]
                rc = self.module.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            self.assertTrue(summary_csv_path.exists())
            self.assertTrue(delta_csv_path.exists())
            self.assertTrue(order_check_path.exists())
            self.assertTrue(summary_json_path.exists())

            with summary_csv_path.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 4)
            p1_b2 = next(r for r in rows if r.get("pair_id") == "1" and r.get("mode") == "B2")
            self.assertEqual(p1_b2.get("process_evidence_method"), "cgroup")
            self.assertEqual(p1_b2.get("pid_warn"), "True")
            self.assertEqual(p1_b2.get("replay_concurrency"), "4")
            self.assertEqual(p1_b2.get("metric_used_replay_p95"), "replay_ttfc_p95_ms")

            with delta_csv_path.open("r", encoding="utf-8", newline="") as fp:
                deltas = list(csv.DictReader(fp))
            self.assertEqual(len(deltas), 2)
            d1 = next(d for d in deltas if d.get("pair_id") == "1")
            d2 = next(d for d in deltas if d.get("pair_id") == "2")
            self.assertAlmostEqual(float(d1["delta_replay_ttft_p95_ms"]), -50.0, places=6)
            self.assertAlmostEqual(float(d2["delta_replay_ttft_p95_ms"]), -30.0, places=6)
            self.assertGreater(float(d1["delta_replay_read_gib"]), 0.0)
            self.assertGreater(float(d2["delta_replay_read_gib"]), 0.0)

            order_check = json.loads(order_check_path.read_text(encoding="utf-8"))
            self.assertIn("metrics", order_check)
            self.assertIn("delta_replay_ttfc_p95_ms", order_check["metrics"])
            self.assertIn("difference_of_means", order_check["metrics"]["delta_replay_ttft_p95_ms"])
            ttfc_stats = order_check["metrics"]["delta_replay_ttfc_p95_ms"]
            self.assertIn("all_pairs", ttfc_stats)
            self.assertIn("min", ttfc_stats["all_pairs"])
            self.assertIn("max", ttfc_stats["all_pairs"])
            self.assertLess(ttfc_stats["all_pairs"]["min"], ttfc_stats["all_pairs"]["max"])
            self.assertIn("approx_ci95_half_width", ttfc_stats["order_ab"])
            self.assertIn("delta_rollups", order_check)
            self.assertIn("delta_replay_ttft_p95_ms", order_check["delta_rollups"])
            self.assertEqual((order_check.get("meta") or {}).get("replay_concurrency"), 4)

            summary_obj = json.loads(summary_json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary_obj.get("pair_count"), 2)
            self.assertEqual(len(summary_obj.get("rows") or []), 4)
            self.assertIn("delta_rollups", summary_obj)
            self.assertIn("delta_replay_read_gib", summary_obj["delta_rollups"])
            self.assertEqual(summary_obj.get("replay_concurrency"), 4)
            self.assertEqual((summary_obj.get("meta") or {}).get("replay_concurrency"), 4)


if __name__ == "__main__":
    unittest.main()

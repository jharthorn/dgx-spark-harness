import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "phase70_check_mechanism_gate.py"
    spec = importlib.util.spec_from_file_location("phase70_check_mechanism_gate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class Phase70MechanismGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_gate_passes_with_ssd_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_ok"
            summary_json = run_dir / "summary.json"
            out_json = root / "gate.json"
            _write_json(
                summary_json,
                {
                    "phase_summaries": [
                        {
                            "phase": "replay",
                            "kvbm_metrics_delta": {
                                "kvbm_matched_tokens_delta": 12.0,
                                "kvbm_onboard_blocks_d2d_delta": 3.0,
                                "kvbm_offload_blocks_h2d_delta": 0.0,
                            },
                            "kvbm_metrics_end": {"kvbm_disk_cache_hit_rate": 0.12},
                        }
                    ]
                },
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "phase70_check_mechanism_gate.py",
                    "--run-dir",
                    str(run_dir),
                    "--json-out",
                    str(out_json),
                ]
                rc = self.module.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(payload.get("pass"))
            self.assertEqual(payload.get("reason_codes"), [])

    def test_gate_fails_without_ssd_mechanism_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_fail"
            summary_json = run_dir / "summary.json"
            out_json = root / "gate.json"
            _write_json(
                summary_json,
                {
                    "phase_summaries": [
                        {
                            "phase": "replay",
                            "kvbm_metrics_delta": {
                                "kvbm_matched_tokens_delta": 0.0,
                                "kvbm_onboard_blocks_d2d_delta": 0.0,
                                "kvbm_offload_blocks_h2d_delta": 0.0,
                            },
                            "kvbm_metrics_end": {"kvbm_disk_cache_hit_rate": 0.0},
                        }
                    ]
                },
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "phase70_check_mechanism_gate.py",
                    "--run-dir",
                    str(run_dir),
                    "--json-out",
                    str(out_json),
                ]
                rc = self.module.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 1)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertFalse(payload.get("pass"))
            self.assertIn("GATE_NO_SSD_MECHANISM_SIGNAL", payload.get("reason_codes") or [])


if __name__ == "__main__":
    unittest.main()

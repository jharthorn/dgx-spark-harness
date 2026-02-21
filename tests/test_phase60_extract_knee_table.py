import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "phase60_extract_knee_table.py"
    spec = importlib.util.spec_from_file_location("phase60_extract_knee_table", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


class Phase60ExtractKneeTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def _invoke(self, argv: list[str]) -> int:
        old_argv = sys.argv
        try:
            sys.argv = argv
            return self.module.main()
        finally:
            sys.argv = old_argv

    def test_extract_knee_table_with_fallback_and_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "phase60_rehydrate_concurrency_sweep_summary_minimal_20260218T120000Z.json"
            out_csv = root / "knee.csv"

            _write_json(
                summary_path,
                {
                    "meta": {"sweep_replay_concurrencies": [1, 2]},
                    "rows": [
                        {
                            "mode": "B1",
                            "concurrency": 1,
                            "replay_concurrency": 1,
                            "point_key": "B1_c1",
                            "status": "ok",
                            "error_rate": 0.0,
                            "replay_ttfc_ms": {"p95": 120.0, "p99": 150.0},
                        },
                        {
                            "mode": "B2",
                            "concurrency": 1,
                            "replay_concurrency": 1,
                            "point_key": "B2_c1",
                            "status": "ok",
                            "error_rate": 0.0,
                            "replay_ttfc_ms": {"p95": 100.0, "p99": 115.0},
                            "kvbm_disk_cache_hit_rate": 0.12,
                            "mechanism": {
                                "kvbm_offload_blocks_h2d_delta_replay_plus_replay2": 5.0,
                                "block_write_bytes_delta_replay_plus_replay2": 1024,
                            },
                        },
                        {
                            "mode": "B1",
                            "concurrency": 2,
                            "replay_concurrency": 2,
                            "point_key": "B1_c2",
                            "status": "ok",
                            "error_rate": 0.0,
                            "replay_ttft_ms": {"p95": 180.0, "p99": 205.0},
                        },
                        {
                            "mode": "B2",
                            "concurrency": 2,
                            "replay_concurrency": 2,
                            "point_key": "B2_c2",
                            "status": "invalid_full",
                            "error_rate": 0.0,
                            "replay_ttft_ms": {"p95": 130.0, "p99": 140.0},
                            "mechanism": {
                                "kvbm_onboard_blocks_d2d_delta_replay_plus_replay2": 2.0,
                                "block_read_bytes_delta_replay_plus_replay2": 4096,
                            },
                        },
                    ],
                },
            )

            rc = self._invoke(
                [
                    "phase60_extract_knee_table.py",
                    "--phase60-summary-json",
                    str(summary_path),
                    "--out-csv",
                    str(out_csv),
                ]
            )
            self.assertEqual(rc, 0)

            rows = _read_csv(out_csv)
            self.assertEqual(len(rows), 4)

            by_key = {(int(row["concurrency"]), row["mode"]): row for row in rows}

            b1_c2 = by_key[(2, "B1")]
            self.assertEqual(b1_c2["metric_source_p95"], "ttft_fallback")
            self.assertEqual(float(b1_c2["ttfc_p95_ms"]), 180.0)
            self.assertIn("ttfc_missing_fallback_ttft", b1_c2["notes"])

            b2_c1 = by_key[(1, "B2")]
            self.assertEqual(b2_c1["ssd_write_signal_present"], "True")
            self.assertEqual(b2_c1["ssd_rehydrate_signal_present"], "True")

            b2_c2 = by_key[(2, "B2")]
            self.assertEqual(b2_c2["status"], "invalid_full")
            self.assertIn("status:invalid_full", b2_c2["notes"])

    def test_missing_mode_row_is_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "phase60_rehydrate_concurrency_sweep_summary_minimal_20260218T120100Z.json"
            out_csv = root / "knee.csv"

            _write_json(
                summary_path,
                {
                    "meta": {"sweep_replay_concurrencies": [1]},
                    "rows": [
                        {
                            "mode": "B2",
                            "concurrency": 1,
                            "replay_concurrency": 1,
                            "point_key": "B2_c1",
                            "status": "ok",
                            "error_rate": 0.0,
                            "replay_ttfc_ms": {"p95": 90.0, "p99": 100.0},
                        }
                    ],
                },
            )

            rc = self._invoke(
                [
                    "phase60_extract_knee_table.py",
                    "--phase60-summary-json",
                    str(summary_path),
                    "--out-csv",
                    str(out_csv),
                ]
            )
            self.assertEqual(rc, 0)

            rows = _read_csv(out_csv)
            self.assertEqual(len(rows), 2)
            missing_b1 = [row for row in rows if row["mode"] == "B1"][0]
            self.assertEqual(missing_b1["status"], "missing")
            self.assertEqual(missing_b1["notes"], "missing_mode_row")

    def test_b1_disk_tier_verified_column_is_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "phase60_rehydrate_concurrency_sweep_summary_minimal_20260218T120200Z.json"
            out_csv = root / "knee.csv"

            _write_json(
                summary_path,
                {
                    "meta": {"sweep_replay_concurrencies": [1]},
                    "rows": [
                        {
                            "mode": "B1",
                            "concurrency": 1,
                            "replay_concurrency": 1,
                            "point_key": "B1_c1",
                            "status": "ok",
                            "error_rate": 0.0,
                            "replay_ttfc_ms": {"p95": 110.0, "p99": 120.0},
                            "b1_disk_tier_verified": False,
                        },
                        {
                            "mode": "B2",
                            "concurrency": 1,
                            "replay_concurrency": 1,
                            "point_key": "B2_c1",
                            "status": "ok",
                            "error_rate": 0.0,
                            "replay_ttfc_ms": {"p95": 100.0, "p99": 110.0},
                        },
                    ],
                },
            )

            rc = self._invoke(
                [
                    "phase60_extract_knee_table.py",
                    "--phase60-summary-json",
                    str(summary_path),
                    "--out-csv",
                    str(out_csv),
                ]
            )
            self.assertEqual(rc, 0)

            rows = _read_csv(out_csv)
            b1_row = [row for row in rows if row["mode"] == "B1"][0]
            self.assertEqual(b1_row["b1_disk_tier_verified"], "False")
            self.assertIn("b1_disk_tier_not_verified", b1_row["notes"])


if __name__ == "__main__":
    unittest.main()

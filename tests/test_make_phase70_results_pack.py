import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "make_phase70_results_pack.py"
    spec = importlib.util.spec_from_file_location("make_phase70_results_pack", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class MakePhase70ResultsPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_pack_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "bench" / "results"
            results.mkdir(parents=True, exist_ok=True)
            ts = "20260218T120000Z"

            manifest = {
                "meta": {
                    "pair_count": 2,
                    "mode_a": "B1",
                    "mode_b": "B2",
                    "replay_concurrency": 4,
                    "order_strategy": "alternating",
                    "pair_washout_s": 10,
                    "stream_metrics_enabled": True,
                    "stream_timeout_s": 120.0,
                    "stream_record_ttfb": True,
                    "io_attrib_enabled": True,
                },
                "runs": [],
            }
            _write_json(results / f"phase70_rehydrate_pair_repeats_manifest_{ts}.json", manifest)

            order_check = {
                "metrics": {
                    "delta_replay_ttfc_p95_ms": {
                        "order_ab_label": "B1_B2",
                        "order_ba_label": "B2_B1",
                        "order_ab": {
                            "n": 1,
                            "mean": -4.0,
                            "stddev": None,
                            "min": -4.0,
                            "max": -4.0,
                            "approx_ci95_low": None,
                            "approx_ci95_high": None,
                        },
                        "order_ba": {
                            "n": 1,
                            "mean": -2.0,
                            "stddev": None,
                            "min": -2.0,
                            "max": -2.0,
                            "approx_ci95_low": None,
                            "approx_ci95_high": None,
                        },
                        "difference_of_means": -2.0,
                        "relative_effect_size": None,
                        "order_effect_flag": False,
                        "order_effect_note": "fixture",
                    }
                }
            }
            _write_json(results / f"phase70_rehydrate_pair_repeats_order_check_{ts}.json", order_check)
            _write_json(
                results / f"phase70_rehydrate_pair_repeats_verdict_{ts}.json",
                {
                    "run_valid": True,
                    "decision_grade": True,
                    "reason_codes": [],
                    "meta": {"replay_concurrency": 4},
                    "checks": {
                        "ssd_write_signal_present": True,
                        "ssd_rehydrate_signal_present": True,
                        "ssd_reuse_signal_present": True,
                        "decision_grade_require_rehydrate": True,
                    },
                },
            )

            summary_rows = []
            deltas_rows = []
            rows_def = [
                ("B1", 1, 1, 120.0, 130.0, 140.0, 8.8, True, "cgroup"),
                ("B2", 1, 2, 116.0, 126.0, 136.0, 9.1, True, "cgroup"),
                ("B2", 2, 1, 118.0, 128.0, 138.0, 9.0, True, "pid"),
                ("B1", 2, 2, 121.0, 131.0, 141.0, 8.9, True, "cgroup"),
            ]
            for mode, pair_id, pair_leg, p50, p95, p99, read_gib, passed, method in rows_def:
                run_dir = results / f"phase70_rehydrate_pair_{mode}_p{pair_id:02d}_l{pair_leg}_{ts}" / f"run_{mode}_{pair_id}_{pair_leg}"
                run_summary = {
                    "phase_summaries": [
                        {
                            "phase": "replay",
                            "ttfc_ms": {"p50": p50, "p95": p95, "p99": p99},
                            "ttft_ms": {"p50": p50, "p95": p95, "p99": p99},
                        }
                    ]
                }
                _write_json(run_dir / "summary.json", run_summary)
                summary_rows.append(
                    {
                        "pair_id": pair_id,
                        "pair_order": "B1_B2" if pair_id == 1 else "B2_B1",
                        "pair_leg": pair_leg,
                        "mode": mode,
                        "replay_concurrency": 4,
                        "run_id": f"run_{mode}_{pair_id}_{pair_leg}",
                        "bundle_id": f"bundle_{mode}_{pair_id}_{pair_leg}",
                        "timestamp_utc": "2026-02-18T12:00:00Z",
                        "status": "ok",
                        "error_rate": 0.0,
                        "stream": True,
                        "stream_record_ttfb": True,
                        "replay_ttfc_p95_ms": p95,
                        "replay_ttfc_p99_ms": p99,
                        "replay_ttft_p95_ms": p95,
                        "replay_ttft_p99_ms": p99,
                        "replay_read_gib": read_gib,
                        "io_attrib_pass": passed,
                        "process_evidence_method": method,
                        "pid_warn": method == "cgroup" and mode == "B2",
                        "kvbm_matched_tokens_delta": 100.0,
                        "kvbm_onboard_blocks_delta": 5.0,
                        "primary_nvme_model": "NVME",
                        "primary_nvme_fw": "FW",
                        "pcie_link": "Gen4 x4",
                        "run_path": str(run_dir),
                        "io_attrib_checked": True,
                        "io_attrib_checker_rc": 0,
                    }
                )
            deltas_rows.extend(
                [
                    {
                        "pair_id": 1,
                        "pair_order": "B1_B2",
                        "mode_a": "B1",
                        "mode_b": "B2",
                        "mode_a_run_id": "run_B1_1_1",
                        "mode_b_run_id": "run_B2_1_2",
                        "delta_replay_ttfc_p95_ms": -4.0,
                        "delta_replay_ttfc_p99_ms": -4.0,
                        "delta_replay_ttft_p95_ms": -4.0,
                        "delta_replay_ttft_p99_ms": -4.0,
                        "delta_replay_read_gib": 0.3,
                        "delta_matched_tokens": 0.0,
                        "delta_onboard_blocks": 0.0,
                    },
                    {
                        "pair_id": 2,
                        "pair_order": "B2_B1",
                        "mode_a": "B1",
                        "mode_b": "B2",
                        "mode_a_run_id": "run_B1_2_2",
                        "mode_b_run_id": "run_B2_2_1",
                        "delta_replay_ttfc_p95_ms": -3.0,
                        "delta_replay_ttfc_p99_ms": -3.0,
                        "delta_replay_ttft_p95_ms": -3.0,
                        "delta_replay_ttft_p99_ms": -3.0,
                        "delta_replay_read_gib": 0.1,
                        "delta_matched_tokens": 0.0,
                        "delta_onboard_blocks": 0.0,
                    },
                ]
            )

            summary_csv = results / f"phase70_rehydrate_pair_repeats_summary_{ts}.csv"
            with summary_csv.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow(row)

            summary_json = {
                "mode_a": "B1",
                "mode_b": "B2",
                "pair_count": 2,
                "rows": summary_rows,
                "pair_deltas": deltas_rows,
            }
            _write_json(results / f"phase70_rehydrate_pair_repeats_summary_{ts}.json", summary_json)

            deltas_csv = results / f"phase70_rehydrate_pair_repeats_deltas_{ts}.csv"
            with deltas_csv.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(deltas_rows[0].keys()))
                writer.writeheader()
                for row in deltas_rows:
                    writer.writerow(row)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "make_phase70_results_pack.py",
                    "--results-root",
                    str(results),
                    "--ts",
                    ts,
                    "--skip-figures",
                ]
                rc = self.module.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            pack_dir = results / "publish" / f"phase70_pairs2_c4_{ts}"
            self.assertTrue((pack_dir / "summary.csv").exists())
            self.assertTrue((pack_dir / "summary.json").exists())
            self.assertTrue((pack_dir / "deltas.csv").exists())
            self.assertTrue((pack_dir / "order_check.json").exists())
            self.assertTrue((pack_dir / "verdict.json").exists())
            self.assertTrue((pack_dir / "methodology.md").exists())
            self.assertTrue((pack_dir / "pack_manifest.json").exists())
            self.assertTrue((pack_dir / "tables" / "table_main_latency.csv").exists())
            self.assertTrue((pack_dir / "tables" / "table_mechanism.csv").exists())
            self.assertTrue((pack_dir / "tables" / "table_order_effect.csv").exists())

            methodology = (pack_dir / "methodology.md").read_text(encoding="utf-8")
            self.assertIn("Headline Claim Structure", methodology)
            self.assertIn("CI95 is an approximate descriptive band", methodology)
            self.assertIn("Replay concurrency: `c=4`", methodology)
            self.assertIn("Mechanism Signal Summary", methodology)
            self.assertIn("SSD rehydrate signal observed: `True`", methodology)

            pack_manifest = json.loads((pack_dir / "pack_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(pack_manifest.get("replay_concurrency"), 4)
            self.assertTrue(pack_manifest.get("ssd_write_signal_present"))
            self.assertTrue(pack_manifest.get("ssd_rehydrate_signal_present"))
            self.assertTrue(pack_manifest.get("ssd_reuse_signal_present"))
            self.assertEqual(pack_manifest.get("source_artifacts", {}).get("verdict_json"), str(results / f"phase70_rehydrate_pair_repeats_verdict_{ts}.json"))
            self.assertEqual(pack_manifest.get("pack_artifacts", {}).get("verdict_json"), str(pack_dir / "verdict.json"))


if __name__ == "__main__":
    unittest.main()

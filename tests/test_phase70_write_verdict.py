import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "phase70_write_verdict.py"
    spec = importlib.util.spec_from_file_location("phase70_write_verdict", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class Phase70WriteVerdictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def _create_inputs(
        self,
        root: Path,
        *,
        pair_count: int = 1,
        matched_tokens_delta: float = 12.0,
        onboard_d2d_delta: float = 4.0,
        offload_h2d_delta: float = 1.0,
        disk_hit_rate: float = 0.22,
        replay_read_gib: float = 9.1,
    ) -> tuple[Path, Path, Path, Path, Path]:
        ts = "20260218T121500Z"
        manifest_path = root / f"phase70_rehydrate_pair_repeats_manifest_{ts}.json"
        summary_json_path = root / f"phase70_rehydrate_pair_repeats_summary_{ts}.json"
        summary_csv_path = root / f"phase70_rehydrate_pair_repeats_summary_{ts}.csv"
        delta_csv_path = root / f"phase70_rehydrate_pair_repeats_deltas_{ts}.csv"
        order_check_path = root / f"phase70_rehydrate_pair_repeats_order_check_{ts}.json"

        run_b1 = root / "bundle_b1" / "run_b1"
        run_b2 = root / "bundle_b2" / "run_b2"
        _write_json(run_b1 / "summary.json", {"phase_summaries": [{"phase": "replay", "kvbm_metrics_delta": {}}]})
        _write_json(
            run_b2 / "summary.json",
            {
                "phase_summaries": [
                    {
                        "phase": "replay",
                        "kvbm_metrics_delta": {
                            "kvbm_matched_tokens_delta": matched_tokens_delta,
                            "kvbm_onboard_blocks_d2d_delta": onboard_d2d_delta,
                            "kvbm_offload_blocks_h2d_delta": offload_h2d_delta,
                        },
                        "kvbm_metrics_end": {"kvbm_disk_cache_hit_rate": disk_hit_rate},
                    }
                ]
            },
        )

        summary_rows = [
            {
                "pair_id": 1,
                "pair_order": "B1_B2",
                "pair_leg": 1,
                "mode": "B1",
                "run_id": "run_b1",
                "bundle_id": "bundle_b1",
                "status": "ok",
                "kvbm_matched_tokens_delta": 0.0,
                "kvbm_onboard_blocks_delta": 0.0,
                "io_attrib_pass": True,
                "replay_read_gib": 0.0,
                "run_path": str(run_b1),
            },
            {
                "pair_id": 1,
                "pair_order": "B1_B2",
                "pair_leg": 2,
                "mode": "B2",
                "run_id": "run_b2",
                "bundle_id": "bundle_b2",
                "status": "ok",
                "kvbm_matched_tokens_delta": matched_tokens_delta,
                "kvbm_onboard_blocks_delta": onboard_d2d_delta,
                "io_attrib_pass": True,
                "replay_read_gib": replay_read_gib,
                "run_path": str(run_b2),
            },
        ]
        _write_json(
            manifest_path,
            {
                "meta": {
                    "pair_count": pair_count,
                    "mode_a": "B1",
                    "mode_b": "B2",
                    "replay_concurrency": 4,
                    "scenario": "rehydrate_replay",
                },
                "runs": [],
            },
        )
        _write_json(
            summary_json_path,
            {
                "mode_a": "B1",
                "mode_b": "B2",
                "pair_count": pair_count,
                "rows": summary_rows,
                "errors": [],
            },
        )
        with summary_csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        with delta_csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "pair_id",
                    "pair_order",
                    "mode_a",
                    "mode_b",
                    "mode_a_run_id",
                    "mode_b_run_id",
                    "delta_replay_ttfc_p95_ms",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "pair_id": 1,
                    "pair_order": "B1_B2",
                    "mode_a": "B1",
                    "mode_b": "B2",
                    "mode_a_run_id": "run_b1",
                    "mode_b_run_id": "run_b2",
                    "delta_replay_ttfc_p95_ms": -2.0,
                }
            )
        _write_json(order_check_path, {"order_effect_summary": {"order_dependent": False}, "metrics": {}})
        return manifest_path, summary_json_path, summary_csv_path, delta_csv_path, order_check_path

    def _invoke_verdict(self, argv: list[str]) -> int:
        old_argv = sys.argv
        try:
            sys.argv = argv
            return self.module.main()
        finally:
            sys.argv = old_argv

    def test_verdict_decision_grade_true_with_rehydrate_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, summary_json_path, summary_csv_path, delta_csv_path, order_check_path = self._create_inputs(root)
            verdict_path = root / "verdict.json"

            rc = self._invoke_verdict(
                [
                    "phase70_write_verdict.py",
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
                    "--out",
                    str(verdict_path),
                    "--io-attrib-enabled",
                    "1",
                    "--decision-grade-hint",
                    "1",
                    "--decision-grade-require-rehydrate",
                    "1",
                ]
            )

            self.assertEqual(rc, 0)
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
            self.assertTrue(verdict.get("run_valid"))
            self.assertTrue(verdict.get("decision_grade"))
            self.assertIn("ORDER_EFFECT_INSUFFICIENT_PAIRS", verdict.get("reason_codes") or [])
            self.assertTrue((verdict.get("checks") or {}).get("ssd_write_signal_present"))
            self.assertTrue((verdict.get("checks") or {}).get("ssd_rehydrate_signal_present"))
            self.assertTrue((verdict.get("checks") or {}).get("ssd_reuse_signal_present"))
            self.assertEqual((verdict.get("meta") or {}).get("replay_concurrency"), 4)

    def test_verdict_write_only_fails_when_rehydrate_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, summary_json_path, summary_csv_path, delta_csv_path, order_check_path = self._create_inputs(
                root,
                matched_tokens_delta=0.0,
                onboard_d2d_delta=0.0,
                offload_h2d_delta=2.0,
                disk_hit_rate=0.0,
                replay_read_gib=0.0,
            )
            verdict_path = root / "verdict.json"

            rc = self._invoke_verdict(
                [
                    "phase70_write_verdict.py",
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
                    "--out",
                    str(verdict_path),
                    "--decision-grade-require-rehydrate",
                    "1",
                ]
            )
            self.assertEqual(rc, 0)
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
            checks = verdict.get("checks") or {}
            reason_codes = verdict.get("reason_codes") or []
            self.assertFalse(verdict.get("decision_grade"))
            self.assertTrue(checks.get("ssd_write_signal_present"))
            self.assertFalse(checks.get("ssd_rehydrate_signal_present"))
            self.assertFalse(checks.get("ssd_reuse_signal_present"))
            self.assertIn("REHYDRATE_SIGNAL_ABSENT_WRITE_ONLY", reason_codes)

    def test_verdict_write_only_can_pass_when_rehydrate_not_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, summary_json_path, summary_csv_path, delta_csv_path, order_check_path = self._create_inputs(
                root,
                matched_tokens_delta=0.0,
                onboard_d2d_delta=0.0,
                offload_h2d_delta=2.0,
                disk_hit_rate=0.0,
                replay_read_gib=0.0,
            )
            verdict_path = root / "verdict.json"

            rc = self._invoke_verdict(
                [
                    "phase70_write_verdict.py",
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
                    "--out",
                    str(verdict_path),
                    "--decision-grade-require-rehydrate",
                    "0",
                ]
            )
            self.assertEqual(rc, 0)
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
            checks = verdict.get("checks") or {}
            reason_codes = verdict.get("reason_codes") or []
            self.assertTrue(verdict.get("decision_grade"))
            self.assertTrue(checks.get("ssd_write_signal_present"))
            self.assertFalse(checks.get("ssd_rehydrate_signal_present"))
            self.assertIn("REHYDRATE_SIGNAL_ABSENT_WRITE_ONLY", reason_codes)

    def test_verdict_flags_order_effect_and_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, summary_json_path, summary_csv_path, delta_csv_path, order_check_path = self._create_inputs(
                root,
                pair_count=6,
            )
            _write_json(order_check_path, {"order_effect_summary": {"order_dependent": True}, "metrics": {}})
            verdict_path = root / "verdict.json"

            rc = self._invoke_verdict(
                [
                    "phase70_write_verdict.py",
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
                    "--out",
                    str(verdict_path),
                    "--decision-grade-hint",
                    "0",
                    "--reason-code",
                    "PREFLIGHT_METRICS_UNAVAILABLE",
                ]
            )

            self.assertEqual(rc, 0)
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
            self.assertFalse(verdict.get("decision_grade"))
            reason_codes = verdict.get("reason_codes") or []
            self.assertIn("PREFLIGHT_METRICS_UNAVAILABLE", reason_codes)
            self.assertIn("ORDER_EFFECT_SUSPECT", reason_codes)
            self.assertNotIn("ORDER_EFFECT_INSUFFICIENT_PAIRS", reason_codes)


if __name__ == "__main__":
    unittest.main()

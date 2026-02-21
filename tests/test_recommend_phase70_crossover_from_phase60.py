import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "recommend_phase70_crossover_from_phase60.py"
    spec = importlib.util.spec_from_file_location("recommend_phase70_crossover_from_phase60", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _row(
    *,
    mode: str,
    conc: int,
    ttfc_p95: float | None,
    ttft_p95: float | None,
    error_rate: float,
    onboard: float,
    disk_hit_rate: float = 0.0,
    status: str = "ok",
    b1_disk_tier_verified: bool | None = None,
) -> dict:
    row = {
        "mode": mode,
        "concurrency": conc,
        "replay_concurrency": conc,
        "point_key": f"{mode}_c{conc}",
        "phase": "sweep_point",
        "status": status,
        "error_rate": error_rate,
        "mechanism": {
            "kvbm_onboard_blocks_d2d_delta_replay_plus_replay2": onboard,
            "kvbm_matched_tokens_delta_replay_plus_replay2": 12.0 if onboard > 0 else 0.0,
        },
        "kvbm_disk_cache_hit_rate": disk_hit_rate,
    }
    if ttfc_p95 is not None:
        row["replay_ttfc_p95_ms"] = ttfc_p95
    if ttft_p95 is not None:
        row["replay_ttft_p95_ms"] = ttft_p95
    if b1_disk_tier_verified is not None:
        row["b1_disk_tier_verified"] = b1_disk_tier_verified
    return row


class RecommendPhase70CrossoverFromPhase60Tests(unittest.TestCase):
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

    def test_ideal_crossover_recommends_c4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ts = "20260218T123456Z"
            summary_path = root / f"phase60_rehydrate_concurrency_sweep_summary_minimal_{ts}.json"
            out_json = root / "recommendation.json"

            summary = {
                "meta": {"sweep_replay_concurrencies": [1, 2, 4]},
                "slo_replay_p95_ms": 110.0,
                "rows": [
                    _row(mode="B1", conc=1, ttfc_p95=95.0, ttft_p95=95.0, error_rate=0.0, onboard=0.0),
                    _row(mode="B2", conc=1, ttfc_p95=92.0, ttft_p95=92.0, error_rate=0.0, onboard=1.0),
                    _row(mode="B1", conc=2, ttfc_p95=120.0, ttft_p95=120.0, error_rate=0.0, onboard=0.0),
                    _row(mode="B2", conc=2, ttfc_p95=98.0, ttft_p95=98.0, error_rate=0.0, onboard=2.0),
                    _row(mode="B1", conc=4, ttfc_p95=170.0, ttft_p95=170.0, error_rate=0.0, onboard=0.0),
                    _row(mode="B2", conc=4, ttfc_p95=100.0, ttft_p95=100.0, error_rate=0.0, onboard=4.0),
                ],
            }
            _write_json(summary_path, summary)

            rc = self._invoke(
                [
                    "recommend_phase70_crossover_from_phase60.py",
                    "--phase60-summary-json",
                    str(summary_path),
                    "--out-json",
                    str(out_json),
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            recommended = payload.get("recommended") or []
            self.assertTrue(recommended)
            self.assertEqual(recommended[0].get("replay_concurrency"), 4)
            self.assertIn("MAX_STABLE_B1_B2_SEPARATION", recommended[0].get("reasons") or [])
            self.assertIn("B2_REHYDRATE_SIGNAL_PRESENT", recommended[0].get("reasons") or [])

    def test_rejects_non_zero_error_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ts = "20260218T130000Z"
            summary_path = root / f"phase60_rehydrate_concurrency_sweep_summary_minimal_{ts}.json"
            out_json = root / "recommendation.json"

            summary = {
                "meta": {"sweep_replay_concurrencies": [2, 4]},
                "slo_replay_p95_ms": 110.0,
                "rows": [
                    _row(mode="B1", conc=2, ttfc_p95=120.0, ttft_p95=120.0, error_rate=0.0, onboard=0.0),
                    _row(mode="B2", conc=2, ttfc_p95=98.0, ttft_p95=98.0, error_rate=0.0, onboard=2.0),
                    _row(mode="B1", conc=4, ttfc_p95=170.0, ttft_p95=170.0, error_rate=0.1, onboard=0.0),
                    _row(mode="B2", conc=4, ttfc_p95=100.0, ttft_p95=100.0, error_rate=0.0, onboard=4.0),
                ],
            }
            _write_json(summary_path, summary)

            rc = self._invoke(
                [
                    "recommend_phase70_crossover_from_phase60.py",
                    "--phase60-summary-json",
                    str(summary_path),
                    "--out-json",
                    str(out_json),
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual((payload.get("recommended") or [])[0].get("replay_concurrency"), 2)
            rejected = ((payload.get("rejection_summary") or {}).get("rejected_for_errors")) or []
            self.assertIn(4, rejected)

    def test_rejects_all_when_b2_rehydrate_required_and_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ts = "20260218T131500Z"
            summary_path = root / f"phase60_rehydrate_concurrency_sweep_summary_minimal_{ts}.json"
            out_json = root / "recommendation.json"

            summary = {
                "meta": {"sweep_replay_concurrencies": [2, 4]},
                "slo_replay_p95_ms": 110.0,
                "rows": [
                    _row(mode="B1", conc=2, ttfc_p95=120.0, ttft_p95=120.0, error_rate=0.0, onboard=0.0),
                    _row(mode="B2", conc=2, ttfc_p95=100.0, ttft_p95=100.0, error_rate=0.0, onboard=0.0, disk_hit_rate=0.0),
                    _row(mode="B1", conc=4, ttfc_p95=170.0, ttft_p95=170.0, error_rate=0.0, onboard=0.0),
                    _row(mode="B2", conc=4, ttfc_p95=108.0, ttft_p95=108.0, error_rate=0.0, onboard=0.0, disk_hit_rate=0.0),
                ],
            }
            _write_json(summary_path, summary)

            rc = self._invoke(
                [
                    "recommend_phase70_crossover_from_phase60.py",
                    "--phase60-summary-json",
                    str(summary_path),
                    "--out-json",
                    str(out_json),
                    "--require-b2-rehydrate",
                    "1",
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("recommended"), [])
            rejected = ((payload.get("rejection_summary") or {}).get("rejected_for_no_b2_rehydrate")) or []
            self.assertEqual(sorted(rejected), [2, 4])

    def test_falls_back_to_ttft_when_ttfc_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ts = "20260218T133000Z"
            summary_path = root / f"phase60_rehydrate_concurrency_sweep_summary_minimal_{ts}.json"
            out_json = root / "recommendation.json"

            summary = {
                "meta": {"sweep_replay_concurrencies": [2, 4]},
                "slo_replay_p95_ms": 110.0,
                "rows": [
                    _row(mode="B1", conc=2, ttfc_p95=None, ttft_p95=120.0, error_rate=0.0, onboard=0.0),
                    _row(mode="B2", conc=2, ttfc_p95=None, ttft_p95=100.0, error_rate=0.0, onboard=2.0),
                    _row(mode="B1", conc=4, ttfc_p95=None, ttft_p95=175.0, error_rate=0.0, onboard=0.0),
                    _row(mode="B2", conc=4, ttfc_p95=None, ttft_p95=99.0, error_rate=0.0, onboard=4.0),
                ],
            }
            _write_json(summary_path, summary)

            rc = self._invoke(
                [
                    "recommend_phase70_crossover_from_phase60.py",
                    "--phase60-summary-json",
                    str(summary_path),
                    "--out-json",
                    str(out_json),
                    "--metric",
                    "ttfc_p95_ms",
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            recommended = payload.get("recommended") or []
            self.assertTrue(recommended)
            self.assertEqual(recommended[0].get("replay_concurrency"), 4)
            b1_evidence = ((recommended[0].get("evidence") or {}).get("b1")) or {}
            self.assertEqual(b1_evidence.get("metric_used"), "ttft_p95_ms")
            warnings = payload.get("warnings") or []
            self.assertIn("ttfc_missing_fell_back_to_ttft", warnings)

    def test_rejects_when_b1_disk_tier_not_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ts = "20260218T134500Z"
            summary_path = root / f"phase60_rehydrate_concurrency_sweep_summary_minimal_{ts}.json"
            out_json = root / "recommendation.json"

            summary = {
                "meta": {"sweep_replay_concurrencies": [2]},
                "slo_replay_p95_ms": 110.0,
                "rows": [
                    _row(
                        mode="B1",
                        conc=2,
                        ttfc_p95=130.0,
                        ttft_p95=130.0,
                        error_rate=0.0,
                        onboard=0.0,
                        b1_disk_tier_verified=False,
                    ),
                    _row(mode="B2", conc=2, ttfc_p95=100.0, ttft_p95=100.0, error_rate=0.0, onboard=2.0),
                ],
            }
            _write_json(summary_path, summary)

            rc = self._invoke(
                [
                    "recommend_phase70_crossover_from_phase60.py",
                    "--phase60-summary-json",
                    str(summary_path),
                    "--out-json",
                    str(out_json),
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("recommended"), [])
            rejected = ((payload.get("rejection_summary") or {}).get("rejected_for_b1_disk_tier_not_disabled")) or []
            self.assertEqual(rejected, [2])


if __name__ == "__main__":
    unittest.main()

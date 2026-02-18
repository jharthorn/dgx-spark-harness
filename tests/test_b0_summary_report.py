import unittest

from bench import run_bench


class TestB0SummaryReport(unittest.TestCase):
    def test_b0_kvbm_summary_without_snapshots(self) -> None:
        summary = run_bench.build_kvbm_metrics_summary(
            snapshots=[],
            phase_deltas={
                "populate": {"available": False, "skipped": True, "reason": "kvbm_disabled"},
                "thrash": {"available": False, "skipped": True, "reason": "kvbm_disabled"},
                "replay": {"available": False, "skipped": True, "reason": "kvbm_disabled"},
            },
            scenario="rehydrate_replay",
            kv_mode="off",
            kvbm_enabled=False,
        )
        status = summary.get("kvbm_metrics_status")
        self.assertIsInstance(status, dict)
        self.assertEqual(status.get("status"), "skipped")
        self.assertFalse(summary.get("available"))
        self.assertIsNone((summary.get("rollup") or {}).get("matched_tokens_total_delta"))
        signal = summary.get("rehydrate_replay_signal")
        self.assertIsInstance(signal, dict)
        self.assertEqual(signal.get("signal"), "skipped")

    def test_report_renders_b0_without_kvbm_metrics(self) -> None:
        payload = {
            "run_id": "run_b0_test",
            "created_utc": "2026-02-13T00:00:00+00:00",
            "run_valid": True,
            "scenario": "rehydrate_replay",
            "tier_mode": "B0",
            "kv_mode": {"mode": "off", "kvbm_enabled": False, "cpu_cache_gb": 0, "disk_cache_gb": 0},
            "kvbm_enabled": False,
            "kvbm_metrics_available": False,
            "kvbm_metrics_status": {
                "status": "skipped",
                "kvbm_enabled": False,
                "metrics_available": False,
                "snapshot_count": 0,
                "reason": "kvbm_disabled",
            },
            "phase_summaries": [
                {
                    "phase": "replay",
                    "error_rate": 0.0,
                    "req_per_s": 1.0,
                    "io_delta": {"read_mib_delta": 0.0, "write_mib_delta": 0.0},
                    "worker_process_io_delta": {"read_mib_delta": 0.0, "write_mib_delta": 0.0},
                    "kvbm_metrics_delta": {"available": False, "skipped": True, "reason": "kvbm_disabled"},
                }
            ],
            "overall_summary": {
                "error_rate": 0.0,
                "req_per_s": 1.0,
                "output_tokens_per_s_est": 1.0,
                "latency_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0},
            },
            "kvbm_metrics": {
                "available": False,
                "reason": "kvbm_disabled",
                "rollup": {
                    "offload_blocks_total_delta": None,
                    "onboard_blocks_total_delta": None,
                    "matched_tokens_total_delta": None,
                },
                "rehydrate_replay_signal": {
                    "signal": "skipped",
                    "reason": "kvbm_disabled",
                    "interpretation": "KVBM counters unavailable for this run.",
                },
            },
            "request_identity": {"available": False},
            "fingerprint": {},
            "invalid_reason": None,
            "invalid_details": [],
            "variant_tags": [],
        }

        report = run_bench.render_report_markdown(payload)
        self.assertIn("KVBM metrics status: `skipped`", report)
        self.assertIn("KVBM rollup offload/onboard/matched: `NA` / `NA` / `NA`", report)
        self.assertIn("offload_blocks_delta=NA", report)


if __name__ == "__main__":
    unittest.main()

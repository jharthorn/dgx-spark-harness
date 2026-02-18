import tempfile
import unittest
from pathlib import Path

from bench import run_bench


class IOAttributionReportTests(unittest.TestCase):
    def test_empty_io_attribution_report_schema(self) -> None:
        report = run_bench._empty_io_attribution_report(
            primary_nvme_device="/dev/nvme0",
            kvbm_disk_path="/mnt/nvme/kvbm",
            phase_windows={"replay": {"start": "2026-02-12T00:00:00+00:00", "end": "2026-02-12T00:01:00+00:00"}},
            error="collector_failed",
            capture_errors=[{"command": "x", "return_code": None, "stderr_snippet": "collector_failed"}],
        )
        for key in (
            "primary_nvme_device",
            "kvbm_disk_path",
            "phase_windows",
            "block_io_by_phase",
            "process_io_by_phase",
            "kvbm_file_io_by_phase",
            "direct_io_evidence",
            "capture_errors",
        ):
            self.assertIn(key, report)
        self.assertFalse(report.get("available"))

    def test_build_report_uses_phase_windows_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            collector = run_bench.IOAttributionCollector(
                run_dir=Path(tmp),
                primary_nvme_device="/dev/nvme0",
                block_device="nvme0n1",
                proc_pattern=r"dynamo\.trtllm",
                kvbm_disk_path="/mnt/nvme/kvbm",
                interval_s=1.0,
                top_procs=4,
                kv_runtime_env={"DYN_KVBM_DISK_CACHE_DIR": "/mnt/nvme/kvbm"},
            )
            collector.phase_windows = {
                "replay": {
                    "start": "2026-02-12T00:00:01+00:00",
                    "end": "2026-02-12T00:00:04+00:00",
                }
            }
            collector.block_samples = [
                {"timestamp_utc": "2026-02-12T00:00:00+00:00", "success": True, "read_sectors": 100, "write_sectors": 50},
                {"timestamp_utc": "2026-02-12T00:00:05+00:00", "success": True, "read_sectors": 150, "write_sectors": 70},
            ]
            collector.process_samples = [
                {
                    "timestamp_utc": "2026-02-12T00:00:00+00:00",
                    "success": True,
                    "selection_mode": "pattern_match",
                    "processes": [
                        {
                            "pid": 111,
                            "cmdline": "python -m dynamo.trtllm",
                            "read_bytes": 1000,
                            "write_bytes": 400,
                            "syscr": 1,
                            "syscw": 1,
                        }
                    ],
                },
                {
                    "timestamp_utc": "2026-02-12T00:00:05+00:00",
                    "success": True,
                    "selection_mode": "pattern_match",
                    "processes": [
                        {
                            "pid": 111,
                            "cmdline": "python -m dynamo.trtllm",
                            "read_bytes": 4500,
                            "write_bytes": 900,
                            "syscr": 2,
                            "syscw": 2,
                        }
                    ],
                },
            ]
            collector.lsof_samples = [
                {
                    "timestamp_utc": "2026-02-12T00:00:02+00:00",
                    "success": True,
                    "entries": [{"pid": 111, "path": "/mnt/nvme/kvbm/block_A"}],
                }
            ]

            report = collector.build_report(phase_block_deltas={}, phase_process_deltas={})

        self.assertTrue(report.get("available"))
        self.assertEqual(report.get("primary_nvme_device"), "/dev/nvme0")
        self.assertEqual(report.get("kvbm_disk_path"), "/mnt/nvme/kvbm")
        self.assertIn("replay", report.get("block_io_by_phase", {}))
        self.assertIn("replay", report.get("process_io_by_phase", {}))
        self.assertIn("replay", (report.get("kvbm_file_io_by_phase") or {}).get("phases", {}))
        replay_block = (report.get("block_io_by_phase") or {}).get("replay") or {}
        replay_proc = (report.get("process_io_by_phase") or {}).get("replay") or {}
        self.assertGreater(int(replay_block.get("read_bytes") or 0), 0)
        self.assertGreater(int(replay_proc.get("read_bytes") or 0), 0)


if __name__ == "__main__":
    unittest.main()

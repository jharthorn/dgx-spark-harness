"""Utility wrappers for shell-based telemetry collectors in `bench/scripts/`."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LOG = logging.getLogger(__name__)


@dataclass
class TelemetryReport:
    started: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TelemetryManager:
    def __init__(
        self,
        run_dir: Path,
        *,
        kvbm_cache_dir: str,
        container_name: str,
        iostat_device: str = "nvme0n1",
        interval_s: int = 1,
        pid_target: str = "ALL",
        scripts_dir: Optional[Path] = None,
    ) -> None:
        self.run_dir = run_dir
        self.kvbm_cache_dir = kvbm_cache_dir
        self.container_name = container_name
        self.iostat_device = iostat_device
        self.interval_s = interval_s
        self.pid_target = pid_target
        self.scripts_dir = scripts_dir or (Path(__file__).resolve().parent / "scripts")

    def start_default(self) -> TelemetryReport:
        report = TelemetryReport()
        self.snapshot_kvbm_dir("start")
        for script_name, args in (
            ("start_iostat.sh", [str(self.run_dir), self.iostat_device, str(self.interval_s)]),
            ("start_pidstat.sh", [str(self.run_dir), self.pid_target, str(self.interval_s)]),
            ("start_gpu_dmon.sh", [str(self.run_dir), str(self.interval_s)]),
        ):
            completed = self._run_script(script_name, args)
            if completed.returncode == 0:
                report.started.append(script_name)
            else:
                report.warnings.append(f"{script_name}: {completed.stderr.strip() or completed.stdout.strip()}")
        return report

    def stop_default(self) -> TelemetryReport:
        report = TelemetryReport()
        for script_name, args in (
            ("stop_iostat.sh", [str(self.run_dir)]),
            ("stop_pidstat.sh", [str(self.run_dir)]),
            ("stop_gpu_dmon.sh", [str(self.run_dir)]),
            ("collect_docker_logs.sh", [str(self.run_dir), self.container_name]),
            ("collect_cufile_logs.sh", [str(self.run_dir), self.container_name]),
        ):
            completed = self._run_script(script_name, args)
            if completed.returncode == 0:
                report.started.append(script_name)
            else:
                report.warnings.append(f"{script_name}: {completed.stderr.strip() or completed.stdout.strip()}")
        self.snapshot_kvbm_dir("end")
        return report

    def snapshot_kvbm_dir(self, label: str) -> subprocess.CompletedProcess:
        return self._run_script(
            "snapshot_kvbm_dir.sh",
            [str(self.run_dir), self.kvbm_cache_dir, label],
        )

    def collect_logs(self) -> TelemetryReport:
        report = TelemetryReport()
        for script_name, args in (
            ("collect_docker_logs.sh", [str(self.run_dir), self.container_name]),
            ("collect_cufile_logs.sh", [str(self.run_dir), self.container_name]),
        ):
            completed = self._run_script(script_name, args)
            if completed.returncode == 0:
                report.started.append(script_name)
            else:
                report.warnings.append(f"{script_name}: {completed.stderr.strip() or completed.stdout.strip()}")
        return report

    def _run_script(self, script_name: str, args: list[str]) -> subprocess.CompletedProcess:
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            msg = f"Missing telemetry script: {script_path}"
            LOG.warning(msg)
            return subprocess.CompletedProcess(args=[str(script_path), *args], returncode=1, stdout="", stderr=msg)
        cmd = [str(script_path), *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=False)


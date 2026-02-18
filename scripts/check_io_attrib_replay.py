#!/usr/bin/env python3
"""Post-run checker for replay I/O attribution evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso8601(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def format_bytes(value: int) -> str:
    amount = max(int(value), 0)
    if amount == 0:
        return "0B"
    gib = 1024**3
    mib = 1024**2
    kib = 1024
    if amount >= gib:
        return f"{amount / gib:.2f}GiB"
    if amount >= mib:
        return f"{amount / mib:.2f}MiB"
    if amount >= kib:
        return f"{amount / kib:.2f}KiB"
    return f"{amount}B"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_mode(summary: dict[str, Any]) -> tuple[str, bool]:
    tier_mode = str(summary.get("tier_mode") or "").strip().upper()
    kv_mode_raw = summary.get("kv_mode")
    if isinstance(kv_mode_raw, dict):
        kv_mode = str(kv_mode_raw.get("mode") or "").strip()
        disk_cache_gb = float(kv_mode_raw.get("disk_cache_gb", 0.0) or 0.0)
    else:
        kv_mode = str(kv_mode_raw or "").strip()
        disk_cache_gb = 0.0
    mode = "unknown"
    if tier_mode in {"B0", "B1", "B2"}:
        mode = tier_mode
    elif kv_mode == "cpu_disk":
        mode = "B2"
    elif kv_mode == "cpu_only":
        mode = "B1"
    elif kv_mode == "off":
        mode = "B0"
    disk_enabled = bool(mode == "B2" or kv_mode == "cpu_disk" or disk_cache_gb > 0.0)
    return mode, disk_enabled


def resolve_replay_phase(report: dict[str, Any]) -> str | None:
    phase_windows = report.get("phase_windows") if isinstance(report.get("phase_windows"), dict) else {}
    for name in ("replay", "replay_A"):
        if name in phase_windows:
            return name
    replay_like = sorted(name for name in phase_windows.keys() if str(name).startswith("replay"))
    return replay_like[0] if replay_like else None


def evaluate(
    *,
    run_dir: Path,
    expect_report: bool,
) -> tuple[dict[str, Any], int]:
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "io" / "io_attribution_report.json"
    verdict_path = run_dir / "io" / "io_attrib_verdict.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)

    checks: list[CheckResult] = []
    summary: dict[str, Any] = {}
    mode = "unknown"
    disk_enabled = False
    if summary_path.exists():
        try:
            summary = load_json(summary_path)
            mode, disk_enabled = resolve_mode(summary)
        except Exception as exc:  # noqa: BLE001
            checks.append(CheckResult("summary_parse", "WARN", f"failed to parse summary.json: {exc}"))
    else:
        checks.append(CheckResult("summary_present", "WARN", "summary.json missing; mode inference unavailable"))

    io_attrib_summary = summary.get("io_attribution") if isinstance(summary.get("io_attribution"), dict) else {}
    io_expected = bool(expect_report or io_attrib_summary.get("enabled"))
    strict_required = bool(mode == "B2" and disk_enabled)

    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = load_json(report_path)
            checks.append(CheckResult("io_report_present", "PASS", str(report_path)))
        except Exception as exc:  # noqa: BLE001
            checks.append(CheckResult("io_report_parse", "FAIL", f"failed to parse io_attribution_report.json: {exc}"))
            report = {}
    else:
        missing_status = "FAIL" if io_expected else "WARN"
        checks.append(CheckResult("io_report_present", missing_status, f"missing: {report_path}"))

    replay_phase = resolve_replay_phase(report)
    phase_windows = report.get("phase_windows") if isinstance(report.get("phase_windows"), dict) else {}
    replay_window = phase_windows.get(replay_phase) if replay_phase else None

    replay_block = ((report.get("block_io_by_phase") or {}).get(replay_phase) or {}) if replay_phase else {}
    replay_proc = ((report.get("process_io_by_phase") or {}).get(replay_phase) or {}) if replay_phase else {}
    replay_block_read = to_int(replay_block.get("read_bytes"))
    replay_proc_read = to_int(replay_proc.get("read_bytes"))
    replay_proc_cgroup_read = to_int(replay_proc.get("cgroup_read_bytes"))

    per_pid_raw = replay_proc.get("per_pid") if isinstance(replay_proc.get("per_pid"), list) else []
    per_pid_readers_available = len(per_pid_raw) > 0
    positive_pid_readers: list[dict[str, int]] = []
    top_pids: list[dict[str, int]] = []
    for item in per_pid_raw:
        if not isinstance(item, dict):
            continue
        pid = to_int(item.get("pid"))
        read_delta = to_int(item.get("read_bytes_delta"))
        if read_delta <= 0:
            read_delta = to_int(item.get("read_bytes"))
        if pid > 0 and read_delta > 0:
            positive_pid_readers.append({"pid": pid, "read_bytes_delta": read_delta})
    top_pids = sorted(positive_pid_readers, key=lambda row: row["read_bytes_delta"], reverse=True)[:10]
    per_pid_readers_nonzero = len(positive_pid_readers) > 0
    per_pid_reader_total = sum(int(item["read_bytes_delta"]) for item in positive_pid_readers)

    # /proc/<pid>/io can miss reads for some containerized/cached paths. Accept
    # cgroup io.stat attribution as process evidence when per-PID deltas are zero.
    process_evidence_method = "none"
    if replay_proc_read > 0:
        process_evidence_method = "pid"
    elif replay_proc_cgroup_read > 0:
        process_evidence_method = "cgroup"

    if strict_required:
        if replay_phase and isinstance(replay_window, dict):
            start_ts = replay_window.get("start")
            end_ts = replay_window.get("end")
            if parse_iso8601(start_ts) and parse_iso8601(end_ts):
                checks.append(CheckResult("replay_phase_window", "PASS", f"phase={replay_phase} start/end present"))
            else:
                checks.append(CheckResult("replay_phase_window", "FAIL", f"phase={replay_phase} has invalid timestamps"))
        else:
            checks.append(CheckResult("replay_phase_window", "FAIL", "missing replay phase window"))

        checks.append(
            CheckResult(
                "replay_block_reads_positive",
                ("PASS" if replay_block_read > 0 else "FAIL"),
                f"replay block read bytes={replay_block_read}",
            )
        )
        checks.append(
            CheckResult(
                "replay_process_reads_positive",
                ("PASS" if process_evidence_method != "none" else "FAIL"),
                (
                    "replay process read bytes "
                    f"pid_total={replay_proc_read} cgroup_total={replay_proc_cgroup_read} "
                    f"method={process_evidence_method}"
                ),
            )
        )
        pid_reader_check_status = "PASS"
        pid_reader_check_detail = f"positive replay reader pids={len(top_pids)}"
        if per_pid_readers_nonzero:
            pid_reader_check_status = "PASS"
            pid_reader_check_detail = f"positive replay reader pids={len(top_pids)}"
        elif process_evidence_method == "none":
            pid_reader_check_status = "FAIL"
            pid_reader_check_detail = "per-pid replay readers absent and no cgroup process evidence"
        elif process_evidence_method == "cgroup":
            if per_pid_readers_available:
                pid_reader_check_status = "WARN"
                pid_reader_check_detail = (
                    f"per-pid replay readers are zero; using cgroup replay read bytes={replay_proc_cgroup_read}"
                )
            else:
                pid_reader_check_status = "WARN"
                pid_reader_check_detail = (
                    f"per-pid replay reader list missing; using cgroup replay read bytes={replay_proc_cgroup_read}"
                )
        else:
            pid_reader_check_status = "WARN"
            pid_reader_check_detail = (
                f"aggregate process replay reads positive ({replay_proc_read}) but no per-pid readers were nonzero"
            )
        checks.append(
            CheckResult(
                "replay_process_pid_readers_present",
                pid_reader_check_status,
                pid_reader_check_detail,
            )
        )
    else:
        checks.append(CheckResult("strict_replay_gate", "PASS", f"strict gate not required for mode={mode}"))

    kvbm_disk_path = str(report.get("kvbm_disk_path") or "")
    file_io = report.get("kvbm_file_io_by_phase") if isinstance(report.get("kvbm_file_io_by_phase"), dict) else {}
    if kvbm_disk_path:
        if file_io.get("available"):
            phase_map = file_io.get("phases") if isinstance(file_io.get("phases"), dict) else {}
            replay_files = phase_map.get(replay_phase) if replay_phase else None
            replay_file_count = len(replay_files) if isinstance(replay_files, list) else 0
            status = "PASS" if replay_file_count > 0 else "WARN"
            checks.append(
                CheckResult(
                    "replay_kvbm_file_observation",
                    status,
                    f"replay observed files under kvbm path={replay_file_count}",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "replay_kvbm_file_observation",
                    "WARN",
                    str(file_io.get("reason") or "file-level attribution unavailable"),
                )
            )
    else:
        checks.append(CheckResult("replay_kvbm_file_observation", "WARN", "kvbm_disk_path missing/empty"))

    fail_count = sum(1 for item in checks if item.status == "FAIL")
    warn_count = sum(1 for item in checks if item.status == "WARN")
    passed = fail_count == 0

    verdict = {
        "pass": passed,
        "mode": mode,
        "strict_replay_gate_required": strict_required,
        "checks": [item.to_dict() for item in checks],
        "replay_phase": replay_phase,
        "replay_read_bytes_block": replay_block_read,
        "replay_read_bytes_process_total": replay_proc_read,
        "replay_read_bytes_cgroup_total": replay_proc_cgroup_read,
        "process_evidence_method": process_evidence_method,
        "per_pid_readers_available": per_pid_readers_available,
        "per_pid_readers_nonzero": per_pid_readers_nonzero,
        "top_pids_replay": top_pids,
        "timestamp_utc": now_utc_iso(),
        "artifacts": {
            "summary_path": str(summary_path),
            "io_attribution_report_path": str(report_path),
            "io_attrib_verdict_path": str(verdict_path),
        },
    }
    verdict_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    label = "PASS" if passed else "FAIL"
    method_bytes = 0
    if process_evidence_method == "pid":
        method_bytes = replay_proc_read
    elif process_evidence_method == "cgroup":
        method_bytes = replay_proc_cgroup_read
    warn_tags: list[str] = []
    if any(item.name == "replay_process_pid_readers_present" and item.status == "WARN" for item in checks):
        warn_tags.append("pid_attrib_zero")
    warn_suffix = f" WARN {','.join(warn_tags)}" if warn_tags else ""
    method_note = " note=cgroup_expected_in_containerized_runs" if process_evidence_method == "cgroup" else ""
    print(
        f"{label} io-attrib replay: block={format_bytes(replay_block_read)} "
        f"proc({process_evidence_method})={format_bytes(method_bytes)} pid={format_bytes(per_pid_reader_total)} "
        f"(method={process_evidence_method}){warn_suffix} "
        f"mode={mode} strict={int(strict_required)} "
        f"replay_block_read={replay_block_read} replay_process_read={replay_proc_read} replay_cgroup_read={replay_proc_cgroup_read} "
        f"warns={warn_count} fails={fail_count}{method_note} verdict={verdict_path}"
    )
    return verdict, (0 if passed else 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check replay I/O attribution evidence for a run bundle.")
    parser.add_argument("--run-dir", required=True, help="Run bundle directory (contains summary.json and io/).")
    parser.add_argument(
        "--expect-report",
        action="store_true",
        help="Fail when io/io_attribution_report.json is missing (use when --io-attrib was enabled).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, rc = evaluate(
        run_dir=Path(args.run_dir),
        expect_report=bool(args.expect_report),
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

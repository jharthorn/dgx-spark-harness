#!/usr/bin/env python3
"""Print a compact progress snapshot for an in-flight Phase70 paired run."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_DIR_RE = re.compile(r"phase70_pair_logs_(\d{8}T\d{6}Z)$")
LEG_LOG_RE = re.compile(
    r"phase70_rehydrate_pair_(?P<mode>B[0-2])_p(?P<pair>\d+)_l(?P<leg>\d+)_(?P<ts>\d{8}T\d{6}Z)\.log$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase70 progress snapshot")
    parser.add_argument("--results-root", default="bench/results")
    parser.add_argument("--ts", default="", help="Optional phase70 timestamp (e.g. 20260217T161447Z).")
    parser.add_argument(
        "--max-completed",
        type=int,
        default=16,
        help="How many completed-leg rows to print (most recent first).",
    )
    parser.add_argument(
        "--stall-warn-min",
        type=float,
        default=25.0,
        help="Emit STALL_WARN=true when minutes since last active-log timestamp exceeds this threshold.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_log_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    text = raw.replace(b"\x00", b"").decode("utf-8", errors="replace")
    return text.splitlines()


def is_timestamped_line(line: str) -> bool:
    return bool(
        re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d{3}", line)
        or re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", line)
        or re.match(r"^==== \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", line)
    )


def parse_timestamp_from_line(line: str) -> datetime | None:
    text = line.strip()
    if text.startswith("==== "):
        text = text[5:].strip()
    if not text:
        return None

    m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d{3})", text)
    if m:
        raw = m.group(1).replace(",", ".")
        try:
            return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return None

    m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", text)
    if m:
        raw = m.group(1).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def minutes_since(dt: datetime) -> float:
    if dt.tzinfo is None:
        now = datetime.now()
    else:
        now = datetime.now(timezone.utc)
    delta = now - dt
    return delta.total_seconds() / 60.0


def last_timestamped_line(path: Path) -> tuple[str, datetime | None]:
    lines = decode_log_lines(path)
    for line in reversed(lines):
        if is_timestamped_line(line):
            return line, parse_timestamp_from_line(line)
    for line in reversed(lines):
        if line.strip():
            return line, None
    return "<empty-log>", None


def last_a2_line(path: Path) -> str:
    lines = decode_log_lines(path)
    for line in reversed(lines):
        if "io-attrib replay:" in line:
            return line
    return "<no-io-attrib-line>"


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def resolve_log_dir(results_root: Path, ts: str) -> tuple[Path | None, str]:
    if ts:
        log_dir = results_root / f"phase70_pair_logs_{ts}"
        return (log_dir if log_dir.exists() else None), ts
    candidates = [p for p in results_root.glob("phase70_pair_logs_*") if p.is_dir()]
    if not candidates:
        return None, ""
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    match = LOG_DIR_RE.search(latest.name)
    return latest, (match.group(1) if match else "")


def active_leg_from_logs(log_dir: Path) -> tuple[Path | None, dict[str, str]]:
    logs = [p for p in log_dir.glob("*.log") if p.is_file()]
    if not logs:
        return None, {}
    active = max(logs, key=lambda p: p.stat().st_mtime)
    match = LEG_LOG_RE.match(active.name)
    if not match:
        return active, {}
    return active, {k: match.group(k) for k in ("mode", "pair", "leg", "ts")}


def find_run_log(log_dir: Path, pair_id: int, pair_leg: int, mode: str) -> Path | None:
    mode = mode.upper()
    pattern = f"phase70_rehydrate_pair_{mode}_p{pair_id:02d}_l{pair_leg}_*.log"
    matches = sorted(log_dir.glob(pattern))
    return matches[0] if matches else None


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    log_dir, ts = resolve_log_dir(results_root, args.ts)

    print("Phase70 Progress Snapshot")
    print(f"- results_root: {results_root}")
    print(f"- selected_ts: {ts or '<none>'}")

    if log_dir is None:
        print("- log_dir: <missing>")
        print("- status: no phase70 log directory found")
        return 0

    manifest_path = results_root / f"phase70_rehydrate_pair_repeats_manifest_{ts}.json"
    print(f"- log_dir: {log_dir}")
    print(f"- manifest: {manifest_path} ({'present' if manifest_path.exists() else 'missing'})")

    active_log, active_meta = active_leg_from_logs(log_dir)
    if active_log is None:
        print("- active_leg: <none>")
    else:
        if active_meta:
            print(
                "- active_leg: "
                f"pair={int(active_meta['pair'])} leg={int(active_meta['leg'])} mode={active_meta['mode']} "
                f"(from {active_log.name})"
            )
        else:
            print(f"- active_leg: {active_log.name} (filename did not match expected pattern)")
        active_line, active_dt = last_timestamped_line(active_log)
        print(f"- active_leg_last_ts_line: {active_line}")
        if active_dt is None:
            print("- minutes_since_last_log_ts: <unknown>")
            print("- STALL_WARN: <unknown>")
        else:
            mins = minutes_since(active_dt)
            stall = mins > float(args.stall_warn_min)
            print(f"- minutes_since_last_log_ts: {mins:.1f}")
            print(f"- STALL_WARN: {'true' if stall else 'false'} (threshold_min={args.stall_warn_min:g})")

    runs: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
            meta = manifest.get("meta") if isinstance(manifest.get("meta"), dict) else {}
        except Exception as exc:  # noqa: BLE001
            print(f"- manifest_parse_error: {exc}")
    print(
        "- runs_recorded: "
        f"{len(runs)}"
        + (
            f" / expected_legs={int(meta.get('pair_count') or 0) * 2}"
            if meta.get("pair_count") is not None
            else ""
        )
    )

    if runs:
        print("- completed_leg_artifacts:")
        recent_runs = sorted(
            (r for r in runs if isinstance(r, dict)),
            key=lambda r: (int(r.get("pair_id") or 0), int(r.get("pair_leg") or 0)),
            reverse=True,
        )[: max(args.max_completed, 0)]
        for run in recent_runs:
            run_dir = Path(str(run.get("run_dir") or ""))
            mode = str(run.get("mode") or "").upper()
            pair_id = int(run.get("pair_id") or 0)
            pair_leg = int(run.get("pair_leg") or 0)
            summary_ok = (run_dir / "summary.json").exists()
            requests_ok = (run_dir / "requests.jsonl").exists()
            verdict_path = run_dir / "io" / "io_attrib_verdict.json"
            verdict_ok = verdict_path.exists()
            req = [summary_ok, requests_ok]
            if mode == "B2":
                req.append(verdict_ok)
                a2_label = "Y" if verdict_ok else "N"
            else:
                a2_label = "NA"
            all_required = all(req)
            print(
                "  - "
                f"pair={pair_id} leg={pair_leg} mode={mode} "
                f"summary={'Y' if summary_ok else 'N'} "
                f"requests={'Y' if requests_ok else 'N'} "
                f"a2_verdict={a2_label} "
                f"required_ok={'Y' if all_required else 'N'}"
            )
    else:
        print("- completed_leg_artifacts: <none-yet>")

    b2_runs = [r for r in runs if isinstance(r, dict) and str(r.get("mode") or "").upper() == "B2"]
    if b2_runs:
        print("- b2_a2_latest:")
        for run in sorted(b2_runs, key=lambda r: (int(r.get("pair_id") or 0), int(r.get("pair_leg") or 0))):
            pair_id = int(run.get("pair_id") or 0)
            pair_leg = int(run.get("pair_leg") or 0)
            mode = str(run.get("mode") or "B2").upper()
            run_dir = Path(str(run.get("run_dir") or ""))
            verdict_path = run_dir / "io" / "io_attrib_verdict.json"
            pass_fail = "UNKNOWN"
            if verdict_path.exists():
                try:
                    verdict = read_json(verdict_path)
                    pass_fail = "PASS" if truthy(verdict.get("pass")) else "FAIL"
                except Exception:  # noqa: BLE001
                    pass_fail = "PARSE_ERR"
            else:
                pass_fail = "MISSING"
            log_path = find_run_log(log_dir, pair_id, pair_leg, mode)
            a2_line = last_a2_line(log_path) if log_path is not None else "<log-not-found>"
            print(f"  - pair={pair_id} leg={pair_leg} mode={mode} verdict={pass_fail}")
            print(f"    {a2_line}")
    else:
        print("- b2_a2_latest: <no-completed-b2-legs-yet>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

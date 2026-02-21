#!/usr/bin/env python3
"""Evaluate early Phase70 mechanism-gate signals from a completed run leg."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_replay_phase(summary: dict[str, Any]) -> dict[str, Any]:
    phases = summary.get("phase_summaries") if isinstance(summary.get("phase_summaries"), list) else []
    for name in ("replay", "replay_A"):
        for phase in phases:
            if isinstance(phase, dict) and str(phase.get("phase")) == name:
                return phase
    for phase in phases:
        if isinstance(phase, dict) and str(phase.get("phase") or "").startswith("replay"):
            return phase
    return {}


def metric_value(kv_delta: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = parse_float(kv_delta.get(key))
        if value is not None:
            return value
    return 0.0


def replay_disk_hit_rate(replay: dict[str, Any]) -> float:
    kv_end = replay.get("kvbm_metrics_end") if isinstance(replay.get("kvbm_metrics_end"), dict) else {}
    kv_start = replay.get("kvbm_metrics_start") if isinstance(replay.get("kvbm_metrics_start"), dict) else {}
    kv_delta = replay.get("kvbm_metrics_delta") if isinstance(replay.get("kvbm_metrics_delta"), dict) else {}

    for candidate in (
        kv_end.get("kvbm_disk_cache_hit_rate"),
        kv_delta.get("kvbm_disk_cache_hit_rate"),
        kv_delta.get("kvbm_disk_cache_hit_rate_delta"),
    ):
        value = parse_float(candidate)
        if value is not None:
            return value

    end_rate = parse_float(kv_end.get("kvbm_disk_cache_hit_rate"))
    start_rate = parse_float(kv_start.get("kvbm_disk_cache_hit_rate"))
    if end_rate is not None and start_rate is not None:
        return end_rate - start_rate
    return 0.0


def detect_run_id(summary: dict[str, Any], run_dir: Path) -> str:
    for key in ("run_id", "id"):
        value = summary.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return run_dir.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check early Phase70 mechanism gate signals.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing summary.json.")
    parser.add_argument("--summary-json", default=None, help="Optional summary.json override path.")
    parser.add_argument("--min-disk-hit-rate", type=float, default=0.01, help="Minimum disk hit-rate considered non-trivial.")
    parser.add_argument("--min-matched-tokens", type=float, default=0.0, help="Minimum matched token delta.")
    parser.add_argument("--require-matched", action="store_true", help="Require matched_tokens signal in addition to hard SSD signal.")
    parser.add_argument("--json-out", default=None, help="Optional output path for gate payload JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_path = Path(args.summary_json) if args.summary_json else (run_dir / "summary.json")
    if not summary_path.exists():
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pass": False,
            "reason_codes": ["GATE_NO_SSD_MECHANISM_SIGNAL"],
            "error": f"missing_summary:{summary_path}",
        }
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, separators=(",", ":")))
        return 1

    summary = load_json(summary_path)
    replay = resolve_replay_phase(summary)
    kv_delta = replay.get("kvbm_metrics_delta") if isinstance(replay.get("kvbm_metrics_delta"), dict) else {}

    matched_tokens = metric_value(kv_delta, ("kvbm_matched_tokens_delta", "matched_tokens_total_delta"))
    onboard_d2d = metric_value(kv_delta, ("kvbm_onboard_blocks_d2d_delta", "onboard_blocks_total_delta"))
    offload_h2d = metric_value(kv_delta, ("kvbm_offload_blocks_h2d_delta", "offload_blocks_total_delta"))
    disk_hit_rate = replay_disk_hit_rate(replay)

    hard_signal = (onboard_d2d > 0.0) or (offload_h2d > 0.0) or (disk_hit_rate >= args.min_disk_hit_rate)
    matched_signal = matched_tokens > args.min_matched_tokens
    passed = hard_signal and (matched_signal if args.require_matched else True)

    reason_codes: list[str] = []
    notes: list[str] = []
    if not hard_signal:
        reason_codes.append("GATE_NO_SSD_MECHANISM_SIGNAL")
    if args.require_matched and not matched_signal:
        reason_codes.append("GATE_NO_SSD_MECHANISM_SIGNAL")
    if hard_signal and not matched_signal:
        notes.append("matched_tokens_signal_missing_soft")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pass": passed,
        "reason_codes": sorted(set(reason_codes)),
        "run_dir": str(run_dir),
        "run_id": detect_run_id(summary, run_dir),
        "summary_json": str(summary_path),
        "thresholds": {
            "min_disk_hit_rate": args.min_disk_hit_rate,
            "min_matched_tokens": args.min_matched_tokens,
            "require_matched": bool(args.require_matched),
        },
        "signals": {
            "kvbm_matched_tokens_delta": matched_tokens,
            "kvbm_onboard_blocks_d2d_delta": onboard_d2d,
            "kvbm_offload_blocks_h2d_delta": offload_h2d,
            "kvbm_disk_cache_hit_rate": disk_hit_rate,
        },
        "hard_signal": hard_signal,
        "matched_signal": matched_signal,
        "notes": notes,
        "references": {
            "nvidia_kvbm_metrics": [
                "kvbm_matched_tokens",
                "kvbm_onboard_blocks_d2d",
                "kvbm_offload_blocks_h2d",
                "kvbm_host_cache_hit_rate",
                "kvbm_disk_cache_hit_rate",
            ],
        },
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

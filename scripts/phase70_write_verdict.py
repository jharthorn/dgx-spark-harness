#!/usr/bin/env python3
"""Write a publishability/decision-grade verdict for Phase70 paired repeats."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MIN_ORDER_EFFECT_PAIRS = 6


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def parse_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_replay_phase(summary: dict[str, Any]) -> dict[str, Any]:
    phase_summaries = summary.get("phase_summaries") if isinstance(summary.get("phase_summaries"), list) else []
    for name in ("replay", "replay_A"):
        for phase in phase_summaries:
            if isinstance(phase, dict) and str(phase.get("phase")) == name:
                return phase
    for phase in phase_summaries:
        if isinstance(phase, dict) and str(phase.get("phase") or "").startswith("replay"):
            return phase
    return {}


def extract_disk_hit_rate(replay: dict[str, Any]) -> float:
    kv_end = replay.get("kvbm_metrics_end") if isinstance(replay.get("kvbm_metrics_end"), dict) else {}
    kv_delta = replay.get("kvbm_metrics_delta") if isinstance(replay.get("kvbm_metrics_delta"), dict) else {}
    kv_start = replay.get("kvbm_metrics_start") if isinstance(replay.get("kvbm_metrics_start"), dict) else {}

    for candidate in (
        kv_end.get("kvbm_disk_cache_hit_rate"),
        kv_delta.get("kvbm_disk_cache_hit_rate"),
        kv_delta.get("kvbm_disk_cache_hit_rate_delta"),
    ):
        value = parse_float(candidate)
        if value is not None:
            return value

    start = parse_float(kv_start.get("kvbm_disk_cache_hit_rate"))
    end = parse_float(kv_end.get("kvbm_disk_cache_hit_rate"))
    if start is not None and end is not None:
        return end - start
    return 0.0


def extract_run_mechanism_signals(run_path: str | None) -> dict[str, float]:
    empty = {
        "matched_tokens": 0.0,
        "onboard_d2d": 0.0,
        "offload_h2d": 0.0,
        "disk_hit_rate": 0.0,
    }
    if not run_path:
        return empty

    summary_path = Path(run_path) / "summary.json"
    if not summary_path.exists():
        return empty
    try:
        summary = load_json(summary_path)
    except Exception:
        return empty

    replay = resolve_replay_phase(summary)
    kv_delta = replay.get("kvbm_metrics_delta") if isinstance(replay.get("kvbm_metrics_delta"), dict) else {}
    matched = parse_float(kv_delta.get("kvbm_matched_tokens_delta"))
    if matched is None:
        matched = parse_float(kv_delta.get("matched_tokens_total_delta"))

    onboard = parse_float(kv_delta.get("kvbm_onboard_blocks_d2d_delta"))
    if onboard is None:
        onboard = parse_float(kv_delta.get("onboard_blocks_total_delta"))

    offload = parse_float(kv_delta.get("kvbm_offload_blocks_h2d_delta"))
    if offload is None:
        offload = parse_float(kv_delta.get("offload_blocks_total_delta"))

    return {
        "matched_tokens": matched or 0.0,
        "onboard_d2d": onboard or 0.0,
        "offload_h2d": offload or 0.0,
        "disk_hit_rate": extract_disk_hit_rate(replay),
    }


def row_status_ok(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "").strip().lower() in {"ok", "success", "valid"}


def parse_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def parse_ts(manifest_path: Path, manifest: dict[str, Any]) -> str | None:
    stem = manifest_path.stem
    m = re.match(r"phase70_rehydrate_pair_repeats_manifest_(.+)$", stem)
    if m:
        return m.group(1)
    created = ((manifest.get("meta") or {}).get("created_utc")) if isinstance(manifest.get("meta"), dict) else None
    if isinstance(created, str) and created:
        return created
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Phase70 decision-grade verdict.")
    parser.add_argument("--manifest", required=True, help="Phase70 manifest JSON path.")
    parser.add_argument("--summary-json", required=True, help="Phase70 summary JSON path.")
    parser.add_argument("--summary-csv", required=True, help="Phase70 summary CSV path.")
    parser.add_argument("--pair-delta-csv", required=True, help="Phase70 pair-delta CSV path.")
    parser.add_argument("--order-check-json", required=True, help="Phase70 order-check JSON path.")
    parser.add_argument("--out", required=True, help="Output verdict JSON path.")
    parser.add_argument("--io-attrib-enabled", default="0", help="Whether strict IO attribution is expected (0/1).")
    parser.add_argument("--decision-grade-hint", default="1", help="External hint forcing non-decision-grade when 0.")
    parser.add_argument(
        "--decision-grade-require-rehydrate",
        default=os.environ.get("BENCH_PHASE70_DECISION_GRADE_REQUIRE_REHYDRATE", "1"),
        help=(
            "Require SSD rehydrate evidence for decision-grade verdicts (0/1). "
            "Default from BENCH_PHASE70_DECISION_GRADE_REQUIRE_REHYDRATE, fallback=1."
        ),
    )
    parser.add_argument("--reason-code", action="append", default=[], help="Additional reason code(s) to include.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    summary_json_path = Path(args.summary_json)
    summary_csv_path = Path(args.summary_csv)
    pair_delta_csv_path = Path(args.pair_delta_csv)
    order_check_path = Path(args.order_check_json)
    out_path = Path(args.out)

    reason_codes = {str(code) for code in args.reason_code if str(code).strip()}
    require_rehydrate_raw = parse_bool(args.decision_grade_require_rehydrate)
    if require_rehydrate_raw is None:
        raise SystemExit(
            "Invalid --decision-grade-require-rehydrate value; expected one of: 0/1 true/false yes/no on/off"
        )
    require_rehydrate = bool(require_rehydrate_raw)
    artifacts = {
        "manifest": {"path": str(manifest_path), "exists": manifest_path.exists()},
        "summary_json": {"path": str(summary_json_path), "exists": summary_json_path.exists()},
        "summary_csv": {"path": str(summary_csv_path), "exists": summary_csv_path.exists()},
        "pair_delta_csv": {"path": str(pair_delta_csv_path), "exists": pair_delta_csv_path.exists()},
        "order_check_json": {"path": str(order_check_path), "exists": order_check_path.exists()},
    }

    if not all(item["exists"] for item in artifacts.values()):
        reason_codes.add("RUN_ERRORS_PRESENT")

    manifest: dict[str, Any] = {}
    summary_obj: dict[str, Any] = {}
    order_check: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    if manifest_path.exists():
        manifest = load_json(manifest_path)
    if summary_json_path.exists():
        summary_obj = load_json(summary_json_path)
        loaded_rows = summary_obj.get("rows")
        if isinstance(loaded_rows, list):
            rows = [row for row in loaded_rows if isinstance(row, dict)]
    if not rows:
        rows = parse_csv_rows(summary_csv_path)
    if order_check_path.exists():
        order_check = load_json(order_check_path)

    if summary_obj.get("errors"):
        reason_codes.add("RUN_ERRORS_PRESENT")
    if any(not row_status_ok(row) for row in rows):
        reason_codes.add("RUN_ERRORS_PRESENT")

    meta = manifest.get("meta") if isinstance(manifest.get("meta"), dict) else {}
    observed_pair_count = parse_int(meta.get("pair_count")) or parse_int(summary_obj.get("pair_count"))
    if observed_pair_count is None:
        pair_ids = {
            int(str(row.get("pair_id")))
            for row in rows
            if str(row.get("pair_id") or "").strip().isdigit()
        }
        observed_pair_count = len(pair_ids) if pair_ids else 0

    order_dependent = False
    order_effect_insufficient_pairs = observed_pair_count < MIN_ORDER_EFFECT_PAIRS
    if order_effect_insufficient_pairs:
        reason_codes.add("ORDER_EFFECT_INSUFFICIENT_PAIRS")
    else:
        if isinstance(order_check.get("order_effect_summary"), dict):
            order_dependent = bool(order_check["order_effect_summary"].get("order_dependent"))
        if not order_dependent:
            metrics = order_check.get("metrics") if isinstance(order_check.get("metrics"), dict) else {}
            order_dependent = any(
                isinstance(metric, dict) and metric.get("order_effect_flag") is True for metric in metrics.values()
            )
        if order_dependent:
            reason_codes.add("ORDER_EFFECT_SUSPECT")

    mode_b = str(meta.get("mode_b") or summary_obj.get("mode_b") or "B2")
    b2_rows = [row for row in rows if str(row.get("mode") or "") == "B2"]
    if not b2_rows and mode_b != "B2":
        b2_rows = [row for row in rows if str(row.get("mode") or "") == mode_b]
    io_attrib_expected = parse_bool(args.io_attrib_enabled) is True

    ssd_write_signal_present = False
    ssd_rehydrate_signal_present = False
    ssd_reuse_signal_present = False
    any_mechanism_signal = False
    io_replay_reads_available = False
    mechanism_details: list[dict[str, Any]] = []
    for row in b2_rows:
        row_signals = extract_run_mechanism_signals(str(row.get("run_path") or ""))
        matched = parse_float(row.get("kvbm_matched_tokens_delta"))
        onboard = parse_float(row.get("kvbm_onboard_blocks_delta"))
        if matched is not None:
            row_signals["matched_tokens"] = max(row_signals["matched_tokens"], matched)
        if onboard is not None:
            row_signals["onboard_d2d"] = max(row_signals["onboard_d2d"], onboard)

        replay_read_gib = parse_float(row.get("replay_read_gib"))
        replay_reads_present = (replay_read_gib or 0.0) > 0.0
        if replay_read_gib is not None:
            io_replay_reads_available = True

        row_write_signal = row_signals["offload_h2d"] > 0.0
        row_rehydrate_signal = (
            row_signals["onboard_d2d"] > 0.0
            or row_signals["disk_hit_rate"] > 0.0
            or (replay_reads_present if io_attrib_expected else False)
        )
        row_reuse_signal = row_signals["matched_tokens"] > 0.0

        signal_present = row_write_signal or row_rehydrate_signal or row_reuse_signal
        if signal_present:
            any_mechanism_signal = True
        ssd_write_signal_present = ssd_write_signal_present or row_write_signal
        ssd_rehydrate_signal_present = ssd_rehydrate_signal_present or row_rehydrate_signal
        ssd_reuse_signal_present = ssd_reuse_signal_present or row_reuse_signal
        mechanism_details.append(
            {
                "run_id": row.get("run_id"),
                "mode": row.get("mode"),
                "signals": row_signals,
                "replay_read_gib": replay_read_gib,
                "replay_reads_present": replay_reads_present,
                "ssd_write_signal": row_write_signal,
                "ssd_rehydrate_signal": row_rehydrate_signal,
                "ssd_reuse_signal": row_reuse_signal,
                "signal_present": signal_present,
            }
        )

    if not any_mechanism_signal:
        reason_codes.add("GATE_NO_SSD_MECHANISM_SIGNAL")
    if ssd_write_signal_present and not ssd_rehydrate_signal_present:
        reason_codes.add("REHYDRATE_SIGNAL_ABSENT_WRITE_ONLY")

    io_attrib_ok = True
    if io_attrib_expected:
        checked_rows = [row for row in rows if str(row.get("mode") or "") != "B0"]
        if not checked_rows:
            io_attrib_ok = False
        for row in checked_rows:
            if parse_bool(row.get("io_attrib_pass")) is not True:
                io_attrib_ok = False
                break
    if not io_attrib_ok:
        reason_codes.add("IO_ATTRIB_MISSING")

    run_valid = all(item["exists"] for item in artifacts.values()) and "RUN_ERRORS_PRESENT" not in reason_codes
    decision_grade = parse_bool(args.decision_grade_hint) is not False
    if not run_valid:
        decision_grade = False
    if not any_mechanism_signal:
        decision_grade = False
    if io_attrib_expected and not io_attrib_ok:
        decision_grade = False
    if order_dependent and not order_effect_insufficient_pairs:
        decision_grade = False
    scenario = str(meta.get("scenario") or summary_obj.get("scenario") or "rehydrate_replay").strip().lower()
    is_rehydrate_workload = "rehydrate" in scenario
    if is_rehydrate_workload and require_rehydrate and not ssd_rehydrate_signal_present:
        decision_grade = False

    replay_concurrency = meta.get("replay_concurrency")
    pair_count = observed_pair_count
    ts = parse_ts(manifest_path, manifest)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_valid": run_valid,
        "decision_grade": decision_grade,
        "reason_codes": sorted(reason_codes),
        "meta": {
            "ts": ts,
            "replay_concurrency": replay_concurrency,
            "pairs": pair_count,
            "modes": {
                "mode_a": meta.get("mode_a") or summary_obj.get("mode_a"),
                "mode_b": mode_b,
            },
        },
        "checks": {
            "artifacts_present": all(item["exists"] for item in artifacts.values()),
            "order_effect_suspect": order_dependent,
            "order_effect_insufficient_pairs": order_effect_insufficient_pairs,
            "min_order_effect_pairs": MIN_ORDER_EFFECT_PAIRS,
            "io_attrib_expected": io_attrib_expected,
            "io_attrib_ok": io_attrib_ok,
            "io_replay_reads_available": io_replay_reads_available,
            "ssd_write_signal_present": ssd_write_signal_present,
            "ssd_rehydrate_signal_present": ssd_rehydrate_signal_present,
            "ssd_reuse_signal_present": ssd_reuse_signal_present,
            "decision_grade_require_rehydrate": require_rehydrate,
            "is_rehydrate_workload": is_rehydrate_workload,
            "mechanism_signal_present": any_mechanism_signal,
            "mechanism_details": mechanism_details,
            "summary_rows": len(rows),
            "summary_errors_count": len(summary_obj.get("errors") or []),
        },
        "artifacts": artifacts,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Extract a compact per-concurrency knee review table from a Phase60 summary."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

TS_RE = re.compile(r"(\d{8}T\d{6}Z)")
MODES = ("B1", "B2")


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_ts(text: str) -> str | None:
    match = TS_RE.search(text)
    return match.group(1) if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a Phase60 knee table CSV.")
    parser.add_argument("--phase60-summary-json", required=True, help="Path to phase60 sweep summary JSON.")
    parser.add_argument("--out-csv", default=None, help="Output knee table CSV path.")
    return parser.parse_args()


def status_rank(status: str) -> int:
    normalized = status.strip().lower()
    if normalized == "ok":
        return 6
    if normalized == "baseline_preflight":
        return 5
    if normalized.startswith("invalid"):
        return 4
    if normalized == "partial":
        return 3
    if normalized == "debug_only":
        return 2
    return 1


def resolve_replay_phase(summary_obj: dict[str, Any]) -> dict[str, Any]:
    phases = summary_obj.get("phase_summaries") if isinstance(summary_obj.get("phase_summaries"), list) else []
    for name in ("replay", "replay_A"):
        for phase in phases:
            if isinstance(phase, dict) and str(phase.get("phase")) == name:
                return phase
    for phase in phases:
        if isinstance(phase, dict) and str(phase.get("phase") or "").startswith("replay"):
            return phase
    return {}


def load_run_summary(row: dict[str, Any], cache: dict[str, dict[str, Any] | None]) -> dict[str, Any] | None:
    run_path_raw = row.get("run_path")
    if not run_path_raw:
        return None

    run_path = str(run_path_raw)
    if run_path in cache:
        return cache[run_path]

    summary_path = Path(run_path) / "summary.json"
    if not summary_path.exists():
        cache[run_path] = None
        return None

    try:
        cache[run_path] = load_json(summary_path)
    except Exception:
        cache[run_path] = None
    return cache[run_path]


def nested_stat(row: dict[str, Any], key: str, stat: str) -> float | None:
    payload = row.get(key)
    if isinstance(payload, dict):
        return parse_float(payload.get(stat))
    return None


def metric_with_fallback(
    row: dict[str, Any],
    cache: dict[str, dict[str, Any] | None],
    stat: str,
) -> tuple[float | None, str]:
    for candidate in (
        parse_float(row.get(f"replay_ttfc_{stat}_ms")),
        parse_float(row.get(f"ttfc_{stat}_ms")),
        nested_stat(row, "replay_ttfc_ms", stat),
        nested_stat(row, "ttfc_ms", stat),
    ):
        if candidate is not None:
            return candidate, "ttfc"

    run_summary = load_run_summary(row, cache)
    if isinstance(run_summary, dict):
        replay = resolve_replay_phase(run_summary)
        replay_ttfc = parse_float(((replay.get("ttfc_ms") or {}).get(stat)) if isinstance(replay, dict) else None)
        if replay_ttfc is not None:
            return replay_ttfc, "ttfc"

    for candidate in (
        parse_float(row.get(f"replay_ttft_{stat}_ms")),
        parse_float(row.get(f"ttft_{stat}_ms")),
        nested_stat(row, "replay_ttft_ms", stat),
        nested_stat(row, "ttft_ms", stat),
    ):
        if candidate is not None:
            return candidate, "ttft_fallback"

    if isinstance(run_summary, dict):
        replay = resolve_replay_phase(run_summary)
        replay_ttft = parse_float(((replay.get("ttft_ms") or {}).get(stat)) if isinstance(replay, dict) else None)
        if replay_ttft is not None:
            return replay_ttft, "ttft_fallback"

    return None, "missing"


def extract_error_rate(row: dict[str, Any], cache: dict[str, dict[str, Any] | None]) -> float | None:
    value = parse_float(row.get("error_rate"))
    if value is not None:
        return value

    run_summary = load_run_summary(row, cache)
    if not isinstance(run_summary, dict):
        return None
    overall = run_summary.get("overall_summary") if isinstance(run_summary.get("overall_summary"), dict) else {}
    return parse_float(overall.get("error_rate"))


def replay_disk_hit_rate(row: dict[str, Any], cache: dict[str, dict[str, Any] | None]) -> float:
    for candidate in (
        row.get("kvbm_disk_cache_hit_rate"),
        ((row.get("mechanism") or {}).get("kvbm_disk_cache_hit_rate") if isinstance(row.get("mechanism"), dict) else None),
        row.get("replay_kvbm_disk_cache_hit_rate"),
    ):
        value = parse_float(candidate)
        if value is not None:
            return value

    run_summary = load_run_summary(row, cache)
    if not isinstance(run_summary, dict):
        return 0.0

    replay = resolve_replay_phase(run_summary)
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


def signal_values(row: dict[str, Any], cache: dict[str, dict[str, Any] | None]) -> dict[str, float]:
    mechanism = row.get("mechanism") if isinstance(row.get("mechanism"), dict) else {}

    def first_float(*values: Any) -> float:
        for value in values:
            parsed = parse_float(value)
            if parsed is not None:
                return parsed
        return 0.0

    matched = first_float(
        mechanism.get("kvbm_matched_tokens_delta_replay_plus_replay2"),
        row.get("kvbm_matched_tokens_delta_replay_plus_replay2"),
        row.get("kvbm_matched_tokens_delta"),
    )
    onboard = first_float(
        mechanism.get("kvbm_onboard_blocks_d2d_delta_replay_plus_replay2"),
        row.get("kvbm_onboard_blocks_d2d_delta_replay_plus_replay2"),
        row.get("kvbm_onboard_blocks_delta"),
        row.get("kvbm_onboard_blocks_d2d_delta"),
    )
    offload = first_float(
        mechanism.get("kvbm_offload_blocks_h2d_delta_replay_plus_replay2"),
        row.get("kvbm_offload_blocks_h2d_delta_replay_plus_replay2"),
        row.get("kvbm_offload_blocks_h2d_delta"),
        row.get("kvbm_offload_blocks_delta"),
    )
    block_read = first_float(
        mechanism.get("block_read_bytes_delta_replay_plus_replay2"),
        ((row.get("io_attribution") or {}).get("replay_block_read_bytes") if isinstance(row.get("io_attribution"), dict) else None),
        ((row.get("io_attribution_verdict") or {}).get("replay_read_bytes_block") if isinstance(row.get("io_attribution_verdict"), dict) else None),
    )
    process_read = first_float(
        mechanism.get("cgroup_read_bytes_delta_replay_plus_replay2"),
        ((row.get("io_attribution") or {}).get("replay_process_read_bytes") if isinstance(row.get("io_attribution"), dict) else None),
        ((row.get("io_attribution_verdict") or {}).get("replay_read_bytes_process_total") if isinstance(row.get("io_attribution_verdict"), dict) else None),
    )
    block_write = first_float(
        mechanism.get("block_write_bytes_delta_replay_plus_replay2"),
        mechanism.get("block_write_bytes_delta"),
    )
    process_write = first_float(
        mechanism.get("cgroup_write_bytes_delta_replay_plus_replay2"),
        mechanism.get("cgroup_write_bytes_delta"),
    )
    disk_hit_rate = replay_disk_hit_rate(row, cache)

    return {
        "kvbm_matched_tokens_delta": matched,
        "kvbm_onboard_blocks_d2d_delta": onboard,
        "kvbm_offload_blocks_h2d_delta": offload,
        "kvbm_disk_cache_hit_rate": disk_hit_rate,
        "replay_block_read_bytes": block_read,
        "replay_process_read_bytes": process_read,
        "block_write_bytes": block_write,
        "process_write_bytes": process_write,
    }


def parse_declared_concurrency(summary_obj: dict[str, Any]) -> list[int]:
    meta = summary_obj.get("meta") if isinstance(summary_obj.get("meta"), dict) else {}
    raw = meta.get("sweep_replay_concurrencies")
    if not isinstance(raw, list):
        return []
    values = []
    for item in raw:
        value = parse_int(item)
        if value is not None and value > 0:
            values.append(value)
    return sorted(set(values))


def select_rows(summary_obj: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    rows = summary_obj.get("rows") if isinstance(summary_obj.get("rows"), list) else []
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    selected_rank: dict[tuple[int, str], tuple[int, int]] = {}

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        mode = str(row.get("mode") or "").upper()
        if mode not in MODES:
            continue
        concurrency = parse_int(row.get("replay_concurrency"))
        if concurrency is None:
            concurrency = parse_int(row.get("concurrency"))
        if concurrency is None:
            continue

        key = (concurrency, mode)
        rank = status_rank(str(row.get("status") or ""))
        current = selected_rank.get(key)
        if current is None or (rank, idx) >= current:
            selected[key] = row
            selected_rank[key] = (rank, idx)

    return selected


def main() -> int:
    args = parse_args()
    summary_path = Path(args.phase60_summary_json)
    if not summary_path.exists():
        print(f"Phase60 summary not found: {summary_path}")
        return 2

    summary_obj = load_json(summary_path)
    ts = extract_ts(summary_path.name) or "unknown"
    out_csv = Path(args.out_csv) if args.out_csv else (summary_path.parent / f"phase60_knee_table_{ts}.csv")

    selected = select_rows(summary_obj)
    declared = parse_declared_concurrency(summary_obj)
    found = sorted({concurrency for (concurrency, _) in selected.keys()})
    concurrencies = sorted(set(declared) | set(found))

    cache: dict[str, dict[str, Any] | None] = {}
    out_rows: list[dict[str, Any]] = []

    for concurrency in concurrencies:
        for mode in MODES:
            row = selected.get((concurrency, mode))
            notes: list[str] = []

            if row is None:
                out_rows.append(
                    {
                        "concurrency": concurrency,
                        "mode": mode,
                        "status": "missing",
                        "point_key": f"{mode}_c{concurrency}",
                        "ttfc_p95_ms": None,
                        "ttfc_p99_ms": None,
                        "ttft_p95_ms": None,
                        "ttft_p99_ms": None,
                        "metric_source_p95": "missing",
                        "metric_source_p99": "missing",
                        "error_rate": None,
                        "ssd_write_signal_present": False,
                        "ssd_rehydrate_signal_present": False,
                        "b1_disk_tier_verified": None,
                        "kvbm_matched_tokens_delta": 0.0,
                        "kvbm_onboard_blocks_d2d_delta": 0.0,
                        "kvbm_offload_blocks_h2d_delta": 0.0,
                        "kvbm_disk_cache_hit_rate": 0.0,
                        "replay_block_read_bytes": 0.0,
                        "replay_process_read_bytes": 0.0,
                        "run_path": None,
                        "notes": "missing_mode_row",
                    }
                )
                continue

            status = str(row.get("status") or "")
            if status and status != "ok":
                notes.append(f"status:{status}")

            ttfc_p95, source_p95 = metric_with_fallback(row, cache, "p95")
            ttfc_p99, source_p99 = metric_with_fallback(row, cache, "p99")
            if source_p95 == "ttft_fallback" or source_p99 == "ttft_fallback":
                notes.append("ttfc_missing_fallback_ttft")
            if source_p95 == "missing":
                notes.append("missing_p95")
            if source_p99 == "missing":
                notes.append("missing_p99")

            b1_disk_tier_verified = parse_bool(row.get("b1_disk_tier_verified")) if mode == "B1" else None
            if mode == "B1":
                if b1_disk_tier_verified is False:
                    notes.append("b1_disk_tier_not_verified")
                elif b1_disk_tier_verified is None:
                    notes.append("b1_disk_tier_verification_missing")

            ttft_p95 = parse_float(row.get("replay_ttft_p95_ms"))
            if ttft_p95 is None:
                ttft_p95 = nested_stat(row, "replay_ttft_ms", "p95")
            ttft_p99 = parse_float(row.get("replay_ttft_p99_ms"))
            if ttft_p99 is None:
                ttft_p99 = nested_stat(row, "replay_ttft_ms", "p99")

            error_rate = extract_error_rate(row, cache)
            if error_rate is None:
                notes.append("missing_error_rate")

            signals = signal_values(row, cache)
            write_signal = (
                signals["kvbm_offload_blocks_h2d_delta"] > 0.0
                or signals["block_write_bytes"] > 0.0
                or signals["process_write_bytes"] > 0.0
            )
            rehydrate_signal = (
                signals["kvbm_onboard_blocks_d2d_delta"] > 0.0
                or signals["kvbm_disk_cache_hit_rate"] > 0.0
                or signals["replay_block_read_bytes"] > 0.0
                or signals["replay_process_read_bytes"] > 0.0
            )

            out_rows.append(
                {
                    "concurrency": concurrency,
                    "mode": mode,
                    "status": status,
                    "point_key": row.get("point_key"),
                    "ttfc_p95_ms": ttfc_p95,
                    "ttfc_p99_ms": ttfc_p99,
                    "ttft_p95_ms": ttft_p95,
                    "ttft_p99_ms": ttft_p99,
                    "metric_source_p95": source_p95,
                    "metric_source_p99": source_p99,
                    "error_rate": error_rate,
                    "ssd_write_signal_present": bool(write_signal),
                    "ssd_rehydrate_signal_present": bool(rehydrate_signal),
                    "b1_disk_tier_verified": b1_disk_tier_verified,
                    "kvbm_matched_tokens_delta": signals["kvbm_matched_tokens_delta"],
                    "kvbm_onboard_blocks_d2d_delta": signals["kvbm_onboard_blocks_d2d_delta"],
                    "kvbm_offload_blocks_h2d_delta": signals["kvbm_offload_blocks_h2d_delta"],
                    "kvbm_disk_cache_hit_rate": signals["kvbm_disk_cache_hit_rate"],
                    "replay_block_read_bytes": signals["replay_block_read_bytes"],
                    "replay_process_read_bytes": signals["replay_process_read_bytes"],
                    "run_path": row.get("run_path"),
                    "notes": ";".join(notes),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "concurrency",
        "mode",
        "status",
        "point_key",
        "ttfc_p95_ms",
        "ttfc_p99_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "metric_source_p95",
        "metric_source_p99",
        "error_rate",
        "ssd_write_signal_present",
        "ssd_rehydrate_signal_present",
        "b1_disk_tier_verified",
        "kvbm_matched_tokens_delta",
        "kvbm_onboard_blocks_d2d_delta",
        "kvbm_offload_blocks_h2d_delta",
        "kvbm_disk_cache_hit_rate",
        "replay_block_read_bytes",
        "replay_process_read_bytes",
        "run_path",
        "notes",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for item in out_rows:
            writer.writerow(item)

    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Recommend Phase70 replay concurrency from a Phase60 fixed-pressure sweep summary."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TS_RE = re.compile(r"(\d{8}T\d{6}Z)")
VALID_MODES = {"B1", "B2"}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def parse_bool_like(value: Any) -> bool | None:
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def extract_ts(text: str) -> str | None:
    match = TS_RE.search(text)
    return match.group(1) if match else None


def to_compact_ts(iso_or_ts: str | None) -> str | None:
    if not iso_or_ts:
        return None
    ts = extract_ts(iso_or_ts)
    if ts:
        return ts
    try:
        parsed = datetime.fromisoformat(iso_or_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_summary_path(args: argparse.Namespace) -> Path:
    if args.phase60_summary_json:
        path = Path(args.phase60_summary_json)
        if not path.exists():
            raise FileNotFoundError(f"Phase60 summary JSON not found: {path}")
        return path

    if not args.ts:
        raise ValueError("Provide either --phase60-summary-json or --ts")

    results_root = Path(args.results_root)
    exact = results_root / f"phase60_rehydrate_concurrency_sweep_summary_minimal_{args.ts}.json"
    if exact.exists():
        return exact

    patterns = [
        f"phase60*_concurrency_sweep_summary*_{args.ts}.json",
        f"phase60*summary*_{args.ts}.json",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(results_root.glob(pattern))

    unique_matches = sorted({path.resolve() for path in matches})
    if not unique_matches:
        raise FileNotFoundError(
            f"No Phase60 summary found for ts={args.ts} under {results_root}; "
            f"expected {exact.name} or a matching phase60*_summary*_{args.ts}.json"
        )
    return unique_matches[0]


def resolve_phase60_ts(summary_path: Path, summary_obj: dict[str, Any], args_ts: str | None) -> str:
    if args_ts:
        return args_ts
    ts = extract_ts(summary_path.name)
    if ts:
        return ts

    meta = summary_obj.get("meta") if isinstance(summary_obj.get("meta"), dict) else {}
    for key in ("created_utc", "resumed_utc"):
        compact = to_compact_ts(meta.get(key))
        if compact:
            return compact
    return now_utc_ts()


def resolve_replay_phase(summary_obj: dict[str, Any]) -> dict[str, Any]:
    phases = summary_obj.get("phase_summaries") if isinstance(summary_obj.get("phase_summaries"), list) else []
    for wanted in ("replay", "replay_A"):
        for phase in phases:
            if isinstance(phase, dict) and str(phase.get("phase")) == wanted:
                return phase
    for phase in phases:
        if isinstance(phase, dict) and str(phase.get("phase") or "").startswith("replay"):
            return phase
    return {}


def status_rank(status: str) -> int:
    normalized = status.strip().lower()
    if normalized == "ok":
        return 4
    if normalized.startswith("invalid"):
        return 3
    if normalized == "partial":
        return 2
    if normalized in {"baseline_preflight", "debug_only"}:
        return 1
    return 1


def is_sweep_row(row: dict[str, Any], mode: str, concurrency: int) -> bool:
    phase = str(row.get("phase") or "").strip().lower()
    if phase and phase != "sweep_point":
        return False

    point_key = str(row.get("point_key") or "").strip()
    if point_key:
        return point_key == f"{mode}_c{concurrency}"
    return True


def collect_mode_rows(summary_obj: dict[str, Any]) -> dict[int, dict[str, dict[str, Any]]]:
    rows = summary_obj.get("rows") if isinstance(summary_obj.get("rows"), list) else []
    out: dict[int, dict[str, dict[str, Any]]] = {}
    rank_cache: dict[tuple[int, str], tuple[int, int]] = {}

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        mode = str(row.get("mode") or "").upper()
        if mode not in VALID_MODES:
            continue
        concurrency = parse_int(row.get("replay_concurrency"))
        if concurrency is None:
            concurrency = parse_int(row.get("concurrency"))
        if concurrency is None:
            continue
        if not is_sweep_row(row, mode, concurrency):
            continue

        status = str(row.get("status") or "")
        rank = status_rank(status)
        key = (concurrency, mode)
        current_rank = rank_cache.get(key)
        if current_rank is None or (rank, idx) >= current_rank:
            out.setdefault(concurrency, {})[mode] = row
            rank_cache[key] = (rank, idx)

    return out


def get_nested_p95(row: dict[str, Any], key: str) -> float | None:
    nested = row.get(key)
    if isinstance(nested, dict):
        return parse_float(nested.get("p95"))
    return None


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


def extract_from_run_summary(
    row: dict[str, Any],
    cache: dict[str, dict[str, Any] | None],
    metric: str,
) -> float | None:
    summary_obj = load_run_summary(row, cache)
    if not isinstance(summary_obj, dict):
        return None

    replay = resolve_replay_phase(summary_obj)
    overall = summary_obj.get("overall_summary") if isinstance(summary_obj.get("overall_summary"), dict) else {}

    if metric == "ttfc_p95_ms":
        replay_val = parse_float(((replay.get("ttfc_ms") or {}).get("p95")) if isinstance(replay, dict) else None)
        if replay_val is not None:
            return replay_val
        return parse_float((overall.get("ttfc_ms") or {}).get("p95"))

    if metric == "ttft_p95_ms":
        replay_val = parse_float(((replay.get("ttft_ms") or {}).get("p95")) if isinstance(replay, dict) else None)
        if replay_val is not None:
            return replay_val
        return parse_float((overall.get("ttft_ms") or {}).get("p95"))

    return None


def extract_metric_p95(
    row: dict[str, Any],
    cache: dict[str, dict[str, Any] | None],
    requested_metric: str,
) -> tuple[float | None, str | None, str | None]:
    def read_row_metric(metric: str) -> float | None:
        if metric == "ttfc_p95_ms":
            for candidate in (
                parse_float(row.get("replay_ttfc_p95_ms")),
                parse_float(row.get("ttfc_p95_ms")),
                get_nested_p95(row, "replay_ttfc_ms"),
                get_nested_p95(row, "ttfc_ms"),
            ):
                if candidate is not None:
                    return candidate
            return None

        for candidate in (
            parse_float(row.get("replay_ttft_p95_ms")),
            parse_float(row.get("ttft_p95_ms")),
            get_nested_p95(row, "replay_ttft_ms"),
            get_nested_p95(row, "ttft_ms"),
        ):
            if candidate is not None:
                return candidate
        return None

    preferred = requested_metric
    val = read_row_metric(preferred)
    if val is not None:
        return val, preferred, "row"

    run_val = extract_from_run_summary(row, cache, preferred)
    if run_val is not None:
        return run_val, preferred, "run_summary"

    if requested_metric == "ttfc_p95_ms":
        fallback = "ttft_p95_ms"
        fallback_val = read_row_metric(fallback)
        if fallback_val is not None:
            return fallback_val, fallback, "row_fallback"
        fallback_run_val = extract_from_run_summary(row, cache, fallback)
        if fallback_run_val is not None:
            return fallback_run_val, fallback, "run_summary_fallback"

    return None, None, None


def extract_error_rate(row: dict[str, Any], cache: dict[str, dict[str, Any] | None]) -> float | None:
    value = parse_float(row.get("error_rate"))
    if value is not None:
        return value

    summary_obj = load_run_summary(row, cache)
    if not isinstance(summary_obj, dict):
        return None
    overall = summary_obj.get("overall_summary") if isinstance(summary_obj.get("overall_summary"), dict) else {}
    return parse_float(overall.get("error_rate"))


def replay_disk_hit_rate_from_phase(replay: dict[str, Any]) -> float:
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


def extract_rehydrate_signals(
    row: dict[str, Any],
    cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    mechanism = row.get("mechanism") if isinstance(row.get("mechanism"), dict) else {}

    def first_float(values: list[Any]) -> float:
        for value in values:
            parsed = parse_float(value)
            if parsed is not None:
                return parsed
        return 0.0

    matched_tokens = first_float(
        [
            mechanism.get("kvbm_matched_tokens_delta_replay_plus_replay2"),
            row.get("kvbm_matched_tokens_delta_replay_plus_replay2"),
            row.get("kvbm_matched_tokens_delta"),
        ]
    )
    onboard_d2d = first_float(
        [
            mechanism.get("kvbm_onboard_blocks_d2d_delta_replay_plus_replay2"),
            row.get("kvbm_onboard_blocks_d2d_delta_replay_plus_replay2"),
            row.get("kvbm_onboard_blocks_delta"),
            row.get("kvbm_onboard_blocks_d2d_delta"),
        ]
    )
    offload_h2d = first_float(
        [
            mechanism.get("kvbm_offload_blocks_h2d_delta_replay_plus_replay2"),
            row.get("kvbm_offload_blocks_h2d_delta_replay_plus_replay2"),
            row.get("kvbm_offload_blocks_h2d_delta"),
            row.get("kvbm_offload_blocks_delta"),
        ]
    )

    disk_hit_rate = first_float(
        [
            row.get("kvbm_disk_cache_hit_rate"),
            mechanism.get("kvbm_disk_cache_hit_rate"),
            row.get("replay_kvbm_disk_cache_hit_rate"),
        ]
    )

    io_attr = row.get("io_attribution") if isinstance(row.get("io_attribution"), dict) else {}
    io_verdict = row.get("io_attribution_verdict") if isinstance(row.get("io_attribution_verdict"), dict) else {}

    io_attr_available = bool(io_attr.get("available")) or bool(io_verdict.get("available"))
    replay_block_read = first_float(
        [
            io_attr.get("replay_block_read_bytes"),
            io_verdict.get("replay_read_bytes_block"),
            mechanism.get("block_read_bytes_delta_replay_plus_replay2"),
        ]
    )
    replay_process_read = first_float(
        [
            io_attr.get("replay_process_read_bytes"),
            io_verdict.get("replay_read_bytes_process_total"),
            mechanism.get("cgroup_read_bytes_delta_replay_plus_replay2"),
        ]
    )

    if disk_hit_rate <= 0.0:
        run_summary = load_run_summary(row, cache)
        if isinstance(run_summary, dict):
            replay = resolve_replay_phase(run_summary)
            if replay:
                disk_hit_rate = replay_disk_hit_rate_from_phase(replay)
                kv_delta = replay.get("kvbm_metrics_delta") if isinstance(replay.get("kvbm_metrics_delta"), dict) else {}
                if matched_tokens <= 0.0:
                    matched_tokens = first_float([kv_delta.get("kvbm_matched_tokens_delta"), kv_delta.get("matched_tokens_total_delta")])
                if onboard_d2d <= 0.0:
                    onboard_d2d = first_float([kv_delta.get("kvbm_onboard_blocks_d2d_delta"), kv_delta.get("onboard_blocks_total_delta")])
                if offload_h2d <= 0.0:
                    offload_h2d = first_float([kv_delta.get("kvbm_offload_blocks_h2d_delta"), kv_delta.get("offload_blocks_total_delta")])

    io_replay_read_signal = io_attr_available and ((replay_block_read > 0.0) or (replay_process_read > 0.0))
    rehydrate_signal = (onboard_d2d > 0.0) or (disk_hit_rate > 0.0) or io_replay_read_signal

    return {
        "matched_tokens_delta": matched_tokens,
        "onboard_blocks_d2d_delta": onboard_d2d,
        "offload_blocks_h2d_delta": offload_h2d,
        "disk_cache_hit_rate": disk_hit_rate,
        "io_attribution_available": io_attr_available,
        "replay_block_read_bytes": replay_block_read,
        "replay_process_read_bytes": replay_process_read,
        "rehydrate_signal": bool(rehydrate_signal),
    }


def row_marked_invalid(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").strip().lower()
    if status and status != "ok":
        return True

    for key in ("valid", "run_valid"):
        if key in row:
            parsed = parse_bool_like(row.get(key))
            if parsed is False:
                return True

    invalid_raw = row.get("invalid")
    parsed_invalid = parse_bool_like(invalid_raw)
    return parsed_invalid is True


def parse_declared_concurrencies(summary_obj: dict[str, Any]) -> list[int]:
    meta = summary_obj.get("meta") if isinstance(summary_obj.get("meta"), dict) else {}
    declared = meta.get("sweep_replay_concurrencies")
    if not isinstance(declared, list):
        return []

    out: list[int] = []
    for item in declared:
        value = parse_int(item)
        if value is not None and value > 0:
            out.append(value)
    return sorted(set(out))


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def evaluate_candidates(
    summary_obj: dict[str, Any],
    *,
    requested_metric: str,
    slo_p95_ms: float | None,
    require_b2_rehydrate: bool,
    top_k: int,
    weights: dict[str, float],
    min_b1_degrade_ratio: float,
    min_tail_gap_ratio: float,
) -> tuple[dict[str, Any], bool]:
    mode_rows = collect_mode_rows(summary_obj)
    declared = parse_declared_concurrencies(summary_obj)
    all_concurrency_values = sorted(set(declared) | set(mode_rows.keys()))

    cache: dict[str, dict[str, Any] | None] = {}
    candidates: list[dict[str, Any]] = []
    all_points: list[dict[str, Any]] = []

    rejected_for_errors: set[int] = set()
    rejected_for_no_b2_rehydrate: set[int] = set()
    rejected_for_no_b1_degradation: set[int] = set()
    rejected_for_missing_artifacts: set[int] = set()
    rejected_for_invalid_rows: set[int] = set()
    rejected_for_missing_metric: set[int] = set()
    rejected_for_b1_disk_tier: set[int] = set()

    metric_used_values: set[str] = set()
    metric_fallback_used = False
    warnings: list[str] = []
    b1_verification_missing_detected = False

    prior_valid_b1_metrics: list[float] = []

    for concurrency in all_concurrency_values:
        row_by_mode = mode_rows.get(concurrency) or {}
        b1_row = row_by_mode.get("B1")
        b2_row = row_by_mode.get("B2")
        rejected_reasons: list[str] = []

        if not b1_row or not b2_row:
            rejected_reasons.append("MISSING_REQUIRED_MODE_ROW")
            rejected_for_missing_artifacts.add(concurrency)
            all_points.append(
                {
                    "replay_concurrency": concurrency,
                    "accepted": False,
                    "rejected_reasons": rejected_reasons,
                }
            )
            continue

        b1_invalid = row_marked_invalid(b1_row)
        b2_invalid = row_marked_invalid(b2_row)
        if b1_invalid or b2_invalid:
            rejected_reasons.append("ROW_MARKED_INVALID")
            rejected_for_invalid_rows.add(concurrency)

        b1_disk_tier_verified = parse_bool_like(b1_row.get("b1_disk_tier_verified"))
        if b1_disk_tier_verified is False:
            rejected_reasons.append("B1_DISK_TIER_NOT_DISABLED")
            rejected_for_invalid_rows.add(concurrency)
            rejected_for_b1_disk_tier.add(concurrency)
        elif b1_disk_tier_verified is None:
            b1_verification_missing_detected = True

        b1_metric, b1_metric_used, _ = extract_metric_p95(b1_row, cache, requested_metric)
        b2_metric, b2_metric_used, _ = extract_metric_p95(b2_row, cache, requested_metric)

        if b1_metric_used:
            metric_used_values.add(b1_metric_used)
            metric_fallback_used = metric_fallback_used or (b1_metric_used != requested_metric)
        if b2_metric_used:
            metric_used_values.add(b2_metric_used)
            metric_fallback_used = metric_fallback_used or (b2_metric_used != requested_metric)

        if b1_metric is None or b2_metric is None:
            rejected_reasons.append("MISSING_METRIC_P95")
            rejected_for_missing_metric.add(concurrency)

        b1_error = extract_error_rate(b1_row, cache)
        b2_error = extract_error_rate(b2_row, cache)
        if b1_error is None or b2_error is None:
            rejected_reasons.append("MISSING_ERROR_RATE")
            rejected_for_missing_artifacts.add(concurrency)
        elif b1_error > 0.0 or b2_error > 0.0:
            rejected_reasons.append("NON_ZERO_ERROR_RATE")
            rejected_for_errors.add(concurrency)

        b2_signals = extract_rehydrate_signals(b2_row, cache)
        b1_signals = extract_rehydrate_signals(b1_row, cache)
        b2_rehydrate = bool(b2_signals.get("rehydrate_signal"))

        if require_b2_rehydrate and not b2_rehydrate:
            rejected_reasons.append("NO_B2_REHYDRATE_SIGNAL")
            rejected_for_no_b2_rehydrate.add(concurrency)

        b1_above_slo = bool(slo_p95_ms is not None and b1_metric is not None and b1_metric > slo_p95_ms)
        b2_within_slo = bool(slo_p95_ms is not None and b2_metric is not None and b2_metric <= slo_p95_ms)

        prior_min = min(prior_valid_b1_metrics) if prior_valid_b1_metrics else None
        sharp_b1_degrade = bool(
            prior_min is not None
            and b1_metric is not None
            and b1_metric >= prior_min * (1.0 + max(0.0, min_b1_degrade_ratio))
        )

        tail_gap_ratio = 0.0
        if b1_metric is not None and b2_metric is not None and b1_metric > 0:
            tail_gap_ratio = max(0.0, (b1_metric - b2_metric) / b1_metric)

        material_tail_gap = tail_gap_ratio >= max(0.0, min_tail_gap_ratio)
        b1_degradation_or_separation = b1_above_slo or sharp_b1_degrade or material_tail_gap
        if not b1_degradation_or_separation:
            rejected_reasons.append("NO_B1_DEGRADATION_OR_SEPARATION")
            rejected_for_no_b1_degradation.add(concurrency)

        if not rejected_reasons:
            slo_factor = 0.0
            if slo_p95_ms is not None and b1_metric is not None and b2_metric is not None:
                if b1_metric > slo_p95_ms and b2_metric <= slo_p95_ms:
                    slo_factor = 1.0
                elif b1_metric > slo_p95_ms and b2_metric > slo_p95_ms and b2_metric < b1_metric:
                    slo_factor = 0.5
                elif b1_metric <= slo_p95_ms and b2_metric <= slo_p95_ms and b2_metric < b1_metric:
                    slo_factor = 0.35

            tail_factor = clamp01(tail_gap_ratio / 0.30)
            rehydrate_factor = 1.0 if b2_rehydrate else 0.0
            headroom_factor = 0.0
            if slo_p95_ms is not None and b2_metric is not None and slo_p95_ms > 0 and b2_metric <= slo_p95_ms:
                headroom_ratio = (slo_p95_ms - b2_metric) / slo_p95_ms
                headroom_factor = clamp01(headroom_ratio / 0.20)

            score_components = {
                "slo_advantage": round(weights["slo_advantage"] * slo_factor, 6),
                "tail_gap": round(weights["tail_gap"] * tail_factor, 6),
                "b2_rehydrate": round(weights["b2_rehydrate"] * rehydrate_factor, 6),
                "headroom_safety": round(weights["headroom_safety"] * headroom_factor, 6),
            }
            score = round(sum(score_components.values()), 6)

            reasons: list[str] = []
            if b1_above_slo and b2_within_slo:
                reasons.append("MAX_STABLE_B1_B2_SEPARATION")
            elif material_tail_gap:
                reasons.append("TAIL_GAP_ADVANTAGE")

            if b2_rehydrate:
                reasons.append("B2_REHYDRATE_SIGNAL_PRESENT")
            if b1_error == 0.0 and b2_error == 0.0:
                reasons.append("ZERO_ERROR_RATE")
            if b1_metric_used != requested_metric or b2_metric_used != requested_metric:
                reasons.append("METRIC_FALLBACK_TTFT_P95")

            candidate = {
                "replay_concurrency": concurrency,
                "score": score,
                "reasons": reasons,
                "evidence": {
                    "b1": {
                        "metric_p95_ms": b1_metric,
                        "metric_used": b1_metric_used,
                        "error_rate": b1_error,
                        "rehydrate_signal": bool(b1_signals.get("rehydrate_signal")),
                        "b1_disk_tier_verified": b1_disk_tier_verified,
                    },
                    "b2": {
                        "metric_p95_ms": b2_metric,
                        "metric_used": b2_metric_used,
                        "error_rate": b2_error,
                        "rehydrate_signal": b2_rehydrate,
                        "kvbm_matched_tokens_delta": b2_signals.get("matched_tokens_delta"),
                        "kvbm_onboard_blocks_d2d_delta": b2_signals.get("onboard_blocks_d2d_delta"),
                        "kvbm_offload_blocks_h2d_delta": b2_signals.get("offload_blocks_h2d_delta"),
                        "kvbm_disk_cache_hit_rate": b2_signals.get("disk_cache_hit_rate"),
                        "replay_block_read_bytes": b2_signals.get("replay_block_read_bytes"),
                        "replay_process_read_bytes": b2_signals.get("replay_process_read_bytes"),
                    },
                    "slo_p95_ms": slo_p95_ms,
                    "tail_gap_ratio": round(tail_gap_ratio, 6),
                    "b1_above_slo": b1_above_slo,
                    "b2_within_slo": b2_within_slo,
                    "sharp_b1_degrade": sharp_b1_degrade,
                    "score_components": score_components,
                },
            }

            if b1_metric_used == "ttfc_p95_ms":
                candidate["evidence"]["b1"]["ttfc_p95_ms"] = b1_metric
            if b1_metric_used == "ttft_p95_ms":
                candidate["evidence"]["b1"]["ttft_p95_ms"] = b1_metric
            if b2_metric_used == "ttfc_p95_ms":
                candidate["evidence"]["b2"]["ttfc_p95_ms"] = b2_metric
            if b2_metric_used == "ttft_p95_ms":
                candidate["evidence"]["b2"]["ttft_p95_ms"] = b2_metric

            candidates.append(candidate)
            all_points.append(
                {
                    "replay_concurrency": concurrency,
                    "accepted": True,
                    "score": score,
                    "reasons": reasons,
                }
            )
        else:
            all_points.append(
                {
                    "replay_concurrency": concurrency,
                    "accepted": False,
                    "rejected_reasons": sorted(set(rejected_reasons)),
                    "evidence": {
                        "b1_metric_p95_ms": b1_metric,
                        "b2_metric_p95_ms": b2_metric,
                        "b1_error_rate": b1_error,
                        "b2_error_rate": b2_error,
                        "b2_rehydrate_signal": b2_rehydrate,
                        "b1_disk_tier_verified": b1_disk_tier_verified,
                    },
                }
            )

        if (
            b1_metric is not None
            and not b1_invalid
            and b1_error is not None
            and b1_error == 0.0
        ):
            prior_valid_b1_metrics.append(b1_metric)

    candidates.sort(key=lambda item: (-float(item.get("score") or 0.0), int(item.get("replay_concurrency") or 0)))
    recommended = candidates[:top_k]

    if not all_concurrency_values:
        warnings.append("no_sweep_rows_found")
    if requested_metric == "ttfc_p95_ms" and metric_fallback_used:
        warnings.append("ttfc_missing_fell_back_to_ttft")
    if requested_metric == "ttfc_p95_ms" and metric_used_values and metric_used_values == {"ttft_p95_ms"}:
        warnings.append("ttfc_absent_all_rows_using_ttft")
    if b1_verification_missing_detected:
        warnings.append("b1_disk_tier_verification_missing_on_some_rows")

    payload = {
        "recommended": recommended,
        "candidates": candidates,
        "all_points": all_points,
        "rejection_summary": {
            "rejected_for_errors": sorted(rejected_for_errors),
            "rejected_for_no_b2_rehydrate": sorted(rejected_for_no_b2_rehydrate),
            "rejected_for_no_b1_degradation": sorted(rejected_for_no_b1_degradation),
            "rejected_for_invalid_rows": sorted(rejected_for_invalid_rows),
            "rejected_for_b1_disk_tier_not_disabled": sorted(rejected_for_b1_disk_tier),
            "rejected_for_missing_artifacts": sorted(rejected_for_missing_artifacts),
            "rejected_for_missing_metric": sorted(rejected_for_missing_metric),
        },
        "warnings": warnings,
    }

    metrics_missing_globally = bool(all_concurrency_values) and len(rejected_for_missing_metric) == len(all_concurrency_values)
    return payload, metrics_missing_globally


def render_console_summary(payload: dict[str, Any], *, out_json: Path, out_md: Path | None) -> None:
    recommended = payload.get("recommended") if isinstance(payload.get("recommended"), list) else []
    rejection_summary = payload.get("rejection_summary") if isinstance(payload.get("rejection_summary"), dict) else {}

    print("Phase60 crossover recommendation:")
    if recommended:
        best = recommended[0]
        reasons = ",".join(best.get("reasons") or [])
        print(
            f"- recommend replay_concurrency={best.get('replay_concurrency')} "
            f"score={best.get('score')} reasons={reasons}"
        )
        if len(recommended) > 1:
            rest = ", ".join(
                f"c={item.get('replay_concurrency')} score={item.get('score')}"
                for item in recommended[1:]
            )
            print(f"- alternate candidates: {rest}")
    else:
        print("- no eligible replay concurrency found")

    print(
        "- rejected: "
        f"errors={rejection_summary.get('rejected_for_errors', [])} "
        f"no_b2_rehydrate={rejection_summary.get('rejected_for_no_b2_rehydrate', [])} "
        f"no_b1_degradation={rejection_summary.get('rejected_for_no_b1_degradation', [])} "
        f"b1_disk_tier_not_disabled={rejection_summary.get('rejected_for_b1_disk_tier_not_disabled', [])}"
    )

    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if warnings:
        print(f"- warnings: {', '.join(str(item) for item in warnings)}")

    print(f"- wrote json: {out_json}")
    if out_md is not None:
        print(f"- wrote markdown: {out_md}")


def write_markdown_report(path: Path, payload: dict[str, Any], summary_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recommended = payload.get("recommended") if isinstance(payload.get("recommended"), list) else []
    rejection_summary = payload.get("rejection_summary") if isinstance(payload.get("rejection_summary"), dict) else {}

    lines: list[str] = []
    lines.append("# Phase60 Crossover Recommendation")
    lines.append("")
    lines.append(f"Source summary: `{summary_path}`")
    lines.append("")

    if recommended:
        lines.append("## Recommended")
        lines.append("")
        lines.append("| Replay Concurrency | Score | Reasons |")
        lines.append("|---|---:|---|")
        for item in recommended:
            reasons = ", ".join(item.get("reasons") or [])
            lines.append(f"| {item.get('replay_concurrency')} | {item.get('score')} | {reasons} |")
        lines.append("")
    else:
        lines.append("## Recommended")
        lines.append("")
        lines.append("No eligible replay concurrency found.")
        lines.append("")

    lines.append("## Rejections")
    lines.append("")
    lines.append(f"- Non-zero errors: {rejection_summary.get('rejected_for_errors', [])}")
    lines.append(f"- Missing B2 rehydrate signal: {rejection_summary.get('rejected_for_no_b2_rehydrate', [])}")
    lines.append(f"- No B1 degradation/separation: {rejection_summary.get('rejected_for_no_b1_degradation', [])}")
    lines.append(f"- Invalid rows: {rejection_summary.get('rejected_for_invalid_rows', [])}")
    lines.append(f"- B1 disk tier not disabled: {rejection_summary.get('rejected_for_b1_disk_tier_not_disabled', [])}")
    lines.append(f"- Missing artifacts: {rejection_summary.get('rejected_for_missing_artifacts', [])}")
    lines.append(f"- Missing metric p95: {rejection_summary.get('rejected_for_missing_metric', [])}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend Phase70 replay concurrency from Phase60 sweep output.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--phase60-summary-json", default=None, help="Path to phase60 sweep summary JSON.")
    input_group.add_argument("--ts", default=None, help="Timestamp token (YYYYMMDDTHHMMSSZ) for convenience lookup.")

    parser.add_argument("--results-root", default="bench/results", help="Results root used with --ts and default output paths.")
    parser.add_argument("--out-json", default=None, help="Output recommendation JSON path.")
    parser.add_argument("--out-md", default=None, help="Optional output markdown report path.")

    parser.add_argument("--metric", choices=["ttfc_p95_ms", "ttft_p95_ms"], default="ttfc_p95_ms")
    parser.add_argument("--slo-p95-ms", type=float, default=None, help="Optional p95 SLO override in ms.")
    parser.add_argument(
        "--require-b2-rehydrate",
        default="1",
        help="Require B2 rehydrate signal (0/1 true/false). Default: 1",
    )
    parser.add_argument("--top-k", type=int, choices=[1, 2, 3], default=1)

    parser.add_argument("--weight-slo-advantage", type=float, default=0.4)
    parser.add_argument("--weight-tail-gap", type=float, default=0.3)
    parser.add_argument("--weight-b2-rehydrate", type=float, default=0.2)
    parser.add_argument("--weight-headroom-safety", type=float, default=0.1)

    parser.add_argument("--min-b1-degrade-ratio", type=float, default=0.15)
    parser.add_argument("--min-tail-gap-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    require_rehydrate = parse_bool_like(args.require_b2_rehydrate)
    if require_rehydrate is None:
        print("Invalid --require-b2-rehydrate value; expected one of: 0/1 true/false yes/no on/off")
        return 2

    try:
        summary_path = resolve_summary_path(args)
    except Exception as exc:
        print(f"Failed to resolve Phase60 summary: {exc}")
        return 2

    try:
        summary_obj = load_json(summary_path)
    except Exception as exc:
        print(f"Failed to parse Phase60 summary JSON at {summary_path}: {exc}")
        return 2

    phase60_ts = resolve_phase60_ts(summary_path, summary_obj, args.ts)

    default_out_json = summary_path.parent / f"phase60_crossover_recommendation_{phase60_ts}.json"
    out_json = Path(args.out_json) if args.out_json else default_out_json

    out_md: Path | None = None
    if args.out_md:
        out_md = Path(args.out_md)

    summary_slo = parse_float(summary_obj.get("slo_replay_p95_ms"))
    slo_p95_ms = parse_float(args.slo_p95_ms)
    if slo_p95_ms is None:
        slo_p95_ms = summary_slo

    weights = {
        "slo_advantage": max(0.0, float(args.weight_slo_advantage)),
        "tail_gap": max(0.0, float(args.weight_tail_gap)),
        "b2_rehydrate": max(0.0, float(args.weight_b2_rehydrate)),
        "headroom_safety": max(0.0, float(args.weight_headroom_safety)),
    }

    scored_payload, metrics_missing_globally = evaluate_candidates(
        summary_obj,
        requested_metric=args.metric,
        slo_p95_ms=slo_p95_ms,
        require_b2_rehydrate=bool(require_rehydrate),
        top_k=args.top_k,
        weights=weights,
        min_b1_degrade_ratio=max(0.0, float(args.min_b1_degrade_ratio)),
        min_tail_gap_ratio=max(0.0, float(args.min_tail_gap_ratio)),
    )

    final_payload = {
        "created_utc": now_utc_iso(),
        "phase60_ts": phase60_ts,
        "source_summary_json": str(summary_path),
        "recommended": scored_payload.get("recommended"),
        "candidates": scored_payload.get("candidates"),
        "rejection_summary": scored_payload.get("rejection_summary"),
        "meta": {
            "slo_p95_ms": slo_p95_ms,
            "metric": args.metric,
            "min_rehydrate_required": bool(require_rehydrate),
            "top_k": args.top_k,
            "weights": weights,
            "min_b1_degrade_ratio": max(0.0, float(args.min_b1_degrade_ratio)),
            "min_tail_gap_ratio": max(0.0, float(args.min_tail_gap_ratio)),
        },
        "warnings": scored_payload.get("warnings"),
        "all_points": scored_payload.get("all_points"),
    }

    write_json(out_json, final_payload)

    if out_md is not None:
        write_markdown_report(out_md, final_payload, summary_path)

    render_console_summary(final_payload, out_json=out_json, out_md=out_md)

    if metrics_missing_globally:
        print(
            "No recommendation candidate had usable p95 metric values "
            f"for requested metric={args.metric} (TTFC fallback to TTFT attempted)."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

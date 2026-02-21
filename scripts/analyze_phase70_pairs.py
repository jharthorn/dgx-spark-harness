#!/usr/bin/env python3
"""Analyze Phase70 pair-local run manifests into CSV/JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def parse_int(value: Any, *, min_value: int | None = None) -> int | None:
    try:
        if value is None:
            return None
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if min_value is not None and parsed < min_value:
        return None
    return parsed


def resolve_replay_phase(summary: dict[str, Any]) -> dict[str, Any]:
    phase_summaries = summary.get("phase_summaries") if isinstance(summary.get("phase_summaries"), list) else []
    for phase in phase_summaries:
        if isinstance(phase, dict) and str(phase.get("phase")) == "replay":
            return phase
    for phase in phase_summaries:
        if isinstance(phase, dict) and str(phase.get("phase")) == "replay_A":
            return phase
    replay_like = [
        phase for phase in phase_summaries if isinstance(phase, dict) and str(phase.get("phase") or "").startswith("replay")
    ]
    return replay_like[0] if replay_like else {}


def resolve_replay_read_gib(replay: dict[str, Any]) -> float | None:
    io_delta = replay.get("io_delta") if isinstance(replay.get("io_delta"), dict) else {}
    read_bytes = parse_float(io_delta.get("read_bytes_delta"))
    if read_bytes is not None:
        return read_bytes / float(1024**3)
    read_mib = parse_float(io_delta.get("read_mib_delta"))
    if read_mib is not None:
        return read_mib / 1024.0
    return None


def resolve_kvbm_delta(replay: dict[str, Any], key_primary: str, key_fallback: str) -> float | None:
    kv_delta = replay.get("kvbm_metrics_delta") if isinstance(replay.get("kvbm_metrics_delta"), dict) else {}
    value = parse_float(kv_delta.get(key_primary))
    if value is not None:
        return value
    return parse_float(kv_delta.get(key_fallback))


def extract_io_verdict(run_dir: Path) -> tuple[bool | None, str | None, bool | None, dict[str, Any] | None]:
    verdict_path = run_dir / "io" / "io_attrib_verdict.json"
    if not verdict_path.exists():
        return None, None, None, None
    try:
        verdict = load_json(verdict_path)
    except Exception:
        return None, None, None, None

    checks = verdict.get("checks") if isinstance(verdict.get("checks"), list) else []
    pid_warn = any(
        isinstance(item, dict)
        and item.get("name") == "replay_process_pid_readers_present"
        and str(item.get("status") or "").upper() == "WARN"
        for item in checks
    )
    method = str(verdict.get("process_evidence_method") or "").strip() or None
    return parse_bool(verdict.get("pass")), method, bool(pid_warn), verdict


def extract_run_row(entry: dict[str, Any], *, replay_concurrency: int | None) -> dict[str, Any]:
    run_dir = Path(str(entry.get("run_dir") or ""))
    summary = load_json(run_dir / "summary.json")

    replay = resolve_replay_phase(summary)
    overall = summary.get("overall_summary") if isinstance(summary.get("overall_summary"), dict) else {}
    storage = summary.get("storage") if isinstance(summary.get("storage"), dict) else {}

    io_attrib_pass, process_evidence_method, pid_warn, io_verdict = extract_io_verdict(run_dir)
    replay_ttfc_p95 = parse_float((replay.get("ttfc_ms") or {}).get("p95"))
    replay_ttfc_p99 = parse_float((replay.get("ttfc_ms") or {}).get("p99"))
    replay_ttft_p95 = parse_float((replay.get("ttft_ms") or {}).get("p95"))
    replay_ttft_p99 = parse_float((replay.get("ttft_ms") or {}).get("p99"))
    metric_used_replay_p95 = "replay_ttfc_p95_ms"
    if replay_ttfc_p95 is None:
        metric_used_replay_p95 = "replay_ttft_p95_ms" if replay_ttft_p95 is not None else "missing"

    row = {
        "pair_id": int(entry.get("pair_id") or 0),
        "pair_order": entry.get("pair_order"),
        "pair_leg": int(entry.get("pair_leg") or 0),
        "mode": entry.get("mode"),
        "replay_concurrency": replay_concurrency,
        "run_id": entry.get("run_id"),
        "bundle_id": entry.get("bundle_id"),
        "timestamp_utc": summary.get("created_utc"),
        "status": "ok" if bool(summary.get("run_valid")) else "invalid",
        "error_rate": parse_float(overall.get("error_rate")),
        "stream": parse_bool(summary.get("stream")),
        "stream_record_ttfb": parse_bool(summary.get("stream_record_ttfb")),
        "replay_ttfc_p95_ms": replay_ttfc_p95,
        "replay_ttfc_p99_ms": replay_ttfc_p99,
        "replay_ttft_p95_ms": replay_ttft_p95,
        "replay_ttft_p99_ms": replay_ttft_p99,
        "metric_preferred_replay_p95": "replay_ttfc_p95_ms",
        "metric_used_replay_p95": metric_used_replay_p95,
        "replay_read_gib": resolve_replay_read_gib(replay),
        "io_attrib_pass": io_attrib_pass,
        "process_evidence_method": process_evidence_method,
        "pid_warn": pid_warn,
        "kvbm_matched_tokens_delta": resolve_kvbm_delta(replay, "kvbm_matched_tokens_delta", "matched_tokens_total_delta"),
        "kvbm_onboard_blocks_delta": resolve_kvbm_delta(replay, "onboard_blocks_total_delta", "kvbm_onboard_blocks_d2d_delta"),
        "primary_nvme_model": storage.get("primary_nvme_model"),
        "primary_nvme_fw": storage.get("primary_nvme_fw"),
        "pcie_link": storage.get("pcie_link"),
        "run_path": str(run_dir),
        "io_attrib_checked": parse_bool(entry.get("io_attrib_checked")),
        "io_attrib_checker_rc": parse_float(entry.get("io_attrib_checker_rc")),
        "io_attrib_verdict_path": str(run_dir / "io" / "io_attrib_verdict.json") if io_verdict is not None else None,
    }
    return row


def delta_value(mode_b_value: Any, mode_a_value: Any) -> float | None:
    b = parse_float(mode_b_value)
    a = parse_float(mode_a_value)
    if a is None or b is None:
        return None
    return b - a


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    var = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var)


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "stddev": None,
            "min": None,
            "max": None,
            "approx_ci95_half_width": None,
            "approx_ci95_low": None,
            "approx_ci95_high": None,
        }

    n = len(values)
    mu = mean(values)
    sd = stddev(values)
    min_v = min(values)
    max_v = max(values)
    ci_half = None
    if sd is not None and n >= 2:
        # Descriptive-only normal approximation; avoids scipy dependency.
        ci_half = 1.96 * (sd / math.sqrt(n))
    ci_low = (mu - ci_half) if (mu is not None and ci_half is not None) else None
    ci_high = (mu + ci_half) if (mu is not None and ci_half is not None) else None
    return {
        "n": n,
        "mean": mu,
        "stddev": sd,
        "min": min_v,
        "max": max_v,
        "approx_ci95_half_width": ci_half,
        "approx_ci95_low": ci_low,
        "approx_ci95_high": ci_high,
    }


def order_metric_stats(
    deltas: list[dict[str, Any]],
    metric: str,
    *,
    order_ab: str,
    order_ba: str,
) -> dict[str, Any]:
    by_order: dict[str, list[float]] = {order_ab: [], order_ba: []}
    for row in deltas:
        order = str(row.get("pair_order") or "")
        value = parse_float(row.get(metric))
        if order in by_order and value is not None:
            by_order[order].append(value)

    stats_ab = summarize_values(by_order[order_ab])
    stats_ba = summarize_values(by_order[order_ba])
    all_values = by_order[order_ab] + by_order[order_ba]
    all_stats = summarize_values(all_values)

    out = {
        "metric": metric,
        "order_ab_label": order_ab,
        "order_ba_label": order_ba,
        "order_ab": stats_ab,
        "order_ba": stats_ba,
        "all_pairs": all_stats,
    }

    mean_ab = stats_ab["mean"]
    mean_ba = stats_ba["mean"]
    diff_of_means = (mean_ab - mean_ba) if (mean_ab is not None and mean_ba is not None) else None
    out["difference_of_means"] = diff_of_means

    std_ab = stats_ab["stddev"]
    std_ba = stats_ba["stddev"]
    if diff_of_means is None:
        out["relative_effect_size"] = None
        out["order_effect_flag"] = None
        out["order_effect_note"] = "Insufficient AB/BA data for this metric."
        return out

    if std_ab is not None and std_ba is not None:
        pooled_std = math.sqrt(((std_ab**2) + (std_ba**2)) / 2.0)
        if pooled_std > 0:
            rel = abs(diff_of_means) / pooled_std
            out["relative_effect_size"] = rel
            out["order_effect_flag"] = bool(rel >= 1.0)
            out["order_effect_note"] = (
                "Potential order dependence; consider larger N or warmup/washout."
                if rel >= 1.0
                else "No large order effect observed in descriptive check."
            )
            return out

    clean_values = list(all_values)
    spread = stddev(clean_values)
    threshold = max(1.0, (spread if spread is not None else 0.0), 0.2 * abs(mean(clean_values) or 0.0))
    flag = abs(diff_of_means) > threshold
    out["relative_effect_size"] = None
    out["order_effect_flag"] = flag
    out["order_effect_note"] = (
        "Potential order dependence (variance estimate limited); consider larger N or warmup/washout."
        if flag
        else "No obvious order effect in descriptive check (variance estimate limited)."
    )
    return out


def build_delta_rollups(pair_deltas: list[dict[str, Any]], metrics: list[str]) -> dict[str, dict[str, Any]]:
    rollups: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        values: list[float] = []
        for row in pair_deltas:
            value = parse_float(row.get(metric))
            if value is not None:
                values.append(value)
        rollups[metric] = summarize_values(values)
    return rollups


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase70 pair-local rehydrate repeats.")
    parser.add_argument("--manifest", required=True, help="Runner manifest JSON from phase70 script.")
    parser.add_argument("--summary-json", required=True, help="Output summary JSON path.")
    parser.add_argument("--summary-csv", required=True, help="Output per-run CSV path.")
    parser.add_argument("--pair-delta-csv", required=True, help="Output per-pair delta CSV path.")
    parser.add_argument("--order-check-json", required=True, help="Output order-effect JSON path.")
    parser.add_argument("--mode-a", default="B1", help="Reference mode in deltas (delta = mode_b - mode_a).")
    parser.add_argument("--mode-b", default="B2", help="Comparison mode in deltas (delta = mode_b - mode_a).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    summary_json_path = Path(args.summary_json)
    summary_csv_path = Path(args.summary_csv)
    pair_delta_csv_path = Path(args.pair_delta_csv)
    order_check_json_path = Path(args.order_check_json)

    manifest = load_json(manifest_path)
    manifest_meta = manifest.get("meta") if isinstance(manifest.get("meta"), dict) else {}
    replay_concurrency = parse_int(manifest_meta.get("replay_concurrency"), min_value=1)
    run_entries = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []

    per_run_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for entry in run_entries:
        if not isinstance(entry, dict):
            continue
        try:
            per_run_rows.append(extract_run_row(entry, replay_concurrency=replay_concurrency))
        except Exception as exc:  # noqa: BLE001
            run_id = entry.get("run_id") if isinstance(entry, dict) else "unknown"
            errors.append(f"failed_to_extract_run:{run_id}:{exc}")

    per_run_rows.sort(key=lambda row: (int(row.get("pair_id") or 0), int(row.get("pair_leg") or 0)))

    per_run_fields = [
        "pair_id",
        "pair_order",
        "pair_leg",
        "mode",
        "replay_concurrency",
        "run_id",
        "bundle_id",
        "timestamp_utc",
        "status",
        "error_rate",
        "stream",
        "stream_record_ttfb",
        "replay_ttfc_p95_ms",
        "replay_ttfc_p99_ms",
        "replay_ttft_p95_ms",
        "replay_ttft_p99_ms",
        "metric_preferred_replay_p95",
        "metric_used_replay_p95",
        "replay_read_gib",
        "io_attrib_pass",
        "process_evidence_method",
        "pid_warn",
        "kvbm_matched_tokens_delta",
        "kvbm_onboard_blocks_delta",
        "primary_nvme_model",
        "primary_nvme_fw",
        "pcie_link",
        "run_path",
        "io_attrib_checked",
        "io_attrib_checker_rc",
    ]
    write_csv(summary_csv_path, per_run_rows, per_run_fields)

    by_pair: dict[int, dict[str, Any]] = {}
    for row in per_run_rows:
        pair_id = int(row.get("pair_id") or 0)
        slot = by_pair.setdefault(
            pair_id,
            {
                "pair_id": pair_id,
                "pair_order": row.get("pair_order"),
                "modes": {},
            },
        )
        slot["pair_order"] = row.get("pair_order") or slot.get("pair_order")
        slot["modes"][str(row.get("mode"))] = row

    pair_deltas: list[dict[str, Any]] = []
    for pair_id in sorted(by_pair):
        slot = by_pair[pair_id]
        mode_a_row = slot["modes"].get(args.mode_a)
        mode_b_row = slot["modes"].get(args.mode_b)
        row = {
            "pair_id": pair_id,
            "pair_order": slot.get("pair_order"),
            "mode_a": args.mode_a,
            "mode_b": args.mode_b,
            "mode_a_run_id": mode_a_row.get("run_id") if isinstance(mode_a_row, dict) else None,
            "mode_b_run_id": mode_b_row.get("run_id") if isinstance(mode_b_row, dict) else None,
            "delta_replay_ttfc_p95_ms": delta_value(
                (mode_b_row or {}).get("replay_ttfc_p95_ms"),
                (mode_a_row or {}).get("replay_ttfc_p95_ms"),
            ),
            "delta_replay_ttfc_p99_ms": delta_value(
                (mode_b_row or {}).get("replay_ttfc_p99_ms"),
                (mode_a_row or {}).get("replay_ttfc_p99_ms"),
            ),
            "delta_replay_ttft_p95_ms": delta_value(
                (mode_b_row or {}).get("replay_ttft_p95_ms"),
                (mode_a_row or {}).get("replay_ttft_p95_ms"),
            ),
            "delta_replay_ttft_p99_ms": delta_value(
                (mode_b_row or {}).get("replay_ttft_p99_ms"),
                (mode_a_row or {}).get("replay_ttft_p99_ms"),
            ),
            "delta_replay_read_gib": delta_value(
                (mode_b_row or {}).get("replay_read_gib"),
                (mode_a_row or {}).get("replay_read_gib"),
            ),
            "delta_matched_tokens": delta_value(
                (mode_b_row or {}).get("kvbm_matched_tokens_delta"),
                (mode_a_row or {}).get("kvbm_matched_tokens_delta"),
            ),
            "delta_onboard_blocks": delta_value(
                (mode_b_row or {}).get("kvbm_onboard_blocks_delta"),
                (mode_a_row or {}).get("kvbm_onboard_blocks_delta"),
            ),
        }
        pair_deltas.append(row)

    delta_fields = [
        "pair_id",
        "pair_order",
        "mode_a",
        "mode_b",
        "mode_a_run_id",
        "mode_b_run_id",
        "delta_replay_ttfc_p95_ms",
        "delta_replay_ttfc_p99_ms",
        "delta_replay_ttft_p95_ms",
        "delta_replay_ttft_p99_ms",
        "delta_replay_read_gib",
        "delta_matched_tokens",
        "delta_onboard_blocks",
    ]
    write_csv(pair_delta_csv_path, pair_deltas, delta_fields)

    order_ab = f"{args.mode_a}_{args.mode_b}"
    order_ba = f"{args.mode_b}_{args.mode_a}"
    order_metrics = {
        "delta_replay_ttfc_p95_ms": order_metric_stats(
            pair_deltas,
            "delta_replay_ttfc_p95_ms",
            order_ab=order_ab,
            order_ba=order_ba,
        ),
        "delta_replay_ttft_p95_ms": order_metric_stats(
            pair_deltas,
            "delta_replay_ttft_p95_ms",
            order_ab=order_ab,
            order_ba=order_ba,
        ),
    }
    delta_rollups = build_delta_rollups(
        pair_deltas,
        [
            "delta_replay_ttfc_p95_ms",
            "delta_replay_ttfc_p99_ms",
            "delta_replay_ttft_p95_ms",
            "delta_replay_ttft_p99_ms",
            "delta_replay_read_gib",
            "delta_matched_tokens",
            "delta_onboard_blocks",
        ],
    )
    any_flagged = any(
        metric.get("order_effect_flag") is True for metric in order_metrics.values() if isinstance(metric, dict)
    )
    order_check_payload = {
        "created_utc": now_utc(),
        "meta": {
            "replay_concurrency": replay_concurrency,
        },
        "pair_count": len(pair_deltas),
        "mode_a": args.mode_a,
        "mode_b": args.mode_b,
        "order_labels": {
            "order_ab": order_ab,
            "order_ba": order_ba,
        },
        "delta_definition": "mode_b_minus_mode_a",
        "note": "Descriptive order-effect check only. Do not overclaim significance for small N.",
        "metrics": order_metrics,
        "order_effect_summary": {
            "order_dependent": any_flagged,
            "recommendation": (
                "Potential order dependence detected. Increase N and/or add warmup/washout controls."
                if any_flagged
                else "No large order effect observed in this descriptive check."
            ),
        },
        "delta_rollups": delta_rollups,
    }
    order_check_json_path.write_text(json.dumps(order_check_payload, indent=2) + "\n", encoding="utf-8")

    summary_meta = dict(manifest_meta)
    if replay_concurrency is not None:
        summary_meta["replay_concurrency"] = replay_concurrency

    summary_payload = {
        "meta": summary_meta,
        "replay_concurrency": replay_concurrency,
        "mode_a": args.mode_a,
        "mode_b": args.mode_b,
        "run_count": len(per_run_rows),
        "pair_count": len(pair_deltas),
        "rows": per_run_rows,
        "pair_deltas": pair_deltas,
        "delta_rollups": delta_rollups,
        "errors": errors,
        "artifacts": {
            "manifest": str(manifest_path),
            "summary_csv": str(summary_csv_path),
            "pair_delta_csv": str(pair_delta_csv_path),
            "order_check_json": str(order_check_json_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

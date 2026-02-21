#!/usr/bin/env python3
"""Create a brief-ready Phase70 results pack from analyzer artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
        ci_half = 1.96 * (sd / math.sqrt(n))
    return {
        "n": n,
        "mean": mu,
        "stddev": sd,
        "min": min_v,
        "max": max_v,
        "approx_ci95_half_width": ci_half,
        "approx_ci95_low": (mu - ci_half) if (mu is not None and ci_half is not None) else None,
        "approx_ci95_high": (mu + ci_half) if (mu is not None and ci_half is not None) else None,
    }


def fmt(value: Any, digits: int = 3) -> str:
    number = parse_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def resolve_latest_ts(results_root: Path) -> str:
    candidates = list(results_root.glob("phase70_rehydrate_pair_repeats_manifest_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no Phase70 manifest files under {results_root}")
    suffixes = [path.stem.replace("phase70_rehydrate_pair_repeats_manifest_", "") for path in candidates]
    return sorted(suffixes)[-1]


def resolve_replay_phase(summary: dict[str, Any]) -> dict[str, Any]:
    phases = summary.get("phase_summaries") if isinstance(summary.get("phase_summaries"), list) else []
    for phase in phases:
        if isinstance(phase, dict) and str(phase.get("phase")) == "replay":
            return phase
    for phase in phases:
        if isinstance(phase, dict) and str(phase.get("phase")) == "replay_A":
            return phase
    for phase in phases:
        if isinstance(phase, dict) and str(phase.get("phase") or "").startswith("replay"):
            return phase
    return {}


def enrich_summary_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        enriched = dict(row)
        run_path = Path(str(row.get("run_path") or ""))
        if run_path.exists():
            summary_path = run_path / "summary.json"
            if summary_path.exists():
                try:
                    run_summary = load_json(summary_path)
                    replay = resolve_replay_phase(run_summary)
                    ttfc = replay.get("ttfc_ms") if isinstance(replay.get("ttfc_ms"), dict) else {}
                    ttft = replay.get("ttft_ms") if isinstance(replay.get("ttft_ms"), dict) else {}
                    enriched["replay_ttfc_p50_ms"] = parse_float(ttfc.get("p50"))
                    enriched["replay_ttft_p50_ms"] = parse_float(ttft.get("p50"))
                except Exception:
                    enriched["replay_ttfc_p50_ms"] = None
                    enriched["replay_ttft_p50_ms"] = None
        if "replay_ttfc_p50_ms" not in enriched:
            enriched["replay_ttfc_p50_ms"] = None
        if "replay_ttft_p50_ms" not in enriched:
            enriched["replay_ttft_p50_ms"] = None

        for key in (
            "pair_id",
            "pair_leg",
            "replay_ttfc_p95_ms",
            "replay_ttfc_p99_ms",
            "replay_ttft_p95_ms",
            "replay_ttft_p99_ms",
            "replay_read_gib",
        ):
            if key in enriched:
                if key in {"pair_id", "pair_leg"}:
                    try:
                        enriched[key] = int(str(enriched[key]))
                    except Exception:
                        pass
                else:
                    enriched[key] = parse_float(enriched.get(key))
        if not str(enriched.get("metric_preferred_replay_p95") or "").strip():
            enriched["metric_preferred_replay_p95"] = "replay_ttfc_p95_ms"
        metric_used_replay_p95 = str(enriched.get("metric_used_replay_p95") or "").strip()
        if not metric_used_replay_p95:
            if parse_float(enriched.get("replay_ttfc_p95_ms")) is not None:
                metric_used_replay_p95 = "replay_ttfc_p95_ms"
            elif parse_float(enriched.get("replay_ttft_p95_ms")) is not None:
                metric_used_replay_p95 = "replay_ttft_p95_ms"
            else:
                metric_used_replay_p95 = "missing"
        enriched["metric_used_replay_p95"] = metric_used_replay_p95
        enriched["io_attrib_pass"] = parse_bool(enriched.get("io_attrib_pass"))
        out.append(enriched)
    return out


def mode_metric_values(rows: list[dict[str, Any]], mode: str, metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if str(row.get("mode")) != mode:
            continue
        value = parse_float(row.get(metric))
        if value is not None:
            values.append(value)
    return values


def pair_metric_deltas(rows: list[dict[str, Any]], mode_a: str, mode_b: str, metric: str) -> list[float]:
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, int):
            continue
        slot = by_pair.setdefault(pair_id, {})
        slot[str(row.get("mode"))] = row

    deltas: list[float] = []
    for pair_id in sorted(by_pair):
        slot = by_pair[pair_id]
        a = parse_float((slot.get(mode_a) or {}).get(metric))
        b = parse_float((slot.get(mode_b) or {}).get(metric))
        if a is None or b is None:
            continue
        deltas.append(b - a)
    return deltas


def build_main_latency_table(rows: list[dict[str, Any]], mode_a: str, mode_b: str) -> list[dict[str, Any]]:
    metrics = [
        ("replay_ttfc_p50_ms", "Replay TTFC p50 (ms)"),
        ("replay_ttfc_p95_ms", "Replay TTFC p95 (ms)"),
        ("replay_ttfc_p99_ms", "Replay TTFC p99 (ms)"),
    ]
    out: list[dict[str, Any]] = []
    for metric_key, metric_label in metrics:
        a_stats = summarize_values(mode_metric_values(rows, mode_a, metric_key))
        b_stats = summarize_values(mode_metric_values(rows, mode_b, metric_key))
        delta_stats = summarize_values(pair_metric_deltas(rows, mode_a, mode_b, metric_key))
        out.append(
            {
                "metric": metric_label,
                "mode_a": mode_a,
                "mode_b": mode_b,
                "mode_a_mean_ms": a_stats["mean"],
                "mode_b_mean_ms": b_stats["mean"],
                "mode_a_min_ms": a_stats["min"],
                "mode_a_max_ms": a_stats["max"],
                "mode_b_min_ms": b_stats["min"],
                "mode_b_max_ms": b_stats["max"],
                "delta_mean_ms": delta_stats["mean"],
                "delta_stddev_ms": delta_stats["stddev"],
                "delta_min_ms": delta_stats["min"],
                "delta_max_ms": delta_stats["max"],
                "delta_ci95_low_ms": delta_stats["approx_ci95_low"],
                "delta_ci95_high_ms": delta_stats["approx_ci95_high"],
                "delta_n_pairs": delta_stats["n"],
            }
        )
    return out


def build_mechanism_table(rows: list[dict[str, Any]], modes: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mode in modes:
        subset = [row for row in rows if str(row.get("mode")) == mode]
        replay_reads = [parse_float(row.get("replay_read_gib")) for row in subset]
        read_values = [value for value in replay_reads if value is not None]
        read_stats = summarize_values(read_values)

        io_checks = [parse_bool(row.get("io_attrib_pass")) for row in subset]
        io_present = [value for value in io_checks if value is not None]
        io_pass = sum(1 for value in io_present if value is True)
        io_total = len(io_present)
        io_rate = (io_pass / io_total) if io_total > 0 else None

        method_counts = {"pid": 0, "cgroup": 0, "none": 0, "other": 0}
        for row in subset:
            method = str(row.get("process_evidence_method") or "").strip().lower()
            if method in method_counts:
                method_counts[method] += 1
            elif method:
                method_counts["other"] += 1

        out.append(
            {
                "mode": mode,
                "run_count": len(subset),
                "replay_read_gib_mean": read_stats["mean"],
                "replay_read_gib_stddev": read_stats["stddev"],
                "replay_read_gib_min": read_stats["min"],
                "replay_read_gib_max": read_stats["max"],
                "io_attrib_pass_rate": io_rate,
                "io_attrib_pass_count": io_pass,
                "io_attrib_total": io_total,
                "process_evidence_pid_count": method_counts["pid"],
                "process_evidence_cgroup_count": method_counts["cgroup"],
                "process_evidence_none_count": method_counts["none"],
                "process_evidence_other_count": method_counts["other"],
            }
        )
    return out


def build_order_effect_table(order_check: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = order_check.get("metrics") if isinstance(order_check.get("metrics"), dict) else {}
    out: list[dict[str, Any]] = []
    for metric_name, payload in metrics.items():
        if not isinstance(payload, dict):
            continue
        ab = payload.get("order_ab") if isinstance(payload.get("order_ab"), dict) else {}
        ba = payload.get("order_ba") if isinstance(payload.get("order_ba"), dict) else {}
        out.append(
            {
                "metric": metric_name,
                "order_ab_label": payload.get("order_ab_label"),
                "order_ba_label": payload.get("order_ba_label"),
                "order_ab_n": ab.get("n"),
                "order_ab_mean": ab.get("mean"),
                "order_ab_stddev": ab.get("stddev"),
                "order_ab_min": ab.get("min"),
                "order_ab_max": ab.get("max"),
                "order_ab_ci95_low": ab.get("approx_ci95_low"),
                "order_ab_ci95_high": ab.get("approx_ci95_high"),
                "order_ba_n": ba.get("n"),
                "order_ba_mean": ba.get("mean"),
                "order_ba_stddev": ba.get("stddev"),
                "order_ba_min": ba.get("min"),
                "order_ba_max": ba.get("max"),
                "order_ba_ci95_low": ba.get("approx_ci95_low"),
                "order_ba_ci95_high": ba.get("approx_ci95_high"),
                "difference_of_means": payload.get("difference_of_means"),
                "relative_effect_size": payload.get("relative_effect_size"),
                "order_effect_flag": payload.get("order_effect_flag"),
                "order_effect_note": payload.get("order_effect_note"),
            }
        )
    return out


def find_metric_row(table: list[dict[str, Any]], metric_label: str) -> dict[str, Any] | None:
    for row in table:
        if str(row.get("metric")) == metric_label:
            return row
    return None


def find_mode_row(table: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    for row in table:
        if str(row.get("mode")) == mode:
            return row
    return None


def maybe_make_figures(
    rows: list[dict[str, Any]],
    main_latency: list[dict[str, Any]],
    mechanism: list[dict[str, Any]],
    order_table: list[dict[str, Any]],
    figures_dir: Path,
    mode_a: str,
    mode_b: str,
) -> list[str]:
    notes: list[str] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        (figures_dir / "README.md").write_text(
            "Figure generation skipped: matplotlib not installed.\n",
            encoding="utf-8",
        )
        return ["matplotlib_not_available"]

    figures_dir.mkdir(parents=True, exist_ok=True)

    p95 = find_metric_row(main_latency, "Replay TTFC p95 (ms)") or {}
    p99 = find_metric_row(main_latency, "Replay TTFC p99 (ms)") or {}

    fig1 = figures_dir / "ttfc_mode_and_deltas.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        [mode_a, mode_b],
        [parse_float(p95.get("mode_a_mean_ms")) or 0.0, parse_float(p95.get("mode_b_mean_ms")) or 0.0],
        color=["#4e79a7", "#f28e2b"],
    )
    axes[0].set_title("Replay TTFC p95 by Mode")
    axes[0].set_ylabel("ms")

    deltas_p95 = pair_metric_deltas(rows, mode_a, mode_b, "replay_ttfc_p95_ms")
    axes[1].hist(deltas_p95, bins=min(max(len(deltas_p95), 3), 10), color="#59a14f")
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_title("Per-pair Delta (B2-B1) TTFC p95")
    axes[1].set_xlabel("ms")
    fig.tight_layout()
    fig.savefig(fig1, dpi=140)
    plt.close(fig)

    fig2 = figures_dir / "replay_read_gib_by_mode.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    mech_a = find_mode_row(mechanism, mode_a) or {}
    mech_b = find_mode_row(mechanism, mode_b) or {}
    ax.bar(
        [mode_a, mode_b],
        [
            parse_float(mech_a.get("replay_read_gib_mean")) or 0.0,
            parse_float(mech_b.get("replay_read_gib_mean")) or 0.0,
        ],
        color=["#76b7b2", "#e15759"],
    )
    ax.set_ylabel("GiB")
    ax.set_title("Replay Read GiB by Mode")
    fig.tight_layout()
    fig.savefig(fig2, dpi=140)
    plt.close(fig)

    if order_table:
        row = order_table[0]
        fig3 = figures_dir / "order_effect_ab_ba.png"
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = [str(row.get("order_ab_label")), str(row.get("order_ba_label"))]
        means = [parse_float(row.get("order_ab_mean")) or 0.0, parse_float(row.get("order_ba_mean")) or 0.0]
        ax.bar(labels, means, color=["#9c755f", "#bab0ab"])
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel("Delta ms")
        ax.set_title(f"Order Effect ({row.get('metric')})")
        fig.tight_layout()
        fig.savefig(fig3, dpi=140)
        plt.close(fig)
    notes.append("figures_generated=true")
    return notes


def write_methodology_md(
    path: Path,
    *,
    ts: str,
    manifest: dict[str, Any],
    main_latency: list[dict[str, Any]],
    mechanism: list[dict[str, Any]],
    order_table: list[dict[str, Any]],
    mode_a: str,
    mode_b: str,
    replay_concurrency: int,
    verdict_checks: dict[str, Any] | None = None,
) -> None:
    meta = manifest.get("meta") if isinstance(manifest.get("meta"), dict) else {}
    pair_count = int(meta.get("pair_count") or 0)
    ttfc_p95 = find_metric_row(main_latency, "Replay TTFC p95 (ms)") or {}
    ttfc_p99 = find_metric_row(main_latency, "Replay TTFC p99 (ms)") or {}
    mech_b = find_mode_row(mechanism, mode_b) or {}
    order_primary = next((row for row in order_table if row.get("metric") == "delta_replay_ttfc_p95_ms"), None) or {}
    verdict_checks = verdict_checks or {}

    evidence_dist = (
        f"pid={int(mech_b.get('process_evidence_pid_count') or 0)}, "
        f"cgroup={int(mech_b.get('process_evidence_cgroup_count') or 0)}, "
        f"none={int(mech_b.get('process_evidence_none_count') or 0)}"
    )

    text = f"""# Phase70 Brief-Ready Methodology

## Run Metadata

- `ts`: `{ts}`
- `pair_count`: `{pair_count}`
- `mode_a`: `{mode_a}`
- `mode_b`: `{mode_b}`
- `replay_concurrency`: `c{replay_concurrency}`
- `order_strategy`: `{meta.get("order_strategy")}`
- `washout_s`: `{meta.get("pair_washout_s")}`
- `stream_metrics_enabled`: `{meta.get("stream_metrics_enabled")}`
- `stream_timeout_s`: `{meta.get("stream_timeout_s")}`
- `stream_record_ttfb`: `{meta.get("stream_record_ttfb")}`
- `io_attrib_enabled`: `{meta.get("io_attrib_enabled")}`

## Definitions

- `TTFC`: time from request start to first non-empty streamed chunk/event.
- `TTFB`: optional time to first response byte/header.
- `A2 PASS`: replay attribution gate succeeded for the run.
- `process_evidence_method`: `pid` or `cgroup` evidence source used by A2.
- In containerized setups, `process_evidence_method=cgroup` is expected when per-PID `/proc/<pid>/io` deltas are zero.

## Methodology

- Pair-local blocked design with AB/BA counterbalancing (`B1_B2` and `B2_B1`).
- Replay concurrency: `c={replay_concurrency}` (replay phase executed with `{replay_concurrency}` concurrent sessions).
- Delta definition is fixed as `mode_b - mode_a` (default `B2 - B1`).
- Primary latency surface: replay `TTFC p95/p99`; legacy `TTFT` is retained for compatibility.
- CI95 is an approximate descriptive band computed from pair deltas (not a formal hypothesis test).

## Mechanism Signal Summary

- SSD write signal observed: `{verdict_checks.get("ssd_write_signal_present")}`
- SSD rehydrate signal observed: `{verdict_checks.get("ssd_rehydrate_signal_present")}`
- SSD reuse signal observed: `{verdict_checks.get("ssd_reuse_signal_present")}`
- Rehydrate required for decision-grade: `{verdict_checks.get("decision_grade_require_rehydrate")}`

## Headline Claim Structure

- Mechanism claim: Disk-backed replay was verified in `{mode_b}` with A2 PASS rate `{fmt((parse_float(mech_b.get("io_attrib_pass_rate")) or 0.0) * 100.0, 1)}%` (`{int(mech_b.get("io_attrib_pass_count") or 0)}/{int(mech_b.get("io_attrib_total") or 0)}`), replay reads mean `{fmt(mech_b.get("replay_read_gib_mean"), 3)} GiB`, evidence distribution `{evidence_dist}`.
- Performance claim: Under this workload shape (`PAIRS={pair_count}`), replay TTFC is near-neutral at p95 (`Δ={fmt(ttfc_p95.get("delta_mean_ms"), 3)} ms`, CI95 `[ {fmt(ttfc_p95.get("delta_ci95_low_ms"), 3)}, {fmt(ttfc_p95.get("delta_ci95_high_ms"), 3)} ]`) and changes at p99 (`Δ={fmt(ttfc_p99.get("delta_mean_ms"), 3)} ms`, CI95 `[ {fmt(ttfc_p99.get("delta_ci95_low_ms"), 3)}, {fmt(ttfc_p99.get("delta_ci95_high_ms"), 3)} ]`).
- Scope claim: Effect size is workload-dependent; this pack includes reproducible knobs and pair-local methodology to explore regimes with larger gains.

## Order-Effect Summary

- Primary metric: `delta_replay_ttfc_p95_ms`
- AB mean/stddev: `{fmt(order_primary.get("order_ab_mean"), 3)} / {fmt(order_primary.get("order_ab_stddev"), 3)}`
- BA mean/stddev: `{fmt(order_primary.get("order_ba_mean"), 3)} / {fmt(order_primary.get("order_ba_stddev"), 3)}`
- Difference-of-means: `{fmt(order_primary.get("difference_of_means"), 3)}`
- Flag: `{order_primary.get("order_effect_flag")}`
- Note: `{order_primary.get("order_effect_note")}`
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a brief-ready Phase70 results pack.")
    parser.add_argument("--results-root", default="bench/results", help="Results root path.")
    parser.add_argument("--ts", default=None, help="Phase70 timestamp token. Defaults to latest manifest.")
    parser.add_argument("--out-root", default=None, help="Output root (default: <results_root>/publish).")
    parser.add_argument("--pack-name", default=None, help="Output folder name under out-root.")
    parser.add_argument("--mode-a", default=None, help="Override mode_a (default from manifest).")
    parser.add_argument("--mode-b", default=None, help="Override mode_b (default from manifest).")
    parser.add_argument("--skip-figures", action="store_true", help="Skip optional PNG figure generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.exists():
        raise SystemExit(f"results root not found: {results_root}")

    ts = args.ts or resolve_latest_ts(results_root)
    manifest_path = results_root / f"phase70_rehydrate_pair_repeats_manifest_{ts}.json"
    summary_json_path = results_root / f"phase70_rehydrate_pair_repeats_summary_{ts}.json"
    summary_csv_path = results_root / f"phase70_rehydrate_pair_repeats_summary_{ts}.csv"
    deltas_csv_path = results_root / f"phase70_rehydrate_pair_repeats_deltas_{ts}.csv"
    order_check_path = results_root / f"phase70_rehydrate_pair_repeats_order_check_{ts}.json"
    verdict_path = results_root / f"phase70_rehydrate_pair_repeats_verdict_{ts}.json"

    for path in (manifest_path, summary_json_path, summary_csv_path, deltas_csv_path, order_check_path):
        if not path.exists():
            raise SystemExit(f"required artifact missing: {path}")

    manifest = load_json(manifest_path)
    manifest_meta = manifest.get("meta") if isinstance(manifest.get("meta"), dict) else {}
    summary_obj = load_json(summary_json_path)
    order_check = load_json(order_check_path)
    summary_meta = summary_obj.get("meta") if isinstance(summary_obj.get("meta"), dict) else {}

    mode_a = str(args.mode_a or manifest_meta.get("mode_a") or summary_obj.get("mode_a") or "B1")
    mode_b = str(args.mode_b or manifest_meta.get("mode_b") or summary_obj.get("mode_b") or "B2")
    pair_count = int(manifest_meta.get("pair_count") or summary_obj.get("pair_count") or 0)
    replay_concurrency = (
        parse_int(manifest_meta.get("replay_concurrency"), min_value=1)
        or parse_int(summary_meta.get("replay_concurrency"), min_value=1)
        or parse_int(summary_obj.get("replay_concurrency"), min_value=1)
        or 1
    )

    out_root = Path(args.out_root) if args.out_root else (results_root / "publish")
    pack_name = args.pack_name or f"phase70_pairs{pair_count}_c{replay_concurrency}_{ts}"
    pack_dir = out_root / pack_name
    tables_dir = pack_dir / "tables"
    figures_dir = pack_dir / "figures"
    pack_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(summary_csv_path, pack_dir / "summary.csv")
    shutil.copy2(summary_json_path, pack_dir / "summary.json")
    shutil.copy2(deltas_csv_path, pack_dir / "deltas.csv")
    shutil.copy2(order_check_path, pack_dir / "order_check.json")
    shutil.copy2(manifest_path, pack_dir / "manifest.json")
    if verdict_path.exists():
        shutil.copy2(verdict_path, pack_dir / "verdict.json")
    verdict_obj = load_json(verdict_path) if verdict_path.exists() else {}
    verdict_checks = verdict_obj.get("checks") if isinstance(verdict_obj.get("checks"), dict) else {}

    with summary_csv_path.open("r", encoding="utf-8", newline="") as fp:
        raw_rows = list(csv.DictReader(fp))
    rows = enrich_summary_rows(raw_rows)

    main_latency = build_main_latency_table(rows, mode_a=mode_a, mode_b=mode_b)
    mechanism = build_mechanism_table(rows, modes=[mode_a, mode_b])
    order_table = build_order_effect_table(order_check)

    write_csv(
        tables_dir / "table_main_latency.csv",
        main_latency,
        [
            "metric",
            "mode_a",
            "mode_b",
            "mode_a_mean_ms",
            "mode_b_mean_ms",
            "mode_a_min_ms",
            "mode_a_max_ms",
            "mode_b_min_ms",
            "mode_b_max_ms",
            "delta_mean_ms",
            "delta_stddev_ms",
            "delta_min_ms",
            "delta_max_ms",
            "delta_ci95_low_ms",
            "delta_ci95_high_ms",
            "delta_n_pairs",
        ],
    )
    write_csv(
        tables_dir / "table_mechanism.csv",
        mechanism,
        [
            "mode",
            "run_count",
            "replay_read_gib_mean",
            "replay_read_gib_stddev",
            "replay_read_gib_min",
            "replay_read_gib_max",
            "io_attrib_pass_rate",
            "io_attrib_pass_count",
            "io_attrib_total",
            "process_evidence_pid_count",
            "process_evidence_cgroup_count",
            "process_evidence_none_count",
            "process_evidence_other_count",
        ],
    )
    write_csv(
        tables_dir / "table_order_effect.csv",
        order_table,
        [
            "metric",
            "order_ab_label",
            "order_ba_label",
            "order_ab_n",
            "order_ab_mean",
            "order_ab_stddev",
            "order_ab_min",
            "order_ab_max",
            "order_ab_ci95_low",
            "order_ab_ci95_high",
            "order_ba_n",
            "order_ba_mean",
            "order_ba_stddev",
            "order_ba_min",
            "order_ba_max",
            "order_ba_ci95_low",
            "order_ba_ci95_high",
            "difference_of_means",
            "relative_effect_size",
            "order_effect_flag",
            "order_effect_note",
        ],
    )

    write_methodology_md(
        pack_dir / "methodology.md",
        ts=ts,
        manifest=manifest,
        main_latency=main_latency,
        mechanism=mechanism,
        order_table=order_table,
        mode_a=mode_a,
        mode_b=mode_b,
        replay_concurrency=replay_concurrency,
        verdict_checks=verdict_checks,
    )

    figure_notes: list[str] = []
    if args.skip_figures:
        (figures_dir / "README.md").write_text("Figure generation skipped (--skip-figures).\n", encoding="utf-8")
        figure_notes.append("figures_skipped_cli=true")
    else:
        figure_notes = maybe_make_figures(
            rows=rows,
            main_latency=main_latency,
            mechanism=mechanism,
            order_table=order_table,
            figures_dir=figures_dir,
            mode_a=mode_a,
            mode_b=mode_b,
        )

    pack_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_ts": ts,
        "pair_count": pair_count,
        "replay_concurrency": replay_concurrency,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "ssd_write_signal_present": verdict_checks.get("ssd_write_signal_present"),
        "ssd_rehydrate_signal_present": verdict_checks.get("ssd_rehydrate_signal_present"),
        "ssd_reuse_signal_present": verdict_checks.get("ssd_reuse_signal_present"),
        "source_artifacts": {
            "manifest": str(manifest_path),
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
            "deltas_csv": str(deltas_csv_path),
            "order_check_json": str(order_check_path),
            "verdict_json": (str(verdict_path) if verdict_path.exists() else None),
        },
        "pack_artifacts": {
            "summary_csv": str(pack_dir / "summary.csv"),
            "summary_json": str(pack_dir / "summary.json"),
            "deltas_csv": str(pack_dir / "deltas.csv"),
            "order_check_json": str(pack_dir / "order_check.json"),
            "verdict_json": (str(pack_dir / "verdict.json") if verdict_path.exists() else None),
            "methodology_md": str(pack_dir / "methodology.md"),
            "tables": {
                "main_latency": str(tables_dir / "table_main_latency.csv"),
                "mechanism": str(tables_dir / "table_mechanism.csv"),
                "order_effect": str(tables_dir / "table_order_effect.csv"),
            },
            "figures_dir": str(figures_dir),
        },
        "notes": figure_notes,
    }
    write_json(pack_dir / "pack_manifest.json", pack_manifest)

    print(f"results_pack={pack_dir}")
    print(f"source_ts={ts}")
    print(f"pair_count={pair_count}")
    print(f"replay_concurrency={replay_concurrency}")
    print(f"mode_a={mode_a} mode_b={mode_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

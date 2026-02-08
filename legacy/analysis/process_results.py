#!/usr/bin/env python3
"""v3.3 analysis for H0–H9 (Test_Plan_v3.3 Appendix C)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
RUN_ROOTS = [ROOT / "results", ROOT / "runs"]
OUT_DIR = Path(__file__).resolve().parent / "figures"
CONFIG_EXPORT_KEYS = [
    "phase",
    "session_phase",
    "tokenizer",
    "max_input_len",
    "input_len_margin",
    "lora_adapter_count",
    "lora_churn_mode",
    "lora_hot_ratio",
    "lora_hot_prob",
    "mix_short_pct",
    "mix_medium_pct",
    "mix_long_pct",
    "mix_short_min",
    "mix_short_max",
    "mix_medium_min",
    "mix_medium_max",
    "mix_long_min",
    "mix_long_max",
    "burstiness",
    "burst_pause_s",
    "session_min_turns",
    "session_max_turns",
    "session_resume_turns",
    "session_idle_s",
]
STORAGE_KNEE_CFG = {
    # Small, documented heuristic because the exporter does not expose full per-tier hits/misses.
    # We look for coincident jumps in KV tier2 ingress and NVMe latency, then require TTFT to jump.
    "baseline_window": 10,
    "ttft_jump_factor": 1.5,
    "tier2_in_jump_factor": 1.5,
    "nvme_await_jump_factor": 1.5,
    "coincident_window_sec": 5.0,
    "min_tier2_bytes_in": 1 * 1024 * 1024,
    "nvme_r_await_floor_ms": 1.0,
}
NVME_JOIN_TOLERANCE_SEC = 0.2


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] * (c - k) + values[c] * (k - f)


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_config(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "config.yaml"
    if not path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_profile(run_dir: Path) -> Optional[str]:
    data = load_config(run_dir)
    profile = data.get("profile") if isinstance(data, dict) else None
    if isinstance(profile, str):
        return profile
    return None


def load_dynkv(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "dynkv.jsonl"
    if not path.exists():
        return {}
    tier2_in = 0.0
    tier2_out = 0.0
    fetch_p50_max = None
    fetch_p95_max = None
    fetch_p99_max = None
    evictions = 0
    prefetches = 0
    samples = 0
    for rec in load_jsonl(path):
        tier2 = rec.get("tier2", {}) if isinstance(rec, dict) else {}
        tier2_in += float(tier2.get("bytes_in", 0) or 0)
        tier2_out += float(tier2.get("bytes_out", 0) or 0)
        f50 = tier2.get("fetch_p50_ms")
        f95 = tier2.get("fetch_p95_ms")
        f99 = tier2.get("fetch_p99_ms")
        if f50 is not None:
            fetch_p50_max = f50 if fetch_p50_max is None else max(fetch_p50_max, f50)
        if f95 is not None:
            fetch_p95_max = f95 if fetch_p95_max is None else max(fetch_p95_max, f95)
        if f99 is not None:
            fetch_p99_max = f99 if fetch_p99_max is None else max(fetch_p99_max, f99)
        evictions += int(rec.get("evictions", 0) or 0)
        prefetches += int(rec.get("prefetches", 0) or 0)
        samples += 1
    return {
        "tier2_bytes_in_total": tier2_in,
        "tier2_bytes_out_total": tier2_out,
        "tier2_fetch_p50_ms": fetch_p50_max,
        "tier2_fetch_p95_ms": fetch_p95_max,
        "tier2_fetch_p99_ms": fetch_p99_max,
        "evictions": evictions,
        "prefetches": prefetches,
        "dynkv_samples": samples,
    }


def load_dynkv_series(run_dir: Path) -> List[Dict[str, float]]:
    """Return per-sample KV deltas for knee detection."""
    series: List[Dict[str, float]] = []
    for rec in load_jsonl(run_dir / "dynkv.jsonl"):
        ts = rec.get("ts")
        if ts is None:
            continue
        tier2 = rec.get("tier2", {}) if isinstance(rec, dict) else {}
        in_delta = rec.get("tier2_bytes_in_delta")
        out_delta = rec.get("tier2_bytes_out_delta")
        evict_delta = rec.get("kv_blocks_evicted_delta")
        # Older runs only have per-sample bytes_in/out; treat them as deltas.
        if in_delta is None:
            in_delta = tier2.get("bytes_in", 0)
        if out_delta is None:
            out_delta = tier2.get("bytes_out", 0)
        if evict_delta is None:
            evict_delta = rec.get("evictions", 0)
        try:
            series.append(
                {
                    "ts": float(ts),
                    "tier2_bytes_in_delta": float(in_delta or 0.0),
                    "tier2_bytes_out_delta": float(out_delta or 0.0),
                    "kv_blocks_evicted_delta": float(evict_delta or 0.0),
                }
            )
        except (TypeError, ValueError):
            continue
    return sorted(series, key=lambda r: r["ts"])


def load_sysmon_series(run_dir: Path) -> List[Dict[str, float]]:
    """Extract NVMe latency samples for knee detection."""
    series: List[Dict[str, float]] = []
    for rec in load_jsonl(run_dir / "sysmon.jsonl"):
        ts = rec.get("ts")
        nvme = rec.get("nvme", {}) if isinstance(rec, dict) else {}
        await_ms = nvme.get("r_await_ms")
        if ts is None or await_ms is None:
            continue
        try:
            series.append({"ts": float(ts), "r_await_ms": float(await_ms)})
        except (TypeError, ValueError):
            continue
    return sorted(series, key=lambda r: r["ts"])


def load_sysmon_nvme(run_dir: Path) -> List[Dict[str, float]]:
    """Return sysmon NVMe samples for latency correlation."""
    samples: List[Dict[str, float]] = []

    def _opt_float(val: object) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    for rec in load_jsonl(run_dir / "sysmon.jsonl"):
        ts = rec.get("ts")
        nvme = rec.get("nvme", {}) if isinstance(rec, dict) else {}
        if ts is None:
            continue
        ts_f = _opt_float(ts)
        if ts_f is None:
            continue
        sample = {
            "ts": ts_f,
            "r_await_ms": _opt_float(nvme.get("r_await_ms")),
            "util_pct": _opt_float(nvme.get("util_pct")),
            "rMBs": _opt_float(nvme.get("rMBs")),
            "rps": _opt_float(nvme.get("rps")),
        }
        samples.append(sample)
    return sorted(samples, key=lambda r: r["ts"])


def load_sysmon_count(run_dir: Path) -> int:
    return len(load_jsonl(run_dir / "sysmon.jsonl"))


def detect_collapse(p50: Optional[float], p99: Optional[float]) -> bool:
    if p50 is None or p99 is None:
        return False
    return p99 > 1000.0 or p99 > 2.0 * p50


def detect_storage_knee(
    metrics: List[Dict[str, object]],
    dynkv_series: List[Dict[str, float]],
    sysmon_series: List[Dict[str, float]],
) -> tuple[bool, Optional[float]]:
    """Approximate storage knee (Exporter does not expose per-tier hit/miss).

    We look for coincident spikes in tier2 ingress (proxy for spill) and NVMe r_await_ms.
    A knee is flagged only if TTFT p95/p99 after the spike exceeds the configured multiple
    of the early-run baseline.
    """

    if not metrics or not dynkv_series or not sysmon_series:
        return False, None

    ok_metrics = [m for m in metrics if m.get("rc") in (0, "ok") and "lat_ttft_ms" in m and "ts" in m]
    if not ok_metrics:
        return False, None

    ttft_series = sorted([(float(m["ts"]), float(m["lat_ttft_ms"])) for m in ok_metrics], key=lambda x: x[0])
    baseline_window = STORAGE_KNEE_CFG["baseline_window"]
    baseline_ttft = [v for _, v in ttft_series[:baseline_window]]
    baseline_p95 = percentile(baseline_ttft, 95)
    baseline_p99 = percentile(baseline_ttft, 99)
    if baseline_p95 is None and baseline_p99 is None:
        return False, None

    tier2_baseline_samples = [s["tier2_bytes_in_delta"] for s in dynkv_series[:baseline_window]]
    tier2_baseline = percentile(tier2_baseline_samples, 50) or 0.0
    tier2_thresh = max(STORAGE_KNEE_CFG["min_tier2_bytes_in"], tier2_baseline * STORAGE_KNEE_CFG["tier2_in_jump_factor"])

    nvme_baseline_samples = [s["r_await_ms"] for s in sysmon_series[:baseline_window]]
    nvme_baseline = percentile(nvme_baseline_samples, 50) or 0.0
    nvme_thresh = max(
        STORAGE_KNEE_CFG["nvme_r_await_floor_ms"],
        nvme_baseline * STORAGE_KNEE_CFG["nvme_await_jump_factor"],
    )

    coincident_window = STORAGE_KNEE_CFG["coincident_window_sec"]
    sysmon_spikes = [s["ts"] for s in sysmon_series if s["r_await_ms"] >= nvme_thresh]

    for kv_sample in dynkv_series:
        if kv_sample["tier2_bytes_in_delta"] < tier2_thresh:
            continue
        if not any(abs(kv_sample["ts"] - ts) <= coincident_window for ts in sysmon_spikes):
            continue
        candidate_ts = kv_sample["ts"]
        ttft_after = [v for ts, v in ttft_series if ts >= candidate_ts]
        if not ttft_after:
            continue
        p95_after = percentile(ttft_after, 95)
        p99_after = percentile(ttft_after, 99)
        meets_ttft = False
        if p95_after is not None and baseline_p95 is not None:
            meets_ttft = p95_after >= baseline_p95 * STORAGE_KNEE_CFG["ttft_jump_factor"]
        if not meets_ttft and p99_after is not None and baseline_p99 is not None:
            meets_ttft = p99_after >= baseline_p99 * STORAGE_KNEE_CFG["ttft_jump_factor"]
        if meets_ttft:
            return True, candidate_ts
    return False, None


def aggregate(run_dir: Path) -> Dict[str, object]:
    metrics = load_jsonl(run_dir / "metrics.jsonl")
    if not metrics:
        return {}
    ok_metrics = [m for m in metrics if m.get("rc") in (0, "ok")]
    if not ok_metrics:
        return {}
    config = load_config(run_dir)
    config = config if isinstance(config, dict) else {}
    cfg_profile = config.get("profile") if isinstance(config.get("profile"), str) else None

    def _coalesce_sample_config(key: str) -> object:
        sample_val = ok_metrics[0].get(key)
        return sample_val if sample_val is not None else config.get(key)

    ttft = [float(m["lat_ttft_ms"]) for m in ok_metrics if "lat_ttft_ms" in m]
    e2e = [float(m["lat_e2e_ms"]) for m in ok_metrics if "lat_e2e_ms" in m]
    if not ttft or not e2e:
        return {}
    sample = ok_metrics[0]
    p50_ttft = percentile(ttft, 50)
    p95_ttft = percentile(ttft, 95)
    p99_ttft = percentile(ttft, 99)
    result: Dict[str, object] = {
        "run_id": sample.get("run_id"),
        "stack": sample.get("stack"),
        "profile": cfg_profile,
        "model": _coalesce_sample_config("model"),
        "workload": _coalesce_sample_config("workload"),
        "context_tokens": _coalesce_sample_config("context_tokens"),
        "concurrency": _coalesce_sample_config("concurrency"),
        "phase": _coalesce_sample_config("phase"),
        "session_phase": _coalesce_sample_config("session_phase"),
        "p50_ttft_ms": p50_ttft,
        "p95_ttft_ms": p95_ttft,
        "p99_ttft_ms": p99_ttft,
        "p50_e2e_ms": percentile(e2e, 50),
        "p95_e2e_ms": percentile(e2e, 95),
        "p99_e2e_ms": percentile(e2e, 99),
        "samples": len(e2e),
        "sysmon_samples": load_sysmon_count(run_dir),
    }
    result.update(load_dynkv(run_dir))
    result["collapse"] = detect_collapse(p50_ttft, p99_ttft)
    storage_knee, knee_ts = detect_storage_knee(ok_metrics, load_dynkv_series(run_dir), load_sysmon_series(run_dir))
    result["storage_knee"] = storage_knee
    result["storage_knee_ts"] = knee_ts
    for key in CONFIG_EXPORT_KEYS:
        val = config.get(key)
        if val is not None:
            result[key] = val
    return result


def join_latency_with_nvme(
    run_dir: Path, agg_row: Optional[Dict[str, object]] = None, tolerance: float = NVME_JOIN_TOLERANCE_SEC
) -> List[Dict[str, object]]:
    metrics = load_jsonl(run_dir / "metrics.jsonl")
    sysmon_samples = load_sysmon_nvme(run_dir)
    if not metrics or not sysmon_samples:
        return []

    def _opt_float(val: object) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    metric_samples: List[Dict[str, object]] = []
    ttft_values: List[float] = []
    e2e_values: List[float] = []
    for rec in metrics:
        if rec.get("rc") not in (0, "ok"):
            continue
        ts_f = _opt_float(rec.get("ts"))
        if ts_f is None:
            continue
        ttft = _opt_float(rec.get("lat_ttft_ms"))
        e2e = _opt_float(rec.get("lat_e2e_ms"))
        if ttft is not None:
            ttft_values.append(ttft)
        if e2e is not None:
            e2e_values.append(e2e)
        metric_samples.append(
            {
                "ts": ts_f,
                "lat_ttft_ms": ttft,
                "lat_e2e_ms": e2e,
                "run_id": rec.get("run_id"),
                "stack": rec.get("stack"),
                "concurrency": rec.get("concurrency"),
                "context_tokens": rec.get("context_tokens"),
                "context_bucket": rec.get("context_bucket") or rec.get("context_tokens"),
                "profile": rec.get("profile"),
            }
        )
    if not metric_samples:
        return []

    def _agg_or_fallback(key: str, fallback: Optional[float]) -> Optional[float]:
        if agg_row is None:
            return fallback
        try:
            val = agg_row.get(key)
            return float(val) if val is not None else fallback
        except (TypeError, ValueError):
            return fallback

    p50_ttft = _agg_or_fallback("p50_ttft_ms", percentile(ttft_values, 50))
    p95_ttft = _agg_or_fallback("p95_ttft_ms", percentile(ttft_values, 95))
    p99_ttft = _agg_or_fallback("p99_ttft_ms", percentile(ttft_values, 99))
    p50_e2e = _agg_or_fallback("p50_e2e_ms", percentile(e2e_values, 50))
    p95_e2e = _agg_or_fallback("p95_e2e_ms", percentile(e2e_values, 95))
    p99_e2e = _agg_or_fallback("p99_e2e_ms", percentile(e2e_values, 99))

    # Nearest-neighbor join to correlate latency samples with adjacent NVMe telemetry.
    joined: List[Dict[str, object]] = []
    sys_idx = 0
    for sample in sorted(metric_samples, key=lambda s: s["ts"]):
        while sys_idx + 1 < len(sysmon_samples) and abs(sysmon_samples[sys_idx + 1]["ts"] - sample["ts"]) <= abs(
            sysmon_samples[sys_idx]["ts"] - sample["ts"]
        ):
            sys_idx += 1
        nearest = sysmon_samples[sys_idx]
        if abs(nearest["ts"] - sample["ts"]) > tolerance:
            continue
        joined.append(
            {
                "run_id": sample.get("run_id"),
                "stack": sample.get("stack"),
                "profile": agg_row.get("profile") if agg_row else sample.get("profile"),
                "concurrency": sample.get("concurrency"),
                "context_tokens": sample.get("context_tokens"),
                "context_bucket": sample.get("context_bucket"),
                "lat_ttft_ms": sample.get("lat_ttft_ms"),
                "lat_e2e_ms": sample.get("lat_e2e_ms"),
                "p50_ttft_ms": p50_ttft,
                "p95_ttft_ms": p95_ttft,
                "p99_ttft_ms": p99_ttft,
                "p50_e2e_ms": p50_e2e,
                "p95_e2e_ms": p95_e2e,
                "p99_e2e_ms": p99_e2e,
                "nvme_r_await_ms": nearest.get("r_await_ms"),
                "nvme_util_pct": nearest.get("util_pct"),
                "nvme_rMBs": nearest.get("rMBs"),
                "nvme_rps": nearest.get("rps"),
                "metrics_ts": sample["ts"],
                "sysmon_ts": nearest["ts"],
                "ts_delta": sample["ts"] - nearest["ts"],
            }
        )
    return joined


def compute_u_work(rows: List[Dict[str, object]], stack: str) -> Optional[int]:
    candidates = []
    for row in rows:
        if row.get("stack") != stack:
            continue
        run_id = row.get("run_id", "")
        if not isinstance(run_id, str) or "H0" not in run_id:
            continue
        p50 = row.get("p50_ttft_ms")
        p99 = row.get("p99_ttft_ms")
        conc = row.get("concurrency")
        if p50 is None or p99 is None or conc is None:
            continue
        meets = p99 <= 2.0 * p50
        candidates.append((int(conc), bool(meets)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    streak = 0
    u_work = None
    for conc, ok in candidates:
        if ok:
            streak += 1
            if streak >= 2:
                u_work = conc
        else:
            streak = 0
    return u_work


def write_csv(path: Path, headers: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def write_summary(rows: List[Dict[str, object]]) -> None:
    summary_path = OUT_DIR / "summary_v3.csv"
    headers = [
        "run_id",
        "stack",
        "profile",
        "model",
        "workload",
        "context_tokens",
        "concurrency",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p50_e2e_ms",
        "p95_e2e_ms",
        "p99_e2e_ms",
        "samples",
        "sysmon_samples",
        "tier2_bytes_in_total",
        "tier2_bytes_out_total",
        "tier2_fetch_p50_ms",
        "tier2_fetch_p95_ms",
        "tier2_fetch_p99_ms",
        "evictions",
        "prefetches",
        "dynkv_samples",
        "collapse",
        "storage_knee",
        "storage_knee_ts",
    ]
    write_csv(summary_path, headers, rows)


def write_uwork(rows: List[Dict[str, object]]) -> None:
    u_stackA = compute_u_work(rows, "stackA")
    u_stackB = compute_u_work(rows, "stackB")
    u_rows = [
        {"stack": "stackA", "u_work": u_stackA},
        {"stack": "stackB", "u_work": u_stackB},
    ]
    uwork_path = OUT_DIR / "uwork.csv"
    write_csv(uwork_path, ["stack", "u_work"], u_rows)
    # Emit helper artifact for runners (prefer Stack B, else Stack A).
    chosen = u_stackB or u_stackA
    if chosen is not None:
        artifact_dir = ROOT / "runs" / "H0"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "uwork.txt").write_text(str(chosen))


def write_h4b_latency_vs_nvme(rows: List[Dict[str, object]]) -> None:
    # H4B: Figure 8.4B – tail latency (p99) vs NVMe latency/utilization under QoS perturbation.
    filtered = [r for r in rows if "H4B" in str(r.get("run_id", ""))]
    if not filtered:
        return
    path = OUT_DIR / "H4B_p99_vs_nvme.csv"
    headers = [
        "run_id",
        "stack",
        "profile",
        "concurrency",
        "context_tokens",
        "context_bucket",
        "lat_ttft_ms",
        "lat_e2e_ms",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p50_e2e_ms",
        "p95_e2e_ms",
        "p99_e2e_ms",
        "nvme_r_await_ms",
        "nvme_util_pct",
        "nvme_rMBs",
        "nvme_rps",
        "metrics_ts",
        "sysmon_ts",
        "ts_delta",
    ]
    write_csv(path, headers, filtered)


def write_h0_concurrency_vs_latency(rows: List[Dict[str, object]]) -> None:
    # H0: Figure 8.0 – concurrency sweep vs TTFT/E2E with collapse markers (Stack A+B).
    filtered = [r for r in rows if "H0" in str(r.get("run_id", ""))]
    if not filtered:
        return
    path = OUT_DIR / "H0_concurrency_vs_latency.csv"
    headers = [
        "run_id",
        "stack",
        "profile",
        "workload",
        "context_tokens",
        "concurrency",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p50_e2e_ms",
        "p95_e2e_ms",
        "p99_e2e_ms",
        "samples",
        "collapse",
        "storage_knee",
        "tier2_bytes_in_total",
        "tier2_bytes_out_total",
        "tier2_fetch_p95_ms",
        "tier2_fetch_p99_ms",
        "evictions",
        "prefetches",
    ]
    sorted_rows = sorted(filtered, key=lambda r: (str(r.get("stack", "")), float(r.get("concurrency") or 0)))
    write_csv(path, headers, sorted_rows)


def write_h2_latency_vs_context(rows: List[Dict[str, object]]) -> None:
    # H2A/H2B: Figure 8.2 – context length vs TTFT/E2E; Stack A flat NVMe, Stack B storage knee.
    filtered = [r for r in rows if "H2" in str(r.get("run_id", ""))]
    if not filtered:
        return
    path = OUT_DIR / "H2_latency_vs_context.csv"
    headers = [
        "run_id",
        "stack",
        "profile",
        "workload",
        "concurrency",
        "context_tokens",
        "tokenizer",
        "max_input_len",
        "input_len_margin",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p50_e2e_ms",
        "p95_e2e_ms",
        "p99_e2e_ms",
        "tier2_bytes_in_total",
        "tier2_bytes_out_total",
        "tier2_fetch_p95_ms",
        "tier2_fetch_p99_ms",
        "storage_knee",
        "storage_knee_ts",
    ]
    sorted_rows = sorted(filtered, key=lambda r: (str(r.get("stack", "")), float(r.get("context_tokens") or 0)))
    write_csv(path, headers, sorted_rows)


def write_h5_workingset_scaling(rows: List[Dict[str, object]]) -> None:
    # H5: Figure 8.5 – KV working-set growth via adapter churn vs p95/p99 and tier2 activity.
    filtered = [r for r in rows if "H5" in str(r.get("run_id", ""))]
    if not filtered:
        return
    path = OUT_DIR / "H5_workingset_scaling.csv"
    headers = [
        "run_id",
        "stack",
        "profile",
        "workload",
        "concurrency",
        "context_tokens",
        "lora_adapter_count",
        "lora_churn_mode",
        "lora_hot_ratio",
        "lora_hot_prob",
        "burstiness",
        "burst_pause_s",
        "mix_short_pct",
        "mix_medium_pct",
        "mix_long_pct",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p50_e2e_ms",
        "p95_e2e_ms",
        "p99_e2e_ms",
        "tier2_bytes_in_total",
        "tier2_bytes_out_total",
        "tier2_fetch_p95_ms",
        "tier2_fetch_p99_ms",
        "evictions",
        "prefetches",
    ]
    sorted_rows = sorted(filtered, key=lambda r: float(r.get("lora_adapter_count") or -1))
    write_csv(path, headers, sorted_rows)


def write_h8_hero_comparison(rows: List[Dict[str, object]]) -> None:
    # H8: Section 6 hero comparison – Stack A vs Stack B at equal UMA budgets.
    filtered = [r for r in rows if "H8" in str(r.get("run_id", ""))]
    if not filtered:
        return
    path = OUT_DIR / "H8_hero_comparison.csv"
    headers = [
        "run_id",
        "stack",
        "profile",
        "workload",
        "context_tokens",
        "concurrency",
        "mix_short_pct",
        "mix_medium_pct",
        "mix_long_pct",
        "burstiness",
        "burst_pause_s",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p50_e2e_ms",
        "p95_e2e_ms",
        "p99_e2e_ms",
        "tier2_bytes_in_total",
        "tier2_bytes_out_total",
        "tier2_fetch_p95_ms",
        "tier2_fetch_p99_ms",
        "evictions",
        "prefetches",
        "collapse",
        "storage_knee",
    ]
    sorted_rows = sorted(filtered, key=lambda r: (str(r.get("stack", "")), float(r.get("context_tokens") or 0)))
    write_csv(path, headers, sorted_rows)


def write_h9_rehydration_cost(rows: List[Dict[str, object]]) -> None:
    # H9: Section 6 re-hydration – build vs resume session latency and KV spill/reload cost.
    filtered = [r for r in rows if "H9" in str(r.get("run_id", ""))]
    if not filtered:
        return
    path = OUT_DIR / "H9_rehydration_cost.csv"
    headers = [
        "run_id",
        "stack",
        "profile",
        "phase",
        "session_phase",
        "workload",
        "context_tokens",
        "concurrency",
        "session_min_turns",
        "session_max_turns",
        "session_resume_turns",
        "session_idle_s",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p50_e2e_ms",
        "p95_e2e_ms",
        "p99_e2e_ms",
        "tier2_bytes_in_total",
        "tier2_bytes_out_total",
        "tier2_fetch_p95_ms",
        "tier2_fetch_p99_ms",
        "evictions",
        "prefetches",
        "storage_knee",
        "storage_knee_ts",
    ]
    sorted_rows = sorted(filtered, key=lambda r: str(r.get("session_phase", "")) or str(r.get("phase", "")))
    write_csv(path, headers, sorted_rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    h4b_rows: List[Dict[str, object]] = []
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        for metrics_path in sorted(root.rglob("metrics.jsonl")):
            agg = aggregate(metrics_path.parent)
            if agg:
                rows.append(agg)
                h4b_rows.extend(join_latency_with_nvme(metrics_path.parent, agg))
    if not rows:
        return 0
    write_summary(rows)
    write_uwork(rows)
    write_h0_concurrency_vs_latency(rows)
    write_h2_latency_vs_context(rows)
    write_h5_workingset_scaling(rows)
    write_h8_hero_comparison(rows)
    write_h9_rehydration_cost(rows)
    write_h4b_latency_vs_nvme(h4b_rows)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

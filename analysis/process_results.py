#!/usr/bin/env python3
"""v3 analysis for H0/H2A (Test_Plan_v3.0 Appendix C)."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
RUN_ROOTS = [
    ROOT / "results",
    ROOT / "runs",
]
OUT_DIR = Path(__file__).resolve().parent / "figures"


def percentile(values: List[float], pct: float) -> float | None:
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


def load_dynkv_kv(run_dir: Path) -> Dict[str, float]:
    """Load normalized KV CSV (if present) and derive aggregates using available metrics."""
    path = run_dir / "dynkv_kv.csv"
    if not path.exists():
        return {}
    total_offload = 0.0
    total_onboard = 0.0
    max_hit_rate_disk = None
    max_hit_rate_host = None
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            def val(key: str):
                v = row.get(key)
                if v in (None, ""):
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            off_d2h = val("derived_offload_bytes_d2h")
            off_h2d = val("derived_offload_bytes_h2d")
            off_d2d = val("derived_offload_bytes_d2d")
            on_d2d = val("derived_onboard_bytes_d2d")
            on_h2d = val("derived_onboard_bytes_h2d")

            for v in (off_d2h, off_h2d, off_d2d):
                if v is not None:
                    total_offload += v
            for v in (on_d2d, on_h2d):
                if v is not None:
                    total_onboard += v

            hr_disk = val("hit_rate_disk")
            hr_host = val("hit_rate_host")
            if hr_disk is not None:
                max_hit_rate_disk = hr_disk if max_hit_rate_disk is None else max(max_hit_rate_disk, hr_disk)
            if hr_host is not None:
                max_hit_rate_host = hr_host if max_hit_rate_host is None else max(max_hit_rate_host, hr_host)

    return {
        "kv_total_offload_bytes": total_offload,
        "kv_total_onboard_bytes": total_onboard,
        "kv_max_hit_rate_disk": max_hit_rate_disk,
        "kv_max_hit_rate_host": max_hit_rate_host,
    }


def load_nvme(run_dir: Path) -> Dict[str, float]:
    """Load NVMe JSONL samples and derive peak utilization/latency/queue depth."""
    path = run_dir / "nvme.jsonl"
    if not path.exists():
        return {}
    util = []
    await_r = []
    await_w = []
    qdepth = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = rec.get("data", {})
            util.append(data.get("util_pct", 0.0))
            await_r.append(data.get("r_await_ms", 0.0))
            await_w.append(data.get("w_await_ms", 0.0))
            qdepth.append(data.get("avg_queue_depth", 0.0))
    return {
        "nvme_max_util_pct": max(util) if util else None,
        "nvme_max_r_await_ms": max(await_r) if await_r else None,
        "nvme_max_w_await_ms": max(await_w) if await_w else None,
        "nvme_max_queue_depth": max(qdepth) if qdepth else None,
    }


def aggregate(run_dir: Path) -> Dict[str, object]:
    metrics = load_jsonl(run_dir / "metrics.jsonl")
    sysmon = load_jsonl(run_dir / "sysmon.jsonl")
    if not metrics:
        return {}
    ttft = [m["lat_ttft_ms"] for m in metrics if m.get("rc") == "ok"]
    e2e = [m["lat_e2e_ms"] for m in metrics if m.get("rc") == "ok"]
    if not ttft or not e2e:
        return {}
    sample = metrics[0]
    result = {
        "run_id": sample.get("run_id"),
        "stack": sample.get("stack"),
        "model": sample.get("model"),
        "workload": sample.get("workload"),
        "context_tokens": sample.get("context_tokens"),
        "concurrency": sample.get("concurrency"),
        "p50_ttft_ms": percentile(ttft, 50),
        "p95_ttft_ms": percentile(ttft, 95),
        "p99_ttft_ms": percentile(ttft, 99),
        "p50_e2e_ms": percentile(e2e, 50),
        "p95_e2e_ms": percentile(e2e, 95),
        "p99_e2e_ms": percentile(e2e, 99),
        "samples": len(e2e),
        "sysmon_samples": len(sysmon),
    }
    result.update(load_dynkv_kv(run_dir))
    result.update(load_nvme(run_dir))
    return result


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        for metrics_path in sorted(root.rglob("metrics.jsonl")):
            agg = aggregate(metrics_path.parent)
            if agg:
                rows.append(agg)
    summary_path = OUT_DIR / "summary_v3.csv"
    headers = [
        "run_id",
        "stack",
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
        "kv_total_offload_bytes",
        "kv_total_onboard_bytes",
        "kv_max_hit_rate_disk",
        "kv_max_hit_rate_host",
        "nvme_max_util_pct",
        "nvme_max_r_await_ms",
        "nvme_max_w_await_ms",
        "nvme_max_queue_depth",
    ]
    with summary_path.open("w", encoding="utf-8") as out:
        out.write(",".join(headers) + "\n")
        for row in rows:
            out.write(",".join(str(row.get(h, "")) for h in headers) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

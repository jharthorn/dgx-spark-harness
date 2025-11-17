#!/usr/bin/env python3
"""v3 analysis for H0/H2A (Test_Plan_v3.0 Appendix C)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs" / "v3"
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
    return {
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for run_dir in sorted(RUNS_ROOT.glob("*")):
        agg = aggregate(run_dir)
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
    ]
    with summary_path.open("w", encoding="utf-8") as out:
        out.write(",".join(headers) + "\n")
        for row in rows:
            out.write(",".join(str(row.get(h, "")) for h in headers) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Fetch Dynamo KVBM Prometheus metrics and emit v3.3 dynkv.jsonl + normalized CSV.

Exporter limitation (current Stack B build):
- Only exposes: matched_tokens; host/disk hit rates; offload_blocks_{d2h,h2d,d2d};
  onboard_blocks_{d2d,h2d}; optional disk_bytes_{read,written}.
- Missing: explicit per-tier hits/misses, tier fetch latency, evictions/prefetches.

Harness approach:
- Emit `dynkv.jsonl` matching Test_Plan_v3.3 schema with best-effort values
  (fill missing counters with zeros).
- Emit `dynkv_kv.csv` with derived bytes per sample for plotting storage knees.
- Block size is taken from --block-size-bytes, env DYN_KVBM_KV_BLOCK_SIZE_BYTES,
  or defaults to 65536 bytes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import re
from typing import Dict, Mapping

PROM_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([0-9.eE+-]+)$")

METRIC_MAP = {
    "kvbm_host_cache_hit_rate": "host_hit_rate",
    "kvbm_disk_cache_hit_rate": "disk_hit_rate",
    "kvbm_offload_blocks_d2h": "offload_d2h",
    "kvbm_offload_blocks_h2d": "offload_h2d",
    "kvbm_offload_blocks_d2d": "offload_d2d",
    "kvbm_onboard_blocks_d2d": "onboard_d2d",
    "kvbm_onboard_blocks_h2d": "onboard_h2d",
    "kvbm_matched_tokens": "matched_tokens",
    "kvbm_disk_bytes_read": "disk_bytes_read",
    "kvbm_disk_bytes_written": "disk_bytes_written",
    # Optional latency counters (if exporter supports)
    "kvbm_disk_fetch_p50_ms": "fetch_p50_ms",
    "kvbm_disk_fetch_p95_ms": "fetch_p95_ms",
    "kvbm_disk_fetch_p99_ms": "fetch_p99_ms",
    "kvbm_cache_evictions": "evictions",
    "kvbm_cache_prefetches": "prefetches",
}


def fetch_metrics(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_prometheus(text: str) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = PROM_LINE_RE.match(line)
        if not m:
            continue
        name, value = m.groups()
        try:
            parsed[name] = float(value)
        except ValueError:
            continue
    return parsed


def build_payload(metrics: Mapping[str, float]) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for src_name, dst_name in METRIC_MAP.items():
        if src_name in metrics:
            payload[dst_name] = metrics[src_name]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamo KV Prometheus JSONL sampler")
    parser.add_argument("--url", default="http://127.0.0.1:6880/metrics", help="Prometheus metrics URL")
    parser.add_argument("--interval-seconds", type=float, default=0.2, help="Sampling interval seconds")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--kv-output",
        default=None,
        help="Optional normalized KV CSV output (defaults to <output> with _kv.csv suffix)",
    )
    parser.add_argument(
        "--block-size-bytes",
        type=int,
        default=None,
        help="KV block size in bytes; overrides env DYN_KVBM_KV_BLOCK_SIZE_BYTES; default 65536",
    )
    return parser.parse_args()


def append_jsonl(path: str, record: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def delta(curr: Mapping[str, float], prev: Mapping[str, float] | None, key: str) -> float:
    if prev is None:
        return 0.0
    return max(0.0, curr.get(key, 0.0) - prev.get(key, 0.0))


def main() -> int:
    args = parse_args()
    kv_output = args.kv_output or args.output.replace(".jsonl", "_kv.csv")
    kv_fields = [
        "ts",
        "hit_rate_host",
        "hit_rate_disk",
        "tier2_bytes_in",
        "tier2_bytes_out",
        "fetch_p50_ms",
        "fetch_p95_ms",
        "fetch_p99_ms",
        "kv_block_size_bytes",
        "tier2_bytes_in_delta",
        "tier2_bytes_out_delta",
        "kv_blocks_evicted_delta",
    ]
    block_size = args.block_size_bytes
    if block_size is None:
        env_block = os.environ.get("DYN_KVBM_KV_BLOCK_SIZE_BYTES")
        if env_block:
            try:
                block_size = int(env_block)
            except ValueError:
                block_size = None
    if block_size is None:
        block_size = 65536
        print("[dynkv_ingest] Block size not provided; defaulting to 65536 bytes", file=sys.stderr)
    last_counters: Dict[str, float] | None = None
    while True:
        ts = time.time()
        try:
            raw = fetch_metrics(args.url)
            parsed = parse_prometheus(raw)
            counters = {
                "matched_tokens": parsed.get("kvbm_matched_tokens", 0.0),
                "hit_rate_host": parsed.get("kvbm_host_cache_hit_rate", 0.0),
                "hit_rate_disk": parsed.get("kvbm_disk_cache_hit_rate", 0.0),
                "offload_blocks_d2h": parsed.get("kvbm_offload_blocks_d2h", 0.0),
                "offload_blocks_h2d": parsed.get("kvbm_offload_blocks_h2d", 0.0),
                "offload_blocks_d2d": parsed.get("kvbm_offload_blocks_d2d", 0.0),
                "onboard_blocks_d2d": parsed.get("kvbm_onboard_blocks_d2d", 0.0),
                "onboard_blocks_h2d": parsed.get("kvbm_onboard_blocks_h2d", 0.0),
                "disk_bytes_read": parsed.get("kvbm_disk_bytes_read", 0.0),
                "disk_bytes_written": parsed.get("kvbm_disk_bytes_written", 0.0),
                "fetch_p50_ms": parsed.get("kvbm_disk_fetch_p50_ms", 0.0),
                "fetch_p95_ms": parsed.get("kvbm_disk_fetch_p95_ms", 0.0),
                "fetch_p99_ms": parsed.get("kvbm_disk_fetch_p99_ms", 0.0),
                "evictions": parsed.get("kvbm_cache_evictions", 0.0),
                "prefetches": parsed.get("kvbm_cache_prefetches", 0.0),
            }

            # Derived bytes per sample using counter deltas.
            derived_offload = {
                "d2h": delta(counters, last_counters, "offload_blocks_d2h") * block_size,
                "h2d": delta(counters, last_counters, "offload_blocks_h2d") * block_size,
                "d2d": delta(counters, last_counters, "offload_blocks_d2d") * block_size,
            }
            derived_onboard = {
                "d2d": delta(counters, last_counters, "onboard_blocks_d2d") * block_size,
                "h2d": delta(counters, last_counters, "onboard_blocks_h2d") * block_size,
            }
            bytes_in = delta(counters, last_counters, "disk_bytes_read") or (derived_onboard["d2d"] + derived_onboard["h2d"])
            bytes_out = delta(counters, last_counters, "disk_bytes_written") or (derived_offload["d2h"] + derived_offload["h2d"] + derived_offload["d2d"])
            evictions_delta = delta(counters, last_counters, "evictions")
            prefetch_delta = delta(counters, last_counters, "prefetches")

            record = {
                "ts": ts,
                "kv_block_size_bytes": block_size,
                "tier0": {"hits": 0, "misses": 0, "bytes_in": 0, "bytes_out": 0},
                "tier1": {"hits": 0, "misses": 0, "bytes_in": 0, "bytes_out": 0},
                "tier2": {
                    "hits": 0,
                    "misses": 0,
                    "bytes_in": int(bytes_in),
                    "bytes_out": int(bytes_out),
                    "fetch_p50_ms": counters.get("fetch_p50_ms", 0.0),
                    "fetch_p95_ms": counters.get("fetch_p95_ms", 0.0),
                    "fetch_p99_ms": counters.get("fetch_p99_ms", 0.0),
                },
                "evictions": int(evictions_delta),
                "prefetches": int(prefetch_delta),
                # Expose deltas explicitly for downstream analysis (storage-knee heuristics).
                "tier2_bytes_in_delta": int(bytes_in),
                "tier2_bytes_out_delta": int(bytes_out),
                "kv_blocks_evicted_delta": int(evictions_delta),
            }
            append_jsonl(args.output, record)

            csv_row = {
                "ts": ts,
                "hit_rate_host": counters.get("hit_rate_host", 0.0),
                "hit_rate_disk": counters.get("hit_rate_disk", 0.0),
                "tier2_bytes_in": bytes_in,
                "tier2_bytes_out": bytes_out,
                "fetch_p50_ms": counters.get("fetch_p50_ms", 0.0),
                "fetch_p95_ms": counters.get("fetch_p95_ms", 0.0),
                "fetch_p99_ms": counters.get("fetch_p99_ms", 0.0),
                "kv_block_size_bytes": block_size,
                "tier2_bytes_in_delta": bytes_in,
                "tier2_bytes_out_delta": bytes_out,
                "kv_blocks_evicted_delta": evictions_delta,
            }
            exists = os.path.exists(kv_output)
            with open(kv_output, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=kv_fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(csv_row)

            last_counters = counters
        except Exception as exc:  # noqa: BLE001
            print(f"[dynkv_ingest] sample failed: {exc}", file=sys.stderr)
        time.sleep(max(args.interval_seconds, 0.05))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Fetch Dynamo KVBM Prometheus metrics and emit JSONL + normalized KV CSV.

Exporter limitation (current Stack B build):
- Only exposes: matched_tokens; host/disk hit rates; offload_blocks_{d2h,h2d,d2d};
  onboard_blocks_{d2d,h2d}.
- Does NOT expose: per-tier hits/misses/bytes, tier fetch latency, evictions/prefetches,
  or block-level timings beyond these counters. No env flag to enable them.

Harness approach:
- Keep emitting legacy JSONL (`dynkv.jsonl`) for backward compatibility.
- Emit `dynkv_kv.csv` with the exposed counters plus derived byte estimates:
    - matched_tokens, hit_rate_host, hit_rate_disk
    - offload/onboard block counters (cumulative)
    - derived_offload_bytes_* and derived_onboard_bytes_* computed as deltas *
      kv_block_size_bytes
- Block size is taken from --block-size-bytes, env DYN_KVBM_KV_BLOCK_SIZE_BYTES, or
  defaults to 65536 bytes with a one-time warning.
- Missing metrics are filled with zero; ingestion continues on malformed lines.
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

# Map Prometheus metric names to output fields
METRIC_MAP = {
    "kvbm_host_cache_hit_rate": "host_hit_rate",
    "kvbm_disk_cache_hit_rate": "disk_hit_rate",
    "kvbm_offload_blocks_d2h": "offload_d2h",
    "kvbm_offload_blocks_h2d": "offload_h2d",
    "kvbm_offload_blocks_d2d": "offload_d2d",
    "kvbm_onboard_blocks_d2d": "onboard_d2d",
    "kvbm_onboard_blocks_h2d": "onboard_h2d",
    "kvbm_matched_tokens": "matched_tokens",
    # The following may or may not be present depending on build
    "kvbm_disk_bytes_read": "disk_bytes_read",
    "kvbm_disk_bytes_written": "disk_bytes_written",
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


def main() -> int:
    args = parse_args()
    kv_output = args.kv_output or args.output.replace(".jsonl", "_kv.csv")
    kv_fields = [
        "ts",
        "matched_tokens",
        "hit_rate_host",
        "hit_rate_disk",
        "offload_blocks_d2h",
        "offload_blocks_h2d",
        "offload_blocks_d2d",
        "onboard_blocks_d2d",
        "onboard_blocks_h2d",
        "derived_offload_bytes_d2h",
        "derived_offload_bytes_h2d",
        "derived_offload_bytes_d2d",
        "derived_onboard_bytes_d2d",
        "derived_onboard_bytes_h2d",
        "kv_block_size_bytes",
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
    last_counters = None
    while True:
        ts = time.time()
        try:
            raw = fetch_metrics(args.url)
            parsed = parse_prometheus(raw)
            payload = build_payload(parsed)
            if payload:
                append_jsonl(args.output, {"ts": ts, "source": "dynkv", "data": {"kvbm": payload}})
            # Prepare normalized row with derived bytes using counter deltas
            counters = {
                "matched_tokens": parsed.get("kvbm_matched_tokens", 0.0),
                "hit_rate_host": parsed.get("kvbm_host_cache_hit_rate", 0.0),
                "hit_rate_disk": parsed.get("kvbm_disk_cache_hit_rate", 0.0),
                "offload_blocks_d2h": parsed.get("kvbm_offload_blocks_d2h", 0.0),
                "offload_blocks_h2d": parsed.get("kvbm_offload_blocks_h2d", 0.0),
                "offload_blocks_d2d": parsed.get("kvbm_offload_blocks_d2d", 0.0),
                "onboard_blocks_d2d": parsed.get("kvbm_onboard_blocks_d2d", 0.0),
                "onboard_blocks_h2d": parsed.get("kvbm_onboard_blocks_h2d", 0.0),
            }
            derived = {k: 0.0 for k in ["derived_offload_bytes_d2h", "derived_offload_bytes_h2d", "derived_offload_bytes_d2d", "derived_onboard_bytes_d2d", "derived_onboard_bytes_h2d"]}
            if last_counters is not None:
                def delta(key: str) -> float:
                    return max(0.0, counters.get(key, 0.0) - last_counters.get(key, 0.0))
                derived["derived_offload_bytes_d2h"] = delta("offload_blocks_d2h") * block_size
                derived["derived_offload_bytes_h2d"] = delta("offload_blocks_h2d") * block_size
                derived["derived_offload_bytes_d2d"] = delta("offload_blocks_d2d") * block_size
                derived["derived_onboard_bytes_d2d"] = delta("onboard_blocks_d2d") * block_size
                derived["derived_onboard_bytes_h2d"] = delta("onboard_blocks_h2d") * block_size
            last_counters = counters
            norm_row = {"ts": ts, **counters, **derived, "kv_block_size_bytes": block_size}
            exists = os.path.exists(kv_output)
            with open(kv_output, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=kv_fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(norm_row)
        except Exception as exc:  # noqa: BLE001
            print(f"[dynkv_ingest] sample failed: {exc}", file=sys.stderr)
        time.sleep(max(args.interval_seconds, 0.05))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

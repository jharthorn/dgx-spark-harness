"""Fetch Dynamo KVBM Prometheus metrics and emit JSONL."""

from __future__ import annotations

import argparse
import json
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
    return parser.parse_args()


def append_jsonl(path: str, record: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> int:
    args = parse_args()
    while True:
        ts = time.time()
        try:
            raw = fetch_metrics(args.url)
            parsed = parse_prometheus(raw)
            payload = build_payload(parsed)
            if payload:
                append_jsonl(args.output, {"ts": ts, "source": "dynkv", "data": {"kvbm": payload}})
        except Exception as exc:  # noqa: BLE001
            print(f"[dynkv_ingest] sample failed: {exc}", file=sys.stderr)
        time.sleep(max(args.interval_seconds, 0.05))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

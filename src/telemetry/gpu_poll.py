"""GPU sampler using nvidia-smi to emit JSONL snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from typing import Dict, List


def query_gpus() -> List[Dict[str, object]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[gpu_poll] nvidia-smi failed: {exc}", file=sys.stderr)
        return []

    gpus: List[Dict[str, object]] = []
    reader = csv.reader(out.splitlines())
    for row in reader:
        if len(row) < 4:
            continue
        uuid, mem_used, mem_total, util = [item.strip() for item in row[:4]]
        try:
            gpus.append(
                {
                    "uuid": uuid,
                    "mem_used_mb": float(mem_used),
                    "mem_total_mb": float(mem_total),
                    "util_pct": float(util),
                }
            )
        except ValueError:
            continue
    return gpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU telemetry sampler (nvidia-smi)")
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
        gpus = query_gpus()
        if gpus:
            record = {"ts": ts, "source": "gpu", "data": {"gpus": gpus}}
            try:
                append_jsonl(args.output, record)
            except Exception as exc:  # noqa: BLE001
                print(f"[gpu_poll] write failed: {exc}", file=sys.stderr)
        time.sleep(max(args.interval_seconds, 0.05))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

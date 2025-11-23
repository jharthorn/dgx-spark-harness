"""NVMe sampler that writes JSONL snapshots using iostat."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Dict, List, Optional


def run_iostat(device: str) -> Optional[Dict[str, float]]:
    try:
        # Single sample (-y to skip first, -d extended, -m MB/sec).
        out = subprocess.check_output(["iostat", "-y", "-dx", "-m", "1", "1"], text=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[nvme_iostat] iostat failed: {exc}", file=sys.stderr)
        return None

    lines = [line for line in out.splitlines() if line.strip()]
    header_idx = next((i for i, l in enumerate(lines) if l.lower().startswith("device")), None)
    if header_idx is None or header_idx + 1 >= len(lines):
        return None
    headers = lines[header_idx].split()
    data_lines = lines[header_idx + 1 :]
    target_line = None
    for line in data_lines:
        if line.split()[0] == device:
            target_line = line
            break
    if target_line is None:
        return None
    fields = target_line.split()
    col_map = {name: idx for idx, name in enumerate(headers)}

    def get(name: str) -> float:
        try:
            return float(fields[col_map[name]])
        except Exception:
            return 0.0

    return {
        "device": device,
        "r_mb_s": get("rMB/s") if "rMB/s" in col_map else get("rkB/s") / 1024,
        "w_mb_s": get("wMB/s") if "wMB/s" in col_map else get("wkB/s") / 1024,
        "r_await_ms": get("r_await") if "r_await" in col_map else get("await"),
        "w_await_ms": get("w_await") if "w_await" in col_map else get("await"),
        "util_pct": get("%util") if "%util" in col_map else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NVMe iostat JSONL sampler")
    parser.add_argument("--device", default="nvme0n1", help="Block device to sample (default: nvme0n1)")
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
        sample = run_iostat(args.device)
        if sample:
            record = {"ts": ts, "source": "nvme", "data": sample}
            try:
                append_jsonl(args.output, record)
            except Exception as exc:  # noqa: BLE001
                print(f"[nvme_iostat] write failed: {exc}", file=sys.stderr)
        time.sleep(max(args.interval_seconds, 0.05))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

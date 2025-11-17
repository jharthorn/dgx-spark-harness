#!/usr/bin/env python3
"""Backfill helper scaffold (Test_Plan_v3.0 Appendix C)."""

from __future__ import annotations

import json
from pathlib import Path

RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs" / "v3"


def main() -> int:
    for run_dir in RUNS_ROOT.glob("*"):
        metrics_path = run_dir / "metrics.jsonl"
        summary_path = run_dir / "summary.json"
        if not metrics_path.exists():
            continue
        summary = {
            "run_id": run_dir.name,
            "metrics_path": str(metrics_path),
            "note": "TODO: compute aggregates and persist for analysis pipeline.",
        }
        summary_path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

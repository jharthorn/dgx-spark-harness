#!/usr/bin/env python3
import json, sys, os
from pathlib import Path
import pandas as pd
import numpy as np # Import numpy

# analysis/backfill_summary.py
# Reads a run's telemetry.csv and summary.json, computes telemetry averages,
# and writes them back into the summary.json's 'avg' block.

HARNESS_DIR = Path(os.environ.get("HARNESS_DIR", "/harness"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(HARNESS_DIR / "results")))

def backfill_summary(run_id):
    """Computes telemetry averages and updates the summary.json file."""
    
    summary_path = RESULTS_DIR / f"{run_id}_summary.json"
    telemetry_path = RESULTS_DIR / f"{run_id}_telemetry.csv"

    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}", file=sys.stderr)
        return
    
    if not telemetry_path.exists():
        print(f"Error: Telemetry file not found: {telemetry_path}", file=sys.stderr)
        return

    try:
        # Read both files
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        
        # Check if summary is just a dummy file (e.g., from a skipped H6 run)
        if summary_data.get("requests_total", 0) == 0:
             print(f"Skipping backfill for {run_id} (dummy summary file).")
             return

        df_telemetry = pd.read_csv(telemetry_path)
        
        # Compute averages from telemetry, handling potential NA values
        avg_io_wait = df_telemetry['vm_wa'].mean()
        avg_qu_sz = df_telemetry['iostat_avg_qu_sz'].mean()
        avg_gpu_util = df_telemetry['gpu_util_pct'].mean()

        # Update the 'avg' block
        # Keep 'rps' from loadgen, overwrite the others
        summary_data['avg']['io_wait_pct'] = round(avg_io_wait, 2) if pd.notna(avg_io_wait) else None
        summary_data['avg']['qu_sz'] = round(avg_qu_sz, 2) if pd.notna(avg_qu_sz) else None
        summary_data['avg']['gpu_util_pct'] = round(avg_gpu_util, 2) if pd.notna(avg_gpu_util) else None

        # Write the updated summary back
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        print(f"Backfilled summary for {run_id}")

    except Exception as e:
        print(f"Error backfilling {run_id}: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_id_1> [run_id_2] ...", file=sys.stderr)
        sys.exit(1)
        
    for run_id in sys.argv[1:]:
        backfill_summary(run_id)
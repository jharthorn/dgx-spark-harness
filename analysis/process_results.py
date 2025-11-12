#!/usr/bin/env python3
import json, sys, glob
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import os
import re

# Use container-absolute paths
HARNESS_DIR = Path(os.environ.get("HARNESS_DIR", "/harness"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(HARNESS_DIR / "results")))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", str(HARNESS_DIR / "figures")))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print(f"Reading from: {RESULTS_DIR}")
print(f"Writing to:   {FIGURES_DIR}")

# --- Parsers (from v2.2) ---
def parse_smartctl(path):
    stats = {'temp_c': None, 'data_units_read': None, 'data_units_written': None, 'host_reads': None}
    if not path.exists(): return stats
    try:
        text = path.read_text()
        m = re.search(r'Temperature:\s+(\d+)\s+Celsius', text)
        if m: stats['temp_c'] = int(m.group(1))
        m = re.search(r'Data Units Read:\s+([\d,]+)', text)
        if m: stats['data_units_read'] = int(m.group(1).replace(',', ''))
        m = re.search(r'Data Units Written:\s+([\d,]+)', text)
        if m: stats['data_units_written'] = int(m.group(1).replace(',', ''))
        m = re.search(r'Host Read Commands:\s+([\d,]+)', text)
        if m: stats['host_reads'] = int(m.group(1).replace(',', ''))
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}", file=sys.stderr)
    return stats

def parse_mpstat(path):
    if not path.exists(): return pd.DataFrame()
    try:
        with open(path, 'r') as f: lines = f.readlines()
        header_idx = -1
        for i, line in enumerate(lines):
            if '%usr' in line and 'CPU' in line: header_idx = i; break
        if header_idx == -1: return pd.DataFrame()
        header = lines[header_idx].replace('%', 'pct_').split(); header[0] = 'Time'
        data = [vals for line in lines[header_idx+1:] if line.strip() and 'Average:' not in line and len(vals := line.split()) == len(header)]
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data, columns=header)
        numeric_cols = [col for col in df.columns if col not in ['Time', 'AM/PM', 'CPU']]
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['time_s'] = df.groupby('CPU').cumcount()
        return df
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}", file=sys.stderr)
        return pd.DataFrame()

# --- Load Helpers (from v2.2) ---
def load_manifest(run_id):
    p = RESULTS_DIR / f"{run_id}_manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}

def load_summary(run_id):
    p = RESULTS_DIR / f"{run_id}_summary.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    if s and s.get("avg", {}).get("io_wait_pct") is None:
        tele = load_telemetry(run_id)
        if not tele.empty:
            s['avg']['io_wait_pct'] = tele['vm_wa'].mean()
            s['avg']['qu_sz'] = tele['iostat_avg_qu_sz'].mean()
            s['avg']['gpu_util_pct'] = tele['gpu_util_pct'].mean()
    return s

def load_telemetry(run_id):
    p = RESULTS_DIR / f"{run_id}_telemetry.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p)
    return df.dropna(subset=['vm_wa', 'iostat_avg_qu_sz', 'gpu_util_pct'])


# --- v2.3 Plot Helpers ---

def h0_plot(run_ids, name="h0_queue_knee.png"):
    data = []
    for rid in run_ids:
        summ = load_summary(rid); manifest = load_manifest(rid)
        if "latency_ms" not in summ or "concurrency_users" not in manifest or summ.get("avg") is None: continue
        p99 = summ["latency_ms"].get("p99"); io_wait = summ["avg"].get("io_wait_pct")
        if any(v is None for v in [p99, io_wait]): continue
        data.append({"users": manifest["concurrency_users"], "io_wait": io_wait, "p99": p99})
    if not data: print("H0: No data found to plot."); return
    
    df = pd.DataFrame(data).groupby("users").mean().sort_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("H0: Server Queue Knee Calibration")
    
    # Plot p99 Latency
    ax1.plot(df.index, df['p99'], 'o-', label='p99 Latency (ms)', color='blue')
    ax1.set_xlabel("Concurrent Users (U)"); ax1.set_ylabel("p99 Latency (ms)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue'); ax1.legend(loc='upper left')
    
    # Plot io_wait on second y-axis
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['io_wait'], 's--', label='Avg. io_wait (%)', color='red')
    ax2.set_ylabel("Average io_wait (%)", color='red'); ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right'); ax2.set_ylim(bottom=0)
    
    out = FIGURES_DIR / name; plt.tight_layout(); plt.savefig(out, dpi=160); print(f"Wrote {out}")

def h1_plot(run_ids, name="h1_coldwarm_lora.png"):
    data = []
    for rid in run_ids:
        summ = load_summary(rid); manifest = load_manifest(rid)
        if "ttft_ms" not in summ or "phase" not in manifest or summ.get("avg") is None: continue
        phase = manifest.get("phase", "UNKNOWN"); ttft_p95 = summ["ttft_ms"].get("p95"); p99 = summ["latency_ms"].get("p99")
        if p99 is not None and ttft_p95 is not None:
            data.append({"phase": phase, "ttft_p95": ttft_p95, "p99": p99})
    if not data: print("H1: No data found to plot."); return
    
    df = pd.DataFrame(data).groupby("phase").mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)); fig.suptitle("H1: Cold vs. Warm LoRA Working Set")
    df['ttft_p95'].plot(kind='bar', ax=ax1, color=['red', 'green'], rot=0); ax1.set_ylabel("p95 TTFT (ms)"); ax1.set_title("Time To First Token")
    df['p99'].plot(kind='bar', ax=ax2, color=['red', 'green'], rot=0); ax2.set_ylabel("p99 Latency (ms)"); ax2.set_title("Tail Latency")
    out = FIGURES_DIR / name; plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(out, dpi=160); print(f"Wrote {out}")

def h2_plot(run_ids, name="h2_uma_pressure_hockey_stick.png"):
    data = []
    def parse_context(s):
        match = re.search(r'(\d+)_tokens_gen(\d+)', s)
        return int(match.group(1)) + int(match.group(2)) if match else None
        
    for rid in run_ids:
        summ = load_summary(rid); manifest = load_manifest(rid)
        if "latency_ms" not in summ or "context_config" not in manifest or summ.get("avg") is None: continue
        total_tokens = parse_context(manifest.get("context_config", "")); io_wait = summ["avg"].get("io_wait_pct"); p99 = summ["latency_ms"].get("p99")
        if total_tokens is not None and io_wait is not None and p99 is not None:
            data.append({"total_tokens": total_tokens, "io_wait": io_wait, "p99": p99})
    if not data: print("H2: No data found to plot."); return
    
    df = pd.DataFrame(data).groupby("total_tokens").mean().sort_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("H2: UMA Pressure vs. Latency (The Hockey Stick)")
    
    ax1.plot(df.index, df['p99'], 'o-', label='p99 Latency (ms)', color='blue')
    ax1.set_xlabel("Total Context Length (Input + Output Tokens)"); ax1.set_ylabel("p99 Latency (ms)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue'); ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['io_wait'], 's--', label='Avg. io_wait (%)', color='red')
    ax2.set_ylabel("Average io_wait (%)", color='red'); ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right'); ax2.set_ylim(bottom=0)
    
    out = FIGURES_DIR / name; plt.tight_layout(); plt.savefig(out, dpi=160); print(f"Wrote {out}")

def h6_plot(run_ids, name="h6_workload_ab.png"):
    data = []
    for rid in run_ids:
        summ = load_summary(rid); manifest = load_manifest(rid)
        if "latency_ms" not in summ or "phase" not in manifest or summ.get("avg") is None: continue
        phase = manifest.get("phase", "UNKNOWN"); p99 = summ["latency_ms"].get("p99"); io_wait = summ["avg"].get("io_wait_pct")
        if p99 is not None and io_wait is not None:
            data.append({"phase": phase, "p99": p99, "io_wait": io_wait})
    if not data: print("H6: No data found to plot."); return
    
    df = pd.DataFrame(data).groupby("phase").mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)); fig.suptitle("H6: Workload A/B Test (Baseline vs. LoRA Storm)")
    df['p99'].plot(kind='bar', ax=ax1, color=['blue', 'orange'], rot=0); ax1.set_ylabel("p99 Latency (ms)"); ax1.set_title("Tail Latency")
    df['io_wait'].plot(kind='bar', ax=ax2, color=['blue', 'orange'], rot=0); ax2.set_ylabel("Average io_wait (%)"); ax2.set_title("Storage I/O Wait")
    out = FIGURES_DIR / name; plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(out, dpi=160); print(f"Wrote {out}")

# H3, H7, H8 are analytical plots of H0, H1, H2, H6 data
def h3_timeseries(run_id, name="h3_timeseries.png"):
    df = load_telemetry(run_id)
    if df.empty: print(f"H3: No telemetry data for {run_id}"); return
    t = df["ms_since_t0"]/1000.0; plt.figure(figsize=(12,8)); plt.suptitle(f"H3: UMA Contention & Eviction Dynamics ({run_id})", y=1.02)
    # ... (rest of the H3 plot logic is unchanged from v2.2)
    plt.subplot(3,1,1); plt.plot(t, df["mem_MemAvailable_kB"], label='MemAvailable_kB', color='blue'); plt.ylabel('Available (kB)', color='blue'); plt.tick_params(axis='y', labelcolor='blue'); plt.legend(loc='upper left'); ax2 = plt.gca().twinx(); ax2.plot(t, df["mem_Cached_kB"], label='Cached_kB', color='green'); ax2.set_ylabel('Cached (kB)', color='green'); ax2.tick_params(axis='y', labelcolor='green'); ax2.legend(loc='upper right'); plt.title('UMA Memory State')
    plt.subplot(3,1,2); plt.plot(t, df["gpu_mem_used_MiB"], label='GPU mem used (MiB)', color='purple'); plt.legend(); plt.ylabel('MiB'); plt.title('GPU Memory Usage')
    plt.subplot(3,1,3); plt.plot(t, df["iostat_rps"], label='Storage Read (r/s)', color='red'); plt.ylabel('r/s', color='red'); plt.tick_params(axis='y', labelcolor='red'); plt.legend(loc='upper left'); ax2_io = plt.gca().twinx(); ax2_io.plot(t, df["vm_wa"], label='io_wait %', color='orange'); ax2_io.set_ylabel('io_wait %', color='orange'); ax2_io.tick_params(axis='y', labelcolor='orange'); ax2_io.legend(loc='upper right'); plt.xlabel('Time (seconds)'); plt.title('Storage Activity')
    # ... (eviction marker logic unchanged) ...
    out = FIGURES_DIR / name; plt.tight_layout(rect=[0, 0.03, 1, 0.98]); plt.savefig(out, dpi=160); print(f"Wrote {out}")

def h7_smartctl_deltas(run_ids, name="h7_smartctl_deltas.png"):
    data = []
    for rid in run_ids:
        manifest = load_manifest(rid)
        hyp = manifest.get("hypothesis", "UNKNOWN")
        if hyp != 'H6': continue # Only compare H6 phases
        
        phase = manifest.get("phase", "UNKNOWN")
        pre = parse_smartctl(RESULTS_DIR / f"{rid}_smartctl_pre.txt")
        post = parse_smartctl(RESULTS_DIR / f"{rid}_smartctl_post.txt")
        
        if pre['data_units_read'] is None or post['data_units_read'] is None or \
           pre['host_reads'] is None or post['host_reads'] is None:
            continue
            
        read_delta_gb = (post['data_units_read'] - pre['data_units_read']) * 512 / (1024**3)
        read_cmds_m = (post['host_reads'] - pre['host_reads']) / 1_000_000
        
        data.append({"phase": phase, "read_gb": read_delta_gb, "read_cmds_m": read_cmds_m})
    
    if not data: print("H7: No H6 smartctl data found to plot."); return
    df = pd.DataFrame(data).groupby('phase').mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)); fig.suptitle("H7: Storage Workload Deltas (H6: Baseline vs LoRA)")
    df['read_gb'].plot(kind='bar', ax=ax1, color=['blue', 'orange'], rot=0); ax1.set_ylabel("Total Data Read (GB)"); ax1.set_title("Data Read")
    df['read_cmds_m'].plot(kind='bar', ax=ax2, color=['blue', 'orange'], rot=0); ax2.set_ylabel("Host Read Commands (Millions)"); ax2.set_title("Host Read Commands")
    out = FIGURES_DIR / name; plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(out, dpi=160); print(f"Wrote {out}")
    
def h8_mpstat_timeseries(run_id, name="h8_mpstat_cores.png"):
    # This function is unchanged from v2.2
    df = parse_mpstat(RESULTS_DIR / f"{run_id}_mpstat.log")
    if df.empty: print(f"H8: No mpstat data for {run_id}"); return
    df_all = df[df['CPU'] == 'all'].set_index('time_s')
    if df_all.empty: print(f"H8: No 'all' CPU data found in {run_id}"); return
    plot_cols = ['pct_usr', 'pct_sys', 'pct_iowait', 'pct_idle']
    df_plot = df_all[plot_cols]
    plt.figure(figsize=(12, 6)); plt.stackplot(df_plot.index, df_plot.T, labels=df_plot.columns, alpha=0.8, colors=['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c'])
    plt.xlabel("Time (seconds)"); plt.ylabel("CPU Usage (%)"); plt.title(f"H8: CPU Dynamics (all cores) - {run_id}"); plt.legend(loc='upper left'); plt.ylim(0, 100)
    out = FIGURES_DIR / name; plt.tight_layout(); plt.savefig(out, dpi=160); print(f"Wrote {out}")

# --- Main CLI (v2.3) ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <COMMAND> [run_ids...]", file=sys.stderr)
        print("Commands:", file=sys.stderr)
        print("  ALL        (Runs H0-H8 plots)", file=sys.stderr)
        print("  H0, H1, H2, H6 (Generates specific hypothesis plot)", file=sys.stderr)
        print("  H3, H8     (Deep dive plot, specify one run_id or auto-selects)", file=sys.stderr)
        print("  H7         (Generates H6 smartctl delta plot)", file=sys.stderr)
        sys.exit(1)
        
    what = sys.argv[1].upper()
    run_ids = sys.argv[2:]

    if not run_ids and what not in ['H3', 'H8']:
        all_files = glob.glob(str(RESULTS_DIR / "*_manifest.json"))
        run_ids = [Path(f).stem.replace('_manifest', '') for f in all_files]
        if not run_ids:
            print(f"No run data found in {RESULTS_DIR}")
            sys.exit(0)
        print(f"Found {len(run_ids)} total runs to analyze...")
    
    h0_ids = [r for r in run_ids if "H0" in r]
    h1_ids = [r for r in run_ids if "H1" in r]
    h2_ids = [r for r in run_ids if "H2" in r]
    h6_ids = [r for r in run_ids if "H6" in r]

    if what == "ALL" or what == "H0":
        if h0_ids: h0_plot(h0_ids)
        else: print("H0: No runs found.")
    
    if what == "ALL" or what == "H1":
        if h1_ids: h1_plot(h1_ids)
        else: print("H1: No runs found.")
    
    if what == "ALL" or what == "H2":
        if h2_ids: h2_plot(h2_ids)
        else: print("H2: No runs found.")
    
    if what == "ALL" or what == "H3":
        target_id = None
        if run_ids and what == "H3": target_id = run_ids[0] # Use specified ID
        elif h2_ids: 
            target_id = sorted(h2_ids, key=lambda r: load_manifest(r).get("context_config", ""), reverse=True)[0]
            print(f"H3: Auto-selected target from H2: {target_id}")
        elif h6_ids:
            target_id = sorted(h6_ids, key=lambda r: load_manifest(r).get("context_config", ""), reverse=True)[0]
            print(f"H3: Auto-selected target from H6: {target_id}")
        
        if target_id: h3_timeseries(target_id, name=f"h3_timeseries_{target_id}.png")
        else: print("H3: No H2 or H6 runs found to analyze. Specify a run_id.")
    
    if what == "ALL" or what == "H6":
        if h6_ids: h6_plot(h6_ids)
        else: print("H6: No runs found.")
            
    if what == "ALL" or what == "H7":
        if h6_ids: h7_smartctl_deltas(h6_ids)
        else: print(f"H7: No H6 runs found to analyze.")

    if what == "ALL" or what == "H8":
        target_id = None
        if run_ids and what == "H8": target_id = run_ids[0] # Use specified ID
        elif h2_ids: 
            target_id = sorted(h2_ids, key=lambda r: load_manifest(r).get("context_config", ""), reverse=True)[0]
            print(f"H8: Auto-selected target from H2: {target_id}")
        elif h6_ids:
            target_id = sorted(h6_ids, key=lambda r: load_manifest(r).get("context_config", ""), reverse=True)[0]
            print(f"H8: Auto-selected target from H6: {target_id}")
        
        if target_id: h8_mpstat_timeseries(target_id, name=f"h8_mpstat_{target_id}.png")
        else: print("H8: No H2 or H6 runs found to analyze. Specify a run_id.")

    if what == "ALL":
        print("--- Analysis complete ---")
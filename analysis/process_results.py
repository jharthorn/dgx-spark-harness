#!/usr/bin/env python3
import json, sys, glob
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import os
import re

# v2.2: Simplified for H1, H2, H6, H7, H8 (H4/H5 removed)

# Use container-absolute paths
HARNESS_DIR = Path(os.environ.get("HARNESS_DIR", "/harness"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(HARNESS_DIR / "results")))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", str(HARNESS_DIR / "figures")))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print(f"Reading from: {RESULTS_DIR}")
print(f"Writing to:   {FIGURES_DIR}")

# --- 5.1 smartctl Parser ---
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

# --- 5.2 mpstat Parser ---
def parse_mpstat(path):
    if not path.exists(): return pd.DataFrame()
    try:
        with open(path, 'r') as f: lines = f.readlines()
        header_idx = -1
        for i, line in enumerate(lines):
            if '%usr' in line and 'CPU' in line: header_idx = i; break
        if header_idx == -1: return pd.DataFrame()
        
        header = lines[header_idx].replace('%', 'pct_').split(); header[0] = 'Time'
        data = []
        for line in lines[header_idx+1:]:
            if not line.strip() or 'Average:' in line: continue
            vals = line.split()
            if len(vals) == len(header): data.append(vals)
        if not data: return pd.DataFrame()

        df = pd.DataFrame(data, columns=header)
        numeric_cols = [col for col in df.columns if col not in ['Time', 'AM/PM', 'CPU']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['time_s'] = df.groupby('CPU').cumcount()
        return df
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}", file=sys.stderr)
        return pd.DataFrame()

# --- Load Helpers ---
def load_manifest(run_id):
    p = RESULTS_DIR / f"{run_id}_manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}

def load_summary(run_id):
    p = RESULTS_DIR / f"{run_id}_summary.json"
    if not p.exists(): return {}
    s = json.loads(p.read_text())
    # Defensive backfill
    if s and s.get("avg", {}).get("io_wait_pct") is None and s.get("requests_total", 0) > 0:
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

# --- Plot Helpers ---

def h1_plot(run_ids, name="h1_cache_cold_warm.png"):
    cold_ids = [r for r in run_ids if "COLD" in r]
    warm_ids = [r for r in run_ids if "WARM" in r]
    cold_ttfts = [load_summary(r)["ttft_ms"]["p50"] for r in cold_ids if "ttft_ms" in load_summary(r) and load_summary(r)["ttft_ms"].get("p50") is not None]
    warm_ttfts = [load_summary(r)["ttft_ms"]["p50"] for r in warm_ids if "ttft_ms" in load_summary(r) and load_summary(r)["ttft_ms"].get("p50") is not None]
    cold_ttft = np.mean(cold_ttfts) if cold_ttfts else 0
    warm_ttft = np.mean(warm_ttfts) if warm_ttfts else 0
    if cold_ttft == 0 and warm_ttft == 0: print("H1: No data found to plot."); return
    plt.figure()
    plt.bar(["Cold Cache (H1)"], [cold_ttft], label=f"Cold (p50 TTFT: {cold_ttft:.0f} ms)", width=0.4)
    plt.bar(["Warm Cache (H1)"], [warm_ttft], label=f"Warm (p50 TTFT: {warm_ttft:.0f} ms)", width=0.4)
    plt.ylabel("p50 TTFT (ms)"); plt.title("H1: Cold vs. Warm Cache TTFT"); plt.legend()
    out = FIGURES_DIR / name; plt.tight_layout(); plt.savefig(out, dpi=160); print(f"Wrote {out}")

def h2_scatter(run_ids, name="h2_io_wait_vs_p99.png"):
    data = []
    for rid in run_ids:
        summ = load_summary(rid); manifest = load_manifest(rid)
        if "latency_ms" not in summ or "concurrency_users" not in manifest or summ.get("avg") is None: continue
        p99 = summ["latency_ms"].get("p99"); ci_lo = summ["latency_ms"].get("p99_ci_low"); ci_hi = summ["latency_ms"].get("p99_ci_high"); io_wait = summ["avg"].get("io_wait_pct")
        if any(v is None for v in [p99, ci_lo, ci_hi, io_wait]): continue
        yerr = (p99 - ci_lo, ci_hi - p99)
        data.append({"users": manifest["concurrency_users"], "io_wait": io_wait, "p99": p99, "yerr_low": yerr[0], "yerr_high": yerr[1]})
    if not data: print("H2: No data found to plot."); return
    df = pd.DataFrame(data).groupby("users").mean()
    plt.figure(); yerr = [df["yerr_low"], df["yerr_high"]]
    plt.errorbar(df["io_wait"], df["p99"], yerr=yerr, fmt='o-', capsize=5)
    plt.xlabel('Average io_wait (%)'); plt.ylabel('p99 Latency (ms)'); plt.title('H2: Storage io_wait vs. p99 Latency (by Concurrency)')
    for u, row in df.iterrows(): plt.annotate(f"U={u}", (row["io_wait"], row["p99"]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    out = FIGURES_DIR / name; plt.tight_layout(); plt.savefig(out, dpi=160); print(f"Wrote {out}")

def h3_timeseries(run_id, name="h3_timeseries.png"):
    df = load_telemetry(run_id)
    if df.empty: print(f"H3: No telemetry data for {run_id}"); return
    t = df["ms_since_t0"]/1000.0; plt.figure(figsize=(12,8)); plt.suptitle(f"H3: UMA Contention & Eviction Dynamics ({run_id})", y=1.02)
    plt.subplot(3,1,1); plt.plot(t, df["mem_MemAvailable_kB"], label='MemAvailable_kB', color='blue'); plt.ylabel('Available (kB)', color='blue'); plt.tick_params(axis='y', labelcolor='blue'); plt.legend(loc='upper left'); ax2 = plt.gca().twinx(); ax2.plot(t, df["mem_Cached_kB"], label='Cached_kB', color='green'); ax2.set_ylabel('Cached (kB)', color='green'); ax2.tick_params(axis='y', labelcolor='green'); ax2.legend(loc='upper right'); plt.title('UMA Memory State')
    plt.subplot(3,1,2); plt.plot(t, df["gpu_mem_used_MiB"], label='GPU mem used (MiB)', color='purple'); plt.legend(); plt.ylabel('MiB'); plt.title('GPU Memory Usage')
    plt.subplot(3,1,3); plt.plot(t, df["iostat_rps"], label='Storage Read (r/s)', color='red'); plt.ylabel('r/s', color='red'); plt.tick_params(axis='y', labelcolor='red'); plt.legend(loc='upper left'); ax2_io = plt.gca().twinx(); ax2_io.plot(t, df["vm_wa"], label='io_wait %', color='orange'); ax2_io.set_ylabel('io_wait %', color='orange'); ax2_io.tick_params(axis='y', labelcolor='orange'); ax2_io.legend(loc='upper right'); plt.xlabel('Time (seconds)'); plt.title('Storage Activity')
    cached = df["mem_Cached_kB"].values
    if len(cached) > 1:
        dc = np.diff(cached, prepend=cached[0]); neg_dc = dc[dc < 0]
        if len(neg_dc) > 1:
            eviction_threshold = np.std(neg_dc) * 3
            if eviction_threshold > 0:
                idx = np.where(dc < -eviction_threshold)[0]
                for ax in plt.gcf().get_axes():
                    for i in idx:
                        if i < len(t): ax.axvline(t.iloc[i], color='gray', alpha=0.4, linestyle='--')
    out = FIGURES_DIR / name; plt.tight_layout(rect=[0, 0.03, 1, 0.98]); plt.savefig(out, dpi=160); print(f"Wrote {out}")

def h2_vs_h6_plot(h2_ids, h6_ids, name="h6_h2_vs_h6_workload.png"):
    def get_data(run_ids):
        data = []
        for rid in run_ids:
            summ = load_summary(rid); manifest = load_manifest(rid)
            if "latency_ms" not in summ or "concurrency_users" not in manifest or summ.get("avg") is None: continue
            if summ.get("requests_total", 0) == 0: continue # Skip dummy H6 runs
            users = manifest.get("concurrency_users"); p99 = summ["latency_ms"].get("p99")
            if users is not None and p99 is not None: data.append({"users": users, "p99": p99})
        if not data: return pd.DataFrame()
        return pd.DataFrame(data).groupby("users").mean().sort_index()
    df_h2 = get_data(h2_ids); df_h6 = get_data(h6_ids)
    if df_h2.empty: print(f"H6: Not enough data for H2 baseline."); return
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_h2.index, df_h2['p99'], 'o-', label='H2: Baseline Workload')
    if not df_h6.empty:
        plt.plot(df_h6.index, df_h6['p99'], 's-', label='H6: LoRA Storm Workload')
    else:
        print("H6: No valid LoRA run data found. Plotting H2 only.")
        
    plt.xlabel("Concurrent Users"); plt.ylabel("p99 Latency (ms)"); plt.title("H6 vs H2: Workload Impact on Tail Latency"); plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    out = FIGURES_DIR / name; plt.tight_layout(); plt.savefig(out, dpi=160); print(f"Wrote {out}")

def h7_smartctl_deltas(run_ids, name="h7_smartctl_deltas.png"):
    data = []
    for rid in run_ids:
        manifest = load_manifest(rid)
        hyp = manifest.get("hypothesis", "UNKNOWN")
        if hyp not in ['H2', 'H6']: continue
        if load_summary(rid).get("requests_total", 0) == 0: continue # Skip dummy H6
        
        users = manifest.get("concurrency_users", 0)
        pre = parse_smartctl(RESULTS_DIR / f"{rid}_smartctl_pre.txt")
        post = parse_smartctl(RESULTS_DIR / f"{rid}_smartctl_post.txt")
        
        if pre.get('data_units_read') is None or post.get('data_units_read') is None or \
           pre.get('host_reads') is None or post.get('host_reads') is None:
            print(f"H7: Skipping {rid}, missing smartctl data.")
            continue
            
        read_delta_gb = (post['data_units_read'] - pre['data_units_read']) * 512 / (1024**3)
        write_delta_gb = (post.get('data_units_written', 0) - pre.get('data_units_written', 0)) * 512 / (1024**3)
        read_cmds_m = (post['host_reads'] - pre['host_reads']) / 1_000_000
        
        data.append({ "run_id": rid, "hyp": hyp, "users": users, "read_gb": read_delta_gb,
                      "write_gb": write_delta_gb, "read_cmds_m": read_cmds_m })
    
    if not data: print("H7: No smartctl data found to plot."); return
    
    df = pd.DataFrame(data)
    df = df.groupby(['hyp', 'users']).mean(numeric_only=True).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("H7: Storage Workload Deltas (from smartctl)")

    df_h2 = df[df['hyp']=='H2'].set_index('users')['read_gb']
    df_h6 = df[df['hyp']=='H6'].set_index('users')['read_gb']
    plot_df = pd.DataFrame({'H2 Baseline (GB)': df_h2, 'H6 LoRA Storm (GB)': df_h6})
    plot_df.plot(kind='bar', ax=ax1, rot=0); ax1.set_ylabel("Total Data Read (GB)"); ax1.set_title("Data Read"); ax1.legend()

    df_h2_cmds = df[df['hyp']=='H2'].set_index('users')['read_cmds_m']
    df_h6_cmds = df[df['hyp']=='H6'].set_index('users')['read_cmds_m']
    plot_df_cmds = pd.DataFrame({'H2 Baseline (M)': df_h2_cmds, 'H6 LoRA Storm (M)': df_h6_cmds})
    plot_df_cmds.plot(kind='bar', ax=ax2, rot=0); ax2.set_xlabel("Concurrent Users"); ax2.set_ylabel("Total Host Reads (Millions)"); ax2.set_title("Host Read Commands"); ax2.legend()

    out = FIGURES_DIR / name
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(out, dpi=160); print(f"Wrote {out}")
    
def h8_mpstat_timeseries(run_id, name="h8_mpstat_cores.png"):
    df = parse_mpstat(RESULTS_DIR / f"{run_id}_mpstat.log")
    if df.empty: print(f"H8: No mpstat data for {run_id}"); return
    
    df_all = df[df['CPU'] == 'all'].set_index('time_s')
    if df_all.empty: print(f"H8: No 'all' CPU data found in {run_id}"); return

    plot_cols = ['pct_usr', 'pct_sys', 'pct_iowait', 'pct_idle']
    df_plot = df_all[plot_cols]
    
    plt.figure(figsize=(12, 6))
    plt.stackplot(df_plot.index, df_plot.T, labels=df_plot.columns, alpha=0.8,
                  colors=['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']) # usr, sys, iowait, idle
    
    plt.xlabel("Time (seconds)"); plt.ylabel("CPU Usage (%)"); plt.title(f"H8: CPU Dynamics (all cores) - {run_id}"); plt.legend(loc='upper left'); plt.ylim(0, 100)
    out = FIGURES_DIR / name
    plt.tight_layout(); plt.savefig(out, dpi=160); print(f"Wrote {out}")
    

# --- Main CLI (v2.2) ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <COMMAND> [run_ids...]", file=sys.stderr)
        print("Commands:", file=sys.stderr)
        print("  ALL        (Runs H1, H2, H6, H7, H8 plots)", file=sys.stderr)
        print("  H1, H2, H6 (Generates specific hypothesis plot)", file=sys.stderr)
        print("  H3, H8     (Deep-dive plot, specify one run_id)", file=sys.stderr)
        print("  H7         (Generates smartctl delta plots for H2/H6)", file=sys.stderr)
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
    
    h1_ids = [r for r in run_ids if "H1" in r]
    h2_ids = [r for r in run_ids if "H2" in r]
    h6_ids = [r for r in run_ids if "H6" in r]

    if what == "ALL" or what == "H1":
        if h1_ids: h1_plot(h1_ids)
        else: print("H1: No runs found.")
    
    if what == "ALL" or what == "H2":
        if h2_ids: h2_scatter(h2_ids)
        else: print("H2: No runs found.")
    
    if what == "ALL" or what == "H3":
        target_id = None
        if run_ids and what == "H3": target_id = run_ids[0] # Use specified ID
        elif h6_ids: # H6 is the most stressful test, use it
            target_id = sorted(h6_ids, key=lambda r: load_manifest(r).get("concurrency_users", 0), reverse=True)[0]
            print(f"H3: Auto-selected target (from H6): {target_id}")
        elif h2_ids: 
            target_id = sorted(h2_ids, key=lambda r: load_manifest(r).get("concurrency_users", 0), reverse=True)[0]
            print(f"H3: Auto-selected target (from H2): {target_id}")
        
        if target_id: h3_timeseries(target_id, name=f"h3_timeseries_{target_id}.png")
        else: print("H3: No H2 or H6 runs found to analyze. Specify a run_id.")
    
    if what == "ALL" or what == "H6":
        if h2_ids: h2_vs_h6_plot(h2_ids, h6_ids) # H6 plot now handles missing H6 data
        else: print(f"H6: Not enough data. H2 runs: {len(h2_ids)}")
    
    if what == "ALL" or what == "H7":
        if h2_ids: h7_smartctl_deltas(h2_ids + h6_ids) # H7 plot handles missing H6 data
        else: print(f"H7: No H2 or H6 runs found to analyze.")

    if what == "ALL" or what == "H8":
        target_id = None
        if run_ids and what == "H8": target_id = run_ids[0] # Use specified ID
        elif h6_ids: # H6 is the most stressful test, use it
            target_id = sorted(h6_ids, key=lambda r: load_manifest(r).get("concurrency_users", 0), reverse=True)[0]
            print(f"H8: Auto-selected target (from H6): {target_id}")
        elif h2_ids: 
            target_id = sorted(h2_ids, key=lambda r: load_manifest(r).get("concurrency_users", 0), reverse=True)[0]
            print(f"H8: Auto-selected target (from H2): {target_id}")
        
        if target_id: h8_mpstat_timeseries(target_id, name=f"h8_mpstat_{target_id}.png")
        else: print("H8: No H2 or H6 runs found to analyze. Specify a run_id.")

    if what == "ALL":
        print("--- Analysis complete ---")
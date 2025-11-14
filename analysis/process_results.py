#!/usr/bin/env python3
"""Generate DGX Spark harness plots from collected run artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# Path helpers


def env_path(name: str, default: Path | str) -> Path:
    """Resolves a filesystem path from either an env var or the provided default."""
    value = os.environ.get(name)
    return Path(value) if value else Path(default)


HARNESS_DIR = env_path("HARNESS_DIR", "/harness")
DEFAULT_RESULTS_DIR = env_path("RESULTS_DIR", HARNESS_DIR / "results")
DEFAULT_FIGURES_DIR = env_path("FIGURES_DIR", HARNESS_DIR / "analysis/figures")
DEFAULT_TABLES_DIR = env_path("TABLES_DIR", HARNESS_DIR / "analysis/tables")


CONTEXT_RE = re.compile(r"(?P<input>\d+)_tokens_gen(?P<output>\d+)")

# -----------------------------------------------------------------------------
# Data access layer


@dataclass
class RunRepository:
    """Lazy loader for manifests, summaries, and telemetry artifacts."""

    results_dir: Path
    _manifest_cache: Dict[str, dict] = field(default_factory=dict)
    _summary_cache: Dict[str, dict] = field(default_factory=dict)
    _telemetry_cache: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def manifest(self, run_id: str) -> dict:
        if run_id not in self._manifest_cache:
            path = self.results_dir / f"{run_id}_manifest.json"
            if path.exists():
                try:
                    self._manifest_cache[run_id] = json.loads(path.read_text())
                except json.JSONDecodeError:
                    logging.error("Failed to decode manifest: %s", path)
                    self._manifest_cache[run_id] = {}
            else:
                self._manifest_cache[run_id] = {}
        return self._manifest_cache[run_id]

    def summary(self, run_id: str) -> dict:
        if run_id not in self._summary_cache:
            path = self.results_dir / f"{run_id}_summary.json"
            if path.exists():
                try:
                    summary = json.loads(path.read_text())
                except json.JSONDecodeError:
                    logging.error("Failed to decode summary: %s", path)
                    summary = {}
            else:
                summary = {}
            # --- UPDATED: Auto-call backfill logic ---
            if summary and summary.get("avg", {}).get("io_wait_pct") is None:
                telemetry = self.telemetry(run_id)
                if not telemetry.empty:
                    avg_block = summary.setdefault("avg", {})
                    avg_block["io_wait_pct"] = telemetry.get("vm_wa", pd.Series(dtype=float)).mean()
                    avg_block["qu_sz"] = telemetry.get("iostat_avg_qu_sz", pd.Series(dtype=float)).mean()
                    avg_block["gpu_util_pct"] = telemetry.get("gpu_util_pct", pd.Series(dtype=float)).mean()
                    # Add new backfills
                    avg_block["r_await_ms"] = telemetry.get("iostat_await_ms", pd.Series(dtype=float)).mean()
                    avg_block["rps_storage"] = telemetry.get("iostat_rps", pd.Series(dtype=float)).mean()
            self._summary_cache[run_id] = summary
        return self._summary_cache[run_id]

    def telemetry(self, run_id: str) -> pd.DataFrame:
        if run_id not in self._telemetry_cache:
            path = self.results_dir / f"{run_id}_telemetry.csv"
            if not path.exists():
                self._telemetry_cache[run_id] = pd.DataFrame()
                return self._telemetry_cache[run_id]
            try:
                df = pd.read_csv(path)
            except Exception as exc:  # pragma: no cover - malformed CSV
                logging.warning("Failed to parse telemetry for %s: %s", run_id, exc)
                df = pd.DataFrame()
            cols = [col for col in ["vm_wa", "iostat_avg_qu_sz", "gpu_util_pct", "iostat_await_ms", "iostat_rps"] if col in df.columns]
            if cols:
                df = df.dropna(subset=cols)
            self._telemetry_cache[run_id] = df
        return self._telemetry_cache[run_id]

    def discover_run_ids(self) -> list[str]:
        if not self.results_dir.exists():
            return []
        manifests = self.results_dir.glob("*_manifest.json")
        return sorted(p.stem.replace("_manifest", "") for p in manifests)


# -----------------------------------------------------------------------------
# Parsers


def parse_smartctl(path: Path) -> dict:
    """Extracts smartctl metrics used in H7 plots."""
    stats = {"temp_c": None, "data_units_read": None, "data_units_written": None, "host_reads": None}
    if not path.exists():
        return stats
    try:
        text = path.read_text()
        matchers = {
            "temp_c": r"Temperature:\s+(\d+)\s+Celsius",
            "data_units_read": r"Data Units Read:\s+([\d,]+)",
            "data_units_written": r"Data Units Written:\s+([\d,]+)",
            "host_reads": r"Host Read Commands:\s+([\d,]+)",
        }
        for key, pattern in matchers.items():
            match = re.search(pattern, text)
            if match:
                stats[key] = int(match.group(1).replace(",", ""))
    except Exception as exc:  # pragma: no cover - file corruption
        logging.warning("smartctl parse failure for %s: %s", path, exc)
    return stats


def parse_mpstat(path: Path) -> pd.DataFrame:
    """Parses mpstat output into a tidy DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except Exception as exc:  # pragma: no cover - encoding issues
        logging.warning("mpstat read failure for %s: %s", path, exc)
        return pd.DataFrame()
    header_idx = next((idx for idx, line in enumerate(lines) if "%usr" in line and "CPU" in line), -1)
    if header_idx == -1:
        return pd.DataFrame()
    header = lines[header_idx].replace("%", "pct_").split()
    header[0] = "Time"
    rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip() or "Average:" in line:
            continue
        values = line.split()
        if len(values) == len(header):
            rows.append(values)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=header)
    numeric_cols = [col for col in df.columns if col not in {"Time", "AM/PM", "CPU"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time_s"] = df.groupby("CPU").cumcount()
    return df


# -----------------------------------------------------------------------------
# Plot helpers


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_table(
    df: pd.DataFrame,
    tables_dir: Path,
    filename: str,
    *,
    index: bool = True,
    index_label: str | None = None,
) -> None:
    """Writes the aggregated dataframe to a CSV table alongside the plots."""
    if df.empty:
        return
    ensure_output_dir(tables_dir)
    table_path = tables_dir / filename
    df.to_csv(table_path, index=index, index_label=index_label)
    logging.info("Wrote table %s", table_path)


def categorize_run_ids(run_ids: Sequence[str], manifests: dict[str, dict]) -> dict[str, list[str]]:
    # UPDATED: Categorize H0-H8
    categories = {key: [] for key in ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8")}
    for run_id in run_ids:
        # Use hypothesis from manifest if available, fallback to run_id parsing
        hyp = manifests.get(run_id, {}).get("hypothesis")
        if hyp in categories:
            categories[hyp].append(run_id)
        else:
            # Fallback for older data or custom runs
            for key in categories:
                if f"_{key}_" in run_id:
                    categories[key].append(run_id)
                    break
    return categories


def context_total_tokens(context_cfg: str | None) -> Optional[int]:
    if not context_cfg:
        return None
    match = CONTEXT_RE.search(context_cfg)
    if not match:
        # Fallback for H5 context strings like '1k_tokens'
        match_simple = re.search(r"(\d+)k_tokens", context_cfg)
        if match_simple:
            return int(match_simple.group(1)) * 1024
        match_simple = re.search(r"(\d+)_tokens", context_cfg)
        if match_simple:
            return int(match_simple.group(1))
        return None
    return int(match.group("input")) + int(match.group("output"))


def plot_h0(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path, model_tag: str) -> bool:
    records = []
    for run_id in run_ids:
        summary = repo.summary(run_id)
        manifest = repo.manifest(run_id)
        if not summary or "latency_ms" not in summary or "avg" not in summary:
            continue
        users = manifest.get("concurrency_users")
        p99 = summary["latency_ms"].get("p99")
        io_wait = summary["avg"].get("io_wait_pct")
        if None in (users, p99, io_wait):
            continue
        records.append({"users": users, "p99": p99, "io_wait": io_wait})
    if not records:
        logging.warning("H0 (%s): no eligible runs.", model_tag)
        return False
    df = pd.DataFrame(records).groupby("users").mean().sort_index()
    export_table(df, tables_dir, f"h0_queue_knee_{model_tag}.csv", index_label="users")
    fig, ax_primary = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"H0: Server Queue Knee Calibration ({model_tag})")
    ax_primary.plot(df.index, df["p99"], "o-", label="p99 Latency (ms)", color="blue")
    ax_primary.set_xlabel("Concurrent Users (U)")
    ax_primary.set_ylabel("p99 Latency (ms)", color="blue")
    ax_primary.tick_params(axis="y", labelcolor="blue")
    ax_primary.legend(loc="upper left")
    ax_secondary = ax_primary.twinx()
    ax_secondary.plot(df.index, df["io_wait"], "s--", label="Avg. io_wait (%)", color="red")
    ax_secondary.set_ylabel("Average io_wait (%)", color="red")
    ax_secondary.tick_params(axis="y", labelcolor="red")
    ax_secondary.legend(loc="upper right")
    ax_secondary.set_ylim(bottom=0)
    output = out_dir / f"h0_queue_knee_{model_tag}.png"
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def plot_h1(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path, model_tag: str) -> bool:
    records = []
    for run_id in run_ids:
        summary = repo.summary(run_id)
        manifest = repo.manifest(run_id)
        if not summary or "ttft_ms" not in summary:
            continue
        phase = manifest.get("phase", "UNKNOWN")
        if phase not in {"COLD", "WARM"}: continue
        ttft_p95 = summary["ttft_ms"].get("p95")
        latency_p99 = summary.get("latency_ms", {}).get("p99")
        if None in (ttft_p95, latency_p99):
            continue
        records.append({"phase": phase, "ttft_p95": ttft_p95, "p99": latency_p99})
    if not records:
        logging.warning("H1 (%s): no eligible runs.", model_tag)
        return False
    df = pd.DataFrame(records).groupby("phase").mean().reindex(["COLD", "WARM"])
    export_table(df, tables_dir, f"h1_coldwarm_lora_{model_tag}.csv", index_label="phase")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"H1: Cold vs. Warm LoRA Working Set ({model_tag})")
    df["ttft_p95"].plot(ax=axes[0], kind="bar", color=["#d9534f", "#5cb85c"], rot=0)
    axes[0].set_ylabel("p95 TTFT (ms)")
    axes[0].set_title("Time To First Token")
    axes[0].set_xlabel("")
    df["p99"].plot(ax=axes[1], kind="bar", color=["#d9534f", "#5cb85c"], rot=0)
    axes[1].set_ylabel("p99 Latency (ms)")
    axes[1].set_title("Tail Latency")
    axes[1].set_xlabel("")
    output = out_dir / f"h1_coldwarm_lora_{model_tag}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def plot_h2(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path, model_tag: str) -> bool:
    records = []
    for run_id in run_ids:
        summary = repo.summary(run_id)
        manifest = repo.manifest(run_id)
        if not summary or "latency_ms" not in summary or "avg" not in summary:
            continue
        total_tokens = context_total_tokens(manifest.get("context_config"))
        p99 = summary["latency_ms"].get("p99")
        io_wait = summary["avg"].get("io_wait_pct")
        if None in (total_tokens, p99, io_wait):
            continue
        records.append({"total_tokens": total_tokens, "p99": p99, "io_wait": io_wait})
    if not records:
        logging.warning("H2 (%s): no eligible runs.", model_tag)
        return False
    df = pd.DataFrame(records).groupby("total_tokens").mean().sort_index()
    export_table(df, tables_dir, f"h2_uma_pressure_hockey_stick_{model_tag}.csv", index_label="total_tokens")
    fig, ax_primary = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"H2: UMA Pressure vs. Latency (Hockey Stick) ({model_tag})")
    ax_primary.plot(df.index, df["p99"], "o-", label="p99 Latency (ms)", color="blue")
    ax_primary.set_xlabel("Total Context Length (Input + Output Tokens)")
    ax_primary.set_ylabel("p99 Latency (ms)", color="blue")
    ax_primary.tick_params(axis="y", labelcolor="blue")
    ax_primary.legend(loc="upper left")
    ax_secondary = ax_primary.twinx()
    ax_secondary.plot(df.index, df["io_wait"], "s--", label="Avg. io_wait (%)", color="red")
    ax_secondary.set_ylabel("Average io_wait (%)", color="red")
    ax_secondary.tick_params(axis="y", labelcolor="red")
    ax_secondary.legend(loc="upper right")
    ax_secondary.set_ylim(bottom=0)
    output = out_dir / f"h2_uma_pressure_hockey_stick_{model_tag}.png"
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True

# --- NEW H4 PLOT ---
def plot_h4(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path, model_tag: str) -> bool:
    """Plots H4: p99 vs storage QoS (fio contention)."""
    records = []
    for run_id in run_ids:
        summary = repo.summary(run_id)
        manifest = repo.manifest(run_id)
        if not summary or "latency_ms" not in summary or "avg" not in summary:
            continue
        phase = manifest.get("phase")
        if phase not in {"BASELINE", "MODERATE", "HEAVY"}:
            continue
        p99 = summary["latency_ms"].get("p99")
        r_await = summary["avg"].get("r_await_ms") # From backfill
        if None in (p99, r_await):
            continue
        records.append({"phase": phase, "p99": p99, "r_await_ms": r_await})
    if not records:
        logging.warning("H4 (%s): no eligible runs.", model_tag)
        return False
    
    df = pd.DataFrame(records).groupby("phase").mean().reindex(["BASELINE", "MODERATE", "HEAVY"])
    export_table(df, tables_dir, f"h4_storage_qos_{model_tag}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"H4: p99 Latency vs. Storage QoS Contention ({model_tag})")

    # Plot 1: p99 vs. Contention Phase (Bar)
    colors = ["#5cb85c", "#f0ad4e", "#d9534f"]
    df["p99"].plot(ax=axes[0], kind="bar", color=colors, rot=0)
    axes[0].set_ylabel("p99 Latency (ms)")
    axes[0].set_title("Latency vs. Contention")
    axes[0].set_xlabel("")

    # Plot 2: p99 vs. Storage Read Await (Scatter)
    df_scatter = pd.DataFrame(records) # Use raw data for scatter
    axes[1].scatter(df_scatter["r_await_ms"], df_scatter["p99"], alpha=0.7, c=df_scatter["phase"].map(colors))
    axes[1].set_xlabel("Average Storage Read Wait (r_await_ms)")
    axes[1].set_ylabel("p99 Latency (ms)")
    axes[1].set_title("Latency vs. Storage Read Wait")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    output = out_dir / f"h4_storage_qos_{model_tag}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True

# --- NEW H5 PLOT ---
def plot_h5(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path, model_tag: str) -> bool:
    """Plots H5: p99 vs. LoRA working set size & churn."""
    records = []
    for run_id in run_ids:
        summary = repo.summary(run_id)
        manifest = repo.manifest(run_id)
        if not summary or "latency_ms" not in summary or "avg" not in summary:
            continue
        
        session_type = manifest.get("lora_session_type") # "sticky" or "random"
        adapter_count = manifest.get("lora_adapter_count")
        if not session_type or not adapter_count:
            continue
            
        p99 = summary["latency_ms"].get("p99")
        rps_storage = summary["avg"].get("rps_storage") # From backfill
        if None in (p99, rps_storage):
            continue
            
        records.append({
            "session_type": "Stormy" if session_type == "random" else "Sticky",
            "adapters": int(adapter_count),
            "p99": p99,
            "rps_storage": rps_storage
        })
    if not records:
        logging.warning("H5 (%s): no eligible runs.", model_tag)
        return False

    df = pd.DataFrame(records)
    df_agg = df.groupby(["adapters", "session_type"]).mean().reset_index()
    export_table(df_agg, tables_dir, f"h5_lora_scaling_{model_tag}.csv", index=False)

    df_pivot_p99 = df_agg.pivot(index="adapters", columns="session_type", values="p99")
    df_pivot_rps = df_agg.pivot(index="adapters", columns="session_type", values="rps_storage")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"H5: LoRA Working-Set Scaling vs. Latency ({model_tag})")

    # Plot 1: p99 Latency vs. Adapter Count
    df_pivot_p99.plot(ax=axes[0], kind="line", style="o-", rot=0)
    axes[0].set_ylabel("p99 Latency (ms)")
    axes[0].set_title("Tail Latency vs. Adapter Count")
    axes[0].set_xlabel("Active LoRA Adapters in Working Set")
    axes[0].set_xticks(df_pivot_p99.index)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Storage Read IOPS vs. Adapter Count
    df_pivot_rps.plot(ax=axes[1], kind="line", style="s--", rot=0)
    axes[1].set_ylabel("Average Storage Read IOPS (iostat_rps)")
    axes[1].set_title("Storage Read Pressure vs. Adapter Count")
    axes[1].set_xlabel("Active LoRA Adapters in Working Set")
    axes[1].set_xticks(df_pivot_rps.index)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    output = out_dir / f"h5_lora_scaling_{model_tag}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def plot_h6(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path, model_tag: str) -> bool:
    records = []
    for run_id in run_ids:
        summary = repo.summary(run_id)
        manifest = repo.manifest(run_id)
        if not summary or "latency_ms" not in summary or "avg" not in summary:
            continue
        phase = manifest.get("phase", "UNKNOWN")
        if phase not in {"BASELINE", "LORA"}: continue # LORA is the "Storm"
        
        # Rename "LORA" to "STORM" for plot clarity
        plot_phase = "STORM" if phase == "LORA" else phase

        p99 = summary["latency_ms"].get("p99")
        io_wait = summary["avg"].get("io_wait_pct")
        if None in (p99, io_wait):
            continue
        records.append({"phase": plot_phase, "p99": p99, "io_wait": io_wait})
    if not records:
        logging.warning("H6 (%s): no eligible runs.", model_tag)
        return False
    df = pd.DataFrame(records).groupby("phase").mean().reindex(["BASELINE", "STORM"])
    export_table(df, tables_dir, f"h6_workload_ab_{model_tag}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"H6: Workload A/B Test (Baseline vs. LoRA Storm) ({model_tag})")
    df["p99"].plot(ax=axes[0], kind="bar", color=["#5bc0de", "#f0ad4e"], rot=0)
    axes[0].set_ylabel("p99 Latency (ms)")
    axes[0].set_title("Tail Latency")
    axes[0].set_xlabel("")
    df["io_wait"].plot(ax=axes[1], kind="bar", color=["#5bc0de", "#f0ad4e"], rot=0)
    axes[1].set_ylabel("Average io_wait (%)")
    axes[1].set_title("Storage I/O Wait")
    axes[1].set_xlabel("")
    output = out_dir / f"h6_workload_ab_{model_tag}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def select_timeseries_target(
    requested_ids: Sequence[str], h2_ids: Sequence[str], h6_ids: Sequence[str], repo: RunRepository
) -> Optional[str]:
    # Try to find the highest-pressure run for timeseries plots
    if requested_ids:
        return requested_ids[0]
    
    all_candidates = h2_ids + h6_ids
    if not all_candidates:
        return None

    def score_run(run_id: str) -> float:
        # Prefer H6-LORA, then H2 with max context
        manifest = repo.manifest(run_id)
        score = 0.0
        if manifest.get("hypothesis") == "H6" and manifest.get("phase") == "LORA":
            score += 100000.0
        context_tok = context_total_tokens(manifest.get("context_config"))
        if context_tok:
            score += context_tok
        return score

    return max(all_candidates, key=score_run)


def plot_h3(
    repo: RunRepository,
    requested_ids: Sequence[str],
    grouped: dict[str, list[str]],
    out_dir: Path,
    tables_dir: Path,
    model_tag: str
) -> bool:
    target = select_timeseries_target(requested_ids, grouped.get("H2", []), grouped.get("H6", []), repo)
    if not target:
        logging.warning("H3 (%s): no target run specified or discovered.", model_tag)
        return False
    telemetry = repo.telemetry(target)
    required_cols = {
        "ms_since_t0",
        "mem_MemAvailable_kB",
        "mem_Cached_kB",
        "gpu_mem_used_MiB",
        "iostat_rps",
        "vm_wa",
    }
    if telemetry.empty or not required_cols.issubset(telemetry.columns):
        logging.warning("H3: telemetry missing required columns for %s.", target)
        return False
    time_axis = telemetry["ms_since_t0"] / 1000.0
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"H3: UMA Contention & Eviction Dynamics ({model_tag})\nRun ID: {target}", y=1.02)
    
    # Plot 1: Memory
    axes[0].plot(time_axis, telemetry["mem_MemAvailable_kB"], label="MemAvailable_kB", color="blue")
    axes[0].set_ylabel("Available (kB)", color="blue")
    axes[0].tick_params(axis="y", labelcolor="blue")
    ax_cached = axes[0].twinx()
    ax_cached.plot(time_axis, telemetry["mem_Cached_kB"], label="Cached_kB", color="green")
    ax_cached.set_ylabel("Cached (kB)", color="green")
    ax_cached.tick_params(axis="y", labelcolor="green")
    axes[0].legend(loc="upper left")
    ax_cached.legend(loc="upper right")
    axes[0].set_title("UMA Memory State")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: GPU
    axes[1].plot(time_axis, telemetry["gpu_mem_used_MiB"], label="GPU mem used (MiB)", color="purple")
    axes[1].set_ylabel("MiB")
    axes[1].set_title("GPU Memory Usage")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Storage
    axes[2].plot(time_axis, telemetry["iostat_rps"], label="Storage Read (r/s)", color="red")
    axes[2].set_ylabel("r/s", color="red")
    axes[2].tick_params(axis="y", labelcolor="red")
    ax_io_wait = axes[2].twinx()
    ax_io_wait.plot(time_axis, telemetry["vm_wa"], label="io_wait %", color="orange")
    ax_io_wait.set_ylabel("io_wait %", color="orange")
    ax_io_wait.tick_params(axis="y", labelcolor="orange")
    axes[2].legend(loc="upper left")
    ax_io_wait.legend(loc="upper right")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_title("Storage Activity")
    axes[2].grid(True, linestyle='--', alpha=0.6)

    timeseries_df = telemetry[["ms_since_t0", "mem_MemAvailable_kB", "mem_Cached_kB", "gpu_mem_used_MiB", "iostat_rps", "vm_wa"]]
    export_table(timeseries_df, tables_dir, f"h3_timeseries_{target}.csv", index=False)
    output = out_dir / f"h3_timeseries_{target}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def plot_h7(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path, model_tag: str) -> bool:
    records = []
    for run_id in run_ids:
        manifest = repo.manifest(run_id)
        if manifest.get("hypothesis") != "H6": # H7 analyzes H6 runs
            continue
        phase = manifest.get("phase", "UNKNOWN")
        if phase not in {"BASELINE", "LORA"}: continue

        # Rename "LORA" to "STORM" for plot clarity
        plot_phase = "STORM" if phase == "LORA" else phase

        pre = parse_smartctl(repo.results_dir / f"{run_id}_smartctl_pre.txt")
        post = parse_smartctl(repo.results_dir / f"{run_id}_smartctl_post.txt")
        if None in (pre["data_units_read"], post["data_units_read"], pre["host_reads"], post["host_reads"]):
            logging.warning("H7: Skipping %s, missing smartctl data.", run_id)
            continue
        read_delta_gb = (post["data_units_read"] - pre["data_units_read"]) * 512 / (1024**3)
        read_cmds_m = (post["host_reads"] - pre["host_reads"]) / 1_000_000
        records.append({"phase": plot_phase, "read_gb": read_delta_gb, "read_cmds_m": read_cmds_m})
    
    if not records:
        logging.warning("H7 (%s): no smartctl data available.", model_tag)
        return False
        
    df = pd.DataFrame(records).groupby("phase").mean().reindex(["BASELINE", "STORM"])
    export_table(df, tables_dir, f"h7_smartctl_deltas_{model_tag}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"H7: Storage Workload Deltas (H6: Baseline vs LoRA) ({model_tag})")
    df["read_gb"].plot(kind="bar", ax=axes[0], color=["#5bc0de", "#f0ad4e"], rot=0)
    axes[0].set_ylabel("Total Data Read (GB)")
    axes[0].set_title("Data Read")
    axes[0].set_xlabel("")
    df["read_cmds_m"].plot(kind="bar", ax=axes[1], color=["#5bc0de", "#f0ad4e"], rot=0)
    axes[1].set_ylabel("Host Read Commands (Millions)")
    axes[1].set_title("Host Read Commands")
    axes[1].set_xlabel("")
    output = out_dir / f"h7_smartctl_deltas_{model_tag}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def plot_h8(
    repo: RunRepository,
    requested_ids: Sequence[str],
    grouped: dict[str, list[str]],
    out_dir: Path,
    tables_dir: Path,
    model_tag: str
) -> bool:
    target = select_timeseries_target(requested_ids, grouped.get("H2", []), grouped.get("H6", []), repo)
    if not target:
        logging.warning("H8 (%s): no target run specified or discovered.", model_tag)
        return False
    mpstat_path = repo.results_dir / f"{target}_mpstat.log"
    df = parse_mpstat(mpstat_path)
    if df.empty:
        logging.warning("H8: no mpstat data for %s.", target)
        return False
    df_all = df[df["CPU"] == "all"].set_index("time_s")
    if df_all.empty:
        logging.warning("H8: missing aggregate CPU row for %s.", target)
        return False
    plot_cols = ["pct_usr", "pct_sys", "pct_iowait", "pct_idle"]
    available_cols = [c for c in plot_cols if c in df_all.columns]
    if not available_cols:
        logging.warning("H8: mpstat columns missing for %s.", target)
        return False
    export_table(df_all[available_cols], tables_dir, f"h8_mpstat_{target}.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    df_all[available_cols].plot.area(ax=ax, stacked=True, alpha=0.8,
                                    color=["#0275d8", "#f0ad4e", "#d9534f", "#e6e6e6"])
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("CPU Usage (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"H8: CPU Dynamics (all cores) - ({model_tag})\nRun ID: {target}")
    ax.legend(loc="upper left")
    output = out_dir / f"h8_mpstat_{target}.png"
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


# -----------------------------------------------------------------------------
# CLI plumbing


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        help="Plot selector (ALL, H0, H1, H2, H3, H4, H5, H6, H7, H8).",
    )
    parser.add_argument("run_ids", nargs="*", help="Optional run IDs to limit the analysis.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory containing *_manifest.json files.")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory where plots are written.")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR), help="Directory for tabular exports.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    command = args.command.upper()
    repo = RunRepository(Path(args.results_dir))
    figures_dir = ensure_output_dir(Path(args.figures_dir))
    tables_dir = ensure_output_dir(Path(args.tables_dir))
    
    requested_ids = args.run_ids
    all_run_ids = requested_ids or repo.discover_run_ids()
    if not all_run_ids:
        logging.warning("No run data found in %s.", repo.results_dir)
        return 0
        
    # --- UPDATED: Pre-cache all manifests and group by model_tag ---
    all_manifests = {run_id: repo.manifest(run_id) for run_id in all_run_ids}
    
    runs_by_model: Dict[str, list[str]] = {}
    for run_id, manifest in all_manifests.items():
        model_tag = manifest.get("model_tag", "UNKNOWN")
        if model_tag not in runs_by_model:
            runs_by_model[model_tag] = []
        runs_by_model[model_tag].append(run_id)

    logging.info("Reading from %s", repo.results_dir)
    logging.info("Writing plots to %s", figures_dir)
    logging.info("Found data for models: %s", list(runs_by_model.keys()))

    plot_commands = {
        "H0": plot_h0,
        "H1": plot_h1,
        "H2": plot_h2,
        "H3": plot_h3,
        "H4": plot_h4, # New
        "H5": plot_h5, # New
        "H6": plot_h6,
        "H7": plot_h7,
        "H8": plot_h8,
    }

    executed_any = False
    
    # --- UPDATED: Loop over each model tag and generate plots for it ---
    for model_tag, model_run_ids in runs_by_model.items():
        logging.info("--- Processing model: %s ---", model_tag)
        model_figures_dir = ensure_output_dir(figures_dir / model_tag)
        model_tables_dir = ensure_output_dir(tables_dir / model_tag)
        
        grouped = categorize_run_ids(model_run_ids, all_manifests)
        
        executed_model = False
        
        for hyp, plot_func in plot_commands.items():
            if command in {"ALL", hyp}:
                run_ids_for_plot = grouped.get(hyp, [])
                
                # H3 and H8 are special (timeseries)
                if hyp in {"H3", "H8"}:
                    # If user specified IDs, use those. Otherwise, let plotter find best.
                    target_ids = requested_ids if requested_ids else []
                    if plot_func(repo, target_ids, grouped, model_figures_dir, model_tables_dir, model_tag):
                        executed_model = True
                # H7 analyzes H6 runs
                elif hyp == "H7":
                    if plot_func(repo, grouped.get("H6", []), model_figures_dir, model_tables_dir, model_tag):
                         executed_model = True
                # Standard hypothesis plots
                elif run_ids_for_plot:
                    if plot_func(repo, run_ids_for_plot, model_figures_dir, model_tables_dir, model_tag):
                        executed_model = True

        if not executed_model:
            logging.warning("Command %s produced no plots for model %s.", command, model_tag)
        
        executed_any |= executed_model

    if command == "ALL":
        logging.info("--- Analysis complete ---")
    if not executed_any:
        logging.warning("Command %s produced no plots for any model.", command)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

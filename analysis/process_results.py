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
                self._manifest_cache[run_id] = json.loads(path.read_text())
            else:
                self._manifest_cache[run_id] = {}
        return self._manifest_cache[run_id]

    def summary(self, run_id: str) -> dict:
        if run_id not in self._summary_cache:
            path = self.results_dir / f"{run_id}_summary.json"
            if path.exists():
                summary = json.loads(path.read_text())
            else:
                summary = {}
            if summary and summary.get("avg", {}).get("io_wait_pct") is None:
                telemetry = self.telemetry(run_id)
                if not telemetry.empty:
                    avg_block = summary.setdefault("avg", {})
                    avg_block["io_wait_pct"] = telemetry.get("vm_wa", pd.Series(dtype=float)).mean()
                    avg_block["qu_sz"] = telemetry.get("iostat_avg_qu_sz", pd.Series(dtype=float)).mean()
                    avg_block["gpu_util_pct"] = telemetry.get("gpu_util_pct", pd.Series(dtype=float)).mean()
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
            cols = [col for col in ["vm_wa", "iostat_avg_qu_sz", "gpu_util_pct"] if col in df.columns]
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


def export_table(df: pd.DataFrame, tables_dir: Path, filename: str, index_label: str | None = None) -> None:
    """Writes the aggregated dataframe to a CSV table alongside the plots."""
    if df.empty:
        return
    ensure_output_dir(tables_dir)
    table_path = tables_dir / filename
    df.to_csv(table_path, index_label=index_label)
    logging.info("Wrote table %s", table_path)


def categorize_run_ids(run_ids: Sequence[str]) -> dict[str, list[str]]:
    categories = {key: [] for key in ("H0", "H1", "H2", "H6")}
    for run_id in run_ids:
        for key in categories:
            if key in run_id:
                categories[key].append(run_id)
    return categories


def context_total_tokens(context_cfg: str | None) -> Optional[int]:
    if not context_cfg:
        return None
    match = CONTEXT_RE.search(context_cfg)
    if not match:
        return None
    return int(match.group("input")) + int(match.group("output"))


def plot_h0(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path) -> bool:
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
        logging.warning("H0: no eligible runs.")
        return False
    df = pd.DataFrame(records).groupby("users").mean().sort_index()
    export_table(df, tables_dir, "h0_queue_knee.csv", index_label="users")
    fig, ax_primary = plt.subplots(figsize=(10, 6))
    fig.suptitle("H0: Server Queue Knee Calibration")
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
    output = out_dir / "h0_queue_knee.png"
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def plot_h1(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path) -> bool:
    records = []
    for run_id in run_ids:
        summary = repo.summary(run_id)
        manifest = repo.manifest(run_id)
        if not summary or "ttft_ms" not in summary:
            continue
        phase = manifest.get("phase", "UNKNOWN")
        ttft_p95 = summary["ttft_ms"].get("p95")
        latency_p99 = summary.get("latency_ms", {}).get("p99")
        if None in (ttft_p95, latency_p99):
            continue
        records.append({"phase": phase, "ttft_p95": ttft_p95, "p99": latency_p99})
    if not records:
        logging.warning("H1: no eligible runs.")
        return False
    df = pd.DataFrame(records).groupby("phase").mean()
    export_table(df, tables_dir, "h1_coldwarm_lora.csv", index_label="phase")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H1: Cold vs. Warm LoRA Working Set")
    df["ttft_p95"].plot(ax=axes[0], kind="bar", color=["red", "green"], rot=0)
    axes[0].set_ylabel("p95 TTFT (ms)")
    axes[0].set_title("Time To First Token")
    df["p99"].plot(ax=axes[1], kind="bar", color=["red", "green"], rot=0)
    axes[1].set_ylabel("p99 Latency (ms)")
    axes[1].set_title("Tail Latency")
    output = out_dir / "h1_coldwarm_lora.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def plot_h2(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path) -> bool:
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
        logging.warning("H2: no eligible runs.")
        return False
    df = pd.DataFrame(records).groupby("total_tokens").mean().sort_index()
    export_table(df, tables_dir, "h2_uma_pressure_hockey_stick.csv", index_label="total_tokens")
    fig, ax_primary = plt.subplots(figsize=(10, 6))
    fig.suptitle("H2: UMA Pressure vs. Latency (Hockey Stick)")
    ax_primary.plot(df.index, df["p99"], "o-", label="p99 Latency (ms)", color="blue")
    ax_primary.set_xlabel("Total Context Length (tokens)")
    ax_primary.set_ylabel("p99 Latency (ms)", color="blue")
    ax_primary.tick_params(axis="y", labelcolor="blue")
    ax_primary.legend(loc="upper left")
    ax_secondary = ax_primary.twinx()
    ax_secondary.plot(df.index, df["io_wait"], "s--", label="Avg. io_wait (%)", color="red")
    ax_secondary.set_ylabel("Average io_wait (%)", color="red")
    ax_secondary.tick_params(axis="y", labelcolor="red")
    ax_secondary.legend(loc="upper right")
    ax_secondary.set_ylim(bottom=0)
    output = out_dir / "h2_uma_pressure_hockey_stick.png"
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def plot_h6(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path) -> bool:
    records = []
    for run_id in run_ids:
        summary = repo.summary(run_id)
        manifest = repo.manifest(run_id)
        if not summary or "latency_ms" not in summary or "avg" not in summary:
            continue
        phase = manifest.get("phase", "UNKNOWN")
        p99 = summary["latency_ms"].get("p99")
        io_wait = summary["avg"].get("io_wait_pct")
        if None in (p99, io_wait):
            continue
        records.append({"phase": phase, "p99": p99, "io_wait": io_wait})
    if not records:
        logging.warning("H6: no eligible runs.")
        return False
    df = pd.DataFrame(records).groupby("phase").mean()
    export_table(df, tables_dir, "h6_workload_ab.csv", index_label="phase")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H6: Workload A/B Test (Baseline vs. LoRA Storm)")
    df["p99"].plot(ax=axes[0], kind="bar", color=["blue", "orange"], rot=0)
    axes[0].set_ylabel("p99 Latency (ms)")
    axes[0].set_title("Tail Latency")
    df["io_wait"].plot(ax=axes[1], kind="bar", color=["blue", "orange"], rot=0)
    axes[1].set_ylabel("Average io_wait (%)")
    axes[1].set_title("Storage I/O Wait")
    output = out_dir / "h6_workload_ab.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def select_timeseries_target(
    requested_ids: Sequence[str], h2_ids: Sequence[str], h6_ids: Sequence[str], repo: RunRepository
) -> Optional[str]:
    if requested_ids:
        return requested_ids[0]
    if h2_ids:
        return max(h2_ids, key=lambda rid: context_total_tokens(repo.manifest(rid).get("context_config")) or 0)
    if h6_ids:
        return max(h6_ids, key=lambda rid: context_total_tokens(repo.manifest(rid).get("context_config")) or 0)
    return None


def plot_h3(
    repo: RunRepository,
    requested_ids: Sequence[str],
    grouped: dict[str, list[str]],
    out_dir: Path,
    tables_dir: Path,
) -> bool:
    target = select_timeseries_target(requested_ids, grouped.get("H2", []), grouped.get("H6", []), repo)
    if not target:
        logging.warning("H3: no target run specified or discovered.")
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
    fig.suptitle(f"H3: UMA Contention & Eviction Dynamics ({target})", y=1.02)
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
    axes[1].plot(time_axis, telemetry["gpu_mem_used_MiB"], label="GPU mem used (MiB)", color="purple")
    axes[1].set_ylabel("MiB")
    axes[1].set_title("GPU Memory Usage")
    axes[1].legend()
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
    timeseries_df = telemetry[["ms_since_t0", "mem_MemAvailable_kB", "mem_Cached_kB", "gpu_mem_used_MiB", "iostat_rps", "vm_wa"]]
    export_table(timeseries_df, tables_dir, f"h3_timeseries_{target}.csv", index_label=None)
    output = out_dir / f"h3_timeseries_{target}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output, dpi=160)
    plt.close(fig)
    logging.info("Wrote %s", output)
    return True


def plot_h7(repo: RunRepository, run_ids: Sequence[str], out_dir: Path, tables_dir: Path) -> bool:
    records = []
    for run_id in run_ids:
        manifest = repo.manifest(run_id)
        if manifest.get("hypothesis") != "H6":
            continue
        phase = manifest.get("phase", "UNKNOWN")
        pre = parse_smartctl(repo.results_dir / f"{run_id}_smartctl_pre.txt")
        post = parse_smartctl(repo.results_dir / f"{run_id}_smartctl_post.txt")
        if None in (pre["data_units_read"], post["data_units_read"], pre["host_reads"], post["host_reads"]):
            continue
        read_delta_gb = (post["data_units_read"] - pre["data_units_read"]) * 512 / (1024**3)
        read_cmds_m = (post["host_reads"] - pre["host_reads"]) / 1_000_000
        records.append({"phase": phase, "read_gb": read_delta_gb, "read_cmds_m": read_cmds_m})
    if not records:
        logging.warning("H7: no smartctl data available.")
        return False
    df = pd.DataFrame(records).groupby("phase").mean()
    export_table(df, tables_dir, "h7_smartctl_deltas.csv", index_label="phase")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H7: Storage Workload Deltas (H6: Baseline vs LoRA)")
    df["read_gb"].plot(kind="bar", ax=axes[0], color=["blue", "orange"], rot=0)
    axes[0].set_ylabel("Total Data Read (GB)")
    axes[0].set_title("Data Read")
    df["read_cmds_m"].plot(kind="bar", ax=axes[1], color=["blue", "orange"], rot=0)
    axes[1].set_ylabel("Host Read Commands (Millions)")
    axes[1].set_title("Host Read Commands")
    output = out_dir / "h7_smartctl_deltas.png"
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
) -> bool:
    target = select_timeseries_target(requested_ids, grouped.get("H2", []), grouped.get("H6", []), repo)
    if not target:
        logging.warning("H8: no target run specified or discovered.")
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
    export_table(df_all[available_cols], tables_dir, f"h8_mpstat_{target}.csv", index_label="time_s")
    fig, ax = plt.subplots(figsize=(12, 6))
    df_all[available_cols].plot.area(ax=ax, stacked=True, alpha=0.8)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("CPU Usage (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"H8: CPU Dynamics (all cores) - {target}")
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
        help="Plot selector (ALL, H0, H1, H2, H3, H6, H7, H8).",
    )
    parser.add_argument("run_ids", nargs="*", help="Optional run IDs to limit the analysis.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory containing *_manifest.json files.")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory where plots are written.")
    parser.add_argument("--tables-dir", default=str(DEFAULT_FIGURES_DIR / "tables"), help="Directory for tabular exports.")
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
    run_ids = requested_ids or repo.discover_run_ids()
    if not run_ids:
        logging.warning("No run data found in %s.", repo.results_dir)
        return 0
    grouped = categorize_run_ids(run_ids)
    logging.info("Reading from %s", repo.results_dir)
    logging.info("Writing plots to %s", figures_dir)
    executed = False
    if command in {"ALL", "H0"}:
        executed |= plot_h0(repo, grouped.get("H0", []), figures_dir, tables_dir)
    if command in {"ALL", "H1"}:
        executed |= plot_h1(repo, grouped.get("H1", []), figures_dir, tables_dir)
    if command in {"ALL", "H2"}:
        executed |= plot_h2(repo, grouped.get("H2", []), figures_dir, tables_dir)
    if command in {"ALL", "H3"}:
        executed |= plot_h3(repo, requested_ids, grouped, figures_dir, tables_dir)
    if command in {"ALL", "H6"}:
        executed |= plot_h6(repo, grouped.get("H6", []), figures_dir, tables_dir)
    if command in {"ALL", "H7"}:
        executed |= plot_h7(repo, grouped.get("H6", []), figures_dir, tables_dir)
    if command in {"ALL", "H8"}:
        executed |= plot_h8(repo, requested_ids, grouped, figures_dir, tables_dir)
    if command == "ALL":
        logging.info("--- Analysis complete ---")
    if not executed:
        logging.warning("Command %s produced no plots.", command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

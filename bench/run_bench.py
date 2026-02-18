#!/usr/bin/env python3
"""Benchmark driver for Dynamo + TRT-LLM + KVBM OpenAI-compatible completions."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .prompts import (
    PromptSpec,
    TokenEstimator,
    generate_local_project_copilot_burst,
    generate_prompt_set,
    generate_rehydrate_replay_sets,
    generate_replay_sets,
    manifest_rows,
)
from .telemetry import TelemetryManager

LOG = logging.getLogger("bench.run_bench")

PROM_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([0-9.eE+-]+)$")
NVME_CONTROLLER_RE = re.compile(r"^nvme\d+$")
NVME_DEVICE_PATH_RE = re.compile(r"^/dev/(nvme\d+)(?:n\d+(?:p\d+)?)?$")
BDF_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")

METRICS_SNAPSHOT_PHASE_SUFFIX: dict[str, str] = {
    "pressure_B": "pressure",
    "thrash": "pressure",
    "replay_A": "replay",
    "replay": "replay",
}

METRIC_INVENTORY_DEFAULT_KEYWORDS: tuple[str, ...] = (
    "kvbm",
    "block_manager",
    "offload",
    "onboard",
    "matched",
    "rehydrate",
    "tier",
    "disk",
    "host",
)

KVBM_COUNTER_METRICS: tuple[str, ...] = (
    "kvbm_offload_blocks_d2h",
    "kvbm_offload_blocks_h2d",
    "kvbm_offload_blocks_d2d",
    "kvbm_onboard_blocks_d2d",
    "kvbm_onboard_blocks_h2d",
    "kvbm_matched_tokens",
    "kvbm_disk_bytes_read",
    "kvbm_disk_bytes_written",
    "kvbm_cache_evictions",
    "kvbm_cache_prefetches",
)

KVBM_GAUGE_METRICS: tuple[str, ...] = (
    "kvbm_host_cache_hit_rate",
    "kvbm_disk_cache_hit_rate",
    "kvbm_disk_fetch_p50_ms",
    "kvbm_disk_fetch_p95_ms",
    "kvbm_disk_fetch_p99_ms",
)

PROMPT_LIMIT_ERROR_RE = re.compile(r"(max_num_tokens|should not exceed max_num_tokens)", re.IGNORECASE)

KVBM_ENV_KEYS: tuple[str, ...] = (
    "BENCH_TIER_MODE",
    "BENCH_KV_MODE",
    "DYN_KVBM_CPU_CACHE_GB",
    "DYN_KVBM_DISK_CACHE_GB",
    "DYN_KVBM_DISK_CACHE_DIR",
    "DYN_KVBM_METRICS",
    "DYN_KVBM_METRICS_PORT",
    "DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER",
    "DYN_SYSTEM_PORT",
    "DYN_REQUEST_PLANE",
)


@dataclass
class PhasePlan:
    name: str
    prompts: list[PromptSpec]
    concurrency: int
    include_in_overall: bool = True


class KVBMMetricsProbe:
    def __init__(self, url: str, timeout_s: float) -> None:
        self.url = url
        self.timeout_s = timeout_s

    def snapshot(self, label: str) -> dict[str, Any]:
        now = now_utc_iso()
        if not self.url:
            return {
                "label": label,
                "timestamp_utc": now,
                "success": False,
                "error": "metrics_url_not_configured",
                "metrics": {},
            }
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout_s) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            parsed = parse_prometheus_metrics(text)
            kvbm = {k: v for k, v in parsed.items() if k.startswith("kvbm_")}
            return {
                "label": label,
                "timestamp_utc": now,
                "success": True,
                "error": None,
                "metrics": kvbm,
            }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {
                "label": label,
                "timestamp_utc": now,
                "success": False,
                "error": str(exc),
                "metrics": {},
            }

    def delta(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        before_metrics = before.get("metrics", {}) if isinstance(before, dict) else {}
        after_metrics = after.get("metrics", {}) if isinstance(after, dict) else {}
        if not isinstance(before_metrics, dict) or not isinstance(after_metrics, dict):
            return {"available": False, "reason": "malformed_snapshot"}
        if not before.get("success") or not after.get("success"):
            return {
                "available": False,
                "reason": "snapshot_missing",
                "before_success": bool(before.get("success")),
                "after_success": bool(after.get("success")),
                "before_error": before.get("error"),
                "after_error": after.get("error"),
            }

        counters_delta: dict[str, float] = {}
        for key in KVBM_COUNTER_METRICS:
            counters_delta[f"{key}_delta"] = max(0.0, float(after_metrics.get(key, 0.0)) - float(before_metrics.get(key, 0.0)))

        gauges_after: dict[str, Optional[float]] = {}
        for key in KVBM_GAUGE_METRICS:
            value = after_metrics.get(key)
            gauges_after[f"{key}_after"] = float(value) if value is not None else None

        offload_total = (
            counters_delta.get("kvbm_offload_blocks_d2h_delta", 0.0)
            + counters_delta.get("kvbm_offload_blocks_h2d_delta", 0.0)
            + counters_delta.get("kvbm_offload_blocks_d2d_delta", 0.0)
        )
        onboard_total = (
            counters_delta.get("kvbm_onboard_blocks_d2d_delta", 0.0)
            + counters_delta.get("kvbm_onboard_blocks_h2d_delta", 0.0)
        )
        disk_read_bytes = counters_delta.get("kvbm_disk_bytes_read_delta", 0.0)
        disk_write_bytes = counters_delta.get("kvbm_disk_bytes_written_delta", 0.0)
        derived = {
            "offload_blocks_total_delta": offload_total,
            "onboard_blocks_total_delta": onboard_total,
            "disk_read_mib_delta": round(disk_read_bytes / (1024 * 1024), 3),
            "disk_write_mib_delta": round(disk_write_bytes / (1024 * 1024), 3),
            "disk_rehydrate_counter_positive": onboard_total > 0.0,
        }
        return {"available": True, **counters_delta, **gauges_after, **derived}


def is_kvbm_enabled(kv_mode: dict[str, Any]) -> bool:
    return bool((kv_mode or {}).get("kvbm_enabled"))


def build_kvbm_metrics_status(
    *,
    kvbm_enabled: bool,
    metrics_available: bool,
    snapshot_count: int,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    if not kvbm_enabled:
        status = "skipped"
        resolved_reason = reason or "kvbm_disabled"
    elif metrics_available:
        status = "ok"
        resolved_reason = None
    else:
        status = "unavailable"
        resolved_reason = reason or "kvbm_metrics_unavailable"
    return {
        "status": status,
        "kvbm_enabled": bool(kvbm_enabled),
        "metrics_available": bool(metrics_available),
        "snapshot_count": int(snapshot_count),
        "reason": resolved_reason,
    }


def make_kvbm_skipped_snapshot(label: str, reason: str) -> dict[str, Any]:
    return {
        "label": label,
        "timestamp_utc": now_utc_iso(),
        "success": False,
        "error": reason,
        "metrics": {},
        "skipped": True,
        "reason": reason,
    }


def make_kvbm_skipped_delta(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "skipped": True,
        "reason": reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DGX Spark Dynamo/TRT-LLM/KVBM benchmark driver.")
    system_port = os.environ.get("DYN_SYSTEM_PORT", "8081")
    kvbm_metrics_port = os.environ.get("DYN_KVBM_METRICS_PORT", "6880")
    default_system_metrics_url = f"http://127.0.0.1:{system_port}/metrics"
    default_kvbm_metrics_url = f"http://127.0.0.1:{kvbm_metrics_port}/metrics"
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Frontend base URL.")
    parser.add_argument("--model-id", default="auto", help="Model ID; `auto` resolves from /v1/models.")
    parser.add_argument(
        "--model-resolve-timeout-s",
        type=float,
        default=180.0,
        help="How long to wait for `/v1/models` to return a model when --model-id=auto.",
    )
    parser.add_argument(
        "--model-resolve-poll-s",
        type=float,
        default=2.0,
        help="Polling interval while waiting for model discovery.",
    )
    parser.add_argument(
        "--scenario",
        choices=["standard", "eviction_replay", "reuse_verify", "local_copilot_burst", "rehydrate_replay"],
        default="standard",
    )
    parser.add_argument("--prompt-set", choices=["short", "long", "mixed"], default="short")
    parser.add_argument("--requests", type=int, default=64, help="Measured requests for standard scenario.")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup requests (excluded from overall summary).")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Stop sequence; can be repeated. Defaults to `<|eot_id|>` when omitted.",
    )
    parser.add_argument("--stream", action="store_true", help="Request streaming responses (enables TTFC capture).")
    parser.add_argument(
        "--stream-metrics",
        dest="stream",
        action="store_true",
        help="Alias for --stream (streamed request path for latency telemetry).",
    )
    parser.add_argument(
        "--stream-timeout-s",
        type=float,
        default=None,
        help="Optional per-request timeout override for stream responses (defaults to --timeout-s).",
    )
    parser.add_argument(
        "--stream-record-ttfb",
        action="store_true",
        help="Record TTFB (time to response headers/first byte) when available.",
    )
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--request-seed",
        type=int,
        default=None,
        help="Optional seed passed to `/v1/completions` request payload.",
    )
    parser.add_argument(
        "--tokenizer",
        default="auto",
        help="Tokenizer path/name for prompt sizing (`auto`, `heuristic`, or local model path).",
    )
    parser.add_argument("--short-range", default="512:2048", help="Input token target range for short prompts.")
    parser.add_argument("--long-range", default="8192:32768", help="Input token target range for long prompts.")
    parser.add_argument("--results-root", default="bench/results")
    parser.add_argument("--run-id", default=None, help="Optional run ID; timestamp is used when omitted.")
    parser.add_argument("--store-responses", action="store_true", help="Persist response text for sanity checks.")

    parser.add_argument("--eviction-a-requests", type=int, default=32, help="Phase A request count.")
    parser.add_argument("--eviction-b-requests", type=int, default=64, help="Phase B pressure request count.")
    parser.add_argument("--eviction-a-concurrency", type=int, default=None, help="Phase A/C concurrency.")
    parser.add_argument("--eviction-b-concurrency", type=int, default=None, help="Phase B concurrency.")
    parser.add_argument(
        "--reuse-repeat-count",
        type=int,
        default=3,
        help="Identical sequential requests for reuse_verify scenario (2 or 3).",
    )
    parser.add_argument(
        "--reuse-prompt-set",
        choices=["short", "long"],
        default="short",
        help="Prompt set for reuse_verify scenario.",
    )

    parser.add_argument(
        "--copilot-session-count",
        type=int,
        default=8,
        help="Session cardinality for local_copilot_burst scenario.",
    )
    parser.add_argument(
        "--copilot-burst-size",
        type=int,
        default=4,
        help="Number of sequential turns per session burst for local_copilot_burst.",
    )
    parser.add_argument(
        "--copilot-shared-prefix-target-tokens",
        type=int,
        default=3072,
        help="Approximate token budget for shared project prefix in local_copilot_burst.",
    )
    parser.add_argument(
        "--rehydrate-populate-sessions",
        type=int,
        default=16,
        help="Session cardinality used to populate reusable KV state before thrash.",
    )
    parser.add_argument(
        "--rehydrate-thrash-sessions",
        type=int,
        default=96,
        help="Unique session cardinality used to apply cache pressure during thrash.",
    )
    parser.add_argument(
        "--rehydrate-turns",
        type=int,
        default=2,
        help="Turns per populate session reused again during replay.",
    )
    parser.add_argument(
        "--rehydrate-prefix-target-tokens",
        type=int,
        default=4096,
        help="Approximate token budget for the shared session prefix in rehydrate_replay.",
    )
    parser.add_argument(
        "--rehydrate-populate-concurrency",
        type=int,
        default=None,
        help="Populate phase concurrency (defaults to --concurrency).",
    )
    parser.add_argument(
        "--rehydrate-thrash-concurrency",
        type=int,
        default=None,
        help="Thrash phase concurrency (defaults to max(populate, 2x --concurrency)).",
    )
    parser.add_argument(
        "--rehydrate-replay-concurrency",
        type=int,
        default=None,
        help="Replay phase concurrency (defaults to populate concurrency).",
    )
    parser.add_argument(
        "--rehydrate-replay-repeats",
        type=int,
        default=1,
        help="Replay phase repetitions for rehydrate_replay scenario.",
    )
    parser.add_argument(
        "--rehydrate-gen-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override used only for rehydrate_replay requests.",
    )

    parser.add_argument("--kv-mode", choices=["off", "cpu_only", "cpu_disk"], default="cpu_disk")
    parser.add_argument(
        "--tier-mode",
        choices=["B0", "B1", "B2", "b0", "b1", "b2"],
        default=None,
        help="Canonical serving mode mapping: B0=off, B1=cpu_only, B2=cpu_disk.",
    )
    parser.add_argument("--kv-cpu-cache-gb", type=float, default=None, help="Resolved CPU cache size tag for artifacts.")
    parser.add_argument("--kv-disk-cache-gb", type=float, default=None, help="Resolved disk cache size tag for artifacts.")
    parser.add_argument("--variant-tag", action="append", default=[], help="Optional variant tags for metadata/report.")
    parser.add_argument("--diagnostic-disable-partial-reuse", action="store_true")
    parser.add_argument("--diagnostic-disable-block-reuse", action="store_true")
    parser.add_argument("--diagnostic-disable-disk-offload-filter", action="store_true")

    parser.add_argument("--engine-max-input-tokens", type=int, default=8192)
    parser.add_argument("--input-token-safety-margin", type=int, default=256)

    parser.add_argument("--phase-io-device", default="nvme0n1", help="Block device for phase read/write deltas.")
    parser.add_argument(
        "--worker-proc-pattern",
        default=r"dynamo\.trtllm",
        help="Regex used to sample worker process /proc/*/io counters at phase boundaries.",
    )
    parser.add_argument(
        "--nvme-device",
        default=os.environ.get("BENCH_NVME_DEVICE", "/dev/nvme0"),
        help="NVMe device path used for identity + SMART pre/post captures.",
    )
    parser.add_argument("--collect-telemetry", action="store_true", help="Start/stop bench/scripts collectors.")
    parser.add_argument("--telemetry-interval-s", type=int, default=1)
    parser.add_argument("--telemetry-pid", default="ALL", help="PID target for pidstat (`ALL` by default).")
    parser.add_argument("--iostat-device", default="nvme0n1")
    parser.add_argument(
        "--io-attrib",
        action="store_true",
        help="Enable additive I/O attribution capture (block/process/path/page-cache hints).",
    )
    parser.add_argument(
        "--io-attrib-interval-s",
        type=float,
        default=1.0,
        help="Sampling interval (seconds) for I/O attribution timeline collectors.",
    )
    parser.add_argument(
        "--io-attrib-top-procs",
        type=int,
        default=12,
        help="When worker PID pattern is unavailable, track top-N /proc I/O processes.",
    )
    parser.add_argument("--kvbm-cache-dir", default="/mnt/nvme/kvbm")
    parser.add_argument("--container-name", default="dyn")

    parser.add_argument("--kvbm-metrics-url", default=default_kvbm_metrics_url)
    parser.add_argument("--kvbm-metrics-timeout-s", type=float, default=3.0)
    parser.add_argument(
        "--capture-metrics-snapshot",
        action="store_true",
        help="Capture raw Prometheus snapshots for pressure/replay from both system and KVBM endpoints.",
    )
    parser.add_argument(
        "--metrics-system-url",
        default=default_system_metrics_url,
        help="System metrics URL (typically DYN_SYSTEM_PORT).",
    )
    parser.add_argument(
        "--metrics-kvbm-url",
        default=default_kvbm_metrics_url,
        help="KVBM metrics URL (typically DYN_KVBM_METRICS_PORT).",
    )
    parser.add_argument(
        "--metrics-snapshot-dir",
        default="",
        help="Output directory for metrics snapshots and inventory files (defaults to run dir).",
    )
    parser.add_argument(
        "--metrics-inventory-keywords",
        default=",".join(METRIC_INVENTORY_DEFAULT_KEYWORDS),
        help="Comma-separated keyword list used for expanded metrics inventory matching.",
    )
    parser.add_argument(
        "--allow-missing-kvbm-metrics",
        action="store_true",
        help="Do not invalidate run when KVBM metrics endpoint is unavailable.",
    )

    parser.add_argument("--report-filename", default="report.md")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.stream_timeout_s is not None and float(args.stream_timeout_s) <= 0:
        parser.error("--stream-timeout-s must be > 0 when provided.")
    return args


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def split_inventory_keywords(raw: str) -> list[str]:
    out: list[str] = []
    for token in re.split(r"[\s,]+", str(raw or "").strip().lower()):
        item = token.strip()
        if item:
            out.append(item)
    return out or list(METRIC_INVENTORY_DEFAULT_KEYWORDS)


def fetch_metrics_text(url: str, timeout_s: float) -> dict[str, Any]:
    if not url:
        return {"success": False, "error": "metrics_url_not_configured", "raw_text": ""}
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return {"success": True, "error": None, "raw_text": text}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"success": False, "error": str(exc), "raw_text": ""}


def write_metrics_snapshot(path: Path, phase_name: str, snap: dict[str, Any]) -> None:
    raw_text = str(snap.get("raw_text") or "")
    if snap.get("success") and raw_text:
        path.write_text(raw_text, encoding="utf-8")
        return
    lines = [f"# snapshot_unavailable phase={phase_name}"]
    if snap.get("skipped"):
        lines.append("# skipped=true")
    reason = snap.get("reason")
    if reason is not None:
        lines.append(f"# reason={reason}")
    lines.append(f"# error={snap.get('error')}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def collect_metric_samples(
    texts: dict[str, str],
    keywords: list[str],
    include_regex: Optional[re.Pattern[str]] = None,
) -> dict[str, str]:
    samples: dict[str, str] = {}
    kw = [k.lower() for k in keywords if k]
    for text in texts.values():
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            match = PROM_LINE_RE.match(s)
            if not match:
                continue
            metric_name = match.group(1)
            lower_name = metric_name.lower()
            keyword_hit = any(token in lower_name for token in kw)
            regex_hit = bool(include_regex and include_regex.search(metric_name))
            if keyword_hit or regex_hit:
                samples.setdefault(metric_name, s)
    return samples


def write_metric_inventory(
    path: Path,
    title: str,
    keywords: list[str],
    samples: dict[str, str],
) -> None:
    lines: list[str] = [
        f"# {title}",
        f"# keywords={','.join(keywords)}",
        f"# metric_count={len(samples)}",
        "",
    ]
    for metric_name in sorted(samples):
        lines.append(metric_name)
        lines.append(f"sample: {samples[metric_name]}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _read_selected_meminfo() -> dict[str, Any]:
    keys = ("Dirty", "Writeback", "Cached", "Buffers")
    out = {key: None for key in keys}
    path = Path("/proc/meminfo")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"success": False, "error": str(exc), "values_kib": out}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        if key not in out:
            continue
        token = rest.strip().split()
        if not token:
            continue
        try:
            out[key] = int(token[0])
        except ValueError:
            continue
    return {"success": True, "error": None, "values_kib": out}


def _read_selected_vmstat() -> dict[str, Any]:
    keys = ("pgfault", "pgmajfault", "pgpgin", "pgpgout")
    out = {key: None for key in keys}
    path = Path("/proc/vmstat")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"success": False, "error": str(exc), "values": out}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        key, value = parts
        if key not in out:
            continue
        try:
            out[key] = int(value)
        except ValueError:
            continue
    return {"success": True, "error": None, "values": out}


def _scan_process_io_snapshot(proc_pattern: str, top_n: int) -> dict[str, Any]:
    timestamp = now_utc_iso()
    try:
        rx = re.compile(proc_pattern)
    except re.error as exc:
        return {
            "timestamp_utc": timestamp,
            "success": False,
            "selection_mode": "invalid_pattern",
            "error": f"invalid_regex:{exc}",
            "processes": [],
        }

    entries: list[dict[str, Any]] = []
    denied = 0
    unreadable = 0
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        cmdline_path = pid_dir / "cmdline"
        io_path = pid_dir / "io"
        try:
            cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        except OSError:
            continue
        if not cmdline:
            continue
        fields: dict[str, int] = {}
        try:
            for raw in io_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" not in raw:
                    continue
                key, value = raw.split(":", 1)
                fields[key.strip()] = int(value.strip())
        except PermissionError:
            denied += 1
            continue
        except OSError:
            unreadable += 1
            continue
        except ValueError:
            unreadable += 1
            continue
        entries.append(
            {
                "pid": int(pid_dir.name),
                "cmdline": cmdline,
                "read_bytes": int(fields.get("read_bytes", 0)),
                "write_bytes": int(fields.get("write_bytes", 0)),
                "syscr": int(fields.get("syscr", 0)),
                "syscw": int(fields.get("syscw", 0)),
                "matched_pattern": bool(rx.search(cmdline)),
            }
        )

    matched = [item for item in entries if item.get("matched_pattern")]
    if matched:
        selected = sorted(matched, key=lambda x: (int(x.get("read_bytes", 0)) + int(x.get("write_bytes", 0))), reverse=True)
        mode = "pattern_match"
    else:
        selected = sorted(entries, key=lambda x: (int(x.get("read_bytes", 0)) + int(x.get("write_bytes", 0))), reverse=True)[
            : max(1, top_n)
        ]
        mode = "top_fallback"

    return {
        "timestamp_utc": timestamp,
        "success": True,
        "selection_mode": mode,
        "error": None,
        "total_processes_scanned": len(entries),
        "matched_process_count": len(matched),
        "permission_denied_count": denied,
        "unreadable_count": unreadable,
        "processes": selected,
    }


def _capture_lsof_entries(
    *,
    pids: list[int],
    kvbm_disk_path: str,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    timestamp = now_utc_iso()
    if not pids:
        return {
            "timestamp_utc": timestamp,
            "success": True,
            "error": None,
            "entries": [],
            "pid_count": 0,
        }
    if not kvbm_disk_path:
        return {
            "timestamp_utc": timestamp,
            "success": True,
            "error": None,
            "entries": [],
            "pid_count": len(pids),
        }
    if shutil.which("lsof") is None:
        return {
            "timestamp_utc": timestamp,
            "success": False,
            "error": "lsof_not_found",
            "entries": [],
            "pid_count": len(pids),
        }
    pid_csv = ",".join(str(pid) for pid in sorted({int(pid) for pid in pids if int(pid) > 0}))
    if not pid_csv:
        return {
            "timestamp_utc": timestamp,
            "success": True,
            "error": None,
            "entries": [],
            "pid_count": 0,
        }
    result = _run_capture_command(["lsof", "-nP", "-p", pid_csv], timeout_s=timeout_s)
    if not result.get("success"):
        return {
            "timestamp_utc": timestamp,
            "success": False,
            "error": str(result.get("error") or "lsof_failed"),
            "stderr": str(result.get("stderr") or "").strip() or None,
            "entries": [],
            "pid_count": len(pids),
        }
    entries: list[dict[str, Any]] = []
    root = str(Path(kvbm_disk_path).resolve())
    for raw in str(result.get("stdout") or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("COMMAND"):
            continue
        parts = line.split(None, 8)
        if len(parts) < 9:
            continue
        command, pid_raw, _user, fd, fd_type, _device, _size_off, _node, name = parts
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        path = name.strip()
        if not path:
            continue
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            resolved = path
        if not (resolved == root or resolved.startswith(root.rstrip("/") + "/")):
            continue
        entries.append(
            {
                "pid": pid,
                "command": command,
                "fd": fd,
                "fd_type": fd_type,
                "path": resolved,
            }
        )
    return {
        "timestamp_utc": timestamp,
        "success": True,
        "error": None,
        "entries": entries,
        "pid_count": len(pids),
    }


def _find_snapshot_for_time(samples: list[dict[str, Any]], target_ts: Optional[str], *, prefer: str) -> Optional[dict[str, Any]]:
    target_dt = _parse_iso8601(target_ts)
    if target_dt is None or not samples:
        return None
    parsed: list[tuple[datetime, dict[str, Any]]] = []
    for sample in samples:
        dt = _parse_iso8601(str(sample.get("timestamp_utc") or ""))
        if dt is None:
            continue
        parsed.append((dt, sample))
    if not parsed:
        return None
    parsed.sort(key=lambda x: x[0])
    if prefer == "before":
        candidate = [item for item in parsed if item[0] <= target_dt]
        if candidate:
            return candidate[-1][1]
        return parsed[0][1]
    candidate = [item for item in parsed if item[0] >= target_dt]
    if candidate:
        return candidate[0][1]
    return parsed[-1][1]


def _block_delta_from_samples(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not before or not after:
        return None
    try:
        read_before = int(before.get("read_sectors", 0))
        read_after = int(after.get("read_sectors", 0))
        write_before = int(before.get("write_sectors", 0))
        write_after = int(after.get("write_sectors", 0))
    except (TypeError, ValueError):
        return None
    read_sectors = max(0, read_after - read_before)
    write_sectors = max(0, write_after - write_before)
    return {
        "read_bytes": int(read_sectors * 512),
        "write_bytes": int(write_sectors * 512),
        "read_mib": round((read_sectors * 512) / (1024 * 1024), 3),
        "write_mib": round((write_sectors * 512) / (1024 * 1024), 3),
    }


def _process_delta_from_samples(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not before or not after:
        return {"available": False, "reason": "phase_sample_missing", "per_pid": []}
    before_list = before.get("processes") if isinstance(before.get("processes"), list) else []
    after_list = after.get("processes") if isinstance(after.get("processes"), list) else []
    before_map = {int(p.get("pid")): p for p in before_list if isinstance(p, dict) and p.get("pid") is not None}
    after_map = {int(p.get("pid")): p for p in after_list if isinstance(p, dict) and p.get("pid") is not None}
    per_pid: list[dict[str, Any]] = []
    total_read = 0
    total_write = 0
    for pid, after_item in after_map.items():
        before_item = before_map.get(pid, {})
        read_delta = max(0, int(after_item.get("read_bytes", 0)) - int(before_item.get("read_bytes", 0)))
        write_delta = max(0, int(after_item.get("write_bytes", 0)) - int(before_item.get("write_bytes", 0)))
        if read_delta <= 0 and write_delta <= 0:
            continue
        total_read += read_delta
        total_write += write_delta
        per_pid.append(
            {
                "pid": pid,
                "cmdline": after_item.get("cmdline"),
                "read_bytes": read_delta,
                "write_bytes": write_delta,
            }
        )
    per_pid.sort(key=lambda item: int(item.get("read_bytes", 0)) + int(item.get("write_bytes", 0)), reverse=True)
    return {
        "available": True,
        "read_bytes": total_read,
        "write_bytes": total_write,
        "per_pid": per_pid,
    }


def _collect_direct_io_hints(kv_runtime_env: dict[str, Any]) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for key, value in (kv_runtime_env or {}).items():
        upper = str(key).upper()
        if "DIRECT" in upper or "CUFILE" in upper or "GDS" in upper:
            hints[key] = value
    kvbm_cfg_path = Path("kvbm/kvbm_llm_api_config.yaml")
    direct_lines: list[str] = []
    if kvbm_cfg_path.exists():
        try:
            for raw in kvbm_cfg_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "direct" in raw.lower():
                    direct_lines.append(raw.strip())
        except OSError:
            direct_lines = []
    return {
        "env_hints": hints,
        "kvbm_config_direct_mentions": direct_lines[:20],
    }


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _empty_io_attribution_report(
    *,
    primary_nvme_device: str,
    kvbm_disk_path: str,
    phase_windows: dict[str, dict[str, str]],
    error: str,
    capture_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "capture_timestamp": now_utc_iso(),
        "enabled": True,
        "available": False,
        "primary_nvme_device": primary_nvme_device,
        "kvbm_disk_path": kvbm_disk_path,
        "phase_windows": phase_windows,
        "block_io_by_phase": {},
        "process_io_by_phase": {},
        "kvbm_file_io_by_phase": {
            "available": False,
            "method": "unavailable",
            "reason": error,
            "phases": {},
        },
        "direct_io_evidence": {
            "method": "unavailable",
            "observed_flags": [],
            "config_hints": {},
            "confidence": "low",
        },
        "error": error,
        "capture_errors": capture_errors,
    }


class IOAttributionCollector:
    def __init__(
        self,
        *,
        run_dir: Path,
        primary_nvme_device: str,
        block_device: str,
        proc_pattern: str,
        kvbm_disk_path: str,
        interval_s: float,
        top_procs: int,
        kv_runtime_env: dict[str, Any],
    ) -> None:
        self.run_dir = run_dir
        self.primary_nvme_device = primary_nvme_device
        self.block_device = block_device
        self.proc_pattern = proc_pattern
        self.kvbm_disk_path = kvbm_disk_path
        self.interval_s = max(0.2, float(interval_s))
        self.top_procs = max(1, int(top_procs))
        self.kv_runtime_env = dict(kv_runtime_env or {})
        self.io_dir = run_dir / "io"

        self.capture_errors: list[dict[str, Any]] = []
        self.phase_windows: dict[str, dict[str, str]] = {}
        self.block_samples: list[dict[str, Any]] = []
        self.process_samples: list[dict[str, Any]] = []
        self.meminfo_snapshots: list[dict[str, Any]] = []
        self.vmstat_snapshots: list[dict[str, Any]] = []
        self.lsof_samples: list[dict[str, Any]] = []
        self.tools_available = {
            "iostat": bool(shutil.which("iostat")),
            "pidstat": bool(shutil.which("pidstat")),
            "lsof": bool(shutil.which("lsof")),
            "bpftrace": bool(shutil.which("bpftrace")),
        }
        self._iostat_proc: Optional[subprocess.Popen[str]] = None
        self._pidstat_proc: Optional[subprocess.Popen[str]] = None
        self._iostat_log_fp = None
        self._pidstat_log_fp = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.running = False

    def start(self) -> None:
        self.io_dir.mkdir(parents=True, exist_ok=True)
        self._start_external_collectors()
        try:
            self._capture_kernel_snapshot(label="run_start", phase=None, boundary="start")
            self._capture_timeline_sample()
        except Exception as exc:  # noqa: BLE001
            _register_capture_error(
                self.capture_errors,
                command=["io_attrib_start_sample"],
                return_code=None,
                stderr=str(exc),
            )
        self.running = True
        self._thread = threading.Thread(target=self._sample_loop, name="io-attrib", daemon=True)
        self._thread.start()

    def mark_phase_boundary(
        self,
        *,
        phase: str,
        boundary: str,
        timestamp_utc: Optional[str] = None,
    ) -> None:
        ts = timestamp_utc or now_utc_iso()
        with self._lock:
            window = self.phase_windows.setdefault(str(phase), {})
            window[str(boundary)] = ts
        try:
            self._capture_kernel_snapshot(label=f"phase_{boundary}", phase=phase, boundary=boundary)
            self._capture_timeline_sample()
        except Exception as exc:  # noqa: BLE001
            _register_capture_error(
                self.capture_errors,
                command=["io_attrib_phase_boundary", str(phase), str(boundary)],
                return_code=None,
                stderr=str(exc),
            )

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(2.0, self.interval_s * 3))
        self._stop_external_collectors()
        try:
            self._capture_timeline_sample()
            self._capture_kernel_snapshot(label="run_end", phase=None, boundary="end")
        except Exception as exc:  # noqa: BLE001
            _register_capture_error(
                self.capture_errors,
                command=["io_attrib_stop_sample"],
                return_code=None,
                stderr=str(exc),
            )
        self.running = False

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                self._capture_timeline_sample()
            except Exception as exc:  # noqa: BLE001
                _register_capture_error(
                    self.capture_errors,
                    command=["io_attrib_sample_loop"],
                    return_code=None,
                    stderr=str(exc),
                )

    def _start_external_collectors(self) -> None:
        sample_interval = str(max(1, int(round(self.interval_s))))
        if self.tools_available.get("iostat"):
            try:
                iostat_path = self.io_dir / "iostat.raw.log"
                self._iostat_log_fp = iostat_path.open("w", encoding="utf-8")
                self._iostat_proc = subprocess.Popen(
                    ["iostat", "-x", "-d", "-t", self.block_device, sample_interval],
                    stdout=self._iostat_log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as exc:  # noqa: BLE001
                _register_capture_error(
                    self.capture_errors,
                    command=["iostat", "-x", "-d", "-t", self.block_device, sample_interval],
                    return_code=None,
                    stderr=str(exc),
                )
        if self.tools_available.get("pidstat"):
            try:
                pidstat_path = self.io_dir / "pidstat.raw.log"
                self._pidstat_log_fp = pidstat_path.open("w", encoding="utf-8")
                self._pidstat_proc = subprocess.Popen(
                    ["pidstat", "-d", "-h", "-p", "ALL", sample_interval],
                    stdout=self._pidstat_log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as exc:  # noqa: BLE001
                _register_capture_error(
                    self.capture_errors,
                    command=["pidstat", "-d", "-h", "-p", "ALL", sample_interval],
                    return_code=None,
                    stderr=str(exc),
                )

    def _stop_external_collectors(self) -> None:
        for proc in (self._iostat_proc, self._pidstat_proc):
            if proc is None:
                continue
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if proc.poll() not in (0, None):
                _register_capture_error(
                    self.capture_errors,
                    command=(proc.args if isinstance(proc.args, list) else [str(proc.args)]),
                    return_code=proc.poll(),
                    stderr="collector_nonzero_exit",
                )
        for fp in (self._iostat_log_fp, self._pidstat_log_fp):
            if fp is None:
                continue
            try:
                fp.close()
            except Exception:
                pass
        self._iostat_proc = None
        self._pidstat_proc = None
        self._iostat_log_fp = None
        self._pidstat_log_fp = None

    def _capture_kernel_snapshot(self, *, label: str, phase: Optional[str], boundary: Optional[str]) -> None:
        timestamp = now_utc_iso()
        meminfo = _read_selected_meminfo()
        vmstat = _read_selected_vmstat()
        mem_row = {
            "timestamp_utc": timestamp,
            "label": label,
            "phase": phase,
            "boundary": boundary,
            **meminfo,
        }
        vm_row = {
            "timestamp_utc": timestamp,
            "label": label,
            "phase": phase,
            "boundary": boundary,
            **vmstat,
        }
        with self._lock:
            self.meminfo_snapshots.append(mem_row)
            self.vmstat_snapshots.append(vm_row)
        if not meminfo.get("success"):
            _register_capture_error(
                self.capture_errors,
                command=["cat", "/proc/meminfo"],
                return_code=None,
                stderr=str(meminfo.get("error") or "meminfo_capture_failed"),
            )
        if not vmstat.get("success"):
            _register_capture_error(
                self.capture_errors,
                command=["cat", "/proc/vmstat"],
                return_code=None,
                stderr=str(vmstat.get("error") or "vmstat_capture_failed"),
            )

    def _capture_timeline_sample(self) -> None:
        timestamp = now_utc_iso()
        block_stats = read_block_device_stats(self.block_device)
        if block_stats is None:
            block_item = {
                "timestamp_utc": timestamp,
                "success": False,
                "error": f"missing_block_device:{self.block_device}",
            }
            _register_capture_error(
                self.capture_errors,
                command=["cat", f"/sys/block/{self.block_device}/stat"],
                return_code=None,
                stderr=f"missing_block_device:{self.block_device}",
            )
        else:
            block_item = {"timestamp_utc": timestamp, "success": True, **block_stats}

        proc_snapshot = _scan_process_io_snapshot(self.proc_pattern, self.top_procs)
        if not proc_snapshot.get("success"):
            _register_capture_error(
                self.capture_errors,
                command=["cat", "/proc/*/io"],
                return_code=None,
                stderr=str(proc_snapshot.get("error") or "process_capture_failed"),
            )

        pids = [int(item.get("pid")) for item in (proc_snapshot.get("processes") or []) if item.get("pid") is not None]
        lsof_capture = _capture_lsof_entries(pids=pids, kvbm_disk_path=self.kvbm_disk_path)
        if not lsof_capture.get("success"):
            _register_capture_error(
                self.capture_errors,
                command=["lsof", "-nP", "-p", ",".join(str(pid) for pid in pids)],
                return_code=None,
                stderr=str(lsof_capture.get("error") or "lsof_failed"),
            )

        with self._lock:
            self.block_samples.append(block_item)
            self.process_samples.append(proc_snapshot)
            self.lsof_samples.append(lsof_capture)

    def _write_mem_vm_jsonl(self) -> None:
        mem_path = self.io_dir / "meminfo_snapshots.jsonl"
        vm_path = self.io_dir / "vmstat_snapshots.jsonl"
        with mem_path.open("w", encoding="utf-8") as fp:
            for item in self.meminfo_snapshots:
                fp.write(json.dumps(item, sort_keys=True) + "\n")
        with vm_path.open("w", encoding="utf-8") as fp:
            for item in self.vmstat_snapshots:
                fp.write(json.dumps(item, sort_keys=True) + "\n")

    def _write_iostat_csv(self) -> None:
        out_path = self.io_dir / "block_stat_timeline.csv"
        fields = [
            "timestamp_utc",
            "device",
            "read_ios_total",
            "write_ios_total",
            "read_bytes_total",
            "write_bytes_total",
            "read_bytes_delta",
            "write_bytes_delta",
            "read_mib_delta",
            "write_mib_delta",
            "source",
        ]
        prev: Optional[dict[str, Any]] = None
        with out_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            for sample in self.block_samples:
                if not sample.get("success"):
                    continue
                read_sectors = int(sample.get("read_sectors", 0))
                write_sectors = int(sample.get("write_sectors", 0))
                read_bytes = int(read_sectors * 512)
                write_bytes = int(write_sectors * 512)
                prev_read_sectors = int(prev.get("read_sectors", read_sectors)) if prev else read_sectors
                prev_write_sectors = int(prev.get("write_sectors", write_sectors)) if prev else write_sectors
                delta_read = max(0, read_sectors - prev_read_sectors) * 512
                delta_write = max(0, write_sectors - prev_write_sectors) * 512
                writer.writerow(
                    {
                        "timestamp_utc": sample.get("timestamp_utc"),
                        "device": self.block_device,
                        "read_ios_total": int(sample.get("read_ios", 0)),
                        "write_ios_total": int(sample.get("write_ios", 0)),
                        "read_bytes_total": read_bytes,
                        "write_bytes_total": write_bytes,
                        "read_bytes_delta": delta_read,
                        "write_bytes_delta": delta_write,
                        "read_mib_delta": round(delta_read / (1024 * 1024), 3),
                        "write_mib_delta": round(delta_write / (1024 * 1024), 3),
                        "source": "procfs_sys_block_stat",
                    }
                )
                prev = sample
        legacy_path = self.io_dir / "iostat.csv"
        shutil.copyfile(out_path, legacy_path)

    def _write_pidstat_csv(self) -> None:
        out_path = self.io_dir / "proc_io_timeline.csv"
        fields = [
            "timestamp_utc",
            "pid",
            "cmdline",
            "selection_mode",
            "read_bytes_total",
            "write_bytes_total",
            "read_bytes_delta",
            "write_bytes_delta",
            "syscr_total",
            "syscw_total",
            "source",
        ]
        prev_by_pid: dict[int, dict[str, Any]] = {}
        with out_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            for sample in self.process_samples:
                if not sample.get("success"):
                    continue
                ts = sample.get("timestamp_utc")
                mode = sample.get("selection_mode")
                for proc in (sample.get("processes") or []):
                    if not isinstance(proc, dict) or proc.get("pid") is None:
                        continue
                    pid = int(proc.get("pid"))
                    read_total = int(proc.get("read_bytes", 0))
                    write_total = int(proc.get("write_bytes", 0))
                    prev = prev_by_pid.get(pid, {})
                    read_delta = max(0, read_total - int(prev.get("read_bytes", read_total)))
                    write_delta = max(0, write_total - int(prev.get("write_bytes", write_total)))
                    writer.writerow(
                        {
                            "timestamp_utc": ts,
                            "pid": pid,
                            "cmdline": proc.get("cmdline"),
                            "selection_mode": mode,
                            "read_bytes_total": read_total,
                            "write_bytes_total": write_total,
                            "read_bytes_delta": read_delta,
                            "write_bytes_delta": write_delta,
                            "syscr_total": int(proc.get("syscr", 0)),
                            "syscw_total": int(proc.get("syscw", 0)),
                            "source": "procfs_io",
                        }
                    )
                    prev_by_pid[pid] = proc
        legacy_path = self.io_dir / "pidstat.csv"
        shutil.copyfile(out_path, legacy_path)

    def _build_kvbm_file_phase_summary(self) -> dict[str, Any]:
        phase_payload: dict[str, Any] = {}
        for phase, window in self.phase_windows.items():
            start = window.get("start")
            end = window.get("end")
            start_dt = _parse_iso8601(start)
            end_dt = _parse_iso8601(end)
            if start_dt is None or end_dt is None:
                phase_payload[phase] = []
                continue
            counts: dict[tuple[int, str], int] = {}
            for sample in self.lsof_samples:
                sample_dt = _parse_iso8601(str(sample.get("timestamp_utc") or ""))
                if sample_dt is None or sample_dt < start_dt or sample_dt > end_dt:
                    continue
                for entry in (sample.get("entries") or []):
                    if not isinstance(entry, dict):
                        continue
                    try:
                        pid = int(entry.get("pid"))
                    except Exception:
                        continue
                    path = str(entry.get("path") or "")
                    if not path:
                        continue
                    key = (pid, path)
                    counts[key] = counts.get(key, 0) + 1
            rows = [
                {
                    "pid": pid,
                    "path": path,
                    "observed_samples": count,
                    "read_bytes": None,
                    "write_bytes": None,
                }
                for (pid, path), count in counts.items()
            ]
            rows.sort(key=lambda item: int(item.get("observed_samples", 0)), reverse=True)
            phase_payload[phase] = rows
        available = any(bool(items) for items in phase_payload.values())
        return {
            "available": available,
            "method": "lsof_open_file_observation",
            "reason": (
                None
                if available
                else "file-level read/write byte attribution unavailable without privileged syscall tracing"
            ),
            "phases": phase_payload,
        }

    def build_report(
        self,
        *,
        phase_block_deltas: dict[str, Optional[dict[str, Any]]],
        phase_process_deltas: dict[str, Optional[dict[str, Any]]],
    ) -> dict[str, Any]:
        block_io_by_phase: dict[str, dict[str, Any]] = {}
        process_io_by_phase: dict[str, dict[str, Any]] = {}
        for phase, window in self.phase_windows.items():
            before_block = _find_snapshot_for_time(self.block_samples, window.get("start"), prefer="before")
            after_block = _find_snapshot_for_time(self.block_samples, window.get("end"), prefer="after")
            sampled_block = _block_delta_from_samples(before_block, after_block)
            phase_block = phase_block_deltas.get(phase) if phase in phase_block_deltas else None
            block_io_by_phase[phase] = {
                "read_bytes": (
                    int(phase_block.get("read_bytes_delta", 0))
                    if isinstance(phase_block, dict)
                    else (sampled_block.get("read_bytes") if sampled_block else None)
                ),
                "write_bytes": (
                    int(phase_block.get("write_bytes_delta", 0))
                    if isinstance(phase_block, dict)
                    else (sampled_block.get("write_bytes") if sampled_block else None)
                ),
                "sampled_read_bytes": (sampled_block.get("read_bytes") if sampled_block else None),
                "sampled_write_bytes": (sampled_block.get("write_bytes") if sampled_block else None),
                "source": (
                    "phase_boundary_block_device_delta" if isinstance(phase_block, dict) else "timeline_sampling_fallback"
                ),
            }

            before_proc = _find_snapshot_for_time(self.process_samples, window.get("start"), prefer="before")
            after_proc = _find_snapshot_for_time(self.process_samples, window.get("end"), prefer="after")
            sampled_proc = _process_delta_from_samples(before_proc, after_proc)
            phase_proc = phase_process_deltas.get(phase) if phase in phase_process_deltas else None
            if isinstance(phase_proc, dict):
                process_io_by_phase[phase] = {
                    "read_bytes": int(phase_proc.get("read_bytes_delta", 0)),
                    "write_bytes": int(phase_proc.get("write_bytes_delta", 0)),
                    "cgroup_read_bytes": phase_proc.get("cgroup_read_bytes_delta"),
                    "cgroup_write_bytes": phase_proc.get("cgroup_write_bytes_delta"),
                    "per_pid": list(phase_proc.get("per_pid_deltas") or []),
                    "source": "phase_boundary_process_delta",
                    "sampled_read_bytes": sampled_proc.get("read_bytes"),
                    "sampled_write_bytes": sampled_proc.get("write_bytes"),
                }
            else:
                process_io_by_phase[phase] = {
                    "read_bytes": sampled_proc.get("read_bytes"),
                    "write_bytes": sampled_proc.get("write_bytes"),
                    "per_pid": list(sampled_proc.get("per_pid") or []),
                    "source": "timeline_sampling_fallback",
                }

        direct_hints = _collect_direct_io_hints(self.kv_runtime_env)
        bpf_probe = None
        observed_flags: list[str] = []
        if self.tools_available.get("bpftrace"):
            bpf_probe = _run_capture_command(["bpftrace", "-e", 'BEGIN { printf("ready\\n"); exit(); }'], timeout_s=4.0)
            if not bpf_probe.get("success"):
                _register_capture_error(
                    self.capture_errors,
                    command=["bpftrace", "-e", 'BEGIN { printf("ready\\n"); exit(); }'],
                    return_code=bpf_probe.get("return_code"),
                    stderr=str(bpf_probe.get("stderr") or bpf_probe.get("error") or "bpftrace_probe_failed"),
                )
        replay_like_phases = [name for name in block_io_by_phase.keys() if str(name).startswith("replay")]
        replay_corroborating_behavior = False
        for phase_name in replay_like_phases:
            block_read = _to_int((block_io_by_phase.get(phase_name) or {}).get("read_bytes"))
            proc_read = _to_int((process_io_by_phase.get(phase_name) or {}).get("read_bytes"))
            if block_read > 0 and proc_read > 0:
                replay_corroborating_behavior = True
                break
        has_config_hints = bool((direct_hints.get("env_hints") or {}) or (direct_hints.get("kvbm_config_direct_mentions") or []))
        confidence = "low"
        direct_method = "none"
        if observed_flags:
            confidence = "high"
            direct_method = "syscall_tracing_observed_o_direct"
        elif has_config_hints and replay_corroborating_behavior:
            confidence = "medium"
            direct_method = "config_hints_plus_replay_io_corroboration"
        elif has_config_hints:
            confidence = "low"
            direct_method = "config_hints_only"
        report = {
            "schema_version": 1,
            "capture_timestamp": now_utc_iso(),
            "enabled": True,
            "available": True,
            "primary_nvme_device": self.primary_nvme_device,
            "kvbm_disk_path": self.kvbm_disk_path,
            "phase_windows": self.phase_windows,
            "block_io_by_phase": block_io_by_phase,
            "process_io_by_phase": process_io_by_phase,
            "kvbm_file_io_by_phase": self._build_kvbm_file_phase_summary(),
            "direct_io_evidence": {
                "method": direct_method,
                "observed_flags": observed_flags,
                "config_hints": direct_hints,
                "corroborating_behavior": {
                    "replay_like_phases": replay_like_phases,
                    "replay_block_and_process_read_positive": replay_corroborating_behavior,
                },
                "bpftrace_probe": {
                    "available": bool(self.tools_available.get("bpftrace")),
                    "success": (bool(bpf_probe.get("success")) if isinstance(bpf_probe, dict) else None),
                    "error": (bpf_probe.get("error") if isinstance(bpf_probe, dict) else None),
                },
                "confidence": confidence,
            },
            "capture_errors": self.capture_errors,
            "collection_methods": {
                "timeline_block_io": "procfs_/sys/block/<dev>/stat (+iostat raw log when installed)",
                "timeline_process_io": "procfs_/proc/<pid>/io (+pidstat raw log when installed)",
                "file_attribution": "lsof_open_file_observation_fallback",
                "meminfo_vmstat": "procfs_snapshots",
            },
            "tools_available": self.tools_available,
            "artifacts": {
                "iostat_csv": str(self.io_dir / "iostat.csv"),
                "pidstat_csv": str(self.io_dir / "pidstat.csv"),
                "block_stat_timeline_csv": str(self.io_dir / "block_stat_timeline.csv"),
                "proc_io_timeline_csv": str(self.io_dir / "proc_io_timeline.csv"),
                "iostat_raw_log": str(self.io_dir / "iostat.raw.log"),
                "pidstat_raw_log": str(self.io_dir / "pidstat.raw.log"),
                "meminfo_snapshots_jsonl": str(self.io_dir / "meminfo_snapshots.jsonl"),
                "vmstat_snapshots_jsonl": str(self.io_dir / "vmstat_snapshots.jsonl"),
            },
        }
        return report

    def finalize(
        self,
        *,
        phase_block_deltas: dict[str, Optional[dict[str, Any]]],
        phase_process_deltas: dict[str, Optional[dict[str, Any]]],
    ) -> dict[str, Any]:
        self.stop()
        self._write_iostat_csv()
        self._write_pidstat_csv()
        self._write_mem_vm_jsonl()
        report = self.build_report(
            phase_block_deltas=phase_block_deltas,
            phase_process_deltas=phase_process_deltas,
        )
        out_path = self.io_dir / "io_attribution_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


async def resolve_model_id_with_retry(client: Any, args: argparse.Namespace) -> str:
    deadline = time.monotonic() + max(0.0, float(args.model_resolve_timeout_s))
    poll_s = max(0.1, float(args.model_resolve_poll_s))
    last_error: Optional[str] = None
    attempt = 0
    while True:
        attempt += 1
        try:
            return await client.fetch_first_model_id()
        except Exception as exc:
            last_error = str(exc)
            now = time.monotonic()
            if now >= deadline:
                break
            remaining = max(0.0, deadline - now)
            LOG.info(
                "Model not discoverable yet (attempt=%s, remaining=%.1fs): %s",
                attempt,
                remaining,
                last_error,
            )
            await asyncio.sleep(min(poll_s, remaining))
    raise RuntimeError(
        "Timed out waiting for model discovery from /v1/models "
        f"after {args.model_resolve_timeout_s:.1f}s. Last error: {last_error}"
    )


async def run_benchmark(args: argparse.Namespace) -> tuple[Path, bool]:
    try:
        from .openai_compat import OpenAICompatClient
    except ModuleNotFoundError as exc:
        if exc.name == "httpx":
            raise RuntimeError("Missing dependency `httpx`; run `pip install -r requirements.txt`.") from exc
        raise

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    telemetry_dir = run_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = (run_dir / "responses") if args.store_responses else None
    if responses_dir is not None:
        responses_dir.mkdir(parents=True, exist_ok=True)

    estimator = TokenEstimator(args.tokenizer)
    short_range = parse_token_range(args.short_range, "short-range")
    long_range = parse_token_range(args.long_range, "long-range")
    kv_mode_resolved = resolve_kv_mode(args)
    kvbm_enabled = is_kvbm_enabled(kv_mode_resolved)
    kvbm_skip_reason = "kvbm_disabled"
    kv_runtime_env = collect_env_snapshot(KVBM_ENV_KEYS)
    phase_plan = build_phase_plan(
        args=args,
        estimator=estimator,
        short_range=short_range,
        long_range=long_range,
    )
    phase_delta_dir = run_dir / "phase_deltas"
    phase_delta_dir.mkdir(parents=True, exist_ok=True)
    phase_delta_artifacts: dict[str, dict[str, str]] = {}
    unique_prompts = dedupe_prompts([p for phase in phase_plan for p in phase.prompts])
    write_prompt_manifest(run_dir / "prompts_manifest.jsonl", unique_prompts)
    write_request_manifest(run_dir / "request_manifest.jsonl", phase_plan)
    preflight = preflight_validate_prompts(unique_prompts, args)

    telemetry_manager: Optional[TelemetryManager] = None
    telemetry_start_report = None
    telemetry_stop_report = None
    kvbm_probe = KVBMMetricsProbe(args.kvbm_metrics_url, args.kvbm_metrics_timeout_s) if kvbm_enabled else None
    metrics_snapshot_dir = Path(args.metrics_snapshot_dir) if args.metrics_snapshot_dir else run_dir
    metrics_capture_records: list[dict[str, Any]] = []
    metrics_capture_texts: dict[str, str] = {}
    metrics_inventory_keywords = split_inventory_keywords(args.metrics_inventory_keywords)
    if args.capture_metrics_snapshot:
        metrics_snapshot_dir.mkdir(parents=True, exist_ok=True)
    kvbm_snapshots: list[dict[str, Any]] = []
    kvbm_phase_deltas: dict[str, dict[str, Any]] = {}
    kvbm_snapshots_path = run_dir / "kvbm_metrics_snapshots.jsonl"

    requests_path = run_dir / "requests.jsonl"
    request_counter = itertools.count(1)
    overall_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    phase_summaries: list[dict[str, Any]] = []
    run_valid = True
    invalid_reason: Optional[str] = None
    invalid_details: list[str] = []
    model_id = args.model_id
    model_count_end: Optional[int] = None

    nvme_identity = collect_nvme_identity(args.nvme_device)
    nvme_smart_pre = collect_nvme_smart(args.nvme_device)
    nvme_smart_post: dict[str, Any] = {}
    device_metadata_pre = collect_device_metadata_safe(
        capture_stage="pre",
        nvme_device_hint=args.nvme_device,
        kvbm_cache_dir=args.kvbm_cache_dir,
        phase_io_device=args.phase_io_device,
    )
    device_metadata_post: dict[str, Any] = {}
    storage_summary_pre = extract_storage_summary(device_metadata_pre)
    io_attribution_collector: Optional[IOAttributionCollector] = None
    io_attribution_report: dict[str, Any] = {}
    phase_windows: dict[str, dict[str, str]] = {}
    phase_block_deltas: dict[str, Optional[dict[str, Any]]] = {}
    phase_process_deltas: dict[str, Optional[dict[str, Any]]] = {}
    (run_dir / "nvme_identity.json").write_text(json.dumps(nvme_identity, indent=2), encoding="utf-8")
    (run_dir / "nvme_smart_pre.json").write_text(json.dumps(nvme_smart_pre, indent=2), encoding="utf-8")
    (run_dir / "device_metadata_pre.json").write_text(json.dumps(device_metadata_pre, indent=2), encoding="utf-8")
    if args.io_attrib:
        try:
            io_attribution_collector = IOAttributionCollector(
                run_dir=run_dir,
                primary_nvme_device=str(storage_summary_pre.get("primary_nvme_device") or args.nvme_device),
                block_device=args.phase_io_device,
                proc_pattern=args.worker_proc_pattern,
                kvbm_disk_path=args.kvbm_cache_dir,
                interval_s=float(args.io_attrib_interval_s),
                top_procs=int(args.io_attrib_top_procs),
                kv_runtime_env=kv_runtime_env,
            )
            io_attribution_collector.start()
        except Exception as exc:  # noqa: BLE001
            LOG.exception("I/O attribution start failed.")
            io_attribution_report = _empty_io_attribution_report(
                primary_nvme_device=str(storage_summary_pre.get("primary_nvme_device") or args.nvme_device),
                kvbm_disk_path=args.kvbm_cache_dir,
                phase_windows=phase_windows,
                error=str(exc),
                capture_errors=[
                    {
                        "command": "io_attribution_start",
                        "return_code": None,
                        "stderr_snippet": str(exc)[:240],
                    }
                ],
            )

    run_config = {
        "run_id": run_id,
        "created_utc": now_utc_iso(),
        "base_url": args.base_url,
        "model_id": model_id,
        "scenario": args.scenario,
        "prompt_set": args.prompt_set,
        "short_range": short_range,
        "long_range": long_range,
        "tokenizer": estimator.tokenizer_name or "heuristic",
        "tier_mode": kv_mode_resolved.get("tier_mode"),
        "kv_mode": kv_mode_resolved,
        "kv_runtime_env": kv_runtime_env,
        "preflight": preflight,
        "client_request_parameters": {
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "request_seed": (int(args.request_seed) if args.request_seed is not None else None),
            "stop": list(args.stop),
            "stream": bool(args.stream),
            "stream_timeout_s": (
                float(args.stream_timeout_s) if args.stream_timeout_s is not None else float(args.timeout_s)
            ),
            "stream_record_ttfb": bool(args.stream_record_ttfb),
        },
        "args": vars(args),
        "phases": [
            {
                "name": phase.name,
                "concurrency": phase.concurrency,
                "requests": len(phase.prompts),
                "include_in_overall": phase.include_in_overall,
            }
            for phase in phase_plan
        ],
    }

    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    write_manifest(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "created_utc": run_config["created_utc"],
            "scenario": args.scenario,
            "tier_mode": kv_mode_resolved.get("tier_mode"),
            "kv_mode": kv_mode_resolved,
            "model_id_requested": args.model_id,
            "base_url": args.base_url,
            "phases": run_config["phases"],
            "kv_runtime_env": kv_runtime_env,
            "nvme_device": args.nvme_device,
            "args": vars(args),
        },
    )

    executed_workload = not preflight["failed"]
    if preflight["failed"]:
        run_valid = False
        invalid_reason = "prompt_preflight_failed"
        invalid_details.extend(preflight["errors"])
        LOG.error("Prompt preflight failed: %s", preflight["errors"])
    else:
        if args.collect_telemetry:
            telemetry_manager = TelemetryManager(
                run_dir=run_dir,
                kvbm_cache_dir=args.kvbm_cache_dir,
                container_name=args.container_name,
                iostat_device=args.iostat_device,
                interval_s=args.telemetry_interval_s,
                pid_target=args.telemetry_pid,
            )
            telemetry_start_report = telemetry_manager.start_default()

        try:
            async with OpenAICompatClient(args.base_url, timeout_s=args.timeout_s) as client:
                if model_id == "auto":
                    model_id = await resolve_model_id_with_retry(client, args)
                    LOG.info("Resolved model ID from /v1/models: %s", model_id)
                run_config["model_id"] = model_id
                (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
                write_manifest(
                    run_dir / "manifest.json",
                    {
                        "run_id": run_id,
                        "created_utc": run_config["created_utc"],
                        "scenario": args.scenario,
                        "tier_mode": kv_mode_resolved.get("tier_mode"),
                        "kv_mode": kv_mode_resolved,
                        "model_id_requested": args.model_id,
                        "model_id_resolved": model_id,
                        "base_url": args.base_url,
                        "phases": run_config["phases"],
                        "kv_runtime_env": kv_runtime_env,
                        "nvme_device": args.nvme_device,
                        "args": vars(args),
                    },
                )

                if kvbm_probe is not None:
                    snap = kvbm_probe.snapshot("run_start")
                    kvbm_snapshots.append(snap)
                    append_jsonl(kvbm_snapshots_path, snap)

                lock = asyncio.Lock()
                with requests_path.open("w", encoding="utf-8") as request_fp:
                    for phase in phase_plan:
                        LOG.info(
                            "Running phase=%s requests=%s concurrency=%s",
                            phase.name,
                            len(phase.prompts),
                            phase.concurrency,
                        )
                        phase_start_ts = now_utc_iso()
                        phase_windows.setdefault(phase.name, {})["start"] = phase_start_ts
                        if io_attribution_collector is not None:
                            io_attribution_collector.mark_phase_boundary(
                                phase=phase.name,
                                boundary="start",
                                timestamp_utc=phase_start_ts,
                            )
                        if telemetry_manager is not None:
                            telemetry_manager.snapshot_kvbm_dir(f"before_{phase.name}")
                        before_io = read_block_device_stats(args.phase_io_device)
                        before_proc_io = read_worker_process_io(args.worker_proc_pattern, args.container_name)
                        if kvbm_probe is not None:
                            before_kvbm = kvbm_probe.snapshot(f"before_{phase.name}")
                            kvbm_snapshots.append(before_kvbm)
                            append_jsonl(kvbm_snapshots_path, before_kvbm)
                        else:
                            before_kvbm = make_kvbm_skipped_snapshot(f"before_{phase.name}", kvbm_skip_reason)

                        phase_rows, phase_duration_s = await run_phase(
                            phase=phase,
                            client=client,
                            model_id=model_id,
                            args=args,
                            estimator=estimator,
                            request_counter=request_counter,
                            requests_fp=request_fp,
                            requests_lock=lock,
                            responses_dir=responses_dir,
                        )

                        phase_end_ts = now_utc_iso()
                        phase_windows.setdefault(phase.name, {})["end"] = phase_end_ts
                        if io_attribution_collector is not None:
                            io_attribution_collector.mark_phase_boundary(
                                phase=phase.name,
                                boundary="end",
                                timestamp_utc=phase_end_ts,
                            )
                        after_io = read_block_device_stats(args.phase_io_device)
                        after_proc_io = read_worker_process_io(args.worker_proc_pattern, args.container_name)
                        if kvbm_probe is not None:
                            after_kvbm = kvbm_probe.snapshot(f"after_{phase.name}")
                            kvbm_snapshots.append(after_kvbm)
                            append_jsonl(kvbm_snapshots_path, after_kvbm)
                        else:
                            after_kvbm = make_kvbm_skipped_snapshot(f"after_{phase.name}", kvbm_skip_reason)
                        if telemetry_manager is not None:
                            telemetry_manager.snapshot_kvbm_dir(f"after_{phase.name}")

                        suffix = METRICS_SNAPSHOT_PHASE_SUFFIX.get(phase.name)
                        if suffix is None and str(phase.name).startswith("replay"):
                            suffix = "replay"
                        if suffix is None and str(phase.name).startswith("thrash"):
                            suffix = "pressure"
                        if args.capture_metrics_snapshot and suffix:
                            system_snap = fetch_metrics_text(args.metrics_system_url, args.kvbm_metrics_timeout_s)
                            if kvbm_probe is not None:
                                kvbm_snap = fetch_metrics_text(args.metrics_kvbm_url, args.kvbm_metrics_timeout_s)
                            else:
                                kvbm_snap = {
                                    "success": False,
                                    "error": kvbm_skip_reason,
                                    "raw_text": "",
                                    "skipped": True,
                                    "reason": kvbm_skip_reason,
                                }
                            metrics_capture_records.append(
                                {
                                    "phase": phase.name,
                                    "suffix": suffix,
                                    "system": {
                                        "url": args.metrics_system_url,
                                        "success": bool(system_snap.get("success")),
                                        "error": system_snap.get("error"),
                                    },
                                    "kvbm": {
                                        "url": args.metrics_kvbm_url,
                                        "success": bool(kvbm_snap.get("success")),
                                        "error": kvbm_snap.get("error"),
                                    },
                                }
                            )
                            if system_snap.get("success"):
                                metrics_capture_texts[f"system_{suffix}"] = str(system_snap.get("raw_text") or "")
                            if kvbm_snap.get("success"):
                                metrics_capture_texts[f"kvbm_{suffix}"] = str(kvbm_snap.get("raw_text") or "")
                            write_metrics_snapshot(
                                metrics_snapshot_dir / f"metrics_system_{suffix}.prom",
                                phase.name,
                                system_snap,
                            )
                            write_metrics_snapshot(
                                metrics_snapshot_dir / f"metrics_kvbm_{suffix}.prom",
                                phase.name,
                                kvbm_snap,
                            )
                            # Keep legacy names for downstream consumers that still expect these.
                            write_metrics_snapshot(
                                metrics_snapshot_dir / f"metrics_{suffix}.prom",
                                phase.name,
                                system_snap,
                            )

                        io_delta = diff_block_device_stats(before_io, after_io)
                        proc_io_delta = diff_process_io_stats(before_proc_io, after_proc_io)
                        phase_block_deltas[phase.name] = io_delta
                        phase_process_deltas[phase.name] = proc_io_delta
                        if kvbm_probe is not None:
                            kvbm_delta = kvbm_probe.delta(before_kvbm, after_kvbm)
                        else:
                            kvbm_delta = make_kvbm_skipped_delta(kvbm_skip_reason)
                        kvbm_phase_deltas[phase.name] = kvbm_delta
                        phase_name_safe = safe_name(phase.name)

                        phase_kvbm_start_path = phase_delta_dir / f"phase_{phase_name_safe}_kvbm_metrics_start.json"
                        phase_kvbm_end_path = phase_delta_dir / f"phase_{phase_name_safe}_kvbm_metrics_end.json"
                        phase_kvbm_delta_path = phase_delta_dir / f"phase_{phase_name_safe}_kvbm_metrics_delta.json"
                        phase_os_start_path = phase_delta_dir / f"phase_{phase_name_safe}_os_io_start.json"
                        phase_os_end_path = phase_delta_dir / f"phase_{phase_name_safe}_os_io_end.json"
                        phase_os_delta_path = phase_delta_dir / f"phase_{phase_name_safe}_os_io_delta.json"

                        phase_os_start_payload = {
                            "phase": phase.name,
                            "block_device": args.phase_io_device,
                            "block_device_stats": before_io,
                            "worker_process_pattern": args.worker_proc_pattern,
                            "worker_process_io": before_proc_io,
                        }
                        phase_os_end_payload = {
                            "phase": phase.name,
                            "block_device": args.phase_io_device,
                            "block_device_stats": after_io,
                            "worker_process_pattern": args.worker_proc_pattern,
                            "worker_process_io": after_proc_io,
                        }
                        phase_os_delta_payload = {
                            "phase": phase.name,
                            "block_device": args.phase_io_device,
                            "block_device_delta": io_delta,
                            "worker_process_pattern": args.worker_proc_pattern,
                            "worker_process_io_delta": proc_io_delta,
                        }

                        write_manifest(phase_kvbm_start_path, before_kvbm)
                        write_manifest(phase_kvbm_end_path, after_kvbm)
                        write_manifest(phase_kvbm_delta_path, kvbm_delta)
                        write_manifest(phase_os_start_path, phase_os_start_payload)
                        write_manifest(phase_os_end_path, phase_os_end_payload)
                        write_manifest(phase_os_delta_path, phase_os_delta_payload)

                        phase_delta_artifacts[phase.name] = {
                            "kvbm_metrics_start": str(phase_kvbm_start_path),
                            "kvbm_metrics_end": str(phase_kvbm_end_path),
                            "kvbm_metrics_delta": str(phase_kvbm_delta_path),
                            "os_io_start": str(phase_os_start_path),
                            "os_io_end": str(phase_os_end_path),
                            "os_io_delta": str(phase_os_delta_path),
                        }

                        summary = summarize_phase(phase_rows, phase_duration_s)
                        summary["phase"] = phase.name
                        summary["phase_start_ts"] = phase_start_ts
                        summary["phase_end_ts"] = phase_end_ts
                        summary["concurrency"] = phase.concurrency
                        summary["requests"] = len(phase.prompts)
                        summary["kvbm_metrics_delta"] = kvbm_delta
                        if io_delta is not None:
                            summary["block_device"] = args.phase_io_device
                            summary["io_delta"] = io_delta
                        if proc_io_delta is not None:
                            summary["worker_process_pattern"] = args.worker_proc_pattern
                            summary["worker_process_io_delta"] = proc_io_delta
                        phase_summaries.append(summary)
                        all_rows.extend(phase_rows)
                        if phase.include_in_overall:
                            overall_rows.extend(phase_rows)

                        if has_prompt_limit_error(phase_rows):
                            run_valid = False
                            invalid_reason = "engine_prompt_limit_exceeded"
                            invalid_details.append(
                                f"Prompt length error observed in phase `{phase.name}`. Reduce long-range or increase engine cap."
                            )
                            LOG.error("Invalidating run due to prompt limit error in phase=%s", phase.name)
                            break

                        if has_status_code(phase_rows, 404) or has_status_code(phase_rows, 500):
                            model_count_end = await client.count_models()
                            summary["models_visible_after_phase"] = model_count_end
                            if model_count_end == 0:
                                run_valid = False
                                invalid_reason = "model_dropped"
                                invalid_details.append(
                                    f"Model registry empty after phase `{phase.name}`. Frontend discovery dropped model."
                                )
                                LOG.error("Invalidating run due to model drop after phase=%s", phase.name)
                                break

                if kvbm_probe is not None:
                    snap = kvbm_probe.snapshot("run_end")
                    kvbm_snapshots.append(snap)
                    append_jsonl(kvbm_snapshots_path, snap)
                model_count_end = await client.count_models()
        except Exception as exc:  # noqa: BLE001
            run_valid = False
            invalid_reason = "runtime_exception"
            invalid_details.append(str(exc))
            LOG.exception("Benchmark runtime failed.")
        finally:
            if telemetry_manager is not None:
                telemetry_stop_report = telemetry_manager.stop_default()

    nvme_smart_post = collect_nvme_smart(args.nvme_device)
    device_metadata_post = collect_device_metadata_safe(
        capture_stage="post",
        nvme_device_hint=args.nvme_device,
        kvbm_cache_dir=args.kvbm_cache_dir,
        phase_io_device=args.phase_io_device,
    )
    (run_dir / "nvme_smart_post.json").write_text(json.dumps(nvme_smart_post, indent=2), encoding="utf-8")
    (run_dir / "device_metadata_post.json").write_text(json.dumps(device_metadata_post, indent=2), encoding="utf-8")
    if io_attribution_collector is not None:
        try:
            io_attribution_report = io_attribution_collector.finalize(
                phase_block_deltas=phase_block_deltas,
                phase_process_deltas=phase_process_deltas,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("I/O attribution finalize failed.")
            io_attribution_report = _empty_io_attribution_report(
                primary_nvme_device=str(storage_summary_pre.get("primary_nvme_device") or args.nvme_device),
                kvbm_disk_path=args.kvbm_cache_dir,
                phase_windows=phase_windows,
                error=str(exc),
                capture_errors=[
                    {
                        "command": "io_attribution_finalize",
                        "return_code": None,
                        "stderr_snippet": str(exc)[:240],
                    }
                ],
            )
    if args.io_attrib:
        io_dir = run_dir / "io"
        io_dir.mkdir(parents=True, exist_ok=True)
        if not io_attribution_report:
            io_attribution_report = _empty_io_attribution_report(
                primary_nvme_device=str(storage_summary_pre.get("primary_nvme_device") or args.nvme_device),
                kvbm_disk_path=args.kvbm_cache_dir,
                phase_windows=phase_windows,
                error="io_attribution_report_unavailable",
                capture_errors=[],
            )
        if not isinstance(io_attribution_report.get("phase_windows"), dict):
            io_attribution_report["phase_windows"] = phase_windows
        (io_dir / "io_attribution_report.json").write_text(
            json.dumps(io_attribution_report, indent=2),
            encoding="utf-8",
        )

    telemetry_payload = {
        "started": telemetry_start_report.__dict__ if telemetry_start_report else None,
        "stopped": telemetry_stop_report.__dict__ if telemetry_stop_report else None,
    }
    (telemetry_dir / "telemetry_report.json").write_text(
        json.dumps(telemetry_payload, indent=2),
        encoding="utf-8",
    )

    if args.capture_metrics_snapshot:
        broad_samples = collect_metric_samples(
            metrics_capture_texts,
            metrics_inventory_keywords,
            include_regex=re.compile(r"dynamo_component_.*kvbm", re.IGNORECASE),
        )
        kvbm_name_samples = collect_metric_samples(metrics_capture_texts, ["kvbm"])
        kvbm_from_6880_texts = {k: v for k, v in metrics_capture_texts.items() if k.startswith("kvbm_")}
        kvbm_onboard_offload_samples = collect_metric_samples(
            kvbm_from_6880_texts,
            ["onboard", "offload", "matched"],
        )
        write_metric_inventory(
            metrics_snapshot_dir / "kvbm_metric_inventory_expanded.txt",
            "Expanded metric inventory",
            metrics_inventory_keywords,
            broad_samples,
        )
        write_metric_inventory(
            metrics_snapshot_dir / "kvbm_metric_inventory.txt",
            "KVBM-prefixed metric inventory",
            ["kvbm"],
            kvbm_name_samples,
        )
        write_metric_inventory(
            metrics_snapshot_dir / "kvbm_metric_inventory_from_6880.txt",
            "KVBM 6880 onboard/offload/matched inventory",
            ["onboard", "offload", "matched"],
            kvbm_onboard_offload_samples,
        )

    kvbm_summary = build_kvbm_metrics_summary(
        kvbm_snapshots,
        kvbm_phase_deltas,
        args.scenario,
        str(kv_mode_resolved.get("mode")),
        kvbm_enabled,
    )
    kvbm_status = kvbm_summary.get("kvbm_metrics_status") if isinstance(kvbm_summary, dict) else None
    if not isinstance(kvbm_status, dict):
        kvbm_status = build_kvbm_metrics_status(
            kvbm_enabled=kvbm_enabled,
            metrics_available=bool((kvbm_summary or {}).get("available")),
            snapshot_count=len(kvbm_snapshots),
            reason=((kvbm_summary or {}).get("reason") if isinstance(kvbm_summary, dict) else None),
        )
    request_identity_summary = analyze_request_identity(all_rows, args.scenario)
    if (
        executed_workload
        and kvbm_enabled
        and not args.allow_missing_kvbm_metrics
        and not kvbm_summary.get("available", False)
    ):
        run_valid = False
        if invalid_reason is None:
            invalid_reason = "kvbm_metrics_unavailable"
        invalid_details.append("KVBM metrics endpoint unavailable; run rejected for measurement-grade validation.")

    if run_valid:
        overall_duration_s = sum(
            float(item.get("duration_s", 0.0))
            for item in phase_summaries
            if item.get("phase") != "warmup" and bool(item.get("phase") in [p.name for p in phase_plan if p.include_in_overall])
        )
        overall_summary = summarize_phase(overall_rows, overall_duration_s)
        overall_summary["phase"] = "overall"
        overall_summary["scenario"] = args.scenario
    else:
        overall_summary = {
            "phase": "overall",
            "scenario": args.scenario,
            "excluded": True,
            "reason": invalid_reason or "invalid_run",
            "details": invalid_details,
        }

    if args.scenario == "eviction_replay":
        overall_summary["eviction_replay_signal_io"] = analyze_eviction_signal_io(phase_summaries)
        overall_summary["eviction_replay_signal_kvbm"] = kvbm_summary.get("eviction_replay_signal")
    if args.scenario == "rehydrate_replay":
        overall_summary["rehydrate_replay_signal_io"] = analyze_rehydrate_signal_io(phase_summaries)
        overall_summary["rehydrate_replay_signal_kvbm"] = kvbm_summary.get("rehydrate_replay_signal")
    if args.scenario == "reuse_verify":
        overall_summary["reuse_verify_signal_kvbm"] = kvbm_summary.get("reuse_verify_signal")
        overall_summary["reuse_verify_identity"] = request_identity_summary.get("reuse_verify_identity")

    fingerprint = collect_fingerprint(args.container_name)
    summary_payload = {
        "run_id": run_id,
        "created_utc": now_utc_iso(),
        "run_valid": run_valid,
        "invalid_reason": invalid_reason,
        "invalid_details": invalid_details,
        "model_id": model_id,
        "model_count_end": model_count_end,
        "scenario": args.scenario,
        "stream": bool(args.stream),
        "stream_timeout_s": (float(args.stream_timeout_s) if args.stream_timeout_s is not None else None),
        "stream_record_ttfb": bool(args.stream_record_ttfb),
        "mode": kv_mode_resolved.get("tier_mode"),
        "tier_mode": kv_mode_resolved.get("tier_mode"),
        "kvbm_enabled": kvbm_enabled,
        "kvbm_metrics_available": bool(kvbm_status.get("metrics_available")),
        "kvbm_metrics_status": kvbm_status,
        "kv_mode": kv_mode_resolved,
        "kv_runtime_env": kv_runtime_env,
        "variant_tags": list(args.variant_tag or []),
        "request_manifest_path": str(run_dir / "request_manifest.jsonl"),
        "phase_delta_artifacts": phase_delta_artifacts,
        "nvme": {
            "device": args.nvme_device,
            "identity": nvme_identity,
            "smart_pre": nvme_smart_pre,
            "smart_post": nvme_smart_post,
            "identity_path": str(run_dir / "nvme_identity.json"),
            "smart_pre_path": str(run_dir / "nvme_smart_pre.json"),
            "smart_post_path": str(run_dir / "nvme_smart_post.json"),
            "device_metadata_pre_path": str(run_dir / "device_metadata_pre.json"),
            "device_metadata_post_path": str(run_dir / "device_metadata_post.json"),
        },
        "storage": {
            **storage_summary_pre,
            "device_metadata_pre_path": str(run_dir / "device_metadata_pre.json"),
            "device_metadata_post_path": str(run_dir / "device_metadata_post.json"),
            "device_metadata_pre_capture_errors": len(device_metadata_pre.get("capture_errors") or []),
            "device_metadata_post_capture_errors": len(device_metadata_post.get("capture_errors") or []),
        },
        "io_attribution": {
            "enabled": bool(args.io_attrib),
            "available": (bool(io_attribution_report.get("available")) if args.io_attrib else False),
            "report_path": (str(run_dir / "io" / "io_attribution_report.json") if args.io_attrib else None),
            "capture_error_count": (
                len(io_attribution_report.get("capture_errors") or []) if args.io_attrib else 0
            ),
        },
        "phase_windows": phase_windows,
        "phase_summaries": phase_summaries,
        "overall_summary": overall_summary,
        "kvbm_metrics": kvbm_summary,
        "metrics_snapshot": {
            "enabled": bool(args.capture_metrics_snapshot),
            "snapshot_dir": str(metrics_snapshot_dir) if args.capture_metrics_snapshot else None,
            "records": metrics_capture_records if args.capture_metrics_snapshot else [],
            "inventory_keywords": metrics_inventory_keywords if args.capture_metrics_snapshot else [],
        },
        "request_identity": request_identity_summary,
        "fingerprint": fingerprint,
        "notes": [
            "TTFC is measured from the first non-empty streamed SSE data payload (or first non-empty stream chunk fallback).",
            "TTFT is retained for backward compatibility and mirrors TTFC for stream runs; non-stream TTFT remains a first-byte proxy.",
            "Invalid runs are explicitly marked and excluded from aggregate conclusions.",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report = render_report_markdown(summary_payload)
    (run_dir / args.report_filename).write_text(report, encoding="utf-8")

    LOG.info("Benchmark complete. Artifacts: %s", run_dir)
    return run_dir, run_valid


def resolve_kv_mode(args: argparse.Namespace) -> dict[str, Any]:
    tier_mode_raw = str(args.tier_mode or "").strip().upper()
    tier_to_mode = {
        "B0": "off",
        "B1": "cpu_only",
        "B2": "cpu_disk",
    }
    mode = tier_to_mode.get(tier_mode_raw, args.kv_mode)
    mode_to_tier = {
        "off": "B0",
        "cpu_only": "B1",
        "cpu_disk": "B2",
    }
    tier_mode = mode_to_tier.get(mode, "B2")
    diagnostic = {
        "disable_partial_reuse": bool(args.diagnostic_disable_partial_reuse),
        "disable_block_reuse": bool(args.diagnostic_disable_block_reuse),
        "disable_disk_offload_filter": bool(args.diagnostic_disable_disk_offload_filter),
    }
    if mode == "off":
        return {
            "mode": mode,
            "tier_mode": tier_mode,
            "kvbm_enabled": False,
            "cpu_cache_gb": 0.0,
            "disk_cache_gb": 0.0,
            "diagnostic": diagnostic,
        }
    if mode == "cpu_only":
        return {
            "mode": mode,
            "tier_mode": tier_mode,
            "kvbm_enabled": True,
            "cpu_cache_gb": float(args.kv_cpu_cache_gb if args.kv_cpu_cache_gb is not None else 8.0),
            "disk_cache_gb": 0.0,
            "diagnostic": diagnostic,
        }
    return {
        "mode": mode,
        "tier_mode": tier_mode,
        "kvbm_enabled": True,
        "cpu_cache_gb": float(args.kv_cpu_cache_gb if args.kv_cpu_cache_gb is not None else 8.0),
        "disk_cache_gb": float(args.kv_disk_cache_gb if args.kv_disk_cache_gb is not None else 32.0),
        "diagnostic": diagnostic,
    }


def preflight_validate_prompts(prompts: list[PromptSpec], args: argparse.Namespace) -> dict[str, Any]:
    max_tokens = resolve_phase_max_tokens(args)
    ceiling = args.engine_max_input_tokens - max_tokens - args.input_token_safety_margin
    if ceiling <= 0:
        return {
            "failed": True,
            "ceiling_tokens": ceiling,
            "errors": [
                "Invalid preflight configuration: computed prompt ceiling <= 0 "
                f"(engine={args.engine_max_input_tokens}, max_tokens={max_tokens}, margin={args.input_token_safety_margin})."
            ],
        }
    failures: list[str] = []
    for prompt in prompts:
        if prompt.prompt_tokens_est > ceiling:
            failures.append(
                f"{prompt.prompt_id}: estimated_tokens={prompt.prompt_tokens_est} exceeds ceiling={ceiling}"
            )
    return {"failed": bool(failures), "ceiling_tokens": ceiling, "errors": failures[:20], "failure_count": len(failures)}


def build_phase_plan(
    *,
    args: argparse.Namespace,
    estimator: TokenEstimator,
    short_range: tuple[int, int],
    long_range: tuple[int, int],
) -> list[PhasePlan]:
    phases: list[PhasePlan] = []
    default_stop = ["<|eot_id|>"] if not args.stop else args.stop
    args.stop = default_stop
    if args.scenario == "standard":
        if args.warmup > 0:
            warmup_prompts = generate_prompt_set(
                prompt_set=args.prompt_set,
                count=args.warmup,
                seed=args.seed + 999,
                estimator=estimator,
                short_range=short_range,
                long_range=long_range,
                prompt_id_prefix="warmup",
            )
            phases.append(PhasePlan(name="warmup", prompts=warmup_prompts, concurrency=max(1, args.concurrency), include_in_overall=False))
        measured = generate_prompt_set(
            prompt_set=args.prompt_set,
            count=args.requests,
            seed=args.seed,
            estimator=estimator,
            short_range=short_range,
            long_range=long_range,
            prompt_id_prefix="main",
        )
        phases.append(PhasePlan(name="main", prompts=measured, concurrency=max(1, args.concurrency), include_in_overall=True))
        return phases

    if args.scenario == "reuse_verify":
        repeats = max(2, min(3, int(args.reuse_repeat_count)))
        prompt_set = args.reuse_prompt_set
        base_prompts = generate_prompt_set(
            prompt_set=prompt_set,
            count=1,
            seed=args.seed,
            estimator=estimator,
            short_range=short_range,
            long_range=long_range,
            prompt_id_prefix="reuse",
        )
        if not base_prompts:
            return phases
        identical_prompt = base_prompts[0]
        phases.append(PhasePlan(name="reuse_1", prompts=[identical_prompt], concurrency=1, include_in_overall=True))
        phases.append(PhasePlan(name="reuse_2", prompts=[identical_prompt], concurrency=1, include_in_overall=True))
        if repeats >= 3:
            phases.append(PhasePlan(name="reuse_3", prompts=[identical_prompt], concurrency=1, include_in_overall=True))
        return phases

    if args.scenario == "local_copilot_burst":
        if args.warmup > 0:
            warmup_prompts = generate_local_project_copilot_burst(
                count=args.warmup,
                seed=args.seed + 17,
                estimator=estimator,
                session_count=max(1, args.copilot_session_count),
                burst_size=max(1, args.copilot_burst_size),
                shared_prefix_target_tokens=max(512, args.copilot_shared_prefix_target_tokens),
                prompt_id_prefix="copilot_warmup",
            )
            phases.append(
                PhasePlan(
                    name="warmup",
                    prompts=warmup_prompts,
                    concurrency=max(1, args.concurrency),
                    include_in_overall=False,
                )
            )
        burst_prompts = generate_local_project_copilot_burst(
            count=args.requests,
            seed=args.seed,
            estimator=estimator,
            session_count=max(1, args.copilot_session_count),
            burst_size=max(1, args.copilot_burst_size),
            shared_prefix_target_tokens=max(512, args.copilot_shared_prefix_target_tokens),
            prompt_id_prefix="copilot_main",
        )
        phases.append(
            PhasePlan(
                name="copilot_burst",
                prompts=burst_prompts,
                concurrency=max(1, args.concurrency),
                include_in_overall=True,
            )
        )
        return phases

    if args.scenario == "rehydrate_replay":
        populate_prompts, thrash_prompts = generate_rehydrate_replay_sets(
            populate_sessions=max(1, int(args.rehydrate_populate_sessions)),
            thrash_sessions=max(1, int(args.rehydrate_thrash_sessions)),
            turns=max(1, int(args.rehydrate_turns)),
            seed=args.seed,
            estimator=estimator,
            prefix_target_tokens=max(512, int(args.rehydrate_prefix_target_tokens)),
            prompt_id_prefix="rehydrate",
        )
        populate_conc = max(1, int(args.rehydrate_populate_concurrency or args.concurrency))
        thrash_default = max(populate_conc, int(args.concurrency) * 2)
        thrash_conc = max(1, int(args.rehydrate_thrash_concurrency or thrash_default))
        replay_conc = max(1, int(args.rehydrate_replay_concurrency or populate_conc))
        phases.append(
            PhasePlan(name="populate", prompts=populate_prompts, concurrency=populate_conc, include_in_overall=True)
        )
        phases.append(PhasePlan(name="thrash", prompts=thrash_prompts, concurrency=thrash_conc, include_in_overall=True))
        replay_repeats = max(1, int(args.rehydrate_replay_repeats))
        for replay_idx in range(replay_repeats):
            replay_name = "replay" if replay_idx == 0 else f"replay_{replay_idx + 1}"
            phases.append(
                PhasePlan(name=replay_name, prompts=populate_prompts, concurrency=replay_conc, include_in_overall=True)
            )
        return phases

    if args.warmup > 0:
        warmup_prompts = generate_prompt_set(
            prompt_set="long",
            count=args.warmup,
            seed=args.seed + 7,
            estimator=estimator,
            short_range=short_range,
            long_range=long_range,
            prompt_id_prefix="warmup",
        )
        phases.append(PhasePlan(name="warmup", prompts=warmup_prompts, concurrency=max(1, args.concurrency), include_in_overall=False))

    a_prompts, b_prompts = generate_replay_sets(
        a_count=max(1, args.eviction_a_requests),
        b_count=max(1, args.eviction_b_requests),
        seed=args.seed,
        estimator=estimator,
        long_range=long_range,
    )
    a_conc = max(1, args.eviction_a_concurrency or args.concurrency)
    b_conc_default = max(a_conc, args.concurrency * 2)
    b_conc = max(1, args.eviction_b_concurrency or b_conc_default)

    phases.append(PhasePlan(name="warm_A", prompts=a_prompts, concurrency=a_conc, include_in_overall=True))
    phases.append(PhasePlan(name="pressure_B", prompts=b_prompts, concurrency=b_conc, include_in_overall=True))
    phases.append(PhasePlan(name="replay_A", prompts=a_prompts, concurrency=a_conc, include_in_overall=True))
    return phases


def resolve_phase_max_tokens(args: argparse.Namespace) -> int:
    if args.scenario == "rehydrate_replay" and args.rehydrate_gen_tokens is not None:
        return max(1, int(args.rehydrate_gen_tokens))
    return max(1, int(args.max_tokens))


async def run_phase(
    *,
    phase: PhasePlan,
    client,
    model_id: str,
    args: argparse.Namespace,
    estimator: TokenEstimator,
    request_counter: itertools.count,
    requests_fp,
    requests_lock: asyncio.Lock,
    responses_dir: Optional[Path],
) -> tuple[list[dict[str, Any]], float]:
    if not phase.prompts:
        return [], 0.0
    phase_max_tokens = resolve_phase_max_tokens(args)

    queue: asyncio.Queue[PromptSpec] = asyncio.Queue()
    for prompt in phase.prompts:
        queue.put_nowait(prompt)

    rows: list[dict[str, Any]] = []
    rows_lock = asyncio.Lock()
    phase_start = time.perf_counter()

    async def worker(worker_idx: int) -> None:
        while True:
            try:
                prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            request_idx = next(request_counter)
            start_ts = now_utc_iso()
            request_identity = build_request_identity(
                model_id=model_id,
                prompt_text=prompt.prompt,
                max_tokens=phase_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=args.stop,
                stream=bool(args.stream),
                request_seed=args.request_seed,
            )
            completion = await client.create_completion(
                model=model_id,
                prompt=prompt.prompt,
                max_tokens=phase_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=args.stop,
                stream=bool(args.stream),
                seed=args.request_seed,
                stream_timeout_s=(float(args.stream_timeout_s) if args.stream_timeout_s is not None else None),
                stream_record_ttfb=bool(args.stream_record_ttfb),
            )
            end_ts = now_utc_iso()
            output_text = completion.text or ""
            output_tokens_est = estimator.estimate(output_text) if output_text else 0
            response_path = None
            if responses_dir is not None and output_text:
                fname = f"{phase.name}_{request_idx:06d}_{safe_name(prompt.prompt_id)}.txt"
                path = responses_dir / fname
                path.write_text(output_text, encoding="utf-8")
                response_path = str(path)
            row: dict[str, Any] = {
                "request_index": request_idx,
                "phase": phase.name,
                "worker_idx": worker_idx,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "request_start_ts": start_ts,
                "request_end_ts": end_ts,
                "latency_ms": round(float(completion.latency_ms), 3),
                "ttft_ms": round(float(completion.ttft_ms), 3) if completion.ttft_ms is not None else None,
                "ttfc_ms": round(float(completion.ttfc_ms), 3) if completion.ttfc_ms is not None else None,
                "ttfb_ms": round(float(completion.ttfb_ms), 3) if completion.ttfb_ms is not None else None,
                "status_code": completion.status_code,
                "http_status": completion.status_code,
                "request_id": completion.request_id,
                "response_id": completion.response_id,
                "prompt_id": prompt.prompt_id,
                "prompt_set": prompt.prompt_set,
                "prompt_target_tokens": prompt.target_tokens,
                "prompt_tokens_est": prompt.prompt_tokens_est,
                "prompt_sha256": request_identity["prompt_sha256"],
                "generation_params": request_identity["generation_params"],
                "generation_params_sha256": request_identity["generation_params_sha256"],
                "request_identity_sha256": request_identity["request_identity_sha256"],
                "request_payload_sha256": request_identity["request_payload_sha256"],
                "output_len_chars": len(output_text),
                "output_tokens_est": output_tokens_est,
                "error": completion.error,
                "stream": bool(args.stream),
                "stream_first_event_type": completion.stream_first_event_type,
                "stream_error": completion.stream_error,
                "response_path": response_path,
                "response_header_hints": completion.response_headers,
                "max_tokens_effective": phase_max_tokens,
            }
            prompt_metadata = dict(prompt.metadata or {})
            if prompt_metadata:
                row["prompt_metadata"] = prompt_metadata
                if prompt_metadata.get("prefix_hash"):
                    row["prefix_hash"] = str(prompt_metadata.get("prefix_hash"))
                if prompt_metadata.get("session_id"):
                    row["session_id"] = str(prompt_metadata.get("session_id"))
                if prompt_metadata.get("session_turn_index") is not None:
                    try:
                        row["session_turn_index"] = int(prompt_metadata.get("session_turn_index"))
                    except Exception:
                        row["session_turn_index"] = prompt_metadata.get("session_turn_index")
            async with requests_lock:
                requests_fp.write(json.dumps(row, sort_keys=True) + "\n")
                requests_fp.flush()
            async with rows_lock:
                rows.append(row)
            queue.task_done()

    workers = [asyncio.create_task(worker(i + 1)) for i in range(max(1, phase.concurrency))]
    await asyncio.gather(*workers)
    duration_s = time.perf_counter() - phase_start
    rows.sort(key=lambda r: r["request_index"])
    return rows, duration_s


def summarize_phase(rows: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "duration_s": 0.0,
            "total_requests": 0,
            "ok_requests": 0,
            "error_requests": 0,
            "error_rate": 0.0,
            "req_per_s": 0.0,
            "input_tokens_per_s_est": 0.0,
            "output_tokens_per_s_est": 0.0,
            "latency_ms": {},
            "ttft_ms": {},
            "ttfc_ms": {},
            "ttfb_ms": {},
        }
    ok_rows = [r for r in rows if not r.get("error") and 200 <= int(r.get("status_code", 0)) < 400]
    err_rows = [r for r in rows if r not in ok_rows]
    latencies = [float(r["latency_ms"]) for r in rows]
    ttft = [float(r["ttft_ms"]) for r in rows if r.get("ttft_ms") is not None]
    ttfc = [float(r["ttfc_ms"]) for r in rows if r.get("ttfc_ms") is not None]
    ttfb = [float(r["ttfb_ms"]) for r in rows if r.get("ttfb_ms") is not None]
    input_tokens = sum(int(r.get("prompt_tokens_est", 0)) for r in rows)
    output_tokens = sum(int(r.get("output_tokens_est", 0)) for r in ok_rows)
    safe_duration = max(duration_s, 1e-9)
    return {
        "duration_s": round(duration_s, 3),
        "total_requests": total,
        "ok_requests": len(ok_rows),
        "error_requests": len(err_rows),
        "error_rate": round(len(err_rows) / total, 6),
        "req_per_s": round(total / safe_duration, 4),
        "input_tokens_per_s_est": round(input_tokens / safe_duration, 3),
        "output_tokens_per_s_est": round(output_tokens / safe_duration, 3),
        "latency_ms": summarize_values(latencies),
        "ttft_ms": summarize_values(ttft),
        "ttfc_ms": summarize_values(ttfc),
        "ttfb_ms": summarize_values(ttfb),
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": round(min(values), 3),
        "p50": round(percentile(values, 50), 3),
        "p90": round(percentile(values, 90), 3),
        "p95": round(percentile(values, 95), 3),
        "p99": round(percentile(values, 99), 3),
        "max": round(max(values), 3),
        "mean": round(sum(values) / len(values), 3),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def parse_token_range(text: str, arg_name: str) -> tuple[int, int]:
    raw = text.strip()
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid {arg_name}: expected MIN:MAX, got {text!r}")
    lo = int(parts[0])
    hi = int(parts[1])
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError(f"Invalid {arg_name}: {text!r}")
    return lo, hi


def write_prompt_manifest(path: Path, prompts: list[PromptSpec]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in manifest_rows(prompts):
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def write_request_manifest(path: Path, phases: list[PhasePlan]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for phase in phases:
            for phase_idx, prompt in enumerate(phase.prompts, start=1):
                row = {
                    "phase": phase.name,
                    "phase_request_index": phase_idx,
                    "prompt_id": prompt.prompt_id,
                    "prompt_set": prompt.prompt_set,
                    "prompt_tokens_est": prompt.prompt_tokens_est,
                    "prompt_sha256": hashlib.sha256(prompt.prompt.encode("utf-8")).hexdigest(),
                }
                metadata = dict(prompt.metadata or {})
                if metadata:
                    row["prompt_metadata"] = metadata
                    if metadata.get("prefix_hash"):
                        row["prefix_hash"] = metadata.get("prefix_hash")
                    if metadata.get("session_id"):
                        row["session_id"] = metadata.get("session_id")
                fp.write(json.dumps(row, sort_keys=True) + "\n")


def collect_env_snapshot(keys: tuple[str, ...]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in keys:
        value = os.environ.get(key)
        if value is not None:
            snapshot[key] = value
    return snapshot


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dedupe_prompts(prompts: list[PromptSpec]) -> list[PromptSpec]:
    seen: dict[str, PromptSpec] = {}
    for prompt in prompts:
        key = f"{prompt.prompt_set}:{prompt.prompt_id}"
        if key not in seen:
            seen[key] = prompt
    return list(seen.values())


def build_request_identity(
    *,
    model_id: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
    stream: bool,
    request_seed: Optional[int],
) -> dict[str, Any]:
    generation_params: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stop": list(stop),
        "stream": bool(stream),
    }
    if request_seed is not None:
        generation_params["seed"] = int(request_seed)

    prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    payload = {
        "model": model_id,
        "prompt": prompt_text,
        **generation_params,
    }
    identity_payload = {
        "model": model_id,
        "prompt_sha256": prompt_sha,
        "generation_params": generation_params,
    }
    return {
        "prompt_sha256": prompt_sha,
        "generation_params": generation_params,
        "generation_params_sha256": hashlib.sha256(_json_bytes(generation_params)).hexdigest(),
        "request_payload_sha256": hashlib.sha256(_json_bytes(payload)).hexdigest(),
        "request_identity_sha256": hashlib.sha256(_json_bytes(identity_payload)).hexdigest(),
    }


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_block_device_stats(device: str) -> Optional[dict[str, Any]]:
    stat_path = Path("/sys/block") / device / "stat"
    if not stat_path.exists():
        return None
    try:
        fields = [int(x) for x in stat_path.read_text(encoding="utf-8").split()]
    except Exception:
        return None
    if len(fields) < 7:
        return None
    return {
        "read_ios": fields[0],
        "read_sectors": fields[2],
        "write_ios": fields[4],
        "write_sectors": fields[6],
        "timestamp_utc": now_utc_iso(),
    }


def diff_block_device_stats(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if before is None or after is None:
        return None
    read_sectors = max(0, after["read_sectors"] - before["read_sectors"])
    write_sectors = max(0, after["write_sectors"] - before["write_sectors"])
    return {
        "read_ios_delta": max(0, after["read_ios"] - before["read_ios"]),
        "write_ios_delta": max(0, after["write_ios"] - before["write_ios"]),
        "read_bytes_delta": int(read_sectors * 512),
        "write_bytes_delta": int(write_sectors * 512),
        "read_mib_delta": round((read_sectors * 512) / (1024 * 1024), 3),
        "write_mib_delta": round((write_sectors * 512) / (1024 * 1024), 3),
        "before": before,
        "after": after,
    }


def read_worker_process_io(proc_pattern: str, container_name: str) -> dict[str, Any]:
    container_snapshot = _read_container_process_io(proc_pattern, container_name)
    host_snapshot = _read_host_process_io(proc_pattern)
    cgroup_snapshot = _read_container_cgroup_io(container_name)

    if container_snapshot.get("available"):
        primary = dict(container_snapshot)
    elif host_snapshot.get("available"):
        primary = dict(host_snapshot)
    else:
        primary = dict(container_snapshot)
        if not primary:
            primary = dict(host_snapshot)
    primary.setdefault("timestamp_utc", now_utc_iso())
    primary.setdefault("available", False)
    primary.setdefault("source", "process_io_unavailable")
    primary.setdefault("pattern", proc_pattern)
    primary.setdefault("processes", [])
    primary["cgroup_io"] = cgroup_snapshot
    primary["attribution_sources"] = {
        "selected_source": primary.get("source"),
        "container_proc_available": bool(container_snapshot.get("available")),
        "host_proc_available": bool(host_snapshot.get("available")),
        "cgroup_io_available": bool(cgroup_snapshot.get("available")),
    }
    if primary.get("source") != "docker_exec":
        primary["container_probe"] = _summarize_io_probe(container_snapshot)
    if primary.get("source") != "host_proc":
        primary["host_probe"] = _summarize_io_probe(host_snapshot)
    return primary


def _read_host_process_io(proc_pattern: str) -> dict[str, Any]:
    timestamp = now_utc_iso()
    try:
        pattern = re.compile(proc_pattern)
    except re.error as exc:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "host_proc",
            "error": f"invalid_regex:{exc}",
            "processes": [],
        }
    processes: list[dict[str, Any]] = []
    errors: list[str] = []
    proc_root = Path("/proc")
    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit():
            continue
        cmdline_path = pid_dir / "cmdline"
        io_path = pid_dir / "io"
        try:
            cmdline_bytes = cmdline_path.read_bytes()
        except OSError:
            continue
        cmdline = cmdline_bytes.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        if not cmdline or not pattern.search(cmdline):
            continue
        io_fields: dict[str, int] = {}
        try:
            for raw in io_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" not in raw:
                    continue
                k, v = raw.split(":", 1)
                io_fields[k.strip()] = int(v.strip())
        except PermissionError:
            errors.append(f"permission_denied:{pid_dir.name}")
            continue
        except OSError:
            errors.append(f"io_unreadable:{pid_dir.name}")
            continue
        processes.append(
            {
                "pid": int(pid_dir.name),
                "cmdline": cmdline,
                "read_bytes": int(io_fields.get("read_bytes", 0)),
                "write_bytes": int(io_fields.get("write_bytes", 0)),
                "syscr": int(io_fields.get("syscr", 0)),
                "syscw": int(io_fields.get("syscw", 0)),
            }
        )

    total_read = sum(int(p.get("read_bytes", 0)) for p in processes)
    total_write = sum(int(p.get("write_bytes", 0)) for p in processes)
    return {
        "timestamp_utc": timestamp,
        "available": bool(processes),
        "source": "host_proc",
        "pattern": proc_pattern,
        "processes": processes,
        "totals": {
            "read_bytes": total_read,
            "write_bytes": total_write,
        },
        "errors": errors,
    }


def _read_container_process_io(proc_pattern: str, container_name: str) -> dict[str, Any]:
    timestamp = now_utc_iso()
    if not container_name:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "docker_exec",
            "error": "container_name_missing",
            "processes": [],
        }
    script = """
import json
import os
import re
from pathlib import Path

pattern = os.environ.get("BENCH_WORKER_PROC_PATTERN", "dynamo\\\\.trtllm")
out = {
    "available": False,
    "source": "docker_exec",
    "pattern": pattern,
    "processes": [],
    "errors": [],
}
try:
    rx = re.compile(pattern)
except re.error as exc:
    out["error"] = f"invalid_regex:{exc}"
    print(json.dumps(out))
    raise SystemExit(0)

for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    cmdline_path = entry / "cmdline"
    io_path = entry / "io"
    try:
        cmdline = cmdline_path.read_bytes().replace(b"\\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        continue
    if not cmdline or not rx.search(cmdline):
        continue
    fields = {}
    try:
        for raw in io_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in raw:
                continue
            key, value = raw.split(":", 1)
            fields[key.strip()] = int(value.strip())
    except OSError as exc:
        out["errors"].append(f"io_unreadable:{entry.name}:{exc}")
        continue
    out["processes"].append(
        {
            "pid": int(entry.name),
            "cmdline": cmdline,
            "read_bytes": int(fields.get("read_bytes", 0)),
            "write_bytes": int(fields.get("write_bytes", 0)),
            "syscr": int(fields.get("syscr", 0)),
            "syscw": int(fields.get("syscw", 0)),
        }
    )

out["available"] = bool(out["processes"])
out["totals"] = {
    "read_bytes": sum(int(p.get("read_bytes", 0)) for p in out["processes"]),
    "write_bytes": sum(int(p.get("write_bytes", 0)) for p in out["processes"]),
}
print(json.dumps(out))
"""
    try:
        completed = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                f"BENCH_WORKER_PROC_PATTERN={proc_pattern}",
                container_name,
                "python3",
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "docker_exec",
            "pattern": proc_pattern,
            "error": str(exc),
            "processes": [],
        }
    if completed.returncode != 0:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "docker_exec",
            "pattern": proc_pattern,
            "error": (completed.stderr or completed.stdout).strip() or f"rc={completed.returncode}",
            "processes": [],
        }
    raw = (completed.stdout or "").strip()
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {
            "available": False,
            "source": "docker_exec",
            "pattern": proc_pattern,
            "error": "invalid_json_response",
            "stdout": raw,
            "processes": [],
        }
    payload.setdefault("available", False)
    payload.setdefault("source", "docker_exec")
    payload.setdefault("pattern", proc_pattern)
    payload.setdefault("processes", [])
    payload["timestamp_utc"] = timestamp
    return payload


def _summarize_io_probe(snapshot: dict[str, Any]) -> dict[str, Any]:
    totals = snapshot.get("totals") if isinstance(snapshot, dict) else {}
    if not isinstance(totals, dict):
        totals = {}
    return {
        "available": bool((snapshot or {}).get("available")),
        "source": (snapshot or {}).get("source"),
        "error": (snapshot or {}).get("error"),
        "process_count": len((snapshot or {}).get("processes") or []),
        "totals": {
            "read_bytes": int(totals.get("read_bytes", 0) or 0),
            "write_bytes": int(totals.get("write_bytes", 0) or 0),
        },
    }


def _read_container_cgroup_io(container_name: str) -> dict[str, Any]:
    timestamp = now_utc_iso()
    if not container_name:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "cgroup_io",
            "error": "container_name_missing",
            "totals": {"read_bytes": 0, "write_bytes": 0},
            "devices": [],
        }
    try:
        inspect = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception as exc:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "cgroup_io",
            "error": str(exc),
            "totals": {"read_bytes": 0, "write_bytes": 0},
            "devices": [],
        }
    if inspect.returncode != 0:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "cgroup_io",
            "error": (inspect.stderr or inspect.stdout).strip() or f"rc={inspect.returncode}",
            "totals": {"read_bytes": 0, "write_bytes": 0},
            "devices": [],
        }
    pid_raw = (inspect.stdout or "").strip()
    try:
        pid = int(pid_raw)
    except Exception:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "cgroup_io",
            "error": f"invalid_container_pid:{pid_raw}",
            "totals": {"read_bytes": 0, "write_bytes": 0},
            "devices": [],
        }
    if pid <= 0:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "cgroup_io",
            "error": f"container_not_running:pid={pid}",
            "totals": {"read_bytes": 0, "write_bytes": 0},
            "devices": [],
        }

    cgroup_rel = None
    cgroup_file = Path(f"/proc/{pid}/cgroup")
    try:
        cgroup_text = cgroup_file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "cgroup_io",
            "error": str(exc),
            "container_pid": pid,
            "totals": {"read_bytes": 0, "write_bytes": 0},
            "devices": [],
        }
    for raw in cgroup_text.splitlines():
        parts = raw.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[1] == "":
            cgroup_rel = parts[2]
            break
    if cgroup_rel is None:
        for raw in cgroup_text.splitlines():
            parts = raw.split(":", 2)
            if len(parts) == 3:
                cgroup_rel = parts[2]
                break
    if not cgroup_rel:
        return {
            "timestamp_utc": timestamp,
            "available": False,
            "source": "cgroup_io",
            "error": "cgroup_path_not_found",
            "container_pid": pid,
            "totals": {"read_bytes": 0, "write_bytes": 0},
            "devices": [],
        }

    cgroup_path = Path("/sys/fs/cgroup") / str(cgroup_rel).lstrip("/")
    io_stat_path = cgroup_path / "io.stat"
    blkio_throttle_path = cgroup_path / "blkio.throttle.io_service_bytes"
    blkio_path = cgroup_path / "blkio.io_service_bytes"

    if io_stat_path.exists():
        devices: list[dict[str, Any]] = []
        read_total = 0
        write_total = 0
        try:
            for raw in io_stat_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                dev = parts[0]
                stats: dict[str, int] = {}
                for token in parts[1:]:
                    if "=" not in token:
                        continue
                    key, value = token.split("=", 1)
                    try:
                        stats[key] = int(value)
                    except ValueError:
                        continue
                read_bytes = int(stats.get("rbytes", 0))
                write_bytes = int(stats.get("wbytes", 0))
                read_total += read_bytes
                write_total += write_bytes
                devices.append({"device": dev, "read_bytes": read_bytes, "write_bytes": write_bytes, "stats": stats})
        except OSError as exc:
            return {
                "timestamp_utc": timestamp,
                "available": False,
                "source": "cgroup_v2_io_stat",
                "error": str(exc),
                "container_pid": pid,
                "cgroup_path": str(cgroup_path),
                "totals": {"read_bytes": 0, "write_bytes": 0},
                "devices": [],
            }
        return {
            "timestamp_utc": timestamp,
            "available": True,
            "source": "cgroup_v2_io_stat",
            "container_pid": pid,
            "cgroup_path": str(cgroup_path),
            "totals": {"read_bytes": int(read_total), "write_bytes": int(write_total)},
            "devices": devices,
        }

    for path in (blkio_throttle_path, blkio_path):
        if not path.exists():
            continue
        devices = {}
        read_total = 0
        write_total = 0
        try:
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                if parts[0].lower() == "total":
                    continue
                dev = parts[0]
                op = parts[1].lower()
                try:
                    value = int(parts[2])
                except ValueError:
                    continue
                item = devices.setdefault(dev, {"device": dev, "read_bytes": 0, "write_bytes": 0})
                if op.startswith("read"):
                    item["read_bytes"] += value
                    read_total += value
                elif op.startswith("write"):
                    item["write_bytes"] += value
                    write_total += value
        except OSError as exc:
            return {
                "timestamp_utc": timestamp,
                "available": False,
                "source": "cgroup_blkio",
                "error": str(exc),
                "container_pid": pid,
                "cgroup_path": str(cgroup_path),
                "totals": {"read_bytes": 0, "write_bytes": 0},
                "devices": [],
            }
        return {
            "timestamp_utc": timestamp,
            "available": True,
            "source": f"cgroup_{path.name}",
            "container_pid": pid,
            "cgroup_path": str(cgroup_path),
            "totals": {"read_bytes": int(read_total), "write_bytes": int(write_total)},
            "devices": list(devices.values()),
        }

    return {
        "timestamp_utc": timestamp,
        "available": False,
        "source": "cgroup_io",
        "error": "io_stat_not_found",
        "container_pid": pid,
        "cgroup_path": str(cgroup_path),
        "totals": {"read_bytes": 0, "write_bytes": 0},
        "devices": [],
    }


def diff_process_io_stats(before: dict[str, Any], after: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    before_procs = before.get("processes") if isinstance(before.get("processes"), list) else []
    after_procs = after.get("processes") if isinstance(after.get("processes"), list) else []
    before_map = {int(p.get("pid")): p for p in before_procs if isinstance(p, dict) and p.get("pid") is not None}
    after_map = {int(p.get("pid")): p for p in after_procs if isinstance(p, dict) and p.get("pid") is not None}

    per_pid: list[dict[str, Any]] = []
    total_read = 0
    total_write = 0
    for pid, after_item in after_map.items():
        before_item = before_map.get(pid, {})
        read_delta = max(0, int(after_item.get("read_bytes", 0)) - int(before_item.get("read_bytes", 0)))
        write_delta = max(0, int(after_item.get("write_bytes", 0)) - int(before_item.get("write_bytes", 0)))
        total_read += read_delta
        total_write += write_delta
        per_pid.append(
            {
                "pid": pid,
                "cmdline": after_item.get("cmdline"),
                "read_bytes_delta": read_delta,
                "write_bytes_delta": write_delta,
            }
        )
    per_pid_sorted = sorted(
        per_pid,
        key=lambda item: int(item.get("read_bytes_delta", 0)) + int(item.get("write_bytes_delta", 0)),
        reverse=True,
    )
    top_writers = sorted(per_pid, key=lambda x: int(x.get("write_bytes_delta", 0)), reverse=True)[:5]
    top_readers = sorted(per_pid, key=lambda x: int(x.get("read_bytes_delta", 0)), reverse=True)[:5]

    cgroup_before = _extract_io_totals((before.get("cgroup_io") or {}))
    cgroup_after = _extract_io_totals((after.get("cgroup_io") or {}))
    cgroup_read_delta: Optional[int] = None
    cgroup_write_delta: Optional[int] = None
    if cgroup_before is not None and cgroup_after is not None:
        cgroup_read_delta = max(0, cgroup_after["read_bytes"] - cgroup_before["read_bytes"])
        cgroup_write_delta = max(0, cgroup_after["write_bytes"] - cgroup_before["write_bytes"])

    return {
        "available": bool(per_pid) or (cgroup_read_delta is not None and cgroup_write_delta is not None),
        "source_before": before.get("source"),
        "source_after": after.get("source"),
        "attribution_sources_before": before.get("attribution_sources"),
        "attribution_sources_after": after.get("attribution_sources"),
        "matched_pids_before": len(before_map),
        "matched_pids_after": len(after_map),
        "read_bytes_delta": total_read,
        "write_bytes_delta": total_write,
        "read_mib_delta": round(total_read / (1024 * 1024), 3),
        "write_mib_delta": round(total_write / (1024 * 1024), 3),
        "per_pid_deltas": per_pid_sorted,
        "cgroup_source_before": ((before.get("cgroup_io") or {}).get("source") if isinstance(before.get("cgroup_io"), dict) else None),
        "cgroup_source_after": ((after.get("cgroup_io") or {}).get("source") if isinstance(after.get("cgroup_io"), dict) else None),
        "cgroup_read_bytes_delta": cgroup_read_delta,
        "cgroup_write_bytes_delta": cgroup_write_delta,
        "cgroup_read_mib_delta": (
            round(cgroup_read_delta / (1024 * 1024), 3) if cgroup_read_delta is not None else None
        ),
        "cgroup_write_mib_delta": (
            round(cgroup_write_delta / (1024 * 1024), 3) if cgroup_write_delta is not None else None
        ),
        "top_writers": top_writers,
        "top_readers": top_readers,
    }


def _extract_io_totals(snapshot: dict[str, Any]) -> Optional[dict[str, int]]:
    if not isinstance(snapshot, dict):
        return None
    totals = snapshot.get("totals")
    if not isinstance(totals, dict):
        return None
    try:
        return {
            "read_bytes": int(totals.get("read_bytes", 0)),
            "write_bytes": int(totals.get("write_bytes", 0)),
        }
    except Exception:
        return None


def _command_to_text(cmd: list[str]) -> str:
    if not cmd:
        return ""
    return " ".join(cmd)


def _register_capture_error(
    capture_errors: list[dict[str, Any]],
    *,
    command: list[str],
    return_code: Optional[int],
    stderr: Optional[str],
    error: Optional[str] = None,
) -> None:
    snippet_source = str((stderr or "").strip() or (error or "").strip() or "unknown_error")
    capture_errors.append(
        {
            "command": _command_to_text(command),
            "return_code": int(return_code) if isinstance(return_code, int) else None,
            "stderr_snippet": snippet_source[:240],
        }
    )


def _capture_json_command(
    cmd: list[str],
    capture_errors: list[dict[str, Any]],
    *,
    timeout_s: float = 10.0,
    record_errors: bool = True,
) -> dict[str, Any]:
    raw = _run_capture_command(cmd, timeout_s=timeout_s)
    out: dict[str, Any] = {
        "command": cmd,
        "success": bool(raw.get("success")),
        "return_code": raw.get("return_code"),
        "error": raw.get("error"),
        "stderr": (str(raw.get("stderr") or "").strip() or None),
        "payload": None,
    }
    if not raw.get("success"):
        if record_errors:
            _register_capture_error(
                capture_errors,
                command=cmd,
                return_code=raw.get("return_code"),
                stderr=str(raw.get("stderr") or raw.get("stdout") or raw.get("error") or ""),
                error=str(raw.get("error") or ""),
            )
        return out
    stdout = str(raw.get("stdout") or "")
    try:
        out["payload"] = json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError as exc:
        out["success"] = False
        out["error"] = f"json_parse_error:{exc}"
        out["raw_stdout"] = stdout
        if record_errors:
            _register_capture_error(
                capture_errors,
                command=cmd,
                return_code=raw.get("return_code"),
                stderr=f"json_parse_error:{exc}",
                error=str(exc),
            )
    return out


def _capture_text_command(
    cmd: list[str],
    capture_errors: list[dict[str, Any]],
    *,
    timeout_s: float = 10.0,
    strip: bool = True,
    record_errors: bool = True,
) -> dict[str, Any]:
    raw = _run_capture_command(cmd, timeout_s=timeout_s)
    out: dict[str, Any] = {
        "command": cmd,
        "success": bool(raw.get("success")),
        "return_code": raw.get("return_code"),
        "error": raw.get("error"),
        "stderr": (str(raw.get("stderr") or "").strip() or None),
        "payload": None,
    }
    if not raw.get("success"):
        if record_errors:
            _register_capture_error(
                capture_errors,
                command=cmd,
                return_code=raw.get("return_code"),
                stderr=str(raw.get("stderr") or raw.get("stdout") or raw.get("error") or ""),
                error=str(raw.get("error") or ""),
            )
        return out
    stdout = str(raw.get("stdout") or "")
    out["payload"] = stdout.strip() if strip else stdout
    return out


def _parse_os_release_text(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(raw or "").splitlines():
        item = line.strip()
        if not item or item.startswith("#") or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        out[key] = value
    return out


def _first_line(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    for line in str(value).splitlines():
        item = line.strip()
        if item:
            return item
    return None


def _controller_from_device_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    match = NVME_DEVICE_PATH_RE.match(str(path).strip())
    if not match:
        return None
    return match.group(1)


def _namespace_from_device_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    match = re.match(r"^/dev/(nvme\d+n\d+)(?:p\d+)?$", str(path).strip())
    if not match:
        return None
    return f"/dev/{match.group(1)}"


def _read_text_file_best_effort(path: Path) -> Optional[str]:
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return text or None
    except Exception:
        return None


def _extract_bdf_from_path(path: Path) -> Optional[str]:
    candidates = [path, *path.parents]
    for item in candidates:
        name = item.name
        if BDF_RE.match(name):
            return name.lower()
    return None


def _make_nvme_controller_record(controller: str) -> dict[str, Any]:
    return {
        "controller": controller,
        "device": f"/dev/{controller}",
        "namespace": None,
        "namespace_candidates": [],
        "bdf": None,
        "sysfs_identity": {
            "model": None,
            "serial": None,
            "firmware_rev": None,
        },
        "nvme_list_entries": [],
    }


def _discover_nvme_controllers() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    sysfs_root = Path("/sys/class/nvme")
    if not sysfs_root.exists():
        return out
    for controller_dir in sorted(sysfs_root.glob("nvme[0-9]*")):
        controller = controller_dir.name
        if not NVME_CONTROLLER_RE.match(controller):
            continue
        record = _make_nvme_controller_record(controller)
        namespace_candidates = [
            str(path)
            for path in sorted(Path("/dev").glob(f"{controller}n*"))
            if re.match(rf"^/dev/{re.escape(controller)}n\d+$", str(path))
        ]
        if namespace_candidates:
            record["namespace"] = namespace_candidates[0]
            record["namespace_candidates"] = namespace_candidates
        record["sysfs_identity"] = {
            "model": _read_text_file_best_effort(controller_dir / "model"),
            "serial": _read_text_file_best_effort(controller_dir / "serial"),
            "firmware_rev": _read_text_file_best_effort(controller_dir / "firmware_rev"),
        }
        try:
            record["bdf"] = _extract_bdf_from_path(controller_dir.resolve())
        except Exception:
            record["bdf"] = None
        out[controller] = record
    return out


def _merge_nvme_list_into_controllers(controllers: dict[str, dict[str, Any]], payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    devices = payload.get("Devices")
    if not isinstance(devices, list):
        return
    for item in devices:
        if not isinstance(item, dict):
            continue
        device_path = str(item.get("DevicePath") or item.get("Device") or "")
        controller = _controller_from_device_path(device_path)
        if controller is None:
            name = str(item.get("Name") or "").strip()
            if NVME_CONTROLLER_RE.match(name):
                controller = name
        if controller is None:
            continue
        record = controllers.setdefault(controller, _make_nvme_controller_record(controller))
        record.setdefault("nvme_list_entries", []).append(item)
        if device_path:
            namespace = _namespace_from_device_path(device_path)
            if namespace and not record.get("namespace"):
                record["namespace"] = namespace


def _extract_bdfs_by_controller_from_subsys(payload: Any) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    stack: list[tuple[Any, Optional[str]]] = [(payload, None)]
    bdf_search = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")
    while stack:
        node, current_controller = stack.pop()
        if isinstance(node, dict):
            node_controller = current_controller
            for key in ("Name", "Device", "Controller", "Path"):
                value = node.get(key)
                if not isinstance(value, str):
                    continue
                controller = _controller_from_device_path(value)
                if controller is None and NVME_CONTROLLER_RE.match(value):
                    controller = value
                if controller:
                    node_controller = controller
            for value in node.values():
                if isinstance(value, str):
                    for bdf in bdf_search.findall(value):
                        if node_controller:
                            mapping.setdefault(node_controller, set()).add(bdf.lower())
                stack.append((value, node_controller))
            continue
        if isinstance(node, list):
            for child in node:
                stack.append((child, current_controller))
    return {controller: sorted(values) for controller, values in mapping.items() if values}


def _merge_nvme_subsys_into_controllers(controllers: dict[str, dict[str, Any]], payload: Any) -> None:
    by_controller = _extract_bdfs_by_controller_from_subsys(payload)
    for controller, bdfs in by_controller.items():
        record = controllers.setdefault(controller, _make_nvme_controller_record(controller))
        if not record.get("bdf") and bdfs:
            record["bdf"] = bdfs[0]


def _extract_nvme_bdfs_from_lspci_scan(raw_text: str) -> list[str]:
    out: list[str] = []
    for line in str(raw_text or "").splitlines():
        if "nvme" not in line.lower():
            continue
        match = re.match(r"^([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])\s+", line)
        if match:
            out.append(match.group(1).lower())
    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _merge_lspci_scan_into_controllers(controllers: dict[str, dict[str, Any]], lspci_scan_text: str) -> None:
    bdfs = _extract_nvme_bdfs_from_lspci_scan(lspci_scan_text)
    if not bdfs:
        return
    unassigned = [name for name, rec in sorted(controllers.items()) if not rec.get("bdf")]
    for controller, bdf in zip(unassigned, bdfs):
        controllers[controller]["bdf"] = bdf


def _resolve_primary_nvme_controller(
    *,
    controllers: dict[str, dict[str, Any]],
    kvbm_source: Optional[str],
    nvme_device_hint: str,
    capture_errors: list[dict[str, Any]],
) -> Optional[str]:
    candidates: list[str] = []
    controller_from_kvbm = _controller_from_device_path(kvbm_source)
    if controller_from_kvbm is None and kvbm_source and kvbm_source.startswith("/dev/"):
        parent = _capture_text_command(
            ["lsblk", "-no", "PKNAME", kvbm_source],
            capture_errors,
            timeout_s=5.0,
        )
        parent_name = _first_line(str(parent.get("payload") or ""))
        if parent_name:
            controller_from_kvbm = _controller_from_device_path(f"/dev/{parent_name}")
    if controller_from_kvbm:
        candidates.append(controller_from_kvbm)
    controller_from_hint = _controller_from_device_path(nvme_device_hint)
    if controller_from_hint:
        candidates.append(controller_from_hint)
    if "nvme0" in controllers:
        candidates.append("nvme0")
    candidates.extend(sorted(controllers.keys()))
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in controllers:
            return candidate
    return None


def _register_capture_errors_from_nvme_probe(capture_errors: list[dict[str, Any]], probe: dict[str, Any]) -> None:
    if not isinstance(probe, dict):
        return
    probe_success = bool(probe.get("success"))
    payload_ok = bool(probe.get("payload_ok", True))
    if probe_success and payload_ok:
        return
    attempts = probe.get("attempts")
    if isinstance(attempts, list) and attempts:
        last = attempts[-1] if isinstance(attempts[-1], dict) else {}
        command = last.get("command")
        if not isinstance(command, list):
            command = probe.get("command") if isinstance(probe.get("command"), list) else []
        _register_capture_error(
            capture_errors,
            command=command,
            return_code=(last.get("return_code") if isinstance(last, dict) else probe.get("return_code")),
            stderr=str((last.get("error") if isinstance(last, dict) else probe.get("error")) or probe.get("stderr") or ""),
            error=str(probe.get("error") or ""),
        )
        return
    _register_capture_error(
        capture_errors,
        command=(probe.get("command") if isinstance(probe.get("command"), list) else []),
        return_code=probe.get("return_code"),
        stderr=str(probe.get("stderr") or probe.get("error") or ""),
        error=str(probe.get("error") or ""),
    )


def _dict_get_first(payload: Any, keys: tuple[str, ...]) -> Optional[Any]:
    if not isinstance(payload, dict):
        return None
    lower = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
        lowered = lower.get(key.lower())
        if lowered not in (None, ""):
            return lowered
    return None


def _flatten_lsblk_nodes(nodes: list[dict[str, Any]], parent_name: Optional[str] = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        item = dict(node)
        item["_parent_name"] = parent_name
        out.append(item)
        children = node.get("children")
        if isinstance(children, list):
            out.extend(_flatten_lsblk_nodes(children, str(node.get("name") or "")))
    return out


def _find_lsblk_size_for_device(lsblk_payload: Any, namespace_path: Optional[str], device_path: str) -> Optional[Any]:
    if not isinstance(lsblk_payload, dict):
        return None
    nodes = lsblk_payload.get("blockdevices")
    if not isinstance(nodes, list):
        return None
    rows = _flatten_lsblk_nodes([n for n in nodes if isinstance(n, dict)])
    targets: set[str] = set()
    if namespace_path:
        targets.add(namespace_path)
        targets.add(Path(namespace_path).name)
    targets.add(device_path)
    targets.add(Path(device_path).name)
    for row in rows:
        path = str(row.get("path") or "")
        name = str(row.get("name") or "")
        kname = str(row.get("kname") or "")
        if path in targets or name in targets or kname in targets:
            size = row.get("size")
            if size not in (None, ""):
                return size
    return None


def _parse_pcie_link_state(raw_text: str) -> dict[str, Optional[str]]:
    text = str(raw_text or "")
    max_match = re.search(r"LnkCap:\s.*?Speed\s+([^,]+),\s*Width\s+([^\s,]+)", text)
    current_match = re.search(r"LnkSta:\s.*?Speed\s+([^,]+),\s*Width\s+([^\s,]+)", text)
    return {
        "negotiated_speed": (current_match.group(1).strip() if current_match else None),
        "negotiated_width": (current_match.group(2).strip() if current_match else None),
        "max_speed": (max_match.group(1).strip() if max_match else None),
        "max_width": (max_match.group(2).strip() if max_match else None),
    }


def _pcie_link_summary(link: dict[str, Any]) -> Optional[str]:
    negotiated_speed = str(link.get("negotiated_speed") or "").strip()
    negotiated_width = str(link.get("negotiated_width") or "").strip()
    max_speed = str(link.get("max_speed") or "").strip()
    max_width = str(link.get("max_width") or "").strip()
    base = " ".join(part for part in (negotiated_speed, negotiated_width) if part)
    max_part = " ".join(part for part in (max_speed, max_width) if part)
    if base and max_part:
        return f"{base} (max {max_part})"
    if base:
        return base
    if max_part:
        return f"max {max_part}"
    return None


def _collect_controller_metadata(
    *,
    controller: str,
    record: dict[str, Any],
    lsblk_payload: Any,
    capture_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    device_path = str(record.get("device") or f"/dev/{controller}")
    namespace_path = str(record.get("namespace") or f"{device_path}n1")
    bdf = str(record.get("bdf") or "").lower() or None
    sysfs_identity = record.get("sysfs_identity") if isinstance(record.get("sysfs_identity"), dict) else {}
    nvme_list_entries = record.get("nvme_list_entries") if isinstance(record.get("nvme_list_entries"), list) else []
    first_list_entry = nvme_list_entries[0] if nvme_list_entries and isinstance(nvme_list_entries[0], dict) else {}

    id_ctrl = _collect_nvme_with_fallback("id-ctrl", device_path)
    _register_capture_errors_from_nvme_probe(capture_errors, id_ctrl)
    id_ns = _collect_nvme_with_fallback("id-ns", namespace_path)
    _register_capture_errors_from_nvme_probe(capture_errors, id_ns)
    smart_log = _collect_nvme_with_fallback("smart-log", device_path)
    _register_capture_errors_from_nvme_probe(capture_errors, smart_log)
    smartctl_payload: Optional[dict[str, Any]] = None
    if not bool(smart_log.get("success")) or not bool(smart_log.get("payload_ok", True)):
        smartctl_payload = _capture_json_command(["smartctl", "-a", "-j", device_path], capture_errors, timeout_s=10.0)

    lspci_vv: Optional[dict[str, Any]] = None
    pcie_link: dict[str, Optional[str]] = {
        "negotiated_speed": None,
        "negotiated_width": None,
        "max_speed": None,
        "max_width": None,
    }
    if bdf:
        lspci_vv = _capture_text_command(["lspci", "-vv", "-s", bdf], capture_errors, timeout_s=10.0, strip=False)
        if lspci_vv.get("success"):
            pcie_link = _parse_pcie_link_state(str(lspci_vv.get("payload") or ""))

    id_ctrl_payload = id_ctrl.get("payload") if isinstance(id_ctrl.get("payload"), dict) else {}
    model = _dict_get_first(id_ctrl_payload, ("mn", "model", "model_number"))
    serial = _dict_get_first(id_ctrl_payload, ("sn", "serial", "serial_number"))
    firmware_rev = _dict_get_first(id_ctrl_payload, ("fr", "firmware_rev", "firmware"))
    if model is None:
        model = _dict_get_first(first_list_entry, ("ModelNumber", "Model", "Model Number", "model"))
    if serial is None:
        serial = _dict_get_first(first_list_entry, ("SerialNumber", "Serial", "serial", "sn"))
    if firmware_rev is None:
        firmware_rev = _dict_get_first(first_list_entry, ("Firmware", "FirmwareRevision", "fw", "fr"))
    if model is None:
        model = _dict_get_first(sysfs_identity, ("model",))
    if serial is None:
        serial = _dict_get_first(sysfs_identity, ("serial",))
    if firmware_rev is None:
        firmware_rev = _dict_get_first(sysfs_identity, ("firmware_rev",))

    size = _dict_get_first(
        first_list_entry,
        ("PhysicalSize", "NamespaceSize", "Size", "TotalNVMCapacity", "UsedBytes"),
    )
    if size is None:
        size = _find_lsblk_size_for_device(lsblk_payload, namespace_path, device_path)

    return {
        "controller": controller,
        "nvme_device": device_path,
        "nvme_namespace": namespace_path,
        "bdf": bdf,
        "sysfs_identity": sysfs_identity,
        "nvme_list_entries": nvme_list_entries,
        "id_ctrl": id_ctrl,
        "id_ns": id_ns,
        "smart_log": smart_log,
        "smartctl": smartctl_payload,
        "lspci_vv": lspci_vv,
        "pcie_link": pcie_link,
        "identity": {
            "model": model,
            "serial": serial,
            "firmware_rev": firmware_rev,
            "size": size,
        },
    }


def collect_device_metadata(
    *,
    capture_stage: str,
    nvme_device_hint: str,
    kvbm_cache_dir: str,
    phase_io_device: str,
) -> dict[str, Any]:
    capture_errors: list[dict[str, Any]] = []
    controllers = _discover_nvme_controllers()

    nvme_list = _capture_json_command(["nvme", "list", "-o", "json"], capture_errors, timeout_s=12.0)
    _merge_nvme_list_into_controllers(controllers, nvme_list.get("payload"))
    nvme_list_subsys = _capture_json_command(["nvme", "list-subsys", "-o", "json"], capture_errors, timeout_s=12.0)
    _merge_nvme_subsys_into_controllers(controllers, nvme_list_subsys.get("payload"))

    lspci_scan = _capture_text_command(["lspci", "-D", "-nn"], capture_errors, timeout_s=12.0)
    _merge_lspci_scan_into_controllers(controllers, str(lspci_scan.get("payload") or ""))

    if not controllers:
        hint_controller = _controller_from_device_path(nvme_device_hint)
        if hint_controller:
            controllers[hint_controller] = _make_nvme_controller_record(hint_controller)

    kvbm_source_lookup = _capture_text_command(
        ["findmnt", "-n", "-o", "SOURCE", "--target", kvbm_cache_dir],
        capture_errors,
        timeout_s=8.0,
    )
    kvbm_source = _first_line(str(kvbm_source_lookup.get("payload") or ""))

    lsblk = _capture_json_command(
        [
            "lsblk",
            "-J",
            "-o",
            "NAME,KNAME,PATH,MODEL,SERIAL,SIZE,ROTA,TYPE,MOUNTPOINT,FSTYPE,UUID,PARTUUID",
        ],
        capture_errors,
        timeout_s=12.0,
    )
    findmnt = _capture_json_command(
        ["findmnt", "-J", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        capture_errors,
        timeout_s=10.0,
    )
    mount_snapshot = _capture_text_command(["mount"], capture_errors, timeout_s=8.0, strip=False)

    uname_snapshot = _capture_text_command(["uname", "-a"], capture_errors, timeout_s=5.0)
    os_release_raw = _capture_text_command(["cat", "/etc/os-release"], capture_errors, timeout_s=5.0, strip=False)
    nvidia_smi = _capture_text_command(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_errors,
        timeout_s=8.0,
        record_errors=False,
    )

    primary_controller = _resolve_primary_nvme_controller(
        controllers=controllers,
        kvbm_source=kvbm_source,
        nvme_device_hint=nvme_device_hint,
        capture_errors=capture_errors,
    )
    if primary_controller is None and controllers:
        primary_controller = sorted(controllers.keys())[0]

    if primary_controller and primary_controller not in controllers:
        controllers[primary_controller] = _make_nvme_controller_record(primary_controller)

    device_entries: list[dict[str, Any]] = []
    pcie_entries: list[dict[str, Any]] = []
    for controller in sorted(controllers.keys()):
        entry = _collect_controller_metadata(
            controller=controller,
            record=controllers[controller],
            lsblk_payload=lsblk.get("payload"),
            capture_errors=capture_errors,
        )
        device_entries.append(entry)
        pcie_entries.append(
            {
                "controller": controller,
                "bdf": entry.get("bdf"),
                "pcie_link": entry.get("pcie_link"),
                "lspci_vv": entry.get("lspci_vv"),
            }
        )

    primary_entry: dict[str, Any] = {}
    if primary_controller:
        for entry in device_entries:
            if entry.get("controller") == primary_controller:
                primary_entry = entry
                break
    if not primary_entry and device_entries:
        primary_entry = device_entries[0]
        primary_controller = str(primary_entry.get("controller") or "")

    primary_identity = primary_entry.get("identity") if isinstance(primary_entry.get("identity"), dict) else {}
    primary_pcie_link = primary_entry.get("pcie_link") if isinstance(primary_entry.get("pcie_link"), dict) else {}

    nvme_devices = [
        {
            "controller": entry.get("controller"),
            "nvme_device": entry.get("nvme_device"),
            "nvme_namespace": entry.get("nvme_namespace"),
            "bdf": entry.get("bdf"),
        }
        for entry in device_entries
    ]

    return {
        "schema_version": 1,
        "capture_stage": capture_stage,
        "capture_timestamp": now_utc_iso(),
        "capture_errors": capture_errors,
        "requested_targets": {
            "nvme_device_hint": nvme_device_hint,
            "kvbm_cache_dir": kvbm_cache_dir,
            "phase_io_device": phase_io_device,
        },
        "resolved_targets": {
            "kvbm_source": kvbm_source,
            "primary_nvme_controller": primary_controller,
            "primary_nvme_device": primary_entry.get("nvme_device"),
            "primary_nvme_namespace": primary_entry.get("nvme_namespace"),
            "primary_bdf": primary_entry.get("bdf"),
            "nvme_devices": nvme_devices,
        },
        "primary_nvme_controller": primary_controller,
        "primary_nvme_device": primary_entry.get("nvme_device"),
        "primary_nvme_namespace": primary_entry.get("nvme_namespace"),
        "nvme_devices": nvme_devices,
        "nvme": {
            "list": nvme_list,
            "list_subsys": nvme_list_subsys,
            "devices": device_entries,
        },
        "pcie": {
            "lspci_scan": lspci_scan,
            "controllers": pcie_entries,
        },
        "block_filesystem": {
            "lsblk": lsblk,
            "findmnt": findmnt,
            "mount": mount_snapshot,
        },
        "system": {
            "uname": uname_snapshot,
            "os_release": {
                "raw": os_release_raw,
                "parsed": _parse_os_release_text(str(os_release_raw.get("payload") or "")),
            },
            "nvidia_smi": nvidia_smi,
        },
        "primary_storage_summary": {
            "model": primary_identity.get("model"),
            "serial": primary_identity.get("serial"),
            "firmware_rev": primary_identity.get("firmware_rev"),
            "size": primary_identity.get("size"),
            "pcie_link": _pcie_link_summary(primary_pcie_link),
        },
    }


def _capture_unavailable_payload(error: str, command: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "command": command or [],
        "success": False,
        "return_code": None,
        "error": error,
        "stderr": error,
        "payload": None,
    }


def collect_device_metadata_safe(
    *,
    capture_stage: str,
    nvme_device_hint: str,
    kvbm_cache_dir: str,
    phase_io_device: str,
) -> dict[str, Any]:
    try:
        return collect_device_metadata(
            capture_stage=capture_stage,
            nvme_device_hint=nvme_device_hint,
            kvbm_cache_dir=kvbm_cache_dir,
            phase_io_device=phase_io_device,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Device metadata capture failed (stage=%s).", capture_stage)
        hint_controller = _controller_from_device_path(nvme_device_hint)
        hint_namespace = _namespace_from_device_path(nvme_device_hint)
        return {
            "schema_version": 1,
            "capture_stage": capture_stage,
            "capture_timestamp": now_utc_iso(),
            "capture_errors": [
                {
                    "command": "collector_internal",
                    "return_code": None,
                    "stderr_snippet": str(exc)[:240],
                }
            ],
            "requested_targets": {
                "nvme_device_hint": nvme_device_hint,
                "kvbm_cache_dir": kvbm_cache_dir,
                "phase_io_device": phase_io_device,
            },
            "resolved_targets": {
                "kvbm_source": None,
                "primary_nvme_controller": hint_controller,
                "primary_nvme_device": (nvme_device_hint if str(nvme_device_hint).startswith("/dev/") else None),
                "primary_nvme_namespace": hint_namespace,
                "primary_bdf": None,
                "nvme_devices": [],
            },
            "primary_nvme_controller": hint_controller,
            "primary_nvme_device": (nvme_device_hint if str(nvme_device_hint).startswith("/dev/") else None),
            "primary_nvme_namespace": hint_namespace,
            "nvme_devices": [],
            "nvme": {
                "list": _capture_unavailable_payload("collector_internal_failure"),
                "list_subsys": _capture_unavailable_payload("collector_internal_failure"),
                "devices": [],
            },
            "pcie": {
                "lspci_scan": _capture_unavailable_payload("collector_internal_failure"),
                "controllers": [],
            },
            "block_filesystem": {
                "lsblk": _capture_unavailable_payload("collector_internal_failure"),
                "findmnt": _capture_unavailable_payload("collector_internal_failure"),
                "mount": _capture_unavailable_payload("collector_internal_failure"),
            },
            "system": {
                "uname": _capture_unavailable_payload("collector_internal_failure"),
                "os_release": {
                    "raw": _capture_unavailable_payload("collector_internal_failure"),
                    "parsed": {},
                },
                "nvidia_smi": _capture_unavailable_payload("collector_internal_failure"),
            },
            "primary_storage_summary": {
                "model": None,
                "serial": None,
                "firmware_rev": None,
                "size": None,
                "pcie_link": None,
            },
        }


def extract_storage_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    summary = metadata.get("primary_storage_summary") if isinstance(metadata.get("primary_storage_summary"), dict) else {}
    resolved = metadata.get("resolved_targets") if isinstance(metadata.get("resolved_targets"), dict) else {}
    return {
        "primary_nvme_controller": resolved.get("primary_nvme_controller"),
        "primary_nvme_device": resolved.get("primary_nvme_device"),
        "primary_nvme_namespace": resolved.get("primary_nvme_namespace"),
        "primary_nvme_model": summary.get("model"),
        "primary_nvme_serial": summary.get("serial"),
        "primary_nvme_fw": summary.get("firmware_rev"),
        "primary_nvme_size": summary.get("size"),
        "pcie_link": summary.get("pcie_link"),
    }


def collect_nvme_identity(device: str) -> dict[str, Any]:
    return _collect_nvme_with_fallback("id-ctrl", device)


def collect_nvme_smart(device: str) -> dict[str, Any]:
    return _collect_nvme_with_fallback("smart-log", device)


def _run_capture_command(cmd: list[str], timeout_s: float = 10.0) -> dict[str, Any]:
    timestamp = now_utc_iso()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        return {
            "timestamp_utc": timestamp,
            "success": False,
            "error": "command_not_found",
            "command": cmd,
        }
    except Exception as exc:
        return {
            "timestamp_utc": timestamp,
            "success": False,
            "error": str(exc),
            "command": cmd,
        }
    stdout = (completed.stdout or "")
    stderr = (completed.stderr or "")
    if completed.returncode != 0:
        return {
            "timestamp_utc": timestamp,
            "success": False,
            "error": (stderr.strip() or stdout.strip() or f"rc={completed.returncode}"),
            "return_code": completed.returncode,
            "command": cmd,
            "stdout": stdout,
            "stderr": stderr,
        }
    return {
        "timestamp_utc": timestamp,
        "success": True,
        "return_code": completed.returncode,
        "command": cmd,
        "stdout": stdout,
        "stderr": stderr,
    }


def _collect_nvme_with_fallback(subcommand: str, device: str, timeout_s: float = 10.0) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    device_candidates = [device]
    if re.match(r"^/dev/nvme\d+$", device):
        device_candidates.append(f"{device}n1")
    seen_commands: set[tuple[str, ...]] = set()
    variants: list[tuple[str, list[str], str]] = []
    for candidate in device_candidates:
        for fmt, cmd in [
            ("json", ["nvme", subcommand, candidate, "--output-format=json"]),
            ("json", ["nvme", subcommand, candidate, "-o", "json"]),
            ("normal", ["nvme", subcommand, candidate, "--output-format=normal"]),
            ("normal", ["nvme", subcommand, candidate]),
        ]:
            key = tuple(cmd)
            if key in seen_commands:
                continue
            seen_commands.add(key)
            variants.append((fmt, cmd, candidate))
    for fmt, cmd, candidate in variants:
        result = _run_capture_command(cmd, timeout_s=timeout_s)
        attempt = {
            "command": cmd,
            "format": fmt,
            "device_candidate": candidate,
            "success": bool(result.get("success")),
            "return_code": result.get("return_code"),
            "error": result.get("error"),
        }
        attempts.append(attempt)
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        if not result.get("success"):
            if fmt == "json" and stdout.strip():
                try:
                    parsed_nonzero = json.loads(stdout)
                except json.JSONDecodeError:
                    parsed_nonzero = None
                if isinstance(parsed_nonzero, dict):
                    attempt["parsed_with_nonzero_status"] = True
                    payload_ok = not (
                        isinstance(parsed_nonzero.get("error"), str)
                        and bool(str(parsed_nonzero.get("error")).strip())
                    )
                    return {
                        "timestamp_utc": now_utc_iso(),
                        "success": True,
                        "device": device,
                        "device_used": candidate,
                        "subcommand": subcommand,
                        "format": "json",
                        "command": cmd,
                        "return_code": result.get("return_code"),
                        "payload": parsed_nonzero,
                        "payload_ok": bool(payload_ok),
                        "stderr": stderr.strip() or None,
                        "attempts": attempts,
                    }
            if result.get("error") == "command_not_found":
                break
            continue
        if fmt == "json":
            try:
                parsed = json.loads(stdout) if stdout.strip() else {}
            except json.JSONDecodeError as exc:
                attempt["parse_error"] = str(exc)
                continue
            payload_ok = not (
                isinstance(parsed, dict)
                and isinstance(parsed.get("error"), str)
                and bool(str(parsed.get("error")).strip())
            )
            return {
                "timestamp_utc": now_utc_iso(),
                "success": True,
                "device": device,
                "device_used": candidate,
                "subcommand": subcommand,
                "format": "json",
                "command": cmd,
                "payload": parsed,
                "payload_ok": bool(payload_ok),
                "stderr": stderr.strip() or None,
                "attempts": attempts,
            }
        payload_ok = "permission denied" not in stdout.lower() and "permission denied" not in stderr.lower()
        return {
            "timestamp_utc": now_utc_iso(),
            "success": True,
            "device": device,
            "device_used": candidate,
            "subcommand": subcommand,
            "format": "normal",
            "command": cmd,
            "payload": {"raw_stdout": stdout},
            "payload_ok": bool(payload_ok),
            "stdout": stdout,
            "stderr": stderr,
            "attempts": attempts,
        }
    return {
        "timestamp_utc": now_utc_iso(),
        "success": False,
        "device": device,
        "subcommand": subcommand,
        "format": None,
        "error": attempts[-1].get("error") if attempts else "unknown_error",
        "payload_ok": False,
        "attempts": attempts,
    }


def analyze_eviction_signal_io(phase_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _analyze_rehydrate_like_signal_io(
        phase_summaries,
        baseline_phase="warm_A",
        pressure_phase="pressure_B",
        replay_phase="replay_A",
    )
    payload["warm_read_mib"] = payload.get("baseline_read_mib")
    return payload


def analyze_rehydrate_signal_io(phase_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return _analyze_rehydrate_like_signal_io(
        phase_summaries,
        baseline_phase="populate",
        pressure_phase="thrash",
        replay_phase="replay",
    )


def _analyze_rehydrate_like_signal_io(
    phase_summaries: list[dict[str, Any]],
    *,
    baseline_phase: str,
    pressure_phase: str,
    replay_phase: str,
) -> dict[str, Any]:
    by_name = {str(p.get("phase")): p for p in phase_summaries}
    baseline = by_name.get(baseline_phase, {})
    pressure = by_name.get(pressure_phase, {})
    replay = by_name.get(replay_phase, {})
    baseline_io = (baseline.get("io_delta") or {}) if isinstance(baseline, dict) else {}
    pressure_io = (pressure.get("io_delta") or {}) if isinstance(pressure, dict) else {}
    replay_io = (replay.get("io_delta") or {}) if isinstance(replay, dict) else {}
    baseline_read = float(baseline_io.get("read_mib_delta", 0.0))
    pressure_write = float(pressure_io.get("write_mib_delta", 0.0))
    replay_read = float(replay_io.get("read_mib_delta", 0.0))
    if replay_read > max(1.0, baseline_read * 1.20):
        signal = "possible_readback"
        interpretation = "Replay phase shows higher NVMe reads than baseline phase."
    elif replay_read > 0.0:
        signal = "weak_read_signal"
        interpretation = "Replay phase has non-zero reads, attribution is weak."
    else:
        signal = "no_clear_readback"
        interpretation = "No replay read spike in block-device counters."
    return {
        "signal": signal,
        "baseline_phase": baseline_phase,
        "pressure_phase": pressure_phase,
        "replay_phase": replay_phase,
        "baseline_read_mib": round(baseline_read, 3),
        "pressure_write_mib": round(pressure_write, 3),
        "replay_read_mib": round(replay_read, 3),
        "interpretation": interpretation,
    }


def build_kvbm_metrics_summary(
    snapshots: list[dict[str, Any]],
    phase_deltas: dict[str, dict[str, Any]],
    scenario: str,
    kv_mode: str,
    kvbm_enabled: bool,
) -> dict[str, Any]:
    available = bool(kvbm_enabled) and any(bool(s.get("success")) and bool((s.get("metrics") or {})) for s in snapshots)
    status_reason = "kvbm_disabled" if not kvbm_enabled else ("kvbm_metrics_unavailable" if not available else None)
    kvbm_status = build_kvbm_metrics_status(
        kvbm_enabled=kvbm_enabled,
        metrics_available=available,
        snapshot_count=len(snapshots),
        reason=status_reason,
    )
    if available:
        total_offload = sum(float((p or {}).get("offload_blocks_total_delta", 0.0)) for p in phase_deltas.values())
        total_onboard = sum(float((p or {}).get("onboard_blocks_total_delta", 0.0)) for p in phase_deltas.values())
        total_matched = sum(float((p or {}).get("kvbm_matched_tokens_delta", 0.0)) for p in phase_deltas.values())
        rollup: dict[str, Optional[float]] = {
            "offload_blocks_total_delta": round(total_offload, 3),
            "onboard_blocks_total_delta": round(total_onboard, 3),
            "matched_tokens_total_delta": round(total_matched, 3),
        }
    else:
        rollup = {
            "offload_blocks_total_delta": None,
            "onboard_blocks_total_delta": None,
            "matched_tokens_total_delta": None,
        }
    summary: dict[str, Any] = {
        "enabled": bool(kvbm_enabled),
        "available": available,
        "metrics_available": available,
        "reason": status_reason,
        "skipped": (not kvbm_enabled),
        "snapshot_count": len(snapshots),
        "phase_deltas": phase_deltas,
        "kv_mode": kv_mode,
        "kvbm_metrics_status": kvbm_status,
        "rollup": rollup,
    }
    if not available:
        if scenario in ("eviction_replay", "rehydrate_replay"):
            signal_key = "eviction_replay_signal" if scenario == "eviction_replay" else "rehydrate_replay_signal"
            summary[signal_key] = {
                "signal": "skipped",
                "reason": str(kvbm_status.get("reason") or "kvbm_metrics_unavailable"),
                "interpretation": "KVBM counters unavailable for this run.",
            }
        if scenario == "reuse_verify":
            summary["reuse_verify_signal"] = {
                "signal": "skipped",
                "reason": str(kvbm_status.get("reason") or "kvbm_metrics_unavailable"),
                "interpretation": "KVBM counters unavailable for this run.",
            }
        return summary
    if scenario in ("eviction_replay", "rehydrate_replay"):
        baseline_phase = "warm_A" if scenario == "eviction_replay" else "populate"
        pressure_phase = "pressure_B" if scenario == "eviction_replay" else "thrash"
        replay_phase = "replay_A" if scenario == "eviction_replay" else "replay"
        signal_key = "eviction_replay_signal" if scenario == "eviction_replay" else "rehydrate_replay_signal"
        baseline = phase_deltas.get(baseline_phase, {})
        pressure = phase_deltas.get(pressure_phase, {})
        replay = phase_deltas.get(replay_phase, {})
        baseline_onboard = float(baseline.get("onboard_blocks_total_delta", 0.0))
        pressure_onboard = float(pressure.get("onboard_blocks_total_delta", 0.0))
        replay_onboard = float(replay.get("onboard_blocks_total_delta", 0.0))
        baseline_matched = float(baseline.get("kvbm_matched_tokens_delta", 0.0))
        pressure_matched = float(pressure.get("kvbm_matched_tokens_delta", 0.0))
        replay_matched = float(replay.get("kvbm_matched_tokens_delta", 0.0))
        if replay_onboard > 0.0:
            signal = "disk_rehydrate_detected"
            interpretation = "Replay phase has non-zero onboard block counters."
            reuse_gate = "reuse_present"
        elif replay_matched <= 0.0:
            signal = "no_onboard_no_reuse_signal"
            interpretation = (
                "Replay onboard counters are zero and matched tokens are zero; "
                "rehydrate path was not eligible because cross-request reuse did not trigger."
            )
            reuse_gate = "reuse_absent"
        elif pressure_onboard > 0.0 or baseline_onboard > 0.0:
            signal = "onboard_outside_replay"
            interpretation = "Onboard counters are non-zero, but replay was zero."
            reuse_gate = "reuse_present"
        else:
            signal = "no_onboard_signal"
            interpretation = "No onboard block counters observed despite non-zero matched tokens."
            reuse_gate = "reuse_present"
        summary[signal_key] = {
            "signal": signal,
            "baseline_phase": baseline_phase,
            "pressure_phase": pressure_phase,
            "replay_phase": replay_phase,
            "baseline_onboard_blocks": round(baseline_onboard, 3),
            "warm_onboard_blocks": round(baseline_onboard, 3),
            "pressure_onboard_blocks": round(pressure_onboard, 3),
            "replay_onboard_blocks": round(replay_onboard, 3),
            "baseline_matched_tokens": round(baseline_matched, 3),
            "warm_matched_tokens": round(baseline_matched, 3),
            "pressure_matched_tokens": round(pressure_matched, 3),
            "replay_matched_tokens": round(replay_matched, 3),
            "replay_onboard_blocks_h2d": round(float(replay.get("kvbm_onboard_blocks_h2d_delta", 0.0)), 3),
            "replay_onboard_blocks_d2d": round(float(replay.get("kvbm_onboard_blocks_d2d_delta", 0.0)), 3),
            "reuse_gate": reuse_gate,
            "interpretation": interpretation,
        }
    if scenario == "reuse_verify":
        reuse_2 = phase_deltas.get("reuse_2", {})
        reuse_3 = phase_deltas.get("reuse_3", {})
        matched_2 = float(reuse_2.get("kvbm_matched_tokens_delta", 0.0))
        matched_3 = float(reuse_3.get("kvbm_matched_tokens_delta", 0.0))
        h2d_2 = float(reuse_2.get("kvbm_onboard_blocks_h2d_delta", 0.0))
        h2d_3 = float(reuse_3.get("kvbm_onboard_blocks_h2d_delta", 0.0))
        d2d_2 = float(reuse_2.get("kvbm_onboard_blocks_d2d_delta", 0.0))
        d2d_3 = float(reuse_3.get("kvbm_onboard_blocks_d2d_delta", 0.0))
        if matched_2 > 0.0 or matched_3 > 0.0:
            signal = "prefix_reuse_detected"
            interpretation = "Matched tokens are non-zero on repeated requests."
        else:
            signal = "no_prefix_reuse_signal"
            interpretation = (
                "Matched tokens are zero for repeated identical requests; "
                "cross-request prefix reuse appears inactive in this serving path."
            )
        summary["reuse_verify_signal"] = {
            "signal": signal,
            "reuse_2_matched_tokens": round(matched_2, 3),
            "reuse_3_matched_tokens": round(matched_3, 3),
            "reuse_2_onboard_blocks_h2d": round(h2d_2, 3),
            "reuse_3_onboard_blocks_h2d": round(h2d_3, 3),
            "reuse_2_onboard_blocks_d2d": round(d2d_2, 3),
            "reuse_3_onboard_blocks_d2d": round(d2d_3, 3),
            "interpretation": interpretation,
        }
    return summary


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        match = PROM_LINE_RE.match(s)
        if not match:
            continue
        name, value = match.groups()
        try:
            parsed[name] = parsed.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return parsed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, sort_keys=True) + "\n")


def analyze_request_identity(rows: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "reason": "no_requests",
            "total_requests": 0,
        }

    prompt_hashes = {str(r.get("prompt_sha256")) for r in rows if r.get("prompt_sha256")}
    params_hashes = {str(r.get("generation_params_sha256")) for r in rows if r.get("generation_params_sha256")}
    identity_hashes = {str(r.get("request_identity_sha256")) for r in rows if r.get("request_identity_sha256")}
    payload_hashes = {str(r.get("request_payload_sha256")) for r in rows if r.get("request_payload_sha256")}
    prefix_hashes = {str(r.get("prefix_hash")) for r in rows if r.get("prefix_hash")}
    session_ids = {str(r.get("session_id")) for r in rows if r.get("session_id")}
    session_counts: dict[str, int] = {}
    for row in rows:
        sid = row.get("session_id")
        if sid:
            key = str(sid)
            session_counts[key] = session_counts.get(key, 0) + 1

    phase_identity: dict[str, dict[str, Any]] = {}
    header_keys: set[str] = set()
    for row in rows:
        phase = str(row.get("phase") or "unknown")
        item = phase_identity.setdefault(
            phase,
            {
                "request_count": 0,
                "request_identity_sha256": [],
                "prompt_sha256": [],
                "generation_params_sha256": [],
            },
        )
        item["request_count"] += 1
        if row.get("request_identity_sha256"):
            item["request_identity_sha256"].append(str(row["request_identity_sha256"]))
        if row.get("prompt_sha256"):
            item["prompt_sha256"].append(str(row["prompt_sha256"]))
        if row.get("generation_params_sha256"):
            item["generation_params_sha256"].append(str(row["generation_params_sha256"]))
        for key in (row.get("response_header_hints") or {}).keys():
            header_keys.add(str(key))

    phase_identity_sorted: dict[str, dict[str, Any]] = {}
    for phase_name in sorted(phase_identity):
        raw = phase_identity[phase_name]
        phase_identity_sorted[phase_name] = {
            "request_count": raw["request_count"],
            "unique_request_identity_sha256": sorted(set(raw["request_identity_sha256"])),
            "unique_prompt_sha256": sorted(set(raw["prompt_sha256"])),
            "unique_generation_params_sha256": sorted(set(raw["generation_params_sha256"])),
        }

    summary: dict[str, Any] = {
        "available": True,
        "total_requests": len(rows),
        "unique_prompt_sha256_count": len(prompt_hashes),
        "unique_generation_params_sha256_count": len(params_hashes),
        "unique_request_identity_sha256_count": len(identity_hashes),
        "unique_request_payload_sha256_count": len(payload_hashes),
        "unique_prefix_hash_count": len(prefix_hashes),
        "unique_session_id_count": len(session_ids),
        "session_request_counts": {k: session_counts[k] for k in sorted(session_counts)},
        "response_header_keys_seen": sorted(header_keys),
        "phase_identity": phase_identity_sorted,
    }

    if scenario == "reuse_verify":
        reuse_rows = sorted(
            [r for r in rows if str(r.get("phase", "")).startswith("reuse_")],
            key=lambda r: int(r.get("request_index", 0)),
        )
        reuse_identity_hashes = [str(r.get("request_identity_sha256")) for r in reuse_rows if r.get("request_identity_sha256")]
        reuse_prompt_hashes = [str(r.get("prompt_sha256")) for r in reuse_rows if r.get("prompt_sha256")]
        reuse_params_hashes = [str(r.get("generation_params_sha256")) for r in reuse_rows if r.get("generation_params_sha256")]
        same_identity = len(reuse_identity_hashes) >= 2 and len(set(reuse_identity_hashes)) == 1
        same_prompt = len(reuse_prompt_hashes) >= 2 and len(set(reuse_prompt_hashes)) == 1
        same_params = len(reuse_params_hashes) >= 2 and len(set(reuse_params_hashes)) == 1
        if same_identity and same_prompt and same_params:
            verdict = "identical_inputs_confirmed"
            interpretation = "Prompt bytes and generation parameters were identical across reuse_verify requests."
        else:
            verdict = "identity_mismatch_detected"
            interpretation = (
                "Prompt or generation parameter identity differs across reuse_verify requests; "
                "reuse counters are not directly comparable."
            )
        summary["reuse_verify_identity"] = {
            "request_count": len(reuse_rows),
            "request_identity_sha256": reuse_identity_hashes,
            "prompt_sha256": reuse_prompt_hashes,
            "generation_params_sha256": reuse_params_hashes,
            "identical_request_identity": same_identity,
            "identical_prompt_bytes": same_prompt,
            "identical_generation_params": same_params,
            "verdict": verdict,
            "interpretation": interpretation,
        }
    return summary


def has_status_code(rows: list[dict[str, Any]], code: int) -> bool:
    for row in rows:
        if int(row.get("status_code", 0)) == code:
            return True
    return False


def has_prompt_limit_error(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        error = str(row.get("error") or "")
        if PROMPT_LIMIT_ERROR_RE.search(error):
            return True
    return False


def collect_fingerprint(container_name: str) -> dict[str, Any]:
    return {
        "timestamp_utc": now_utc_iso(),
        "host_platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_sha": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "container_name": container_name,
        "container_image": run_cmd(["docker", "inspect", "--format", "{{.Config.Image}}", container_name]),
        "container_image_digest": run_cmd(["docker", "inspect", "--format", "{{.Image}}", container_name]),
    }


def run_cmd(cmd: list[str]) -> Optional[str]:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    out = completed.stdout.strip()
    return out if out else None


def render_report_markdown(summary: dict[str, Any]) -> str:
    run_id = summary.get("run_id")
    created = summary.get("created_utc")
    run_valid = bool(summary.get("run_valid"))
    overall = summary.get("overall_summary", {})
    kv_mode = summary.get("kv_mode", {})
    scenario = summary.get("scenario")
    fingerprint = summary.get("fingerprint", {})
    invalid_reason = summary.get("invalid_reason")
    invalid_details = summary.get("invalid_details", [])
    phase_summaries = summary.get("phase_summaries", [])
    kvbm = summary.get("kvbm_metrics", {})
    request_identity = summary.get("request_identity", {})
    phase_by_name = {str(p.get("phase")): p for p in phase_summaries}
    kvbm_status = summary.get("kvbm_metrics_status")
    if not isinstance(kvbm_status, dict):
        kvbm_status = kvbm.get("kvbm_metrics_status") if isinstance(kvbm, dict) else None
    if not isinstance(kvbm_status, dict):
        kvbm_status = build_kvbm_metrics_status(
            kvbm_enabled=bool(kv_mode.get("kvbm_enabled")),
            metrics_available=bool((kvbm or {}).get("available")),
            snapshot_count=int((kvbm or {}).get("snapshot_count") or 0),
            reason=((kvbm or {}).get("reason") if isinstance(kvbm, dict) else None),
        )
    kvbm_status_name = str(kvbm_status.get("status") or "unavailable")
    kvbm_metrics_ok = kvbm_status_name == "ok"

    def display_metric(value: Any) -> str:
        if value is None:
            return "NA"
        return str(value)

    def phase_kvbm_value(phase_name: str, key: str) -> float:
        phase = phase_by_name.get(phase_name, {})
        kv = (phase.get("kvbm_metrics_delta") or {}) if isinstance(phase, dict) else {}
        try:
            return float(kv.get(key, 0.0))
        except Exception:
            return 0.0

    rollup = kvbm.get("rollup")
    if not isinstance(rollup, dict):
        if kvbm_metrics_ok:
            rollup = {
                "offload_blocks_total_delta": round(
                    sum(phase_kvbm_value(name, "offload_blocks_total_delta") for name in phase_by_name),
                    3,
                ),
                "onboard_blocks_total_delta": round(
                    sum(phase_kvbm_value(name, "onboard_blocks_total_delta") for name in phase_by_name),
                    3,
                ),
                "matched_tokens_total_delta": round(
                    sum(phase_kvbm_value(name, "kvbm_matched_tokens_delta") for name in phase_by_name),
                    3,
                ),
            }
        else:
            rollup = {
                "offload_blocks_total_delta": None,
                "onboard_blocks_total_delta": None,
                "matched_tokens_total_delta": None,
            }

    lines = [
        f"# Benchmark Report: {run_id}",
        "",
        "## Run Status",
        f"- Created (UTC): `{created}`",
        f"- Scenario: `{scenario}`",
        f"- Run valid: `{run_valid}`",
    ]
    if not run_valid:
        lines.append(f"- Invalid reason: `{invalid_reason}`")
        for detail in invalid_details:
            lines.append(f"- Invalid detail: {detail}")

    lines.extend(
        [
            "",
            "## Platform + Software Fingerprint",
            f"- Git SHA: `{fingerprint.get('git_sha')}`",
            f"- Git branch: `{fingerprint.get('git_branch')}`",
            f"- Container image: `{fingerprint.get('container_image')}`",
            f"- Container digest: `{fingerprint.get('container_image_digest')}`",
            f"- Host platform: `{fingerprint.get('host_platform')}`",
            f"- Python: `{fingerprint.get('python_version')}`",
            "",
            "## KV Configuration",
            f"- Tier mode: `{summary.get('tier_mode')}`",
            f"- KV mode: `{kv_mode.get('mode')}`",
            f"- KVBM enabled: `{kv_mode.get('kvbm_enabled')}`",
            f"- CPU cache GB: `{kv_mode.get('cpu_cache_gb')}`",
            f"- Disk cache GB: `{kv_mode.get('disk_cache_gb')}`",
            f"- Variant tags: `{', '.join(summary.get('variant_tags', [])) or 'none'}`",
            "",
            "## Overall Results",
            f"- Error rate: `{overall.get('error_rate')}`",
            f"- Throughput (req/s): `{overall.get('req_per_s')}`",
            f"- Output tokens/s (est): `{overall.get('output_tokens_per_s_est')}`",
            f"- Latency p50/p95/p99 (ms): `{(overall.get('latency_ms') or {}).get('p50')}` / `{(overall.get('latency_ms') or {}).get('p95')}` / `{(overall.get('latency_ms') or {}).get('p99')}`",
            f"- Legacy TTFT p50/p95/p99 (ms): `{(overall.get('ttft_ms') or {}).get('p50')}` / `{(overall.get('ttft_ms') or {}).get('p95')}` / `{(overall.get('ttft_ms') or {}).get('p99')}`",
            f"- KVBM rollup offload/onboard/matched: `{display_metric(rollup.get('offload_blocks_total_delta'))}` / "
            f"`{display_metric(rollup.get('onboard_blocks_total_delta'))}` / "
            f"`{display_metric(rollup.get('matched_tokens_total_delta'))}`",
        ]
    )
    ttfc_overall = overall.get("ttfc_ms") if isinstance(overall.get("ttfc_ms"), dict) else {}
    if ttfc_overall:
        lines.append(
            f"- Streaming TTFC p50/p95/p99 (ms): `{ttfc_overall.get('p50')}` / `{ttfc_overall.get('p95')}` / `{ttfc_overall.get('p99')}`"
        )
    ttfb_overall = overall.get("ttfb_ms") if isinstance(overall.get("ttfb_ms"), dict) else {}
    if ttfb_overall:
        lines.append(
            f"- Streaming TTFB p50/p95/p99 (ms): `{ttfb_overall.get('p50')}` / `{ttfb_overall.get('p95')}` / `{ttfb_overall.get('p99')}`"
        )

    lines.extend(["", "## Phase Results"])
    for phase in phase_summaries:
        phase_name = phase.get("phase")
        io_delta = phase.get("io_delta") or {}
        proc_io_delta = phase.get("worker_process_io_delta") or {}
        kvbm_delta = phase.get("kvbm_metrics_delta") or {}
        offload_value = kvbm_delta.get("offload_blocks_total_delta") if kvbm_metrics_ok else None
        onboard_value = kvbm_delta.get("onboard_blocks_total_delta") if kvbm_metrics_ok else None
        matched_value = kvbm_delta.get("kvbm_matched_tokens_delta") if kvbm_metrics_ok else None
        lines.append(
            f"- `{phase_name}`: err={phase.get('error_rate')} req/s={phase.get('req_per_s')} "
            f"nvme_write_mib={io_delta.get('write_mib_delta')} nvme_read_mib={io_delta.get('read_mib_delta')} "
            f"proc_write_mib={proc_io_delta.get('write_mib_delta')} proc_read_mib={proc_io_delta.get('read_mib_delta')} "
            f"ttft_p95_ms={display_metric((phase.get('ttft_ms') or {}).get('p95'))} "
            f"ttfc_p95_ms={display_metric((phase.get('ttfc_ms') or {}).get('p95'))} "
            f"offload_blocks_delta={display_metric(offload_value)} "
            f"onboard_blocks_delta={display_metric(onboard_value)} "
            f"matched_tokens_delta={display_metric(matched_value)}"
        )

    lines.extend(["", "## KVBM Metrics Signal"])
    lines.append(f"- KVBM metrics status: `{kvbm_status_name}`")
    lines.append(f"- KVBM metrics available: `{kvbm_status.get('metrics_available')}`")
    if kvbm_status.get("reason"):
        lines.append(f"- KVBM metrics reason: `{kvbm_status.get('reason')}`")
    evict_signal_raw = kvbm.get("eviction_replay_signal")
    signal_label = "Eviction replay"
    if not isinstance(evict_signal_raw, dict):
        evict_signal_raw = kvbm.get("rehydrate_replay_signal")
        signal_label = "Rehydrate replay"
    evict_signal: dict[str, Any] = dict(evict_signal_raw) if isinstance(evict_signal_raw, dict) else {}
    reuse_signal = kvbm.get("reuse_verify_signal")
    if kvbm_metrics_ok and evict_signal:
        replay_phase_name = str(evict_signal.get("replay_phase") or "replay_A")
        replay_matched = evict_signal.get("replay_matched_tokens")
        if replay_matched is None:
            replay_matched = round(phase_kvbm_value(replay_phase_name, "kvbm_matched_tokens_delta"), 3)
            evict_signal["replay_matched_tokens"] = replay_matched
        replay_h2d = evict_signal.get("replay_onboard_blocks_h2d")
        if replay_h2d is None:
            replay_h2d = round(phase_kvbm_value(replay_phase_name, "kvbm_onboard_blocks_h2d_delta"), 3)
            evict_signal["replay_onboard_blocks_h2d"] = replay_h2d
        replay_d2d = evict_signal.get("replay_onboard_blocks_d2d")
        if replay_d2d is None:
            replay_d2d = round(phase_kvbm_value(replay_phase_name, "kvbm_onboard_blocks_d2d_delta"), 3)
            evict_signal["replay_onboard_blocks_d2d"] = replay_d2d
        reuse_gate = evict_signal.get("reuse_gate")
        if reuse_gate is None:
            reuse_gate = "reuse_absent" if float(replay_matched or 0.0) <= 0.0 else "reuse_present"
            evict_signal["reuse_gate"] = reuse_gate
        interpretation = str(evict_signal.get("interpretation") or "")
        if reuse_gate == "reuse_absent" and (
            not interpretation or interpretation.strip() == "No onboard block counters observed."
        ):
            evict_signal["interpretation"] = (
                "Replay onboard counters are zero and matched tokens are zero; "
                "rehydrate path was not eligible because cross-request reuse did not trigger."
            )
        elif not interpretation:
            evict_signal["interpretation"] = "Replay onboard counters are zero despite non-zero matched tokens."

        lines.append(f"- {signal_label} signal: `{evict_signal.get('signal')}`")
        lines.append(
            f"- Replay matched/onboard(h2d,d2d): `{evict_signal.get('replay_matched_tokens')}` / "
            f"`{evict_signal.get('replay_onboard_blocks_h2d')}`, `{evict_signal.get('replay_onboard_blocks_d2d')}`"
        )
        lines.append(f"- Reuse gate: `{evict_signal.get('reuse_gate')}`")
        lines.append(f"- Interpretation: {evict_signal.get('interpretation')}")
    elif not kvbm_metrics_ok:
        lines.append("- KVBM signal analysis skipped because counters were not collected for this run.")
        if evict_signal:
            lines.append(f"- {signal_label} signal: `{evict_signal.get('signal')}`")
            lines.append(f"- Interpretation: {evict_signal.get('interpretation')}")
    if kvbm_metrics_ok and isinstance(reuse_signal, dict):
        lines.append(f"- Reuse verify signal: `{reuse_signal.get('signal')}`")
        lines.append(
            f"- reuse_2 matched/onboard(h2d,d2d): `{reuse_signal.get('reuse_2_matched_tokens')}` / "
            f"`{reuse_signal.get('reuse_2_onboard_blocks_h2d')}`, `{reuse_signal.get('reuse_2_onboard_blocks_d2d')}`"
        )
        lines.append(
            f"- reuse_3 matched/onboard(h2d,d2d): `{reuse_signal.get('reuse_3_matched_tokens')}` / "
            f"`{reuse_signal.get('reuse_3_onboard_blocks_h2d')}`, `{reuse_signal.get('reuse_3_onboard_blocks_d2d')}`"
        )
        lines.append(f"- Interpretation: {reuse_signal.get('interpretation')}")

    lines.extend(["", "## Request Identity"])
    if request_identity.get("available"):
        lines.append(f"- Unique prompt hashes: `{request_identity.get('unique_prompt_sha256_count')}`")
        lines.append(
            f"- Unique generation param hashes: `{request_identity.get('unique_generation_params_sha256_count')}`"
        )
        lines.append(
            f"- Unique request identity hashes: `{request_identity.get('unique_request_identity_sha256_count')}`"
        )
        reuse_identity = request_identity.get("reuse_verify_identity")
        if isinstance(reuse_identity, dict):
            lines.append(f"- Reuse identity verdict: `{reuse_identity.get('verdict')}`")
            lines.append(f"- Interpretation: {reuse_identity.get('interpretation')}")
    else:
        lines.append("- Request identity data unavailable.")

    lines.extend(["", "## Anomalies + Limitations"])
    if not run_valid:
        lines.append("- Run is invalid and excluded from aggregate conclusions.")
    if isinstance(evict_signal, dict) and evict_signal.get("reuse_gate") == "reuse_absent":
        lines.append("- Offload/spill activity does not imply rehydrate readiness when matched tokens stay zero.")
    if isinstance(reuse_signal, dict) and reuse_signal.get("signal") == "no_prefix_reuse_signal":
        lines.append("- No prefix reuse signal: disk onboarding is gated behind cross-request matching.")
    lines.append(
        "- TTFC is the preferred streaming-first-output metric; TTFT is kept for backward compatibility and remains a proxy in non-stream runs."
    )
    lines.append("- NVMe counters are supporting evidence; KVBM counters are primary for offload/onboard proof when enabled.")

    return "\n".join(lines).strip() + "\n"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        run_dir, run_valid = asyncio.run(run_benchmark(args))
    except Exception:
        LOG.exception("Benchmark run failed before artifact generation.")
        return 1
    print(run_dir)
    return 0 if run_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

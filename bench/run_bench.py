#!/usr/bin/env python3
"""Benchmark driver for Dynamo + TRT-LLM + KVBM OpenAI-compatible completions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import logging
import math
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .prompts import PromptSpec, TokenEstimator, generate_prompt_set, generate_replay_sets, manifest_rows
from .telemetry import TelemetryManager

LOG = logging.getLogger("bench.run_bench")

PROM_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([0-9.eE+-]+)$")

METRICS_SNAPSHOT_PHASE_SUFFIX: dict[str, str] = {
    "pressure_B": "pressure",
    "replay_A": "replay",
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
    parser.add_argument("--scenario", choices=["standard", "eviction_replay", "reuse_verify"], default="standard")
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
    parser.add_argument("--stream", action="store_true", help="Request streaming responses for TTFT proxy.")
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

    parser.add_argument("--kv-mode", choices=["off", "cpu_only", "cpu_disk"], default="cpu_disk")
    parser.add_argument("--kv-cpu-cache-gb", type=float, default=None, help="Resolved CPU cache size tag for artifacts.")
    parser.add_argument("--kv-disk-cache-gb", type=float, default=None, help="Resolved disk cache size tag for artifacts.")
    parser.add_argument("--variant-tag", action="append", default=[], help="Optional variant tags for metadata/report.")
    parser.add_argument("--diagnostic-disable-partial-reuse", action="store_true")
    parser.add_argument("--diagnostic-disable-block-reuse", action="store_true")
    parser.add_argument("--diagnostic-disable-disk-offload-filter", action="store_true")

    parser.add_argument("--engine-max-input-tokens", type=int, default=8192)
    parser.add_argument("--input-token-safety-margin", type=int, default=256)

    parser.add_argument("--phase-io-device", default="nvme0n1", help="Block device for phase read/write deltas.")
    parser.add_argument("--collect-telemetry", action="store_true", help="Start/stop bench/scripts collectors.")
    parser.add_argument("--telemetry-interval-s", type=int, default=1)
    parser.add_argument("--telemetry-pid", default="ALL", help="PID target for pidstat (`ALL` by default).")
    parser.add_argument("--iostat-device", default="nvme0n1")
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
    return parser.parse_args()


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
    path.write_text(
        (
            f"# snapshot_unavailable phase={phase_name}\n"
            f"# error={snap.get('error')}\n"
        ),
        encoding="utf-8",
    )


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
    phase_plan = build_phase_plan(
        args=args,
        estimator=estimator,
        short_range=short_range,
        long_range=long_range,
    )
    unique_prompts = dedupe_prompts([p for phase in phase_plan for p in phase.prompts])
    write_prompt_manifest(run_dir / "prompts_manifest.jsonl", unique_prompts)
    preflight = preflight_validate_prompts(unique_prompts, args)

    telemetry_manager: Optional[TelemetryManager] = None
    telemetry_start_report = None
    telemetry_stop_report = None
    kvbm_probe = KVBMMetricsProbe(args.kvbm_metrics_url, args.kvbm_metrics_timeout_s)
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
        "kv_mode": kv_mode_resolved,
        "preflight": preflight,
        "client_request_parameters": {
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "request_seed": (int(args.request_seed) if args.request_seed is not None else None),
            "stop": list(args.stop),
            "stream": bool(args.stream),
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
                        if telemetry_manager is not None:
                            telemetry_manager.snapshot_kvbm_dir(f"before_{phase.name}")
                        before_io = read_block_device_stats(args.phase_io_device)
                        before_kvbm = kvbm_probe.snapshot(f"before_{phase.name}")
                        kvbm_snapshots.append(before_kvbm)
                        append_jsonl(kvbm_snapshots_path, before_kvbm)

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

                        after_io = read_block_device_stats(args.phase_io_device)
                        after_kvbm = kvbm_probe.snapshot(f"after_{phase.name}")
                        kvbm_snapshots.append(after_kvbm)
                        append_jsonl(kvbm_snapshots_path, after_kvbm)
                        if telemetry_manager is not None:
                            telemetry_manager.snapshot_kvbm_dir(f"after_{phase.name}")

                        suffix = METRICS_SNAPSHOT_PHASE_SUFFIX.get(phase.name)
                        if args.capture_metrics_snapshot and suffix:
                            system_snap = fetch_metrics_text(args.metrics_system_url, args.kvbm_metrics_timeout_s)
                            kvbm_snap = fetch_metrics_text(args.metrics_kvbm_url, args.kvbm_metrics_timeout_s)
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
                        kvbm_delta = kvbm_probe.delta(before_kvbm, after_kvbm)
                        kvbm_phase_deltas[phase.name] = kvbm_delta

                        summary = summarize_phase(phase_rows, phase_duration_s)
                        summary["phase"] = phase.name
                        summary["concurrency"] = phase.concurrency
                        summary["requests"] = len(phase.prompts)
                        summary["kvbm_metrics_delta"] = kvbm_delta
                        if io_delta is not None:
                            summary["block_device"] = args.phase_io_device
                            summary["io_delta"] = io_delta
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

    kvbm_summary = build_kvbm_metrics_summary(kvbm_snapshots, kvbm_phase_deltas, args.scenario, args.kv_mode)
    request_identity_summary = analyze_request_identity(all_rows, args.scenario)
    if executed_workload and args.kv_mode != "off" and not args.allow_missing_kvbm_metrics and not kvbm_summary.get("available", False):
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
        "kv_mode": kv_mode_resolved,
        "variant_tags": list(args.variant_tag or []),
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
            "TTFT proxy is available only when --stream is enabled and the server emits stream chunks.",
            "Invalid runs are explicitly marked and excluded from aggregate conclusions.",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report = render_report_markdown(summary_payload)
    (run_dir / args.report_filename).write_text(report, encoding="utf-8")

    LOG.info("Benchmark complete. Artifacts: %s", run_dir)
    return run_dir, run_valid


def resolve_kv_mode(args: argparse.Namespace) -> dict[str, Any]:
    mode = args.kv_mode
    if mode == "off":
        return {
            "mode": mode,
            "kvbm_enabled": False,
            "cpu_cache_gb": 0.0,
            "disk_cache_gb": 0.0,
            "diagnostic": {
                "disable_partial_reuse": bool(args.diagnostic_disable_partial_reuse),
                "disable_block_reuse": bool(args.diagnostic_disable_block_reuse),
                "disable_disk_offload_filter": bool(args.diagnostic_disable_disk_offload_filter),
            },
        }
    if mode == "cpu_only":
        return {
            "mode": mode,
            "kvbm_enabled": True,
            "cpu_cache_gb": float(args.kv_cpu_cache_gb if args.kv_cpu_cache_gb is not None else 8.0),
            "disk_cache_gb": 0.0,
            "diagnostic": {
                "disable_partial_reuse": bool(args.diagnostic_disable_partial_reuse),
                "disable_block_reuse": bool(args.diagnostic_disable_block_reuse),
                "disable_disk_offload_filter": bool(args.diagnostic_disable_disk_offload_filter),
            },
        }
    return {
        "mode": mode,
        "kvbm_enabled": True,
        "cpu_cache_gb": float(args.kv_cpu_cache_gb if args.kv_cpu_cache_gb is not None else 8.0),
        "disk_cache_gb": float(args.kv_disk_cache_gb if args.kv_disk_cache_gb is not None else 32.0),
        "diagnostic": {
            "disable_partial_reuse": bool(args.diagnostic_disable_partial_reuse),
            "disable_block_reuse": bool(args.diagnostic_disable_block_reuse),
            "disable_disk_offload_filter": bool(args.diagnostic_disable_disk_offload_filter),
        },
    }


def preflight_validate_prompts(prompts: list[PromptSpec], args: argparse.Namespace) -> dict[str, Any]:
    ceiling = args.engine_max_input_tokens - args.max_tokens - args.input_token_safety_margin
    if ceiling <= 0:
        return {
            "failed": True,
            "ceiling_tokens": ceiling,
            "errors": [
                "Invalid preflight configuration: computed prompt ceiling <= 0 "
                f"(engine={args.engine_max_input_tokens}, max_tokens={args.max_tokens}, margin={args.input_token_safety_margin})."
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
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=args.stop,
                stream=bool(args.stream),
                request_seed=args.request_seed,
            )
            completion = await client.create_completion(
                model=model_id,
                prompt=prompt.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=args.stop,
                stream=bool(args.stream),
                seed=args.request_seed,
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
                "latency_ms": round(float(completion.latency_ms), 3),
                "ttft_ms": round(float(completion.ttft_ms), 3) if completion.ttft_ms is not None else None,
                "status_code": completion.status_code,
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
                "response_path": response_path,
                "response_header_hints": completion.response_headers,
            }
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
        }
    ok_rows = [r for r in rows if not r.get("error") and 200 <= int(r.get("status_code", 0)) < 400]
    err_rows = [r for r in rows if r not in ok_rows]
    latencies = [float(r["latency_ms"]) for r in rows]
    ttft = [float(r["ttft_ms"]) for r in rows if r.get("ttft_ms") is not None]
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


def analyze_eviction_signal_io(phase_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {p.get("phase"): p for p in phase_summaries}
    warm = by_name.get("warm_A", {})
    pressure = by_name.get("pressure_B", {})
    replay = by_name.get("replay_A", {})
    warm_io = (warm.get("io_delta") or {}) if isinstance(warm, dict) else {}
    pressure_io = (pressure.get("io_delta") or {}) if isinstance(pressure, dict) else {}
    replay_io = (replay.get("io_delta") or {}) if isinstance(replay, dict) else {}
    warm_read = float(warm_io.get("read_mib_delta", 0.0))
    pressure_write = float(pressure_io.get("write_mib_delta", 0.0))
    replay_read = float(replay_io.get("read_mib_delta", 0.0))

    if replay_read > max(1.0, warm_read * 1.20):
        signal = "possible_readback"
        interpretation = "Replay phase shows higher NVMe reads than warm phase."
    elif replay_read > 0.0:
        signal = "weak_read_signal"
        interpretation = "Replay phase has non-zero reads, attribution is weak."
    else:
        signal = "no_clear_readback"
        interpretation = "No replay read spike in block-device counters."
    return {
        "signal": signal,
        "warm_read_mib": round(warm_read, 3),
        "pressure_write_mib": round(pressure_write, 3),
        "replay_read_mib": round(replay_read, 3),
        "interpretation": interpretation,
    }


def build_kvbm_metrics_summary(
    snapshots: list[dict[str, Any]],
    phase_deltas: dict[str, dict[str, Any]],
    scenario: str,
    kv_mode: str,
) -> dict[str, Any]:
    available = any(bool(s.get("success")) and bool((s.get("metrics") or {})) for s in snapshots)
    total_offload = sum(float((p or {}).get("offload_blocks_total_delta", 0.0)) for p in phase_deltas.values())
    total_onboard = sum(float((p or {}).get("onboard_blocks_total_delta", 0.0)) for p in phase_deltas.values())
    total_matched = sum(float((p or {}).get("kvbm_matched_tokens_delta", 0.0)) for p in phase_deltas.values())
    summary: dict[str, Any] = {
        "available": available,
        "snapshot_count": len(snapshots),
        "phase_deltas": phase_deltas,
        "kv_mode": kv_mode,
        "rollup": {
            "offload_blocks_total_delta": round(total_offload, 3),
            "onboard_blocks_total_delta": round(total_onboard, 3),
            "matched_tokens_total_delta": round(total_matched, 3),
        },
    }
    if scenario == "eviction_replay":
        warm = phase_deltas.get("warm_A", {})
        pressure = phase_deltas.get("pressure_B", {})
        replay = phase_deltas.get("replay_A", {})
        warm_onboard = float(warm.get("onboard_blocks_total_delta", 0.0))
        pressure_onboard = float(pressure.get("onboard_blocks_total_delta", 0.0))
        replay_onboard = float(replay.get("onboard_blocks_total_delta", 0.0))
        warm_matched = float(warm.get("kvbm_matched_tokens_delta", 0.0))
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
        elif pressure_onboard > 0.0 or warm_onboard > 0.0:
            signal = "onboard_outside_replay"
            interpretation = "Onboard counters are non-zero, but replay was zero."
            reuse_gate = "reuse_present"
        else:
            signal = "no_onboard_signal"
            interpretation = "No onboard block counters observed despite non-zero matched tokens."
            reuse_gate = "reuse_present"
        summary["eviction_replay_signal"] = {
            "signal": signal,
            "warm_onboard_blocks": round(warm_onboard, 3),
            "pressure_onboard_blocks": round(pressure_onboard, 3),
            "replay_onboard_blocks": round(replay_onboard, 3),
            "warm_matched_tokens": round(warm_matched, 3),
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

    def phase_kvbm_value(phase_name: str, key: str) -> float:
        phase = phase_by_name.get(phase_name, {})
        kv = (phase.get("kvbm_metrics_delta") or {}) if isinstance(phase, dict) else {}
        try:
            return float(kv.get(key, 0.0))
        except Exception:
            return 0.0

    rollup = kvbm.get("rollup")
    if not isinstance(rollup, dict):
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
            f"- KVBM rollup offload/onboard/matched: `{rollup.get('offload_blocks_total_delta')}` / "
            f"`{rollup.get('onboard_blocks_total_delta')}` / "
            f"`{rollup.get('matched_tokens_total_delta')}`",
        ]
    )

    lines.extend(["", "## Phase Results"])
    for phase in phase_summaries:
        phase_name = phase.get("phase")
        io_delta = phase.get("io_delta") or {}
        kvbm_delta = phase.get("kvbm_metrics_delta") or {}
        lines.append(
            f"- `{phase_name}`: err={phase.get('error_rate')} req/s={phase.get('req_per_s')} "
            f"nvme_write_mib={io_delta.get('write_mib_delta')} nvme_read_mib={io_delta.get('read_mib_delta')} "
            f"offload_blocks_delta={kvbm_delta.get('offload_blocks_total_delta')} "
            f"onboard_blocks_delta={kvbm_delta.get('onboard_blocks_total_delta')} "
            f"matched_tokens_delta={kvbm_delta.get('kvbm_matched_tokens_delta')}"
        )

    lines.extend(["", "## KVBM Metrics Signal"])
    lines.append(f"- KVBM metrics available: `{kvbm.get('available')}`")
    evict_signal_raw = kvbm.get("eviction_replay_signal")
    evict_signal: dict[str, Any] = dict(evict_signal_raw) if isinstance(evict_signal_raw, dict) else {}
    if evict_signal:
        replay_matched = evict_signal.get("replay_matched_tokens")
        if replay_matched is None:
            replay_matched = round(phase_kvbm_value("replay_A", "kvbm_matched_tokens_delta"), 3)
            evict_signal["replay_matched_tokens"] = replay_matched
        replay_h2d = evict_signal.get("replay_onboard_blocks_h2d")
        if replay_h2d is None:
            replay_h2d = round(phase_kvbm_value("replay_A", "kvbm_onboard_blocks_h2d_delta"), 3)
            evict_signal["replay_onboard_blocks_h2d"] = replay_h2d
        replay_d2d = evict_signal.get("replay_onboard_blocks_d2d")
        if replay_d2d is None:
            replay_d2d = round(phase_kvbm_value("replay_A", "kvbm_onboard_blocks_d2d_delta"), 3)
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

        lines.append(f"- Eviction replay signal: `{evict_signal.get('signal')}`")
        lines.append(
            f"- Replay matched/onboard(h2d,d2d): `{evict_signal.get('replay_matched_tokens')}` / "
            f"`{evict_signal.get('replay_onboard_blocks_h2d')}`, `{evict_signal.get('replay_onboard_blocks_d2d')}`"
        )
        lines.append(f"- Reuse gate: `{evict_signal.get('reuse_gate')}`")
        lines.append(f"- Interpretation: {evict_signal.get('interpretation')}")
    reuse_signal = kvbm.get("reuse_verify_signal")
    if isinstance(reuse_signal, dict):
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
    lines.append("- TTFT remains a proxy unless streaming chunk emission is available.")
    lines.append("- NVMe counters are supporting evidence; KVBM counters are primary for offload/onboard proof.")

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

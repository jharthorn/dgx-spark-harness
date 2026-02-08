#!/usr/bin/env python3
"""Benchmark driver for Dynamo + TRT-LLM + KVBM OpenAI-compatible completions."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import math
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
    parser.add_argument("--scenario", choices=["standard", "eviction_replay"], default="standard")
    parser.add_argument("--prompt-set", choices=["short", "long", "mixed"], default="short")
    parser.add_argument("--requests", type=int, default=64, help="Measured requests for standard scenario.")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup requests (excluded from overall summary).")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
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

    parser.add_argument("--kvbm-metrics-url", default="http://127.0.0.1:6880/metrics")
    parser.add_argument("--kvbm-metrics-timeout-s", type=float, default=3.0)
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
    kvbm_snapshots: list[dict[str, Any]] = []
    kvbm_phase_deltas: dict[str, dict[str, Any]] = {}
    kvbm_snapshots_path = run_dir / "kvbm_metrics_snapshots.jsonl"

    requests_path = run_dir / "requests.jsonl"
    request_counter = itertools.count(1)
    overall_rows: list[dict[str, Any]] = []
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

    kvbm_summary = build_kvbm_metrics_summary(kvbm_snapshots, kvbm_phase_deltas, args.scenario, args.kv_mode)
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
            completion = await client.create_completion(
                model=model_id,
                prompt=prompt.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stop=args.stop,
                stream=bool(args.stream),
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
                "output_len_chars": len(output_text),
                "output_tokens_est": output_tokens_est,
                "error": completion.error,
                "stream": bool(args.stream),
                "response_path": response_path,
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
    summary: dict[str, Any] = {
        "available": available,
        "snapshot_count": len(snapshots),
        "phase_deltas": phase_deltas,
        "kv_mode": kv_mode,
    }
    if scenario == "eviction_replay":
        warm = phase_deltas.get("warm_A", {})
        pressure = phase_deltas.get("pressure_B", {})
        replay = phase_deltas.get("replay_A", {})
        warm_onboard = float(warm.get("onboard_blocks_total_delta", 0.0))
        pressure_onboard = float(pressure.get("onboard_blocks_total_delta", 0.0))
        replay_onboard = float(replay.get("onboard_blocks_total_delta", 0.0))
        if replay_onboard > 0.0:
            signal = "disk_rehydrate_detected"
            interpretation = "Replay phase has non-zero onboard block counters."
        elif pressure_onboard > 0.0 or warm_onboard > 0.0:
            signal = "onboard_outside_replay"
            interpretation = "Onboard counters are non-zero, but replay was zero."
        else:
            signal = "no_onboard_signal"
            interpretation = "No onboard block counters observed."
        summary["eviction_replay_signal"] = {
            "signal": signal,
            "warm_onboard_blocks": round(warm_onboard, 3),
            "pressure_onboard_blocks": round(pressure_onboard, 3),
            "replay_onboard_blocks": round(replay_onboard, 3),
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
            f"onboard_blocks_delta={kvbm_delta.get('onboard_blocks_total_delta')}"
        )

    lines.extend(["", "## KVBM Metrics Signal"])
    lines.append(f"- KVBM metrics available: `{kvbm.get('available')}`")
    evict_signal = kvbm.get("eviction_replay_signal")
    if isinstance(evict_signal, dict):
        lines.append(f"- Eviction replay signal: `{evict_signal.get('signal')}`")
        lines.append(f"- Interpretation: {evict_signal.get('interpretation')}")

    lines.extend(["", "## Anomalies + Limitations"])
    if not run_valid:
        lines.append("- Run is invalid and excluded from aggregate conclusions.")
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

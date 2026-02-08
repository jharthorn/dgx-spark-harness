#!/usr/bin/env python3
"""Benchmark driver for Dynamo + TRT-LLM + KVBM OpenAI-compatible completions."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .prompts import PromptSpec, TokenEstimator, generate_prompt_set, generate_replay_sets, manifest_rows
from .telemetry import TelemetryManager

LOG = logging.getLogger("bench.run_bench")


@dataclass
class PhasePlan:
    name: str
    prompts: list[PromptSpec]
    concurrency: int
    include_in_overall: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DGX Spark Dynamo/TRT-LLM/KVBM benchmark driver.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Frontend base URL.")
    parser.add_argument("--model-id", default="auto", help="Model ID; `auto` resolves from /v1/models.")
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

    parser.add_argument("--phase-io-device", default="nvme0n1", help="Block device for phase read/write deltas.")
    parser.add_argument("--collect-telemetry", action="store_true", help="Start/stop bench/scripts collectors.")
    parser.add_argument("--telemetry-interval-s", type=int, default=1)
    parser.add_argument("--telemetry-pid", default="ALL", help="PID target for pidstat (`ALL` by default).")
    parser.add_argument("--iostat-device", default="nvme0n1")
    parser.add_argument("--kvbm-cache-dir", default="/mnt/nvme/kvbm")
    parser.add_argument("--container-name", default="dyn")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def run_benchmark(args: argparse.Namespace) -> Path:
    try:
        from .openai_compat import OpenAICompatClient
    except ModuleNotFoundError as exc:
        if exc.name == "httpx":
            raise RuntimeError("Missing dependency `httpx`; run `pip install -r requirements.txt`.") from exc
        raise

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "telemetry").mkdir(parents=True, exist_ok=True)
    responses_dir = (run_dir / "responses") if args.store_responses else None
    if responses_dir is not None:
        responses_dir.mkdir(parents=True, exist_ok=True)

    estimator = TokenEstimator(args.tokenizer)
    short_range = parse_token_range(args.short_range, "short-range")
    long_range = parse_token_range(args.long_range, "long-range")

    telemetry_manager: Optional[TelemetryManager] = None
    telemetry_start_report = None
    telemetry_stop_report = None
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
            model_id = args.model_id
            if model_id == "auto":
                model_id = await client.fetch_first_model_id()
                LOG.info("Resolved model ID from /v1/models: %s", model_id)

            phase_plan = build_phase_plan(
                args=args,
                estimator=estimator,
                short_range=short_range,
                long_range=long_range,
            )
            unique_prompts = dedupe_prompts([p for phase in phase_plan for p in phase.prompts])
            write_prompt_manifest(run_dir / "prompts_manifest.jsonl", unique_prompts)

            run_config = {
                "run_id": run_id,
                "base_url": args.base_url,
                "model_id": model_id,
                "scenario": args.scenario,
                "prompt_set": args.prompt_set,
                "short_range": short_range,
                "long_range": long_range,
                "tokenizer": estimator.tokenizer_name or "heuristic",
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

            requests_path = run_dir / "requests.jsonl"
            request_counter = itertools.count(1)
            overall_rows: list[dict] = []
            phase_summaries: list[dict] = []
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
                    if telemetry_manager is not None:
                        telemetry_manager.snapshot_kvbm_dir(f"after_{phase.name}")

                    io_delta = diff_block_device_stats(before_io, after_io)
                    summary = summarize_phase(phase_rows, phase_duration_s)
                    summary["phase"] = phase.name
                    summary["concurrency"] = phase.concurrency
                    summary["requests"] = len(phase.prompts)
                    if io_delta is not None:
                        summary["block_device"] = args.phase_io_device
                        summary["io_delta"] = io_delta
                    phase_summaries.append(summary)
                    if phase.include_in_overall:
                        overall_rows.extend(phase_rows)

            overall_summary = summarize_phase(overall_rows, sum(p["duration_s"] for p in phase_summaries if p["phase"] != "warmup"))
            overall_summary["phase"] = "overall"
            overall_summary["scenario"] = args.scenario
            if args.scenario == "eviction_replay":
                overall_summary["eviction_replay_signal"] = analyze_eviction_signal(phase_summaries)

            summary_payload = {
                "run_id": run_id,
                "created_utc": now_utc_iso(),
                "model_id": model_id,
                "scenario": args.scenario,
                "stream": bool(args.stream),
                "phase_summaries": phase_summaries,
                "overall_summary": overall_summary,
                "notes": [
                    "TTFT proxy is available only when --stream is enabled and the server emits stream chunks.",
                    "Without true streaming support, focus on end-to-end latency and throughput.",
                ],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    finally:
        if telemetry_manager is not None:
            telemetry_stop_report = telemetry_manager.stop_default()
        telemetry_payload = {
            "started": asdict(telemetry_start_report) if telemetry_start_report else None,
            "stopped": asdict(telemetry_stop_report) if telemetry_stop_report else None,
        }
        (run_dir / "telemetry" / "telemetry_report.json").write_text(
            json.dumps(telemetry_payload, indent=2),
            encoding="utf-8",
        )

    LOG.info("Benchmark complete. Artifacts: %s", run_dir)
    return run_dir


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
            phases.append(
                PhasePlan(name="warmup", prompts=warmup_prompts, concurrency=max(1, args.concurrency), include_in_overall=False)
            )
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

    # eviction_replay
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
        phases.append(
            PhasePlan(name="warmup", prompts=warmup_prompts, concurrency=max(1, args.concurrency), include_in_overall=False)
        )

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
    client: OpenAICompatClient,
    model_id: str,
    args: argparse.Namespace,
    estimator: TokenEstimator,
    request_counter: itertools.count,
    requests_fp,
    requests_lock: asyncio.Lock,
    responses_dir: Optional[Path],
) -> tuple[list[dict], float]:
    if not phase.prompts:
        return [], 0.0

    queue: asyncio.Queue[PromptSpec] = asyncio.Queue()
    for prompt in phase.prompts:
        queue.put_nowait(prompt)

    rows: list[dict] = []
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
            row = {
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


def summarize_phase(rows: list[dict], duration_s: float) -> dict:
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


def summarize_values(values: list[float]) -> dict:
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


def read_block_device_stats(device: str) -> Optional[dict]:
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


def diff_block_device_stats(before: Optional[dict], after: Optional[dict]) -> Optional[dict]:
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


def analyze_eviction_signal(phase_summaries: list[dict]) -> dict:
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
        interpretation = "Replay phase shows higher NVMe reads than initial warm phase; disk rehydrate is plausible."
        signal = "possible_readback"
    elif replay_read > 0.0:
        interpretation = "Replay phase has non-zero reads, but attribution to KV readback is weak."
        signal = "weak_read_signal"
    else:
        interpretation = (
            "No clear replay read spike. This can happen when reuse is served from in-memory tiers, "
            "eviction pressure was insufficient, or disk rehydrate is not active in this mode."
        )
        signal = "no_clear_readback"
    return {
        "signal": signal,
        "warm_read_mib": round(warm_read, 3),
        "pressure_write_mib": round(pressure_write, 3),
        "replay_read_mib": round(replay_read, 3),
        "interpretation": interpretation,
    }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        run_dir = asyncio.run(run_benchmark(args))
    except Exception:
        LOG.exception("Benchmark run failed.")
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

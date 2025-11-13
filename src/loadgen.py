#!/usr/bin/env python3
"""Async load generator for DGX Spark LLM Storage Test Harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import httpx

# -----------------------------------------------------------------------------
# Environment configuration

HARNESS_DIR = Path(os.environ.get("HARNESS_DIR", "/harness"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", HARNESS_DIR / "results"))
INPUTS_DIR = Path(os.environ.get("INPUTS_DIR", HARNESS_DIR / "inputs"))
MODEL_HANDLE = os.environ.get("MODEL_HANDLE", "openai/gpt-oss-120b")

# -----------------------------------------------------------------------------
# API configuration


@dataclass(frozen=True, slots=True)
class ApiConfig:
    """Connection details for a target inference API."""

    name: str
    url: str
    description: str
    extra_headers: dict[str, str] = field(default_factory=dict)


API_CONFIGS: dict[str, ApiConfig] = {
    "openai": ApiConfig(
        name="openai",
        url="http://127.0.0.1:8355/v1/completions",
        description="OpenAI-compatible (trtllm-serve, default for No-LoRA)",
    ),
    "triton": ApiConfig(
        name="triton",
        url="http://127.0.0.1:8000/v2/models/ensemble/generate_stream",
        description="NVIDIA Triton (required for LoRA support)",
        extra_headers={"Accept": "text/event-stream"},
    ),
}

# -----------------------------------------------------------------------------
# Small utilities


@dataclass(slots=True)
class RequestMetrics:
    """Latency measurements for a single inference request."""

    ttft_ms: Optional[float]
    latency_ms: Optional[float]
    success: bool


def percentile(values: Sequence[float], percent: float) -> Optional[float]:
    """Returns the percentile using linear interpolation (inclusive)."""
    cleaned = sorted(v for v in values if v is not None)
    if not cleaned:
        return None
    if percent <= 0:
        return cleaned[0]
    if percent >= 100:
        return cleaned[-1]
    rank = (len(cleaned) - 1) * (percent / 100.0)
    low = int(rank)
    high = min(low + 1, len(cleaned) - 1)
    fraction = rank - low
    if low == high:
        return cleaned[low]
    return cleaned[low] * (1.0 - fraction) + cleaned[high] * fraction


def bootstrap_percentile_ci(
    values: Sequence[float],
    percent: float = 99.0,
    iterations: int = 5000,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Bootstrap percentile confidence interval."""
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return (None, None, None)
    rng = random.Random(1234)  # deterministic for reproducibility
    samples: List[float] = []
    for _ in range(iterations):
        bucket = [cleaned[rng.randrange(len(cleaned))] for _ in cleaned]
        samples.append(percentile(bucket, percent) or 0.0)
    samples.sort()
    lo_index = int((alpha / 2.0) * iterations)
    hi_index = int((1 - alpha / 2.0) * iterations)
    return (
        percentile(cleaned, percent),
        samples[lo_index],
        samples[min(hi_index, len(samples) - 1)],
    )


def resolve_path(candidate: str, base_dir: Path) -> Path:
    """Resolves relative paths against a base directory."""
    path = Path(candidate)
    return path if path.is_absolute() else base_dir / path


def read_prompt_files(files: Iterable[str]) -> list[str]:
    """Reads prompts from provided files, falling back to a default prompt."""
    prompts: list[str] = []
    for prompt_file in files:
        resolved = resolve_path(prompt_file, INPUTS_DIR)
        if not resolved.exists():
            logging.warning("Prompt file not found: %s", resolved)
            continue
        prompts.append(resolved.read_text())
    if not prompts:
        logging.warning("No prompt files supplied; using default stub prompt.")
        prompts = ["Hello world."]
    return prompts


def load_adapters(lora_list: Optional[str]) -> list[str]:
    """Loads LoRA adapter names when the --lora-list flag is provided."""
    if not lora_list:
        return []
    resolved = resolve_path(lora_list, INPUTS_DIR)
    if not resolved.exists():
        raise FileNotFoundError(
            f"LoRA manifest not found at {resolved}. Provide an absolute path "
            "or a path relative to $INPUTS_DIR."
        )
    adapters = [
        line.strip()
        for line in resolved.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not adapters:
        raise ValueError(
            f"--lora-list specified but no adapters detected in {resolved}."
        )
    logging.info("Loaded %d LoRA adapters from %s.", len(adapters), resolved)
    return adapters


def build_payload(
    args: argparse.Namespace, prompt: str, adapter: Optional[str], api: ApiConfig
) -> dict:
    """Constructs the request payload for the selected API."""
    base_payload = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "stream": True,
    }
    if api.name == "triton":
        payload = {"prompt": prompt, **base_payload}
        if adapter:
            payload["lora_config"] = {"lora_name": adapter}
        return payload
    
    # OpenAI compatible mode (No LoRA)
    return {
        "model": MODEL_HANDLE,
        "prompt": prompt,
        **base_payload,
    }


async def stream_request(
    client: httpx.AsyncClient,
    payload: dict,
    api: ApiConfig,
    timeout: httpx.Timeout,
) -> RequestMetrics:
    """Executes one streaming inference request."""
    start = time.perf_counter()
    ttft_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    success = False
    headers = {"Content-Type": "application/json", **api.extra_headers}
    try:
        async with client.stream(
            "POST", api.url, json=payload, timeout=timeout, headers=headers
        ) as response:
            async for _chunk in response.aiter_raw():
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000.0
            latency_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            success = True
    except Exception as exc:  # pragma: no cover - network errors
        logging.error("Request failed: %s", exc)
    return RequestMetrics(ttft_ms=ttft_ms, latency_ms=latency_ms, success=success)


async def worker(
    *,
    args: argparse.Namespace,
    prompts: Sequence[str],
    adapters: Sequence[str],
    api: ApiConfig,
) -> list[RequestMetrics]:
    """Continuously issues requests until the duration deadline is reached."""
    timeout = httpx.Timeout(args.timeout)
    deadline = time.monotonic() + args.duration
    rng = random.Random()
    collected: list[RequestMetrics] = []

    # --- H5 Sticky-mode logic ---
    # If adapters are present and session is sticky, pick *one* for this worker.
    sticky_adapter: Optional[str] = None
    if adapters and args.lora_session == "sticky":
        sticky_adapter = rng.choice(adapters)

    async with httpx.AsyncClient(timeout=timeout) as client:
        while time.monotonic() < deadline:
            prompt = rng.choice(prompts)
            
            # Select adapter for *this* request
            chosen_adapter: Optional[str] = None
            if adapters:
                if args.lora_session == "sticky":
                    chosen_adapter = sticky_adapter
                else:  # 'random' (Stormy)
                    chosen_adapter = rng.choice(adapters)
            
            # If no adapters were loaded (e.g. H0, H2, H4), chosen_adapter remains None.
            # This is correct for the OpenAI API path.

            payload = build_payload(args, prompt, chosen_adapter, api)
            collected.append(
                await stream_request(client, payload, api=api, timeout=timeout)
            )
    return collected


def summarise_results(
    results: Sequence[RequestMetrics], duration: int
) -> dict[str, object]:
    """Converts raw request metrics into the summary JSON document."""
    ttfts = [r.ttft_ms for r in results if r.ttft_ms is not None]
    latencies = [r.latency_ms for r in results if r.success and r.latency_ms]
    total_reqs = len(results)
    success_reqs = len(latencies)
    admission_pct = 100.0 * (success_reqs / max(1, total_reqs))
    throughput_rps = success_reqs / max(1.0, duration)
    p99, ci_lo, ci_hi = bootstrap_percentile_ci(latencies, percent=99.0)
    return {
        "requests_total": total_reqs,
        "admission_pct": admission_pct,
        "throughput_rps": throughput_rps,
        "ttft_ms": {
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "p99": percentile(ttfts, 99),
        },
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p95": percentile(latencies, 95),
            "p99": p99,
            "p99_ci_low": ci_lo,
            "p99_ci_high": ci_hi,
        },
        "avg": {
            # These are backfilled by backfill_summary.py
            "io_wait_pct": None,
            "qu_sz": None,
            "rps": throughput_rps,
            "gpu_util_pct": None,
            "r_await_ms": None,
            "rps_storage": None,
        },
    }


def persist_summary(run_id: str, summary: dict[str, object], api: ApiConfig) -> Path:
    """Writes the summary JSON file to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}_summary.json"
    payload = {
        **summary,
        "meta": {
            "run_id": run_id,
            "api_mode": api.name,
            "target_url": api.url,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def resolve_api_mode(args: argparse.Namespace, adapters: list[str]) -> ApiConfig:
    """Determines which API endpoint should be used for this run."""
    forced_mode = args.api_mode
    if forced_mode not in {"auto", *API_CONFIGS.keys()}:
        raise ValueError(f"Unsupported --api-mode value: {forced_mode}")

    if adapters and forced_mode == "openai":
        raise ValueError("--lora-list requires Triton. Remove --api-mode openai or the adapter list.")

    selected = forced_mode
    if forced_mode == "auto":
        selected = "triton" if adapters else "openai"
    elif forced_mode == "triton" and not adapters:
        logging.info("Forcing Triton API without LoRA adapters (baseline run on LoRA server).")

    return API_CONFIGS[selected]


async def run_loadgen(args: argparse.Namespace) -> None:
    """Entrypoint for the async load generator."""
    logging.info("Starting loadgen for run %s.", args.run_id)
    adapters = load_adapters(args.lora_list)
    api = resolve_api_mode(args, adapters)
    logging.info("API Mode: %s (%s)", api.name.upper(), api.url)
    if adapters:
        logging.info("LoRA Mode: %s sessions", args.lora_session.upper())
        
    prompts = read_prompt_files(args.prompt_file or [])
    logging.info(
        "Running %s users for %ss with %d prompt(s).",
        args.users,
        args.duration,
        len(prompts),
    )
    worker_tasks = [
        worker(args=args, prompts=prompts, adapters=adapters, api=api)
        for _ in range(args.users)
    ]
    results_nested = await asyncio.gather(*worker_tasks)
    all_results = [metric for worker_results in results_nested for metric in worker_results]
    logging.info("Completed load generation: %d samples collected.", len(all_results))
    summary = summarise_results(all_results, args.duration)
    out_path = persist_summary(args.run_id, summary, api)
    logging.info("Wrote summary: %s", out_path)


def build_parser() -> argparse.ArgumentParser:
    """Creates the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique identifier for this run.")
    parser.add_argument(
        "-U",
        "--users",
        type=int,
        required=True,
        help="Number of concurrent workers (users).",
    )
    parser.add_argument(
        "-P",
        "--prompt-file",
        action="append",
        default=[],
        help="Prompt file (relative to $INPUTS_DIR unless absolute). Can repeat.",
    )
    parser.add_argument(
        "--lora-list",
        default=None,
        help="Path to lora_list.txt; requires Triton API mode.",
    )
    # --- NEW ARGUMENT FOR H5 ---
    parser.add_argument(
        "--lora-session",
        choices=["random", "sticky"],
        default="random",
        help="LoRA session type. 'random' = stormy (default), 'sticky' = affinity.",
    )
    # ---
    parser.add_argument(
        "--api-mode",
        choices=["auto", "openai", "triton"],
        default="auto",
        help="Override API selection. Use 'triton' to hit the LoRA server without adapters.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Test duration in seconds.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed passed through to the API.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds.",
    )
    return parser


def configure_logging(verbose: bool = False) -> None:
    """Initialises root logger once for the module."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI wrapper to parse args and execute the async entrypoint."""
    parser = build_parser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        asyncio.run(run_loadgen(args))
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

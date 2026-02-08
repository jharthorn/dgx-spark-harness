#!/usr/bin/env python3
"""Load generator (Test_Plan_v3.3 Sections 5-7) with real HTTP for Stack A/B."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import string
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import httpx

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

SRC_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = Path(__file__).resolve().parent.parent / "inputs"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from workloads import fixed_context, sessioned_chat  # noqa: E402
from workloads.sessioned_chat import SessionTurn  # noqa: E402

METRICS_FILENAME = "metrics.jsonl"
NONCE_ALPHABET = string.ascii_uppercase + string.digits
NONCE_LEN = 12

LOG = logging.getLogger("loadgen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DGX Spark v3 LoadGen")
    parser.add_argument("--config", help="YAML run config (stack/model/workload/etc.)")
    parser.add_argument("--run-id", default=None, help="Optional run identifier")
    parser.add_argument("--endpoint", required=False)
    parser.add_argument("--stack", required=False)
    parser.add_argument("--model", required=False)
    parser.add_argument("--workload", default=None, choices=["fixed_context", "mixed_context", "sessioned_chat"])
    parser.add_argument("--context_tokens", type=int, required=False)
    parser.add_argument("--concurrency", type=int, required=False)
    parser.add_argument("--mix_short_pct", type=float, default=None, help="Weight for short contexts (mixed_context)")
    parser.add_argument("--mix_medium_pct", type=float, default=None, help="Weight for medium contexts (mixed_context)")
    parser.add_argument("--mix_long_pct", type=float, default=None, help="Weight for long contexts (mixed_context)")
    parser.add_argument("--mix_short_min", type=int, default=None, help="Min tokens for short bucket (mixed_context)")
    parser.add_argument("--mix_short_max", type=int, default=None, help="Max tokens for short bucket (mixed_context)")
    parser.add_argument("--mix_medium_min", type=int, default=None, help="Min tokens for medium bucket (mixed_context)")
    parser.add_argument("--mix_medium_max", type=int, default=None, help="Max tokens for medium bucket (mixed_context)")
    parser.add_argument("--mix_long_min", type=int, default=None, help="Min tokens for long bucket (mixed_context)")
    parser.add_argument("--mix_long_max", type=int, default=None, help="Max tokens for long bucket (mixed_context)")
    parser.add_argument(
        "--burstiness",
        choices=["even", "bursty"],
        default=None,
        help="Request spacing pattern (evenly spaced vs short bursts)",
    )
    parser.add_argument("--burst_pause_s", type=float, default=None, help="Average idle gap before a burst when burstiness=bursty")
    parser.add_argument(
        "--duration_s",
        type=int,
        default=None,
        help="Run duration seconds; required unless provided via --config",
    )
    parser.add_argument("--max_input_len", type=int, default=None, help="Token-level guard (truncate prompt to this many tokens)")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer name_or_path; required with --max_input_len for accurate truncation")
    parser.add_argument("--input_len_margin", type=int, default=64, help="Safety margin tokens below max_input_len to avoid BOS/extra tokens overflow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nonce_per_user", action="store_true")
    parser.add_argument("--output-dir", default=".", help="Directory for metrics.jsonl")
    parser.add_argument("--prompt-file", default=None, help="Optional prompt file path")
    parser.add_argument("--lora_adapter_count", type=int, default=None, help="Number of LoRA adapters to cycle (generates adapter_### names)")
    parser.add_argument("--lora_adapter_list", default=None, help="Optional path to adapter_id list (one per line)")
    parser.add_argument(
        "--lora_churn_mode",
        default="round_robin",
        choices=["round_robin", "random", "hot_cold"],
        help="Adapter selection mode",
    )
    parser.add_argument("--lora_hot_ratio", type=float, default=0.2, help="Fraction of adapters considered hot (hot_cold mode)")
    parser.add_argument("--lora_hot_prob", type=float, default=0.7, help="Probability of choosing a hot adapter (hot_cold mode)")
    parser.add_argument("--session_phase", choices=["auto", "build", "resume"], default=None, help="Session workload phase control")
    parser.add_argument("--session_min_turns", type=int, default=None, help="Minimum build turns per session")
    parser.add_argument("--session_max_turns", type=int, default=None, help="Maximum build turns per session")
    parser.add_argument("--session_resume_turns", type=int, default=None, help="Follow-up turns emitted after idle")
    parser.add_argument("--session_idle_s", type=float, default=None, help="Idle gap before resume turn (sessioned_chat)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def load_config(path: str | None) -> Dict[str, object]:
    if not path:
        return {}
    if yaml is None:
        raise ImportError("pyyaml is required to load config files")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_config(args: argparse.Namespace, config: Dict[str, object]) -> argparse.Namespace:
    merged = dict(config)
    merged.update({k: v for k, v in vars(args).items() if v is not None})
    required = ["endpoint", "stack", "model", "workload", "context_tokens", "concurrency"]
    missing = [key for key in required if merged.get(key) is None]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")
    merged.setdefault("mix_short_pct", 0.6)
    merged.setdefault("mix_medium_pct", 0.3)
    merged.setdefault("mix_long_pct", 0.1)
    merged.setdefault("mix_short_min", 200)
    merged.setdefault("mix_short_max", 500)
    merged.setdefault("mix_medium_min", 800)
    merged.setdefault("mix_medium_max", 1500)
    merged.setdefault("mix_long_min", 2500)
    merged.setdefault("mix_long_max", 3200)
    merged.setdefault("burstiness", "even")
    merged.setdefault("burst_pause_s", 1.0)
    merged.setdefault("session_phase", "auto")
    merged.setdefault("session_min_turns", 3)
    merged.setdefault("session_max_turns", 10)
    merged.setdefault("session_resume_turns", 1)
    merged.setdefault("session_idle_s", 20.0)
    # Map generic phase -> session_phase when not explicitly set so H9 runner can drive build/resume.
    if merged.get("session_phase") == "auto" and merged.get("phase") in ("build", "resume"):
        merged["session_phase"] = merged["phase"]
    args.__dict__.update(merged)
    return args


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_prompt_file(path: Optional[str], length_tokens: int, seed: int) -> str:
    if path:
        resolved = Path(path)
    else:
        resolved = INPUTS_DIR / "prompts" / f"{length_tokens}_tokens.txt"
    if resolved.exists():
        return resolved.read_text()
    rng = random.Random(seed)
    approx_chars = max(length_tokens, 1) * 4
    return "".join(rng.choice(string.ascii_lowercase + " ") for _ in range(approx_chars))


def build_workload(name: str, tokens: int, seed: int, prompt_text: str, args: argparse.Namespace) -> Iterable[str]:
    if name == "fixed_context":
        return fixed_context.prompt_stream(tokens, seed, prompt_text)
    if name == "mixed_context":
        mix = fixed_context.buckets_from_args(args)
        if not getattr(args, "_mix_logged", False):
            LOG.info(
                "Mixed-context workload: %s | burstiness=%s (pause≈%ss)",
                fixed_context.format_mix_summary(mix),
                getattr(args, "burstiness", "even"),
                getattr(args, "burst_pause_s", 1.0),
            )
            args._mix_logged = True
        return fixed_context.mixed_prompt_stream(
            seed=seed,
            base_prompt=prompt_text,
            mix=mix,
            burstiness=getattr(args, "burstiness", "even"),
            burst_pause_s=float(getattr(args, "burst_pause_s", 1.0)),
        )
    if name == "sessioned_chat":
        return sessioned_chat.prompt_stream(
            tokens,
            seed,
            prompt_text,
            session_phase=getattr(args, "session_phase", "auto"),
            session_min_turns=int(getattr(args, "session_min_turns", 3)),
            session_max_turns=int(getattr(args, "session_max_turns", 10)),
            session_resume_turns=int(getattr(args, "session_resume_turns", 1)),
            session_idle_s=float(getattr(args, "session_idle_s", 20.0)),
        )
    raise NotImplementedError(f"Workload '{name}' not implemented yet")


def build_tokenizer(tokenizer_ref: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("transformers is required when --tokenizer/--max_input_len is set") from exc
    return AutoTokenizer.from_pretrained(tokenizer_ref, use_fast=True)


def truncate_prompt(prompt: str, tokenizer, max_input_len: int, margin: int) -> str:
    target = max_input_len - max(margin, 0)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) <= target:
        return prompt
    trimmed = tokens[-target:]
    new_prompt = tokenizer.decode(trimmed, skip_special_tokens=True)
    LOG.info("Truncated prompt from %s tokens to %s tokens (max_input_len=%s, margin=%s)", len(tokens), len(trimmed), max_input_len, margin)
    return new_prompt


def _nonce(rng: random.Random, user_id: int) -> str:
    tag = "".join(rng.choice(NONCE_ALPHABET) for _ in range(NONCE_LEN))
    return f"[NONCE:{user_id}:{tag}]"


async def send_request(
    client: httpx.AsyncClient,
    endpoint: str,
    prompt: str,
    model: str,
    adapter_id: str | None,
    session_id: str | None = None,
) -> tuple[float, float, int]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    }
    if adapter_id:
        payload["adapter_id"] = adapter_id
    if session_id:
        payload["session_id"] = session_id
    start = time.perf_counter()
    ttft: Optional[float] = None
    try:
        async with client.stream("POST", endpoint, json=payload, timeout=httpx.Timeout(120)) as resp:
            async for _chunk in resp.aiter_raw():
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            e2e = (time.perf_counter() - start) * 1000
            return ttft or e2e, e2e, 0
    except httpx.HTTPStatusError as exc:  # pragma: no cover - network errors
        elapsed = (time.perf_counter() - start) * 1000
        body = ""
        try:
            if exc.response is not None:
                await exc.response.aread()
                body = exc.response.text
        except Exception as body_exc:  # pragma: no cover - best-effort body logging
            LOG.debug("Failed to read error body: %s", body_exc)
        status = exc.response.status_code if exc.response else "?"
        LOG.error("Request failed: status=%s detail=%s", status, body[:512])
        return ttft or elapsed, elapsed, 1
    except Exception as exc:  # pragma: no cover - network errors
        LOG.error("Request failed: %s", exc)
        elapsed = (time.perf_counter() - start) * 1000
        return ttft or elapsed, elapsed, 1


def build_adapter_pool(args: argparse.Namespace) -> list[str]:
    adapters: list[str] = []
    if args.lora_adapter_list:
        path = Path(args.lora_adapter_list)
        if path.exists():
            for line in path.read_text().splitlines():
                name = line.strip()
                if name:
                    adapters.append(name)
    if args.lora_adapter_count:
        count = max(0, int(args.lora_adapter_count))
        if count > 0 and not adapters:
            adapters = [f"adapter_{idx:03d}" for idx in range(1, count + 1)]
        elif count > 0 and adapters:
            adapters = adapters[:count]
    return adapters


def make_adapter_selector(args: argparse.Namespace, adapters: list[str]):
    if not adapters:
        return lambda rng, user_idx: None

    mode = args.lora_churn_mode
    hot_count = max(1, int(len(adapters) * max(0.0, min(args.lora_hot_ratio, 1.0))))
    hot_pool = adapters[:hot_count]
    cursor = {"idx": 0}

    def choose(rng: random.Random, user_idx: int) -> str:
        if mode == "round_robin":
            val = adapters[cursor["idx"] % len(adapters)]
            cursor["idx"] += 1
            return val
        if mode == "hot_cold":
            if rng.random() < args.lora_hot_prob:
                return rng.choice(hot_pool)
            return rng.choice(adapters)
        # default random
        return rng.choice(adapters)

    return choose


async def worker(
    user_id: int,
    args: argparse.Namespace,
    prompt_iter: Iterator[str],
    rng: random.Random,
    metrics_path: Path,
    client: httpx.AsyncClient,
    adapter_selector,
) -> None:
    end_time = time.monotonic() + int(args.duration_s)
    with metrics_path.open("a", encoding="utf-8") as sink:
        while time.monotonic() < end_time:
            turn = next(prompt_iter)
            session_id: Optional[str] = None
            turn_idx: Optional[int] = None
            turns_in_session: Optional[int] = None
            is_resume: Optional[bool] = None
            sleep_before = 0.0
            context_tokens = int(args.context_tokens)
            context_bucket: Optional[str] = None
            if isinstance(turn, SessionTurn):
                session_id = turn.session_id
                prompt = turn.prompt
                turn_idx = turn.turn_idx
                turns_in_session = turn.turns_in_session
                is_resume = turn.is_resume
                sleep_before = max(0.0, float(turn.sleep_before_s))
            elif isinstance(turn, fixed_context.PromptEvent):
                prompt = turn.prompt
                sleep_before = max(0.0, float(turn.sleep_before_s))
                context_tokens = int(turn.context_tokens)
                context_bucket = turn.bucket
            else:
                prompt = str(turn)

            if sleep_before > 0:
                await asyncio.sleep(min(sleep_before, max(0.0, end_time - time.monotonic())))
            if args.nonce_per_user:
                prompt = f"{prompt}\n{_nonce(rng, user_id)}"
            if getattr(args, "tokenizer_obj", None) is not None and args.max_input_len is not None:
                prompt = truncate_prompt(prompt, args.tokenizer_obj, int(args.max_input_len), int(args.input_len_margin))
            adapter_id = adapter_selector(rng, user_id)
            if adapter_id:
                prompt = f"[ADAPTER:{adapter_id}] {prompt}"
            ttft, e2e, rc = await send_request(client, args.endpoint, prompt, args.model, adapter_id, session_id)
            record = {
                "run_id": args.run_id,
                "ts": time.time(),
                "stack": args.stack,
                "model": args.model,
                "workload": args.workload,
                "context_tokens": context_tokens,
                "concurrency": int(args.concurrency),
                "lat_ttft_ms": ttft,
                "lat_e2e_ms": e2e,
                "rc": rc,
            }
            if context_bucket:
                record["context_bucket"] = context_bucket
            if adapter_id:
                record["adapter_id"] = adapter_id
            if session_id:
                record["session_id"] = session_id
            if turn_idx is not None:
                record["turn_idx"] = turn_idx
            if turns_in_session is not None:
                record["turns_in_session"] = turns_in_session
            if is_resume is not None:
                record["resume"] = bool(is_resume)
            if getattr(args, "phase", None):
                record["phase"] = getattr(args, "phase")
            if getattr(args, "session_phase", None):
                record["session_phase"] = getattr(args, "session_phase")
            sink.write(json.dumps(record) + "\n")


async def run_loadgen(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    prompt_text = load_prompt_file(args.prompt_file, int(args.context_tokens), args.seed)

    tokenizer = None
    if args.max_input_len is not None:
        if args.tokenizer is None:
            raise ValueError("--max_input_len requires --tokenizer to compute token counts accurately")
        tokenizer = build_tokenizer(args.tokenizer)
        prompt_text = truncate_prompt(prompt_text, tokenizer, int(args.max_input_len), int(args.input_len_margin))
    args.tokenizer_obj = tokenizer
    user_generators: list[Iterator[str]] = []
    for user_idx in range(int(args.concurrency)):
        user_generators.append(build_workload(args.workload, int(args.context_tokens), args.seed + user_idx, prompt_text, args))

    adapters = build_adapter_pool(args)
    adapter_selector = make_adapter_selector(args, adapters)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / METRICS_FILENAME
    args.run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{args.stack}_{args.model}"

    LOG.info("Starting loadgen run_id=%s stack=%s model=%s users=%s ctx=%s", args.run_id, args.stack, args.model, args.concurrency, args.context_tokens)
    async with httpx.AsyncClient() as client:
        tasks = [
            worker(idx, args, user_generators[idx], random.Random(args.seed + idx), metrics_path, client, adapter_selector)
            for idx in range(int(args.concurrency))
        ]
        await asyncio.gather(*tasks)
    LOG.info("Loadgen complete: metrics -> %s", metrics_path)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    cfg = load_config(args.config)
    args = apply_config(args, cfg)
    asyncio.run(run_loadgen(args))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

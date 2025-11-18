#!/usr/bin/env python3
"""Load generator (Test_Plan_v3.0 Sections 5-7) with real HTTP for Stack A."""

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
    parser.add_argument("--workload", default="fixed_context", choices=["fixed_context", "sessioned_chat"])
    parser.add_argument("--context_tokens", type=int, required=False)
    parser.add_argument("--concurrency", type=int, required=False)
    parser.add_argument("--duration_s", type=int, default=60)
    parser.add_argument("--max_input_len", type=int, default=None, help="Token-level guard (truncate prompt to this many tokens)")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer name_or_path; required with --max_input_len for accurate truncation")
    parser.add_argument("--input_len_margin", type=int, default=64, help="Safety margin tokens below max_input_len to avoid BOS/extra tokens overflow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nonce_per_user", action="store_true")
    parser.add_argument("--output-dir", default=".", help="Directory for metrics.jsonl")
    parser.add_argument("--prompt-file", default=None, help="Optional prompt file path")
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


def build_workload(name: str, tokens: int, seed: int, prompt_text: str) -> Iterable[str]:
    if name == "fixed_context":
        return fixed_context.prompt_stream(tokens, seed, prompt_text)
    if name == "sessioned_chat":
        return sessioned_chat.prompt_stream(tokens, seed, prompt_text)
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


async def send_request(client: httpx.AsyncClient, endpoint: str, prompt: str, model: str) -> tuple[float, float, str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    }
    start = time.perf_counter()
    ttft: Optional[float] = None
    try:
        async with client.stream("POST", endpoint, json=payload, timeout=httpx.Timeout(120)) as resp:
            async for _chunk in resp.aiter_raw():
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            e2e = (time.perf_counter() - start) * 1000
            return ttft or e2e, e2e, "ok"
    except Exception as exc:  # pragma: no cover - network errors
        LOG.error("Request failed: %s", exc)
        elapsed = (time.perf_counter() - start) * 1000
        return ttft or elapsed, elapsed, "error"


async def worker(user_id: int, args: argparse.Namespace, prompt_iter: Iterator[str], rng: random.Random, metrics_path: Path, client: httpx.AsyncClient) -> None:
    end_time = time.monotonic() + int(args.duration_s)
    with metrics_path.open("a", encoding="utf-8") as sink:
        while time.monotonic() < end_time:
            prompt = next(prompt_iter)
            if args.nonce_per_user:
                prompt = f"{prompt}\n{_nonce(rng, user_id)}"
            if getattr(args, "tokenizer_obj", None) is not None and args.max_input_len is not None:
                prompt = truncate_prompt(prompt, args.tokenizer_obj, int(args.max_input_len), int(args.input_len_margin))
            ttft, e2e, rc = await send_request(client, args.endpoint, prompt, args.model)
            record = {
                "run_id": args.run_id,
                "ts": time.time(),
                "stack": args.stack,
                "model": args.model,
                "workload": args.workload,
                "context_tokens": int(args.context_tokens),
                "concurrency": int(args.concurrency),
                "lat_ttft_ms": ttft,
                "lat_e2e_ms": e2e,
                "rc": rc,
            }
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
        user_generators.append(build_workload(args.workload, int(args.context_tokens), args.seed + user_idx, prompt_text))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / METRICS_FILENAME
    args.run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{args.stack}_{args.model}"

    LOG.info("Starting loadgen run_id=%s stack=%s model=%s users=%s ctx=%s", args.run_id, args.stack, args.model, args.concurrency, args.context_tokens)
    async with httpx.AsyncClient() as client:
        tasks = [
            worker(idx, args, user_generators[idx], random.Random(args.seed + idx), metrics_path, client)
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

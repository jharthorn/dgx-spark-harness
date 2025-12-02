"""Fixed-context and mixed-context workloads (Test_Plan_v3.3 Section 5.1)."""

from __future__ import annotations

from dataclasses import dataclass
import random
import string
from typing import Iterator, Sequence

ALPHABET = string.ascii_lowercase + " "
CHARS_PER_TOKEN = 4  # rough estimate until tokenizer integration


@dataclass
class PromptEvent:
    prompt: str
    context_tokens: int
    sleep_before_s: float = 0.0
    bucket: str | None = None


@dataclass
class MixBucket:
    name: str
    min_tokens: int
    max_tokens: int
    weight: float


DEFAULT_MIX: tuple[MixBucket, ...] = (
    MixBucket("short", 200, 500, 0.6),
    MixBucket("medium", 800, 1500, 0.3),
    MixBucket("long", 2500, 3200, 0.1),
)


def prompt_stream(token_length: int, seed: int = 0, base_prompt: str | None = None) -> Iterator[str]:
    """Yield prompts approximating the requested token length."""
    rng = random.Random(seed)
    approx_chars = max(token_length, 1) * CHARS_PER_TOKEN
    while True:
        if base_prompt:
            yield base_prompt
            continue
        payload = "".join(rng.choice(ALPHABET) for _ in range(approx_chars)).strip()
        yield f"[CTX:{token_length}] {payload}"


def _render_payload(token_length: int, rng: random.Random, base_prompt: str | None) -> str:
    if base_prompt:
        return base_prompt
    approx_chars = max(token_length, 1) * CHARS_PER_TOKEN
    payload = "".join(rng.choice(ALPHABET) for _ in range(approx_chars)).strip()
    return f"[CTX:{token_length}] {payload}"


def _normalize_mix(mix: Sequence[MixBucket] | None) -> list[MixBucket]:
    buckets = list(mix or DEFAULT_MIX)
    total = sum(max(bucket.weight, 0.0) for bucket in buckets)
    if total <= 0:
        return list(DEFAULT_MIX)
    return [MixBucket(bucket.name, bucket.min_tokens, bucket.max_tokens, bucket.weight / total) for bucket in buckets]


def format_mix_summary(mix: Sequence[MixBucket]) -> str:
    parts = []
    for bucket in mix:
        parts.append(f"{bucket.name} {bucket.weight * 100:.0f}% ({bucket.min_tokens}-{bucket.max_tokens})")
    return ", ".join(parts)


def mixed_prompt_stream(
    seed: int = 0,
    base_prompt: str | None = None,
    mix: Sequence[MixBucket] | None = None,
    burstiness: str = "even",
    burst_pause_s: float = 1.0,
) -> Iterator[PromptEvent]:
    """Yield prompts with a configurable context-length mix and simple burstiness."""

    rng = random.Random(seed)
    buckets = _normalize_mix(mix)

    def choose_bucket(sample: float) -> MixBucket:
        cursor = 0.0
        for bucket in buckets:
            cursor += bucket.weight
            if sample <= cursor:
                return bucket
        return buckets[-1]

    burst_remaining = 0
    burstiness = burstiness or "even"
    burst_pause_s = max(0.0, burst_pause_s)

    while True:
        bucket = choose_bucket(rng.random())
        token_length = rng.randint(bucket.min_tokens, bucket.max_tokens)
        sleep_before = 0.0
        if burstiness == "bursty":
            if burst_remaining <= 0:
                burst_remaining = rng.randint(3, 7)
                sleep_before = rng.uniform(burst_pause_s * 0.5, burst_pause_s * 1.5) if burst_pause_s > 0 else 0.0
            burst_remaining -= 1
        payload = _render_payload(token_length, rng, base_prompt)
        yield PromptEvent(prompt=payload, context_tokens=token_length, sleep_before_s=sleep_before, bucket=bucket.name)


def buckets_from_args(args: object) -> list[MixBucket]:
    def _val(name: str, default: int) -> int:
        return int(getattr(args, name, default) or default)

    mix_short_pct = float(getattr(args, "mix_short_pct", DEFAULT_MIX[0].weight))
    mix_medium_pct = float(getattr(args, "mix_medium_pct", DEFAULT_MIX[1].weight))
    mix_long_pct = float(getattr(args, "mix_long_pct", DEFAULT_MIX[2].weight))
    buckets: list[MixBucket] = [
        MixBucket("short", _val("mix_short_min", DEFAULT_MIX[0].min_tokens), _val("mix_short_max", DEFAULT_MIX[0].max_tokens), mix_short_pct),
        MixBucket("medium", _val("mix_medium_min", DEFAULT_MIX[1].min_tokens), _val("mix_medium_max", DEFAULT_MIX[1].max_tokens), mix_medium_pct),
        MixBucket("long", _val("mix_long_min", DEFAULT_MIX[2].min_tokens), _val("mix_long_max", DEFAULT_MIX[2].max_tokens), mix_long_pct),
    ]
    return _normalize_mix(buckets)

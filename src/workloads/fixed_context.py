"""Fixed-context workload scaffolding (Test_Plan_v3.0.md Section 5.1)."""

from __future__ import annotations

import random
import string
from typing import Iterator

ALPHABET = string.ascii_lowercase + " "
CHARS_PER_TOKEN = 4  # rough estimate until tokenizer integration


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

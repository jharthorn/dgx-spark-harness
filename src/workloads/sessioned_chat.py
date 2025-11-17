"""Sessioned chat workload placeholder per Test_Plan_v3.0.md Section 5.2."""

from __future__ import annotations

from typing import Iterator


def prompt_stream(token_length: int, seed: int = 0, base_prompt: str | None = None) -> Iterator[str]:
    """Yield simple chat-style prompts (stub)."""
    # TODO: Implement multi-turn chat stateful prompts.
    prompt = base_prompt or f"[CHAT_CTX:{token_length}] user: hello\nassistant:"
    while True:
        yield prompt

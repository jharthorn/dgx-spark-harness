"""Sessioned chat workload for H5/H9 multi-turn + re-hydration."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator, Iterable


@dataclass
class SessionTurn:
    session_id: str
    prompt: str
    turn_idx: int
    turns_in_session: int
    is_resume: bool
    sleep_before_s: float = 0.0


def _chunk_text(text: str, size: int) -> Iterable[str]:
    for idx in range(0, len(text), max(size, 1)):
        yield text[idx : idx + size]


def _render_prompt(session_id: str, history: list[tuple[str, str]], next_user_msg: str, turns_in_session: int) -> str:
    lines = [f"[SESSION_ID:{session_id}] multi-turn context build (turns={turns_in_session})"]
    for idx, (user_msg, assistant_msg) in enumerate(history, start=1):
        lines.append(f"User {idx}: {user_msg}")
        lines.append(f"Assistant {idx}: {assistant_msg}")
    lines.append(f"User {len(history) + 1}: {next_user_msg}")
    lines.append(f"Assistant {len(history) + 1}:")
    return "\n".join(lines)


def prompt_stream(
    token_length: int,
    seed: int = 0,
    base_prompt: str | None = None,
    session_phase: str = "auto",
    session_min_turns: int = 3,
    session_max_turns: int = 10,
    session_resume_turns: int = 1,
    session_idle_s: float = 20.0,
) -> Iterator[SessionTurn]:
    """Yield stateful chat prompts with stable session ids.

    session_phase:
      - build: emit only the build-up turns (for Phase 1).
      - resume: emit only the follow-up turns after an idle (Phase 3).
      - auto: emit build + idle gap + resume in one stream.
    Assumptions are intentionally documented here so the runner can stay simple:
      * The session id is deterministic (seed + session counter) so Phase 3 can reuse it.
      * Context grows by appending a new long-form user chunk every build turn.
      * Resume turns are short prompts that rely on the cached KV instead of replaying the full history.
    """

    rng = random.Random(seed)
    base_text = base_prompt or f"[CHAT_CTX:{token_length}] seed={seed}"
    # Split the base text into moderately sized chunks; we reuse them to grow the context.
    chunk_size = max(len(base_text) // max(session_min_turns, 1), 256)
    chunk_cycle = list(_chunk_text(base_text, chunk_size)) or [base_text]

    session_idx = 0
    while True:
        session_idx += 1
        session_id = f"s{seed}-u{session_idx:04d}"
        build_turns = max(session_min_turns, min(session_max_turns, rng.randint(session_min_turns, session_max_turns)))
        resume_turns = max(1, session_resume_turns)
        turns_in_session = build_turns + resume_turns
        history: list[tuple[str, str]] = []

        build_events: list[SessionTurn] = []
        for turn in range(build_turns):
            chunk = chunk_cycle[(turn + session_idx) % len(chunk_cycle)]
            user_msg = (
                f"Please weave this chunk into the shared thread (chunk {turn + 1}/{build_turns}):\n"
                f"{chunk}\nKeep replies concise; we are stress testing context length."
            )
            prompt = _render_prompt(session_id, history, user_msg, turns_in_session)
            history.append((user_msg, f"Noted chunk {turn + 1}, continuing the session state."))
            build_events.append(
                SessionTurn(
                    session_id=session_id,
                    prompt=prompt,
                    turn_idx=turn + 1,
                    turns_in_session=turns_in_session,
                    is_resume=False,
                )
            )

        idle_delay = max(session_idle_s, 0.0)
        resume_events: list[SessionTurn] = []
        for resume_idx in range(resume_turns):
            follow_up = (
                "Resume after the idle pause. Provide a brief status update and the next step "
                f"(follow-up {resume_idx + 1}/{resume_turns})."
            )
            prompt = "\n".join(
                [
                    f"[SESSION_ID:{session_id}] resume turn",
                    f"Prior turns cached: {len(history)}",
                    f"User {len(history) + 1}: {follow_up}",
                    f"Assistant {len(history) + 1}:",
                ]
            )
            resume_events.append(
                SessionTurn(
                    session_id=session_id,
                    prompt=prompt,
                    turn_idx=build_turns + resume_idx + 1,
                    turns_in_session=turns_in_session,
                    is_resume=True,
                    sleep_before_s=idle_delay if resume_idx == 0 and session_phase == "auto" else 0.0,
                )
            )

        if session_phase in ("build", "auto"):
            for ev in build_events:
                yield ev
        if session_phase == "auto":
            for ev in resume_events:
                yield ev
        if session_phase == "resume":
            for ev in resume_events:
                yield ev

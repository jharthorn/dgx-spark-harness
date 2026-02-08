"""Deterministic prompt generation for OpenAI-compatible `/v1/completions` benchmarks."""

from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

LOG = logging.getLogger(__name__)

LLAMA3_USER_PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT_PREFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

SHORT_TOKEN_CHOICES: tuple[int, ...] = (512, 768, 1024, 1536, 2048)
LONG_TOKEN_CHOICES: tuple[int, ...] = (8192, 12288, 16384, 24576, 32768)

_TOPICS: tuple[str, ...] = (
    "request scheduling",
    "token budget policy",
    "storage queue behavior",
    "cache eviction workflow",
    "fault tolerance and retries",
    "tenant isolation controls",
    "latency observability",
    "service degradation playbook",
    "throughput stabilization plan",
    "maintenance and incident response",
)

_SUBJECTS: tuple[str, ...] = (
    "The runtime coordinator",
    "The storage planner",
    "The batch scheduler",
    "The policy engine",
    "The telemetry worker",
    "The frontend router",
    "The cache reconciler",
    "The replay evaluator",
    "The token allocator",
    "The audit process",
)

_ACTIONS: tuple[str, ...] = (
    "records",
    "re-evaluates",
    "normalizes",
    "aggregates",
    "describes",
    "compares",
    "escalates",
    "bounds",
    "prioritizes",
    "documents",
)

_QUALIFIERS: tuple[str, ...] = (
    "under bursty traffic",
    "during long-context prompts",
    "for mixed tenant load",
    "while preserving deterministic replay",
    "with explicit timeout guards",
    "while reducing tail latency spikes",
    "without changing request semantics",
    "under constrained memory tiers",
    "before and after eviction pressure",
    "for compatibility-mode file I/O",
)

_FALLBACK_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    prompt_set: str
    target_tokens: int
    prompt_tokens_est: int
    prompt: str


class TokenEstimator:
    """Estimate tokens; optionally use a local tokenizer when available."""

    def __init__(self, tokenizer: str = "auto") -> None:
        self._tokenizer = None
        self.tokenizer_name: Optional[str] = None
        if tokenizer != "heuristic":
            for candidate in self._candidate_tokenizers(tokenizer):
                tok = self._try_load_tokenizer(candidate)
                if tok is not None:
                    self._tokenizer = tok
                    self.tokenizer_name = candidate
                    LOG.info("Using tokenizer for prompt sizing: %s", candidate)
                    break
        if self._tokenizer is None:
            LOG.info("Falling back to heuristic token estimation.")

    @property
    def using_tokenizer(self) -> bool:
        return self._tokenizer is not None

    def estimate(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return len(_FALLBACK_TOKEN_RE.findall(text))

    def tune_to_target(self, text: str, target_tokens: int, tolerance_tokens: int = 64) -> tuple[str, int]:
        if target_tokens <= 0:
            return text, self.estimate(text)
        if self._tokenizer is not None:
            return self._tune_with_tokenizer(text, target_tokens, tolerance_tokens)
        return self._tune_heuristic(text, target_tokens, tolerance_tokens)

    def _candidate_tokenizers(self, tokenizer: str) -> list[str]:
        if tokenizer not in ("", "auto", None):
            return [tokenizer]
        candidates: list[str] = []
        for env_var in ("BENCH_TOKENIZER", "MODEL_HANDLE"):
            value = os.environ.get(env_var)
            if value:
                candidates.append(value)
        hf_root = Path("/root/.cache/huggingface/hub")
        try:
            if hf_root.exists():
                snapshots = sorted(hf_root.glob("models--*/snapshots/*"))
                if snapshots:
                    candidates.append(str(snapshots[-1]))
        except PermissionError:
            LOG.debug("Skipping unreadable tokenizer cache path: %s", hf_root)
        candidates.extend(
            [
                "nvidia/Llama-3.1-8B-Instruct-FP8",
                "meta-llama/Llama-3.1-8B-Instruct",
            ]
        )
        seen: set[str] = set()
        unique: list[str] = []
        for c in candidates:
            if c not in seen:
                unique.append(c)
                seen.add(c)
        return unique

    def _try_load_tokenizer(self, candidate: str):
        try:
            from transformers import AutoTokenizer
        except Exception:
            LOG.debug("transformers is unavailable; skipping tokenizer load.")
            return None
        try:
            kwargs = {"use_fast": True, "trust_remote_code": True}
            if not Path(candidate).exists():
                kwargs["local_files_only"] = True
            tok = AutoTokenizer.from_pretrained(candidate, **kwargs)
            _ = tok.encode("tokenizer_probe", add_special_tokens=False)
            return tok
        except Exception as exc:
            LOG.debug("Tokenizer load failed for %s: %s", candidate, exc)
            return None

    def _tune_with_tokenizer(self, text: str, target: int, tolerance: int) -> tuple[str, int]:
        tok = self._tokenizer
        assert tok is not None
        token_ids = tok.encode(text, add_special_tokens=False)
        if len(token_ids) > target + tolerance:
            token_ids = token_ids[:target]
            tuned = tok.decode(token_ids, skip_special_tokens=False)
            return tuned, len(token_ids)
        filler = (
            " deterministic checkpoint replay baseline cache latency throughput "
            "scheduler queue storage observability policy appendix"
        )
        loops = 0
        tuned = text
        while len(token_ids) < target and loops < 64:
            tuned = f"{tuned}{filler}"
            token_ids = tok.encode(tuned, add_special_tokens=False)
            loops += 1
        if len(token_ids) > target + tolerance:
            token_ids = token_ids[:target]
            tuned = tok.decode(token_ids, skip_special_tokens=False)
        return tuned, len(token_ids)

    def _tune_heuristic(self, text: str, target: int, tolerance: int) -> tuple[str, int]:
        tuned = text
        est = self.estimate(tuned)
        filler_words = "deterministic cache replay appendix checkpoint scheduler storage latency".split()

        if est < target:
            missing = target - est
            repeats = max(1, (missing // len(filler_words)) + 2)
            tuned = f"{tuned} {' '.join((filler_words * repeats)[:missing])}"
            est = self.estimate(tuned)

        if est <= target + tolerance:
            return tuned, est

        words = tuned.split()
        low = 1
        high = len(words)
        best_text = tuned
        best_est = est
        while low <= high:
            mid = (low + high) // 2
            candidate = " ".join(words[:mid])
            cand_est = self.estimate(candidate)
            if cand_est > target + tolerance:
                high = mid - 1
            else:
                best_text = candidate
                best_est = cand_est
                low = mid + 1

        if best_est < target:
            current_words = best_text.split()
            idx = 0
            while best_est < target:
                current_words.append(filler_words[idx % len(filler_words)])
                idx += 1
                best_text = " ".join(current_words)
                best_est = self.estimate(best_text)
            if best_est > target + tolerance:
                best_text = " ".join(current_words[:-1])
                best_est = self.estimate(best_text)

        return best_text, best_est


def render_llama3_completion_prompt(user_text: str) -> str:
    return f"{LLAMA3_USER_PREFIX}{user_text}{LLAMA3_ASSISTANT_PREFIX}"


def generate_prompt_set(
    prompt_set: str,
    count: int,
    seed: int,
    estimator: TokenEstimator,
    short_range: tuple[int, int] = (512, 2048),
    long_range: tuple[int, int] = (8192, 32768),
    prompt_id_prefix: Optional[str] = None,
) -> list[PromptSpec]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    specs: list[PromptSpec] = []
    prefix = prompt_id_prefix or prompt_set
    for i in range(count):
        target = _choose_target(prompt_set, rng, short_range=short_range, long_range=long_range)
        topic = rng.choice(_TOPICS)
        payload = _build_payload(target_tokens=target, rng=rng, topic=topic, doc_id=f"{prefix}-{i:05d}")
        prompt_text, est_tokens = estimator.tune_to_target(
            render_llama3_completion_prompt(payload),
            target_tokens=target,
        )
        specs.append(
            PromptSpec(
                prompt_id=f"{prefix}-{i:05d}",
                prompt_set=prompt_set,
                target_tokens=target,
                prompt_tokens_est=est_tokens,
                prompt=prompt_text,
            )
        )
    return specs


def generate_replay_sets(
    a_count: int,
    b_count: int,
    seed: int,
    estimator: TokenEstimator,
    long_range: tuple[int, int] = (8192, 32768),
) -> tuple[list[PromptSpec], list[PromptSpec]]:
    a_base = generate_prompt_set(
        prompt_set="long",
        count=a_count,
        seed=seed,
        estimator=estimator,
        long_range=long_range,
        prompt_id_prefix="A",
    )
    b_base = generate_prompt_set(
        prompt_set="long",
        count=b_count,
        seed=seed + 100_003,
        estimator=estimator,
        long_range=long_range,
        prompt_id_prefix="B",
    )
    a = [replace(p, prompt_set="replay_a") for p in a_base]
    b = [replace(p, prompt_set="replay_b") for p in b_base]
    return a, b


def manifest_rows(specs: Sequence[PromptSpec]) -> Iterable[dict]:
    for spec in specs:
        yield {
            "prompt_id": spec.prompt_id,
            "prompt_set": spec.prompt_set,
            "target_tokens": spec.target_tokens,
            "prompt_tokens_est": spec.prompt_tokens_est,
            "prompt_chars": len(spec.prompt),
            "prompt_sha256": _sha256_hex(spec.prompt),
        }


def _choose_target(
    prompt_set: str,
    rng: random.Random,
    short_range: tuple[int, int],
    long_range: tuple[int, int],
) -> int:
    short_min, short_max = short_range
    long_min, long_max = long_range
    if prompt_set == "short":
        return _sample_target(rng, SHORT_TOKEN_CHOICES, short_min, short_max)
    if prompt_set == "long":
        return _sample_target(rng, LONG_TOKEN_CHOICES, long_min, long_max)
    if prompt_set == "mixed":
        if rng.random() < 0.5:
            return _sample_target(rng, SHORT_TOKEN_CHOICES, short_min, short_max)
        return _sample_target(rng, LONG_TOKEN_CHOICES, long_min, long_max)
    raise ValueError(f"Unsupported prompt_set={prompt_set}")


def _sample_target(rng: random.Random, defaults: Sequence[int], min_tokens: int, max_tokens: int) -> int:
    in_range = [v for v in defaults if min_tokens <= v <= max_tokens]
    if in_range:
        return rng.choice(in_range)
    if min_tokens == max_tokens:
        return min_tokens
    return rng.randint(min_tokens, max_tokens)


def _build_payload(target_tokens: int, rng: random.Random, topic: str, doc_id: str) -> str:
    # Heuristic: long English text is typically around 1.2-1.4 tokens per word.
    target_words = max(220, int(target_tokens * 0.66))
    lines: list[str] = [
        f"Document ID: {doc_id}",
        f"Topic: {topic}",
        "Instruction: summarize key controls, constraints, and ordered actions in clear prose.",
        "",
    ]
    words_used = sum(len(line.split()) for line in lines)
    section = 1
    while words_used < target_words:
        sec_lines = _render_section(section, topic, rng)
        lines.extend(sec_lines)
        words_used += sum(len(line.split()) for line in sec_lines)
        section += 1
    return "\n".join(lines)


def _render_section(section: int, topic: str, rng: random.Random) -> list[str]:
    title = f"Section {section}: {topic} operating notes"
    lines = [title]
    bullets = 7 if section % 2 == 0 else 6
    for idx in range(1, bullets + 1):
        subj = rng.choice(_SUBJECTS)
        action = rng.choice(_ACTIONS)
        qualifier = rng.choice(_QUALIFIERS)
        lines.append(
            f"{section}.{idx} {subj} {action} the {topic} workflow {qualifier}, "
            f"tracks queue depth and service latency, and emits deterministic replay markers."
        )
    lines.append(
        f"Checklist {section}: verify ordering, compare deltas, reconcile counters, and persist trace evidence."
    )
    lines.append("")
    return lines


def _sha256_hex(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()

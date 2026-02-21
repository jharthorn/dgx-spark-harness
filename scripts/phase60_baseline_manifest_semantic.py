#!/usr/bin/env python3
"""Semantic baseline-manifest hashing and policy helpers for Phase60.

This module intentionally hashes only semantic benchmark configuration fields
so timestamp/path/container drift does not invalidate comparability checks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NON_SEMANTIC_KEYS = {
    "created_utc",
    "created_at",
    "timestamp",
    "timestamp_utc",
    "ts",
    "run_id",
    "run_path",
    "results_root",
    "cwd",
    "hostname",
    "container_id",
    "git_dirty",
    "build_id",
}


def parse_bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _drop_nones(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, raw in value.items():
            reduced = _drop_nones(raw)
            if reduced is None:
                continue
            out[key] = reduced
        return out
    if isinstance(value, list):
        out_list = []
        for item in value:
            reduced = _drop_nones(item)
            if reduced is None:
                continue
            out_list.append(reduced)
        return out_list
    return value


def normalize_manifest_for_semantic_hash(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize a manifest by removing known non-semantic and unstable fields."""

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key in sorted(value.keys()):
                if key in NON_SEMANTIC_KEYS:
                    continue
                if key.endswith("_path") or key.endswith("_dir"):
                    continue
                if key in {"path", "paths"}:
                    continue
                child = walk(value[key])
                if child is None:
                    continue
                out[key] = child
            return out
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, float):
            # Canonical numeric handling for deterministic JSON hashing.
            return float(f"{value:.12g}")
        return value

    normalized = walk(copy.deepcopy(manifest))
    return _drop_nones(normalized)


def semantic_hash(manifest_norm: dict[str, Any]) -> str:
    blob = json.dumps(manifest_norm, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_semantic_manifest(run_dir: Path, context: dict[str, Any]) -> dict[str, Any]:
    config = _load_json(run_dir / "config.json")
    args = config.get("args") if isinstance(config.get("args"), dict) else {}
    kv_mode = config.get("kv_mode") if isinstance(config.get("kv_mode"), dict) else {}
    diag = kv_mode.get("diagnostic") if isinstance(kv_mode.get("diagnostic"), dict) else {}

    analysis_manifest = _load_json(run_dir.parent / "analysis" / "manifest.json")
    runtime_manifest = _load_json(run_dir.parent / "analysis" / "worker_runtime_manifest.json")
    runtime_env = runtime_manifest.get("env") if isinstance(runtime_manifest.get("env"), dict) else {}

    phases = config.get("phases") if isinstance(config.get("phases"), list) else []
    phase_layout: list[dict[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase_layout.append(
            {
                "name": phase.get("name"),
                "concurrency": _to_int_or_none(phase.get("concurrency")),
                "requests": _to_int_or_none(phase.get("requests")),
            }
        )

    # Keep these fields explicit so changes are auditable and reproducible.
    semantic: dict[str, Any] = {
        "schema_version": 1,
        "phase_family": "phase60_rehydrate_minimal_sweep",
        "scenario": config.get("scenario") or args.get("scenario") or context.get("scenario"),
        "model_profile": context.get("model_profile"),
        "model_id": config.get("model_id") or analysis_manifest.get("model_handle"),
        "tier_mode": config.get("tier_mode") or args.get("tier_mode"),
        "kv_mode": {
            "mode": kv_mode.get("mode") or args.get("kv_mode"),
            "cpu_cache_gb": (
                _to_float_or_none(args.get("kv_cpu_cache_gb"))
                if _to_float_or_none(args.get("kv_cpu_cache_gb")) is not None
                else _to_float_or_none(kv_mode.get("cpu_cache_gb"))
            ),
            "disk_cache_gb": (
                _to_float_or_none(args.get("kv_disk_cache_gb"))
                if _to_float_or_none(args.get("kv_disk_cache_gb")) is not None
                else _to_float_or_none(kv_mode.get("disk_cache_gb"))
            ),
            "diagnostic_disable_disk_offload_filter": parse_bool(
                args.get("diagnostic_disable_disk_offload_filter")
                if "diagnostic_disable_disk_offload_filter" in args
                else diag.get("disable_disk_offload_filter", False)
            ),
            "runtime_env": {
                "DYN_KVBM_CPU_CACHE_GB": runtime_env.get("DYN_KVBM_CPU_CACHE_GB"),
                "DYN_KVBM_DISK_CACHE_GB": runtime_env.get("DYN_KVBM_DISK_CACHE_GB"),
                "DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER": runtime_env.get("DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER"),
            },
        },
        "workload_shape": {
            "rehydrate_populate_sessions": _to_int_or_none(args.get("rehydrate_populate_sessions")),
            "rehydrate_thrash_sessions": _to_int_or_none(args.get("rehydrate_thrash_sessions")),
            "rehydrate_turns": _to_int_or_none(args.get("rehydrate_turns")),
            "rehydrate_prefix_target_tokens": _to_int_or_none(args.get("rehydrate_prefix_target_tokens")),
            "rehydrate_populate_concurrency": _to_int_or_none(args.get("rehydrate_populate_concurrency")),
            "rehydrate_thrash_concurrency": _to_int_or_none(args.get("rehydrate_thrash_concurrency")),
            "rehydrate_replay_concurrency": _to_int_or_none(args.get("rehydrate_replay_concurrency")),
            "rehydrate_replay_repeats": _to_int_or_none(args.get("rehydrate_replay_repeats")),
            "rehydrate_gen_tokens": _to_int_or_none(args.get("rehydrate_gen_tokens")),
            "seed": _to_int_or_none(args.get("seed")),
            "request_seed": _to_int_or_none(args.get("request_seed")),
        },
        "phase_layout": phase_layout,
        "sweep_policy": {
            "sweep_replay_concurrencies": sorted(
                [_to_int_or_none(x) for x in (context.get("sweep_replay_concurrencies") or []) if _to_int_or_none(x) is not None]
            ),
            "baseline_replay_concurrency": _to_int_or_none(context.get("baseline_replay_concurrency")),
            "pressure_populate_concurrency": _to_int_or_none(context.get("pressure_populate_concurrency")),
            "pressure_thrash_concurrency": _to_int_or_none(context.get("pressure_thrash_concurrency")),
            "include_b0": parse_bool(context.get("include_b0")),
            "run_order_per_concurrency": context.get("run_order_per_concurrency"),
            "b1_disk_tier_enforced": parse_bool(context.get("b1_disk_tier_enforced", True)),
            "b1_disk_cache_gb": _to_float_or_none(context.get("b1_disk_cache_gb")),
            "b1_read_bytes_threshold": _to_int_or_none(context.get("b1_read_bytes_threshold")),
            "b1_disk_hit_rate_threshold": _to_float_or_none(context.get("b1_disk_hit_rate_threshold")),
        },
        "gates": {
            "require_b2_rehydrate": parse_bool(context.get("require_b2_rehydrate", True)),
            "io_attrib_enabled": parse_bool(context.get("io_attrib_enabled", False)),
        },
        "metric_policy": {
            "preferred": context.get("metric_preferred") or "ttfc_ms",
            "fallback": "ttft_ms",
        },
    }
    return normalize_manifest_for_semantic_hash(semantic)


def build_semantic_payload(run_dir: Path, context: dict[str, Any]) -> dict[str, Any]:
    manifest = build_semantic_manifest(run_dir, context)
    return {
        "semantic_hash": semantic_hash(manifest),
        "semantic_manifest": manifest,
        "run_path": str(run_dir),
        "manifest_path": str(run_dir / "config.json"),
    }


def diff_manifest_fields(left: Any, right: Any, prefix: str = "") -> list[str]:
    if left == right:
        return []
    out: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left.keys()) | set(right.keys()))
        for key in keys:
            sub_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(diff_manifest_fields(left.get(key), right.get(key), sub_prefix))
        return out
    if isinstance(left, list) and isinstance(right, list):
        max_len = max(len(left), len(right))
        for idx in range(max_len):
            li = left[idx] if idx < len(left) else None
            ri = right[idx] if idx < len(right) else None
            sub_prefix = f"{prefix}[{idx}]"
            out.extend(diff_manifest_fields(li, ri, sub_prefix))
        return out
    out.append(prefix or "<root>")
    return out


@dataclass(frozen=True)
class BaselineDecision:
    action: str
    should_stop: bool
    reason_code: str | None
    warning_code: str | None
    known_hash: str
    current_hash: str
    diff_fields: list[str]


def evaluate_baseline_hash_mismatch(
    *,
    known_payload: dict[str, Any],
    current_payload: dict[str, Any],
    strict: bool,
    accept_new: bool,
) -> BaselineDecision:
    known_hash = str(known_payload.get("semantic_hash") or "")
    current_hash = str(current_payload.get("semantic_hash") or "")
    known_manifest = known_payload.get("semantic_manifest") if isinstance(known_payload.get("semantic_manifest"), dict) else {}
    current_manifest = current_payload.get("semantic_manifest") if isinstance(current_payload.get("semantic_manifest"), dict) else {}
    diffs = diff_manifest_fields(known_manifest, current_manifest)

    if known_hash == current_hash:
        return BaselineDecision(
            action="match",
            should_stop=False,
            reason_code=None,
            warning_code=None,
            known_hash=known_hash,
            current_hash=current_hash,
            diff_fields=diffs,
        )
    if accept_new:
        return BaselineDecision(
            action="accept",
            should_stop=False,
            reason_code=None,
            warning_code="BASELINE_MANIFEST_HASH_ACCEPTED",
            known_hash=known_hash,
            current_hash=current_hash,
            diff_fields=diffs,
        )
    if strict:
        return BaselineDecision(
            action="stop",
            should_stop=True,
            reason_code="baseline_manifest_hash_mismatch",
            warning_code=None,
            known_hash=known_hash,
            current_hash=current_hash,
            diff_fields=diffs,
        )
    return BaselineDecision(
        action="warn",
        should_stop=False,
        reason_code=None,
        warning_code="BASELINE_MANIFEST_HASH_MISMATCH_WARNING",
        known_hash=known_hash,
        current_hash=current_hash,
        diff_fields=diffs,
    )


def accept_new_baseline(
    *,
    baseline_file: Path,
    audit_jsonl: Path,
    known_payload: dict[str, Any],
    current_payload: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    audit_jsonl.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    known_hash = str(known_payload.get("semantic_hash") or "")
    current_hash = str(current_payload.get("semantic_hash") or "")
    known_manifest = known_payload.get("semantic_manifest") if isinstance(known_payload.get("semantic_manifest"), dict) else {}
    current_manifest = current_payload.get("semantic_manifest") if isinstance(current_payload.get("semantic_manifest"), dict) else {}
    changed_fields = diff_manifest_fields(known_manifest, current_manifest)

    baseline_record = {
        "updated_utc": now,
        "semantic_hash": current_hash,
        "source_run_path": current_payload.get("run_path"),
        "manifest_path": current_payload.get("manifest_path"),
        "semantic_manifest": current_manifest,
    }
    baseline_file.write_text(json.dumps(baseline_record, indent=2) + "\n", encoding="utf-8")

    audit_entry = {
        "timestamp_utc": now,
        "action": "accept_new_baseline_manifest",
        "reason": reason,
        "previous_semantic_hash": known_hash,
        "new_semantic_hash": current_hash,
        "baseline_file": str(baseline_file),
        "known_source_run_path": known_payload.get("run_path") or known_payload.get("source_run_path"),
        "new_source_run_path": current_payload.get("run_path"),
        "changed_fields": changed_fields,
        "semantic_manifest_summary": {
            "scenario": (current_manifest.get("scenario") if isinstance(current_manifest, dict) else None),
            "tier_mode": ((current_manifest.get("tier_mode")) if isinstance(current_manifest, dict) else None),
            "kv_mode": (((current_manifest.get("kv_mode") or {}).get("mode")) if isinstance(current_manifest, dict) else None),
            "model_profile": ((current_manifest.get("model_profile")) if isinstance(current_manifest, dict) else None),
            "sweep_replay_concurrencies": (
                ((current_manifest.get("sweep_policy") or {}).get("sweep_replay_concurrencies"))
                if isinstance(current_manifest, dict)
                else None
            ),
        },
    }
    with audit_jsonl.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(audit_entry, separators=(",", ":"), ensure_ascii=True) + "\n")

    return {
        "baseline_file": str(baseline_file),
        "audit_jsonl": str(audit_jsonl),
        "previous_semantic_hash": known_hash,
        "new_semantic_hash": current_hash,
        "changed_fields": changed_fields,
    }


def _cmd_compute(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    context = json.loads(args.context_json) if args.context_json else {}
    payload = build_semantic_payload(run_dir, context if isinstance(context, dict) else {})
    print(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
    return 0


def _cmd_decide(args: argparse.Namespace) -> int:
    known = json.loads(args.known_json)
    current = json.loads(args.current_json)
    decision = evaluate_baseline_hash_mismatch(
        known_payload=known if isinstance(known, dict) else {},
        current_payload=current if isinstance(current, dict) else {},
        strict=parse_bool(args.strict),
        accept_new=parse_bool(args.accept_new),
    )
    print(
        json.dumps(
            {
                "action": decision.action,
                "should_stop": decision.should_stop,
                "reason_code": decision.reason_code,
                "warning_code": decision.warning_code,
                "known_hash": decision.known_hash,
                "current_hash": decision.current_hash,
                "diff_fields": decision.diff_fields,
            },
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


def _cmd_accept(args: argparse.Namespace) -> int:
    known = json.loads(args.known_json)
    current = json.loads(args.current_json)
    result = accept_new_baseline(
        baseline_file=Path(args.baseline_file),
        audit_jsonl=Path(args.audit_jsonl),
        known_payload=known if isinstance(known, dict) else {},
        current_payload=current if isinstance(current, dict) else {},
        reason=args.reason,
    )
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    compute = sub.add_parser("compute", help="Compute semantic manifest/hash payload for a run dir.")
    compute.add_argument("--run-dir", required=True, help="Run directory containing config.json")
    compute.add_argument("--context-json", default="{}", help="JSON context with sweep policy fields")
    compute.set_defaults(func=_cmd_compute)

    decide = sub.add_parser("decide", help="Evaluate mismatch policy for known vs current semantic payload.")
    decide.add_argument("--known-json", required=True, help="Known semantic payload JSON")
    decide.add_argument("--current-json", required=True, help="Current semantic payload JSON")
    decide.add_argument("--strict", default="0", help="Strict mode boolean")
    decide.add_argument("--accept-new", default="0", help="Accept-new-baseline boolean")
    decide.set_defaults(func=_cmd_decide)

    accept = sub.add_parser("accept", help="Persist new known-good semantic hash and append audit entry.")
    accept.add_argument("--baseline-file", required=True, help="Known-good baseline semantic hash file")
    accept.add_argument("--audit-jsonl", required=True, help="Append-only baseline audit JSONL")
    accept.add_argument("--known-json", required=True, help="Previous known semantic payload JSON")
    accept.add_argument("--current-json", required=True, help="Current semantic payload JSON")
    accept.add_argument("--reason", default="operator_accept_new_baseline", help="Audit reason string")
    accept.set_defaults(func=_cmd_accept)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

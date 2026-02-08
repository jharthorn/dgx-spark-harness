#!/usr/bin/env python3
"""Emit export lines for Stack B tier env vars based on a config YAML."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency should exist via requirements
    raise SystemExit("PyYAML is required; install with pip install pyyaml") from exc


UNIT_RE = re.compile(r"(?i)^(\d+(?:\.\d+)?)([a-z]+)?$")
UNIT_SCALE = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "mib": 1024**2,
    "gb": 1024**3,
    "gib": 1024**3,
    "tb": 1024**4,
    "tib": 1024**4,
}
PROFILE_CONFIGS = {
    "comfy": "stackB_llama70b_dynamo_tiered.yaml",
    "spill": "stackB_llama70b_dynamo_tiered_spill.yaml",
    "stress": "stackB_llama70b_dynamo_tiered_stress.yaml",
}


def parse_bytes(val: Any, field: str) -> int:
    """Parse an integer or string with units into bytes."""
    if isinstance(val, (int, float)):
        return int(val)
    if not isinstance(val, str):
        raise ValueError(f"{field}: expected int or string with units, got {type(val)}")
    match = UNIT_RE.match(val.strip())
    if not match:
        raise ValueError(f"{field}: could not parse value '{val}'")
    qty, unit = match.groups()
    unit = (unit or "b").lower()
    scale = UNIT_SCALE.get(unit)
    if scale is None:
        raise ValueError(f"{field}: unsupported unit '{unit}' in '{val}'")
    return int(float(qty) * scale)


def load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Config root must be a mapping")
    return data


def parse_int_field(val: Any, field: str) -> int | None:
    """Parse an int-like field; returns None if unset."""
    if val is None:
        return None
    if isinstance(val, bool):
        raise ValueError(f"{field}: expected integer, got bool")
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str) and val.strip().isdigit():
        return int(val.strip())
    raise ValueError(f"{field}: expected integer, got {type(val)}")


def resolve_config(profile: str | None, config_arg: str | None) -> tuple[Path, str | None]:
    """Resolve the config path from profile or explicit path."""
    repo_root = Path(__file__).resolve().parent.parent
    configs_dir = repo_root / "configs"

    if config_arg:
        cfg_path = Path(config_arg).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        profile_name = next((p for p, name in PROFILE_CONFIGS.items() if name == cfg_path.name), None)
        return cfg_path, profile_name

    default_profile = profile or "comfy"
    cfg_name = PROFILE_CONFIGS.get(default_profile)
    if cfg_name is None:
        raise SystemExit(f"Unknown profile '{default_profile}' (expected one of {', '.join(PROFILE_CONFIGS)})")
    return (configs_dir / cfg_name).resolve(), default_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit tier env exports from a Stack B YAML")
    parser.add_argument("--config", help="Path to stackB_*_dynamo_tiered.yaml")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CONFIGS),
        help="Profile name to map to a stock stackB_llama70b_dynamo_tiered_* config",
    )
    args = parser.parse_args()

    cfg_path, profile = resolve_config(args.profile, args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    try:
        cfg = load_yaml(cfg_path)
        model_cfg = cfg.get("model", {})
        max_input_len = parse_int_field(model_cfg.get("max_input_len"), "model.max_input_len")
        max_seq_len = parse_int_field(model_cfg.get("max_seq_len"), "model.max_seq_len")
        max_num_tokens = parse_int_field(model_cfg.get("max_num_tokens"), "model.max_num_tokens")
        max_batch_size = parse_int_field(model_cfg.get("max_batch_size"), "model.max_batch_size")
        runtime = cfg.get("runtime", {})
        dynamo = runtime.get("dynamo", {})
        kv_block_size = dynamo.get("kv_block_size_bytes")
        tier0 = dynamo.get("tier0") or {}
        tier1 = dynamo.get("tier1") or {}
        tier2 = dynamo.get("tier2") or {}
        t0_bytes = parse_bytes(tier0.get("capacity_bytes"), "tier0.capacity_bytes")
        t1_bytes = parse_bytes(tier1.get("capacity_bytes"), "tier1.capacity_bytes")
        t2_bytes = parse_bytes(tier2.get("allocate_bytes"), "tier2.allocate_bytes")
        t2_path = tier2.get("path")
        if not t2_path:
            raise ValueError("tier2.path is required")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error parsing config: {exc}") from exc

    profile_name = profile or "custom"
    print(f"export STACKB_PROFILE={profile_name}")
    print(f"export STACKB_TIER_CONFIG={cfg_path}")
    print(f"export DYN_KVBM_TIER0_BYTES={t0_bytes}")
    print(f"export DYN_KVBM_TIER1_BYTES={t1_bytes}")
    print(f"export DYN_KVBM_TIER2_BYTES={t2_bytes}")
    print(f"export DYN_KVBM_TIER2_PATH={t2_path}")
    if kv_block_size is not None:
        print(f"export DYN_KVBM_KV_BLOCK_SIZE_BYTES={parse_bytes(kv_block_size, 'dynamo.kv_block_size_bytes')}")
    if max_input_len is not None:
        print(f"export STACKB_MAX_INPUT_LEN={max_input_len}")
    if max_seq_len is not None:
        print(f"export STACKB_MAX_SEQ_LEN={max_seq_len}")
    if max_num_tokens is not None:
        print(f"export STACKB_MAX_NUM_TOKENS={max_num_tokens}")
    if max_batch_size is not None:
        print(f"export STACKB_MAX_BATCH_SIZE={max_batch_size}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit tier env exports from a Stack B YAML")
    parser.add_argument("--config", required=True, help="Path to stackB_*_dynamo_tiered.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    try:
        cfg = load_yaml(cfg_path)
        runtime = cfg.get("runtime", {})
        dynamo = runtime.get("dynamo", {})
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

    print(f"export DYN_KVBM_TIER0_BYTES={t0_bytes}")
    print(f"export DYN_KVBM_TIER1_BYTES={t1_bytes}")
    print(f"export DYN_KVBM_TIER2_BYTES={t2_bytes}")
    print(f"export DYN_KVBM_TIER2_PATH={t2_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

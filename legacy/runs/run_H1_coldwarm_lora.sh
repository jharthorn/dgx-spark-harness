#!/usr/bin/env bash
# Hypothesis: H1 – LoRA cold/warm churn (shim)
# Typical profile: Spill (Stack B)
# Expected behavior: delegate to run_H1_lora_thrash.sh for Tier2-backed adapter churn.
# See docs/Test_Plan_v3.3.md, section H1.
set -euo pipefail

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
exec "$HARNESS_DIR/runs/run_H1_lora_thrash.sh" "$@"

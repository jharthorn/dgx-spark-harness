#!/usr/bin/env bash
# Backward-compatible shim to the v3.3 H1 runner.
set -euo pipefail

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
exec "$HARNESS_DIR/runs/run_H1_lora_thrash.sh" "$@"

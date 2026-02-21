#!/usr/bin/env bash
set -euo pipefail

# Canonical Phase70 paired-repeat runner entrypoint.
# BENCH_PHASE70_REPLAY_CONCURRENCY controls replay-phase concurrency for both legs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/bench_phase70_rehydrate_c1_pair_repeats.sh" "$@"

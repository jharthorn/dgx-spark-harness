#!/usr/bin/env bash
set -euo pipefail

# H2A UMA pressure (Test_Plan_v3.0 Section 8.2A, Stack A control)

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

ENDPOINT=${ENDPOINT:-${1:-http://stackA-baseline:8355/v1/completions}}
STACK="stackA"
MODEL="L70B"
WORKLOAD="fixed_context"
CONCURRENCY=${CONCURRENCY:-32}
DURATION=${DURATION:-180}
PROMPTS=(512 1024 2048 4096)
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}

for CTX in "${PROMPTS[@]}"; do
  RUN_ID="$(rt_ts)_H2A_${STACK}_${MODEL}_${CTX}"
  RUN_DIR="$RESULTS_BASE/${RUN_ID}"
  ensure_run_dir "$RUN_DIR"

  cat > "$RUN_DIR/config.yaml" <<EOF
stack: ${STACK}
model: ${MODEL}
workload: ${WORKLOAD}
context_tokens: ${CTX}
concurrency: ${CONCURRENCY}
duration_s: ${DURATION}
endpoint: ${ENDPOINT}
nonce_per_user: true
seed: 42
EOF

  start_sysmon "$RUN_DIR" "A"
  python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR" --endpoint "${ENDPOINT}"
  stop_sysmon "$RUN_DIR"
done

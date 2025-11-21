#!/usr/bin/env bash
set -euo pipefail

# H1 cold vs warm LoRA (Test_Plan_v3.0 Section 8.1, Stack A control)

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackA"
MODEL="L70B"
WORKLOAD="fixed_context"
CTX=256
CONCURRENCY=32
DURATION=120
ENDPOINT=${ENDPOINT:-http://127.0.0.1:8355/v1/completions}
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}

for PHASE in COLD WARM; do
  RUN_ID="$(rt_ts)_H1_${PHASE}_${STACK}_${MODEL}"
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
nonce_per_user: false
seed: 42
phase: ${PHASE}
EOF

  start_sysmon "$RUN_DIR" "A"
  python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR"
  stop_sysmon "$RUN_DIR"
done

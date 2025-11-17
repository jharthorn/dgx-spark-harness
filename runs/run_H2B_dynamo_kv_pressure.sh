#!/usr/bin/env bash
set -euo pipefail

# H2B Dynamo KV pressure (Test_Plan_v3.0 Section 8.2B, Stack B tiered)

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackB"
MODEL="L70B"
WORKLOAD="fixed_context"
CONCURRENCY=32
DURATION=180

PROMPTS=(1024 2048 4096 6144)
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}

for CTX in "${PROMPTS[@]}"; do
  RUN_ID="$(rt_ts)_H2B_${STACK}_${MODEL}_${CTX}"
  RUN_DIR="$RESULTS_BASE/${RUN_ID}"
  ensure_run_dir "$RUN_DIR"

  cat > "$RUN_DIR/config.yaml" <<EOF
stack: ${STACK}
model: ${MODEL}
workload: ${WORKLOAD}
context_tokens: ${CTX}
concurrency: ${CONCURRENCY}
duration_s: ${DURATION}
endpoint: http://stackB-dynamo:9000/v1/completions
nonce_per_user: true
seed: 42
EOF

  start_sysmon "$RUN_DIR" "B"
  start_dynkv "$RUN_DIR"
  python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR"
  stop_sysmon "$RUN_DIR"
  stop_dynkv "$RUN_DIR"
done

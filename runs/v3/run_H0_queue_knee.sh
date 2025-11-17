#!/usr/bin/env bash
set -euo pipefail

# H0 queue knee (Test_Plan_v3.0 Section 8.0, Stack A)

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$HARNESS_DIR/runs/v3/_lib_v3.sh"

ENDPOINT=${ENDPOINT:-${1:-http://stackA-baseline:8355/v1/completions}}
STACK="stackA"
MODEL="L70B"
WORKLOAD="fixed_context"
CTX=256
DURATION=${DURATION:-120}
CONCURRENCY_LIST=(8 16 32 64)

for U in "${CONCURRENCY_LIST[@]}"; do
  RUN_ID="$(rt_ts)_H0_${STACK}_${MODEL}_U${U}"
  RUN_DIR="$HARNESS_DIR/runs/v3/${RUN_ID}"
  ensure_run_dir "$RUN_DIR"

  cat > "$RUN_DIR/config.yaml" <<EOF
stack: ${STACK}
model: ${MODEL}
workload: ${WORKLOAD}
context_tokens: ${CTX}
concurrency: ${U}
duration_s: ${DURATION}
endpoint: ${ENDPOINT}
nonce_per_user: false
seed: 42
EOF

  start_sysmon "$RUN_DIR" "A"
  python3 "$HARNESS_DIR/src/loadgen_v3.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR" --endpoint "${ENDPOINT}"
  stop_sysmon "$RUN_DIR"
done

#!/usr/bin/env bash
set -euo pipefail

# H4B Dynamo storage QoS (Test_Plan_v3.0 Section 8.4B, Stack B)

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackB"
# Default to Llama 3.3 70B FP4; override MODEL env to switch.
MODEL=${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
WORKLOAD="fixed_context"
CTX=2048
CONCURRENCY=32
DURATION=180
ENDPOINT=${ENDPOINT:-http://127.0.0.1:9000/v1/completions}
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}

for PHASE in baseline qos_degraded; do
  RUN_ID="$(rt_ts)_H4B_${PHASE}_${STACK}_${MODEL}"
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
phase: ${PHASE}
EOF

  start_sysmon "$RUN_DIR" "B"
  start_telemetry "$RUN_DIR"
  # TODO: apply NVMe QoS controls for ${PHASE} when implemented.
  python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR"
  stop_sysmon "$RUN_DIR"
  stop_telemetry "$RUN_DIR"
done

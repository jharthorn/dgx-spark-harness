#!/usr/bin/env bash
# Hypothesis: H7 – KV telemetry cadence sweep on Stack B
# Typical profile: Stress
# Expected behavior: maximize tier2 fetches to validate telemetry cadence impact and overhead.
# See docs/Test_Plan_v3.3.md, section H7 (8.7).
set -euo pipefail

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackB"
PROFILE=${PROFILE:-stress}  # Stress profile to maximize tier2 fetches for telemetry cadence sweeps
apply_profile_env "$PROFILE"
# Default to Llama 3.3 70B NVFP4; override MODEL env to switch.
MODEL=${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
MODEL_TAG=${MODEL_TAG:-$(model_tag "$MODEL")}
WORKLOAD="fixed_context"
CTX=1024
CONCURRENCY=32
DURATION=120
ENDPOINT=${ENDPOINT:-http://127.0.0.1:9000/v1/completions}
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}

TELEMETRY_INTERVALS=(${TELEMETRY_INTERVALS:-0.1 0.2 0.5 1 5})

for INTERVAL in "${TELEMETRY_INTERVALS[@]}"; do
  RUN_ID="$(rt_ts)_H7_${PROFILE}_${STACK}_${MODEL_TAG}_${INTERVAL}s"
  RUN_DIR="$RESULTS_BASE/${RUN_ID}"
  ensure_run_dir "$RUN_DIR"

  cat > "$RUN_DIR/config.yaml" <<EOF
stack: ${STACK}
profile: ${PROFILE}
model: ${MODEL}
workload: ${WORKLOAD}
context_tokens: ${CTX}
concurrency: ${CONCURRENCY}
duration_s: ${DURATION}
endpoint: ${ENDPOINT}
nonce_per_user: true
seed: 42
EOF

  TELEMETRY_INTERVAL="$INTERVAL" start_sysmon "$RUN_DIR" "B"
  TELEMETRY_INTERVAL="$INTERVAL" start_telemetry "$RUN_DIR"
  TELEMETRY_INTERVAL="$INTERVAL" python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR"
  stop_sysmon "$RUN_DIR"
  stop_telemetry "$RUN_DIR"
done

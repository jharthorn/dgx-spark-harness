#!/usr/bin/env bash
# Hypothesis: H6 – Tier sizing and eviction policy on Stack B
# Typical profile: Spill
# Expected behavior: stress tier sizing and eviction policy under fixed-context load.
# See docs/Test_Plan_v3.3.md, section H6 (8.6).
set -euo pipefail

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackB"
PROFILE=${PROFILE:-spill}  # Spill profile to evaluate eviction policy under tier pressure
apply_profile_env "$PROFILE"
# Default to Llama 3.3 70B NVFP4; override MODEL env to switch.
MODEL=${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
MODEL_TAG=${MODEL_TAG:-$(model_tag "$MODEL")}
WORKLOAD="fixed_context"
CTX=1024
CONCURRENCY=32
DURATION=180
ENDPOINT=${ENDPOINT:-http://127.0.0.1:9000/v1/completions}

POLICIES=(fifo lru)
TIER0_BYTES_LIST=(${TIER0_BYTES_LIST:-$DYN_KVBM_TIER0_BYTES})
TIER1_BYTES_LIST=(${TIER1_BYTES_LIST:-$DYN_KVBM_TIER1_BYTES})
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}

for POLICY in "${POLICIES[@]}"; do
  for T0_BYTES in "${TIER0_BYTES_LIST[@]}"; do
    for T1_BYTES in "${TIER1_BYTES_LIST[@]}"; do
      export DYN_KVBM_TIER0_BYTES="$T0_BYTES"
      export DYN_KVBM_TIER1_BYTES="$T1_BYTES"
      RUN_ID="$(rt_ts)_H6_${PROFILE}_${STACK}_${MODEL_TAG}_${POLICY}_t0${T0_BYTES}_t1${T1_BYTES}"
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
eviction_policy: ${POLICY}
tier0_bytes: ${T0_BYTES}
tier1_bytes: ${T1_BYTES}
EOF

      start_sysmon "$RUN_DIR" "B"
      start_telemetry "$RUN_DIR"
      python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR"
      stop_sysmon "$RUN_DIR"
      stop_telemetry "$RUN_DIR"
    done
  done
done

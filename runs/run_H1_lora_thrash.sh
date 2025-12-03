#!/usr/bin/env bash
# Hypothesis: H1 – LoRA bandwidth/adapter churn ("Infinite LoRA")
# Typical profile: Spill (Stack B)
# Expected behavior: Tier2-backed adapters, nonce_per_user on; expect adapter churn bandwidth limits.
# See docs/Test_Plan_v3.3.md, section H1.
# Warmup: send a small ping (e.g., single POST) after the worker finishes attention workspace resize before starting full load.
set -euo pipefail

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackB"
PROFILE=${PROFILE:-spill}  # Spill profile to force tier2-backed adapters during churn
apply_profile_env "$PROFILE"

MODEL=${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
MODEL_TAG=${MODEL_TAG:-$(model_tag "$MODEL")}
WORKLOAD="fixed_context"
CTX=${CTX:-4096}
CONCURRENCY=${CONCURRENCY:-32}
DURATION=${DURATION:-180}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:9000/v1/completions}
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}
ADAPTER_COUNTS=(${ADAPTER_COUNTS:-100 200 300})
LORA_CHURN_MODE=${LORA_CHURN_MODE:-random}

for COUNT in "${ADAPTER_COUNTS[@]}"; do
  RUN_ID="$(rt_ts)_H1_${PROFILE}_${STACK}_${MODEL_TAG}_${COUNT}adpt"
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
lora_adapter_count: ${COUNT}
lora_churn_mode: ${LORA_CHURN_MODE}
EOF

  start_sysmon "$RUN_DIR" "B"
  start_telemetry "$RUN_DIR"
  python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR"
  stop_sysmon "$RUN_DIR"
  stop_telemetry "$RUN_DIR"
done

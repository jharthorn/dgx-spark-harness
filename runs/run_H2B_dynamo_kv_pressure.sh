#!/usr/bin/env bash
set -euo pipefail

# H2B Dynamo KV pressure (Test_Plan_v3.3 Section 8.2B, Stack B tiered)

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackB"
PROFILE=${PROFILE:-spill}  # Spill profile to drive steady tier2 IO for KV pressure
apply_profile_env "$PROFILE"
# Default to 70B dev path; set MODEL=nvidia/Llama-3.1-8B-Instruct-NVFP4 for the 8B sweep.
MODEL=${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
MODEL_TAG=${MODEL_TAG:-$(model_tag "$MODEL")}
WORKLOAD="fixed_context"
if [[ -z "${CONCURRENCY:-}" ]] && [[ -f "$HARNESS_DIR/runs/H0/uwork.txt" ]]; then
  CONCURRENCY="$(<"$HARNESS_DIR/runs/H0/uwork.txt")"
fi
CONCURRENCY=${CONCURRENCY:-32}
DURATION=${DURATION:-180}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:9000/v1/completions}

PROMPTS=(${PROMPTS_OVERRIDE:-256 512 1024 2048 4096})
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}
# Tokenizer-aware truncation (align with Stack B engine admit per profile).
TOKENIZER=${TOKENIZER:-$MODEL}
MAX_INPUT_LEN=${MAX_INPUT_LEN:-${STACKB_MAX_INPUT_LEN:-4096}}
INPUT_LEN_MARGIN=${INPUT_LEN_MARGIN:-64}

for CTX in "${PROMPTS[@]}"; do
  RUN_ID="$(rt_ts)_H2B_${STACK}_${MODEL_TAG}_${CTX}"
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
tokenizer: ${TOKENIZER}
max_input_len: ${MAX_INPUT_LEN}
input_len_margin: ${INPUT_LEN_MARGIN}
EOF

  echo "---- Run $RUN_ID ----"
  echo "  endpoint         : ${ENDPOINT}"
  echo "  context_tokens   : ${CTX}"
  echo "  concurrency      : ${CONCURRENCY}"
  echo "  tokenizer        : ${TOKENIZER}"
  echo "  max_input_len    : ${MAX_INPUT_LEN}"
  echo "  input_len_margin : ${INPUT_LEN_MARGIN}"

  start_sysmon "$RUN_DIR" "B"
  start_telemetry "$RUN_DIR"
  python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR"
  stop_sysmon "$RUN_DIR"
  stop_telemetry "$RUN_DIR"
done

#!/usr/bin/env bash
set -euo pipefail

# H2B Dynamo KV pressure (Test_Plan_v3.0 Section 8.2B, Stack B tiered)

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackB"
# Default to 8B dev path; set MODEL=nvidia/Llama-3.3-70B-Instruct-NVFP4 for the 70B sweep.
MODEL=${MODEL:-nvidia/Llama-3.1-8B-Instruct-NVFP4}
WORKLOAD="fixed_context"
CONCURRENCY=${CONCURRENCY:-32}
DURATION=${DURATION:-180}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:9000/v1/completions}

PROMPTS=(1024 2048 4096 6144 8192)
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}
# Tokenizer-aware truncation (align with Stack B engine admit, defaults assume Llama-3.1-8B build at 8k).
TOKENIZER=${TOKENIZER:-$MODEL}
MAX_INPUT_LEN=${MAX_INPUT_LEN:-8192}
INPUT_LEN_MARGIN=${INPUT_LEN_MARGIN:-64}

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

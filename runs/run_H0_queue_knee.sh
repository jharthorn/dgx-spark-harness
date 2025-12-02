#!/usr/bin/env bash
set -euo pipefail

# H0 queue knee (Test_Plan_v3.3 Section 8.0, Stack A/B Comfy)

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK=${STACK:-stackA}          # stackA (TRT-LLM UMA) or stackB (Dynamo tiered)
PROFILE=${PROFILE:-comfy}       # Stack B comfy baseline to match transparency checks in H0B
if [[ "$STACK" == "stackB" ]]; then
  apply_profile_env "$PROFILE"
fi
# Stack A default (localhost smoke) is 8355; Stack B default is 9000.
if [[ "$STACK" == "stackB" ]]; then
  ENDPOINT=${ENDPOINT:-${1:-http://127.0.0.1:9000/v1/completions}}
else
  ENDPOINT=${ENDPOINT:-${1:-http://127.0.0.1:8355/v1/completions}}
fi
MODEL=${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
MODEL_TAG=${MODEL_TAG:-$(model_tag "$MODEL")}
WORKLOAD="fixed_context"
CTX=${CTX:-${H0_CONTEXT_TOKENS:-2048}}
DURATION=${DURATION:-120}
CONCURRENCY_LIST=(${CONCURRENCY_LIST:-1 2 4 8 16 32 64 96 128 160 192 224 256})
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}
UWORK_MANIFEST=${UWORK_MANIFEST:-$HARNESS_DIR/runs/H0/uwork_manifest.txt}
mkdir -p "$(dirname "$UWORK_MANIFEST")"
: >"$UWORK_MANIFEST"

for U in "${CONCURRENCY_LIST[@]}"; do
  RUN_ID="$(rt_ts)_H0_${PROFILE}_${STACK}_${MODEL_TAG}_U${U}"
  RUN_DIR="$RESULTS_BASE/${RUN_ID}"
  ensure_run_dir "$RUN_DIR"

  cat > "$RUN_DIR/config.yaml" <<EOF
stack: ${STACK}
profile: ${PROFILE}
model: ${MODEL}
workload: ${WORKLOAD}
context_tokens: ${CTX}
concurrency: ${U}
duration_s: ${DURATION}
endpoint: ${ENDPOINT}
nonce_per_user: false
seed: 42
EOF

  echo "${RUN_ID},${RUN_DIR},${PROFILE},${U}" >>"$UWORK_MANIFEST"

  stack_tag="A"
  if [[ "$STACK" == "stackB" ]]; then
    stack_tag="B"
  fi

  start_sysmon "$RUN_DIR" "$stack_tag"
  if [[ "$STACK" == "stackB" ]]; then
    start_telemetry "$RUN_DIR"
  fi
  python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR" --endpoint "${ENDPOINT}"
  stop_sysmon "$RUN_DIR"
  if [[ "$STACK" == "stackB" ]]; then
    stop_telemetry "$RUN_DIR"
  fi
done

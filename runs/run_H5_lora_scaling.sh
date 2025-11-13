#!/usr/bin/env bash
set -euo pipefail

# v2.5: Implements H5 (LoRA Working-Set Scaling) from Test_Plan_v2.5.md
# Usage: ./run_H5_lora_scaling.sh L70B
#
# This test MUST run against the 70B model to be in the paging regime.
# It runs a sweep across:
# 1. Adapter Set Size (Small=4, Medium=16, Large=64)
# 2. Session Type (Sticky, Stormy/Random)
#
# This test MUST use LoRA and hits the Triton (port 8000) endpoint.

# Paths & defaults
export HARNESS_DIR=${HARNESS_DIR:-/harness}
export SRC_DIR=${SRC_DIR:-$HARNESS_DIR/src}
export RESULTS_DIR=${RESULTS_DIR:-$HARNESS_DIR/results}
export INPUTS_DIR=${INPUTS_DIR:-$HARNESS_DIR/inputs}
export ANALYSIS_DIR=${ANALYSIS_DIR:-$HARNESS_DIR/analysis}
export NVME_DEVICE=${NVME_DEVICE:-nvme0n1}
export MASTER_LOG=${MASTER_LOG:-$HARNESS_DIR/master_run.log}

# --- v2.5: Source Model Env ---
# This script is intended for the 70B model to force paging.
if [[ "$1" != "L70B" ]]; then
  echo "--- WARNING ---" | tee -a "$MASTER_LOG"
  echo "H5 is designed to test the paging regime."
  echo "Running with '$1' may not produce the intended UMA pressure." | tee -a "$MASTER_LOG"
  read -p "Press Enter to continue with $1, or Ctrl+C to abort."
fi
source "$HARNESS_DIR/runs/model_env.sh" $1

source "$HARNESS_DIR/runs/_lib_quiescence.sh"

HYP=H5
DUR=300
U=64 # U_work (from H0)
# Ck (from H2, the paging regime). 1536_tokens is a good default.
CONTEXT_CONFIG_PROMPT="prompts/1536_tokens.txt" 
REPEAT=3

# --- H5 Test Matrix ---
declare -a LORA_LIST_FILES=(
    "$INPUTS_DIR/lora_adapters/lora_list.txt"
    "$INPUTS_DIR/lora_adapters/lora_list_16.txt"
    "$INPUTS_DIR/lora_adapters/lora_list_64.txt"
)
declare -a LORA_SET_TAGS=( "S_4" "S_16" "S_64" )
declare -a SESSION_TYPES=( "sticky" "random" ) # 'random' is "Stormy"

mkdir -p "$RESULTS_DIR"
echo "--- RUNNING H5 (LoRA Working-Set Scaling) for $MODEL_TAG_SHORT ---" | tee -a "$MASTER_LOG"

echo "--- PRE-FLIGHT CHECK FOR H5 ---" | tee -a "$MASTER_LOG"
echo "This test (H5) MUST run against the TRITON (LoRA) server for $MODEL_TAG_SHORT."
echo "This is the server on port 8000."
read -p "Press Enter to continue..."

PROMPT_PATH="$INPUTS_DIR/$CONTEXT_CONFIG_PROMPT"
if [[ ! -f "$PROMPT_PATH" ]]; then echo "ERROR: H5 prompt file not found: $PROMPT_PATH" | tee -a "$MASTER_LOG"; exit 1; fi
PROMPT_SHA=$(sha256sum "$PROMPT_PATH" | awk '{print $1}')

# --- Iterate over the test matrix ---
for i in "${!LORA_LIST_FILES[@]}"; do
  LORA_LIST_FILE="${LORA_LIST_FILES[$i]}"
  LORA_TAG="${LORA_SET_TAGS[$i]}" # e.g., "S_4"
  ADAPTER_COUNT=$(echo "$LORA_TAG" | cut -d_ -f2)

  if [[ ! -f "$LORA_LIST_FILE" ]]; then
    echo "SKIPPING $LORA_TAG: File not found $LORA_LIST_FILE" | tee -a "$MASTER_LOG"
    continue
  fi

  for SESSION_TYPE in "${SESSION_TYPES[@]}"; do
    SESSION_TAG=$( [ "$SESSION_TYPE" = "random" ] && echo "STORM" || echo "STICKY" ) # STICKY or STORM

    echo "--- H5 Phase: $LORA_TAG ($ADAPTER_COUNT adapters), Session: $SESSION_TAG ---" | tee -a "$MASTER_LOG"
    # CRITICAL: Drop caches before each *new adapter set* to ensure
    # we measure the cost of pulling them from disk, especially for sticky.
    echo "Dropping OS page cache before new test phase..." | tee -a "$MASTER_LOG"
    drop_caches
    # We must also ask user to restart Triton server to clear its *internal* cache
    echo "Please RESTART the TRITON server (port 8000) to clear in-memory LoRA caches."
    read -p "Press Enter once the $MODEL_TAG_SHORT Triton server is back online."

    for r in $(seq 1 $REPEAT); do
      RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_${MODEL_TAG_SHORT}_${LORA_TAG}_${SESSION_TAG}_U${U}_r${r}"
      export RUN_ID
      echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

      # Manifest
      cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}", "hypothesis":"${HYP}", 
  "phase": "${LORA_TAG}_${SESSION_TAG}",
  "lora_session_type": "${SESSION_TYPE}",
  "lora_adapter_count": ${ADAPTER_COUNT},
  "model_handle":"${MODEL_HANDLE}",
  "model_tag":"${MODEL_TAG_SHORT}",
  "engine_profile":"bs256_ctx2k_triton", "kv_cache":"ON",
  "concurrency_users": ${U}, "run_duration_sec": ${DUR},
  "context": "$(basename $PROMPT_PATH)",
  "prompts_sha256":["${PROMPT_SHA}"]
}
EOF
      
      smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_pre.txt" 2>/dev/null || true
      NVME_DEVICE="${NVME_DEVICE}" RUN_ID="${RUN_ID}" "${SRC_DIR}/sysmon.sh" & SYSMON_PID=$!
      mpstat -P ALL 1 "${DUR}" > "${RESULTS_DIR}/${RUN_ID}_mpstat.log" 2>/dev/null || true
      
      # Loadgen (with --lora-list AND --lora-session)
      "${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" \
        -U ${U} \
        -P "$PROMPT_PATH" \
        --duration ${DUR} \
        --lora-list "$LORA_LIST_FILE" \
        --lora-session "$SESSION_TYPE" || true

      smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
      kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
      kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
      "${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"
      
      echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
      sleep 5 # Cooldown
    done
  done
done

echo "--- H5 (LoRA Working-Set Scaling) for $MODEL_TAG_SHORT COMPLETE ---" | tee -a "$MASTER_LOG"
run_fstrim

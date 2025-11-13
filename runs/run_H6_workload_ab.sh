#!/usr/bin/env bash
set -euo pipefail

# v2.5: Parameterized model tag
# Usage: ./run_H6_workload_ab.sh L8B
#    or: ./run_H6_workload_ab.sh L70B

# Paths & defaults
export HARNESS_DIR=${HARNESS_DIR:-/harness}
export SRC_DIR=${SRC_DIR:-$HARNESS_DIR/src}
export RESULTS_DIR=${RESULTS_DIR:-$HARNESS_DIR/results}
export INPUTS_DIR=${INPUTS_DIR:-$HARNESS_DIR/inputs}
export ANALYSIS_DIR=${ANALYSIS_DIR:-$HARNESS_DIR/analysis}
export NVME_DEVICE=${NVME_DEVICE:-nvme0n1}
export MASTER_LOG=${MASTER_LOG:-$HARNESS_DIR/master_run.log}

# --- NEW v2.5: Source Model Env ---
source "$HARNESS_DIR/runs/model_env.sh" $1

source "$HARNESS_DIR/runs/_lib_quiescence.sh"

HYP=H6
DUR=300
U=64 # U_work (from H0)
CONTEXT_CONFIG_PROMPT="prompts/1536_tokens.txt" # Ck (from H2, the paging regime)
LORA_LIST_FILE="$INPUTS_DIR/lora_adapters/lora_list.txt"
REPEAT=3

mkdir -p "$RESULTS_DIR"
echo "--- RUNNING H6 (Workload A/B Test) for $MODEL_TAG_SHORT ---" | tee -a "$MASTER_LOG"

echo "--- PRE-FLIGHT CHECK FOR H6 ---" | tee -a "$MASTER_LOG"
echo "This test (H6) MUST run against the TRITON server for $MODEL_TAG_SHORT."
read -p "Press Enter to continue..."

PROMPT_PATH="$INPUTS_DIR/$CONTEXT_CONFIG_PROMPT"
if [[ ! -f "$PROMPT_PATH" ]]; then echo "ERROR: H2 prompt file not found: $PROMPT_PATH" | tee -a "$MASTER_LOG"; exit 1; fi
PROMPT_SHA=$(sha256sum "$PROMPT_PATH" | awk '{print $1}')
if [[ ! -f "$LORA_LIST_FILE" ]]; then echo "ERROR: LoRA list not found: $LORA_LIST_FILE" | tee -a "$MASTER_LOG"; exit 1; fi

# --- PHASE 1: BASELINE (H6-Base) ---
echo "--- RUNNING H6 (Baseline Workload) for $MODEL_TAG_SHORT ---" | tee -a "$MASTER_LOG"
for r in $(seq 1 $REPEAT); do
  RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_${MODEL_TAG_SHORT}_BASELINE_U${U}_r${r}"
  export RUN_ID
  echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

  # Manifest
  cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}", "hypothesis":"${HYP}", "phase": "BASELINE",
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
  mpstat -P ALL 1 "${DUR}" > "${RESULTS_DIR}/${RUN_ID}_mpstat.log" 2>/dev/null & MPSTAT_PID=$!
  
  # Loadgen (Baseline: NO --lora-list)
  "${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" -U ${U} -P "$PROMPT_PATH" --duration ${DUR} || true

  smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
  kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
  kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
  "${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"
  
  echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
done
echo "--- H6 (Baseline) for $MODEL_TAG_SHORT COMPLETE ---" | tee -a "$MASTER_LOG"

# --- PHASE 2: LORA STORM (H6-LoRA) ---
echo "--- RUNNING H6 (LoRA Storm Workload) for $MODEL_TAG_SHORT ---" | tee -a "$MASTER_LOG"
for r in $(seq 1 $REPEAT); do
  RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_${MODEL_TAG_SHORT}_LORA_U${U}_r${r}"
  export RUN_ID
  echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

  # Manifest
  cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}", "hypothesis":"${HYP}", "phase": "LORA",
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
  mpstat -P ALL 1 "${DUR}" > "${RESULTS_DIR}/${RUN_ID}_mpstat.log" 2>/dev/null & MPSTAT_PID=$!

  # Loadgen (LoRA Storm: ADD --lora-list)
  "${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" -U ${U} -P "$PROMPT_PATH" --duration ${DUR} \
    --lora-list "$LORA_LIST_FILE" || true

  smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
  kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
  kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
  "${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"
  
  echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
done
echo "--- H6 (LoRA Storm) for $MODEL_TAG_SHORT COMPLETE ---" | tee -a "$MASTER_LOG"

run_fstrim
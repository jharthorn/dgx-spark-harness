#!/usr/bin/env bash
set -euo pipefail

# v2.5: Parameterized model tag
# Usage: ./run_H1_coldwarm_lora.sh L8B
#    or: ./run_H1_coldwarm_lora.sh L70B

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

HYP=H1
DUR=300
U=64 # U_work, TBD from H0
LORA_LIST_FILE="$INPUTS_DIR/lora_adapters/lora_list.txt"
PROMPT_FILE="$INPUTS_DIR/prompts/256_tokens.txt" # Use a small prompt

mkdir -p "$RESULTS_DIR"
echo "--- RUNNING H1 (LoRA Cold/Warm) for $MODEL_TAG_SHORT ---" | tee -a "$MASTER_LOG"

echo "--- PRE-FLIGHT CHECK FOR H1 ---" | tee -a "$MASTER_LOG"
echo "This test (H1) MUST run against the TRITON server for $MODEL_TAG_SHORT."
read -p "Press Enter to continue..."

if [[ ! -f "$LORA_LIST_FILE" ]]; then echo "ERROR: LoRA list not found: $LORA_LIST_FILE" | tee -a "$MASTER_LOG"; exit 1; fi
if [[ ! -f "$PROMPT_FILE" ]]; then echo "ERROR: Base prompt file not found: $PROMPT_FILE" | tee -a "$MASTER_LOG"; exit 1; fi
PROMPT_SHA=$(sha256sum "$PROMPT_FILE" | awk '{print $1}')

# --- PHASE 1: COLD ---
echo "--- RUNNING H1 (COLD) ---" | tee -a "$MASTER_LOG"
echo "Stopping server to clear in-memory LoRA cache. Please RESTART the server..." | tee -a "$MASTER_LOG"
read -p "Press Enter after the TRITON server for $MODEL_TAG_SHORT has RESTARTED."
drop_caches # CRITICAL: Call the helper to drop OS page cache

RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_${MODEL_TAG_SHORT}_COLD_U${U}_r1"
export RUN_ID
echo "Starting run: $RUN_ID (First run after restart)" | tee -a "$MASTER_LOG"

# Manifest
cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}", "hypothesis":"${HYP}", "phase": "COLD",
  "model_handle":"${MODEL_HANDLE}",
  "model_tag":"${MODEL_TAG_SHORT}",
  "engine_profile":"bs256_ctx2k_triton", "kv_cache":"ON",
  "concurrency_users": ${U}, "run_duration_sec": ${DUR},
  "prompts_sha256":["${PROMPT_SHA}"]
}
EOF

smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_pre.txt" 2>/dev/null || true
NVME_DEVICE="${NVME_DEVICE}" RUN_ID="${RUN_ID}" "${SRC_DIR}/sysmon.sh" & SYSMON_PID=$!
mpstat -P ALL 1 "${DUR}" > "${RESULTS_DIR}/${RUN_ID}_mpstat.log" 2>/dev/null & MPSTAT_PID=$!

"${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" -U ${U} -P "$PROMPT_FILE" --duration ${DUR} \
  --lora-list "$LORA_LIST_FILE" || true

smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
"${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"

echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
echo "--- H1 (COLD) COMPLETE ---" | tee -a "$MASTER_LOG"
sleep 10 # Let system settle

# --- PHASE 2: WARM ---
echo "--- RUNNING H1 (WARM) ---" | tee -a "$MASTER_LOG"
echo "Server is now WARM (LoRA adapters are cached in memory)." | tee -a "$MASTER_LOG"
# *** DO NOT CALL drop_caches or restart server ***
RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_${MODEL_TAG_SHORT}_WARM_U${U}_r1"
export RUN_ID
echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

# Manifest
cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}", "hypothesis":"${HYP}", "phase": "WARM",
  "model_handle":"${MODEL_HANDLE}",
  "model_tag":"${MODEL_TAG_SHORT}",
  "engine_profile":"bs256_ctx2k_triton", "kv_cache":"ON",
  "concurrency_users": ${U}, "run_duration_sec": ${DUR},
  "prompts_sha256":["${PROMPT_SHA}"]
}
EOF

smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_pre.txt" 2>/dev/null || true
NVME_DEVICE="${NVME_DEVICE}" RUN_ID="${RUN_ID}" "${SRC_DIR}/sysmon.sh" & SYSMON_PID=$!
mpstat -P ALL 1 "${DUR}" > "${RESULTS_DIR}/${RUN_ID}_mpstat.log" 2>/dev/null & MPSTAT_PID=$!

"${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" -U ${U} -P "$PROMPT_FILE" --duration ${DUR} \
  --lora-list "$LORA_LIST_FILE" || true

smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
"${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"

echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
echo "--- H1 (WARM) COMPLETE ---" | tee -a "$MASTER_LOG"

run_fstrim
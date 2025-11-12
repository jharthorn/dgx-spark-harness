#!/usr/bin/env bash
set -euo pipefail

# runs/run_H1_coldwarm_lora.sh
# Runs H1 (Revised) - Cold vs Warm LoRA working set.
# This test MUST run against the Triton server (port 8000).

# Paths & defaults
export HARNESS_DIR=${HARNESS_DIR:-/harness}
export SRC_DIR=${SRC_DIR:-$HARNESS_DIR/src}
export RESULTS_DIR=${RESULTS_DIR:-$HARNESS_DIR/results}
export INPUTS_DIR=${INPUTS_DIR:-$HARNESS_DIR/inputs}
export ANALYSIS_DIR=${ANALYSIS_DIR:-$HARNESS_DIR/analysis}
export NVME_DEVICE=${NVME_DEVICE:-nvme0n1}
export MASTER_LOG=${MASTER_LOG:-$HARNESS_DIR/master_run.log}

source "$HARNESS_DIR/runs/_lib_quiescence.sh"

HYP=H1
DUR=300
U_WORK=64 # U_work from H0 (e.g., 64)
REPEAT=3
PROMPT_FILE="$INPUTS_DIR/prompts/1024_tokens.txt" # Use a mid-size prompt
MAX_TOKENS=512
LORA_LIST_FILE="$INPUTS_DIR/lora_adapters/lora_list.txt"

mkdir -p "$RESULTS_DIR"
echo "--- RUNNING H1 (Cold/Warm LoRA) ---" | tee -a "$MASTER_LOG"
read -p "Ensure the TRITON (LoRA) server is running on port 8000. Press Enter to continue..."

if [[ ! -f "$PROMPT_FILE" || ! -f "$LORA_LIST_FILE" ]]; then
    echo "ERROR: Files not found. Prompt: $PROMPT_FILE, LoRA: $LORA_LIST_FILE" | tee -a "$MASTER_LOG"
    exit 1
fi
PROMPT_SHA=$(sha256sum "$PROMPT_FILE" | awk '{print $1}')

# --- PHASE 1: COLD ---
echo "--- H1 (COLD) - Running 3 repeats ---" | tee -a "$MASTER_LOG"
drop_caches # CRITICAL: Call the helper
for r in $(seq 1 $REPEAT); do
    RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_COLD_U${U_WORK}_r${r}"
    export RUN_ID
    echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

    # Manifest
    cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}", "hypothesis":"${HYP}", "phase": "COLD",
  "engine_profile": "bs256_ctx2k", "concurrency_users": ${U_WORK},
  "run_duration_sec": ${DUR}, "max_tokens": ${MAX_TOKENS},
  "prompts_sha256":["${PROMPT_SHA}"], "lora_enabled": true
}
EOF
    
    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_pre.txt" 2>/dev/null || true
    NVME_DEVICE="${NVME_DEVICE}" RUN_ID="${RUN_ID}" "${SRC_DIR}/sysmon.sh" & SYSMON_PID=$!
    mpstat -P ALL 1 "${DUR}" > "${RESULTS_DIR}/${RUN_ID}_mpstat.log" 2>/dev/null & MPSTAT_PID=$!

    "${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" \
      -U "${U_WORK}" \
      -P "$PROMPT_FILE" \
      --max-tokens ${MAX_TOKENS} \
      --lora-list "$LORA_LIST_FILE" \
      --duration "${DUR}" || true

    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
    kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
    kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
    
    "${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"
    echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
done

# --- PHASE 2: WARM (Priming + Measurement) ---
echo "--- H1 (WARM) - Running 1 priming run... ---" | tee -a "$MASTER_LOG"
# *** DO NOT CALL drop_caches ***
"${SRC_DIR}/loadgen.py" --run-id "PRIMING_RUN" -U ${U_WORK} -P "$PROMPT_FILE" --max-tokens ${MAX_TOKENS} --lora-list "$LORA_LIST_FILE" --duration 300 || true
echo "--- H1 (WARM) - Priming complete. Running 3 measurement runs... ---" | tee -a "$MASTER_LOG"

for r in $(seq 1 $REPEAT); do
    RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_WARM_U${U_WORK}_r${r}"
    export RUN_ID
    echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

    # Manifest
    cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}", "hypothesis":"${HYP}", "phase": "WARM",
  "engine_profile": "bs256_ctx2k", "concurrency_users": ${U_WORK},
  "run_duration_sec": ${DUR}, "max_tokens": ${MAX_TOKENS},
  "prompts_sha256":["${PROMPT_SHA}"], "lora_enabled": true
}
EOF

    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_pre.txt" 2>/dev/null || true
    NVME_DEVICE="${NVME_DEVICE}" RUN_ID="${RUN_ID}" "${SRC_DIR}/sysmon.sh" & SYSMON_PID=$!
    mpstat -P ALL 1 "${DUR}" > "${RESULTS_DIR}/${RUN_ID}_mpstat.log" 2>/dev/null & MPSTAT_PID=$!

    "${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" \
      -U "${U_WORK}" \
      -P "$PROMPT_FILE" \
      --max-tokens ${MAX_TOKENS} \
      --lora-list "$LORA_LIST_FILE" \
      --duration "${DUR}" || true

    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
    kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
    kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
    
    "${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"
    echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
done

echo "--- H1 (Cold/Warm LoRA) COMPLETE ---" | tee -a "$MASTER_LOG"
run_fstrim
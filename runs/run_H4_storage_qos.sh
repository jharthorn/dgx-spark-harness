#!/usr/bin/env bash
set -euo pipefail

# v2.5: Implements H4 (Storage QoS Sensitivity) from Test_Plan_v2.5.md
# Usage: ./run_H4_storage_qos.sh L70B
#
# This test MUST run against the 70B model to be in the paging regime.
# It runs three phases:
# 1. BASELINE: No storage interference.
# 2. MODERATE: 'fio' generates moderate read-only contention.
# 3. HEAVY: 'fio' generates heavy read-only contention.
#
# This test does NOT use LoRA, per the test plan, to isolate KV paging.

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
  echo "H4 is designed to test the paging regime."
  echo "Running with '$1' may not produce the intended UMA pressure." | tee -a "$MASTER_LOG"
  read -p "Press Enter to continue with $1, or Ctrl+C to abort."
fi
source "$HARNESS_DIR/runs/model_env.sh" $1

source "$HARNESS_DIR/runs/_lib_quiescence.sh"

HYP=H4
DUR=300
U=64 # U_work (from H0)
# Ck (from H2, the paging regime). 1536_tokens is a good default.
CONTEXT_CONFIG_PROMPT="prompts/1536_tokens.txt" 
REPEAT=3

# FIO configuration
FIO_FILE="/workspace/fio_test_file.dat" # Assumes /workspace is on the NVMe
FIO_SIZE="50G" # Use a large file to ensure it hits storage
FIO_RUNTIME=$((DUR + 15)) # Run fio slightly longer than loadgen
FIO_LOG_DIR="$RESULTS_DIR/fio_logs"
mkdir -p "$FIO_LOG_DIR"

echo "--- RUNNING H4 (Storage QoS Sensitivity) for $MODEL_TAG_SHORT ---" | tee -a "$MASTER_LOG"

echo "--- PRE-FLIGHT CHECK FOR H4 ---" | tee -a "$MASTER_LOG"
echo "This test (H4) MUST run against the BASELINE (No LoRA) server for $MODEL_TAG_SHORT."
echo "This is the OpenAI-compatible server on port 8355."
read -p "Press Enter to continue..."

PROMPT_PATH="$INPUTS_DIR/$CONTEXT_CONFIG_PROMPT"
if [[ ! -f "$PROMPT_PATH" ]]; then echo "ERROR: H4 prompt file not found: $PROMPT_PATH" | tee -a "$MASTER_LOG"; exit 1; fi
PROMPT_SHA=$(sha256sum "$PROMPT_PATH" | awk '{print $1}')

echo "Preparing fio test file ($FIO_SIZE) at $FIO_FILE. This may take a moment..." | tee -a "$MASTER_LOG"
fio --name=setup --filename=$FIO_FILE --size=$FIO_SIZE --rw=write --bs=1M --direct=1 --output-format=terse \
  --output="$FIO_LOG_DIR/fio_setup.log" || echo "FIO setup failed, continuing..."
echo "fio test file prepared." | tee -a "$MASTER_LOG"


# --- Function to run one test phase ---
run_phase() {
  local PHASE_NAME=$1 # e.g., BASELINE, MODERATE, HEAVY
  local FIO_ARGS=$2   # Fio command string, or "NONE"

  echo "--- RUNNING H4 (Phase: $PHASE_NAME) for $MODEL_TAG_SHORT ---" | tee -a "$MASTER_LOG"
  
  local FIO_PID=""
  if [[ "$FIO_ARGS" != "NONE" ]]; then
    echo "Starting FIO ($PHASE_NAME) in background..." | tee -a "$MASTER_LOG"
    # Run fio with the specified args
    fio --name="h4_contention_$PHASE_NAME" \
        --filename=$FIO_FILE \
        --size=$FIO_SIZE \
        --direct=1 \
        --time_based --runtime=$FIO_RUNTIME \
        --rw=randread --bs=64k \
        $FIO_ARGS \
        --output-format=terse \
        --output="$FIO_LOG_DIR/fio_${HYP}_${MODEL_TAG_SHORT}_${PHASE_NAME}.log" &
    FIO_PID=$!
    sleep 5 # Let fio ramp up
  fi

  for r in $(seq 1 $REPEAT); do
    RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_${MODEL_TAG_SHORT}_${PHASE_NAME}_U${U}_r${r}"
    export RUN_ID
    echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

    # Manifest
    cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}", "hypothesis":"${HYP}", "phase": "${PHASE_NAME}",
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
    sleep 5 # Cooldown
  done
  
  if [[ -n "$FIO_PID" ]]; then
    echo "Stopping FIO ($PHASE_NAME)..." | tee -a "$MASTER_LOG"
    kill "$FIO_PID" 2>/dev/null || true
    wait "$FIO_PID" 2>/dev/null || true
  fi
  
  echo "--- H4 (Phase: $PHASE_NAME) for $MODEL_TAG_SHORT COMPLETE ---" | tee -a "$MASTER_LOG"
}

# --- Execute Phases ---

# Phase 1: Baseline
run_phase "BASELINE" "NONE"

# Phase 2: Moderate Contention
# Per test plan: "fio QD8–16, read-heavy, ~40% NVMe util"
# We'll use iodepth=8 and rate-limit to achieve partial utilization.
# This may need tuning for your specific drive. Start with 500MB/s.
run_phase "MODERATE" "--iodepth=8 --rate_iops=8000" # Rate limit to 8k IOPS (approx 500MB/s @ 64k)

# Phase 3: Heavy Contention
# Per test plan: "Higher QD, or cgroup IOPS/bandwidth limiting"
# We'll use a higher iodepth and no rate limit to try and saturate.
run_phase "HEAVY" "--iodepth=32" # No rate limit, just higher QD

echo "--- H4 (Storage QoS Sensitivity) for $MODEL_TAG_SHORT COMPLETE ---" | tee -a "$MASTER_LOG"
echo "Cleaning up fio test file..." | tee -a "$MASTER_LOG"
rm -f "$FIO_FILE"

run_fstrim

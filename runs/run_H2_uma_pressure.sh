#!/usr/bin/env bash
set -euo pipefail

# v2.5: Parameterized model tag
# Usage: ./run_H2_uma_pressure.sh L8B
#    or: ./run_H2_uma_pressure.sh L70B

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

HYP=H2
DUR=300
U_WORK=64 # U_work from H0 (e.g., 64)
REPEAT=3

# Define the context sweeps (Prompt File, Max_Tokens)
# Note: Engine is built for max_input_len=2048, max_seq_len=3072
declare -a PROMPTS=(
    "$INPUTS_DIR/prompts/256_tokens.txt"
    "$INPUTS_DIR/prompts/512_tokens.txt"
    "$INPUTS_DIR/prompts/1024_tokens.txt"
    "$INPUTS_DIR/prompts/1536_tokens.txt"
)
declare -a MAX_TOKENS_LIST=(
    256
    512
    512
    512
)

mkdir -p "$RESULTS_DIR"
echo "--- RUNNING H2 (UMA Pressure Sweep) for $MODEL_TAG_SHORT ---" | tee -a "$MASTER_LOG"
read -p "Ensure the TRITON server for $MODEL_TAG_SHORT is running on port 8000. Press Enter to continue..."

for i in "${!PROMPTS[@]}"; do
  PROMPT_FILE="${PROMPTS[$i]}"
  MAX_TOKENS="${MAX_TOKENS_LIST[$i]}"
  
  PROMPT_NAME=$(basename "$PROMPT_FILE" .txt) # e.g., "256_tokens"
  CONTEXT_TAG="${PROMPT_NAME}_gen${MAX_TOKENS}"  # e.g., "256_tokens_gen256"

  if [[ ! -f "$PROMPT_FILE" ]]; then
      echo "ERROR: Base prompt file not found: $PROMPT_FILE" | tee -a "$MASTER_LOG"
      exit 1
  fi
  PROMPT_SHA=$(sha256sum "$PROMPT_FILE" | awk '{print $1}')

  for r in $(seq 1 $REPEAT); do
    RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_${MODEL_TAG_SHORT}_${CONTEXT_TAG}_U${U_WORK}_r${r}"
    export RUN_ID
    echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

    # Manifest
    cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}",
  "timestamp_utc":"$(date -u +%Y%m%dT%H:%M:%SZ)",
  "hypothesis":"${HYP}",
  "model_handle":"${MODEL_HANDLE}",
  "model_tag":"${MODEL_TAG_SHORT}",
  "engine_profile": "bs256_ctx2k_triton",
  "context_config": "${CONTEXT_TAG}",
  "concurrency_users": ${U_WORK},
  "run_duration_sec": ${DUR},
  "max_tokens": ${MAX_TOKENS},
  "sampling":{"temperature":0.0,"top_p":1.0,"seed":42},
  "prompts_sha256":["${PROMPT_SHA}"],
  "storage":{"device":"/dev/${NVME_DEVICE}","model":"$(cat /sys/block/${NVME_DEVICE}/device/model 2>/dev/null || echo unknown)","sched":"$(cat /sys/block/${NVME_DEVICE}/queue/scheduler 2>/dev/null || echo unknown)","read_ahead_kb":"$(cat /sys/block/${NVME_DEVICE}/queue/read_ahead_kb 2>/dev/null || echo unknown)"},
  "os":{"kernel":"$(uname -r)","driver":"$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo n/a)","dgx_os":"$(. /etc/os-release && echo $PRETTY_NAME 2>/dev/null || echo n/a)"},
  "cpu":{"governor":"performance","isolation":"n/a"},
  "saturated": false
}
EOF

    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_pre.txt" 2>/dev/null || true
    NVME_DEVICE="${NVME_DEVICE}" RUN_ID="${RUN_ID}" "${SRC_DIR}/sysmon.sh" & SYSMON_PID=$!
    mpstat -P ALL 1 "${DUR}" > "${RESULTS_DIR}/${RUN_ID}_mpstat.log" 2>/dev/null & MPSTAT_PID=$!

    # Run loadgen (v2.3) - NO --lora-list flag
    "${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" \
      -U "${U_WORK}" \
      -P "$PROMPT_FILE" \
      --max-tokens ${MAX_TOKENS} \
      --duration "${DUR}" || true

    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
    kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
    kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true

    "${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"
    echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
    sleep 5
  done
done
echo "--- H2 (UMA Pressure Sweep) for $MODEL_TAG_SHORT COMPLETE ---" | tee -a "$MASTER_LOG"
run_fstrim
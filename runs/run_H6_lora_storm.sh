#!/usr/bin/env bash
set -euo pipefail

# runs/run_H6_lora_storm.sh
# Runs H6 (LoRA Storm) CONCURRENCY SWEEP test.
# v2.4: Re-added fio pre-conditioning, but in a *safe* file-based mode.

# Paths & defaults
export HARNESS_DIR=${HARNESS_DIR:-/harness}
export SRC_DIR=${SRC_DIR:-$HARNESS_DIR/src}
export RESULTS_DIR=${RESULTS_DIR:-$HARNESS_DIR/results}
export INPUTS_DIR=${INPUTS_DIR:-$HARNESS_DIR/inputs}
export ANALYSIS_DIR=${ANALYSIS_DIR:-$HARNESS_DIR/analysis}
export NVME_DEVICE=${NVME_DEVICE:-nvme0n1}
export MASTER_LOG=${MASTER_LOG:-$HARNESS_DIR/master_run.log}

source "$HARNESS_DIR/runs/_lib_quiescence.sh"

HYP=H6
DUR=300
USERS_LIST=("8" "16" "32" "64" "128" "192") # Same as H2
REPEAT=3
LORA_LIST_FILE="$INPUTS_DIR/lora_adapters/lora_list.txt"
PROMPT_FILE="$INPUTS_DIR/prompts/1k_tokens.txt"
PROMPT_SHA=$(sha256sum "$PROMPT_FILE" | awk '{print $1}')
ENGINE_PROFILE="bs64_triton_lora" # H6 MUST run against Triton server

if [[ ! -f "$LORA_LIST_FILE" ]]; then
    echo "ERROR: LoRA list not found at $LORA_LIST_FILE" | tee -a "$MASTER_LOG"
    exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Base prompt file not found: $PROMPT_FILE" | tee -a "$MASTER_LOG"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# --- PRE-FLIGHT CHECK ---
echo "--- PRE-FLIGHT CHECK FOR H6 ---" | tee -a "$MASTER_LOG"
echo "ERROR: The 'trtllm-serve' (OpenAI API) server does NOT support LoRA." | tee -a "$MASTER_LOG"
echo "To run this H6 test, you MUST be running the TRITON server (on port 8000)." | tee -a "$MASTER_LOG"
read -p "Ensure the TRITON (bs=64) server is running on port 8000. Press Enter to continue..."

echo "--- RUNNING H6 (LoRA Storm Concurrency Sweep) ---" | tee -a "$MASTER_LOG"
echo "Users: ${USERS_LIST[@]}, Repeats: $REPEAT" | tee -a "$MASTER_LOG"

# --- v2.4: SAFE FIO PRECONDITIONING (FILE-BASED) ---
FIO_TEST_FILE="$RESULTS_DIR/fio_precondition.tmp" # Write to a temp file
trap 'rm -f "$FIO_TEST_FILE"' EXIT INT TERM # Ensure cleanup

echo "Starting 100G sequential write pre-conditioning (to file)..." | tee -a "$MASTER_LOG"
fio --name=precondition_fill \
    --filename=$FIO_TEST_FILE \
    --rw=write \
    --bs=128k \
    --size=100G \
    --direct=1 \
    --iodepth=32 \
    --loops=1 \
    --eta=always \
    || echo "WARNING: fio pre-conditioning failed." | tee -a "$MASTER_LOG"

echo "Pre-conditioning complete. Cleaning up test file..." | tee -a "$MASTER_LOG"
rm -f "$FIO_TEST_FILE"
trap - EXIT INT TERM # Clear trap
echo "Sleeping 30s for drive to settle..." | tee -a "$MASTER_LOG"
sleep 30
# --- END FIO SECTION ---


for U in "${USERS_LIST[@]}"; do
  for r in $(seq 1 $REPEAT); do
    RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_U${U}_r${r}"
    export RUN_ID
    echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

    # Manifest
    DRIVE_MODEL=$(cat /sys/block/${NVME_DEVICE}/device/model 2>/dev/null || echo unknown)
    cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}",
  "timestamp_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hypothesis":"${HYP}",
  "engine_profile": "${ENGINE_PROFILE}",
  "model_name":"gpt-oss-120B", "kv_cache":"ON",
  "concurrency_users": ${U}, "run_duration_sec": ${DUR},
  "sampling":{"temperature":0.0,"top_p":1.0,"seed":42},
  "prompts_sha256":["${PROMPT_SHA}"],
  "storage":{"device":"/dev/${NVME_DEVICE}","model":"${DRIVE_MODEL}","sched":"$(cat /sys/block/${NVME_DEVICE}/queue/scheduler 2>/dev/null || echo unknown)","read_ahead_kb":"$(cat /sys/block/${NVME_DEVICE}/queue/read_ahead_kb 2>/dev/null || echo unknown)"},
  "os":{"kernel":"$(uname -r)","driver":"$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo n/a)","dgx_os":"$(. /etc/os-release && echo $PRETTY_NAME 2>/dev/null || echo n/a)"},
  "cpu":{"governor":"performance","isolation":"server:0-9, loadgen:10-19"},
  "saturated": false
}
EOF

    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_pre.txt" 2>/dev/null || true
    NVME_DEVICE="${NVME_DEVICE}" RUN_ID="${RUN_ID}" "${SRC_DIR}/sysmon.sh" & SYSMON_PID=$!
    mpstat -P ALL 1 "${DUR}" > "${RESULTS_DIR}/${RUN_ID}_mpstat.log" 2>/dev/null & MPSTAT_PID=$!

    # Loadgen → add --lora-list and -U
    "${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" -U ${U} -P "$PROMPT_FILE" --lora-list "$LORA_LIST_FILE" --duration ${DUR} || true
    
    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
    kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
    kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
    "${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"

    echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
    sleep 5
  done
done
echo "--- H6 (LoRA Storm Concurrency Sweep) COMPLETE ---" | tee -a "$MASTER_LOG"

run_fstrim
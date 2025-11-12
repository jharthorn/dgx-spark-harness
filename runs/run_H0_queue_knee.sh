#!/usr/bin/env bash
set -euo pipefail

# runs/run_H0_queue_knee.sh
# Runs H0 (Calibration) test to find the server's application queue knee.
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

HYP=H0
DUR=300
USERS_LIST=("8" "16" "32" "64" "96" "128" "192")
REPEAT=3
PROMPT_FILE="$INPUTS_DIR/prompts/256_tokens.txt" # Use a small, standard prompt
MAX_TOKENS=32

mkdir -p "$RESULTS_DIR"
echo "--- RUNNING H0 (Queue Knee Calibration) ---" | tee -a "$MASTER_LOG"
read -p "Ensure the TRITON (LoRA) server is running on port 8000. Press Enter to continue..."

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Base prompt file not found: $PROMPT_FILE" | tee -a "$MASTER_LOG"
    exit 1
fi
PROMPT_SHA=$(sha256sum "$PROMPT_FILE" | awk '{print $1}')

for U in "${USERS_LIST[@]}"; do
  for r in $(seq 1 $REPEAT); do
    RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_U${U}_r${r}"
    export RUN_ID
    echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

    # Manifest
    cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}",
  "timestamp_utc":"$(date -u +%Y%m%dT%H:%M:%SZ)",
  "hypothesis":"${HYP}",
  "engine_profile": "bs256_ctx2k",
  "concurrency_users": ${U},
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
      -U "${U}" \
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
echo "--- H0 (Queue Knee Calibration) COMPLETE ---" | tee -a "$MASTER_LOG"
run_fstrim
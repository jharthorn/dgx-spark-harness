#!/usr/bin/env bash
set -euo pipefail

# runs/run_H2_concurrency.sh
# Runs H2 (Concurrency Sweep) test.
# v2.2: Targets the trtllm-serve container on port 8355.

# Paths & defaults
export HARNESS_DIR=${HARNESS_DIR:-/harness}
export SRC_DIR=${SRC_DIR:-$HARNESS_DIR/src}
export RESULTS_DIR=${RESULTS_DIR:-$HARNESS_DIR/results}
export INPUTS_DIR=${INPUTS_DIR:-$HARNESS_DIR/inputs}
export ANALYSIS_DIR=${ANALYSIS_DIR:-$HARNESS_DIR/analysis}
export NVME_DEVICE=${NVME_DEVICE:-nvme0n1}
export MASTER_LOG=${MASTER_LOG:-$HARNESS_DIR/master_run.log}

source "$HARNESS_DIR/runs/_lib_quiescence.sh"

HYP=H2
DUR=300
USERS_LIST=("8" "16" "32" "64" "128" "192") # Test beyond the 64 batch size
REPEAT=3
PROMPT_FILE="$INPUTS_DIR/prompts/1k_tokens.txt"
PROMPT_SHA=$(sha256sum "$PROMPT_FILE" | awk '{print $1}')
ENGINE_PROFILE="bs64_default_ctx" # v2.2: High-Throughput Server

mkdir -p "$RESULTS_DIR"

# --- PRE-FLIGHT CHECK ---
echo "--- PRE-FLIGHT CHECK FOR H2 ---" | tee -a "$MASTER_LOG"
read -p "This test (H2) MUST run against the trtllm-serve (bs=64) container. Ensure it is running on port 8355. Press Enter to continue..."

echo "--- RUNNING H2 (Concurrency Sweep) ---" | tee -a "$MASTER_LOG"
echo "Users: ${USERS_LIST[@]}, Repeats: $REPEAT" | tee -a "$MASTER_LOG"

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Base prompt file not found: $PROMPT_FILE" | tee -a "$MASTER_LOG"
    exit 1
fi

for U in "${USERS_LIST[@]}"; do
  for r in $(seq 1 $REPEAT); do
    RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_U${U}_r${r}"
    export RUN_ID
    echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

    # Manifest
    cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}",
  "timestamp_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hypothesis":"${HYP}",
  "engine_profile": "${ENGINE_PROFILE}",
  "model_name":"gpt-oss-120b",
  "tokenizer_sha":"unknown",
  "kv_cache":"ON",
  "concurrency_users": ${U},
  "run_duration_sec": ${DUR},
  "sampling":{"temperature":0.0,"top_p":1.0,"seed":42},
  "prompts_sha256":["${PROMPT_SHA}"],
  "storage":{"device":"/dev/${NVME_DEVICE}","model":"$(cat /sys/block/${NVME_DEVICE}/device/model 2>/dev/null || echo unknown)","sched":"$(cat /sys/block/${NVME_DEVICE}/queue/scheduler 2>/dev/null || echo unknown)","read_ahead_kb":"$(cat /sys/block/${NVME_DEVICE}/queue/read_ahead_kb 2>/dev/null || echo unknown)"},
  "os":{"kernel":"$(uname -r)","driver":"$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo n/a)","dgx_os":"$(. /etc/os-release && echo $PRETTY_NAME 2>/dev/null || echo n/a)"},
  "cpu":{"governor":"performance","isolation":"server:0-9, loadgen:10-19"},
  "saturated": false
}
EOF

    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_pre.txt" 2>/dev/null || true
    NVME_DEVICE="${NVME_DEVICE}" RUN_ID="${RUN_ID}" "${SRC_DIR}/sysmon.sh" & SYSMON_PID=$!
    mpstat -P ALL 1 "${DUR}" > "${RESULTS_DIR}/${RUN_ID}_mpstat.log" 2>/dev/null & MPSTAT_PID=$!

    "${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" -U "${U}" -P "$PROMPT_FILE" --duration "${DUR}" || true

    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
    kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
    kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
    "${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"

    echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
    sleep 5
  done
done
echo "--- H2 (Concurrency Sweep) COMPLETE ---" | tee -a "$MASTER_LOG"

run_fstrim
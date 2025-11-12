#!/usr/bin/env bash
set -euo pipefail

# runs/run_H5_context.sh
# Runs H5 (Context Length Sweep) test.
# --- V2.1: This test MUST run against ENGINE A (Long Context) ---

# Paths & defaults
export HARNESS_DIR=${HARNESS_DIR:-/harness}
export SRC_DIR=${SRC_DIR:-$HARNESS_DIR/src}
export RESULTS_DIR=${RESULTS_DIR:-$HARNESS_DIR/results}
export INPUTS_DIR=${INPUTS_DIR:-$HARNESS_DIR/inputs}
export ANALYSIS_DIR=${ANALYSIS_DIR:-$HARNESS_DIR/analysis}
export NVME_DEVICE=${NVME_DEVICE:-nvme0n1}
export MASTER_LOG=${MASTER_LOG:-$HARNESS_DIR/master_run.log}

source "$HARNESS_DIR/runs/_lib_quiescence.sh"

HYP=H5
DUR=120 # Duration per prompt
U=1     # Single user to isolate context length impact
PROMPT_FILES=(
  "prompts/1k_tokens.txt"
  "prompts/2k_tokens.txt"
  "prompts/4k_tokens.txt"
  "prompts/8k_tokens.txt"
  "prompts/16k_tokens.txt"
  "prompts/32k_tokens.txt"
)
REPEAT=3
ENGINE_PROFILE="ctx32k_bs1" # --- V2.1: Engine A ---

mkdir -p "$RESULTS_DIR"

# --- V2.1: Safety Check ---
echo "--- PRE-FLIGHT CHECK FOR H5 ---" | tee -a "$MASTER_LOG"
read -p "This test (H5) MUST run against ENGINE A (ctx32k, bs=1). Ensure it is running. Press Enter to continue..."

echo "--- RUNNING H5 (Context Length Sweep) ---" | tee -a "$MASTER_LOG"

for P_FILE in "${PROMPT_FILES[@]}"; do
  PROMPT_NAME=$(basename "$P_FILE" .txt)
  PROMPT_PATH="$INPUTS_DIR/$P_FILE"
  if [[ ! -f "$PROMPT_PATH" ]]; then
    echo "SKIPPING: $PROMPT_PATH not found" | tee -a "$MASTER_LOG"
    continue
  fi
  PROMPT_SHA=$(sha256sum "$PROMPT_PATH" | awk '{print $1}')

  for r in $(seq 1 $REPEAT); do
    RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_${PROMPT_NAME}_U${U}_r${r}"
    export RUN_ID
    echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

    # Manifest
    cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}",
  "timestamp_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hypothesis":"${HYP}", "context": "${PROMPT_NAME}",
  "engine_profile": "${ENGINE_PROFILE}",
  "model_name":"gpt-oss-120b", "kv_cache":"ON",
  "concurrency_users": ${U}, "run_duration_sec": ${DUR},
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
    
    "${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" -U ${U} -P "$PROMPT_PATH" --duration ${DUR} || true

    smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
    kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
    kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
    "${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"

    echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
    sleep 5
  done
done
echo "--- H5 (Context Length Sweep) COMPLETE ---" | tee -a "$MASTER_LOG"

run_fstrim
#!/usr/bin/env bash
set -euo pipefail

# runs/run_H4_kv_policy.sh
# Runs H4 (KV Cache Policy) A/B test.
# --- V2.1: This test MUST run against ENGINE B (High Throughput) ---

# Paths & defaults
export HARNESS_DIR=${HARNESS_DIR:-/harness}
export SRC_DIR=${SRC_DIR:-$HARNESS_DIR/src}
export RESULTS_DIR=${RESULTS_DIR:-$HARNESS_DIR/results}
export INPUTS_DIR=${INPUTS_DIR:-$HARNESS_DIR/inputs}
export ANALYSIS_DIR=${ANALYSIS_DIR:-$HARNESS_DIR/analysis}
export NVME_DEVICE=${NVME_DEVICE:-nvme0n1}
export MASTER_LOG=${MASTER_LOG:-$HARNESS_DIR/master_run.log}

source "$HARNESS_DIR/runs/_lib_quiescence.sh"

HYP=H4
DUR=300
U=64 # Use a high-pressure user count
REPEAT=3
PROMPT_FILE="$INPUTS_DIR/prompts/1k_tokens.txt" # Use 1k tokens (fits in 2k engine)
PROMPT_SHA=$(sha256sum "$PROMPT_FILE" | awk '{print $1}')
ENGINE_PROFILE="bs256_ctx2k" # --- V2.1: Engine B ---

mkdir -p "$RESULTS_DIR"

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Base prompt file not found: $PROMPT_FILE" | tee -a "$MASTER_LOG"
    exit 1
fi

# --- PHASE 1: KV-Cache ON ---
echo "--- RUNNING H4 (KV-Cache ON) ---" | tee -a "$MASTER_LOG"
# --- V2.1: Updated Prompt ---
read -p "Ensure ENGINE B (bs256, ctx=2k) is running WITH KV-Cache ENABLED. Press Enter to continue..."
KV_STATE="ON"
for r in $(seq 1 $REPEAT); do
  RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_KV-ON_U${U}_r${r}"
  export RUN_ID
  echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

  # Manifest
  cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}",
  "timestamp_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hypothesis":"${HYP}", "kv_cache":"${KV_STATE}",
  "engine_profile": "${ENGINE_PROFILE}",
  "model_name":"gpt-oss-120b", "tokenizer_sha":"unknown",
  "concurrency_users": ${U}, "run_duration_sec": ${DUR},
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
done

# --- PHASE 2: KV-Cache OFF ---
echo "--- RUNNING H4 (KV-Cache OFF) ---" | tee -a "$MASTER_LOG"
# --- V2.1: Updated Prompt ---
read -p "RESTART ENGINE B (bs256, ctx=2k) WITH KV-Cache DISABLED. Press Enter to continue..."
KV_STATE="OFF"
for r in $(seq 1 $REPEAT); do
  RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_KV-OFF_U${U}_r${r}"
  export RUN_ID
  echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

  # Manifest
  cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}",
  "timestamp_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hypothesis":"${HYP}", "kv_cache":"${KV_STATE}",
  "engine_profile": "${ENGINE_PROFILE}",
  "model_name":"gpt-oss-120b", "tokenizer_sha":"unknown",
  "concurrency_users": ${U}, "run_duration_sec": ${DUR},
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
done
echo "--- H4 (KV-Cache Policy) COMPLETE ---" | tee -a "$MASTER_LOG"

run_fstrim
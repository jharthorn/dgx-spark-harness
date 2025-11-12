#!/usr/bin/env bash
set -euo pipefail

# runs/run_H1_cache.sh
# Runs H1 (Cold vs Warm) test. Requires --privileged container for drop_caches.
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

HYP=H1
DUR=60 # Short run, just need TTFT
U=1
PROMPT_FILE="$INPUTS_DIR/prompts/1k_tokens.txt"
PROMPT_SHA=$(sha256sum "$PROMPT_FILE" | awk '{print $1}')
ENGINE_PROFILE="bs64_default_ctx" # v2.2: High-Throughput Server

mkdir -p "$RESULTS_DIR"

# --- PRE-FLIGHT CHECK ---
echo "--- PRE-FLIGHT CHECK FOR H1 ---" | tee -a "$MASTER_LOG"
read -p "This test (H1) MUST run against the trtllm-serve (bs=64) container. Ensure it is running on port 8355. Press Enter to continue..."

# --- PHASE 1: COLD ---
echo "--- RUNNING H1 (COLD) ---" | tee -a "$MASTER_LOG"
drop_caches # CRITICAL: Call the helper
RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_COLD_U${U}_r1"
export RUN_ID
echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

# Manifest
cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}",
  "timestamp_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hypothesis":"${HYP}", "phase": "COLD",
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

# Wait for server to load model (this run *is* the load)
"${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" -U ${U} -P "$PROMPT_FILE" --duration ${DUR} || true

smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
"${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"

echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
echo "--- H1 (COLD) COMPLETE ---" | tee -a "$MASTER_LOG"
sleep 10 # Let system settle

# --- PHASE 2: WARM ---
echo "--- RUNNING H1 (WARM) ---" | tee -a "$MASTER_LOG"
# *** DO NOT CALL drop_caches ***
RUN_ID="$(date -u +%Y%m%d_%H%M)_${HYP}_WARM_U${U}_r1"
export RUN_ID
echo "Starting run: $RUN_ID" | tee -a "$MASTER_LOG"

# Manifest
cat > "$RESULTS_DIR/${RUN_ID}_manifest.json" <<EOF
{
  "run_id":"${RUN_ID}",
  "timestamp_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hypothesis":"${HYP}", "phase": "WARM",
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

"${SRC_DIR}/loadgen.py" --run-id "${RUN_ID}" -U ${U} -P "$PROMPT_FILE" --duration ${DUR} || true

smartctl -a "/dev/${NVME_DEVICE}" > "${RESULTS_DIR}/${RUN_ID}_smartctl_post.txt" 2>/dev/null || true
kill "${SYSMON_PID}" 2>/dev/null || true; wait "${SYSMON_PID}" 2>/dev/null || true
kill "${MPSTAT_PID}" 2>/dev/null || true; wait "${MPSTAT_PID}" 2>/dev/null || true
"${ANALYSIS_DIR}/backfill_summary.py" "${RUN_ID}"

echo "Completed run: $RUN_ID" | tee -a "$MASTER_LOG"
echo "--- H1 (WARM) COMPLETE ---" | tee -a "$MASTER_LOG"

run_fstrim
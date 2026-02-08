#!/usr/bin/env bash
# Hypothesis: H4A – Storage QoS sensitivity control on Stack A
# Typical profile: Comfy
# Expected behavior: baseline NVMe QoS sensitivity without tiering; control vs H4B.
# See docs/Test_Plan_v3.3.md, section H4A (8.4A).
set -euo pipefail

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackA"
# Default to Llama 3.3 70B NVFP4; override MODEL env to switch (e.g., 8B for smoke).
MODEL=${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
WORKLOAD="fixed_context"
CTX=2048
CONCURRENCY=32
DURATION=180
ENDPOINT=${ENDPOINT:-http://127.0.0.1:8355/v1/completions}
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}
PROFILE=${PROFILE:-comfy}

for PHASE in baseline qos_degraded; do
  RUN_ID="$(rt_ts)_H4A_${PHASE}_${STACK}_${MODEL}"
  RUN_DIR="$RESULTS_BASE/${RUN_ID}"
  ensure_run_dir "$RUN_DIR"

  cat > "$RUN_DIR/config.yaml" <<EOF
stack: ${STACK}
profile: ${PROFILE}
model: ${MODEL}
workload: ${WORKLOAD}
context_tokens: ${CTX}
concurrency: ${CONCURRENCY}
duration_s: ${DURATION}
endpoint: ${ENDPOINT}
nonce_per_user: true
seed: 42
phase: ${PHASE}
EOF

  # Optional QoS perturbation: run a background fio load during qos_degraded to demonstrate UMA insensitivity.
  qos_pid=""
  if [[ "$PHASE" == "qos_degraded" && "${H4A_QOS_FIO:-1}" == "1" ]]; then
    QOS_PATH=${QOS_PATH:-/nvme/kvbm/l70b}
    QOS_FILE=${QOS_FILE:-"$QOS_PATH/.qos_noise"}
    QOS_SIZE=${QOS_SIZE:-4G}
    QOS_IODEPTH=${QOS_IODEPTH:-8}
    QOS_BS=${QOS_BS:-128k}
    QOS_RW=${QOS_RW:-randread}
    mkdir -p "$QOS_PATH"
    fio --name=qos_noise --filename="$QOS_FILE" --size="$QOS_SIZE" --rw="$QOS_RW" \
        --direct=1 --ioengine=libaio --iodepth="$QOS_IODEPTH" --bs="$QOS_BS" \
        --time_based --runtime="$DURATION" --group_reporting --numjobs=1 \
        >"$RUN_DIR/qos_fio.log" 2>&1 &
    qos_pid=$!
    echo "$qos_pid" >"$RUN_DIR/qos_fio.pid"
  fi

  start_sysmon "$RUN_DIR" "A"
  python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR"
  stop_sysmon "$RUN_DIR"
  if [[ -n "$qos_pid" ]]; then
    kill "$qos_pid" 2>/dev/null || true
  fi
done

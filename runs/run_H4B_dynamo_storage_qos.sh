#!/usr/bin/env bash
# Hypothesis: H4B – Dynamo storage QoS sensitivity on Stack B
# Typical profile: Spill
# Expected behavior: observe NVMe QoS impact with tiered KV; compare against H4A control.
# See docs/Test_Plan_v3.3.md, section H4B (8.4B).
set -euo pipefail

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackB"
PROFILE=${PROFILE:-spill}  # Spill profile to keep 4k context with sustained tier2 traffic
apply_profile_env "$PROFILE"
# Default to Llama 3.3 70B NVFP4; override MODEL env to switch.
MODEL=${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
MODEL_TAG=${MODEL_TAG:-$(model_tag "$MODEL")}
WORKLOAD="fixed_context"
# Default to 2k unless a high-context engine is advertised via STACKB_MAX_INPUT_LEN (>=4k).
CTX_DEFAULT=2048
if [[ "${STACKB_MAX_INPUT_LEN:-}" =~ ^[0-9]+$ ]] && [[ "${STACKB_MAX_INPUT_LEN}" -ge 4096 ]]; then
  CTX_DEFAULT=4096
fi
CTX=${H4B_CONTEXT_TOKENS:-$CTX_DEFAULT}
CONCURRENCY=32
DURATION=180
ENDPOINT=${ENDPOINT:-http://127.0.0.1:9000/v1/completions}
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}

# Quick dependency sanity check so we fail fast before creating run dirs.
python3 - <<'PY'
import importlib
for mod in ("httpx", "yaml"):
    importlib.import_module(mod)
PY

for PHASE in baseline qos_degraded; do
  RUN_ID="$(rt_ts)_H4B_${PROFILE}_${PHASE}_${STACK}_${MODEL_TAG}"
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

  # Optional QoS degradation during qos_degraded phase: inject background I/O on the tier2 path.
  qos_pid=""
  if [[ "$PHASE" == "qos_degraded" ]]; then
    QOS_PATH=${QOS_PATH:-${DYN_KVBM_TIER2_PATH:-/nvme/kvbm/l70b}}
    QOS_FILE=${QOS_FILE:-"$QOS_PATH/.qos_noise"}
    QOS_SIZE=${QOS_SIZE:-8G}
    # Default to a realistic mixed workload: 70% randread / 30% randwrite, 16k, iodepth 32, 4 jobs.
    QOS_IODEPTH=${QOS_IODEPTH:-32}
    QOS_BS=${QOS_BS:-16k}
    QOS_RW=${QOS_RW:-randrw}
    QOS_RWMIXREAD=${QOS_RWMIXREAD:-70}
    QOS_JOBS=${QOS_JOBS:-4}
    mkdir -p "$QOS_PATH"
    # Run fio in the background for the duration; stdout/stderr go to the run dir.
    fio --name=qos_noise \
        --directory="$QOS_PATH" \
        --filename=".qos_noise" \
        --size="$QOS_SIZE" \
        --rw="$QOS_RW" \
        --rwmixread="$QOS_RWMIXREAD" \
        --direct=1 \
        --ioengine=libaio \
        --iodepth="$QOS_IODEPTH" \
        --bs="$QOS_BS" \
        --numjobs="$QOS_JOBS" \
        --time_based --runtime="$DURATION" --group_reporting \
        >"$RUN_DIR/qos_fio.log" 2>&1 &
    qos_pid=$!
    echo "$qos_pid" >"$RUN_DIR/qos_fio.pid"
  fi

  start_sysmon "$RUN_DIR" "B"
  start_telemetry "$RUN_DIR"
  run_status=0
  python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR" || run_status=$?
  stop_sysmon "$RUN_DIR"
  stop_telemetry "$RUN_DIR"
  if [[ -n "$qos_pid" ]]; then
    kill "$qos_pid" 2>/dev/null || true
  fi
  if [[ $run_status -ne 0 ]]; then
    echo "loadgen_failed" >"$RUN_DIR/FAILED"
    exit "$run_status"
  fi
done

#!/usr/bin/env bash
set -euo pipefail

# Runner helper (Test_Plan_v3.3 Section 8 scaffolding)

rt_ts() { date -u +"%Y%m%d_%H%M%S"; }

ensure_run_dir() {
  local dir=$1
  mkdir -p "$dir"
}

model_tag() {
  local name=$1
  echo "${name//\//-}"
}

record_pid() {
  local pid=$1 run_dir=$2
  echo "$pid" >>"$run_dir/telemetry.pids"
}

start_sysmon() {
  local run_dir=$1 stack_tag=$2
  bash "$HARNESS_DIR/src/telemetry/sysmon.sh" "$run_dir" "$stack_tag" >"$run_dir/sysmon.log" 2>&1 &
  echo $! > "$run_dir/sysmon.pid"
}

stop_sysmon() {
  local run_dir=$1
  if [[ -f "$run_dir/sysmon.pid" ]]; then
    kill "$(cat "$run_dir/sysmon.pid")" 2>/dev/null || true
  fi
}

start_telemetry() {
  local run_dir=$1
  local interval=${TELEMETRY_INTERVAL:-0.2} # 200 ms default per v3.3
  local nvme_dev=${NVME_DEVICE:-nvme0n1}
  local metrics_port=${DYN_KVBM_METRICS_PORT:-6880}
  : >"$run_dir/telemetry.pids"

  # NVMe
  python3 "$HARNESS_DIR/src/telemetry/nvme_iostat.py" \
    --device "$nvme_dev" \
    --interval-seconds "$interval" \
    --output "$run_dir/nvme.jsonl" \
    >"$run_dir/nvme.log" 2>&1 &
  record_pid $! "$run_dir"

  # GPU/UMA
  python3 "$HARNESS_DIR/src/telemetry/gpu_poll.py" \
    --interval-seconds "$interval" \
    --output "$run_dir/gpu.jsonl" \
    >"$run_dir/gpu.log" 2>&1 &
  record_pid $! "$run_dir"

  # Dynamo KV metrics (Prometheus endpoint exposed when DYN_KVBM_METRICS=true)
  python3 "$HARNESS_DIR/src/telemetry/dynkv_ingest.py" \
    --url "http://127.0.0.1:${metrics_port}/metrics" \
    --interval-seconds "$interval" \
    --output "$run_dir/dynkv.jsonl" \
    >"$run_dir/dynkv.log" 2>&1 &
  record_pid $! "$run_dir"
}

stop_telemetry() {
  local run_dir=$1
  if [[ -f "$run_dir/telemetry.pids" ]]; then
    while read -r pid; do
      kill "$pid" 2>/dev/null || true
    done <"$run_dir/telemetry.pids"
  fi
}

# Apply Stack B profile defaults unless already exported by caller.
apply_profile_env() {
  local profile=${1:-comfy}
  local tier_env_script="${HARNESS_DIR:-}/scripts/stackB_tier_env.py"

  if [[ -f "$tier_env_script" ]]; then
    if env_output=$(python3 "$tier_env_script" --profile "$profile"); then
      eval "$env_output"
    else
      echo "Warning: failed to load Stack B tier env for profile '${profile}'" >&2
    fi
  fi

  case "$profile" in
    comfy)
      export STACKB_PROFILE=${STACKB_PROFILE:-comfy}
      export STACKB_MAX_INPUT_LEN=${STACKB_MAX_INPUT_LEN:-4096}
      export STACKB_MAX_SEQ_LEN=${STACKB_MAX_SEQ_LEN:-16384}
      export STACKB_MAX_NUM_TOKENS=${STACKB_MAX_NUM_TOKENS:-16384}
      export DYN_KVBM_TIER0_BYTES=${DYN_KVBM_TIER0_BYTES:-$((8 * 1024**3))}
      export DYN_KVBM_TIER1_BYTES=${DYN_KVBM_TIER1_BYTES:-$((8 * 1024**3))}
      export DYN_KVBM_TIER2_BYTES=${DYN_KVBM_TIER2_BYTES:-$((256 * 1024**3))}
      ;;
    spill)
      export STACKB_PROFILE=${STACKB_PROFILE:-spill}
      export STACKB_MAX_INPUT_LEN=${STACKB_MAX_INPUT_LEN:-4096}
      export STACKB_MAX_SEQ_LEN=${STACKB_MAX_SEQ_LEN:-16384}
      export STACKB_MAX_NUM_TOKENS=${STACKB_MAX_NUM_TOKENS:-32000}
      export DYN_KVBM_TIER0_BYTES=${DYN_KVBM_TIER0_BYTES:-$((2 * 1024**3))}
      export DYN_KVBM_TIER1_BYTES=${DYN_KVBM_TIER1_BYTES:-$((4 * 1024**3))}
      export DYN_KVBM_TIER2_BYTES=${DYN_KVBM_TIER2_BYTES:-$((64 * 1024**3))}
      ;;
    stress)
      export STACKB_PROFILE=${STACKB_PROFILE:-stress}
      export STACKB_MAX_INPUT_LEN=${STACKB_MAX_INPUT_LEN:-8192}
      export STACKB_MAX_SEQ_LEN=${STACKB_MAX_SEQ_LEN:-16384}
      export STACKB_MAX_NUM_TOKENS=${STACKB_MAX_NUM_TOKENS:-32000}
      export DYN_KVBM_TIER0_BYTES=${DYN_KVBM_TIER0_BYTES:-$((512 * 1024**2))}
      export DYN_KVBM_TIER1_BYTES=${DYN_KVBM_TIER1_BYTES:-$((1 * 1024**3))}
      export DYN_KVBM_TIER2_BYTES=${DYN_KVBM_TIER2_BYTES:-$((128 * 1024**3))}
      ;;
    *)
      echo "Unknown profile '$profile' (expected comfy|spill|stress)" >&2
      return 1
      ;;
  esac
  export DYN_KVBM_KV_BLOCK_SIZE_BYTES=${DYN_KVBM_KV_BLOCK_SIZE_BYTES:-65536}
  export DYN_KVBM_TIER2_PATH=${DYN_KVBM_TIER2_PATH:-/nvme/kvbm/l70b}
  export DYN_KVBM_METRICS_PORT=${DYN_KVBM_METRICS_PORT:-6880}
}

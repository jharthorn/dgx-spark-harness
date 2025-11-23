#!/usr/bin/env bash
set -euo pipefail

# Runner helper (Test_Plan_v3.0 Section 8 scaffolding)

rt_ts() { date -u +"%Y%m%d_%H%M%S"; }

ensure_run_dir() {
  local dir=$1
  mkdir -p "$dir"
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
  local interval=${TELEMETRY_INTERVAL:-0.2}
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

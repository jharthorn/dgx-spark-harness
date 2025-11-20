#!/usr/bin/env bash
set -euo pipefail

# Runner helper (Test_Plan_v3.0 Section 8 scaffolding)

rt_ts() { date -u +"%Y%m%d_%H%M%S"; }

ensure_run_dir() {
  local dir=$1
  mkdir -p "$dir"
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

start_dynkv() {
  local run_dir=$1
  # TODO: replace with real Dynamo/KVBM telemetry ingestion.
  echo "{}" > "$run_dir/dynkv.jsonl"
}

stop_dynkv() {
  local run_dir=$1
  return 0
}

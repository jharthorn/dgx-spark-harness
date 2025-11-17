#!/usr/bin/env bash
set -euo pipefail

# v3 runner helper (Test_Plan_v3.0 Section 8 scaffolding)

rt_ts() { date -u +"%Y%m%d_%H%M%S"; }

ensure_run_dir() {
  local dir=$1
  mkdir -p "$dir"
}

start_sysmon() {
  local run_dir=$1 stack_tag=$2
  bash "$HARNESS_DIR/src/telemetry/sysmon_v3.sh" "$run_dir" "$stack_tag" >/dev/null 2>&1 &
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
  : > "$run_dir/dynkv.jsonl"
  ( while :; do echo "{\"ts\":\"$(date -Is)\",\"tier\":\"nvme\",\"iops\":0}" >> "$run_dir/dynkv.jsonl"; sleep 2; done ) &
  echo $! > "$run_dir/dynkv.pid"
}

stop_dynkv() {
  local run_dir=$1
  if [[ -f "$run_dir/dynkv.pid" ]]; then
    kill "$(cat "$run_dir/dynkv.pid")" 2>/dev/null || true
  fi
}

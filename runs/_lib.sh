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
  local tier_path=${DYN_KVBM_TIER2_PATH:-/nvme/kvbm}
  local interval=${DYN_KV_TELEMETRY_INTERVAL:-5}

  (
    while :; do
      ts=$(date -Ins)
      size_bytes=$(du -sb "$tier_path" 2>/dev/null | awk '{print $1+0}')
      if [[ -z "$size_bytes" ]]; then size_bytes=0; fi
      df_line=$(df -B1 "$tier_path" 2>/dev/null | awk 'NR==2{print $2,$3,$4}')
      read -r fs_size fs_used fs_avail <<<"${df_line:-0 0 0}"
      cat <<EOF >> "$run_dir/dynkv.jsonl"
{"ts":"${ts}","tier_path":"${tier_path}","bytes_used":${size_bytes},"fs_total_bytes":${fs_size:-0},"fs_used_bytes":${fs_used:-0},"fs_avail_bytes":${fs_avail:-0}}
EOF
      sleep "$interval"
    done
  ) >"$run_dir/dynkv.log" 2>&1 &
  echo $! > "$run_dir/dynkv.pid"
}

stop_dynkv() {
  local run_dir=$1
  if [[ -f "$run_dir/dynkv.pid" ]]; then
    kill "$(cat "$run_dir/dynkv.pid")" 2>/dev/null || true
  fi
}

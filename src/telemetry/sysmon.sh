#!/usr/bin/env bash
# sysmon_v3.sh -- Test_Plan_v3.3 Section 6.4
# Collects host + GPU + NVMe stats into sysmon.jsonl for v3 runs (Stack A/B) at ~200 ms cadence.
# Usage: sysmon_v3.sh <run_dir> <stack_tag>
set -euo pipefail

RUN_DIR=${1:?run dir required}
STACK_TAG=${2:-A}
OUT="$RUN_DIR/sysmon.jsonl"
NVME_DEVICE=${NVME_DEVICE:-nvme0n1}

# TODO: add per-GPU metrics when multiple devices are present; capture GPU memory bandwidth if available.
# TODO: add network telemetry (bytes/pps) and NUMA locality markers for KV pressure runs.
# TODO: add NVMe namespace detection instead of fixed ${NVME_DEVICE}.
SAMPLE_INTERVAL=${SYS_SAMPLE_INTERVAL:-${TELEMETRY_INTERVAL:-0.2}}

MPSTAT_OK=0
IOSTAT_OK=0
NVIDIA_OK=0
command -v mpstat >/dev/null 2>&1 && MPSTAT_OK=1 || true
command -v iostat >/dev/null 2>&1 && IOSTAT_OK=1 || true
command -v nvidia-smi >/dev/null 2>&1 && NVIDIA_OK=1 || true

mkdir -p "$RUN_DIR"
: > "$OUT"

ts() { date -Ins; }
ts_float() { date +%s.%N; }

sanitize_num() {
  local v=$1
  if [[ "$v" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
    printf "%s" "$v"
  else
    printf "0"
  fi
}

collect_sample() {
  local TS=${TS_NOW:-$(ts_float)}
  local usr sys iowait memAvail memCached swapTotal swapFree rps wps rMB wMB await avgrq avgqu util gpu_util gpu_mem_used gpu_mem_total

  if [[ $MPSTAT_OK -eq 1 ]]; then
    mp_line=$(mpstat 1 1 2>/dev/null | awk '/Average:/ {print $3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13}') || mp_line=""
  else
    mp_line=""
  fi
  read -r usr nice sys iowait irq softirq steal guest guestnice idle <<<"${mp_line:-0 0 0 0 0 0 0 0 0 0 0}"

  read -r memAvail memCached swapTotal swapFree < <(awk '/MemAvailable:/{a=$2}/^Cached:/{c=$2}/SwapTotal:/{s=$2}/SwapFree:/{f=$2} END{print a,c,s,f}' /proc/meminfo)

  if [[ $IOSTAT_OK -eq 1 ]]; then
    io_line=$(iostat -x -m 1 2 2>/dev/null | awk -v dev="${NVME_DEVICE}" 'tolower($1)==tolower(dev){rps=$4;wps=$5;rMB=$6;wMB=$7;await=$10;avgrq=$9;avgqu=$8;util=$NF} END{print rps,wps,rMB,wMB,await,avgrq,avgqu,util}') || io_line=""
  else
    io_line=""
  fi
  read -r rps wps rMB wMB await avgrq avgqu util <<<"${io_line:-0 0 0 0 0 0 0 0}"

  gpu_util=0
  gpu_mem_used=0
  gpu_mem_total=0
  if [[ $NVIDIA_OK -eq 1 ]]; then
    # Primary: query util/mem via nvidia-smi; some GB/UMA platforms report mem as N/A.
    gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' %MiB') || gpu_line=""
    read -r gpu_util gpu_mem_used gpu_mem_total <<<"${gpu_line:-0 0 0}"
    gpu_util=$(sanitize_num "$gpu_util")
    # Fallback: if util is zero and mem is N/A/0 (UMA reporting), try dmon for util only.
    if [[ "$gpu_util" == "0" ]]; then
      dmon_line=$(nvidia-smi dmon -s u -c 1 2>/dev/null | awk 'NR==3{print $2}') || dmon_line=""
      gpu_util=$(sanitize_num "${dmon_line:-$gpu_util}")
    fi
    gpu_mem_used=$(sanitize_num "$gpu_mem_used")
    gpu_mem_total=$(sanitize_num "$gpu_mem_total")
  fi
  usr=$(sanitize_num "$usr"); sys=$(sanitize_num "$sys"); iowait=$(sanitize_num "$iowait")
  memAvail=$(sanitize_num "$memAvail"); memCached=$(sanitize_num "$memCached"); swapTotal=$(sanitize_num "$swapTotal"); swapFree=$(sanitize_num "$swapFree")
  rps=$(sanitize_num "$rps"); rMB=$(sanitize_num "$rMB"); await=$(sanitize_num "$await"); avgqu=$(sanitize_num "$avgqu"); util=$(sanitize_num "$util")

  cat <<EOF >> "$OUT"
{"ts":${TS},"stack":"${STACK_TAG}","cpu":{"user":${usr:-0},"system":${sys:-0},"iowait":${iowait:-0}},"mem":{"MemAvailable":${memAvail:-0},"Cached":${memCached:-0},"SwapTotal":${swapTotal:-0},"SwapFree":${swapFree:-0}},"nvme":{"rps":${rps:-0},"rMBs":${rMB:-0},"r_await_ms":${await:-0},"aqu_sz":${avgqu:-0},"util_pct":${util:-0}},"gpu":{"util":${gpu_util:-0},"mem_used_bytes":$(( (${gpu_mem_used:-0}) * 1024 * 1024 ))}}
EOF
}

trap 'exit 0' INT TERM

while :; do
  TS_NOW=$(ts_float)
  collect_sample
  sleep "$SAMPLE_INTERVAL"
done

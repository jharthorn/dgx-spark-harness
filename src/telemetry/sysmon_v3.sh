#!/usr/bin/env bash
set -euo pipefail

# sysmon_v3.sh -- Test_Plan_v3.0 Section 6.4
# Collects host + GPU + NVMe stats into sysmon.jsonl for v3 runs (Stack A/B).
# Usage: sysmon_v3.sh <run_dir> <stack_tag>

RUN_DIR=${1:?run dir required}
STACK_TAG=${2:-A}
OUT="$RUN_DIR/sysmon.jsonl"

CPU_INTERVAL=0.5
IOSTAT_INTERVAL=0.5
NVIDIA_INTERVAL=1

mkdir -p "$RUN_DIR"
: > "$OUT"

ts() { date -Ins; }

collect_sample() {
  local TS
  TS=$(ts)

  # mpstat single sample
  read -r _ _ usr nice sys iowait irq softirq steal guest guestnice idle < <(mpstat 1 1 | awk '/Average:/ {print $3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13}')
  # meminfo
  read -r memAvail memCached swapTotal swapFree < <(awk '/MemAvailable:/{a=$2}/^Cached:/{c=$2}/SwapTotal:/{s=$2}/SwapFree:/{f=$2} END{print a,c,s,f}' /proc/meminfo)
  # iostat
  read -r rps wps rMB wMB await avgrq avgqu util < <(iostat -x -m 1 2 | awk 'tolower($1)==tolower(dev){rps=$4;wps=$5;rMB=$6;wMB=$7;await=$10;avgrq=$9;avgqu=$8;util=$NF} END{print rps,wps,rMB,wMB,await,avgrq,avgqu,util}' dev="${NVME_DEVICE:-nvme0n1}")
  # gpu
  read -r gpu_util gpu_mem_used gpu_mem_total < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' %MiB')

  cat <<EOF >> "$OUT"
{"ts":"${TS}","stack":"${STACK_TAG}","cpu":{"user":${usr:-0},"system":${sys:-0},"iowait":${iowait:-0}},"mem":{"MemAvailable":${memAvail:-0},"Cached":${memCached:-0},"SwapTotal":${swapTotal:-0},"SwapFree":${swapFree:-0}},"nvme":{"rps":${rps:-0},"rMBs":${rMB:-0},"r_await_ms":${await:-0},"aqu_sz":${avgqu:-0},"util_pct":${util:-0}},"gpu":{"util":${gpu_util:-0},"mem_used_bytes":$(( (${gpu_mem_used:-0}) * 1024 * 1024 ))}}
EOF
}

trap 'exit 0' INT TERM

while :; do
  collect_sample
  sleep "$CPU_INTERVAL"
done

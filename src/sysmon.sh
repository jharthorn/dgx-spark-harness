#!/usr/bin/env bash
set -euo pipefail

# src/sysmon.sh
# DGX Spark telemetry collector (1 Hz)
# Outputs a single, wide CSV with the schema required by v1.3.
# v2.2: Added robustness to commands

# ---- Paths & env ----
HARNESS_DIR=${HARNESS_DIR:-/harness}
RESULTS_DIR=${RESULTS_DIR:-$HARNESS_DIR/results}
mkdir -p "$RESULTS_DIR"

RUN_ID=${RUN_ID:?set RUN_ID}
DEVICE=${NVME_DEVICE:-nvme0n1}
OUT_FILE="$RESULTS_DIR/${RUN_ID}_telemetry.csv"

export TZ=UTC

# ---- t0 (monotonic-ish) ----
T0_MS=$(date +%s%3N)

# ---- NVMe temp path (best-effort) ----
find_nvme_temp_path() {
  for H in /sys/class/hwmon/hwmon*; do
    [[ -r "$H/name" ]] || continue
    if grep -qi nvme "$H/name"; then
      for S in "$H"/temp*_input; do
        [[ -r "$S" ]] && { echo "$S"; return; }
      done
    fi
  done
  echo ""
}
NVME_TEMP_PATH=$(find_nvme_temp_path)

# ---- CSV header ----
printf '%s\n' \
  'ms_since_t0,vm_r,vm_b,vm_us,vm_sy,vm_id,vm_wa,mem_MemAvailable_kB,mem_Cached_kB,iostat_rps,iostat_wps,iostat_rMBps,iostat_wMBps,iostat_await_ms,iostat_avg_rqsz,iostat_avg_qu_sz,iostat_util_pct,gpu_util_pct,gpu_mem_used_MiB,gpu_mem_total_MiB,ssd_temp_C' \
  > "$OUT_FILE"

now_ms() { date +%s%3N; }

read_vmstat() {
  # vmstat -n 1 2 prints two samples; take the 2nd line of data
  vmstat -n 1 2 | awk 'NR==4{print $1","$2","$13","$14","$15","$16}'
}

read_meminfo() {
  awk '/MemAvailable:/{a=$2}/^Cached:/{c=$2} END{printf "%s,%s", a,c}' /proc/meminfo
}

read_iostat() {
  # iostat -x -m 1 2 for the device, grab the last line matching DEVICE
  iostat -x -m 1 2 | awk -v dev="${DEVICE}" 'tolower($1)==tolower(dev){rps=$4; wps=$5; rMB=$6; wMB=$7; await=$10; avgrq=$9; avgqu=$8; util=$NF; out=rps","wps","rMB","wMB","await","avgrq","avgqu","util} END{if(out!="") print out; else print "NA,NA,NA,NA,NA,NA,NA,NA";}'
}

read_gpu() {
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
  | head -n1 | tr -d ' %MiB'
}

read_nvme_temp() {
  if [[ -n "$NVME_TEMP_PATH" && -r "$NVME_TEMP_PATH" ]]; then
    awk '{printf "%.1f", $1/1000}' "$NVME_TEMP_PATH"
  else
    printf "NA"
  fi
}

trap 'exit 0' INT TERM
while :; do
  TS_MS=$(now_ms); MS_SINCE=$(( TS_MS - T0_MS ))
  
  # Protect against commands failing
  VM=$(read_vmstat || echo "NA,NA,NA,NA,NA,NA")
  MEM=$(read_meminfo || echo "NA,NA")
  IO=$(read_iostat || echo "NA,NA,NA,NA,NA,NA,NA,NA")
  GPU=$(read_gpu || echo "NA,NA,NA")
  TEMP=$(read_nvme_temp || echo "NA")
  
  printf '%s,%s,%s,%s,%s\n' "$MS_SINCE" "$VM" "$MEM" "$IO" "$GPU,$TEMP" >> "$OUT_FILE"
  
  # Sleep to maintain ~1 Hz
  sleep 1
done
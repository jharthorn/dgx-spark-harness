#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: start_gpu_dmon.sh RUN_DIR [INTERVAL_S]}
INTERVAL_S=${2:-1}

OUT_DIR="${RUN_DIR}/telemetry"
PID_FILE="${OUT_DIR}/gpu_dmon.pid"
LOG_FILE="${OUT_DIR}/nvidia_smi_dmon.log"
SNAPSHOT_FILE="${OUT_DIR}/nvidia_smi_snapshot.txt"

mkdir -p "${OUT_DIR}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "gpu dmon already running (pid=$(cat "${PID_FILE}"))"
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; skipping GPU collector." >&2
  exit 0
fi

nvidia-smi >"${SNAPSHOT_FILE}" 2>&1 || true

if nvidia-smi dmon -s pucvmet -d "${INTERVAL_S}" -c 1 >/dev/null 2>&1; then
  nohup bash -lc "exec nvidia-smi dmon -s pucvmet -d ${INTERVAL_S}" >"${LOG_FILE}" 2>&1 &
  echo $! >"${PID_FILE}"
  echo "started nvidia-smi dmon pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
else
  echo "nvidia-smi dmon unavailable; kept one-shot snapshot at ${SNAPSHOT_FILE}" >&2
fi


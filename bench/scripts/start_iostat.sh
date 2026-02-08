#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: start_iostat.sh RUN_DIR [DEVICE] [INTERVAL_S]}
DEVICE=${2:-nvme0n1}
INTERVAL_S=${3:-1}

OUT_DIR="${RUN_DIR}/telemetry"
PID_FILE="${OUT_DIR}/iostat.pid"
LOG_FILE="${OUT_DIR}/iostat_${DEVICE}.log"

mkdir -p "${OUT_DIR}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "iostat already running (pid=$(cat "${PID_FILE}"))"
  exit 0
fi

if ! command -v iostat >/dev/null 2>&1; then
  echo "iostat not found; skipping collector." >&2
  exit 0
fi

nohup bash -lc "exec iostat -x -d ${DEVICE} ${INTERVAL_S}" >"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"
echo "started iostat pid=$(cat "${PID_FILE}") log=${LOG_FILE}"


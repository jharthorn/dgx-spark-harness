#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: start_pidstat.sh RUN_DIR [PID|ALL] [INTERVAL_S]}
PID_TARGET=${2:-ALL}
INTERVAL_S=${3:-1}

OUT_DIR="${RUN_DIR}/telemetry"
PID_FILE="${OUT_DIR}/pidstat.pid"
LOG_FILE="${OUT_DIR}/pidstat.log"

mkdir -p "${OUT_DIR}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "pidstat already running (pid=$(cat "${PID_FILE}"))"
  exit 0
fi

if ! command -v pidstat >/dev/null 2>&1; then
  echo "pidstat not found; skipping collector." >&2
  exit 0
fi

nohup bash -lc "exec pidstat -d -r -u -h -p ${PID_TARGET} ${INTERVAL_S}" >"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"
echo "started pidstat pid=$(cat "${PID_FILE}") log=${LOG_FILE}"


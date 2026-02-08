#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: stop_gpu_dmon.sh RUN_DIR}
PID_FILE="${RUN_DIR}/telemetry/gpu_dmon.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "no gpu_dmon pid file"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" >/dev/null 2>&1; then
  kill "${PID}" >/dev/null 2>&1 || true
  wait "${PID}" 2>/dev/null || true
  echo "stopped gpu dmon pid=${PID}"
else
  echo "gpu dmon pid ${PID} not running"
fi
rm -f "${PID_FILE}"


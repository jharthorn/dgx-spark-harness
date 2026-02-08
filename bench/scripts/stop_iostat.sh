#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: stop_iostat.sh RUN_DIR}
PID_FILE="${RUN_DIR}/telemetry/iostat.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "no iostat pid file"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" >/dev/null 2>&1; then
  kill "${PID}" >/dev/null 2>&1 || true
  wait "${PID}" 2>/dev/null || true
  echo "stopped iostat pid=${PID}"
else
  echo "iostat pid ${PID} not running"
fi
rm -f "${PID_FILE}"


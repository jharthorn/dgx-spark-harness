#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: stop_pidstat.sh RUN_DIR}
PID_FILE="${RUN_DIR}/telemetry/pidstat.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "no pidstat pid file"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" >/dev/null 2>&1; then
  kill "${PID}" >/dev/null 2>&1 || true
  wait "${PID}" 2>/dev/null || true
  echo "stopped pidstat pid=${PID}"
else
  echo "pidstat pid ${PID} not running"
fi
rm -f "${PID_FILE}"


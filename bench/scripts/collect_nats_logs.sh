#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: collect_nats_logs.sh RUN_DIR [NATS_CONTAINER]}
NATS_CONTAINER=${2:-${BENCH_NATS_CONTAINER_NAME:-bench-nats}}
NATS_LOG_FILE="${BENCH_NATS_LOG_FILE:-/tmp/bench-logs/nats.log}"

OUT_DIR="${RUN_DIR}/telemetry"
mkdir -p "${OUT_DIR}"

if [[ -f "${NATS_LOG_FILE}" ]]; then
  cp -f "${NATS_LOG_FILE}" "${OUT_DIR}/nats_server.log"
  echo "copied nats server log from ${NATS_LOG_FILE}"
else
  echo "nats log file not found: ${NATS_LOG_FILE}" >&2
fi

if command -v docker >/dev/null 2>&1; then
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${NATS_CONTAINER}"; then
    docker logs --timestamps "${NATS_CONTAINER}" >"${OUT_DIR}/docker_${NATS_CONTAINER}_logs.txt" 2>&1 || true
    docker inspect "${NATS_CONTAINER}" >"${OUT_DIR}/docker_${NATS_CONTAINER}_inspect.json" 2>/dev/null || true
    echo "collected docker logs for ${NATS_CONTAINER}"
  fi
fi

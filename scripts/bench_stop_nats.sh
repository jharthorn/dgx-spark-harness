#!/usr/bin/env bash
set -euo pipefail

NATS_CONTAINER_NAME="${BENCH_NATS_CONTAINER_NAME:-bench-nats}"
NATS_LOG_DIR="${BENCH_NATS_LOG_DIR:-/tmp/bench-logs}"
NATS_DOCKER_LOG_FILE="${NATS_DOCKER_LOG_FILE:-${NATS_LOG_DIR}/nats_docker.log}"

mkdir -p "${NATS_LOG_DIR}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${NATS_CONTAINER_NAME}"; then
  docker logs --timestamps "${NATS_CONTAINER_NAME}" >"${NATS_DOCKER_LOG_FILE}" 2>&1 || true
  docker rm -f "${NATS_CONTAINER_NAME}" >/dev/null
  echo "NATS stopped: container=${NATS_CONTAINER_NAME}"
  echo "Captured docker logs: ${NATS_DOCKER_LOG_FILE}"
else
  echo "NATS container not present: ${NATS_CONTAINER_NAME}"
fi

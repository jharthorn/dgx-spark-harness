#!/usr/bin/env bash
set -euo pipefail

NATS_CONTAINER_NAME="${BENCH_NATS_CONTAINER_NAME:-bench-nats}"
NATS_IMAGE="${BENCH_NATS_IMAGE:-nats:2.10-alpine}"
NATS_HOST="${BENCH_NATS_HOST:-127.0.0.1}"
NATS_CLIENT_PORT="${BENCH_NATS_CLIENT_PORT:-4222}"
NATS_MONITOR_PORT="${BENCH_NATS_MONITOR_PORT:-8222}"
NATS_STORE_DIR="${BENCH_NATS_STORE_DIR:-/tmp/bench-logs/nats-store}"
NATS_LOG_DIR="${BENCH_NATS_LOG_DIR:-/tmp/bench-logs}"
NATS_LOG_BASENAME="${BENCH_NATS_LOG_BASENAME:-nats.log}"
NATS_LOG_FILE="${NATS_LOG_DIR}/${NATS_LOG_BASENAME}"
NATS_SERVER_URL="${BENCH_NATS_SERVER:-${NATS_SERVER:-nats://${NATS_HOST}:${NATS_CLIENT_PORT}}}"

mkdir -p "${NATS_STORE_DIR}" "${NATS_LOG_DIR}"
touch "${NATS_LOG_FILE}"

docker rm -f "${NATS_CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d --name "${NATS_CONTAINER_NAME}" \
  --network host \
  -v "${NATS_STORE_DIR}:/nats-store" \
  -v "${NATS_LOG_DIR}:/nats-logs" \
  "${NATS_IMAGE}" \
  -js \
  -a 0.0.0.0 \
  -p "${NATS_CLIENT_PORT}" \
  -m "${NATS_MONITOR_PORT}" \
  -sd /nats-store \
  -l "/nats-logs/${NATS_LOG_BASENAME}" \
  >/dev/null

echo "NATS started: container=${NATS_CONTAINER_NAME}"
echo "NATS server URL: ${NATS_SERVER_URL}"
echo "NATS monitor URL: http://${NATS_HOST}:${NATS_MONITOR_PORT}"
echo "NATS log file: ${NATS_LOG_FILE}"
docker ps --filter "name=${NATS_CONTAINER_NAME}"

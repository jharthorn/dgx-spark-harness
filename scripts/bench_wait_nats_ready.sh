#!/usr/bin/env bash
set -euo pipefail

TIMEOUT_S="${1:-120}"
SLEEP_S="${2:-2}"
NATS_HOST="${BENCH_NATS_HOST:-127.0.0.1}"
NATS_CLIENT_PORT="${BENCH_NATS_CLIENT_PORT:-4222}"
NATS_MONITOR_PORT="${BENCH_NATS_MONITOR_PORT:-8222}"

check_client_port() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 1 bash -c ":</dev/tcp/${NATS_HOST}/${NATS_CLIENT_PORT}" >/dev/null 2>&1
  else
    bash -c ":</dev/tcp/${NATS_HOST}/${NATS_CLIENT_PORT}" >/dev/null 2>&1
  fi
}

check_monitor_health() {
  curl -sf "http://${NATS_HOST}:${NATS_MONITOR_PORT}/healthz" >/dev/null 2>&1
}

check_jetstream_enabled() {
  curl -sf "http://${NATS_HOST}:${NATS_MONITOR_PORT}/varz" | jq -e '.jetstream != null' >/dev/null 2>&1
}

deadline=$((SECONDS + TIMEOUT_S))
while (( SECONDS < deadline )); do
  port_ok=0
  monitor_ok=0
  jetstream_ok=0
  if check_client_port; then
    port_ok=1
  fi
  if check_monitor_health; then
    monitor_ok=1
  fi
  if check_jetstream_enabled; then
    jetstream_ok=1
  fi
  if [[ "${port_ok}" == "1" && "${monitor_ok}" == "1" && "${jetstream_ok}" == "1" ]]; then
    echo "NATS ready: nats://${NATS_HOST}:${NATS_CLIENT_PORT} (monitor=http://${NATS_HOST}:${NATS_MONITOR_PORT})"
    exit 0
  fi
  echo "Waiting for NATS readiness (client_port=${port_ok}, monitor_health=${monitor_ok}, jetstream=${jetstream_ok})..."
  sleep "${SLEEP_S}"
done

echo "Timeout waiting for NATS readiness: nats://${NATS_HOST}:${NATS_CLIENT_PORT}" >&2
curl -s "http://${NATS_HOST}:${NATS_MONITOR_PORT}/healthz" || true
curl -s "http://${NATS_HOST}:${NATS_MONITOR_PORT}/varz" | jq . || true
exit 1

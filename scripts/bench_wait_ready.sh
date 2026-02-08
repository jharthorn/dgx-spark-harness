#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
TIMEOUT_S="${1:-300}"
SLEEP_S="${2:-5}"

deadline=$((SECONDS + TIMEOUT_S))
while (( SECONDS < deadline )); do
  health="$(curl -sf "${BASE_URL}/health" 2>/dev/null || true)"
  models="$(curl -sf "${BASE_URL}/v1/models" 2>/dev/null || true)"
  model_count="$(jq -r '.data | length' <<<"${models:-{\"data\":[]}}" 2>/dev/null || echo 0)"
  if [[ -n "${health}" && "${model_count}" != "0" ]]; then
    echo "Ready: ${BASE_URL} (models=${model_count})"
    exit 0
  fi
  echo "Waiting for ${BASE_URL} models>0 (health/model discovery not ready yet)..."
  sleep "${SLEEP_S}"
done

echo "Timeout waiting for readiness at ${BASE_URL}" >&2
curl -s "${BASE_URL}/health" || true
curl -s "${BASE_URL}/v1/models" || true
exit 1


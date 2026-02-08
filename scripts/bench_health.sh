#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
KVBM_METRICS_URL="${KVBM_METRICS_URL:-http://127.0.0.1:6880/metrics}"

echo "== health =="
curl -s "${BASE_URL}/health" | jq .
echo
echo "== models =="
curl -s "${BASE_URL}/v1/models" | jq .
echo
echo "== kvbm metrics (head, best effort) =="
curl -s "${KVBM_METRICS_URL}" | head -n 20 || true

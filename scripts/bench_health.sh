#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"

echo "== health =="
curl -s "${BASE_URL}/health" | jq .
echo
echo "== models =="
curl -s "${BASE_URL}/v1/models" | jq .


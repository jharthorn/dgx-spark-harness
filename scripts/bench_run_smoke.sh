#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
RUN_ID="${1:-smoke_short_c1_$(date -u +%Y%m%dT%H%M%SZ)}"

python3 -m bench.run_bench \
  --base-url "${BASE_URL}" \
  --scenario standard \
  --prompt-set short \
  --requests 4 \
  --warmup 1 \
  --concurrency 1 \
  --max-tokens 64 \
  --temperature 0.2 \
  --stop "<|eot_id|>" \
  --collect-telemetry \
  --container-name "${BENCH_CONTAINER_NAME:-dyn}" \
  --kvbm-cache-dir "${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}" \
  --run-id "${RUN_ID}"

RUN_DIR="bench/results/${RUN_ID}"
jq '.overall_summary' "${RUN_DIR}/summary.json"


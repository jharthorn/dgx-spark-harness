#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_profile_lib.sh
source "${SCRIPT_DIR}/bench_profile_lib.sh"

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
RUN_ID="${BENCH_RUN_ID:-local_copilot_burst_$(date -u +%Y%m%dT%H%M%SZ)}"
REQUESTS="${BENCH_COPILOT_REQUESTS:-48}"
WARMUP="${BENCH_COPILOT_WARMUP:-6}"
CONCURRENCY="${BENCH_COPILOT_CONCURRENCY:-6}"
SESSION_COUNT="${BENCH_COPILOT_SESSION_COUNT:-8}"
BURST_SIZE="${BENCH_COPILOT_BURST_SIZE:-4}"
SHARED_PREFIX_TOKENS="${BENCH_COPILOT_SHARED_PREFIX_TOKENS:-3072}"

bench_resolve_tier_mode "${BENCH_TIER_MODE:-${BENCH_KV_MODE:-}}"
bench_defaults_for_tier_mode "${BENCH_TIER_MODE_RESOLVED}"

KV_MODE="${BENCH_KV_MODE_RESOLVED}"
TIER_MODE="${BENCH_TIER_MODE_RESOLVED}"
KV_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-${BENCH_CPU_CACHE_GB_DEFAULT}}"
KV_DISK_CACHE_GB="${DYN_KVBM_DISK_CACHE_GB:-${BENCH_DISK_CACHE_GB_DEFAULT}}"

python3 -m bench.run_bench \
  --base-url "${BASE_URL}" \
  --tier-mode "${TIER_MODE}" \
  --kv-mode "${KV_MODE}" \
  --kv-cpu-cache-gb "${KV_CPU_CACHE_GB}" \
  --kv-disk-cache-gb "${KV_DISK_CACHE_GB}" \
  --variant-tag "workload:local_project_copilot_shared_prefix_burst" \
  --variant-tag "tier_mode:${TIER_MODE}" \
  --scenario local_copilot_burst \
  --requests "${REQUESTS}" \
  --warmup "${WARMUP}" \
  --concurrency "${CONCURRENCY}" \
  --copilot-session-count "${SESSION_COUNT}" \
  --copilot-burst-size "${BURST_SIZE}" \
  --copilot-shared-prefix-target-tokens "${SHARED_PREFIX_TOKENS}" \
  --max-tokens "${BENCH_COPILOT_MAX_TOKENS:-192}" \
  --temperature "${BENCH_COPILOT_TEMPERATURE:-0.2}" \
  --seed "${BENCH_COPILOT_SEED:-1337}" \
  --request-seed "${BENCH_COPILOT_REQUEST_SEED:-1337}" \
  --stop "<|eot_id|>" \
  --collect-telemetry \
  --container-name "${BENCH_CONTAINER_NAME:-dyn}" \
  --kvbm-cache-dir "${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}" \
  --run-id "${RUN_ID}"

echo "Completed local_copilot_burst run: bench/results/${RUN_ID}"

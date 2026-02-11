#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_profile_lib.sh
source "${SCRIPT_DIR}/bench_profile_lib.sh"

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
MODES="${BENCH_KV_MODE_LIST:-B0 B1 B2}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
READY_REQUIRE_ENDPOINTS="${BENCH_COMPARE_READY_REQUIRE_ENDPOINTS:-0}"
READY_CONSECUTIVE="${BENCH_COMPARE_READY_CONSECUTIVE:-2}"
READY_TIMEOUT_S="${BENCH_COMPARE_READY_TIMEOUT_S:-300}"
READY_SLEEP_S="${BENCH_COMPARE_READY_SLEEP_S:-5}"
SKIP_READY="${BENCH_COMPARE_SKIP_READY:-0}"
MODEL_RESOLVE_TIMEOUT_S="${BENCH_COMPARE_MODEL_RESOLVE_TIMEOUT_S:-300}"
MODEL_RESOLVE_POLL_S="${BENCH_COMPARE_MODEL_RESOLVE_POLL_S:-2}"
CURRENT_MODE=""

on_error() {
  local rc=$?
  echo "Mode comparison failed (mode=${CURRENT_MODE:-unknown}, rc=${rc})." >&2
  echo "== bench_health ==" >&2
  scripts/bench_health.sh >&2 || true
  echo "== worker log (tail) ==" >&2
  docker exec "${CONTAINER_NAME}" bash -lc "tail -n 120 /tmp/bench-logs/worker.log" >&2 || true
  echo "== frontend log (tail) ==" >&2
  docker exec "${CONTAINER_NAME}" bash -lc "tail -n 120 /tmp/bench-logs/frontend.log" >&2 || true
  exit "${rc}"
}
trap on_error ERR

for mode in ${MODES}; do
  CURRENT_MODE="${mode}"
  bench_resolve_tier_mode "${mode}"
  bench_defaults_for_tier_mode "${BENCH_TIER_MODE_RESOLVED}"

  export BENCH_TIER_MODE="${BENCH_TIER_MODE_RESOLVED}"
  export BENCH_KV_MODE="${BENCH_KV_MODE_RESOLVED}"

  echo "=== Restarting worker/frontend for tier_mode=${BENCH_TIER_MODE_RESOLVED} kv_mode=${BENCH_KV_MODE_RESOLVED} ==="
  scripts/bench_start_worker.sh
  scripts/bench_start_frontend.sh
  if [[ "${SKIP_READY}" == "1" ]]; then
    echo "[ASSUMPTION] Skipping readiness wait for mode=${mode} because BENCH_COMPARE_SKIP_READY=1."
  else
    BENCH_READY_REQUIRE_ENDPOINTS="${READY_REQUIRE_ENDPOINTS}" \
    BENCH_READY_CONSECUTIVE="${READY_CONSECUTIVE}" \
    scripts/bench_wait_ready.sh "${READY_TIMEOUT_S}" "${READY_SLEEP_S}"
  fi

  cpu_cache_gb="${DYN_KVBM_CPU_CACHE_GB:-${BENCH_CPU_CACHE_GB_DEFAULT}}"
  disk_cache_gb="${DYN_KVBM_DISK_CACHE_GB:-${BENCH_DISK_CACHE_GB_DEFAULT}}"

  python3 -m bench.run_bench \
    --base-url "${BASE_URL}" \
    --model-resolve-timeout-s "${MODEL_RESOLVE_TIMEOUT_S}" \
    --model-resolve-poll-s "${MODEL_RESOLVE_POLL_S}" \
    --tier-mode "${BENCH_TIER_MODE_RESOLVED}" \
    --kv-mode "${BENCH_KV_MODE_RESOLVED}" \
    --kv-cpu-cache-gb "${cpu_cache_gb}" \
    --kv-disk-cache-gb "${disk_cache_gb}" \
    --variant-tag "tier_mode:${BENCH_TIER_MODE_RESOLVED}" \
    --variant-tag "kv_mode:${BENCH_KV_MODE_RESOLVED}" \
    --scenario standard \
    --prompt-set short \
    --requests "${BENCH_COMPARE_REQUESTS:-32}" \
    --warmup "${BENCH_COMPARE_WARMUP:-4}" \
    --concurrency "${BENCH_COMPARE_CONCURRENCY:-4}" \
    --max-tokens "${BENCH_COMPARE_MAX_TOKENS:-128}" \
    --temperature "${BENCH_COMPARE_TEMPERATURE:-0.2}" \
    --stop "<|eot_id|>" \
    --collect-telemetry \
    --container-name "${BENCH_CONTAINER_NAME:-dyn}" \
    --kvbm-cache-dir "${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}" \
    --run-id "compare_${BENCH_TIER_MODE_RESOLVED}_${TS}"
done

echo "Mode comparison runs complete for timestamp ${TS}"

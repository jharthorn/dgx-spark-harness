#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_profile_lib.sh
source "${SCRIPT_DIR}/bench_profile_lib.sh"

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
CONCURRENCIES="${BENCH_CONCURRENCIES:-1 4 8}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

bench_resolve_tier_mode "${BENCH_TIER_MODE:-${BENCH_KV_MODE:-}}"
bench_defaults_for_tier_mode "${BENCH_TIER_MODE_RESOLVED}"
KV_MODE="${BENCH_KV_MODE_RESOLVED}"
TIER_MODE="${BENCH_TIER_MODE_RESOLVED}"
KV_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-${BENCH_CPU_CACHE_GB_DEFAULT}}"
KV_DISK_CACHE_GB="${DYN_KVBM_DISK_CACHE_GB:-${BENCH_DISK_CACHE_GB_DEFAULT}}"

COMMON_ARGS=(
  --base-url "${BASE_URL}"
  --tier-mode "${TIER_MODE}"
  --kv-mode "${KV_MODE}"
  --kv-cpu-cache-gb "${KV_CPU_CACHE_GB}"
  --kv-disk-cache-gb "${KV_DISK_CACHE_GB}"
  --variant-tag "tier_mode:${TIER_MODE}"
  --variant-tag "kv_mode:${KV_MODE}"
  --variant-tag "cpu_cache_gb:${KV_CPU_CACHE_GB}"
  --variant-tag "disk_cache_gb:${KV_DISK_CACHE_GB}"
)

if [[ "${BENCH_DISABLE_PARTIAL_REUSE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--diagnostic-disable-partial-reuse --variant-tag "diag:disable_partial_reuse")
fi
if [[ "${BENCH_DISABLE_BLOCK_REUSE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--diagnostic-disable-block-reuse --variant-tag "diag:disable_block_reuse")
fi
if [[ "${BENCH_DISABLE_DISK_OFFLOAD_FILTER:-0}" == "1" ]]; then
  COMMON_ARGS+=(--diagnostic-disable-disk-offload-filter --variant-tag "diag:disable_disk_offload_filter")
fi

for c in ${CONCURRENCIES}; do
  python3 -m bench.run_bench \
    "${COMMON_ARGS[@]}" \
    --scenario standard \
    --prompt-set short \
    --requests "${BENCH_SHORT_REQUESTS:-64}" \
    --warmup "${BENCH_SHORT_WARMUP:-8}" \
    --concurrency "${c}" \
    --max-tokens "${BENCH_MAX_TOKENS:-256}" \
    --temperature "${BENCH_TEMPERATURE:-0.2}" \
    --stop "<|eot_id|>" \
    --collect-telemetry \
    --container-name "${BENCH_CONTAINER_NAME:-dyn}" \
    --kvbm-cache-dir "${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}" \
    --run-id "short_c${c}_${TS}"
done

for c in ${CONCURRENCIES}; do
  python3 -m bench.run_bench \
    "${COMMON_ARGS[@]}" \
    --scenario standard \
    --prompt-set long \
    --long-range "${BENCH_LONG_RANGE:-6000:7600}" \
    --requests "${BENCH_LONG_REQUESTS:-24}" \
    --warmup "${BENCH_LONG_WARMUP:-4}" \
    --concurrency "${c}" \
    --max-tokens "${BENCH_MAX_TOKENS:-256}" \
    --temperature "${BENCH_TEMPERATURE:-0.2}" \
    --stop "<|eot_id|>" \
    --collect-telemetry \
    --container-name "${BENCH_CONTAINER_NAME:-dyn}" \
    --kvbm-cache-dir "${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}" \
    --run-id "long_c${c}_${TS}"
done

python3 -m bench.run_bench \
  "${COMMON_ARGS[@]}" \
  --scenario eviction_replay \
  --warmup "${BENCH_EVICT_WARMUP:-2}" \
  --eviction-a-requests "${BENCH_EVICT_A_REQUESTS:-8}" \
  --eviction-b-requests "${BENCH_EVICT_B_REQUESTS:-12}" \
  --eviction-a-concurrency "${BENCH_EVICT_A_CONCURRENCY:-2}" \
  --eviction-b-concurrency "${BENCH_EVICT_B_CONCURRENCY:-4}" \
  --long-range "${BENCH_LONG_RANGE:-6000:7600}" \
  --max-tokens "${BENCH_MAX_TOKENS:-256}" \
  --temperature "${BENCH_TEMPERATURE:-0.2}" \
  --stop "<|eot_id|>" \
  --collect-telemetry \
  --container-name "${BENCH_CONTAINER_NAME:-dyn}" \
  --kvbm-cache-dir "${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}" \
  --run-id "eviction_replay_${TS}"

echo "Completed matrix at timestamp ${TS}"

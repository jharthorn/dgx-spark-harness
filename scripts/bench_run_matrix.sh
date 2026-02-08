#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
CONCURRENCIES="${BENCH_CONCURRENCIES:-1 4 8}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

for c in ${CONCURRENCIES}; do
  python3 -m bench.run_bench \
    --base-url "${BASE_URL}" \
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
    --base-url "${BASE_URL}" \
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
  --base-url "${BASE_URL}" \
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


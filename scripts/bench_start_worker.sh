#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/root/.cache/huggingface/hub/models--nvidia--Llama-3.1-8B-Instruct-FP8/snapshots/42d9515ebd69eea3a87351d079c671c3c5ff0a31}"
KVBM_CONFIG_IN_CONTAINER="${KVBM_CONFIG_IN_CONTAINER:-/tmp/kvbm_llm_api_config.yaml}"
WORKER_LOG="${WORKER_LOG:-/tmp/bench-logs/worker.log}"
KV_MODE="${BENCH_KV_MODE:-cpu_disk}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
KVBM_METRICS_PORT="${DYN_KVBM_METRICS_PORT:-6880}"
ENABLE_LOCAL_INDEXER="${BENCH_ENABLE_LOCAL_INDEXER:-false}"
PUBLISH_EVENTS_AND_METRICS="${BENCH_PUBLISH_EVENTS_AND_METRICS:-0}"
REQUEST_PLANE="${BENCH_REQUEST_PLANE:-tcp}"
NATS_SERVER_URL="${BENCH_NATS_SERVER:-${NATS_SERVER:-nats://127.0.0.1:4222}}"

case "${REQUEST_PLANE}" in
  nats|http|tcp)
    ;;
  *)
    echo "Unsupported BENCH_REQUEST_PLANE=${REQUEST_PLANE} (expected nats/http/tcp)." >&2
    exit 1
    ;;
esac

WORKER_EXTRA_ARGS=""
case "${PUBLISH_EVENTS_AND_METRICS}" in
  1|true|TRUE|on|ON)
    WORKER_EXTRA_ARGS+=" --publish-events-and-metrics"
    RESOLVED_PUBLISH_EVENTS_AND_METRICS="true"
    ;;
  0|false|FALSE|off|OFF|"")
    RESOLVED_PUBLISH_EVENTS_AND_METRICS="false"
    ;;
  *)
    echo "Unsupported BENCH_PUBLISH_EVENTS_AND_METRICS=${PUBLISH_EVENTS_AND_METRICS} (expected on/off)." >&2
    exit 1
    ;;
esac

case "${KV_MODE}" in
  off)
    STORE_KV_ARG="--store-kv file"
    EXTRA_ENGINE_ARGS=""
    RESOLVED_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-0}"
    RESOLVED_DISK_CACHE_GB="${DYN_KVBM_DISK_CACHE_GB:-0}"
    RESOLVED_KVBM_METRICS="${DYN_KVBM_METRICS:-false}"
    ;;
  cpu_only)
    STORE_KV_ARG="--store-kv file"
    EXTRA_ENGINE_ARGS="--extra-engine-args '${KVBM_CONFIG_IN_CONTAINER}'"
    RESOLVED_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-8}"
    RESOLVED_DISK_CACHE_GB="${DYN_KVBM_DISK_CACHE_GB:-0}"
    RESOLVED_KVBM_METRICS="${DYN_KVBM_METRICS:-true}"
    ;;
  cpu_disk)
    STORE_KV_ARG="--store-kv file"
    EXTRA_ENGINE_ARGS="--extra-engine-args '${KVBM_CONFIG_IN_CONTAINER}'"
    RESOLVED_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-8}"
    RESOLVED_DISK_CACHE_GB="${DYN_KVBM_DISK_CACHE_GB:-32}"
    RESOLVED_KVBM_METRICS="${DYN_KVBM_METRICS:-true}"
    ;;
  *)
    echo "Unsupported BENCH_KV_MODE=${KV_MODE}" >&2
    exit 1
    ;;
esac

docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
pkill -f '^python3 -m dynamo\.trtllm( |$)' >/dev/null 2>&1 || true
mkdir -p /tmp/bench-logs
: > '${WORKER_LOG}'
export DYN_KVBM_CPU_CACHE_GB='${RESOLVED_CPU_CACHE_GB}'
export DYN_KVBM_DISK_CACHE_GB='${RESOLVED_DISK_CACHE_GB}'
export DYN_KVBM_DISK_CACHE_DIR=\${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}
export DYN_KVBM_METRICS='${RESOLVED_KVBM_METRICS}'
export DYN_KVBM_METRICS_PORT='${KVBM_METRICS_PORT}'
export DYN_SYSTEM_PORT='${SYSTEM_PORT}'
export DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=\${DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER:-${BENCH_DISABLE_DISK_OFFLOAD_FILTER:-0}}
export DYN_REQUEST_PLANE='${REQUEST_PLANE}'
export NATS_SERVER='${NATS_SERVER_URL}'
{
  echo \"DYN_SYSTEM_PORT=\${DYN_SYSTEM_PORT}\"
  echo \"DYN_KVBM_METRICS_PORT=\${DYN_KVBM_METRICS_PORT}\"
  echo \"DYN_REQUEST_PLANE=\${DYN_REQUEST_PLANE}\"
  echo \"TRTLLM_EXTRA_ENGINE_ARGS=${EXTRA_ENGINE_ARGS}\"
  echo \"TRTLLM_STORE_KV_ARG=${STORE_KV_ARG}\"
} >> '${WORKER_LOG}'
nohup python3 -m dynamo.trtllm \
  --model-path '${MODEL_SNAPSHOT}' \
  --endpoint dyn://dynamo.tensorrt_llm.generate \
  --request-plane '${REQUEST_PLANE}' \
  ${EXTRA_ENGINE_ARGS} \
  ${STORE_KV_ARG} \
  ${WORKER_EXTRA_ARGS} \
  --enable-local-indexer '${ENABLE_LOCAL_INDEXER}' \
  > '${WORKER_LOG}' 2>&1 < /dev/null &
"

echo "Worker started in ${CONTAINER_NAME}. Log: ${WORKER_LOG}"
echo "Resolved KV mode: ${KV_MODE} (cpu_cache_gb=${RESOLVED_CPU_CACHE_GB}, disk_cache_gb=${RESOLVED_DISK_CACHE_GB}, kvbm_metrics=${RESOLVED_KVBM_METRICS}, kvbm_metrics_port=${KVBM_METRICS_PORT}, system_port=${SYSTEM_PORT}, local_indexer=${ENABLE_LOCAL_INDEXER}, publish_events=${RESOLVED_PUBLISH_EVENTS_AND_METRICS}, request_plane=${REQUEST_PLANE}, nats_server=${NATS_SERVER_URL})"
echo "Final TRT-LLM args: --model-path '${MODEL_SNAPSHOT}' --endpoint dyn://dynamo.tensorrt_llm.generate --request-plane '${REQUEST_PLANE}' ${EXTRA_ENGINE_ARGS} ${STORE_KV_ARG} ${WORKER_EXTRA_ARGS} --enable-local-indexer '${ENABLE_LOCAL_INDEXER}'"
docker exec "${CONTAINER_NAME}" bash -lc "
sleep 2
if ! pgrep -f '^python3 -m dynamo\\.trtllm( |$)' >/dev/null 2>&1; then
  echo 'Worker process exited during startup.' >&2
  tail -n 120 '${WORKER_LOG}' >&2 || true
  exit 1
fi
tail -n 40 '${WORKER_LOG}'
"

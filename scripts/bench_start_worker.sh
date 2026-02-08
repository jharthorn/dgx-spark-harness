#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/root/.cache/huggingface/hub/models--nvidia--Llama-3.1-8B-Instruct-FP8/snapshots/42d9515ebd69eea3a87351d079c671c3c5ff0a31}"
KVBM_CONFIG_IN_CONTAINER="${KVBM_CONFIG_IN_CONTAINER:-/tmp/kvbm_llm_api_config.yaml}"
WORKER_LOG="${WORKER_LOG:-/tmp/bench-logs/worker.log}"

docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
pkill -f 'python3 -m dynamo.trtllm' >/dev/null 2>&1 || true
mkdir -p /tmp/bench-logs
export DYN_KVBM_DISK_CACHE_GB=\${DYN_KVBM_DISK_CACHE_GB:-32}
export DYN_KVBM_DISK_CACHE_DIR=\${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}
nohup python3 -m dynamo.trtllm \
  --model-path '${MODEL_SNAPSHOT}' \
  --endpoint dyn://dynamo.tensorrt_llm.generate \
  --extra-engine-args '${KVBM_CONFIG_IN_CONTAINER}' \
  --store-kv file \
  > '${WORKER_LOG}' 2>&1 < /dev/null &
"

echo "Worker started in ${CONTAINER_NAME}. Log: ${WORKER_LOG}"
docker exec "${CONTAINER_NAME}" bash -lc "sleep 2; tail -n 40 '${WORKER_LOG}'"


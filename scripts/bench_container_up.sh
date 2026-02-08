#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
IMAGE="${BENCH_IMAGE:-trtllm-rc6-dynamo-nixl:latest}"
MODEL_HANDLE="${MODEL_HANDLE:-nvidia/Llama-3.1-8B-Instruct-FP8}"
DYN_KVBM_DISK_CACHE_GB="${DYN_KVBM_DISK_CACHE_GB:-32}"
DYN_KVBM_DISK_CACHE_DIR="${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}"
KVBM_CONFIG_HOST="${KVBM_CONFIG_HOST:-/mnt/nvme/kvbm/kvbm_llm_api_config.yaml}"

if [[ ! -f "${KVBM_CONFIG_HOST}" ]]; then
  echo "Missing KVBM config: ${KVBM_CONFIG_HOST}" >&2
  exit 1
fi

DEV_ARGS=()
for d in /dev/nvidia-fs*; do
  if [[ -e "${d}" ]]; then
    DEV_ARGS+=(--device="${d}")
  fi
done

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc host \
  --network host \
  "${DEV_ARGS[@]}" \
  --device=/dev/nvme0n1 \
  --device=/dev/nvme0n1p2 \
  --ulimit memlock=-1 \
  --cap-add IPC_LOCK \
  --cap-add SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -e MODEL_HANDLE="${MODEL_HANDLE}" \
  -e DYN_KVBM_DISK_CACHE_GB="${DYN_KVBM_DISK_CACHE_GB}" \
  -e DYN_KVBM_DISK_CACHE_DIR="${DYN_KVBM_DISK_CACHE_DIR}" \
  -e CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON:-/etc/cufile/cufile.json}" \
  -v /mnt/nvme:/mnt/nvme:rshared \
  -v "${KVBM_CONFIG_HOST}:/tmp/kvbm_llm_api_config.yaml:ro" \
  -v "${HOME}/.cache/huggingface/:/root/.cache/huggingface/" \
  -v /run/udev:/run/udev:ro \
  -v /dev/disk:/dev/disk:ro \
  "${IMAGE}" \
  bash -lc "sleep infinity"

echo "Container started: ${CONTAINER_NAME}"
docker ps --filter "name=${CONTAINER_NAME}"
docker exec "${CONTAINER_NAME}" bash -lc "findmnt -T ${DYN_KVBM_DISK_CACHE_DIR} -o TARGET,SOURCE,FSTYPE,OPTIONS"


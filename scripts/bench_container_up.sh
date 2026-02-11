#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_profile_lib.sh
source "${SCRIPT_DIR}/bench_profile_lib.sh"

CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
IMAGE="${BENCH_IMAGE:-trtllm-rc6-dynamo-nixl:latest}"

bench_resolve_tier_mode "${BENCH_TIER_MODE:-${BENCH_KV_MODE:-}}"
bench_defaults_for_tier_mode "${BENCH_TIER_MODE_RESOLVED}"
bench_resolve_model_env "${BENCH_MODEL_PROFILE:-}"

MODEL_PROFILE="${BENCH_MODEL_PROFILE_RESOLVED}"
MODEL_HANDLE="${MODEL_HANDLE:-${BENCH_MODEL_HANDLE_DEFAULT}}"
MODEL_NAME="${MODEL_NAME:-${BENCH_MODEL_NAME_DEFAULT}}"
DYN_KVBM_DISK_CACHE_GB="${DYN_KVBM_DISK_CACHE_GB:-${BENCH_DISK_CACHE_GB_DEFAULT}}"
DYN_KVBM_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-${BENCH_CPU_CACHE_GB_DEFAULT}}"
DYN_KVBM_DISK_CACHE_DIR="${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}"
KVBM_CONFIG_HOST="${KVBM_CONFIG_HOST:-/mnt/nvme/kvbm/kvbm_llm_api_config.yaml}"
DYN_KVBM_METRICS="${DYN_KVBM_METRICS:-${BENCH_KVBM_METRICS_DEFAULT}}"
DYN_KVBM_METRICS_PORT="${DYN_KVBM_METRICS_PORT:-6880}"
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"

if [[ "${BENCH_TIER_MODE_RESOLVED}" != "B0" && ! -f "${KVBM_CONFIG_HOST}" ]]; then
  echo "Missing KVBM config: ${KVBM_CONFIG_HOST}" >&2
  exit 1
fi

DEV_ARGS=()
for d in /dev/nvidia-fs*; do
  if [[ -e "${d}" ]]; then
    DEV_ARGS+=(--device="${d}")
  fi
done

VOLUME_ARGS=(
  -v /mnt/nvme:/mnt/nvme:rshared
  -v "${HOME}/.cache/huggingface/:/root/.cache/huggingface/"
  -v /run/udev:/run/udev:ro
  -v /dev/disk:/dev/disk:ro
)
if [[ -f "${KVBM_CONFIG_HOST}" ]]; then
  VOLUME_ARGS+=(-v "${KVBM_CONFIG_HOST}:/tmp/kvbm_llm_api_config.yaml:ro")
fi

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
  -e BENCH_MODEL_PROFILE="${MODEL_PROFILE}" \
  -e BENCH_TIER_MODE="${BENCH_TIER_MODE_RESOLVED}" \
  -e MODEL_HANDLE="${MODEL_HANDLE}" \
  -e MODEL_NAME="${MODEL_NAME}" \
  -e DYN_KVBM_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB}" \
  -e DYN_KVBM_DISK_CACHE_GB="${DYN_KVBM_DISK_CACHE_GB}" \
  -e DYN_KVBM_DISK_CACHE_DIR="${DYN_KVBM_DISK_CACHE_DIR}" \
  -e DYN_KVBM_METRICS="${DYN_KVBM_METRICS}" \
  -e DYN_KVBM_METRICS_PORT="${DYN_KVBM_METRICS_PORT}" \
  -e DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT}" \
  -e CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON:-/etc/cufile/cufile.json}" \
  "${VOLUME_ARGS[@]}" \
  "${IMAGE}" \
  bash -lc "sleep infinity"

echo "Container started: ${CONTAINER_NAME}"
echo "Resolved tier/model: tier_mode=${BENCH_TIER_MODE_RESOLVED} model_profile=${MODEL_PROFILE} model_handle=${MODEL_HANDLE}"
docker ps --filter "name=${CONTAINER_NAME}"
docker exec "${CONTAINER_NAME}" bash -lc "findmnt -T ${DYN_KVBM_DISK_CACHE_DIR} -o TARGET,SOURCE,FSTYPE,OPTIONS"

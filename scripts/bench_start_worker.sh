#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_profile_lib.sh
source "${SCRIPT_DIR}/bench_profile_lib.sh"

CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
KVBM_CONFIG_IN_CONTAINER="${KVBM_CONFIG_IN_CONTAINER:-/tmp/kvbm_llm_api_config.yaml}"
WORKER_LOG="${WORKER_LOG:-/tmp/bench-logs/worker.log}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
KVBM_METRICS_PORT="${DYN_KVBM_METRICS_PORT:-6880}"
ENABLE_LOCAL_INDEXER="${BENCH_ENABLE_LOCAL_INDEXER:-false}"
PUBLISH_EVENTS_AND_METRICS="${BENCH_PUBLISH_EVENTS_AND_METRICS:-0}"
REQUEST_PLANE="${BENCH_REQUEST_PLANE:-tcp}"
NATS_SERVER_URL="${BENCH_NATS_SERVER:-${NATS_SERVER:-nats://127.0.0.1:4222}}"

bench_resolve_tier_mode "${BENCH_TIER_MODE:-${BENCH_KV_MODE:-}}"
bench_defaults_for_tier_mode "${BENCH_TIER_MODE_RESOLVED}"
bench_resolve_model_env "${BENCH_MODEL_PROFILE:-}"

KV_MODE="${BENCH_KV_MODE_RESOLVED}"
TIER_MODE="${BENCH_TIER_MODE_RESOLVED}"
MODEL_PROFILE="${BENCH_MODEL_PROFILE_RESOLVED}"

MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-}"
if [[ -z "${MODEL_SNAPSHOT}" ]]; then
  MODEL_SNAPSHOT="$(docker exec -e MODEL_SNAPSHOT_GLOB="${MODEL_SNAPSHOT_GLOB}" "${CONTAINER_NAME}" bash -lc "python3 -c 'import glob, os; pattern = os.environ.get(\"MODEL_SNAPSHOT_GLOB\", \"\"); paths = sorted(glob.glob(pattern)) if pattern else []; print(paths[-1] if paths else \"\")'")"
fi
if [[ -z "${MODEL_SNAPSHOT}" ]]; then
  echo "Unable to resolve model snapshot inside ${CONTAINER_NAME}. MODEL_SNAPSHOT_GLOB=${MODEL_SNAPSHOT_GLOB}" >&2
  exit 1
fi

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
    # B0 must not touch etcd; file-backed store keeps discovery local and shared with frontend.
    STORE_KV_ARG="--store-kv file"
    EXTRA_ENGINE_ARGS=""
    ;;
  cpu_only|cpu_disk)
    STORE_KV_ARG="--store-kv file"
    EXTRA_ENGINE_ARGS="--extra-engine-args ${KVBM_CONFIG_IN_CONTAINER}"
    ;;
  *)
    echo "Unsupported BENCH_KV_MODE=${KV_MODE}" >&2
    exit 1
    ;;
esac

USE_KVBM_ENV="0"
if [[ "${KV_MODE}" != "off" ]]; then
  USE_KVBM_ENV="1"
fi
RESOLVED_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-${BENCH_CPU_CACHE_GB_DEFAULT}}"
RESOLVED_DISK_CACHE_GB="${DYN_KVBM_DISK_CACHE_GB:-${BENCH_DISK_CACHE_GB_DEFAULT}}"
RESOLVED_KVBM_METRICS="${DYN_KVBM_METRICS:-${BENCH_KVBM_METRICS_DEFAULT}}"
if [[ "${USE_KVBM_ENV}" != "1" ]]; then
  RESOLVED_CPU_CACHE_GB="0"
  RESOLVED_DISK_CACHE_GB="0"
  RESOLVED_KVBM_METRICS="false"
fi
KVBM_METRICS_PORT_DISPLAY="${KVBM_METRICS_PORT}"
if [[ "${USE_KVBM_ENV}" != "1" ]]; then
  KVBM_METRICS_PORT_DISPLAY="disabled"
fi
RUNTIME_MANIFEST="/tmp/bench-logs/worker_runtime_manifest.json"

# BENCH_GPU_ARCH_PREFLIGHT=1 keeps startup failures obvious on unsupported GPU/PyTorch builds.
if [[ "${BENCH_GPU_ARCH_PREFLIGHT:-0}" == "1" ]]; then
  docker exec "${CONTAINER_NAME}" bash -lc "python3 - <<'PY'
import torch
print('torch_version=', torch.__version__)
print('torch_cuda_version=', torch.version.cuda)
if torch.cuda.is_available():
  print('device_name=', torch.cuda.get_device_name(0))
  print('device_capability=', torch.cuda.get_device_capability(0))
else:
  print('device_name=none')
  print('device_capability=none')
PY"
fi

docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
pkill -f '^python3 -m dynamo\.trtllm( |$)' >/dev/null 2>&1 || true
mkdir -p /tmp/bench-logs
: > '${WORKER_LOG}'
export BENCH_TIER_MODE='${TIER_MODE}'
export BENCH_KV_MODE='${KV_MODE}'
export BENCH_MODEL_PROFILE='${MODEL_PROFILE}'
export MODEL_HANDLE='${MODEL_HANDLE}'
export MODEL_NAME='${MODEL_NAME}'
export MODEL_SNAPSHOT='${MODEL_SNAPSHOT}'
export DYN_SYSTEM_PORT='${SYSTEM_PORT}'
export DYN_REQUEST_PLANE='${REQUEST_PLANE}'
export NATS_SERVER='${NATS_SERVER_URL}'
if [[ '${USE_KVBM_ENV}' == '1' ]]; then
  export KVBM_CONFIG_IN_CONTAINER='${KVBM_CONFIG_IN_CONTAINER}'
  export DYN_KVBM_CPU_CACHE_GB='${RESOLVED_CPU_CACHE_GB}'
  export DYN_KVBM_DISK_CACHE_GB='${RESOLVED_DISK_CACHE_GB}'
  export DYN_KVBM_DISK_CACHE_DIR=\${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}
  export DYN_KVBM_METRICS='${RESOLVED_KVBM_METRICS}'
  export DYN_KVBM_METRICS_PORT='${KVBM_METRICS_PORT}'
  export DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=\${DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER:-${BENCH_DISABLE_DISK_OFFLOAD_FILTER:-0}}
else
  unset KVBM_CONFIG_IN_CONTAINER DYN_KVBM_CPU_CACHE_GB DYN_KVBM_DISK_CACHE_GB DYN_KVBM_DISK_CACHE_DIR
  unset DYN_KVBM_METRICS DYN_KVBM_METRICS_PORT DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER
fi
{
  echo DYN_SYSTEM_PORT=\${DYN_SYSTEM_PORT}
  echo DYN_KVBM_METRICS_PORT=\${DYN_KVBM_METRICS_PORT:-disabled}
  echo DYN_REQUEST_PLANE=\${DYN_REQUEST_PLANE}
  echo TRTLLM_EXTRA_ENGINE_ARGS=${EXTRA_ENGINE_ARGS}
  echo TRTLLM_STORE_KV_ARG=${STORE_KV_ARG:-none}
  echo MODEL_SNAPSHOT=\${MODEL_SNAPSHOT}
  echo MODEL_PROFILE=\${BENCH_MODEL_PROFILE}
} >> '${WORKER_LOG}'
nohup python3 -m dynamo.trtllm \
  --model-path '${MODEL_SNAPSHOT}' \
  --endpoint dyn://dynamo.tensorrt_llm.generate \
  --request-plane '${REQUEST_PLANE}' \
  ${EXTRA_ENGINE_ARGS} \
  ${STORE_KV_ARG} \
  ${WORKER_EXTRA_ARGS} \
  --enable-local-indexer '${ENABLE_LOCAL_INDEXER}' \
  >> '${WORKER_LOG}' 2>&1 < /dev/null &
"

RUNTIME_ENV_ARGS=(
  -e BENCH_TIER_MODE="${TIER_MODE}"
  -e BENCH_KV_MODE="${KV_MODE}"
  -e BENCH_MODEL_PROFILE="${MODEL_PROFILE}"
  -e MODEL_HANDLE="${MODEL_HANDLE}"
  -e MODEL_NAME="${MODEL_NAME}"
  -e MODEL_SNAPSHOT="${MODEL_SNAPSHOT}"
  -e DYN_SYSTEM_PORT="${SYSTEM_PORT}"
  -e DYN_REQUEST_PLANE="${REQUEST_PLANE}"
  -e BENCH_STORE_KV_ARG="${STORE_KV_ARG}"
  -e BENCH_EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS}"
  -e BENCH_RUNTIME_MANIFEST="${RUNTIME_MANIFEST}"
)
if [[ "${USE_KVBM_ENV}" == "1" ]]; then
  RUNTIME_ENV_ARGS+=(
    -e KVBM_CONFIG_IN_CONTAINER="${KVBM_CONFIG_IN_CONTAINER}"
    -e DYN_KVBM_METRICS_PORT="${KVBM_METRICS_PORT}"
    -e DYN_KVBM_METRICS="${RESOLVED_KVBM_METRICS}"
    -e DYN_KVBM_CPU_CACHE_GB="${RESOLVED_CPU_CACHE_GB}"
    -e DYN_KVBM_DISK_CACHE_GB="${RESOLVED_DISK_CACHE_GB}"
    -e DYN_KVBM_DISK_CACHE_DIR="${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}"
    -e DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER="${DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER:-${BENCH_DISABLE_DISK_OFFLOAD_FILTER:-0}}"
  )
fi
docker exec \
  "${RUNTIME_ENV_ARGS[@]}" \
  "${CONTAINER_NAME}" \
  python3 -c 'import hashlib, json, os; from datetime import datetime, timezone; from pathlib import Path; cfg_path = os.environ.get("KVBM_CONFIG_IN_CONTAINER", ""); cfg_sha = hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest() if cfg_path and Path(cfg_path).exists() else None; payload = {"captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tier_mode": os.environ.get("BENCH_TIER_MODE"), "kv_mode": os.environ.get("BENCH_KV_MODE"), "model_profile": os.environ.get("BENCH_MODEL_PROFILE"), "model_handle": os.environ.get("MODEL_HANDLE"), "model_name": os.environ.get("MODEL_NAME"), "model_snapshot": os.environ.get("MODEL_SNAPSHOT"), "kvbm_config_in_container": cfg_path, "kvbm_config_sha256": cfg_sha, "store_kv_arg": os.environ.get("BENCH_STORE_KV_ARG"), "extra_engine_args": os.environ.get("BENCH_EXTRA_ENGINE_ARGS"), "env": {"DYN_SYSTEM_PORT": os.environ.get("DYN_SYSTEM_PORT"), "DYN_KVBM_METRICS_PORT": os.environ.get("DYN_KVBM_METRICS_PORT"), "DYN_KVBM_METRICS": os.environ.get("DYN_KVBM_METRICS"), "DYN_KVBM_CPU_CACHE_GB": os.environ.get("DYN_KVBM_CPU_CACHE_GB"), "DYN_KVBM_DISK_CACHE_GB": os.environ.get("DYN_KVBM_DISK_CACHE_GB"), "DYN_KVBM_DISK_CACHE_DIR": os.environ.get("DYN_KVBM_DISK_CACHE_DIR"), "DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER": os.environ.get("DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER"), "DYN_REQUEST_PLANE": os.environ.get("DYN_REQUEST_PLANE")}}; Path(os.environ.get("BENCH_RUNTIME_MANIFEST", "/tmp/bench-logs/worker_runtime_manifest.json")).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")'

echo "Worker started in ${CONTAINER_NAME}. Log: ${WORKER_LOG}"
echo "Resolved mode: tier_mode=${TIER_MODE} kv_mode=${KV_MODE} (cpu_cache_gb=${RESOLVED_CPU_CACHE_GB}, disk_cache_gb=${RESOLVED_DISK_CACHE_GB}, kvbm_metrics=${RESOLVED_KVBM_METRICS}, kvbm_metrics_port=${KVBM_METRICS_PORT_DISPLAY}, system_port=${SYSTEM_PORT}, local_indexer=${ENABLE_LOCAL_INDEXER}, publish_events=${RESOLVED_PUBLISH_EVENTS_AND_METRICS}, request_plane=${REQUEST_PLANE}, nats_server=${NATS_SERVER_URL})"
echo "Model profile: ${MODEL_PROFILE} handle=${MODEL_HANDLE} snapshot=${MODEL_SNAPSHOT}"
echo "Final TRT-LLM args: --model-path '${MODEL_SNAPSHOT}' --endpoint dyn://dynamo.tensorrt_llm.generate --request-plane '${REQUEST_PLANE}' ${EXTRA_ENGINE_ARGS} ${STORE_KV_ARG} ${WORKER_EXTRA_ARGS} --enable-local-indexer '${ENABLE_LOCAL_INDEXER}'"
echo "Runtime manifest: ${RUNTIME_MANIFEST}"

docker exec "${CONTAINER_NAME}" bash -lc "
sleep 2
if ! pgrep -f '^python3 -m dynamo\\.trtllm( |$)' >/dev/null 2>&1; then
  echo 'Worker process exited during startup.' >&2
  tail -n 120 '${WORKER_LOG}' >&2 || true
  exit 1
fi
tail -n 40 '${WORKER_LOG}'
"

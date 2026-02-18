#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_profile_lib.sh
source "${SCRIPT_DIR}/bench_profile_lib.sh"

CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
FRONTEND_LOG="${FRONTEND_LOG:-/tmp/bench-logs/frontend.log}"
BENCH_HTTP_PORT="${BENCH_HTTP_PORT:-8000}"
ROUTER_MODE="${BENCH_ROUTER_MODE:-}"
KV_EVENTS_MODE="${BENCH_KV_EVENTS:-}"
ROUTER_RESET_STATES="${BENCH_ROUTER_RESET_STATES:-0}"
REQUEST_PLANE="${BENCH_REQUEST_PLANE:-tcp}"
NATS_SERVER_URL="${BENCH_NATS_SERVER:-${NATS_SERVER:-nats://127.0.0.1:4222}}"

bench_resolve_tier_mode "${BENCH_TIER_MODE:-${BENCH_KV_MODE:-}}"
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

case "${KV_MODE}" in
  off)
    # B0 must not touch etcd; file-backed store keeps discovery local and shared with worker.
    STORE_KV_ARG="--store-kv file"
    ;;
  cpu_only|cpu_disk)
    STORE_KV_ARG="--store-kv file"
    ;;
  *)
    echo "Unsupported BENCH_KV_MODE=${KV_MODE}" >&2
    exit 1
    ;;
esac

case "${REQUEST_PLANE}" in
  nats|http|tcp)
    ;;
  *)
    echo "Unsupported BENCH_REQUEST_PLANE=${REQUEST_PLANE} (expected nats/http/tcp)." >&2
    exit 1
    ;;
esac

FRONTEND_EXTRA_ARGS=""
if [[ -n "${ROUTER_MODE}" ]]; then
  FRONTEND_EXTRA_ARGS+=" --router-mode '${ROUTER_MODE}'"
fi
case "${KV_EVENTS_MODE}" in
  1|true|TRUE|on|ON)
    FRONTEND_EXTRA_ARGS+=" --kv-events"
    ;;
  0|false|FALSE|off|OFF)
    FRONTEND_EXTRA_ARGS+=" --no-kv-events"
    ;;
  "")
    ;;
  *)
    echo "Unsupported BENCH_KV_EVENTS=${KV_EVENTS_MODE} (expected on/off)." >&2
    exit 1
    ;;
esac
case "${ROUTER_RESET_STATES}" in
  1|true|TRUE|on|ON)
    FRONTEND_EXTRA_ARGS+=" --router-reset-states"
    RESOLVED_ROUTER_RESET_STATES="true"
    ;;
  0|false|FALSE|off|OFF|"")
    RESOLVED_ROUTER_RESET_STATES="false"
    ;;
  *)
    echo "Unsupported BENCH_ROUTER_RESET_STATES=${ROUTER_RESET_STATES} (expected on/off)." >&2
    exit 1
    ;;
esac

docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
pkill -f '^python3 -m dynamo\.frontend( |$)' >/dev/null 2>&1 || true
mkdir -p /tmp/bench-logs
: > '${FRONTEND_LOG}'
export DYN_REQUEST_PLANE='${REQUEST_PLANE}'
export NATS_SERVER='${NATS_SERVER_URL}'
export BENCH_TIER_MODE='${TIER_MODE}'
export BENCH_KV_MODE='${KV_MODE}'
export BENCH_MODEL_PROFILE='${MODEL_PROFILE}'
nohup python3 -m dynamo.frontend \
  --http-host 0.0.0.0 \
  --http-port '${BENCH_HTTP_PORT}' \
  --request-plane '${REQUEST_PLANE}' \
  ${STORE_KV_ARG} \
  ${FRONTEND_EXTRA_ARGS} \
  --exp-python-factory \
  --model-name '${MODEL_NAME}' \
  --model-path '${MODEL_SNAPSHOT}' \
  > '${FRONTEND_LOG}' 2>&1 < /dev/null &
"

echo "Frontend started in ${CONTAINER_NAME}. Log: ${FRONTEND_LOG}"
echo "Resolved mode for frontend: tier_mode=${TIER_MODE} kv_mode=${KV_MODE} (router_mode=${ROUTER_MODE:-default}, kv_events=${KV_EVENTS_MODE:-default}, router_reset_states=${RESOLVED_ROUTER_RESET_STATES}, request_plane=${REQUEST_PLANE}, nats_server=${NATS_SERVER_URL})"
echo "Model profile: ${MODEL_PROFILE} model_name=${MODEL_NAME} model_snapshot=${MODEL_SNAPSHOT}"
docker exec "${CONTAINER_NAME}" bash -lc "sleep 2; tail -n 40 '${FRONTEND_LOG}'"

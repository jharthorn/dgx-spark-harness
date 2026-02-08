#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
FRONTEND_LOG="${FRONTEND_LOG:-/tmp/bench-logs/frontend.log}"
BENCH_HTTP_PORT="${BENCH_HTTP_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-nvidia/Llama-3.1-8B-Instruct-FP8}"
KV_MODE="${BENCH_KV_MODE:-cpu_disk}"
ROUTER_MODE="${BENCH_ROUTER_MODE:-}"
KV_EVENTS_MODE="${BENCH_KV_EVENTS:-}"

case "${KV_MODE}" in
  off)
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

docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
pkill -f '^python3 -m dynamo\.frontend( |$)' >/dev/null 2>&1 || true
mkdir -p /tmp/bench-logs
MODEL_DIR=\$(python3 - <<'PY'
import glob
paths = sorted(glob.glob('/root/.cache/huggingface/hub/models--nvidia--Llama-3.1-8B-Instruct-FP8/snapshots/*'))
print(paths[-1] if paths else '')
PY
)
test -d \"\${MODEL_DIR}\"
nohup python3 -m dynamo.frontend \
  --http-host 0.0.0.0 \
  --http-port '${BENCH_HTTP_PORT}' \
  ${STORE_KV_ARG} \
  ${FRONTEND_EXTRA_ARGS} \
  --exp-python-factory \
  --model-name '${MODEL_NAME}' \
  --model-path \"\${MODEL_DIR}\" \
  > '${FRONTEND_LOG}' 2>&1 < /dev/null &
"

echo "Frontend started in ${CONTAINER_NAME}. Log: ${FRONTEND_LOG}"
echo "Resolved KV mode for frontend: ${KV_MODE} (router_mode=${ROUTER_MODE:-default}, kv_events=${KV_EVENTS_MODE:-default})"
docker exec "${CONTAINER_NAME}" bash -lc "sleep 2; tail -n 40 '${FRONTEND_LOG}'"

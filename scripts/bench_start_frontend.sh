#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
FRONTEND_LOG="${FRONTEND_LOG:-/tmp/bench-logs/frontend.log}"
BENCH_HTTP_PORT="${BENCH_HTTP_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-nvidia/Llama-3.1-8B-Instruct-FP8}"

docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
pkill -f 'python3 -m dynamo.frontend' >/dev/null 2>&1 || true
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
  --store-kv file \
  --exp-python-factory \
  --model-name '${MODEL_NAME}' \
  --model-path \"\${MODEL_DIR}\" \
  > '${FRONTEND_LOG}' 2>&1 < /dev/null &
"

echo "Frontend started in ${CONTAINER_NAME}. Log: ${FRONTEND_LOG}"
docker exec "${CONTAINER_NAME}" bash -lc "sleep 2; tail -n 40 '${FRONTEND_LOG}'"


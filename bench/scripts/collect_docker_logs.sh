#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: collect_docker_logs.sh RUN_DIR [CONTAINER]}
CONTAINER=${2:-dyn}

OUT_DIR="${RUN_DIR}/telemetry"
mkdir -p "${OUT_DIR}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found; skipping docker log collection." >&2
  exit 0
fi

docker logs --timestamps "${CONTAINER}" >"${OUT_DIR}/docker_${CONTAINER}_logs.txt" 2>&1 || true
docker inspect "${CONTAINER}" >"${OUT_DIR}/docker_${CONTAINER}_inspect.json" 2>/dev/null || true
echo "collected docker logs for ${CONTAINER}"


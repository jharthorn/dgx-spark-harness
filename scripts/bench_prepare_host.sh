#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${BENCH_HOST_KVBM_DIR:-/mnt/nvme/kvbm}"
DEST_CONFIG="${DEST_DIR}/kvbm_llm_api_config.yaml"
SRC_CONFIG="${BENCH_KVBM_TEMPLATE:-kvbm/kvbm_llm_api_config.yaml}"

if [[ ! -f "${SRC_CONFIG}" ]]; then
  echo "Missing template config: ${SRC_CONFIG}" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
cp -f "${SRC_CONFIG}" "${DEST_CONFIG}"

echo "Prepared host KVBM config:"
ls -l "${DEST_CONFIG}"


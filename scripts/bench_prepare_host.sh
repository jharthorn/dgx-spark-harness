#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${BENCH_HOST_KVBM_DIR:-/mnt/nvme/kvbm}"
DEST_CONFIG="${DEST_DIR}/kvbm_llm_api_config.yaml"
SRC_CONFIG="${BENCH_KVBM_TEMPLATE:-kvbm/kvbm_llm_api_config.yaml}"
DISABLE_PARTIAL_REUSE="${BENCH_DISABLE_PARTIAL_REUSE:-0}"
DISABLE_BLOCK_REUSE="${BENCH_DISABLE_BLOCK_REUSE:-0}"
DISABLE_DISK_OFFLOAD_FILTER="${BENCH_DISABLE_DISK_OFFLOAD_FILTER:-0}"

if [[ ! -f "${SRC_CONFIG}" ]]; then
  echo "Missing template config: ${SRC_CONFIG}" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
cp -f "${SRC_CONFIG}" "${DEST_CONFIG}"

if [[ "${DISABLE_PARTIAL_REUSE}" == "1" || "${DISABLE_BLOCK_REUSE}" == "1" ]]; then
  python3 - "${DEST_CONFIG}" "${DISABLE_PARTIAL_REUSE}" "${DISABLE_BLOCK_REUSE}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
disable_partial = sys.argv[2] == "1"
disable_block = sys.argv[3] == "1"
text = path.read_text(encoding="utf-8")

if disable_partial:
    text = re.sub(r"^(\s*enable_partial_reuse:\s*)true\s*$", r"\1false", text, flags=re.MULTILINE)
if disable_block:
    text = re.sub(r"^(\s*enable_block_reuse:\s*)true\s*$", r"\1false", text, flags=re.MULTILINE)

path.write_text(text, encoding="utf-8")
PY
fi

echo "Prepared host KVBM config:"
ls -l "${DEST_CONFIG}"
echo "Diagnostic toggles: disable_partial_reuse=${DISABLE_PARTIAL_REUSE} disable_block_reuse=${DISABLE_BLOCK_REUSE} disable_disk_offload_filter=${DISABLE_DISK_OFFLOAD_FILTER}"
if [[ "${DISABLE_DISK_OFFLOAD_FILTER}" == "1" ]]; then
  echo "[ASSUMPTION] Set BENCH_DISABLE_DISK_OFFLOAD_FILTER=1 and start worker with scripts/bench_start_worker.sh to export DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=1."
fi

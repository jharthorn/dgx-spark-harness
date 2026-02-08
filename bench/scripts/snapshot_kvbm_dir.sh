#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: snapshot_kvbm_dir.sh RUN_DIR CACHE_DIR [LABEL]}
CACHE_DIR=${2:?usage: snapshot_kvbm_dir.sh RUN_DIR CACHE_DIR [LABEL]}
LABEL=${3:-snapshot}

OUT_DIR="${RUN_DIR}/telemetry"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_FILE="${OUT_DIR}/kvbm_${LABEL}_${TS}.txt"
LIST_FILE="${OUT_DIR}/kvbm_${LABEL}_${TS}_files.txt"

mkdir -p "${OUT_DIR}"

{
  echo "timestamp_utc=${TS}"
  echo "cache_dir=${CACHE_DIR}"
  if [[ -d "${CACHE_DIR}" ]]; then
    echo "--- du -sh ---"
    du -sh "${CACHE_DIR}" || true
    echo "--- du depth=2 ---"
    du -h --max-depth=2 "${CACHE_DIR}" 2>/dev/null || true
    echo "--- file_count ---"
    find "${CACHE_DIR}" -type f 2>/dev/null | wc -l || true
  else
    echo "cache_dir_missing=1"
  fi
} >"${OUT_FILE}"

if [[ -d "${CACHE_DIR}" ]]; then
  find "${CACHE_DIR}" -maxdepth 5 -type f -printf "%TY-%Tm-%TdT%TTZ %12s %p\n" 2>/dev/null | sort >"${LIST_FILE}" || true
fi

echo "${OUT_FILE}"


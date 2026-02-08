#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: collect_cufile_logs.sh RUN_DIR [CONTAINER]}
CONTAINER=${2:-dyn}

OUT_DIR="${RUN_DIR}/telemetry"
DEST_DIR="${OUT_DIR}/cufile_logs"
mkdir -p "${DEST_DIR}"

if [[ -d /var/log/cufile ]]; then
  mkdir -p "${DEST_DIR}/host"
  cp -a /var/log/cufile/. "${DEST_DIR}/host/" 2>/dev/null || true
fi

if command -v docker >/dev/null 2>&1; then
  docker cp "${CONTAINER}:/var/log/cufile" "${DEST_DIR}/container_${CONTAINER}" >/dev/null 2>&1 || true
  docker exec "${CONTAINER}" bash -lc "grep -RinE 'compat|POSIX|GDS|ERROR|WARN' /var/log/cufile 2>/dev/null | tail -n 200" \
    >"${OUT_DIR}/cufile_${CONTAINER}_grep.txt" 2>&1 || true
fi

echo "collected cuFile logs (best-effort)"

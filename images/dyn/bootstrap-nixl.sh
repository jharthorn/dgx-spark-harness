#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# 0) Helpers
# -----------------------------
log() { echo "[$(date -Is)] bootstrap: $*" >&2; }

# -----------------------------
# 1) NVIDIA FS device alias
# -----------------------------
# Many stacks look for /dev/nvidia-fs, but the driver exposes /dev/nvidia-fs0, /dev/nvidia-fs1, ...
if [[ -e /dev/nvidia-fs0 && ! -e /dev/nvidia-fs ]]; then
  ln -sf /dev/nvidia-fs0 /dev/nvidia-fs
fi

# -----------------------------
# 2) NIXL plugins + wheel libs
# -----------------------------
export NIXL_PLUGIN_DIR="${NIXL_PLUGIN_DIR:-/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/plugins}"

PREFIX="/usr/local/lib/python3.12/dist-packages/nixl_cu12.libs:/usr/local/lib/python3.12/dist-packages/nixl_cu12.libs/nixl"
export LD_LIBRARY_PATH="${PREFIX}:${LD_LIBRARY_PATH:-}"

# -----------------------------
# 3) cuFile config + logging
# -----------------------------
export CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON:-/etc/cufile/cufile.json}"

if [[ ! -f "${CUFILE_ENV_PATH_JSON}" ]]; then
  log "WARNING: CUFILE_ENV_PATH_JSON=${CUFILE_ENV_PATH_JSON} not found; falling back to /etc/cufile/cufile.json"
  export CUFILE_ENV_PATH_JSON=/etc/cufile/cufile.json
fi

# Create the default log dir regardless
mkdir -p /var/log/cufile || true
chmod 777 /var/log/cufile || true

# If config specifies a different log dir, ensure it exists too
if [[ -f "${CUFILE_ENV_PATH_JSON}" ]]; then
  log_dir="$(
    python3 - <<'PY'
import json, os
p=os.environ.get("CUFILE_ENV_PATH_JSON")
try:
    cfg=json.load(open(p))
    print(cfg.get("logging",{}).get("dir",""))
except Exception:
    print("")
PY
  )"
  if [[ -n "${log_dir}" ]]; then
    mkdir -p "${log_dir}" || true
    chmod 777 "${log_dir}" || true
  fi
fi

# -----------------------------
# 4) (Optional) Mount NVMe inside container
# -----------------------------
# If you PASS THROUGH the host mount correctly (recommended), you don't need this.
# If you cannot, set MOUNT_NVME_IN_CONTAINER=1 and ensure you:
#   - pass --device=/dev/nvme0n1p2
#   - pass --cap-add SYS_ADMIN
#   - and /mnt/nvme exists
if [[ "${MOUNT_NVME_IN_CONTAINER:-0}" == "1" ]]; then
  mkdir -p /mnt/nvme
  if ! findmnt -n /mnt/nvme >/dev/null 2>&1; then
    log "Mounting /dev/nvme0n1p2 -> /mnt/nvme inside container namespace"
    mount -t ext4 /dev/nvme0n1p2 /mnt/nvme
  else
    log "/mnt/nvme already mounted in this container"
  fi
fi

# -----------------------------
# 5) Sanity print for cuFile: mount visibility
# -----------------------------
if command -v findmnt >/dev/null 2>&1; then
  if [[ -d /mnt/nvme ]]; then
    log "findmnt -T /mnt/nvme (if mounted):"
    findmnt -T /mnt/nvme -o TARGET,SOURCE,FSTYPE,OPTIONS || true
  fi
  if [[ -d /mnt/nvme/kvbm ]]; then
    log "findmnt -T /mnt/nvme/kvbm:"
    findmnt -T /mnt/nvme/kvbm -o TARGET,SOURCE,FSTYPE,OPTIONS || true
  fi
fi

exec "$@"

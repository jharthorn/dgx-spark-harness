#!/usr/bin/env bash
set -euo pipefail

# runs/_lib_quiescence.sh
# Helper script for system quiescence and privileged operations.
# Sourced by other run scripts.

# Paths
HARNESS_DIR=${HARNESS_DIR:-/harness}
RESULTS_DIR=${RESULTS_DIR:-$HARNESS_DIR/results}
NVME_DEVICE=${NVME_DEVICE:-nvme0n1}
MASTER_LOG=${MASTER_LOG:-$HARNESS_DIR/master_run.log} # Added for logging

as_root() {
  if [[ $EUID -ne 0 ]]; then sudo -- "$@"; else "$@"; fi
}

# Optional: stop periodic services (best effort in container)
echo "Stopping periodic services (best effort)..." | tee -a "$MASTER_LOG"
as_root systemctl stop updatedb.timer 2>/dev/null || true
as_root systemctl stop fstrim.timer  2>/dev/null || true

# Governor (requires privileged)
echo "Setting CPU governor to 'performance'..." | tee -a "$MASTER_LOG"
if [[ -d /sys/devices/system/cpu ]]; then
  as_root bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [[ -w $g ]] && echo performance > $g || true; done' || true
fi

# drop_caches helper (only call where the plan requires it)
drop_caches() {
  echo "Dropping page caches..." | tee -a "$MASTER_LOG"
  as_root sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || echo "WARNING: drop_caches not permitted; run container with --privileged and -v /proc:/proc"
}

# --- NEW: Function for 5.3 ---
run_fstrim() {
  echo "Running fstrim on / ..." | tee -a "$MASTER_LOG"
  as_root fstrim -v / || echo "WARNING: fstrim failed. Run privileged."
}
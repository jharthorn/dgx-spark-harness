#!/usr/bin/env bash
set -euo pipefail

# Archive harness outputs (v2.5 + v3) into a timestamped tarball.
# Captures results/, runs/v3, runs/v2.5 (if present), and analysis/figures.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARCHIVE_NAME=${1:-"results_$(date -u +%Y%m%d_%H%M%S).tar.gz"}
OUT_PATH="$ROOT/archives/$ARCHIVE_NAME"

mkdir -p "$ROOT/archives"

paths=()
for p in results runs/v3 runs/v2.5 analysis/figures; do
  if [[ -e "$ROOT/$p" ]]; then
    paths+=("$p")
  fi
done

if [[ ${#paths[@]} -eq 0 ]]; then
  echo "No result paths found to archive." >&2
  exit 1
fi

echo "Archiving: ${paths[*]} -> $OUT_PATH"
tar -czf "$OUT_PATH" -C "$ROOT" "${paths[@]}" --ignore-failed-read
echo "Done."

#!/usr/bin/env bash
set -euo pipefail

# Archive harness outputs into a timestamped tarball and prune archived dirs.
# Usage:
#   ./scripts/archive_results.sh [FILTER] [ARCHIVE_NAME]
#     FILTER       Optional glob fragment to select result subdirs (e.g., H2B).
#                  If omitted, archives all subdirs under results/.
#     ARCHIVE_NAME Optional archive filename (defaults to results_<UTC timestamp>.tar.gz).

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_FILTER=${1:-}
DEFAULT_LABEL=${RUN_FILTER:-all}
ARCHIVE_NAME=${2:-"results_${DEFAULT_LABEL}_$(date -u +%Y%m%d_%H%M%S).tar.gz"}
OUT_PATH="$ROOT/archives/$ARCHIVE_NAME"

mkdir -p "$ROOT/archives"

paths=()
if [[ -n "$RUN_FILTER" ]]; then
  while IFS= read -r -d '' dir; do
    paths+=("${dir#"$ROOT"/}")
  done < <(find "$ROOT/results" -maxdepth 1 -mindepth 1 -type d -name "*${RUN_FILTER}*" -print0 2>/dev/null)
else
  while IFS= read -r -d '' dir; do
    paths+=("${dir#"$ROOT"/}")
  done < <(find "$ROOT/results" -maxdepth 1 -mindepth 1 -type d -print0 2>/dev/null)
fi

if [[ ${#paths[@]} -eq 0 ]]; then
  echo "No result paths found to archive (filter='$RUN_FILTER')." >&2
  exit 1
fi

echo "Archiving: ${paths[*]} -> $OUT_PATH"
tar -czf "$OUT_PATH" -C "$ROOT" "${paths[@]}" --ignore-failed-read
echo "Archive complete; pruning archived result directories..."
for p in "${paths[@]}"; do
  rm -rf "$ROOT/$p"
done
echo "Done."

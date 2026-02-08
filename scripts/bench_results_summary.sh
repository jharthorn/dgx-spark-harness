#!/usr/bin/env bash
set -euo pipefail

PATTERN="${1:-bench/results/*/summary.json}"

for s in ${PATTERN}; do
  [[ -f "${s}" ]] || continue
  echo "==== ${s}"
  jq '{scenario, overall: .overall_summary, phases: [.phase_summaries[] | {phase, req_per_s, error_rate, io_delta}]}' "${s}"
done


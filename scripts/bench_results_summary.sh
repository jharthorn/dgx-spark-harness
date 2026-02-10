#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-bench/results/*/summary.json}"

expand_to_summaries() {
  local value="$1"
  if [[ -d "${value}" ]]; then
    printf '%s/summary.json\n' "${value}"
    return 0
  fi
  if [[ "${value}" == *"/summary.json" ]]; then
    printf '%s\n' "${value}"
    return 0
  fi
  printf '%s\n' "${value}"
}

for p in $(expand_to_summaries "${INPUT}"); do
  for s in ${p}; do
  [[ -f "${s}" ]] || continue
  echo "==== ${s}"
  jq '{
    run_valid,
    invalid_reason,
    scenario,
    kv_mode: .kv_mode.mode,
    overall: .overall_summary,
    kvbm_signal: .overall_summary.eviction_replay_signal_kvbm,
    reuse_verify_signal: .overall_summary.reuse_verify_signal_kvbm,
    request_identity: .request_identity.reuse_verify_identity,
    phases: [.phase_summaries[] | {phase, req_per_s, error_rate, io_delta, kvbm_metrics_delta}]
  }' "${s}"
  done
done

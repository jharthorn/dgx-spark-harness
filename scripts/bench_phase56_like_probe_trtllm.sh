#!/usr/bin/env bash
set -euo pipefail

normalize_bool() {
  local raw="${1:-0}"
  case "${raw,,}" in
    1|true|yes|on) echo "1" ;;
    0|false|no|off|"") echo "0" ;;
    *)
      echo "Invalid boolean value: ${raw}" >&2
      exit 1
      ;;
  esac
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_ROOT="${BENCH_RESULTS_ROOT:-bench/results}"
BUNDLE_ID="${BENCH_PHASE56_LIKE_BUNDLE_ID:-phase56_like_trtllm_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${BENCH_PHASE56_LIKE_RUN_ID:-run_trtllm_$(date -u +%Y%m%dT%H%M%SZ)}"
BUNDLE_DIR="${RESULTS_ROOT}/${BUNDLE_ID}"
ANALYSIS_DIR="${BUNDLE_DIR}/analysis"
RUN_DIR="${BUNDLE_DIR}/${RUN_ID}"
mkdir -p "${ANALYSIS_DIR}"

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
KV_MODE="${BENCH_KV_MODE:-cpu_disk}"
KV_CPU_CACHE_GB="${BENCH_PHASE56_CPU_CACHE_GB:-${DYN_KVBM_CPU_CACHE_GB:-8}}"
KV_DISK_CACHE_GB="${BENCH_PHASE56_DISK_CACHE_GB:-${DYN_KVBM_DISK_CACHE_GB:-32}}"
A_REQ="${BENCH_PHASE56_A_REQ:-8}"
B_REQ="${BENCH_PHASE56_B_REQ:-16}"
A_CONC="${BENCH_PHASE56_A_CONC:-2}"
B_CONC="${BENCH_PHASE56_B_CONC:-4}"
LONG_RANGE="${BENCH_PHASE56_LONG_RANGE:-6000:7600}"
MAX_TOKENS="${BENCH_PHASE56_MAX_TOKENS:-256}"
TEMPERATURE="${BENCH_PHASE56_TEMPERATURE:-0.2}"
SEED="${BENCH_PHASE56_SEED:-1337}"
REQUEST_SEED="${BENCH_PHASE56_REQUEST_SEED:-1337}"
COLLECT_TELEMETRY="$(normalize_bool "${BENCH_PHASE56_COLLECT_TELEMETRY:-0}")"

METRICS_SYSTEM_PORT="${BENCH_PHASE56_METRICS_SYSTEM_PORT:-${DYN_SYSTEM_PORT:-8081}}"
METRICS_KVBM_PORT="${BENCH_PHASE56_METRICS_KVBM_PORT:-${DYN_KVBM_METRICS_PORT:-6880}}"
METRICS_SYSTEM_URL="${BENCH_PHASE56_METRICS_SYSTEM_URL:-http://127.0.0.1:${METRICS_SYSTEM_PORT}/metrics}"
METRICS_KVBM_URL="${BENCH_PHASE56_METRICS_KVBM_URL:-http://127.0.0.1:${METRICS_KVBM_PORT}/metrics}"
PYTHON_BIN="${BENCH_PHASE56_PYTHON_BIN:-python3}"

if ! "${PYTHON_BIN}" -c "import httpx" >/dev/null 2>&1; then
  if [[ -x "${REPO_ROOT}/.venv-bench/bin/python3" ]] && "${REPO_ROOT}/.venv-bench/bin/python3" -c "import httpx" >/dev/null 2>&1; then
    PYTHON_BIN="${REPO_ROOT}/.venv-bench/bin/python3"
  elif [[ -x "/home/jharthorn/dgx-spark-harness/.venv-bench/bin/python3" ]] && /home/jharthorn/dgx-spark-harness/.venv-bench/bin/python3 -c "import httpx" >/dev/null 2>&1; then
    PYTHON_BIN="/home/jharthorn/dgx-spark-harness/.venv-bench/bin/python3"
  else
    echo "Missing dependency: httpx (python binary: ${PYTHON_BIN})." >&2
    exit 1
  fi
fi

STARTUP_EXTRACT="${ANALYSIS_DIR}/worker_kvbm_startup_extract.log"
QUICK_SUMMARY_PATH="${ANALYSIS_DIR}/quick_summary.json"
SUMMARY_PATH="${RUN_DIR}/summary.json"

scripts/bench_start_worker.sh
scripts/bench_start_frontend.sh
scripts/bench_wait_ready.sh

BENCH_ARGS=(
  --base-url "${BASE_URL}"
  --results-root "${BUNDLE_DIR}"
  --run-id "${RUN_ID}"
  --kv-mode "${KV_MODE}"
  --kv-cpu-cache-gb "${KV_CPU_CACHE_GB}"
  --kv-disk-cache-gb "${KV_DISK_CACHE_GB}"
  --scenario eviction_replay
  --warmup 0
  --eviction-a-requests "${A_REQ}"
  --eviction-b-requests "${B_REQ}"
  --eviction-a-concurrency "${A_CONC}"
  --eviction-b-concurrency "${B_CONC}"
  --long-range "${LONG_RANGE}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
  --request-seed "${REQUEST_SEED}"
  --stop "<|eot_id|>"
  --container-name "${CONTAINER_NAME}"
  --kvbm-cache-dir "${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}"
  --allow-missing-kvbm-metrics
  --capture-metrics-snapshot
  --metrics-snapshot-dir "${ANALYSIS_DIR}"
  --metrics-system-url "${METRICS_SYSTEM_URL}"
  --metrics-kvbm-url "${METRICS_KVBM_URL}"
  --kvbm-metrics-url "${METRICS_KVBM_URL}"
)
if [[ "${COLLECT_TELEMETRY}" == "1" ]]; then
  BENCH_ARGS+=(--collect-telemetry)
fi

set +e
"${PYTHON_BIN}" -m bench.run_bench "${BENCH_ARGS[@]}"
BENCH_RC=$?
set -e

if (( BENCH_RC != 0 && BENCH_RC != 2 )); then
  echo "bench.run_bench failed with rc=${BENCH_RC}" >&2
  exit "${BENCH_RC}"
fi

if ! docker exec "${CONTAINER_NAME}" bash -lc "test -f /tmp/bench-logs/worker.log"; then
  echo "worker log not found in container ${CONTAINER_NAME}" > "${STARTUP_EXTRACT}"
else
  docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
{
  echo \"# captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
  echo \"# first_200_lines\"
  sed -n '1,200p' /tmp/bench-logs/worker.log
  echo
  echo \"# key_matches\"
  grep -nEi 'DYN_SYSTEM_PORT|TRTLLM|kvbm|disk_cache|metrics|prometheus|store-kv|extra-engine-args' /tmp/bench-logs/worker.log || true
} " > "${STARTUP_EXTRACT}" 2>/dev/null || true
fi

if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "missing run summary at ${SUMMARY_PATH}" >&2
  exit 1
fi

run_valid="$(jq '.run_valid' "${SUMMARY_PATH}")"
invalid_reason="$(jq -r '.invalid_reason // ""' "${SUMMARY_PATH}")"
overall_p50="$(jq '.overall_summary.latency_ms.p50 // null' "${SUMMARY_PATH}")"
overall_p99="$(jq '.overall_summary.latency_ms.p99 // null' "${SUMMARY_PATH}")"
warm_p50="$(jq '[.phase_summaries[] | select(.phase=="warm_A") | .latency_ms.p50][0] // null' "${SUMMARY_PATH}")"
pressure_p50="$(jq '[.phase_summaries[] | select(.phase=="pressure_B") | .latency_ms.p50][0] // null' "${SUMMARY_PATH}")"
replay_p50="$(jq '[.phase_summaries[] | select(.phase=="replay_A") | .latency_ms.p50][0] // null' "${SUMMARY_PATH}")"

system_pressure_ok=0
system_replay_ok=0
kvbm_pressure_ok=0
kvbm_replay_ok=0
if [[ -s "${ANALYSIS_DIR}/metrics_system_pressure.prom" ]] && ! grep -q "snapshot_unavailable" "${ANALYSIS_DIR}/metrics_system_pressure.prom"; then
  system_pressure_ok=1
fi
if [[ -s "${ANALYSIS_DIR}/metrics_system_replay.prom" ]] && ! grep -q "snapshot_unavailable" "${ANALYSIS_DIR}/metrics_system_replay.prom"; then
  system_replay_ok=1
fi
if [[ -s "${ANALYSIS_DIR}/metrics_kvbm_pressure.prom" ]] && ! grep -q "snapshot_unavailable" "${ANALYSIS_DIR}/metrics_kvbm_pressure.prom"; then
  kvbm_pressure_ok=1
fi
if [[ -s "${ANALYSIS_DIR}/metrics_kvbm_replay.prom" ]] && ! grep -q "snapshot_unavailable" "${ANALYSIS_DIR}/metrics_kvbm_replay.prom"; then
  kvbm_replay_ok=1
fi

expanded_metric_count="$(grep -c '^sample:' "${ANALYSIS_DIR}/kvbm_metric_inventory_expanded.txt" 2>/dev/null || echo 0)"
from_6880_metric_count="$(grep -c '^sample:' "${ANALYSIS_DIR}/kvbm_metric_inventory_from_6880.txt" 2>/dev/null || echo 0)"
if [[ -z "${expanded_metric_count}" ]]; then
  expanded_metric_count=0
fi
if [[ -z "${from_6880_metric_count}" ]]; then
  from_6880_metric_count=0
fi

jq -n \
  --arg backend "trtllm" \
  --arg bundle_id "${BUNDLE_ID}" \
  --arg run_id "${RUN_ID}" \
  --arg summary_path "${SUMMARY_PATH}" \
  --arg startup_extract_path "${STARTUP_EXTRACT}" \
  --arg metrics_system_pressure "${ANALYSIS_DIR}/metrics_system_pressure.prom" \
  --arg metrics_system_replay "${ANALYSIS_DIR}/metrics_system_replay.prom" \
  --arg metrics_kvbm_pressure "${ANALYSIS_DIR}/metrics_kvbm_pressure.prom" \
  --arg metrics_kvbm_replay "${ANALYSIS_DIR}/metrics_kvbm_replay.prom" \
  --arg inventory_expanded "${ANALYSIS_DIR}/kvbm_metric_inventory_expanded.txt" \
  --arg inventory_from_6880 "${ANALYSIS_DIR}/kvbm_metric_inventory_from_6880.txt" \
  --arg invalid_reason "${invalid_reason}" \
  --argjson run_valid "${run_valid}" \
  --argjson overall_p50 "${overall_p50}" \
  --argjson overall_p99 "${overall_p99}" \
  --argjson warm_p50 "${warm_p50}" \
  --argjson pressure_p50 "${pressure_p50}" \
  --argjson replay_p50 "${replay_p50}" \
  --argjson system_pressure_ok "${system_pressure_ok}" \
  --argjson system_replay_ok "${system_replay_ok}" \
  --argjson kvbm_pressure_ok "${kvbm_pressure_ok}" \
  --argjson kvbm_replay_ok "${kvbm_replay_ok}" \
  --argjson expanded_metric_count "${expanded_metric_count:-0}" \
  --argjson from_6880_metric_count "${from_6880_metric_count:-0}" \
  --argjson bench_rc "${BENCH_RC}" \
  '{
    backend: $backend,
    bundle_id: $bundle_id,
    run_id: $run_id,
    run_valid: $run_valid,
    invalid_reason: (if $invalid_reason == "" then null else $invalid_reason end),
    bench_return_code: $bench_rc,
    latency_ms: {
      overall: {p50: $overall_p50, p99: $overall_p99},
      warm_A: {p50: $warm_p50},
      pressure_B: {p50: $pressure_p50},
      replay_A: {p50: $replay_p50}
    },
    metrics: {
      system_endpoint: {
        pressure_snapshot_present: ($system_pressure_ok == 1),
        replay_snapshot_present: ($system_replay_ok == 1)
      },
      kvbm_endpoint: {
        pressure_snapshot_present: ($kvbm_pressure_ok == 1),
        replay_snapshot_present: ($kvbm_replay_ok == 1)
      },
      inventory_counts: {
        expanded: $expanded_metric_count,
        from_6880: $from_6880_metric_count
      }
    },
    artifacts: {
      summary_path: $summary_path,
      worker_kvbm_startup_extract: $startup_extract_path,
      metrics_system_pressure: $metrics_system_pressure,
      metrics_system_replay: $metrics_system_replay,
      metrics_kvbm_pressure: $metrics_kvbm_pressure,
      metrics_kvbm_replay: $metrics_kvbm_replay,
      kvbm_metric_inventory_expanded: $inventory_expanded,
      kvbm_metric_inventory_from_6880: $inventory_from_6880
    }
  }' > "${QUICK_SUMMARY_PATH}"

echo "${BUNDLE_DIR}"

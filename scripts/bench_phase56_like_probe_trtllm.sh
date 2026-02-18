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
# shellcheck source=scripts/bench_profile_lib.sh
source "${SCRIPT_DIR}/bench_profile_lib.sh"

RESULTS_ROOT="${BENCH_RESULTS_ROOT:-bench/results}"
BUNDLE_ID="${BENCH_PHASE56_LIKE_BUNDLE_ID:-phase56_like_trtllm_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${BENCH_PHASE56_LIKE_RUN_ID:-run_trtllm_$(date -u +%Y%m%dT%H%M%SZ)}"
BUNDLE_DIR="${RESULTS_ROOT}/${BUNDLE_ID}"
ANALYSIS_DIR="${BUNDLE_DIR}/analysis"
RUN_DIR="${BUNDLE_DIR}/${RUN_ID}"
mkdir -p "${ANALYSIS_DIR}"

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
bench_resolve_tier_mode "${BENCH_TIER_MODE:-${BENCH_KV_MODE:-}}"
bench_defaults_for_tier_mode "${BENCH_TIER_MODE_RESOLVED}"
bench_resolve_model_env "${BENCH_MODEL_PROFILE:-}"
KV_MODE="${BENCH_KV_MODE_RESOLVED}"
TIER_MODE="${BENCH_TIER_MODE_RESOLVED}"
MODEL_PROFILE="${BENCH_MODEL_PROFILE_RESOLVED}"
KV_CPU_CACHE_GB="${BENCH_PHASE56_CPU_CACHE_GB:-${DYN_KVBM_CPU_CACHE_GB:-${BENCH_CPU_CACHE_GB_DEFAULT}}}"
KV_DISK_CACHE_GB="${BENCH_PHASE56_DISK_CACHE_GB:-${DYN_KVBM_DISK_CACHE_GB:-${BENCH_DISK_CACHE_GB_DEFAULT}}}"
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
SCENARIO="${BENCH_PHASE56_SCENARIO:-eviction_replay}"
DISK_OFFLOAD_FILTER_OVERRIDE="${BENCH_PHASE56_DISABLE_DISK_OFFLOAD_FILTER:-${DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER:-}}"

REHYDRATE_POPULATE_SESSIONS="${BENCH_PHASE56_REHYDRATE_POPULATE_SESSIONS:-${A_REQ}}"
REHYDRATE_THRASH_SESSIONS="${BENCH_PHASE56_REHYDRATE_THRASH_SESSIONS:-${B_REQ}}"
REHYDRATE_TURNS="${BENCH_PHASE56_REHYDRATE_TURNS:-2}"
REHYDRATE_PREFIX_TARGET_TOKENS="${BENCH_PHASE56_REHYDRATE_PREFIX_TARGET_TOKENS:-4096}"
REHYDRATE_POPULATE_CONC="${BENCH_PHASE56_REHYDRATE_POPULATE_CONC:-${A_CONC}}"
REHYDRATE_THRASH_CONC="${BENCH_PHASE56_REHYDRATE_THRASH_CONC:-${B_CONC}}"
REHYDRATE_REPLAY_CONC="${BENCH_PHASE56_REHYDRATE_REPLAY_CONC:-${A_CONC}}"
REHYDRATE_REPLAY_REPEATS="${BENCH_PHASE56_REHYDRATE_REPLAY_REPEATS:-1}"
REHYDRATE_GEN_TOKENS="${BENCH_PHASE56_REHYDRATE_GEN_TOKENS:-${MAX_TOKENS}}"
IO_ATTRIB="$(normalize_bool "${BENCH_PHASE56_IO_ATTRIB:-0}")"
IO_ATTRIB_INTERVAL_S="${BENCH_PHASE56_IO_ATTRIB_INTERVAL_S:-1}"
STREAM_METRICS="$(normalize_bool "${BENCH_PHASE56_STREAM_METRICS:-0}")"
STREAM_TIMEOUT_S="${BENCH_PHASE56_STREAM_TIMEOUT_S:-}"
STREAM_RECORD_TTFB="$(normalize_bool "${BENCH_PHASE56_STREAM_RECORD_TTFB:-0}")"

METRICS_SYSTEM_PORT="${BENCH_PHASE56_METRICS_SYSTEM_PORT:-${DYN_SYSTEM_PORT:-8081}}"
METRICS_KVBM_PORT="${BENCH_PHASE56_METRICS_KVBM_PORT:-${DYN_KVBM_METRICS_PORT:-6880}}"
METRICS_SYSTEM_URL="${BENCH_PHASE56_METRICS_SYSTEM_URL:-http://127.0.0.1:${METRICS_SYSTEM_PORT}/metrics}"
METRICS_KVBM_URL="${BENCH_PHASE56_METRICS_KVBM_URL:-http://127.0.0.1:${METRICS_KVBM_PORT}/metrics}"
KVBM_CACHE_DIR_RUN="${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}"
if [[ "${TIER_MODE}" == "B0" ]]; then
  KVBM_CACHE_DIR_RUN=""
fi
PYTHON_BIN="${BENCH_PHASE56_PYTHON_BIN:-python3}"
KVBM_INVENTORY_FROM_6880="${ANALYSIS_DIR}/kvbm_metric_inventory_from_6880.txt"

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
RUNTIME_MANIFEST_PATH="${ANALYSIS_DIR}/worker_runtime_manifest.json"
BUNDLE_MANIFEST_PATH="${ANALYSIS_DIR}/manifest.json"

BENCH_TIER_MODE="${TIER_MODE}" \
BENCH_KV_MODE="${KV_MODE}" \
BENCH_MODEL_PROFILE="${MODEL_PROFILE}" \
DYN_KVBM_CPU_CACHE_GB="${KV_CPU_CACHE_GB}" \
DYN_KVBM_DISK_CACHE_GB="${KV_DISK_CACHE_GB}" \
DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER="${DISK_OFFLOAD_FILTER_OVERRIDE}" \
scripts/bench_start_worker.sh
BENCH_TIER_MODE="${TIER_MODE}" \
BENCH_KV_MODE="${KV_MODE}" \
BENCH_MODEL_PROFILE="${MODEL_PROFILE}" \
DYN_KVBM_CPU_CACHE_GB="${KV_CPU_CACHE_GB}" \
DYN_KVBM_DISK_CACHE_GB="${KV_DISK_CACHE_GB}" \
DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER="${DISK_OFFLOAD_FILTER_OVERRIDE}" \
scripts/bench_start_frontend.sh
scripts/bench_wait_ready.sh

BENCH_ARGS=(
  --base-url "${BASE_URL}"
  --results-root "${BUNDLE_DIR}"
  --run-id "${RUN_ID}"
  --tier-mode "${TIER_MODE}"
  --kv-mode "${KV_MODE}"
  --kv-cpu-cache-gb "${KV_CPU_CACHE_GB}"
  --kv-disk-cache-gb "${KV_DISK_CACHE_GB}"
  --variant-tag "tier_mode:${TIER_MODE}"
  --variant-tag "kv_mode:${KV_MODE}"
  --variant-tag "model_profile:${MODEL_PROFILE}"
  --scenario "${SCENARIO}"
  --warmup 0
  --eviction-a-requests "${A_REQ}"
  --eviction-b-requests "${B_REQ}"
  --eviction-a-concurrency "${A_CONC}"
  --eviction-b-concurrency "${B_CONC}"
  --rehydrate-populate-sessions "${REHYDRATE_POPULATE_SESSIONS}"
  --rehydrate-thrash-sessions "${REHYDRATE_THRASH_SESSIONS}"
  --rehydrate-turns "${REHYDRATE_TURNS}"
  --rehydrate-prefix-target-tokens "${REHYDRATE_PREFIX_TARGET_TOKENS}"
  --rehydrate-populate-concurrency "${REHYDRATE_POPULATE_CONC}"
  --rehydrate-thrash-concurrency "${REHYDRATE_THRASH_CONC}"
  --rehydrate-replay-concurrency "${REHYDRATE_REPLAY_CONC}"
  --rehydrate-replay-repeats "${REHYDRATE_REPLAY_REPEATS}"
  --rehydrate-gen-tokens "${REHYDRATE_GEN_TOKENS}"
  --long-range "${LONG_RANGE}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
  --request-seed "${REQUEST_SEED}"
  --worker-proc-pattern "${BENCH_WORKER_PROC_PATTERN:-dynamo\\.trtllm}"
  --nvme-device "${BENCH_NVME_DEVICE:-/dev/nvme0}"
  --stop "<|eot_id|>"
  --container-name "${CONTAINER_NAME}"
  --kvbm-cache-dir "${KVBM_CACHE_DIR_RUN}"
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
if [[ "${IO_ATTRIB}" == "1" ]]; then
  BENCH_ARGS+=(--io-attrib --io-attrib-interval-s "${IO_ATTRIB_INTERVAL_S}")
fi
if [[ "${STREAM_METRICS}" == "1" ]]; then
  BENCH_ARGS+=(--stream-metrics)
  if [[ -n "${STREAM_TIMEOUT_S}" ]]; then
    BENCH_ARGS+=(--stream-timeout-s "${STREAM_TIMEOUT_S}")
  fi
  if [[ "${STREAM_RECORD_TTFB}" == "1" ]]; then
    BENCH_ARGS+=(--stream-record-ttfb)
  fi
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

if docker exec "${CONTAINER_NAME}" bash -lc "test -f /tmp/bench-logs/worker_runtime_manifest.json"; then
  docker exec "${CONTAINER_NAME}" bash -lc "cat /tmp/bench-logs/worker_runtime_manifest.json" > "${RUNTIME_MANIFEST_PATH}" 2>/dev/null || true
else
  jq -n \
    --arg note "worker_runtime_manifest_missing" \
    --arg container "${CONTAINER_NAME}" \
    '{note: $note, container: $container}' > "${RUNTIME_MANIFEST_PATH}"
fi

jq -n \
  --arg bundle_id "${BUNDLE_ID}" \
  --arg run_id "${RUN_ID}" \
  --arg scenario "${SCENARIO}" \
  --arg tier_mode "${TIER_MODE}" \
  --arg kv_mode "${KV_MODE}" \
  --arg model_profile "${MODEL_PROFILE}" \
  --arg model_handle "${MODEL_HANDLE}" \
  --arg model_name "${MODEL_NAME}" \
  --arg container_name "${CONTAINER_NAME}" \
  --argjson kv_cpu_cache_gb "${KV_CPU_CACHE_GB}" \
  --argjson kv_disk_cache_gb "${KV_DISK_CACHE_GB}" \
  --arg metrics_system_port "${METRICS_SYSTEM_PORT}" \
  --arg metrics_kvbm_port "${METRICS_KVBM_PORT}" \
  --arg disk_offload_filter_override "${DISK_OFFLOAD_FILTER_OVERRIDE}" \
  --arg runtime_manifest_path "${RUNTIME_MANIFEST_PATH}" \
  --arg summary_path "${SUMMARY_PATH}" \
  '{
    bundle_id: $bundle_id,
    run_id: $run_id,
    scenario: $scenario,
    tier_mode: $tier_mode,
    kv_mode: $kv_mode,
    model_profile: $model_profile,
    model_handle: $model_handle,
    model_name: $model_name,
    container_name: $container_name,
    kvbm: {
      cpu_cache_gb: $kv_cpu_cache_gb,
      disk_cache_gb: $kv_disk_cache_gb,
      metrics_system_port: $metrics_system_port,
      metrics_kvbm_port: $metrics_kvbm_port,
      disk_offload_filter_override: (if $disk_offload_filter_override == "" then null else $disk_offload_filter_override end)
    },
    artifacts: {
      worker_runtime_manifest: $runtime_manifest_path,
      run_summary: $summary_path
    }
  }' > "${BUNDLE_MANIFEST_PATH}"

if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "missing run summary at ${SUMMARY_PATH}" >&2
  exit 1
fi

run_valid="$(jq '.run_valid' "${SUMMARY_PATH}")"
invalid_reason="$(jq -r '.invalid_reason // ""' "${SUMMARY_PATH}")"
scenario_name="$(jq -r '.scenario // "unknown"' "${SUMMARY_PATH}")"
overall_p50="$(jq '.overall_summary.latency_ms.p50 // null' "${SUMMARY_PATH}")"
overall_p95="$(jq '.overall_summary.latency_ms.p95 // null' "${SUMMARY_PATH}")"
overall_p99="$(jq '.overall_summary.latency_ms.p99 // null' "${SUMMARY_PATH}")"
overall_error_rate="$(jq '.overall_summary.error_rate // null' "${SUMMARY_PATH}")"
overall_ttft_p50="$(jq '.overall_summary.ttft_ms.p50 // null' "${SUMMARY_PATH}")"
overall_ttft_p95="$(jq '.overall_summary.ttft_ms.p95 // null' "${SUMMARY_PATH}")"
overall_ttft_p99="$(jq '.overall_summary.ttft_ms.p99 // null' "${SUMMARY_PATH}")"
warm_p50="$(jq '([.phase_summaries[] | select(.phase=="warm_A") | .latency_ms.p50][0] // [.phase_summaries[] | select(.phase=="populate") | .latency_ms.p50][0] // null)' "${SUMMARY_PATH}")"
warm_p95="$(jq '([.phase_summaries[] | select(.phase=="warm_A") | .latency_ms.p95][0] // [.phase_summaries[] | select(.phase=="populate") | .latency_ms.p95][0] // null)' "${SUMMARY_PATH}")"
pressure_p50="$(jq '([.phase_summaries[] | select(.phase=="pressure_B") | .latency_ms.p50][0] // [.phase_summaries[] | select(.phase=="thrash") | .latency_ms.p50][0] // null)' "${SUMMARY_PATH}")"
pressure_p95="$(jq '([.phase_summaries[] | select(.phase=="pressure_B") | .latency_ms.p95][0] // [.phase_summaries[] | select(.phase=="thrash") | .latency_ms.p95][0] // null)' "${SUMMARY_PATH}")"
replay_p50="$(jq '([.phase_summaries[] | select(.phase=="replay_A") | .latency_ms.p50][0] // [.phase_summaries[] | select(.phase=="replay") | .latency_ms.p50][0] // null)' "${SUMMARY_PATH}")"
replay_p95="$(jq '([.phase_summaries[] | select(.phase=="replay_A") | .latency_ms.p95][0] // [.phase_summaries[] | select(.phase=="replay") | .latency_ms.p95][0] // null)' "${SUMMARY_PATH}")"
replay_ttft_p50="$(jq '([.phase_summaries[] | select(.phase=="replay_A") | .ttft_ms.p50][0] // [.phase_summaries[] | select(.phase=="replay") | .ttft_ms.p50][0] // null)' "${SUMMARY_PATH}")"
replay_ttft_p95="$(jq '([.phase_summaries[] | select(.phase=="replay_A") | .ttft_ms.p95][0] // [.phase_summaries[] | select(.phase=="replay") | .ttft_ms.p95][0] // null)' "${SUMMARY_PATH}")"
replay_ttft_p99="$(jq '([.phase_summaries[] | select(.phase=="replay_A") | .ttft_ms.p99][0] // [.phase_summaries[] | select(.phase=="replay") | .ttft_ms.p99][0] // null)' "${SUMMARY_PATH}")"
populate_p50="$(jq '[.phase_summaries[] | select(.phase=="populate") | .latency_ms.p50][0] // null' "${SUMMARY_PATH}")"
populate_p95="$(jq '[.phase_summaries[] | select(.phase=="populate") | .latency_ms.p95][0] // null' "${SUMMARY_PATH}")"
thrash_p50="$(jq '[.phase_summaries[] | select(.phase=="thrash") | .latency_ms.p50][0] // null' "${SUMMARY_PATH}")"
thrash_p95="$(jq '[.phase_summaries[] | select(.phase=="thrash") | .latency_ms.p95][0] // null' "${SUMMARY_PATH}")"

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

metrics_scan_json="$(
  "${PYTHON_BIN}" - \
    "${ANALYSIS_DIR}/metrics_system_pressure.prom" \
    "${ANALYSIS_DIR}/metrics_kvbm_pressure.prom" \
    "${ANALYSIS_DIR}/metrics_kvbm_replay.prom" \
    "${KVBM_INVENTORY_FROM_6880}" <<'PY'
import collections
import json
import re
import sys
from pathlib import Path

system_pressure_path = Path(sys.argv[1])
kvbm_pressure_path = Path(sys.argv[2])
kvbm_replay_path = Path(sys.argv[3])
inventory_out_path = Path(sys.argv[4])

metric_line_re = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+")
keywords = ("offload", "onboard", "matched", "tier", "disk", "host")


def iter_metric_lines(path: Path):
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = metric_line_re.match(line)
        if not m:
            continue
        yield m.group(1), line


system_names = {name for name, _ in iter_metric_lines(system_pressure_path) or []}
system_trtllm_prefix_count = len(
    {name for name in system_names if name.startswith("trtllm_") or name.startswith("tensorrt_llm_")}
)
system_dynamo_component_prefix_count = len({name for name in system_names if name.startswith("dynamo_component_")})

keyword_counter: collections.Counter[str] = collections.Counter()
keyword_samples: dict[str, str] = {}
for path in (kvbm_pressure_path, kvbm_replay_path):
    for name, line in iter_metric_lines(path) or []:
        lower = name.lower()
        if any(token in lower for token in keywords):
            keyword_counter[name] += 1
            keyword_samples.setdefault(name, line)

top_names = [name for name, _ in keyword_counter.most_common(20)]
keyword_match_count = len(keyword_counter)

inventory_lines = [
    "# KVBM metrics keyword inventory from 6880 snapshots",
    f"# keywords={','.join(keywords)}",
    f"# keyword_match_count={keyword_match_count}",
    "",
]
for name in top_names:
    inventory_lines.append(name)
    inventory_lines.append(f"sample: {keyword_samples.get(name, '')}")
    inventory_lines.append("")
inventory_out_path.write_text("\n".join(inventory_lines).rstrip() + "\n", encoding="utf-8")

print(
    json.dumps(
        {
            "system_metrics_trtllm_prefix_count": int(system_trtllm_prefix_count),
            "system_metrics_dynamo_component_prefix_count": int(system_dynamo_component_prefix_count),
            "kvbm_metrics_keyword_match_count": int(keyword_match_count),
            "kvbm_metrics_top_names": top_names,
        }
    )
)
PY
)"

system_metrics_trtllm_prefix_count="$(jq '.system_metrics_trtllm_prefix_count' <<< "${metrics_scan_json}")"
system_metrics_dynamo_component_prefix_count="$(jq '.system_metrics_dynamo_component_prefix_count' <<< "${metrics_scan_json}")"
kvbm_metrics_keyword_match_count="$(jq '.kvbm_metrics_keyword_match_count' <<< "${metrics_scan_json}")"
kvbm_metrics_top_names_json="$(jq '.kvbm_metrics_top_names' <<< "${metrics_scan_json}")"
from_6880_metric_count="${kvbm_metrics_keyword_match_count}"

kvbm_metrics_classification="kvbm_metrics_endpoint_missing"
if [[ "${TIER_MODE}" == "B0" ]]; then
  kvbm_metrics_classification="kvbm_disabled"
elif [[ "${kvbm_pressure_ok}" == "1" || "${kvbm_replay_ok}" == "1" ]]; then
  if [[ "${kvbm_metrics_keyword_match_count}" -gt 0 ]]; then
    kvbm_metrics_classification="kvbm_metrics_keywords_found"
  else
    kvbm_metrics_classification="kvbm_metrics_endpoint_reachable_but_no_matching_counters"
  fi
fi

sanitize_argjson_var() {
  local var_name="$1"
  local value="${!var_name-}"
  if [[ -z "${value}" ]] || ! jq -e . <<<"${value}" >/dev/null 2>&1; then
    printf -v "${var_name}" 'null'
  fi
}

ARGJSON_FIELDS=(
  run_valid
  overall_p50
  overall_p95
  overall_p99
  overall_error_rate
  overall_ttft_p50
  overall_ttft_p95
  overall_ttft_p99
  warm_p50
  warm_p95
  pressure_p50
  pressure_p95
  replay_p50
  replay_p95
  replay_ttft_p50
  replay_ttft_p95
  replay_ttft_p99
  populate_p50
  populate_p95
  thrash_p50
  thrash_p95
  system_pressure_ok
  system_replay_ok
  kvbm_pressure_ok
  kvbm_replay_ok
  system_metrics_trtllm_prefix_count
  system_metrics_dynamo_component_prefix_count
  kvbm_metrics_keyword_match_count
  kvbm_metrics_top_names_json
  expanded_metric_count
  from_6880_metric_count
  BENCH_RC
)
for field in "${ARGJSON_FIELDS[@]}"; do
  sanitize_argjson_var "${field}"
done

jq -n \
  --arg backend "trtllm" \
  --arg bundle_id "${BUNDLE_ID}" \
  --arg run_id "${RUN_ID}" \
  --arg tier_mode "${TIER_MODE}" \
  --arg kv_mode "${KV_MODE}" \
  --arg scenario "${scenario_name}" \
  --arg model_profile "${MODEL_PROFILE}" \
  --arg summary_path "${SUMMARY_PATH}" \
  --arg startup_extract_path "${STARTUP_EXTRACT}" \
  --arg runtime_manifest_path "${RUNTIME_MANIFEST_PATH}" \
  --arg bundle_manifest_path "${BUNDLE_MANIFEST_PATH}" \
  --arg nvme_identity_path "${RUN_DIR}/nvme_identity.json" \
  --arg nvme_smart_pre_path "${RUN_DIR}/nvme_smart_pre.json" \
  --arg nvme_smart_post_path "${RUN_DIR}/nvme_smart_post.json" \
  --arg device_metadata_pre_path "${RUN_DIR}/device_metadata_pre.json" \
  --arg device_metadata_post_path "${RUN_DIR}/device_metadata_post.json" \
  --arg io_attribution_report_path "${RUN_DIR}/io/io_attribution_report.json" \
  --arg metrics_system_pressure "${ANALYSIS_DIR}/metrics_system_pressure.prom" \
  --arg metrics_system_replay "${ANALYSIS_DIR}/metrics_system_replay.prom" \
  --arg metrics_kvbm_pressure "${ANALYSIS_DIR}/metrics_kvbm_pressure.prom" \
  --arg metrics_kvbm_replay "${ANALYSIS_DIR}/metrics_kvbm_replay.prom" \
  --arg inventory_expanded "${ANALYSIS_DIR}/kvbm_metric_inventory_expanded.txt" \
  --arg inventory_from_6880 "${ANALYSIS_DIR}/kvbm_metric_inventory_from_6880.txt" \
  --arg invalid_reason "${invalid_reason}" \
  --argjson run_valid "${run_valid}" \
  --argjson overall_p50 "${overall_p50}" \
  --argjson overall_p95 "${overall_p95}" \
  --argjson overall_p99 "${overall_p99}" \
  --argjson overall_error_rate "${overall_error_rate}" \
  --argjson overall_ttft_p50 "${overall_ttft_p50}" \
  --argjson overall_ttft_p95 "${overall_ttft_p95}" \
  --argjson overall_ttft_p99 "${overall_ttft_p99}" \
  --argjson warm_p50 "${warm_p50}" \
  --argjson warm_p95 "${warm_p95}" \
  --argjson pressure_p50 "${pressure_p50}" \
  --argjson pressure_p95 "${pressure_p95}" \
  --argjson replay_p50 "${replay_p50}" \
  --argjson replay_p95 "${replay_p95}" \
  --argjson replay_ttft_p50 "${replay_ttft_p50}" \
  --argjson replay_ttft_p95 "${replay_ttft_p95}" \
  --argjson replay_ttft_p99 "${replay_ttft_p99}" \
  --argjson populate_p50 "${populate_p50}" \
  --argjson populate_p95 "${populate_p95}" \
  --argjson thrash_p50 "${thrash_p50}" \
  --argjson thrash_p95 "${thrash_p95}" \
  --argjson system_pressure_ok "${system_pressure_ok}" \
  --argjson system_replay_ok "${system_replay_ok}" \
  --argjson kvbm_pressure_ok "${kvbm_pressure_ok}" \
  --argjson kvbm_replay_ok "${kvbm_replay_ok}" \
  --arg kvbm_metrics_classification "${kvbm_metrics_classification}" \
  --argjson system_metrics_trtllm_prefix_count "${system_metrics_trtllm_prefix_count}" \
  --argjson system_metrics_dynamo_component_prefix_count "${system_metrics_dynamo_component_prefix_count}" \
  --argjson kvbm_metrics_keyword_match_count "${kvbm_metrics_keyword_match_count}" \
  --argjson kvbm_metrics_top_names "${kvbm_metrics_top_names_json}" \
  --argjson expanded_metric_count "${expanded_metric_count:-0}" \
  --argjson from_6880_metric_count "${from_6880_metric_count:-0}" \
  --argjson bench_rc "${BENCH_RC}" \
  '{
    backend: $backend,
    bundle_id: $bundle_id,
    run_id: $run_id,
    tier_mode: $tier_mode,
    kv_mode: $kv_mode,
    scenario: $scenario,
    model_profile: $model_profile,
    run_valid: $run_valid,
    invalid_reason: (if $invalid_reason == "" then null else $invalid_reason end),
    bench_return_code: $bench_rc,
    error_rate: $overall_error_rate,
    system_metrics_trtllm_prefix_count: $system_metrics_trtllm_prefix_count,
    system_metrics_dynamo_component_prefix_count: $system_metrics_dynamo_component_prefix_count,
    kvbm_metrics_keyword_match_count: $kvbm_metrics_keyword_match_count,
    kvbm_metrics_top_names: $kvbm_metrics_top_names,
    latency_ms: {
      overall: {p50: $overall_p50, p95: $overall_p95, p99: $overall_p99},
      warm_A: {p50: $warm_p50, p95: $warm_p95},
      pressure_B: {p50: $pressure_p50, p95: $pressure_p95},
      replay_A: {p50: $replay_p50, p95: $replay_p95},
      populate: {p50: $populate_p50, p95: $populate_p95},
      thrash: {p50: $thrash_p50, p95: $thrash_p95},
      replay: {p50: $replay_p50, p95: $replay_p95}
    },
    ttft_ms: {
      overall: {p50: $overall_ttft_p50, p95: $overall_ttft_p95, p99: $overall_ttft_p99},
      replay_A: {p50: $replay_ttft_p50, p95: $replay_ttft_p95, p99: $replay_ttft_p99},
      replay: {p50: $replay_ttft_p50, p95: $replay_ttft_p95, p99: $replay_ttft_p99}
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
      kvbm_classification: $kvbm_metrics_classification,
      inventory_counts: {
        expanded: $expanded_metric_count,
        from_6880: $from_6880_metric_count
      }
    },
    artifacts: {
      summary_path: $summary_path,
      worker_kvbm_startup_extract: $startup_extract_path,
      worker_runtime_manifest: $runtime_manifest_path,
      bundle_manifest: $bundle_manifest_path,
      nvme_identity: $nvme_identity_path,
      nvme_smart_pre: $nvme_smart_pre_path,
      nvme_smart_post: $nvme_smart_post_path,
      device_metadata_pre: $device_metadata_pre_path,
      device_metadata_post: $device_metadata_post_path,
      io_attribution_report: $io_attribution_report_path,
      metrics_system_pressure: $metrics_system_pressure,
      metrics_system_replay: $metrics_system_replay,
      metrics_kvbm_pressure: $metrics_kvbm_pressure,
      metrics_kvbm_replay: $metrics_kvbm_replay,
      kvbm_metric_inventory_expanded: $inventory_expanded,
      kvbm_metric_inventory_from_6880: $inventory_from_6880
    }
  }' > "${QUICK_SUMMARY_PATH}"

echo "${BUNDLE_DIR}"

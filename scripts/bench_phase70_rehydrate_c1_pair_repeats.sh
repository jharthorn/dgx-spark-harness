#!/usr/bin/env bash
set -euo pipefail

# Back-compat entrypoint for Phase70 paired repeats.
# Replay-phase concurrency is configurable via BENCH_PHASE70_REPLAY_CONCURRENCY (>=1).

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

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

now_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_ROOT="${BENCH_RESULTS_ROOT:-bench/results}"
TS="${BENCH_PHASE70_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${RESULTS_ROOT}/phase70_pair_logs_${TS}"
mkdir -p "${LOG_DIR}"

CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
PYTHON_BIN="${BENCH_PHASE70_PYTHON_BIN:-python3}"
if ! "${PYTHON_BIN}" -c "import json" >/dev/null 2>&1; then
  if [[ -x "${REPO_ROOT}/.venv-bench/bin/python3" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv-bench/bin/python3"
  fi
fi

MODEL_PROFILE="${BENCH_MODEL_PROFILE:-llama33_70b_nvfp4}"
SCENARIO="${BENCH_PHASE70_SCENARIO:-rehydrate_replay}"
PAIR_COUNT="${BENCH_PHASE70_PAIRS:-${BENCH_PHASE70_REPEATS:-5}}"
REPLAY_CONC="${BENCH_PHASE70_REPLAY_CONCURRENCY:-1}"
ORDER_STRATEGY="${BENCH_PHASE70_ORDER_STRATEGY:-alternating}"
ORDER_SEED="${BENCH_PHASE70_ORDER_SEED:-${BENCH_PHASE70_SEED:-20260212}}"
REQUEST_SEED="${BENCH_PHASE70_REQUEST_SEED:-20260212}"
CPU_CACHE_GB="${BENCH_PHASE70_CPU_CACHE_GB:-2}"
DISK_CACHE_GB="${BENCH_PHASE70_DISK_CACHE_GB:-32}"
MAX_TOKENS="${BENCH_PHASE70_MAX_TOKENS:-128}"
TEMPERATURE="${BENCH_PHASE70_TEMPERATURE:-0.2}"
POP_SESSIONS="${BENCH_PHASE70_REHYDRATE_POPULATE_SESSIONS:-16}"
THRASH_SESSIONS="${BENCH_PHASE70_REHYDRATE_THRASH_SESSIONS:-192}"
TURNS="${BENCH_PHASE70_REHYDRATE_TURNS:-2}"
PREFIX_TOKENS="${BENCH_PHASE70_REHYDRATE_PREFIX_TARGET_TOKENS:-4096}"
REPLAY_REPEATS="${BENCH_PHASE70_REHYDRATE_REPLAY_REPEATS:-2}"
REHYDRATE_GEN_TOKENS="${BENCH_PHASE70_REHYDRATE_GEN_TOKENS:-128}"
PRESSURE_POPULATE_CONC="${BENCH_PHASE70_PRESSURE_POPULATE_CONCURRENCY:-2}"
PRESSURE_THRASH_CONC="${BENCH_PHASE70_PRESSURE_THRASH_CONCURRENCY:-2}"
IO_ATTRIB="$(normalize_bool "${BENCH_PHASE70_IO_ATTRIB:-0}")"
IO_ATTRIB_INTERVAL_S="${BENCH_PHASE70_IO_ATTRIB_INTERVAL_S:-1}"
STREAM_METRICS="$(normalize_bool "${BENCH_PHASE70_STREAM_METRICS:-0}")"
STREAM_TIMEOUT_S="${BENCH_PHASE70_STREAM_TIMEOUT_S:-}"
STREAM_RECORD_TTFB="$(normalize_bool "${BENCH_PHASE70_STREAM_RECORD_TTFB:-0}")"
PAIR_WASHOUT_S="${BENCH_PHASE70_PAIR_WASHOUT_S:-0}"
PAIR_WASHOUT_SYNC="$(normalize_bool "${BENCH_PHASE70_PAIR_WASHOUT_SYNC:-0}")"
PAIR_WASHOUT_DROP_CACHES="$(normalize_bool "${BENCH_PHASE70_PAIR_WASHOUT_DROP_CACHES:-0}")"
SKIP_PREFLIGHT="$(normalize_bool "${BENCH_PHASE70_SKIP_PREFLIGHT:-0}")"
ALLOW_MISSING_KVBM_METRICS="$(normalize_bool "${BENCH_PHASE70_ALLOW_MISSING_KVBM_METRICS:-0}")"
DISABLE_MECHANISM_GATE="$(normalize_bool "${BENCH_PHASE70_DISABLE_MECHANISM_GATE:-0}")"
DECISION_GRADE_REQUIRE_REHYDRATE="$(normalize_bool "${BENCH_PHASE70_DECISION_GRADE_REQUIRE_REHYDRATE:-1}")"
PREFLIGHT_BASE_URL="${BENCH_PHASE70_BASE_URL:-${BENCH_BASE_URL:-http://127.0.0.1:8000}}"
PREFLIGHT_METRICS_KVBM_URL="${BENCH_PHASE70_METRICS_KVBM_URL:-${BENCH_METRICS_KVBM_URL:-http://127.0.0.1:6880/metrics}}"
PREFLIGHT_TIMEOUT_S="${BENCH_PHASE70_PREFLIGHT_TIMEOUT_S:-25}"
PREFLIGHT_SLEEP_S="${BENCH_PHASE70_PREFLIGHT_SLEEP_S:-2}"
MECHANISM_GATE_MIN_DISK_HIT_RATE="${BENCH_PHASE70_GATE_MIN_DISK_HIT_RATE:-0.01}"
MECHANISM_GATE_MIN_MATCHED_TOKENS="${BENCH_PHASE70_GATE_MIN_MATCHED_TOKENS:-0}"
MECHANISM_GATE_REQUIRE_MATCHED="$(normalize_bool "${BENCH_PHASE70_GATE_REQUIRE_MATCHED:-0}")"

MODE_A="${BENCH_PHASE70_MODE_A:-B1}"
MODE_B="${BENCH_PHASE70_MODE_B:-B2}"
if [[ -n "${BENCH_PHASE70_MODES:-}" ]]; then
  read -r -a MODES_ARR <<< "${BENCH_PHASE70_MODES}"
  if (( ${#MODES_ARR[@]} != 2 )); then
    echo "BENCH_PHASE70_MODES must contain exactly two modes for paired repeats; got: ${BENCH_PHASE70_MODES}" >&2
    exit 1
  fi
  MODE_A="${MODES_ARR[0]}"
  MODE_B="${MODES_ARR[1]}"
fi
MODE_A="${MODE_A^^}"
MODE_B="${MODE_B^^}"

for mode in "${MODE_A}" "${MODE_B}"; do
  if [[ "${mode}" != "B0" && "${mode}" != "B1" && "${mode}" != "B2" ]]; then
    echo "Unsupported mode for Phase70 paired runner: ${mode} (expected B0/B1/B2)" >&2
    exit 1
  fi
done
if [[ "${MODE_A}" == "${MODE_B}" ]]; then
  echo "MODE_A and MODE_B must differ; both are ${MODE_A}" >&2
  exit 1
fi
if ! [[ "${PAIR_COUNT}" =~ ^[0-9]+$ ]] || (( PAIR_COUNT <= 0 )); then
  echo "BENCH_PHASE70_PAIRS/REPEATS must be a positive integer; got ${PAIR_COUNT}" >&2
  exit 1
fi
if ! [[ "${REPLAY_CONC}" =~ ^[0-9]+$ ]] || (( REPLAY_CONC < 1 )); then
  echo "BENCH_PHASE70_REPLAY_CONCURRENCY must be an integer >= 1; got ${REPLAY_CONC}" >&2
  exit 1
fi
if ! [[ "${PAIR_WASHOUT_S}" =~ ^[0-9]+$ ]]; then
  echo "BENCH_PHASE70_PAIR_WASHOUT_S must be a non-negative integer; got ${PAIR_WASHOUT_S}" >&2
  exit 1
fi
if ! [[ "${PREFLIGHT_TIMEOUT_S}" =~ ^[0-9]+$ ]] || (( PREFLIGHT_TIMEOUT_S <= 0 )); then
  echo "BENCH_PHASE70_PREFLIGHT_TIMEOUT_S must be a positive integer; got ${PREFLIGHT_TIMEOUT_S}" >&2
  exit 1
fi
if ! [[ "${PREFLIGHT_SLEEP_S}" =~ ^[0-9]+$ ]] || (( PREFLIGHT_SLEEP_S <= 0 )); then
  echo "BENCH_PHASE70_PREFLIGHT_SLEEP_S must be a positive integer; got ${PREFLIGHT_SLEEP_S}" >&2
  exit 1
fi
if [[ "${ORDER_STRATEGY}" != "alternating" && "${ORDER_STRATEGY}" != "random" ]]; then
  echo "BENCH_PHASE70_ORDER_STRATEGY must be alternating or random; got ${ORDER_STRATEGY}" >&2
  exit 1
fi

if (( PAIR_COUNT % 2 == 1 )); then
  echo "Phase70 note: pair count ${PAIR_COUNT} is odd; AB/BA counts differ by at most one. Prefer N=6 or N=8 for exact balance." >&2
fi

KVBM_CACHE_BASE_DIR="${BENCH_KVBM_CACHE_BASE_DIR:-/mnt/nvme/kvbm/phase70_pair_${TS}}"
IO_ATTRIB_CHECKER="${REPO_ROOT}/scripts/check_io_attrib_replay.py"
ANALYZER_SCRIPT="${REPO_ROOT}/scripts/analyze_phase70_pairs.py"
PREFLIGHT_SCRIPT="${REPO_ROOT}/scripts/bench_phase70_preflight.sh"
MECHANISM_GATE_SCRIPT="${REPO_ROOT}/scripts/phase70_check_mechanism_gate.py"
VERDICT_SCRIPT="${REPO_ROOT}/scripts/phase70_write_verdict.py"

MANIFEST_JSON="${RESULTS_ROOT}/phase70_rehydrate_pair_repeats_manifest_${TS}.json"
SUMMARY_JSON="${RESULTS_ROOT}/phase70_rehydrate_pair_repeats_summary_${TS}.json"
SUMMARY_CSV="${RESULTS_ROOT}/phase70_rehydrate_pair_repeats_summary_${TS}.csv"
DELTAS_CSV="${RESULTS_ROOT}/phase70_rehydrate_pair_repeats_deltas_${TS}.csv"
ORDER_CHECK_JSON="${RESULTS_ROOT}/phase70_rehydrate_pair_repeats_order_check_${TS}.json"
VERDICT_JSON="${RESULTS_ROOT}/phase70_rehydrate_pair_repeats_verdict_${TS}.json"
PREFLIGHT_JSON="${RESULTS_ROOT}/phase70_rehydrate_pair_repeats_preflight_${TS}.json"
MECHANISM_GATE_JSON="${RESULTS_ROOT}/phase70_rehydrate_pair_repeats_gate_${TS}.json"

declare -a PHASE70_REASON_CODES=()
RUNS_EXECUTED=0
ANALYZER_DONE=0
DECISION_GRADE_HINT="1"
MECHANISM_GATE_DONE="0"
LAST_RUN_DIR=""
if is_truthy "${DISABLE_MECHANISM_GATE}"; then
  DECISION_GRADE_HINT="0"
fi

add_reason_code() {
  local code="${1:-}"
  if [[ -z "${code}" ]]; then
    return 0
  fi
  local existing
  for existing in "${PHASE70_REASON_CODES[@]}"; do
    if [[ "${existing}" == "${code}" ]]; then
      return 0
    fi
  done
  PHASE70_REASON_CODES+=("${code}")
}

ingest_reason_codes_from_json() {
  local json_path="$1"
  if [[ ! -f "${json_path}" ]]; then
    return 0
  fi
  mapfile -t _codes < <(
    "${PYTHON_BIN}" - "${json_path}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    obj = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

codes = obj.get("reason_codes") if isinstance(obj.get("reason_codes"), list) else []
for code in codes:
    if isinstance(code, str) and code.strip():
        print(code.strip())
PY
  )
  local code
  for code in "${_codes[@]}"; do
    add_reason_code "${code}"
    if [[ "${code}" == "PREFLIGHT_METRICS_UNAVAILABLE" ]]; then
      DECISION_GRADE_HINT="0"
    fi
  done
}

run_preflight() {
  if is_truthy "${SKIP_PREFLIGHT}"; then
    echo "Phase70 preflight skipped (BENCH_PHASE70_SKIP_PREFLIGHT=1)."
    return 0
  fi
  if [[ ! -x "${PREFLIGHT_SCRIPT}" ]]; then
    echo "Phase70 preflight helper missing or not executable: ${PREFLIGHT_SCRIPT}" >&2
    return 1
  fi
  local preflight_rc=0
  set +e
  "${PREFLIGHT_SCRIPT}" \
    --base-url "${PREFLIGHT_BASE_URL}" \
    --metrics-kvbm-url "${PREFLIGHT_METRICS_KVBM_URL}" \
    --timeout-s "${PREFLIGHT_TIMEOUT_S}" \
    --sleep-s "${PREFLIGHT_SLEEP_S}" \
    --allow-missing-kvbm-metrics "${ALLOW_MISSING_KVBM_METRICS}" \
    --python-bin "${PYTHON_BIN}" \
    --json-out "${PREFLIGHT_JSON}"
  preflight_rc=$?
  set -e
  ingest_reason_codes_from_json "${PREFLIGHT_JSON}"
  if (( preflight_rc != 0 )); then
    echo "Phase70 preflight failed; see ${PREFLIGHT_JSON}" >&2
    return "${preflight_rc}"
  fi
  echo "Phase70 preflight complete."
}

maybe_run_mechanism_gate() {
  local mode="$1"
  local pair_id="$2"
  local pair_leg="$3"
  local run_dir="$4"
  if [[ "${mode}" != "B2" ]]; then
    return 0
  fi
  if [[ "${MECHANISM_GATE_DONE}" == "1" ]]; then
    return 0
  fi
  if is_truthy "${DISABLE_MECHANISM_GATE}"; then
    echo "Phase70 mechanism gate disabled by override (BENCH_PHASE70_DISABLE_MECHANISM_GATE=1)." >&2
    DECISION_GRADE_HINT="0"
    MECHANISM_GATE_DONE="1"
    return 0
  fi
  if [[ ! -f "${run_dir}/summary.json" ]]; then
    add_reason_code "GATE_NO_SSD_MECHANISM_SIGNAL"
    echo "Phase70 mechanism gate failed (missing summary.json for first B2 leg)." >&2
    return 1
  fi
  if [[ ! -f "${MECHANISM_GATE_SCRIPT}" ]]; then
    add_reason_code "RUN_ERRORS_PRESENT"
    echo "Phase70 mechanism gate helper missing: ${MECHANISM_GATE_SCRIPT}" >&2
    return 1
  fi

  local gate_rc=0
  local gate_args=(
    --run-dir "${run_dir}"
    --json-out "${MECHANISM_GATE_JSON}"
    --min-disk-hit-rate "${MECHANISM_GATE_MIN_DISK_HIT_RATE}"
    --min-matched-tokens "${MECHANISM_GATE_MIN_MATCHED_TOKENS}"
  )
  if is_truthy "${MECHANISM_GATE_REQUIRE_MATCHED}"; then
    gate_args+=(--require-matched)
  fi

  set +e
  "${PYTHON_BIN}" "${MECHANISM_GATE_SCRIPT}" "${gate_args[@]}" >> "${LOG_DIR}/phase70_mechanism_gate_${TS}.log" 2>&1
  gate_rc=$?
  set -e
  if (( gate_rc != 0 )); then
    add_reason_code "GATE_NO_SSD_MECHANISM_SIGNAL"
    echo "Phase70 mechanism gate failed pair_id=${pair_id} pair_leg=${pair_leg} run_dir=${run_dir} reason=NO_SSD_MECHANISM_SIGNAL" >&2
    return "${gate_rc}"
  fi
  MECHANISM_GATE_DONE="1"
  echo "Phase70 mechanism gate passed pair_id=${pair_id} pair_leg=${pair_leg} run_dir=${run_dir}"
}

run_analyzer() {
  if [[ "${ANALYZER_DONE}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "${MANIFEST_JSON}" ]]; then
    echo "Phase70 analyzer skipped (manifest missing)." >&2
    return 1
  fi
  "${PYTHON_BIN}" "${ANALYZER_SCRIPT}" \
    --manifest "${MANIFEST_JSON}" \
    --summary-json "${SUMMARY_JSON}" \
    --summary-csv "${SUMMARY_CSV}" \
    --pair-delta-csv "${DELTAS_CSV}" \
    --order-check-json "${ORDER_CHECK_JSON}" \
    --mode-a "${MODE_A}" \
    --mode-b "${MODE_B}"
  ANALYZER_DONE="1"
}

write_verdict() {
  if (( RUNS_EXECUTED <= 0 )); then
    return 0
  fi
  if [[ ! -f "${VERDICT_SCRIPT}" ]]; then
    echo "Phase70 verdict helper missing: ${VERDICT_SCRIPT}" >&2
    return 1
  fi
  local reason_args=()
  local code
  for code in "${PHASE70_REASON_CODES[@]}"; do
    reason_args+=(--reason-code "${code}")
  done
  "${PYTHON_BIN}" "${VERDICT_SCRIPT}" \
    --manifest "${MANIFEST_JSON}" \
    --summary-json "${SUMMARY_JSON}" \
    --summary-csv "${SUMMARY_CSV}" \
    --pair-delta-csv "${DELTAS_CSV}" \
    --order-check-json "${ORDER_CHECK_JSON}" \
    --out "${VERDICT_JSON}" \
    --io-attrib-enabled "${IO_ATTRIB}" \
    --decision-grade-hint "${DECISION_GRADE_HINT}" \
    --decision-grade-require-rehydrate "${DECISION_GRADE_REQUIRE_REHYDRATE}" \
    "${reason_args[@]}"
}

phase70_on_exit() {
  local rc="$1"
  trap - EXIT
  if (( RUNS_EXECUTED > 0 )); then
    if (( rc != 0 )); then
      add_reason_code "RUN_ERRORS_PRESENT"
    fi
    if [[ "${ANALYZER_DONE}" != "1" ]]; then
      set +e
      run_analyzer
      local analyzer_rc=$?
      set -e
      if (( analyzer_rc != 0 )); then
        add_reason_code "RUN_ERRORS_PRESENT"
      fi
    fi
    set +e
    write_verdict
    local verdict_rc=$?
    set -e
    if (( verdict_rc != 0 )); then
      echo "Phase70 verdict generation failed rc=${verdict_rc}" >&2
    fi
  fi
  exit "${rc}"
}
trap 'phase70_on_exit $?' EXIT

ensure_container() {
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is not running." >&2
    exit 1
  fi
}

mk_cache_dir() {
  local cache_dir="$1"
  docker exec "${CONTAINER_NAME}" bash -lc "mkdir -p '${cache_dir}' && rm -rf '${cache_dir}'/*"
}

init_manifest() {
  "${PYTHON_BIN}" - "${MANIFEST_JSON}" "${TS}" "${MODEL_PROFILE}" "${SCENARIO}" "${PAIR_COUNT}" "${MODE_A}" "${MODE_B}" "${REPLAY_CONC}" "${ORDER_STRATEGY}" "${ORDER_SEED}" "${IO_ATTRIB}" "${IO_ATTRIB_INTERVAL_S}" "${STREAM_METRICS}" "${STREAM_TIMEOUT_S}" "${STREAM_RECORD_TTFB}" "${PAIR_WASHOUT_S}" "${PAIR_WASHOUT_SYNC}" "${PAIR_WASHOUT_DROP_CACHES}" "${SKIP_PREFLIGHT}" "${ALLOW_MISSING_KVBM_METRICS}" "${DISABLE_MECHANISM_GATE}" "${DECISION_GRADE_REQUIRE_REHYDRATE}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "meta": {
        "created_utc": sys.argv[2],
        "model_profile": sys.argv[3],
        "scenario": sys.argv[4],
        "pair_count": int(sys.argv[5]),
        "mode_a": sys.argv[6],
        "mode_b": sys.argv[7],
        "replay_concurrency": int(sys.argv[8]),
        "order_strategy": sys.argv[9],
        "order_seed": sys.argv[10],
        "io_attrib_enabled": bool(int(sys.argv[11])),
        "io_attrib_interval_s": float(sys.argv[12]),
        "stream_metrics_enabled": bool(int(sys.argv[13])),
        "stream_timeout_s": (float(sys.argv[14]) if sys.argv[14] else None),
        "stream_record_ttfb": bool(int(sys.argv[15])),
        "pair_washout_s": int(sys.argv[16]),
        "pair_washout_sync": bool(int(sys.argv[17])),
        "pair_washout_drop_caches": bool(int(sys.argv[18])),
        "skip_preflight": bool(int(sys.argv[19])),
        "allow_missing_kvbm_metrics": bool(int(sys.argv[20])),
        "disable_mechanism_gate": bool(int(sys.argv[21])),
        "decision_grade_require_rehydrate": bool(int(sys.argv[22])),
        "methodology": {
            "design": "pair_local_blocked_matched_pairs",
            "counterbalancing": "AB_BA",
            "delta_definition": "mode_b_minus_mode_a",
            "recommendation": "Use even pair counts (N=6 or N=8) for exact AB/BA balance.",
        },
    },
    "runs": [],
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

append_manifest_run() {
  local row_json="$1"
  "${PYTHON_BIN}" - "${MANIFEST_JSON}" "${row_json}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
row = json.loads(sys.argv[2])
obj = json.loads(path.read_text(encoding="utf-8"))
obj.setdefault("runs", []).append(row)
path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

generate_pair_orders() {
  # Pair-local blocked runs use AB/BA counterbalancing to reduce drift and order bias.
  mapfile -t PAIR_ORDERS < <(
    "${PYTHON_BIN}" - "${PAIR_COUNT}" "${ORDER_STRATEGY}" "${ORDER_SEED}" "${MODE_A}" "${MODE_B}" <<'PY'
import math
import random
import sys

n = int(sys.argv[1])
strategy = sys.argv[2]
seed = sys.argv[3]
mode_a = sys.argv[4]
mode_b = sys.argv[5]
order_ab = f"{mode_a}_{mode_b}"
order_ba = f"{mode_b}_{mode_a}"

orders: list[str] = []
if strategy == "alternating":
    for idx in range(n):
        orders.append(order_ab if idx % 2 == 0 else order_ba)
elif strategy == "random":
    n_ab = int(math.ceil(n / 2.0))
    n_ba = n - n_ab
    orders = [order_ab] * n_ab + [order_ba] * n_ba
    rng = random.Random(seed)
    rng.shuffle(orders)
else:
    raise SystemExit(f"unsupported strategy: {strategy}")

for order in orders:
    print(order)
PY
  )
  if (( ${#PAIR_ORDERS[@]} != PAIR_COUNT )); then
    echo "Failed to generate pair order schedule." >&2
    exit 1
  fi
}

run_pair_washout() {
  local pair_id="$1"
  local pair_order="$2"
  local first_mode="$3"
  local second_mode="$4"
  local drop_requested="$5"
  local run_drop="0"

  if (( PAIR_WASHOUT_S <= 0 )) && ! is_truthy "${drop_requested}"; then
    return 0
  fi
  if is_truthy "${drop_requested}"; then
    run_drop="1"
  fi

  echo "Phase70 washout pair_id=${pair_id} order=${pair_order} between ${first_mode}->${second_mode}: sleep=${PAIR_WASHOUT_S}s sync=${PAIR_WASHOUT_SYNC} drop_caches=${run_drop}"

  if (( PAIR_WASHOUT_S > 0 )); then
    sleep "${PAIR_WASHOUT_S}"
  fi

  if is_truthy "${PAIR_WASHOUT_SYNC}"; then
    sync || true
  fi

  if [[ "${run_drop}" == "1" ]]; then
    if [[ "$(id -u)" != "0" ]]; then
      echo "Phase70 washout: drop_caches requested but skipped (not root)." >&2
      return 0
    fi
    if [[ ! -w "/proc/sys/vm/drop_caches" ]]; then
      echo "Phase70 washout: drop_caches requested but skipped (/proc/sys/vm/drop_caches not writable)." >&2
      return 0
    fi
    if is_truthy "${PAIR_WASHOUT_SYNC}"; then
      sync || true
    fi
    echo 3 > /proc/sys/vm/drop_caches || {
      echo "Phase70 washout: drop_caches write failed." >&2
      return 0
    }
    echo "Phase70 washout: dropped page cache/dentries/inodes."
  fi
}

run_probe() {
  local pair_id="$1"
  local pair_order="$2"
  local pair_leg="$3"
  local mode="$4"

  local pair_id_fmt
  pair_id_fmt="$(printf '%02d' "${pair_id}")"
  local bundle_id="phase70_rehydrate_pair_${mode}_p${pair_id_fmt}_l${pair_leg}_${TS}"
  local run_id="run_${mode}_p${pair_id_fmt}_l${pair_leg}_${TS}"
  local cache_dir="${KVBM_CACHE_BASE_DIR}/pair_${pair_id_fmt}/${mode}/leg_${pair_leg}"
  local log_path="${LOG_DIR}/${bundle_id}.log"
  local started_utc
  started_utc="$(now_utc)"

  if [[ "${mode}" != "B0" ]]; then
    mk_cache_dir "${cache_dir}"
  else
    cache_dir=""
  fi

  {
    echo "==== $(now_utc) phase70 run start pair_id=${pair_id} pair_order=${pair_order} pair_leg=${pair_leg} mode=${mode} ===="
  } >> "${log_path}"

  set +e
  BENCH_RESULTS_ROOT="${RESULTS_ROOT}" \
  BENCH_PHASE56_LIKE_BUNDLE_ID="${bundle_id}" \
  BENCH_PHASE56_LIKE_RUN_ID="${run_id}" \
  BENCH_TIER_MODE="${mode}" \
  BENCH_MODEL_PROFILE="${MODEL_PROFILE}" \
  BENCH_PHASE56_SCENARIO="${SCENARIO}" \
  BENCH_PHASE56_CPU_CACHE_GB="${CPU_CACHE_GB}" \
  BENCH_PHASE56_DISK_CACHE_GB="${DISK_CACHE_GB}" \
  BENCH_PHASE56_MAX_TOKENS="${MAX_TOKENS}" \
  BENCH_PHASE56_TEMPERATURE="${TEMPERATURE}" \
  BENCH_PHASE56_SEED="${ORDER_SEED}" \
  BENCH_PHASE56_REQUEST_SEED="${REQUEST_SEED}" \
  BENCH_PHASE56_REHYDRATE_POPULATE_SESSIONS="${POP_SESSIONS}" \
  BENCH_PHASE56_REHYDRATE_THRASH_SESSIONS="${THRASH_SESSIONS}" \
  BENCH_PHASE56_REHYDRATE_TURNS="${TURNS}" \
  BENCH_PHASE56_REHYDRATE_PREFIX_TARGET_TOKENS="${PREFIX_TOKENS}" \
  BENCH_PHASE56_REHYDRATE_POPULATE_CONC="${PRESSURE_POPULATE_CONC}" \
  BENCH_PHASE56_REHYDRATE_THRASH_CONC="${PRESSURE_THRASH_CONC}" \
  BENCH_PHASE56_REHYDRATE_REPLAY_CONC="${REPLAY_CONC}" \
  BENCH_PHASE56_REHYDRATE_REPLAY_REPEATS="${REPLAY_REPEATS}" \
  BENCH_PHASE56_REHYDRATE_GEN_TOKENS="${REHYDRATE_GEN_TOKENS}" \
  BENCH_PHASE56_COLLECT_TELEMETRY="0" \
  BENCH_PHASE56_IO_ATTRIB="${IO_ATTRIB}" \
  BENCH_PHASE56_IO_ATTRIB_INTERVAL_S="${IO_ATTRIB_INTERVAL_S}" \
  BENCH_PHASE56_STREAM_METRICS="${STREAM_METRICS}" \
  BENCH_PHASE56_STREAM_TIMEOUT_S="${STREAM_TIMEOUT_S}" \
  BENCH_PHASE56_STREAM_RECORD_TTFB="${STREAM_RECORD_TTFB}" \
  DYN_KVBM_DISK_CACHE_DIR="${cache_dir}" \
  BENCH_CONTAINER_NAME="${CONTAINER_NAME}" \
  scripts/bench_phase56_like_probe_trtllm.sh >> "${log_path}" 2>&1
  local rc=$?
  set -e
  if (( rc != 0 )); then
    add_reason_code "RUN_ERRORS_PRESENT"
    echo "Phase70 run failed pair_id=${pair_id} mode=${mode} rc=${rc}; see ${log_path}" >&2
    exit "${rc}"
  fi

  local run_dir="${RESULTS_ROOT}/${bundle_id}/${run_id}"
  local io_checked="0"
  local io_checker_rc=""
  if is_truthy "${IO_ATTRIB}" && [[ "${mode}" != "B0" ]]; then
    io_checked="1"
    {
      echo "==== $(now_utc) io_attrib checker start mode=${mode} run_dir=${run_dir} ===="
    } >> "${log_path}"
    set +e
    "${PYTHON_BIN}" "${IO_ATTRIB_CHECKER}" --run-dir "${run_dir}" --expect-report >> "${log_path}" 2>&1
    io_checker_rc=$?
    set -e
    {
      echo "==== $(now_utc) io_attrib checker done mode=${mode} rc=${io_checker_rc} ===="
    } >> "${log_path}"
    if (( io_checker_rc != 0 )); then
      add_reason_code "IO_ATTRIB_MISSING"
      echo "io_attrib replay gate failed for mode=${mode} pair_id=${pair_id}; rc=${io_checker_rc}; see ${log_path}" >&2
      exit "${io_checker_rc}"
    fi
  fi

  local completed_utc
  completed_utc="$(now_utc)"
  local row_json
  row_json="$(
    "${PYTHON_BIN}" - "${pair_id}" "${pair_order}" "${pair_leg}" "${mode}" "${bundle_id}" "${run_id}" "${run_dir}" "${log_path}" "${started_utc}" "${completed_utc}" "${io_checked}" "${io_checker_rc}" <<'PY'
import json
import sys

payload = {
    "pair_id": int(sys.argv[1]),
    "pair_order": sys.argv[2],
    "pair_leg": int(sys.argv[3]),
    "mode": sys.argv[4],
    "bundle_id": sys.argv[5],
    "run_id": sys.argv[6],
    "run_dir": sys.argv[7],
    "log_path": sys.argv[8],
    "started_utc": sys.argv[9],
    "completed_utc": sys.argv[10],
    "io_attrib_checked": bool(int(sys.argv[11])),
    "io_attrib_checker_rc": (int(sys.argv[12]) if sys.argv[12] else None),
}
print(json.dumps(payload, separators=(",", ":")))
PY
  )"
  append_manifest_run "${row_json}"
  RUNS_EXECUTED=$((RUNS_EXECUTED + 1))
  LAST_RUN_DIR="${run_dir}"

  echo "Phase70 pair_id=${pair_id} pair_order=${pair_order} pair_leg=${pair_leg} mode=${mode} run_dir=${run_dir}"
}

ensure_container
run_preflight
init_manifest
generate_pair_orders

for pair_id in $(seq 1 "${PAIR_COUNT}"); do
  pair_order="${PAIR_ORDERS[$((pair_id - 1))]}"
  first_mode="${pair_order%%_*}"
  second_mode="${pair_order##*_}"
  if [[ -z "${first_mode}" || -z "${second_mode}" ]]; then
    echo "Invalid pair order token: ${pair_order}" >&2
    exit 1
  fi

  echo "Phase70 pair ${pair_id}/${PAIR_COUNT}: order=${pair_order}"
  run_probe "${pair_id}" "${pair_order}" 1 "${first_mode}"
  maybe_run_mechanism_gate "${first_mode}" "${pair_id}" 1 "${LAST_RUN_DIR}"
  run_pair_washout "${pair_id}" "${pair_order}" "${first_mode}" "${second_mode}" "${PAIR_WASHOUT_DROP_CACHES}"
  run_probe "${pair_id}" "${pair_order}" 2 "${second_mode}"
  maybe_run_mechanism_gate "${second_mode}" "${pair_id}" 2 "${LAST_RUN_DIR}"
done

run_analyzer

echo "PHASE70_MANIFEST_JSON=${MANIFEST_JSON}"
echo "PHASE70_SUMMARY_JSON=${SUMMARY_JSON}"
echo "PHASE70_SUMMARY_CSV=${SUMMARY_CSV}"
echo "PHASE70_DELTAS_CSV=${DELTAS_CSV}"
echo "PHASE70_ORDER_CHECK_JSON=${ORDER_CHECK_JSON}"
echo "PHASE70_VERDICT_JSON=${VERDICT_JSON}"
echo "PHASE70_LOG_DIR=${LOG_DIR}"

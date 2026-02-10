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

is_uint() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

is_long_range() {
  [[ "${1}" =~ ^[0-9]+:[0-9]+$ ]]
}

parse_series() {
  local raw="$1"
  local -n out_ref="$2"
  raw="${raw//,/ }"
  raw="${raw//;/ }"
  read -r -a out_ref <<< "${raw}"
}

pick_series_value() {
  local idx="$1"
  local -n arr_ref="$2"
  local len="${#arr_ref[@]}"
  if (( len == 0 )); then
    echo ""
    return
  fi
  if (( idx < len )); then
    echo "${arr_ref[$idx]}"
  else
    echo "${arr_ref[$((len - 1))]}"
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_ROOT="${BENCH_RESULTS_ROOT:-bench/results}"
BACKEND="${BENCH_BACKEND:-trtllm}"
PATTERN="${BENCH_PHASE58_PATTERN:-progressive_thrash}"
BUNDLE_ID="${BENCH_PHASE58_BUNDLE_ID:-phase58_${BACKEND}_${PATTERN}_${TS}}"
BUNDLE_DIR="${RESULTS_ROOT}/${BUNDLE_ID}"
TRIALS_ROOT="${BUNDLE_DIR}/trials"
ANALYSIS_DIR="${BUNDLE_DIR}/analysis"
LOG_DIR="${BUNDLE_DIR}/logs"
mkdir -p "${TRIALS_ROOT}" "${ANALYSIS_DIR}" "${LOG_DIR}"

if [[ "${PATTERN}" != "progressive_thrash" ]]; then
  echo "Unsupported BENCH_PHASE58_PATTERN=${PATTERN}. Supported: progressive_thrash." >&2
  exit 1
fi

DRY_RUN="$(normalize_bool "${BENCH_PHASE58_DRY_RUN:-0}")"
STOP_ON_SUCCESS="$(normalize_bool "${BENCH_PHASE58_STOP_ON_SUCCESS:-1}")"
MAX_ATTEMPTS="${BENCH_PHASE58_MAX_ATTEMPTS:-0}"
COLLECT_TELEMETRY="$(normalize_bool "${BENCH_PHASE58_COLLECT_TELEMETRY:-0}")"
IDENTICAL_SEEDS="$(normalize_bool "${BENCH_PHASE58_IDENTICAL_SEEDS:-0}")"
METRICS_SYSTEM_PORT="${BENCH_PHASE58_METRICS_SYSTEM_PORT:-${DYN_SYSTEM_PORT:-8081}}"
METRICS_KVBM_PORT="${BENCH_PHASE58_METRICS_KVBM_PORT:-${DYN_KVBM_METRICS_PORT:-6880}}"

A_REQ="${BENCH_PHASE58_A_REQ:-2}"
A_CONC="${BENCH_PHASE58_A_CONC:-1}"
SEED_BASE="${BENCH_PHASE58_SEED_BASE:-1337}"
REQUEST_SEED_BASE="${BENCH_PHASE58_REQUEST_SEED_BASE:-1337}"

B_REQ_SERIES_RAW="${BENCH_PHASE58_B_REQ_SERIES:-96,128,160,192}"
B_CONC_SERIES_RAW="${BENCH_PHASE58_B_CONC_SERIES:-4,4,6,8}"
LONG_RANGE_SERIES_RAW="${BENCH_PHASE58_LONG_RANGE_SERIES:-6144:6272,6144:6912,6144:7680,6144:8192}"

if ! is_uint "${A_REQ}" || (( A_REQ < 1 )); then
  echo "BENCH_PHASE58_A_REQ must be an integer >= 1 (got ${A_REQ})" >&2
  exit 1
fi
if ! is_uint "${A_CONC}" || (( A_CONC < 1 )); then
  echo "BENCH_PHASE58_A_CONC must be an integer >= 1 (got ${A_CONC})" >&2
  exit 1
fi
if ! is_uint "${SEED_BASE}" || ! is_uint "${REQUEST_SEED_BASE}"; then
  echo "BENCH_PHASE58_SEED_BASE and BENCH_PHASE58_REQUEST_SEED_BASE must be integers >= 0" >&2
  exit 1
fi
if ! is_uint "${MAX_ATTEMPTS}"; then
  echo "BENCH_PHASE58_MAX_ATTEMPTS must be an integer >= 0 (got ${MAX_ATTEMPTS})" >&2
  exit 1
fi

declare -a B_REQ_SERIES=()
declare -a B_CONC_SERIES=()
declare -a LONG_RANGE_SERIES=()
parse_series "${B_REQ_SERIES_RAW}" B_REQ_SERIES
parse_series "${B_CONC_SERIES_RAW}" B_CONC_SERIES
parse_series "${LONG_RANGE_SERIES_RAW}" LONG_RANGE_SERIES

max_len="${#B_REQ_SERIES[@]}"
if (( ${#B_CONC_SERIES[@]} > max_len )); then
  max_len="${#B_CONC_SERIES[@]}"
fi
if (( ${#LONG_RANGE_SERIES[@]} > max_len )); then
  max_len="${#LONG_RANGE_SERIES[@]}"
fi
if (( max_len == 0 )); then
  echo "Phase58 schedule is empty." >&2
  exit 1
fi
if (( MAX_ATTEMPTS > 0 && MAX_ATTEMPTS < max_len )); then
  max_len="${MAX_ATTEMPTS}"
fi

{
  echo "backend=${BACKEND}"
  echo "pattern=${PATTERN}"
  echo "bundle_id=${BUNDLE_ID}"
  echo "dry_run=${DRY_RUN}"
  echo "stop_on_success=${STOP_ON_SUCCESS}"
  echo "attempt_count=${max_len}"
  echo "a_req=${A_REQ}"
  echo "a_conc=${A_CONC}"
  echo "b_req_series=${B_REQ_SERIES_RAW}"
  echo "b_conc_series=${B_CONC_SERIES_RAW}"
  echo "long_range_series=${LONG_RANGE_SERIES_RAW}"
  echo "metrics_system_port=${METRICS_SYSTEM_PORT}"
  echo "metrics_kvbm_port=${METRICS_KVBM_PORT}"
  echo "collect_telemetry=${COLLECT_TELEMETRY}"
  echo "identical_seeds=${IDENTICAL_SEEDS}"
} > "${LOG_DIR}/phase58_start.log"

TRIALS_JSONL="${ANALYSIS_DIR}/trials.jsonl"
: > "${TRIALS_JSONL}"

for ((i = 0; i < max_len; i++)); do
  attempt_num="$((i + 1))"
  trial_id="$(printf 'trial_%02d' "${attempt_num}")"
  trial_bundle_id="${trial_id}_${PATTERN}_${TS}"
  trial_run_id="phase58_${BACKEND}_${PATTERN}_${attempt_num}_${TS}"
  trial_dir="${TRIALS_ROOT}/${trial_bundle_id}"
  trial_log="${LOG_DIR}/${trial_id}.log"
  quick_summary_path="${trial_dir}/analysis/quick_summary.json"
  run_summary_path="${trial_dir}/${trial_run_id}/summary.json"

  b_req="$(pick_series_value "${i}" B_REQ_SERIES)"
  b_conc="$(pick_series_value "${i}" B_CONC_SERIES)"
  long_range="$(pick_series_value "${i}" LONG_RANGE_SERIES)"

  if [[ -z "${b_req}" || -z "${b_conc}" || -z "${long_range}" ]] || \
     ! is_uint "${b_req}" || ! is_uint "${b_conc}" || ! is_long_range "${long_range}"; then
    jq -c -n \
      --arg backend "${BACKEND}" \
      --arg pattern "${PATTERN}" \
      --arg trial_id "${trial_id}" \
      --arg trial_bundle_id "${trial_bundle_id}" \
      --arg trial_run_id "${trial_run_id}" \
      --arg b_req "${b_req}" \
      --arg b_conc "${b_conc}" \
      --arg long_range "${long_range}" \
      --argjson attempt_num "${attempt_num}" \
      '{
        attempt_num: $attempt_num,
        backend: $backend,
        pattern: $pattern,
        trial_id: $trial_id,
        trial_bundle_id: $trial_bundle_id,
        trial_run_id: $trial_run_id,
        status: "invalid_schedule_row",
        params_raw: {b_req: $b_req, b_conc: $b_conc, long_range: $long_range}
      }' >> "${TRIALS_JSONL}"
    continue
  fi

  if [[ "${IDENTICAL_SEEDS}" == "1" ]]; then
    trial_seed="${SEED_BASE}"
    trial_request_seed="${REQUEST_SEED_BASE}"
  else
    trial_seed="$((SEED_BASE + i))"
    trial_request_seed="$((REQUEST_SEED_BASE + i))"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    jq -c -n \
      --arg backend "${BACKEND}" \
      --arg pattern "${PATTERN}" \
      --arg trial_id "${trial_id}" \
      --arg trial_bundle_id "${trial_bundle_id}" \
      --arg trial_run_id "${trial_run_id}" \
      --arg quick_summary_path "${quick_summary_path}" \
      --arg run_summary_path "${run_summary_path}" \
      --arg trial_log "${trial_log}" \
      --arg long_range "${long_range}" \
      --argjson attempt_num "${attempt_num}" \
      --argjson seed "${trial_seed}" \
      --argjson request_seed "${trial_request_seed}" \
      --argjson a_req "${A_REQ}" \
      --argjson b_req "${b_req}" \
      --argjson a_conc "${A_CONC}" \
      --argjson b_conc "${b_conc}" \
      '{
        attempt_num: $attempt_num,
        backend: $backend,
        pattern: $pattern,
        trial_id: $trial_id,
        trial_bundle_id: $trial_bundle_id,
        trial_run_id: $trial_run_id,
        status: "dry_run",
        seed: $seed,
        request_seed: $request_seed,
        params: {
          a_req: $a_req,
          b_req: $b_req,
          a_conc: $a_conc,
          b_conc: $b_conc,
          long_range: $long_range
        },
        artifacts: {
          trial_log: $trial_log,
          quick_summary_path: $quick_summary_path,
          run_summary_path: $run_summary_path
        }
      }' >> "${TRIALS_JSONL}"
    continue
  fi

  {
    echo "attempt=${attempt_num}"
    echo "backend=${BACKEND}"
    echo "trial_bundle_id=${trial_bundle_id}"
    echo "trial_run_id=${trial_run_id}"
    echo "a_req=${A_REQ} b_req=${b_req} a_conc=${A_CONC} b_conc=${b_conc} long_range=${long_range}"
    echo "seed=${trial_seed} request_seed=${trial_request_seed}"
  } > "${trial_log}"

  set +e
  if [[ "${BACKEND}" == "trtllm" ]]; then
    BENCH_RESULTS_ROOT="${TRIALS_ROOT}" \
    BENCH_PHASE56_LIKE_BUNDLE_ID="${trial_bundle_id}" \
    BENCH_PHASE56_LIKE_RUN_ID="${trial_run_id}" \
    BENCH_PHASE56_A_REQ="${A_REQ}" \
    BENCH_PHASE56_B_REQ="${b_req}" \
    BENCH_PHASE56_A_CONC="${A_CONC}" \
    BENCH_PHASE56_B_CONC="${b_conc}" \
    BENCH_PHASE56_LONG_RANGE="${long_range}" \
    BENCH_PHASE56_SEED="${trial_seed}" \
    BENCH_PHASE56_REQUEST_SEED="${trial_request_seed}" \
    BENCH_PHASE56_COLLECT_TELEMETRY="${COLLECT_TELEMETRY}" \
    BENCH_PHASE56_METRICS_SYSTEM_PORT="${METRICS_SYSTEM_PORT}" \
    BENCH_PHASE56_METRICS_KVBM_PORT="${METRICS_KVBM_PORT}" \
    DYN_SYSTEM_PORT="${METRICS_SYSTEM_PORT}" \
    DYN_KVBM_METRICS_PORT="${METRICS_KVBM_PORT}" \
    scripts/bench_phase56_like_probe_trtllm.sh >> "${trial_log}" 2>&1
    rc=$?
  elif [[ "${BACKEND}" == "vllm" ]]; then
    if [[ ! -x scripts/bench_phase56_l2_probe.sh ]]; then
      echo "scripts/bench_phase56_l2_probe.sh not found on this branch." >> "${trial_log}"
      rc=127
    else
      BENCH_RESULTS_ROOT="${TRIALS_ROOT}" \
      BENCH_PHASE56_L2_BUNDLE_ID="${trial_bundle_id}" \
      BENCH_PHASE56_L2_RUN_ID="${trial_run_id}" \
      BENCH_PHASE56_A_REQ="${A_REQ}" \
      BENCH_PHASE56_B_REQ="${b_req}" \
      BENCH_PHASE56_A_CONC="${A_CONC}" \
      BENCH_PHASE56_B_CONC="${b_conc}" \
      BENCH_PHASE56_LONG_RANGE="${long_range}" \
      BENCH_PHASE56_SEED="${trial_seed}" \
      BENCH_PHASE56_REQUEST_SEED="${trial_request_seed}" \
      scripts/bench_phase56_l2_probe.sh >> "${trial_log}" 2>&1
      rc=$?
    fi
  else
    echo "Unsupported BENCH_BACKEND=${BACKEND}" >> "${trial_log}"
    rc=2
  fi
  set -e

  status="ok"
  if (( rc != 0 )); then
    status="probe_failed"
  elif [[ ! -f "${quick_summary_path}" || ! -f "${run_summary_path}" ]]; then
    status="missing_summary_artifacts"
  fi

  run_valid="null"
  invalid_reason=""
  system_pressure_ok=0
  kvbm_pressure_ok=0
  if [[ -f "${quick_summary_path}" ]]; then
    run_valid="$(jq '.run_valid // null' "${quick_summary_path}")"
    invalid_reason="$(jq -r '.invalid_reason // ""' "${quick_summary_path}")"
    if jq -e '.metrics.system_endpoint.pressure_snapshot_present == true' "${quick_summary_path}" >/dev/null 2>&1; then
      system_pressure_ok=1
    fi
    if jq -e '.metrics.kvbm_endpoint.pressure_snapshot_present == true' "${quick_summary_path}" >/dev/null 2>&1; then
      kvbm_pressure_ok=1
    fi
  fi

  jq -c -n \
    --arg backend "${BACKEND}" \
    --arg pattern "${PATTERN}" \
    --arg trial_id "${trial_id}" \
    --arg trial_bundle_id "${trial_bundle_id}" \
    --arg trial_run_id "${trial_run_id}" \
    --arg status "${status}" \
    --arg trial_dir "${trial_dir}" \
    --arg trial_log "${trial_log}" \
    --arg quick_summary_path "${quick_summary_path}" \
    --arg run_summary_path "${run_summary_path}" \
    --arg invalid_reason "${invalid_reason}" \
    --arg long_range "${long_range}" \
    --argjson attempt_num "${attempt_num}" \
    --argjson return_code "${rc}" \
    --argjson run_valid "${run_valid}" \
    --argjson seed "${trial_seed}" \
    --argjson request_seed "${trial_request_seed}" \
    --argjson a_req "${A_REQ}" \
    --argjson b_req "${b_req}" \
    --argjson a_conc "${A_CONC}" \
    --argjson b_conc "${b_conc}" \
    --argjson system_pressure_ok "${system_pressure_ok}" \
    --argjson kvbm_pressure_ok "${kvbm_pressure_ok}" \
    '{
      attempt_num: $attempt_num,
      backend: $backend,
      pattern: $pattern,
      trial_id: $trial_id,
      trial_bundle_id: $trial_bundle_id,
      trial_run_id: $trial_run_id,
      status: $status,
      return_code: $return_code,
      run_valid: $run_valid,
      invalid_reason: (if $invalid_reason == "" then null else $invalid_reason end),
      seed: $seed,
      request_seed: $request_seed,
      params: {
        a_req: $a_req,
        b_req: $b_req,
        a_conc: $a_conc,
        b_conc: $b_conc,
        long_range: $long_range
      },
      metrics: {
        system_pressure_snapshot_present: ($system_pressure_ok == 1),
        kvbm_pressure_snapshot_present: ($kvbm_pressure_ok == 1)
      },
      artifacts: {
        trial_dir: $trial_dir,
        trial_log: $trial_log,
        quick_summary_path: $quick_summary_path,
        run_summary_path: $run_summary_path
      },
      phase56_runs: [
        {
          bundle_id: $trial_bundle_id,
          run_id: $trial_run_id,
          quick_summary_path: $quick_summary_path,
          run_summary_path: $run_summary_path
        }
      ]
    }' >> "${TRIALS_JSONL}"

  if [[ "${status}" == "ok" && "${STOP_ON_SUCCESS}" == "1" ]]; then
    break
  fi
done

METRICS_DIFF_SUMMARY_PATH="${ANALYSIS_DIR}/metrics_diff_summary.json"
python3 - "${TRIALS_JSONL}" "${METRICS_DIFF_SUMMARY_PATH}" <<'PY'
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

trials_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
metric_line_re = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[Nn]a[Nn]|[+-]?[Ii]nf(?:inity)?)")
eps = 1e-9


def load_trials(path: Path):
    rows = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def parse_metric_file(path: Path):
    names = set()
    sums = defaultdict(float)
    if not path.exists():
        return names, sums
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = metric_line_re.match(line)
        if not m:
            continue
        name = m.group(1)
        try:
            value = float(m.group(2))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        names.add(name)
        sums[name] += value
    return names, sums


def trial_metric_state(trial):
    quick_summary = Path(trial.get("artifacts", {}).get("quick_summary_path", ""))
    analysis_dir = quick_summary.parent if quick_summary.name else Path()
    pressure_files = [
        analysis_dir / "metrics_system_pressure.prom",
        analysis_dir / "metrics_kvbm_pressure.prom",
    ]
    replay_files = [
        analysis_dir / "metrics_system_replay.prom",
        analysis_dir / "metrics_kvbm_replay.prom",
    ]

    all_names = set()
    pressure_sums = defaultdict(float)
    replay_sums = defaultdict(float)

    for p in pressure_files:
        names, sums = parse_metric_file(p)
        all_names.update(names)
        for name, value in sums.items():
            pressure_sums[name] += value
    for p in replay_files:
        names, sums = parse_metric_file(p)
        all_names.update(names)
        for name, value in sums.items():
            replay_sums[name] += value

    deltas = {}
    for name in set(pressure_sums) | set(replay_sums):
        deltas[name] = replay_sums.get(name, 0.0) - pressure_sums.get(name, 0.0)

    return {
        "names": all_names,
        "deltas": deltas,
    }


trials = load_trials(trials_path)
successes = [t for t in trials if t.get("status") == "ok"]

summary = {
    "successful_trials": len(successes),
    "new_metric_names_added": [],
    "metric_names_missing": [],
    "counters_with_nonzero_delta_count": 0,
}

if len(successes) >= 2:
    first = successes[0]
    second = successes[1]
    state_a = trial_metric_state(first)
    state_b = trial_metric_state(second)
    names_a = state_a["names"]
    names_b = state_b["names"]

    summary["compared_trials"] = [
        {
            "trial_id": first.get("trial_id"),
            "attempt_num": first.get("attempt_num"),
        },
        {
            "trial_id": second.get("trial_id"),
            "attempt_num": second.get("attempt_num"),
        },
    ]
    summary["new_metric_names_added"] = sorted(names_b - names_a)
    summary["metric_names_missing"] = sorted(names_a - names_b)

    common_names = set(state_a["deltas"]) & set(state_b["deltas"])
    nonzero_in_both = [
        name
        for name in common_names
        if abs(state_a["deltas"][name]) > eps and abs(state_b["deltas"][name]) > eps
    ]
    summary["counters_with_nonzero_delta_count"] = len(nonzero_in_both)
    summary["trial_1_nonzero_delta_count"] = sum(
        1 for value in state_a["deltas"].values() if abs(value) > eps
    )
    summary["trial_2_nonzero_delta_count"] = sum(
        1 for value in state_b["deltas"].values() if abs(value) > eps
    )
    summary["status"] = "ok"
else:
    summary["status"] = "insufficient_successful_trials"

out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

jq -s \
  --arg bundle_id "${BUNDLE_ID}" \
  --arg backend "${BACKEND}" \
  --arg pattern "${PATTERN}" \
  '{
    bundle_id: $bundle_id,
    backend: $backend,
    pattern: $pattern,
    total_attempts: length,
    success_count: ([ .[] | select(.status == "ok") ] | length),
    failed_count: ([ .[] | select(.status != "ok") ] | length),
    first_success: ([ .[] | select(.status == "ok") ] | .[0] // null),
    trials: .
  }' "${TRIALS_JSONL}" > "${ANALYSIS_DIR}/quick_summary.json"

echo "${BUNDLE_DIR}"

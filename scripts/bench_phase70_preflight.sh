#!/usr/bin/env bash
set -euo pipefail

normalize_bool() {
  local raw="${1:-0}"
  case "${raw,,}" in
    1|true|yes|on) echo "1" ;;
    0|false|no|off|"") echo "0" ;;
    *)
      echo "Invalid boolean value: ${raw}" >&2
      exit 2
      ;;
  esac
}

snippet() {
  local raw="${1:-}"
  raw="${raw//$'\n'/ }"
  printf '%s' "${raw:0:200}"
}

fetch_url() {
  local url="$1"
  local body_var="$2"
  local status_var="$3"
  local err_var="$4"
  local rc_var="$5"

  local tmp_body
  local tmp_err
  tmp_body="$(mktemp)"
  tmp_err="$(mktemp)"

  local http_status
  set +e
  http_status="$(curl -sS --max-time 4 -o "${tmp_body}" -w '%{http_code}' "${url}" 2>"${tmp_err}")"
  local rc=$?
  set -e

  local body
  local err
  body="$(cat "${tmp_body}")"
  err="$(cat "${tmp_err}")"
  rm -f "${tmp_body}" "${tmp_err}"

  printf -v "${body_var}" '%s' "${body}"
  printf -v "${status_var}" '%s' "${http_status}"
  printf -v "${err_var}" '%s' "${err}"
  printf -v "${rc_var}" '%s' "${rc}"
}

json_extract() {
  local payload="$1"
  local expr="$2"
  local out_var="$3"
  local out
  set +e
  out="$(printf '%s' "${payload}" | jq -er "${expr}" 2>/dev/null)"
  local rc=$?
  set -e
  if (( rc != 0 )); then
    return 1
  fi
  printf -v "${out_var}" '%s' "${out}"
}

BASE_URL="${BENCH_PHASE70_BASE_URL:-${BENCH_BASE_URL:-http://127.0.0.1:8000}}"
METRICS_KVBM_URL="${BENCH_PHASE70_METRICS_KVBM_URL:-${BENCH_METRICS_KVBM_URL:-http://127.0.0.1:6880/metrics}}"
TIMEOUT_S="${BENCH_PHASE70_PREFLIGHT_TIMEOUT_S:-25}"
SLEEP_S="${BENCH_PHASE70_PREFLIGHT_SLEEP_S:-2}"
ALLOW_MISSING_KVBM_METRICS="$(normalize_bool "${BENCH_PHASE70_ALLOW_MISSING_KVBM_METRICS:-0}")"
PYTHON_BIN="${BENCH_PHASE70_PYTHON_BIN:-python3}"
JSON_OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --metrics-kvbm-url)
      METRICS_KVBM_URL="$2"
      shift 2
      ;;
    --timeout-s)
      TIMEOUT_S="$2"
      shift 2
      ;;
    --sleep-s)
      SLEEP_S="$2"
      shift 2
      ;;
    --allow-missing-kvbm-metrics)
      ALLOW_MISSING_KVBM_METRICS="$(normalize_bool "$2")"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --json-out)
      JSON_OUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! [[ "${TIMEOUT_S}" =~ ^[0-9]+$ ]] || (( TIMEOUT_S <= 0 )); then
  echo "TIMEOUT_S must be a positive integer; got ${TIMEOUT_S}" >&2
  exit 2
fi
if ! [[ "${SLEEP_S}" =~ ^[0-9]+$ ]] || (( SLEEP_S <= 0 )); then
  echo "SLEEP_S must be a positive integer; got ${SLEEP_S}" >&2
  exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for Phase70 preflight." >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required for Phase70 preflight JSON parsing." >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python is required for Phase70 preflight (missing: ${PYTHON_BIN})." >&2
  exit 2
fi

echo "Phase70 preflight config: base_url=${BASE_URL} metrics_kvbm_url=${METRICS_KVBM_URL}"

reason_code=""
parse_target=""
ok="1"
decision_grade="1"
model_count="0"
kvbm_metrics_ok="0"
kvbm_metrics_missing=""

health_payload=""
models_payload=""
metrics_payload=""
health_err=""
models_err=""
metrics_err=""
health_http_status="000"
models_http_status="000"
metrics_http_status="000"

# 1) Frontend health gate.
fetch_url "${BASE_URL}/health" health_payload health_http_status health_err health_rc
if [[ "${health_rc}" != "0" || ! "${health_http_status}" =~ ^2 ]]; then
  reason_code="PREFLIGHT_FRONTEND_UNHEALTHY"
else
  health_state=""
  if ! json_extract "${health_payload}" '.status // empty' health_state; then
    reason_code="PREFLIGHT_PARSE_ERROR"
    parse_target="health"
  else
    health_state="${health_state,,}"
    if [[ "${health_state}" != "ok" && "${health_state}" != "healthy" && "${health_state}" != "ready" ]]; then
      reason_code="PREFLIGHT_FRONTEND_UNHEALTHY"
    fi
  fi
fi

# 2) Model registration gate.
if [[ -z "${reason_code}" ]]; then
  fetch_url "${BASE_URL}/v1/models" models_payload models_http_status models_err models_rc
  if [[ "${models_rc}" != "0" || ! "${models_http_status}" =~ ^2 ]]; then
    reason_code="PREFLIGHT_NO_MODELS"
  else
    if ! json_extract "${models_payload}" 'if (.data | type) == "array" then (.data | length) else 0 end' model_count; then
      reason_code="PREFLIGHT_PARSE_ERROR"
      parse_target="models"
    elif ! [[ "${model_count}" =~ ^[0-9]+$ ]] || (( model_count <= 0 )); then
      reason_code="PREFLIGHT_NO_MODELS"
    fi
  fi
fi

# 3) KVBM metrics gate.
if [[ -z "${reason_code}" ]]; then
  fetch_url "${METRICS_KVBM_URL}" metrics_payload metrics_http_status metrics_err metrics_rc
  if [[ "${metrics_rc}" != "0" || ! "${metrics_http_status}" =~ ^2 ]]; then
    reason_code="PREFLIGHT_METRICS_UNAVAILABLE"
  else
    declare -a required_metrics=(
      "kvbm_matched_tokens"
      "kvbm_onboard_blocks_d2d"
      "kvbm_offload_blocks_h2d"
      "kvbm_host_cache_hit_rate"
      "kvbm_disk_cache_hit_rate"
    )
    missing=()
    for name in "${required_metrics[@]}"; do
      if ! grep -Eq "^${name}([[:space:]{]|$)" <<< "${metrics_payload}"; then
        missing+=("${name}")
      fi
    done
    if (( ${#missing[@]} == 0 )); then
      kvbm_metrics_ok="1"
    else
      kvbm_metrics_missing="$(IFS=,; echo "${missing[*]}")"
      reason_code="PREFLIGHT_METRICS_MISSING_COUNTERS"
    fi
  fi
fi

if [[ -n "${reason_code}" ]]; then
  case "${reason_code}" in
    PREFLIGHT_METRICS_UNAVAILABLE|PREFLIGHT_METRICS_MISSING_COUNTERS)
      if [[ "${ALLOW_MISSING_KVBM_METRICS}" == "1" ]]; then
        ok="1"
        decision_grade="0"
      else
        ok="0"
        decision_grade="0"
      fi
      ;;
    *)
      ok="0"
      decision_grade="0"
      ;;
  esac
fi

echo "Phase70 preflight http_status: health=${health_http_status} models=${models_http_status} metrics=${metrics_http_status}"

health_snippet="$(snippet "${health_payload}")"
models_snippet="$(snippet "${models_payload}")"
metrics_snippet="$(snippet "${metrics_payload}")"
health_err_snippet="$(snippet "${health_err}")"
models_err_snippet="$(snippet "${models_err}")"
metrics_err_snippet="$(snippet "${metrics_err}")"

if [[ -n "${JSON_OUT}" ]]; then
  PREFLIGHT_JSON_OUT="${JSON_OUT}" \
  PREFLIGHT_OK="${ok}" \
  PREFLIGHT_DECISION_GRADE="${decision_grade}" \
  PREFLIGHT_BASE_URL="${BASE_URL}" \
  PREFLIGHT_METRICS_URL="${METRICS_KVBM_URL}" \
  PREFLIGHT_HEALTH_STATUS="${health_http_status}" \
  PREFLIGHT_MODELS_STATUS="${models_http_status}" \
  PREFLIGHT_METRICS_STATUS="${metrics_http_status}" \
  PREFLIGHT_MODEL_COUNT="${model_count}" \
  PREFLIGHT_KVBM_METRICS_OK="${kvbm_metrics_ok}" \
  PREFLIGHT_KVBM_METRICS_MISSING="${kvbm_metrics_missing}" \
  PREFLIGHT_REASON_CODE="${reason_code}" \
  PREFLIGHT_PARSE_TARGET="${parse_target}" \
  PREFLIGHT_HEALTH_SNIPPET="${health_snippet}" \
  PREFLIGHT_MODELS_SNIPPET="${models_snippet}" \
  PREFLIGHT_METRICS_SNIPPET="${metrics_snippet}" \
  PREFLIGHT_HEALTH_ERR="${health_err_snippet}" \
  PREFLIGHT_MODELS_ERR="${models_err_snippet}" \
  PREFLIGHT_METRICS_ERR="${metrics_err_snippet}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import pathlib
from datetime import datetime, timezone

reason = os.environ.get("PREFLIGHT_REASON_CODE", "").strip()
payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "ok": bool(int(os.environ.get("PREFLIGHT_OK", "0"))),
    "decision_grade": bool(int(os.environ.get("PREFLIGHT_DECISION_GRADE", "0"))),
    "base_url": os.environ.get("PREFLIGHT_BASE_URL"),
    "metrics_kvbm_url": os.environ.get("PREFLIGHT_METRICS_URL"),
    "health_http_status": os.environ.get("PREFLIGHT_HEALTH_STATUS"),
    "models_http_status": os.environ.get("PREFLIGHT_MODELS_STATUS"),
    "metrics_http_status": os.environ.get("PREFLIGHT_METRICS_STATUS"),
    "model_count": int(os.environ.get("PREFLIGHT_MODEL_COUNT", "0") or 0),
    "kvbm_metrics_ok": bool(int(os.environ.get("PREFLIGHT_KVBM_METRICS_OK", "0"))),
    "kvbm_metrics_missing": [
        item for item in (os.environ.get("PREFLIGHT_KVBM_METRICS_MISSING", "") or "").split(",") if item
    ],
    "reason_code": (reason or None),
    "reason_codes": ([reason] if reason else []),
    "parse_target": (os.environ.get("PREFLIGHT_PARSE_TARGET", "") or None),
    "response_snippets": {
        "health": os.environ.get("PREFLIGHT_HEALTH_SNIPPET"),
        "models": os.environ.get("PREFLIGHT_MODELS_SNIPPET"),
        "metrics": os.environ.get("PREFLIGHT_METRICS_SNIPPET"),
    },
    "curl_error_snippets": {
        "health": os.environ.get("PREFLIGHT_HEALTH_ERR"),
        "models": os.environ.get("PREFLIGHT_MODELS_ERR"),
        "metrics": os.environ.get("PREFLIGHT_METRICS_ERR"),
    },
}

path = pathlib.Path(os.environ["PREFLIGHT_JSON_OUT"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(path)
PY
fi

if [[ "${ok}" == "1" && -z "${reason_code}" ]]; then
  echo "Phase70 preflight PASS base_url=${BASE_URL} models=${model_count} kvbm_metrics_ok=${kvbm_metrics_ok}"
  exit 0
fi
if [[ "${ok}" == "1" && -n "${reason_code}" ]]; then
  echo "Phase70 preflight PASS_NON_DECISION_GRADE reason=${reason_code} models=${model_count} kvbm_metrics_ok=${kvbm_metrics_ok}"
  exit 0
fi

echo "Phase70 preflight FAIL reason=${reason_code} base_url=${BASE_URL} metrics_url=${METRICS_KVBM_URL}" >&2
echo "health_snippet=${health_snippet}" >&2
echo "models_snippet=${models_snippet}" >&2
echo "metrics_snippet=${metrics_snippet}" >&2
if [[ -n "${health_err_snippet}" ]]; then
  echo "health_curl_error=${health_err_snippet}" >&2
fi
if [[ -n "${models_err_snippet}" ]]; then
  echo "models_curl_error=${models_err_snippet}" >&2
fi
if [[ -n "${metrics_err_snippet}" ]]; then
  echo "metrics_curl_error=${metrics_err_snippet}" >&2
fi
exit 1

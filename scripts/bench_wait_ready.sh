#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
TIMEOUT_S="${1:-300}"
SLEEP_S="${2:-5}"
REQUIRED_CONSECUTIVE="${BENCH_READY_CONSECUTIVE:-2}"
REQUIRE_ENDPOINTS="${BENCH_READY_REQUIRE_ENDPOINTS:-0}"

extract_count() {
  local payload="$1"
  local expr="$2"
  local out
  out="$(printf '%s' "${payload}" | jq -r "${expr}" 2>/dev/null || true)"
  out="$(printf '%s' "${out}" | tr -dc '0-9')"
  if [[ -n "${out}" ]]; then
    printf '%s\n' "${out}"
    return 0
  fi
  printf '0\n'
}

deadline=$((SECONDS + TIMEOUT_S))
ready_streak=0
while (( SECONDS < deadline )); do
  health="$(curl -sf "${BASE_URL}/health" 2>/dev/null || true)"
  models="$(curl -sf "${BASE_URL}/v1/models" 2>/dev/null || true)"
  endpoint_count="$(extract_count "${health:-{}}" 'if (.endpoints | type) == "array" then (.endpoints | length) else 0 end')"
  model_count="$(extract_count "${models:-{}}" 'if (.data | type) == "array" then (.data | length) else 0 end')"
  ready_now=0
  if [[ "${model_count}" -gt 0 ]]; then
    if [[ "${REQUIRE_ENDPOINTS}" == "1" ]]; then
      if [[ "${endpoint_count}" -gt 0 ]]; then
        ready_now=1
      fi
    else
      ready_now=1
    fi
  fi
  if [[ "${ready_now}" -eq 1 ]]; then
    ready_streak=$((ready_streak + 1))
    if [[ "${ready_streak}" -ge "${REQUIRED_CONSECUTIVE}" ]]; then
      echo "Ready: ${BASE_URL} (endpoints=${endpoint_count}, models=${model_count}, streak=${ready_streak}, require_endpoints=${REQUIRE_ENDPOINTS})"
      exit 0
    fi
  else
    ready_streak=0
  fi
  if [[ "${REQUIRE_ENDPOINTS}" == "1" ]]; then
    echo "Waiting for ${BASE_URL} endpoints>0 and models>0 (now endpoints=${endpoint_count}, models=${model_count}, streak=${ready_streak}/${REQUIRED_CONSECUTIVE})..."
  else
    echo "Waiting for ${BASE_URL} models>0 (now endpoints=${endpoint_count}, models=${model_count}, streak=${ready_streak}/${REQUIRED_CONSECUTIVE})..."
  fi
  sleep "${SLEEP_S}"
done

echo "Timeout waiting for readiness at ${BASE_URL}" >&2
curl -s "${BASE_URL}/health" || true
curl -s "${BASE_URL}/v1/models" || true
exit 1

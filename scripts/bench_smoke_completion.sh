#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
USER_MSG="${1:-Give me 3 bullets about NVMe KV offload benchmarking.}"

MODEL_ID="$(curl -s "${BASE_URL}/v1/models" | jq -r '.data[0].id')"
if [[ -z "${MODEL_ID}" || "${MODEL_ID}" == "null" ]]; then
  echo "No model id returned from ${BASE_URL}/v1/models" >&2
  exit 1
fi

PROMPT=$'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'"${USER_MSG}"$'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

curl -sS "${BASE_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg model "${MODEL_ID}" \
    --arg prompt "${PROMPT}" \
    '{
      model: $model,
      prompt: $prompt,
      max_tokens: 128,
      temperature: 0.2,
      stop: ["<|eot_id|>"],
      stream: false
    }')" \
  | jq -r '.choices[0].text'


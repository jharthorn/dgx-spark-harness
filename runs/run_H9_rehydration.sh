#!/usr/bin/env bash
set -euo pipefail

# H9 context re-hydration (Test_Plan_v3.3 Section 6)
# Build a session, allow KV eviction to Tier2, then resume.

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackB"
PROFILE=${PROFILE:-stress}  # Stress profile to force eviction + rehydration from tier2
apply_profile_env "$PROFILE"
MODEL=${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
MODEL_TAG=${MODEL_TAG:-$(model_tag "$MODEL")}
WORKLOAD=${WORKLOAD:-sessioned_chat}
CTX=${CTX:-4096}
CONCURRENCY=${CONCURRENCY:-8}
DURATION_BUILD=${DURATION_BUILD:-180}
DURATION_RESUME=${DURATION_RESUME:-90}
EVICT_WAIT=${EVICT_WAIT:-60}
SESSION_IDLE_S=${SESSION_IDLE_S:-30}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:9000/v1/completions}
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}
# Keep the same seed across phases so session_ids line up for resume.
SEED=${SEED:-4242}

RUN_ID_BASE="$(rt_ts)_H9_${PROFILE}_${STACK}_${MODEL_TAG}_ctx${CTX}"

# Phase 1: build session
RUN_ID_BUILD="${RUN_ID_BASE}_build"
RUN_DIR_BUILD="$RESULTS_BASE/${RUN_ID_BUILD}"
ensure_run_dir "$RUN_DIR_BUILD"

cat > "$RUN_DIR_BUILD/config.yaml" <<EOF
stack: ${STACK}
profile: ${PROFILE}
model: ${MODEL}
workload: ${WORKLOAD}
context_tokens: ${CTX}
concurrency: ${CONCURRENCY}
duration_s: ${DURATION_BUILD}
endpoint: ${ENDPOINT}
nonce_per_user: true
seed: ${SEED}
phase: build
session_phase: build
session_min_turns: 4
session_max_turns: 10
session_resume_turns: 2
session_idle_s: ${SESSION_IDLE_S}
EOF

start_sysmon "$RUN_DIR_BUILD" "B"
start_telemetry "$RUN_DIR_BUILD"
python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR_BUILD/config.yaml" --run-id "$RUN_ID_BUILD" --output-dir "$RUN_DIR_BUILD" --endpoint "${ENDPOINT}"
stop_sysmon "$RUN_DIR_BUILD"
stop_telemetry "$RUN_DIR_BUILD"

echo "Sleeping ${EVICT_WAIT}s to encourage KV eviction to tier2..."
if [[ -n "${EVICT_LOAD_CMD:-}" ]]; then
  echo "Background load during idle: ${EVICT_LOAD_CMD}"
  bash -c "${EVICT_LOAD_CMD}" >/tmp/h9_eviction_load.log 2>&1 &
  BG_LOAD_PID=$!
fi
sleep "$EVICT_WAIT"
if [[ -n "${BG_LOAD_PID:-}" ]]; then
  wait "${BG_LOAD_PID}" || true
fi

# Phase 2: resume session to measure re-hydration
RUN_ID_RESUME="${RUN_ID_BASE}_resume"
RUN_DIR_RESUME="$RESULTS_BASE/${RUN_ID_RESUME}"
ensure_run_dir "$RUN_DIR_RESUME"

cat > "$RUN_DIR_RESUME/config.yaml" <<EOF
stack: ${STACK}
profile: ${PROFILE}
model: ${MODEL}
workload: ${WORKLOAD}
context_tokens: ${CTX}
concurrency: ${CONCURRENCY}
duration_s: ${DURATION_RESUME}
endpoint: ${ENDPOINT}
nonce_per_user: true
seed: ${SEED}
phase: resume
session_phase: resume
session_min_turns: 4
session_max_turns: 10
session_resume_turns: 2
session_idle_s: ${SESSION_IDLE_S}
EOF

start_sysmon "$RUN_DIR_RESUME" "B"
start_telemetry "$RUN_DIR_RESUME"
python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR_RESUME/config.yaml" --run-id "$RUN_ID_RESUME" --output-dir "$RUN_DIR_RESUME" --endpoint "${ENDPOINT}"
stop_sysmon "$RUN_DIR_RESUME"
stop_telemetry "$RUN_DIR_RESUME"

# Guardrails: ensure the run actually exercised sessioned_chat with session ids and resume turns.
if ! jq 'select(.workload=="sessioned_chat")' "$RUN_DIR_BUILD/metrics.jsonl" | head -1 | grep -q .; then
  echo "H9 FAILED: no sessioned_chat rows in build phase" >&2
  exit 1
fi
if ! jq 'select(.session_id!=null)' "$RUN_DIR_BUILD/metrics.jsonl" | head -1 | grep -q .; then
  echo "H9 FAILED: no session_id in build metrics" >&2
  exit 1
fi
if ! jq 'select(.resume==true)' "$RUN_DIR_RESUME/metrics.jsonl" | head -1 | grep -q .; then
  echo "H9 FAILED: no resume turns in resume phase" >&2
  exit 1
fi

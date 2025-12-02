#!/usr/bin/env bash
# Hypothesis: H8 – Hero side-by-side Stack A vs Stack B
# Typical profile: Spill
# Expected behavior: mixed-context hero run with aligned UMA budgets; compare Stack A/B behavior.
# See docs/Test_Plan_v3.3.md, section H8.
set -euo pipefail

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}
DURATION=${DURATION:-180}
CONCURRENCY=${CONCURRENCY:-}
if [[ -z "$CONCURRENCY" && -f "$HARNESS_DIR/runs/H0/uwork.txt" ]]; then
  CONCURRENCY="$(<"$HARNESS_DIR/runs/H0/uwork.txt")"
fi
CONCURRENCY=${CONCURRENCY:-32}
WORKLOAD=${WORKLOAD:-mixed_context}
MIX_SHORT_PCT=${MIX_SHORT_PCT:-0.6}
MIX_MEDIUM_PCT=${MIX_MEDIUM_PCT:-0.3}
MIX_LONG_PCT=${MIX_LONG_PCT:-0.1}
MIX_SHORT_MIN=${MIX_SHORT_MIN:-200}
MIX_SHORT_MAX=${MIX_SHORT_MAX:-500}
MIX_MEDIUM_MIN=${MIX_MEDIUM_MIN:-800}
MIX_MEDIUM_MAX=${MIX_MEDIUM_MAX:-1500}
MIX_LONG_MIN=${MIX_LONG_MIN:-2500}
MIX_LONG_MAX=${MIX_LONG_MAX:-3200}
BURSTINESS=${BURSTINESS:-bursty}
BURST_PAUSE_S=${BURST_PAUSE_S:-1.5}

# Stack A (control)
STACKA_PROFILE=${STACKA_PROFILE:-comfy}
STACKA_MODEL=${STACKA_MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
STACKA_MODEL_TAG=${STACKA_MODEL_TAG:-$(model_tag "$STACKA_MODEL")}
STACKA_ENDPOINT=${STACKA_ENDPOINT:-http://127.0.0.1:8355/v1/completions}
STACKA_CONTEXT=${STACKA_CONTEXT:-2048}

RUN_ID_A="$(rt_ts)_H8_${STACKA_PROFILE}_stackA_${STACKA_MODEL_TAG}_${STACKA_CONTEXT}"
RUN_DIR_A="$RESULTS_BASE/${RUN_ID_A}"
ensure_run_dir "$RUN_DIR_A"

cat > "$RUN_DIR_A/config.yaml" <<EOF
stack: stackA
profile: ${STACKA_PROFILE}
model: ${STACKA_MODEL}
workload: ${WORKLOAD}
context_tokens: ${STACKA_CONTEXT}
concurrency: ${CONCURRENCY}
duration_s: ${DURATION}
endpoint: ${STACKA_ENDPOINT}
nonce_per_user: true
seed: 42
mix_short_pct: ${MIX_SHORT_PCT}
mix_medium_pct: ${MIX_MEDIUM_PCT}
mix_long_pct: ${MIX_LONG_PCT}
mix_short_min: ${MIX_SHORT_MIN}
mix_short_max: ${MIX_SHORT_MAX}
mix_medium_min: ${MIX_MEDIUM_MIN}
mix_medium_max: ${MIX_MEDIUM_MAX}
mix_long_min: ${MIX_LONG_MIN}
mix_long_max: ${MIX_LONG_MAX}
burstiness: ${BURSTINESS}
burst_pause_s: ${BURST_PAUSE_S}
EOF

start_sysmon "$RUN_DIR_A" "A"
python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR_A/config.yaml" --run-id "$RUN_ID_A" --output-dir "$RUN_DIR_A" --endpoint "${STACKA_ENDPOINT}"
stop_sysmon "$RUN_DIR_A"

# Stack B (tiered)
STACKB_PROFILE=${STACKB_PROFILE:-spill}  # Spill profile to mirror H4B/H5 steady-state tier2 behavior
apply_profile_env "$STACKB_PROFILE"
STACKB_MODEL=${STACKB_MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
STACKB_MODEL_TAG=${STACKB_MODEL_TAG:-$(model_tag "$STACKB_MODEL")}
STACKB_ENDPOINT=${STACKB_ENDPOINT:-http://127.0.0.1:9000/v1/completions}
STACKB_CONTEXT=${STACKB_CONTEXT:-4096}

RUN_ID_B="$(rt_ts)_H8_${STACKB_PROFILE}_stackB_${STACKB_MODEL_TAG}_${STACKB_CONTEXT}"
RUN_DIR_B="$RESULTS_BASE/${RUN_ID_B}"
ensure_run_dir "$RUN_DIR_B"

cat > "$RUN_DIR_B/config.yaml" <<EOF
stack: stackB
profile: ${STACKB_PROFILE}
model: ${STACKB_MODEL}
workload: ${WORKLOAD}
context_tokens: ${STACKB_CONTEXT}
concurrency: ${CONCURRENCY}
duration_s: ${DURATION}
endpoint: ${STACKB_ENDPOINT}
nonce_per_user: true
seed: 42
mix_short_pct: ${MIX_SHORT_PCT}
mix_medium_pct: ${MIX_MEDIUM_PCT}
mix_long_pct: ${MIX_LONG_PCT}
mix_short_min: ${MIX_SHORT_MIN}
mix_short_max: ${MIX_SHORT_MAX}
mix_medium_min: ${MIX_MEDIUM_MIN}
mix_medium_max: ${MIX_MEDIUM_MAX}
mix_long_min: ${MIX_LONG_MIN}
mix_long_max: ${MIX_LONG_MAX}
burstiness: ${BURSTINESS}
burst_pause_s: ${BURST_PAUSE_S}
EOF

start_sysmon "$RUN_DIR_B" "B"
start_telemetry "$RUN_DIR_B"
python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR_B/config.yaml" --run-id "$RUN_ID_B" --output-dir "$RUN_DIR_B" --endpoint "${STACKB_ENDPOINT}"
stop_sysmon "$RUN_DIR_B"
stop_telemetry "$RUN_DIR_B"

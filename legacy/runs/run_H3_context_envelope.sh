#!/usr/bin/env bash
# Hypothesis: H3 – Context scaling envelope across Stack A/B
# Typical profile: Stress (per-stack override allowed)
# Expected behavior: sweep context lengths to find collapse/limit points for each stack.
# See docs/Test_Plan_v3.3.md, section H3.
set -euo pipefail

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACKS=(${STACKS:-stackA stackB})
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}
PROFILE_STACKA=${PROFILE_STACKA:-stress}
PROFILE_STACKB=${PROFILE_STACKB:-stress}  # Stress profile to chase collapse points at high context
DURATION=${DURATION:-180}
CONCURRENCY=${CONCURRENCY:-}
if [[ -z "$CONCURRENCY" && -f "$HARNESS_DIR/runs/H0/uwork.txt" ]]; then
  CONCURRENCY="$(<"$HARNESS_DIR/runs/H0/uwork.txt")"
fi
CONCURRENCY=${CONCURRENCY:-32}
WORKLOAD=${H3_WORKLOAD:-fixed_context}
MIX_SHORT_PCT=${MIX_SHORT_PCT:-0.6}
MIX_MEDIUM_PCT=${MIX_MEDIUM_PCT:-0.3}
MIX_LONG_PCT=${MIX_LONG_PCT:-0.1}
MIX_SHORT_MIN=${MIX_SHORT_MIN:-200}
MIX_SHORT_MAX=${MIX_SHORT_MAX:-500}
MIX_MEDIUM_MIN=${MIX_MEDIUM_MIN:-800}
MIX_MEDIUM_MAX=${MIX_MEDIUM_MAX:-1500}
MIX_LONG_MIN=${MIX_LONG_MIN:-2500}
MIX_LONG_MAX=${MIX_LONG_MAX:-3200}
BURSTINESS=${BURSTINESS:-even}
BURST_PAUSE_S=${BURST_PAUSE_S:-1.5}

for STACK in "${STACKS[@]}"; do
  if [[ "$STACK" == "stackB" ]]; then
    PROFILE=${PROFILE_STACKB}
    apply_profile_env "$PROFILE"
    ENDPOINT=${ENDPOINT_STACKB:-http://127.0.0.1:9000/v1/completions}
    MODEL=${MODEL_STACKB:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
    MODEL_TAG=${MODEL_TAG_STACKB:-$(model_tag "$MODEL")}
    CONTEXTS=(${H3_CONTEXTS_STACKB:-2048 4096 6144 8192})
  else
    PROFILE=${PROFILE_STACKA}
    ENDPOINT=${ENDPOINT_STACKA:-http://127.0.0.1:8355/v1/completions}
    MODEL=${MODEL_STACKA:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
    MODEL_TAG=${MODEL_TAG_STACKA:-$(model_tag "$MODEL")}
    CONTEXTS=(${H3_CONTEXTS_STACKA:-1024 2048 4096})
  fi

  for CTX in "${CONTEXTS[@]}"; do
    RUN_ID="$(rt_ts)_H3_${PROFILE}_${STACK}_${MODEL_TAG}_${CTX}"
    RUN_DIR="$RESULTS_BASE/${RUN_ID}"
    ensure_run_dir "$RUN_DIR"

    cat > "$RUN_DIR/config.yaml" <<EOF
stack: ${STACK}
profile: ${PROFILE}
model: ${MODEL}
workload: ${WORKLOAD}
context_tokens: ${CTX}
concurrency: ${CONCURRENCY}
duration_s: ${DURATION}
endpoint: ${ENDPOINT}
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

    stack_tag="A"
    if [[ "$STACK" == "stackB" ]]; then
      stack_tag="B"
    fi
    start_sysmon "$RUN_DIR" "$stack_tag"
    if [[ "$STACK" == "stackB" ]]; then
      start_telemetry "$RUN_DIR"
    fi
    python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR" --endpoint "${ENDPOINT}"
    stop_sysmon "$RUN_DIR"
    if [[ "$STACK" == "stackB" ]]; then
      stop_telemetry "$RUN_DIR"
    fi
  done
done

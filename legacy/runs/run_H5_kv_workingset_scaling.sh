#!/usr/bin/env bash
# Hypothesis: H5 – KV working-set scaling on Stack B
# Typical profile: Spill
# Expected behavior: mixed-context load scaling adapter sets; expect Tier2 churn with growing working set.
# See docs/Test_Plan_v3.3.md, section H5 (8.5).
set -euo pipefail

HARNESS_DIR=${HARNESS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$HARNESS_DIR/runs/_lib.sh"

STACK="stackB"
PROFILE=${PROFILE:-spill}  # Spill profile to exercise tier2 churn while scaling adapter sets
apply_profile_env "$PROFILE"
# Default to Llama 3.3 70B NVFP4; override MODEL env to switch.
MODEL=${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}
MODEL_TAG=${MODEL_TAG:-$(model_tag "$MODEL")}
WORKLOAD=${WORKLOAD:-mixed_context}
CTX=1024
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
CONCURRENCY=32
DURATION=180
ADAPTER_PROXY=${ADAPTER_PROXY:-true}
if [[ "$ADAPTER_PROXY" == "true" ]]; then
  ENDPOINT=${ENDPOINT:-http://127.0.0.1:9100/v1/completions}  # adapter_proxy strips adapter_id and generates tier2 IO
  ADAPTER_SETS=(${ADAPTER_SETS:-4 16 64})
else
  ENDPOINT=${ENDPOINT:-http://127.0.0.1:9000/v1/completions}
  ADAPTER_SETS=(${ADAPTER_SETS:-0})  # disable adapters if proxy not in use
fi
RESULTS_BASE=${RESULTS_BASE:-$HARNESS_DIR/results}
LORA_CHURN_MODE=${LORA_CHURN_MODE:-hot_cold}
LORA_HOT_RATIO=${LORA_HOT_RATIO:-0.2}
LORA_HOT_PROB=${LORA_HOT_PROB:-0.7}

for COUNT in "${ADAPTER_SETS[@]}"; do
  RUN_ID="$(rt_ts)_H5_${PROFILE}_${STACK}_${MODEL_TAG}_${COUNT}adpt"
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
lora_adapter_count: ${COUNT}
lora_churn_mode: ${LORA_CHURN_MODE}
lora_hot_ratio: ${LORA_HOT_RATIO}
lora_hot_prob: ${LORA_HOT_PROB}
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

  start_sysmon "$RUN_DIR" "B"
  start_telemetry "$RUN_DIR"
  python3 "$HARNESS_DIR/src/loadgen.py" --config "$RUN_DIR/config.yaml" --run-id "$RUN_ID" --output-dir "$RUN_DIR"
  stop_sysmon "$RUN_DIR"
  stop_telemetry "$RUN_DIR"
done

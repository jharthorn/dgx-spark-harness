#!/usr/bin/env bash
set -euo pipefail

# runs/model_env.sh
# Centralizes model-specific metadata for run scripts.
# Usage: source ./model_env.sh <MODEL_TAG>  (MODEL_TAG ∈ {L8B, L70B})

if [[ $# -lt 1 ]]; then
  echo "Usage: source runs/model_env.sh <L8B|L70B>" >&2
  exit 1
fi

MODEL_TAG_REQUEST="$1"

case "$MODEL_TAG_REQUEST" in
  L8B|l8b)
    export MODEL_TAG_SHORT="L8B"
    export MODEL_HANDLE=${MODEL_HANDLE:-"nvidia/Llama-3.3-8B-Instruct-FP4"}
    ;;
  L70B|l70b)
    export MODEL_TAG_SHORT="L70B"
    export MODEL_HANDLE=${MODEL_HANDLE:-"nvidia/Llama-3.3-70B-Instruct-FP4"}
    ;;
  *)
    echo "Unsupported model tag: $MODEL_TAG_REQUEST (expected L8B or L70B)" >&2
    exit 1
    ;;
esac

echo "[model_env] Using model ${MODEL_TAG_SHORT} (handle=${MODEL_HANDLE})"

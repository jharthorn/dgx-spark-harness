#!/usr/bin/env bash
set -euo pipefail

# Model metadata helper shared by setup/launch scripts.
# Usage: source runs/model_env.sh <L8B|L70B>

MODEL_TAG=${1:?Usage: source runs/model_env.sh <L8B|L70B>}

case "$MODEL_TAG" in
  L70B)
    export MODEL_TAG_SHORT="L70B"
    # Default to NVIDIA's FP4 Llama 3.3 70B Instruct checkpoint; override via MODEL_HANDLE env if needed.
    export MODEL_HANDLE=${MODEL_HANDLE:-"nvidia/Llama-3.3-70B-Instruct-FP4"}
    ;;
  L8B)
    export MODEL_TAG_SHORT="L8B"
    # Default to Meta's Llama 3.1 8B Instruct; override via MODEL_HANDLE env if needed.
    export MODEL_HANDLE=${MODEL_HANDLE:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
    ;;
  *)
    echo "Unsupported model tag: $MODEL_TAG (expected L8B or L70B)" >&2
    exit 1
    ;;
esac

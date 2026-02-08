#!/usr/bin/env bash
set -euo pipefail

# launch_triton_server.sh
# Boots the multi-LoRA Triton server used by H1/H5/H6 on port 8000.
#
# Usage:
#   ./launch_triton_server.sh L70B
#   ./launch_triton_server.sh L8B
#
# Environment overrides:
#   TRITON_IMAGE  - Container image tag (default: nvcr.io/nvidia/tritonserver:24.08-py3)
#   MODEL_REPO    - Model repository path (default: $HARNESS_DIR/model_repository)
#   MULTI_LORA_CONFIG - Path to multi_lora.json (default: $HARNESS_DIR/multi_lora.json)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export HARNESS_DIR=${HARNESS_DIR:-$SCRIPT_DIR}
MODEL_TAG=${1:-L70B}

source "$HARNESS_DIR/runs/model_env.sh" "$MODEL_TAG"

TRITON_IMAGE=${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.08-py3}
MODEL_REPO=${MODEL_REPO:-$HARNESS_DIR/model_repository}
MULTI_LORA_CONFIG=${MULTI_LORA_CONFIG:-$HARNESS_DIR/multi_lora.json}

if [[ ! -d "$MODEL_REPO" ]]; then
  echo "ERROR: Model repository not found at $MODEL_REPO" >&2
  exit 1
fi
if [[ ! -f "$MULTI_LORA_CONFIG" ]]; then
  echo "ERROR: multi_lora.json not found at $MULTI_LORA_CONFIG" >&2
  exit 1
fi

echo "--- Launching Triton server for $MODEL_TAG_SHORT ---"
echo "  Model repository : $MODEL_REPO"
echo "  multi_lora.json  : $MULTI_LORA_CONFIG"
echo "  Container image  : $TRITON_IMAGE"

docker run --rm -it \
  --name triton_lora_server \
  --gpus all \
  --ipc host \
  --network host \
  -e MODEL_TAG="$MODEL_TAG_SHORT" \
  -e TRT_LLM_MULTI_LORA_CONFIG="/workspace/multi_lora.json" \
  -v "$MODEL_REPO:/models" \
  -v "$MULTI_LORA_CONFIG:/workspace/multi_lora.json:ro" \
  "$TRITON_IMAGE" \
  tritonserver \
    --model-repository /models \
    --http-port 8000 \
    --grpc-port 8001 \
    --metrics-port 8002 \
    --log-verbose 0 \
    --strict-readiness=false

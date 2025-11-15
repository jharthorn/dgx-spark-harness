#!/usr/bin/env bash
set -euo pipefail

# setup_triton_server_llama.sh
# Rebuilds the TensorRT-LLM engine + Triton model repo for Llama 3.x with custom context limits.
#
# Usage:
#   ./setup_triton_server_llama.sh L70B --max-input-len 16384 --max-seq-len 32768
#   ./setup_triton_server_llama.sh L8B   # uses defaults (8192/16384)
#
# The script launches an NVIDIA TensorRT-LLM container, downloads the Hugging Face checkpoint,
# builds the engine with the requested sequence lengths, and syncs it into model_repository/.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <L8B|L70B> [--max-input-len TOKENS] [--max-seq-len TOKENS] [--engine-dir PATH]" >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

validate_positive_int() {
  local value=$1
  if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
    echo "Expected positive integer, got '$value'" >&2
    exit 1
  fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export HARNESS_DIR=${HARNESS_DIR:-$SCRIPT_DIR}
MODEL_TAG="$1"; shift

MAX_INPUT_LEN=${MAX_INPUT_LEN:-8192}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-16384}
ENGINE_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-input-len)
      MAX_INPUT_LEN="$2"; shift 2;;
    --max-seq-len)
      MAX_SEQ_LEN="$2"; shift 2;;
    --engine-dir)
      ENGINE_DIR_OVERRIDE="$2"; shift 2;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

validate_positive_int "$MAX_INPUT_LEN"
validate_positive_int "$MAX_SEQ_LEN"

if (( MAX_SEQ_LEN < MAX_INPUT_LEN )); then
  echo "max-seq-len ($MAX_SEQ_LEN) must be >= max-input-len ($MAX_INPUT_LEN)" >&2
  exit 1
fi

require_cmd docker

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be exported so the container can download the model." >&2
  exit 1
fi

source "$HARNESS_DIR/runs/model_env.sh" "$MODEL_TAG"

TRT_LLM_IMAGE=${TRT_LLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
ENGINE_DIR=${ENGINE_DIR_OVERRIDE:-$HARNESS_DIR/trt_engine_${MODEL_TAG_SHORT}_ctx${MAX_INPUT_LEN}}
MODEL_REPO_TARGET=${MODEL_REPO_TARGET:-$HARNESS_DIR/model_repository/tensorrt_llm/1}

case "$ENGINE_DIR" in
  "$HARNESS_DIR"/*) ;;
  *) echo "ENGINE_DIR must live under $HARNESS_DIR (was $ENGINE_DIR)" >&2; exit 1;;
esac

case "$MODEL_REPO_TARGET" in
  "$HARNESS_DIR"/*) ;;
  *) echo "MODEL_REPO_TARGET must live under $HARNESS_DIR (was $MODEL_REPO_TARGET)" >&2; exit 1;;
esac

ENGINE_SUBPATH=${ENGINE_DIR#"$HARNESS_DIR"/}

mkdir -p "$HF_CACHE" "$ENGINE_DIR"

echo "--- Rebuilding TensorRT-LLM engine ---"
echo " Model        : $MODEL_HANDLE"
echo " Max input    : $MAX_INPUT_LEN tokens"
echo " Max sequence : $MAX_SEQ_LEN tokens"
echo " Output dir   : $ENGINE_DIR"

docker run --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$HARNESS_DIR:/workspace/harness" \
  "$TRT_LLM_IMAGE" \
  bash -c "
    set -euo pipefail
    BUILD_DIR=/workspace/engine_build
    rm -rf \"\$BUILD_DIR\"
    mkdir -p \"\$BUILD_DIR\"
    echo 'Downloading $MODEL_HANDLE ...'
    huggingface-cli download \"$MODEL_HANDLE\" --cache-dir /root/.cache/huggingface --resume-download >/tmp/hf_download.log 2>&1
    echo 'Running trtllm-build ...'
    trtllm-build \
      --hf-model \"$MODEL_HANDLE\" \
      --max-input-len $MAX_INPUT_LEN \
      --max-seq-len $MAX_SEQ_LEN \
      --world-size 1 \
      --tp-size 1 \
      --pp-size 1 \
      --output-dir \"\$BUILD_DIR\" \
      --quantization-mode force_fp4 \
      --enable-gemm-plugin \
      --use-fused-mha
    echo 'Copying engine artifacts...'
    rm -rf \"/workspace/harness/$ENGINE_SUBPATH\"
    mkdir -p \"/workspace/harness/$ENGINE_SUBPATH\"
    cp -r \"\$BUILD_DIR\"/* \"/workspace/harness/$ENGINE_SUBPATH\"
  "

echo "--- Syncing Triton model_repository ---"
rm -rf -- "$MODEL_REPO_TARGET"
mkdir -p "$MODEL_REPO_TARGET"
cp -r "$ENGINE_DIR"/. "$MODEL_REPO_TARGET"/

CONFIG_FILE="$HARNESS_DIR/model_repository/ensemble/1/config.pbtxt"
if [[ -f "$CONFIG_FILE" ]]; then
  perl -0pi -e "s/max_context_length: \\d+/max_context_length: $MAX_INPUT_LEN/g" "$CONFIG_FILE"
  perl -0pi -e "s/max_tokens_in_output: \\d+/max_tokens_in_output: $MAX_SEQ_LEN/g" "$CONFIG_FILE"
fi

echo "Engine rebuild complete. Restart serve_llama33_70b_fp4.sh or launch_triton_server.sh to pick up the new context limits."

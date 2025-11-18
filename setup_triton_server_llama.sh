#!/usr/bin/env bash
set -euo pipefail

# setup_triton_server_llama.sh
# Rebuilds the TensorRT-LLM engine + Triton model repo for Llama 3.x with custom context limits.
# Supports L8B / L70B; tested with Stack B 8B flow as of 2025-11-18.
#
# Usage:
#   ./setup_triton_server_llama.sh L70B --max-input-len 16384 --max-seq-len 32768 --max-num-tokens 16384
#   ./setup_triton_server_llama.sh L8B   # uses defaults (8192/16384/8192)
#
# The script launches an NVIDIA TensorRT-LLM container, downloads the Hugging Face checkpoint,
# builds the engine with the requested sequence lengths, and syncs it into model_repository/.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <L8B|L70B> [--max-input-len TOKENS] [--max-seq-len TOKENS] [--max-num-tokens TOKENS] [--engine-dir PATH]" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/hftoken.txt" ]]; then
  export HF_TOKEN="$(<"$HOME/hftoken.txt")"
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
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-8192}
ENGINE_DIR_OVERRIDE=""
CKPT_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-input-len)
      MAX_INPUT_LEN="$2"; shift 2;;
    --max-seq-len)
      MAX_SEQ_LEN="$2"; shift 2;;
    --engine-dir)
      ENGINE_DIR_OVERRIDE="$2"; shift 2;;
    --max-num-tokens)
      MAX_NUM_TOKENS="$2"; shift 2;;
    --checkpoint-dir)
      CKPT_DIR_OVERRIDE="$2"; shift 2;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

validate_positive_int "$MAX_INPUT_LEN"
validate_positive_int "$MAX_SEQ_LEN"
validate_positive_int "$MAX_NUM_TOKENS"

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
CKPT_DIR_HOST=${CKPT_DIR_OVERRIDE:-}

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
echo " Max tokens   : $MAX_NUM_TOKENS tokens (build_config.max_num_tokens)"
echo " Output dir   : $ENGINE_DIR"
[[ -n "$CKPT_DIR_HOST" ]] && echo " Checkpoint   : $CKPT_DIR_HOST"

docker run --rm -i \
  --gpus all \
  --ipc host \
  --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e MODEL_TAG_SHORT="$MODEL_TAG_SHORT" \
  -e MAX_INPUT_LEN="$MAX_INPUT_LEN" \
  -e MAX_SEQ_LEN="$MAX_SEQ_LEN" \
  -e MAX_NUM_TOKENS="$MAX_NUM_TOKENS" \
  -e ENGINE_SUBPATH="$ENGINE_SUBPATH" \
  ${CKPT_DIR_HOST:+-e CKPT_DIR_OVERRIDE="/workspace/harness/${CKPT_DIR_HOST#$HARNESS_DIR/}"} \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$HARNESS_DIR:/workspace/harness" \
  --entrypoint /bin/bash \
  "$TRT_LLM_IMAGE" \
  -s <<'EOS'
	set -euxo pipefail
BUILD_DIR=/workspace/engine_build
ENGINE_DIR="/workspace/harness/${ENGINE_SUBPATH}"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
if [[ -n "${CKPT_DIR_OVERRIDE:-}" ]]; then
  CKPT_DIR="$CKPT_DIR_OVERRIDE"
  echo "Using provided checkpoint directory: $CKPT_DIR"
else
  echo "Downloading ${MODEL_HANDLE} ..."
  DOWNLOAD_LOG=/tmp/hf_download.log
  if ! huggingface-cli download "$MODEL_HANDLE" --cache-dir /root/.cache/huggingface --resume-download --local-dir /workspace/hf_ckpt --local-dir-use-symlinks False >"$DOWNLOAD_LOG" 2>&1; then
    echo "ERROR: huggingface-cli download failed; log follows:" >&2
    cat "$DOWNLOAD_LOG" >&2
    exit 1
  fi
  CKPT_DIR="/workspace/hf_ckpt"
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: could not locate checkpoint directory at $CKPT_DIR" >&2
    exit 1
  fi
  # If HF snapshot does not contain rank0.safetensors, convert to TRT-LLM format (tp1) so trtllm-build can load it.
  if [[ ! -f "$CKPT_DIR/rank0.safetensors" ]]; then
    CONVERTED_DIR="/workspace/harness/trt_llm_ckpt_${MODEL_TAG_SHORT}_tp1"
    echo "Converting HF checkpoint to TRT-LLM format at $CONVERTED_DIR ..."
    rm -rf "$CONVERTED_DIR"
    python3 /workspace/harness/scripts/convert_llama_checkpoint.py \
      --model_dir "$CKPT_DIR" \
      --output_dir "$CONVERTED_DIR" \
      --tp_size 1 \
      --dtype bfloat16
    CKPT_DIR="$CONVERTED_DIR"
  fi
fi
MODEL_CONFIG=/tmp/model_config.json
export CKPT_DIR MODEL_CONFIG
python3 - <<'PY'
import json, os

ckpt = os.environ["CKPT_DIR"]
cfg = os.path.join(ckpt, "config.json")
with open(cfg, "r") as f:
    data = json.load(f)

arch = data.get("architecture")
if not arch:
    arch_candidates = data.get("architectures") or []
    arch = arch_candidates[0] if arch_candidates else data.get("model_type", "")
if not arch:
    arch = "LlamaForCausalLM"

data["architecture"] = arch
data["architectures"] = [arch]
data["dtype"] = str(data.get("torch_dtype", "bfloat16"))

model_config_path = os.environ["MODEL_CONFIG"]
with open(model_config_path, "w") as f:
    json.dump(data, f)

# Also patch the checkpoint config.json in-place so trtllm-build picks up the architecture key.
with open(cfg, "w") as f:
    json.dump(data, f)

print(f"Wrote model config to {model_config_path} with architecture={arch}")
print(open(model_config_path, "r").read())
PY
echo "Using checkpoint: $CKPT_DIR"
echo "Running trtllm-build ..."
trtllm-build \
  --checkpoint_dir "$CKPT_DIR" \
  --output_dir "$BUILD_DIR" \
  --max_batch_size 64 \
  --max_input_len "$MAX_INPUT_LEN" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --max_num_tokens "$MAX_NUM_TOKENS" \
  --model_config "$MODEL_CONFIG" \
  --gemm_plugin nvfp4 \
  --remove_input_padding enable \
  --context_fmha enable \
  --kv_cache_type paged
echo "Copying engine artifacts..."
rm -rf "$ENGINE_DIR"
mkdir -p "$ENGINE_DIR"
cp -r "$BUILD_DIR"/* "$ENGINE_DIR"
echo "Build dir contents:"
ls -la "$BUILD_DIR"
echo "Engine dir contents:"
ls -la "$ENGINE_DIR"
EOS

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

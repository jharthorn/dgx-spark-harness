#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN=${HF_TOKEN:?HF_TOKEN must be set}
export MODEL_HANDLE="${MODEL_HANDLE:-nvidia/Llama-3.3-70B-Instruct-FP4}"
export HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
export ENGINE_DIR="${ENGINE_DIR:-}"                 # Optional: host path to a prebuilt TRT-LLM engine dir
export TOKENIZER_DIR="${TOKENIZER_DIR:-}"           # Optional: host path to tokenizer (defaults to HF cache)
export PORT="${PORT:-8355}"
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
export MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"     # server-side guardrail; set to 16000 for custom ctx
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"           # align with engine build (e.g., 32000 for ctx16k build)

mkdir -p "$HF_CACHE"

docker run --name trtllm_llm_server --rm -it \
  --gpus all --ipc host --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e ENGINE_DIR="$ENGINE_DIR" \
  -e TOKENIZER_DIR="$TOKENIZER_DIR" \
  -e PORT="$PORT" \
  -e MAX_BATCH_SIZE="$MAX_BATCH_SIZE" \
  -e MAX_NUM_TOKENS="$MAX_NUM_TOKENS" \
  -e MAX_SEQ_LEN="$MAX_SEQ_LEN" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  ${ENGINE_DIR:+-v "$ENGINE_DIR:/workspace/engine"} \
  ${TOKENIZER_DIR:+-v "$TOKENIZER_DIR:/workspace/tokenizer"} \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c '
    set -e
    use_engine=false
    model_arg="$MODEL_HANDLE"
    tokenizer_arg="$TOKENIZER_DIR"

    if [[ -n "$ENGINE_DIR" ]]; then
      use_engine=true
      model_arg="/workspace/engine"
      # When serving a local engine we must request backend=trt and point to tokenizer assets.
      backend_flag="--backend trt"
    else
      backend_flag=""
    fi

    # Ensure tokenizer path is available (for both HF + engine flows).
    if [[ -z "$tokenizer_arg" ]]; then
      echo "Downloading tokenizer/model assets for $MODEL_HANDLE ..."
      hf download "$MODEL_HANDLE"
      tokenizer_arg=$(python3 - <<PY
import glob, os
handle = os.environ.get("MODEL_HANDLE", "").replace("/", "--")
paths = glob.glob(f"/root/.cache/huggingface/hub/models--{handle}*/snapshots/*")
paths = [p for p in paths if os.path.isdir(p)]
paths.sort(key=os.path.getmtime, reverse=True)
print(paths[0] if paths else "")
PY
)
    else
      tokenizer_arg="/workspace/tokenizer"
    fi

    if [[ ! -d "$tokenizer_arg" ]]; then
      echo "ERROR: tokenizer directory not found at $tokenizer_arg" >&2
      exit 1
    fi

    # Ensure tokenizer config declares a known model_type (older snapshots may miss it).
    export TOKENIZER_ARG="$tokenizer_arg"
    python3 - <<'PY'
import json, os, sys

cfg = os.path.join(os.environ.get("TOKENIZER_ARG", ""), "config.json")
if not os.path.isfile(cfg):
    sys.exit(0)

with open(cfg, "r") as f:
    data = json.load(f)

changed = False
if "model_type" not in data:
    data["model_type"] = "llama"
    changed = True
if "architectures" not in data:
    arch = data.get("architecture") or data.get("model_type") or "LlamaForCausalLM"
    data["architectures"] = [arch]
    changed = True

if changed:
    with open(cfg, "w") as f:
        json.dump(data, f)
    print(f"Patched tokenizer config model_type/architectures in {cfg}")
PY

    echo "Writing extra-llm-api-config.yml..."
    cat > /tmp/extra-llm-api-config.yml <<EOF
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.90
EOF

    if $use_engine; then
      echo "Starting trtllm-serve on :$PORT with engine from $ENGINE_DIR (backend=trt) ..."
    else
      echo "Starting trtllm-serve on :$PORT with HF model $MODEL_HANDLE ..."
    fi

    trtllm-serve "$model_arg" \
      ${backend_flag} \
      --tokenizer "$tokenizer_arg" \
      --max_batch_size "$MAX_BATCH_SIZE" \
      --max_num_tokens "$MAX_NUM_TOKENS" \
      --max_seq_len "$MAX_SEQ_LEN" \
      --trust_remote_code \
      --port "$PORT" \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
  '

#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN=${HF_TOKEN:?HF_TOKEN must be set}
export MODEL_HANDLE="nvidia/Llama-3.3-70B-Instruct-FP4"
export HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

mkdir -p "$HF_CACHE"

docker run --name trtllm_llm_server --rm -it \
  --gpus all --ipc host --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c '
    set -e
    echo "Downloading model $MODEL_HANDLE to HF cache..."
    hf download "$MODEL_HANDLE"

    echo "Writing extra-llm-api-config.yml..."
    cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.90
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF

    echo "Starting trtllm-serve on :8355 with $MODEL_HANDLE ..."
    trtllm-serve "$MODEL_HANDLE" \
      --max_batch_size 64 \
      --trust_remote_code \
      --port 8355 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
  '

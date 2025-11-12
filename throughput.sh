# (On the host - Terminal 1)
export MODEL_HANDLE="openai/gpt-oss-120b"

# This is your working server command from earlier
docker run --name trtllm_llm_server --rm -it --gpus all --ipc host --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c '
    # Set up the tokenizer files
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
    
    # Download the model (will use cache)
    hf download $MODEL_HANDLE && \
    
    # Create the config file
    cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
    
    # Run the server
    trtllm-serve "$MODEL_HANDLE" \
      --max_batch_size 64 \
      --trust_remote_code \
      --port 8355 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
'
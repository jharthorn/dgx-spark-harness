#!/usr/bin/env bash
set -euo pipefail

# v1.5: Fixes SyntaxError in python one-liner

# This script prepares the entire environment needed to run
# the Triton Inference Server with TensorRT-LLM LoRA support.

# --- 1. Define Paths ---
export HARNESS_DIR="$HOME/dgx_spark_harness"
export MODEL_HANDLE="openai/gpt-oss-120b"
export HF_CACHE="$HOME/.cache/huggingface"
# Use the same container as the trtllm-serve, as it has the build tools
export TRT_CONTAINER="nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev"

# Paths this script will create:
export ENGINE_DIR="$HARNESS_DIR/trt_engine_bs256"
export MODEL_REPO_DIR="$HARNESS_DIR/model_repository"
export LORA_DIR="$HARNESS_DIR/inputs/lora_adapters" # We'll mount this

echo "--- Starting Triton Setup ---"
echo "Creating directories..."
mkdir -p "$ENGINE_DIR"
mkdir -p "$MODEL_REPO_DIR"
mkdir -p "$LORA_DIR"

# --- 2. Build the TRT-LLM Engine ---
# We are building the "High Throughput" (bs=256, ctx=2048) engine.
# This runs `trtllm-build` inside the container.

echo "Building TensorRT-LLM Engine (bs=256, ctx=2048)..."
docker run --rm -it --gpus all --ipc host \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$ENGINE_DIR:/engine" \
  -v "$LORA_DIR:/lora_adapters" \
  "$TRT_CONTAINER" \
  bash -c '
    set -e
    echo "Starting engine build..."
    
    # --- FIX v1.4: Find the exact model path ---
    echo "Downloading model '"$MODEL_HANDLE"' to cache..."
    hf download '"$MODEL_HANDLE"'
    
    echo "Finding model snapshot path..."
    # --- FIX v1.5: Corrected python quoting ---
    MODEL_PATH=$(python3 -c "from huggingface_hub import hf_hub_download; f = hf_hub_download(repo_id=\"$MODEL_HANDLE\", filename='config.json'); print(f.rsplit('/',1)[0])")
    
    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: Could not find model snapshot path."
        exit 1
    fi
    echo "Found model snapshot at: $MODEL_PATH"
    # --- END FIX ---

    # Now, build using the explicit path
    trtllm-build \
      --checkpoint_dir "$MODEL_PATH" \
      --output_dir /engine \
      --gemm_plugin bfloat16 \
      --moe_plugin bfloat16 \
      --max_batch_size 256 \
      --max_input_len 2048 \
      --max_seq_len 3072 \
      --max_beam_width 1 \
      --max_lora_rank 64 \
      --lora_target_modules "attn_qkv" "mlp_gate_up" "mlp_4h_to_h"
    
    echo "Engine build complete. Files are in /engine."
'

# --- 3. Create the Model Repository ---
echo "Creating Triton Model Repository structure..."
mkdir -p "$MODEL_REPO_DIR/ensemble/1"
mkdir -p "$MODEL_REPO_DIR/preprocessing/1"
mkdir -p "$MODEL_REPO_DIR/postprocessing/1"
mkdir -p "$MODEL_REPO_DIR/tensorrt_llm/1"

# --- 4. Write ALL config.pbtxt files ---
echo "Writing config.pbtxt files..."

# --- Config for the core LLM (tensorrt_llm/config.pbtxt) ---
cat > "$MODEL_REPO_DIR/tensorrt_llm/config.pbtxt" <<EOF
name: "tensorrt_llm"
backend: "tensorrtllm"
model_repository {
  root_location: "/engine"
}
model_version_policy {
  specific {
    versions: 1
  }
}
parameters {
  key: "max_beam_width"
  value: { string_value: "1" }
}
parameters {
  key: "gpt_model_type"
  value: { string_value: "gpt" }
}
parameters {
  key: "lora_cache_size_bytes"
  value: { string_value: "8589934592" } # 8 GiB
}
parameters {
  key: "lora_dir"
  value: { string_value: "/lora_adapters" }
}
dynamic_batching {
  max_queue_delay_microseconds: 10000 # 10ms
}
EOF

# --- Config for the Preprocessing step ---
cat > "$MODEL_REPO_DIR/preprocessing/config.pbtxt" <<EOF
name: "preprocessing"
backend: "python"
max_batch_size: 256
inputs [
  {
    name: "QUERY"
    data_type: TYPE_STRING
    dims: [ -1 ]
  },
  {
    name: "LORA_ADAPTER_NAME"
    data_type: TYPE_STRING
    dims: [ -1 ]
    optional: true
  }
]
outputs [
  {
    name: "INPUT_ID"
    data_type: TYPE_INT32
    dims: [ -1, -1 ]
  },
  {
    name: "REQUEST_INPUT_LEN"
    data_type: TYPE_INT32
    dims: [ -1, 1 ]
  },
  {
    name: "LORA_NAMES"
    data_type: TYPE_STRING
    dims: [ -1, 1 ]
    optional: true
  }
]
EOF

# --- Config for the Postprocessing step ---
cat > "$MODEL_REPO_DIR/postprocessing/config.pbtxt" <<EOF
name: "postprocessing"
backend: "python"
max_batch_size: 256
inputs [
  {
    name: "OUTPUT_ID"
    data_type: TYPE_INT32
    dims: [ -1, -1 ]
  }
]
outputs [
  {
    name: "RESPONSE"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]
EOF

# --- Config for the main Ensemble ---
cat > "$MODEL_REPO_DIR/ensemble/config.pbtxt" <<EOF
name: "ensemble"
platform: "ensemble"
max_batch_size: 256
inputs [
  {
    name: "QUERY"
    data_type: TYPE_STRING
    dims: [ -1 ]
  },
  {
    name: "LORA_ADAPTER_NAME"
    data_type: TYPE_STRING
    dims: [ -1 ]
    optional: true
  }
]
outputs [
  {
    name: "RESPONSE"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

# Ensemble Chaining:
# QUERY -> preprocessing -> INPUT_ID -> tensorrt_llm
# LORA_ADAPTER_NAME -> preprocessing -> LORA_NAMES -> tensorrt_llm
# tensorrt_llm -> OUTPUT_ID -> postprocessing -> RESPONSE
ensemble_scheduling {
  step [
    {
      model_name: "preprocessing"
      model_version: -1
      input_map [
        {
          key: "QUERY"
          value: "QUERY"
        },
        {
          key: "LORA_ADAPTER_NAME"
          value: "LORA_ADAPTER_NAME"
        }
      ]
      output_map [
        {
          key: "INPUT_ID"
          value: "ensemble_input_id"
        },
        {
          key: "REQUEST_INPUT_LEN"
          value: "ensemble_request_input_len"
        },
        {
          key: "LORA_NAMES"
          value: "ensemble_lora_names"
        }
      ]
    },
    {
      model_name: "tensorrt_llm"
      model_version: -1
      input_map [
        {
          key: "input_ids"
          value: "ensemble_input_id"
        },
        {
          key: "input_lengths"
          value: "ensemble_request_input_len"
        },
        {
          key: "lora_names"
          value: "ensemble_lora_names"
        }
      ]
      output_map [
        {
          key: "output_ids"
          value: "ensemble_output_id"
        }
      ]
    },
    {
      model_name: "postprocessing"
      model_version: -1
      input_map [
        {
          key: "OUTPUT_ID"
          value: "ensemble_output_id"
        }
      ]
      output_map [
        {
          key:"RESPONSE"
          value: "RESPONSE"
        }
      ]
    }
  ]
}
EOF

echo "--- Triton Setup Complete ---"
echo "You are now ready to launch the Triton server."
echo "Engine is in: $ENGINE_DIR"
echo "Model Repo is in: $MODEL_REPO_DIR"
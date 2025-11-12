# DGX Spark LLM Storage Test Harness (v2.2)

This harness is designed to test the performance of the NVIDIA DGX Spark's internal NVMe SSD under a memory-pressure-bound LLM inference workload.

This v2.2 plan is simplified to target the default trtllm-serve server (which is a "High Throughput" profile, bs=64). It focuses on H1 (Cache), H2 (Concurrency), and H6 (LoRA Storm) tests.

Harness Architecture

The harness is organized into five distinct directories, all managed from the /harness mount point.

/harness/
├── src/                # Core tools: sysmon.sh (telemetry), loadgen.py (load)
├── runs/               # Orchestration scripts: run_H1_cache.sh, run_H2_...sh, etc.
├── inputs/             # Static data: prompts/ (for H2/H6), lora_adapters/ (for H6)
├── results/            # (Git-ignored) RAW DATA: all .csv, .json, and .log files
└── analysis/           # Analysis tools: process_results.py, backfill_summary.py
    └── figures/        # (Git-ignored) Final plots (PNG)



Quick Start: Go-Live Checklist

This plan uses a two-container model: one for the (unprivileged) LLM server, and one for our (privileged) test harness.

Step 1: Run the LLM Server (Terminal 1)

This is the command you provided. It starts the trtllm-serve container on port 8355.

# Set the model to test
export MODEL_HANDLE="openai/gpt-oss-120b"

# Run the server container
docker run --name trtllm_llm_server --rm -it --gpus all --ipc host --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c '
    # Set up tokenizer files
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE [https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken](https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken) && \
    wget -P $TIKTOKEN_ENCODINGS_BASE [https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken) && \
    
    # Download the model (will use cache)
    hf download $MODEL_HANDLE && \
    
    # Create the config file (forces KV cache ON)
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


Step 2: Run the Test Harness (Terminal 2)

Once the server is running, open a new terminal.

Launch the Privileged Harness Container:
This container must be privileged to access host-level sysstat and smartctl.

docker run --gpus all -it --rm \
  --network host \
  --shm-size=1g \
  --privileged --pid=host \
  -v /my-models:/workspace \
  -v ~/dgx_spark_harness:/harness \
  -v /sys:/sys \
  -v /proc:/proc \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash


Install Dependencies (Inside Harness Container):

# (Inside container)
apt-get update && apt-get install -y sysstat smartmontools fio
pip install httpx pandas matplotlib


Run the Test Suite (Inside Harness Container):
The server is already running on localhost:8355. Our scripts are pre-configured to target it.

# (Inside container)
cd /harness/runs

# Run H1 (Cold/Warm)
bash ./run_H1_cache.sh

# Run H2 (Baseline Concurrency)
bash ./run_H2_concurrency.sh

# Run H6 (LoRA Storm Concurrency)
bash ./run_H6_lora_storm.sh


Step 3: Analyze Results (Inside Harness Container)

After the tests are complete, run the analysis script.

# (Inside container)
cd /harness/analysis

# Run ALL analysis and generate ALL plots
python3 ./process_results.py ALL


This will generate all plots (H1, H2, H6, H7, H8) in the /harness/analysis/figures/ directory.

Test Suite Overview (v2.2)

Script

Test ID

Purpose

run_H1_cache.sh

H1

Cache QoS: Measures p50 TTFT for a COLD vs. WARM model load.

run_H2_concurrency.sh

H2

Baseline Latency: Generates the p99 latency vs. io_wait curve under load.

run_H6_lora_storm.sh

H6

Workload A/B Test: Runs a concurrency sweep with LoRA switching.

(Analyzes H2/H6 Data)

H3

Eviction Analysis: Deep-dive time-series of a high-load H2/H6 run.

(Analyzes H2/H6 Data)

H7

smartctl Deltas: Plots total GB read and total read commands.

(Analyzes H2/H6 Data)

H8

mpstat Dynamics: Plots per-core CPU usage (usr, sys, iowait).
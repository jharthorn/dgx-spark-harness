# DGX Spark LLM Storage Harness

Full-detail working reference for maintaining and running Stack A (TRT-LLM UMA) and Stack B (Dynamo tiered KV) under Test_Plan_v3.0.

This harness drives UMA, KV cache, and NVMe under realistic LLM inference workloads on unified memory systems. It measures TTFT and E2E latency under queue pressure, context scaling, KV tiering, NVMe QoS perturbations, and KV working set growth.

---

# Table of contents

1. [Overview](#overview)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Quick start (Stack A, H0 smoke test)](#quick-start-stack-a-h0-smoke-test)
5. [Stack overview](#stack-overview)
6. [Stack B bring-up (Dynamo tiered KV)](#stack-b-bring-up-dynamo-tiered-kv)

   * [Infrastructure and discovery](#infrastructure-and-discovery)
   * [Frontend](#frontend)
   * [Worker (8B)](#worker-8b)
   * [Worker (70B)](#worker-70b)
   * [Sanity checks and H2B](#sanity-checks-and-h2b)
7. [Prompt length guardrail](#prompt-length-guardrail)
8. [Advanced: building a custom Dynamo + KVBM worker image](#advanced-building-a-custom-dynamo--kvbm-worker-image)
9. [Notes and TODOs](#notes-and-todos)

---

# Overview

This repository implements the v3 test harness defined in `Test_Plan_v3.0.md`.
It supports two serving stacks:

### Stack A (TRT-LLM UMA only)

* Triton + TRT-LLM
* UMA only; no Dynamo; no tiered KV
* Uses TRT engines and checkpoints if present
* Used for UMA control data: H0, H1, H2A, H3, H4A, H8A

### Stack B (Dynamo tiered KV)

* Dynamo frontend (HTTP) + `dynamo.trtllm` worker
* Tiered KV cache (HBM → UMA → NVMe)
* Does not use any TRT artifacts from Stack A
* Used for KV and storage tests: H0, H2B, H3, H4B, H5, H6, H7, H8B

The harness includes a load generator, realistic long prompts, concurrency and sessioned workloads, telemetry collectors, runner scripts, and summary analysis.

---

# Repository layout

```
configs/
  stackA_llama70b_baseline.yaml
  stackB_llama70b_dynamo_tiered.yaml
  stackB_llama8b_dynamo_tiered.yaml
  kvbm_llm_api_8b.yaml
  kvbm_llm_api_70b.yaml

src/
  loadgen.py
  workloads/
    fixed_context.py
    sessioned_chat.py
  telemetry/
    sysmon.sh
    (NVMe, GPU, Dynamo stubs)

runs/
  run_H0_queue_knee.sh
  run_H1_coldwarm_lora.sh
  run_H2A_uma_pressure.sh
  run_H2B_dynamo_kv_pressure.sh
  run_H4A_storage_qos.sh
  run_H4B_dynamo_storage_qos.sh
  run_H5_kv_workingset_scaling.sh
  run_H6_tier_sizing_policy.sh
  run_H7_kv_telemetry_sweep.sh
  _lib.sh

analysis/
  process_results.py
  backfill_summary.py
  figures/

inputs/
  prompts/        (128 to 64k token realistic incident prompts)
  lora_lists/

cards/
  Llama-3.1-8B-Instruct.json   (discovery card for Dynamo)

scripts/
  archive_results.sh
  patch_nixl_opt_in.sh
```

---

# Requirements

### Hardware

* UMA GPU system such as DGX Spark
* 128 GB unified memory recommended
* NVMe local storage for KV tier 2

### Software

* Ubuntu 22.04 or DGX OS
* Python 3.10 or 3.12
* Docker and docker compose
* Hugging Face token for:

  * meta-llama/Meta-Llama-3.1-8B-Instruct
  * meta-llama/Llama-3.3-70B-Instruct

Set your token:

```bash
export HF_TOKEN="$(<~/hftoken.txt)"
```

---

# Quick start (Stack A, H0 smoke test)

This validates the harness against a UMA-only TRT LLM endpoint on port 8355.

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start a TRT-LLM server

Example:

```bash
bash ./serve_llama33_70b_fp4.sh
```

Ensure the endpoint matches `configs/stackA_llama70b_baseline.yaml`.

### 3. Optional: harness inside container

```bash
docker run --gpus all -it --rm \
  --network host --ipc host \
  -v ~/dgx_spark_harness:/harness \
  -w /harness \
  -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nvidia/pytorch:xx.xx-py3
```

### 4. Run H0 queue knee

```bash
./runs/run_H0_queue_knee.sh http://127.0.0.1:8355/v1/completions
```

Output appears under:

```
results/<timestamp>_H0_stackA_*/{config.yaml,metrics.jsonl,sysmon.jsonl}
```

### 5. Analyze

```bash
python3 analysis/process_results.py
cat analysis/figures/summary_v3.csv
```

### 6. Archive results

```bash
./scripts/archive_results.sh
```

---

# Stack overview

## Stack A: UMA only, TRT-LLM

* Endpoint: 8355
* Config: `configs/stackA_llama70b_baseline.yaml`
* Uses TRT engines and checkpoints if present
* Test coverage: H0, H1, H2A, H3, H4A, H8A
* Default TRT engine is normally sufficient; custom engines only required if:

  * You need > default max sequence or input length
  * You want specific max_tokens limits for experiments

## Stack B: Dynamo tiered KV

* Endpoint: 9000
* Configs:

  * `stackB_llama70b_dynamo_tiered.yaml`
  * `stackB_llama8b_dynamo_tiered.yaml`
* KV cache tiers:

  * tier0: GPU HBM
  * tier1: UMA
  * tier2: NVMe
* Does not use any TRT engines from Stack A
* Test coverage: H0, H2B, H3, H4B, H5, H6, H7, H8B

## 8B testbed

* Lighter worker for development
* Uses `configs/stackB_llama8b_dynamo_tiered.yaml`
* Good for early H2B, QoS, and discovery validation

---

# Stack B bring-up (Dynamo tiered KV)

This section describes reliable bring-up for 8B and 70B stacks with Dynamo 0.7.0.

## Infrastructure and discovery

Start etcd + nats:

```bash
cd ~/dynamo
docker compose -f deploy/docker-compose.yml up -d
```

Seed the discovery store (only after a reset or if discovery is corrupt):

```bash
docker exec deploy-etcd-server-1 etcdctl del --prefix v1/mdc/dynamo/tensorrt_llm

docker exec -i deploy-etcd-server-1 etcdctl put \
  v1/mdc/dynamo/tensorrt_llm/generate/694d9aa7979575ab \
  < ~/dgx_spark_harness/cards/Llama-3.1-8B-Instruct.json
```

## Frontend

Run in a venv with the local wheels installed:

```bash
source ~/dyn-venv/bin/activate

export HF_TOKEN="$(<~/hftoken.txt)"
export DYN_DISCOVERY_BACKEND=kv_store
export DYN_STORE_KV=etcd
export DYN_NAMESPACE=dynamo

python3 -m dynamo.frontend --http-port 9000
```

The worker commands below pass `--extra-engine-args /workspace/kvbm_llm_api_config.yaml`, which mounts `configs/kvbm_llm_api_8b.yaml` or `configs/kvbm_llm_api_70b.yaml` into the container and applies their KV cache / CUDA graph settings. If you change those files, restart the worker to pick them up.

## Worker (8B)

```bash
eval "$(python3 scripts/stackB_tier_env.py --config configs/stackB_llama8b_dynamo_tiered.yaml)"
TIER2_ROOT=$(dirname "$DYN_KVBM_TIER2_PATH")

docker run --gpus all --ipc host --network host --rm -it \
  -e HF_TOKEN="$(<~/hftoken.txt)" \
  -e DYN_KVBM_METRICS=true \
  -e DYN_KVBM_METRICS_PORT=${DYN_KVBM_METRICS_PORT:-6880} \
  -e DYN_KVBM_TIER0_BYTES="$DYN_KVBM_TIER0_BYTES" \
  -e DYN_KVBM_TIER1_BYTES="$DYN_KVBM_TIER1_BYTES" \
  -e DYN_KVBM_TIER2_BYTES="$DYN_KVBM_TIER2_BYTES" \
  -e DYN_KVBM_CPU_CACHE_GB=12 \
  -e DYN_KVBM_DISK_CACHE_GB=256 \
  -e DYN_DISCOVERY_KV_EXPORT_ENABLED=false \
  -e DYN_KVBM_TIER2_PATH="$DYN_KVBM_TIER2_PATH" \
  -p ${DYN_KVBM_METRICS_PORT:-6880}:${DYN_KVBM_METRICS_PORT:-6880} \
  -v "$TIER2_ROOT":"$TIER2_ROOT" \
  -v ~/dgx_spark_harness:/workspace/harness \
  -v ~/dgx_spark_harness/configs/kvbm_llm_api_8b.yaml:/workspace/kvbm_llm_api_config.yaml \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  spark-dynamo-worker:latest \
  bash -lc "cd /workspace/harness/scripts && ./patch_nixl_opt_in.sh && \
            python3 -m dynamo.trtllm \
              --model-path nvidia/Llama-3.1-8B-Instruct-NVFP4 \
              --served-model-name nvidia/Llama-3.1-8B-Instruct-NVFP4 \
              --max-num-tokens 16000 \
              --max-batch-size 2 \
              --kv-block-size 32 \
              --extra-engine-args /workspace/kvbm_llm_api_config.yaml"
```

## Worker (70B)

```bash
eval "$(python3 scripts/stackB_tier_env.py --config configs/stackB_llama70b_dynamo_tiered.yaml)"
TIER2_ROOT=$(dirname "$DYN_KVBM_TIER2_PATH")

# Optional: shrink tiers to force spill for H2B storage pressure runs
export DYN_KVBM_TIER0_BYTES=${DYN_KVBM_TIER0_BYTES:-$((2 * 1024**3))}
export DYN_KVBM_TIER1_BYTES=${DYN_KVBM_TIER1_BYTES:-$((8 * 1024**3))}
export DYN_KVBM_TIER2_BYTES=${DYN_KVBM_TIER2_BYTES:-$((64 * 1024**3))}
export DYN_KVBM_CPU_CACHE_GB=${DYN_KVBM_CPU_CACHE_GB:-4}
export DYN_KVBM_DISK_CACHE_GB=${DYN_KVBM_DISK_CACHE_GB:-128}
# Let dynkv_ingest derive bytes accurately
export DYN_KVBM_KV_BLOCK_SIZE_BYTES=${DYN_KVBM_KV_BLOCK_SIZE_BYTES:-65536}

docker run --gpus all --ipc host --network host --rm -it \
  -e HF_TOKEN="$(<~/hftoken.txt)" \
  -e DYN_KVBM_METRICS=true \
  -e DYN_KVBM_METRICS_PORT=${DYN_KVBM_METRICS_PORT:-6880} \
  -e DYN_KVBM_TIER0_BYTES="$DYN_KVBM_TIER0_BYTES" \
  -e DYN_KVBM_TIER1_BYTES="$DYN_KVBM_TIER1_BYTES" \
  -e DYN_KVBM_TIER2_BYTES="$DYN_KVBM_TIER2_BYTES" \
  -e DYN_KVBM_CPU_CACHE_GB=12 \
  -e DYN_KVBM_DISK_CACHE_GB=256 \
  -e DYN_DISCOVERY_KV_EXPORT_ENABLED=false \
  -e DYN_KVBM_TIER2_PATH="$DYN_KVBM_TIER2_PATH" \
  -p ${DYN_KVBM_METRICS_PORT:-6880}:${DYN_KVBM_METRICS_PORT:-6880} \
  -v "$TIER2_ROOT":"$TIER2_ROOT" \
  -v ~/dgx_spark_harness/configs/kvbm_llm_api_70b.yaml:/workspace/kvbm_llm_api_config.yaml \
  -v ~/dgx_spark_harness:/workspace/harness \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  spark-dynamo-worker:latest \
  bash -lc "cd /workspace/harness/scripts && ./patch_nixl_opt_in.sh && \
            python3 -m dynamo.trtllm \
              --model-path nvidia/Llama-3.3-70B-Instruct-NVFP4 \
              --served-model-name nvidia/Llama-3.3-70B-Instruct-NVFP4 \
              --max-num-tokens 16000 \
              --max-batch-size 4 \
              --kv-block-size 32 \
              --extra-engine-args /workspace/kvbm_llm_api_config.yaml"
```

## Sanity checks and H2B

Check models:

```bash
curl -s http://127.0.0.1:9000/v1/models | jq .
```

Simple completion:

```bash
curl -i -X POST http://127.0.0.1:9000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/Llama-3.1-8B-Instruct-NVFP4","prompt":"Hello","max_tokens":64}'
```
```bash
curl -i -X POST http://127.0.0.1:9000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/Llama-3.3-70B-Instruct-NVFP4","prompt":"Hello","max_tokens":64}'
```

Run H2B:

```bash
MODEL=nvidia/Llama-3.1-8B-Instruct-NVFP4 \
ENDPOINT=http://127.0.0.1:9000/v1/completions \
CONCURRENCY=4 \
DURATION=30 \
./runs/run_H2B_dynamo_kv_pressure.sh
```

For 70B, set `MODEL=nvidia/Llama-3.3-70B-Instruct-NVFP4` and launch the worker with `configs/stackB_llama70b_dynamo_tiered.yaml` via `stackB_tier_env.py`.

Runners start telemetry automatically and write JSONL into `results/<run_id>/`: `sysmon.jsonl` (CPU/mem/NVMe/gpu summary), `nvme.jsonl` (iostat-derived), `gpu.jsonl` (per-GPU util/memory), and `dynkv.jsonl` (KVBM Prometheus scrape).

---

# Prompt length guardrail

For Stack A or Stack B, use tokenizer-aware truncation:

```bash
python3 src/loadgen.py \
  --endpoint http://127.0.0.1:8355/v1/completions \
  --stack stackA \
  --model nvidia/Llama-3.3-70B-Instruct-FP4 \
  --tokenizer nvidia/Llama-3.3-70B-Instruct-FP4 \
  --max_input_len 16000 \
  --input_len_margin 64 \
  --workload fixed_context \
  --context_tokens 16384 \
  --prompt-file inputs/prompts/16384_tokens.txt \
  --concurrency 1 \
  --duration_s 20 \
  --nonce_per_user
```

`MAX_INPUT_LEN` also controls H2A behavior in runner scripts.

---

# Advanced: building a custom Dynamo + KVBM worker image

This accelerates rebuilds and ensures workers always start with known-good wheel versions.

### Build wheels

Local Dockerfile: `container/Dockerfile`

Example:

```bash
cd ~/dynamo

BASE_IMAGE=nvcr.io/nvidia/cuda
BASE_IMAGE_TAG=12.6.3-devel-ubuntu22.04

./container/build.sh \
  --framework none \
  --enable-kvbm \
  --tag kvbm-wheel \
  --build-arg ARCH=arm64 \
  --build-arg ARCH_ALT=aarch64 \
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG \
  --build-arg NIXL_UCX_REF=1.20.0 \
  --platform linux/arm64
```

Extract wheels:

```bash
docker create --name kvbm-wheel kvbm-wheel
docker cp kvbm-wheel:/opt/dynamo/wheelhouse ./wheelhouse
docker rm kvbm-wheel
```

### Persist a worker image

```bash
docker run --gpus all --ipc host --network host --rm -d \
  --name spark-worker-setup \
  -v ~/dgx_spark_harness:/workspace/harness \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  spark-dynamo-worker sleep infinity

docker exec spark-worker-setup bash -lc "\
  pip install --no-cache-dir \
    /workspace/harness/wheelhouse/ai_dynamo_runtime-0.7.0-*.whl \
    /workspace/harness/wheelhouse/ai_dynamo-0.7.0-*.whl \
    /workspace/harness/wheelhouse/nixl/*.whl \
    /workspace/harness/wheelhouse/kvbm*.whl || true"

docker commit spark-worker-setup spark-dynamo-worker:0.7.0
docker rm -f spark-worker-setup
```

### Frontend using local wheels

```bash
source ~/dyn-venv/bin/activate

pip install ~/dgx_spark_harness/wheelhouse/ai_dynamo_runtime-0.7.0-*.whl
pip install ~/dgx_spark_harness/wheelhouse/ai_dynamo-0.7.0-py3-none-any.whl
```

---

# Notes and TODOs

* Stack B telemetry is partially stubbed; expand NVMe, GPU, Dynamo ingestion.
* Extend `sessioned_chat.py` for multi turn state reuse.
* Enhance `analysis/process_results.py` to correlate sysmon and dynkv.
* Add KV hit/miss and tier residency tracking once telemetry is integrated.
* Custom TRT engine instructions can be added in a separate doc if extended admits become required.
* Runner gap: Stack B tier0/1/2 sizing/path in `configs/stackB_*_dynamo_tiered.yaml` is not wired into the worker launch; add a shim to read those YAMLs and export tier capacities/paths for the worker.
If you ever see HF cache permission errors (common after running containerized servers that wrote root-owned files under `~/.cache/huggingface`), either chown the cache `sudo chown -R $USER:$USER ~/.cache/huggingface` or keep a harness-local cache by exporting:

```bash
mkdir -p .cache/hf
export HF_HOME=$PWD/.cache/hf
export HF_HUB_CACHE=$PWD/.cache/hf
```

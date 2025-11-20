# DGX Spark LLM Storage Harness (v3, Test_Plan_v3.0)

This repository implements the v3 harness from `Test_Plan_v3.0.md`: Stack A (UMA-only TRT-LLM) vs Stack B (Dynamo tiered KV), hypotheses H0–H8B, and storage-aware latency analysis. Legacy v2.5 artifacts have been removed from the tree (available in git history).

## Current State & Verified Functionality
- Stack configs: `configs/stackA_llama70b_baseline.yaml` (UMA control), `configs/stackB_llama70b_dynamo_tiered.yaml` (Dynamo tiered KV).
- Load generator: `src/loadgen.py` issues real HTTP requests (Stack A), supports `fixed_context` and `sessioned_chat`, optional nonce per user, writes `metrics.jsonl` with lat_ttft_ms/lat_e2e_ms.
- Workloads: `src/workloads/fixed_context.py`, `src/workloads/sessioned_chat.py`.
- Telemetry: `src/telemetry/sysmon.sh` emits `sysmon.jsonl`; other ingestion scripts are stubs for NVMe/GPU/Dynamo.
- Runners: `runs/run_H0_queue_knee.sh`, `run_H1_coldwarm_lora.sh`, `run_H2A_uma_pressure.sh`, `run_H2B_dynamo_kv_pressure.sh`, `run_H4A_storage_qos.sh`, `run_H4B_dynamo_storage_qos.sh`, `run_H5_kv_workingset_scaling.sh`, `run_H6_tier_sizing_policy.sh`, `run_H7_kv_telemetry_sweep.sh` plus `runs/_lib.sh`. Outputs land in `results/<run_id>/` with `config.yaml`, `metrics.jsonl`, `sysmon.jsonl`, and `dynkv.jsonl` (Stack B).
- Analysis: `analysis/process_results.py` scans `results/` (and `runs/` if present), computes p50/p95/p99 TTFT/E2E, and writes `analysis/figures/summary_v3.csv`. `analysis/backfill_summary.py` is a stub.
- Prompts: `inputs/prompts/` includes realistic incident-style text from 128 to 64k tokens (aliases for 1k/2k/4k/etc.).

## Spin-Up Guide (Stack A smoke test: H0)

1) Python env (host or container):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2) Terminal 1: start Stack A server (TRT-LLM OpenAI-compatible on 8355). Example:
   ```bash
   bash ./serve_llama33_70b_fp4.sh
   ```
   Adjust to your deployment; ensure the endpoint matches `configs/stackA_llama70b_baseline.yaml`.

3) Terminal 2: harness container (if using a dockerized harness workspace):
   ```bash
   docker run --gpus all -it --rm \
     --network host --ipc host \
     -v ~/dgx_spark_harness:/harness \
     -w /harness \
     nvcr.io/nvidia/pytorch:xx.xx-py3  # or your base image
   ```
   Inside container: `cd /harness` and activate `.venv` if mounted.

4) Run H0 sweep:
   ```bash
   ./runs/run_H0_queue_knee.sh http://127.0.0.1:8355/v1/completions
   ```
   Outputs: `results/<timestamp>_H0_stackA_L70B_U*/{config.yaml,metrics.jsonl,sysmon.jsonl}`.

5) Analyze:
   ```bash
   python3 analysis/process_results.py
   cat analysis/figures/summary_v3.csv
   ```

6) Archive (optional):
   ```bash
   ./scripts/archive_results.sh
   ```

Docker run example (tested):
```bash
docker run --gpus all -it --rm \
  --network host \
  --shm-size=1g \
  --privileged --pid=host \
  -v /my-models:/workspace \
  -v ~/dgx_spark_harness:/harness \
  -v /sys:/sys \
  -v /proc:/proc \
  -w /harness \
  -e HF_TOKEN="$HF_TOKEN" \
  spark-harness:v1 \
  bash
```

## Stack Overview
- **Stack A (UMA-only)**: TRT-LLM, KV sharing off, no Dynamo tiers. Endpoint default: 8355. Use `stackA_llama70b_baseline.yaml`.
- **Stack B (Dynamo tiered KV)**: TRT-LLM + Dynamo KV manager with tier0 (hbm), tier1 (uma), tier2 (nvme). Endpoint default: 9000. Use `stackB_llama70b_dynamo_tiered.yaml`. Telemetry stubs exist; QoS/tier controls to be implemented.
- **Stack B (8B testbed)**: Use `stackB_llama8b_dynamo_tiered.yaml` plus `serve_llama31_8b_fp4.sh` (or your own Dynamo front-end) for Llama 3.1 8B FP4 when 70B is too heavy.

## Custom Dynamo image with KVBM (arm64)

Upstream Dockerfile: https://github.com/ai-dynamo/dynamo/blob/main/container/Dockerfile  
Local customized Dockerfile: `container/Dockerfile` (arm64-friendly, KVBM/NIXL fixes, NIXL opt-in).

Key changes vs upstream:
- `NIXL_UCX_REF` bumped to v1.20.0; apt installs gcc-12 and sets CC/CXX in base.
- x86 builds UCX/NIXL/GDRCopy from source; arm skips UCX/GDRCopy and uses pip NIXL wheel.
- Manylinux wheel-builder venv uses `/opt/python/cp312-cp312`.
- Arm NIXL path copies any `libnixl*.so*` from the pip-installed wheel into `/opt/nvidia/nvda_nixl/lib64` and creates linker symlinks; headers copied from `src/api/cpp`.
- TRT-LLM NIXL is opt-in (`--use-nixl-connect` off by default) via patches in `components/src/dynamo/trtllm/main.py` and `utils/trtllm_utils.py`.

Build (arm64 example):
```bash
cd ~/dynamo
BASE_IMAGE=nvcr.io/nvidia/cuda
BASE_IMAGE_TAG=12.6.3-devel-ubuntu22.04
./container/build.sh --framework none --enable-kvbm --tag kvbm-wheel \
  --build-arg ARCH=arm64 --build-arg ARCH_ALT=aarch64 \
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

Install inside TRT-LLM worker (example):
```bash
pip install /workspace/harness/wheelhouse/ai_dynamo_runtime*.whl
pip install /workspace/harness/wheelhouse/ai_dynamo*any.whl
pip install /workspace/harness/wheelhouse/nixl/*.whl
pip install /workspace/harness/wheelhouse/kvbm*.whl   # if present
```

Run worker without NIXL connect (KVBM enabled, CUDA KV off):
```bash
python3 -m dynamo.trtllm \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --served-model-name llama3-8b \
  --max-num-tokens 512 \
  --max-batch-size 2 \
  --kv-block-size 32 \
  --extra-engine-args /workspace/kvbm_llm_api_config.yaml
# omit --use-nixl-connect to keep NIXL transport off
```

## Prompt length guardrail (16k engine)

When exercising large contexts, use loadgen’s tokenizer-aware truncation to stay under the engine’s admit limit:

```bash
python3 src/loadgen.py \
  --endpoint http://127.0.0.1:8356/v1/completions \
  --stack stackA --model nvidia/Llama-3.3-70B-Instruct-FP4 \
  --tokenizer nvidia/Llama-3.3-70B-Instruct-FP4 \
  --max_input_len 16000 --input_len_margin 64 \
  --workload fixed_context --context_tokens 16384 \
  --prompt-file inputs/prompts/16384_tokens.txt \
  --concurrency 1 --duration_s 20 --nonce_per_user
```

`--input_len_margin` subtracts a small safety cushion from `max_input_len` to account for BOS/metadata; adjust as needed. Runner `runs/run_H2A_uma_pressure.sh` will automatically inject `tokenizer/max_input_len/input_len_margin` when `MAX_INPUT_LEN` is exported. When `MAX_INPUT_LEN` is set, H2A enforces tokenizer-aware truncation; reported context lengths in plots should be derived from logged/tokenized counts rather than nominal prompt file size.

## Dynamo worker config (8B)

For the 8B Stack B worker, a sample KV/LLM API config lives at `configs/kvbm_llm_api_8b.yaml` (80% GPU memory for KV cache, padding-enabled cuda graphs). Mount it into the TRT-LLM container and pass via `--extra-engine-args /workspace/kvbm_llm_api_8b.yaml` when launching `dynamo.trtllm`.

## File Map
- `configs/`: stackA/stackB YAML.
- `src/loadgen.py`: load generator.
- `src/workloads/`: workload generators.
- `src/telemetry/`: sysmon + stubs.
- `runs/`: runner scripts + `_lib.sh`.
- `results/`: per-run outputs (config/metrics/telemetry).
- `analysis/`: `process_results.py`, `backfill_summary.py`, `figures/`.
- `inputs/prompts/`, `inputs/lora_lists/`.
- `scripts/archive_results.sh`: bundle results/runs/analysis figs into `archives/`.

## Notes / TODOs
- Stack B/Dynamo telemetry and QoS controls are stubbed; implement real ingestion and controls per Test_Plan_v3.0 Sections 6.5/8B.
- Sessioned chat workload is minimal; extend to handle multi-turn state.
- Analysis currently reports latency percentiles; add sysmon/dynkv aggregates as needed.

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

## Stack B (Dynamo) reproducible bring-up for Llama 3.1 8B
Known-good sequence to avoid malformed discovery and HF 404s:

1) Versions: ensure frontend and worker both use ai-dynamo/ai-dynamo-runtime 0.7.0 (`pip show ai-dynamo ai-dynamo-runtime` in both containers).
2) Infra: `docker compose -f ~/dynamo/deploy/docker-compose.yml up -d` (etcd + nats).
3) Seed discovery with the canonical card (one-time after a clean etcd):
   ```bash
   docker exec deploy-etcd-server-1 etcdctl del --prefix v1/mdc/dynamo/tensorrt_llm
   docker exec -i deploy-etcd-server-1 etcdctl put \
     v1/mdc/dynamo/tensorrt_llm/generate/694d9aa7979575ab \
     < ~/dgx_spark_harness/cards/Llama-3.1-8B-Instruct.json
   ```
   The card uses `hf_repo_id: meta-llama/Meta-Llama-3.1-8B-Instruct` and snapshot paths.
4) Frontend (host venv):
   ```bash
   export HF_TOKEN="$(<~/hftoken.txt)"
   export DYN_DISCOVERY_BACKEND=kv_store
   export DYN_STORE_KV=etcd
   export DYN_NAMESPACE=dynamo
   python3 -m dynamo.frontend --http-port 9000
   ```
5) Worker (container):
   ```bash
   python3 -m dynamo.trtllm \
     --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
     --served-model-name Meta-Llama-3.1-8B-Instruct \
     --max-num-tokens 512 --max-batch-size 2 --kv-block-size 32 \
     --extra-engine-args /workspace/kvbm_llm_api_config.yaml
   ```
   Mount HF cache (`-v $HOME/.cache/huggingface:/root/.cache/huggingface`) and set `HF_TOKEN` in the container.
6) Smoke test:
   ```bash
   curl -s http://127.0.0.1:9000/v1/models | jq .
   curl -s -X POST http://127.0.0.1:9000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
   ```
   If the model list is empty, verify the namespace (`DYN_NAMESPACE`) and etcd keys under `v1/mdc/dynamo/tensorrt_llm`.
7) Runner scripts: point Stack B endpoints at `http://127.0.0.1:9000` and use the model name `meta-llama/Meta-Llama-3.1-8B-Instruct` to match discovery.

### Persist a worker image pinned to ai-dynamo 0.7.0
Avoid re-installing wheels every run by baking a tagged image:
```bash
# Start a temp container, upgrade wheels from wheelhouse, and commit
docker run --gpus all --ipc host --network host --rm -d --name spark-worker-setup \
  -v ~/dgx_spark_harness:/workspace/harness \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  spark-dynamo-worker sleep infinity
docker exec spark-worker-setup bash -lc "pip install --no-cache-dir /workspace/harness/wheelhouse/ai_dynamo_runtime-0.7.0-*.whl /workspace/harness/wheelhouse/ai_dynamo-0.7.0-*.whl"
docker commit spark-worker-setup spark-dynamo-worker:0.7.0
docker rm -f spark-worker-setup
```
Use `spark-dynamo-worker:0.7.0` in runner/env variables so future runs always launch the correct Dynamo version.

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

# Install ai-dynamo CLI in the frontend venv (uses local wheels)
```bash
source ~/dyn-venv/bin/activate
pip install ~/dgx_spark_harness/wheelhouse/ai_dynamo_runtime-0.7.0-*.whl
pip install ~/dgx_spark_harness/wheelhouse/ai_dynamo-0.7.0-py3-none-any.whl
```

Start Dynamo frontend
```bash
python3 -m dynamo.frontend --http-port 9000
```

One-time infra (etcd + nats) using docker compose
-------------------------------------------------
Run on the host once per system so the discovery and request plane services are up:
```bash
cd ~/dynamo
docker compose -f deploy/docker-compose.yml up -d
```

Discovery card for Llama 3.1 8B (static file or etcd payload)
--------------------------------------------------------------
We standardize on the canonical HF repo `meta-llama/Llama-3.1-8B-Instruct`. A ready-to-use card is in `cards/Llama-3.1-8B-Instruct.json`; it includes the correct paths/checksums to the cached snapshot (`0e9e39f...`). Use one of the two flows:

- Static discovery (frontend):
  ```bash
  DYNAMO_DISCOVERY_BACKEND=static \
  DYNAMO_DISCOVERY_STATIC_DIR=~/dgx_spark_harness/cards \
  python3 -m dynamo.frontend --http-port 9000
  ```

- Etcd discovery (push from host):
  ```bash
  docker exec deploy-etcd-server-1 etcdctl del --prefix v1/mdc/dynamo/tensorrt_llm
  docker exec -i deploy-etcd-server-1 etcdctl put \
    v1/mdc/dynamo/tensorrt_llm/generate/7587890969240119941 \
    < cards/Llama-3.1-8B-Instruct.json
  ```
  (Run this only after a fresh/cleared etcd or if a bad record was written. In steady state you seed once unless you wipe or corrupt the discovery store. If the worker exports clean records you can skip the manual put.)

TRT-LLM worker container example (interactive):
```bash
docker run --gpus all --ipc host --network host --rm -it \
  -e HF_TOKEN="$(<~/hftoken.txt)" \
  -e DYN_KVBM_TIER2_PATH=/nvme/kvbm/l8b \
  -v /nvme/kvbm:/nvme/kvbm \
  -v ~/dgx_spark_harness/configs/kvbm_llm_api_8b.yaml:/workspace/kvbm_llm_api_config.yaml \
  -v ~/dgx_spark_harness:/workspace/harness \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  spark-dynamo-worker \
  bash
```

If you restart a clean worker image, re-apply the NIXL opt-in patch and stay in the shell:
```bash
docker run --gpus all --ipc host --network host --rm -it \
  -e HF_TOKEN="$(<~/hftoken.txt)" \
  -e DYN_KVBM_TIER2_PATH=/nvme/kvbm/l8b \
  -v /nvme/kvbm:/nvme/kvbm \
  -v ~/dgx_spark_harness/configs/kvbm_llm_api_8b.yaml:/workspace/kvbm_llm_api_config.yaml \
  -v ~/dgx_spark_harness:/workspace/harness \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  spark-dynamo-worker \
  bash -lc "cd /workspace/harness/scripts && ./patch_nixl_opt_in.sh && exec bash"
```

Run worker without NIXL connect (KVBM enabled, CUDA KV off):
```bash
python3 -m dynamo.trtllm \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --served-model-name Meta-Llama-3.1-8B-Instruct \
  --max-num-tokens 512 \
  --max-batch-size 2 \
  --kv-block-size 32 \
  --extra-engine-args /workspace/kvbm_llm_api_config.yaml
# omit --use-nixl-connect to keep NIXL transport off
```
If the worker writes malformed discovery records while we debug, start it with export disabled:
```bash
DYNAMO_DISCOVERY_KV_EXPORT_ENABLED=false \
python3 -m dynamo.trtllm ...
```

Example request (matches the slug in the discovery card):
```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3-1-8b-instruct","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
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

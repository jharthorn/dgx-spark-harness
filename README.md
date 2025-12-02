# DGX Spark LLM Storage Harness

Implements the DGX Spark LLM Storage Harness for **Test Plan v3.3** (docs/Test_Plan_v3.3.md). v3.0 is superseded. Supports:
- **Stack A** – UMA-only TRT-LLM (control)
- **Stack B** – Dynamo + tiered KV (treatment)

Designed to drive Comfy / Spill / Stress profiles across H0–H9 to measure queue knees (U_work), KV spill behavior, NVMe QoS sensitivity, context collapse points, and LoRA churn/resume behavior.

---

## Overview
- Harness = loadgen, runners, telemetry, analysis for Stack A/B side-by-side.
- Uses long prompts, sessioned chat, and tier-aware env helpers.
- Runners live in `runs/`; configs in `configs/`; telemetry + analysis in `analysis/`.

## Test Plan & Hypotheses
- Current plan: `docs/Test_Plan_v3.3.md` (v3.0 kept only for history).
- H0–H9 implemented via runner scripts listed below.

## Environment Setup

Python virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

HF token:
```bash
export HF_TOKEN="$(<~/hftoken.txt)"
```

Optional: run the harness inside a container:
```bash
docker run --gpus all -it --rm \
  --network host --ipc host \
  -v ~/dgx_spark_harness:/harness \
  -w /harness \
  -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nvidia/pytorch:xx.xx-py3
```

## Stacks & Profiles

### Stack A – UMA-only (TRT-LLM control)
- TRT-LLM on UMA; no tiered KV.
- Intended for H0, H2A, H4A, H8A and baselines.
- Config: `configs/stackA_llama70b_baseline.yaml` (endpoint must match your TRT server).

### Stack B – Dynamo tiered KV (treatment)
- Dynamo frontend + `dynamo.trtllm` worker with KVBM.
- KV tiers: Tier0/Tier1 in UMA, Tier2 on NVMe.
- Used for H0, H2B, H3, H4B, H5, H6, H7, H8, and LoRA/H9 paths when supported.

### Profiles (Comfy / Spill / Stress)
- **Comfy** – KV fits in UMA; minimal Tier2 usage.
- **Spill** – tiering active; realistic KV spill.
- **Stress** – heavy spill, high concurrency, often with background I/O.

Profile configs:
- `configs/stackB_llama70b_dynamo_tiered.yaml` → Comfy
- `configs/stackB_llama70b_dynamo_tiered_spill.yaml` → Spill
- `configs/stackB_llama70b_dynamo_tiered_stress.yaml` → Stress

`scripts/stackB_tier_env.py` now accepts `--profile` and is used by runners to set `DYN_KVBM_TIER*` envs for each profile.

## Launching Servers

### Stack A backend (TRT-LLM UMA)
```bash
bash ./serve_llama33_70b_fp4.sh
```
Listens on port **8355** by default. Ensure `configs/stackA_llama70b_baseline.yaml` matches the endpoint and model handle.

### Stack B frontend
Start discovery services (etcd + nats) once per host:
```bash
cd ~/dynamo
docker compose -f deploy/docker-compose.yml up -d   # starts etcd + nats
```
```bash
source ~/dyn-venv/bin/activate

export HF_TOKEN="$(<~/hftoken.txt)"
export DYN_DISCOVERY_BACKEND=kv_store
export DYN_STORE_KV=etcd
export DYN_NAMESPACE=dynamo

python3 -m dynamo.frontend --http-port 9000
```

### Stack B worker (profile-aware)
Bring-up that applies profile KV sizing and mounts the KV config:
```bash
PROFILE=spill  # comfy|spill|stress
CONFIG=configs/stackB_llama70b_dynamo_tiered_${PROFILE}.yaml
eval "$(python3 scripts/stackB_tier_env.py --profile "$PROFILE" --config "$CONFIG")"
TIER2_ROOT=$(dirname "$DYN_KVBM_TIER2_PATH")

docker run --gpus all --ipc host --network host --rm -it \
  -e HF_TOKEN="$(<~/hftoken.txt)" \
  -e DYN_KVBM_METRICS=true \
  -e DYN_KVBM_METRICS_PORT=${DYN_KVBM_METRICS_PORT:-6880} \
  -e DYN_KVBM_TIER0_BYTES="$DYN_KVBM_TIER0_BYTES" \
  -e DYN_KVBM_TIER1_BYTES="$DYN_KVBM_TIER1_BYTES" \
  -e DYN_KVBM_TIER2_BYTES="$DYN_KVBM_TIER2_BYTES" \
  -e DYN_KVBM_KV_BLOCK_SIZE_BYTES=${DYN_KVBM_KV_BLOCK_SIZE_BYTES:-65536} \
  -e DYN_KVBM_TIER2_PATH="$DYN_KVBM_TIER2_PATH" \
  -e DYN_DISCOVERY_KV_EXPORT_ENABLED=false \
  -p ${DYN_KVBM_METRICS_PORT:-6880}:${DYN_KVBM_METRICS_PORT:-6880} \
  -v "$TIER2_ROOT":"$TIER2_ROOT" \
  -v ~/dgx_spark_harness:/workspace/harness \
  -v ~/dgx_spark_harness/configs/kvbm_llm_api_70b.yaml:/workspace/kvbm_llm_api_config.yaml \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  spark-dynamo-worker:latest \
  bash -lc "cd /workspace/harness/scripts && ./patch_nixl_opt_in.sh && \
            python3 -m dynamo.trtllm \
              --model-path ${MODEL_HANDLE:-nvidia/Llama-3.3-70B-Instruct-NVFP4} \
              --served-model-name ${MODEL_HANDLE:-nvidia/Llama-3.3-70B-Instruct-NVFP4} \
              --kv-block-size ${KV_BLOCK_SIZE:-32} \
              --extra-engine-args /workspace/kvbm_llm_api_config.yaml"
```
Simplified 8B dev worker: set `PROFILE=comfy` and `CONFIG=configs/stackB_llama8b_dynamo_tiered.yaml`, then change `MODEL_HANDLE` to `nvidia/Llama-3.1-8B-Instruct-NVFP4` and mount `configs/kvbm_llm_api_8b.yaml`.

## Running Tests (H0–H9)

### Quick path (v3.3)
```bash
# After Stack A (8355) and Stack B (9000) are online:
./runs/run_H0_queue_knee.sh          # Stack A/B; computes U_work
./runs/run_H2B_dynamo_kv_pressure.sh # KV spill behavior (Spill/Stress)
./runs/run_H4B_dynamo_storage_qos.sh # QoS under storage noise
./runs/run_H8_hero_scenario.sh       # Hero Stack A vs Stack B

python3 analysis/process_results.py
```
Shows U_work queue knees, spill behavior, storage sensitivity, and Stack A vs B hero comparison.

### Hypothesis → runner map
- H0 — `run_H0_queue_knee.sh` (Stack A/B, Comfy) — finds U_work queue knee.
- H1 — `run_H1_lora_thrash.sh` (Stack B + LoRA proxy, Spill) — adapter churn + Tier2 traffic.
- H2A — `run_H2A_uma_pressure.sh` (Stack A, Comfy/Spill) — UMA pressure vs U_work.
- H2B — `run_H2B_dynamo_kv_pressure.sh` (Stack B, Spill/Stress) — KV spill pressure.
- H3 — `run_H3_context_envelope.sh` (Stack A/B, Stress) — context window collapse limits.
- H4A — `run_H4A_storage_qos.sh` (Stack A, Spill) — storage QoS without tiering.
- H4B — `run_H4B_dynamo_storage_qos.sh` (Stack B, Spill/Stress + fio) — NVMe QoS impact.
- H5 — `run_H5_kv_workingset_scaling.sh` (Stack B, Spill + LoRA proxy) — KV working set growth.
- H6 — `run_H6_tier_sizing_policy.sh` (Stack B, Spill) — Tier0/1 sizing sweeps.
- H7 — `run_H7_kv_telemetry_sweep.sh` (Stack B, Stress) — telemetry cadence + collapse.
- H8 — `run_H8_hero_scenario.sh` (Stack A vs B, mixed) — side-by-side hero comparison.
- H9 — `run_H9_rehydration.sh` (Stack B, Stress) — session resume; currently blocked (see status).

## LoRA Adapter Churn (H1, H5)
H1/H5 simulate high-cardinality LoRA usage by adding per-request `adapter_id` plus Tier2 I/O:
- Loadgen flags: `--lora_adapter_count`, `--lora_adapter_list`, `--lora_churn_mode`, `--lora_hot_ratio`, `--lora_hot_prob`.
- `adapter_id` is carried into metrics for churn analysis.

`scripts/adapter_proxy.py`:
- Listens on port 9100.
- Accepts OpenAI-style requests with `adapter_id`, strips it, forwards to the real backend (e.g., `http://127.0.0.1:9000/v1/completions`).
- For each adapter, performs a 1 MiB write + read under `$DYN_KVBM_TIER2_PATH/lora/<adapter_id>` to drive Tier2 traffic.

Usage:
```bash
# Terminal 1
BACKEND_URL=http://127.0.0.1:9000/v1/completions \
DYN_KVBM_TIER2_PATH=/nvme/kvbm/l70b \
python3 scripts/adapter_proxy.py

# Terminal 2
ENDPOINT=http://127.0.0.1:9100/v1/completions \
./runs/run_H1_lora_thrash.sh
```
Synthetic but storage-realistic: churn hits Tier2 even though the worker does not yet natively load LoRA weights.

## Telemetry & Analysis
- Per-run outputs under `results/<run_id>/`:
  - `metrics.jsonl` (request-level metrics incl. adapter_id, workload, session info)
  - `sysmon.jsonl` (CPU, memory, NVMe, GPU)
  - `dynkv.jsonl` + `dynkv_kv.csv` (KV movement, tier2_bytes_in_delta, etc.)
- Aggregate and derive U_work / knees / storage heuristics:
```bash
python3 analysis/process_results.py
```
Writes `analysis/figures/summary_v3.csv`, `analysis/figures/uwork.csv`, and per-hypothesis CSVs (H4B_p99_vs_nvme.csv, H2, H5, H8, H9 as implemented).

## Known Limitations / Status per Hypothesis
- H0/H2/H4/H5/H6/H7/H8: runnable under v3.3 harness; see Test Plan for nuances.
- H1: LoRA behavior simulated via adapter proxy; Tier2 churn is real but worker does not natively load LoRA adapters.
- H9: Blocked by current Dynamo OpenAI frontend; harness sends `session_id`/resume metadata but frontend 400s it, so full KV rehydration requires stack changes.

## Advanced Topics
- Custom worker images or wheel builds (see `container/` and `wheelhouse/` for local builds).
- HF cache permissions: if container runs leave root-owned files, fix with `sudo chown -R $USER:$USER ~/.cache/huggingface` or set a harness-local cache via:
  ```bash
  mkdir -p .cache/hf
  export HF_HOME=$PWD/.cache/hf
  export HF_HUB_CACHE=$PWD/.cache/hf
  ```
- Prompt length guardrails and TRT engine overrides remain in the runner/env defaults; adjust `STACKB_MAX_*` and `DYN_KVBM_*` before launching workers for custom admits.

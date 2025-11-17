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
  spark-harness:v1 \
  bash
```

## Stack Overview
- **Stack A (UMA-only)**: TRT-LLM, KV sharing off, no Dynamo tiers. Endpoint default: 8355. Use `stackA_llama70b_baseline.yaml`.
- **Stack B (Dynamo tiered KV)**: TRT-LLM + Dynamo KV manager with tier0 (hbm), tier1 (uma), tier2 (nvme). Endpoint default: 9000. Use `stackB_llama70b_dynamo_tiered.yaml`. Telemetry stubs exist; QoS/tier controls to be implemented.

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

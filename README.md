# DGX Spark LLM Storage Harness

This repository supports two harness tracks:

- **v3 (canonical)** — Implements Test_Plan_v3.0.md: Stack A (UMA-only) vs Stack B (Dynamo tiered KV), hypotheses H0–H8B, KV telemetry, and storage-aware latency analysis.
- **v2.5 (legacy)** — Keeps the prior UMA paging + LoRA-focused workflows for historical runs (see Test_Plan_v2.5.md and src/run scripts in `runs/`).

Use v3 for all new work; keep v2.5 runnable for comparison.

## Repo Layout (v3)

- `Test_Plan_v3.0.md` — Canonical plan (objectives, stacks, experiments, telemetry, analysis).
- `configs/` — Stack configs:
  - `stackA_llama70b_baseline.yaml` (UMA-only control)
  - `stackB_llama70b_dynamo_tiered.yaml` (Dynamo tiered KV)
- `src/`
  - `loadgen_v3.py` — Simulated async loadgen; reads config, emits `metrics.jsonl`.
  - `workloads/` — `fixed_context.py`, `sessioned_chat.py`.
  - `telemetry/` — `_v3` stubs for sysmon, NVMe, GPU, Dynamo ingest.
- `runs/v3/` — Runner stubs for H0–H7 (Stack A/B), plus `_lib_v3.sh` helper.
- `analysis/` — `process_results_v3.py`, `backfill_summary_v3.py`, `figures/`.
- `inputs/` — Shared prompts and LoRA lists (usable by both v2.5 and v3).
- `runs/v2.5` + `src/loadgen.py` + `analysis/process_results.py` — Legacy harness.

## Running v3 (high level)

1) Start the appropriate server:
   - Stack A: UMA-only TRT-LLM (no Dynamo).
   - Stack B: TRT-LLM + Dynamo KV tiering (tiers per `configs/stackB_llama70b_dynamo_tiered.yaml`).
2) Launch a runner from `runs/v3/` (e.g., `run_H2A_uma_pressure.sh` or `run_H2B_dynamo_kv_pressure.sh`). Each script writes a run directory under `runs/v3/YYYYMMDD_*` containing `config.yaml`, `metrics.jsonl`, and telemetry stubs (`sysmon.jsonl`, `dynkv.jsonl` for Stack B).
3) Analyze with `python3 analysis/process_results_v3.py` (writes summary CSVs to `analysis/figures/`).

### Archiving runs

Use `scripts/archive_results.sh [archive_name.tar.gz]` to bundle `results/`, `runs/v3`, `runs/v2.5`, and `analysis/figures/` into `archives/` for safekeeping or transfer.

## Legacy v2.5 Workflow

The previous UMA paging/LoRA harness remains available (scripts under `runs/`, loadgen at `src/loadgen.py`, analysis at `analysis/process_results.py`). See `Test_Plan_v2.5.md` and the README_harness.md scaffold for details. Treat this track as non-canonical going forward.

## Notes

- All v3 scripts are scaffolds aligned to Test_Plan_v3.0.md sections. TODO markers indicate where real server calls, telemetry parsing, and QoS controls must be implemented.
- Nonce tagging is available in `loadgen_v3.py` to keep per-user KV entries unique when required (H2A/H2B).

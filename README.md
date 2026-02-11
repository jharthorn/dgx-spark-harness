# DGX Spark Harness

This repository now centers on a reproducible benchmarking workflow for **DGX Spark + Dynamo + TRT-LLM + KVBM** focused on **SSD KV offload behavior visible to serving**.

Primary docs:

- `RUNBOOK.md`: canonical end-to-end setup and benchmark procedure.
- `CANONICAL_TEST_PLAN.md`: canonical project goals and whitepaper-grade test/evidence plan.
- `bench/README.md`: benchmark driver CLI, telemetry outputs, and interpretation notes.

## Quick Start (Current Workflow)

1. Prepare runtime prerequisites:

```bash
python3 -m venv ~/dynamo-venv
source ~/dynamo-venv/bin/activate
pip install -r requirements.txt
```

2. Start benchmark container:

```bash
scripts/bench_prepare_host.sh
scripts/bench_container_up.sh
```

Mode/model switches (no manual config edits required):

- `BENCH_TIER_MODE=B0|B1|B2` maps to `off|cpu_only|cpu_disk`.
- `BENCH_MODEL_PROFILE=llama31_8b_fp8|llama33_70b_nvfp4` selects baseline vs pressure-model defaults.

3. Start worker and frontend:

```bash
scripts/bench_start_worker.sh
scripts/bench_start_frontend.sh
scripts/bench_wait_ready.sh
```

For KV-router control-plane bring-up (Phase 2), start NATS first and set router toggles:

```bash
scripts/bench_start_nats.sh
scripts/bench_wait_nats_ready.sh
BENCH_ENABLE_LOCAL_INDEXER=true BENCH_PUBLISH_EVENTS_AND_METRICS=0 scripts/bench_start_worker.sh
BENCH_ROUTER_MODE=kv BENCH_KV_EVENTS=on scripts/bench_start_frontend.sh
scripts/bench_wait_ready.sh
```

4. Verify health and run a completions smoke test:

```bash
scripts/bench_health.sh
scripts/bench_smoke_completion.sh
```

5. Run benchmark smoke:

```bash
source ~/dynamo-venv/bin/activate
scripts/bench_run_smoke.sh
```

6. Run matrix (short/long/eviction replay):

```bash
source ~/dynamo-venv/bin/activate
scripts/bench_run_matrix.sh
```

7. Optional baseline vs offload mode compare:

```bash
source ~/dynamo-venv/bin/activate
BENCH_COMPARE_SKIP_READY=1 BENCH_KV_MODE_LIST="cpu_only cpu_disk" scripts/bench_run_mode_compare.sh
```

8. Summarize outputs:

```bash
scripts/bench_results_summary.sh
```

Artifacts are written under `bench/results/<run_id>/`.

## Validated Bring-Up Notes (2026-02-08)

- Worker/frontend startup can race model discovery. `bench.run_bench` now waits for `/v1/models` to return at least one model before failing.
- `scripts/bench_wait_ready.sh` supports model-based readiness by default (more reliable on this build) and optional strict endpoint gating.
- `scripts/bench_run_mode_compare.sh` supports:
  - `BENCH_COMPARE_SKIP_READY=1` to bypass readiness gating when the control plane is flaky.
  - `BENCH_COMPARE_MODEL_RESOLVE_TIMEOUT_S` to tolerate slow model registration.
- `off` mode can trigger discovery-store behavior that is unstable on this stack; for current Spark validation, use `cpu_only` vs `cpu_disk` for baseline/offload contrast.

## New Harness Components

- `bench/run_bench.py`: OpenAI-compatible `/v1/completions` benchmark runner with:
  - KVBM metrics snapshots/deltas
  - phase delta artifacts for KVBM metrics + OS I/O (`phase_deltas/`)
  - `reuse_verify` scenario for identical-request prefix-reuse checks
  - `local_copilot_burst` workload with deterministic `request_manifest.jsonl` (`prefix_hash`, `session_id`)
  - request identity hashes (prompt bytes + generation params)
  - `--tier-mode {B0,B1,B2}` + `--kv-mode {off,cpu_only,cpu_disk}`
  - NVMe identity + SMART pre/post capture (`nvme_identity.json`, `nvme_smart_pre.json`, `nvme_smart_post.json`)
  - prompt preflight guardrails
  - invalid-run labeling
  - auto `report.md` generation
- `bench/prompts.py`: deterministic short/long/mixed prompt generation with replay sets.
- `bench/openai_compat.py`: model discovery + completion client.
- `bench/telemetry.py`: orchestration for telemetry script collectors.
- `bench/scripts/`: `iostat`, `pidstat`, `nvidia-smi`, cache snapshots, docker/cufile logs.
- `scripts/bench_*.sh`: operator wrappers for container lifecycle, health checks, smoke runs, and matrix execution.
- `scripts/bench_start_nats.sh`, `scripts/bench_wait_nats_ready.sh`, `scripts/bench_stop_nats.sh`: NATS control-plane runbook helpers.
- `scripts/bench_run_mode_compare.sh`: mode-controlled baseline vs offload runs.
- `images/dyn/`: benchmark container Docker build context.
- `kvbm/kvbm_llm_api_config.yaml`: tracked KVBM template used by `scripts/bench_prepare_host.sh`.

## Notes About Engine Limits

[ASSUMPTION: many local engines are built with `max_num_tokens=8192`.]

If requests exceed that limit, the worker can unregister and frontend model discovery will drop to empty (`/v1/models` returns no models). For this case, keep long prompt generation under the engine cap (for example `--long-range 6000:7600`) unless you rebuild the TRT-LLM engine with higher context/token limits.

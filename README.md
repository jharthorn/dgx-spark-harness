# DGX Spark Harness

This repository now centers on a reproducible benchmarking workflow for **DGX Spark + Dynamo + TRT-LLM + KVBM** focused on **SSD KV offload behavior visible to serving**.

Primary docs:

- `RUNBOOK.md`: canonical end-to-end setup and benchmark procedure.
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
scripts/bench_container_up.sh
```

3. Start worker and frontend:

```bash
scripts/bench_start_worker.sh
scripts/bench_start_frontend.sh
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

7. Summarize outputs:

```bash
scripts/bench_results_summary.sh
```

Artifacts are written under `bench/results/<run_id>/`.

## New Harness Components

- `bench/run_bench.py`: OpenAI-compatible `/v1/completions` benchmark runner.
- `bench/prompts.py`: deterministic short/long/mixed prompt generation with replay sets.
- `bench/openai_compat.py`: model discovery + completion client.
- `bench/telemetry.py`: orchestration for telemetry script collectors.
- `bench/scripts/`: `iostat`, `pidstat`, `nvidia-smi`, cache snapshots, docker/cufile logs.
- `scripts/bench_*.sh`: operator wrappers for container lifecycle, health checks, smoke runs, and matrix execution.

## Notes About Engine Limits

[ASSUMPTION: many local engines are built with `max_num_tokens=8192`.]

If requests exceed that limit, the worker can unregister and frontend model discovery will drop to empty (`/v1/models` returns no models). For this case, keep long prompt generation under the engine cap (for example `--long-range 6000:7600`) unless you rebuild the TRT-LLM engine with higher context/token limits.

## Legacy v3.3 Harness Paths

The prior H0-H9 matrix and analysis paths are still present:

- `docs/Test_Plan_v3.3.md`
- `docs/v3.3_sweep_playbook.md`
- `runs/run_H*.sh`
- `src/loadgen.py` and `analysis/process_results.py`

Those are kept for historical comparison, but the recommended active workflow is `RUNBOOK.md` + `bench/`.

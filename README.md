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

9. Run a Phase58 progressive-thrash trial bundle (backend-agnostic wrapper, TRT-LLM default):

```bash
BENCH_BACKEND=trtllm \
BENCH_PHASE58_PATTERN=progressive_thrash \
BENCH_PHASE58_MAX_ATTEMPTS=1 \
scripts/bench_phase58_eviction_thrash.sh
```

10. Run the Phase60 fixed-pressure minimal sweep (replay concurrency sweep with baseline + resume support):

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
BENCH_PHASE60_TS=$TS \
BENCH_PHASE60_SWEEP_CONCURRENCIES="1 2 4" \
BENCH_PHASE60_PRESSURE_POPULATE_CONCURRENCY=2 \
BENCH_PHASE60_PRESSURE_THRASH_CONCURRENCY=2 \
BENCH_PHASE60_BASELINE_REPLAY_CONCURRENCY=1 \
BENCH_PHASE60_INCLUDE_B0=1 \
BENCH_PHASE60_FORCE_NEW_SUMMARY=true \
scripts/bench_phase60_rehydrate_minimal_sweep.sh
```

11. Run the Phase70 c=1 paired repeats helper (pair-local AB/BA counterbalanced design):

```bash
BENCH_PHASE70_PAIRS=6 \
BENCH_PHASE70_REPLAY_CONCURRENCY=1 \
BENCH_PHASE70_IO_ATTRIB=1 \
BENCH_PHASE70_STREAM_METRICS=1 \
BENCH_PHASE70_STREAM_TIMEOUT_S=120 \
BENCH_PHASE70_STREAM_RECORD_TTFB=1 \
scripts/bench_phase70_rehydrate_c1_pair_repeats.sh
```

12. Build a brief-ready results pack (single command + folder):

```bash
python3 scripts/make_phase70_results_pack.py \
  --results-root bench/results \
  --ts <ts>
```

This writes `bench/results/publish/phase70_pairs<N>_<ts>/` with summary artifacts, review tables, and methodology text.

Artifacts are written under `bench/results/<run_id>/`.

## Methodology Terms

- `B0`: KVBM disabled end-to-end (`off`).
- `B1`: KVBM CPU tier only (`cpu_only`).
- `B2`: KVBM CPU + SSD tier (`cpu_disk`).
- `TTFC`: streaming time to first non-empty output chunk (primary publishability latency metric).
- `TTFT`: legacy compatibility metric; retained in reports/summaries.
- `TTFB`: optional time-to-first-byte/header when streaming capture enables it.
- `A1`: storage/device metadata evidence capture (pre/post).
- `A2`: replay I/O attribution gate for disk-tier mechanism evidence.
- `A2 process evidence`: `pid` or `cgroup`; `cgroup` is expected in many containerized runs.
- `pair-local AB/BA`: blocked paired repeats with counterbalanced order (`B1_B2`, `B2_B1`).

## Naming Conventions

- Timestamp token: `<ts>=YYYYMMDDTHHMMSSZ`.
- Phase70 pair legs: `phase70_rehydrate_pair_<mode>_p<pair_id>_l<pair_leg>_<ts>`.
- Phase70 aggregate outputs:
  - `phase70_rehydrate_pair_repeats_manifest_<ts>.json`
  - `phase70_rehydrate_pair_repeats_summary_<ts>.csv`
  - `phase70_rehydrate_pair_repeats_deltas_<ts>.csv`
  - `phase70_rehydrate_pair_repeats_order_check_<ts>.json`
- Per-pair deltas are always `mode_b - mode_a` (default `B2 - B1`).

## Workload Mapping

- Phase56-like probe: mechanism/debug run (`scripts/bench_phase56_like_probe_trtllm.sh`).
- Phase60 sweep: baseline + concurrency sweep (`scripts/bench_phase60_rehydrate_minimal_sweep.sh`).
- Phase70 repeats: publishability gate run with AB/BA order checks (`scripts/bench_phase70_rehydrate_c1_pair_repeats.sh`).

## Results Hygiene

Keep `bench/results/` root high-signal and archive trial/legacy runs:

```bash
scripts/bench_results_hygiene.py --results-root bench/results
scripts/bench_results_hygiene.py --results-root bench/results --execute \
  --keep-modern-ts 20260217T161447Z --keep-modern-ts 20260213T003400Z
```

Notes:

- Dry-run is default; `--execute` applies moves.
- Trial-like artifacts are moved under `bench/results/trials/<archive_tag>/`.
- Non-modern/older artifacts are moved under `bench/results/archive/<archive_tag>/`.

## Validated Bring-Up Notes (2026-02-08)

- Worker/frontend startup can race model discovery. `bench.run_bench` now waits for `/v1/models` to return at least one model before failing.
- `scripts/bench_wait_ready.sh` supports model-based readiness by default (more reliable on this build) and optional strict endpoint gating.
- `scripts/bench_run_mode_compare.sh` supports:
  - `BENCH_COMPARE_SKIP_READY=1` to bypass readiness gating when the control plane is flaky.
  - `BENCH_COMPARE_MODEL_RESOLVE_TIMEOUT_S` to tolerate slow model registration.
- `B0`/`off` now disables KVBM integration in worker/frontend startup paths (no KVBM config injection, no KVBM store-kv args).

## New Harness Components

- `bench/run_bench.py`: OpenAI-compatible `/v1/completions` benchmark runner with:
  - KVBM metrics snapshots/deltas
  - phase delta artifacts for KVBM metrics + OS I/O (`phase_deltas/`)
  - `reuse_verify` scenario for identical-request prefix-reuse checks
  - `rehydrate_replay` scenario for phase-structured populate/thrash/replay validation
  - `local_copilot_burst` workload with deterministic `request_manifest.jsonl` (`prefix_hash`, `session_id`)
  - request identity hashes (prompt bytes + generation params)
  - `--tier-mode {B0,B1,B2}` + `--kv-mode {off,cpu_only,cpu_disk}`
  - NVMe identity + SMART pre/post capture (`nvme_identity.json`, `nvme_smart_pre.json`, `nvme_smart_post.json`)
  - publish-grade storage metadata pre/post capture (`device_metadata_pre.json`, `device_metadata_post.json`)
  - optional replay-phase I/O attribution bundle via `--io-attrib` (`io/iostat.csv`, `io/pidstat.csv`, `io/io_attribution_report.json`)
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
- `scripts/bench_phase60_rehydrate_minimal_sweep.sh`: fixed-pressure replay-concurrency sweep with baseline/SLO persistence and resume checkpoints.
- `scripts/bench_phase70_rehydrate_c1_pair_repeats.sh`: pair-local blocked repeat runner for c=1 rehydrate replay with AB/BA counterbalancing and order-effect output.
- `scripts/analyze_phase70_pairs.py`: manifest-driven analyzer for per-run CSV, per-pair deltas, and order-effect summaries.
- `scripts/make_phase70_results_pack.py`: one-command packager for brief-ready Phase70 folders (`publish/phase70_pairs<N>_<ts>/`).
- `images/dyn/`: benchmark container Docker build context.
- `kvbm/kvbm_llm_api_config.yaml`: tracked KVBM template used by `scripts/bench_prepare_host.sh`.

## Notes About Engine Limits

[ASSUMPTION: many local engines are built with `max_num_tokens=8192`.]

If requests exceed that limit, the worker can unregister and frontend model discovery will drop to empty (`/v1/models` returns no models). For this case, keep long prompt generation under the engine cap (for example `--long-range 6000:7600`) unless you rebuild the TRT-LLM engine with higher context/token limits.

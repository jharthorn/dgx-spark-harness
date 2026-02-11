# Bench Harness (`bench/`)

This package adds a focused benchmark harness for DGX Spark Dynamo + TRT-LLM + KVBM SSD KV offload testing using **`/v1/completions`**.

## Files

- `bench/run_bench.py`: benchmark CLI driver (`standard`, `eviction_replay`, `reuse_verify`, `local_copilot_burst`, `rehydrate_replay`).
  - Includes KVBM metrics snapshots/deltas by phase.
  - Can capture raw Prometheus snapshots from both Dynamo system metrics and KVBM metrics endpoints.
  - Includes per-request identity hashes (prompt bytes + generation params).
  - Includes canonical mode mapping `--tier-mode {B0,B1,B2}` plus `--kv-mode {off,cpu_only,cpu_disk}` metadata.
  - Emits per-phase delta artifacts for KVBM metrics and OS I/O (block device + container/worker IO deltas).
  - Captures NVMe identity + SMART pre/post snapshots into every run bundle.
  - Enforces prompt preflight guardrails against engine token limits.
  - Marks invalid runs explicitly and emits `report.md`.
- `bench/prompts.py`: deterministic short/long/mixed and replay prompt generation.
- `bench/openai_compat.py`: async OpenAI-compatible client (`/v1/models`, `/v1/completions`).
- `bench/telemetry.py`: Python wrappers around shell collectors.
- `bench/scripts/*.sh`: iostat/pidstat/GPU/cufile/docker/cache-dir collection scripts.

## Benchmark Driver Usage

[ASSUMPTION: run from repo root with Python env that has `httpx`; optional `transformers` is only used if locally available.]
[ASSUMPTION: many local TRT-LLM builds on this stack enforce `max_num_tokens=8192`; use `--long-range` below that cap.]

### 1) Short-context run

```bash
python3 -m bench.run_bench \
  --base-url http://127.0.0.1:8000 \
  --kv-mode cpu_disk \
  --scenario standard \
  --prompt-set short \
  --requests 64 \
  --warmup 8 \
  --concurrency 4 \
  --max-tokens 256 \
  --collect-telemetry \
  --kvbm-cache-dir /mnt/nvme/kvbm \
  --container-name dyn
```

### 2) Long-context run

```bash
python3 -m bench.run_bench \
  --base-url http://127.0.0.1:8000 \
  --kv-mode cpu_disk \
  --scenario standard \
  --prompt-set long \
  --long-range 6000:7600 \
  --requests 32 \
  --warmup 4 \
  --concurrency 4 \
  --max-tokens 256 \
  --collect-telemetry \
  --kvbm-cache-dir /mnt/nvme/kvbm \
  --container-name dyn
```

### 3) Eviction + replay scenario

```bash
python3 -m bench.run_bench \
  --base-url http://127.0.0.1:8000 \
  --kv-mode cpu_disk \
  --scenario eviction_replay \
  --warmup 4 \
  --eviction-a-requests 24 \
  --eviction-b-requests 48 \
  --eviction-a-concurrency 4 \
  --eviction-b-concurrency 8 \
  --long-range 6000:7600 \
  --max-tokens 256 \
  --collect-telemetry \
  --kvbm-cache-dir /mnt/nvme/kvbm \
  --container-name dyn
```

### 3a) Dual metrics snapshots (system + kvbm)

```bash
python3 -m bench.run_bench \
  --base-url http://127.0.0.1:8000 \
  --kv-mode cpu_disk \
  --scenario eviction_replay \
  --eviction-a-requests 8 \
  --eviction-b-requests 16 \
  --eviction-a-concurrency 2 \
  --eviction-b-concurrency 4 \
  --long-range 6000:7600 \
  --allow-missing-kvbm-metrics \
  --capture-metrics-snapshot \
  --metrics-system-url "http://127.0.0.1:${DYN_SYSTEM_PORT:-8081}/metrics" \
  --metrics-kvbm-url "http://127.0.0.1:${DYN_KVBM_METRICS_PORT:-6880}/metrics"
```

This writes:

- `metrics_system_pressure.prom`, `metrics_system_replay.prom`
- `metrics_kvbm_pressure.prom`, `metrics_kvbm_replay.prom`
- `kvbm_metric_inventory.txt`, `kvbm_metric_inventory_expanded.txt`, `kvbm_metric_inventory_from_6880.txt`

### 4) Reuse verification scenario (identical request replay)

```bash
python3 -m bench.run_bench \
  --base-url http://127.0.0.1:8000 \
  --kv-mode cpu_disk \
  --scenario reuse_verify \
  --reuse-prompt-set short \
  --reuse-repeat-count 3 \
  --max-tokens 64 \
  --temperature 0 \
  --top-p 1 \
  --request-seed 1337 \
  --stop "<|eot_id|>"
```

Inspect:

- `.overall_summary.reuse_verify_signal_kvbm`
- `.request_identity.reuse_verify_identity`

### 5) TTFT collection modes (best effort)

```bash
python3 -m bench.run_bench \
  --base-url http://127.0.0.1:8000 \
  --scenario standard \
  --prompt-set short \
  --requests 32 \
  --concurrency 2 \
  --stream
```

Without `--stream`, the client still records a first-byte TTFT proxy for non-stream completions. With `--stream`, TTFT uses first chunk arrival and is typically a higher-fidelity proxy.

### 6) Phase60 fixed-pressure minimal sweep wrapper

This is the canonical decision-grade B1 vs B2 sweep flow. Populate/thrash pressure stays fixed while replay concurrency is swept.

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
BENCH_PHASE60_TS=$TS \
BENCH_PHASE60_SWEEP_CONCURRENCIES="1 2 4" \
BENCH_PHASE60_PRESSURE_POPULATE_CONCURRENCY=2 \
BENCH_PHASE60_PRESSURE_THRASH_CONCURRENCY=2 \
BENCH_PHASE60_BASELINE_REPLAY_CONCURRENCY=1 \
BENCH_PHASE60_FORCE_NEW_SUMMARY=true \
scripts/bench_phase60_rehydrate_minimal_sweep.sh
```

## Operator Scripts

For the scripted workflow around this harness, use:

- `scripts/bench_prepare_host.sh`
- `scripts/bench_container_up.sh`
- `scripts/bench_start_worker.sh`
- `scripts/bench_start_frontend.sh`
- `scripts/bench_start_nats.sh`
- `scripts/bench_wait_nats_ready.sh`
- `scripts/bench_stop_nats.sh`
- `scripts/bench_health.sh`
- `scripts/bench_smoke_completion.sh`
- `scripts/bench_run_smoke.sh`
- `scripts/bench_run_matrix.sh`
- `scripts/bench_results_summary.sh`
- `scripts/bench_run_mode_compare.sh`
- `scripts/bench_phase56_like_probe_trtllm.sh`
- `scripts/bench_phase58_eviction_thrash.sh`
- `scripts/bench_phase60_rehydrate_minimal_sweep.sh`
- `scripts/bench_workload_local_copilot_burst.sh`
  - For current Spark stability: `BENCH_COMPARE_SKIP_READY=1 BENCH_KV_MODE_LIST="cpu_only cpu_disk" scripts/bench_run_mode_compare.sh`

## Concurrency Sweep Example

```bash
for c in 1 2 4 8 16; do
  python3 -m bench.run_bench \
    --base-url http://127.0.0.1:8000 \
    --scenario standard \
    --prompt-set short \
    --requests 64 \
    --warmup 8 \
    --concurrency "$c" \
    --max-tokens 256 \
    --collect-telemetry \
    --container-name dyn \
    --kvbm-cache-dir /mnt/nvme/kvbm \
    --run-id "short_c${c}_$(date -u +%Y%m%dT%H%M%SZ)"
done
```

## Output Artifacts

Each run writes to `bench/results/<run_id>/`:

- `config.json`: run parameters, phase plan, tokenizer mode.
- `manifest.json`: canonical run manifest with tier mode + KVBM env snapshot.
- `prompts_manifest.jsonl`: prompt IDs + token targets/estimates + hashes.
- `request_manifest.jsonl`: deterministic per-request schedule including `prefix_hash`/`session_id` where available.
- `requests.jsonl`: per-request raw records:
  `start_ts`, `end_ts`, `latency_ms`, `status_code`, `prompt_id`, `prompt_tokens_est`, `output_len_chars`, `error`, `request_id`, `ttft_ms`.
- `kvbm_metrics_snapshots.jsonl`: raw `/metrics` snapshots at phase boundaries.
- `summary.json`: includes run validity, per-phase + overall metrics, KVBM deltas, eviction replay signals.
- `phase_deltas/`: `phase_<name>_kvbm_metrics_{start,end,delta}.json` and `phase_<name>_os_io_{start,end,delta}.json`.
- `nvme_identity.json`, `nvme_smart_pre.json`, `nvme_smart_post.json`: NVMe identity + health snapshots.
- `report.md`: human-readable run report (brief-ready).
- `telemetry/`: iostat, pidstat, nvidia-smi, KVBM cache snapshots, docker logs, cuFile logs.
- `responses/` (optional with `--store-responses`): raw text outputs.

## Standalone Telemetry Script Usage

```bash
RUN_DIR=bench/results/manual_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_DIR"
bench/scripts/start_iostat.sh "$RUN_DIR" nvme0n1 1
bench/scripts/start_pidstat.sh "$RUN_DIR" ALL 1
bench/scripts/start_gpu_dmon.sh "$RUN_DIR" 1
bench/scripts/snapshot_kvbm_dir.sh "$RUN_DIR" /mnt/nvme/kvbm before_manual
# ...run workload...
bench/scripts/snapshot_kvbm_dir.sh "$RUN_DIR" /mnt/nvme/kvbm after_manual
bench/scripts/stop_iostat.sh "$RUN_DIR"
bench/scripts/stop_pidstat.sh "$RUN_DIR"
bench/scripts/stop_gpu_dmon.sh "$RUN_DIR"
bench/scripts/collect_docker_logs.sh "$RUN_DIR" dyn
bench/scripts/collect_nats_logs.sh "$RUN_DIR" bench-nats
bench/scripts/collect_cufile_logs.sh "$RUN_DIR" dyn
```

## Findings (Placeholder)

What to inspect for KV offload evidence:

- `summary.json` phase `io_delta` fields (`read_mib_delta`, `write_mib_delta`).
- `telemetry/iostat_*.log` for read/write MB/s, IOPS, queue depth (`aqu-sz`), and await.
- `telemetry/kvbm_*_files.txt` snapshots for file count/size/mtime churn in cache directory.
- `telemetry/cufile_*` and `telemetry/cufile_logs/` for compat/POSIX path confirmation.

Signals of pressure/eviction:

- rising NVMe writes during long-context or higher concurrency pressure phases,
- larger cache-dir churn during pressure than warmup,
- replay phase reads above warm phase reads (possible rehydrate signal).

Known limitations:

- TTFT is a proxy metric (non-stream uses first-byte timing, stream uses first chunk timing).
- If replay reads do not appear and `kvbm_matched_tokens_delta` is zero, disk rehydrate is gated because no cross-request reuse path activated.

## Interactive Validation Notes (2026-02-08)

- Mode-compare runs are stable with `cpu_only` and `cpu_disk`.
- `off` mode can be runtime-fragile in this build and is not required for Phase 1 evidence.
- If startup/model registration races occur, use:
  - `--model-resolve-timeout-s` / `--model-resolve-poll-s` in `bench.run_bench`,
  - `BENCH_COMPARE_MODEL_RESOLVE_TIMEOUT_S` in `scripts/bench_run_mode_compare.sh`.

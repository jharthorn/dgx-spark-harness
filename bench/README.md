# Bench Harness (`bench/`)

This package adds a focused benchmark harness for DGX Spark Dynamo + TRT-LLM + KVBM SSD KV offload testing using **`/v1/completions`**.

## Files

- `bench/run_bench.py`: benchmark CLI driver (standard + `eviction_replay` scenario).
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

### 4) Streaming TTFT proxy (best effort)

```bash
python3 -m bench.run_bench \
  --base-url http://127.0.0.1:8000 \
  --scenario standard \
  --prompt-set short \
  --requests 32 \
  --concurrency 2 \
  --stream
```

If the server does not stream chunks, `ttft_ms` stays empty and you should rely on end-to-end latency and throughput.

## Operator Scripts

For the scripted workflow around this harness, use:

- `scripts/bench_prepare_host.sh`
- `scripts/bench_container_up.sh`
- `scripts/bench_start_worker.sh`
- `scripts/bench_start_frontend.sh`
- `scripts/bench_health.sh`
- `scripts/bench_smoke_completion.sh`
- `scripts/bench_run_smoke.sh`
- `scripts/bench_run_matrix.sh`
- `scripts/bench_results_summary.sh`

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
- `prompts_manifest.jsonl`: prompt IDs + token targets/estimates + hashes.
- `requests.jsonl`: per-request raw records:
  `start_ts`, `end_ts`, `latency_ms`, `status_code`, `prompt_id`, `prompt_tokens_est`, `output_len_chars`, `error`, `request_id`, `ttft_ms` (if stream).
- `summary.json`: per-phase + overall aggregate metrics and eviction replay signal notes.
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

- TTFT is only a proxy unless streaming chunks are actually emitted.
- If replay reads do not appear, reuse may be served from in-memory tiers, eviction may be insufficient, or disk rehydrate may not be active in this mode.

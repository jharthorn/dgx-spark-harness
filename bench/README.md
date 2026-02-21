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
  - Optional `--io-attrib` mode writes replay-phase attribution artifacts under `io/`.
  - Enforces prompt preflight guardrails against engine token limits.
  - Marks invalid runs explicitly and emits `report.md`.
- `bench/prompts.py`: deterministic short/long/mixed and replay prompt generation.
- `bench/openai_compat.py`: async OpenAI-compatible client (`/v1/models`, `/v1/completions`).
- `bench/telemetry.py`: Python wrappers around shell collectors.
- `bench/scripts/*.sh`: iostat/pidstat/GPU/cufile/docker/cache-dir collection scripts.

## Methodology Glossary

- `B0`: KVBM fully disabled (`kv_mode=off`), no KVBM integration in serving stack.
- `B1`: KVBM enabled with CPU tier only (`kv_mode=cpu_only`).
- `B2`: KVBM enabled with CPU + SSD tier (`kv_mode=cpu_disk`).
- `TTFT`: legacy time-to-first-token proxy retained for compatibility.
- `TTFC`: streaming time to first non-empty streamed chunk/event (primary user-latency metric).
- `TTFB`: optional time to first response byte/header.
- `A1`: storage/device metadata capture (`device_metadata_pre.json`, `device_metadata_post.json`).
- `A2`: replay I/O attribution gate (`io/io_attribution_report.json` + `io/io_attrib_verdict.json`).
- `pair-local blocked design`: each pair is a local block where both modes run adjacently.
- `AB/BA counterbalancing`: alternating pair order (`B1_B2`, `B2_B1`) to mitigate order effects.
- `process_evidence_method`: process attribution source used by A2 verdict (`pid`, `cgroup`, `none`).
  In containerized setups, `cgroup` is expected and acceptable when per-PID `/proc/<pid>/io` deltas are zero.

## Naming Conventions

- UTC timestamp token: `<ts>=YYYYMMDDTHHMMSSZ`.
- Phase60 sweep run directories:
  - `phase60_rehydrate_minimal_sweep_<mode>_c<replay_concurrency>_<ts>`
  - `phase60_rehydrate_minimal_preflight_B2_c1_<ts>`
- Phase60 baseline semantic-hash policy artifacts:
  - `phase60_known_good_baseline_manifest_semantic_hash.json`
  - `phase60_baseline_manifest_audit.jsonl`
- Phase70 pair leg run directories:
  - `phase70_rehydrate_pair_<mode>_p<pair_id>_l<pair_leg>_<ts>`
  - Example: `phase70_rehydrate_pair_B2_p04_l1_<ts>`
- Phase70 aggregate artifacts:
  - `phase70_rehydrate_pair_repeats_manifest_<ts>.json`
  - `phase70_rehydrate_pair_repeats_summary_<ts>.csv`
  - `phase70_rehydrate_pair_repeats_deltas_<ts>.csv`
  - `phase70_rehydrate_pair_repeats_order_check_<ts>.json`
  - `phase70_rehydrate_pair_repeats_verdict_<ts>.json`
- Delta fields are always `mode_b - mode_a` and encoded as `delta_*`.

## Workload Mapping

- `scripts/bench_phase56_like_probe_trtllm.sh`:
  mechanism/debug probe, not publishability-grade evidence.
- `scripts/bench_phase60_rehydrate_minimal_sweep.sh`:
  baseline/sweep methodology run (B0/B1/B2, replay concurrency sweep).
- `scripts/bench_phase70_rehydrate_pair_repeats.sh`:
  publishability repeatability run (pair-local AB/BA, order-check output).

## Results Hygiene

- Keep `bench/results/` root high-signal for current methodology (modern Phase60/Phase70 artifacts).
- Archive older or trial artifacts with:

```bash
scripts/bench_results_hygiene.py --results-root bench/results
scripts/bench_results_hygiene.py --results-root bench/results --execute
```

- Defaults:
  - dry-run first,
  - keeps only latest modern timestamp group (`--keep-latest-ts=1`),
  - supports pinned keep timestamps via repeated `--keep-modern-ts <ts>`.
- Moved artifacts are separated into:
  - `bench/results/archive/<archive_tag>/` (legacy/non-modern),
  - `bench/results/trials/<archive_tag>/` (trial/debug/smoke-like artifacts).

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

### 3b) I/O Attribution Mode (Optional)

```bash
python3 -m bench.run_bench \
  --base-url http://127.0.0.1:8000 \
  --scenario rehydrate_replay \
  --kv-mode cpu_disk \
  --io-attrib \
  --io-attrib-interval-s 1 \
  --kvbm-cache-dir /mnt/nvme/kvbm \
  --container-name dyn
```

Optional dependencies:

- `sysstat` for host tools like `iostat` and `pidstat` (collector degrades to `/proc`-based sampling when unavailable).
- `lsof` for file-level open-file attribution fallback under the KVBM disk path.
- `bpftrace` for privileged syscall tracing (collector probes availability and degrades gracefully without it).

Install example (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y sysstat lsof bpftrace
```

### 3c) Replay Attribution Gate Checker (A2)

Validate replay attribution evidence for a completed run bundle:

```bash
python3 scripts/check_io_attrib_replay.py \
  --run-dir bench/results/<bundle>/<run_id> \
  --expect-report
```

Behavior:

- Writes `io/io_attrib_verdict.json` inside the run bundle.
- Prints one-line `PASS`/`FAIL` summary.
- Exit code `0` for pass (including warn-only checks), `2` for hard fail.
- Strict replay checks apply only to B2 + disk-tier-enabled runs.
- In strict B2 runs, replay process evidence accepts either `process_io_by_phase.<replay>.read_bytes` (PID aggregate) or `process_io_by_phase.<replay>.cgroup_read_bytes` (cgroup fallback).
- Verdicts include `process_evidence_method` (`pid`, `cgroup`, `none`) plus per-PID availability/nonzero flags so cgroup-backed passes are explicit.
- Reviewer note: in containerized runs, `process_evidence_method=cgroup` is often expected and is treated as valid process attribution.

`scripts/bench_phase60_rehydrate_minimal_sweep.sh` automatically runs this checker when `BENCH_PHASE60_IO_ATTRIB=1`, and records verdict details in each sweep row under `io_attribution_verdict`.

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

### 5) Streaming TTFC / optional TTFB collection

```bash
python3 -m bench.run_bench \
  --base-url http://127.0.0.1:8000 \
  --scenario standard \
  --prompt-set short \
  --requests 32 \
  --concurrency 2 \
  --stream \
  --stream-timeout-s 120 \
  --stream-record-ttfb
```

Behavior:

- `--stream` (or alias `--stream-metrics`) enables streamed requests.
- `ttfc_ms` is measured from request start to first non-empty streamed SSE `data:` payload (fallback: first non-empty stream chunk).
- `ttfb_ms` is optional and emitted only when `--stream-record-ttfb` is set.
- `ttft_ms` is retained for backward compatibility; for stream runs it mirrors TTFC.
- Without `--stream`, TTFT remains a first-byte proxy and TTFC is absent.

### 6) Phase60 fixed-pressure minimal sweep wrapper

This is the canonical decision-grade B1 vs B2 sweep flow. Populate/thrash pressure stays fixed while replay concurrency is swept.
Canonical Phase 1 loop is: Phase60 sweep -> knee table -> crossover recommendation -> Phase70 paired repeats -> pack generation.

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

When `BENCH_PHASE60_INCLUDE_B0=1`, each replay-concurrency point runs in a consistent order `B2 -> B1 -> B0` and emits a combined JSON + CSV summary (KVBM counters are blank in CSV for B0 rows).

Phase60 primary outputs:

- `bench/results/phase60_rehydrate_concurrency_sweep_summary_minimal_<ts>.json`
- `bench/results/phase60_rehydrate_concurrency_sweep_summary_minimal_<ts>.csv`
- stop/fail-only: `bench/results/phase60_matrix_stop_verdict_minimal_<ts>.json`
- stop/fail diagnostics: `bench/results/phase60_sweep_b2c1_failure_diagnosis_<ts>.json`

Baseline semantic-hash guardrail knobs:

- `BENCH_PHASE60_STRICT_BASELINE_HASH=0|1` (default `0`): strict mode stops on `B2@c1` semantic-hash mismatch; discovery mode records warning `BASELINE_MANIFEST_HASH_MISMATCH_WARNING` and continues.
- `BENCH_PHASE60_ACCEPT_NEW_BASELINE_MANIFEST=1`: on mismatch, accepts current semantic baseline hash and appends an audit entry.
- Baseline semantic hash file (default): `bench/results/phase60_known_good_baseline_manifest_semantic_hash.json`
- Audit trail (default): `bench/results/phase60_baseline_manifest_audit.jsonl`
- `BENCH_PHASE60_ENFORCE_B1_DISK_TIER_OFF=1` (default): forces `B1` to run with disk tier disabled and validates replay-phase B1 mechanism sanity.
- `BENCH_PHASE60_B1_DISK_TIER_READ_BYTES_THRESHOLD` / `BENCH_PHASE60_B1_DISK_TIER_HIT_RATE_THRESHOLD`: tune B1 replay read/hit-rate sanity thresholds.
- Phase60 rows expose `b1_disk_tier_verified` and `metric_used`; summary-level `meta.metric_policy` records run-wide metric usage (`ttfc_ms` preferred, `ttft_ms` fallback).

### 6a) Phase60 knee table extraction

```bash
python3 scripts/phase60_extract_knee_table.py \
  --phase60-summary-json bench/results/phase60_rehydrate_concurrency_sweep_summary_minimal_<ts>.json
```

Primary output:

- `bench/results/phase60_knee_table_<ts>.csv`

### 6b) Phase60 crossover recommender (pick Phase70 replay concurrency)

Use the Phase60 summary to select the most publishable Phase70 replay concurrency.

```bash
python3 scripts/recommend_phase70_crossover_from_phase60.py \
  --ts "$TS" \
  --results-root bench/results \
  --out-md "bench/results/phase60_crossover_recommendation_${TS}.md"
```

Alternative direct-input mode:

```bash
python3 scripts/recommend_phase70_crossover_from_phase60.py \
  --phase60-summary-json bench/results/phase60_rehydrate_concurrency_sweep_summary_minimal_<ts>.json
```

Primary output:

- `bench/results/phase60_crossover_recommendation_<ts>.json`
- optional: `bench/results/phase60_crossover_recommendation_<ts>.md`

### 6c) Phase70 paired repeats (pair-local AB/BA counterbalanced, replay concurrency `c>=1`)

```bash
BENCH_PHASE70_PAIRS=6 \
BENCH_PHASE70_REPLAY_CONCURRENCY=4 \
BENCH_PHASE70_IO_ATTRIB=1 \
BENCH_PHASE70_STREAM_METRICS=1 \
BENCH_PHASE70_STREAM_TIMEOUT_S=120 \
BENCH_PHASE70_STREAM_RECORD_TTFB=1 \
scripts/bench_phase70_rehydrate_pair_repeats.sh
```

Behavior:

- Pair-local blocked design: each pair runs back-to-back in one block (`pair_id`).
- `BENCH_PHASE70_REPLAY_CONCURRENCY` controls replay concurrency for both legs (`BENCH_PHASE56_REHYDRATE_REPLAY_CONC` pass-through).
- Preflight fast-fail checks run before pair legs (frontend health, model registration, KVBM metrics reachability).
- Early mechanism gate runs after the first `B2` leg; if SSD-tier mechanism signals are absent, the run aborts with `GATE_NO_SSD_MECHANISM_SIGNAL`.
- A post-run verdict is written to `bench/results/phase70_rehydrate_pair_repeats_verdict_<ts>.json`.
- Verdict mechanism checks are split into `ssd_write_signal_present`, `ssd_rehydrate_signal_present`, and `ssd_reuse_signal_present`.
- For `rehydrate_replay`, decision-grade defaults to requiring rehydrate evidence. Write-only evidence records `REHYDRATE_SIGNAL_ABSENT_WRITE_ONLY`.
- Preflight/gate/verdict reason codes are explicit (`PREFLIGHT_*`, `GATE_NO_SSD_MECHANISM_SIGNAL`, `RUN_ERRORS_PRESENT`, `ORDER_EFFECT_*`, `REHYDRATE_SIGNAL_ABSENT_WRITE_ONLY`).
- Counterbalanced order: default `BENCH_PHASE70_ORDER_STRATEGY=alternating` gives AB/BA by pair.
- Optional randomized balanced order: `BENCH_PHASE70_ORDER_STRATEGY=random` with `BENCH_PHASE70_ORDER_SEED=<seed>`.
- Delta definition is fixed as `(mode_b - mode_a)`; defaults are `mode_a=B1`, `mode_b=B2`.
- Optional pair washout between legs: `BENCH_PHASE70_PAIR_WASHOUT_S=0` by default.
  - Set `BENCH_PHASE70_PAIR_WASHOUT_S=10` or `30` when debugging order effects.
  - Optional guarded extras: `BENCH_PHASE70_PAIR_WASHOUT_SYNC=1` and `BENCH_PHASE70_PAIR_WASHOUT_DROP_CACHES=1` (drop-caches runs only if root and writable; otherwise skipped).
- Recommended pair counts are even (`N=6` or `N=8`) for exact AB/BA balance.
- Override knobs:
  - `BENCH_PHASE70_ALLOW_MISSING_KVBM_METRICS=1` (continue but non-decision-grade)
  - `BENCH_PHASE70_DISABLE_MECHANISM_GATE=1` (continue without early gate, non-decision-grade)
  - `BENCH_PHASE70_SKIP_PREFLIGHT=1` (skip preflight checks)
  - `BENCH_PHASE70_DECISION_GRADE_REQUIRE_REHYDRATE=0` (allow write-only evidence for decision-grade; verdict still records rehydrate-absent reason)

Artifacts written by the runner:

- `bench/results/phase70_rehydrate_pair_repeats_manifest_<ts>.json`
- `bench/results/phase70_rehydrate_pair_repeats_summary_<ts>.json`
- `bench/results/phase70_rehydrate_pair_repeats_summary_<ts>.csv` (per-run)
- `bench/results/phase70_rehydrate_pair_repeats_deltas_<ts>.csv` (per-pair deltas)
- `bench/results/phase70_rehydrate_pair_repeats_order_check_<ts>.json` (descriptive order-effect check with AB/BA min/max and approximate 95% bands)
- `bench/results/phase70_rehydrate_pair_repeats_verdict_<ts>.json` (run-valid + decision-grade verdict and reason codes)

### 6d) Phase70 brief-ready results pack (single command)

Generate a publish folder that can be dropped into a whitepaper/brief repo:

```bash
python3 scripts/make_phase70_results_pack.py \
  --results-root bench/results \
  --ts <ts>
```

Output folder:

- `bench/results/publish/phase70_pairs<N>_c<replay_concurrency>_<ts>/`
- includes: `summary.csv`, `summary.json`, `deltas.csv`, `order_check.json`, `methodology.md`
- includes: `verdict.json` when source verdict exists (`phase70_rehydrate_pair_repeats_verdict_<ts>.json` copied into the pack)
- includes: `tables/table_main_latency.csv`, `tables/table_mechanism.csv`, `tables/table_order_effect.csv`
- includes: `figures/` (PNG files when matplotlib is installed; otherwise a README note)

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
- `scripts/phase60_extract_knee_table.py`
- `scripts/phase60_baseline_manifest_semantic.py`
- `scripts/recommend_phase70_crossover_from_phase60.py`
- `scripts/bench_phase70_rehydrate_pair_repeats.sh` (legacy alias: `scripts/bench_phase70_rehydrate_c1_pair_repeats.sh`)
- `scripts/analyze_phase70_pairs.py`
- `scripts/make_phase70_results_pack.py`
- `scripts/bench_phase70_preflight.sh`
- `scripts/phase70_check_mechanism_gate.py`
- `scripts/phase70_write_verdict.py`
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
  `start_ts`, `end_ts`, `latency_ms`, `status_code`, `http_status`, `prompt_id`, `prompt_tokens_est`, `output_len_chars`, `error`, `request_id`, `ttft_ms`, `ttfc_ms` (stream mode), `ttfb_ms` (optional), `stream_first_event_type`, `stream_error`.
- `kvbm_metrics_snapshots.jsonl`: raw `/metrics` snapshots at phase boundaries.
- `summary.json`: includes run validity, per-phase + overall metrics (latency + TTFT + TTFC/TTFB when present), KVBM deltas, eviction replay signals.
- `summary.json` also includes top-level `mode`, `kvbm_enabled`, `kvbm_metrics_available`, and `kvbm_metrics_status`.
- `phase_deltas/`: `phase_<name>_kvbm_metrics_{start,end,delta}.json` and `phase_<name>_os_io_{start,end,delta}.json`.
- `nvme_identity.json`, `nvme_smart_pre.json`, `nvme_smart_post.json`: legacy NVMe identity + SMART snapshots.
- `device_metadata_pre.json`, `device_metadata_post.json`: standardized storage metadata snapshots
  (NVMe identity/SMART, PCIe link state, block/filesystem mapping, kernel/OS info, and capture errors).
- `io/`: optional I/O attribution bundle (enabled with `--io-attrib`):
  - `io/block_stat_timeline.csv` (canonical procfs block timeline)
  - `io/proc_io_timeline.csv` (canonical procfs per-process timeline)
  - `io/iostat.csv` (legacy compatibility copy of `block_stat_timeline.csv`)
  - `io/pidstat.csv` (legacy compatibility copy of `proc_io_timeline.csv`)
  - `io/iostat.raw.log` (tool output when `iostat` is installed)
  - `io/pidstat.raw.log` (tool output when `pidstat` is installed)
  - `io/meminfo_snapshots.jsonl`
  - `io/vmstat_snapshots.jsonl`
  - `io/io_attribution_report.json`
  - `io/io_attrib_verdict.json` (A2 replay gate verdict from checker)
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

- TTFC is preferred for streamed latency; TTFT remains for backward compatibility and is a proxy in non-stream mode.
- Some current runs still lack TTFC capture; knee/recommender paths fall back to TTFT and mark fallback usage in outputs.
- If replay reads do not appear and `kvbm_matched_tokens_delta` is zero, disk rehydrate is gated because no cross-request reuse path activated.
- Dataset L (`>=32k`) remains blocked on local engines with `max_num_tokens=8192`; use a rebuilt context ladder (16k -> 32k -> higher) before publishing L-tier claims.

## Interactive Validation Notes (2026-02-08)

- Mode-compare runs are stable with `cpu_only` and `cpu_disk`.
- `off` mode can be runtime-fragile in this build and is not required for Phase 1 evidence.
- If startup/model registration races occur, use:
  - `--model-resolve-timeout-s` / `--model-resolve-poll-s` in `bench.run_bench`,
  - `BENCH_COMPARE_MODEL_RESOLVE_TIMEOUT_S` in `scripts/bench_run_mode_compare.sh`.

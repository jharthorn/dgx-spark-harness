# DGX Spark Dynamo + TRT-LLM + KVBM Benchmark Runbook

This runbook is the canonical path to reproduce SSD KV offload benchmarking on DGX Spark using Dynamo + TensorRT-LLM + KVBM.

- Platform assumptions: DGX Spark (aarch64, unified memory), no GPUDirect RDMA fast path.
- Endpoint used for load: `/v1/completions` only.
- Model prompt format: Llama 3 chat template wrapped into completion prompt.

## Scripted Quick Path

If you prefer an operator script flow instead of manual command blocks:

```bash
scripts/bench_prepare_host.sh
scripts/bench_container_up.sh
scripts/bench_start_worker.sh
scripts/bench_start_frontend.sh
scripts/bench_wait_ready.sh
scripts/bench_health.sh
scripts/bench_smoke_completion.sh
source ~/dynamo-venv/bin/activate
scripts/bench_run_smoke.sh
scripts/bench_run_matrix.sh
BENCH_COMPARE_SKIP_READY=1 BENCH_KV_MODE_LIST="cpu_only cpu_disk" scripts/bench_run_mode_compare.sh
scripts/bench_results_summary.sh
```

## 1) Prepare Host Directories and Config

```bash
set -euxo pipefail

scripts/bench_prepare_host.sh
```

Expected: `/mnt/nvme/kvbm/kvbm_llm_api_config.yaml` exists and is readable.

## 2) Build (Optional) and Run Container

Build wrapper image from this repo:

```bash
docker build -t dgx-spark-dynamo-bench:latest images/dyn
```

[ASSUMPTION: base image `trtllm-rc6-dynamo-nixl:latest` is already present locally.]

Run container:

```bash
set -euxo pipefail
DEV_ARGS=""
for d in /dev/nvidia-fs*; do
  DEV_ARGS+=" --device=$d"
done

docker rm -f dyn >/dev/null 2>&1 || true

docker run -d --name dyn \
  --gpus all \
  --ipc host \
  --network host \
  $DEV_ARGS \
  --device=/dev/nvme0n1 \
  --device=/dev/nvme0n1p2 \
  --ulimit memlock=-1 \
  --cap-add IPC_LOCK \
  --cap-add SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -e MODEL_HANDLE="nvidia/Llama-3.1-8B-Instruct-FP8" \
  -e DYN_KVBM_DISK_CACHE_GB=32 \
  -e DYN_KVBM_DISK_CACHE_DIR=/mnt/nvme/kvbm \
  -e CUFILE_ENV_PATH_JSON=/etc/cufile/cufile.json \
  -v /mnt/nvme:/mnt/nvme:rshared \
  -v /mnt/nvme/kvbm/kvbm_llm_api_config.yaml:/tmp/kvbm_llm_api_config.yaml:ro \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  -v /run/udev:/run/udev:ro \
  -v /dev/disk:/dev/disk:ro \
  dgx-spark-dynamo-bench:latest \
  bash -lc "sleep infinity"
```

Expected:

```bash
docker ps --filter name=dyn
docker exec dyn bash -lc "findmnt -T /mnt/nvme/kvbm -o TARGET,SOURCE,FSTYPE,OPTIONS"
```

## 3) Start TRT-LLM Worker (Inside Container)

```bash
docker exec -d dyn bash -lc '
set -euxo pipefail
mkdir -p /tmp/bench-logs
export DYN_KVBM_DISK_CACHE_GB=${DYN_KVBM_DISK_CACHE_GB:-32}
export DYN_KVBM_DISK_CACHE_DIR=${DYN_KVBM_DISK_CACHE_DIR:-/mnt/nvme/kvbm}

python3 -m dynamo.trtllm \
  --model-path /root/.cache/huggingface/hub/models--nvidia--Llama-3.1-8B-Instruct-FP8/snapshots/42d9515ebd69eea3a87351d079c671c3c5ff0a31 \
  --endpoint dyn://dynamo.tensorrt_llm.generate \
  --extra-engine-args /tmp/kvbm_llm_api_config.yaml \
  --store-kv file \
  > /tmp/bench-logs/worker.log 2>&1
'
```

Expected: worker stays running, no immediate process exit.

```bash
docker exec dyn bash -lc "tail -n 50 /tmp/bench-logs/worker.log"
```

## 4) Start HTTP Frontend (Inside Container)

```bash
docker exec -d dyn bash -lc '
set -euxo pipefail
mkdir -p /tmp/bench-logs
MODEL_DIR="$(python3 - <<PY
import glob
paths=glob.glob("/root/.cache/huggingface/hub/models--nvidia--Llama-3.1-8B-Instruct-FP8/snapshots/*")
print(sorted(paths)[-1] if paths else "")
PY
)"
test -d "$MODEL_DIR"

python3 -m dynamo.frontend \
  --http-host 0.0.0.0 \
  --http-port 8000 \
  --store-kv file \
  --exp-python-factory \
  --model-name "nvidia/Llama-3.1-8B-Instruct-FP8" \
  --model-path "$MODEL_DIR" \
  > /tmp/bench-logs/frontend.log 2>&1
'
```

Expected: frontend process listening on port `8000`.

## 5) Validate Health and Model Registration

```bash
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8000/v1/models | jq .
```

Expected:

- `/health` returns healthy JSON.
- `/v1/models` returns non-empty `.data`.

## 6) Smoke Test `/v1/completions`

```bash
MODEL_ID="$(curl -s http://localhost:8000/v1/models | jq -r '.data[0].id')"
USER_MSG="What can you tell me about the band Sleep Token?"
PROMPT=$'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'"$USER_MSG"$'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

curl -sS http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg model "$MODEL_ID" \
    --arg prompt "$PROMPT" \
    '{
      model: $model,
      prompt: $prompt,
      max_tokens: 256,
      temperature: 0.7,
      stop: ["<|eot_id|>"],
      stream: false
    }')" | jq -r '.choices[0].text'
```

Expected: non-empty completion text.

## 7) Run Short-Context Benchmarks

Sweep `1/4/8` first (expand to `2/16/32/64` as stable):

```bash
cd ~/dgx-spark-harness

for c in 1 4 8; do
  python3 -m bench.run_bench \
    --base-url http://127.0.0.1:8000 \
    --kv-mode cpu_disk \
    --scenario standard \
    --prompt-set short \
    --requests 64 \
    --warmup 8 \
    --concurrency "$c" \
    --max-tokens 256 \
    --temperature 0.7 \
    --stop "<|eot_id|>" \
    --collect-telemetry \
    --container-name dyn \
    --kvbm-cache-dir /mnt/nvme/kvbm \
    --run-id "short_c${c}_$(date -u +%Y%m%dT%H%M%SZ)"
done
```

## 8) Run Long-Context Benchmarks

[ASSUMPTION: this local engine is often built with `max_num_tokens=8192`; use a capped long range to avoid fatal worker unregister.]

```bash
cd ~/dgx-spark-harness

for c in 1 4 8; do
  python3 -m bench.run_bench \
    --base-url http://127.0.0.1:8000 \
    --kv-mode cpu_disk \
    --scenario standard \
    --prompt-set long \
    --long-range 6000:7600 \
    --requests 24 \
    --warmup 4 \
    --concurrency "$c" \
    --max-tokens 256 \
    --temperature 0.7 \
    --stop "<|eot_id|>" \
    --collect-telemetry \
    --container-name dyn \
    --kvbm-cache-dir /mnt/nvme/kvbm \
    --run-id "long_c${c}_$(date -u +%Y%m%dT%H%M%SZ)"
done
```

## 9) Run Eviction/Replay Scenario

This executes warm set A, pressure set B, then replays A.

```bash
cd ~/dgx-spark-harness

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
  --temperature 0.7 \
  --stop "<|eot_id|>" \
  --collect-telemetry \
  --container-name dyn \
  --kvbm-cache-dir /mnt/nvme/kvbm \
  --run-id "eviction_replay_$(date -u +%Y%m%dT%H%M%SZ)"
```

Check replay signal:

```bash
jq '.run_valid, .overall_summary.eviction_replay_signal_kvbm, .overall_summary.eviction_replay_signal_io' bench/results/eviction_replay_*/summary.json
```

## 9a) Run Reuse Verification (Phase 1.5 Gate)

This runs 3 identical sequential requests (`reuse_1`, `reuse_2`, `reuse_3`) and reports matched/onboard deltas per request window.

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
  --stop "<|eot_id|>" \
  --run-id "reuse_verify_$(date -u +%Y%m%dT%H%M%SZ)"
```

Check reuse gate:

```bash
jq '.overall_summary.reuse_verify_signal_kvbm, .request_identity.reuse_verify_identity' bench/results/reuse_verify_*/summary.json
```

## 9b) Baseline vs Offload Mode Compare

Same workload, only KV mode changes:

```bash
source ~/dynamo-venv/bin/activate
BENCH_COMPARE_SKIP_READY=1 BENCH_KV_MODE_LIST="cpu_only cpu_disk" scripts/bench_run_mode_compare.sh
```

Recommended on this stack:

- Compare `cpu_only` vs `cpu_disk` for stable baseline/offload contrast.
- `off` mode may fail due to runtime discovery-store behavior in this build; treat as optional diagnostic only.
- If startup is stable and you want strict gating, set `BENCH_COMPARE_READY_REQUIRE_ENDPOINTS=1`.

## 10) Collect and Summarize Artifacts

Per run, inspect:

- `bench/results/<run_id>/summary.json`
- `bench/results/<run_id>/report.md`
- `bench/results/<run_id>/requests.jsonl`
- `bench/results/<run_id>/telemetry/iostat_*.log`
- `bench/results/<run_id>/telemetry/pidstat.log`
- `bench/results/<run_id>/telemetry/nvidia_smi_*.txt`
- `bench/results/<run_id>/telemetry/kvbm_*`
- `bench/results/<run_id>/telemetry/cufile_*`
- `bench/results/<run_id>/telemetry/docker_dyn_logs.txt`

Quick summary helper:

```bash
scripts/bench_results_summary.sh "bench/results/*/summary.json"
```

## Expected Evidence for KV Offload

- NVMe writes increase under long-context and/or higher concurrency.
- KVBM cache directory snapshots show active file churn and size movement.
- Eviction replay run shows phase-level read deltas (`read_mib_delta`) if rehydrate happens.
- If `kvbm_matched_tokens_delta` stays zero in replay/reuse verification, disk onboarding is gated and rehydrate cannot be observed.
- If replay reads do not appear, document likely causes:
  - replay served from in-memory tiers,
  - insufficient eviction pressure,
  - disk rehydrate not active in this runtime mode.

## Troubleshooting (Validated)

- `worker.log` / `frontend.log` missing:
  - confirm startup wrappers create `/tmp/bench-logs` and that prior `pkill` patterns are not matching the shell itself.
- Ready loop appears stuck but `/v1/models` is non-empty in another shell:
  - use model-based readiness (default),
  - or bypass compare-loop readiness with `BENCH_COMPARE_SKIP_READY=1`.
- Compare run exits early with "No models returned by /v1/models":
  - rerun with a larger model resolve timeout:
    `BENCH_COMPARE_MODEL_RESOLVE_TIMEOUT_S=300`.

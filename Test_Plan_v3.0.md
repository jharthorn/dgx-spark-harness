# DGX Spark Storage-Aware Inference Test Plan (v3.0)

> Superseded by `docs/Test_Plan_v3.3.md` (kept for historical reference)

Target platform: **NVIDIA DGX Spark**, GB10 Grace Blackwell
128 GB coherent unified memory • 4 TB NVMe • 8B and 70B Llama models
*(Source: Test Plan 3.0 ODT)* 

---

## 0. Change Log

* **v3.0 initial:**

  * Replaces v2.5’s paging-only assumptions with an explicit **KV tiering** design using **Dynamo**.
  * Adds **test definitions per hypothesis**,
  * Expands **harness + telemetry** specifications,
  * Defines **analysis outputs** and **quality gates**.

---

## 1. Objective and Scope

### Objective

Quantify **when and how storage affects real-time LLM inference** on a UMA system. Produce comparable evidence for two regimes:

### Stack A — UMA-Only

* KV and weights stored in unified memory
* NVMe **not** part of KV hot path
* Tail latency dominated by GPU + scheduler

### Stack B — Dynamo Tiered KV

* KV overflows into tiered storage
* **Tier 2 = NVMe**
* NVMe latency directly shapes p95 and p99

### In Scope

* Queue saturation / usable concurrency
* KV pressure (context + concurrency)
* Storage QoS sensitivity under KV spill
* Tier sizing and eviction policies
* Correlation of KV telemetry ↔ storage metrics ↔ tail latency

### Out of Scope

* Distributed inference
* Dataset curation
* Fine-tuning model-quality comparisons

---

## 2. Hypotheses and Experiment Map

Each hypothesis corresponds to a runnable experiment in Section 8.

| ID      | Stack | Hypothesis                                                     | Expected Outcome                                         |
| ------- | ----- | -------------------------------------------------------------- | -------------------------------------------------------- |
| **H0**  | A, B  | Queue saturation defines usable concurrency U_work.            | p99 stays within a fixed multiple of p50 until the knee. |
| **H1**  | A     | LoRA cold vs warm affects TTFT but not KV-driven tails.        | LoRA loads are a weight-storage story, not KV.           |
| **H2A** | A     | More context/concurrency does *not* create KV storage traffic. | NVMe flat; p99 from queue/GPU only.                      |
| **H2B** | B     | RAM tier overflow → NVMe tier-2 fetches → storage knee.        | First clear storage-dependent latency knee.              |
| **H3**  | A, B  | OS UMA counters informative for A, insufficient for B.         | Motivates KV-aware telemetry.                            |
| **H4A** | A     | NVMe QoS degradation barely changes p99.                       | Storage-indifferent control.                             |
| **H4B** | B     | Degraded NVMe QoS increases TTFT/p99 proportionally.           | Direct coupling: NVMe latency → p99.                     |
| **H5**  | B     | Larger KV working sets worsen p95/p99 at fixed compute.        | Multi-tenant KV scaling effects.                         |
| **H6**  | B     | Larger RAM tiers or smarter eviction shift the knee.           | Practical guidance for tier sizing.                      |
| **H7**  | B     | Dynamo fetch latency explains NVMe IOPS + p99.                 | Mechanistic mapping: KV → storage → tails.               |
| **H8A** | A     | CPU iowait flat when queue-bound.                              | Why host storage signals are flat.                       |
| **H8B** | B     | iowait + r_await track p95/p99 once NVMe engaged.              | Telemetry-complete storage story.                        |

### Definition: U_work

For a fixed prompt distribution:

* Sweep concurrency
* Find first point where **p99 > 2.0 × p50 for two consecutive steps**
* Choose **U_work = one step below knee**

---

## 3. Platform Under Test

### Hardware

* DGX Spark (GB10 Grace-Blackwell)
* 128 GB coherent unified memory
* 4 TB NVMe M.2 SSD
* NIC present, unused

### OS & Drivers

* DGX OS baseline
* CUDA-compatible NVIDIA drivers
* TRT-LLM release matching engine builds

### Power/Clocks

* Defaults unless test requires static clocks
* Document any deviations

---

## 4. Software Stacks

### 4.1 Stack A — UMA Only (Control)

* TRT-LLM treats UMA as device memory
* KV sharing off for regression clarity
* Server rejects requests before UMA exhaustion
* Expect **near-zero NVMe reads** for KV

### 4.2 Stack B — Dynamo Tiered KV

* TRT-LLM + Dynamo Distributed KV Cache Manager + KVBM
* **Tiers:**

  * Tier0: GPU or reserved UMA
  * Tier1: CPU or lower-priority UMA
  * Tier2: NVMe SSD
* **Config Parameters:**

  * `tier*.capacity_bytes`
  * `tier2.path` + `allocate_bytes`
  * `kv_block_size_bytes`
  * `eviction_policy: LRU|LFU|segmented`
  * `prefetch_enable`, `prefetch_depth`
  * `telemetry_enable` + interval
* Expected: Tier0+Tier1 undersized to ensure tier2 activity

---

## 5. Models and Workloads

### 5.1 Models

* **Llama 3 8B** — H0 queue knee, no KV pressure
* **Llama 3 70B FP4** — primary for H2–H8

  * Same weights used for Stack A and B

### 5.2 Workload Classes

* **Fixed context uniform** — H0, H2A, H2B
* **Sessioned multi-turn** — H5, H6
* **LoRA adapter suite** — H1

### Global Defaults

* Duration: **300 s**
* Repetitions: **3**
* Uniform arrivals unless specified
* Seed all workloads

---

## 6. Metrics and Telemetry

### 6.1 Latency & Throughput

* p50 / p95 / p99 TTFT
* End-to-end latency
* RPS admitted/completed/rejected

### 6.2 System & GPU

* CPU: usr, sys, iowait
* Memory: MemAvailable, Cached
* GPU: util, mem used, outstanding requests

### 6.3 Storage

* `iostat -x` at 100–250 ms

  * r/s, rMB/s, r_await, aqu-sz, util%
* smartctl (host read counters)
* Optional: blktrace/bpftrace deep dives

### 6.4 Dynamo KV Telemetry (Stack B)

Example line:

```json
{
 "ts": 1731624501.123,
 "tier0": {"hits": 12345, "misses": 678, "bytes_in": 987654321, "bytes_out": 456789},
 "tier1": {"hits": 2345, "misses": 789, "bytes_in": 1234567, "bytes_out": 7654321},
 "tier2": {
   "hits": 456, "misses": 0,
   "bytes_in": 2222222, "bytes_out": 1111111,
   "fetch_p50_ms": 0.42, "fetch_p95_ms": 1.8, "fetch_p99_ms": 4.7
 },
 "kv_block_size_bytes": 65536,
 "evictions": 321,
 "prefetches": 0
}
```

* Cadence: **200 ms**, aligned with sysmon timestamps

---

## 7. Harness and Repository Structure

```
harness/
  README.md
  configs/
    stackA_llama70b_baseline.yaml
    stackB_llama70b_dynamo_tiered.yaml
  inputs/
    prompts/
    lora_lists/
  src/
    loadgen.py
    workloads/
      fixed_context.py
      sessioned_chat.py
    telemetry/
      sysmon.sh
      nvme_iostat.py
      gpu_poll.py
      dynkv_ingest.py
  runners/
    run_H0_queue_knee.sh
    run_H1_coldwarm_lora.sh
    run_H2A_uma_pressure.sh
    run_H2B_dynamo_kv_pressure.sh
    run_H4A_storage_qos.sh
    run_H4B_dynamo_storage_qos.sh
    run_H5_kv_workingset_scaling.sh
    run_H6_tier_sizing_policy.sh
    run_H7_kv_telemetry_sweep.sh
  analysis/
    process_results.py
    backfill_summary.py
    figures/
  runs/
    2025-11-15_H2B_ctx512_conc32/
```

### 7.1 Load Generator Flags

```
--endpoint URL
--stack A|B
--model L8B|L70B
--workload fixed_context|sessioned_chat
--context_tokens INT
--concurrency INT
--nonce_per_user BOOL
--duration_s INT
--seed INT
```

### 7.2 Sysmon

Runs:

* `iostat -x 0.2`
* `mpstat 0.5`
* `free -s 1`
* `nvidia-smi dmon`
* Stack B: tail Dynamo logs

### 7.3 Results Schema

* `metrics.jsonl`
* `sysmon.jsonl`
* Include stack, model, context, concurrency, seed, notes

---

## 8. Test Definitions

Each test corresponds directly to the hypotheses in Section 2.

### 8.1 H0 — Queue Knee / U_work

* **Model:** L8B
* **Sweep:** concurrency 1 → 256
* Output: p50/p95/p99 curves, U_work

### 8.2 H1 — Cold vs Warm LoRA

* **Model:** L70B
* Cold: restart server + `drop_caches`
* Warm: rerun
* Output: TTFT distributions, NVMe deltas

### 8.3 H2A — UMA-Only KV Behavior

* **Model:** L70B
* Sweep: context (256 → 2048), concurrency
* Output: NVMe flatline; failures = OOM/rejections

### 8.4 H2B — Dynamo KV Tiering Knee

* **Model:** L70B
* Sweep: context 256 → 4096
* Output: p99 knee, tier hits, NVMe r_await

### 8.5 H3 — UMA vs Tiered KV Interpretation

* Compare H2A vs H2B OS counters

### 8.6 H4A — Storage QoS (UMA-Only Control)

* `fio` contention
* p99 barely moves despite NVMe load

### 8.7 H4B — Storage QoS (Tiered KV)

* Tier2 hit rate ≥ 20–30 percent with 4k prompts (`STACKB_MAX_INPUT_LEN>=4096`, `STACKB_MAX_SEQ_LEN≈16384`, `STACKB_MAX_NUM_TOKENS≈32000`, `STACKB_MAX_BATCH_SIZE=4`)
* Use spill config `configs/stackB_llama70b_dynamo_tiered_spill.yaml` (T0≈512 MiB, T1≈1 GiB, T2≈64 GiB, block_size=64 KiB) to guarantee tier2 traffic on UMA
* Run H4B with `H4B_CONTEXT_TOKENS` unset (defaults to 4096 when high-context envs are present) and `CONCURRENCY=32`
* fio QoS workload: randrw 70/30, 16k, iodepth 32, 4 jobs on tier2 path
* p99 tracks `tier2_fetch_p95_ms`; NVMe util/latency rises above fio-only baseline

### 8.8 H5 — KV Working-Set Scaling

* Sessioned multi-turn
* Sweep: session length, reuse ratio
* Output: p95/p99 vs reuse; NVMe IOPS

### 8.9 H6 — Tier Sizing & Eviction

* Sweep Tier0+Tier1 capacities and policies
* Output: knee shift + p99 correlation

### 8.10 H7 — KV Telemetry → Storage Mapping

* Predict NVMe traffic from KV telemetry
* Correlate predicted vs observed

### 8.11 H8A/B — CPU iowait Narrative

* A: flat iowait vs queueing p99
* B: iowait tracks tier2 activity

---

## 9. Analysis Outputs & Figure Checklist

* H0: p50/p95/p99 vs concurrency
* H2A vs H2B: p99 vs context
* H2B: tier hit ratio vs context
* H4B: p99 vs NVMe r_await (annotate fio profiles)
* H5: p99 vs KV working-set size
* H6: knee vs tier capacity
* H7: predicted vs observed NVMe r/s + p99 vs fetch_p95
* H8: iowait vs p99 (A and B)

Export all backing data as CSV.

---

## 10. Quality Gates & Acceptance Criteria

* ≥95 percent aligned samples (latency, sysmon, Dynamo)
* CoV(p99) < 10 percent across repetitions
* **Sanity checks:**

  * Stack A H2A: NVMe r/s within 10 percent of idle
  * Stack B H2B: tier2 hits ≥ 20 percent at knee
* Reproducibility: p99 within 5 percent

---

## 11. Risks & Mitigations

* Missing telemetry in Stack B → add sidecar
* OS cache interference → cold then warm passes
* Clock skew → unify time source
* Disk thermal throttling → pre-warm + large fio files

---

## 12. Runbooks

### 12.1 Stack A

* Launch UMA-only TRT-LLM
* Disable swap
* Set `max_num_tokens`
* Run H0 then H2A

### 12.2 Stack B

* Build TRT-LLM with Dynamo
* Create NVMe directory for KVBM
* Provide YAML config
* Confirm telemetry at 200 ms cadence

---

## 13. Configuration Template (YAML)

```yaml
run:
  id: "H2B_ctx_sweep_2025-11-15"
  stack: "B"
  model: "L70B"
  workload: "fixed_context"
  context_tokens: [256, 512, 1024, 2048, 4096]
  concurrency: 32
  duration_s: 300
  repetitions: 3
  seed: 42

server:
  endpoint: "http://127.0.0.1:8000"
  nonce_per_user: true

dynamo:
  kv_block_size_bytes: 65536
  eviction_policy: "LRU"
  tier0: {capacity_bytes: 8GiB}
  tier1: {capacity_bytes: 24GiB}
  tier2: {path: "/nvme/kvbm/l70b", allocate_bytes: 512GiB}

telemetry:
  cadence_ms: 200
  enable_dynkv: true
  enable_iostat: true

notes: "Baseline H2B sweep for knee discovery"
```

---

## 14. Data Management

* Runs stored under `runs/YYYY-MM-DD_testid/`
* Include: config.yaml, raw logs, metrics.jsonl, figures
* Do **not** store models

---

## 15. Success Criteria

* **Stack A:** NVMe flat in H2A/H4A; p99 growth explained by queue/GPU.
* **Stack B:** Tier2 hits + NVMe r_await explain p95/p99 in H2B/H4B.
* **H6:** Tier sizing shifts knee; policy affects hit rate.
* **H7:** KV telemetry quantitatively maps to NVMe + p99.

---

## 16. Appendix A — CLI Examples

```
./run_H0_queue_knee.sh --stack A --model L8B --context_tokens 256

./run_H2A_uma_pressure.sh --model L70B \
  --concurrency $(cat runs/H0/uwork.txt) \
  --contexts 256,512,1024,2048 --nonce_per_user

./run_H2B_dynamo_kv_pressure.sh --model L70B \
  --concurrency $(cat runs/H0/uwork.txt) \
  --contexts 256,512,1024,2048,4096 \
  --config configs/stackB_llama70b_dynamo_tiered.yaml

./run_H4B_dynamo_storage_qos.sh --fio_profile moderate
./run_H4B_dynamo_storage_qos.sh --fio_profile heavy
```

---

## 17. Appendix B — Minimal Ingestion Schema

### `metrics.jsonl`

```json
{
 "run_id": "H2B_ctx_sweep_2025-11-15",
 "ts": 1731624500.412,
 "stack": "B",
 "model": "L70B",
 "context_tokens": 2048,
 "concurrency": 32,
 "lat_ttft_ms": 311.4,
 "lat_e2e_ms": 1890.2,
 "rc": 200
}
```

### `sysmon.jsonl`

```json
{
 "ts": 1731624500.600,
 "cpu": {"user": 31.2, "system": 9.8, "iowait": 4.3},
 "mem": {"MemAvailable": 27456798720, "Cached": 3355443200,
         "SwapTotal": 0, "SwapFree": 0},
 "gpu": {"util": 92, "mem_used_bytes": 92771225600},
 "nvme": {
   "rps": 2200, "rMBs": 350,
   "r_await_ms": 1.7, "aqu_sz": 7.4,
   "util_pct": 78.3
 }
}
```

### `dynkv.jsonl`

Follows the schema in Section 6.4.

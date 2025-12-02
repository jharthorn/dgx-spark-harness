DGX Spark LLM Storage-Aware Inference Test Plan (v3.3)
======================================================

Status: Formal Engineering Specification for Whitepaper  
Target Model: Llama 3.3 70B (NVFP4)  
System Under Test: NVIDIA DGX Spark (128 GiB UMA, NVMe Tier2)  
Date: 2025-11

0. Change Log
-------------

v3.3 (This Revision)

* Merges conceptual framing of v3.2 with full operational specificity of v3.0
* Restores strict definitions: U_work, Collapse Point, Storage Knee
* Restores missing telemetry schemas and cadence requirements
* Restores full hypothesis → experiment → expected output mapping
* Restores runbooks for Stack A and B
* Adds Profile-Driven Testing Framework (Comfy / Spill / Stress) across all hypotheses
* Integrates “Infinite LoRA” and “Context Re-Hydration” (H1, H9)
* Adds updated KV tier config templates and environment variable documentation

v3.2

* Introduced execution profiles
* Added H1 LoRA bandwidth scenario and H9 re-hydration
* Strengthened collapse-point and storage-knee definitions

v3.0

* Established foundational hypothesis map, harness structure, telemetry schemas, YAML templates

1. Executive Summary
--------------------

NVIDIA DGX Spark is a unified memory architecture (UMA) inference platform where CPU and GPU share a coherent 128 GiB memory pool. This test plan evaluates how large-model KV cache behavior interacts with:

* UMA-only serving (Stack A)
* Dynamo tiered KV cache spanning UMA and NVMe (Stack B)
* Background storage activity
* Stress conditions such as large context windows, high concurrency, and LoRA churn

This document characterizes:

* The UMA-only collapse point (Stack A) under large context + concurrency.
* The resilience and extended safety net provided by Dynamo KV Tier0/Tier1 UMA + Tier2 NVMe.
* The conditions under which storage becomes a first-class latency determinant (p95/p99 TTFT).
* The operational envelope for developers serving 70B-class models on DGX Spark.

The v3.3 plan produces reproducible, engineering-grade evidence supporting NVIDIA messaging around:

* predictable behavior of storage-aware inference
* feasibility of LoRA-rich deployments
* capability of UMA-based devices for edge-scale inference workloads

2. Objective and Scope
----------------------

### 2.1 Objectives

Quantify Baseline UMA Behavior  
Establish Stack A’s concurrency knee, context envelope, and tail-latency characteristics.

Validate Transparency of Dynamo Under “Comfy” Loads  
Confirm Stack B performs within ±5% of Stack A when KV fits in UMA.

Characterize KV Spill Behavior  
Measure how Stack B behaves as KV exceeds UMA tiers and flows to NVMe.

Define Collapse Points and Storage Knees  
Establish inflection points where UMA usage or NVMe fetch latency dominates tails.

Demonstrate High-Cardinality LoRA Support  
Validate ability to store dozens–hundreds of LoRA adapters on NVMe without service collapse.

Measure Context Re-Hydration Latency  
Quantify the cost of restoring dormant session KV blocks from NVMe.

Produce Engineering-Ready Guidance  
Recommend UMA KV budgets, tier splits, and telemetry profiles for DGX Spark deployments.

### 2.2 In Scope

* Llama 3.3 70B NVFP4 inference
* UMA-only vs tiered inference behavior
* Context-length scaling
* Concurrency scaling
* LoRA swap-in/out load
* Background storage I/O interference
* Tier sizing and eviction policy sensitivity
* KV telemetry fidelity and sampling analysis

### 2.3 Out of Scope

* Multi-GPU training
* Multi-node inference
* Dataset quality, alignment, or hallucination metrics
* Model fine-tuning quality evaluation

3. Definitions
--------------

These definitions are required for whitepaper-quality reproducibility and were restored from v3.0.

### Usable Concurrency (U_work)

The highest concurrency level where:

* p99 ≤ 2.0 × p50 for two consecutive concurrency steps.
* Used as baseline concurrency for H2, H4, H5, H6, H7 tests.

### Collapse Point

The concurrency or effective throughput at which the system breaches SLA such that:

* p99 TTFT > 1000 ms, **or**
* p99 grows uncontrollably with increasing concurrency, **or**
* RPS admitted collapses due to scheduler failure or memory exhaustion.

### Storage Knee (Stack B only)

The first point where:

* Tier2 Fetch p95 ≥ 50% of TTFT p95
* This indicates NVMe latency is now the dominant component of tail behavior.

4. System Under Test (SUT)
--------------------------

### 4.1 Hardware

* NVIDIA DGX Spark workstation
* 128 GiB coherent UMA (LPDDR5X)
* 4 TB NVMe SSD (Tier2)
* Grace CPU + Blackwell GPU (GB10)

### 4.2 Software

* NVIDIA DGX OS
* TensorRT-LLM (Stack A and B)
* Dynamo KV Tiering (Stack B)
* KVBM exporter enabled
* Python harness and loadgen provided

### 4.3 Model

* Llama-3.3-70B-Instruct-NVFP4 (primary)
* Optional: Llama-3-8B for H0 baseline concurrency

5. Execution Profiles
---------------------

This framework (introduced in v3.2) applies across all hypotheses and test sequences.

| Profile | UMA KV Budget (T0+T1) | Context Window | Concurrency | Expected Behavior |
| --- | --- | --- | --- | --- |
| Comfy | 8–16 GiB | 2k–4k | Moderate | Minimal spill, NVMe idle |
| Spill | 4–8 GiB | 4k | Moderate–High | Significant KV spill |
| Stress | 1–2 GiB | 4k–8k | High (≥32) | Heavy spill; NVMe 50–70% util |

6. Hypothesis Map (H0–H9)
-------------------------

This table integrates the descriptive framing of v3.2 with the operational specificity of v3.0.

### H0 — Queue Knee & UMA Baseline Capacity

**H0A — Stack A**

* Profile: Comfy
* Sweep: concurrency 1 → 256
* Context: 2k–4k
* Outputs:
  * p50/p95/p99 vs concurrency
  * U_work
  * GPU/CPU saturation markers

**H0B — Stack B (Comfy Transparency)**

* Same workload as H0A.  
* Success: Stack B within ±5% latency and throughput of Stack A.

### H1 — LoRA Storage Bandwidth (“Infinite LoRA”)

* Profile: Spill
* Workload:
  * 100–300 LoRA adapters stored in Tier2
  * nonce_per_user=True to force adapter thrash
* Metrics:
  * NVMe random-read IOPS
  * TTFT drift
  * Tier2 fetch bytes
* Expected:
  * Stack A: OOM or severe degradation
  * Stack B: controlled tails, stable throughput

### H2A/B — UMA vs KV Pressure

**H2A — UMA Pressure (Stack A)**

* Profile: Comfy → Stress
* Sweep: context 1k → 2k → 4k
* Outcomes:
  * Collapse point for UMA-only
  * NVMe flatline (control validation)

**H2B — KV Pressure (Stack B)**

* Profile: Spill
* Sweep: context 256 → 4096
* Metrics:
  * p99 vs context
  * NVMe r_await vs p99
  * tier2_hits %, spill volume
* Expected:
  * Higher context/concurrency ceiling vs H2A
  * Gradual rather than abrupt collapse

### H3 — Context Scaling Envelope

* Goal: Max safe context per stack.
* Stack A: context increases until OOM or collapse
* Stack B: extended context via tiering

### H4A/B — QoS Under Background Storage Noise

**H4A — UMA-Only (Control)**

* Background fio should not affect p99.

**H4B — Tiered KV**

* Profile: Spill & Stress
* fio profile:
  * randrw, 70/30, 16k, iodepth 32, numjobs 4
* Expect:
  * p99 drift proportional to NVMe r_await
  * Stack B remains stable (no crash)

### H5 — KV Working-Set Scaling (Session Behavior)

* Simulates real traffic:
  * Mixed context sizes
  * Multi-turn chats
  * Bursty arrival patterns
* Outputs:
  * Hit ratios, spill volume, prefetch efficiency
  * p95/p99 drift under realistic load

### H6 — Tier Sizing & Policy Sensitivity

* Sweeps:
  * T0/T1 split at fixed total UMA KV
  * Total UMA KV at fixed ratio
  * Policy variants (LRU, LFU, guaranteed-no-evict)

### H7 — Telemetry Fidelity

* Sampling intervals: 100 ms, 200 ms, 500 ms, 1 s, 5 s
* Criteria:
  * <3% overhead
  * Ability to attribute tail events clearly

### H8 — Capability vs Resource Summary (“Hero Scenario”)

* Deliverables:
  * Side-by-side Stack A vs Stack B at equal UMA budgets
  * Show that Stack B can serve wider contexts or higher concurrency at similar QoS

### H9 — Context Re-Hydration

* Scenario:
  * Build session → let KV evict → resume session
* Metrics:
  * TTFT on the next turn
  * Tier2 sequential read bytes

7. Test Harness and Repository Structure
---------------------------------------

```
harness/
  configs/
    stackA_llama70b_baseline.yaml
    stackB_llama70b_dynamo_tiered.yaml
  src/
    loadgen.py
    workloads/
    telemetry/
      sysmon.sh
      nvme_iostat.py
      gpu_poll.py
      dynkv_ingest.py
  runners/
    run_H0_queue_knee.sh
    run_H1_lora_thrash.sh
    run_H2A_uma_pressure.sh
    run_H2B_dynamo_kv_pressure.sh
    run_H4A_storage_qos.sh
    run_H4B_dynamo_storage_qos.sh
    run_H5_workingset.sh
    run_H6_tier_sizing.sh
    run_H7_telemetry_sweep.sh
  analysis/
    process_results.py
    figures/
  runs/
```

8. Telemetry Schemas
--------------------

### 8.1 metrics.jsonl

```
{
  "ts": <float>,
  "stack": "A|B",
  "model": "L70B",
  "context_tokens": <int>,
  "concurrency": <int>,
  "lat_ttft_ms": <float>,
  "lat_e2e_ms": <float>,
  "rc": <int>
}
```

### 8.2 sysmon.jsonl

```
{
  "ts": <float>,
  "cpu": {"user": <...>, "system": <...>, "iowait": <...>},
  "mem": {"MemAvailable": <...>, "Cached": <...>},
  "gpu": {"util": <int>, "mem_used_bytes": <int>},
  "nvme": {
    "rps": <int>, "rMBs": <float>,
    "r_await_ms": <float>, "aqu_sz": <float>, "util_pct": <float>
  }
}
```

### 8.3 dynkv.jsonl

```
{
  "ts": <float>,
  "kv_block_size_bytes": 65536,
  "tier0": {...},
  "tier1": {...},
  "tier2": {
    "hits": <int>, "misses": <int>,
    "bytes_in": <int>, "bytes_out": <int>,
    "fetch_p50_ms": <float>,
    "fetch_p95_ms": <float>,
    "fetch_p99_ms": <float>
  },
  "evictions": <int>,
  "prefetches": <int>
}
```

Cadence: 200 ms (strict)

9. Analysis Checklist
---------------------

Each run produces the following figures:

* H0: p50/p95/p99 vs concurrency
* H2A vs H2B: latency vs context
* H2B: tier2 hit ratio vs context
* H4B: p99 vs NVMe r_await under fio
* H5: p99 vs working-set size
* H6: Knee vs tier capacity
* H7: Telemetry sampling cost
* H8: Hero scenario comparison
* H9: Re-hydration cost vs bytes loaded

10. Quality Gates and Acceptance Criteria
-----------------------------------------

* ≥95% aligned telemetry samples (sysmon + dynkv + metrics)
* CoV(p99) < 10% across repetitions
* Stack A H2A: NVMe r/s within ±10% of idle
* Stack B H2B: tier2 hits ≥ 20% near knee
* Reproducibility: p99 within ±5%

11. Runbooks
------------

### 11.1 Stack A Runbook

* Launch UMA-only TRT-LLM server
* Disable swap
* Set max_total_tokens
* Run H0 → H2A → H4A

### 11.2 Stack B Runbook

* Build TRT-LLM with Dynamo
* Allocate Tier2 directory (/nvme/kvbm/l70b/)
* Set tier capacities via env variables
* Validate dynkv telemetry (200 ms)
* Run H0B → H2B → H4B → H5–H9

12. Configuration Templates
---------------------------

### 12.1 YAML Template (Stack B)

```
dynamo:
  kv_block_size_bytes: 65536
  eviction_policy: "LRU"
  tier0: {capacity_bytes: 8GiB}
  tier1: {capacity_bytes: 24GiB}
  tier2: {path: "/nvme/kvbm/l70b", allocate_bytes: 512GiB}
telemetry:
  cadence_ms: 200
```

13. Environment Variables
-------------------------

Restoring full table from v3.0 and extending with v3.2 additions:

* STACKB_MAX_INPUT_LEN
* STACKB_MAX_SEQ_LEN
* STACKB_MAX_NUM_TOKENS
* DYN_KVBM_TIER0_BYTES
* DYN_KVBM_TIER1_BYTES
* DYN_KVBM_TIER2_BYTES
* DYN_KVBM_KV_BLOCK_SIZE_BYTES
* DYN_KVBM_CPU_CACHE_GB
* DYN_KVBM_DISK_CACHE_GB
* H4B_CONTEXT_TOKENS
* CONCURRENCY

14. Data Management
-------------------

* Each run stored under runs/YYYYMMDD_testid/
* Must include: configs, metrics.jsonl, sysmon.jsonl, dynkv.jsonl, figures, notes

15. Success Criteria
--------------------

* Stack B matches Stack A under Comfy
* Stack B extends collapse point significantly under Spill
* Tier2 fetch latency explains tail latency
* Re-hydration latency scales with NVMe sequential bandwidth

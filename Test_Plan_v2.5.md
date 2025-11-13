# DGX Spark Storage-Focused Inference Test Plan (v2.5)

**Target Platform:** NVIDIA DGX Spark — 128 GB coherent unified memory, 4 TB NVMe, GB10 Grace-Blackwell Superchip

---

## 1. Objective & Scope

This test plan defines a consistent methodology for evaluating **storage-driven inference performance** on UMA (Unified Memory Architecture) systems, using **DGX Spark** as the reference platform.

DGX Spark provides:

- **128 GB** coherent unified system memory (CPU+GPU, NVLink-C2C)  
- **1 PFLOP FP4** compute for large-model inference  
- **4 TB NVMe M.2** local storage  

These properties make Spark uniquely suited for experiments where model size, KV cache growth, and multi-tenant LoRA access patterns can induce **storage-backed paging**, enabling clean measurement of the latency contribution of **storage QoS**.

**New in Test Plan v2.5:**

- **H4 — Storage QoS Sensitivity:** explicitly vary NVMe QoS to quantify tail-latency sensitivity once paging begins.  
- **H5 — LoRA Working-Set Scaling:** quantify how adapter count and churn drive storage pressure and latency.

---

## 2. Hypotheses (v2.5)

| ID | Hypothesis | Purpose |
|----|------------|---------|
| **H0** | Queue saturation occurs before hardware saturation for some servers; identify usable concurrency (U_work). | Establish safe operating region for all tests. |
| **H1** | Cold vs warm LoRA adapters exhibit different I/O profiles; warming reduces TTFT and p99. | Establish LoRA storage footprint behavior. |
| **H2** | Increasing context or concurrency exceeds UMA headroom → KV paging → sharp rise in tail latency. | Identify the “UMA → paging knee.” |
| **H3** (Analytical) | UMA + swap counters correlate with onset of paging. | Validate OS metric interpretation. |
| **H4** | At fixed UMA pressure, degraded NVMe QoS causes predictable increases in TTFT/p99. | Isolate storage QoS sensitivity. |
| **H5** | Larger LoRA working-set & churn increase NVMe reads → degrade p95/p99. | Characterize multi-tenant LoRA scaling. |
| **H6** | High-entropy LoRA “storms” cause worst-case thrash vs. baseline. | Establish upper-bound behavior. |
| **H7** (Analytical) | NVMe telemetry scales with workload access patterns. | Validate storage-level measurements. |
| **H8** (Analytical) | CPU iowait% correlates with storage-bound stalls. | Integrate CPU profiling into conclusions. |

---

## 3. Platform Configuration

### System
- **NVIDIA DGX Spark (GB10 Grace-Blackwell)**  
- 128 GB LPDDR5x coherent unified memory  
- 4 TB M.2 NVMe  
- 1 PFLOP FP4 compute  

### Software Stack
- DGX OS + NVIDIA container stack  
- Triton (or similar inference server)  
- Harness + `loadgen.py` (patched for model routing + LoRA handling)

### Models
- **8B FP4** baseline  
- **70B FP4** (“hero” model, ~40 GiB effective) for UMA stress

### LoRA Adapters
- 2–64 adapters (`.safetensors` or `.pt`)  
- Stored on local NVMe for realistic access patterns  

---

## 4. Test Definitions

All tests use:

- **Load duration:** 300 seconds  
- **Repetitions:** 3 runs  
- **Client profile:** uniform or Poisson (per test)  
- **Telemetry Collected:**  
  - Latency metrics: p50/p95/p99 TTFT + E2E  
  - CPU: usr/sys/iowait%  
  - Memory: MemAvailable, Cached  
  - Storage: rMB/s, read IOPS, r_await, util% (iostat -x)  
  - NVMe counters: Host Read Commands, Data Units Read  

---

### 4.1 H0 — Queue Knee / Scheduler Saturation

**Goal:** Determine `U_queue_knee` and compute `U_work = 0.6–0.7 × U_queue_knee`.

**Procedure:**
- Sweep concurrency: **1 → 256**  
- Use **8B FP4** (avoid paging)  
- Track **p99** and queue depth vs. hardware utilization  

**Expected:** Sharp p99 rise **before** hardware saturation → safe operating point.

---

### 4.2 H1 — Cold vs Warm LoRA

**Goal:** Understand LoRA disk-load vs. cached behavior.

- Use **8B or 70B**  
- Test with **4–8 LoRAs**  
- Compare:  
  - **Cold:** first load from NVMe  
  - **Warm:** after OS cache fill  

---

### 4.3 H2 — UMA Pressure → Paging Knee

**Goal:** Identify context length `Ck` or concurrency `U` where paging begins.

**Procedure:**
- Fix concurrency = **U_work**  
- Sweep context length: C1 → C2 → C3 …  
- OR fix context and sweep concurrency  

**Expected:**  
- Latency **hockey stick** at paging onset  
- **MemAvailable↓**  
- **r_await↑**

---

### 4.4 H3 — UMA Counter Interpretation (Analytical)

Cross-reference:

- `MemAvailable_kB`  
- `Cached_kB`  
- `page-in/out` counters  
- `swapcache`  

Verify paging signals precede latency spike (H2).

---

### 4.5 H4 — Storage QoS Sensitivity *(New in v2.5)*

**Goal:** Prove inference latency becomes **storage-QoS-limited** when paging.

**Preconditions:**
- Use 70B FP4  
- Choose **Ck** just past H2 knee  
- Concurrency = **U_work**  
- **No LoRA**  

**Variants:**

1. **H4-A:** Baseline NVMe  
2. **H4-B:** Moderate contention  
   - `fio` read-heavy, QD8–16, ~40% NVMe util  
3. **H4-C:** Heavy contention  
   - Higher QD or cgroup IOPS/BW limits  

**Expected:**  
Higher **r_await** → higher **TTFT / p99** in predictable relationship.

---

### 4.6 H5 — LoRA Working-Set Scaling *(New in v2.5)*

**Goal:** Quantify how LoRA adapter count + churn increase storage pressure under paging.

**Preconditions:**
- Use **70B FP4**  
- Use paging regime from H2  

**Adapter set sizes:**
- **4 adapters** (S_small)  
- **16 adapters** (S_med)  
- **64 adapters** (S_large)  

**Churn types:**
- **Sticky sessions** — adapter reuse  
- **Stormy** — per-request random selection  

**Expected:**
- Host Read Commands ↑ with |S| and churn  
- p99 ↑ even with unchanged QoS  
- Scaling curve illuminates multi-tenant LoRA cost

---

### 4.7 H6 — LoRA Storm vs Baseline

**Goal:** Establish worst-case access patterns.

Compare:
- **No LoRA**  
- **High-entropy adapter switching**  

Uses **U_work** & **Ck** from H2.

---

### 4.8 H7 — NVMe Telemetry Interpretation (Analytical)

Map storage I/O to:
- Host Read Commands  
- Data Units Read  
- r_await  
- util%  

Validate H4/H5 results at the NVMe level.

---

### 4.9 H8 — CPU iowait Correlation (Analytical)

Show latency aligns with:
- pct_iowait  
- elevated r_await  
- MemAvailable collapse  

This ties CPU symptoms to storage stalls.

---

## 5. Scripts & Repository Structure

| Script | Purpose |
|--------|---------|
| `run_H0_queue_knee.sh` | Concurrency sweep |
| `run_H1_cold_warm_lora.sh` | LoRA cold/warm |
| `run_H2_uma_pressure.sh` | Paging knee discovery |
| `run_H4_storage_qos.sh` | NVMe QoS degradation |
| `run_H5_lora_scaling.sh` | LoRA working-set scaling |
| `run_H6_lora_storm.sh` | Baseline vs storm A/B |
| `process_results.py` | Unified summary + plots |

*(Ensure consistent naming in README.)*

---

## 6. Outputs for Whitepaper

v2.5 produces the following figures for direct inclusion in the DGX Spark whitepaper:

1. **Queue knee curve (H0)**  
2. **Cold vs warm storage TTFT (H1)**  
3. **UMA → paging hockey stick (H2)**  
4. **p99 vs storage QoS (H4)** — *primary storage impact plot*  
5. **p99 vs LoRA working-set size (H5)**  
6. **Baseline vs LoRA storm access patterns (H6)**  
7. **Host Read Commands correlation (H7)**  
8. **iowait ↔ tail latency correlation (H8)**  

Together, these form a cohesive narrative demonstrating that once DGX Spark’s UMA architecture is pushed into paging, **storage QoS becomes a first-class determinant of tail-latency behavior**—the core result this whitepaper aims to convey.

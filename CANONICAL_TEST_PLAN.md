# Canonical Test Plan and Project Goals

Below is a canonical, comprehensive plan aimed at producing whitepaper-grade evidence that **local SSD tiering + KV reuse improves runtime inference** on **DGX Spark (GB10)** with **Samsung PM9E1 4TB M.2 2242**, using **Dynamo + TRT-LLM + KVBM** as the serving stack. It is structured as two tracks:

* **Plumbing** (make the platform provably capable), and
* **Workload shaping** (make the loadgen compelling and high-signal).

This plan is written so that “done” is only one of two outcomes:

* **Conclusive proof we are blocked** from making the platform deliver/measure SSD value; or
* **Strong, high-value SSD signals** suitable for a vendor brief / technical whitepaper.

---

## 0) Target claim and framing

### Target claim style (your requirement)

We will demonstrate either:

1. **Improved latency at fixed feature set**, or
2. **Improved feature set at fixed latency**.

“Feature set” in near-edge/workstation terms means:

* higher concurrent interactive sessions at a given SLO,
* longer context window supported without tail-latency collapse,
* higher KV reuse effectiveness (prefix reuse, multi-turn continuity),
* greater stability (fewer stalls / fewer OOMs / fewer tail spikes).

### Why this aligns with NVIDIA’s “storage as part of the stack”

NVIDIA’s ICMS/ICMSP framing explicitly treats KV cache as latency-critical but ephemeral, and discusses a tiered hierarchy spanning GPU memory, host memory, local SSDs, and shared flash tiers, orchestrated by software like Dynamo + KV managers + NIXL. ([NVIDIA Developer][1])
That is the north star; we will start with **local NVMe on DGX Spark** and keep the plan structurally compatible with future shared storage phases.

---

## 1) Testbed definition

### Testbed baseline

* Platform profile: **single-node DGX Spark** workstation / near-edge deployment baseline.
* Memory and storage envelope: **128GB unified coherent memory + local 4TB NVMe**.
* Network expansion path: optional 2-node scaling and future shared-tier experiments via ConnectX/NIXL-compatible topology.
* Scope for this phase: **local SSD tier characterization only**; no cross-vendor SSD comparison.

### Hardware (fixed for this phase)

* System: NVIDIA DGX Spark (GB10 Grace-Blackwell), 128GB unified coherent memory, 4TB NVMe, ConnectX networking.
* SSD under test: Samsung PM9E1 4TB M.2 22×42, PCIe Gen5; Samsung positions it as complementing DGX Spark and quotes read/write capability. ([Samsung Semiconductor Global][2])
* No vendor comparisons in this phase (single-device characterization only).

### Models (canonical set)

We will keep today’s baseline and add pressure models deliberately:

* Baseline: **Llama-3.1-8B-Instruct FP8** (current). ([NVIDIA Build][3])
* Pressure model A: **Llama-3.3-70B-Instruct NVFP4** (bigger KV footprint; more likely to force tiering/reuse to matter). ([NVIDIA Build][3])
* Pressure model B: **GPT-OSS-120B MXFP4** (max pressure within single-node Spark-friendly list). ([NVIDIA Build][3])

This model list is taken from NVIDIA’s DGX Spark TRT-LLM “Model Support Matrix.” ([NVIDIA Build][3])

---

## 2) Canonical “Definition of Done”

### Done outcome A: Blocked (conclusive proof we cannot meet goals)

We declare “blocked” only if we can demonstrate one or more of the following, with artifacts:

1. **Control gap:** We cannot create a credible A/B where “SSD tier used” is the independent variable (cannot enable/disable disk tier, cannot limit it, or behavior is uncontrolled).
2. **Mechanism invisibility:** We cannot verify eviction/offload/rehydrate/reuse with sufficient fidelity using KVBM metrics + OS I/O.
3. **Attribution failure:** Disk I/O exists but cannot be attributed to KV cache activity (for example, dominated by unrelated processes or page cache artifacts).
4. **Reproducibility failure:** Effects are too noisy/unstable to support a decision-grade conclusion despite strict controls.
5. **Long-context enablement failure (Dataset L):** We cannot build/serve an engine configuration at ≥32k input tokens reliably enough to collect decision-grade data.

**Deliverable:** a blocker dossier with minimal repro steps, exact missing hooks, and what must change in the platform.

### Done outcome B: Strong SSD value signals (whitepaper-grade)

We declare success if we produce repeatable evidence that SSD-backed tiering/reuse improves:

* **Latency at fixed feature set** (e.g., meaningful p95/p99 TTFT or replay tail reduction), and/or
* **Feature set at fixed latency** (e.g., higher max concurrency or longer context meets the same SLO),

…and we can show a credible chain of evidence:

1. **Mechanism proof:** KVBM counters show offload/onboard/hit activity (deltas aligned to workload phases). ([NVIDIA Docs][4])
2. **OS corroboration:** NVMe read/write behavior correlates with those phases.
3. **User-visible impact:** Request-plane latency/throughput moves in the predicted direction.
4. **Repeatability:** Effects survive repeats and are robust against known confounders.

**Long-context gate (Dataset L):** At ≥32k inputs, SSD-assisted tiering must show (i) rehydrate/read activity during replay, (ii) matched-token reuse, and (iii) either improved tail latency on replay or increased max concurrency at fixed SLO versus the no-SSD baseline.

**Deliverable:** an “evidence pack” suitable to become (or feed) a technical brief.

---

# Track 1 — Plumbing work

Goal: ensure the serving stack is capable of producing controlled, attributable, reproducible evidence of SSD value.

## P1) Establish the “KV tiering control surface” (must-have)

KVBM is explicitly configurable via env vars for CPU and disk tiers, including disk-only experimental modes. ([NVIDIA Docs][4]) We will standardize three canonical modes:

### Mode A — “No SSD tier” baseline

* KVBM disk tier disabled
* optionally also a “no KVBM” baseline if needed (see P3)

### Mode B — “SSD tier enabled” (primary)

* GPU → CPU pinned → SSD (tiered offload)
* disk cache size controlled

### Mode C — “SSD tier enabled but constrained” (sensitivity)

* same SSD, but restrict disk cache size or apply throttling
* purpose: show monotonic relationship between storage capability and inference outcome

**Implementation requirement:** a single config/flag selects modes and writes the chosen mode into `manifest.json`.

Grounding in docs: disk+CPU tiering is configured by `DYN_KVBM_CPU_CACHE_GB` and `DYN_KVBM_DISK_CACHE_GB` (or override block counts). ([NVIDIA Docs][4])

---

## P2) Make disk-offload policy explicit (SSD endurance vs signal)

KVBM enables disk offload filtering by default to extend SSD lifespan, using a frequency-based policy; you can disable it via `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true`. ([NVIDIA Docs][4])

Canonical policy:

* **Realism runs:** keep filtering enabled (represent typical safe ops).
* **Mechanism stress runs:** disable filtering to guarantee SSD-tier activity (used only when we need hard proof of mechanism).

Every run must record:

* whether filtering was enabled,
* cache sizes and override counts,
* whether O_DIRECT was in play (and if disabled for troubleshooting).

Docs mention fallbacks and O_DIRECT disable as troubleshooting steps. ([NVIDIA Docs][4])

---

## P2.1) Long-context engine profile (≥32k) and build provenance (must-have for Dataset L)

Dataset L requires a serving engine built/configured for **≥32k** max input tokens. We define one or two canonical long-context targets and record build provenance.

Canonical targets:

* **Ctx32k (required):** ≥32k max input tokens
* **Ctx64k (optional):** ≥64k if supported and stable

Requirements:

* A named engine/profile for long context (e.g., `llama33_70b_nvfp4_ctx32k`) that selects the correct build.
* Manifest records:

  * `max_sequence_length` (or equivalent),
  * engine build identifier/hash,
  * container image + Dynamo + TRT-LLM versions,
  * any KV cache connector settings relevant to reuse/tiering.

Validity gate for Dataset L:

* Engine must load and serve a smoke test; if not, Dataset L is “blocked” and we capture a blocker dossier.

---

## P3) Baseline without KVBM (for credibility)

Even if the project centers on KVBM, whitepaper-grade work benefits from a clean “no KVBM” baseline.

NVIDIA’s KVBM guide includes explicit baseline guidance for TRT-LLM without KVBM (config without kv_connector_config; `enable_partial_reuse: false`). ([NVIDIA Docs][4])

Canonical baselines:

* **B0:** TRT-LLM without KVBM (pure serving baseline)
* **B1:** TRT-LLM + KVBM, CPU tier only
* **B2:** TRT-LLM + KVBM, CPU + SSD tier (target)
* (Optional) **B3:** disk-only (explicitly labeled experimental)

This creates clean comparisons:

* “Does KVBM help at all?”
* “Does SSD tier add incremental value beyond CPU tier?”

---

## P4) Mechanism observability: metrics + deltas per phase (must-have)

KVBM provides explicit counters and hit rates (matched tokens, offload blocks device→host, host→disk, onboard disk→device, hit rates). ([NVIDIA Docs][4])

Canonical rule: all key metrics are tracked as **phase deltas**, not raw values.

### Required KVBM metrics captured

At minimum (names from docs):

* `kvbm_matched_tokens`
* `kvbm_offload_blocks_d2h`
* `kvbm_offload_blocks_h2d` (host→disk)
* `kvbm_onboard_blocks_d2d` (disk→device)
* `kvbm_host_cache_hit_rate`
* `kvbm_disk_cache_hit_rate` ([NVIDIA Docs][4])

### Required phase boundaries

Every scenario is decomposed into phases with start/end snapshots:

* warmup
* pressure/thrash
* replay/rehydrate
* steady-state mixed load

Deliverables:

* `phase_<name>_kvbm_metrics_start.json`
* `phase_<name>_kvbm_metrics_end.json`
* `phase_<name>_kvbm_metrics_delta.json`

---

## P5) OS truth: storage measurement and attribution (must-have)

To claim “SSD value add,” we must show the SSD did real work at the right time.

Minimum OS instrumentation:

* `iostat -x` (device throughput, await, queue depth)
* per-process IO (`pidstat -d`, and/or cgroup IO stats if containerized)
* NVMe health snapshot (`nvme smart-log`, temps, media errors) pre/post run

Attribution upgrade (strongly recommended):

* file-level tracing of the KVBM disk store path (eBPF tools, opensnoop/filetop equivalents) so reviewers cannot dismiss IO as journaling noise.

Deliverables:

* `os_iostat.csv`
* `os_pid_io.csv` (or cgroup equivalent)
* `nvme_smart_pre.json`, `nvme_smart_post.json`
* `io_attribution_report.json` (even if partial)

---

## P6) Run validity gates (operator-grade)

A run is only decision-grade if it passes gates.

### Hard fail (run invalid)

* Serving not healthy / model not loaded
* high error rate above threshold
* metrics endpoints missing (KVBM metrics unreachable when SSD tier test selected)

### Soft fail (run valid but not decision-grade)

* disk tier configured but no disk IO observed
* disk IO observed but KVBM deltas do not corroborate
* excessive system noise detected (background jobs, thermal throttling indicators)

Deliverables:

* `verdict.json` with:

  * `run_valid: true/false`
  * `decision_grade: true/false`
  * `reason_codes: [...]`

---

## P7) One-command “phase bundle runner” (canonical UX)

A whitepaper effort dies if runs are ad-hoc. Canonical UX:

* `run_bundle --model llama8b_fp8 --workload W1 --mode B2 --repeats 10`

Output is always a self-describing bundle:

* `manifest.json`
* `summary.json`
* `quick_summary.json`
* `verdict.json`
* `metrics_delta/` (phase deltas)
* `os/` (iostat/pid/io attribution)
* `charts/`
* `raw/`

---

## P8) Scale-forward plumbing hooks (future phases, not executed now)

DGX Spark has networking intended for scaling/connecting systems. NVIDIA’s ICMS framing explicitly calls out a hierarchy including local SSDs and a shared flash context tier orchestrated by Dynamo/NIXL/KV managers. ([NVIDIA Developer][1])

So we bake in now:

* a storage tier abstraction in manifests (local_nvme now; shared_nvmeof later),
* networking telemetry capture stubs,
* the ability to swap the disk tier path/endpoint.

---

# Track 2 — Workload shaping

Goal: ensure loadgen is near-edge/workstation realistic and produces high-signal SSD/KV reuse behavior.

## W0) Workload principles (defensibility rules)

1. Interactive first: emphasize TTFT and tail latency (p95/p99), not just average TPS.
2. KV reuse opportunities must be forced intentionally: reuse does not happen accidentally.
3. Token distributions must be plausible: long prefixes + short outputs (common in RAG and instruction-following) are where TTFT reuse can matter.
4. Arrival patterns must be realistic: bursts, think time, concurrency variability.
5. Reproducible datasets: deterministic seeds, stable prompt sets, logged request manifests.

### W0.1 Dataset tiers (publishable structure)

We will collect two publishable datasets to reflect realistic workstation/near-edge usage as well as long-context scaling pressure:

* **Dataset S (Short/Interactive):** prompts and sessions with **≤8k** input tokens (typical interactive assistant + light RAG).
  Goal: demonstrate repeatable latency/capacity wins with clean A/B controls and strong mechanism proof.

* **Dataset L (Long-Context):** prompts with **≥32k** input tokens (heavy RAG summarization, long documents, large shared prefixes).
  Goal: amplify KV footprint and eviction frequency to make SSD tiering/reuse value clearer (tail latency and/or max concurrency at fixed SLO).

**Note:** Dataset L requires a **TRT-LLM engine rebuild** for the target max sequence length. Dataset S is the default baseline and must be completed first to lock methodology and avoid moving goalposts.

---

## W1) Canonical near-edge/workstation use cases

These map to what a Spark-class workstation is for: local development, privacy-sensitive/offline inference, and project-context assistants.

### Workload family A — “Local project copilot” (shared-prefix heavy)

* Very long shared prefix (team coding standards + repo context summary + system prompt)
* Many short user queries appended
* Expectation: high prefix reuse; SSD tier helps preserve reusable KV blocks under concurrency
* **Dataset S:** shared prefix fits ≤8k
* **Dataset L:** shared prefix ≥32k with short turns (optional but compelling)

### Workload family B — “Multi-turn agent loop” (session persistence)

* A single session lasts minutes, with tool-use style turns
* Think time between turns
* Expectation: session KV grows; eviction occurs; SSD tier helps keep continuity without recompute
* Primarily **Dataset S** (Dataset L optional)

### Workload family C — “Offline RAG summarization” (long prefill pressure)

* long retrieved documents (prefill-heavy), modest generation
* Expectation: massive KV footprint; SSD tiering makes “feature set at fixed latency” possible (longer docs, more concurrent summarizations)
* **Dataset L is canonical here** (≥32k)

### Workload family D — “Mixed local inference” (multi-tenant realism)

* blend of short chat, medium RAG, occasional long summarization
* Expectation: cache churn; SSD-tier usefulness shows up as improved tail stability and higher sustainable concurrency
* Primarily **Dataset S** (Dataset L optional)

---

## W2) Canonical “feature set” knobs

For each workload, evaluate two experimental styles:

### Style 1 — Fixed feature set → measure latency

Fix:

* context length distribution
* output length distribution
* concurrency / arrival pattern
* model

Measure:

* TTFT p50/p95/p99
* tokens/sec
* throughput at fixed error rate

Compare across modes (no SSD vs SSD tier).

### Style 2 — Fixed latency SLO → measure feature set

Fix SLO (example template; tune per model):

* TTFT p95 ≤ X ms
* TTFT p99 ≤ Y ms
* error rate ≤ Z%

Then search for max:

* concurrency
* sustainable QPS
* maximum context length supported

Compare “max feature set under SLO” across modes.

This directly satisfies your requirement (“improved feature set at fixed latency”).

---

## W3) KV reuse validation logic (must-have)

To claim reuse, we must prove the workload actually created reuse opportunities.

Canonical checks per run:

* request manifest includes a stable `prefix_hash` and `session_id`
* the loadgen reports:

  * reuse opportunity rate
  * observed reuse proxy from KVBM:

    * `kvbm_matched_tokens` increase
    * disk/host hit rates non-trivial ([NVIDIA Docs][4])

If “reuse opportunity high” but “matched_tokens ~0,” the run is automatically labeled not decision-grade (it was a bad reuse test).

---

## W4) “Make SSD matter” shaping (pressure profiles)

SSD value will not show if you never pressure KV capacity.

Each workload has at least one pressure profile:

* increase context length until GPU KV cache must evict
* increase concurrency until eviction is frequent
* enforce mixed churn to defeat purely in-GPU caching

Strict rule: pressure profiles must remain plausibly near-edge (no absurd QPS, no synthetic extremes that a workstation would never see).

---

## W5) Model scaling strategy (why 70B / 120B belongs)

DGX Spark is explicitly positioned for large models locally, and NVIDIA’s TRT-LLM Spark playbook lists supported quantized larger models (70B NVFP4; GPT-OSS-120B MXFP4). ([NVIDIA Build][3])

Canonical approach:

* Use 8B to debug harness correctness and reduce iteration time.
* Use 70B and 120B to create realistic KV pressure where tiering/reuse and local NVMe can plausibly deliver benefits.

---

# 3) Canonical experiment matrix (single-node phase)

This is the minimum that can produce strong signals without being bloated.

## Axis A — Serving mode (independent variable)

* **B0:** TRT-LLM no KVBM baseline ([NVIDIA Docs][4])
* **B1:** KVBM CPU tier only ([NVIDIA Docs][4])
* **B2:** KVBM CPU + SSD tier ([NVIDIA Docs][4])
* (Optional) **B2′:** B2 with disk filter disabled (mechanism stress) ([NVIDIA Docs][4])

## Axis B — Workloads (near-edge suite)

* **A:** Local project copilot (shared prefix)
* **B:** Multi-turn agent loop (session persistence)
* **C:** Offline RAG summarization (long prefill pressure)
* **D:** Mixed local inference (multi-tenant)

## Axis C — Models (progression)

* 8B FP8 → 70B NVFP4 → 120B MXFP4 ([NVIDIA Build][3])

## Axis D — Dataset tier (context length tier)

* **S:** ≤8k input tokens (no engine rebuild; primary baseline)
* **L:** ≥32k input tokens (engine rebuild required; captured in manifest)

### Decision-grade minimum set

**Dataset S (≤8k):**

* Run (A, B, C) on 8B across B0/B1/B2
* Run (A, C) on 70B across B1/B2

**Dataset L (≥32k):**

* Run (C) on 70B across B1/B2 at ≥32k
* (Optional) Run (C) on 120B across B1/B2 at ≥32k if feasible and stable on Spark
* (Optional) Run (A) in L-variant (≥32k shared prefix) across B1/B2 as a “developer copilot long-context” story

This gives:

* a correctness foundation,
* near-edge relevance,
* and at least one pressure regime where SSD tier should matter strongly.

---

# 4) Analysis outputs and whitepaper readiness

## Canonical claims we want the data to support

1. Mechanism: KVBM offload/onboard/hit metrics increase in the expected phases. ([NVIDIA Docs][4])
2. Storage involvement: NVMe reads/writes correlate with those phases.
3. User experience: TTFT tail improves or max concurrency improves at a fixed SLO.
4. Tradeoffs: identify regimes where SSD hurts (e.g., extra I/O overhead) vs helps.

## Canonical charts (repeatable templates)

* TTFT p95/p99 vs concurrency (per workload, per model)
* “Max concurrency under SLO” bar chart (feature set at fixed latency)
* KVBM metric deltas over time aligned to phases ([NVIDIA Docs][4])
* NVMe iostat timeline aligned to phases
* Correlation: disk await vs TTFT tail events

## Canonical report pack structure

* `TECHBRIEF_EVIDENCE.md` (method + claims + results)
* `METHOD.md` (reproducibility details, environment capture)
* `RESULTS/` (run bundles + summary rollups)
* `FIGURES/` (exported plots)

---

# 5) Known risks and how we neutralize them

1. “You benchmarked page cache, not SSD.”

   * Use direct I/O when possible; record O_DIRECT settings; corroborate via attribution. ([NVIDIA Docs][4])

2. “Disk writes are unrelated to KV.”

   * Per-process + file-level attribution required for decision-grade runs.

3. “Reuse did not actually happen.”

   * Require reuse opportunity accounting + `kvbm_matched_tokens` evidence. ([NVIDIA Docs][4])

4. “This does not generalize beyond a workstation.”

   * Tie methodology to ICMS-style tiering hierarchy and keep storage tier abstraction for future networked tests. ([NVIDIA Developer][1])

5. “Long-context results are a special-case build.”

   * Capture engine build provenance and max sequence length in manifest (P2.1), and treat Dataset L as a named dataset tier with explicit gates.

---

# 6) Immediate “start here” implementation order (fastest path to signal)

Shortest path to compelling results:

1. Finalize mode switches (B0/B1/B2) and ensure manifests record them. ([NVIDIA Docs][4])
2. Phase-based deltas for KVBM metrics + OS IO (required for mechanism proof). ([NVIDIA Docs][4])
3. Implement two workloads first:

   * Shared-prefix “local copilot” (reuse signal)
   * RAG summarization (pressure signal; becomes Dataset L anchor once engine supports ≥32k)
4. Add attribution (per-process minimum; file-level best-effort)
5. Run the 8B suite to validate harness; then escalate to 70B/120B for pressure. ([NVIDIA Build][3])
6. Produce the first decision-grade evidence bundle, then lock it as baseline.
7. Build Dataset L engine profile (P2.1) and run the minimal Dataset L matrix (C on 70B across B1/B2 at ≥32k).

---

## Notes specific to PM9E1 narrative positioning

Samsung explicitly positions PM9E1 as a compact, AI-optimized PCIe Gen5 4TB M.2 22×42 SSD that complements the GB10-powered DGX Spark, and quotes high sequential performance and power-efficiency claims. ([Samsung Semiconductor Global][2])
We should treat those as vendor assertions and focus our evidence on runtime inference outcomes (TTFT, tail latency, max concurrency/context under SLO) that directly match the project goal.

---

[1]: https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/ "Introducing NVIDIA BlueField-4-Powered Inference Context Memory Storage Platform for the Next Frontier of AI | NVIDIA Technical Blog"
[2]: https://semiconductor.samsung.com/news-events/tech-blog/samsung-pm9e1-inside-the-leading-ai-optimized-pcie-gen5-ssd/ "Samsung PM9E1: Inside the Leading AI-optimized PCIe Gen5 SSD | Samsung Semiconductor Global"
[3]: https://build.nvidia.com/spark/trt-llm "TRT LLM for Inference | DGX Spark"
[4]: https://docs.nvidia.com/dynamo/dev/components/kvbm/kvbm_guide.html "KVBM Guide — NVIDIA Dynamo Documentation"

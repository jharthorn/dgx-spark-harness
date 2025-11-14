# DGX Spark LLM Storage Test Harness (v2.5)

This harness is designed to test the performance of the NVIDIA DGX Spark's internal NVMe SSD under a memory-pressure-bound LLM inference workload, based on the canonical test plan **Test_Plan_v2.5.md**.

The goal is to quantify the impact of storage QoS on inference tail latency (p99) once the 128GB Unified Memory Architecture (UMA) is saturated and begins paging model data (KV cache or LoRA adapters) from the local 4TB NVMe drive.

This harness is configured to test the **Llama 3 8B (L8B)** and **70B (L70B)** models.

---

## Test Plan & Hypotheses (v2.5)

This harness executes the tests defined in **Test_Plan_v2.5.md**.

**H0 — Queue Knee Calibration:**  
Find the maximum stable concurrency (U_work) before queue delays dominate.

**H1 — Cold vs. Warm LoRA:**  
Measure p99 TTFT difference between cold OS cache vs. warm cache LoRA loads.

**H2 — UMA Pressure "Hockey Stick":**  
Identify the paging knee where KV cache paging to NVMe causes a sharp p99 rise.

**H3 (Analytical):**  
Correlate OS paging/swap metrics with the H2 latency knee.

**H4 — Storage QoS Sensitivity:**  
Primary goal. While paging, artificially degrade NVMe QoS using `fio`, proving that p99 latency scales with NVMe read latency (r_await).

**H5 — LoRA Working-Set Scaling:**  
Primary goal. Test how p99 latency and storage IOPS scale with LoRA adapter count (4, 16, 64) and session churn.

**H6 — LoRA Storm A/B Test:**  
Worst-case thrash scenario comparing baseline vs. LoRA-storm workloads.

**H7 (Analytical):**  
Correlate smartctl host read commands with LoRA access patterns.

**H8 (Analytical):**  
Correlate CPU iowait% with storage-bound tail latency.

---

## Harness Architecture

The architecture separates the **baseline (no-LoRA)** server from the **multi-adapter LoRA** server.

### Baseline Server (OpenAI API)

- Runs on **port 8355**
- Launched via `serve_llama33_70b_fp4.sh`
- Used for **H0, H2, H4**

### LoRA Server (Triton API)

- Runs on **port 8000**
- Launched via `launch_triton_server.sh`
- Used for **H1, H5, H6**

```
/harness/
├── src/
│   ├── loadgen.py          # Async load generator (Updated for H5)
│   └── sysmon.sh           # System telemetry collector
├── runs/
│   ├── _lib_quiescence.sh  # System prep (drop caches, etc.)
│   ├── model_env.sh        # Sets L8B/L70B paths
│   │
│   ├── run_H0_queue_knee.sh
│   ├── run_H1_coldwarm_lora.sh
│   ├── run_H2_uma_pressure.sh
│   ├── run_H4_storage_qos.sh    # NEW SCRIPT
│   ├── run_H5_lora_scaling.sh   # NEW SCRIPT
│   └── run_H6_workload_ab.sh
│
├── inputs/
│   ├── prompts/             # .txt files for different context lengths
│   └── lora_adapters/
│       ├── lora_list.txt        # 4 adapters (H1, H6, H5-Small)
│       ├── lora_list_16.txt     # NEW: 16 adapters (H5-Medium)
│       └── lora_list_64.txt     # NEW: 64 adapters (H5-Large)
│ 
├── analysis/
│   ├── backfill_summary.py  # Updated
│   ├── process_results.py   # Updated for H4, H5
│   ├── figures/             # (Git-ignored) Final plots
│   └── tables/              # (Git-ignored) Final tables
│
├── results/                 # (Git-ignored) Raw data
│
├── serve_llama33_70b_fp4.sh # Baseline server (OpenAI API @ 8355)
└── launch_triton_server.sh  # LoRA server (Triton API @ 8000)
```

---

## Load Generator (src/loadgen.py)

- Automatically chooses the baseline OpenAI API unless `--lora-list` is supplied or `--api-mode` overrides the selection.
- `--api-mode triton` lets you hit the Triton server even without LoRA adapters (used by H6 Baseline to keep both phases symmetric).
- `--lora-session random|sticky` controls adapter churn for H5.
- Summaries now capture additional telemetry placeholders (`io_wait_pct`, `r_await_ms`, `rps_storage`) that are backfilled by `analysis/backfill_summary.py`.

---

## Test Execution Workflow (v2.5)

### Step 1: Launch Harness Container

```bash
# (On the host - Terminal 1)
docker run --gpus all -it --rm \
  --network host \
  --shm-size=1g \
  --privileged --pid=host \
  -v /my-models:/workspace \
  -v ~/dgx_spark_harness:/harness \
  -v /sys:/sys \
  -v /proc:/proc \
  spark-harness:v1 \
  bash
```

---

## Step 2: Run Baseline (No-LoRA) Tests — H0, H2, H4

These tests target the baseline server on **port 8355**.

### A. Launch the Baseline 70B Server (Terminal 2)

```bash
# (On the host - Terminal 2)
cd ~/dgx_spark_harness

# Starts the server on port 8355
bash ./serve_llama33_70b_fp4.sh
```

Leave this running.

### B. Run the Baseline Tests (Terminal 1)

```bash
# (Inside Harness Container - Terminal 1)
cd /harness/runs

# 1. Calibrate queue knee (H0)
bash ./run_H0_queue_knee.sh L70B

# 2. Find the UMA paging knee (H2)
bash ./run_H2_uma_pressure.sh L70B

# 3. Run the Storage QoS test (H4)
# Requires the paging regime found in H2
bash ./run_H4_storage_qos.sh L70B
```

### C. Stop Baseline Server (Terminal 2)

Press **Ctrl+C**.

---

## Step 3: Run LoRA-Based Tests — H1, H5, H6

These use the Triton LoRA server on **port 8000**.

### A. Launch the 70B Triton LoRA Server (Terminal 2)

```bash
# (On the host - Terminal 2)
cd ~/dgx_spark_harness

bash ./launch_triton_server.sh L70B
```

Leave this running.  
The server must be restarted when prompted by H1 and H5 scripts.

### B. Run the LoRA Tests (Terminal 1)

```bash
# (Inside Harness Container - Terminal 1)
cd /harness/runs

# 1. LoRA Cold/Warm (H1)
# *** This script will ask you to restart the server in Terminal 2 ***
bash ./run_H1_coldwarm_lora.sh L70B

# 2. LoRA Storm A/B (H6)
bash ./run_H6_workload_ab.sh L70B

# 3. LoRA Scaling (H5)
# *** This script will ask for multiple server restarts ***
bash ./run_H5_lora_scaling.sh L70B
```

### C. Stop the LoRA Server (Terminal 2)

Press **Ctrl+C**.

---

## Step 4: Analyze All Results

```bash
# (Inside Harness Container - Terminal 1)
cd /harness/analysis

# Generate ALL plots for L70B
python3 ./process_results.py ALL
```

This will create all plots (H0, H1, H2, H4, H5, H6, etc.) in:

```
analysis/figures/L70B/
analysis/figures/tables/L70B/
```

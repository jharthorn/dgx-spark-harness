# DGX Spark LLM Storage Test Harness (v2.5 - Llama 3.x)

This harness is designed to test the performance of the NVIDIA DGX Spark's internal NVMe SSD under a memory-pressure-bound LLM inference workload, based on the v2.4 Test Plan.

It has been re-architected to test the Llama 3.x model family (8B and 70B) on a Triton Inference Server with full LoRA support.

---

## Harness Architecture (v2.5)

The architecture is parameterized to support multiple model engines.

```
/harness/
├── src/                # Core tools: sysmon.sh, loadgen.py (v2.3)
├── runs/               # Orchestration scripts (v2.5)
│   ├── model_env.sh    # NEW: Sets paths for L8B or L70B
│   ├── run_H0_...sh
│   ├── run_H1_...sh
│   ├── run_H2_...sh
│   └── run_H6_...sh
├── inputs/             # Static data: prompts/, lora_adapters/
├── results/            # (Git-ignored) RAW DATA
├── analysis/           # Analysis tools
│   └── figures/        # (Git-ignored) Final plots
│
├── setup_triton_server_llama.sh # NEW (v2.5): Builds Llama engine
└── launch_triton_server.sh      # NEW (v2.5): Launches Triton server
```

---

## Quick Start: The v2.5 Workflow

### Step 1: Launch Harness Container

In Terminal 1, launch your persistent `spark-harness:v1` container (which has sysstat, fio, pandas, etc. installed).

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

### Step 2: Build Engines

On the host (Terminal 2), you must build the TRT-LLM engines once for each model. This will take time.

```bash
# (On the host - Terminal 2)
cd ~/dgx_spark_harness

# Build the 8B engine
bash ./setup_triton_server_llama.sh L8B

# Build the 70B engine (this will take a while)
bash ./setup_triton_server_llama.sh L70B
```

---

### Step 3: Run the 8B (L8B) Test Suite

This is an “end-to-end” flow for testing the 8B model.

#### A. Launch the 8B Triton Server (Terminal 2)

```bash
# (On the host - Terminal 2)
cd ~/dgx_spark_harness
bash ./launch_triton_server.sh L8B
```

Leave this terminal running. You will see the Triton logs here.

#### B. Run the 8B Tests (Harness Container — Terminal 1)

```bash
# (Inside Harness Container - Terminal 1)
cd /harness/runs

# 1. Calibrate the 8B server
bash ./run_H0_queue_knee.sh L8B

# 2. Find the 8B UMA/storage bottleneck
bash ./run_H2_uma_pressure.sh L8B

# 3. Run the 8B LoRA A/B test
bash ./run_H6_workload_ab.sh L8B

# 4. Run the 8B LoRA Cold/Warm test
bash ./run_H1_coldwarm_lora.sh L8B
```

#### C. Stop the 8B Server (Terminal 2)

Press **Ctrl+C** in Terminal 2 to stop the Triton server.

---

### Step 4: Run the 70B (L70B) Test Suite

Now, repeat the process for the “hero” 70B model.

#### A. Launch the 70B Triton Server (Terminal 2)

```bash
# (On the host - Terminal 2)
cd ~/dgx_spark_harness
bash ./launch_triton_server.sh L70B
```

Leave this running.

#### B. Run the 70B Tests (Harness Container — Terminal 1)

```bash
# (Inside Harness Container - Terminal 1)
cd /harness/runs

bash ./run_H0_queue_knee.sh L70B
bash ./run_H2_uma_pressure.sh L70B
bash ./run_H6_workload_ab.sh L70B
bash ./run_H1_coldwarm_lora.sh L70B
```

#### C. Stop the 70B Server (Terminal 2)

Press **Ctrl+C**.

---

### Step 5: Analyze All Results

You have now collected all data for both models.

```bash
# (Inside Harness Container - Terminal 1)
cd /harness/analysis

# Run ALL analysis and generate ALL plots
python3 ./process_results.py ALL
```

This will create all plots (H0, H1, H2, H6, etc.) in `analysis/figures/`.

The `process_results.py` script (v2.3) is already smart enough to create **separate plots for L8B and L70B** data by reading the `model_tag` from the manifest.

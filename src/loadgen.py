#!/usr/bin/env python3
import argparse, asyncio, json, time, random, os
import sys
from pathlib import Path
import httpx

# v2.3: "Smart" Loadgen
# - Auto-detects H6 (LoRA) test and switches to Triton API (port 8000)
# - All other tests (H1, H2) default to OpenAI API (port 8355)

# --- API Endpoints ---
API_CONFIGS = {
    "openai": {
        "url": "http://127.0.0.1:8355/v1/completions",
        "doc": "OpenAI-compatible (trtllm-serve, default)"
    },
    "triton": {
        "url": "http://127.0.0.1:8000/v2/models/ensemble/generate_stream",
        "doc": "NVIDIA Triton (for LoRA support)"
    }
}
# ---

# Paths (container-absolute defaults)
HARNESS_DIR = Path(os.environ.get("HARNESS_DIR", "/harness"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(HARNESS_DIR / "results")))
INPUTS_DIR  = Path(os.environ.get("INPUTS_DIR",  str(HARNESS_DIR / "inputs")))
MODEL_HANDLE = os.environ.get("MODEL_HANDLE", "openai/gpt-oss-120b")

def percentile(xs, p):
    if not xs: return None
    xs_sorted = sorted(xs)
    k = (len(xs_sorted)-1) * (p/100.0)
    f = int(k); c = min(f+1, len(xs_sorted)-1)
    if f == c: return xs_sorted[f]
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return d0 + d1

def bootstrap_ci(xs, p=99, iters=10000, alpha=0.05):
    if not xs: return (None, None, None)
    import random
    n = len(xs); samples = []
    for _ in range(iters):
        s = [xs[random.randrange(n)] for _ in range(n)]
        samples.append(percentile(s, p))
    samples.sort()
    lo = samples[int((alpha/2)*iters)]
    hi = samples[int((1-alpha/2)*iters)]
    return (percentile(xs, p), lo, hi)

async def stream_request(client, url, payload, api_mode, timeout):
    """Sends one request, measures client-side TTFT and total latency."""
    t_send_end = time.time()
    ttft = None; total_latency = None; success = False
    
    headers = {"Content-Type": "application/json"}
    if api_mode == "triton":
        headers["Accept"] = "text/event-stream"
        
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout, headers=headers) as r:
            async for chunk in r.aiter_raw():
                if ttft is None: # First chunk received
                    ttft = (time.time() - t_send_end) * 1000.0
            
            # After stream ends
            total_latency = (time.time() - t_send_end) * 1000.0
            r.raise_for_status() # Raise HTTPStatusError for 4xx/5xx
            success = True
            
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        success = False
        
    return ttft, total_latency, success

def build_payload(args, prompt, adapter, api_mode):
    """Builds the JSON payload based on the API mode."""
    
    if api_mode == "triton":
        return {
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "stream": True,
            "lora_config": {
                "lora_name": adapter
            }
        }
    else: # openai
        return {
            "model": MODEL_HANDLE,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "stream": True
        }

async def worker(args, prompts, adapters, results_queue, api_mode):
    """A single concurrent worker task."""
    url = API_CONFIGS[api_mode]["url"]
    deadline = time.time() + args.duration
    timeout = httpx.Timeout(args.timeout)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        while time.time() < deadline:
            prompt = random.choice(prompts)
            adapter = random.choice(adapters) if adapters else None
            
            payload = build_payload(args, prompt, adapter, api_mode)

            ttft, lat, ok = await stream_request(client, url, payload, api_mode, timeout)
            await results_queue.put((time.time(), ttft, lat, ok))

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("-U","--users", type=int, required=True, help="Number of concurrent users")
    ap.add_argument("-P","--prompt-file", action="append", default=[], help="Path to prompt file(s). Relative to INPUTS_DIR or absolute.")
    ap.add_argument("--lora-list", default=None, help="Path to lora_list.txt. Using this flag *forces* Triton API mode.")
    ap.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    args = ap.parse_args()

    # --- v2.3: Auto-detect API mode ---
    api_mode = "openai"
    adapters = None
    if args.lora_list:
        api_mode = "triton" # H6 LoRA test MUST use Triton
        pth = Path(args.lora_list)
        if not pth.is_absolute():
            pth = Path(INPUTS_DIR) / pth
        if pth.exists():
            lines = pth.read_text().splitlines()
            adapters = [
                x.strip() for x in lines 
                if x.strip() and not x.strip().startswith("#")
            ]
        else:
            print(f"Error: LoRA list not found: {pth}", file=sys.stderr)
            return

        if not adapters:
            print(f"Error: --lora-list specified but no valid adapters found in {pth}", file=sys.stderr)
            return
    # ---------------------------------

    print(f"--- Starting Loadgen for Run ID: {args.run_id} ---")
    print(f"API Mode: {api_mode.upper()} (Target: {API_CONFIGS[api_mode]['url']})")

    # Input prompts
    prompts = []
    for p in args.prompt_file:
        pth = Path(p)
        if not pth.is_absolute():
            pth = Path(INPUTS_DIR) / p
        if not pth.exists():
            print(f"Warning: Prompt file not found: {pth}", file=sys.stderr)
            continue
        prompts.append(pth.read_text())
    if not prompts:
        prompts = ["Hello world."] # Default prompt

    results_queue = asyncio.Queue()
    tasks = [worker(args, prompts, adapters, results_queue, api_mode) for _ in range(args.users)]
    
    print(f"Starting load: {args.users} users for {args.duration}s...")
    await asyncio.gather(*tasks)
    print("Loadgen complete. Summarizing...")

    results = []
    while not results_queue.empty():
        results.append(results_queue.get_nowait())

    # --- Summaries ---
    ttfts = [r[1] for r in results if r[1] is not None]
    lats  = [r[2] for r in results if r[3] and r[2] is not None] # Only successful, completed latencies
    total_reqs = len(results)
    success_reqs = len(lats)
    admission = 100.0 * (success_reqs / max(1, total_reqs))
    rps = success_reqs / max(1.0, args.duration)

    p99, ci_lo, ci_hi = bootstrap_ci(lats, 99)
    summary = {
      "requests_total": total_reqs,
      "admission_pct": admission,
      "throughput_rps": rps,
      "ttft_ms": {"p50": percentile(ttfts,50), "p95": percentile(ttfts,95), "p99": percentile(ttfts,99)},
      "latency_ms": {"p50": percentile(lats,50), "p90": percentile(lats,90), "p95": percentile(lats,95),
                     "p99": p99, "p99_ci_low": ci_lo, "p99_ci_high": ci_hi},
      "avg": {"io_wait_pct": None, "qu_sz": None, "rps": rps, "gpu_util_pct": None} # Populated by analysis script
    }

    out_dir = Path(RESULTS_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_id}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary: {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
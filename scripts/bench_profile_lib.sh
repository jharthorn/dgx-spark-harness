#!/usr/bin/env bash

bench_resolve_tier_mode() {
  local raw="${1:-${BENCH_TIER_MODE:-${BENCH_KV_MODE:-}}}"
  local normalized="${raw,,}"
  case "${normalized}" in
    ""|b2|cpu_disk|cpu-disk|cpu+disk)
      BENCH_TIER_MODE_RESOLVED="B2"
      BENCH_KV_MODE_RESOLVED="cpu_disk"
      ;;
    b1|cpu_only|cpu-only|cpu)
      BENCH_TIER_MODE_RESOLVED="B1"
      BENCH_KV_MODE_RESOLVED="cpu_only"
      ;;
    b0|off|none|no_kvbm)
      BENCH_TIER_MODE_RESOLVED="B0"
      BENCH_KV_MODE_RESOLVED="off"
      ;;
    *)
      echo "Unsupported tier/mode value: ${raw} (expected B0/B1/B2 or off/cpu_only/cpu_disk)" >&2
      return 1
      ;;
  esac
  export BENCH_TIER_MODE_RESOLVED BENCH_KV_MODE_RESOLVED
}

bench_defaults_for_tier_mode() {
  local mode="${1:-${BENCH_TIER_MODE_RESOLVED:-B2}}"
  case "${mode}" in
    B0)
      BENCH_CPU_CACHE_GB_DEFAULT="0"
      BENCH_DISK_CACHE_GB_DEFAULT="0"
      BENCH_KVBM_METRICS_DEFAULT="false"
      ;;
    B1)
      BENCH_CPU_CACHE_GB_DEFAULT="8"
      BENCH_DISK_CACHE_GB_DEFAULT="0"
      BENCH_KVBM_METRICS_DEFAULT="true"
      ;;
    B2)
      BENCH_CPU_CACHE_GB_DEFAULT="8"
      BENCH_DISK_CACHE_GB_DEFAULT="32"
      BENCH_KVBM_METRICS_DEFAULT="true"
      ;;
    *)
      echo "Unsupported tier mode for defaults: ${mode}" >&2
      return 1
      ;;
  esac
  export BENCH_CPU_CACHE_GB_DEFAULT BENCH_DISK_CACHE_GB_DEFAULT BENCH_KVBM_METRICS_DEFAULT
}

bench_resolve_model_profile() {
  local raw="${1:-${BENCH_MODEL_PROFILE:-llama31_8b_fp8}}"
  local normalized="${raw,,}"
  case "${normalized}" in
    ""|llama31_8b_fp8|8b|baseline)
      BENCH_MODEL_PROFILE_RESOLVED="llama31_8b_fp8"
      BENCH_MODEL_HANDLE_DEFAULT="nvidia/Llama-3.1-8B-Instruct-FP8"
      BENCH_MODEL_NAME_DEFAULT="nvidia/Llama-3.1-8B-Instruct-FP8"
      ;;
    llama33_70b_nvfp4|70b|70b_nvfp4|pressure_70b)
      BENCH_MODEL_PROFILE_RESOLVED="llama33_70b_nvfp4"
      BENCH_MODEL_HANDLE_DEFAULT="nvidia/Llama-3.3-70B-Instruct-NVFP4"
      BENCH_MODEL_NAME_DEFAULT="nvidia/Llama-3.3-70B-Instruct-NVFP4"
      ;;
    gpt_oss_120b_mxfp4|120b|120b_mxfp4)
      BENCH_MODEL_PROFILE_RESOLVED="gpt_oss_120b_mxfp4"
      BENCH_MODEL_HANDLE_DEFAULT="openai/gpt-oss-120b-mxfp4"
      BENCH_MODEL_NAME_DEFAULT="openai/gpt-oss-120b-mxfp4"
      ;;
    *)
      echo "Unsupported BENCH_MODEL_PROFILE=${raw}" >&2
      return 1
      ;;
  esac
  export BENCH_MODEL_PROFILE_RESOLVED BENCH_MODEL_HANDLE_DEFAULT BENCH_MODEL_NAME_DEFAULT
}

bench_model_handle_to_snapshot_glob() {
  local handle="${1:?model handle is required}"
  local cache_key
  cache_key="${handle//\//--}"
  printf '/root/.cache/huggingface/hub/models--%s/snapshots/*\n' "${cache_key}"
}

bench_resolve_model_env() {
  bench_resolve_model_profile "${1:-${BENCH_MODEL_PROFILE:-}}"
  MODEL_HANDLE="${MODEL_HANDLE:-${BENCH_MODEL_HANDLE_DEFAULT}}"
  MODEL_NAME="${MODEL_NAME:-${BENCH_MODEL_NAME_DEFAULT}}"
  MODEL_SNAPSHOT_GLOB="${MODEL_SNAPSHOT_GLOB:-$(bench_model_handle_to_snapshot_glob "${MODEL_HANDLE}")}"
  export MODEL_HANDLE MODEL_NAME MODEL_SNAPSHOT_GLOB
}

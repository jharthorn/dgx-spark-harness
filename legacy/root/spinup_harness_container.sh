#!/bin/bash
set -e

MODEL_DIR=${1:-/my-models}
HARNESS_DIR=${2:-$HOME/dgx_spark_harness}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}

# Populate HF_TOKEN from host file if not already set.
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/hftoken.txt" ]]; then
  HF_TOKEN="$(<"$HOME/hftoken.txt")"
fi

echo "Using model directory: $MODEL_DIR"
echo "Using harness directory: $HARNESS_DIR"
mkdir -p "$HF_CACHE"

docker run --gpus all -it --rm \
  --network host \
  --shm-size=1g \
  --privileged --pid=host \
  -v ${MODEL_DIR}:/workspace \
  -v ${HARNESS_DIR}:/harness \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  -v /sys:/sys \
  -v /proc:/proc \
  -w /harness \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  spark-harness:v1 \
  bash

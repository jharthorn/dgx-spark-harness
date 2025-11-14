#!/bin/bash
set -e

MODEL_DIR=${1:-/my-models}
HARNESS_DIR=${2:-$HOME/dgx_spark_harness}

echo "Using model directory: $MODEL_DIR"
echo "Using harness directory: $HARNESS_DIR"

docker run --gpus all -it --rm \
  --network host \
  --shm-size=1g \
  --privileged --pid=host \
  -v ${MODEL_DIR}:/workspace \
  -v ${HARNESS_DIR}:/harness \
  -v /sys:/sys \
  -v /proc:/proc \
  spark-harness:v1 \
  bash

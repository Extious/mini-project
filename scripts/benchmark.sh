#!/usr/bin/env bash
set -euo pipefail

python -m src.cli \
  --model resnet50 \
  --dataset synthetic \
  --mode sequential \
  --batch-size 32 \
  --num-samples 2048 \
  --precision fp32 \
  --metrics-path data/sequential.json

python -m src.cli \
  --model resnet50 \
  --dataset synthetic \
  --mode batched \
  --batch-size 256 \
  --num-samples 4096 \
  --precision fp16 \
  --metrics-path data/batched.json

if command -v torchrun >/dev/null 2>&1; then
  torchrun --nproc_per_node=2 -m src.cli \
    --model resnet50 \
    --dataset synthetic \
    --mode distributed \
    --batch-size 128 \
    --num-samples 8192 \
    --precision fp16 \
    --metrics-path data/distributed.json
fi

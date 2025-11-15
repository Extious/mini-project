#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-"data/.cache/huggingface"}
mkdir -p data metrics data/profiler

python -m src.cli \
  --model distilgpt2 \
  --dataset synthetic \
  --mode sequential \
  --batch-size 8 \
  --num-samples 256 \
  --max-length 256 \
  --metrics-path data/sequential_prefill.json

python -m src.cli \
  --model distilgpt2 \
  --dataset curated \
  --mode batched \
  --batch-size 64 \
  --num-samples 1024 \
  --precision fp16 \
  --max-length 512 \
  --metrics-path data/batched_fp16.json

python -m src.cli \
  --model gpt2 \
  --dataset synthetic \
  --mode batched \
  --batch-size 32 \
  --num-samples 512 \
  --precision fp16 \
  --profile \
  --profile-dir data/profiler/gpt2_fp16 \
  --metrics-path data/gpt2_fp16_profiled.json

if command -v torchrun >/dev/null 2>&1; then
  torchrun --nproc_per_node=2 -m src.cli \
    --model gpt2 \
    --dataset synthetic \
    --mode distributed \
    --batch-size 64 \
    --num-samples 2048 \
    --precision fp16 \
    --profile \
    --profile-dir data/profiler/ddp_fp16 \
    --metrics-path data/distributed_fp16.json
fi

# GPU Language Model Inference Mini-Project

## Overview
- Target workload: Hugging Face causal language models (e.g., `distilgpt2`, `gpt2`, `gpt2-medium`) under multi-GPU inference.
- Parallel strategies: sequential micro-batching, large-batch single GPU, `torch.nn.DataParallel`, and `torchrun`-based DistributedDataParallel.
- Performance levers: mixed precision, CUDA graph capture (single-rank), prompt batching, and `torch.profiler` guided tuning.
- Outputs: JSON metrics (samples/s, tokens/s, latency stats), optional TensorBoard traces, and a structured report template.

## Repository Layout
- `src/`: Python package for dataset synthesis, Hugging Face model loading, execution loop, and profiling utilities.
- `scripts/`: Ready-to-run shell scripts for baseline sweeps and profiler demonstrations.
- `slurm/`: Example SLURM submission script using `torchrun` on multi-GPU nodes.
- `report/`: Markdown report template plus artifact directory for plots/traces.
- `data/`: Cache for downloaded tokenizers/models, metrics JSON, and profiler outputs (ignored by git).

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify GPU visibility:
```bash
python - <<'PY'
import torch
print("cuda?", torch.cuda.is_available())
print("gpus", torch.cuda.device_count())
PY
```

## Running Benchmarks
All experiments use `python -m src.cli` with the following key arguments:
- `--model`: `distilgpt2`, `gpt2`, or `gpt2-medium`.
- `--dataset`: `synthetic` (randomized prompt permutations) or `curated` (fixed research prompts).
- `--mode`: `sequential`, `batched`, `dataparallel`, or `distributed`.
- `--batch-size`: Prompts per step; sequential mode automatically micro-batches to size 1.
- `--max-length`: Prompt token length after padding (default 256).
- `--precision`: `fp32`, `fp16`, or `bf16`.
- `--profile`: Enable `torch.profiler`; combine with `--profile-dir` for TensorBoard traces.

### Example commands
```bash
# Single-GPU latency baseline (prefill only)
python -m src.cli --model distilgpt2 --dataset synthetic --mode sequential --batch-size 8 --num-samples 256

# Batched throughput with fp16
python -m src.cli --model distilgpt2 --dataset curated --mode batched --batch-size 64 --precision fp16 --max-length 512 --num-samples 1024

# Two-GPU DDP run with profiling enabled
torchrun --nproc_per_node=2 -m src.cli \
  --model gpt2 \
  --dataset synthetic \
  --mode distributed \
  --batch-size 64 \
  --precision fp16 \
  --profile \
  --profile-dir data/profiler/ddp \
  --metrics-path data/ddp_fp16.json
```

## Torch Profiler Workflow
`src.cli` exposes three knobs:
- `--profile`: turn profiling on (disabled by default).
- `--profile-dir`: optional TensorBoard log directory (`data/profiler/...` recommended).
- `--profile-wait`, `--profile-warmup`, `--profile-active`: schedule tuning for steady-state capture.

Launch TensorBoard after a profiled run:
```bash
tensorboard --logdir data/profiler --bind_all
```

## SLURM Execution
`slurm/run_inference.sbatch` demonstrates:
- Module/env activation (conda + CUDA).
- Resource requests for one node with four GPUs.
- `torchrun` launch plus profiler directory selection.

Customize partition, account, `module load` statements, and storage paths to match your cluster. Remember the course policy: compile LaTeX reports with `xelatex`.

## Reporting
- Fill `report/report.md` with background, experimental matrix, profiler screenshots, and tables of throughput vs. hardware.
- Place large assets (TensorBoard screenshots, Nsight traces) under `report/artifacts/`.
- Keep raw metrics JSON (from `--metrics-path`) under `data/` for reproducibility.

# GPU Inference Parallelism Mini-Project

## Overview
This mini-project targets optimizing deep learning inference workloads on GPUs. It demonstrates how to:
- Benchmark single-GPU, batched, and multi-GPU inference pipelines.
- Apply practical optimization levers such as mixed precision, CUDA graph capture, and asynchronous data loading.
- Scale runs on a cluster with SLURM using a torchrun-based launcher.
- Capture performance telemetry for later analysis and reporting.

## Repository Layout
- `src/`: Core Python modules for data preparation, model loading, inference pipelines, and performance accounting.
- `scripts/`: Convenience shell scripts to replicate benchmark sweeps.
- `slurm/`: Example SLURM submission scripts for cluster execution.
- `report/`: Markdown template capturing the full project report, including background, methodology, and results tables.
- `data/`: Placeholder directory for cached datasets or exported metrics.

## Quickstart
1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Verify GPU availability (optional)**
   ```bash
   python - <<'PY'
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   PY
   ```

## Running Benchmarks
All experiments are driven by `python -m src.cli`. Key arguments:
- `--model`: Backbone to load (`resnet50`, `efficientnet_b0`, `vit_b_16`).
- `--dataset`: `synthetic` (random tensors) or `cifar10`.
- `--mode`: `sequential`, `batched`, `dataparallel`, or `distributed`.
- `--batch-size`: Micro-batch per device.
- `--num-samples`: Total samples to process for the run.
- `--precision`: `fp32`, `fp16`, or `bf16`.
- `--capture-graph`: Enable CUDA graph replay to reduce launch overhead.

### Examples
```bash
# Baseline single-GPU FP32 run
python -m src.cli --model resnet50 --dataset synthetic --mode sequential --batch-size 32 --num-samples 2048

# DataParallel across all visible GPUs with mixed precision
python -m src.cli --model resnet50 --dataset cifar10 --mode dataparallel --batch-size 128 --precision fp16

# Distributed execution (2 GPUs) using torchrun
torchrun --nproc_per_node=2 -m src.cli --model vit_b_16 --mode distributed --dataset synthetic --num-samples 4096
```

## SLURM Usage
The `slurm/run_inference.sbatch` script illustrates:
- Module/environment setup.
- Resource requests for multi-GPU nodes.
- Launching `torchrun` with node-local ranks and aggregated metrics.

Adapt node counts, partition names, and paths to your cluster.

## Reporting
Populate `report/report.md` once experiments finish. Document:
- Problem definition and datasets.
- Parallel strategy (e.g., DataParallel vs. DistributedDataParallel) and optimization toggles.
- Hardware configurations tried and their outcomes.
- Tables/plots summarizing throughput, latency, utilization, and scaling behavior.

Add screenshots or profiler exports under `report/artifacts/` if needed.

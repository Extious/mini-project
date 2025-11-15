# GPU Inference Parallelization Report

## 1. Problem Definition
- **Research goal:** Reduce latency and improve throughput of pretrained vision models during batch inference on GPU clusters.
- **Baseline workload:** Torchvision `resnet50` evaluated on synthetic ImageNet-sized tensors (optionally CIFAR10) with 2,048 samples.
- **Metrics:** Samples per second (throughput), mean latency per batch, GPU memory footprint.

## 2. System Design
- **Data pipeline:** Configurable datasets via `src/data.py` with synthetic tensors or CIFAR10; deterministic sharding for distributed ranks.
- **Model stack:** Torchvision backbones (`resnet50`, `efficientnet_b0`, `vit_b_16`) loaded through `src/model_loader.py`, with optional TorchScript tracing.
- **Execution modes:** Sequential, batched, `torch.nn.DataParallel`, and DistributedDataParallel (torchrun) plus optional CUDA graph capture.
- **Optimization levers:** Mixed precision (`fp16`, `bf16`), CUDA graph replay, warm-up trimming, asynchronous DataLoader workers, SLURM job orchestration.

## 3. Implementation Summary
- `src/cli.py` serves as the experiment driver with CLI flags for all toggles.
- `scripts/benchmark.sh` showcases a reproducible sweep (sequential vs batched vs distributed).
- `slurm/run_inference.sbatch` enables cluster submission with four GPUs per node.
- Metrics persist to JSON (`--metrics-path`) for downstream visualization (e.g., pandas, matplotlib).

## 4. Experiment Plan
| Run ID | Mode | GPUs | Batch Size | Precision | CUDA Graph | Notes |
|--------|------|------|------------|-----------|------------|-------|
| baseline | sequential | 1 | 32 | fp32 | off | Reference latency. |
| batched | batched | 1 | 256 | fp16 | off | Demonstrates batching gains + mixed precision. |
| dp | dataparallel | 4 | 128 | fp16 | off | Highlights multi-GPU data parallel scaling. |
| ddp-graph | distributed | 4 | 128 | fp16 | on | Uses torchrun + CUDA graphs to cut launch overhead. |

## 5. Sample Results (fill with real data)
| Run ID | Samples/s | Avg Latency (s) | P95 Latency (s) | Max GPU Memory (MB) |
|--------|-----------|-----------------|-----------------|---------------------|
| baseline | TBD | TBD | TBD | TBD |
| batched | TBD | TBD | TBD | TBD |
| dp | TBD | TBD | TBD | TBD |
| ddp-graph | TBD | TBD | TBD | TBD |

Add profiler screenshots or Nsight timelines to `report/artifacts/`.

## 6. Analysis Template
1. Compare throughput scaling efficiency between single GPU and DDP runs.
2. Quantify benefits of mixed precision on both throughput and memory.
3. Discuss CUDA graph capture impacts on small batch scenarios.
4. Identify remaining bottlenecks (data loading, kernel launch, comms) and propose next steps (e.g., TensorRT conversion, pipeline parallelism).

## 7. Reproducibility Checklist
- [ ] Python environment recreated via `requirements.txt`.
- [ ] Dataset availability documented.
- [ ] CLI commands (or SLURM scripts) recorded for each result row.
- [ ] Metrics JSON artifacts versioned under `data/`.
- [ ] Random seeds logged (`--seed` flag).

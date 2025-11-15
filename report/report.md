# GPU LLM Inference Report Template

## 1. Problem Definition
- **Research focus:** Accelerating autoregressive language model inference for interactive assistants running on multi-GPU clusters.
- **Models:** Hugging Face `distilgpt2`, `gpt2`, `gpt2-medium`.
- **Workload:** Prompt prefill only (no generation) over 256–512 token sequences, comparing sequential vs. batched vs. multi-GPU execution.
- **Metrics:** Samples per second, tokens per second, batch latency (avg/p95), max GPU memory, profiler-derived hotspots.

## 2. System Architecture
- **Data pipeline (`src/data.py`):** Synthetic or curated prompts padded to a fixed `max_length`, deterministic sharding per rank.
- **Execution driver (`src/cli.py`):** CLI toggles for precision, CUDA graphs, DistributedDataParallel, and `torch.profiler` schedule.
- **Profiling (`src/profiler.py`):** Collects latency summaries and optional TensorBoard traces (CPU+CUDA activities, stack traces, memory).
- **Automation:** `scripts/benchmark.sh` for local sweeps; `slurm/run_inference.sbatch` for cluster jobs with profiler outputs.

## 3. Experiment Matrix (example)
| Run ID | Model | Mode | GPUs | Batch | Precision | Max Length | Profile? | Notes |
|--------|-------|------|------|-------|-----------|------------|----------|-------|
| seq-fp32 | distilgpt2 | sequential | 1 | 8 | fp32 | 256 | no | Latency baseline |
| batched-fp16 | distilgpt2 | batched | 1 | 64 | fp16 | 512 | no | Throughput focus |
| dp-fp16 | gpt2 | dataparallel | 4 | 64 | fp16 | 256 | no | Scale-up with NCCL |
| ddp-prof | gpt2-medium | distributed | 4 | 96 | fp16 | 512 | yes | Trace analysis |

## 4. Results Table (fill with actual numbers)
| Run ID | Samples/s | Tokens/s | Avg Latency (s) | P95 Latency (s) | Max GPU Mem (MB) |
|--------|-----------|----------|-----------------|-----------------|------------------|
| seq-fp32 | TBD | TBD | TBD | TBD | TBD |
| batched-fp16 | TBD | TBD | TBD | TBD | TBD |
| dp-fp16 | TBD | TBD | TBD | TBD | TBD |
| ddp-prof | TBD | TBD | TBD | TBD | TBD |

Store supporting JSON metrics under `data/` (same names as `Run ID`) for reproducibility.

## 5. Profiler Findings
1. Attach TensorBoard screenshots (`report/artifacts/`) showing kernel time distribution and memory usage.
2. Summarize key bottlenecks (e.g., attention kernels, embedding lookups, NCCL all-reduce).
3. Link optimizations to evidence (e.g., CUDA graph reduced launch overhead by X% according to profiler traces).

## 6. Analysis Checklist
- Compare scaling efficiency: `(tokens/s) / (#GPUs)` across sequential, batched, DDP runs.
- Quantify precision impact: fp32 vs. fp16 throughput, latency, and memory.
- Evaluate CUDA graph benefits for steady micro-batches.
- Identify remaining issues (tokenizer speed, DataLoader overhead, communication hotspots) and propose next steps (KV cache reuse, tensor parallelism, quantization).

## 7. Reproducibility
- [ ] Record exact CLI invocations (or SLURM scripts) for every table entry.
- [ ] Track environment info (PyTorch, transformers, CUDA, driver).
- [ ] Preserve raw profiler directories (`data/profiler/...`) and mention how to open them with TensorBoard.
- [ ] Document random seeds and prompt sources.

> Reminder: compile the final PDF with `xelatex` per course policy.

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch
from torch import distributed as dist

from .data import DataConfig, build_dataloader, build_dataset, shard_dataset
from .model_loader import load_model
from .profiler import MetricsBuffer, gather_gpu_stats

MODEL_NAMES = ("resnet50", "efficientnet_b0", "vit_b_16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU inference benchmarking toolkit")
    parser.add_argument("--model", type=str, default="resnet50", choices=MODEL_NAMES)
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "cifar10"])
    parser.add_argument("--mode", type=str, default="sequential", choices=["sequential", "batched", "dataparallel", "distributed"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--capture-graph", action="store_true")
    parser.add_argument("--use-torchscript", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def init_distributed_if_needed() -> Dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return {"world_size": world_size, "rank": rank, "local_rank": local_rank}


class GraphRunner:
    def __init__(self, model, dtype):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graph capture requires a CUDA device")
        self.model = model
        self.dtype = dtype
        self.graph = None
        self.static_batch = None
        self.static_output = None

    def __call__(self, batch):
        if self.graph is None:
            self.static_batch = torch.empty_like(batch)
            self.static_output = None
            self.graph = torch.cuda.CUDAGraph()
            self.static_batch.copy_(batch)
            autocast_cm = torch.cuda.amp.autocast(dtype=self.dtype) if self.dtype else nullcontext()
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph):
                with autocast_cm:
                    self.static_output = self.model(self.static_batch)
            torch.cuda.synchronize()
        elif batch.shape != self.static_batch.shape:
            return self.model(batch)
        else:
            self.static_batch.copy_(batch)
        self.graph.replay()
        return self.static_output


def prepare_model(args, device, dist_state):
    model = load_model(
        name=args.model,
        precision=args.precision,
        device=device,
        image_shape=(3, 224, 224),
        use_torchscript=args.use_torchscript,
    )
    if args.mode == "dataparallel":
        if torch.cuda.device_count() < 2:
            raise RuntimeError("DataParallel mode requires at least 2 CUDA devices")
        model = torch.nn.DataParallel(model)
    elif args.mode == "distributed":
        if not dist.is_initialized():
            raise RuntimeError("Distributed mode requires torch.distributed initialization")
        device_ids = [device.index] if device.type == "cuda" else None
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids, output_device=device.index)
    return model


def run_loop(args, model, dataloader, device) -> Dict[str, Any]:
    total_processed = 0
    metrics = MetricsBuffer()
    autocast_dtype = None
    if args.precision == "fp16":
        autocast_dtype = torch.float16
    elif args.precision == "bf16":
        autocast_dtype = torch.bfloat16
    amp_context = torch.cuda.amp.autocast(dtype=autocast_dtype) if (autocast_dtype and torch.cuda.is_available()) else nullcontext()
    graph_runner = GraphRunner(model, autocast_dtype) if args.capture_graph else None
    warmup_left = args.warmup_steps
    with torch.no_grad():
        for step, (inputs, _) in enumerate(dataloader):
            if total_processed >= args.num_samples:
                break
            inputs = inputs.to(device, non_blocking=True)
            current_batch = inputs.shape[0]
            if args.mode == "sequential":
                inputs = inputs.chunk(current_batch, dim=0)
            else:
                inputs = [inputs]
            for chunk in inputs:
                if total_processed >= args.num_samples:
                    break
                use_cuda = torch.cuda.is_available()
                if use_cuda:
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    start = end = None
                with amp_context:
                    if graph_runner:
                        _ = graph_runner(chunk)
                    else:
                        _ = model(chunk)
                if end is not None:
                    end.record()
                    torch.cuda.synchronize()
                    duration = start.elapsed_time(end) / 1000.0
                else:
                    duration = 0.0
                if warmup_left > 0:
                    warmup_left -= 1
                    continue
                total_processed += chunk.shape[0]
                metrics.record(duration if duration > 0 else 0.0)
                if args.log_interval and len(metrics.latencies) % args.log_interval == 0 and dist.get_rank() == 0 if dist.is_initialized() else True:
                    print(f"[step {step}] processed={total_processed} samples, last_latency={duration:.4f}s")
    return metrics.summary(processed=total_processed, batch_size=args.batch_size)


def aggregate_metrics(local_metrics: Dict[str, Any], dist_state: Dict[str, int]) -> Dict[str, Any]:
    if dist_state["world_size"] == 1:
        return local_metrics
    keys = ["samples", "batches", "total_time_s"]
    target_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.zeros(len(keys), device=target_device)
    for idx, key in enumerate(keys):
        tensor[idx] = float(local_metrics.get(key, 0.0))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    merged = dict(local_metrics)
    for idx, key in enumerate(keys):
        merged[key] = tensor[idx].item()
    merged["throughput_sps"] = merged["samples"] / merged["total_time_s"] if merged["total_time_s"] > 0 else 0.0
    return merged


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dist_state = init_distributed_if_needed()
    device = torch.device("cuda", dist_state["local_rank"]) if torch.cuda.is_available() else torch.device("cpu")
    data_cfg = DataConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dataset = build_dataset(data_cfg)
    dataset = shard_dataset(dataset, dist_state["world_size"], dist_state["rank"])
    dataloader = build_dataloader(data_cfg, dataset)
    model = prepare_model(args, device, dist_state)
    metrics = run_loop(args, model, dataloader, device)
    metrics["gpu"] = gather_gpu_stats()
    metrics["mode"] = args.mode
    metrics["model_name"] = args.model
    metrics["precision"] = args.precision
    metrics["world_size"] = dist_state["world_size"]
    summary = aggregate_metrics(metrics, dist_state)
    if dist_state["rank"] == 0:
        print(json.dumps(summary, indent=2))
        if args.metrics_path:
            args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_path.write_text(json.dumps(summary, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

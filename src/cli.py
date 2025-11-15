import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import distributed as dist

from .data import DataConfig, build_dataloader, build_dataset, shard_dataset
from .model_loader import list_models, load_model, load_tokenizer
from .profiler import MetricsBuffer, build_profiler, gather_gpu_stats

MODEL_NAMES = list_models()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU language model inference benchmarking toolkit")
    parser.add_argument("--model", type=str, default=MODEL_NAMES[0], choices=MODEL_NAMES)
    parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer name; defaults to the model name.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "curated"])
    parser.add_argument("--mode", type=str, default="sequential", choices=["sequential", "batched", "dataparallel", "distributed"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--capture-graph", action="store_true")
    parser.add_argument("--use-torchscript", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--profile-wait", type=int, default=2)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-active", type=int, default=10)
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
        self.static_inputs: Optional[Dict[str, torch.Tensor]] = None
        self.static_output = None

    def _shapes_match(self, batch: Dict[str, torch.Tensor]) -> bool:
        assert self.static_inputs is not None
        for key, tensor in self.static_inputs.items():
            if key not in batch or batch[key].shape != tensor.shape:
                return False
        return True

    def __call__(self, batch: Dict[str, torch.Tensor]):
        if self.graph is None:
            self.static_inputs = {key: torch.empty_like(value) for key, value in batch.items()}
            self.graph = torch.cuda.CUDAGraph()
            autocast_cm = torch.cuda.amp.autocast(dtype=self.dtype) if self.dtype else nullcontext()
            for key, tensor in self.static_inputs.items():
                tensor.copy_(batch[key])
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph):
                with autocast_cm:
                    self.static_output = self.model(**self.static_inputs)
            torch.cuda.synchronize()
        elif not self._shapes_match(batch):
            return self.model(**batch)
        else:
            for key, tensor in self.static_inputs.items():
                tensor.copy_(batch[key])
        self.graph.replay()
        return self.static_output


def prepare_model(args, device, dist_state):
    model = load_model(
        name=args.model,
        precision=args.precision,
        device=device,
        use_torchscript=args.use_torchscript,
    )
    if args.mode == "dataparallel":
        if torch.cuda.device_count() < 2:
            raise RuntimeError("DataParallel mode requires at least 2 CUDA devices")
        if args.capture_graph:
            raise RuntimeError("CUDA graph capture is not supported with DataParallel.")
        model = torch.nn.DataParallel(model)
    elif args.mode == "distributed":
        if not dist.is_initialized():
            raise RuntimeError("Distributed mode requires torch.distributed initialization")
        device_ids = [device.index] if device.type == "cuda" else None
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids, output_device=device.index)
    return model


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def split_batch(batch: Dict[str, torch.Tensor], sequential: bool) -> List[Dict[str, torch.Tensor]]:
    if not sequential:
        return [batch]
    batch_size = next(iter(batch.values())).shape[0]
    return [{key: value[i : i + 1] for key, value in batch.items()} for i in range(batch_size)]


def batch_size_from(batch: Dict[str, torch.Tensor]) -> int:
    return next(iter(batch.values())).shape[0]


def token_count(batch: Dict[str, torch.Tensor]) -> int:
    if "attention_mask" in batch:
        return int(batch["attention_mask"].sum().item())
    return int(batch_size_from(batch) * batch["input_ids"].shape[1])


def run_loop(args, model, dataloader, device, profiler) -> Dict[str, Any]:
    total_processed = 0
    total_tokens = 0
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
        for step, batch in enumerate(dataloader):
            if total_processed >= args.num_samples:
                break
            batch = move_to_device(batch, device)
            micro_batches = split_batch(batch, sequential=(args.mode == "sequential"))
            for chunk in micro_batches:
                if total_processed >= args.num_samples:
                    break
                use_cuda = torch.cuda.is_available() and device.type == "cuda"
                if use_cuda:
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    start_time = time.perf_counter()
                    start = end = None
                with amp_context:
                    if graph_runner:
                        _ = graph_runner(chunk)
                    else:
                        _ = model(**chunk)
                if end is not None:
                    end.record()
                    torch.cuda.synchronize()
                    duration = start.elapsed_time(end) / 1000.0
                else:
                    duration = time.perf_counter() - start_time
                if profiler is not None:
                    profiler.step()
                if warmup_left > 0:
                    warmup_left -= 1
                    continue
                current_bs = batch_size_from(chunk)
                total_processed += current_bs
                total_tokens += token_count(chunk)
                metrics.record(max(duration, 0.0))
                should_log = True
                if dist.is_initialized():
                    should_log = dist.get_rank() == 0
                if args.log_interval and len(metrics.latencies) % args.log_interval == 0 and should_log:
                    print(f"[step {step}] processed={total_processed} samples, last_latency={duration:.4f}s")
    return metrics.summary(processed=total_processed, batch_size=args.batch_size, tokens=total_tokens)


def aggregate_metrics(local_metrics: Dict[str, Any], dist_state: Dict[str, int]) -> Dict[str, Any]:
    if dist_state["world_size"] == 1:
        return local_metrics
    keys = ["samples", "batches", "total_time_s", "tokens"]
    target_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.zeros(len(keys), device=target_device)
    for idx, key in enumerate(keys):
        tensor[idx] = float(local_metrics.get(key, 0.0))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    merged = dict(local_metrics)
    for idx, key in enumerate(keys):
        merged[key] = tensor[idx].item()
    merged["throughput_sps"] = merged["samples"] / merged["total_time_s"] if merged["total_time_s"] > 0 else 0.0
    merged["tokens_per_second"] = merged["tokens"] / merged["total_time_s"] if merged["total_time_s"] > 0 else 0.0
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
        max_length=args.max_length,
        prompt_file=args.prompt_file,
    )
    tokenizer_name = args.tokenizer or args.model
    tokenizer = load_tokenizer(tokenizer_name)
    dataset = build_dataset(data_cfg, tokenizer)
    dataset = shard_dataset(dataset, dist_state["world_size"], dist_state["rank"])
    dataloader = build_dataloader(data_cfg, dataset)
    model = prepare_model(args, device, dist_state)
    profile_dir = args.profile_dir
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
    with build_profiler(
        enable=args.profile,
        trace_dir=profile_dir,
        rank=dist_state["rank"],
        wait=args.profile_wait,
        warmup=args.profile_warmup,
        active=args.profile_active,
    ) as profiler:
        metrics = run_loop(args, model, dataloader, device, profiler)
    metrics["gpu"] = gather_gpu_stats()
    metrics["mode"] = args.mode
    metrics["model_name"] = args.model
    metrics["tokenizer"] = tokenizer_name
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

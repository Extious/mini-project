from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


@dataclass
class MetricsBuffer:
    latencies: List[float] = field(default_factory=list)

    def record(self, duration: float) -> None:
        self.latencies.append(duration)

    def summary(self, processed: int, batch_size: int, tokens: int) -> Dict[str, float]:
        if not self.latencies:
            return {}
        total_time = sum(self.latencies)
        throughput = processed / total_time if total_time > 0 else 0.0
        token_throughput = tokens / total_time if total_time > 0 else 0.0
        sorted_lat = sorted(self.latencies)
        idx = max(int(0.95 * (len(sorted_lat) - 1)), 0)
        return {
            "samples": processed,
            "tokens": tokens,
            "batches": len(self.latencies),
            "avg_latency_s": total_time / len(self.latencies),
            "p95_latency_s": sorted_lat[idx],
            "total_time_s": total_time,
            "throughput_sps": throughput,
            "tokens_per_second": token_throughput,
            "batch_size": batch_size,
        }


def gather_gpu_stats() -> Dict[str, float]:
    stats = {}
    if not torch.cuda.is_available():
        return stats
    stats["device_count"] = torch.cuda.device_count()
    stats["devices"] = ",".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    stats["max_memory_mb"] = max(torch.cuda.max_memory_allocated(i) / (1024**2) for i in range(torch.cuda.device_count()))
    return stats


@contextmanager
def build_profiler(
    enable: bool,
    trace_dir: Optional[Path],
    rank: int,
    wait: int,
    warmup: int,
    active: int,
):
    if not enable:
        yield None
        return
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    handler = None
    if trace_dir is not None:
        trace_dir = trace_dir / f"rank{rank}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        handler = tensorboard_trace_handler(str(trace_dir))
    with profile(
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield prof

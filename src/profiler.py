import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

import torch


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def cuda_timer():
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = end - start
        setattr(cuda_timer, "last_elapsed", elapsed)


@dataclass
class MetricsBuffer:
    latencies: List[float] = field(default_factory=list)

    def record(self, duration: float) -> None:
        self.latencies.append(duration)

    def summary(self, processed: int, batch_size: int) -> Dict[str, float]:
        if not self.latencies:
            return {}
        total_time = sum(self.latencies)
        throughput = processed / total_time if total_time > 0 else 0.0
        sorted_lat = sorted(self.latencies)
        idx = max(int(0.95 * (len(sorted_lat) - 1)), 0)
        return {
            "samples": processed,
            "batches": len(self.latencies),
            "avg_latency_s": total_time / len(self.latencies),
            "p95_latency_s": sorted_lat[idx],
            "total_time_s": total_time,
            "throughput_sps": throughput,
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

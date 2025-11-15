import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from torch.utils.data import DataLoader, Dataset, Subset
from transformers import PreTrainedTokenizerBase


DEFAULT_PROMPTS: List[str] = [
    "Summarize the following research abstract in one sentence.",
    "Generate a concise title for a paper about reinforcement learning with transformers.",
    "Explain how rotary positional embeddings impact autoregressive decoding.",
    "Outline the steps required to fine-tune a causal language model on legal documents.",
    "Compare tensor parallelism and pipeline parallelism for large language model inference.",
    "Provide three risks associated with deploying conversational agents in healthcare.",
    "Describe how key-value caching accelerates autoregressive decoding.",
    "List optimization strategies for batching multilingual prompts on GPUs.",
    "Predict how quantization-aware training influences downstream accuracy.",
    "Suggest an evaluation plan for latency-sensitive dialogue models.",
]


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    batch_size: int = 8
    num_samples: int = 512
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    seed: int = 42
    max_length: int = 256
    prompt_file: Optional[str] = None
    prompts: List[str] = field(default_factory=list)


class PromptDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: DataConfig) -> None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.config = config
        self.prompts = self._materialize_prompts(config)

    def _materialize_prompts(self, config: DataConfig) -> List[str]:
        source: List[str] = []
        if config.prompt_file:
            path = Path(config.prompt_file)
            if path.exists():
                source = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not source:
            source = config.prompts or DEFAULT_PROMPTS
        rng = random.Random(config.seed)
        prompts: List[str] = []
        while len(prompts) < config.num_samples:
            base = rng.choice(source)
            if config.dataset == "synthetic":
                suffix = rng.randint(0, 10**6)
                prompts.append(f"{base} #{suffix}")
            else:
                prompts.append(base)
        return prompts[: config.num_samples]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int):
        encoded = self.tokenizer(
            self.prompts[index],
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


def build_dataset(config: DataConfig, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    if config.dataset not in {"synthetic", "curated"}:
        raise ValueError(f"Unknown dataset '{config.dataset}'")
    return PromptDataset(tokenizer=tokenizer, config=config)


def shard_dataset(dataset: Dataset, world_size: int, rank: int) -> Dataset:
    if world_size <= 1:
        return dataset
    shard_size = math.ceil(len(dataset) / world_size)
    start = rank * shard_size
    end = min((rank + 1) * shard_size, len(dataset))
    indices = list(range(start, end))
    return Subset(dataset, indices)


def build_dataloader(
    config: DataConfig,
    dataset: Optional[Dataset] = None,
) -> DataLoader:
    if dataset is None:
        raise ValueError("Dataset must be provided when using language model workloads.")
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        shuffle=False,
        drop_last=False,
    )

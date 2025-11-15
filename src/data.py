import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torchvision import datasets, transforms
except ImportError:  # pragma: no cover - torchvision might be missing on cpu-only envs
    datasets = None
    transforms = None


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    batch_size: int = 32
    num_samples: int = 2048
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    seed: int = 42
    image_size: Tuple[int, int, int] = (3, 224, 224)
    num_classes: int = 1000


class SyntheticImageDataset(Dataset):
    """Synthetic dataset that mimics ImageNet-sized tensors."""

    def __init__(
        self,
        num_samples: int,
        image_size: Tuple[int, int, int],
        num_classes: int,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        g = torch.Generator()
        g.manual_seed(seed)
        self.data = torch.randn((num_samples, *image_size), generator=g)
        self.targets = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]


def _build_cifar10_dataset(num_samples: int) -> Dataset:
    if datasets is None or transforms is None:
        raise RuntimeError("torchvision is required for CIFAR10 dataset")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    if num_samples < len(dataset):
        dataset = Subset(dataset, torch.arange(num_samples))
    return dataset


def build_dataset(config: DataConfig) -> Dataset:
    if config.dataset == "synthetic":
        return SyntheticImageDataset(
            num_samples=config.num_samples,
            image_size=config.image_size,
            num_classes=config.num_classes,
            seed=config.seed,
        )
    if config.dataset == "cifar10":
        return _build_cifar10_dataset(config.num_samples)
    raise ValueError(f"Unknown dataset '{config.dataset}'")


def shard_dataset(dataset: Dataset, world_size: int, rank: int) -> Dataset:
    if world_size <= 1:
        return dataset
    shard_size = math.ceil(len(dataset) / world_size)
    start = rank * shard_size
    end = min((rank + 1) * shard_size, len(dataset))
    indices = torch.arange(start, end)
    return Subset(dataset, indices)


def build_dataloader(config: DataConfig, dataset: Optional[Dataset] = None) -> DataLoader:
    dataset = dataset or build_dataset(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        shuffle=False,
        drop_last=False,
    )

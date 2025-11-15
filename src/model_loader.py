from functools import lru_cache
from typing import Callable, Dict

import torch

try:
    from torchvision import models
except ImportError:  # pragma: no cover
    models = None


def _build_model_factory() -> Dict[str, Callable[[], torch.nn.Module]]:
    if models is None:
        raise RuntimeError("torchvision is required for pretrained backbones")

    return {
        "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
        "efficientnet_b0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "vit_b_16": lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),
    }


@lru_cache(maxsize=1)
def _factory_cache() -> Dict[str, Callable[[], torch.nn.Module]]:
    return _build_model_factory()


def list_models() -> Dict[str, Callable[[], torch.nn.Module]]:
    return _factory_cache()


def load_model(
    name: str,
    precision: str,
    device: torch.device,
    image_shape=(3, 224, 224),
    use_torchscript: bool = False,
) -> torch.nn.Module:
    factory = _factory_cache()
    if name not in factory:
        raise ValueError(f"Unknown model '{name}'")
    model = factory[name]()
    model.eval()
    model.to(device, non_blocking=True)

    if precision == "fp16":
        model.half()
    elif precision == "bf16":
        model.bfloat16()

    if use_torchscript:
        example = torch.randn((1, *image_shape), device=device)
        model = torch.jit.trace(model, example)
        model = torch.jit.freeze(model)

    return model

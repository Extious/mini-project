from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

SUPPORTED_MODELS: Tuple[str, ...] = ("distilgpt2", "gpt2", "gpt2-medium")


def list_models() -> Tuple[str, ...]:
    return SUPPORTED_MODELS


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    name: str,
    precision: str,
    device: torch.device,
    use_torchscript: bool = False,
) -> PreTrainedModel:
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{name}'")
    if use_torchscript:
        raise ValueError("TorchScript is not supported for Hugging Face causal LM models in this toolkit.")
    dtype = torch.float32
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.eval()
    model.to(device, non_blocking=True)
    return model

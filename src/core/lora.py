"""PEFT LoRA construction and checkpoint identity, without merging base weights."""

from enum import Enum
from math import isfinite

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import PreTrainedModel


def apply_lora(model, *, rank: int = 16, alpha: float = 32.0) -> PeftModel:
    """Wrap an exclusively owned actor in place; keep its reference model separate.

    Only adapters are trainable and cast to FP32; frozen base weights retain their
    dtype. Checkpoints preserve the frozen tensors either inline or in a
    hash-verified compact checkpoint dependency. Preserve FP32 adapter storage.
    """
    assert type(rank) is int and rank > 0, "rank must be a positive integer"
    assert type(alpha) in (int, float) and isfinite(alpha) and alpha > 0, (
        "alpha must be finite and positive"
    )
    assert isinstance(model, PreTrainedModel), "Expected an unwrapped HF model"
    assert not getattr(model, "peft_config", None) and not any(
        isinstance(module, BaseTunerLayer) for module in model.modules()
    ), "Model already has adapters"
    assert not model.config.is_encoder_decoder, "Expected a decoder-only model"
    assert not getattr(model, "is_quantized", False), "Quantization is not supported"
    config = LoraConfig(
        r=rank,
        lora_alpha=float(alpha),
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    wrapped = get_peft_model(model, config, autocast_adapter_dtype=True)
    adapters = {
        id(parameter)
        for module in wrapped.modules()
        if isinstance(module, LoraLayer)
        for adapter in (module.lora_A, module.lora_B)
        for parameter in adapter.parameters()
    }
    assert adapters, "No LoRA parameters were created"
    for name, parameter in wrapped.named_parameters():
        assert parameter.requires_grad == (id(parameter) in adapters), (
            f"Only LoRA adapters must be trainable: {name}"
        )
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=torch.float32)
    return wrapped


def _primitive(value):
    """Normalize configs deterministically; reject opaque checkpoint objects."""
    if isinstance(value, Enum):
        return _primitive(value.value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        assert isfinite(value), "Nonfinite LoRA identity value"
        return value
    if isinstance(value, dict):
        assert all(type(key) is str for key in value), "Expected string config keys"
        return {key: _primitive(value[key]) for key in sorted(value)}
    if isinstance(value, (set, frozenset)):
        return sorted(_primitive(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    raise TypeError(f"Unsupported LoRA identity value: {type(value).__name__}")


def lora_identity(model) -> dict | None:
    """Return primitive config and effective adapter state for resume validation.

    Include actual per-layer scaling: it can change independently of config or
    tensor shapes. Active adapter order, disabled state, and merges also affect
    behavior. No tensors or PEFT enums enter weights_only checkpoint metadata.
    """
    configs = getattr(model, "peft_config", None)
    layers = {
        name: {
            "rank": module.r,
            "alpha": module.lora_alpha,
            "scaling": module.scaling,
            "active_adapters": module.active_adapters,
            "disabled": module.disable_adapters,
            "merged_adapters": module.merged_adapters,
        }
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer)
    }
    if not configs:
        assert not layers, "LoRA layers without PEFT configuration"
        return None
    assert all(isinstance(config, LoraConfig) for config in configs.values()), (
        "Expected LoRA configurations"
    )
    assert layers, "LoRA configuration without adapter layers"
    return _primitive(
        {
            "adapters": {name: config.to_dict() for name, config in configs.items()},
            "layers": layers,
        }
    )

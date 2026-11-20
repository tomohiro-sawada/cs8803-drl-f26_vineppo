"""Precision checks shared by training, direct updates, and checkpoint restore."""

import torch


def trainable_fp32_parameters(model):
    parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.dtype == torch.float32, (
                f"Trainable parameter {name} requires FP32 storage, got {parameter.dtype}"
            )
            parameters.append(parameter)
    return parameters


def assert_fp32_optimizer_state(state):
    for slot in state.values():
        for name, value in slot.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert value.dtype == torch.float32, (
                    f"Optimizer state {name} requires FP32 storage, got {value.dtype}"
                )

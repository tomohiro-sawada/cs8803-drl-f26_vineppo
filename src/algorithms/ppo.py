"""Critic-based PPO primitives with fixed targets and separate optimizers.

Cache values before either model updates. Compute GAE once, then reuse its
returns throughout critic optimization. Actor/backbone independence is the
caller's responsibility. Response ends, including truncations, are terminal.
"""

from dataclasses import dataclass, fields
from math import isfinite

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from algorithms.base import AlgorithmConfig
from core.data import CriticBatch
from core.lora import apply_lora
from core.optimization import assert_fp32_optimizer_state, trainable_fp32_parameters


@dataclass
class PPOConfig(AlgorithmConfig):
    gamma: float = 1.0
    lam: float = 0.95
    value_clip_eps: float | None = None

    def __post_init__(self):
        assert isfinite(self.gamma) and 0 <= self.gamma <= 1
        assert isfinite(self.lam) and 0 <= self.lam <= 1
        assert self.value_clip_eps is None or 0 < self.value_clip_eps < 1


class TokenCritic(nn.Module):
    """Scalar value head on a caller-supplied decoder-only base AutoModel.

    With a LoRA rank, only the FP32 adapters and FP32 value head train. The
    backbone keeps its loaded precision and is independent of the actor.
    """

    def __init__(self, backbone: PreTrainedModel, *, lora_rank: int | None = None):
        super().__init__()
        assert isinstance(backbone, PreTrainedModel)
        assert not backbone.config.is_encoder_decoder, "Expected a decoder-only model"
        self.lora_rank = lora_rank
        parameter = next(backbone.parameters())
        self.backbone = (
            backbone
            if lora_rank is None
            else apply_lora(
                backbone,
                rank=lora_rank,
                alpha=2 * lora_rank,
                task_type="FEATURE_EXTRACTION",
            )
        )
        self.value_head = nn.Linear(
            backbone.config.hidden_size,
            1,
            device=parameter.device,
            dtype=parameter.dtype if lora_rank is None else torch.float32,
        )

    @property
    def config(self):
        return self.backbone.config

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        use_cache: bool = False,
    ) -> Tensor:
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            return_dict=True,
        ).last_hidden_state
        assert hidden.shape == (*input_ids.shape, self.config.hidden_size)
        return self.value_head(hidden.to(self.value_head.weight.dtype)).squeeze(-1)


def token_values(
    critic: TokenCritic, input_ids: Tensor, attention_mask: Tensor
) -> Tensor:
    """Float32 [B, T-1] state-before-action values; eval mode, autograd preserved.

    Mask-derived positions isolate left/right padding. Off-response values have
    no meaning; the caller must apply its shifted response mask.
    """
    assert input_ids.ndim == 2 and input_ids.shape[1] >= 2
    assert input_ids.dtype == torch.long
    assert attention_mask.shape == input_ids.shape
    assert attention_mask.device == input_ids.device
    assert ((attention_mask == 0) | (attention_mask == 1)).all()
    assert attention_mask.bool().any(dim=1).all(), "Every row needs a prefix"
    critic.eval()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return critic(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )[:, :-1].float()


def _validate_masked_pair(first: Tensor, second: Tensor, mask: Tensor) -> None:
    assert mask.ndim == 2 and mask.dtype == torch.bool
    assert mask.shape[0] > 0 and mask.any(dim=1).all(), (
        "Every sequence needs response tokens"
    )
    for value in (first, second):
        assert value.shape == mask.shape and value.device == mask.device
        assert value.is_floating_point()
        assert torch.isfinite(value[mask]).all(), "Nonfinite response data"


def _validate_contiguous(mask: Tensor) -> None:
    starts = mask[:, :1].sum(dim=1) + (mask[:, 1:] & ~mask[:, :-1]).sum(dim=1)
    assert (starts == 1).all(), "Response masks must be contiguous"


@torch.no_grad()
def masked_gae(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    *,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Detached raw GAE and returns, zero off response, without mutating inputs.

    Each row contains one nonempty contiguous response, with optional leading
    and trailing false entries. No bootstrap follows the final response token.
    """
    PPOConfig(gamma=gamma, lam=lam)
    _validate_masked_pair(rewards, values, mask)
    _validate_contiguous(mask)
    rewards = rewards.detach().masked_fill(~mask, 0)
    values = values.detach().masked_fill(~mask, 0)
    advantages = torch.zeros_like(values)
    following = torch.zeros_like(values[:, 0])
    carry = torch.zeros_like(following)
    for t in reversed(range(mask.shape[1])):
        delta = rewards[:, t] + gamma * following - values[:, t]
        carry = (delta + gamma * lam * carry).masked_fill(~mask[:, t], 0)
        advantages[:, t] = carry
        following = values[:, t]
    returns = (advantages + values).masked_fill(~mask, 0)
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()
    return advantages, returns


def value_loss(
    values: Tensor,
    returns: Tensor,
    mask: Tensor,
    *,
    num_tokens: int | None = None,
    old_values: Tensor | None = None,
    value_clip_eps: float | None = None,
) -> Tensor:
    """Half masked MSE, clipped against fixed pre-update values when requested."""
    _validate_masked_pair(values, returns, mask)
    tokens = int(mask.sum())
    num_tokens = tokens if num_tokens is None else num_tokens
    assert type(num_tokens) is int and num_tokens >= tokens
    current, target = values[mask], returns.detach()[mask]
    error = (current - target).square()
    if value_clip_eps is not None:
        assert old_values is not None and 0 < value_clip_eps < 1
        _validate_masked_pair(old_values, returns, mask)
        old = old_values.detach()[mask]
        clipped = old + (current - old).clamp(-value_clip_eps, value_clip_eps)
        error = torch.maximum(error, (clipped - target).square())
    loss = 0.5 * error.sum() / num_tokens
    assert torch.isfinite(loss), "Nonfinite value loss"
    return loss


def update_critic(
    critic: TokenCritic,
    optimizer: torch.optim.Optimizer,
    batch: CriticBatch,
    *,
    epochs: int,
    minibatch_size: int,
    microbatch_size: int,
    max_grad_norm: float = 1.0,
    generator: torch.Generator | None = None,
    value_clip_eps: float | None = None,
) -> list[dict[str, float]]:
    """Optimize fixed returns; report pre-step token-mean loss and pre-clip norm.

    CPU or critic-device batches are accepted. Only microbatches move to the
    critic. A CPU generator shuffles each epoch; partial batches are included.
    Clip once per logical minibatch, never regenerate targets, and clear stale
    and exit gradients (including failures). Dropout stays disabled.
    """
    optimizer.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    try:
        assert all(
            type(n) is int and n > 0 for n in (epochs, minibatch_size, microbatch_size)
        )
        assert isfinite(max_grad_norm) and max_grad_norm > 0
        assert torch.is_grad_enabled(), "Critic update requires autograd"
        assert generator is None or generator.device.type == "cpu", (
            "Expected a CPU generator"
        )
        assert batch.input_ids.ndim == 2 and batch.input_ids.dtype == torch.long
        size, width = batch.input_ids.shape
        assert size > 0 and width >= 2
        assert batch.attention_mask.shape == batch.input_ids.shape
        assert batch.attention_mask.dtype == torch.bool
        assert batch.response_mask.shape == (size, width - 1)
        for field in fields(batch):
            assert getattr(batch, field.name).device == batch.input_ids.device
        _validate_masked_pair(batch.returns, batch.old_values, batch.response_mask)
        _validate_contiguous(batch.response_mask)
        assert not (
            batch.response_mask
            & ~(batch.attention_mask[:, :-1] & batch.attention_mask[:, 1:])
        ).any(), "Response tokens need attended states and actions"
        parameters = trainable_fp32_parameters(critic)
        owned = [p for group in optimizer.param_groups for p in group["params"]]
        assert len(owned) == len({id(p) for p in owned}), (
            "Duplicate optimizer parameters"
        )
        assert parameters and {id(p) for p in owned} == {id(p) for p in parameters}, (
            "Optimizer must own exactly the trainable critic parameters"
        )
        assert_fp32_optimizer_state(optimizer.state)
        device = parameters[0].device
        assert all(p.device == device for p in critic.parameters()), (
            "Single-device update only"
        )
        assert batch.input_ids.device.type == "cpu" or batch.input_ids.device == device
        targets = batch.returns.detach()
        history = []
        for _ in range(epochs):
            order = torch.randperm(size, generator=generator, device="cpu").to(
                batch.input_ids.device
            )
            for indices in order.split(minibatch_size):
                tokens = int(batch.response_mask[indices].sum())
                total_loss = 0.0
                for rows in indices.split(microbatch_size):
                    current = token_values(
                        critic,
                        batch.input_ids[rows].to(device),
                        batch.attention_mask[rows].to(device),
                    )
                    loss = value_loss(
                        current,
                        targets[rows].to(device),
                        batch.response_mask[rows].to(device),
                        num_tokens=tokens,
                        old_values=batch.old_values[rows].to(device),
                        value_clip_eps=value_clip_eps,
                    )
                    loss.backward()
                    total_loss += loss.item()
                norm = nn.utils.clip_grad_norm_(
                    parameters, max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                history.append(
                    {
                        "value_loss": total_loss,
                        "grad_norm": norm.item(),
                        "num_tokens": tokens,
                        "num_sequences": len(indices),
                    }
                )
        return history
    finally:
        optimizer.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)

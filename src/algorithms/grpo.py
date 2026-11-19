"""Group-relative advantages and sequence-normalized token-ratio GRPO."""

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from algorithms.base import AlgorithmConfig


@dataclass
class GRPOConfig(AlgorithmConfig):
    eps: float = 1e-6

    def __post_init__(self):
        assert isfinite(self.eps) and self.eps > 0


def _validate_rewards(rewards: Tensor, minimum_group: int) -> None:
    assert rewards.ndim == 2 and rewards.shape[0] > 0
    assert rewards.shape[1] >= minimum_group, f"Requires G >= {minimum_group}"
    assert rewards.is_floating_point() and torch.isfinite(rewards).all()


@torch.no_grad()
def group_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """Detached [B, G] population-standardized rewards, independently per prompt.

    Constant groups (including G=1) yield zero. Trainers must require G >= 2;
    this helper preserves the homework's G=1 contract. No token whitening.
    """
    _validate_rewards(rewards, 1)
    GRPOConfig(eps=eps)
    # Translation leaves the estimator unchanged and makes tied rewards exactly
    # zero even when their common non-integer value has an inexact FP32 mean.
    shifted = rewards - rewards[:, :1]
    result = (shifted - shifted.mean(dim=1, keepdim=True)) / (
        shifted.std(dim=1, correction=0, keepdim=True) + eps
    )
    assert torch.isfinite(result).all(), "Nonfinite group advantages"
    return result


def _grouped_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    ref_logprobs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    sequence_ratio: bool,
    clip_eps: float,
    kl_coef: float,
    num_tokens: int | None,
    num_sequences: int | None,
) -> tuple[Tensor, dict[str, Tensor]]:
    assert mask.ndim == 2 and mask.dtype == torch.bool
    assert isfinite(clip_eps) and 0 < clip_eps < 1
    assert isfinite(kl_coef) and kl_coef >= 0
    for value in (logprobs, old_logprobs, ref_logprobs, advantages):
        assert value.shape == mask.shape and value.device == mask.device
        assert value.is_floating_point()
        assert torch.isfinite(value[mask]).all(), "Nonfinite response data"
    nonempty = mask.any(dim=1)
    sequences = int(nonempty.sum())
    if sequences == 0:
        raise ValueError("At least one nonempty completion is required")
    num_sequences = sequences if num_sequences is None else num_sequences
    assert type(num_sequences) is int and num_sequences >= sequences
    if num_tokens is not None:
        assert type(num_tokens) is int and num_tokens >= int(mask.sum())
    mask = mask[nonempty]
    current = logprobs[nonempty].masked_fill(~mask, 0)
    old, ref, adv = (
        value.detach()[nonempty].masked_fill(~mask, 0)
        for value in (old_logprobs, ref_logprobs, advantages)
    )
    lengths = mask.sum(dim=1)
    response_adv = adv.gather(1, mask.long().argmax(dim=1, keepdim=True))
    assert ((adv == response_adv) | ~mask).all(), (
        "Advantages must be constant within each response"
    )
    log_ratio = current - old
    d = ref - current
    reference = torch.expm1(d) - d
    assert torch.isfinite(reference).all(), "Score exponential overflow"
    if sequence_ratio:
        log_ratio = log_ratio.sum(dim=1)
        adv = response_adv.squeeze(1)
    ratio = log_ratio.exp()
    assert torch.isfinite(ratio).all(), "Score exponential overflow"
    surrogate = -torch.minimum(
        ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    )

    def reduce(value: Tensor) -> Tensor:
        if sequence_ratio:
            return value.sum() / num_sequences
        return (value.masked_fill(~mask, 0).sum(dim=1) / lengths).sum() / num_sequences

    policy_loss = reduce(surrogate)
    reference_kl = (
        reference.sum() / num_sequences if sequence_ratio else reduce(reference)
    )
    loss = policy_loss + kl_coef * reference_kl
    with torch.no_grad():
        info = {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "reference_kl": reference_kl.detach(),
            "kl_loss": kl_coef * reference_kl.detach(),
            "ratio_mean": reduce(ratio),
            "clip_frac": reduce(((ratio - 1).abs() > clip_eps).to(ratio.dtype)),
            "approx_kl": reduce(0.5 * log_ratio.square()),
        }
    assert all(torch.isfinite(value) for value in info.values()), (
        "Nonfinite grouped loss"
    )
    return loss, info


def grpo_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    ref_logprobs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 1e-4,
    num_tokens: int | None = None,
    num_sequences: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Clipped token ratios plus unclamped k3, averaged within then across responses.

    All inputs are actor-compatible [B, T-1]; advantages must be expanded and
    constant on each response. Old/reference scores and advantages are detached,
    never whitened. Mask before exponentiation; ignore empty rows, reject an
    entirely empty batch. Only current scores receive gradients.

    All metrics use the same equal-completion reduction. For accumulation,
    num_sequences is the full logical minibatch's nonempty count; num_tokens
    is accepted for updater compatibility but does not determine the reduction.
    k3 = exp(ref-current) - (ref-current) - 1 is an old-rollout regularizer,
    not an unbiased estimate of current-policy KL after updating the policy.
    """
    return _grouped_loss(
        logprobs,
        old_logprobs,
        ref_logprobs,
        advantages,
        mask,
        sequence_ratio=False,
        clip_eps=clip_eps,
        kl_coef=kl_coef,
        num_tokens=num_tokens,
        num_sequences=num_sequences,
    )

"""Leave-one-out advantages with a PPO-style clipped sequence-ratio objective."""

from dataclasses import dataclass

import torch
from torch import Tensor

from algorithms.base import AlgorithmConfig
from algorithms.grpo import _grouped_loss, _validate_rewards


@dataclass
class RLOOConfig(AlgorithmConfig):
    """Distinct configuration identity; no RLOO-specific settings."""


@torch.no_grad()
def rloo_advantages(rewards: Tensor) -> Tensor:
    """Detached r_i - mean(other rewards), independently per prompt; G >= 2.

    No standard deviation scaling, token whitening, or KL reward shaping.
    """
    _validate_rewards(rewards, 2)
    shifted = rewards - rewards[:, :1]
    result = shifted - (shifted.sum(dim=1, keepdim=True) - shifted) / (
        rewards.shape[1] - 1
    )
    assert torch.isfinite(result).all(), "Nonfinite leave-one-out advantages"
    return result


def rloo_loss(
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
    """PPO-style sequence-ratio RLOO, not exact vanilla on-policy REINFORCE.

    Actor-compatible [B, T-1] inputs have constant expanded response advantages.
    The importance ratio is exp(SUM_response(new-old)); optimize the sequence
    mean of -min(ratio*A, clipped_ratio*A). Separately SUM unclamped token k3
    within each sequence, then average sequences. No whitening or reward shaping.

    ratio_mean, clip_frac, and approx_kl describe SEQUENCE ratios/log-ratios,
    not tokens. Fixed scores/advantages are detached. Empty rows are ignored;
    all-empty batches fail. num_sequences overrides the accumulation denominator;
    num_tokens is accepted but never used as a reduction denominator. As in GRPO,
    old-rollout k3 is not an unbiased current-policy KL after policy updates.
    """
    return _grouped_loss(
        logprobs,
        old_logprobs,
        ref_logprobs,
        advantages,
        mask,
        sequence_ratio=True,
        clip_eps=clip_eps,
        kl_coef=kl_coef,
        num_tokens=num_tokens,
        num_sequences=num_sequences,
    )

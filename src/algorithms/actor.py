"""Fixed-data, single-device VinePPO actor optimization; no collection or critic.

Numerics follow McGill-NLP/VinePPO at 9f716e68bb934e71da86c226c1699c36b5553faf:
configs/trainers/{ppo_MATH,refKl0.0001,klLoss}.jsonnet and
src/treetune/trainers/ppo_trainer.py (_training_step, _compute_actor_loss).
Policy is a response-token mean; reference k3 is clamped to [0, 10], summed
per sequence, then sequence-averaged. These are different denominators.

Whitening uses expanded tokens and sample variance, once per logical optimizer
minibatch. Unlike upstream with accumulation > 1, memory-only microbatch splits
must not change whitening statistics or either loss denominator. The paper's
trajectory-summed surrogate is not the upstream token-mean policy reduction.
"""

from dataclasses import fields
from math import isfinite
from typing import Literal

import torch
from torch import Tensor
from transformers import PreTrainedModel

from algorithms.grpo import grpo_loss
from algorithms.rloo import rloo_loss
from core.data import PolicyGradientBatch
from core.inference import token_logprobs


def normalize_advantages(advantages: Tensor, mask: Tensor) -> Tensor:
    """Whiten raw expanded advantages over valid tokens; never mutate targets.

    Call once per logical minibatch, not per step, sequence, or microbatch.
    N > 1 is required for Bessel-corrected variance. Constant inputs yield zero.
    """
    assert mask.dtype == torch.bool and mask.shape == advantages.shape
    values = advantages.detach()[mask]
    assert values.numel() > 1, "Sample variance requires at least two response tokens"
    assert torch.isfinite(values).all(), "Nonfinite advantages"
    centered = values - values.mean()
    variance = centered.square().sum() / (values.numel() - 1)
    result = torch.zeros_like(advantages)
    result[mask] = centered / torch.sqrt(variance + 1e-8)
    return result


def actor_loss(
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
    """Clipped policy + explicit reference loss; advantages are already whitened.

    All tensors are [B, T-1], aligned to input_ids[:, 1:]. Only current scores
    receive gradients. Select valid entries BEFORE exponentiation so arbitrary
    prompt/padding scores cannot poison loss or gradients. Optional denominators
    are the full logical minibatch counts when accumulating microbatches.
    Diagnostics use the same denominators and can therefore be summed.
    """
    assert mask.ndim == 2 and mask.dtype == torch.bool
    assert mask.any(dim=1).all(), "Every sequence needs response tokens"
    assert all(
        x.shape == mask.shape
        for x in (logprobs, old_logprobs, ref_logprobs, advantages)
    )
    assert isfinite(clip_eps) and 0 < clip_eps < 1
    assert isfinite(kl_coef) and kl_coef >= 0
    tokens = int(mask.sum())
    sequences = mask.shape[0]
    num_tokens = tokens if num_tokens is None else num_tokens
    num_sequences = sequences if num_sequences is None else num_sequences
    assert num_tokens >= tokens and num_sequences >= sequences
    current = logprobs[mask]
    old, ref, adv = (x.detach()[mask] for x in (old_logprobs, ref_logprobs, advantages))
    assert all(torch.isfinite(x).all() for x in (current, old, ref, adv))

    log_ratio = current - old
    ratio = log_ratio.exp()
    surrogate = -torch.minimum(
        ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    )
    d = ref - current
    ref_kl = d.exp() - d - 1
    assert torch.isfinite(ratio).all() and torch.isfinite(ref_kl).all(), (
        "Score exponential overflow"
    )
    policy_loss = surrogate.sum() / num_tokens
    reference_kl = ref_kl.clamp(0, 10).sum() / num_sequences
    loss = policy_loss + kl_coef * reference_kl
    assert torch.isfinite(loss), "Nonfinite actor loss"
    with torch.no_grad():
        info = {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "reference_kl": reference_kl.detach(),
            "kl_loss": kl_coef * reference_kl.detach(),
            "ratio_mean": ratio.sum() / num_tokens,
            "clip_frac": ((ratio - 1).abs() > clip_eps).sum() / num_tokens,
            "approx_kl": 0.5 * log_ratio.square().sum() / num_tokens,
        }
    return loss, info


def update_actor(
    actor: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    batch: PolicyGradientBatch,
    *,
    epochs: int,
    minibatch_size: int,
    microbatch_size: int,
    clip_eps: float = 0.2,
    kl_coef: float = 1e-4,
    max_grad_norm: float = 1.0,
    generator: torch.Generator | None = None,
    objective: Literal["ppo", "grpo", "rloo"] = "ppo",
) -> list[dict[str, float]]:
    """Shuffle fixed data each epoch; return pre-step diagnostics per optimizer step.

    Caller supplies RAW token advantages and cached T=1 old/reference scores;
    PPO whitens these once per minibatch; grouped objectives use fixed expanded
    response advantages without further whitening. The caller must construct
    grouped advantages with G >= 2 before flattening responses into this batch.
    Scores/targets are detached, never regenerated or changed.
    CPU or actor-device batches are accepted; only each microbatch moves to actor.
    The caller owns optimizer creation/state, scheduling, and collection ordering.
    All trainable actor parameters must belong to this optimizer, with no foreign
    parameters. Existing gradients are cleared; gradients are also cleared on exit.
    Partial minibatches/microbatches are included. A CPU generator controls shuffle.
    Scoring leaves the actor in eval mode (dropout off, autograd on).
    No DDP, AMP scaler, scheduler, reference model, or reward shaping lives here.
    """
    assert objective in ("ppo", "grpo", "rloo"), "Unknown actor objective"
    loss_fn = {"ppo": actor_loss, "grpo": grpo_loss, "rloo": rloo_loss}[objective]
    assert all(
        type(n) is int and n > 0 for n in (epochs, minibatch_size, microbatch_size)
    )
    assert isfinite(max_grad_norm) and max_grad_norm > 0
    assert isfinite(clip_eps) and 0 < clip_eps < 1
    assert isfinite(kl_coef) and kl_coef >= 0
    assert torch.is_grad_enabled(), "Actor update requires autograd"
    size, width = batch.input_ids.shape
    assert size > 0 and width >= 2
    assert batch.attention_mask.shape == batch.input_ids.shape
    assert batch.attention_mask.dtype == torch.bool
    assert batch.response_mask.dtype == torch.bool
    for field in fields(batch):
        value = getattr(batch, field.name)
        assert value.device == batch.input_ids.device
        if field.name not in ("input_ids", "attention_mask"):
            assert value.shape == (size, width - 1)
    assert batch.response_mask.any(dim=1).all()
    assert not (
        batch.response_mask
        & ~(batch.attention_mask[:, :-1] & batch.attention_mask[:, 1:])
    ).any()
    for value in (batch.old_logprobs, batch.ref_logprobs, batch.advantages):
        assert torch.isfinite(value[batch.response_mask]).all()
    parameters = [p for p in actor.parameters() if p.requires_grad]
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(owned) == len({id(p) for p in owned}), "Duplicate optimizer parameters"
    assert parameters and {id(p) for p in owned} == {id(p) for p in parameters}, (
        "Optimizer must own exactly the trainable actor parameters"
    )
    device = parameters[0].device
    assert all(p.device == device for p in parameters), "Single-device update only"
    history = []
    optimizer.zero_grad(set_to_none=True)
    try:
        for _ in range(epochs):
            order = torch.randperm(size, generator=generator).to(batch.input_ids.device)
            for indices in order.split(minibatch_size):
                mask = batch.response_mask[indices]
                advantages = (
                    normalize_advantages(batch.advantages[indices], mask)
                    if objective == "ppo"
                    else batch.advantages[indices].detach()
                )
                tokens = int(mask.sum())
                sequences = len(indices)
                metrics: dict[str, float] = {}
                for start in range(0, sequences, microbatch_size):
                    rows = indices[start : start + microbatch_size]
                    current = token_logprobs(
                        actor,
                        batch.input_ids[rows].to(device),
                        batch.attention_mask[rows].to(device),
                    )
                    loss, info = loss_fn(
                        current,
                        batch.old_logprobs[rows].to(device),
                        batch.ref_logprobs[rows].to(device),
                        advantages[start : start + microbatch_size].to(device),
                        batch.response_mask[rows].to(device),
                        clip_eps=clip_eps,
                        kl_coef=kl_coef,
                        num_tokens=tokens,
                        num_sequences=sequences,
                    )
                    loss.backward()
                    for key, value in info.items():
                        metrics[key] = metrics.get(key, 0.0) + value.item()
                assert metrics["ratio_mean"] <= 10, (
                    "Mean policy ratio exceeds upstream safety threshold 10"
                )
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, max_grad_norm, error_if_nonfinite=True
                )
                metrics["grad_norm"] = norm.item()  # Before clipping.
                metrics["num_tokens"] = tokens
                metrics["num_sequences"] = sequences
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                history.append(metrics)
    finally:
        optimizer.zero_grad(set_to_none=True)
    return history

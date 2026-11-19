"""Analytic grouped objectives, masking, detachment, and accumulation contracts."""

from dataclasses import fields
from math import exp, log

import pytest
import torch

from algorithms.base import AlgorithmConfig
from algorithms.grpo import GRPOConfig, group_advantages, grpo_loss
from algorithms.rloo import RLOOConfig, rloo_loss


def test_config_and_population_groups():
    assert isinstance(GRPOConfig(), AlgorithmConfig)
    assert GRPOConfig().eps == 1e-6
    assert type(GRPOConfig()) is not type(RLOOConfig())
    assert not fields(RLOOConfig())
    rewards = torch.tensor([[1.0, 3.0, 5.0], [100.0, 100.0, 100.0]], requires_grad=True)
    before = rewards.detach().clone()
    result = group_advantages(rewards)
    expected = (before - before.mean(1, keepdim=True)) / (
        before.std(1, correction=0, keepdim=True) + 1e-6
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(group_advantages(rewards[:1]), result[:1])
    assert not result.requires_grad and rewards.grad is None
    assert torch.equal(rewards, before)
    assert torch.equal(group_advantages(rewards[:, :1]), torch.zeros(2, 1))
    assert torch.equal(result[1], torch.zeros(3))


@pytest.mark.parametrize("group_size", [1, 2, 3, 8, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_noninteger_ties_have_exact_zero_advantages_and_policy_gradients(
    group_size, dtype
):
    rewards = (
        torch.tensor([[-0.1], [0.1]], dtype=dtype)
        .repeat(1, group_size)
        .requires_grad_()
    )
    advantages = group_advantages(rewards)
    assert torch.equal(advantages, torch.zeros_like(rewards))
    assert not advantages.requires_grad

    current = (
        torch.linspace(-2.3, -1.7, rewards.numel() * 3, dtype=dtype)
        .reshape(-1, 3)
        .requires_grad_()
    )
    mask = torch.ones_like(current, dtype=torch.bool)
    mask[::2, 1:] = False
    loss, info = grpo_loss(
        current,
        torch.full_like(current, -2),
        torch.full_like(current, -1.5),
        advantages.reshape(-1, 1).expand_as(current),
        mask,
        kl_coef=0,
    )
    assert loss.item() == info["policy_loss"].item() == 0
    assert info["reference_kl"].item() > 0
    loss.backward()
    assert torch.equal(current.grad, torch.zeros_like(current))
    assert rewards.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_group_advantages_translation_invariance(dtype):
    rewards = torch.tensor([[-0.7, 0.1, 1.3, 2.4], [3.2, -1.1, 0.6, 0.2]], dtype=dtype)
    offsets = torch.tensor([[3.25], [-2.5]], dtype=dtype)
    expected = (rewards - rewards.mean(1, keepdim=True)) / (
        rewards.std(1, correction=0, keepdim=True) + 1e-6
    )
    original = group_advantages(rewards)
    translated = group_advantages(rewards + offsets)
    assert torch.isfinite(translated).all()
    torch.testing.assert_close(original, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(translated, original, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("eps", [0, -1, float("inf"), float("nan")])
def test_invalid_epsilon(eps):
    with pytest.raises(AssertionError):
        GRPOConfig(eps)
    with pytest.raises(AssertionError):
        group_advantages(torch.ones(1, 2), eps)


def test_grpo_equal_completion_weight_and_unclamped_reference():
    mask = torch.tensor([[True, False, False], [True, True, True]])
    current = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    old = -torch.tensor([[0.9, 1, 1], [1.1, 1.1, 1.1]], dtype=torch.float64).log()
    ref = torch.tensor([[log(2), 0, 0], [4.0, 4.0, 4.0]], dtype=torch.float64)
    adv = torch.tensor([[2.0, 0, 0], [-1.0, -1.0, -1.0]], dtype=torch.float64)
    loss, info = grpo_loss(current, old, ref, adv, mask, kl_coef=0.2)
    policy = (-0.9 * 2 + 1.1) / 2
    kl = (1 - log(2) + exp(4) - 5) / 2
    assert info["policy_loss"].item() == pytest.approx(policy)
    assert info["reference_kl"].item() == pytest.approx(kl)
    assert loss.item() == pytest.approx(policy + 0.2 * kl)
    loss.backward()
    assert current.grad[1, 0].item() == pytest.approx(1.1 / 6 + 0.2 * (1 - exp(4)) / 6)


@pytest.mark.parametrize("loss_fn", [grpo_loss, rloo_loss])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_ratio_clipping_signs_and_gradients(loss_fn, sign):
    current = (
        torch.tensor([[0.5], [1.0], [1.5]], dtype=torch.float64).log().requires_grad_()
    )
    old = torch.zeros_like(current, requires_grad=True)
    ref = torch.zeros_like(current, requires_grad=True)
    adv = torch.full_like(current, sign, requires_grad=True)
    loss, info = loss_fn(
        current, old, ref, adv, torch.ones_like(current, dtype=torch.bool), kl_coef=0
    )
    expected = [-0.5, -1.0, -1.2] if sign > 0 else [0.8, 1.0, 1.5]
    assert loss.item() == pytest.approx(sum(expected) / 3)
    assert info["ratio_mean"].item() == pytest.approx(1)
    assert info["clip_frac"].item() == pytest.approx(2 / 3)
    loss.backward()
    grad = [-0.5, -1.0, 0.0] if sign > 0 else [0.0, 1.0, 1.5]
    torch.testing.assert_close(
        current.grad[:, 0], torch.tensor(grad, dtype=torch.float64) / 3
    )
    assert all(x.grad is None for x in (old, ref, adv))


@pytest.mark.parametrize("loss_fn", [grpo_loss, rloo_loss])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_mask_poison_empty_rows_and_detached_targets(loss_fn, bad):
    mask = torch.tensor([[False, True, True, False], [False] * 4])
    values = [
        torch.tensor(
            [[bad, a, b, bad], [bad] * 4], dtype=torch.float64, requires_grad=True
        )
        for a, b in [(-1.1, -0.9), (-1.0, -1.0), (-0.8, -1.2), (2.0, 2.0)]
    ]
    clean = [x.detach()[:1, 1:3].clone().requires_grad_() for x in values]
    expected, expected_info = loss_fn(
        *clean, torch.ones((1, 2), dtype=torch.bool), kl_coef=0.3
    )
    loss, info = loss_fn(*values, mask, kl_coef=0.3)
    loss.backward()
    expected.backward()
    for key in info:
        torch.testing.assert_close(info[key], expected_info[key])
        assert not info[key].requires_grad
    torch.testing.assert_close(values[0].grad[:1, 1:3], clean[0].grad)
    assert (values[0].grad[~mask] == 0).all()
    assert all(x.grad is None for x in values[1:])
    with pytest.raises(ValueError, match="nonempty"):
        loss_fn(*values, torch.zeros_like(mask))


@pytest.mark.parametrize("loss_fn", [grpo_loss, rloo_loss])
@pytest.mark.parametrize("field", range(4))
def test_nonfinite_valid_data_rejected(loss_fn, field):
    values = [torch.zeros(1, 2) for _ in range(4)]
    values[field][0, 0] = float("nan")
    with pytest.raises(AssertionError, match="Nonfinite"):
        loss_fn(*values, torch.ones((1, 2), dtype=torch.bool))


@pytest.mark.parametrize("loss_fn", [grpo_loss, rloo_loss])
@pytest.mark.parametrize("field", [1, 2])
def test_finite_exponential_overflow_rejected_even_without_kl(loss_fn, field):
    values = [torch.zeros(1, 2) for _ in range(4)]
    values[field].fill_(-1000 if field == 1 else 1000)
    with pytest.raises(AssertionError, match="overflow"):
        loss_fn(*values, torch.ones((1, 2), dtype=torch.bool), kl_coef=0)


@pytest.mark.parametrize("loss_fn", [grpo_loss, rloo_loss])
def test_advantages_must_be_constant_and_clip_strict(loss_fn):
    scores = torch.zeros(1, 2)
    mask = torch.ones_like(scores, dtype=torch.bool)
    with pytest.raises(AssertionError, match="constant"):
        loss_fn(scores, scores, scores, torch.tensor([[1.0, 2.0]]), mask)
    for clip in [0, 1, -0.1, float("nan")]:
        with pytest.raises(AssertionError):
            loss_fn(scores, scores, scores, scores, mask, clip_eps=clip)


@pytest.mark.parametrize("loss_fn", [grpo_loss, rloo_loss])
@pytest.mark.parametrize("microbatch", [1, 2])
def test_loss_accumulation(loss_fn, microbatch):
    mask = torch.tensor([[True, False, False], [True, True, True], [True, True, False]])
    current = torch.tensor(
        [[-2.1, 0, 0], [-1.8, -2.3, -2.0], [-1.9, -2.2, 0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    old, ref = torch.full_like(current, -2), torch.full_like(current, -1.5)
    adv = torch.tensor([[2.0], [-1.0], [3.0]], dtype=torch.float64).expand_as(current)
    full, expected = loss_fn(current, old, ref, adv, mask, kl_coef=0.4)
    full.backward()
    split = current.detach().clone().requires_grad_()
    infos = []
    for start in range(0, 3, microbatch):
        rows = slice(start, start + microbatch)
        loss, info = loss_fn(
            split[rows],
            old[rows],
            ref[rows],
            adv[rows],
            mask[rows],
            kl_coef=0.4,
            num_sequences=3,
            num_tokens=6,
        )
        loss.backward()
        infos.append(info)
    torch.testing.assert_close(split.grad, current.grad)
    for key in expected:
        torch.testing.assert_close(sum(info[key] for info in infos), expected[key])

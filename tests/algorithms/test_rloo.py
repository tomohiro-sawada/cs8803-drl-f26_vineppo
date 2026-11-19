"""Leave-one-out baselines and sequence (not token) importance ratios."""

from math import exp, log

import pytest
import torch

from algorithms.base import AlgorithmConfig
from algorithms.rloo import RLOOConfig, rloo_advantages, rloo_loss


def test_leave_one_out_analytic_groups_detached_and_unscaled():
    assert isinstance(RLOOConfig(), AlgorithmConfig)
    rewards = torch.tensor([[1.0, 3.0, 8.0], [100.0, 100.0, 100.0]], requires_grad=True)
    before = rewards.detach().clone()
    result = rloo_advantages(rewards)
    torch.testing.assert_close(
        result, torch.tensor([[-4.5, -1.5, 6.0], [0.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(rloo_advantages(rewards[:1]), result[:1])
    torch.testing.assert_close(rloo_advantages(rewards * 2), result * 2)
    torch.testing.assert_close(
        rloo_advantages(torch.tensor([[2.0, 5.0]])), torch.tensor([[-3.0, 3.0]])
    )
    assert not result.requires_grad and rewards.grad is None
    assert torch.equal(rewards, before)


@pytest.mark.parametrize("group_size", [2, 3, 8, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_noninteger_ties_have_exact_zero_advantages_and_policy_gradients(
    group_size, dtype
):
    rewards = (
        torch.tensor([[-0.1], [0.1]], dtype=dtype)
        .repeat(1, group_size)
        .requires_grad_()
    )
    advantages = rloo_advantages(rewards)
    assert torch.equal(advantages, torch.zeros_like(rewards))
    assert not advantages.requires_grad

    current = (
        torch.linspace(-2.3, -1.7, rewards.numel() * 3, dtype=dtype)
        .reshape(-1, 3)
        .requires_grad_()
    )
    mask = torch.ones_like(current, dtype=torch.bool)
    mask[::2, 1:] = False
    loss, info = rloo_loss(
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
def test_rloo_advantages_translation_invariance(dtype):
    rewards = torch.tensor([[-0.7, 0.1, 1.3, 2.4], [3.2, -1.1, 0.6, 0.2]], dtype=dtype)
    offsets = torch.tensor([[3.25], [-2.5]], dtype=dtype)
    expected = rewards - (rewards.sum(1, keepdim=True) - rewards) / (
        rewards.shape[1] - 1
    )
    original = rloo_advantages(rewards)
    translated = rloo_advantages(rewards + offsets)
    assert torch.isfinite(translated).all()
    torch.testing.assert_close(original, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(translated, original, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "rewards",
    [
        torch.ones(2, 1),
        torch.ones(2, 0),
        torch.ones(0, 2),
        torch.ones(2),
        torch.ones(2, 2, dtype=torch.long),
        torch.tensor([[1.0, float("inf")]]),
    ],
)
def test_invalid_groups(rewards):
    with pytest.raises(AssertionError):
        rloo_advantages(rewards)


def test_sequence_ratios_and_summed_unclamped_reference():
    mask = torch.tensor([[True, False], [True, True]])
    current = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    old = -torch.tensor([[log(0.9), 0], [log(1.1), log(1.1)]], dtype=torch.float64)
    ref = torch.full_like(current, 4)
    adv = torch.tensor([[2.0, 2.0], [-1.0, -1.0]], dtype=torch.float64)
    loss, info = rloo_loss(current, old, ref, adv, mask, kl_coef=0.2)
    policy = (-0.9 * 2 + 1.21) / 2
    kl = 3 * (exp(4) - 5) / 2
    assert info["policy_loss"].item() == pytest.approx(policy)
    assert info["reference_kl"].item() == pytest.approx(kl)
    assert info["ratio_mean"].item() == pytest.approx((0.9 + 1.21) / 2)
    assert info["clip_frac"].item() == pytest.approx(0.5)
    assert info["approx_kl"].item() == pytest.approx(
        (log(0.9) ** 2 + log(1.21) ** 2) / 4
    )
    assert loss.item() == pytest.approx(policy + 0.2 * kl)
    loss.backward()
    torch.testing.assert_close(
        current.grad,
        torch.tensor(
            [[-0.9 + 0.1 * (1 - exp(4)), 0], [0.605 + 0.1 * (1 - exp(4))] * 2],
            dtype=torch.float64,
        ),
    )


def test_sequence_ratio_overflow_when_individual_ratios_are_finite():
    current = torch.full((1, 2), 50.0)
    zeros = torch.zeros_like(current)
    with pytest.raises(AssertionError, match="overflow"):
        rloo_loss(
            current, zeros, current, zeros, torch.ones_like(current, dtype=torch.bool)
        )

"""Hand-computed GAE and download-free critic optimization contracts."""

from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch
from transformers import GPT2Config, GPT2Model

from algorithms.base import AlgorithmConfig
from algorithms.ppo import (
    PPOConfig,
    TokenCritic,
    masked_gae,
    token_values,
    update_critic,
    value_loss,
)
from core.data import Episode, PPOValueEstimate, collate_critic


@pytest.mark.parametrize("field", ["gamma", "lam"])
@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_config_rejects_invalid_discount(field, bad):
    with pytest.raises(AssertionError):
        PPOConfig(**{field: bad})


def test_config_defaults_and_endpoints():
    config = PPOConfig()
    assert isinstance(config, AlgorithmConfig)
    assert (config.gamma, config.lam) == (1, 0.95)
    PPOConfig(gamma=0, lam=1)
    PPOConfig(gamma=1, lam=0)


@pytest.mark.parametrize(
    "gamma,lam,expected",
    [
        (1, 0, [1.5, 2.5, 1.5]),
        (1, 1, [5.5, 4, 1.5]),
        (0.5, 0, [1, 1.75, 1.5]),
        (0.5, 1, [2.25, 2.5, 1.5]),
        (0.5, 0.5, [1.53125, 2.125, 1.5]),
        (0, 0.95, [0.5, 1, 1.5]),
    ],
)
def test_hand_gae_terminal_zero_and_detached(gamma, lam, expected):
    rewards = torch.tensor([[1.0, 2, 3]], requires_grad=True)
    values = torch.tensor([[0.5, 1, 1.5]], requires_grad=True)
    before = [x.detach().clone() for x in (rewards, values)]
    advantages, returns = masked_gae(
        rewards, values, torch.ones_like(values, dtype=torch.bool), gamma=gamma, lam=lam
    )
    torch.testing.assert_close(advantages, torch.tensor([expected]))
    torch.testing.assert_close(returns, advantages + values.detach())
    assert not advantages.requires_grad and not returns.requires_grad
    for actual, original in zip((rewards, values), before):
        torch.testing.assert_close(actual, original)
        assert actual.grad is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_ragged_gae_padding_isolation(bad):
    mask = torch.tensor(
        [[False, True, True, True, False], [True, False, False, False, False]]
    )
    rewards = torch.full((2, 5), bad)
    values = torch.full((2, 5), bad)
    rewards[mask] = torch.tensor([1.0, 2, 3, 4])
    values[mask] = torch.tensor([0.5, 1, 1.5, 7])
    before = [x.clone() for x in (rewards, values)]
    advantages, returns = masked_gae(rewards, values, mask, gamma=0.5, lam=1)
    torch.testing.assert_close(advantages[mask], torch.tensor([2.25, 2.5, 1.5, -3]))
    torch.testing.assert_close(returns[mask], torch.tensor([2.75, 3.5, 3, 4]))
    assert (advantages[~mask] == 0).all() and (returns[~mask] == 0).all()
    for actual, original in zip((rewards, values), before):
        torch.testing.assert_close(actual, original, equal_nan=True)


@pytest.mark.parametrize(
    "mask",
    [
        [[False, False, False]],
        [[True, False, True]],
    ],
)
def test_gae_rejects_empty_or_discontinuous_responses(mask):
    with pytest.raises(AssertionError):
        masked_gae(torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor(mask))


@pytest.mark.parametrize("function", [masked_gae, value_loss])
@pytest.mark.parametrize(
    "invalid", ["nonfinite", "shape", "dtype", "mask_dtype", "device"]
)
def test_masked_primitives_validate_data(function, invalid):
    first, second = torch.zeros(1, 2), torch.ones(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    if invalid == "nonfinite":
        second[0, 0] = float("nan")
    elif invalid == "shape":
        second = second[:, :1]
    elif invalid == "dtype":
        first = first.long()
    elif invalid == "mask_dtype":
        mask = mask.long()
    else:
        second = second.to("meta")
    with pytest.raises(AssertionError):
        function(first, second, mask)


@pytest.mark.parametrize("microbatch_size", [1, 2])
def test_value_loss_detaches_targets_and_accumulates_token_mean(microbatch_size):
    mask = torch.tensor([[True, False, False], [True, True, True], [False, True, True]])
    values = (
        torch.arange(9, dtype=torch.float64)
        .reshape(3, 3)
        .masked_fill(~mask, float("nan"))
        .requires_grad_()
    )
    targets = torch.ones_like(values, requires_grad=True)
    loss = value_loss(values, targets, mask)
    assert loss.item() == pytest.approx(0.5 * (1 + 4 + 9 + 16 + 36 + 49) / 6)
    loss.backward()
    split = values.detach().clone().requires_grad_()
    losses = []
    for start in range(0, 3, microbatch_size):
        rows = slice(start, start + microbatch_size)
        part = value_loss(split[rows], targets[rows], mask[rows], num_tokens=6)
        part.backward()
        losses.append(part.detach())
    torch.testing.assert_close(sum(losses), loss)
    torch.testing.assert_close(split.grad, values.grad)
    assert (values.grad[~mask] == 0).all()
    assert targets.grad is None
    with pytest.raises(AssertionError):
        value_loss(values, targets, mask, num_tokens=5)


@pytest.fixture
def local_batch():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            critic = TokenCritic(
                GPT2Model(
                    GPT2Config(
                        vocab_size=23,
                        n_positions=16,
                        n_embd=16,
                        n_layer=1,
                        n_head=2,
                        resid_pdrop=0.5,
                        embd_pdrop=0.5,
                        attn_pdrop=0.5,
                        attn_implementation="sdpa",
                    )
                )
            )
        episodes = [
            Episode([1], [4], 1, "length"),
            Episode([1, 6, 7], [8, 9, 10], 0, "length"),
            Episode([1, 3], [11, 12, 13, 2], 1, "stop"),
            Episode([1, 4, 5, 6], [14, 2], 0, "stop"),
            Episode([1, 8], [15, 16, 2], 1, "stop"),
        ]
        batch = collate_critic(
            [
                PPOValueEstimate(
                    str(i),
                    episode,
                    [0.0] * len(episode.response_ids),
                    [float(i + j) / 3 for j in range(len(episode.response_ids))],
                )
                for i, episode in enumerate(episodes)
            ],
            pad_token_id=0,
        )
        yield critic, batch
    finally:
        torch.set_num_threads(threads)


def test_token_values_alignment_padding_dropout_and_autograd(local_batch):
    critic, batch = local_batch
    assert critic.config is critic.backbone.config
    scores = token_values(critic, batch.input_ids, batch.attention_mask)
    assert scores.shape == batch.response_mask.shape and scores.dtype == torch.float32
    assert scores.requires_grad and not critic.training and not critic.backbone.training
    torch.testing.assert_close(
        scores, token_values(critic, batch.input_ids, batch.attention_mask)
    )
    for row in range(len(batch.input_ids)):
        length = int(batch.attention_mask[row].sum())
        ids = batch.input_ids[row : row + 1, :length]
        attention = torch.ones_like(ids, dtype=torch.bool)
        hidden = critic.backbone(
            input_ids=ids, attention_mask=attention, use_cache=False
        ).last_hidden_state
        expected = critic.value_head(hidden).squeeze(-1)[:, :-1]
        torch.testing.assert_close(scores[row : row + 1, : length - 1], expected)
        left_ids = torch.cat([torch.zeros(1, 2, dtype=torch.long), ids], dim=1)
        left_mask = torch.cat([torch.zeros(1, 2, dtype=torch.bool), attention], dim=1)
        torch.testing.assert_close(
            token_values(critic, left_ids, left_mask)[:, 2:], expected
        )
    scores.retain_grad()
    value_loss(scores, batch.returns, batch.response_mask).backward()
    assert (scores.grad[~batch.response_mask] == 0).all()
    assert critic.backbone.wte.weight.grad.abs().sum() > 0
    with torch.no_grad():
        assert not token_values(
            critic, batch.input_ids, batch.attention_mask
        ).requires_grad


@pytest.mark.parametrize("microbatch_size", [1, 2])
def test_update_matches_full_batches_fixed_targets_and_clipping(
    local_batch, microbatch_size
):
    critic, batch = local_batch
    manual = deepcopy(critic)
    before = deepcopy(critic.state_dict())
    batch_before = {f.name: getattr(batch, f.name).clone() for f in fields(batch)}
    batch.returns.requires_grad_()
    optimizer = torch.optim.SGD(critic.parameters(), lr=0.02)
    manual_optimizer = torch.optim.SGD(manual.parameters(), lr=0.02)
    for parameter in critic.parameters():
        parameter.grad = torch.ones_like(parameter)
    history = update_critic(
        critic,
        optimizer,
        batch,
        epochs=2,
        minibatch_size=3,
        microbatch_size=microbatch_size,
        max_grad_norm=0.01,
        generator=torch.Generator().manual_seed(19),
    )
    expected = []
    generator = torch.Generator().manual_seed(19)
    for _ in range(2):
        for rows in torch.randperm(5, generator=generator).split(3):
            mask = batch.response_mask[rows]
            scores = token_values(
                manual, batch.input_ids[rows], batch.attention_mask[rows]
            )
            # Independent full-minibatch regression against the ORIGINAL targets.
            loss = (
                0.5
                * (scores[mask] - batch_before["returns"][rows][mask]).square().mean()
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                manual.parameters(), 0.01, error_if_nonfinite=True
            )
            expected.append(
                {
                    "value_loss": loss.item(),
                    "grad_norm": norm.item(),
                    "num_tokens": int(mask.sum()),
                    "num_sequences": len(rows),
                }
            )
            manual_optimizer.step()
            manual_optimizer.zero_grad(set_to_none=True)
    assert len(history) == 4
    assert [h["num_sequences"] for h in history] == [3, 2, 3, 2]
    assert all(h["grad_norm"] > 0.01 for h in history)
    for actual, wanted in zip(history, expected):
        assert actual == pytest.approx(wanted, abs=2e-6, rel=2e-5)
    for actual, wanted in zip(critic.parameters(), manual.parameters()):
        torch.testing.assert_close(actual, wanted, atol=2e-7, rtol=2e-5)
        assert actual.grad is None
    assert not torch.equal(critic.backbone.wte.weight, before["backbone.wte.weight"])
    assert not torch.equal(critic.value_head.weight, before["value_head.weight"])
    assert batch.returns.grad is None and not critic.training
    for name, original in batch_before.items():
        torch.testing.assert_close(getattr(batch, name), original, atol=0, rtol=0)


@pytest.mark.parametrize(
    "invalid",
    ["duplicate", "foreign", "missing", "shape", "attention", "nonfinite", "epochs"],
)
def test_update_validation_clears_gradients(local_batch, invalid):
    critic, batch = local_batch
    optimizer = torch.optim.SGD(critic.parameters(), lr=0.01)
    epochs = 1
    if invalid == "duplicate":
        optimizer.param_groups[0]["params"].append(next(critic.parameters()))
    elif invalid == "foreign":
        optimizer.param_groups[0]["params"].append(torch.nn.Parameter(torch.ones(1)))
    elif invalid == "missing":
        optimizer.param_groups[0]["params"].pop()
    elif invalid == "shape":
        batch = replace(batch, returns=batch.returns[:, :-1])
    elif invalid == "attention":
        batch.attention_mask[0, 0] = False
    elif invalid == "nonfinite":
        batch.returns[0, 0] = float("nan")
    else:
        epochs = 0
    for parameter in critic.parameters():
        parameter.grad = torch.ones_like(parameter)
    with pytest.raises(AssertionError):
        update_critic(
            critic, optimizer, batch, epochs=epochs, minibatch_size=3, microbatch_size=1
        )
    assert all(p.grad is None for p in critic.parameters())


def test_critic_update_rejects_bf16_trainable_parameters(local_batch):
    critic, batch = local_batch
    critic.bfloat16()
    optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-6)
    with pytest.raises(AssertionError, match="Trainable parameter .* requires FP32"):
        update_critic(
            critic, optimizer, batch, epochs=1, minibatch_size=3, microbatch_size=1
        )


def test_update_clears_partial_accumulation_on_forward_failure(
    local_batch, monkeypatch
):
    critic, batch = local_batch
    optimizer = torch.optim.SGD(critic.parameters(), lr=0.01)
    original = critic.forward
    calls = 0

    def failing_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert any(p.grad is not None for p in critic.parameters())
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(critic, "forward", failing_forward)
    before = deepcopy(critic.state_dict())
    with pytest.raises(RuntimeError, match="injected failure"):
        update_critic(
            critic, optimizer, batch, epochs=1, minibatch_size=5, microbatch_size=1
        )
    assert all(p.grad is None for p in critic.parameters())
    for name, value in critic.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)

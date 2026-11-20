"""Fixed-data actor numerics and local, download-free optimizer contracts."""

from copy import deepcopy
from dataclasses import fields
from math import log, sqrt

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from algorithms.actor import actor_loss, normalize_advantages, update_actor
from algorithms.grpo import grpo_loss
from algorithms.rloo import rloo_loss
from core.data import EpisodeWithAdvantages, collate
from core.inference import token_logprobs


def test_clipping_values_and_gradients_for_both_advantage_signs():
    # Six externally normalized advantages (zero mean, unit sample variance).
    a = sqrt(5 / 6)
    advantages = torch.tensor([[a] * 3, [-a] * 3], dtype=torch.float64)
    current = (
        torch.tensor([[0.5, 1, 1.5]] * 2, dtype=torch.float64).log().requires_grad_()
    )
    old = torch.zeros_like(current, requires_grad=True)
    ref = torch.zeros_like(current, requires_grad=True)
    advantages.requires_grad_()
    loss, info = actor_loss(
        current,
        old,
        ref,
        advantages,
        torch.ones_like(current, dtype=torch.bool),
        kl_coef=0,
    )
    # Positive: [0.5, 1, 1.2]; negative: [0.8, 1, 1.5].
    assert loss.item() == pytest.approx(a * 0.6 / 6)
    assert info["clip_frac"].item() == pytest.approx(4 / 6)
    assert info["ratio_mean"].item() == pytest.approx(1)
    loss.backward()
    torch.testing.assert_close(
        current.grad,
        a / 6 * torch.tensor([[-0.5, -1, 0], [0, 1, 1.5]], dtype=torch.float64),
    )
    assert all(x.grad is None for x in (old, ref, advantages))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_masked_nonfinite_scores_and_targets_are_ignored(bad):
    mask = torch.tensor([[False, True, True, False]])
    current = torch.tensor([[bad, -1.1, -0.9, bad]], requires_grad=True)
    old = torch.tensor([[bad, -1, -1, bad]], requires_grad=True)
    ref = torch.tensor([[bad, -0.8, -1.2, bad]], requires_grad=True)
    advantages = torch.tensor([[bad, -sqrt(0.5), sqrt(0.5), bad]], requires_grad=True)
    clean = [x.detach()[:, 1:3].clone() for x in (current, old, ref, advantages)]
    clean[0].requires_grad_()
    expected, expected_info = actor_loss(
        *clean, torch.ones((1, 2), dtype=torch.bool), kl_coef=0.3
    )
    loss, info = actor_loss(current, old, ref, advantages, mask, kl_coef=0.3)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    for key in info:
        torch.testing.assert_close(info[key], expected_info[key])
        assert not info[key].requires_grad
    torch.testing.assert_close(current.grad[:, 1:3], clean[0].grad)
    assert torch.equal(current.grad[~mask], torch.zeros(2))
    assert all(x.grad is None for x in (old, ref, advantages))


def test_unequal_lengths_policy_token_mean_kl_sequence_mean_and_clamp_gradients():
    mask = torch.tensor([[True, False, False], [True, True, True]])
    current = torch.full((2, 3), -5.0, dtype=torch.float64, requires_grad=True)
    ratios = torch.tensor([[0.9, 1, 1], [1.1, 1.1, 1.1]], dtype=torch.float64)
    old = (current.detach() - ratios.log()).requires_grad_()
    d = torch.tensor([[log(2), 0, 0], [-log(2), 4, -12]], dtype=torch.float64)
    ref = (current.detach() + d).requires_grad_()
    advantages = torch.tensor(
        [[-0.5, 0, 0], [-0.5, -0.5, 1.5]], dtype=torch.float64, requires_grad=True
    )
    loss, info = actor_loss(current, old, ref, advantages, mask, kl_coef=0.2)
    policy = -(-0.5 * 0.9 + 0.5 * 1.1) / 4
    kl = ((1 - log(2)) + (log(2) - 0.5) + 10 + 10) / 2
    assert info["policy_loss"].item() == pytest.approx(policy)
    assert info["reference_kl"].item() == pytest.approx(kl)
    assert info["kl_loss"].item() == pytest.approx(0.2 * kl)
    assert loss.item() == pytest.approx(policy + 0.2 * kl)
    loss.backward()
    # Both large positive and negative d saturate; only policy gradients remain.
    expected = torch.tensor(
        [[0.45 / 4 - 0.1, 0, 0], [0.55 / 4 + 0.05, 0.55 / 4, -1.65 / 4]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(current.grad, expected)
    assert all(x.grad is None for x in (old, ref, advantages))


def test_normalization_uses_expanded_tokens_and_sample_variance():
    raw = torch.tensor(
        [[1.0, 1, 1, 3, float("nan")]], dtype=torch.float64, requires_grad=True
    )
    mask = torch.tensor([[True, True, True, True, False]])
    before = raw.detach().clone()
    result = normalize_advantages(raw, mask)
    expected = torch.tensor([[-0.5, -0.5, -0.5, 1.5, 0]], dtype=torch.float64) / sqrt(
        1 + 1e-8
    )
    torch.testing.assert_close(result, expected)
    assert not result.requires_grad
    # Whitening the two steps first would instead produce +/- 1/sqrt(2).
    steps = normalize_advantages(
        torch.tensor([[1.0, 3.0]]), torch.ones((1, 2), dtype=torch.bool)
    )
    assert not torch.allclose(result[0, :4].float(), steps[0, [0, 0, 0, 1]])
    torch.testing.assert_close(raw, before, equal_nan=True)
    assert raw.grad is None


def test_constant_advantages_are_zero_and_single_token_is_rejected():
    raw = torch.tensor([[7.0, 7, float("inf")]], requires_grad=True)
    torch.testing.assert_close(
        normalize_advantages(raw, torch.tensor([[True, True, False]])),
        torch.zeros_like(raw),
    )
    with pytest.raises(AssertionError, match="at least two response tokens"):
        normalize_advantages(raw, torch.tensor([[True, False, False]]))


@pytest.mark.parametrize("microbatch_size", [1, 2])
def test_loss_accumulation_preserves_separate_denominators(microbatch_size):
    mask = torch.tensor([[True, False, False], [True, True, True], [True, True, False]])
    current = torch.tensor(
        [[-2.1, 0, 0], [-1.8, -2.3, -2.0], [-1.9, -2.2, 0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    old = torch.full_like(current, -2)
    ref = torch.full_like(current, -1.5)
    advantages = normalize_advantages(
        torch.arange(9, dtype=torch.float64).reshape(3, 3), mask
    )
    full, full_info = actor_loss(current, old, ref, advantages, mask, kl_coef=0.4)
    full.backward()
    split = current.detach().clone().requires_grad_()
    infos = []
    for start in range(0, 3, microbatch_size):
        rows = slice(start, start + microbatch_size)
        loss, info = actor_loss(
            split[rows],
            old[rows],
            ref[rows],
            advantages[rows],
            mask[rows],
            kl_coef=0.4,
            num_tokens=6,
            num_sequences=3,
        )
        loss.backward()
        infos.append(info)
    torch.testing.assert_close(split.grad, current.grad)
    for key in full_info:
        torch.testing.assert_close(sum(info[key] for info in infos), full_info[key])


@pytest.fixture
def local_batch():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            actor = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=23,
                    n_positions=16,
                    n_embd=16,
                    n_layer=1,
                    n_head=2,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                    resid_pdrop=0.5,
                    embd_pdrop=0.5,
                    attn_pdrop=0.5,
                    attn_implementation="sdpa",
                )
            )
        reference = deepcopy(actor)
        with torch.no_grad():
            reference.transformer.wte.weight.mul_(0.7)
        episodes = [
            EpisodeWithAdvantages([1], [4, 5], 1, "length", [1, 3]),
            EpisodeWithAdvantages([1, 6, 7], [8, 9, 10], 0, "length", [-2, -2, 4]),
            EpisodeWithAdvantages([1, 3], [11, 12, 13, 2], 1, "stop", [1, 1, 1, 3]),
            EpisodeWithAdvantages([1, 4, 5, 6], [14, 2], 0, "stop", [-5, 2]),
            EpisodeWithAdvantages([1, 8], [15, 16, 2], 1, "stop", [2, 6, 6]),
        ]
        dummy = [[0.0] * len(e.response_ids) for e in episodes]
        layout = collate(
            episodes, old_logprobs=dummy, ref_logprobs=dummy, pad_token_id=0
        )
        with torch.no_grad():
            old = token_logprobs(actor, layout.input_ids, layout.attention_mask)
            ref = token_logprobs(reference, layout.input_ids, layout.attention_mask)
        batch = collate(
            episodes,
            old_logprobs=[
                row[mask].tolist() for row, mask in zip(old, layout.response_mask)
            ],
            ref_logprobs=[
                row[mask].tolist() for row, mask in zip(ref, layout.response_mask)
            ],
            pad_token_id=0,
        )
        yield actor, reference, batch
    finally:
        torch.set_num_threads(threads)


def test_variable_length_causal_alignment_and_loss_mask(local_batch):
    actor, _, batch = local_batch
    current = token_logprobs(actor, batch.input_ids, batch.attention_mask)
    current.retain_grad()
    # Independent next-token cross entropy, one unpadded sequence at a time.
    for row in range(len(batch.input_ids)):
        length = int(batch.attention_mask[row].sum())
        ids = batch.input_ids[row : row + 1, :length]
        logits = actor(input_ids=ids, use_cache=False).logits[0, :-1].float()
        expected = -torch.nn.functional.cross_entropy(
            logits, ids[0, 1:], reduction="none"
        )
        torch.testing.assert_close(current[row, : length - 1], expected)
    assert batch.response_mask[0].nonzero().flatten().tolist() == [0, 1]
    assert batch.response_mask[1].nonzero().flatten().tolist() == [2, 3, 4]
    loss, _ = actor_loss(
        current,
        batch.old_logprobs,
        batch.ref_logprobs,
        normalize_advantages(batch.advantages, batch.response_mask),
        batch.response_mask,
    )
    loss.backward()
    assert (current.grad[~batch.response_mask] == 0).all()
    assert (current.grad[batch.response_mask] != 0).any()


def test_duplicate_optimizer_parameters_rejected(local_batch):
    actor, _, batch = local_batch
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)
    optimizer.param_groups[0]["params"].append(next(actor.parameters()))
    with pytest.raises(AssertionError, match="Duplicate optimizer parameters"):
        update_actor(
            actor, optimizer, batch, epochs=1, minibatch_size=5, microbatch_size=2
        )


def record_step_gradients(optimizer):
    gradients = []
    optimizer.register_step_pre_hook(
        lambda opt, args, kwargs: gradients.append(
            torch.cat(
                [
                    p.grad.detach().flatten().clone()
                    for group in opt.param_groups
                    for p in group["params"]
                ]
            )
        )
    )
    return gradients


@pytest.mark.parametrize("objective", ["ppo", "grpo", "rloo"])
@pytest.mark.parametrize("microbatch_size", [1, 2])
def test_update_matches_full_minibatches_with_epochs_partial_batches_and_clipping(
    local_batch, microbatch_size, objective, monkeypatch
):
    actor, reference, batch = local_batch
    loss_fn = {"ppo": actor_loss, "grpo": grpo_loss, "rloo": rloo_loss}[objective]
    if objective != "ppo":
        batch.advantages.copy_(
            torch.tensor([[2.0], [-1.0], [3.0], [-2.0], [1.0]]).expand_as(
                batch.advantages
            )
        )

        def forbidden_whitening(*args, **kwargs):
            pytest.fail("Grouped objectives must not whiten advantages")

        monkeypatch.setattr(
            "algorithms.actor.normalize_advantages", forbidden_whitening
        )
    manual = deepcopy(actor)
    before = {f.name: getattr(batch, f.name).clone() for f in fields(batch)}
    reference_before = deepcopy(reference.state_dict())
    actor_before = deepcopy(actor.state_dict())
    for target in (batch.old_logprobs, batch.ref_logprobs, batch.advantages):
        target.requires_grad_()
    current = token_logprobs(actor, batch.input_ids, batch.attention_mask)
    torch.testing.assert_close(
        (current - batch.old_logprobs)[batch.response_mask].exp(),
        torch.ones(int(batch.response_mask.sum())),
    )
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.02)
    manual_optimizer = torch.optim.SGD(manual.parameters(), lr=0.02)
    actual_grads = record_step_gradients(optimizer)
    expected_grads = record_step_gradients(manual_optimizer)
    # Stale gradients must be discarded before the first minibatch.
    for p in actor.parameters():
        p.grad = torch.ones_like(p)
    max_norm = 0.01
    history = update_actor(
        actor,
        optimizer,
        batch,
        epochs=2,
        minibatch_size=3,
        microbatch_size=microbatch_size,
        kl_coef=0.3,
        max_grad_norm=max_norm,
        generator=torch.Generator().manual_seed(19),
        objective=objective,
    )
    expected_history = []
    generator = torch.Generator().manual_seed(19)
    for _ in range(2):
        for rows in torch.randperm(5, generator=generator).split(3):
            mask = batch.response_mask[rows]
            raw = batch.advantages.detach()[rows]
            normalized = torch.zeros_like(raw)
            values = raw[mask]
            normalized[mask] = (values - values.mean()) / (
                values.var(correction=1) + 1e-8
            ).sqrt()
            scores = token_logprobs(
                manual, batch.input_ids[rows], batch.attention_mask[rows]
            )
            loss, info = loss_fn(
                scores,
                batch.old_logprobs[rows],
                batch.ref_logprobs[rows],
                normalized if objective == "ppo" else raw,
                mask,
                kl_coef=0.3,
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                manual.parameters(), max_norm, error_if_nonfinite=True
            )
            expected_history.append(
                {
                    **{k: v.item() for k, v in info.items()},
                    "grad_norm": norm.item(),
                    "num_tokens": int(mask.sum()),
                    "num_sequences": len(rows),
                }
            )
            manual_optimizer.step()
            manual_optimizer.zero_grad(set_to_none=True)
    assert len(history) == len(actual_grads) == len(expected_grads) == 4
    assert [h["num_sequences"] for h in history] == [3, 2, 3, 2]
    assert sum(h["num_tokens"] for h in history) == 2 * int(batch.response_mask.sum())
    assert history[0]["ratio_mean"] == pytest.approx(1)
    assert all(h["grad_norm"] > max_norm for h in history)
    for actual, expected, grad, expected_grad in zip(
        history, expected_history, actual_grads, expected_grads
    ):
        assert actual == pytest.approx(expected, abs=2e-6, rel=2e-5)
        torch.testing.assert_close(grad, expected_grad, atol=2e-7, rtol=2e-4)
        assert grad.norm().item() == pytest.approx(max_norm, abs=1e-6)
    for actual, expected in zip(actor.parameters(), manual.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        assert actual.grad is None
    assert not actor.training
    assert any(
        not torch.equal(v, actor_before[k]) for k, v in actor.state_dict().items()
    )
    with torch.no_grad():
        after = token_logprobs(actor, batch.input_ids, batch.attention_mask)
    assert not torch.equal(
        after[batch.response_mask], batch.old_logprobs[batch.response_mask]
    )
    for name, value in before.items():
        torch.testing.assert_close(getattr(batch, name), value, atol=0, rtol=0)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, reference_before[name], atol=0, rtol=0)
    assert all(p.grad is None for p in reference.parameters())
    assert all(
        t.grad is None
        for t in (batch.old_logprobs, batch.ref_logprobs, batch.advantages)
    )


@pytest.mark.parametrize("objective", ["grpo", "rloo"])
def test_grouped_single_token_minibatch_without_whitening(
    local_batch, objective, monkeypatch
):
    actor, _, _ = local_batch
    episode = EpisodeWithAdvantages([1], [4], 1, "length", [2.0])
    batch = collate(
        [episode], old_logprobs=[[-3.0]], ref_logprobs=[[-2.8]], pad_token_id=0
    )
    before = {f.name: getattr(batch, f.name).clone() for f in fields(batch)}
    loss_fn = {"grpo": grpo_loss, "rloo": rloo_loss}[objective]
    expected, _ = loss_fn(
        token_logprobs(actor, batch.input_ids, batch.attention_mask),
        batch.old_logprobs,
        batch.ref_logprobs,
        batch.advantages,
        batch.response_mask,
    )

    def forbidden_whitening(*args, **kwargs):
        pytest.fail("Single-token grouped minibatch must not normalize")

    monkeypatch.setattr("algorithms.actor.normalize_advantages", forbidden_whitening)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)
    history = update_actor(
        actor,
        optimizer,
        batch,
        epochs=2,
        minibatch_size=1,
        microbatch_size=1,
        objective=objective,
    )
    assert history[0]["loss"] == pytest.approx(expected.item())
    assert history[0]["grad_norm"] > 0
    assert len(history) == 2
    for name, value in before.items():
        assert torch.equal(getattr(batch, name), value)

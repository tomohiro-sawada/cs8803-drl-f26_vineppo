from copy import deepcopy
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from core.data import (
    CriticBatch,
    Episode,
    EpisodeWithAdvantages,
    Example,
    PolicyGradientBatch,
    PPOValueEstimate,
    TrainBatch,
    VinePPOValueEstimate,
    collate,
    collate_critic,
)


def episode(**overrides):
    values = {
        "prompt_ids": [11, 12],
        "response_ids": [21, 22],
        "reward": 1.0,
        "finish_reason": "stop",
        "advantages": [0.25, 0.5],
    }
    return EpisodeWithAdvantages(**(values | overrides))


def assert_batch(batch, episodes):
    assert isinstance(batch, PolicyGradientBatch)
    batch_size = len(episodes)
    width = max(len(e.prompt_ids) + len(e.response_ids) for e in episodes)
    dtypes = {
        "input_ids": torch.int64,
        "attention_mask": torch.bool,
        "response_mask": torch.bool,
        "old_logprobs": torch.float32,
        "ref_logprobs": torch.float32,
        "advantages": torch.float32,
    }
    for name, dtype in dtypes.items():
        value = getattr(batch, name)
        assert isinstance(value, torch.Tensor)
        assert value.shape == (
            batch_size,
            width if name in ("input_ids", "attention_mask") else width - 1,
        )
        assert value.dtype == dtype
        assert value.device.type == "cpu"
        assert not value.requires_grad
        assert value.grad_fn is None
    for i, e in enumerate(episodes):
        prompt_len, response_len = len(e.prompt_ids), len(e.response_ids)
        assert batch.attention_mask[i].tolist() == [True] * (
            prompt_len + response_len
        ) + [False] * (width - prompt_len - response_len)
        assert batch.response_mask[i].tolist() == [False] * (prompt_len - 1) + [
            True
        ] * response_len + [False] * (width - prompt_len - response_len)
        assert batch.attention_mask[i].sum().item() == prompt_len + response_len
        assert batch.response_mask[i].sum().item() == response_len
    for name in ("old_logprobs", "ref_logprobs", "advantages"):
        assert (
            torch.count_nonzero(getattr(batch, name)[~batch.response_mask]).item() == 0
        )


def test_episode_stage_has_no_advantages():
    result = Episode(
        prompt_ids=[11], response_ids=[21], reward=1.0, finish_reason="stop"
    )
    assert {f.name for f in fields(result)} == {
        "prompt_ids",
        "response_ids",
        "reward",
        "finish_reason",
    }
    for name in (
        "advantages",
        "raw_step_advantages",
        "step_boundaries",
        "mc_prefix_values",
    ):
        assert not hasattr(result, name)
    assert isinstance(episode(), Episode)


def test_valid_episode_with_advantages():
    prompt, response, advantages = [11, 12], [21, 22], [0.25, 0.5]
    result = episode(
        prompt_ids=prompt,
        response_ids=response,
        advantages=advantages,
    )
    assert result.prompt_ids is prompt
    assert result.response_ids is response
    assert result.advantages is advantages
    assert {f.name for f in fields(result)} == {
        "prompt_ids",
        "response_ids",
        "reward",
        "finish_reason",
        "advantages",
    }


@pytest.mark.parametrize(
    "boundaries,mc_values,raw",
    [((0, 2), (0.75,), (0.25,)), ((0, 1, 2), (0.75, 0.5), (-0.25, 0.5))],
    ids=["single-step", "multi-step"],
)
def test_vineppo_estimate_keeps_step_diagnostics(boundaries, mc_values, raw):
    source = Episode([11, 12], [21, 22], 1.0, "stop")
    result = VinePPOValueEstimate("id", source, boundaries, mc_values, raw)
    assert result.episode_id == "id"
    assert result.episode is source
    assert result.step_boundaries == boundaries
    assert result.mc_prefix_values == mc_values
    assert result.raw_step_advantages == raw


RECORDS = [
    Example("id", "question", "answer"),
    Episode([11], [21], 1.0, "stop"),
    episode(),
    VinePPOValueEstimate(
        "id", Episode([11], [21], 1.0, "stop"), (0, 1), (0.75,), (0.25,)
    ),
    TrainBatch(*(torch.zeros(1, 1) for _ in range(3))),
    PolicyGradientBatch(*(torch.zeros(1, 1) for _ in range(6))),
]


@pytest.mark.parametrize(
    "record,name",
    [(record, f.name) for record in RECORDS for f in fields(record)],
    ids=[
        f"{type(record).__name__}.{f.name}"
        for record in RECORDS
        for f in fields(record)
    ],
)
def test_records_reject_field_reassignment(record, name):
    with pytest.raises(FrozenInstanceError):
        setattr(record, name, getattr(record, name))


@pytest.mark.parametrize(
    "advantages",
    [[], [0.25], [0.25, 0.5, 0.75]],
    ids=["empty", "short", "long"],
)
def test_episode_rejects_advantage_length_mismatch(advantages):
    with pytest.raises(AssertionError):
        episode(advantages=advantages)


def test_collate_exact_two_rows_and_no_input_mutation():
    episodes = [
        episode(),
        episode(
            prompt_ids=[31],
            response_ids=[41],
            advantages=[-0.75],
        ),
    ]
    old, ref = [[-0.125, -0.25], [-0.5]], [[-1.25, -1.5], [-1.75]]
    snapshot = deepcopy((episodes, old, ref))
    retained = [(e.prompt_ids, e.response_ids, e.advantages) for e in episodes]
    batch = collate(episodes, old_logprobs=old, ref_logprobs=ref, pad_token_id=99)
    assert_batch(batch, episodes)
    expected = {
        "input_ids": [[11, 12, 21, 22], [31, 41, 99, 99]],
        "attention_mask": [[True, True, True, True], [True, True, False, False]],
        "response_mask": [[False, True, True], [True, False, False]],
        "old_logprobs": [[0.0, -0.125, -0.25], [-0.5, 0.0, 0.0]],
        "ref_logprobs": [[0.0, -1.25, -1.5], [-1.75, 0.0, 0.0]],
        "advantages": [[0.0, 0.25, 0.5], [-0.75, 0.0, 0.0]],
    }
    for name, values in expected.items():
        assert getattr(batch, name).tolist() == values
    assert (episodes, old, ref) == snapshot
    for e, originals in zip(episodes, retained):
        for name, original in zip(
            ("prompt_ids", "response_ids", "advantages"),
            originals,
        ):
            assert getattr(e, name) is original


@pytest.mark.parametrize(
    "prompts,responses",
    [
        ([[11]], [[21]]),
        ([[11, 12]], [[21, 22]]),
        ([[99], [31, 32]], [[99], [41, 99]]),
        ([[11, 12], [31]], [[21], [41, 42]]),
    ],
    ids=[
        "minimum-singleton",
        "singleton",
        "real-pad-token",
        "equal-total-different-prompts",
    ],
)
def test_collate_edges(prompts, responses):
    episodes = tuple(
        episode(
            prompt_ids=p,
            response_ids=r,
            advantages=[0.25 * (j + 1) for j in range(len(r))],
        )
        for p, r in zip(prompts, responses)
    )
    old = tuple(
        tuple(-float(i + j + 1) for j in range(len(r))) for i, r in enumerate(responses)
    )
    ref = tuple(tuple(value - 4.0 for value in row) for row in old)
    batch = collate(episodes, old_logprobs=old, ref_logprobs=ref, pad_token_id=99)
    assert_batch(batch, episodes)
    width = batch.input_ids.shape[1]
    for i, e in enumerate(episodes):
        tokens = e.prompt_ids + e.response_ids
        assert batch.input_ids[i].tolist() == tokens + [99] * (width - len(tokens))
        for name, values in (
            ("old_logprobs", old[i]),
            ("ref_logprobs", ref[i]),
            ("advantages", e.advantages),
        ):
            assert getattr(batch, name)[i, batch.response_mask[i]].tolist() == list(
                values
            )


def test_collate_rejects_empty_batch():
    with pytest.raises(AssertionError):
        collate([], old_logprobs=[], ref_logprobs=[], pad_token_id=99)


@pytest.mark.parametrize("name", ["old_logprobs", "ref_logprobs"])
@pytest.mark.parametrize(
    "scores",
    [
        [],
        [[-1.0, -2.0]] * 3,
        [[], [-1.0, -2.0]],
        [[-1.0, -2.0], [-1.0]],
        [[-1.0, -2.0], [-1.0, -2.0, -3.0]],
    ],
    ids=["outer-short", "outer-long", "inner-empty", "inner-short", "inner-long"],
)
def test_collate_rejects_score_length_mismatch(name, scores):
    kwargs = {
        "old_logprobs": [[-1.0, -2.0], [-3.0, -4.0]],
        "ref_logprobs": [[-5.0, -6.0], [-7.0, -8.0]],
    }
    kwargs[name] = scores
    with pytest.raises(AssertionError):
        collate([episode(), episode()], **kwargs, pad_token_id=99)


@pytest.mark.parametrize(
    "name,replacement",
    [
        ("prompt_ids", []),
        ("response_ids", []),
        ("response_ids", [21]),
        ("response_ids", [21, 22, 23]),
        ("advantages", []),
        ("advantages", [0.25]),
        ("advantages", [0.25, 0.5, 0.75]),
    ],
    ids=[
        "empty-prompt",
        "empty-response",
        "response-short",
        "response-long",
        "adv-empty",
        "adv-short",
        "adv-long",
    ],
)
def test_collate_revalidates_mutated_lists(name, replacement):
    invalid = episode()
    getattr(invalid, name)[:] = replacement

    episodes = [episode(), invalid]
    old = [[-1.0] * len(e.response_ids) for e in episodes]
    ref = [[-2.0] * len(e.response_ids) for e in episodes]
    snapshot = deepcopy((episodes, old, ref))
    with pytest.raises(AssertionError):
        collate(episodes, old_logprobs=old, ref_logprobs=ref, pad_token_id=99)
    assert (episodes, old, ref) == snapshot


@pytest.mark.parametrize(
    "prompts,responses",
    [
        ([[11, 12], [31]], [[21, 22], [41]]),
        ([[11]], [[21]]),
        ([[99], [31, 32]], [[99], [41, 99]]),
        ([[11, 12], [31]], [[21], [41, 42]]),
    ],
    ids=["mixed-lengths", "minimum-singleton", "real-pad-token", "equal-totals"],
)
def test_collate_critic_layout_and_targets(prompts, responses):
    estimates = tuple(
        PPOValueEstimate(
            str(i),
            Episode(p, r, 1.0, "stop"),
            [-10.0] * len(r),
            [float(i + j + 1) for j in range(len(r))],
        )
        for i, (p, r) in enumerate(zip(prompts, responses, strict=True))
    )
    snapshot = deepcopy(estimates)
    batch = collate_critic(estimates, pad_token_id=99)
    assert isinstance(batch, CriticBatch)
    width = max(len(p) + len(r) for p, r in zip(prompts, responses, strict=True))
    for name, dtype in (
        ("input_ids", torch.int64),
        ("attention_mask", torch.bool),
        ("response_mask", torch.bool),
        ("returns", torch.float32),
    ):
        tensor = getattr(batch, name)
        assert tensor.shape == (
            len(estimates),
            width if name in ("input_ids", "attention_mask") else width - 1,
        )
        assert tensor.dtype == dtype
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
    for i, estimate in enumerate(estimates):
        p, r = estimate.episode.prompt_ids, estimate.episode.response_ids
        padding = width - len(p) - len(r)
        assert batch.input_ids[i].tolist() == p + r + [99] * padding
        assert (
            batch.attention_mask[i].tolist()
            == [True] * (len(p) + len(r)) + [False] * padding
        )
        assert (
            batch.response_mask[i].tolist()
            == [False] * (len(p) - 1) + [True] * len(r) + [False] * padding
        )
        assert batch.returns[i].tolist() == (
            [0.0] * (len(p) - 1) + estimate.return_estimates + [0.0] * padding
        )
    assert estimates == snapshot


def test_collate_critic_rejects_empty_batch():
    with pytest.raises(AssertionError, match="episodes must be nonempty"):
        collate_critic([], pad_token_id=99)


@pytest.mark.parametrize(
    "name,replacement,message",
    [
        ("prompt_ids", [], "prompt must be nonempty"),
        ("response_ids", [], "response must be nonempty"),
        ("response_ids", [21], "must match response length"),
        ("response_ids", [21, 22, 23], "must match response length"),
        ("critic_value_estimates", [], "must match response length"),
        ("critic_value_estimates", [0.5], "must match response length"),
        ("critic_value_estimates", [0.5] * 3, "must match response length"),
        ("return_estimates", [], "must match response length"),
        ("return_estimates", [1.0], "must match response length"),
        ("return_estimates", [1.0] * 3, "must match response length"),
    ],
)
def test_collate_critic_revalidates_mutated_lists(name, replacement, message):
    estimate = PPOValueEstimate(
        "id", Episode([11], [21, 22], 1.0, "stop"), [0.5, 0.5], [1.0, 1.0]
    )
    owner = estimate.episode if name in ("prompt_ids", "response_ids") else estimate
    getattr(owner, name)[:] = replacement
    snapshot = deepcopy(estimate)
    with pytest.raises(AssertionError, match=message):
        collate_critic([estimate], pad_token_id=99)
    assert estimate == snapshot

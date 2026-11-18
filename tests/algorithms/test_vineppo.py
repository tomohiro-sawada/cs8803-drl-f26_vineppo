"""Deterministic VinePPO contract tests; no models, sampling, or accelerators."""

from copy import deepcopy
from dataclasses import replace

import pytest

from algorithms.vineppo import (
    compute_step_advantages,
    estimate_vineppo,
    vineppo_advantage_method,
)
from core.data import Episode, MCContinuation, PrefixSamples, VinePPOValueEstimate
from core.inference import Completion

BOUNDARIES = (0, 2, 5, 6)
INVALID_BOUNDARIES = [
    (),
    (0,),
    (1, 6),
    (0, 5),
    (0, 7),
    (0, 2, 2, 6),
    (0, 4, 2, 6),
    (0, -1, 6),
    (0, 7, 6),
    (False, 6),
    (0, True, 6),
    (0, 2.0, 6),
    (0, "2", 6),
]


@pytest.mark.parametrize("sequence_type", [list, tuple])
@pytest.mark.parametrize(
    "values, reward, expected",
    [
        ((0.25, 0.75, 0.5), 1.0, (0.5, -0.25, 0.5)),
        ((0.75, 0.5, 0.25), 0.0, (-0.25, -0.25, -0.25)),
        ((0.5,), 0.75, (0.25,)),
        ((0.5,), 0.0, (-0.5,)),
        ((-1.0, 2.0, -0.5, 1.5), -0.25, (3.0, -2.5, 2.0, -1.75)),
    ],
)
def test_compute_step_advantages(values, reward, expected, sequence_type):
    prefix_values = sequence_type(values)
    before = deepcopy(prefix_values)
    advantages = compute_step_advantages(prefix_values, reward)
    assert isinstance(advantages, tuple)
    assert advantages == expected
    assert sum(advantages) == reward - values[0]
    assert prefix_values == before


@pytest.mark.parametrize("values", [[], ()])
def test_compute_step_advantages_rejects_empty_values(values):
    with pytest.raises(ValueError, match="prefix_values must be nonempty"):
        compute_step_advantages(values, 1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_compute_step_advantages_rejects_nonfinite_values(value, index):
    values = [0.25, 0.75, 0.5]
    values[index] = value
    with pytest.raises(ValueError, match="prefix_values must be finite"):
        compute_step_advantages(values, 1.0)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
def test_compute_step_advantages_rejects_nonfinite_reward(reward):
    with pytest.raises(ValueError, match="reward must be finite"):
        compute_step_advantages((0.25, 0.75, 0.5), reward)


class FakeInference:
    def __init__(self, groups):
        self.groups = groups
        self.calls = []
        self.transform = lambda groups: groups

    def generate(self, *, prefixes, num_samples, max_new_tokens, temperature):
        self.calls.append(
            (deepcopy(prefixes), num_samples, max_new_tokens, temperature)
        )
        return self.transform([self.groups[tuple(prefix)] for prefix in prefixes])


@pytest.fixture
def episode():
    return Episode([90, 91], [10, 11, 12, 13, 14, 15], 1.0, "stop")


@pytest.fixture
def inference(episode):
    return FakeInference(
        {
            tuple(episode.prompt_ids): [
                Completion([20], "stop"),
                Completion([21], "length"),
            ],
            tuple(episode.prompt_ids + [10, 11]): [
                Completion([22], "length"),
                Completion([23], "stop"),
            ],
            tuple(episode.prompt_ids + [10, 11, 12, 13, 14]): [
                Completion([24], "stop"),
                Completion([25], "length"),
            ],
        }
    )


def score_response(response_ids, finish_reason):
    return {20: 0.0, 21: 0.5, 22: 0.5, 23: 1.0, 24: 1.0, 25: 0.0}[response_ids[-1]]


def estimate(episode, inference, **overrides):
    options = {
        "step_boundaries": BOUNDARIES,
        "score_response": score_response,
        "num_samples": 2,
        "max_new_tokens": 3,
        "temperature": 0.7,
    }
    options.update(overrides)
    return estimate_vineppo("episode-1", episode, inference=inference, **options)


@pytest.fixture
def value_estimate(episode):
    return VinePPOValueEstimate(
        episode_id="episode-1",
        episode=episode,
        step_boundaries=BOUNDARIES,
        mc_prefix_values=(0.25, 0.75, 0.5),
        raw_step_advantages=(0.5, -0.25, 0.5),
    )


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_estimation_and_diagnostics(episode, inference, value_estimate, finish_reason):
    episode = replace(episode, finish_reason=finish_reason)
    before = deepcopy((episode, inference.groups))
    scored = []
    records = []

    def scorer(response_ids, reason):
        assert isinstance(response_ids, list)
        scored.append((response_ids.copy(), reason))
        return score_response(response_ids, reason)

    plain = estimate(episode, inference, score_response=scorer)
    recorded = estimate(episode, inference, record_prefix_samples=records.append)
    assert isinstance(plain, VinePPOValueEstimate)
    assert plain == recorded == replace(value_estimate, episode=episode)
    assert scored == [
        ([20], "stop"),
        ([21], "length"),
        ([10, 11, 22], "length"),
        ([10, 11, 23], "stop"),
        ([10, 11, 12, 13, 14, 24], "stop"),
        ([10, 11, 12, 13, 14, 25], "length"),
    ]
    assert [prefix for prefixes, *_ in inference.calls for prefix in prefixes] == [
        [90, 91],
        [90, 91, 10, 11],
        [90, 91, 10, 11, 12, 13, 14],
    ] * 2
    assert all(call[1:] == (2, 3, 0.7) for call in inference.calls)
    assert records == [
        PrefixSamples(
            "episode-1",
            0,
            (
                MCContinuation((20,), "stop", 0.0),
                MCContinuation((21,), "length", 0.5),
            ),
        ),
        PrefixSamples(
            "episode-1",
            2,
            (
                MCContinuation((22,), "length", 0.5),
                MCContinuation((23,), "stop", 1.0),
            ),
        ),
        PrefixSamples(
            "episode-1",
            5,
            (
                MCContinuation((24,), "stop", 1.0),
                MCContinuation((25,), "length", 0.0),
            ),
        ),
    ]
    assert (episode, inference.groups) == before
    assert vineppo_advantage_method(recorded) == [0.5, 0.5, -0.25, -0.25, -0.25, 0.5]


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_single_step_greedy(episode, finish_reason):
    episode = replace(episode, finish_reason=finish_reason, reward=0.75)
    inference = FakeInference({tuple(episode.prompt_ids): [Completion([21], "length")]})
    result = estimate(
        episode,
        inference,
        step_boundaries=(0, 6),
        num_samples=1,
        temperature=0.0,
    )
    assert result.mc_prefix_values == (0.5,)
    assert result.raw_step_advantages == (0.25,)
    assert vineppo_advantage_method(result) == [0.25] * 6
    assert inference.calls == [([episode.prompt_ids], 1, 3, 0.0)]


@pytest.mark.parametrize("boundaries", INVALID_BOUNDARIES)
def test_invalid_boundaries_before_inference(episode, inference, boundaries):
    with pytest.raises(ValueError):
        estimate(episode, inference, step_boundaries=boundaries)
    assert inference.calls == []


@pytest.mark.parametrize("field", ["prompt_ids", "response_ids"])
def test_empty_episode_before_inference(episode, inference, field):
    with pytest.raises(ValueError):
        estimate(replace(episode, **{field: []}), inference)
    assert inference.calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"num_samples": 0},
        {"num_samples": -1},
        {"max_new_tokens": 0},
        {"max_new_tokens": -1},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": -float("inf")},
        {"temperature": 0.0, "num_samples": 2},
    ],
)
def test_invalid_sampling_before_inference(episode, inference, options):
    with pytest.raises(ValueError):
        estimate(episode, inference, **options)
    assert inference.calls == []


@pytest.mark.parametrize(
    "transform",
    [
        lambda groups: groups[:-1],
        lambda groups: groups + [groups[0]],
        lambda groups: [[] for group in groups],
        lambda groups: [group[:-1] for group in groups],
        lambda groups: [group + [group[0]] for group in groups],
    ],
)
def test_wrong_generation_cardinality(episode, inference, transform):
    inference.transform = transform
    with pytest.raises(ValueError):
        estimate(episode, inference)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("source", ["episode", "sample"])
def test_nonfinite_rewards(episode, inference, reward, source):
    options = {}
    if source == "episode":
        episode = replace(episode, reward=reward)
    else:
        options["score_response"] = lambda response_ids, reason: reward
    with pytest.raises(ValueError):
        estimate(episode, inference, **options)


@pytest.mark.parametrize("source", ["inference", "scorer", "diagnostics"])
def test_errors_propagate(episode, inference, source):
    error = RuntimeError(source)

    def fail(*args, **kwargs):
        raise error

    options = {}
    if source == "inference":
        inference.transform = fail
    elif source == "scorer":
        options["score_response"] = fail
    else:
        options["record_prefix_samples"] = fail
    with pytest.raises(RuntimeError) as caught:
        estimate(episode, inference, **options)
    assert caught.value is error


def test_advantage_expansion_without_mutation(value_estimate):
    before = deepcopy(value_estimate)
    assert vineppo_advantage_method(value_estimate) == [
        0.5,
        0.5,
        -0.25,
        -0.25,
        -0.25,
        0.5,
    ]
    assert value_estimate == before


@pytest.mark.parametrize("boundaries", INVALID_BOUNDARIES)
def test_advantage_rejects_invalid_boundaries(value_estimate, boundaries):
    with pytest.raises(ValueError):
        vineppo_advantage_method(replace(value_estimate, step_boundaries=boundaries))


@pytest.mark.parametrize("field", ["mc_prefix_values", "raw_step_advantages"])
@pytest.mark.parametrize("values", [(), (0.25, 0.75), (0.25, 0.75, 0.5, 1.0)])
def test_advantage_rejects_wrong_value_counts(value_estimate, field, values):
    with pytest.raises(ValueError):
        vineppo_advantage_method(replace(value_estimate, **{field: values}))

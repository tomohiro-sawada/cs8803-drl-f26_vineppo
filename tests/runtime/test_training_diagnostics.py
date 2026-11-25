"""Cheap diagnostics describe primary data, never resample or change RNG state."""

import random
from math import isfinite

import pytest
import torch

from algorithms.grpo import GRPOConfig
from algorithms.rloo import RLOOConfig
from core.data import EpisodeWithAdvantages
from core.inference import Completion
from core.segmentation import SplitterRejection
from runtime import train as train_module
from runtime.train import _Moments, _training_diagnostics
from tests.runtime import test_ppo_train, test_train

training_case = test_train.training_case
ppo_case = test_ppo_train.ppo_case


def test_summary_token_weighting_constant_and_unequal_groups_without_rng_changes():
    episodes = [
        EpisodeWithAdvantages([1], [2], 1.0, "stop", [0.0]),
        EpisodeWithAdvantages([1], [2, 3], 1.0, "stop", [0.0, 0.0]),
        EpisodeWithAdvantages([1], [2], 0.0, "stop", [2.0]),
        EpisodeWithAdvantages([1], [2, 3, 4], -3.0, "length", [-2.0] * 3),
    ]
    rng, python_rng = torch.get_rng_state(), random.getstate()
    metrics = _training_diagnostics(episodes, group_size=2)
    values = torch.tensor([0, 0, 0, 2, -2, -2, -2], dtype=torch.float64)
    assert metrics == pytest.approx(
        {
            "response_length_mean": 7 / 4,
            "response_length_max": 3,
            "raw_advantage_token_zero_fraction": 3 / 7,
            "raw_advantage_token_std": values.std(correction=0).item(),
            "stopped_correctness_mean": 2 / 3,
            "group_zero_reward_variance_fraction": 0.5,
        }
    )
    moments = _Moments()
    moments.add([1, 2])
    moments.add([3])
    assert moments.count == 3
    assert moments.metrics("value") == pytest.approx(
        {"value_mean": 2, "value_std": (2 / 3) ** 0.5}
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert python_rng == random.getstate()


def test_single_token_all_truncated_constant_advantages_are_finite():
    metrics = _training_diagnostics(
        [EpisodeWithAdvantages([1], [2], -3.0, "length", [0.0])], group_size=None
    )
    assert "stopped_correctness_mean" not in metrics
    assert "group_zero_reward_variance_fraction" not in metrics
    assert metrics["raw_advantage_token_std"] == 0
    assert metrics["raw_advantage_token_zero_fraction"] == 1
    assert all(isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("algorithm", [GRPOConfig(), RLOOConfig()])
def test_group_training_metrics(training_case, monkeypatch, algorithm):
    make, inputs = training_case
    run = make(algorithm_config=algorithm)
    tokenizer = inputs["tokenizer"]

    def generate(prefixes, **kwargs):
        return [
            [
                Completion(tokenizer.encode("#### 1"), "stop"),
                Completion(tokenizer.encode("x"), "length"),
            ]
            if tokenizer.decode(prefix) == "Q"
            else [Completion(tokenizer.encode("#### 0"), "stop") for _ in range(2)]
            for prefix in prefixes
        ]

    monkeypatch.setattr(run.inference, "generate", generate)
    captured = []
    cache = run._cache_scores

    def capture(episodes):
        captured.extend(episodes)
        return cache(episodes)

    monkeypatch.setattr(run, "_cache_scores", capture)
    logged = []
    metrics = run.train(**inputs, record_metrics=logged.append)[0]
    assert logged == [metrics]
    assert metrics["group_zero_reward_variance_fraction"] == 0.5
    assert metrics["stopped_correctness_mean"] == pytest.approx(1 / 3)
    assert metrics["truncation_fraction"] == 0.25
    assert metrics["reward_mean"] == -0.5
    for key, value in _training_diagnostics(captured, group_size=2).items():
        assert metrics[key] == value
    assert not any(key.startswith("vineppo/") for key in metrics)
    assert all(isfinite(value) for value in metrics.values())


def test_splitter_rejection_trains_as_one_counted_step(training_case, monkeypatch):
    make, inputs = training_case
    run = make()
    segment, estimate = train_module.step_boundaries_from_tokens, run.estimate_vineppo_batch
    expected, items = [], []

    def reject_first(response_ids, *args, **kwargs):
        expected.append(segment(response_ids, *args, **kwargs))
        if len(expected) == 1:
            expected[0] = (0, len(response_ids))
            raise SplitterRejection("nested placeholder")
        return expected[-1]

    def capture(batch, **kwargs):
        items.extend(batch)
        return estimate(batch, **kwargs)

    monkeypatch.setattr(train_module, "step_boundaries_from_tokens", reject_first)
    monkeypatch.setattr(run, "estimate_vineppo_batch", capture)
    metrics = run.train(**inputs)[0]
    assert metrics["vineppo/splitter_rejected_responses"] == 1
    assert [boundaries for _, _, boundaries, _ in items] == expected
    assert len(expected[1]) > 2 and metrics["vineppo/step_count"] == sum(
        len(boundaries) - 1 for boundaries in expected
    )


def test_vine_metrics_stream_estimates_without_enabling_prefix_payloads(
    training_case, monkeypatch
):
    make, inputs = training_case
    run = make()
    estimates, logged = [], []
    estimate = run.estimate_vineppo_batch

    def capture(*args, **kwargs):
        assert kwargs["record_prefix_samples"] is None
        result = estimate(*args, **kwargs)
        estimates.extend(result)
        return result

    monkeypatch.setattr(run, "estimate_vineppo_batch", capture)
    metrics = run.train(**inputs, record_metrics=logged.append)[0]
    values = torch.tensor(
        [v for estimate in estimates for v in estimate.mc_prefix_values],
        dtype=torch.float64,
    )
    assert logged == [metrics]
    assert metrics["vineppo/step_count"] == len(values)
    assert metrics["vineppo/splitter_rejected_responses"] == 0
    assert metrics["vineppo/steps_per_response_mean"] == len(values) / len(estimates)
    assert metrics["vineppo/mc_prefix_value_mean"] == pytest.approx(
        values.mean().item()
    )
    assert metrics["vineppo/mc_prefix_value_std"] == pytest.approx(
        values.std(correction=0).item()
    )
    assert metrics["stopped_correctness_mean"] == 0.5
    assert metrics["truncation_fraction"] == 0.5
    assert "group_zero_reward_variance_fraction" not in metrics
    assert all(isfinite(value) for value in metrics.values())


def test_ppo_has_primary_and_raw_token_metrics_only(ppo_case):
    make, inputs = ppo_case
    metrics = make().train(**inputs)[0]
    assert {
        "response_length_mean",
        "response_length_max",
        "raw_advantage_token_zero_fraction",
        "raw_advantage_token_std",
    } <= metrics.keys()
    assert "group_zero_reward_variance_fraction" not in metrics
    assert not any(key.startswith("vineppo/") for key in metrics)
    assert all(isfinite(value) for value in metrics.values())

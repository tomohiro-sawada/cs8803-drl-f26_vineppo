"""Reject unsupported training policies without restricting evaluation."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from algorithms.vineppo import VinePPOConfig
from runtime.config import PPOTrainerConfig


def config(**kwargs):
    return PPOTrainerConfig(
        checkpoint_dir=Path("unused-checkpoints"),
        algorithm_config=VinePPOConfig(),
        **kwargs,
    )


@pytest.mark.parametrize("temperature", [1, 1.0])
def test_training_accepts_only_unit_temperature(temperature):
    settings = config(temperature=temperature)
    assert settings.temperature == temperature
    assert config().temperature == 1.0


@pytest.mark.parametrize(
    "temperature",
    [
        0,
        0.6,
        1.000001,
        2,
        -1,
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        False,
        "1",
        None,
    ],
)
def test_training_rejects_instead_of_overriding_temperature(temperature):
    with pytest.raises(ValueError, match="temperature=1"):
        config(temperature=temperature)


def test_length_budget_defaults():
    assert config().max_sequence_length == 2048
    assert config().max_new_tokens == 1024


@pytest.mark.parametrize(
    "name",
    [
        "max_sequence_length",
        "max_new_tokens",
        "num_iterations",
        "prompts_per_iteration",
        "num_samples",
        "mc_num_samples",
        "update_epochs",
        "minibatch_size",
        "microbatch_size",
        "checkpoint_every",
    ],
)
@pytest.mark.parametrize("value", [0, -1, 1.0, 1.5, True, False, "2048", None])
def test_counts_and_budgets_require_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        config(**{name: value})


def test_temperature_cannot_change_after_validation():
    settings = config()
    with pytest.raises(FrozenInstanceError):
        settings.temperature = 0.6
    with pytest.raises(ValueError, match="temperature=1"):
        replace(settings, temperature=0.6)


def test_training_defaults():
    settings = config()
    assert settings.num_iterations == 100
    assert settings.prompts_per_iteration == settings.num_samples == 1
    assert settings.mc_num_samples == 4
    assert (
        settings.update_epochs
        == settings.minibatch_size
        == settings.microbatch_size
        == 1
    )
    assert settings.clip_eps == 0.2
    assert settings.kl_coef == 1e-4
    assert settings.max_grad_norm == 1.0
    assert settings.truncation_reward == -1.0
    assert settings.seed == 0 and settings.checkpoint_every == 1


@pytest.mark.parametrize(
    "name", ["clip_eps", "kl_coef", "max_grad_norm", "truncation_reward"]
)
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), True, False, "1", None]
)
def test_numeric_controls_must_be_finite_numbers(name, value):
    with pytest.raises(ValueError, match=name):
        config(**{name: value})


@pytest.mark.parametrize(
    "name,value",
    [
        ("clip_eps", 0),
        ("clip_eps", -0.1),
        ("clip_eps", 1),
        ("clip_eps", 1.1),
        ("kl_coef", -0.1),
        ("max_grad_norm", 0),
        ("max_grad_norm", -1),
    ],
)
def test_numeric_control_ranges(name, value):
    with pytest.raises(ValueError, match=name):
        config(**{name: value})


@pytest.mark.parametrize("seed", [-1, 2**64, 1.0, True, False, "0", None])
def test_seed_rejects_invalid_values(seed):
    with pytest.raises(ValueError, match="seed"):
        config(seed=seed)


@pytest.mark.parametrize("seed", [0, 2**64 - 1])
def test_seed_accepts_unsigned_64_bit_boundaries(seed):
    assert config(seed=seed).seed == seed


@pytest.mark.parametrize("reward", [-7, 0, 2.5])
def test_finite_truncation_reward_and_zero_kl_are_allowed(reward):
    settings = config(
        truncation_reward=reward, kl_coef=0, clip_eps=0.999, max_grad_norm=2
    )
    assert settings.truncation_reward == reward
    assert settings.kl_coef == 0


def test_positive_counts_are_independent_not_silently_clamped():
    settings = config(
        max_sequence_length=8,
        max_new_tokens=16,
        prompts_per_iteration=3,
        num_samples=2,
        mc_num_samples=5,
        update_epochs=4,
        minibatch_size=3,
        microbatch_size=7,
        checkpoint_every=9,
        num_iterations=2,
    )
    assert settings.max_new_tokens == 16
    assert settings.microbatch_size == 7
    assert settings.checkpoint_every == 9
    with pytest.raises(FrozenInstanceError):
        settings.update_epochs = 1
    with pytest.raises(ValueError, match="update_epochs"):
        replace(settings, update_epochs=0)

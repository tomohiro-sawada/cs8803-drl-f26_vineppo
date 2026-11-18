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


def test_temperature_cannot_change_after_validation():
    settings = config()
    with pytest.raises(FrozenInstanceError):
        settings.temperature = 0.6
    with pytest.raises(ValueError, match="temperature=1"):
        replace(settings, temperature=0.6)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True])
def test_iteration_count_must_be_positive_integer(iterations):
    with pytest.raises(ValueError, match="num_iterations"):
        config(num_iterations=iterations)

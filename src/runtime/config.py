"""Application-owned training configuration; evaluation settings are independent."""

from dataclasses import dataclass
from pathlib import Path

from algorithms.base import AlgorithmConfig


@dataclass(frozen=True, kw_only=True)
class PPOTrainerConfig:
    """One sampling policy for primary rollouts and MC continuations.

    Raw actor scores are normalized log-probabilities without temperature
    scaling. Until differentiable scoring supports other temperatures, reject
    them here rather than silently overriding the requested policy. T=1 still
    samples categorically; it is not greedy decoding.
    """

    checkpoint_dir: Path
    algorithm_config: AlgorithmConfig
    num_iterations: int = 100
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if type(self.temperature) not in (int, float) or self.temperature != 1.0:
            raise ValueError(
                "PPO training requires temperature=1 for rollouts and MC estimation"
            )
        if type(self.num_iterations) is not int or self.num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer")

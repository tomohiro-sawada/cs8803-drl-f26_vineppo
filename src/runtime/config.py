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

    max_sequence_length bounds prompt plus response, including special tokens.
    max_new_tokens caps each primary/MC continuation independently; remaining
    total context may shorten it. It is not a cumulative response-length cap.
    """

    checkpoint_dir: Path
    algorithm_config: AlgorithmConfig
    num_iterations: int = 100
    temperature: float = 1.0
    max_sequence_length: int = 2048
    max_new_tokens: int = 1024

    def __post_init__(self) -> None:
        for name in ("max_sequence_length", "max_new_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.temperature) not in (int, float) or self.temperature != 1.0:
            raise ValueError(
                "PPO training requires temperature=1 for rollouts and MC estimation"
            )
        if type(self.num_iterations) is not int or self.num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer")

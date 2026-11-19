"""Application-owned training configuration; evaluation settings are independent."""

from dataclasses import dataclass
from math import isfinite
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
    prompts_per_iteration samples without replacement within each iteration;
    num_samples and mc_num_samples control primary and prefix continuations.
    minibatch_size counts episodes per optimizer step; microbatch_size only
    bounds scoring/backpropagation memory. update_epochs reuse fixed data.
    Length-truncated primary and MC responses both receive truncation_reward.
    seed controls prompt/update shuffling; callers seed torch for generation.
    Checkpoints are saved every checkpoint_every iterations and at completion.
    """

    checkpoint_dir: Path
    algorithm_config: AlgorithmConfig
    num_iterations: int = 100
    temperature: float = 1.0
    max_sequence_length: int = 2048
    max_new_tokens: int = 1024
    prompts_per_iteration: int = 1
    num_samples: int = 1
    mc_num_samples: int = 4
    update_epochs: int = 1
    minibatch_size: int = 1
    microbatch_size: int = 1
    clip_eps: float = 0.2
    kl_coef: float = 1e-4
    max_grad_norm: float = 1.0
    truncation_reward: float = -1.0
    seed: int = 0
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        for name in (
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
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.temperature) not in (int, float) or self.temperature != 1.0:
            raise ValueError(
                "PPO training requires temperature=1 for rollouts and MC estimation"
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        for name in ("clip_eps", "kl_coef", "max_grad_norm", "truncation_reward"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.clip_eps < 1:
            raise ValueError("clip_eps must be in (0, 1)")
        if self.kl_coef < 0:
            raise ValueError("kl_coef must be nonnegative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

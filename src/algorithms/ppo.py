"""
PPO algorithm implementation.


Training:
    For critic-based PPO, the iteration is:
    Collect and verify episodes.
    Cache old actor log-probabilities and critic values before any updates.
    Compute raw advantages and fixed return targets.
    Construct policy and critic batches.
    Optimize the actor using the policy batch.
    Optimize the critic using the critic batch.
    Repeat optimization epochs against the same cached targets; do not recompute those targets from the changing critic mid-update.

Separate actor and critic models/optimizers are the simplest initial implementation. If you later share a backbone, the update schedule and gradient ownership need an explicit design.
"""

from dataclasses import dataclass

from core.algorithm.base import AlgorithmConfig


@dataclass
class PPOConfig(AlgorithmConfig):
    pass


def ppo_advantage_method():
    pass

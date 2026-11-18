"""
Contracts for the core RL training logic.
Enforce compatibility as we add new baselines to the codebase.
"""

from dataclasses import dataclass
from abc import ABC

from transformers import PreTrainedModel


@dataclass
class AlgorithmConfig:
    """
    Configuration for specific algorithms.
    """


class RolloutGenerator:
    def __init__(self, policy: PreTrainedModel):
        """
        Sample trajectories under a frozen policy snapshot.
        Use cases: Primary rollouts, MC continuations
        """
        self.policy = policy


class StepSegmenter:
    def __init__(self) -> None:
        """
        Identify reasoning-step boundaries
        Use cases: Newline, punctuation, fixed-token
        """
        pass



class AdvantageEstimator(ABC):
    def __init__(self, policy: PreTrainedModel):
        """
        Estimate the advantage of a state or trajectory.
        This is the most important class in this codebase.
        """
        self.policy = policy


#

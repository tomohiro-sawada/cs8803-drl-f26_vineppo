"""
Contracts for the core RL training logic.
Enforce compatibility as we add new baselines to the codebase.
"""

from dataclasses import dataclass


@dataclass
class AlgorithmConfig:
    """
    Configuration for specific algorithms.
    """

"""
"""


from dataclasses import dataclass

from core.types import Episode
from core.algorithm.base import AlgorithmConfig


@dataclass
class VinePPOConfig(AlgorithmConfig):
    def __init__(self):
        super().__init__()
        pass




def vineppo_advantage_method(
    episodes: list[Episode]
) -> list[float]:
    """
    Computes the advantage for each episode using the VinePPO advantage method.
    """
    def _compute_advantage(episode: Episode) -> float:
        """
        Computes the advantage for a single episode.

        VinePPO advantage computes:

            advantage = (total reward for whole episode) - baseline

        where baseline is obtained by averaging total reward for total rewards over rollouts generated from partial context.
        """
        return

    # validate that the Episode objects have the appropriate fields?

    # Compute the advantage for each episode
    # TODO: can we broadcast this????
    advantages = []
    for episode in episodes:
        advantages.append(_compute_advantage(episode))
    return advantages

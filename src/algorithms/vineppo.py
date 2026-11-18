"""
VinePPO.


Training should:
    For VinePPO:
    Collect and verify episodes.
    Estimate prefix values using MC continuations.
    Convert step advantages to token advantages.
    Construct the policy batch.
    Optimize the actor.
    There is no critic batch or critic update in VinePPO.
    Separate actor and critic models/optimizers are the simplest initial implementation. If you later share a backbone, the update schedule and gradient ownership need an explicit design.
"""

# typing
from dataclasses import dataclass
from typing import Callable


from algorithms.base import AlgorithmConfig
from core.data import Episode, PrefixSamples, MCContinuation, VinePPOValueEstimate
from core.inference import HFInference


@dataclass
class VinePPOConfig(AlgorithmConfig):
    def __init__(self):
        super().__init__()
        pass


def estimate_vineppo(
    episode_id: str,
    episode: Episode,
    *,
    step_boundaries: tuple[int, ...],
    inference: HFInference,
    score_response: Callable[[list[int], str], float],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    record_prefix_samples: Callable[[PrefixSamples], None] | None = None,
) -> VinePPOValueEstimate:
    ...

    """
    Estimates the value of a state under a given policy using the VinePPO algorithm.

    Inputs:

        - score_response receives: The complete response token sequence being evaluated. The continuation's finish reason. It can be a callable that captures the tokenizer and the example’s reference answer. This keeps those dependencies explicit without teaching the estimator about dataset loading.

    Diagnostics: record_prefix_samples. This optional branch is not a fallback: it is an explicit feature choice. Estimation is identical whether evidence is retained or discarded. Recording failures should propagate rather than silently disabling auditing.

    Usage:
        # Training without retained diagnostics:
        estimate = estimate_vineppo(...)

        # Small debugging run:
        records = []
        estimate = estimate_vineppo(
            ...,
            record_prefix_samples=records.append,
        )

        # Larger audited run:
        estimate = estimate_vineppo(
            ...,
            record_prefix_samples=diagnostic_writer.write,
        )
    """
    # For each prefix, the estimator:
    # Generates continuations.
    # Verifies their rewards.
    # Computes the prefix value from those rewards.
    # If recording was explicitly requested, sends a PrefixSamples record to the callback.
    # Releases the samples when no longer needed.
    # The caller chooses the retention policy:


def vineppo_advantage_method(episodes: list[Episode]) -> list[float]:
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

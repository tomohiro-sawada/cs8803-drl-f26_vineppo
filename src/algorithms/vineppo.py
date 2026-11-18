"""Monte Carlo prefix-value estimation and step-to-token advantages for VinePPO.

Estimation precedes token expansion, policy-batch collation, and actor updates.
VinePPO does not require a critic batch or critic update.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

from algorithms.base import AlgorithmConfig
from core.data import Episode, MCContinuation, PrefixSamples, VinePPOValueEstimate
from core.inference import HFInference


@dataclass
class VinePPOConfig(AlgorithmConfig):
    pass


def _validate_boundaries(episode: Episode, boundaries: tuple[int, ...]) -> None:
    if not episode.prompt_ids or not episode.response_ids:
        raise ValueError("episode prompt and response must be nonempty")
    if len(boundaries) < 2:
        raise ValueError("step_boundaries must contain at least two boundaries")
    if any(type(boundary) is not int for boundary in boundaries):
        raise ValueError("step_boundaries must contain integers")
    if boundaries[0] != 0 or boundaries[-1] != len(episode.response_ids):
        raise ValueError("step_boundaries must start at 0 and end at response length")
    if any(left >= right for left, right in pairwise(boundaries)):
        raise ValueError("step_boundaries must be strictly increasing")


def _get_prefixes(
    episode: Episode, step_boundaries: tuple[int, ...]
) -> list[list[int]]:
    return [
        episode.prompt_ids + episode.response_ids[:boundary]
        for boundary in step_boundaries[:-1]
    ]


def compute_step_advantages(
    prefix_values: Sequence[float], reward: float
) -> tuple[float, ...]:
    """Compute raw advantages for terminal-only reward, gamma=1, and terminal V=0.

    prefix_values contains one nonterminal pre-step value per step, excluding
    the terminal value. Each advantage is the next value minus the current
    value, except the last, which is reward minus the final pre-step value.
    """
    if len(prefix_values) == 0:
        raise ValueError("prefix_values must be nonempty")
    if any(not isfinite(value) for value in prefix_values):
        raise ValueError("prefix_values must be finite")
    if not isfinite(reward):
        raise ValueError("reward must be finite")
    return tuple(
        next_value - current_value
        for current_value, next_value in pairwise(prefix_values)
    ) + (reward - prefix_values[-1],)


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
    """Estimate pre-step values and raw advantages using sampled continuations.

    Rewards are terminal-only, gamma is 1, and terminal value is zero. Both
    stopped and length-truncated episodes are treated as terminal with their
    supplied reward; sampled truncations are scored by score_response.

    The scorer receives response-prefix plus continuation tokens (no prompt),
    and the sampled finish reason. Generation budgets apply per continuation.
    Optional diagnostics retain every scored sample; omitting the callback
    retains only aggregate values. Generation, scoring, and recording errors
    propagate. The caller owns the inference model and its lifetime.
    """
    _validate_boundaries(episode, step_boundaries)
    if type(num_samples) is not int or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if (
        not isinstance(temperature, (int, float))
        or not isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and nonnegative")
    if temperature == 0 and num_samples != 1:
        raise ValueError("greedy decoding requires one sample")
    if not isfinite(episode.reward):
        raise ValueError("episode reward must be finite")

    prefix_values = []
    prefixes = _get_prefixes(episode, step_boundaries)
    for boundary, prefix in zip(step_boundaries[:-1], prefixes, strict=True):
        groups = inference.generate(
            prefixes=[prefix],
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if len(groups) != 1 or len(groups[0]) != num_samples:
            raise ValueError(
                "generation must return one group with num_samples completions"
            )
        rewards = []
        samples = []
        for continuation in groups[0]:
            response_tokens = episode.response_ids[:boundary] + continuation.token_ids
            reward = score_response(response_tokens, continuation.finish_reason)
            if not isfinite(reward):
                raise ValueError("sample reward must be finite")
            rewards.append(reward)
            if record_prefix_samples is not None:
                samples.append(
                    MCContinuation(
                        response_ids=tuple(continuation.token_ids),
                        finish_reason=continuation.finish_reason,
                        reward=reward,
                    )
                )
        prefix_values.append(sum(rewards) / len(rewards))
        if record_prefix_samples is not None:
            record_prefix_samples(
                PrefixSamples(
                    episode_id=episode_id,
                    response_prefix_length=boundary,
                    continuations=tuple(samples),
                )
            )

    return VinePPOValueEstimate(
        episode_id=episode_id,
        episode=episode,
        step_boundaries=step_boundaries,
        mc_prefix_values=tuple(prefix_values),
        raw_step_advantages=compute_step_advantages(prefix_values, episode.reward),
    )


def vineppo_advantage_method(estimate: VinePPOValueEstimate) -> list[float]:
    """Expand one estimate's raw step advantages to response tokens, unnormalized."""
    _validate_boundaries(estimate.episode, estimate.step_boundaries)
    num_steps = len(estimate.step_boundaries) - 1
    if (
        len(estimate.mc_prefix_values) != num_steps
        or len(estimate.raw_step_advantages) != num_steps
    ):
        raise ValueError("each step must have one prefix value and one raw advantage")
    return [
        advantage
        for start, end, advantage in zip(
            estimate.step_boundaries[:-1],
            estimate.step_boundaries[1:],
            estimate.raw_step_advantages,
            strict=True,
        )
        for _ in range(end - start)
    ]

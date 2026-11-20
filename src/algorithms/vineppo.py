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
from core.inference import Completion, HFInference

ScoreResponses = Callable[[Sequence[Completion]], Sequence[float]]


@dataclass
class VinePPOConfig(AlgorithmConfig):
    pass


def continuation_budget(
    prefix_length: int, *, max_sequence_length: int, max_new_tokens: int
) -> int:
    """Cap new tokens by remaining total context, not remaining response budget.

    The full prefix includes prompt/BOS and any existing response tokens. Like
    upstream EfficientIIDExpander._compute_max_tokens, the response prefix is
    subtracted only from total context, never separately from max_new_tokens.
    Exact token IDs include special tokens (including generated EOS); unlike
    upstream's text-tokenized 2047 context allowance, no extra BOS reservation
    is needed inside the default 2048-token total limit.
    """
    for name, value in (
        ("prefix_length", prefix_length),
        ("max_sequence_length", max_sequence_length),
        ("max_new_tokens", max_new_tokens),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    remaining = max_sequence_length - prefix_length
    if remaining <= 0:
        raise ValueError("prefix exhausts max_sequence_length; no continuation fits")
    return min(max_new_tokens, remaining)


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


def estimate_vineppo_batch(
    items: Sequence[tuple[str, Episode, tuple[int, ...], ScoreResponses]],
    *,
    inference: HFInference,
    num_samples: int,
    max_new_tokens: int = 1024,
    max_sequence_length: int = 2048,
    temperature: float,
    record_prefix_samples: Callable[[PrefixSamples], None] | None = None,
) -> list[VinePPOValueEstimate]:
    """Estimate pre-step values and raw advantages for sampled episodes.

    Every item is validated and every nonterminal prefix of every episode is
    submitted in one generate_each call before any result is awaited, so a
    concurrent sampler sees the whole iteration's work. Results are reduced in
    submission order: sampler arrival order cannot change an estimate. Each
    scorer is bound to its episode's reference answer and sees only response
    prefix plus continuation tokens and the finish reason. Each branch uses
    its remaining context budget. Stopped and truncated episodes use their
    supplied terminal reward and zero terminal value.
    """
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
    prefixes, budgets = [], []
    for _, episode, step_boundaries, _ in items:
        _validate_boundaries(episode, step_boundaries)
        if not isfinite(episode.reward):
            raise ValueError("episode reward must be finite")
        episode_prefixes = _get_prefixes(episode, step_boundaries)
        budgets += [
            continuation_budget(
                len(prefix),
                max_sequence_length=max_sequence_length,
                max_new_tokens=max_new_tokens,
            )
            for prefix in episode_prefixes
        ]
        prefixes += episode_prefixes
        if len(episode.prompt_ids) + len(episode.response_ids) > max_sequence_length:
            raise ValueError("episode exceeds max_sequence_length")
    groups = inference.generate_each(
        prefixes, budgets, num_samples=num_samples, temperature=temperature
    )
    if len(groups) != len(prefixes) or any(len(g) != num_samples for g in groups):
        raise ValueError("generation returned invalid completion cardinality")

    groups = iter(groups)
    estimates = []
    for episode_id, episode, step_boundaries, score_responses in items:
        prefix_values = []
        for boundary in step_boundaries[:-1]:
            group = next(groups)
            responses = [
                Completion(
                    episode.response_ids[:boundary] + continuation.token_ids,
                    continuation.finish_reason,
                )
                for continuation in group
            ]
            try:
                rewards = score_responses(responses)
            except Exception as error:
                error.add_note(
                    f"MC episode {episode_id}, response prefix length {boundary}"
                )
                raise
            if len(rewards) != len(group) or not all(
                isfinite(reward) for reward in rewards
            ):
                raise ValueError("Expected one finite reward per MC sample")
            prefix_values.append(sum(rewards) / len(rewards))
            if record_prefix_samples is not None:
                record_prefix_samples(
                    PrefixSamples(
                        episode_id=episode_id,
                        response_prefix_length=boundary,
                        continuations=tuple(
                            MCContinuation(
                                tuple(continuation.token_ids),
                                continuation.finish_reason,
                                reward,
                            )
                            for continuation, reward in zip(group, rewards, strict=True)
                        ),
                    )
                )
        estimates.append(
            VinePPOValueEstimate(
                episode_id=episode_id,
                episode=episode,
                step_boundaries=step_boundaries,
                mc_prefix_values=tuple(prefix_values),
                raw_step_advantages=compute_step_advantages(
                    prefix_values, episode.reward
                ),
            )
        )
    return estimates


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

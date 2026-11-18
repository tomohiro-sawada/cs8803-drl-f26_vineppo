"""Shared dataset records and training data structures.

Evaluation consumes a ``Sequence[Example]``, not an ``Episode`` or ``TrainBatch``.
Dataset-specific loading normalizes source rows into examples; application
assembly selects the split/subset and records dataset provenance. Loading and
prompt formatting belong outside these records.

The evaluator owns generation batching. It preserves example order, includes the
final partial batch, and associates each completion with an example ID and sample
index. Repeated evaluation must reuse the same selection without consuming or
mutating it. An evaluation selection must be nonempty and have unique IDs;
validation of these requirements belongs at the evaluation boundary.

Training may share examples and source-loading functions, but owns its sampling
state and tensor collation separately. The training structures below are not part
of the evaluation input contract.
"""

# typing
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
from jaxtyping import Bool, Float32, Int64
from torch import Tensor


@dataclass(frozen=True)
class Example:
    """One normalized, backend-independent dataset example.

    Attributes:
        id: Nonempty identifier for the original source row, unique within its
            dataset revision/configuration/split. Selection, shuffling, and
            batching must preserve it rather than assign a new positional ID.
            Dataset provenance is recorded separately in run metadata; the ID
            alone need not identify a row across different datasets.
        question: Nonempty problem text before prompt formatting. It contains
            neither an appended reference solution nor model-specific tokens or
            chat-template wrappers. The prompt formatter receives this field,
            not the reference answer.
        reference_answer: Nonempty target answer in the representation expected
            by the task's verifier. For GSM8K this is the extracted final answer,
            not the full worked solution. Keep it as text to avoid premature
            numeric conversion; numeric equivalence belongs to the verifier.

    Dataset loaders must reject malformed source rows rather than silently skip
    them or substitute answers. These are producer obligations, not constructor
    validation implemented by this record. A malformed generated answer is a
    separate, valid evaluation outcome.

    One example may yield multiple completions. Sample indices belong to the
    evaluation results, not this record. Tokenization, generation settings,
    rewards, and advantages likewise do not belong to a source example.
    """

    id: str
    question: str
    reference_answer: str


@dataclass(frozen=True)
class Episode:
    """
    Represents a single episode of the agent-environment interaction *after* verification (hence `reward` field).
    """

    prompt_ids: list[int]
    response_ids: list[int]
    reward: float
    finish_reason: str


@dataclass(frozen=True)
class EpisodeWithAdvantages(Episode):
    """
    Represents a single episode of the agent-environment interaction with advantages.

    Fields:
        step_boundaries: contains response-relative token offsets, including 0 and len(response_ids); length K+1
        mc_prefix_values[i]: estimates the value of the prefix before step i; length K
    """

    advantages: list[float]
    raw_step_advantages: list[float]
    step_boundaries: tuple[int, ...]
    mc_prefix_values: tuple[float, ...]

    def __post_init__(self):
        # validate invariants
        b = self.step_boundaries
        assert len(b) >= 2, f"{b=} must contain at least a start and end boundary"
        assert b[0] == 0, f"{b[0]=} must be 0 (the start boundary)"
        assert b[-1] == len(self.response_ids), (
            f"{b[-1]=} must equal the response length {len(self.response_ids)=}"
        )
        assert all(start < end for start, end in pairwise(b)), (
            f"{b=} must be strictly increasing"
        )
        assert len(self.mc_prefix_values) == len(b) - 1, (
            f"{self.mc_prefix_values=} must have one value per step ({len(b) - 1=})"
        )
        assert len(self.raw_step_advantages) == len(b) - 1, (
            f"{self.raw_step_advantages=} must have one value per step ({len(b) - 1=})"
        )
        assert len(self.advantages) == len(self.response_ids), (
            f"{self.advantages=} must have one value per response token ({len(self.response_ids)=})"
        )


@dataclass(frozen=True)
class VinePPOEstimate:
    """
    Represents the estimated value of a state under a given policy.
    """

    episode_id: str
    episode: Episode


@dataclass(frozen=True)
class TrainBatch:
    """
    Represents a batch of training data for policy gradient-based methods.
    Supports non-policy gradient baselines such as RestEM.

    input_ids, attention_mask: [B, T].
    response_mask, scores, advantages: [B, T-1].

    Reason:: token_logprobs() returns [B, T-1], scoring input_ids[:, 1:].
    """

    input_ids: Int64[Tensor, "B T"]
    attention_mask: Bool[Tensor, "B T"]
    response_mask: Bool[Tensor, "B T-1"]
    old_logprobs: Float32[Tensor, "B T-1"]
    ref_logprobs: Float32[Tensor, "B T-1"]


@dataclass(frozen=True)
class PolicyGradientBatch:
    """
    Represents a batch of training data for policy gradient-based methods.
    """

    input_ids: Int64[Tensor, "B T"]
    attention_mask: Bool[Tensor, "B T"]
    response_mask: Bool[Tensor, "B T-1"]
    old_logprobs: Float32[Tensor, "B T-1"]
    advantages: Float32[Tensor, "B T-1"]
    ref_logprobs: Float32[Tensor, "B T-1"]


def collate(
    episodes: Sequence[EpisodeWithAdvantages],
    *,
    old_logprobs: Sequence[Sequence[float]],
    ref_logprobs: Sequence[Sequence[float]],
    pad_token_id: int,
) -> PolicyGradientBatch:
    """Construct a right-padded CPU batch, preserving episode order.

    Episodes, prompts, and responses must be nonempty. Both score inputs have
    one row per episode and one value per response token. Mutable episode
    contents are revalidated here. No input is modified.

    IDs are int64 and attention masks are bool, shaped [B, T]. Response masks
    are bool; scores and advantages are detached float32, shaped [B, T-1]
    and aligned with input_ids[:, 1:]. Off-response numeric values are zero.
    Masks depend on lengths, never on token IDs. Device transfer, model scoring,
    verification, and advantage estimation belong outside this function.
    """
    assert len(episodes) > 0, "episodes must be nonempty"
    assert len(old_logprobs) == len(episodes), (
        "old_logprobs must have one row per episode"
    )
    assert len(ref_logprobs) == len(episodes), (
        "ref_logprobs must have one row per episode"
    )
    for episode, old, ref in zip(episodes, old_logprobs, ref_logprobs, strict=True):
        assert len(episode.prompt_ids) > 0, "prompt must be nonempty"
        assert len(episode.response_ids) > 0, "response must be nonempty"
        episode.__post_init__()
        assert len(old) == len(episode.response_ids), (
            "old_logprobs must have one value per response token"
        )
        assert len(ref) == len(episode.response_ids), (
            "ref_logprobs must have one value per response token"
        )

    batch_size = len(episodes)
    width = max(len(e.prompt_ids) + len(e.response_ids) for e in episodes)
    input_ids = torch.full(
        (batch_size, width), pad_token_id, dtype=torch.int64, device="cpu"
    )
    attention_mask = torch.zeros((batch_size, width), dtype=torch.bool, device="cpu")
    response_mask = torch.zeros((batch_size, width - 1), dtype=torch.bool, device="cpu")
    old_scores, advantages, ref_scores = (
        torch.zeros((batch_size, width - 1), dtype=torch.float32, device="cpu")
        for _ in range(3)
    )
    for i, (episode, old, ref) in enumerate(
        zip(episodes, old_logprobs, ref_logprobs, strict=True)
    ):
        prompt_length = len(episode.prompt_ids)
        length = prompt_length + len(episode.response_ids)
        targets = slice(prompt_length - 1, length - 1)
        input_ids[i, :length] = torch.tensor(
            episode.prompt_ids + episode.response_ids, dtype=torch.int64, device="cpu"
        )
        attention_mask[i, :length] = True
        response_mask[i, targets] = True
        for output, values in (
            (old_scores, old),
            (advantages, episode.advantages),
            (ref_scores, ref),
        ):
            output[i, targets] = torch.tensor(values, dtype=torch.float32, device="cpu")

    return PolicyGradientBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        old_logprobs=old_scores,
        advantages=advantages,
        ref_logprobs=ref_scores,
    )

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
from dataclasses import dataclass
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


@dataclass
class Episode:
    prompt_ids: list[int]
    response_ids: list[int]
    reward: float
    finish_reason: str
    advantages: list[float]


class TrainBatch:
    def __init__(self,
        input_ids: list[int],
        attention_mask: list[int],
        response_mask: list[int],
        old_logprobs: list[float],
        advantages: list[float],
        ref_logprobs: list[float],
        returns: list[float]):

        self.input_ids       = input_ids
        self.attention_mask = attention_mask
        self.response_mask   = response_mask
        self.old_logprobs   = old_logprobs
        self.advantages     = advantages
        self.ref_logprobs   = ref_logprobs
        self.returns        = returns

        self._as_tensor()


    def _as_tensor(self) -> "TrainBatch":
        return TrainBatch(
            input_ids=Tensor(self.input_ids),
            attention_mask=Tensor(self.attention_mask),
            response_mask=Tensor(self.response_mask),
            old_logprobs=Tensor(self.old_logprobs),
            advantages=Tensor(self.advantages),
            ref_logprobs=Tensor(self.ref_logprobs),
            returns=Tensor(self.returns),
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter(self._as_tensor())

    def __len__(self):
        return len(self.input_ids)

    def sort_prompts(self):
        """
        Sorts the prompts in the batch by their length.
        """
        sorted_indices = sorted(range(len(self.input_ids)), key=lambda i: len(self.input_ids[i]))
        self.input_ids = [self.input_ids[i] for i in sorted_indices]
        self.attention_mask = [self.attention_mask[i] for i in sorted_indices]
        self.response_mask = [self.response_mask[i] for i in sorted_indices]
        self.old_logprobs = [self.old_logprobs[i] for i in sorted_indices]
        self.advantages = [self.advantages[i] for i in sorted_indices]
        self.ref_logprobs = [self.ref_logprobs[i] for i in sorted_indices]
        self.returns = [self.returns[i] for i in sorted_indices]

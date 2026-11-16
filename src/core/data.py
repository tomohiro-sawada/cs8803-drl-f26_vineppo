"""
Process the data.
This includes:


- Dataloader, batching
-

"""


# typing
from dataclasses import dataclass
from torch.nn import Tensor
# from jaxtyping import


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

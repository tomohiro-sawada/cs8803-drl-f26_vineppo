"""Single-model HF rollouts and differentiable selected-token log-probabilities.

Finish all rollouts (including MC branches) and cache detached old-policy scores
before updating the shared actor. No old-policy copy or weight synchronization is
needed. Tokenization and response/loss masks belong to the caller.

TextGenerator is the text-only evaluation contract, including remote backends.
Training uses exact token prefixes through HFInference, never a decode/send/
retokenize round trip. Its temperature restriction belongs to the trainer.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from types import TracebackType
from typing import Literal, Protocol, Self

import torch
from torch import Tensor
from transformers import PreTrainedModel


@dataclass(frozen=True)
class TextCompletion:
    """Continuation text with normal stopping or token-budget exhaustion.

    ``stop`` means EOS or a configured stop condition; ``length`` means the
    generation token budget was exhausted. Other reasons are errors, not valid
    completions, and must never be silently mapped to either reason.
    """

    text: str
    finish_reason: Literal["stop", "length"]

    def __post_init__(self) -> None:
        if self.finish_reason not in ("stop", "length"):
            raise ValueError(f"Unsupported finish reason: {self.finish_reason!r}")


class TextGenerator(Protocol):
    """Text-only evaluation interface for trained models and blackbox APIs.

    ``generate`` returns exactly ``len(prompts)`` groups in input prompt order,
    including separate groups for duplicate prompts. Each group contains exactly
    ``num_samples`` completions in backend sample order, never ranked or filtered.
    Empty prompts input returns an empty list. Completion text is continuation
    only, without the prompt; empty continuation text is allowed.

    ``num_samples`` and ``max_new_tokens`` must be positive integers; temperature
    must be finite and nonnegative. Zero temperature requests greedy decoding
    with one sample. The token budget applies to each continuation independently.

    Implementations must reject unsupported finish reasons and malformed result
    cardinality explicitly. Backend, decoding, and validation exceptions propagate
    to the caller: do not return partial batches, fabricate completions, or turn
    failures into empty text. This structural protocol documents obligations;
    implementations are responsible for enforcing them.
    """

    def generate(
        self,
        prompts: Sequence[str],
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
    ) -> list[list[TextCompletion]]: ...


@dataclass
class Completion:
    token_ids: list[int]  # Continuation only, including EOS when sampled.
    finish_reason: Literal["stop", "length"]
    logprobs: list[float] | None = None  # Sampling distribution, one per token.


def token_logprobs(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Return float32 [B, T-1] normalized, unscaled log p(input_ids[:, 1:]).

    "Raw" scores are normalized log-probabilities without temperature scaling,
    not logits. PPO forward passes use dropout-disabled behavior, with gradients
    enabled during optimization. This function preserves the caller's grad mode;
    collection explicitly uses no_grad(). eval() currently disables dropout.
    Checkpointing that requires training mode needs a revised contract and tests,
    not a silent switch to train().

    Padding positions have no meaningful scores: apply a separate response mask
    aligned with the targets, not just attention_mask. Positions derived from the
    mask make valid scores invariant to left/right padding.
    """
    assert not model.config.is_encoder_decoder, "Expected a decoder-only model"
    assert input_ids.ndim == 2 and input_ids.shape[1] >= 2
    assert input_ids.dtype == torch.long
    assert attention_mask.shape == input_ids.shape
    assert attention_mask.device == input_ids.device
    assert ((attention_mask == 0) | (attention_mask == 1)).all()
    assert attention_mask.bool().any(dim=1).all(), "Every row needs a prefix"

    model.eval()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    logits = (
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        .logits[:, :-1]
        .float()
    )
    return logits.log_softmax(-1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)


class HFInference:
    """Generate from the training model itself, without a backend hierarchy.

    Supply padding explicitly (using EOS as padding is supported); EOS IDs come
    from model.generation_config. No tokenizer loading or model copying occurs.
    PPO forward passes use dropout-disabled behavior, with gradients enabled
    during optimization. Generation itself runs without gradients. Currently
    eval() implements the dropout policy and calls leave the model in that mode.
    The caller owns the model: close() only releases this wrapper's reference,
    without moving or mutating the model or flushing global allocator caches.
    Close is idempotent; generation and context re-entry after close raise.
    A context manager closes on both normal and exceptional exit, without
    suppressing exceptions. Generation and close must not run concurrently.
    """

    def __init__(self, model: PreTrainedModel, *, pad_token_id: int):
        assert not model.config.is_encoder_decoder, "Expected a decoder-only model"
        assert type(pad_token_id) is int and 0 <= pad_token_id < model.config.vocab_size
        self.model: PreTrainedModel | None = model
        self.pad_token_id = pad_token_id

    def close(self) -> None:
        """Release the wrapper reference, leaving external model owners intact."""
        self.model = None

    def __enter__(self) -> Self:
        if self.model is None:
            raise RuntimeError("HFInference is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def sync(self, version: int) -> None:
        """The sampler is the live actor itself: there is nothing to publish."""

    def generate_each(
        self,
        prefixes: Sequence[Sequence[int]],
        budgets: Sequence[int],
        *,
        num_samples: int,
        temperature: float,
    ) -> list[list[Completion]]:
        """One generate call per prefix, each with its own exact budget.

        Sequential calls bound KV memory to one prefix's siblings and keep the
        global RNG stream independent of how prefixes would be batched. A
        malformed result fails before any further prefix is sampled.
        """
        results = []
        for prefix, budget in zip(prefixes, budgets, strict=True):
            groups = self.generate(
                prefixes=[prefix],
                num_samples=num_samples,
                max_new_tokens=budget,
                temperature=temperature,
            )
            if len(groups) != 1 or len(groups[0]) != num_samples:
                raise ValueError("generation returned invalid completion cardinality")
            results += groups
        return results

    @torch.no_grad()
    def generate(
        self,
        prefixes: Sequence[Sequence[int]],
        *,
        num_samples: int = 1,
        max_new_tokens: int,
        temperature: float = 1.0,
        return_logprobs: bool = False,
    ) -> list[list[Completion]]:
        """Return continuation-only completions in [prefix][sample] order.

        Prefixes are exact, unpadded token IDs, including any trajectory tokens
        preceding an MC branch. max_new_tokens is a per-continuation budget;
        longest prefix + budget must fit the model context (no truncation).

        Temperature 0 selects greedy evaluation (one sample); positive values
        sample without top-k/nucleus filtering. The PPO correctness default is
        temperature 1, deliberately unlike the paper's 0.6/top_p=0.9 recipe.
        Optional logprobs describe the temperature-adjusted sampling distribution;
        greedy decoding returns raw actor scores, not point-mass logprobs. For
        PPO, use temperature 1 and cache detached token_logprobs() before the
        first optimizer step.
        """
        if self.model is None:
            raise RuntimeError("HFInference is closed")
        assert type(num_samples) is int and num_samples > 0
        assert type(max_new_tokens) is int and max_new_tokens > 0
        assert isinstance(temperature, (int, float)) and isfinite(temperature)
        assert temperature >= 0
        assert temperature > 0 or num_samples == 1, "Greedy decoding needs one sample"
        assert len(prefixes) > 0 and all(len(prefix) > 0 for prefix in prefixes)
        assert all(
            type(token) is int and 0 <= token < self.model.config.vocab_size
            for prefix in prefixes
            for token in prefix
        ), "Prefixes must contain valid integer token IDs"
        width = max(map(len, prefixes))
        assert width + max_new_tokens <= self.model.config.max_position_embeddings, (
            "Longest prefix + max_new_tokens exceeds the model context"
        )
        eos = self.model.generation_config.eos_token_id
        eos_ids = [eos] if type(eos) is int else eos
        assert isinstance(eos_ids, list) and eos_ids, "Configure EOS token IDs"
        assert all(
            type(token) is int and 0 <= token < self.model.config.vocab_size
            for token in eos_ids
        ), "Invalid EOS token IDs"

        device = self.model.device
        input_ids = torch.full(
            (len(prefixes), width), self.pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, prefix in enumerate(prefixes):
            input_ids[row, -len(prefix) :] = torch.tensor(prefix, device=device)
            attention_mask[row, -len(prefix) :] = 1

        self.model.eval()
        # Explicit overrides prevent checkpoint sampling/penalty settings from
        # silently changing the behavior distribution or forcing EOS at the budget.
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.pad_token_id,
            eos_token_id=eos_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_samples,
            num_beams=1,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_k=0,
            top_p=1.0,
            typical_p=1.0,
            min_p=None,
            top_h=None,
            epsilon_cutoff=0.0,
            eta_cutoff=0.0,
            repetition_penalty=1.0,
            encoder_repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            encoder_no_repeat_ngram_size=0,
            min_length=0,
            min_new_tokens=0,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            bad_words_ids=None,
            suppress_tokens=None,
            begin_suppress_tokens=None,
            sequence_bias=None,
            exponential_decay_length_penalty=None,
            guidance_scale=None,
            watermarking_config=None,
            token_healing=False,
            stop_strings=None,
            max_time=None,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=return_logprobs,
            output_logits=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        continuations = output.sequences[:, width:].tolist()
        scores = None
        if return_logprobs:
            scores = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            ).tolist()

        completions = []
        for row, tokens in enumerate(continuations):
            end = next(
                (i + 1 for i, token in enumerate(tokens) if token in eos_ids), None
            )
            assert end is not None or len(tokens) == max_new_tokens, (
                "Generation stopped without EOS before exhausting its budget"
            )
            completions.append(
                Completion(
                    token_ids=tokens[:end],
                    finish_reason="stop" if end is not None else "length",
                    logprobs=scores[row][:end] if scores is not None else None,
                )
            )
        return [
            completions[start : start + num_samples]
            for start in range(0, len(completions), num_samples)
        ]

"""PPO collection/scoring boundary; the optimization loop is not implemented.

PPO forward passes use dropout-disabled behavior, with gradients enabled during
optimization. Collection uses T=1 for both primary and MC rollouts. The caller
must not update the actor until all rollouts, value estimates, and detached old
scores have been collected. Evaluation has independent decoding settings.

These methods borrow a shared actor and do not own an optimizer or close the
inference wrapper. The future train() loop must own optimizer mutation and
checkpointing; the collection helpers alone cannot enforce phase ordering.
"""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from transformers import PreTrainedModel

from algorithms.vineppo import continuation_budget, estimate_vineppo
from core.data import Episode, PrefixSamples, VinePPOValueEstimate
from core.inference import Completion, HFInference, token_logprobs
from runtime.config import PPOTrainerConfig


class PPOTrainer:
    def __init__(
        self,
        config: PPOTrainerConfig,
        policy: PreTrainedModel,
        inference: HFInference,
    ):
        assert inference.model is policy, "Inference must share the training actor"
        capacity = policy.config.max_position_embeddings
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("model max_position_embeddings must be a positive integer")
        if config.max_sequence_length > capacity:
            raise ValueError("max_sequence_length exceeds model context capacity")
        self.config = config
        self.policy = policy
        self.inference = inference

    def _generation_cap(self, requested: int | None) -> int:
        if requested is None:
            return self.config.max_new_tokens
        if (
            type(requested) is not int
            or not 0 < requested <= self.config.max_new_tokens
        ):
            raise ValueError(
                "max_new_tokens must be positive and within the configured cap"
            )
        return requested

    def collect_completions(
        self,
        prefixes: Sequence[Sequence[int]],
        *,
        num_samples: int,
        max_new_tokens: int | None = None,
    ) -> list[list[Completion]]:
        """Collect within total context and the configured per-continuation cap.

        An explicit max_new_tokens may lower, but not exceed, the configured cap.
        Group by effective budget so long prompts do not shorten other rollouts.
        Verification belongs to the caller; input tokens are never truncated.
        """
        cap = self._generation_cap(max_new_tokens)
        if not prefixes:
            raise ValueError("prefixes must be nonempty")
        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        groups: dict[int, list[int]] = {}
        for index, prefix in enumerate(prefixes):
            budget = continuation_budget(
                len(prefix),
                max_sequence_length=self.config.max_sequence_length,
                max_new_tokens=cap,
            )
            groups.setdefault(budget, []).append(index)
        results: dict[int, list[Completion]] = {}
        for budget, indices in groups.items():
            completions = self.inference.generate(
                [prefixes[index] for index in indices],
                num_samples=num_samples,
                max_new_tokens=budget,
                temperature=self.config.temperature,
            )
            if len(completions) != len(indices) or any(
                len(group) != num_samples for group in completions
            ):
                raise ValueError("generation returned invalid completion cardinality")
            results.update(zip(indices, completions, strict=True))
        return [results[index] for index in range(len(prefixes))]

    def estimate_vineppo(
        self,
        episode_id: str,
        episode: Episode,
        *,
        step_boundaries: tuple[int, ...],
        score_response: Callable[[list[int], str], float],
        num_samples: int,
        max_new_tokens: int | None = None,
        record_prefix_samples: Callable[[PrefixSamples], None] | None = None,
    ) -> VinePPOValueEstimate:
        """Estimate values under the same T=1 policy used for primary rollouts.

        The caller binds the scorer to this episode's reference answer and
        training truncation policy, not the evaluator's scoring policy.
        """
        return estimate_vineppo(
            episode_id,
            episode,
            step_boundaries=step_boundaries,
            inference=self.inference,
            score_response=score_response,
            num_samples=num_samples,
            max_new_tokens=self._generation_cap(max_new_tokens),
            max_sequence_length=self.config.max_sequence_length,
            temperature=self.config.temperature,
            record_prefix_samples=record_prefix_samples,
        )

    @torch.no_grad()
    def cache_old_logprobs(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Snapshot normalized, unscaled [B, T-1] scores before actor updates.

        Apply response masks when using the scores. During optimization call
        token_logprobs() directly with gradients enabled, and reuse this cache
        unchanged across epochs rather than rescoring the old policy.
        """
        return token_logprobs(self.policy, input_ids, attention_mask).detach()

    def train(self) -> None:
        """Future loop: collect/verify -> estimate -> cache -> collate -> update.

        Complete every collection phase before any optimizer step. Reuse fixed
        old scores/advantages across update epochs. Log representative responses,
        step boundaries, MC returns and advantages, then save resumable state.
        Critic-based PPO additionally needs its own critic targets and updates;
        it is not selected by swapping a single advantage function.
        """
        raise NotImplementedError(
            "Training phase ordering and optimization are not implemented"
        )

    def update_policy(self, batch) -> None:
        raise NotImplementedError(
            "PPO loss and optimizer ownership are not implemented"
        )

    def save_checkpoint(self) -> None:
        raise NotImplementedError("Resumable training checkpoints are not implemented")

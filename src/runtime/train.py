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

from algorithms.vineppo import estimate_vineppo
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
        self.config = config
        self.policy = policy
        self.inference = inference

    def collect_completions(
        self,
        prefixes: Sequence[Sequence[int]],
        *,
        num_samples: int,
        max_new_tokens: int,
    ) -> list[list[Completion]]:
        """Collect exact-token continuations; verification belongs to the caller."""
        return self.inference.generate(
            prefixes,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=self.config.temperature,
        )

    def estimate_vineppo(
        self,
        episode_id: str,
        episode: Episode,
        *,
        step_boundaries: tuple[int, ...],
        score_response: Callable[[list[int], str], float],
        num_samples: int,
        max_new_tokens: int,
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
            max_new_tokens=max_new_tokens,
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

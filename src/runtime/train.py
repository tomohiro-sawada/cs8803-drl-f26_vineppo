"""Single-device policy-gradient orchestration with iteration-boundary checkpoints.

Borrow models, optimizers and inference wrapper; never close them.
Collection helpers also work without optimizers/reference or an algorithm config.
Training completes all rollouts, estimates and CPU score caches before updating.
PPO uses independent actor/critic models and fixed GAE targets, updating the actor
then the critic. VinePPO has no critic. GRPO/RLOO use fixed per-prompt group
advantages without segmentation, MC estimation, or a critic.
Callers must recreate consistent models, tokenizer, formatter and ordered data
for resume. Formatters and diagnostic callbacks must be deterministic and must
not mutate models, data or RNGs. Failed iterations are not resumable in memory:
recreate the trainer and load the last completed checkpoint.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from hashlib import sha256
from math import ceil, isfinite, sqrt
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from algorithms.actor import update_actor
from algorithms.grpo import GRPOConfig, group_advantages
from algorithms.ppo import (
    PPOConfig,
    TokenCritic,
    masked_gae,
    token_values,
    update_critic,
)
from algorithms.rloo import RLOOConfig, rloo_advantages
from algorithms.vineppo import (
    ScoreResponses,
    VinePPOConfig,
    continuation_budget,
    estimate_vineppo_batch,
    vineppo_advantage_method,
)
from core.data import (
    CriticBatch,
    Episode,
    EpisodeWithAdvantages,
    Example,
    PolicyGradientBatch,
    PPOValueEstimate,
    PrefixSamples,
    VinePPOValueEstimate,
    _collate_token_layout,
    collate,
    collate_critic,
)
from core.inference import Completion, HFInference, token_logprobs
from core.lora import lora_identity
from core.optimization import assert_fp32_optimizer_state, trainable_fp32_parameters
from core.segmentation import step_boundaries_from_tokens
from core.verifier import verify_math_responses, verify_response
from runtime.checkpoints import (
    COMPACT_LORA,
    FULL,
    checkpoint_models,
    file_identity,
    file_sha256,
    publish_checkpoint,
    trainable_state_keys,
    validate_tensor_state,
)
from runtime.config import PPOTrainerConfig
from runtime.schedule import PolynomialSchedule


class _Moments:
    """Streaming population moments; no samples or trajectories retained."""

    def __init__(self):
        self.count = 0
        self.mean = self.m2 = 0.0

    def add(self, values: Sequence[float]) -> None:
        for value in values:
            assert isfinite(value), "Diagnostic inputs must be finite"
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)

    def metrics(self, prefix: str) -> dict[str, float]:
        assert self.count > 0
        return {
            f"{prefix}_mean": self.mean,
            f"{prefix}_std": sqrt(self.m2 / self.count),
        }


def _training_diagnostics(
    episodes: Sequence[EpisodeWithAdvantages], *, group_size: int | None
) -> dict[str, float]:
    """Primary-only lengths/correctness and pre-normalization token advantages.

    Length includes generated special tokens. Correctness is conditional on stop:
    length-limited responses have penalty rewards, not correctness labels. Omit
    correctness when no stopped responses exist. All standard deviations are
    population statistics, not sample uncertainty estimates.
    """
    lengths = [len(e.response_ids) for e in episodes]
    advantages = torch.tensor(
        [a for e in episodes for a in e.advantages], dtype=torch.float64
    )
    assert advantages.numel() and torch.isfinite(advantages).all()
    metrics = {
        "response_length_mean": sum(lengths) / len(lengths),
        "response_length_max": float(max(lengths)),
        "raw_advantage_token_zero_fraction": (advantages == 0).double().mean().item(),
        "raw_advantage_token_std": advantages.std(correction=0).item(),
    }
    stopped = [e.reward for e in episodes if e.finish_reason == "stop"]
    if stopped:
        metrics["stopped_correctness_mean"] = sum(stopped) / len(stopped)
    if group_size is not None:
        rewards = torch.tensor(
            [e.reward for e in episodes], dtype=torch.float64
        ).reshape(-1, group_size)
        metrics["group_zero_reward_variance_fraction"] = (
            (rewards == rewards[:, :1]).all(dim=1).double().mean().item()
        )
    return metrics


class PPOTrainer:
    def __init__(
        self,
        config: PPOTrainerConfig,
        policy: PreTrainedModel,
        inference: HFInference,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        reference_policy: PreTrainedModel | None = None,
        critic: TokenCritic | None = None,
        critic_optimizer: torch.optim.Optimizer | None = None,
        compact_lora_checkpoints: bool = False,
        math_grader_workers: int = 1,
        polynomial_schedule: bool = False,
    ):
        # validate the inputs
        assert inference.model is policy, "Inference must share the training actor"
        capacity = policy.config.max_position_embeddings
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("model max_position_embeddings must be a positive integer")
        if config.max_sequence_length > capacity:
            raise ValueError("max_sequence_length exceeds model context capacity")

        # initialize the trainer state
        self.config = config
        self.policy = policy
        self.inference = inference
        self.optimizer = optimizer
        self.reference_policy = reference_policy
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        assert not compact_lora_checkpoints or lora_identity(policy) is not None, (
            "Compact checkpoints require a LoRA actor"
        )
        self.compact_lora_checkpoints = compact_lora_checkpoints
        assert type(math_grader_workers) is int and math_grader_workers > 0
        self.math_grader_workers = math_grader_workers
        self.schedule = None
        if polynomial_schedule:
            assert optimizer is not None
            self.steps_per_iteration = config.update_epochs * ceil(
                config.prompts_per_iteration
                * config.num_samples
                / config.minibatch_size
            )
            self.schedule = PolynomialSchedule(
                [o for o in (optimizer, critic_optimizer) if o is not None],
                self.steps_per_iteration * config.num_iterations,
            )
        self._frozen_checkpoint = None
        self.iteration = 0  # Number of fully completed iterations.
        self._generator: torch.Generator | None = None
        self._identity: str | None = None
        # Runner-owned identity for held-out evaluation; legacy checkpoints lack it.
        self.evaluation_provenance: dict | None = None
        self._at_boundary = True

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

    def estimate_vineppo_batch(
        self,
        items: Sequence[tuple[str, Episode, tuple[int, ...], ScoreResponses]],
        *,
        record_prefix_samples: Callable[[PrefixSamples], None] | None = None,
    ) -> list[VinePPOValueEstimate]:
        """Estimate every episode of an iteration from one batch of MC requests."""
        return estimate_vineppo_batch(
            items,
            inference=self.inference,
            num_samples=self.config.mc_num_samples,
            max_new_tokens=self.config.max_new_tokens,
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

    def _validate_training(self) -> None:
        assert isinstance(
            self.config.algorithm_config,
            (VinePPOConfig, PPOConfig, GRPOConfig, RLOOConfig),
        ), "Unsupported training algorithm"
        if isinstance(self.config.algorithm_config, (GRPOConfig, RLOOConfig)):
            assert self.config.num_samples >= 2, (
                "Grouped training requires num_samples >= 2"
            )
        assert self.optimizer is not None, "Training requires an explicit optimizer"
        assert self.reference_policy is not None, "Training requires a reference policy"
        assert self.inference.model is self.policy, (
            "Inference must share the training actor"
        )
        actor = list(self.policy.parameters())
        reference = list(self.reference_policy.parameters())
        assert actor and reference
        assert not {id(p) for p in actor} & {id(p) for p in reference}, (
            "Reference must not share actor parameters"
        )
        actor_tensors = actor + list(self.policy.buffers())
        ref_tensors = reference + list(self.reference_policy.buffers())
        assert all(t.device.type != "meta" for t in actor_tensors + ref_tensors)
        storage = {
            (t.device, t.untyped_storage().data_ptr())
            for t in actor_tensors
            if t.numel()
        }
        assert not storage.intersection(
            (t.device, t.untyped_storage().data_ptr()) for t in ref_tensors if t.numel()
        ), "Reference must not share actor storage"
        owned = [p for group in self.optimizer.param_groups for p in group["params"]]
        trainable = trainable_fp32_parameters(self.policy)
        assert trainable and len(owned) == len({id(p) for p in owned})
        assert {id(p) for p in owned} == {id(p) for p in trainable}, (
            "Optimizer must own exactly the trainable actor parameters"
        )
        assert len({p.device for p in actor}) == len({p.device for p in reference}) == 1
        assert not self.reference_policy.config.is_encoder_decoder
        assert self.reference_policy.config.vocab_size == self.policy.config.vocab_size
        assert (
            self.reference_policy.config.max_position_embeddings
            >= self.config.max_sequence_length
        )

        if isinstance(self.config.algorithm_config, PPOConfig):
            self.config.algorithm_config.__post_init__()
            assert isinstance(self.critic, TokenCritic), "PPO requires a TokenCritic"
            assert self.critic_optimizer is not None, "PPO requires a critic optimizer"
            critic_tensors = list(self.critic.parameters()) + list(
                self.critic.buffers()
            )
            assert all(t.device.type != "meta" for t in critic_tensors)
            assert len({p.device for p in self.critic.parameters()}) == 1
            other_storage = storage | {
                (t.device, t.untyped_storage().data_ptr())
                for t in ref_tensors
                if t.numel()
            }
            assert not other_storage.intersection(
                (t.device, t.untyped_storage().data_ptr())
                for t in critic_tensors
                if t.numel()
            ), "Critic must not share actor/reference storage"
            owned = [
                p
                for group in self.critic_optimizer.param_groups
                for p in group["params"]
            ]
            trainable = trainable_fp32_parameters(self.critic)
            assert trainable and len(owned) == len({id(p) for p in owned})
            assert {id(p) for p in owned} == {id(p) for p in trainable}, (
                "Critic optimizer must own exactly the trainable critic parameters"
            )
            assert self.critic.config.vocab_size == self.policy.config.vocab_size
            assert (
                self.critic.config.max_position_embeddings
                >= self.config.max_sequence_length
            )
        else:
            assert self.critic is None and self.critic_optimizer is None, (
                "VinePPO/grouped training must not receive a critic or critic optimizer"
            )

        self.reference_policy.requires_grad_(False)
        self.reference_policy.zero_grad(set_to_none=True)
        self.reference_policy.eval()
        if self._generator is None:
            self._generator = torch.Generator(device="cpu").manual_seed(
                self.config.seed
            )

    def _prepare_inputs(
        self,
        examples: Sequence[Example],
        tokenizer: PreTrainedTokenizerBase,
        task: Literal["math", "gsm8k"],
        formatter: Callable[[str], str],
        reference_is_solution: bool,
    ) -> tuple[list[list[int]], str]:
        assert len(examples) >= self.config.prompts_per_iteration
        assert task in ("math", "gsm8k")
        assert not reference_is_solution or task == "math"
        assert callable(formatter), "An explicit prompt formatter is required"
        assert tokenizer.is_fast, "Exact segmentation requires a fast tokenizer"
        assert len({e.id for e in examples}) == len(examples), (
            "Example IDs must be unique"
        )
        digest = sha256()

        def add(value) -> None:
            digest.update(
                json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
            )
            digest.update(b"\n")

        add([task, reference_is_solution, self.inference.pad_token_id])
        add(tokenizer.backend_tokenizer.to_str())
        add(tokenizer.all_special_ids)
        add(self.policy.generation_config.to_dict())
        prompts = []
        for example in examples:
            assert all(
                isinstance(s, str) and s.strip()
                for s in (
                    example.id,
                    example.question,
                    example.reference_answer,
                )
            ), "Example fields must be nonempty strings"
            prompt = formatter(example.question)
            assert isinstance(prompt, str) and prompt, (
                "Formatter must return nonempty text"
            )
            ids = tokenizer(
                prompt, add_special_tokens=True, padding=False, truncation=False
            )["input_ids"]
            assert ids and all(
                type(t) is int and 0 <= t < self.policy.config.vocab_size for t in ids
            )
            prompts.append(ids)
            add([asdict(example), prompt, ids])
        return prompts, digest.hexdigest()

    def train(
        self,
        examples: Sequence[Example],
        tokenizer: PreTrainedTokenizerBase,
        *,
        task: Literal["math", "gsm8k"],
        formatter: Callable[[str], str],
        reference_is_solution: bool = False,
        record_prefix_samples: Callable[[PrefixSamples], None] | None = None,
        record_value_estimate: Callable[[VinePPOValueEstimate | PPOValueEstimate], None]
        | None = None,
        record_metrics: Callable[[dict[str, float]], None] | None = None,
        stop_at_boundary: Callable[[], bool] | None = None,
    ) -> list[dict[str, float]]:
        """Run to num_iterations total; return only this call's iteration metrics.

        Actor/critic diagnostics are arithmetic means over their optimizer steps
        (not a global token-weighted mean). Estimate callbacks are synchronous and their
        payloads are not retained. Iteration metrics use one-based completed counts.
        Prompt sampling and update shuffles share a dedicated seeded CPU generator;
        generation uses the caller's global torch RNG, restored on checkpoint load.
        GRPO/RLOO reject value/prefix callbacks: they do not estimate values.
        """
        assert self._at_boundary, (
            "An incomplete iteration requires a fresh trainer/load"
        )
        self._validate_training()
        algorithm = self.config.algorithm_config
        grouped = isinstance(algorithm, (GRPOConfig, RLOOConfig))
        if grouped:
            assert record_value_estimate is None and record_prefix_samples is None, (
                "Grouped training does not support value/prefix callbacks"
            )
        prompts, identity = self._prepare_inputs(
            examples,
            tokenizer,
            task,
            formatter,
            reference_is_solution,
        )
        assert self._identity is None or self._identity == identity, (
            "Dataset/prompt identity mismatch"
        )
        self._identity = identity
        assert self.iteration <= self.config.num_iterations
        eos = self.policy.generation_config.eos_token_id
        eos_ids = [eos] if type(eos) is int else eos
        assert isinstance(eos_ids, list) and eos_ids
        assert all(
            type(t) is int and 0 <= t < self.policy.config.vocab_size for t in eos_ids
        )
        history = []
        while self.iteration < self.config.num_iterations:
            if stop_at_boundary is not None and stop_at_boundary():
                break
            self._at_boundary = False
            # Freeze the behavior actor: every sampler serves this iteration's weights.
            self.inference.sync(self.iteration)
            indices = torch.randperm(len(examples), generator=self._generator)[
                : self.config.prompts_per_iteration
            ].tolist()
            selected = [prompts[i] for i in indices]
            # collect_completions validates EVERY selected budget before generating.
            completions = self.collect_completions(
                selected, num_samples=self.config.num_samples
            )
            episodes = []
            prefix_values = _Moments()
            step_counts = _Moments()
            raw_episodes, episode_ids, scorers = [], [], []
            for index, groups in zip(indices, completions, strict=True):
                example = examples[index]

                def score_responses(
                    responses: Sequence[Completion],
                    example: Example = example,
                ) -> list[float]:
                    rewards = [float(self.config.truncation_reward)] * len(responses)
                    stopped = {}
                    for sample, response in enumerate(responses):
                        assert response.finish_reason in ("stop", "length"), (
                            "Unsupported finish reason"
                        )
                        if response.finish_reason == "stop":
                            stopped[sample] = tokenizer.decode(
                                response.token_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                    try:
                        if task == "math" and self.math_grader_workers > 1:
                            verified = verify_math_responses(
                                list(stopped.values()),
                                example.reference_answer,
                                question=example.question,
                                reference_is_solution=reference_is_solution,
                                workers=self.math_grader_workers,
                            )
                        else:
                            verified = [
                                verify_response(
                                    text,
                                    example.reference_answer,
                                    task=task,
                                    question=example.question,
                                    reference_is_solution=reference_is_solution,
                                )
                                for text in stopped.values()
                            ]
                    except Exception as error:
                        error.add_note(
                            f"Verification example {example.id}; stopped samples "
                            f"{list(stopped.items())!r:.512}"
                        )
                        raise
                    for sample, result in zip(stopped, verified, strict=True):
                        rewards[sample] = float(result.correct)
                    return rewards

                rewards = score_responses(groups)
                for sample_index, (completion, reward) in enumerate(
                    zip(groups, rewards, strict=True)
                ):
                    episode = Episode(
                        prompts[index],
                        completion.token_ids,
                        reward,
                        completion.finish_reason,
                    )
                    raw_episodes.append(episode)
                    episode_ids.append(f"{self.iteration}:{example.id}:{sample_index}")
                    scorers.append(score_responses)
            if isinstance(algorithm, VinePPOConfig):
                # Segment every primary response, then queue all MC work at once.
                estimates = self.estimate_vineppo_batch(
                    [
                        (
                            episode_id,
                            episode,
                            step_boundaries_from_tokens(
                                episode.response_ids,
                                tokenizer,
                                dataset=task,
                                eos_token_ids=eos_ids,
                                context_ids=episode.prompt_ids,
                            ),
                            scorer,
                        )
                        for episode_id, episode, scorer in zip(
                            episode_ids, raw_episodes, scorers, strict=True
                        )
                    ],
                    record_prefix_samples=record_prefix_samples,
                )
                for episode, estimate in zip(raw_episodes, estimates, strict=True):
                    prefix_values.add(estimate.mc_prefix_values)
                    step_counts.add([len(estimate.mc_prefix_values)])
                    if record_value_estimate is not None:
                        record_value_estimate(estimate)
                    episodes.append(
                        EpisodeWithAdvantages(
                            episode.prompt_ids,
                            episode.response_ids,
                            episode.reward,
                            episode.finish_reason,
                            vineppo_advantage_method(estimate),
                        )
                    )
            critic_batch = None
            if grouped:
                rewards = torch.tensor([e.reward for e in raw_episodes]).reshape(
                    len(indices), self.config.num_samples
                )
                advantages = (
                    group_advantages(rewards, eps=algorithm.eps)
                    if isinstance(algorithm, GRPOConfig)
                    else rloo_advantages(rewards)
                )
                episodes = [
                    EpisodeWithAdvantages(
                        **asdict(episode),
                        advantages=[advantage] * len(episode.response_ids),
                    )
                    for episode, advantage in zip(
                        raw_episodes, advantages.flatten().tolist(), strict=True
                    )
                ]
            if isinstance(self.config.algorithm_config, PPOConfig):
                episodes, estimates = self._estimate_ppo(raw_episodes, episode_ids)
                critic_batch = collate_critic(
                    estimates, pad_token_id=self.inference.pad_token_id
                )
                if record_value_estimate is not None:
                    for estimate in estimates:
                        record_value_estimate(estimate)
            diagnostics = _training_diagnostics(
                episodes, group_size=self.config.num_samples if grouped else None
            )
            if isinstance(algorithm, VinePPOConfig):
                diagnostics.update(step_counts.metrics("vineppo/steps_per_response"))
                diagnostics["vineppo/step_count"] = float(prefix_values.count)
                diagnostics.update(prefix_values.metrics("vineppo/mc_prefix_value"))
            old, ref = self._cache_scores(episodes)
            batch = collate(
                episodes,
                old_logprobs=old,
                ref_logprobs=ref,
                pad_token_id=self.inference.pad_token_id,
            )
            updates = self.update_policy(batch)
            metrics = {
                key: sum(row[key] for row in updates) / len(updates)
                for key in updates[0]
            }
            if critic_batch is not None:
                critic_updates = self.update_value(critic_batch)
                metrics.update(
                    {
                        f"critic/{key}": sum(row[key] for row in critic_updates)
                        / len(critic_updates)
                        for key in critic_updates[0]
                    }
                )
                metrics["critic/optimizer_steps"] = len(critic_updates)
            metrics.update(
                iteration=self.iteration + 1,
                optimizer_steps=len(updates),
                episodes=len(episodes),
                reward_mean=sum(e.reward for e in episodes) / len(episodes),
                truncation_fraction=sum(e.finish_reason == "length" for e in episodes)
                / len(episodes),
            )
            metrics.update(diagnostics)
            self.iteration += 1
            self._at_boundary = True
            history.append(metrics)
            if record_metrics is not None:
                record_metrics(dict(metrics))
            if (
                self.iteration % self.config.checkpoint_every == 0
                or self.iteration == self.config.num_iterations
            ):
                self.save_checkpoint()
        return history

    @torch.no_grad()
    def _estimate_ppo(
        self, episodes: Sequence[Episode], episode_ids: Sequence[str]
    ) -> tuple[list[EpisodeWithAdvantages], list[PPOValueEstimate]]:
        """Cache pre-update values and GAE targets; terminal rewards, no KL shaping.

        Stops and length limits are both terminal under the training reward policy.
        Actor KL is handled separately by update_actor, just as in VinePPO.
        """
        config = self.config.algorithm_config
        assert isinstance(config, PPOConfig) and self.critic is not None
        assert len(episodes) == len(episode_ids) > 0
        device = next(self.critic.parameters()).device
        prepared, estimates = [], []
        for start in range(0, len(episodes), self.config.microbatch_size):
            chunk = episodes[start : start + self.config.microbatch_size]
            layout = _collate_token_layout(
                chunk, pad_token_id=self.inference.pad_token_id
            )
            values = (
                token_values(
                    self.critic,
                    layout.input_ids.to(device),
                    layout.attention_mask.to(device),
                )
                .detach()
                .cpu()
            )
            rewards = torch.zeros_like(values)
            for row, episode in enumerate(chunk):
                assert episode.finish_reason in ("stop", "length")
                last = len(episode.prompt_ids) + len(episode.response_ids) - 2
                rewards[row, last] = episode.reward
            advantages, returns = masked_gae(
                rewards,
                values,
                layout.response_mask,
                gamma=config.gamma,
                lam=config.lam,
            )
            for row, episode in enumerate(chunk):
                mask = layout.response_mask[row]
                prepared.append(
                    EpisodeWithAdvantages(
                        **asdict(episode), advantages=advantages[row, mask].tolist()
                    )
                )
                estimates.append(
                    PPOValueEstimate(
                        episode_id=episode_ids[start + row],
                        episode=episode,
                        critic_value_estimates=values[row, mask].tolist(),
                        return_estimates=returns[row, mask].tolist(),
                    )
                )
        return prepared, estimates

    @torch.no_grad()
    def _cache_scores(
        self, episodes: Sequence[Episode]
    ) -> tuple[list[list[float]], list[list[float]]]:
        assert self.reference_policy is not None
        old, ref = [], []
        for start in range(0, len(episodes), self.config.microbatch_size):
            layout = _collate_token_layout(
                episodes[start : start + self.config.microbatch_size],
                pad_token_id=self.inference.pad_token_id,
            )
            for model, output in ((self.policy, old), (self.reference_policy, ref)):
                ids = layout.input_ids.to(model.device)
                mask = layout.attention_mask.to(model.device)
                scores = (
                    (
                        self.cache_old_logprobs(ids, mask)
                        if model is self.policy
                        else token_logprobs(model, ids, mask)
                    )
                    .detach()
                    .cpu()
                )
                output.extend(
                    row[positions].tolist()
                    for row, positions in zip(scores, layout.response_mask, strict=True)
                )
        return old, ref

    def update_policy(self, batch: PolicyGradientBatch) -> list[dict[str, float]]:
        """Delegate all epochs/minibatches/microbatches; standalone calls invalidate checkpoints."""
        self._validate_training()
        size = len(batch.input_ids)
        objective = (
            "grpo"
            if isinstance(self.config.algorithm_config, GRPOConfig)
            else "rloo"
            if isinstance(self.config.algorithm_config, RLOOConfig)
            else "ppo"
        )
        if objective == "ppo" and (
            self.config.minibatch_size == 1 or size % self.config.minibatch_size == 1
        ):
            assert (batch.response_mask.sum(dim=1) > 1).all(), (
                "Singleton minibatches require at least two response tokens per episode"
            )
        self._at_boundary = False
        return update_actor(
            self.policy,
            self.optimizer,
            batch,
            epochs=self.config.update_epochs,
            minibatch_size=self.config.minibatch_size,
            microbatch_size=self.config.microbatch_size,
            clip_eps=self.config.clip_eps,
            kl_coef=self.config.kl_coef,
            max_grad_norm=self.config.max_grad_norm,
            generator=self._generator,
            objective=objective,
        )

    def update_value(self, batch: CriticBatch) -> list[dict[str, float]]:
        """Optimize fixed critic returns after actor epochs; never regenerate GAE."""
        self._validate_training()
        assert isinstance(self.config.algorithm_config, PPOConfig)
        self._at_boundary = False
        return update_critic(
            self.critic,
            self.critic_optimizer,
            batch,
            epochs=self.config.update_epochs,
            minibatch_size=self.config.minibatch_size,
            microbatch_size=self.config.microbatch_size,
            max_grad_norm=self.config.max_grad_norm,
            generator=self._generator,
            value_clip_eps=self.config.algorithm_config.value_clip_eps,
        )

    def _config_state(self) -> dict:
        state = asdict(self.config)
        state["checkpoint_dir"] = str(self.config.checkpoint_dir)
        algorithm = type(self.config.algorithm_config)
        if (
            algorithm is PPOConfig
            and state["algorithm_config"]["value_clip_eps"] is None
        ):
            del state["algorithm_config"]["value_clip_eps"]
        state["algorithm_type"] = f"{algorithm.__module__}.{algorithm.__qualname__}"
        return state

    @staticmethod
    def _optimizer_identity(model, optimizer) -> dict:
        names = {id(p): name for name, p in model.named_parameters()}
        return {
            "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "parameters": [
                [names[id(p)] for p in group["params"]]
                for group in optimizer.param_groups
            ],
        }

    def _model_identity(self) -> list[dict]:
        return [
            {
                "class": f"{type(model).__module__}.{type(model).__qualname__}",
                "config": model.config.to_dict(),
                **(
                    {"lora": identity}
                    if (identity := lora_identity(model)) is not None
                    else {}
                ),
                "dtypes": {
                    name: str(value.dtype) for name, value in model.state_dict().items()
                },
            }
            for model in (
                [self.policy, self.reference_policy]
                + ([self.critic] if self.critic is not None else [])
            )
        ]

    def _separate_lora_state(self):
        trainable = trainable_state_keys(self.policy)
        policy = self.policy.state_dict()
        assert trainable and trainable <= policy.keys(), (
            "Checkpoint omits trainable parameters"
        )
        frozen = {
            "policy": {
                key: value for key, value in policy.items() if key not in trainable
            },
            "reference": self.reference_policy.state_dict(),
        }
        # Include persistent buffers in the immutable snapshot and its mutation check.
        signature = {
            (model, key): (value.data_ptr(), value._version, value.dtype, value.shape)
            for model, weights in frozen.items()
            for key, value in weights.items()
        }
        return {key: policy[key] for key in sorted(trainable)}, frozen, signature

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save a completed boundary; default checkpoint_dir/iteration-N.pt.

        Tensor/state-dict payloads plus primitive metadata only; no model objects,
        tokenizer, formatter, examples, diagnostics or global Python/NumPy RNGs.
        Compact LoRA checkpoints share a verified frozen actor/reference file.
        """
        assert self._at_boundary and self._identity is not None, (
            "Checkpoint requires a completed training boundary"
        )
        self._validate_training()
        path = (
            Path(path)
            if path is not None
            else Path(self.config.checkpoint_dir) / f"iteration-{self.iteration}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "optimizer": self.optimizer.state_dict(),
            "optimizer_identity": self._optimizer_identity(self.policy, self.optimizer),
            "model_identity": self._model_identity(),
            "config": self._config_state(),
            "iteration": self.iteration,
            "identity": self._identity,
            "evaluation_provenance": self.evaluation_provenance,
            "generator": self._generator.get_state(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "mps_rng": torch.mps.get_rng_state()
            if torch.backends.mps.is_available()
            else None,
        }
        if self.compact_lora_checkpoints:
            trainable, frozen, signature = self._separate_lora_state()
            if self._frozen_checkpoint is None:
                blob = (path.parent / "frozen-lora.pt").resolve()
                publish_checkpoint(blob, frozen)
                spec = {"path": str(blob), "sha256": file_sha256(blob)}
                self._frozen_checkpoint = spec, file_identity(blob), signature
            spec, identity, original = self._frozen_checkpoint
            assert signature == original, "Frozen model parameters or buffers changed"
            assert file_identity(Path(spec["path"])) == identity, (
                "Frozen model file changed"
            )
            state.update(format_version=COMPACT_LORA, policy=trainable, frozen=spec)
        else:
            state.update(
                format_version=FULL,
                policy=self.policy.state_dict(),
                reference=self.reference_policy.state_dict(),
            )
        if self.critic is not None:
            state["critic"] = self.critic.state_dict()
            state["critic_optimizer"] = self.critic_optimizer.state_dict()
            state["critic_optimizer_identity"] = self._optimizer_identity(
                self.critic, self.critic_optimizer
            )
            state["critic_backbone_class"] = (
                f"{type(self.critic.backbone).__module__}.{type(self.critic.backbone).__qualname__}"
            )
        if self.schedule is not None:
            state["scheduler"] = self.schedule.state_dict()
        publish_checkpoint(path, state)
        return path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        examples: Sequence[Example],
        tokenizer: PreTrainedTokenizerBase,
        task: Literal["math", "gsm8k"],
        formatter: Callable[[str], str],
        reference_is_solution: bool = False,
    ) -> None:
        """Restore a boundary after validating caller-recreated inputs.

        Only num_iterations (total target), checkpoint_every and checkpoint_dir
        may differ. Same optimizer class/parameter ordering and RNG device topology
        are required. Bitwise reproducibility still depends on deterministic kernels
        and identical software/hardware. Errors propagate; a failed load should be
        discarded rather than used as a partially restored trainer.
        """
        assert self._at_boundary, "Cannot load during an incomplete iteration"
        self._validate_training()
        path = Path(path)
        checkpoint_identity = file_identity(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        current = self._config_state()
        ignored = {"num_iterations", "checkpoint_every", "checkpoint_dir"}
        assert {k: v for k, v in state["config"].items() if k not in ignored} == {
            k: v for k, v in current.items() if k not in ignored
        }, "Checkpoint training config mismatch"
        assert (
            type(state["iteration"]) is int
            and 0 <= state["iteration"] <= self.config.num_iterations
        )
        _, identity = self._prepare_inputs(
            examples, tokenizer, task, formatter, reference_is_solution
        )
        assert identity == state["identity"], "Dataset/prompt identity mismatch"
        assert self._model_identity() == state["model_identity"], (
            "Model execution identity mismatch"
        )
        assert (
            self._optimizer_identity(self.policy, self.optimizer)
            == state["optimizer_identity"]
        ), "Optimizer identity mismatch"
        assert ("scheduler" in state) == (self.schedule is not None), (
            "Checkpoint learning-rate schedule presence mismatch"
        )
        if self.schedule is not None:
            assert state["scheduler"]["identity"] == self.schedule.identity(), (
                "Checkpoint learning-rate schedule mismatch"
            )
        assert_fp32_optimizer_state(state["optimizer"]["state"])
        if self.critic is not None:
            assert (
                self._optimizer_identity(self.critic, self.critic_optimizer)
                == state["critic_optimizer_identity"]
            ), "Critic optimizer identity mismatch"
            assert_fp32_optimizer_state(state["critic_optimizer"]["state"])
            assert state["critic_backbone_class"] == (
                f"{type(self.critic.backbone).__module__}.{type(self.critic.backbone).__qualname__}"
            ), "Critic backbone identity mismatch"
        assert len(state["cuda_rng"]) == torch.cuda.device_count(), (
            "CUDA RNG topology mismatch"
        )
        assert (state["mps_rng"] is not None) == torch.backends.mps.is_available(), (
            "MPS RNG availability mismatch"
        )
        compact = state.get("format_version") == COMPACT_LORA
        if compact:
            assert state["policy"].keys() == trainable_state_keys(self.policy), (
                "Compact policy does not contain exactly the trainable state"
            )
            spec = state["frozen"]
            frozen_path = Path(spec["path"])
            frozen_identity = file_identity(frozen_path)
        policy, reference = checkpoint_models(state)
        assert reference is not None, "Training requires a checkpoint reference model"
        validate_tensor_state(self.policy, policy)
        validate_tensor_state(self.reference_policy, reference)
        if self.critic is not None:
            validate_tensor_state(self.critic, state["critic"])
        self._at_boundary = False
        self.policy.load_state_dict(policy, strict=True)
        self.reference_policy.load_state_dict(reference, strict=True)
        assert file_identity(path) == checkpoint_identity, (
            "Checkpoint changed while loading"
        )
        if compact:
            assert file_identity(frozen_path) == frozen_identity, (
                "Frozen model changed while loading"
            )
        if compact and self.compact_lora_checkpoints:
            _, _, signature = self._separate_lora_state()
            self._frozen_checkpoint = spec, frozen_identity, signature
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer.zero_grad(set_to_none=True)
        if self.critic is not None:
            self.critic.load_state_dict(state["critic"], strict=True)
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.critic_optimizer.zero_grad(set_to_none=True)
        if self.schedule is not None:
            self.schedule.load_state_dict(
                state["scheduler"], state["iteration"] * self.steps_per_iteration
            )
        self._generator.set_state(state["generator"])
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state["mps_rng"] is not None:
            torch.mps.set_rng_state(state["mps_rng"])
        self.iteration = state["iteration"]
        self._identity = identity
        self._at_boundary = True

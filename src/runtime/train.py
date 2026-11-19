"""Single-device VinePPO orchestration with iteration-boundary checkpoints.

Borrow the actor, optimizer, reference and inference wrapper; never close them.
Collection helpers also work without an optimizer/reference or VinePPO config.
Training completes all rollouts, estimates and CPU score caches before updating.
Callers must recreate consistent models, tokenizer, formatter and ordered data
for resume. Formatters and diagnostic callbacks must be deterministic and must
not mutate models, data or RNGs. Failed iterations are not resumable in memory:
recreate the trainer and load the last completed checkpoint.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from algorithms.actor import update_actor
from algorithms.vineppo import (
    VinePPOConfig,
    continuation_budget,
    estimate_vineppo,
    vineppo_advantage_method,
)
from core.data import (
    Episode,
    EpisodeWithAdvantages,
    Example,
    PolicyGradientBatch,
    PrefixSamples,
    VinePPOValueEstimate,
    _collate_token_layout,
    collate,
)
from core.inference import Completion, HFInference, token_logprobs
from core.segmentation import step_boundaries_from_tokens
from core.verifier import verify_response
from runtime.config import PPOTrainerConfig


class PPOTrainer:
    def __init__(
        self,
        config: PPOTrainerConfig,
        policy: PreTrainedModel,
        inference: HFInference,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        reference_policy: PreTrainedModel | None = None,
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
        self.iteration = 0  # Number of fully completed iterations.
        self._generator: torch.Generator | None = None
        self._identity: str | None = None
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

    def _validate_training(self) -> None:
        assert isinstance(self.config.algorithm_config, VinePPOConfig), (
            "Training supports only VinePPOConfig"
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
        trainable = [p for p in actor if p.requires_grad]
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
        record_value_estimate: Callable[[VinePPOValueEstimate], None] | None = None,
        record_metrics: Callable[[dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        """Run to num_iterations total; return only this call's iteration metrics.

        Actor diagnostics are arithmetic means over optimizer steps (not a global
        token-weighted mean). Raw MC/estimate callbacks are synchronous and their
        payloads are not retained. Iteration metrics use one-based completed counts.
        Prompt sampling and update shuffles share a dedicated seeded CPU generator;
        generation uses the caller's global torch RNG, restored on checkpoint load.
        """
        assert self._at_boundary, (
            "An incomplete iteration requires a fresh trainer/load"
        )
        self._validate_training()
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
            self._at_boundary = False
            indices = torch.randperm(len(examples), generator=self._generator)[
                : self.config.prompts_per_iteration
            ].tolist()
            selected = [prompts[i] for i in indices]
            # collect_completions validates EVERY selected budget before generating.
            completions = self.collect_completions(
                selected, num_samples=self.config.num_samples
            )
            episodes = []
            for index, groups in zip(indices, completions, strict=True):
                example = examples[index]

                def score_response(
                    ids: list[int],
                    reason: str,
                    example: Example = example,
                ) -> float:
                    assert reason in ("stop", "length"), "Unsupported finish reason"
                    if reason == "length":
                        return float(self.config.truncation_reward)
                    text = tokenizer.decode(
                        ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    return float(
                        verify_response(
                            text,
                            example.reference_answer,
                            task=task,
                            question=example.question,
                            reference_is_solution=reference_is_solution,
                        ).correct
                    )

                for sample_index, completion in enumerate(groups):
                    episode = Episode(
                        prompts[index],
                        completion.token_ids,
                        score_response(completion.token_ids, completion.finish_reason),
                        completion.finish_reason,
                    )
                    estimate = self.estimate_vineppo(
                        f"{self.iteration}:{example.id}:{sample_index}",
                        episode,
                        step_boundaries=step_boundaries_from_tokens(
                            episode.response_ids,
                            tokenizer,
                            dataset=task,
                            eos_token_ids=eos_ids,
                        ),
                        score_response=score_response,
                        num_samples=self.config.mc_num_samples,
                        record_prefix_samples=record_prefix_samples,
                    )
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
            metrics.update(
                iteration=self.iteration + 1,
                optimizer_steps=len(updates),
                episodes=len(episodes),
                reward_mean=sum(e.reward for e in episodes) / len(episodes),
                truncation_fraction=sum(e.finish_reason == "length" for e in episodes)
                / len(episodes),
            )
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
        if self.config.minibatch_size == 1 or size % self.config.minibatch_size == 1:
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
        )

    def _config_state(self) -> dict:
        state = asdict(self.config)
        state["checkpoint_dir"] = str(self.config.checkpoint_dir)
        return state

    def _optimizer_identity(self) -> dict:
        names = {id(p): name for name, p in self.policy.named_parameters()}
        return {
            "class": f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}",
            "parameters": [
                [names[id(p)] for p in group["params"]]
                for group in self.optimizer.param_groups
            ],
        }

    def _model_identity(self) -> list[dict]:
        return [
            {
                "class": f"{type(model).__module__}.{type(model).__qualname__}",
                "config": model.config.to_dict(),
                "dtypes": {
                    name: str(value.dtype) for name, value in model.state_dict().items()
                },
            }
            for model in (self.policy, self.reference_policy)
        ]

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save a completed boundary; default checkpoint_dir/iteration-N.pt.

        Tensor/state-dict payloads plus primitive metadata only; no model objects,
        tokenizer, formatter, examples, diagnostics or global Python/NumPy RNGs.
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
            "policy": self.policy.state_dict(),
            "reference": self.reference_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_identity": self._optimizer_identity(),
            "model_identity": self._model_identity(),
            "config": self._config_state(),
            "iteration": self.iteration,
            "identity": self._identity,
            "generator": self._generator.get_state(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "mps_rng": torch.mps.get_rng_state()
            if torch.backends.mps.is_available()
            else None,
        }
        # Replace only after a complete write, preserving any previous checkpoint.
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
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
        assert self._optimizer_identity() == state["optimizer_identity"], (
            "Optimizer identity mismatch"
        )
        assert len(state["cuda_rng"]) == torch.cuda.device_count(), (
            "CUDA RNG topology mismatch"
        )
        assert (state["mps_rng"] is not None) == torch.backends.mps.is_available(), (
            "MPS RNG availability mismatch"
        )
        self._at_boundary = False
        self.policy.load_state_dict(state["policy"], strict=True)
        self.reference_policy.load_state_dict(state["reference"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer.zero_grad(set_to_none=True)
        self._generator.set_state(state["generator"])
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state["mps_rng"] is not None:
            torch.mps.set_rng_state(state["mps_rng"])
        self.iteration = state["iteration"]
        self._identity = identity
        self._at_boundary = True

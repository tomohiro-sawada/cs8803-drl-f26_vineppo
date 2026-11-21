"""Exercise the training collection boundary with a real, local tiny HF model."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

import algorithms.actor as actor_module
import runtime.train as train_module
from algorithms.grpo import GRPOConfig
from algorithms.rloo import RLOOConfig
from algorithms.vineppo import VinePPOConfig
from core.data import Episode, EpisodeWithAdvantages, Example, collate
from core.inference import Completion, HFInference, token_logprobs
from runtime.config import PPOTrainerConfig
from runtime.train import PPOTrainer


@pytest.fixture
def trainer(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=23,
                n_positions=24,
                n_embd=16,
                n_layer=1,
                n_head=2,
                eos_token_id=2,
                pad_token_id=0,
                resid_pdrop=0.5,
                embd_pdrop=0.5,
                attn_pdrop=0.5,
                attn_implementation="sdpa",
            )
        )
    return PPOTrainer(
        PPOTrainerConfig(
            checkpoint_dir=tmp_path,
            algorithm_config=VinePPOConfig(),
            max_sequence_length=24,
            max_new_tokens=3,
        ),
        model,
        HFInference(model, pad_token_id=0),
    )


def test_collection_mc_and_cached_scores_use_one_unchanged_policy(trainer, monkeypatch):
    policy = trainer.policy
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    generate = trainer.inference.generate
    temperatures = []
    grad_modes = []

    def record_generate(*args, **kwargs):
        temperatures.append(kwargs["temperature"])
        return generate(*args, **kwargs)

    def record_forward(module, args):
        assert not any(child.training for child in module.modules())
        grad_modes.append(torch.is_grad_enabled())

    monkeypatch.setattr(trainer.inference, "generate", record_generate)
    handle = policy.register_forward_pre_hook(record_forward)
    try:
        policy.train()  # Nonzero dropout makes the disabled-dropout contract observable.
        prompt = [4, 5]
        completion = trainer.collect_completions(
            [prompt], num_samples=1, max_new_tokens=3
        )[0][0]
        episode = Episode(prompt, completion.token_ids, 1.0, completion.finish_reason)
        (estimate,) = trainer.estimate_vineppo_batch(
            [
                (
                    "example-0",
                    episode,
                    (0, len(episode.response_ids)),
                    lambda responses: [0.25] * len(responses),
                )
            ]
        )
        assert estimate.mc_prefix_values == (0.25,)
        assert estimate.raw_step_advantages == (0.75,)
        assert temperatures == [1.0, 1.0]

        ids = torch.tensor([episode.prompt_ids + episode.response_ids])
        mask = torch.ones_like(ids, dtype=torch.bool)
        old = trainer.cache_old_logprobs(ids, mask)
        assert old.grad_fn is None and not old.requires_grad
        assert grad_modes and not any(grad_modes)
        for name, tensor in policy.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

        cached = old.clone()
        response = slice(len(prompt) - 1, None)
        new = token_logprobs(policy, ids, mask)
        assert new.requires_grad and grad_modes[-1]
        ratios = (new[:, response] - old[:, response]).exp()
        torch.testing.assert_close(
            ratios, torch.ones_like(ratios), rtol=1e-5, atol=1e-6
        )

        # Synthetic optimization checks cache lifetime, not an implemented PPO loss.
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        for _ in range(2):
            optimizer.zero_grad()
            new = token_logprobs(policy, ids, mask)
            (-new[:, response].mean()).backward()
            assert any(
                p.grad is not None and torch.count_nonzero(p.grad)
                for p in policy.parameters()
            )
            optimizer.step()
            torch.testing.assert_close(old, cached, rtol=0, atol=0)
        changed = token_logprobs(policy, ids, mask)
        assert not torch.allclose(changed[:, response], old[:, response])
        assert not old.requires_grad
    finally:
        handle.remove()


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_evaluation_decoding_remains_independent(trainer, temperature):
    results = trainer.inference.generate(
        [[4, 5]], max_new_tokens=2, temperature=temperature
    )
    assert len(results) == 1 and len(results[0]) == 1
    assert 1 <= len(results[0][0].token_ids) <= 2
    assert trainer.config.temperature == 1.0


def test_primary_budgets_preserve_order_and_do_not_shorten_other_prompts(
    trainer, monkeypatch
):
    prefixes = [[4, 5], [6] * 23, [7] * 22, [8, 9]]
    calls = []

    def generate(batch, **kwargs):
        calls.append((batch, kwargs))
        return [[Completion([prefix[0]], "stop")] for prefix in batch]

    monkeypatch.setattr(trainer.inference, "generate", generate)
    results = trainer.collect_completions(prefixes, num_samples=1)
    assert [group[0].token_ids for group in results] == [[4], [6], [7], [8]]
    assert [kwargs["max_new_tokens"] for _, kwargs in calls] == [3, 1, 2]
    assert calls[0][0] == [prefixes[0], prefixes[3]]
    assert all(kwargs["temperature"] == 1.0 for _, kwargs in calls)


@pytest.mark.parametrize("prefixes", [[], [[]], [[4], []], [[4], [5] * 24], [[5] * 25]])
def test_invalid_primary_prefix_lengths_fail_before_generation(
    trainer, monkeypatch, prefixes
):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid input reached generation")

    monkeypatch.setattr(trainer.inference, "generate", unexpected)
    with pytest.raises(ValueError):
        trainer.collect_completions(prefixes, num_samples=1)


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, 4])
def test_invalid_or_excessive_generation_override(trainer, cap):
    with pytest.raises(ValueError, match="configured cap"):
        trainer.collect_completions([[4]], num_samples=1, max_new_tokens=cap)


@pytest.mark.parametrize("capacity", [0, True, 24.0])
def test_invalid_model_capacity(trainer, capacity):
    trainer.policy.config.max_position_embeddings = capacity
    with pytest.raises(ValueError, match="max_position_embeddings"):
        PPOTrainer(trainer.config, trainer.policy, trainer.inference)


def test_configured_total_must_fit_actual_model(trainer):
    with pytest.raises(ValueError, match="model context capacity"):
        PPOTrainer(
            replace(trainer.config, max_sequence_length=25),
            trainer.policy,
            trainer.inference,
        )


def test_real_mc_late_prefix_uses_last_available_token(trainer, monkeypatch):
    calls = []
    generate = trainer.inference.generate

    def record(**kwargs):
        calls.append(kwargs["max_new_tokens"])
        return generate(**kwargs)

    monkeypatch.setattr(trainer.inference, "generate", record)
    (result,) = trainer.estimate_vineppo_batch(
        [
            (
                "late",
                Episode([4] * 20, [5, 6, 7, 8], 1.0, "length"),
                (0, 2, 3, 4),
                lambda responses: [0.5] * len(responses),
            )
        ]
    )
    assert calls == [3, 2, 1]
    assert result.mc_prefix_values == (0.5, 0.5, 0.5)


def test_training_requires_shared_actor(trainer):
    with pytest.raises(AssertionError, match="share the training actor"):
        PPOTrainer(trainer.config, object(), trainer.inference)


def test_checkpoint_before_training_fails_loudly(trainer):
    with pytest.raises(AssertionError, match="completed training boundary"):
        trainer.save_checkpoint()


@pytest.fixture
def training_case(tmp_path, monkeypatch):
    """Local real tokenizer/model; only sampling is controlled, never optimization."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>"
    )
    examples = [Example("one", "Q", "1"), Example("two", "RR", "2")]
    inputs = {
        "examples": examples,
        "tokenizer": tokenizer,
        "task": "gsm8k",
        "formatter": str,
    }
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def make(
        name="run", *, random_answers=False, polynomial_schedule=False, **overrides
    ):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            policy = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=len(tokenizer),
                    n_positions=64,
                    n_embd=8,
                    n_layer=1,
                    n_head=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    resid_pdrop=0.4,
                    embd_pdrop=0.4,
                    attn_pdrop=0.4,
                    attn_implementation="eager",
                )
            ).cpu()
        settings = {
            "checkpoint_dir": tmp_path / name,
            "algorithm_config": VinePPOConfig(),
            "num_iterations": 1,
            "max_sequence_length": 64,
            "max_new_tokens": 16,
            "prompts_per_iteration": 2,
            "num_samples": 2,
            "mc_num_samples": 2,
            "update_epochs": 2,
            "minibatch_size": 2,
            "microbatch_size": 1,
            "truncation_reward": -3.0,
            "seed": 23,
        }
        settings.update(overrides)
        result = PPOTrainer(
            PPOTrainerConfig(**settings),
            policy,
            HFInference(policy, pad_token_id=tokenizer.pad_token_id),
            optimizer=torch.optim.AdamW(policy.parameters(), lr=0.003),
            reference_policy=deepcopy(policy),
            polynomial_schedule=polynomial_schedule,
        )

        def generate(prefixes, *, num_samples, max_new_tokens, temperature):
            assert temperature == 1.0 and num_samples == 2
            assert all(len(p) + max_new_tokens <= 64 for p in prefixes)
            # Consume global RNG and, for resume tests, make it affect the data.
            draw = torch.randint(1, 3, ()).item()
            answer = str(draw) if random_answers else "1"
            primary = len(prefixes) == 2
            text = "a\nb\n#### " + answer if primary else "\n#### " + answer
            stopped = tokenizer.encode(text) + [tokenizer.eos_token_id]
            truncated = tokenizer.encode("a\nb\n#### 2".ljust(max_new_tokens))
            assert len(stopped) <= max_new_tokens == len(truncated)
            return [
                [
                    Completion(stopped.copy(), "stop"),
                    Completion(truncated.copy(), "length"),
                ]
                for _ in prefixes
            ]

        monkeypatch.setattr(result.inference, "generate", generate)
        return result

    with torch.random.fork_rng(devices=[]):
        try:
            yield make, inputs
        finally:
            torch.set_num_threads(threads)


def assert_exact(actual, expected):
    """Recursively compare optimizer/model/RNG state, without numeric tolerances."""
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_exact(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_exact(left, right)
    else:
        assert actual == expected


def test_real_loop_orders_phases_binds_references_and_reuses_old_cache(
    training_case,
    monkeypatch,
):
    make, inputs = training_case
    run = make()
    initial = deepcopy(run.policy.state_dict())
    reference = deepcopy(run.reference_policy.state_dict())
    events, estimates, prefixes, verified, batches = [], [], [], [], []
    old_rows = Counter()
    real_generate = run.inference.generate
    real_verify = train_module.verify_response
    real_cache = run.cache_old_logprobs
    real_update = run.update_policy
    real_loss = actor_module.actor_loss
    real_step = run.optimizer.step

    def unchanged():
        assert_exact(run.policy.state_dict(), initial)
        assert not run.optimizer.state

    def generate(*args, **kwargs):
        unchanged()
        assert "cache" not in events and "step" not in events
        events.append("generate")
        return real_generate(*args, **kwargs)

    def verify(text, reference_answer, **kwargs):
        verified.append((text, reference_answer, kwargs))
        return real_verify(text, reference_answer, **kwargs)

    def record_prefix(value):
        unchanged()
        prefixes.append(value)
        events.append("mc")

    def record_estimate(value):
        unchanged()
        estimates.append(value)
        events.append("estimate")

    def cache(*args):
        unchanged()
        assert len(estimates) == 4 and len(prefixes) == 8
        events.append("cache")
        value = real_cache(*args)
        assert not value.requires_grad
        return value

    def update(batch):
        assert events.count("cache") == 4
        batches.append(deepcopy(batch))
        result = real_update(batch)
        assert_exact(batch.old_logprobs, batches[0].old_logprobs)
        assert_exact(batch.ref_logprobs, batches[0].ref_logprobs)
        assert_exact(batch.advantages, batches[0].advantages)
        return result

    def loss(current, old, ref, advantages, mask, **kwargs):
        assert current.requires_grad and not old.requires_grad and not ref.requires_grad
        batch = batches[0]
        matches = [
            i for i, row in enumerate(batch.old_logprobs) if torch.equal(row, old[0])
        ]
        assert len(matches) == 1
        index = matches[0]
        old_rows[index] += 1
        assert_exact(ref[0], batch.ref_logprobs[index])
        return real_loss(current, old, ref, advantages, mask, **kwargs)

    def step(*args, **kwargs):
        assert events.count("cache") == 4 and len(estimates) == 4
        assert any(
            p.grad is not None and p.grad.count_nonzero()
            for p in run.policy.parameters()
        )
        events.append("step")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(run.inference, "generate", generate)
    monkeypatch.setattr(train_module, "verify_response", verify)
    monkeypatch.setattr(run, "cache_old_logprobs", cache)
    monkeypatch.setattr(run, "update_policy", update)
    monkeypatch.setattr(actor_module, "actor_loss", loss)
    monkeypatch.setattr(run.optimizer, "step", step)
    recorded_metrics = []
    history = run.train(
        **inputs,
        record_prefix_samples=record_prefix,
        record_value_estimate=record_estimate,
        record_metrics=recorded_metrics.append,
    )
    assert events[0] == "generate"
    # One primary call, then one per distinct MC prefix: each prompt's two
    # responses share its bare prompt and their first step.
    assert events.count("generate") == 5
    assert events[-8:] == ["cache"] * 4 + ["step"] * 4
    assert old_rows == Counter({i: 2 for i in range(4)})
    assert {e.episode_id for e in estimates} == {
        f"0:{name}:{sample}" for name in ("one", "two") for sample in range(2)
    }
    for estimate in estimates:
        _, name, sample = estimate.episode_id.split(":")
        correct = float(name == "one")
        reward = correct if sample == "0" else -3.0
        value = (correct - 3.0) / 2
        assert estimate.episode.reward == reward
        assert estimate.step_boundaries == (0, 1, len(estimate.episode.response_ids))
        assert estimate.mc_prefix_values == (value, value)
        assert estimate.raw_step_advantages == (0.0, reward - value)
        records = [p for p in prefixes if p.episode_id == estimate.episode_id]
        assert [p.response_prefix_length for p in records] == [0, 1]
        assert all(
            tuple(c.reward for c in p.continuations) == (correct, -3.0) for p in records
        )
    # Only stopped responses reach verification; MC scoring includes the sampled prefix.
    assert len(verified) == 10
    assert Counter(reference for _, reference, _ in verified) == {"1": 5, "2": 5}
    assert any(text.startswith("a\n####") for text, _, _ in verified)
    for _, answer, kwargs in verified:
        assert kwargs == {
            "task": "gsm8k",
            "question": "Q" if answer == "1" else "RR",
            "reference_is_solution": False,
        }
    assert history == recorded_metrics
    assert history[0]["optimizer_steps"] == 4
    assert history[0]["episodes"] == 4
    assert history[0]["reward_mean"] == -1.25
    assert history[0]["truncation_fraction"] == 0.5
    assert run.iteration == 1
    assert any(
        not torch.equal(tensor, initial[name])
        for name, tensor in run.policy.state_dict().items()
    )
    assert_exact(run.reference_policy.state_dict(), reference)
    assert all(
        not p.requires_grad and p.grad is None
        for p in run.reference_policy.parameters()
    )
    assert not run.reference_policy.training
    assert (run.config.checkpoint_dir / "iteration-1.pt").is_file()


def update_batch(run, lengths):
    episodes = [
        EpisodeWithAdvantages(
            [4], [5] * length, 1.0, "stop", [float(i + j) for j in range(length)]
        )
        for i, length in enumerate(lengths)
    ]
    old, ref = run._cache_scores(episodes)
    return collate(
        episodes,
        old_logprobs=old,
        ref_logprobs=ref,
        pad_token_id=run.inference.pad_token_id,
    )


@pytest.mark.parametrize("size,minibatch_size", [(3, 2), (5, 2), (2, 1), (1, 4)])
def test_singleton_plan_rejected_before_any_optimizer_changes(
    training_case,
    monkeypatch,
    size,
    minibatch_size,
):
    make, inputs = training_case
    run = make()
    run.train(**inputs)
    run.config = replace(run.config, minibatch_size=minibatch_size)
    run._generator.manual_seed(23)
    order = torch.randperm(size, generator=torch.Generator().manual_seed(23)).tolist()
    lengths = [2] * size
    lengths[order[-1]] = 1
    batch = update_batch(run, lengths)
    # Without preflight, all preceding minibatches can step before the final one fails.
    assert batch.response_mask[order[-1]].sum() == 1
    assert all(batch.response_mask[i].sum() == 2 for i in order[:-1])
    for parameter in run.policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    policy = deepcopy(run.policy.state_dict())
    optimizer = deepcopy(run.optimizer.state_dict())
    gradients = [p.grad.clone() for p in run.policy.parameters()]
    rng = run._generator.get_state().clone()

    def unexpected(*args, **kwargs):
        pytest.fail("invalid minibatch plan reached optimizer activity")

    monkeypatch.setattr(run.optimizer, "step", unexpected)
    monkeypatch.setattr(run.optimizer, "zero_grad", unexpected)
    with pytest.raises(
        AssertionError,
        match="Singleton minibatches require at least two response tokens",
    ):
        run.update_policy(batch)
    assert_exact(run.policy.state_dict(), policy)
    assert_exact(run.optimizer.state_dict(), optimizer)
    assert_exact([p.grad for p in run.policy.parameters()], gradients)
    assert_exact(run._generator.get_state(), rng)
    assert run.iteration == 1 and run._at_boundary


@pytest.mark.parametrize(
    "lengths,minibatch_size",
    [
        ([1, 1, 1, 1], 2),
        ([1, 2, 1], 3),
        ([2, 2, 2], 2),
        ([2], 4),
    ],
)
def test_valid_minibatch_plans_still_perform_real_updates(
    training_case, lengths, minibatch_size
):
    make, _ = training_case
    run = make(minibatch_size=minibatch_size)
    batch = update_batch(run, lengths)
    initial = deepcopy(run.policy.state_dict())
    history = run.update_policy(batch)
    steps = 2 * ((len(lengths) + minibatch_size - 1) // minibatch_size)
    assert len(history) == steps
    assert {int(state["step"]) for state in run.optimizer.state.values()} == {steps}
    assert any(
        not torch.equal(value, initial[name])
        for name, value in run.policy.state_dict().items()
    )


def test_resume_rejects_bf16_optimizer_moments_before_casting(training_case):
    make, inputs = training_case
    original = make("original")
    original.train(**inputs)
    path = original.config.checkpoint_dir / "iteration-1.pt"
    saved = torch.load(path, weights_only=True)
    slot = next(iter(saved["optimizer"]["state"].values()))
    slot["exp_avg"] = slot["exp_avg"].bfloat16()
    torch.save(saved, path)
    resumed = make("resumed")
    with pytest.raises(AssertionError, match="Optimizer state exp_avg requires FP32"):
        resumed.load_checkpoint(path, **inputs)
    assert resumed.iteration == 0 and not resumed.optimizer.state


@pytest.mark.parametrize("target", ["policy", "reference_policy"])
@pytest.mark.parametrize("mismatch", ["config", "dtype", "class"])
def test_resume_rejects_shape_compatible_model_identity_before_restore(
    training_case,
    monkeypatch,
    target,
    mismatch,
):
    make, inputs = training_case
    original = make("original")
    original.train(**inputs)
    path = original.config.checkpoint_dir / "iteration-1.pt"
    saved = torch.load(path, weights_only=True, map_location="cpu")
    for identity, model in zip(
        saved["model_identity"],
        (original.policy, original.reference_policy),
        strict=True,
    ):
        assert identity == {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "config": model.config.to_dict(),
            "dtypes": {
                name: str(value.dtype) for name, value in model.state_dict().items()
            },
        }
    other = make("other", num_iterations=2)
    model = getattr(other, target)
    if mismatch == "config":
        model.config.layer_norm_epsilon *= 2
    elif mismatch == "dtype":
        model.double()
    else:

        class AlternateGPT2(GPT2LMHeadModel):
            pass

        model = AlternateGPT2(deepcopy(model.config))
        setattr(other, target, model)
        if target == "policy":
            other.inference.model = model
            other.optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    state_key = "policy" if target == "policy" else "reference"
    # Ordinary strict tensor loading accepts these changes; execution identity must not.
    compatible = deepcopy(model)
    compatible.load_state_dict(saved[state_key], strict=True)
    assert {name: value.shape for name, value in model.state_dict().items()} == {
        name: value.shape for name, value in saved[state_key].items()
    }
    before = [deepcopy(m.state_dict()) for m in (other.policy, other.reference_policy)]
    optimizer = deepcopy(other.optimizer.state_dict())
    rng = torch.get_rng_state().clone()

    def unexpected(*args, **kwargs):
        pytest.fail("model identity mismatch reached state restoration")

    for owner in (other.policy, other.reference_policy, other.optimizer):
        monkeypatch.setattr(owner, "load_state_dict", unexpected)
    message = (
        "Trainable parameter .* requires FP32"
        if (target, mismatch) == ("policy", "dtype")
        else "Model execution identity mismatch"
    )
    with pytest.raises(AssertionError, match=message):
        other.load_checkpoint(path, **inputs)
    assert_exact(other.policy.state_dict(), before[0])
    assert_exact(other.reference_policy.state_dict(), before[1])
    assert_exact(other.optimizer.state_dict(), optimizer)
    assert_exact(torch.get_rng_state(), rng)
    assert other.iteration == 0 and other._at_boundary


def test_checkpoint_resume_is_bitwise_equal_to_uninterrupted(training_case):
    make, inputs = training_case
    inputs = dict(inputs, examples=inputs["examples"] + [Example("three", "SSS", "2")])
    full = make("full", num_iterations=3, random_answers=True)
    initial = deepcopy(full.policy.state_dict())
    torch.manual_seed(1234)
    full_history = full.train(**inputs)
    final_rng = torch.get_rng_state().clone()
    next_draws = torch.rand(8)

    first = make("first", random_answers=True)
    torch.manual_seed(1234)
    first_history = first.train(**inputs)
    path = first.config.checkpoint_dir / "iteration-1.pt"
    saved = torch.load(path, weights_only=True, map_location="cpu")
    assert saved["iteration"] == 1 and saved["optimizer"]["state"]
    resumed = make("resumed", num_iterations=3, checkpoint_every=2, random_answers=True)
    torch.manual_seed(999)
    resumed.load_checkpoint(path, **inputs)
    assert resumed.iteration == 1
    assert_exact(resumed.policy.state_dict(), first.policy.state_dict())
    assert_exact(resumed.optimizer.state_dict(), first.optimizer.state_dict())
    assert_exact(torch.get_rng_state(), saved["cpu_rng"])
    assert_exact(resumed._generator.get_state(), saved["generator"])
    resumed_history = resumed.train(**inputs)
    assert first_history + resumed_history == full_history
    assert_exact(resumed.policy.state_dict(), full.policy.state_dict())
    assert_exact(
        resumed.reference_policy.state_dict(), full.reference_policy.state_dict()
    )
    assert_exact(resumed.optimizer.state_dict(), full.optimizer.state_dict())
    assert_exact(resumed._generator.get_state(), full._generator.get_state())
    assert_exact(torch.get_rng_state(), final_rng)
    assert_exact(torch.rand(8), next_draws)
    assert any(
        not torch.equal(tensor, initial[name])
        for name, tensor in full.policy.state_dict().items()
    )
    assert {int(state["step"]) for state in resumed.optimizer.state.values()} == {12}
    assert sorted(p.name for p in resumed.config.checkpoint_dir.iterdir()) == [
        "iteration-2.pt",
        "iteration-3.pt",
    ]
    assert resumed.train(**inputs) == []


@pytest.mark.parametrize("cuts", [(1,), (2,), (3,), (1, 2, 3)])
@pytest.mark.parametrize("algorithm", [VinePPOConfig, GRPOConfig, RLOOConfig])
def test_polynomial_resume_at_boundaries(training_case, algorithm, cuts):
    make, inputs = training_case
    settings = {
        "num_iterations": 4,
        "checkpoint_every": 4,
        "random_answers": True,
        "algorithm_config": algorithm(),
        "polynomial_schedule": True,
    }
    full = make("full-schedule", **settings)
    torch.manual_seed(1234)
    expected = full.train(**inputs)
    final = torch.load(full.config.checkpoint_dir / "iteration-4.pt", weights_only=True)
    trainer = make("split-0", **settings)
    torch.manual_seed(1234)
    actual = []
    for index, cut in enumerate(cuts):
        actual.extend(
            trainer.train(
                **inputs,
                stop_at_boundary=lambda trainer=trainer, cut=cut: (
                    trainer.iteration == cut
                ),
            )
        )
        assert not list(trainer.config.checkpoint_dir.glob("iteration-*.pt"))
        path = trainer.save_checkpoint(
            trainer.config.checkpoint_dir / f"resume-{cut}.pt"
        )
        trainer = make(f"split-{index + 1}", **settings)
        trainer.load_checkpoint(path, **inputs)
    actual.extend(trainer.train(**inputs))
    checkpoint = torch.load(
        trainer.config.checkpoint_dir / "iteration-4.pt", weights_only=True
    )
    assert actual == expected
    for field in (
        "policy",
        "reference",
        "optimizer",
        "scheduler",
        "generator",
        "cpu_rng",
    ):
        assert_exact(checkpoint[field], final[field])


@pytest.mark.parametrize(
    "mismatch", ["order", "answer", "formatter", "tokenizer", "config", "optimizer"]
)
def test_resume_rejects_mismatched_inputs(training_case, mismatch):
    make, inputs = training_case
    run = make("original")
    run.train(**inputs)
    other = make("other", num_iterations=2)
    changed = dict(inputs)
    if mismatch == "order":
        changed["examples"] = list(reversed(inputs["examples"]))
    elif mismatch == "answer":
        changed["examples"] = [
            replace(inputs["examples"][0], reference_answer="9"),
            inputs["examples"][1],
        ]
    elif mismatch == "formatter":
        changed["formatter"] = lambda text: text + "?"
    elif mismatch == "tokenizer":
        changed["tokenizer"] = deepcopy(inputs["tokenizer"])
        changed["tokenizer"].add_tokens(["unused-added-token"])
    elif mismatch == "config":
        other.config = replace(other.config, clip_eps=0.1)
    else:
        other.optimizer = torch.optim.SGD(other.policy.parameters(), lr=0.003)
    message = {
        "config": "training config mismatch",
        "optimizer": "Optimizer identity mismatch",
    }.get(mismatch, "Dataset/prompt identity mismatch")
    initial = deepcopy(other.policy.state_dict())
    with pytest.raises(AssertionError, match=message):
        other.load_checkpoint(run.config.checkpoint_dir / "iteration-1.pt", **changed)
    assert other.iteration == 0 and not other.optimizer.state
    assert_exact(other.policy.state_dict(), initial)


@pytest.mark.parametrize("alias", ["model", "parameter", "storage", "buffer"])
def test_training_rejects_reference_alias_before_collection(
    training_case, monkeypatch, alias
):
    make, inputs = training_case
    run = make()
    if alias == "model":
        run.reference_policy = run.policy
    elif alias == "parameter":
        run.reference_policy.transformer.wpe.weight = run.policy.transformer.wpe.weight
    elif alias == "storage":
        run.reference_policy.transformer.wpe.weight = torch.nn.Parameter(
            run.policy.transformer.wpe.weight.detach()
        )
    else:
        run.reference_policy.register_buffer(
            "actor_alias", run.policy.transformer.wpe.weight.detach()
        )
    monkeypatch.setattr(
        run.inference,
        "generate",
        lambda *a, **k: pytest.fail("aliased reference reached collection"),
    )
    with pytest.raises(AssertionError, match="Reference must not share actor"):
        run.train(**inputs)
    assert not run.optimizer.state and run.iteration == 0
    assert all(p.requires_grad for p in run.policy.parameters())


@pytest.mark.parametrize("failure", ["primary", "mc", "segmentation"])
def test_failed_collection_never_steps_or_checkpoints(
    training_case, monkeypatch, failure
):
    make, inputs = training_case
    run = make()
    initial = deepcopy(run.policy.state_dict())
    real_generate = run.inference.generate
    calls = []

    def generate(prefixes, **kwargs):
        calls.append(len(prefixes))
        if failure == "primary" or (failure == "mc" and len(prefixes) == 1):
            return []
        groups = real_generate(prefixes, **kwargs)
        if failure == "segmentation":
            groups[0][0] = Completion([inputs["tokenizer"].pad_token_id], "stop")
        return groups

    monkeypatch.setattr(run.inference, "generate", generate)
    monkeypatch.setattr(
        run.optimizer, "step", lambda *a, **k: pytest.fail("failed collection stepped")
    )
    monkeypatch.setattr(
        run,
        "cache_old_logprobs",
        lambda *a, **k: pytest.fail("failed collection cached"),
    )
    error, message = (
        (AssertionError, "special tokens")
        if failure == "segmentation"
        else (ValueError, "(cardinality|one group)")
    )
    with pytest.raises(error, match=message):
        run.train(**inputs)
    assert calls == ([2, 1] if failure == "mc" else [2])
    assert_exact(run.policy.state_dict(), initial)
    assert run.iteration == 0 and not run.optimizer.state
    assert not run.config.checkpoint_dir.exists()
    with pytest.raises(AssertionError, match="completed training boundary"):
        run.save_checkpoint()
    with pytest.raises(AssertionError, match="incomplete iteration"):
        run.train(**inputs)

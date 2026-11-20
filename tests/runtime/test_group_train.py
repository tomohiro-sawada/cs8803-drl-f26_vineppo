"""Grouped training uses real local actor updates, without value estimation."""

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

import algorithms.actor as actor_module
import runtime.train as train_module
from algorithms.grpo import GRPOConfig
from algorithms.rloo import RLOOConfig
from core.inference import Completion
from tests.runtime import test_train
from tests.runtime.test_train import assert_exact

training_case = test_train.training_case


def forbidden(*args, **kwargs):
    pytest.fail("Grouped training entered a forbidden path")


@pytest.mark.parametrize("algorithm", [GRPOConfig(eps=0.01), RLOOConfig()])
def test_group_advantages_order_lengths_and_fixed_updates(
    training_case, monkeypatch, algorithm
):
    make, inputs = training_case
    run = make(algorithm_config=algorithm, num_samples=3, minibatch_size=2)
    tokenizer = inputs["tokenizer"]
    collected = []

    def generate(prefixes, *, num_samples, max_new_tokens, temperature):
        assert num_samples == 3 and temperature == 1
        collected.extend(prefixes)
        return [
            [
                Completion(tokenizer.encode("#### 1"), "stop"),
                Completion(tokenizer.encode("x\n#### 0"), "stop"),
                Completion(tokenizer.encode("x" * max_new_tokens), "length")
                if tokenizer.decode(prefix) == "Q"
                else Completion(tokenizer.encode("long\n#### 0"), "stop"),
            ]
            for prefix in prefixes
        ]

    monkeypatch.setattr(run.inference, "generate", generate)
    for name in (
        "step_boundaries_from_tokens",
        "estimate_vineppo_batch",
        "token_values",
        "update_critic",
    ):
        monkeypatch.setattr(train_module, name, forbidden)
    monkeypatch.setattr(actor_module, "normalize_advantages", forbidden)
    monkeypatch.setattr(run, "_estimate_ppo", forbidden)
    captured = []
    cache = run._cache_scores
    update = run.update_policy

    def capture_cache(episodes):
        captured.extend(episodes)
        return cache(episodes)

    def capture_update(batch):
        fixed = deepcopy(batch)
        result = update(batch)
        for name in ("advantages", "old_logprobs", "ref_logprobs", "response_mask"):
            assert_exact(getattr(batch, name), getattr(fixed, name))
        return result

    monkeypatch.setattr(run, "_cache_scores", capture_cache)
    monkeypatch.setattr(run, "update_policy", capture_update)
    before = deepcopy(run.policy.state_dict())
    reference = deepcopy(run.reference_policy.state_dict())
    history = run.train(**inputs)
    assert len(collected) == 2 and len(captured) == 6
    for group, prompt in zip((captured[:3], captured[3:]), collected, strict=True):
        is_one = tokenizer.decode(prompt) == "Q"
        rewards = [1.0, 0.0, -3.0] if is_one else [0.0, 0.0, 0.0]
        assert [e.reward for e in group] == rewards
        # Population variance of [1, 0, -3] is 26/9; LOO is 3/2 * centered.
        centered = torch.tensor([5 / 3, 2 / 3, -7 / 3]) if is_one else torch.zeros(3)
        expected = (
            centered / ((26 / 9) ** 0.5 + algorithm.eps)
            if isinstance(algorithm, GRPOConfig)
            else centered * 1.5
        )
        assert len({len(e.response_ids) for e in group}) == 3
        for episode, advantage in zip(group, expected, strict=True):
            assert episode.prompt_ids == prompt
            torch.testing.assert_close(
                torch.tensor(episode.advantages),
                advantage.expand(len(episode.response_ids)),
            )
    assert history[0]["episodes"] == 6 and history[0]["optimizer_steps"] == 6
    assert run.critic is None and run.critic_optimizer is None
    assert any(
        not torch.equal(value, before[name])
        for name, value in run.policy.state_dict().items()
    )
    assert_exact(run.reference_policy.state_dict(), reference)


@pytest.mark.parametrize("algorithm", [GRPOConfig(), RLOOConfig()])
def test_group_single_token_singleton_updates(training_case, monkeypatch, algorithm):
    make, inputs = training_case
    run = make(algorithm_config=algorithm, minibatch_size=1)
    tokenizer = inputs["tokenizer"]
    monkeypatch.setattr(
        run.inference,
        "generate",
        lambda prefixes, **kwargs: [
            [
                Completion([tokenizer.eos_token_id], "stop"),
                Completion(tokenizer.encode("#### 1"), "stop"),
            ]
            for _ in prefixes
        ],
    )
    monkeypatch.setattr(actor_module, "normalize_advantages", forbidden)
    assert run.train(**inputs)[0]["optimizer_steps"] == 8


@pytest.mark.parametrize("algorithm", [GRPOConfig(), RLOOConfig()])
def test_g1_rejected_before_collection(training_case, monkeypatch, algorithm):
    make, inputs = training_case
    with pytest.raises(ValueError, match="num_samples >= 2"):
        make(algorithm_config=algorithm, num_samples=1)
    run = make(algorithm_config=algorithm)
    # Also guard against a configuration changed after construction.
    object.__setattr__(run.config, "num_samples", 1)
    monkeypatch.setattr(run.inference, "generate", forbidden)
    with pytest.raises(AssertionError, match="num_samples >= 2"):
        run.train(**inputs)


@pytest.mark.parametrize("algorithm", [GRPOConfig(), RLOOConfig()])
@pytest.mark.parametrize(
    "invalid",
    ["record_value_estimate", "record_prefix_samples", "critic", "critic_optimizer"],
)
def test_group_rejects_value_interfaces(training_case, monkeypatch, algorithm, invalid):
    make, inputs = training_case
    run = make(algorithm_config=algorithm)
    monkeypatch.setattr(run.inference, "generate", forbidden)
    if invalid.startswith("record_"):
        inputs = dict(inputs, **{invalid: forbidden})
    else:
        setattr(run, invalid, object())
    with pytest.raises(AssertionError, match="callbacks|critic"):
        run.train(**inputs)


@pytest.mark.parametrize("algorithm", [GRPOConfig(), RLOOConfig()])
def test_group_resume_exact(training_case, algorithm):
    make, inputs = training_case
    full = make(
        "full", algorithm_config=algorithm, num_iterations=2, random_answers=True
    )
    torch.manual_seed(123)
    expected = full.train(**inputs)
    rng = torch.get_rng_state()
    first = make("first", algorithm_config=algorithm, random_answers=True)
    torch.manual_seed(123)
    history = first.train(**inputs)
    resumed = make(
        "resumed", algorithm_config=algorithm, num_iterations=2, random_answers=True
    )
    resumed.load_checkpoint(first.config.checkpoint_dir / "iteration-1.pt", **inputs)
    assert history + resumed.train(**inputs) == expected
    for name in ("policy", "reference_policy", "optimizer"):
        assert_exact(
            getattr(resumed, name).state_dict(), getattr(full, name).state_dict()
        )
    assert_exact(torch.get_rng_state(), rng)
    assert_exact(resumed._generator.get_state(), full._generator.get_state())


def test_algorithm_identity_rejected_even_with_identical_fields(
    training_case, monkeypatch
):
    make, inputs = training_case
    first = make(algorithm_config=GRPOConfig())
    first.train(**inputs)
    path = first.config.checkpoint_dir / "iteration-1.pt"
    state = torch.load(path, weights_only=True)
    resumed = make("resumed", algorithm_config=RLOOConfig())
    # Neutralize field differences to test the type discriminator independently.
    state["config"]["algorithm_config"] = asdict(resumed.config.algorithm_config)
    torch.save(state, path)
    monkeypatch.setattr(resumed.policy, "load_state_dict", forbidden)
    with pytest.raises(AssertionError, match="training config mismatch"):
        resumed.load_checkpoint(path, **inputs)

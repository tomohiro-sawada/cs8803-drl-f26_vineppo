"""Downloads-free PPO integration tests; generation alone is controlled."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model, PreTrainedTokenizerFast

import algorithms.ppo as ppo_module
import runtime.train as train_module
from algorithms.ppo import PPOConfig, TokenCritic
from core.data import Episode, EpisodeWithAdvantages, Example, PPOValueEstimate
from core.inference import Completion, HFInference
from core.lora import apply_lora
from runtime.config import PPOTrainerConfig
from runtime.train import PPOTrainer


def assert_exact(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_exact(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_exact(left, right)
    else:
        assert actual == expected


def states(run):
    return deepcopy(
        {
            name: getattr(run, name).state_dict()
            for name in (
                "policy",
                "reference_policy",
                "critic",
                "optimizer",
                "critic_optimizer",
            )
            if getattr(run, name) is not None
        }
    )


def forbidden(*args, **kwargs):
    pytest.fail("PPO must not enter MC/segmentation or update invalid training state")


@pytest.fixture
def ppo_case(tmp_path, monkeypatch):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>"
    )
    inputs = {
        "examples": [Example("one", "Q", "1"), Example("two", "RR", "2")],
        "tokenizer": tokenizer,
        "task": "gsm8k",
        "formatter": str,
    }

    def make(name="run", *, polynomial_schedule=False, compact_lora=False, **overrides):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            config = GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=32,
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
            actor = GPT2LMHeadModel(deepcopy(config)).cpu()
            critic = TokenCritic(GPT2Model(deepcopy(config))).cpu()
            reference = deepcopy(actor)
            if compact_lora:
                actor = apply_lora(actor, rank=2, alpha=4)
        settings = {
            "checkpoint_dir": tmp_path / name,
            "algorithm_config": PPOConfig(gamma=1, lam=0.95),
            "num_iterations": 1,
            "max_sequence_length": 32,
            "max_new_tokens": 12,
            "prompts_per_iteration": 2,
            "num_samples": 2,
            "update_epochs": 2,
            "minibatch_size": 2,
            "microbatch_size": 1,
            "truncation_reward": -3.0,
            "seed": 23,
        }
        settings.update(overrides)
        run = PPOTrainer(
            PPOTrainerConfig(**settings),
            actor,
            HFInference(actor, pad_token_id=tokenizer.pad_token_id),
            optimizer=torch.optim.AdamW(
                [p for p in actor.parameters() if p.requires_grad], lr=0.003
            ),
            reference_policy=reference,
            critic=critic,
            critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=0.003),
            polynomial_schedule=polynomial_schedule,
            compact_lora_checkpoints=compact_lora,
        )

        def generate(prefixes, *, num_samples, max_new_tokens, temperature):
            assert num_samples == 2 and temperature == 1 and max_new_tokens == 12
            answer = str(torch.randint(1, 3, ()).item())
            return [
                [
                    Completion(
                        tokenizer.encode("a" * len(p) + "\n#### " + answer)
                        + [tokenizer.eos_token_id],
                        "stop",
                    ),
                    Completion(tokenizer.encode("x" * max_new_tokens), "length"),
                ]
                for p in prefixes
            ]

        monkeypatch.setattr(run.inference, "generate", generate)
        return run

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        try:
            yield make, inputs
        finally:
            torch.set_num_threads(threads)


@pytest.mark.parametrize("microbatch_size", [1, 3])
def test_estimates_align_states_and_terminal_gae(ppo_case, microbatch_size):
    make, _ = ppo_case
    run = make(microbatch_size=microbatch_size)
    episodes = [
        Episode([4], [5, 6, 7], 2.0, "stop"),
        Episode([8, 9, 10], [11], -3.0, "length"),
        Episode([12, 13], [14, 15], 1.0, "length"),
    ]
    before = states(run)
    prepared, estimates = run._estimate_ppo(episodes, ["a", "b", "c"])
    assert [e.episode_id for e in estimates] == ["a", "b", "c"]
    for episode, ready, estimate in zip(episodes, prepared, estimates, strict=True):
        assert isinstance(ready, EpisodeWithAdvantages)
        assert isinstance(estimate, PPOValueEstimate)
        assert estimate.episode == episode
        assert {k: v for k, v in asdict(ready).items() if k != "advantages"} == asdict(
            episode
        )
        # Independently read each unpadded state BEFORE its action, not token_values/GAE.
        ids = torch.tensor([episode.prompt_ids + episode.response_ids])
        with torch.no_grad():
            values = run.critic(
                ids, torch.ones_like(ids), torch.arange(ids.shape[1])[None]
            )
        start = len(episode.prompt_ids) - 1
        expected = values[0, start:-1].tolist()
        assert estimate.critic_value_estimates == pytest.approx(expected, abs=1e-6)
        advantages = [0.0] * len(expected)
        following = carry = 0.0
        for i in reversed(range(len(expected))):
            reward = episode.reward if i == len(expected) - 1 else 0.0
            carry = reward + following - expected[i] + 0.95 * carry
            advantages[i] = carry
            following = expected[i]
        assert ready.advantages == pytest.approx(advantages, abs=1e-6)
        assert estimate.return_estimates == pytest.approx(
            [v + a for v, a in zip(expected, advantages, strict=True)], abs=1e-6
        )
        assert estimate.return_estimates[-1] == pytest.approx(episode.reward, abs=1e-6)
    assert_exact(states(run), before)
    assert all(p.grad is None for p in run.critic.parameters())


def test_real_updates_reuse_fixed_targets_after_all_actor_epochs(ppo_case, monkeypatch):
    make, inputs = ppo_case
    run = make()
    initial = states(run)
    events, estimates, cache_calls, target_rows = [], [], [], Counter()
    batches = {}
    for name in ("step_boundaries_from_tokens", "estimate_vineppo_batch"):
        monkeypatch.setattr(train_module, name, forbidden)
    monkeypatch.setattr(run, "estimate_vineppo_batch", forbidden)
    real_values = train_module.token_values

    def cache(*args):
        assert not torch.is_grad_enabled() and not events
        assert_exact(states(run), initial)
        result = real_values(*args)
        assert not result.requires_grad
        cache_calls.append(result.clone())
        return result

    monkeypatch.setattr(train_module, "token_values", cache)

    def record(estimate):
        assert not events
        estimates.append(estimate)

    for name, label in (("update_policy", "actor"), ("update_value", "critic")):
        real_update = getattr(run, name)

        def update(batch, real_update=real_update, label=label):
            assert len(cache_calls) == len(estimates) == 4
            saved = deepcopy(asdict(batch))
            batches[label] = batch
            if label == "critic":
                assert events == ["actor"] * 4
                for row, estimate in enumerate(estimates):
                    assert (
                        batch.returns[row, batch.response_mask[row]].tolist()
                        == estimate.return_estimates
                    )
            result = real_update(batch)
            assert_exact(asdict(batch), saved)
            return result

        monkeypatch.setattr(run, name, update)

    real_actor = train_module.update_actor

    def actor_update(*args, **kwargs):
        assert kwargs["kl_coef"] == run.config.kl_coef > 0
        batch = batches["actor"]
        for row, estimate in enumerate(estimates):
            expected = [
                r - v
                for r, v in zip(
                    estimate.return_estimates,
                    estimate.critic_value_estimates,
                    strict=True,
                )
            ]
            assert batch.advantages[
                row, batch.response_mask[row]
            ].tolist() == pytest.approx(expected, abs=1e-6)
        return real_actor(*args, **kwargs)

    monkeypatch.setattr(train_module, "update_actor", actor_update)
    frozen_estimates = []
    for optimizer, model, label in (
        (run.optimizer, run.policy, "actor"),
        (run.critic_optimizer, run.critic, "critic"),
    ):
        real_step = optimizer.step

        def step(*args, real_step=real_step, model=model, label=label, **kwargs):
            if not frozen_estimates:
                frozen_estimates.extend(deepcopy(estimates))
            assert estimates == frozen_estimates
            assert len(cache_calls) == 4
            assert any(
                p.grad is not None and p.grad.count_nonzero()
                for p in model.parameters()
            )
            if label == "actor":
                assert "critic" not in events
                assert_exact(run.critic.state_dict(), initial["critic"])
            events.append(label)
            return real_step(*args, **kwargs)

        monkeypatch.setattr(optimizer, "step", step)

    real_values_update, real_loss = ppo_module.token_values, ppo_module.value_loss
    current_rows = []

    def values_update(critic, ids, mask):
        batch = batches["critic"]
        current_rows[:] = [
            next(i for i, row in enumerate(batch.input_ids) if torch.equal(row, ids[0]))
        ]
        return real_values_update(critic, ids, mask)

    def loss(values, returns, mask, **kwargs):
        index = current_rows[0]
        batch = batches["critic"]
        assert values.requires_grad and not returns.requires_grad
        assert_exact(returns[0], batch.returns[index])
        assert_exact(mask[0], batch.response_mask[index])
        target_rows[index] += 1
        return real_loss(values, returns, mask, **kwargs)

    monkeypatch.setattr(ppo_module, "token_values", values_update)
    monkeypatch.setattr(ppo_module, "value_loss", loss)
    metrics = []
    history = run.train(
        **inputs,
        record_prefix_samples=forbidden,
        record_value_estimate=record,
        record_metrics=metrics.append,
    )
    assert events == ["actor"] * 4 + ["critic"] * 4
    assert target_rows == Counter({i: 2 for i in range(4)})
    assert estimates == frozen_estimates
    assert history == metrics and run.iteration == 1
    assert history[0]["optimizer_steps"] == history[0]["critic/optimizer_steps"] == 4
    assert history[0]["episodes"] == 4 and history[0]["truncation_fraction"] == 0.5
    assert history[0]["reward_mean"] == -1.25
    assert {e.episode_id for e in estimates} == {
        f"0:{name}:{i}" for name in ("one", "two") for i in range(2)
    }
    for name in ("policy", "critic"):
        assert any(
            not torch.equal(v, initial[name][k])
            for k, v in getattr(run, name).state_dict().items()
        )
    assert_exact(run.reference_policy.state_dict(), initial["reference_policy"])
    assert not run.reference_policy.training
    assert all(
        not p.requires_grad and p.grad is None
        for p in run.reference_policy.parameters()
    )


@pytest.mark.parametrize("cuts", [(1,), (2,), (1, 2)])
@pytest.mark.parametrize("polynomial", [False, True])
def test_resume_is_bitwise_exact_for_models_optimizers_rng_and_metrics(
    ppo_case, cuts, polynomial
):
    make, inputs = ppo_case
    settings = {
        "num_iterations": 3,
        "checkpoint_every": 3,
        "polynomial_schedule": polynomial,
        "algorithm_config": PPOConfig(lam=1.0, value_clip_eps=0.2),
    }
    full, split = make("full", **settings), make("split", **settings)
    torch.manual_seed(101)
    expected_metrics = full.train(**inputs)
    expected = torch.load(
        full.config.checkpoint_dir / "iteration-3.pt", weights_only=True
    )
    torch.manual_seed(101)
    metrics = []
    for cut in cuts:
        metrics.extend(
            split.train(
                **inputs,
                stop_at_boundary=lambda split=split, cut=cut: split.iteration == cut,
            )
        )
        path = split.save_checkpoint(split.config.checkpoint_dir / f"resume-{cut}.pt")
        saved = torch.load(path, weights_only=True)
        assert_exact(saved["critic"], split.critic.state_dict())
        assert_exact(saved["critic_optimizer"], split.critic_optimizer.state_dict())
        assert len(saved["model_identity"]) == 3
        resumed = make(f"resumed-{cut}", **settings)
        torch.rand(31)
        resumed.load_checkpoint(path, **inputs, audit=True)
        assert_exact(states(resumed), states(split))
        assert_exact(torch.get_rng_state(), saved["cpu_rng"])
        assert_exact(resumed._generator.get_state(), saved["generator"])
        split = resumed
    assert_exact(metrics + split.train(**inputs), expected_metrics)
    actual = torch.load(
        split.config.checkpoint_dir / "iteration-3.pt", weights_only=True
    )
    for key in expected.keys() - {"config"}:
        assert_exact(actual[key], expected[key])


def test_split_critic_round_trip(ppo_case):
    make, inputs = ppo_case
    settings = {"compact_lora": True, "num_iterations": 2, "checkpoint_every": 2}
    source = make(**settings)
    source.train(**inputs, stop_at_boundary=lambda: source.iteration == 1)
    path = source.save_checkpoint(
        source.config.checkpoint_dir / "resume-1.pt", split_critic=True
    )
    saved = torch.load(path, weights_only=True)
    assert {"critic", "critic_optimizer"}.isdisjoint(saved)
    resumed = make("resumed", **settings)
    resumed.load_checkpoint(path, **inputs, audit=True)
    assert_exact(states(resumed), states(source))
    assert resumed.critic_checkpoint == source.critic_checkpoint


@pytest.mark.parametrize(
    "invalid",
    [
        "critic",
        "critic_optimizer",
        "actor_storage",
        "reference_storage",
        "wrong_optimizer",
    ],
)
def test_invalid_critic_rejected_before_collection_or_updates(
    ppo_case, monkeypatch, invalid
):
    make, inputs = ppo_case
    run = make()
    if invalid in ("critic", "critic_optimizer"):
        setattr(run, invalid, None)
    elif invalid == "wrong_optimizer":
        run.critic_optimizer = torch.optim.AdamW(run.policy.parameters())
    else:
        other = run.policy if invalid == "actor_storage" else run.reference_policy
        # Distinct Parameter objects aliasing storage must also be rejected.
        run.critic.backbone.wte.weight.data = other.transformer.wte.weight.data
        assert run.critic.backbone.wte.weight is not other.transformer.wte.weight
    before = states(run)
    monkeypatch.setattr(run.inference, "generate", forbidden)
    monkeypatch.setattr(run, "update_policy", forbidden)
    monkeypatch.setattr(run, "update_value", forbidden)
    with pytest.raises(AssertionError, match="critic|Critic"):
        run.train(**inputs)
    assert_exact(states(run), before)
    assert run.iteration == 0


@pytest.mark.parametrize(
    "mismatch", ["model", "backbone", "config", "dtype", "optimizer", "order"]
)
def test_resume_critic_identity_rejected_before_weights_mutate(ppo_case, mismatch):
    make, inputs = ppo_case
    source = make("source")
    source.train(**inputs)
    path = source.config.checkpoint_dir / "iteration-1.pt"
    target = make("target")
    if mismatch == "model":

        class OtherCritic(TokenCritic):
            pass

        target.critic.__class__ = OtherCritic
    elif mismatch == "backbone":

        class OtherBackbone(GPT2Model):
            pass

        target.critic.backbone.__class__ = OtherBackbone
    elif mismatch == "config":
        target.critic.config.layer_norm_epsilon *= 2
    elif mismatch == "dtype":
        target.critic.double()
    elif mismatch == "optimizer":
        target.critic_optimizer = torch.optim.SGD(target.critic.parameters(), lr=0.003)
    else:
        target.critic_optimizer = torch.optim.AdamW(
            list(target.critic.parameters())[::-1], lr=0.003
        )
    before = states(target)
    rng = torch.get_rng_state().clone()
    message = (
        "Trainable parameter .* requires FP32"
        if mismatch == "dtype"
        else "identity mismatch"
    )
    with pytest.raises(AssertionError, match=message):
        target.load_checkpoint(path, **inputs)
    assert_exact(states(target), before)
    assert_exact(torch.get_rng_state(), rng)
    assert target.iteration == 0

"""Exercise the training collection boundary with a real, local tiny HF model."""

from dataclasses import replace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from algorithms.vineppo import VinePPOConfig
from core.data import Episode
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
        estimate = trainer.estimate_vineppo(
            "example-0",
            episode,
            step_boundaries=(0, len(episode.response_ids)),
            score_response=lambda response, reason: 0.25,
            num_samples=2,
            max_new_tokens=3,
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
@pytest.mark.parametrize("mc", [False, True])
def test_invalid_or_excessive_generation_override(trainer, cap, mc):
    with pytest.raises(ValueError, match="configured cap"):
        if mc:
            trainer.estimate_vineppo(
                "invalid",
                Episode([4], [5], 1.0, "stop"),
                step_boundaries=(0, 1),
                score_response=lambda ids, reason: 1.0,
                num_samples=1,
                max_new_tokens=cap,
            )
        else:
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
    result = trainer.estimate_vineppo(
        "late",
        Episode([4] * 20, [5, 6, 7, 8], 1.0, "length"),
        step_boundaries=(0, 2, 3, 4),
        score_response=lambda ids, reason: 0.5,
        num_samples=1,
    )
    assert calls == [3, 2, 1]
    assert result.mc_prefix_values == (0.5, 0.5, 0.5)


def test_training_requires_shared_actor(trainer):
    with pytest.raises(AssertionError, match="share the training actor"):
        PPOTrainer(trainer.config, object(), trainer.inference)


@pytest.mark.parametrize(
    "method,args", [("train", ()), ("update_policy", (None,)), ("save_checkpoint", ())]
)
def test_unimplemented_training_phases_fail_loudly(trainer, method, args):
    with pytest.raises(NotImplementedError):
        getattr(trainer, method)(*args)

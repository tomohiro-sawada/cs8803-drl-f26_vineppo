"""Contracts for shared-model HF rollouts and differentiable token rescoring."""

from copy import deepcopy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.generation.utils import GenerateDecoderOnlyOutput

from core.inference import Completion, HFInference, token_logprobs


@pytest.fixture
def model():
    # Local random weights only; nonzero dropout makes missing eval() observable.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        config = GPT2Config(
            vocab_size=23,
            n_positions=24,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            resid_pdrop=0.5,
            embd_pdrop=0.5,
            attn_pdrop=0.5,
            attn_implementation="sdpa",
        )
        yield GPT2LMHeadModel(config)


def scripted_generate(monkeypatch, model, continuations):
    """Replace decoding, retaining HF's real transition-score computation."""
    calls = []

    def generate(input_ids, **kwargs):
        assert not torch.is_grad_enabled()
        assert not any(module.training for module in model.modules())
        config = deepcopy(kwargs.get("generation_config", model.generation_config))
        config.update(**{k: v for k, v in kwargs.items() if k != "generation_config"})
        prompts = input_ids.repeat_interleave(config.num_return_sequences, dim=0)
        responses = torch.tensor(continuations, device=input_ids.device)
        assert len(prompts) == len(responses)
        assert responses.shape[1] <= config.max_new_tokens
        scores = tuple(
            torch.sin(
                torch.arange(
                    len(prompts) * model.config.vocab_size,
                    device=input_ids.device,
                    dtype=torch.float32,
                ).reshape(len(prompts), -1)
                + step
            )
            for step in range(responses.shape[1])
        )
        sequences = torch.cat((prompts, responses), dim=1)
        calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": kwargs["attention_mask"].clone(),
                "config": config,
                "scores": scores,
            }
        )
        if config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=sequences, scores=scores)
        return sequences

    monkeypatch.setattr(model, "generate", generate)
    return calls


def test_completion_logprobs_are_optional():
    completion = Completion(token_ids=[4, 2], finish_reason="stop")
    assert completion.token_ids == [4, 2]
    assert completion.finish_reason == "stop"
    assert completion.logprobs is None
    scored = Completion([5, 6], "length", [-1.25, -2.5])
    assert scored.logprobs == [-1.25, -2.5]


def test_generate_preserves_prefixes_batch_sample_order_and_inputs(model, monkeypatch):
    prefixes = ([2, 3, 0, 4], [5, 6])
    original = deepcopy(prefixes)
    continuations = [[7, 8], [9, 10], [11, 12], [13, 14]]
    calls = scripted_generate(monkeypatch, model, continuations)
    results = HFInference(model, pad_token_id=0).generate(
        prefixes, num_samples=2, max_new_tokens=2
    )
    assert [[c.token_ids for c in group] for group in results] == [
        continuations[:2],
        continuations[2:],
    ]
    assert all(isinstance(c, Completion) for group in results for c in group)
    assert all(c.finish_reason == "length" for group in results for c in group)
    assert all(c.logprobs is None for group in results for c in group)
    assert prefixes == original
    assert len(calls) == 1
    call = calls[0]
    config = call["config"]
    assert config.do_sample is True
    assert (config.temperature, config.top_k, config.top_p) == (1.0, 0, 1.0)
    repeats = config.num_return_sequences
    torch.testing.assert_close(
        call["input_ids"].repeat_interleave(repeats, 0),
        torch.tensor([[2, 3, 0, 4]] * 2 + [[0, 0, 5, 6]] * 2),
    )
    torch.testing.assert_close(
        call["attention_mask"].repeat_interleave(repeats, 0),
        torch.tensor([[1, 1, 1, 1]] * 2 + [[0, 0, 1, 1]] * 2),
    )


@pytest.mark.parametrize(
    "eos,pad,continuations,expected,reasons",
    [
        (
            2,
            0,
            [[7, 2, 0, 0], [8, 9, 10, 2]],
            [[7, 2], [8, 9, 10, 2]],
            ["stop", "stop"],
        ),
        ([2, 3], 0, [[7, 3, 0, 0], [8, 2, 0, 0]], [[7, 3], [8, 2]], ["stop", "stop"]),
        (2, 2, [[2, 2, 2, 2], [7, 8, 9, 2]], [[2], [7, 8, 9, 2]], ["stop", "stop"]),
        (
            2,
            0,
            [[0, 7, 0, 0], [8, 0, 2, 0]],
            [[0, 7, 0, 0], [8, 0, 2]],
            ["length", "stop"],
        ),
    ],
    ids=["eos-at-budget", "multiple-eos", "pad-is-eos", "generated-pad-is-valid"],
)
@pytest.mark.parametrize("return_logprobs", [False, True])
def test_generate_termination_and_score_alignment(
    model, monkeypatch, eos, pad, continuations, expected, reasons, return_logprobs
):
    model.generation_config.eos_token_id = eos
    calls = scripted_generate(monkeypatch, model, continuations)
    results = HFInference(model, pad_token_id=pad).generate(
        [[4, 5]], num_samples=2, max_new_tokens=4, return_logprobs=return_logprobs
    )[0]
    assert [c.token_ids for c in results] == expected
    assert [c.finish_reason for c in results] == reasons
    for row, completion in enumerate(results):
        if not return_logprobs:
            assert completion.logprobs is None
            continue
        expected_scores = [
            scores[row].log_softmax(-1)[token].item()
            for scores, token in zip(calls[0]["scores"], completion.token_ids)
        ]
        assert isinstance(completion.logprobs, list)
        assert all(isinstance(score, float) for score in completion.logprobs)
        assert completion.logprobs == pytest.approx(expected_scores, abs=1e-6)


def test_generate_accepts_globally_early_eos(model, monkeypatch):
    scripted_generate(monkeypatch, model, [[2]])
    completion = HFInference(model, pad_token_id=0).generate(
        [[4, 5]], max_new_tokens=6
    )[0][0]
    assert completion.token_ids == [2]
    assert completion.finish_reason == "stop"


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
def test_generate_overrides_checkpoint_sampling_settings(
    model, monkeypatch, temperature
):
    config = model.generation_config
    config.do_sample = False
    config.temperature = 0.2
    config.top_k = 1
    config.top_p = 0.1
    config.num_beams = 3
    config.num_return_sequences = 3
    config.repetition_penalty = 2.0
    config.encoder_repetition_penalty = 2.0
    config.no_repeat_ngram_size = 2
    config.encoder_no_repeat_ngram_size = 1
    config.forced_eos_token_id = 2
    before = deepcopy(config.to_dict())
    calls = scripted_generate(monkeypatch, model, [[7, 8]])
    HFInference(model, pad_token_id=0).generate(
        [[4, 5]], max_new_tokens=2, temperature=temperature
    )
    effective = calls[0]["config"]
    assert effective.do_sample is (temperature > 0)
    if temperature > 0:
        assert effective.temperature == temperature
        assert effective.top_k == 0
        assert effective.top_p == 1.0
    assert effective.num_beams == 1
    assert effective.num_return_sequences == 1
    assert effective.repetition_penalty == 1.0
    assert effective.encoder_repetition_penalty == 1.0
    assert effective.no_repeat_ngram_size == 0
    assert effective.encoder_no_repeat_ngram_size == 0
    assert effective.forced_eos_token_id is None
    assert config.to_dict() == before


def test_generate_is_no_grad_and_leaves_shared_model_in_eval(model):
    inference = HFInference(model, pad_token_id=0)
    model.train()
    seen = []

    def check_forward(module, args, kwargs):
        seen.append(torch.is_grad_enabled())
        assert not any(child.training for child in module.modules())

    handle = model.register_forward_pre_hook(check_forward, with_kwargs=True)
    try:
        inference.generate([[4, 5]], max_new_tokens=2, temperature=0)
    finally:
        handle.remove()
    assert seen and not any(seen)
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_greedy_logprobs_are_raw_actor_scores(model):
    prefix = [4, 5]
    completion = HFInference(model, pad_token_id=0).generate(
        [prefix], max_new_tokens=3, temperature=0, return_logprobs=True
    )[0][0]
    ids = torch.tensor([prefix + completion.token_ids])
    with torch.no_grad():
        expected = token_logprobs(model, ids, torch.ones_like(ids))[
            0, len(prefix) - 1 :
        ]
    torch.testing.assert_close(
        torch.tensor(completion.logprobs), expected, rtol=1e-5, atol=1e-6
    )


def test_generate_observes_shared_weight_updates_without_sync(model):
    model.lm_head = torch.nn.Linear(model.config.n_embd, model.config.vocab_size)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[7] = 10
    inference = HFInference(model, pad_token_id=0)
    first = inference.generate([[4, 5]], max_new_tokens=2, temperature=0)[0][0]
    assert first.token_ids == [7, 7]
    with torch.no_grad():
        model.lm_head.bias[8] = 20
    second = inference.generate([[4, 5]], max_new_tokens=2, temperature=0)[0][0]
    assert second.token_ids == [8, 8]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_token_logprobs_match_shifted_raw_logits_in_float32(model, dtype):
    model.to(dtype=dtype).eval()
    input_ids = torch.tensor([[4, 5, 6, 2], [7, 8, 9, 10]])
    mask = torch.ones_like(input_ids)
    logits = model(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=torch.arange(input_ids.shape[1]).expand_as(input_ids),
        use_cache=False,
    ).logits
    expected = (
        logits[:, :-1]
        .float()
        .log_softmax(-1)
        .gather(-1, input_ids[:, 1:, None])
        .squeeze(-1)
    )
    actual = token_logprobs(model, input_ids, mask)
    assert actual.shape == (2, 3)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
def test_token_logprobs_are_padding_invariant(model, side):
    sequences = [[4, 5, 6, 7], [8, 9]]
    padding = [0, 0]
    short = padding + sequences[1] if side == "left" else sequences[1] + padding
    short_mask = [0, 0, 1, 1] if side == "left" else [1, 1, 0, 0]
    input_ids = torch.tensor([sequences[0], short])
    mask = torch.tensor([[1, 1, 1, 1], short_mask])
    actual = token_logprobs(model, input_ids, mask)
    valid = mask[:, :-1].bool() & mask[:, 1:].bool()
    changed_padding = input_ids.masked_fill(mask == 0, 17)
    changed = token_logprobs(model, changed_padding, mask)
    torch.testing.assert_close(actual[valid], changed[valid], rtol=1e-5, atol=1e-6)
    for row, sequence in enumerate(sequences):
        unpadded = torch.tensor([sequence])
        expected = token_logprobs(model, unpadded, torch.ones_like(unpadded))[0]
        torch.testing.assert_close(
            actual[row][valid[row]], expected, rtol=1e-5, atol=1e-6
        )


def test_token_logprobs_use_mask_positions_no_cache_and_eval(model):
    input_ids = torch.tensor([[0, 0, 4, 5], [6, 7, 8, 9]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    calls = []

    def check_forward(module, args, kwargs):
        assert not any(child.training for child in module.modules())
        assert kwargs["use_cache"] is False
        positions = kwargs["position_ids"]
        torch.testing.assert_close(
            positions[mask.bool()], (mask.cumsum(-1) - 1)[mask.bool()]
        )
        torch.testing.assert_close(kwargs["attention_mask"], mask)
        calls.append(True)

    model.train()
    handle = model.register_forward_pre_hook(check_forward, with_kwargs=True)
    try:
        first = token_logprobs(model, input_ids, mask)
        model.train()
        second = token_logprobs(model, input_ids, mask)
    finally:
        handle.remove()
    assert len(calls) == 2
    assert not model.training
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_token_logprobs_backpropagate_to_shared_model(model):
    input_ids = torch.tensor([[4, 5, 6, 2]])
    mask = torch.ones_like(input_ids)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = token_logprobs(model, input_ids, mask)
    assert before.requires_grad
    (-before.mean()).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.count_nonzero(grad) for grad in gradients)
    optimizer.step()
    after = token_logprobs(model, input_ids, mask)
    assert not torch.allclose(before, after)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.7])
@pytest.mark.parametrize(
    "checkpoint_settings",
    [{}, {"encoder_repetition_penalty": 2.0}, {"encoder_no_repeat_ngram_size": 1}],
)
def test_rollout_logprobs_match_sampling_distribution_and_rescoring(
    model, temperature, checkpoint_settings
):
    model.generation_config.update(**checkpoint_settings)
    prefixes = [[4, 5, 6], [7]]
    results = HFInference(model, pad_token_id=0).generate(
        prefixes,
        num_samples=2,
        max_new_tokens=3,
        temperature=temperature,
        return_logprobs=True,
    )
    assert [len(group) for group in results] == [2, 2]
    for prefix, group in zip(prefixes, results):
        for completion in group:
            assert 1 <= len(completion.token_ids) <= 3
            ids = torch.tensor([prefix + completion.token_ids])
            mask = torch.ones_like(ids)
            with torch.no_grad():
                logits = model(
                    input_ids=ids, attention_mask=mask, use_cache=False
                ).logits
            expected = (
                (logits[:, :-1].float() / temperature)
                .log_softmax(-1)
                .gather(-1, ids[:, 1:, None])
                .squeeze(-1)[0, len(prefix) - 1 :]
            )
            rollout = torch.tensor(completion.logprobs)
            torch.testing.assert_close(rollout, expected, rtol=1e-5, atol=1e-6)
            if temperature == 1:
                rescored = token_logprobs(model, ids, mask)[0, len(prefix) - 1 :]
                ratios = (rescored - rollout).exp()
                torch.testing.assert_close(
                    ratios, torch.ones_like(ratios), rtol=1e-5, atol=1e-6
                )


def test_generate_context_budget_uses_longest_prefix_and_accepts_boundary(
    model, monkeypatch
):
    inference = HFInference(model, pad_token_id=0)
    scripted_generate(monkeypatch, model, [[7, 8], [9, 10]])
    limit = model.config.max_position_embeddings
    results = inference.generate([[4] * (limit - 2), [5]], max_new_tokens=2)
    assert len(results) == 2
    with pytest.raises((AssertionError, ValueError)):
        inference.generate([[4] * (limit - 1), [5]], max_new_tokens=2)


@pytest.mark.parametrize(
    "prefixes,kwargs",
    [
        ([], {}),
        ([[]], {}),
        ([[4], []], {}),
        ([[-1]], {}),
        ([[23]], {}),
        ([[3.5]], {}),
        ([[True]], {}),
        ([["4"]], {}),
        *[
            ([[4]], {key: value})
            for key in ("num_samples", "max_new_tokens")
            for value in (0, -1, 1.5, True, None, "1")
        ],
        *[
            ([[4]], {"temperature": value})
            for value in (-0.1, float("nan"), float("inf"), -float("inf"), None, "hot")
        ],
        ([[4]], {"num_samples": 2, "temperature": 0}),
    ],
)
def test_generate_rejects_malformed_arguments(model, monkeypatch, prefixes, kwargs):
    def unexpected_generate(*args, **kwargs):
        pytest.fail("invalid arguments reached model.generate")

    monkeypatch.setattr(model, "generate", unexpected_generate)
    inference = HFInference(model, pad_token_id=0)
    with pytest.raises((AssertionError, TypeError, ValueError)):
        inference.generate(prefixes, **({"max_new_tokens": 2} | kwargs))


@pytest.mark.parametrize("pad", [-1, 23, 0.5, True, None])
def test_hf_inference_rejects_invalid_pad_ids(model, pad):
    with pytest.raises((AssertionError, TypeError, ValueError)):
        HFInference(model, pad_token_id=pad).generate([[4]], max_new_tokens=1)


def test_hf_inference_requires_explicit_pad_id(model):
    with pytest.raises(TypeError):
        HFInference(model)


@pytest.mark.parametrize("eos", [None, [], -1, 23, [2, 23], [2, -1], 2.5, True])
def test_hf_inference_requires_legal_eos_ids(model, eos):
    model.generation_config.eos_token_id = eos
    with pytest.raises((AssertionError, TypeError, ValueError)):
        HFInference(model, pad_token_id=0).generate([[4]], max_new_tokens=1)

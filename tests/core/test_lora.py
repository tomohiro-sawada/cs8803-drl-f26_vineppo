"""Local CPU contracts for PEFT actors; no pretrained downloads."""

import io
import json
from copy import deepcopy

import pytest
import torch
from peft import LoraConfig, PeftModel
from peft.tuners.lora import LoraLayer
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from core.inference import HFInference, token_logprobs
from core.lora import apply_lora, lora_identity


@pytest.fixture(params=["llama", "gpt2"])
def model(request):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        common = {
            "vocab_size": 23,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }
        if request.param == "llama":
            return LlamaForCausalLM(
                LlamaConfig(
                    **common,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=24,
                    attention_dropout=0.0,
                )
            )
        return GPT2LMHeadModel(
            GPT2Config(
                **common,
                n_positions=24,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )


def test_defaults_all_linear_and_zero_init(model):
    reference = deepcopy(model).eval().requires_grad_(False)
    expected_targets = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and module is not model.get_output_embeddings()
    }
    actor = apply_lora(model)
    assert isinstance(actor, PeftModel)
    assert actor.get_base_model() is model
    config = actor.peft_config["default"]
    assert (config.r, config.lora_alpha, config.lora_dropout, config.bias) == (
        16,
        32,
        0,
        "none",
    )
    assert config.task_type == "CAUSAL_LM"
    assert not isinstance(model.get_output_embeddings(), LoraLayer)
    targets = {
        name for name, module in model.named_modules() if isinstance(module, LoraLayer)
    }
    assert expected_targets <= targets
    assert any("mlp" in name for name in targets)
    assert any("attn" in name for name in targets)
    ids = torch.tensor([[1, 4, 5, 6]])
    with torch.no_grad():
        torch.testing.assert_close(
            token_logprobs(actor, ids, torch.ones_like(ids)),
            token_logprobs(reference, ids, torch.ones_like(ids)),
            rtol=0,
            atol=0,
        )
    assert all(
        torch.count_nonzero(module.lora_B["default"].weight) == 0
        for module in actor.modules()
        if isinstance(module, LoraLayer)
    )
    assert {p.data_ptr() for p in actor.parameters()}.isdisjoint(
        p.data_ptr() for p in reference.parameters()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_only_fp32_adapters_update_base_and_reference_unchanged(model, dtype):
    if dtype == torch.bfloat16:
        try:
            torch.ones(2, 2, dtype=dtype) @ torch.ones(2, 2, dtype=dtype)
        except RuntimeError as error:
            pytest.skip(f"CPU BF16 matmul unsupported: {error}")
    model.to(dtype=dtype)
    reference = deepcopy(model).eval().requires_grad_(False)
    reference_before = {
        name: value.clone() for name, value in reference.state_dict().items()
    }
    actor = apply_lora(model, rank=2, alpha=4.0)
    before = {name: p.detach().clone() for name, p in actor.named_parameters()}
    trainable = {name: p for name, p in actor.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_" in name for name in trainable)
    assert all(p.dtype == torch.float32 for p in trainable.values())
    assert all(p.dtype == dtype for p in actor.parameters() if not p.requires_grad)
    optimizer = torch.optim.SGD(trainable.values(), lr=0.5)
    ids = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]])
    scores = token_logprobs(actor, ids, torch.ones_like(ids))
    assert scores.dtype == torch.float32 and scores.requires_grad
    assert torch.isfinite(scores).all()
    (-scores.mean()).backward()
    for name, p in actor.named_parameters():
        if name in trainable:
            assert p.grad is not None and p.grad.dtype == torch.float32
            assert torch.isfinite(p.grad).all()
        else:
            assert p.grad is None
    assert any(torch.count_nonzero(p.grad) > 0 for p in trainable.values())
    optimizer.step()
    assert any(not torch.equal(p, before[name]) for name, p in trainable.items())
    for name, p in actor.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, before[name])
    for name, value in reference.state_dict().items():
        assert torch.equal(value, reference_before[name])
    assert all(p.grad is None for p in reference.parameters())


def test_generation_uses_actor_and_matches_production_rescoring(model):
    actor = apply_lora(model, rank=2)
    with torch.no_grad():
        for module in actor.modules():
            if isinstance(module, LoraLayer):
                module.lora_B["default"].weight.fill_(0.1)
    backend = HFInference(actor, pad_token_id=0)
    assert backend.model is actor
    completion = backend.generate(
        [[1, 4]],
        max_new_tokens=3,
        temperature=0,
        return_logprobs=True,
    )[0][0]
    assert completion.token_ids
    ids = torch.tensor([[1, 4] + completion.token_ids])
    expected = token_logprobs(actor, ids, torch.ones_like(ids))[0, 1:]
    torch.testing.assert_close(torch.tensor(completion.logprobs), expected)
    backend.close()
    assert actor.get_base_model() is model


def test_identity_is_primitive_sensitive_and_full_checkpoint_roundtrips(model):
    assert lora_identity(model) is None
    other_base = deepcopy(model)
    actor = apply_lora(model, rank=2, alpha=4.0)
    other = apply_lora(other_base, rank=2, alpha=8.0)
    identity = lora_identity(actor)
    assert identity != lora_identity(other)
    assert {k: v.shape for k, v in actor.state_dict().items()} == {
        k: v.shape for k, v in other.state_dict().items()
    }
    assert identity == json.loads(json.dumps(identity, allow_nan=False))
    assert identity["adapters"]["default"]["peft_type"] == "LORA"
    assert type(identity["adapters"]["default"]["peft_type"]) is str
    assert isinstance(identity["adapters"]["default"]["target_modules"], list)
    buffer = io.BytesIO()
    torch.save({"identity": identity, "model": actor.state_dict()}, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert loaded["identity"] == identity
    actor.load_state_dict(loaded["model"], strict=True)
    layer = next(module for module in actor.modules() if isinstance(module, LoraLayer))
    layer.scaling["default"] *= 2
    assert lora_identity(actor) != identity
    layer.scaling["default"] /= 2
    with actor.disable_adapter():
        assert lora_identity(actor) != identity
    assert lora_identity(actor) == identity
    actor.add_adapter(
        "second", LoraConfig(r=2, lora_alpha=4, target_modules="all-linear")
    )
    assert set(lora_identity(actor)["adapters"]) == {"default", "second"}
    both = lora_identity(actor)
    actor.set_adapter("second")
    assert lora_identity(actor) != both


@pytest.mark.parametrize("rank", [0, -1, True, 1.5, "2"])
def test_invalid_rank_fails_before_mutation(model, rank):
    with pytest.raises(AssertionError, match="rank"):
        apply_lora(model, rank=rank)
    assert lora_identity(model) is None


@pytest.mark.parametrize("alpha", [0, -1, True, float("nan"), float("inf"), "4"])
def test_invalid_alpha_fails_before_mutation(model, alpha):
    with pytest.raises(AssertionError, match="alpha"):
        apply_lora(model, alpha=alpha)
    assert lora_identity(model) is None


def test_rejects_double_wrapping_and_quantization(model):
    model.is_quantized = True
    with pytest.raises(AssertionError, match="Quantization"):
        apply_lora(model)
    model.is_quantized = False
    actor = apply_lora(model)
    with pytest.raises(AssertionError):
        apply_lora(actor)
    with pytest.raises(AssertionError, match="already has adapters"):
        apply_lora(model)

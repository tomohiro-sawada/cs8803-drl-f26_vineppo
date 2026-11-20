"""Offline CPU checks against real locally saved HF Llama models/tokenizers."""

import json
from hashlib import sha256

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from core.inference import token_logprobs
from core.lora import apply_lora, lora_identity
from runtime.checkpoint_actor import load_checkpoint_actor
from runtime.checkpoints import (
    COMPACT_LORA,
    file_sha256,
    publish_checkpoint,
    trainable_state_keys,
)


@pytest.fixture
def local_actor(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "local-model"
    backend = Tokenizer(
        models.WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                **{f"word{i}": i for i in range(4, 16)},
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=32,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                attention_dropout=0.0,
            )
        )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    def make(*, lora=False, dtype=torch.float32):
        actor = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=dtype,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
        if lora:
            actor = apply_lora(actor, rank=2, alpha=4.0)
        # Nonzero actor/adapter updates, not an identity round trip of base weights.
        with torch.no_grad():
            for parameter in actor.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.03125)
        actor.eval()
        state = {
            "policy": actor.state_dict(),
            "model_identity": [
                {
                    "class": f"{type(actor).__module__}.{type(actor).__qualname__}",
                    "config": actor.config.to_dict(),
                    "dtypes": {
                        key: str(value.dtype)
                        for key, value in actor.state_dict().items()
                    },
                    **({"lora": lora_identity(actor)} if lora else {}),
                }
            ],
            "config": {
                "num_iterations": 3,
                "algorithm_type": "algorithms.ppo.PPOConfig",
            },
            "iteration": 2,
            "evaluation_provenance": {
                "schema_version": 1,
                "task": "math",
                "prompt": "[MATH_TASK] Problem:\n{question}\n\nSolution:",
                "tokenizer_sha256": sha256(
                    tokenizer.backend_tokenizer.to_str().encode("utf-8")
                ).hexdigest(),
                "dataset": {"path": "local-training-data", "split": "train"},
            },
        }
        checkpoint = tmp_path / "iteration-2.pt"
        torch.save(state, checkpoint)
        return actor, state, checkpoint

    return path, make


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_scores_and_state(local_actor, lora, dtype, monkeypatch):
    path, make = local_actor
    original, state, checkpoint = make(lora=lora, dtype=dtype)
    # Unused components are allowed but not constructed or consumed by the loader.
    state.update(
        reference={"unused": torch.ones(8)},
        optimizer={"unused": torch.ones(8)},
        critic={"unused": torch.ones(8)},
        critic_optimizer={"unused": torch.ones(8)},
    )
    torch.save(state, checkpoint)
    load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return load(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Must not construct an optimizer or reload base weights")

    monkeypatch.setattr(torch, "load", checked_load)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    restored, tokenizer, info = load_checkpoint_actor(checkpoint, path, device="cpu")
    assert calls == [{"weights_only": True, "map_location": "cpu", "mmap": True}]
    assert not restored.training and not any(
        p.requires_grad for p in restored.parameters()
    )
    assert restored.config._attn_implementation == "sdpa"
    assert restored.state_dict().keys() == original.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        if "lora_" in name:
            assert value.dtype == torch.float32
            assert torch.count_nonzero(value)
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 9, 2]])
    with torch.no_grad():
        expected = token_logprobs(original, ids, torch.ones_like(ids, dtype=torch.bool))
        actual = token_logprobs(restored, ids, torch.ones_like(ids, dtype=torch.bool))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert info == {
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "iteration": 2,
        "finetuning": "lora" if lora else "full",
        "dtype": str(dtype).removeprefix("torch."),
        "lora": lora_identity(original),
        "tokenizer_sha256": sha256(
            tokenizer.backend_tokenizer.to_str().encode()
        ).hexdigest(),
        "training_config": state["config"],
        "evaluation_provenance": state["evaluation_provenance"],
    }
    json.dumps(info)


def test_compact_actor_matches_legacy_and_verifies_both_digests(local_actor):
    path, make = local_actor
    original, state, checkpoint = make(lora=True, dtype=torch.bfloat16)
    keys = trainable_state_keys(original)
    frozen = checkpoint.with_name("frozen-lora.pt")
    publish_checkpoint(
        frozen,
        {
            "policy": {
                name: value
                for name, value in state["policy"].items()
                if name not in keys
            },
            "reference": {},
        },
    )
    compact = checkpoint.with_name("compact.pt")
    publish_checkpoint(
        compact,
        state
        | {
            "format_version": COMPACT_LORA,
            "policy": {name: state["policy"][name] for name in keys},
            "frozen": {"path": str(frozen), "sha256": file_sha256(frozen)},
        },
    )
    legacy, _, _ = load_checkpoint_actor(checkpoint, path, device="cpu")
    restored, _, provenance = load_checkpoint_actor(
        compact, path, device="cpu", expected_sha256=file_sha256(compact)
    )
    assert provenance["checkpoint_sha256"] == file_sha256(compact)
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 9, 2]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    torch.testing.assert_close(
        token_logprobs(restored, ids, mask),
        token_logprobs(legacy, ids, mask),
        rtol=0,
        atol=0,
    )
    with pytest.raises(AssertionError, match="Selected checkpoint hash mismatch"):
        load_checkpoint_actor(compact, path, device="cpu", expected_sha256="0" * 64)


@pytest.mark.parametrize("lora", [False, True])
def test_relocated_model_origin(local_actor, lora):
    path, make = local_actor
    _, _, checkpoint = make(lora=lora)
    moved = path.with_name("relocated")
    path.rename(moved)
    load_checkpoint_actor(checkpoint, moved, device="cpu")


@pytest.mark.parametrize(
    "mismatch",
    [
        "class",
        "config",
        "rank",
        "alpha",
        "scaling",
        "dtype",
        "shape",
        "keys",
        "adapter_dtype",
    ],
)
def test_identity_rejected_before_weight_load(local_actor, mismatch, monkeypatch):
    path, make = local_actor
    _, state, checkpoint = make(lora=True)
    identity = state["model_identity"][0]
    if mismatch == "class":
        identity["class"] = "transformers.NotLlama"
    elif mismatch == "config":
        identity["config"]["rope_theta"] = 123.0
    elif mismatch in ("rank", "alpha"):
        identity["lora"]["adapters"]["default"][
            "r" if mismatch == "rank" else "lora_alpha"
        ] *= 2
    elif mismatch == "scaling":
        next(iter(identity["lora"]["layers"].values()))["scaling"]["default"] *= 2
    else:
        key = next(key for key in state["policy"] if "lora_A" in key)
        if mismatch == "keys":
            del state["policy"][key]
            del identity["dtypes"][key]
        elif mismatch == "shape":
            state["policy"][key] = state["policy"][key][:1]
        else:
            state["policy"][key] = state["policy"][key].to(torch.bfloat16)
            if mismatch == "adapter_dtype":
                identity["dtypes"][key] = "torch.bfloat16"
    torch.save(state, checkpoint)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid identity must fail before loading actor weights")

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden)
    with pytest.raises(
        AssertionError,
        match="identity mismatch|shape mismatch|keys mismatch|dtype mismatch",
    ):
        load_checkpoint_actor(checkpoint, path, device="cpu")


def test_incompatible_local_config(local_actor):
    path, make = local_actor
    _, _, checkpoint = make()
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config["rms_norm_eps"] *= 10
    config_path.write_text(json.dumps(config))
    with pytest.raises(AssertionError, match="config identity mismatch"):
        load_checkpoint_actor(checkpoint, path, device="cpu")


@pytest.mark.parametrize("mismatch", ["special", "vocab"])
def test_incompatible_tokenizer(local_actor, mismatch):
    path, make = local_actor
    _, _, checkpoint = make()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if mismatch == "special":
        tokenizer.eos_token = "word4"
    else:
        tokenizer.add_tokens(["out_of_range"])
    tokenizer.save_pretrained(path)
    with pytest.raises(
        AssertionError, match="Tokenizer special ID mismatch|exceeds actor vocabulary"
    ):
        load_checkpoint_actor(checkpoint, path, device="cpu")


def test_tokenizer_content_mismatch_without_vocab_size_change(local_actor):
    path, make = local_actor
    _, _, checkpoint = make()
    load_checkpoint_actor(checkpoint, path, device="cpu")
    tokenizer_path = path / "tokenizer.json"
    content = json.loads(tokenizer_path.read_text())
    vocab = content["model"]["vocab"]
    vocab["word4"], vocab["word5"] = vocab["word5"], vocab["word4"]
    tokenizer_path.write_text(json.dumps(content))
    with pytest.raises(AssertionError, match="tokenizer content hash mismatch"):
        load_checkpoint_actor(checkpoint, path, device="cpu")


@pytest.mark.parametrize(
    "invalid",
    [
        "absent",
        None,
        [],
        {},
        "missing_field",
        "extra_field",
        "version",
        "bool_version",
        "task",
        "prompt",
        "hash",
        "dataset",
    ],
)
def test_invalid_evaluation_provenance_before_model_construction(
    local_actor, monkeypatch, invalid
):
    path, make = local_actor
    _, state, checkpoint = make()
    provenance = state["evaluation_provenance"]
    if invalid == "absent":
        del state["evaluation_provenance"]
    elif invalid in (None, [], {}):
        state["evaluation_provenance"] = invalid
    elif invalid == "missing_field":
        del provenance["dataset"]
    elif invalid == "extra_field":
        provenance["unexpected"] = True
    else:
        field, value = {
            "version": ("schema_version", 2),
            "bool_version": ("schema_version", True),
            "task": ("task", "unknown"),
            "prompt": ("prompt", ""),
            "hash": ("tokenizer_sha256", "z" * 64),
            "dataset": ("dataset", {}),
        }[invalid]
        provenance[field] = value
    torch.save(state, checkpoint)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid provenance must fail before constructing the actor")

    monkeypatch.setattr(AutoModelForCausalLM, "from_config", forbidden)
    with pytest.raises(AssertionError, match="evaluation_provenance"):
        load_checkpoint_actor(checkpoint, path, device="cpu")


def test_corrupt_checkpoint_fails(local_actor):
    path, make = local_actor
    _, _, checkpoint = make()
    checkpoint.write_bytes(checkpoint.read_bytes()[:100])
    with pytest.raises(RuntimeError):
        load_checkpoint_actor(checkpoint, path, device="cpu")


def test_legacy_serialization_has_no_fallback(local_actor):
    path, make = local_actor
    _, state, checkpoint = make()
    torch.save(state, checkpoint, _use_new_zipfile_serialization=False)
    with pytest.raises(RuntimeError, match="mmap"):
        load_checkpoint_actor(checkpoint, path, device="cpu")

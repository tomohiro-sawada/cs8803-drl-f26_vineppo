"""Evaluation-only restoration of full actor states from trainer checkpoints.

No trainer, examples, reference, critic, or optimizer is constructed. Requires
modern torch ZIP checkpoints supporting weights_only=True and mmap=True; errors
propagate without retry. mmap avoids materializing unused tensor storages (their
Python metadata is still unpickled). Checkpoints must not change during loading.
"""

import json
from hashlib import sha256
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from core.lora import apply_lora, lora_identity


def _execution_config(config: dict) -> dict:
    # Origin is provenance, not execution semantics. All other fields stay strict,
    # including library version, dtype, RoPE, special IDs and architecture.
    return {key: value for key, value in config.items() if key != "_name_or_path"}


def load_checkpoint_actor(
    checkpoint: Path, model_path: Path, *, device: str = "cuda:0"
) -> tuple[torch.nn.Module, object, dict]:
    """Return a frozen eval actor, local fast tokenizer, and JSON-safe metadata.

    Supports full FP32/BF16 actors and apply_lora actors with FP32 adapters over
    FP32/BF16 bases. Reconstructs from local config, since trainer checkpoints
    contain *all* actor weights. No merge, quantization, dtype coercion, remote
    code, or network access. Only config._name_or_path may differ; the saved
    origin is retained when recreating PEFT so its full identity matches exactly.

    info keys: checkpoint_sha256 (str), iteration (int), finetuning (full/lora),
    dtype (float32/bfloat16, base dtype), lora (exact identity dict or None),
    tokenizer_sha256 (SHA256 of backend_tokenizer.to_str() UTF-8), and
    training_config (checkpoint config dict), and evaluation_provenance (required
    schema-v1 dict with task, prompt, tokenizer_sha256 and dataset). Legacy
    checkpoints without embedded provenance are rejected. The local tokenizer
    must match its embedded hash. The caller MUST compare evaluation_provenance
    exactly with independently constructed task/prompt/tokenizer/dataset metadata.
    A computed checkpoint digest is provenance, not an authenticated checksum.
    Generation settings are local defaults, not recovered training RNG state.
    """
    assert checkpoint.is_file(), f"Checkpoint is not a file: {checkpoint}"
    assert model_path.is_dir(), f"Local model directory not found: {model_path}"
    digest = sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    provenance = state.get("evaluation_provenance")
    assert isinstance(provenance, dict), (
        "Checkpoint requires embedded evaluation_provenance; legacy checkpoints are unsupported"
    )
    assert set(provenance) == {
        "schema_version",
        "task",
        "prompt",
        "tokenizer_sha256",
        "dataset",
    }, "Invalid evaluation_provenance fields"
    assert (
        type(provenance["schema_version"]) is int and provenance["schema_version"] == 1
    ), "Unsupported evaluation_provenance schema_version"
    assert provenance["task"] in ("math", "gsm8k"), "Invalid evaluation_provenance task"
    assert isinstance(provenance["prompt"], str) and provenance["prompt"].strip(), (
        "Invalid evaluation_provenance prompt"
    )
    tokenizer_hash = provenance["tokenizer_sha256"]
    assert (
        isinstance(tokenizer_hash, str)
        and len(tokenizer_hash) == 64
        and all(char in "0123456789abcdef" for char in tokenizer_hash)
    ), "Invalid evaluation_provenance tokenizer_sha256"
    assert isinstance(provenance["dataset"], dict) and provenance["dataset"], (
        "Invalid evaluation_provenance dataset"
    )
    # Dataset schema belongs to the runner; reject tensors/non-JSON payloads here.
    json.dumps(provenance, allow_nan=False)
    identity = state["model_identity"][0]
    weights = state["policy"]
    assert isinstance(state["config"], dict), "Missing training config provenance"
    assert type(state["iteration"]) is int and state["iteration"] >= 0, (
        "Invalid checkpoint iteration"
    )
    assert state["iteration"] <= state["config"]["num_iterations"], (
        "Checkpoint iteration exceeds training config"
    )
    assert weights and all(
        isinstance(value, torch.Tensor) for value in weights.values()
    )
    assert identity["dtypes"] == {
        name: str(value.dtype) for name, value in weights.items()
    }, "Checkpoint state dtype identity mismatch"
    base_dtypes = {
        value.dtype for name, value in weights.items() if "lora_" not in name
    }
    assert len(base_dtypes) == 1, "Expected a uniform base dtype"
    dtype = next(iter(base_dtypes))
    assert dtype in (torch.float32, torch.bfloat16), "Unsupported actor dtype"

    config = AutoConfig.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    # from_pretrained training can override the on-disk dtype. Constructing with
    # the checkpoint's actual dtype reproduces that same runtime config change.
    config._name_or_path = identity["config"].get("_name_or_path", "")
    model = AutoModelForCausalLM.from_config(
        config, dtype=dtype, trust_remote_code=False, attn_implementation="sdpa"
    )
    assert _execution_config(model.config.to_dict()) == _execution_config(
        identity["config"]
    ), "Actor config identity mismatch"
    saved_lora = identity.get("lora")
    if saved_lora is not None:
        assert set(saved_lora["adapters"]) == {"default"}, (
            "Expected one default adapter"
        )
        adapter = saved_lora["adapters"]["default"]
        model = apply_lora(model, rank=adapter["r"], alpha=adapter["lora_alpha"])
    assert lora_identity(model) == saved_lora, "LoRA execution identity mismatch"
    assert (
        f"{type(model).__module__}.{type(model).__qualname__}" == identity["class"]
    ), "Actor class identity mismatch"
    current = model.state_dict()
    assert current.keys() == weights.keys(), "Actor state keys mismatch"
    for name, value in current.items():
        assert value.dtype == weights[name].dtype, f"Actor dtype mismatch: {name}"
        assert value.shape == weights[name].shape, f"Actor shape mismatch: {name}"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    assert tokenizer.is_fast, "Expected a fast tokenizer"
    vocab = tokenizer.get_vocab()
    assert vocab and len(tokenizer) == len(vocab), "Tokenizer vocabulary mismatch"
    assert all(
        type(index) is int and 0 <= index < config.vocab_size
        for index in vocab.values()
    ), "Tokenizer vocabulary exceeds actor vocabulary"
    assert all(index in vocab.values() for index in tokenizer.all_special_ids), (
        "Tokenizer special IDs are outside vocabulary"
    )
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        expected = getattr(config, name)
        actual = getattr(tokenizer, name)
        assert actual == expected or (
            isinstance(expected, list) and actual in expected
        ), f"Tokenizer special ID mismatch: {name}"
    tokenizer_sha256 = sha256(
        tokenizer.backend_tokenizer.to_str().encode("utf-8")
    ).hexdigest()
    assert tokenizer_sha256 == tokenizer_hash, (
        "Checkpoint tokenizer content hash mismatch"
    )
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False).eval().to(device=device)
    return (
        model,
        tokenizer,
        {
            "checkpoint_sha256": digest.hexdigest(),
            "iteration": state["iteration"],
            "finetuning": "lora" if saved_lora is not None else "full",
            "dtype": str(dtype).removeprefix("torch."),
            "lora": saved_lora,
            "tokenizer_sha256": tokenizer_sha256,
            "training_config": state["config"],
            "evaluation_provenance": provenance,
        },
    )

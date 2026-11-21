"""Immutable checkpoint publication and explicit full/compact format readers."""

import os
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

FULL = 1
COMPACT_LORA = 2


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def publish_checkpoint(path: Path, state: dict) -> None:
    """Expose a complete file atomically. Existing checkpoint names are reserved."""
    assert not path.exists() and not path.is_symlink(), f"Checkpoint exists: {path}"
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    ) as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, path)


def load_frozen(spec: dict) -> dict:
    assert set(spec) == {"path", "sha256"}, "Invalid frozen model reference"
    path = Path(spec["path"]).resolve(strict=True)
    assert str(path) == spec["path"], "Frozen model path must be canonical"
    before = file_identity(path)
    assert file_sha256(path) == spec["sha256"], "Frozen model hash mismatch"
    frozen = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert file_identity(path) == before, "Frozen model changed while loading"
    assert set(frozen) == {"policy", "reference"}, "Invalid frozen model fields"
    return frozen


def checkpoint_models(state: dict) -> tuple[dict, dict | None]:
    version = state.get("format_version", FULL)
    assert type(version) is int and version in (FULL, COMPACT_LORA), (
        f"Unsupported checkpoint format: {version}"
    )
    if version == FULL:
        return state["policy"], state.get("reference")
    assert "lora" in state["model_identity"][0], "Compact checkpoints require LoRA"
    assert "reference" not in state, "Compact checkpoints store the reference once"
    frozen = load_frozen(state["frozen"])
    assert not frozen["policy"].keys() & state["policy"].keys(), (
        "Compact trainable and frozen checkpoint keys overlap"
    )
    return frozen["policy"] | state["policy"], frozen["reference"]


def trainable_state_keys(model) -> set[str]:
    return {
        name
        for name, value in model.named_parameters(remove_duplicate=False)
        if value.requires_grad
    }


def validate_tensor_state(model, weights: dict) -> None:
    current = model.state_dict()
    assert current.keys() == weights.keys(), "Model state keys mismatch"
    for name, value in current.items():
        assert value.dtype == weights[name].dtype, f"Model dtype mismatch: {name}"
        assert value.shape == weights[name].shape, f"Model shape mismatch: {name}"


def assert_state_equal(actual, expected, path="state"):
    """Assert exact checkpoint tensor dtype, shape and values across devices."""
    assert type(actual) is type(expected), f"Type mismatch: {path}"
    if isinstance(actual, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, path
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu()), path
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), f"Keys mismatch: {path}"
        for key, value in actual.items():
            assert_state_equal(value, expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)):
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_state_equal(left, right, f"{path}.{i}")
    else:
        assert actual == expected, path

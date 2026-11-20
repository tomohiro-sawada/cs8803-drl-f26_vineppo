"""Full-weight LoRA checkpoints and real adapter-only actor optimization."""

from copy import deepcopy

import pytest
import torch
from peft.tuners.lora import LoraLayer

from algorithms.grpo import GRPOConfig
from algorithms.rloo import RLOOConfig
from algorithms.vineppo import VinePPOConfig
from core.lora import apply_lora
from runtime.checkpoints import COMPACT_LORA, checkpoint_models
from tests.runtime import test_train
from tests.runtime.test_train import assert_exact

training_case = test_train.training_case


@pytest.fixture
def lora_case(training_case):
    make, inputs = training_case

    def build(name="run", *, alpha=4.0, reference_lora=False, **kwargs):
        run = make(name, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            run.policy = apply_lora(run.policy, rank=2, alpha=alpha)
            if reference_lora:
                run.reference_policy = apply_lora(
                    run.reference_policy, rank=2, alpha=4.0
                )
        run.inference.model = run.policy
        run.optimizer = torch.optim.AdamW(
            [p for p in run.policy.parameters() if p.requires_grad], lr=0.003
        )
        return run

    return build, inputs


@pytest.mark.parametrize("algorithm", [VinePPOConfig(), GRPOConfig(), RLOOConfig()])
@pytest.mark.parametrize("compact", [False, True])
def test_lora_train_and_resume_exact(lora_case, monkeypatch, algorithm, compact):
    make, inputs = lora_case
    full = make(
        "full", algorithm_config=algorithm, num_iterations=2, random_answers=True
    )
    initial = deepcopy(full.policy.state_dict())
    reference = deepcopy(full.reference_policy.state_dict())
    steps = []
    step = full.optimizer.step

    def checked_step(*args, **kwargs):
        gradients = []
        for name, parameter in full.policy.named_parameters():
            assert parameter.requires_grad == ("lora_" in name)
            if parameter.requires_grad:
                assert (
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                )
                gradients.append(parameter.grad.abs().sum().item())
            else:
                assert parameter.grad is None
        assert sum(gradients) > 0
        assert all(
            p.grad is None and not p.requires_grad
            for p in full.reference_policy.parameters()
        )
        steps.append(1)
        return step(*args, **kwargs)

    monkeypatch.setattr(full.optimizer, "step", checked_step)
    torch.manual_seed(123)
    expected = full.train(**inputs)
    rng = torch.get_rng_state()
    assert steps
    assert_exact(full.reference_policy.state_dict(), reference)
    changed = [
        name
        for name, value in full.policy.state_dict().items()
        if not torch.equal(value, initial[name])
    ]
    assert changed and all("lora_" in name for name in changed)

    first = make("first", algorithm_config=algorithm, random_answers=True)
    first.compact_lora_checkpoints = compact
    torch.manual_seed(123)
    history = first.train(**inputs)
    path = first.config.checkpoint_dir / "iteration-1.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    policy, reference_state = checkpoint_models(state)
    assert_exact(dict(policy), dict(first.policy.state_dict()))
    assert_exact(reference_state, first.reference_policy.state_dict())
    assert any("base_layer.weight" in name for name in policy)
    if compact:
        assert state["format_version"] == COMPACT_LORA
        assert all("lora_" in name for name in state["policy"])
        assert (
            path.stat().st_size
            < (full.config.checkpoint_dir / "iteration-1.pt").stat().st_size
        )
    assert state["model_identity"][0]["lora"] is not None
    resumed = make(
        "resumed", algorithm_config=algorithm, num_iterations=2, random_answers=True
    )
    resumed.compact_lora_checkpoints = compact
    # Different base/reference weights must be restored from the checkpoint too.
    with torch.no_grad():
        for model in (resumed.policy, resumed.reference_policy):
            for parameter in model.parameters():
                parameter.add_(0.1)
    resumed.load_checkpoint(path, **inputs)
    assert_exact(resumed.policy.state_dict(), first.policy.state_dict())
    assert_exact(resumed.reference_policy.state_dict(), reference)
    assert history + resumed.train(**inputs) == expected
    if compact:
        new = torch.load(
            resumed.config.checkpoint_dir / "iteration-2.pt", weights_only=True
        )
        assert new["frozen"] == state["frozen"]
        assert not (resumed.config.checkpoint_dir / "frozen-lora.pt").exists()
    for name in ("policy", "reference_policy", "optimizer"):
        assert_exact(
            getattr(resumed, name).state_dict(), getattr(full, name).state_dict()
        )
    assert_exact(resumed._generator.get_state(), full._generator.get_state())
    assert_exact(torch.get_rng_state(), rng)


def test_compact_checkpoint_refuses_mutation_and_requires_frozen_blob(lora_case):
    make, inputs = lora_case
    original = make("original")
    original.compact_lora_checkpoints = True
    original.train(**inputs)
    with pytest.raises(AssertionError, match="Checkpoint exists"):
        original.save_checkpoint()
    with torch.no_grad():
        next(original.reference_policy.parameters()).add_(1)
    with pytest.raises(
        AssertionError, match="Frozen model parameters or buffers changed"
    ):
        original.save_checkpoint(original.config.checkpoint_dir / "alternate.pt")
    path = original.config.checkpoint_dir / "iteration-1.pt"
    restored = make("restored")
    blob = original.config.checkpoint_dir / "frozen-lora.pt"
    blob.rename(blob.with_suffix(".moved"))
    with pytest.raises(FileNotFoundError):
        restored.load_checkpoint(path, **inputs)
    blob.with_suffix(".moved").rename(blob)
    with blob.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(AssertionError, match="Frozen model hash mismatch"):
        restored.load_checkpoint(path, **inputs)


@pytest.mark.parametrize("target", ["policy", "reference_policy"])
@pytest.mark.parametrize("mismatch", ["alpha", "scaling"])
def test_lora_identity_mismatch_before_weights_load(
    lora_case, monkeypatch, target, mismatch
):
    make, inputs = lora_case
    first = make(algorithm_config=GRPOConfig(), reference_lora=True)
    first.train(**inputs)
    resumed = make(
        "resumed",
        algorithm_config=GRPOConfig(),
        reference_lora=True,
        alpha=8.0 if target == "policy" and mismatch == "alpha" else 4.0,
    )
    model = getattr(resumed, target)
    if mismatch == "scaling":
        layer = next(
            module for module in model.modules() if isinstance(module, LoraLayer)
        )
        layer.scaling["default"] *= 2
    elif target == "reference_policy":
        model.peft_config["default"].lora_alpha = 8.0

    def forbidden(*args, **kwargs):
        pytest.fail("Identity mismatch must fail before any weights load")

    for component in (resumed.policy, resumed.reference_policy, resumed.optimizer):
        monkeypatch.setattr(component, "load_state_dict", forbidden)
    with pytest.raises(AssertionError, match="Model execution identity mismatch"):
        resumed.load_checkpoint(
            first.config.checkpoint_dir / "iteration-1.pt", **inputs
        )

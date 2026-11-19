"""Full-weight LoRA checkpoints and real adapter-only actor optimization."""

from copy import deepcopy

import pytest
import torch
from peft.tuners.lora import LoraLayer

from algorithms.grpo import GRPOConfig
from algorithms.rloo import RLOOConfig
from algorithms.vineppo import VinePPOConfig
from core.lora import apply_lora
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
def test_lora_train_and_resume_exact(lora_case, monkeypatch, algorithm):
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
    torch.manual_seed(123)
    history = first.train(**inputs)
    path = first.config.checkpoint_dir / "iteration-1.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert_exact(state["policy"], first.policy.state_dict())
    assert_exact(state["reference"], first.reference_policy.state_dict())
    assert any("base_layer.weight" in name for name in state["policy"])
    assert state["model_identity"][0]["lora"] is not None
    resumed = make(
        "resumed", algorithm_config=algorithm, num_iterations=2, random_answers=True
    )
    # Different base/reference weights must be restored from the checkpoint too.
    with torch.no_grad():
        for model in (resumed.policy, resumed.reference_policy):
            for parameter in model.parameters():
                parameter.add_(0.1)
    resumed.load_checkpoint(path, **inputs)
    assert_exact(resumed.policy.state_dict(), first.policy.state_dict())
    assert_exact(resumed.reference_policy.state_dict(), reference)
    assert history + resumed.train(**inputs) == expected
    for name in ("policy", "reference_policy", "optimizer"):
        assert_exact(
            getattr(resumed, name).state_dict(), getattr(full, name).state_dict()
        )
    assert_exact(resumed._generator.get_state(), full._generator.get_state())
    assert_exact(torch.get_rng_state(), rng)


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

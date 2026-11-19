"""Offline helpers and real tiny CPU actor updates; never CUDA acceptance results."""

import json
from dataclasses import replace

import pytest
import torch

from scripts.diagnostics import check_training_numerics as check


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_evidence_retains_failures_and_rejects_overwrite(tmp_path):
    evidence = check.Evidence(tmp_path / "result")
    evidence.check("good", True)
    evidence.close("bad", torch.tensor([1.]), torch.tensor([0.]), check.LIMITS["fp32"])
    evidence.close("nan", torch.tensor([float("nan")]), torch.tensor([0.]), check.LIMITS["fp32"])
    records = [json.loads(line) for line in evidence.path.read_text().splitlines()]
    assert [r["passed"] for r in records] == [True, False, False]
    assert records[1]["max_abs"] == 1 and records[2]["max_abs"] is None
    assert evidence.failures == ["bad", "nan"]
    with pytest.raises(FileExistsError):
        check.Evidence(tmp_path / "result")


def test_tensor_hash_dtype_shape_and_one_bit():
    tensor = torch.tensor([[0., 1.]], dtype=torch.bfloat16)
    assert check.tensor_hash(tensor) == check.tensor_hash(tensor.clone())
    assert check.tensor_hash(tensor) != check.tensor_hash(tensor.float())
    assert check.tensor_hash(tensor) != check.tensor_hash(tensor.flatten())
    changed = tensor.clone()
    changed[0, 0] = torch.finfo(tensor.dtype).tiny
    assert check.tensor_hash(tensor) != check.tensor_hash(changed)
    assert check.tensor_hash(torch.tensor(3))


def batch(objective="rloo"):
    return check.make_batch([([3, 4], [5, 2]), ([3, 4], [6, 7, 8, 2])], 0, objective)


def test_masked_accumulated_ratios_and_clipping():
    data = batch()
    current = torch.full_like(data.old_logprobs, float("nan"))
    current[data.response_mask] = 0.1
    ratios = check.ratio_views(current, data)
    torch.testing.assert_close(ratios["token"], torch.full((6,), 0.1).exp())
    torch.testing.assert_close(ratios["sequence"], torch.tensor([0.2, 0.4]).exp())
    decisions = check.clipping(torch.tensor([0.7, 0.7, 1.3, 1.3]), torch.tensor([-1., 1., -1., 1.]))
    assert decisions[2].tolist() == [True, False, False, True]


def test_cross_precision_catches_sequence_drift_hidden_by_token_error(tmp_path):
    evidence = check.Evidence(tmp_path / "cross")
    data = check.make_batch([([3], [4] * 100), ([3], [5] * 100)], 0, "rloo")
    current = torch.full_like(data.old_logprobs, 0.003)
    check.cross_precision(evidence, "drift", current, torch.zeros_like(current), data, "rloo")
    assert "drift/logp" not in evidence.failures
    assert "drift/token_ratio" not in evidence.failures
    assert "drift/sequence_ratio" in evidence.failures
    assert "drift/sequence_clipping" in evidence.failures


def test_group_ties_are_exact_cpu(tmp_path):
    evidence = check.Evidence(tmp_path / "ties")
    check.grouped_ties(evidence, "cpu")
    assert not evidence.failures


@pytest.mark.parametrize("objective", ["ppo", "grpo", "rloo"])
def test_real_runtime_microbatch_and_frozen_contract(tmp_path, objective):
    from transformers import GPT2Config, GPT2LMHeadModel

    from core.lora import apply_lora

    torch.manual_seed(42)
    actor = apply_lora(GPT2LMHeadModel(GPT2Config(
        vocab_size=13, n_positions=16, n_embd=16, n_layer=1, n_head=2,
        eos_token_id=2, pad_token_id=0, resid_pdrop=0, embd_pdrop=0,
        attn_pdrop=0, attn_implementation="sdpa",
    )), rank=16, alpha=32)
    initial = {n: p.detach().clone() for n, p in actor.named_parameters() if p.requires_grad}
    frozen = check.frozen(actor)
    data = batch(objective)
    old = check.score(actor, data)
    data = replace(data, old_logprobs=old, ref_logprobs=old.clone())
    before = check.batch_hash(data)
    evidence = check.Evidence(tmp_path / objective)
    one = check.optimize(actor, data, objective, 1, evidence, "one")
    assert check.frozen(actor) == frozen and check.batch_hash(data) == before
    assert any(torch.count_nonzero(d) for n, d in one[2][0].items() if "lora_B." in n)
    with torch.no_grad():
        for n, p in actor.named_parameters():
            if p.requires_grad:
                p.copy_(initial[n])
    two = check.optimize(actor, data, objective, 2, evidence, "two")
    check.compare_updates(evidence, "micro", one, two, "float32")
    assert not evidence.failures
    assert check.frozen(actor) == frozen and check.batch_hash(data) == before


def test_cli_requires_explicit_a40():
    common = ["--model-path", "/model", "--dataset-path", "/data", "--output", "/out", "--task", "math"]
    with pytest.raises(SystemExit):
        check.parse_args(common)
    with pytest.raises(SystemExit):
        check.parse_args(common + ["--expected-gpu", "H100"])
    assert check.parse_args(common + ["--expected-gpu", "A40"]).expected_gpu == "A40"


def test_no_cuda_fails_without_loading_models(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    evidence = check.Evidence(tmp_path / "no_cuda")
    with pytest.raises(AssertionError, match="CUDA required"):
        check.run(None, evidence)


def test_main_persists_fatal_before_rethrow(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    def fail(*args):
        raise RuntimeError("deliberate failure")
    monkeypatch.setattr(check, "run", fail)
    out = tmp_path / "fatal"
    with pytest.raises(RuntimeError, match="deliberate"):
        check.main(["--model-path", str(tmp_path), "--dataset-path", str(tmp_path),
                    "--output", str(out), "--task", "gsm8k", "--expected-gpu", "A40"])
    lines = [json.loads(line) for line in (out / "evidence.jsonl").read_text().splitlines()]
    assert lines[0]["stage"] == "start"
    assert lines[-1]["stage"] == "fatal" and "deliberate failure" in lines[-1]["traceback"]


def test_reward_groups_cannot_mix_prompts():
    with pytest.raises(AssertionError, match="Mixed prompts"):
        check.make_batch([([3], [4, 2]), ([5], [6, 2])], 0, "grpo")

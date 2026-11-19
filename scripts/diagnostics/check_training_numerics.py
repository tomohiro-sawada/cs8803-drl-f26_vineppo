"""Offline A40 fixed-data acceptance, NOT learning/throughput validation.

Local DatasetDict/train uses question/answer (gsm8k) or problem/solution (math).
Short probes use two real solutions, each with unequal prefix lengths and EOS.
The 2048-token probe packs contiguous real corpus tokens, NOT sampled rollouts.
Rewards [0, 1] are deliberately synthetic diagnostics, never verifier accuracy.
Only production scoring, collation, advantages and update_actor optimize weights.
Evidence is fsynced incrementally; numerical failures are recorded, never waived.
Trainer checkpoint/resume, collection/MC and critic checks remain explicit gaps.
"""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import fields, replace
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

# Acceptance budgets are intentionally fixed, not CLI knobs to loosen after failure.
# Cross-precision ratio budget is 1/10 of clip width; any decision change fails.
LIMITS = {
    "fp32": {"atol": 1e-6, "rtol": 1e-5},
    "bf16_repeat": {"atol": 2e-3, "rtol": 0.0},
    "cross_logp": {"atol": 0.02, "rtol": 0.0},
    "cross_ratio": {"atol": 0.02, "rtol": 0.0},
    "cross_loss": {"atol": 0.02, "rtol": 0.02},
    "bf16_micro": {"atol": 2e-5, "rtol": 0.02},
    "fp32_delta": {"atol": 1e-7, "rtol": 1e-4},
    "bf16_delta": {"atol": 2e-7, "rtol": 0.02},
}
GAPS = [
    "Real PPOTrainer checkpoint/load/two-iteration resume and negative identity tests untested; no optimizer-only substitute.",
    "Fresh-process RNG and sampled-token replay untested.",
    "Collection, verifier rewards, segmentation, generation KV cache, MC accounting and learned critic untested.",
    "2048-token packed real text is a training-memory stress probe, not a sampled completion or throughput estimate.",
    "PPO actor uses diagnostic raw advantages; no claim about PPO critic or VinePPO MC estimation.",
]


class Evidence:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=False)
        self.path = directory / "evidence.jsonl"
        self.failures = []

    def write(self, stage, **values):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"stage": stage, **values}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def check(self, name, passed, **values):
        passed = bool(passed)
        self.write(name, passed=passed, **values)
        if not passed:
            self.failures.append(name)

    def close(self, name, actual, expected, budget):
        actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
        assert actual.shape == expected.shape and actual.numel(), name
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        difference = (actual - expected).abs()
        self.check(
            name, finite and torch.allclose(actual, expected, **budget),
            finite=finite, max_abs=float(difference.max()) if finite else None,
            mean_abs=float(difference.mean()) if finite else None, budget=budget,
        )


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def hashes(tensors):
    return {name: tensor_hash(value) for name, value in tensors.items()}


def batch_hash(batch):
    return hashes({field.name: getattr(batch, field.name) for field in fields(batch)})


def frozen(model):
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    return hashes({name: value for name, value in model.state_dict().items() if name not in trainable})


def ratio_views(current, batch):
    mask = batch.response_mask
    log_ratio = (current.cpu() - batch.old_logprobs).masked_fill(~mask, 0)
    return {"token": log_ratio[mask].exp(), "sequence": log_ratio.sum(1).exp()}


def clipping(ratio, advantages):
    # Both clip-boundary classification and the actual sign-dependent surrogate branch.
    return torch.stack((ratio < 0.8, ratio > 1.2,
                        ((ratio < 0.8) & (advantages < 0)) |
                        ((ratio > 1.2) & (advantages > 0))))


def forbid(*args, **kwargs):
    raise AssertionError("Grouped actor entered whitening/generation/MC path")


def grouped_ties(evidence, device):
    from algorithms.grpo import group_advantages, grpo_loss
    from algorithms.rloo import rloo_advantages, rloo_loss

    for name, advantage_fn, loss_fn in (
        ("grpo", group_advantages, grpo_loss), ("rloo", rloo_advantages, rloo_loss)
    ):
        rewards = torch.full((2, 8), -0.1, device=device)
        adv = advantage_fn(rewards)
        current = torch.zeros((16, 3), device=device, requires_grad=True)
        mask = torch.ones_like(current, dtype=torch.bool)
        loss, _ = loss_fn(current, current.detach(), current.detach(),
                          adv.reshape(-1, 1).expand_as(current), mask, kl_coef=0)
        loss.backward()
        evidence.check(name + "/ties_exact_zero",
                       torch.count_nonzero(adv) == 0 and loss.item() == 0 and
                       torch.count_nonzero(current.grad) == 0)
        pair = advantage_fn(torch.tensor([[0., 1.], [2., 3.]], device=device))
        evidence.close(name + "/group_isolation_shift", pair,
                       torch.tensor([[-1., 1.], [-1., 1.]]), LIMITS["fp32"])
    evidence.close("rloo/three_rewards", rloo_advantages(torch.tensor([[0., 0., 1.]], device=device)),
                   torch.tensor([[-0.5, -0.5, 1.]]), LIMITS["fp32"])


def load_tokens(args, evidence):
    from datasets import DatasetDict, load_from_disk
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    assert tokenizer.is_fast and tokenizer.eos_token_id is not None
    # Same explicit EOS-as-pad convention as the training launcher, not a retry.
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    dataset = load_from_disk(str(args.dataset_path))
    assert isinstance(dataset, DatasetDict) and "train" in dataset
    short, corpus, sources = [], [], []
    for index, row in enumerate(dataset["train"]):
        if index >= 256:
            break
        question = row["question" if args.task == "gsm8k" else "problem"]
        solution = row["answer" if args.task == "gsm8k" else "solution"]
        assert isinstance(question, str) and question.strip()
        assert isinstance(solution, str) and solution.strip()
        prompt = tokenizer("[MATH_TASK] Problem:\n" + question.strip() + "\n\nSolution:",
                           add_special_tokens=True)["input_ids"]
        response = tokenizer(solution, add_special_tokens=False)["input_ids"]
        assert prompt and len(response) >= 4
        sources.append({"row": index, "text_sha256": hashlib.sha256((question + "\n" + solution).encode()).hexdigest()})
        corpus.extend(prompt + response + [tokenizer.eos_token_id])
        if len(short) < 4:
            assert len(prompt) + min(64, len(response)) + 1 <= 2048, "Short probe exceeds context"
            for length in (min(16, len(response) // 2), min(64, len(response))):
                short.append((prompt, response[:length] + [tokenizer.eos_token_id]))
        if len(short) == 4 and len(corpus) >= 4096:
            break
    assert len(short) == 4 and len(corpus) >= 4096, "Need two examples and 4096 corpus tokens within first 256 rows"
    # Two distinct packed blocks, unequal response lengths; never repeat one token.
    long = [(corpus[:128], corpus[128:2047] + [tokenizer.eos_token_id]),
            (corpus[:128], corpus[2176:4094] + [tokenizer.eos_token_id])]
    evidence.write("fixed_data", sources=sources, task=args.task, pad_token_id=pad,
                   eos_token_id=tokenizer.eos_token_id, tokenizer_sha256=hashlib.sha256(
                       tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                   short=short, long=long, reward_label="SYNTHETIC DIAGNOSTIC [0,1] per group; NOT accuracy",
                   long_label="contiguous real corpus packing; NOT sampled rollouts")
    return {"short": short, "2048": long}, pad


def make_batch(tokens, pad, objective):
    from algorithms.grpo import group_advantages
    from algorithms.rloo import rloo_advantages
    from core.data import EpisodeWithAdvantages, collate

    assert tokens and len(tokens) % 2 == 0
    assert all(tokens[i][0] == tokens[i + 1][0] for i in range(0, len(tokens), 2)), "Mixed prompts in reward group"
    rewards = torch.tensor([[0., 1.]] * (len(tokens) // 2))
    advantages = (rloo_advantages(rewards) if objective == "rloo" else group_advantages(rewards)).flatten()
    episodes = [EpisodeWithAdvantages(prompt, response, float(i % 2), "stop",
                                      [float(advantages[i])] * len(response))
                for i, (prompt, response) in enumerate(tokens)]
    zeros = [[0.] * len(response) for _, response in tokens]
    return collate(episodes, old_logprobs=zeros, ref_logprobs=zeros, pad_token_id=pad)


@torch.no_grad()
def score(model, batch, micro=1):
    from core.inference import token_logprobs

    return torch.cat([token_logprobs(model, ids.to(model.device), mask.to(model.device)).cpu()
                      for ids, mask in zip(batch.input_ids.split(micro),
                                           batch.attention_mask.split(micro), strict=True)])


def optimize(actor, batch, objective, micro, evidence, label, epochs=2):
    import runtime.train as train_module
    from algorithms.actor import update_actor
    from core.inference import HFInference

    params = {name: p for name, p in actor.named_parameters() if p.requires_grad}
    assert params and all(p.dtype == torch.float32 and ("lora_A." in n or "lora_B." in n)
                          for n, p in params.items())
    optimizer = torch.optim.AdamW(list(params.values()), lr=1e-5, weight_decay=0.0)
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(owned) == len(params) and {id(p) for p in owned} == {id(p) for p in params.values()}
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    cache = batch_hash(batch)
    gradients, deltas = [], []
    step = optimizer.step

    def inspected_step(*args, **kwargs):
        assert all(p.grad is None for p in actor.parameters() if not p.requires_grad)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
        grad = {n: p.grad.detach().cpu().clone() for n, p in params.items()}
        gradients.append(grad)
        evidence.check(label + f"/step{len(gradients)}/B_gradient",
                       any(torch.count_nonzero(g) > 0 for n, g in grad.items() if "lora_B." in n))
        result = step(*args, **kwargs)
        delta = {n: p.detach().cpu() - initial[n] for n, p in params.items()}
        deltas.append(delta)
        evidence.write(label + f"/step{len(gradients)}/adapters", tensors={
            n: {"grad_norm_after_clip": float(grad[n].norm()), "delta_norm": float(d.norm()),
                "fraction_changed": float((d != 0).float().mean())} for n, d in delta.items()})
        evidence.check(label + f"/step{len(gradients)}/B_update",
                       any(torch.count_nonzero(d) > 0 for n, d in delta.items() if "lora_B." in n))
        assert len(optimizer.state) == len(params)
        for state in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                assert state[key].dtype == torch.float32 and torch.isfinite(state[key]).all()
        assert batch_hash(batch) == cache, "Update mutated cached rollout"
        return result

    optimizer.step = inspected_step
    # Production updater only; tripwires ensure grouped probes cannot whiten or branch.
    with patch.object(HFInference, "generate", forbid), patch.object(train_module, "estimate_vineppo", forbid):
        if objective != "ppo":
            with patch("algorithms.actor.normalize_advantages", forbid):
                history = update_actor(actor, optimizer, batch, epochs=epochs,
                                       minibatch_size=len(batch.input_ids), microbatch_size=micro,
                                       objective=objective, generator=torch.Generator().manual_seed(42))
        else:
            history = update_actor(actor, optimizer, batch, epochs=epochs,
                                   minibatch_size=len(batch.input_ids), microbatch_size=micro,
                                   objective=objective, generator=torch.Generator().manual_seed(42))
    evidence.write(label + "/history", history=history, optimizer="AdamW", lr=1e-5,
                   weight_decay=0, moment_dtype="float32", optimizer_ownership=True,
                   mc_tokens=0, mc_continuations=0, mc_expansions=0,
                   grouped_no_mc="actor-only tripwires; collection accounting untested")
    return history, gradients, deltas, optimizer


def compare_updates(evidence, label, left, right, dtype):
    history_a, grads_a, deltas_a = left[:3]
    history_b, grads_b, deltas_b = right[:3]
    budget = LIMITS["fp32" if dtype == "float32" else "bf16_micro"]
    for i, (a, b) in enumerate(zip(history_a, history_b, strict=True)):
        for key in a:
            evidence.close(f"{label}/step{i}/{key}", torch.tensor(a[key]), torch.tensor(b[key]), budget)
    for kind, aa, bb in (("gradient_after_clip", grads_a, grads_b), ("delta", deltas_a, deltas_b)):
        tol = LIMITS["fp32_delta" if dtype == "float32" else "bf16_delta"] if kind == "delta" else budget
        for i, (a, b) in enumerate(zip(aa, bb, strict=True)):
            assert a.keys() == b.keys()
            # Per-element tolerance, not just agreement of norms.
            evidence.close(f"{label}/step{i}/{kind}", torch.cat([x.flatten() for x in a.values()]),
                           torch.cat([x.flatten() for x in b.values()]), tol)


def cross_precision(evidence, label, current, baseline, batch, objective):
    from algorithms.actor import actor_loss, normalize_advantages
    from algorithms.grpo import grpo_loss
    from algorithms.rloo import rloo_loss

    evidence.close(label + "/logp", current[batch.response_mask], baseline[batch.response_mask], LIMITS["cross_logp"])
    for name, ratios in ratio_views(current, batch).items():
        expected = ratio_views(baseline, batch)[name]
        evidence.close(label + "/" + name + "_ratio", ratios, expected, LIMITS["cross_ratio"])
        adv = batch.advantages[batch.response_mask] if name == "token" else batch.advantages.sum(1)
        evidence.check(label + "/" + name + "_clipping", torch.equal(clipping(ratios, adv), clipping(expected, adv)),
                       changed_decisions=int((clipping(ratios, adv) != clipping(expected, adv)).sum()))
    adv = normalize_advantages(batch.advantages, batch.response_mask) if objective == "ppo" else batch.advantages
    loss_fn = {"ppo": actor_loss, "grpo": grpo_loss, "rloo": rloo_loss}[objective]
    _, aa = loss_fn(current, batch.old_logprobs, batch.ref_logprobs, adv, batch.response_mask)
    _, bb = loss_fn(baseline, batch.old_logprobs, batch.ref_logprobs, adv, batch.response_mask)
    for name in aa:
        evidence.close(label + "/loss/" + name, aa[name], bb[name], LIMITS["cross_loss"])


def run(args, evidence):
    from transformers import AutoModelForCausalLM

    from core.lora import apply_lora, lora_identity

    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    assert torch.cuda.device_count() == 1, "Expose exactly one CUDA device"
    device_name = torch.cuda.get_device_name(0)
    assert device_name in ("NVIDIA A40", "A40"), f"A40 required, found {device_name}"
    assert torch.cuda.is_bf16_supported(), "Native BF16 required"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    evidence.write("environment", gpu=device_name, cuda=torch.version.cuda,
                   packages={name: version(name) for name in ("torch", "transformers", "peft", "datasets")},
                   nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True),
                   tf32=False, deterministic=True, attention="sdpa", seed=42,
                   rank=16, alpha=32, lr=1e-5, clip_eps=0.2, kl_coef=1e-4,
                   max_grad_norm=1, epochs=2, logical_batch="4 short / 2 packed", limits=LIMITS)
    tokens, pad = load_tokens(args, evidence)
    grouped_ties(evidence, "cuda:0")
    baseline = {}
    for dtype in ("float32", "bfloat16"):
        evidence.write(dtype + "/loading")
        kwargs = {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True,
                  "dtype": getattr(torch, dtype), "attn_implementation": "sdpa"}
        actor = AutoModelForCausalLM.from_pretrained(str(args.model_path), **kwargs).to("cuda:0")
        reference = AutoModelForCausalLM.from_pretrained(str(args.model_path), **kwargs).to("cuda:0")
        reference.requires_grad_(False).eval()
        assert actor.config.max_position_embeddings >= 2048
        assert hashes(actor.state_dict()) == hashes(reference.state_dict()), "Initial base/reference mismatch"
        assert not ({p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in reference.parameters()})
        torch.manual_seed(42)
        actor = apply_lora(actor, rank=16, alpha=32)
        assert all(p.dtype == getattr(torch, dtype) for p in actor.parameters() if not p.requires_grad)
        assert all(p.dtype == getattr(torch, dtype) for p in reference.parameters())
        initial = {n: p.detach().cpu().clone() for n, p in actor.named_parameters() if p.requires_grad}
        assert all(torch.count_nonzero(p) == 0 for n, p in initial.items() if "lora_B." in n)
        if dtype == "float32":
            initial_fp32 = initial
        else:
            # PEFT initializes in the base dtype before promoting adapters; reuse
            # FP32 A tensors rather than comparing different random initializations.
            with torch.no_grad():
                for n, p in actor.named_parameters():
                    if p.requires_grad:
                        p.copy_(initial_fp32[n])
            initial = {n: p.detach().cpu().clone() for n, p in actor.named_parameters() if p.requires_grad}
        assert hashes(initial) == hashes(initial_fp32), "Cross-precision adapter initialization differs"
        frozen_before, ref_before = frozen(actor), frozen(reference)
        evidence.write(dtype + "/identity", lora=lora_identity(actor), base_hashes=frozen_before,
                       reference_hashes=ref_before, initial_adapters=hashes(initial))

        def reset(model, state):
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.copy_(state[n])
            torch.manual_seed(42)

        for length, fixed_tokens in tokens.items():
            for objective in ("ppo", "grpo", "rloo"):
                label = f"{dtype}/{length}/{objective}"
                reset(actor, initial)
                batch = make_batch(fixed_tokens, pad, objective)
                current, ref = score(actor, batch), score(reference, batch)
                repeat_budget = LIMITS["fp32" if dtype == "float32" else "bf16_repeat"]
                evidence.close(label + "/zero_init", current[batch.response_mask], ref[batch.response_mask], repeat_budget)
                evidence.close(label + "/cached_ratio", (score(actor, batch) - current)[batch.response_mask].exp(),
                               torch.ones_like(current[batch.response_mask]), repeat_budget)
                # Compare genuinely unpadded forwards against padded forwards, including EOS.
                for i in range(len(batch.input_ids)):
                    end = int(batch.attention_mask[i].sum())
                    single = replace(batch, input_ids=batch.input_ids[i:i+1, :end],
                                     attention_mask=batch.attention_mask[i:i+1, :end])
                    unpadded = score(actor, single)[0]
                    mask = batch.response_mask[i, :end-1]
                    evidence.close(label + f"/padding/{i}", unpadded[mask], current[i, :end-1][mask], repeat_budget)
                key = (length, objective)
                if dtype == "float32":
                    batch = replace(batch, old_logprobs=current.masked_fill(~batch.response_mask, 0),
                                    ref_logprobs=ref.masked_fill(~batch.response_mask, 0))
                    baseline[key] = {"batch": batch, "initial": current}
                else:
                    assert batch_hash(batch) == batch_hash(replace(baseline[key]["batch"],
                           old_logprobs=batch.old_logprobs, ref_logprobs=batch.ref_logprobs))
                    batch = baseline[key]["batch"]
                    cross_precision(evidence, label + "/cross_initial", current, baseline[key]["initial"], batch, objective)
                cache_before = batch_hash(batch)
                results = []
                for micro in (1, 2):
                    reset(actor, initial)
                    result = optimize(actor, batch, objective, micro, evidence, label + f"/micro{micro}")
                    after = score(actor, batch)
                    evidence.check(label + f"/micro{micro}/scores_move",
                                   torch.count_nonzero((after-current)[batch.response_mask]) > 0)
                    assert frozen(actor) == frozen_before and frozen(reference) == ref_before, "Frozen tensors changed"
                    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
                    assert batch_hash(batch) == cache_before
                    evidence.write(label + f"/micro{micro}/immutable", base=True, reference=True, cache=True)
                    if micro == 1:
                        if dtype == "float32":
                            baseline[key]["after"] = after
                        else:
                            cross_precision(evidence, label + "/cross_after", after, baseline[key]["after"], batch, objective)
                    if length == "2048":
                        # Warm optimizer already contains moments; measure another actual update.
                        from algorithms.actor import update_actor
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        update_actor(actor, result[3], batch, epochs=1, minibatch_size=len(batch.input_ids),
                                     microbatch_size=micro, objective=objective,
                                     generator=torch.Generator().manual_seed(42))
                        torch.cuda.synchronize()
                        evidence.write(label + f"/micro{micro}/memory_after_optimizer_init",
                                       allocated_peak=torch.cuda.max_memory_allocated(),
                                       reserved_peak=torch.cuda.max_memory_reserved(),
                                       process_memory=subprocess.check_output([
                                           "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"], text=True),
                                       pid=os.getpid(), tokens=2048)
                        assert frozen(actor) == frozen_before and frozen(reference) == ref_before
                    # Drop optimizer GPU states before the other microbatch experiment.
                    del result[3].step
                    results.append(result[:3])
                    del result
                # Memory probes append a third inspected step for long batches; compare it too.
                compare_updates(evidence, label + "/micro_invariance", results[0], results[1], dtype)
                del results
        del actor, reference, initial
        gc.collect()
        torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "dataset-path", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--expected-gpu", choices=("A40",), required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    evidence = Evidence(args.output)
    try:
        evidence.write("start", argv=sys.argv if argv is None else argv, gaps=GAPS, limits=LIMITS,
                       harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       commit=subprocess.check_output(["git", "--no-pager", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       dirty=subprocess.check_output(["git", "--no-pager", "status", "--short"], cwd=ROOT, text=True),
                       args={k: str(v) for k, v in vars(args).items()})
        if not __debug__:
            raise RuntimeError("Do not run with python -O")
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Launch with CUBLAS_WORKSPACE_CONFIG=:4096:8"
        for path in (args.model_path, args.dataset_path):
            assert path.is_dir(), f"Expected local directory: {path}"
        run(args, evidence)
    except BaseException:
        evidence.write("fatal", traceback=traceback.format_exc(), prior_failures=evidence.failures)
        raise
    evidence.write("complete", passed=not evidence.failures, failures=evidence.failures, untested=GAPS)
    return int(bool(evidence.failures))


if __name__ == "__main__":
    sys.exit(main())

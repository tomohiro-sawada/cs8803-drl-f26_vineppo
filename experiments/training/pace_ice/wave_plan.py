"""Generate the pinned RhoMath and OLMo experiment waves for the PACE watcher."""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from time import time

from experiments.training.resume import atomic_record

PACE = "experiments/training/pace_ice"


def generate(args):
    repo, scratch, control = (
        p.resolve(strict=True) for p in (args.repo, args.scratch, args.control)
    )
    pinned = Path.cwd().resolve()
    assert scratch / "worktrees" in pinned.parents
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()
    for required in ("7cddd63", "7e0a2ea"):
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", required, commit], check=True
        )
    logs = repo / "artifacts/training/pace_ice/slurm"
    out = scratch / "training/pace_ice"
    env = {
        "PYTHON": str(repo / ".venv/bin/python"),
        "VLLM_PYTHON": str(repo / ".venv-inference-glibc2.34/bin/python"),
        "TRAINING_CACHE": str(scratch / "training-cache"),
        "PACE_CHAIN_LOG_DIR": str(logs),
        "TMPDIR": "/tmp",
        "WANDB_BASE_URL": "https://api.forge.coreweave.com",
    }
    jobs, archives = [], []
    suffix = args.suffix
    common_gates = [args.chain_smoke / marker for marker in ("COMPLETE", "ROLLOVER")]
    common_gates += [
        scratch / f"handoff-20261120/control-03/{name}"
        for name in ("g1-exact.json", "g2-requests.json")
    ]

    def job(key, command, root, gib, gates, *, entry=None, queue=jobs):
        name = f"vp26-{key}-{suffix}"
        queue.append(
            {
                "name": name,
                "repo": str(pinned),
                "commit": commit,
                "submit": [
                    f"--output={logs}/%x-%j.out",
                    f"--error={logs}/%x-%j.err",
                    *map(str, command),
                ],
                "afterok": ["cpu"],
                "gates": list(map(str, gates)),
                "env": env | ({"TRAINING_ENTRY": entry} if entry else {}),
                "disk_root": str(root),
                "space_gib": gib,
            }
        )
        return name

    def archive(key, root):
        receipt = control / f"{root.name}.pruned.jsonl"
        complete = receipt.with_suffix(".complete.json")
        job(
            "arc-" + key,
            [f"{PACE}/archive_run.slurm", root, receipt],
            complete,
            0.25,
            [root / "COMPLETE"],
            queue=archives,
        )

    def options(method, task, model, *, context="rhomath"):
        dataset = scratch / (
            "cache/datasets/gsm8k" if task == "gsm8k" else "cache/math/datasets/math"
        )
        return [
            "--algorithm",
            method,
            "--task",
            task,
            "--model-protocol",
            context,
            "--model-path",
            model,
            "--dataset-path",
            dataset,
            "--dtype",
            "bfloat16",
            "--finetuning",
            "lora",
            "--lora-rank",
            16,
            "--lora-alpha",
            32,
            "--learning-rate",
            "0.00001",
            "--backend",
            "vllm",
            "--num-workers",
            1,
            "--worker-placement",
            "colocated",
            "--eval-batch-size",
            256,
            "--wandb-mode",
            "online",
            "--wandb-entity",
            "tsawada",
            "--wandb-group",
            "handoff-20261120",
        ]

    def train(
        key,
        method,
        *,
        seed=42,
        steps=200,
        prompts=16,
        batch=32,
        epochs=2,
        period=25,
        resume=5,
        micro=4,
        task="gsm8k",
        olmo=False,
        limit="05:00:00",
        samples=8,
        rounds=4,
        evaluation="validation-test",
        gates=(),
    ):
        root = out / f"handoff-{key}-{suffix}"
        assert not root.exists(), root
        model = args.olmo_model if olmo else scratch / "cache/models/rho-1b-sft-GSM8K"
        keep = 2 if steps == 200 and method != "ppo" else 1
        if args.archive_checkpoints == "no":
            keep = (steps + period - 1) // period
        command = [
            f"--time={limit}",
            "--constraint=gpu-h200" if olmo else "--constraint=gpu-h100",
            f"{PACE}/train_resumable.slurm",
            root,
            *options(method, task, model, context="olmo3-sft" if olmo else "rhomath"),
            "--seed",
            seed,
            "--num-iterations",
            steps,
            "--prompts-per-iteration",
            prompts,
            "--num-samples",
            samples,
            "--minibatch-size",
            batch,
            "--microbatch-size",
            micro,
            "--update-epochs",
            epochs,
            "--mc-num-samples",
            9,
            "--checkpoint-every",
            period,
            "--resume-checkpoint-every",
            resume,
            "--checkpoint-format",
            "compact-lora",
            "--keep-checkpoints",
            keep,
            "--evaluation",
            evaluation,
            "--validation-rounds",
            rounds,
            "--eval-seed",
            42,
            "--lr-schedule",
            "polynomial" if steps == 650 else "constant",
            "--kl-coef",
            0.003
            if method in {"grpo", "rloo"}
            else 0
            if method == "restem"
            else 0.0001,
            "--archive-checkpoints",
            args.archive_checkpoints,
            "--math-grader-workers",
            4 if task == "math" else 1,
            "--wandb-name",
            root.name,
        ]
        gib = 44 if olmo else 32 if method == "ppo" else 10
        job(key, command, root, gib, [*common_gates, *gates])
        archive(key, root)
        return root

    long = {"steps": 650, "prompts": 64, "batch": 64, "limit": "08:00:00"}
    for seed in (42, 43, 44):
        train(f"rho-rloo-200-s{seed}", "rloo", seed=seed)
    rehearsal = train("rho-vineppo-650-s42", "vineppo", **long)
    for seed in (42, *args.extra_long_seed):
        for method in ("vineppo", "rloo", "grpo", "ppo"):
            if (seed, method) == (42, "vineppo"):
                continue
            gates = [rehearsal / "ROLLOVER"]
            if method == "ppo":
                gates.append(args.ppo_smoke / "COMPLETE")
            train(f"rho-{method}-650-s{seed}", method, seed=seed, **long, gates=gates)
    for method in ("grpo", "vineppo", "ppo"):
        for seed in (42, 43, 44):
            gates = [args.ppo_smoke / "COMPLETE"] if method == "ppo" else []
            train(f"rho-{method}-200-s{seed}", method, seed=seed, gates=gates)
    restem_budget = train(
        "restem-budget-check",
        "restem",
        steps=1,
        prompts=320,
        batch=64,
        period=1,
        resume=1,
        epochs=8,
        limit="04:00:00",
        evaluation="disabled",
    )
    train(
        "rho-restem-match-s42",
        "restem",
        steps=10,
        prompts=320,
        batch=64,
        period=1,
        resume=1,
        epochs=8,
        limit="03:00:00",
        gates=[restem_budget / "COMPLETE"],
    )

    olmo_gates = [args.olmo_acceptance / "acceptance.json"]
    for task in ("math", "gsm8k"):
        key = f"olmo3-{task}-sft"
        root = out / f"handoff-{key}-{suffix}"
        job(
            key,
            [
                "--constraint=gpu-h200",
                "--cpus-per-task=12",
                "--mem-per-cpu=8G",
                "--time=04:00:00",
                f"{PACE}/train.slurm",
                "--output",
                root,
                *options("grpo", task, args.olmo_model, context="olmo3-sft"),
                "--wandb-name",
                root.name,
            ],
            root,
            2,
            [*common_gates, *olmo_gates],
            entry="scripts/diagnostics/evaluate_model_baseline.py",
        )
        archive(key, root)
        ready = root
        if task == "math":
            ready = train(
                "olmo3-math-smoke",
                "grpo",
                steps=4,
                prompts=4,
                batch=8,
                samples=4,
                rounds=1,
                period=2,
                resume=1,
                micro=1,
                task=task,
                olmo=True,
                limit="02:00:00",
                gates=[root / "COMPLETE"],
            )
        trained = train(
            f"olmo3-{task}-grpo-50-s42",
            "grpo",
            steps=50,
            period=10,
            resume=2,
            micro=1,
            task=task,
            olmo=True,
            limit="12:00:00",
            gates=[ready / "COMPLETE"],
        )
        olmo_gates.append(trained / "COMPLETE")
    plan = {
        "jobs": archives + jobs,
        "faults": [],
        "cache": env["TRAINING_CACHE"],
        "scratch_mount": "/storage/ice1",
        "deadline": args.deadline,
        "since": datetime.now().isoformat(timespec="seconds"),
    }
    assert time() < args.deadline
    for name in ("wave-plan.json", "watch-state.json"):
        assert not (control / name).exists()
    atomic_record(control / "wave-plan.json", plan)
    atomic_record(
        control / "watch-state.json", {"jobs": {"cpu": args.cpu_job}, "faults": {}}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "repo",
        "scratch",
        "control",
        "chain-smoke",
        "ppo-smoke",
        "olmo-acceptance",
        "olmo-model",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--cpu-job", required=True)
    parser.add_argument("--suffix", default="03")
    parser.add_argument("--archive-checkpoints", choices=("yes", "no"), default="yes")
    parser.add_argument(
        "--extra-long-seed", type=int, choices=(43, 44), action="append", default=[]
    )
    generate(parser.parse_args())

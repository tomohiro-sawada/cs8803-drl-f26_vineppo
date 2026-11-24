"""Summarize PPO runs and select the seed-42 LoRA rank by validation Pass@1.

    python vp-critic-rank-results.py RUN_ROOT [RUN_ROOT ...]
"""

import json
import runpy
import subprocess
import sys
from pathlib import Path

analysis = runpy.run_path(str(Path(__file__).with_name("vp-training-analysis.py")))
extract, read = analysis["extract"], analysis["read"]


def describe(root):
    assert (root / "COMPLETE").is_file(), root
    options = read(root / "chain-plan.json")["options"]
    args = dict(zip(options[::2], options[1::2], strict=True))
    assert args["--algorithm"] == "ppo" and float(args["--kl-coef"]) == 1e-4
    rank = int(args["--critic-lora-rank"]) if "--critic-lora-rank" in args else None
    run, _ = extract(root)
    pointer = read(root / "test-complete.json")
    winner = read(Path(pointer["selection"]))["best"]
    iteration = run["selection_iteration"]
    actor = Path(winner["checkpoint"]["path"])
    if rank is None:
        critic = actor.with_name(f"critic-{iteration}.pt")
        checkpoint_bytes = sum(
            read(path.with_suffix(".archive.json"))["bytes"] for path in (actor, critic)
        )
    else:
        assert run["logged_iterations"] == run["expected_iterations"]
        checkpoint_bytes = actor.stat().st_size
    accounting = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(run["slurm"])]
        + ["-o", "JobIDRaw,ElapsedRaw"],
        text=True,
    )
    elapsed = dict(line.split("|") for line in accounting.splitlines())
    assert elapsed.keys() == set(run["slurm"]), accounting
    n = run["logged_iterations"]
    phases = run["phase_seconds"]
    return {
        "run": root.name, "rank": rank, "seed": run["seed"],
        "model_protocol": args["--model-protocol"], "task": run["task"],
        "iterations": n, "selected_iteration": iteration,
        "validation_pass_at_1": winner["accuracy"],
        "test_pass_at_1": run["test_pass_at_1"],
        "critic_loss": run["training_health_mean"]["critic/value_loss"],
        "mean_iteration_seconds": phases["iteration_seconds"] / n,
        "mean_critic_update_seconds": phases["critic_update_seconds"] / n,
        "mean_actor_update_seconds": phases["actor_update_seconds"] / n,
        "checkpoint_seconds": phases["checkpoint_seconds"],
        "selected_checkpoint_bytes": checkpoint_bytes,
        "gpu_wall_seconds": sum(map(int, elapsed.values())),
        "primary_tokens": run["primary_tokens"],
        "jobs": run["slurm"], "wandb": run["wandb"],
    }


def main(roots):
    results = [describe(Path(root)) for root in roots]
    sweep = [
        run for run in results
        if run["rank"] is not None and run["seed"] == 42
        and run["model_protocol"] == "rhomath"
    ]
    if sweep:
        assert sorted(run["rank"] for run in sweep) == [16, 64, 256]
        best = max(sweep, key=lambda run: (run["validation_pass_at_1"], -run["rank"]))
        print(json.dumps({"selected_rank": best["rank"], "selection": "validation"}))
    for run in results:
        print(json.dumps(run))


if __name__ == "__main__":
    main(sys.argv[1:])

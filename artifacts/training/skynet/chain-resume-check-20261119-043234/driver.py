import json, subprocess, time
from pathlib import Path
root, base = Path('/coc/flash13/tsawada7/Files/projects/cs8803-drl-f26_vineppo'), Path('/coc/flash13/tsawada7/Files/projects/cs8803-drl-f26_vineppo/artifacts/training/skynet/chain-resume-check-20261119-043234')
srun = ["srun", "--jobid=4794852", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=6",
        "--immediate=30", f"--chdir={root}"]
common = [str(root / ".venv/bin/python"), "-u", "experiments/training/run.py", "--task", "gsm8k",
          "--model-path", str(root / ".cache/models/rho-1b-sft-GSM8K"),
          "--dataset-path", str(root / ".cache/datasets/gsm8k"), "--expected-gpu", "A40",
          "--finetuning", "lora", "--dtype", "bfloat16", "--lora-rank", "16", "--lora-alpha", "32",
          "--learning-rate", "1e-5", "--seed", "42", "--wandb-mode", "disabled"]
status = {}
def save():
    (base / "status.json").write_text(json.dumps(status, indent=2) + "\n")
with (base / "rloo-gsm8k.log").open("x") as log:
    status["rloo_gsm8k_exit"] = subprocess.run(
        srun + common + ["--mode", "benchmark", "--algorithm", "rloo", "--num-samples", "8",
                         "--prompts-per-iteration", "16", "--minibatch-size", "64", "--microbatch-size", "4",
                         "--update-epochs", "2", "--num-iterations", "2", "--warmup-iterations", "1",
                         "--output", str(base / "rloo-gsm8k")], stdout=log, stderr=subprocess.STDOUT).returncode
save()
selection = ["--mode", "train", "--algorithm", "grpo", "--num-samples", "4", "--prompts-per-iteration", "2",
             "--minibatch-size", "2", "--microbatch-size", "1", "--num-iterations", "4",
             "--checkpoint-every", "1", "--keep-checkpoints", "2", "--evaluation", "validation",
             "--validation-rounds", "1", "--eval-batch-size", "32", "--eval-timeout-seconds", "1800"]
def attempt(index, resume, stop_after):
    """Kill once `stop_after` candidates exist and the next checkpoint is saved but unvalidated."""
    output = base / f"attempt-{index:02d}"
    argv = srun + common + selection + ["--output", str(output)]
    if resume is not None:
        argv += ["--resume", str(resume), "--resume-selection", str(resume.parent.parent / "selection.json")]
    with (base / f"attempt-{index:02d}.log").open("x") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        target = output / f"checkpoints/iteration-{stop_after + 1}.pt"
        while process.poll() is None:
            manifest = output / "selection.json"
            done = len(json.loads(manifest.read_text())["candidates"]) if manifest.exists() else 0
            if stop_after is not None and done == stop_after and target.exists():
                process.terminate()
                break
            time.sleep(1)
        status[f"attempt_{index}_exit"] = process.wait()
    save()
    time.sleep(20)
    return target
first = attempt(0, None, 1)
second = attempt(1, first, 2)
attempt(2, second, None)
status["finished"] = True
save()

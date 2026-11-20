import json, subprocess, time
from pathlib import Path
root, base = Path('/coc/flash13/tsawada7/Files/projects/cs8803-drl-f26_vineppo'), Path('/coc/flash13/tsawada7/Files/projects/cs8803-drl-f26_vineppo/artifacts/training/skynet/selection-resume-check-20261119-034544')
srun = ["srun", "--jobid=4794852", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=6",
        "--immediate=30", f"--chdir={root}"]
python = str(root / ".venv/bin/python")
def train(task, model, *extra):
    return [python, "-u", "experiments/training/run.py", "--task", task,
            "--model-path", str(root / ".cache/models" / model),
            "--dataset-path", str(root / ".cache/datasets" / task), "--expected-gpu", "A40",
            "--finetuning", "lora", "--dtype", "bfloat16", "--lora-rank", "16", "--lora-alpha", "32",
            "--learning-rate", "1e-5", "--seed", "42", "--wandb-mode", "disabled", *extra]
selection = ["--mode", "train", "--algorithm", "grpo", "--num-samples", "4", "--prompts-per-iteration", "2",
             "--minibatch-size", "2", "--microbatch-size", "1", "--num-iterations", "3",
             "--checkpoint-every", "1", "--keep-checkpoints", "2", "--evaluation", "validation",
             "--validation-rounds", "1", "--eval-batch-size", "32", "--eval-timeout-seconds", "1800"]
a, b = base / "interrupted", base / "resumed"
status = {}
def save():
    (base / "status.json").write_text(json.dumps(status, indent=2) + "\n")
with (base / "interrupted.log").open("x") as log:
    process = subprocess.Popen(srun + train("gsm8k", "rho-1b-sft-GSM8K", *selection, "--output", str(a)),
                               stdout=log, stderr=subprocess.STDOUT)
    # Interrupt once iteration 1 is validated and iteration 2 is saved but not yet
    # validated: the resumed run must then validate iteration 2 before any update.
    while process.poll() is None:
        manifest = a / "selection.json"
        done = len(json.loads(manifest.read_text())["candidates"]) if manifest.exists() else 0
        if done >= 1 and (a / "checkpoints/iteration-2.pt").exists():
            status["interrupted_with_candidates"] = done
            process.terminate()
            break
        time.sleep(1)
    status["interrupted_exit"] = process.wait()
save()
time.sleep(20)
status["gpu_processes_after_interrupt"] = subprocess.run(
    srun + ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip()
save()
if status.get("interrupted_with_candidates") == 1 and not status["gpu_processes_after_interrupt"]:
    with (base / "resumed.log").open("x") as log:
        status["resumed_exit"] = subprocess.run(
            srun + train("gsm8k", "rho-1b-sft-GSM8K", *selection, "--output", str(b), "--resume",
                         str(a / "checkpoints/iteration-2.pt"), "--resume-selection", str(a / "selection.json")),
            stdout=log, stderr=subprocess.STDOUT).returncode
    save()
with (base / "rloo-math.log").open("x") as log:
    status["rloo_math_exit"] = subprocess.run(
        srun + train("math", "rho-1b-sft-MATH", "--mode", "benchmark", "--algorithm", "rloo",
                     "--num-samples", "8", "--prompts-per-iteration", "8", "--minibatch-size", "16",
                     "--microbatch-size", "4", "--update-epochs", "2", "--num-iterations", "2",
                     "--warmup-iterations", "1", "--output", str(base / "rloo-math")),
        stdout=log, stderr=subprocess.STDOUT).returncode
status["finished"] = True
save()

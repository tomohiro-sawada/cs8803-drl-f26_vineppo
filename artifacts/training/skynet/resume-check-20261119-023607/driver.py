import json, subprocess
from pathlib import Path
root, base = Path('/coc/flash13/tsawada7/Files/projects/cs8803-drl-f26_vineppo'), Path('/coc/flash13/tsawada7/Files/projects/cs8803-drl-f26_vineppo/artifacts/training/skynet/resume-check-20261119-023607')
srun = ["srun", "--jobid=4794852", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=6",
        "--immediate=30", f"--chdir={root}"]
python = str(root / ".venv/bin/python")
common = [python, "-u", "experiments/training/run.py", "--mode", "train", "--algorithm", "grpo",
          "--task", "gsm8k", "--model-path", str(root / ".cache/models/rho-1b-sft-GSM8K"),
          "--dataset-path", str(root / ".cache/datasets/gsm8k"), "--expected-gpu", "A40",
          "--finetuning", "lora", "--dtype", "bfloat16", "--lora-rank", "16", "--lora-alpha", "32",
          "--learning-rate", "1e-5", "--num-samples", "4", "--prompts-per-iteration", "2",
          "--minibatch-size", "2", "--microbatch-size", "1", "--num-iterations", "2",
          "--checkpoint-every", "1", "--keep-checkpoints", "2", "--seed", "42",
          "--wandb-mode", "disabled"]
full, resumed = base / "full", base / "resumed"
steps = [
    ("idle", ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"]),
    ("full", common + ["--output", str(full)]),
    ("resumed", common + ["--output", str(resumed), "--resume", str(full / "checkpoints/iteration-1.pt")]),
    ("compare", [python, str(base / "compare.py"), str(full / "checkpoints/iteration-2.pt"),
                 str(resumed / "checkpoints/iteration-2.pt")]),
]
status = {}
for name, argv in steps:
    with (base / f"{name}.log").open("x") as log:
        code = subprocess.run(srun + argv, stdout=log, stderr=subprocess.STDOUT).returncode
    status[name] = code
    (base / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    if code or (name == "idle" and (base / "idle.log").read_text().strip()):
        status["stopped_at"] = name
        (base / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        break
else:
    status["finished"] = True
    (base / "status.json").write_text(json.dumps(status, indent=2) + "\n")

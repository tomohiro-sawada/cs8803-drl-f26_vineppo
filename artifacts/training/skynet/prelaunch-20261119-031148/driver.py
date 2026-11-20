import json, subprocess
from pathlib import Path
root, base = Path('/coc/flash13/tsawada7/Files/projects/cs8803-drl-f26_vineppo'), Path('/coc/flash13/tsawada7/Files/projects/cs8803-drl-f26_vineppo/artifacts/training/skynet/prelaunch-20261119-031148')
srun = ["srun", "--jobid=4794852", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=6",
        "--immediate=30", f"--chdir={root}"]
common = [str(root / ".venv/bin/python"), "-u", "experiments/training/run.py", "--mode", "benchmark",
          "--task", "gsm8k", "--model-path", str(root / ".cache/models/rho-1b-sft-GSM8K"),
          "--dataset-path", str(root / ".cache/datasets/gsm8k"), "--expected-gpu", "A40",
          "--finetuning", "lora", "--dtype", "bfloat16", "--lora-rank", "16", "--lora-alpha", "32",
          "--learning-rate", "1e-5", "--seed", "42", "--wandb-mode", "disabled",
          "--num-iterations", "2", "--warmup-iterations", "1", "--update-epochs", "2"]
steps = [
    ("vineppo-smoke", common + ["--algorithm", "vineppo", "--prompts-per-iteration", "4", "--num-samples", "4",
                                "--mc-num-samples", "9", "--minibatch-size", "8", "--microbatch-size", "4",
                                "--output", str(base / "vineppo-smoke")]),
    ("grpo-scale", common + ["--algorithm", "grpo", "--prompts-per-iteration", "64", "--num-samples", "8",
                             "--minibatch-size", "64", "--microbatch-size", "4",
                             "--output", str(base / "grpo-scale")]),
]
status = {}
for name, argv in steps:
    with (base / f"{name}.log").open("x") as log:
        status[name] = subprocess.run(srun + argv, stdout=log, stderr=subprocess.STDOUT).returncode
    (base / "status.json").write_text(json.dumps(status, indent=2) + "\n")
status["finished"] = True
(base / "status.json").write_text(json.dumps(status, indent=2) + "\n")

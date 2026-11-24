from pathlib import Path
import json
import os
import shlex
import subprocess

repo = Path.cwd().resolve()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
pinned = scratch / "worktrees/8726b44"
logs = repo / "artifacts/training/pace_ice/slurm"
home_scratch = Path("/home/hice1/tsawada7/scratch/cs8803-drl-f26_vineppo")
control = scratch / "handoff-20261120/control-06"
old_control = control.with_name("control-05")
chain = home_scratch / "training/pace_ice/handoff-production-chain-smoke-02"
ppo = home_scratch / "training/pace_ice/handoff-smoke-ppo-05"
python = repo / ".venv/bin/python"
env = os.environ | {
    "PYTHON": str(python),
    "VLLM_PYTHON": str(repo / ".venv-inference-glibc2.34/bin/python"),
    "TRAINING_CACHE": str(scratch / "training-cache"),
    "PACE_CHAIN_LOG_DIR": str(logs),
    "TMPDIR": "/tmp",
}
env = {key: v for key, v in env.items() if not key.startswith(("SLURM_", "SBATCH_"))}
out = [f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err"]
cpu_options = ["--partition=ice-cpu", "--account=math", "--qos=coc-ice", *out]
def run(*args):
    return subprocess.check_output(args, cwd=pinned, env=env, text=True).strip()
def submit(*args):
    job = run("sbatch", "--parsable", *args).split(";")[0]
    assert job.isdecimal()
    return job

assert run("git", "rev-parse", "--short", "HEAD") == "8726b44"
assert not run("git", "status", "--porcelain")
old_state = json.loads((old_control / "watch-state.json").read_text())
assert set(old_state["jobs"]) == {"cpu"}, old_state
queued = dict(row.split("|") for row in run("squeue", "--me", "--noheader", "--format=%i|%T").splitlines())
assert all(queued[job] == "PENDING" for job in ("7225325", "7225326", "7225327", "7225328", "7225420"))
quota, = [line.split() for line in run("lfs", "quota", "-u", os.environ["USER"], "/storage/ice1").splitlines() if line.split()[:1] == ["/storage/ice1"]]
assert (int(quota[3]) - int(quota[1].rstrip("*"))) / 1024**2 >= 20 + 10 + 32 + 2
run("scancel", "7225325", "7225326", "7225327", "7225328", "7225329", "7225420")
control.mkdir()
cpu = submit(*cpu_options, "--cpus-per-task=4", "--mem=16G", "--time=00:30:00",
             "--job-name=vp26-production-cpu-v2",
             "--wrap=" + shlex.join(["env", f"PYTHONPATH={pinned}/src:{pinned}", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", str(python), "-m", "pytest", "-q", "--tb=short"]))
print("cpu", cpu, flush=True)
gate = [f"--dependency=afterok:{cpu}", "--kill-on-invalid-dep=yes"]
train = ["--constraint=gpu-h100", "--time=00:45:00", *out, *gate]
script = "experiments/training/pace_ice/train_resumable.slurm"
common = ["--task", "gsm8k", "--model-path", str(home_scratch / "cache/models/rho-1b-sft-GSM8K"),
          "--dataset-path", str(home_scratch / "cache/datasets/gsm8k"), "--seed", "42",
          "--microbatch-size", "4", "--update-epochs", "2", "--dtype", "bfloat16", "--finetuning", "lora",
          "--backend", "vllm", "--num-workers", "1", "--worker-placement", "colocated",
          "--checkpoint-format", "compact-lora", "--resume-checkpoint-every", "1", "--keep-checkpoints", "1",
          "--archive-checkpoints", "yes", "--evaluation", "validation-test", "--validation-rounds", "1",
          "--eval-batch-size", "256", "--wandb-group", "handoff-20261120"]
grpo_id = submit(*train, "--job-name=vp26-production-chain-smoke-v2", script, str(chain), *common,
                 "--algorithm", "grpo", "--kl-coef", "0.003", "--checkpoint-every", "4", "--num-iterations", "8",
                 "--prompts-per-iteration", "16", "--num-samples", "8", "--minibatch-size", "32", "--wandb-mode", "disabled")
print("grpo", grpo_id, flush=True)
ppo_id = submit(*train, "--job-name=vp26-ppo-storage-smoke-v3", script, str(ppo), *common,
                "--algorithm", "ppo", "--kl-coef", "0.0001", "--checkpoint-every", "2", "--num-iterations", "4",
                "--prompts-per-iteration", "4", "--num-samples", "4", "--minibatch-size", "8",
                "--wandb-mode", "online", "--wandb-name", ppo.name)
print("ppo", ppo_id, flush=True)
for jid, root in ((grpo_id, chain), (ppo_id, ppo)):
    status = shlex.quote(str(root / "status.json"))
    test = shlex.quote(f'.iteration >= 1 and .job == "{jid}" and .state == "running"')
    command = f"set -eu\nwhile test ! -f {status} || ! jq -e {test} {status} >/dev/null; do sleep 1; done\nscancel --batch --signal=USR1 {jid}\n"
    print("injector", submit(*cpu_options, f"--dependency=after:{jid}", "--kill-on-invalid-dep=yes",
                             "--cpus-per-task=1", "--mem=1G", "--time=00:16:00", f"--job-name=vp26-signal-{jid}",
                             "--wrap=" + command), flush=True)
deadline = json.loads((old_control / "wave-plan.json").read_text())["deadline"]
run(str(python), "-m", "experiments.training.pace_ice.wave_plan", "--repo", str(repo), "--scratch", str(scratch),
    "--control", str(control), "--chain-smoke", str(chain), "--ppo-smoke", str(ppo),
    "--olmo-acceptance", str(home_scratch / "training/pace_ice/olmo3-sft-sampler-acceptance-06"),
    "--olmo-model", str(scratch / "cache/math/models/Olmo-3-7B-Instruct-SFT-e1452fc"),
    "--deadline", str(deadline), "--cpu-job", cpu)
print("watcher", submit(*gate, f"--output={control}/controller-%j.log", f"--error={control}/controller-%j.err",
                        "experiments/training/pace_ice/watch_resumable.slurm", str(control / "wave-plan.json")), flush=True)

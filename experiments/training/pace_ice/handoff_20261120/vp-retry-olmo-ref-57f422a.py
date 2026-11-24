import json
import os
from pathlib import Path
import subprocess
import time

repo = Path.cwd()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
out = scratch / "training/pace_ice"
pinned = scratch / "worktrees/57f422a"
old_control = scratch / "handoff-20261120/control-15"
control = old_control.with_name("control-16")
logs = repo / "artifacts/training/pace_ice/slurm"
python = repo / ".venv/bin/python"
env = {
    **{k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))},
    "PYTHON": str(python),
    "VLLM_PYTHON": str(repo / ".venv-inference-glibc2.34/bin/python"),
    "TRAINING_CACHE": str(scratch / "training-cache"),
    "PACE_CHAIN_LOG_DIR": str(logs),
    "TMPDIR": "/tmp",
}

def run(*args):
    return subprocess.check_output(args, cwd=pinned, env=env, text=True).strip()

def read(path):
    return json.loads(path.read_text())

def write(name, data):
    path = control / name
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    pending.rename(path)

assert run("git", "rev-parse", "--short", "HEAD") == pinned.name
assert not run("git", "status", "--porcelain")
model = scratch / "cache/math/models/Olmo-3-7B-Instruct-SFT-e1452fc"
old_plan = read(old_control / "wave-plan.json")
control.mkdir()
run(str(python), "-m", "experiments.training.pace_ice.wave_plan", "--repo", str(repo), "--scratch", str(scratch),
    "--control", str(control), "--chain-smoke", str(out / "handoff-production-chain-smoke-03"),
    "--ppo-smoke", str(out / "handoff-smoke-ppo-06"),
    "--olmo-acceptance", str(out / "olmo3-sft-sampler-acceptance-06"), "--olmo-model", str(model),
    "--deadline", str(old_plan["deadline"]), "--cpu-job", "7233459", "--suffix", "10", "--archive-checkpoints", "no")
new_plan = read(control / "wave-plan.json")
replacement = {
    j["name"].removesuffix("-10"): j | {
        "afterok": ["olmo-reference-cpu"],
        "gates": [g.replace("handoff-olmo3-math-smoke-10/", "handoff-olmo3-math-smoke-08/") for g in j["gates"]],
    }
    for j in new_plan["jobs"] if "olmo3-gsm8k-" in j["name"] or "olmo3-math-grpo-50-" in j["name"]
}
assert len(replacement) == 8
old_name = "vp26-olmo3-math-grpo-50-s42-09"
failed_root = out / "handoff-olmo3-math-grpo-50-s42-09"
assert not (failed_root / "COMPLETE").is_file()
failed = read(failed_root / "status.json")
assert failed["iteration"] == 0 and failed["state"] in ("failed", "halted")
assert not list(failed_root.glob("attempt-*/checkpoints/*.pt"))
queue = set(run("squeue", "--me", "-h", "-o", "%i").splitlines())
old_watchers = {"7232995", "7233002"}
assert old_watchers <= queue
to_cancel = old_watchers | ({"7233190"} & queue)
run("scancel", *sorted(to_cancel))
for _ in range(30):
    if not set(run("squeue", "--me", "-h", "-o", "%i").splitlines()) & to_cancel:
        break
    time.sleep(1)
else:
    raise RuntimeError("Superseded allocations did not stop")
state = read(old_control / "watch-state.json")
assert state["jobs"].pop(old_name) == "7233011"
assert not (replacement.keys() & {i.removesuffix("-09") for i in state["jobs"]})
state["jobs"]["olmo-reference-cpu"] = "7233459"
old_plan["jobs"] = [replacement.get(j["name"].removesuffix("-09"), j) for j in old_plan["jobs"]]
deferred = old_plan["deferred_archives"]
deferred["jobs"] = [replacement.get(j["name"].removesuffix("-09"), j) for j in deferred["jobs"]]
deferred["recovery_sources"]["vp26-olmo3-math-grpo-50-s42-09"] = {
    "source": str(failed_root), "job": "7233011", "iteration": 1, "reason": "malformed gold training solution before the first checkpoint",
}
write("wave-plan.json", old_plan)
write("watch-state.json", state)
quota = run("lfs", "quota", "-u", os.environ["USER"], "/storage/ice1").splitlines()[-1].split()
assert quota[0] == "/storage/ice1" and (int(quota[3]) - int(quota[1].rstrip("*"))) / 1024**2 >= 20.25
job = run("sbatch", "--parsable", f"--output={control}/controller-%j.log", f"--error={control}/controller-%j.err",
          "experiments/training/pace_ice/watch_resumable.slurm", str(control / "wave-plan.json")).split(";")[0]
assert job.isdecimal()
print("watcher", job, "previous...", failed["state"], flush=True)

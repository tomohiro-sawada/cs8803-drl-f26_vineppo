from pathlib import Path
import json
import os
import shlex
import subprocess
import time

repo = Path.cwd().resolve()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
pinned = scratch / "worktrees/6c839a7"
logs = repo / "artifacts/training/pace_ice/slurm"
old_control = scratch / "handoff-20261120/control-09"
control = old_control.with_name("control-10")
model = scratch / "cache/math/models/Olmo-3-7B-Instruct-SFT-e1452fc"
out = scratch / "training/pace_ice"
python = repo / ".venv/bin/python"
env = os.environ | {
    "PYTHON": str(python),
    "VLLM_PYTHON": str(repo / ".venv-inference-glibc2.34/bin/python"),
    "TRAINING_CACHE": str(scratch / "training-cache"),
    "PACE_CHAIN_LOG_DIR": str(logs),
    "TMPDIR": "/tmp",
}
env = {key: v for key, v in env.items() if not key.startswith(("SLURM_", "SBATCH_"))}

def run(*args):
    return subprocess.check_output(args, cwd=pinned, env=env, text=True).strip()

def submit(*args):
    job = run("sbatch", "--parsable", *args).split(";")[0]
    assert job.isdecimal()
    return job

def read(directory, name):
    return json.loads((directory / name).read_text())

def write(name, data):
    target = control / name
    pending = target.with_suffix(".pending.json")
    pending.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    pending.rename(target)

assert run("git", "rev-parse", "--short", "HEAD") == pinned.name
assert not run("git", "status", "--porcelain")
assert (out / "handoff-olmo3-math-smoke-04/HALTED").is_file()
assert (out / "handoff-olmo3-math-sft-04/COMPLETE").is_file()
quota, = [line.split() for line in run("lfs", "quota", "-u", os.environ["USER"], "/storage/ice1").splitlines() if line.split()[:1] == ["/storage/ice1"]]
assert (int(quota[3]) - int(quota[1].rstrip("*"))) / 1024**2 > 20
pream = shlex.join(["env", f"PYTHONPATH={pinned}/src:{pinned}", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", str(python)])
cpu = submit("--partition=ice-cpu", "--account=math", "--qos=coc-ice",
             "--cpus-per-task=4", "--mem=16G", "--time=00:30:00",
             "--job-name=vp26-olmo-verifier-cpu", f"--output={logs}/%x-%j.out",
             f"--error={logs}/%x-%j.err", "--wrap=" + pream + " -m pytest -q --tb=short")
print("cpu", cpu, flush=True)
old_plan = read(old_control, "wave-plan.json")
control.mkdir()
run(str(python), "-m", "experiments.training.pace_ice.wave_plan", "--repo", str(repo), "--scratch", str(scratch),
    "--control", str(control), "--chain-smoke", str(out / "handoff-production-chain-smoke-03"),
    "--ppo-smoke", str(out / "handoff-smoke-ppo-06"),
    "--olmo-acceptance", str(out / "olmo3-sft-sampler-acceptance-06"), "--olmo-model", str(model),
    "--deadline", str(old_plan["deadline"]), "--cpu-job", cpu, "--suffix", "05")
new = read(control, "wave-plan.json")
updates = {
    j["name"].removesuffix("-05"): j | {
        "afterok": ["olmo-retry-cpu"],
        "gates": [g.replace("handoff-olmo3-math-sft-05/", "handoff-olmo3-math-sft-04/") for g in j["gates"]],
    }
    for j in new["jobs"] if "olmo3-" in j["name"] and "olmo3-math-sft-" not in j["name"]
}
assert len(updates) == 10
assert {j["name"].removesuffix("-04") for j in old_plan["jobs"] if "olmo3-" in j["name"] and "olmo3-math-sft-" not in j["name"]} == updates.keys()
old_watchers = {"7229086", "7229094"}
queue = set(run("squeue", "--me", "-h", "-o", "%i").splitlines())
assert old_watchers <= queue
run("scancel", *sorted(old_watchers))
for _ in range(30):
    if not (set(run("squeue", "--me", "-h", "-o", "%i").splitlines()) & old_watchers):
        break
    time.sleep(1)
else:
    raise RuntimeError("Old watcher did not stop")
state = read(old_control, "watch-state.json")
assert state["jobs"].pop("vp26-olmo3-math-smoke-04") == "7230728"
assert not (updates.keys() & {name.removesuffix("-04") for name in state["jobs"]})
state["jobs"]["olmo-retry-cpu"] = cpu
old_plan["jobs"] = [updates.get(j["name"].removesuffix("-04"), j) for j in old_plan["jobs"]]
write("wave-plan.json", old_plan)
write("watch-state.json", state)
jid = submit(f"--output={control}/controller-%j.log", f"--error={control}/controller-%j.err",
             "experiments/training/pace_ice/watch_resumable.slurm", str(control / "wave-plan.json"))
print("watcher", jid, flush=True)

from pathlib import Path
import json
import os
import subprocess
import time

repo = Path.cwd().resolve()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
pinned = scratch / "worktrees/d15e1c7"
old_control = scratch / "handoff-20261120/control-12"
control = old_control.with_name("control-13")
out = scratch / "training/pace_ice"
model = scratch / "cache/math/models/Olmo-3-7B-Instruct-SFT-e1452fc"
source = out / "handoff-olmo3-math-smoke-07/attempt-0000/checkpoints/iteration-2.pt"
python = repo / ".venv/bin/python"
env = os.environ | {
    "PYTHON": str(python),
    "VLLM_PYTHON": str(repo / ".venv-inference-glibc2.34/bin/python"),
    "TRAINING_CACHE": str(scratch / "training-cache"),
    "PACE_CHAIN_LOG_DIR": str(repo / "artifacts/training/pace_ice/slurm"),
    "TMPDIR": "/tmp",
}
env = {key: v for key, v in env.items() if not key.startswith(("SLURM_", "SBATCH_"))}
def run(*args):
    return subprocess.check_output(args, cwd=pinned, env=env, text=True).strip()
def read(p):
    return json.loads(p.read_text())
def write(name, data):
    path = control / name
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    pending.rename(path)

assert run("git", "rev-parse", "--short", "HEAD") == pinned.name
assert not run("git", "status", "--porcelain")
assert source.is_file() and (source.parents[2] / "HALTED").is_file()
old_plan = read(old_control / "wave-plan.json")
control.mkdir()
run(str(python), "-m", "experiments.training.pace_ice.wave_plan", "--repo", str(repo), "--scratch", str(scratch),
    "--control", str(control), "--chain-smoke", str(out / "handoff-production-chain-smoke-03"),
    "--ppo-smoke", str(out / "handoff-smoke-ppo-06"),
    "--olmo-acceptance", str(out / "olmo3-sft-sampler-acceptance-06"), "--olmo-model", str(model),
    "--deadline", str(old_plan["deadline"]), "--cpu-job", "7232029", "--suffix", "08", "--archive-checkpoints", "no")
new = read(control / "wave-plan.json")
replacement = {
    j["name"].removesuffix("-08"): j | {
        "afterok": ["olmo-actor-cpu"], "gates": [g.replace("handoff-olmo3-math-sft-08/", "handoff-olmo3-math-sft-04/") for g in j["gates"]],
    }
    for j in new["jobs"] if "olmo3-" in j["name"] and "olmo3-math-sft-" not in j["name"]
}
assert len(replacement) == 10
replacement["vp26-olmo3-math-smoke"]["submit"] += ["--resume", str(source)]
old_watchers = {"7231703", "7231704"}
queue = set(run("squeue", "--me", "-h", "-o", "%i").splitlines())
assert old_watchers <= queue
run("scancel", *sorted(old_watchers))
for _ in range(30):
    if not (set(run("squeue", "--me", "-h", "-o", "%i").splitlines()) & old_watchers):
        break
    time.sleep(1)
else:
    raise RuntimeError("Old watcher did not stop")
state = read(old_control / "watch-state.json")
assert state["jobs"].pop("vp26-olmo3-math-smoke-07") == "7231707"
assert not (replacement.keys() & {i.removesuffix("-07") for i in state["jobs"]})
state["jobs"]["olmo-actor-cpu"] = "7232029"
old_plan["jobs"] = [replacement.get(j["name"].removesuffix("-07"), j) for j in old_plan["jobs"]]
deferred = old_plan["deferred_archives"]
deferred["jobs"] = [replacement.get(j["name"].removesuffix("-07"), j) for j in deferred["jobs"]]
deferred["recovery_sources"]["vp26-olmo3-math-smoke"] = {
    "source": str(source.parents[2]), "checkpoint": str(source), "iteration": 2,
}
write("wave-plan.json", old_plan)
write("watch-state.json", state)
jid = run("sbatch", "--parsable", f"--output={control}/controller-%j.log", f"--error={control}/controller-%j.err",
          "experiments/training/pace_ice/watch_resumable.slurm", str(control / "wave-plan.json")).split(";")[0]
assert jid.isdecimal()
print("watcher", jid)

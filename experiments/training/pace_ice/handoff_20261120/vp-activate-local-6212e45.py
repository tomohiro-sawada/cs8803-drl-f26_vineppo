from pathlib import Path
import json
import os
import subprocess
import time

repo = Path.cwd().resolve()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
pinned = scratch / "worktrees/6212e45"
old_control = scratch / "handoff-20261120/control-11"
control = old_control.with_name("control-12")
out = scratch / "training/pace_ice"
model = scratch / "cache/math/models/Olmo-3-7B-Instruct-SFT-e1452fc"
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

def read(directory, name):
    return json.loads((directory / name).read_text())

def write(name, data):
    target = control / name
    pending = target.with_suffix(".pending.json")
    pending.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    pending.rename(target)

def queue():
    return dict(line.split("|") for line in run("squeue", "--me", "-h", "-o", "%i|%T").splitlines())

assert run("git", "rev-parse", "--short", "HEAD") == "6212e45"
assert not run("git", "status", "--porcelain")
old_plan = read(old_control, "wave-plan.json")
planned = read(control, "wave-plan.json")
new = {j["name"].removesuffix("-07"): j for j in planned["jobs"]}
rho = {f"vp26-rho-{arm}-650-s42" for arm in ("grpo", "rloo", "ppo")}
recover = {f"vp26-rho-ppo-200-s{seed}": out / f"handoff-rho-ppo-200-s{seed}-03" for seed in (43, 44)}
olmo = {key for key in new if "olmo3-" in key and "olmo3-math-sft" not in key and not key.startswith("vp26-arc-")}
keys = rho | recover.keys() | olmo
assert len(keys) == 10
blocked = []
for name in keys:
    j = new[name]
    j["afterok"] = ["storage-cpu"]
    j["gates"] = [
        p.replace("handoff-olmo3-math-sft-07/", "handoff-olmo3-math-sft-04/")
         .replace("handoff-rho-vineppo-650-s42-07/", "handoff-rho-vineppo-650-s42-03/")
        for p in j["gates"]
    ]
    blocked.append(new[name.replace("vp26-", "vp26-arc-", 1)])
assert not (queue().keys() & {"7231461", "7231470", "7229238", "7231654", "7231492"})
sources = {}
for name, root in recover.items():
    status = read(root, "status.json")
    expected = 200 if name.endswith("43") else 175
    assert status["iteration"] == expected
    matches = sorted(root.glob(f"attempt-*/checkpoints/iteration-{expected}.pt"))
    assert len(matches) == 1, matches
    source = matches[0]
    assert source.with_name(f"critic-{expected}.pt").is_file()
    selection = read(source.parent.parent, "selection.json")
    assert selection["candidates"][-1]["checkpoint"]["path"] == str(source)
    new[name]["submit"] += ["--resume", str(source)]
    new[name]["submit"][2] = "--time=00:45:00" if expected == 200 else "--time=01:15:00"
    sources[name] = {"source": str(root), "checkpoint": str(source), "iteration": expected}
state = read(old_control, "watch-state.json")
assert state["jobs"].pop("vp26-rho-ppo-200-s43-03") == "7226940"
assert state["jobs"].pop("vp26-rho-ppo-200-s44-03") == "7227708"
assert state["jobs"].pop("vp26-olmo3-math-smoke-06") == "7231492"
state["jobs"]["storage-cpu"] = "7231649"
updated = []
replaced = set()
for job in old_plan["jobs"]:
    jobkey = job["name"].rsplit("-", 1)[0]
    if jobkey in keys:
        assert job["name"] not in state["jobs"], job["name"]
        updated.append(new[jobkey])
        replaced.add(jobkey)
    elif jobkey.replace("vp26-arc-", "vp26-", 1) not in keys:
        updated.append(job)
    else:
        assert job["name"] not in state["jobs"], job["name"]
assert replaced == keys
old_plan["jobs"] = updated
old_plan["deferred_archives"] = {"reason": "Hugging Face write-quota 403", "jobs": blocked, "recovery_sources": sources}
write("wave-plan.json", old_plan)
write("watch-state.json", state)
jid = run("sbatch", "--parsable", f"--output={control}/controller-%j.log", f"--error={control}/controller-%j.err",
          "experiments/training/pace_ice/watch_resumable.slurm", str(control / "wave-plan.json")).split(";")[0]
assert jid.isdecimal()
print(json.dumps({"watcher": jid, "deferred_archives": len(blocked), "recovery_sources": sources}, sort_keys=True))

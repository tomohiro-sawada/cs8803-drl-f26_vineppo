from pathlib import Path
import json
import os
import subprocess
import time

repo = Path.cwd().resolve()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
pinned = scratch / "worktrees/2d27896"
old_control = scratch / "handoff-20261120/control-08"
control = old_control.with_name("control-09")
model = scratch / "cache/math/models/Olmo-3-7B-Instruct-SFT-e1452fc"
out = scratch / "training/pace_ice"
python = repo / ".venv/bin/python"
env = os.environ | {
    "PYTHON": str(python),
    "VLLM_PYTHON": str(repo / ".venv-inference-glibc2.34/bin/python"),
    "TRAINING_CACHE": str(scratch / "training-cache"),
    "PACE_CHAIN_LOG_DIR": str(repo / "artifacts/training/pace_ice/slurm"),
    "TMPDIR": "/tmp",
}
env = {key: value for key, value in env.items() if not key.startswith(("SLURM_", "SBATCH_"))}

def run(*args):
    return subprocess.check_output(args, cwd=pinned, env=env, text=True).strip()

def read(directory, name):
    return json.loads((directory / name).read_text())

def write(name, data):
    target = control / name
    pending = target.with_suffix(".pending.json")
    pending.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    pending.rename(target)

assert run("git", "rev-parse", "--short", "HEAD") == "2d27896"
assert not run("git", "status", "--porcelain")
old_plan = read(old_control, "wave-plan.json")
control.mkdir()
run(str(python), "-m", "experiments.training.pace_ice.wave_plan", "--repo", str(repo), "--scratch", str(scratch),
    "--control", str(control), "--chain-smoke", str(out / "handoff-production-chain-smoke-03"),
    "--ppo-smoke", str(out / "handoff-smoke-ppo-06"),
    "--olmo-acceptance", str(out / "olmo3-sft-sampler-acceptance-06"), "--olmo-model", str(model),
    "--deadline", str(old_plan["deadline"]), "--cpu-job", "7225658", "--suffix", "05",
    "--extra-long-seed=43", "--extra-long-seed=44")
proposed = read(control, "wave-plan.json")
expect_extra = {f"vp26-rho-{method}-650-s{seed}-05" for method in ("vineppo", "rloo", "grpo", "ppo") for seed in (43, 44)}
assert {j["name"] for j in proposed["jobs"] if "-650-s4" in j["name"] and "s42" not in j["name"] and "-arc-" not in j["name"]} == expect_extra
assert len(proposed["jobs"]) == len(old_plan["jobs"]) + 16
starts = {f"vp26-rho-{m}-650-s42" for m in ("vineppo", "rloo", "grpo", "ppo")} | {"vp26-rho-restem-match-s42"}
limits = {j["name"].removesuffix("-05"): next(arg for arg in j["submit"] if arg.startswith("--time=")) for j in proposed["jobs"] if j["name"].removesuffix("-05") in starts}
assert limits.keys() == starts
assert set(limits.values()) == {"--time=08:00:00", "--time=03:00:00"}
assert limits["vp26-rho-restem-match-s42"] == "--time=03:00:00"
assert all(limits[key] == "--time=08:00:00" for key in starts if "650" in key)
old_watchers = {"7228434", "7228439"}
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
assert not any(name in state["jobs"] for name in ("vp26-rho-rloo-650-s42-03", "vp26-rho-grpo-650-s42-03", "vp26-rho-ppo-650-s42-03"))
for job in old_plan["jobs"]:
    key = job["name"].removesuffix("-03")
    if key in limits:
        assert sum(arg.startswith("--time=") for arg in job["submit"]) == 1
        job["submit"] = [limits[key] if arg.startswith("--time=") else arg for arg in job["submit"]]
assert {job["commit"][:7] for job in old_plan["jobs"]} == {"1c7ea5d", "8052fab"}
assert not any(key.removesuffix("-05") in {job["name"].removesuffix("-03") for job in old_plan["jobs"]} for key in expect_extra)
write("wave-plan.json", old_plan)
write("watch-state.json", state)
watcher = run("sbatch", "--parsable", f"--output={control}/controller-%j.log", f"--error={control}/controller-%j.err",
              "experiments/training/pace_ice/watch_resumable.slurm", str(control / "wave-plan.json")).split(";")[0]
assert watcher.isdecimal()
print(json.dumps({"watcher": watcher, "limits": limits, "optional_specs_validated": len(expect_extra), "admitted_optional": 0}))

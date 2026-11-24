import json
import os
from pathlib import Path
import subprocess
import time

repo = Path.cwd()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
pinned = scratch / "worktrees/6c05c85"
old = scratch / "handoff-20261120/control-17"
control = old.with_name("control-18")
proof = old.with_name("control-18-disk-audit")
env = {k: v for k, v in os.environ.items() if not k.startswith(("SBATCH_", "SLURM_"))}
env["PYTHON"] = str(repo / ".venv/bin/python")

def command(*args):
    return subprocess.check_output(args, cwd=pinned, env=env, text=True).strip()

def read(p):
    return json.loads(p.read_text())

assert not command("git", "status", "--porcelain")
assert command("sacct", "-X", "-n", "-P", "-j", "7236543", "-o", "State,ExitCode") == "COMPLETED|0:0"
verified_plan = read(proof / "wave-plan.json")
required = next(j for j in verified_plan["jobs"] if j["name"] == "vp26-olmo3-gsm8k-grpo-50-s42-disk-audit")
plan = read(old / "wave-plan.json")
name = "vp26-olmo3-gsm8k-grpo-50-s42-10"
gate_job = next(j for j in plan["jobs"] if j["name"] == name)
assert gate_job["space_gib"] == 44 and gate_job["submit"][2] == "--time=12:00:00"
gate_job["space_gib"] = required["space_gib"]
gate_job["submit"][2] = required["submit"][2]
gate_job["afterok"].append("olmo-gsm-disk-cpu")
assert Path(gate_job["gates"][-1]).is_file()
controller = {"7233855", "7233857"}
assert controller <= set(command("squeue", "--me", "-h", "-o", "%i").splitlines())
command("scancel", *sorted(controller))
for _ in range(30):
    if not set(command("squeue", "--me", "-h", "-o", "%i").splitlines()) & controller:
        break
    time.sleep(1)
else:
    raise AssertionError("Old watcher still active")
state = read(old / "watch-state.json")
assert name not in state["jobs"]
state["jobs"]["olmo-gsm-disk-cpu"] = "7236543"
control.mkdir()
for file,data in (("wave-plan.json",plan), ("watch-state.json",state)):
    (control / file).write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
quota = command("lfs", "quota", "-u", os.environ["USER"], "/storage/ice1").splitlines()[-1].split()
assert quota[0] == "/storage/ice1" and (int(quota[3]) - int(quota[1])) / 1024**2 >= 20.25
print(command("sbatch", "--parsable", f"--output={control}/controller-%j.log", f"--error={control}/controller-%j.err",
              "experiments/training/pace_ice/watch_resumable.slurm", str(control / "wave-plan.json")))

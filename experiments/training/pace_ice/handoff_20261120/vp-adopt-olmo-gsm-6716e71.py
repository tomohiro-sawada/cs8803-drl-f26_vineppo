import json
import os
from pathlib import Path
import subprocess
import time

repo = Path.cwd()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
pinned = scratch / "worktrees/6716e71"
old = scratch / "handoff-20261120/control-16"
control = old.with_name("control-17")
proof = old.with_name("control-17-gate-audit")
env = {k: v for k, v in os.environ.items() if not k.startswith(("SBATCH_", "SLURM_"))}
env["PYTHON"] = str(repo / ".venv/bin/python")

def command(*args):
    return subprocess.check_output(args, cwd=pinned, env=env, text=True).strip()

def read(p):
    return json.loads(p.read_text())

assert not command("git", "status", "--porcelain")
assert command("sacct", "-X", "-n", "-P", "-j", "7233837", "-o", "State,ExitCode") == "COMPLETED|0:0"
verified_plan = read(proof / "wave-plan.json")
required = next(j for j in verified_plan["jobs"] if j["name"] == "vp26-olmo3-gsm8k-grpo-50-s42-gate-audit")
gate = required["gates"][-1].replace("gate-audit", "10")
plan = read(old / "wave-plan.json")
current = {j["name"]: j for j in plan["jobs"]}
gate_job = current["vp26-olmo3-gsm8k-grpo-50-s42-10"]
assert gate_job["gates"][-1].endswith("/handoff-olmo3-gsm8k-smoke-10/COMPLETE")
assert gate == current["vp26-olmo3-gsm8k-sft-10"]["disk_root"] + "/COMPLETE"
assert not Path(gate).is_file()
gate_job["gates"][-1] = gate
gate_job["afterok"].append("olmo-gsm-gate-cpu")
smoke_name = "vp26-olmo3-gsm8k-smoke-10"
archive_name = "vp26-arc-olmo3-gsm8k-sft-10"
deferred = plan["deferred_archives"]["jobs"]
small = next(j for j in deferred if j["name"] == archive_name)
assert small["gates"] == [gate]
plan["jobs"] = [small, *[j for j in plan["jobs"] if j["name"] != smoke_name]]
plan["deferred_archives"]["jobs"] = [
    j for j in deferred if j["name"] not in (archive_name, "vp26-arc-olmo3-gsm8k-smoke-10")
]
controller = {"7233461", "7233463"}
assert controller <= set(command("squeue", "--me", "-h", "-o", "%i").splitlines())
command("scancel", *sorted(controller))
for _ in range(30):
    if not set(command("squeue", "--me", "-h", "-o", "%i").splitlines()) & controller:
        break
    time.sleep(1)
else:
    raise AssertionError("Old watcher still active")
state = read(old / "watch-state.json")
assert smoke_name not in state["jobs"]
state["jobs"]["olmo-gsm-gate-cpu"] = "7233837"
control.mkdir()
for file,data in (("wave-plan.json",plan), ("watch-state.json",state)):
    (control / file).write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
quota = command("lfs", "quota", "-u", os.environ["USER"], "/storage/ice1").splitlines()[-1].split()
assert quota[0] == "/storage/ice1" and (int(quota[3]) - int(quota[1])) / 1024**2 >= 20.25
print(command("sbatch", "--parsable", f"--output={control}/controller-%j.log", f"--error={control}/controller-%j.err",
              "experiments/training/pace_ice/watch_resumable.slurm", str(control / "wave-plan.json")))

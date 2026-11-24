import json
import os
from pathlib import Path
import subprocess
import time

repo = Path.cwd()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
old = scratch / "handoff-20261120/control-18"
control = old.with_name("control-19")
pinned = scratch / "worktrees/6212e45"
watch_pin = scratch / "worktrees/6c05c85"
old_root = scratch / "training/pace_ice/handoff-rho-vineppo-650-s42-03"
new_root = old_root.with_name("handoff-rho-vineppo-650-s42-11")
old_job = "vp26-rho-vineppo-650-s42-03"
new_job = "vp26-rho-vineppo-650-s42-11"
old_archive = "vp26-arc-rho-vineppo-650-s42-03"
new_archive = "vp26-arc-rho-vineppo-650-s42-11"
env = {k:v for k,v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
env["PYTHON"] = str(repo / ".venv/bin/python")

def command(*args):
    return subprocess.check_output(args, cwd=watch_pin, env=env, text=True).strip()
def read(p):
    return json.loads(p.read_text())
assert not command("git", "status", "--porcelain")
assert not command("git", "-C", str(pinned), "status", "--porcelain")
assert not new_root.exists()
status = read(old_root/"status.json")
assert status["job"] == "7234149" and status["successor"] == "7239479"
assert status["iteration"] == 625 and status["position"] == [625, False]
metrics = old_root / "attempt-0002/metrics.jsonl"
assert not metrics.exists() or not metrics.stat().st_size, "Successor already progressed"
selection = read(old_root/"attempt-0001/selection.json")
best = selection["candidates"][-1]["checkpoint"]
source = Path(best["path"])
assert source == old_root/"attempt-0001/checkpoints/iteration-625.pt" and best["iteration"] == 625
assert command("sha256sum",str(source)).split()[0] == best["sha256"]
assert not (old_root/"COMPLETE").exists() and "403 Forbidden" in next(
    (repo / "artifacts/training/pace_ice/slurm").glob("*7227188.err")
).read_text()
queue = set(command("squeue","--me","-h","-o","%i").splitlines())
controller = {"7236554","7236557"}
assert controller <= queue and "7234149" in queue and "7239479" in queue
command("scancel","7239479","7234149",*sorted(controller))
targets = controller | {"7234149","7239479"}
for _ in range(40):
    if not set(command("squeue","--me","-h","-o","%i").splitlines()) & targets:
        break
    time.sleep(1)
else:
    raise AssertionError("The superseded continuation or watcher is still active")
plan = read(old/"wave-plan.json")
state = read(old/"watch-state.json")
job, = (j for j in plan["jobs"] if j["name"] == old_job)
arc, = (j for j in plan["jobs"] if j["name"] == old_archive)
assert old_job in state["jobs"] and old_archive not in state["jobs"]
state["jobs"].pop(old_job)
new = job | {"name":new_job,"repo":str(pinned),"commit":command("git","-C",str(pinned),"rev-parse","HEAD"),"disk_root":str(new_root)}
opts = [s.replace(old_root.name,new_root.name) for s in job["submit"]]
assert opts[2] == "--time=08:00:00"
opts[2] = "--time=02:00:00"
assert opts[opts.index("--archive-checkpoints")+1] == "yes"
opts[opts.index("--archive-checkpoints")+1] = "no"
opts[opts.index("--keep-checkpoints")+1] = "26"
new["submit"] = opts + ["--resume",str(source)]
arc = arc | {"name":new_archive,"repo":str(pinned),"commit":new["commit"],
             "disk_root":arc["disk_root"].replace(old_root.name,new_root.name),
             "submit":[s.replace(old_root.name,new_root.name) for s in arc["submit"]],
             "gates":[str(new_root/"COMPLETE")]}
plan["jobs"] = [new if j["name"] == old_job else j for j in plan["jobs"] if j["name"] != old_archive]
deferred = plan["deferred_archives"]
deferred["jobs"].append(arc)
deferred["recovery_sources"][old_job] = {"source":str(old_root),"checkpoint":str(source),"iteration":625,"job":"7227188"}
control.mkdir()
for name,data in (("wave-plan.json",plan),("watch-state.json",state)):
    (control/name).write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
quota = command("lfs","quota","-u",os.environ["USER"],"/storage/ice1").splitlines()[-1].split()
assert quota[0] == "/storage/ice1" and (int(quota[3])-int(quota[1]))/1024**2 >= 20.25
print(command("sbatch","--parsable",f"--output={control}/controller-%j.log",f"--error={control}/controller-%j.err",
              "experiments/training/pace_ice/watch_resumable.slurm",str(control/"wave-plan.json")))

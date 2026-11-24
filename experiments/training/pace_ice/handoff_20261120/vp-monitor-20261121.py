from pathlib import Path
import json
import subprocess
import sys
import time

def read(path):
    return json.loads(path.read_text())

def slurm(*args):
    return subprocess.check_output(args, text=True).splitlines()

scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
control = scratch / "handoff-20261120"
snapshots = {}
for watch in sorted(control.glob("control-*/watch-status.json")):
    snapshots[watch.parent] = read(watch)
directory = control / (sys.argv[1] if len(sys.argv) > 1 else "control-13")
watch = snapshots[directory]
plan = read(directory / "wave-plan.json")
jobs = read(directory / "watch-state.json")["jobs"]
scheduler = {fields[0]: fields[1:] for line in slurm("squeue", "-u", "tsawada7", "-h", "-o", "%i|%j|%T|%M|%r") if (fields := line.split("|"))}
paths = sorted((scratch / "training/pace_ice").glob("*/status.json"))
status = {p.parent: read(p) for p in paths}
root_jobs = {Path(j["disk_root"]): j["name"] for j in plan["jobs"]}
archive_roots = {root for root in root_jobs if root.suffix == ".json"}
current = {root: name for root, name in root_jobs.items() if name in jobs}
past_ids = {jobs[name] for name in current.values()}
gate_producers = {}
for spec in plan["jobs"]:
    root = Path(spec["disk_root"])
    for marker in ("COMPLETE", "ROLLOVER", "acceptance.json"):
        gate_producers[root / marker] = spec["name"]
    if root.suffix == ".json":
        gate_producers[root] = spec["name"]
for root, s in status.items():
    if root in current:
        past_ids.update(s[k] for k in ("job", "successor") if k in s)
missing = sorted(past_ids.difference(scheduler))
accounting = {}
if missing:
    accounting = {p[0]: p[1:] for row in slurm("sacct", "-X", "-n", "-P", "-j", ",".join(missing), "--format=JobIDRaw,State,ExitCode,Elapsed") if (p := row.split("|"))}
stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
if (directory / "watch-complete.json").is_file():
    assert not watch["pending"]
    print(stamp, directory.name, "all admissions completed")
else:
    print(stamp, directory.name, "watch heartbeat seconds ago", round(time.time() - watch["updated"]), "pending", len(watch["pending"]))
assert (directory / "watch-status.json").is_file()
if (directory / "watch-halted.json").is_file():
    print("WATCH HALTED", read(directory / "watch-halted.json"))
if watch["submission_errors"] or watch["dependency_wait"]:
    print("WATCH ERRORS", watch["submission_errors"], watch["dependency_wait"])
for job, values in sorted(scheduler.items()):
    print("QUEUE", job, *values)
for root, name in sorted(current.items()):
    job = jobs[name]
    s = status.get(root, {})
    latest = s.get("job", job)
    if root in archive_roots:
        if root.is_file():
            print("VERIFIED", name, job)
            continue
    elif (root / "COMPLETE").is_file():
        print("COMPLETE", name, s.get("iteration", "SFT"), latest)
        continue
    if (root / "HALTED").is_file():
        print("HALTED", name, latest, s.get("error"))
        continue
    if latest in scheduler:
        print("ACTIVE", name, latest, s.get("iteration", 0))
    elif s.get("successor") in scheduler:
        print("WAIT SUCCESSOR", name, s["successor"], s.get("iteration"))
    else:
        print("TERMINATED OR MISSING MARKER", name, latest, accounting.get(latest), "successor", s.get("successor"), accounting.get(s.get("successor")))
for spec in plan["jobs"]:
    name = spec["name"]
    if name in jobs:
        continue
    for value in spec.get("gates", ()):
        gate = Path(value)
        if gate.is_file():
            continue
        producer = gate_producers.get(gate)
        producer_job = jobs.get(producer)
        if (gate.parent / "HALTED").is_file():
            print("DEAD GATE", name, gate, "producer halted")
        elif (gate.parent / "COMPLETE").is_file():
            print("DEAD GATE", name, gate, "producer completed without marker")
        elif producer is None:
            print("UNMAPPED GATE", name, gate)
        elif producer_job is not None and producer_job in accounting:
            state = status.get(gate.parent, {})
            if not any(state.get(k) in scheduler for k in ("job", "successor")):
                last = state.get("job", producer_job)
                if last in accounting and not accounting[last][0].startswith(("RUNNING", "PENDING")):
                    print("DEAD GATE", name, gate, "producer left Slurm", last, accounting[last])
        print("GATE", name, gate.name, gate.parent.name)
archives = sorted(control.glob("control-*/*.pruned.complete.json"))
print("archives verified", len(archives), "all status files read", len(status), "all watcher statuses read", len(snapshots))
for name, info in watch["disk_wait"].items():
    print("DISK", name, round(info["free_gib"], 1), round(info["required_gib"], 1))

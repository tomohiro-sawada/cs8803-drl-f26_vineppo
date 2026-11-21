"""Submit gated PACE waves and inject the handoff's one-shot Slurm test faults."""

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from time import sleep, time

from experiments.training.resume import atomic_record


def slurm(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def submit(job, since):
    name = job["name"]
    previous = slurm(
        "sacct",
        "-X",
        "-n",
        "-P",
        "--starttime",
        since,
        "--name",
        name,
        "--format=JobIDRaw,JobName%100",
    ).splitlines()
    ids = [line.split("|")[0] for line in previous if line.split("|")[1] == name]
    if not ids:
        ids = slurm(
            "squeue", "--me", "--noheader", "--name", name, "--format=%A"
        ).split()
    assert len(set(ids)) <= 1, f"Duplicate submissions for {name}: {ids}"
    if ids:
        return ids[0]
    env = os.environ | job["env"]
    repo = Path(job["repo"]).resolve(strict=True)
    assert slurm("git", "rev-parse", "HEAD", cwd=repo) == job["commit"]
    assert not slurm("git", "status", "--porcelain", cwd=repo), (
        f"Dirty worktree: {repo}"
    )
    result = slurm(
        "sbatch", "--parsable", f"--job-name={name}", *job["submit"], cwd=repo, env=env
    ).split(";")[0]
    assert result.isdecimal()
    print(f"submitted {name} {result}", flush=True)
    return result


def inject(spec, job, cache):
    root = Path(spec["root"])
    status_file = root / "status.json"
    if (
        not status_file.is_file()
        or (root / "COMPLETE").exists()
        or (root / "HALTED").exists()
    ):
        return None
    status = json.loads(status_file.read_text())
    if status["job"] != job or status["restart"] != 0 or status["state"] != "running":
        return None
    kind = spec["kind"]
    attempt = root / f"attempt-{status['attempt']:04d}"
    after = spec.get("after", 1)
    if kind == "checkpoint":
        ready = status["iteration"] >= after and list(
            (attempt / "checkpoints").glob(".iteration-*.pt.*.tmp")
        )
    elif kind in ("validation", "test"):
        evaluation = f"validation-{after}" if kind == "validation" else "test-selected"
        directory = attempt / "evaluations" / evaluation
        ready = (directory / "progress.json").is_file() and not (
            directory / "result.json"
        ).exists()
    else:
        ready = status["iteration"] >= after
    if not ready:
        return None
    if kind == "requeue":
        command = ("scontrol", "requeue", job)
    elif kind == "signal":
        command = ("scancel", "--batch", "--signal=USR1", job)
    else:
        step = status["learner_step"]
        if kind == "worker":
            identity = json.loads(
                (cache / "workers" / f"{job}-0/worker-0.json").read_text()
            )
            assert identity["slurm_job_id"] == job and identity["rank"] == 0
            step = identity["slurm_step_id"]
        assert step.isdecimal()
        command = ("scancel", "--signal=KILL", f"{job}.{step}")
    slurm(*command)
    event = {
        "kind": kind,
        "job": job,
        "command": command,
        "time": time(),
        "attempt": str(attempt),
        "iteration": status["iteration"],
    }
    print(
        f"injected {kind} in job {job} at iteration {status['iteration']}", flush=True
    )
    return event


def watch(plan_path):
    directory = plan_path.parent
    state_path = directory / "watch-state.json"
    plan = json.loads(plan_path.read_text())
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"jobs": {}, "faults": {}}
    )
    jobs = plan["jobs"]
    assert len({job["name"] for job in jobs}) == len(jobs)
    while time() < plan["deadline"]:
        for job in jobs:
            name = job["name"]
            gates = job.get("gates", [])
            if name not in state["jobs"] and all(
                Path(path).is_file() for path in gates
            ):
                options = job
                if dependencies := job.get("afterok"):
                    assert all(dep in state["jobs"] for dep in dependencies)
                    job_ids = ":".join(state["jobs"][dep] for dep in dependencies)
                    options = job | {
                        "submit": [
                            f"--dependency=afterok:{job_ids}",
                            "--kill-on-invalid-dep=yes",
                            *job["submit"],
                        ]
                    }
                state["jobs"][name] = submit(options, plan["since"])
                atomic_record(state_path, state)
        outstanding = False
        for spec in plan.get("faults", []):
            name = spec["name"]
            if name in state["faults"] or name not in state["jobs"]:
                continue
            outstanding |= (Path(spec["root"]) / "status.json").is_file()
            if event := inject(spec, state["jobs"][name], Path(plan["cache"])):
                atomic_record(directory / f"{name}.fault.json", event)
                state["faults"][name] = event
                atomic_record(state_path, state)
        pending = [job["name"] for job in jobs if job["name"] not in state["jobs"]]
        atomic_record(
            directory / "watch-status.json", {"pending": pending, "updated": time()}
        )
        if not pending and len(state["faults"]) == len(plan.get("faults", [])):
            return
        sleep(1 if outstanding else 10)
    raise TimeoutError("PACE wave watcher reached the handoff budget")


if __name__ == "__main__":
    plan_path = Path(sys.argv[1]).resolve(strict=True)
    with (plan_path.parent / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watch(plan_path)

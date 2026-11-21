"""Submit gated PACE waves and inject the handoff's one-shot Slurm test faults."""

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from time import sleep, time

from experiments.training.pace_ice.chain import submission_environment
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
    env = submission_environment(os.environ | job["env"])
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


def free_scratch_gib(mount):
    entries = slurm("lfs", "quota", "-u", os.environ["USER"], mount).splitlines()
    quota = [line.split() for line in entries if line.split()[:1] == [mount]]
    assert len(quota) == 1, f"No scratch quota for {mount}"
    used, soft_limit, hard_limit = (int(field.rstrip("*")) for field in quota[0][1:4])
    assert 0 < soft_limit <= hard_limit
    return (hard_limit - used) / 1024**2


def disk_reservation(job):
    root = Path(job["disk_root"])
    if root.is_file() or any(
        (root / marker).is_file() for marker in ("COMPLETE", "HALTED")
    ):
        return 0
    used = (
        int(slurm("du", "-sk", "--", str(root)).split()[0]) / 1024**2
        if root.is_dir()
        else 0
    )
    return max(0, job["space_gib"] - used)


def dependency_blockers(dependencies, receipts, accounting):
    """Require durable success before submitting a job whose evidence gates opened."""
    missing = {
        name: "awaiting submission" for name in dependencies if name not in receipts
    }
    ids = {name: receipts[name] for name in dependencies if name in receipts}
    unknown = sorted(set(ids.values()) - accounting.keys())
    if unknown:
        rows = slurm(
            "sacct",
            "-X",
            "-n",
            "-P",
            "--jobs",
            ",".join(unknown),
            "--format=JobIDRaw,State,ExitCode",
        ).splitlines()
        accounting.update(dict.fromkeys(unknown, "awaiting accounting"))
        for row in rows:
            job, state, code = row.split("|")
            assert job in unknown, f"Unexpected dependency accounting row: {row}"
            accounting[job] = f"{state.strip()} {code}"
    return missing | {
        name: accounting.get(job, "awaiting accounting")
        for name, job in ids.items()
        if accounting.get(job) != "COMPLETED 0:0"
    }


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
    names = {job["name"] for job in jobs}
    assert len(names) == len(jobs)
    assert all(
        set(job.get("afterok", [])) <= names | state["jobs"].keys() for job in jobs
    ), "Unknown wave dependency"
    retry_at = {}
    while time() < plan["deadline"]:
        reserved = None
        disk_wait = {}
        evidence_wait = {}
        dependency_wait = {}
        submission_errors = {}
        accounting = {}
        for job in jobs:
            name = job["name"]
            gates = job.get("gates", [])
            if name not in state["jobs"]:
                missing = next(
                    (path for path in gates if not Path(path).is_file()), None
                )
                if missing is not None:
                    evidence_wait[name] = missing
                    continue
            if name not in state["jobs"]:
                try:
                    blockers = dependency_blockers(
                        job.get("afterok", []), state["jobs"], accounting
                    )
                except subprocess.CalledProcessError as error:
                    dependency_wait[name] = {
                        "accounting": f"sacct exit {error.returncode}"
                    }
                    continue
                if blockers:
                    dependency_wait[name] = blockers
                    continue
                if "disk_root" in job:
                    free = free_scratch_gib(plan["scratch_mount"])
                    if reserved is None:
                        reserved = sum(
                            disk_reservation(active)
                            for active in jobs
                            if active["name"] in state["jobs"] and "disk_root" in active
                        )
                    required = job["space_gib"] + reserved + 20
                    if free < required:
                        disk_wait[name] = {
                            "free_gib": free,
                            "required_gib": required,
                        }
                        continue
                if time() < retry_at.get(name, 0):
                    submission_errors[name] = "waiting to retry scheduler submission"
                    continue
                try:
                    state["jobs"][name] = submit(job, plan["since"])
                except subprocess.CalledProcessError as error:
                    submission_errors[name] = f"{error.cmd[0]} exit {error.returncode}"
                    print(
                        f"submission of {name} failed: {submission_errors[name]}",
                        flush=True,
                    )
                    retry_at[name] = time() + 60
                    continue
                atomic_record(state_path, state)
                if "disk_root" in job:
                    reserved += disk_reservation(job)
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
            directory / "watch-status.json",
            {
                "pending": pending,
                "evidence_wait": evidence_wait,
                "dependency_wait": dependency_wait,
                "disk_wait": disk_wait,
                "submission_errors": submission_errors,
                "updated": time(),
            },
        )
        if not pending and len(state["faults"]) == len(plan.get("faults", [])):
            return
        sleep(1 if outstanding else 10)
    raise TimeoutError("PACE wave watcher reached the handoff budget")


def run_allocation(plan_path):
    directory = plan_path.parent
    complete, halted = (
        directory / "watch-complete.json",
        directory / "watch-halted.json",
    )
    if complete.exists() or halted.exists():
        return
    job = os.environ["SLURM_JOB_ID"]
    name = f"vp26-watch-after-{job}"
    jobs = slurm("squeue", "--me", "--name", name, "--noheader", "--format=%A").split()
    assert len(jobs) <= 1
    if jobs:
        successor = jobs[0]
    else:
        successor = slurm(
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{job}",
            f"--job-name={name}",
            f"--output={directory}/controller-%j.log",
            f"--error={directory}/controller-%j.err",
            "experiments/training/pace_ice/watch_resumable.slurm",
            str(plan_path),
            env=submission_environment(os.environ),
        ).split(";")[0]
    assert successor.isdecimal()
    print(f"watch allocation {job}, successor {successor}", flush=True)
    try:
        watch(plan_path)
    except Exception as error:
        atomic_record(halted, {"job": job, "error": str(error)})
        raise
    else:
        atomic_record(complete, {"job": job, "time": time()})
    finally:
        subprocess.run(["scancel", successor], check=True)


if __name__ == "__main__":
    plan_path = Path(sys.argv[1]).resolve(strict=True)
    with (plan_path.parent / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run_allocation if sys.argv[2:] == ["--slurm"] else watch)(plan_path)

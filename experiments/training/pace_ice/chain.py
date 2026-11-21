"""Hold one afterany successor and resume the highest published boundary."""

import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from experiments.training.resume import atomic_record, checkpoint_position

SCRIPT = "experiments/training/pace_ice/train_resumable.slurm"
LAUNCH = ["bash", "experiments/training/launch.sh", "pace_ice"]


def latest_checkpoint(attempts):
    paths = [
        path
        for attempt in attempts
        for pattern in ("iteration-*.pt", "resume-*.pt")
        for path in (attempt / "checkpoints").glob(pattern)
    ]
    return max(
        paths,
        key=lambda path: (*checkpoint_position(path), path.parent.parent.name),
        default=None,
    )


def run_command(*command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def submission_environment(env):
    """Keep caller configuration and let Slurm set the new allocation identity."""
    return {
        key: value
        for key, value in env.items()
        if not key.startswith(("SLURM_", "SBATCH_"))
    }


def successor(job, command, env):
    name = f"vineppo-after-{job}"
    ids = run_command(
        "squeue", "--noheader", "--me", "--name", name, "--format=%A"
    ).split()
    assert len(ids) <= 1, f"Multiple successors of {job}: {ids}"
    if ids:
        return ids[0]
    limit, constraint = run_command(
        "squeue", "--noheader", f"--jobs={job}", "--format=%l|%f"
    ).split("|", 1)
    assert limit and constraint and constraint != "(null)", (
        "Expected a pinned GPU constraint"
    )
    fields = list(map(int, limit.split(":")))
    assert len(fields) in (2, 3)
    assert (
        0 < sum(value * 60**i for i, value in enumerate(reversed(fields))) <= 16 * 3600
    )
    cpus = env["SLURM_CPUS_PER_TASK"]
    memory = (
        f"--mem-per-cpu={env['SLURM_MEM_PER_CPU']}"
        if "SLURM_MEM_PER_CPU" in env
        else f"--mem={env['SLURM_MEM_PER_NODE']}"
    )
    warning = int(env.get("PACE_CHAIN_WARNING_SECONDS", "600"))
    assert warning > 0
    log = Path(env["PACE_CHAIN_LOG_DIR"]).resolve(strict=True)
    assert log.is_dir()
    result = run_command(
        "sbatch",
        "--parsable",
        f"--dependency=afterany:{job}",
        f"--job-name={name}",
        f"--time={limit}",
        f"--constraint={constraint}",
        f"--cpus-per-task={cpus}",
        memory,
        f"--signal=B:USR1@{warning}",
        f"--output={log}/%x-%j.out",
        f"--error={log}/%x-%j.err",
        SCRIPT,
        *command,
        env=submission_environment(env),
    ).split(";")[0]
    assert result.isdecimal(), f"Unexpected successor job ID: {result}"
    return result


def supervise(root, options, env):
    job = env["SLURM_JOB_ID"]
    restart = int(env.get("SLURM_RESTART_COUNT", "0"))
    status_path = root / "status.json"
    plan_path = root / "chain-plan.json"
    plan = {"repo": str(Path.cwd()), "options": options}
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, "Launch plan changed"
    else:
        atomic_record(plan_path, plan)
    if (root / "COMPLETE").exists():
        return 0
    assert not (root / "HALTED").exists(), f"Run is halted: {root}"

    stop_file = root / f".stop-{job}-{restart}"
    signal.signal(signal.SIGUSR1, lambda *_: stop_file.touch())
    env = env | {
        "PACE_CHAIN_STOP_FILE": str(stop_file),
        "PACE_CHAIN_STATUS_FILE": str(status_path),
    }
    attempts = sorted(root.glob("attempt-*"))
    assert all(path.name == f"attempt-{i:04d}" for i, path in enumerate(attempts))
    checkpoint = latest_checkpoint(attempts)
    frozen = list(root.glob("attempt-*/selection-frozen.json"))
    assert len(frozen) <= 1, "Multiple frozen selections"
    iteration = checkpoint_position(checkpoint)[0] if checkpoint is not None else 0
    position = [iteration, bool(frozen)]
    previous = json.loads(status_path.read_text()) if status_path.exists() else None
    if previous is not None:
        assert position >= previous["position"], "Latest published boundary regressed"

    child = successor(job, [str(root), *options], env)
    status = {
        "job": job,
        "restart": restart,
        "successor": child,
        "position": position,
        "attempt": len(attempts),
        "state": "running",
        "iteration": iteration,
    }
    if (
        previous is not None
        and position == previous["position"]
        and not (previous["job"] == job and restart == 1)
    ):
        reason = "Allocation restarted without a newer completed boundary"
        halted = status | {"state": "halted", "error": reason}
        atomic_record(root / "HALTED", halted)
        atomic_record(status_path, halted)
        subprocess.run(["scancel", child], check=True)
        print(f"{reason}: {root}", file=sys.stderr)
        return 1
    atomic_record(status_path, status)
    args = dict(zip(options[::2], options[1::2], strict=True))
    evaluation = args.get("--evaluation", "disabled")
    if not frozen:
        output = root / f"attempt-{len(attempts):04d}"
        resume = [] if checkpoint is None else ["--resume", str(checkpoint)]
        if checkpoint is not None and evaluation != "disabled":
            selection = checkpoint.parent.parent / "selection.json"
            assert selection.is_file(), f"Resume boundary has no selection: {selection}"
            resume.extend(("--resume-selection", str(selection)))
        rc = subprocess.call(
            [*LAUNCH, "train", *options, "--output", str(output), *resume], env=env
        )
        if rc:
            status = json.loads(status_path.read_text())
            atomic_record(status_path, status | {"state": "failed", "exit_code": rc})
            return rc
        iteration = json.loads((output / "summary.json").read_text())[
            "completed_iteration"
        ]
        if iteration < int(args["--num-iterations"]):
            atomic_record(
                status_path, status | {"state": "yielded", "iteration": iteration}
            )
            return 0
        if evaluation != "disabled":
            frozen = [output / "selection-frozen.json"]
            assert frozen[0].is_file(), "Completed training did not freeze selection"
            if (output / "test-deferred.json").exists():
                assert evaluation == "validation-test"
                atomic_record(
                    status_path, status | {"state": "yielded", "iteration": iteration}
                )
                return 0
    if frozen:
        rc = subprocess.call(
            [
                *LAUNCH,
                "evaluate",
                *options,
                "--output",
                str(root),
                "--resume-selection",
                str(frozen[0]),
            ],
            env=env,
        )
        if rc:
            atomic_record(status_path, status | {"state": "failed", "exit_code": rc})
            return rc
    complete = status | {"state": "complete", "iteration": iteration}
    atomic_record(root / "COMPLETE", complete)
    atomic_record(status_path, complete)
    subprocess.run(["scancel", child], check=True)
    return 0


def main(argv):
    root, *options = argv
    assert root.startswith("/") and not Path(root).is_symlink(), (
        "Use an absolute run root"
    )
    assert len(options) % 2 == 0 and all(opt.startswith("--") for opt in options[::2])
    keys = options[::2]
    assert len(set(keys)) == len(keys), "Duplicate runner option"
    assert not {"--output", "--resume", "--resume-selection"}.intersection(keys)
    assert "--num-iterations" in keys and "--resume-checkpoint-every" in keys
    assert os.environ["SLURM_NTASKS"] == "1", (
        "PACE chaining needs a single GPU allocation"
    )
    assert "TRAINING_ENTRY" not in os.environ, "Chained runs use the training runner"
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return supervise(root, options, dict(os.environ))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Exercise the PACE supervisor and commit ordering without a GPU allocation."""

import fcntl
import json
import subprocess

import pytest

from experiments.training.pace_ice import chain
from experiments.training.resume import BoundaryDeadline, ResumeSlot


def test_deadline_uses_longest_iteration_checkpoint_and_warning(tmp_path):
    now = [10]
    flag = tmp_path / "signal"
    deadline = BoundaryDeadline(100, flag, clock=lambda: now[0], margin=10)
    deadline.observe(40, 3)
    deadline.observe(30, 5)
    now[0] = 44
    assert not deadline.should_stop()
    now[0] = 45
    assert deadline.should_stop()
    now[0] = 10
    flag.touch()
    assert deadline.should_stop()
    flag.unlink()
    deadline.request_stop()
    assert deadline.should_stop()


def test_resume_slot_keeps_previous_until_atomic_publication(tmp_path):
    def commit(path):
        path.write_bytes(b"valid")
        return path

    def interrupted(path):
        path.with_suffix(".tmp").touch()
        raise OSError("interrupted")

    slot = ResumeSlot(tmp_path)
    previous = slot.save(3, commit)
    with pytest.raises(OSError, match="interrupted"):
        slot.save(6, interrupted)
    assert previous.read_bytes() == b"valid"
    latest = slot.save(6, commit)
    assert slot.save(6, interrupted) == latest
    assert list(tmp_path.glob("*.pt")) == [latest]


def test_recovery_orders_numeric_boundaries_and_ignores_tmp(tmp_path):
    attempts = [tmp_path / f"attempt-{i:04d}" for i in range(2)]
    for attempt in attempts:
        (attempt / "checkpoints").mkdir(parents=True)
    for attempt, name in [
        (attempts[0], "iteration-80.pt"),
        (attempts[0], "resume-100.pt"),
        (attempts[1], "iteration-100.pt"),
        (attempts[1], "resume-103.pt"),
        (attempts[1], ".resume-106.pt.tmp"),
    ]:
        (attempt / "checkpoints" / name).touch()
    latest = chain.latest_checkpoint(attempts)
    assert latest.name == "resume-103.pt"
    latest.unlink()
    assert chain.latest_checkpoint(attempts).name == "iteration-100.pt"


@pytest.fixture
def simulated_slurm(monkeypatch):
    calls = []
    monkeypatch.setattr(chain.signal, "signal", lambda *args: None)
    monkeypatch.setattr(chain, "successor", lambda job, *_: str(int(job) + 1))

    def cancel(command, *, check):
        assert command[0] == "scancel" and check
        calls.append(command)

    monkeypatch.setattr(chain.subprocess, "run", cancel)
    return calls


def test_failed_chain_halts_after_no_progress(tmp_path, simulated_slurm, monkeypatch):
    monkeypatch.setattr(chain.subprocess, "call", lambda *args, **kwargs: 89)
    args = ["--num-iterations", "4"]
    assert chain.supervise(tmp_path, args, {"SLURM_JOB_ID": "1"}) == 89
    assert chain.supervise(tmp_path, args, {"SLURM_JOB_ID": "2"}) == 1
    assert simulated_slurm == [["scancel", "3"]]
    halted = json.loads((tmp_path / "HALTED").read_text())
    assert halted["position"] == [0, False] and halted["state"] == "halted"


def test_chain_rejects_another_active_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "1")
    options = [str(tmp_path), "--num-iterations", "4", "--resume-checkpoint-every", "1"]
    with (tmp_path / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            chain.main(options)


def test_successor_runs_frozen_test_and_records_completion(
    tmp_path, simulated_slurm, monkeypatch
):
    owner = tmp_path / "attempt-0000"
    (owner / "checkpoints").mkdir(parents=True)
    frozen = owner / "selection-frozen.json"
    frozen.touch()
    (owner / "checkpoints" / "iteration-4.pt").touch()

    def finish(command, *, env):
        assert command[3] == "evaluate"
        assert command[-2:] == ["--resume-selection", str(frozen)]
        return 0

    monkeypatch.setattr(subprocess, "call", finish)
    args = ["--num-iterations", "4", "--evaluation", "validation-test"]
    assert chain.supervise(tmp_path, args, {"SLURM_JOB_ID": "1"}) == 0
    assert chain.supervise(tmp_path, args, {"SLURM_JOB_ID": "2"}) == 0
    assert simulated_slurm == [["scancel", "2"]]
    assert json.loads((tmp_path / "COMPLETE").read_text())["iteration"] == 4

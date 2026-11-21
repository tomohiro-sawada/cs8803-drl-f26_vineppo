"""Check that old allocations gate waves through durable accounting."""

import json
import subprocess

import pytest

from experiments.training.pace_ice import watch


def test_dependencies_require_clean_terminal_accounting(monkeypatch):
    def sacct(*command):
        assert (
            command[0] == "sacct" and command[command.index("--jobs") + 1] == "1,2,3,4"
        )
        return "1|COMPLETED|0:0\n2|COMPLETED|1:0\n3|RUNNING|0:0"

    monkeypatch.setattr(watch, "slurm", sacct)
    receipts = {"old": "1", "failed": "2", "active": "3", "unknown": "4"}
    assert watch.dependency_blockers([*receipts, "future"], receipts, {}) == {
        "failed": "COMPLETED 1:0",
        "active": "RUNNING 0:0",
        "unknown": "awaiting accounting",
        "future": "awaiting submission",
    }


def test_old_dependencies_and_rejected_submissions_do_not_stall_waves(
    tmp_path, monkeypatch
):
    plan = tmp_path / "wave-plan.json"
    jobs = [
        {"name": "rejected", "afterok": ["control"], "submit": ["--wrap=probe"]},
        {"name": "ready", "afterok": ["control"], "submit": ["--wrap=train"]},
        {"name": "blocked", "afterok": ["bad"], "submit": ["--wrap=train"]},
    ]
    plan.write_text(json.dumps({"jobs": jobs, "deadline": 200, "since": "yesterday"}))
    state = tmp_path / "watch-state.json"
    state.write_text(json.dumps({"jobs": {"control": "11", "bad": "12"}, "faults": {}}))
    submitted = []

    def accounting(*command):
        return {
            "11": "11|COMPLETED|0:0",
            "12": "12|FAILED|1:0",
        }[command[command.index("--jobs") + 1]]

    def submit(job, since):
        assert since == "yesterday"
        submitted.append(job["submit"])
        if job["name"] == "rejected":
            raise subprocess.CalledProcessError(1, ["sbatch"])
        assert job["name"] == "ready"
        return "13"

    class EndPoll(Exception):
        pass

    def sleep(_):
        raise EndPoll

    monkeypatch.setattr(watch, "slurm", accounting)
    monkeypatch.setattr(watch, "submit", submit)
    monkeypatch.setattr(watch, "time", lambda: 100)
    monkeypatch.setattr(watch, "sleep", sleep)
    with pytest.raises(EndPoll):
        watch.watch(plan)
    assert submitted == [["--wrap=probe"], ["--wrap=train"]]
    assert json.loads(state.read_text())["jobs"]["ready"] == "13"
    status = json.loads((tmp_path / "watch-status.json").read_text())
    assert status["dependency_wait"] == {"blocked": {"bad": "FAILED 1:0"}}
    assert status["submission_errors"] == {"rejected": "sbatch exit 1"}

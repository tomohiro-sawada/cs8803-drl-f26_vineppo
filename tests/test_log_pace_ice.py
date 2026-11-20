"""Offline regression tests for the read-only scheduler logger."""

import json
import subprocess

import pytest

from scripts.diagnostics import log_pace_ice as logger


def test_collect_preserves_success_failure_and_timeout(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs == {"capture_output": True, "text": True, "timeout": 60, "check": False}
        if len(calls) == 2:
            return subprocess.CompletedProcess(argv, 255, "", "Connection refused")
        if len(calls) == 3:
            raise subprocess.TimeoutExpired(argv, 60, output=b"partial", stderr=b"slow")
        return subprocess.CompletedProcess(argv, 0, "raw scheduler output\n", "")

    monkeypatch.setattr(logger.subprocess, "run", run)
    results = logger.collect(logger.ssh_prefix("ice"), 60)
    assert results["nodes"]["stdout"] == "raw scheduler output\n"
    assert results["partitions"]["returncode"] == 255
    assert results["reservations"]["timed_out"]
    assert results["reservations"]["stdout"] == "partial"
    assert results["reservations"]["returncode"] is None
    assert results["pending_jobs"]["returncode"] == 0
    assert len(calls) == 4
    assert all("BatchMode=yes" in call and "StrictHostKeyChecking=yes" in call for call in calls)
    assert all("export TZ=UTC LC_ALL=C;" in call[-1] for call in calls)


def test_append_retains_previous_records(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    for record in [{"ok": True}, {"ok": False}]:
        with path.open("a") as stream:
            logger.append(stream, record)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"ok": True}, {"ok": False},
    ]


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("local", [False, True])
def test_once_records_provenance_and_status(monkeypatch, tmp_path, failure, local):
    path = tmp_path / "snapshots.jsonl"
    transport = ["--local"] if local else ["--host", "ice"]
    monkeypatch.setattr("sys.argv", ["log_pace_ice.py", *transport, "--once", "--output", str(path)])
    monkeypatch.setattr(logger.subprocess, "check_output", lambda *a, **kw: "abcdef\n")

    def collect(prefix, timeout):
        assert prefix == (["bash", "-c"] if local else logger.ssh_prefix("ice"))
        return {"nodes": {"returncode": 255 if failure else 0,
                          "stderr": "error" if failure else "", "timed_out": False}}

    monkeypatch.setattr(logger, "collect", collect)
    monkeypatch.setattr(logger.time, "sleep", lambda *a: pytest.fail("--once must not sleep"))
    assert logger.main() == int(failure)
    record = json.loads(path.read_text())
    assert record["ok"] is not failure
    assert record["git_commit"] == "abcdef"
    assert record["transport"] == ("local" if local else "ssh")
    assert record["host"] == (logger.os.uname().nodename if local else "ice")
    assert len(record["script_sha256"]) == 64
    assert record["started_at"].endswith("+00:00")
    assert record["started_at_eastern"][-6:] in {"-04:00", "-05:00"}


def test_duplicate_writer_rejected(monkeypatch, tmp_path):
    path = tmp_path / "snapshots.jsonl"
    monkeypatch.setattr("sys.argv", ["log_pace_ice.py", "--host", "ice", "--once", "--output", str(path)])
    monkeypatch.setattr(logger.subprocess, "check_output", lambda *a, **kw: "abcdef\n")
    with path.open("a") as stream:
        logger.fcntl.flock(stream, logger.fcntl.LOCK_EX | logger.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            logger.main()


@pytest.mark.parametrize("extra", [["--interval-seconds", "0"], ["--timeout-seconds", "0"],
                                   ["--interval-seconds", "300", "--timeout-seconds", "100"]])
def test_invalid_poll_settings_rejected(monkeypatch, extra):
    monkeypatch.setattr("sys.argv", ["log_pace_ice.py", "--host", "ice", *extra])
    with pytest.raises(SystemExit, match="2"):
        logger.main()

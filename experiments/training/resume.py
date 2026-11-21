"""Iteration-boundary resume utilities shared by the learner and PACE wrapper."""

import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

CHECKPOINT = re.compile(r"(iteration|resume)-(0|[1-9][0-9]*)\.pt\Z")


def checkpoint_position(path):
    match = CHECKPOINT.fullmatch(Path(path).name)
    assert match is not None, f"Invalid checkpoint name: {path}"
    kind, iteration = match.groups()
    return int(iteration), kind == "iteration"


class ResumeSlot:
    """Keep one rolling resume boundary. Publish the new file before unlinking old."""

    def __init__(self, directory):
        self.directory = directory

    def save(self, iteration, publish):
        path = self.directory / f"resume-{iteration}.pt"
        if not path.is_file():
            assert publish(path) == path
            sync_directory(path.parent)
        for previous in self.directory.glob("resume-*.pt"):
            if previous != path:
                assert checkpoint_position(previous)[0] < iteration
                previous.unlink()
        return path


class BoundaryDeadline:
    def __init__(self, end, stop_file, *, clock, margin=120):
        self.end = end
        self.stop_file = stop_file
        self.clock = clock
        self.margin = margin
        self.requested = False
        self.longest_iteration = self.longest_checkpoint = 0.0

    def request_stop(self, *_):
        self.requested = True

    def observe(self, iteration_seconds, checkpoint_seconds):
        self.longest_iteration = max(self.longest_iteration, iteration_seconds)
        self.longest_checkpoint = max(self.longest_checkpoint, checkpoint_seconds)

    def should_stop(self):
        needed = self.longest_iteration + self.longest_checkpoint + self.margin
        return (
            self.requested
            or self.stop_file.exists()
            or self.clock() + needed >= self.end
        )


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_record(path, data):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.") as temp:
        temp.write((json.dumps(data, sort_keys=True) + "\n").encode())
        temp.flush()
        os.fsync(temp.fileno())
        staging = temp.name + ".publish"
        os.link(temp.name, staging)
        os.replace(staging, path)
        sync_directory(path.parent)

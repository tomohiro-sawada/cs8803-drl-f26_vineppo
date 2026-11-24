"""Read-only on run directories: does a compact checkpoint load when its frozen base is a hard link, and when it is a symlink?"""

import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path[:0] = ["/home/hice1/tsawada7/Files/projects/cs8803-drl-f26_vineppo/src"]
from runtime.checkpoints import checkpoint_models

T = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo/training/pace_ice")
base = T / "handoff-rho-vineppo-650-s42-03/attempt-0000/checkpoints/frozen-lora.pt"
checkpoint = (
    T / "handoff-rho-vineppo-650-s42-03/attempt-0001/checkpoints/iteration-550.pt"
)
state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
print(
    "checkpoint expects:",
    state["frozen"]["path"],
    state["frozen"]["sha256"][:16],
    flush=True,
)
# A scratch-side stand-in for "the path the checkpoint records", on the same filesystem as the base.
with tempfile.TemporaryDirectory(
    dir=T.parent.parent / "handoff-20261120/control-21", prefix="restore-"
) as tmp:
    for how in ("hard link", "symlink"):
        target = Path(tmp) / how.replace(" ", "-") / "frozen-lora.pt"
        target.parent.mkdir()
        (os.link if how == "hard link" else os.symlink)(base, target)
        trial = dict(
            state, frozen={"path": str(target), "sha256": state["frozen"]["sha256"]}
        )
        try:
            policy, reference = checkpoint_models(trial)
            print(
                f"{how}: loaded, {len(policy)} policy tensors, {len(reference)} reference tensors",
                flush=True,
            )
        except AssertionError as error:
            print(f"{how}: refused: {error}", flush=True)

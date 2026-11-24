import json
from pathlib import Path
import sys

from experiments.training.archive import archive_checkpoint
from runtime.checkpoints import file_sha256
from scripts.remote_sync.hf_bucket_sync import push, verify, RUNS

run = Path(sys.argv[1])
marker = run / "test-complete.json"
test = json.loads(marker.read_text())
selection = Path(test["selection"])
result = Path(test["result"])
assert file_sha256(selection) == test["selection_sha256"]
assert file_sha256(result) == test["result_sha256"]
best = json.loads(selection.read_text())["best"]["checkpoint"]
actor = Path(best["path"])
assert file_sha256(actor) == test["checkpoint_sha256"] == best["sha256"]
for artifact in (actor, selection, marker):
    archive_checkpoint(artifact)
from argparse import Namespace
args = Namespace(root=RUNS, runs=[result.parent])
push(args)
verify(args)

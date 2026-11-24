import json
import os
import re
import sys
from pathlib import Path
root = Path("/home/hice1/tsawada7/scratch/cs8803-drl-f26_vineppo").resolve()
receipt = root / "handoff-20261120/pruned-verified-checkpoints-20261121.jsonl"
logs = Path("artifacts/training/pace_ice/slurm")
prior = root / "handoff-20261120/cleanup-receipt.json"
removed = {os.path.relpath(row["path"], root): row["bytes"] for row in json.loads(prior.read_text())["files"]}
if receipt.exists():
    for line in receipt.read_text().splitlines():
        row = json.loads(line)
        removed[row["path"]] = row["bytes"]
proofs = {}
for job in (7208659, 7209032, 7224877, 7225330, 7226701, 7226702):
    log, = logs.glob(f"*-{job}.out")
    for match in re.finditer(r"^verified (training/pace_ice/[a-z0-9][a-z0-9-]*): (\d+) files, (\d+) bytes$", log.read_text(), re.M):
        path, count, size = match.groups()
        proofs[path] = (int(count), int(size), job)
selected = []
for prefix, (count, size, job) in sorted(proofs.items()):
    run = root / prefix
    assert run.resolve() == run and run.is_dir(), run
    files = [p for p in run.rglob("*") if p.is_file() and "wandb" not in p.relative_to(run).parts[:-1]]
    checkpoints = [p for p in files if p.suffix == ".pt" and "checkpoints" in p.relative_to(run).parts[:-1]]
    if not checkpoints:
        continue
    taken = [b for p,b in removed.items() if p.startswith(prefix + "/")]
    assert len(files) + len(taken) == count and sum(p.stat().st_size for p in files) + sum(taken) == size, (prefix, count, size)
    for path in checkpoints:
        assert path.resolve() == path and not path.is_symlink(), path
        selected.append((path, {"path":str(path.relative_to(root)), "bytes":path.stat().st_size, "verified_job":job, "run":prefix}))
print(len(selected), sum(row["bytes"] for _,row in selected), "bytes", len({row["run"] for _,row in selected}), "runs")
if sys.argv[1:] == ["--apply"]:
    with receipt.open("a") as stream:
        for path, row in selected:
            path.unlink()
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(receipt)

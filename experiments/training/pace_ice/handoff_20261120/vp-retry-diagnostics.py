from pathlib import Path
import os
import shlex
import subprocess

repo = Path.cwd().resolve()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
pinned = scratch / "worktrees/1c7ea5d"
logs = repo / "artifacts/training/pace_ice/slurm"
env = os.environ | {
    "PYTHON": str(repo / ".venv/bin/python"),
    "VLLM_PYTHON": str(repo / ".venv-inference-glibc2.34/bin/python"),
    "TRAINING_CACHE": str(scratch / "training-cache"),
    "PACE_CHAIN_LOG_DIR": str(logs),
    "TMPDIR": "/tmp",
}
env = {key: v for key, v in env.items() if not key.startswith(("SLURM_", "SBATCH_"))}
def run(*args):
    return subprocess.check_output(args, cwd=pinned, env=env, text=True).strip()
def submit(*args):
    value = run("sbatch", "--parsable", *args).split(";")[0]
    assert value.isdecimal()
    return value

assert run("git", "rev-parse", "--short", "HEAD") == "1c7ea5d"
assert not run("git", "status", "--porcelain")
assert run("sacct", "-X", "-n", "-P", "-j", "7225658", "--format=State,ExitCode") == "COMPLETED|0:0"
quota, = [line.split() for line in run("lfs", "quota", "-u", os.environ["USER"], "/storage/ice1").splitlines() if line.split()[:1] == ["/storage/ice1"]]
assert (int(quota[3]) - int(quota[1].rstrip("*"))) / 1024**2 >= 20 + 10 + 32 + 2 + 10 + 10

options = [f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err"]
script = "experiments/training/pace_ice/train_resumable.slurm"
for prior, old, new, name in (
    ("7224875", "handoff-d9-auto-warning-03", "handoff-d9-auto-warning-04", "vp26-d9-auto-warning-v4"),
    ("7210896", "handoff-frozen-test-defer-01", "handoff-frozen-test-defer-02", "vp26-frozen-test-defer-v2"),
):
    line = run("sacct", "-X", "-n", "-P", "-j", prior, "--format=SubmitLine%5000")
    original = shlex.split(line)
    assert original[0] == "sbatch" and script in original
    idx = original.index(script)
    output = Path(original[idx + 1])
    assert output.name == old
    output = output.with_name(new)
    assert not output.exists()
    request = [arg for arg in original[:idx] if arg.startswith(("--time=", "--constraint="))]
    assert len(request) == 2
    flags = [arg if arg != old else new for arg in original[idx+2:]]
    jid = submit(*request, *options, f"--job-name={name}", script, str(output), *flags)
    print(name, jid, flush=True)
    if "frozen" not in name:
        continue
    validation = output / "attempt-0000/evaluations/validation-2"
    command = (
        f"set -eu\nwhile test ! -f {validation}/progress.json; do sleep 1; done\n"
        f"test ! -f {validation}/result.json\nscancel --batch --signal=USR1 {jid}"
    )
    print("injector", submit("--partition=ice-cpu", "--account=math", "--qos=coc-ice",
                             "--cpus-per-task=1", "--mem=1G", "--time=00:20:00", *options,
                             f"--dependency=after:{jid}", "--kill-on-invalid-dep=yes",
                             f"--job-name=vp26-frozen-test-inject-{jid}", "--wrap=" + command), flush=True)

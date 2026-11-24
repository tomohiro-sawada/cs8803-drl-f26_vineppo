from pathlib import Path
import json
import os
import shlex
import subprocess
import time

repo = Path.cwd().resolve()
scratch = Path("/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo")
pinned = scratch / "worktrees/8052fab"
logs = repo / "artifacts/training/pace_ice/slurm"
old_control = scratch / "handoff-20261120/control-07"
control = old_control.with_name("control-08")
model = scratch / "cache/math/models/Olmo-3-7B-Instruct-SFT-e1452fc"
out = scratch / "training/pace_ice"
python = repo / ".venv/bin/python"
env = os.environ | {
    "PYTHON": str(python),
    "VLLM_PYTHON": str(repo / ".venv-inference-glibc2.34/bin/python"),
    "TRAINING_CACHE": str(scratch / "training-cache"),
    "PACE_CHAIN_LOG_DIR": str(logs),
    "TMPDIR": "/tmp",
    "WANDB_BASE_URL": "https://api.forge.coreweave.com",
}
env = {key: value for key, value in env.items() if not key.startswith(("SLURM_", "SBATCH_"))}

def run(*args):
    return subprocess.check_output(args, cwd=pinned, env=env, text=True).strip()

def submit(*args):
    job = run("sbatch", "--parsable", *args).split(";")[0]
    assert job.isdecimal()
    return job

def read(directory, name):
    return json.loads((directory / name).read_text())

def write(name, data):
    target = control / name
    pending = target.with_suffix(".pending.json")
    pending.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    pending.rename(target)

assert run("git", "rev-parse", "--short", "HEAD") == "8052fab"
assert not run("git", "status", "--porcelain")
assert run("sacct", "-X", "-n", "-P", "-j", "7226484", "--format=State,ExitCode") == "FAILED|1:0"
assert not (out / "handoff-olmo3-math-sft-03/COMPLETE").exists()
quota, = [line.split() for line in run("lfs", "quota", "-u", os.environ["USER"], "/storage/ice1").splitlines() if line.split()[:1] == ["/storage/ice1"]]
assert (int(quota[3]) - int(quota[1].rstrip("*"))) / 1024**2 > 20
pream = shlex.join(["env", f"PYTHONPATH={pinned}/src:{pinned}", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", str(python)])
probe = r"""
import json
from pathlib import Path
import sys
from transformers import AutoTokenizer
from core.prompt_protocol import OLMO3, prompt_protocol
model = Path(sys.argv[1])
t = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
eos = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
assert eos == [100265, 100257]
assert 100264 not in t.all_special_ids and 100265 not in t.all_special_ids
for task in ("math", "gsm8k"):
    p = prompt_protocol(OLMO3, task)
    assert p.eos_ids(t) == eos and p.context_special_ids(t) == (100264, *eos)
    prompt, = p.encode(t, [p.format(t, "What is two plus two?")])
    assert 100264 in prompt and 100265 in prompt
print(json.dumps({"added_chat_start": 100264, "generation_stop": eos, "tasks": ["math", "gsm8k"]}), flush=True)
""".strip()
wrapped = f"set -eu\n{pream} - {shlex.quote(str(model))} <<'PY'\n{probe}\nPY\n{pream} -m pytest -q --tb=short"
cpu = submit("--partition=ice-cpu", "--account=math", "--qos=coc-ice",
             "--cpus-per-task=4", "--mem=16G", "--time=00:30:00",
             "--job-name=vp26-olmo-stop-cpu", f"--output={logs}/%x-%j.out",
             f"--error={logs}/%x-%j.err", "--wrap=" + wrapped)
print("cpu", cpu, flush=True)
old_plan = read(old_control, "wave-plan.json")
control.mkdir()
run(str(python), "-m", "experiments.training.pace_ice.wave_plan", "--repo", str(repo), "--scratch", str(scratch),
    "--control", str(control), "--chain-smoke", str(out / "handoff-production-chain-smoke-03"),
    "--ppo-smoke", str(out / "handoff-smoke-ppo-06"),
    "--olmo-acceptance", str(out / "olmo3-sft-sampler-acceptance-06"), "--olmo-model", str(model),
    "--deadline", str(old_plan["deadline"]), "--cpu-job", cpu, "--suffix", "04")
new = read(control, "wave-plan.json")
updates = {
    j["name"].removesuffix("-04"): j | {"afterok": ["olmo-cpu"]}
    for j in new["jobs"] if "olmo3-" in j["name"]
}
assert len(updates) == 12
assert {j["name"].removesuffix("-03") for j in old_plan["jobs"] if "olmo3-" in j["name"]} == updates.keys()
old_watchers = {"7225663", "7225697"}
queue = set(run("squeue", "--me", "-h", "-o", "%i").splitlines())
assert old_watchers <= queue
run("scancel", *sorted(old_watchers))
for _ in range(30):
    if not (set(run("squeue", "--me", "-h", "-o", "%i").splitlines()) & old_watchers):
        break
    time.sleep(1)
else:
    raise RuntimeError("Old watcher did not stop")
state = read(old_control, "watch-state.json")
assert {k: v for k, v in state["jobs"].items() if "olmo3-" in k} == {"vp26-olmo3-math-sft-03": "7226484"}
state["jobs"].pop("vp26-olmo3-math-sft-03")
state["jobs"]["olmo-cpu"] = cpu
old_plan["jobs"] = [updates.get(j["name"].removesuffix("-03"), j) for j in old_plan["jobs"]]
write("wave-plan.json", old_plan)
write("watch-state.json", state)
jid = submit(f"--output={control}/controller-%j.log", f"--error={control}/controller-%j.err",
             "experiments/training/pace_ice/watch_resumable.slurm", str(control / "wave-plan.json"))
print("watcher", jid, flush=True)

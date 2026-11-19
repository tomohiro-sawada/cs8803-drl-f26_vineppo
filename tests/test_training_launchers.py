"""Execute real wrappers with fake srun/python; validate argv with the real parser."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "experiments/training"
spec = importlib.util.spec_from_file_location(
    "launcher_training_runner", TRAINING / "run.py"
)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.fixture
def launch(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Run only the stdlib containment preflight, never run.py or any model code.
    stub = (
        f"#!{sys.executable}\n"
        + """import json, os, pathlib, sys
if pathlib.Path(sys.argv[0]).name == "srun":
    pathlib.Path(os.environ["SRUN_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))
    assert sys.argv[1] == "--ntasks=1"
    os.execv(sys.argv[2], sys.argv[2:])
elif sys.argv[1] == "-c":
    code = sys.argv[2]
    sys.argv = ["-c", *sys.argv[3:]]
    exec(compile(code, "<launcher-preflight>", "exec"))
else:
    assert sys.argv[1] == "experiments/training/run.py"
    pathlib.Path(os.environ["PYTHON_CAPTURE"]).write_text(json.dumps({
        "argv": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd(),
    }))
"""
    )
    for name in ("srun", "python"):
        path = bin_dir / name
        path.write_text(stub)
        path.chmod(0o755)
    model, dataset = tmp_path / "model with spaces", tmp_path / "dataset with spaces"
    model.mkdir()
    dataset.mkdir()
    output = tmp_path / "new output"
    cache = tmp_path / "shared cache"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("SLURM_", "TRAINING_", "HF_", "PYTHON", "OMP_", "TOKENIZERS_")
        )
    }
    env.update(
        PATH=f"{bin_dir}:/usr/bin:/bin",
        PYTHON=str(bin_dir / "python"),
        SLURM_SUBMIT_DIR=str(ROOT),
        SLURM_CPUS_PER_TASK="6",
        TRAINING_CACHE=str(cache),
        SRUN_CAPTURE=str(tmp_path / "srun.json"),
        PYTHON_CAPTURE=str(tmp_path / "python.json"),
    )
    args = [
        "--task",
        "gsm8k",
        "--model-path",
        str(model),
        "--dataset-path",
        str(dataset),
        "--output",
        str(output),
    ]

    def run(cluster="skynet", mode="benchmark", extra=(), omit=()):
        argv = [
            item
            for index in range(0, len(args), 2)
            if args[index] not in omit
            for item in args[index : index + 2]
        ]
        return subprocess.run(
            ["bash", str(TRAINING / cluster / f"{mode}.slurm"), *argv, *extra],
            env=env,
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    return SimpleNamespace(
        run=run, env=env, output=output, cache=cache, model=model, dataset=dataset
    )


@pytest.mark.parametrize(
    "cluster,gpu", [("skynet", "A40"), ("pace_ice", "H100"), ("pace_ice", "H200")]
)
@pytest.mark.parametrize("mode", ["train", "benchmark"])
def test_wrapper_defaults_are_valid_runner_argv_and_offline(launch, cluster, gpu, mode):
    extra = [] if cluster == "skynet" else ["--expected-gpu", gpu]
    result = launch.run(cluster, mode, extra)
    assert result.returncode == 0, result.stderr
    captured = json.loads(Path(launch.env["PYTHON_CAPTURE"]).read_text())
    srun = json.loads(Path(launch.env["SRUN_CAPTURE"]).read_text())
    assert srun == ["--ntasks=1", launch.env["PYTHON"], *captured["argv"]]
    assert captured["cwd"] == str(ROOT)
    args = runner.parse_args(captured["argv"][1:])
    assert args.wandb_mode == "online"
    assert (args.mode, args.algorithm, args.expected_gpu, args.dtype) == (
        mode,
        "vineppo",
        gpu,
        "float32",
    )
    assert (args.finetuning, args.lora_rank, args.lora_alpha, args.learning_rate) == (
        "full",
        None,
        None,
        1e-6,
    )
    assert (args.num_iterations, args.warmup_iterations, args.checkpoint_every) == (
        (100, 0, 10) if mode == "train" else (3, 1, None)
    )
    assert (args.prompts_per_iteration, args.num_samples, args.mc_num_samples) == (
        2,
        2,
        2,
    )
    assert (args.minibatch_size, args.microbatch_size, args.keep_checkpoints) == (
        2,
        1,
        2,
    )
    assert args.model_path == launch.model.resolve()
    assert args.dataset_path == launch.dataset.resolve()
    assert args.output == launch.output.resolve()
    assert not launch.output.exists()
    env = captured["env"]
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONUNBUFFERED"):
        assert env[name] == "1"
    assert env["OMP_NUM_THREADS"] == "6" and env["TOKENIZERS_PARALLELISM"] == "false"
    for name, suffix in (
        ("HF_HOME", "huggingface"),
        ("XDG_CACHE_HOME", "xdg"),
        ("TORCH_HOME", "torch"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        assert env[name] == str(launch.cache / suffix)
        assert Path(env[name]).is_dir()


@pytest.mark.parametrize(
    "cluster,algorithm",
    [
        (cluster, algorithm)
        for cluster in ("skynet", "pace_ice")
        for algorithm in ("vineppo", "ppo", "grpo", "rloo")
        if (cluster, algorithm) != ("skynet", "ppo")
    ],
)
@pytest.mark.parametrize("mode", ["train", "benchmark"])
@pytest.mark.parametrize("learning_rate", [None, "2e-7"])
def test_explicit_lora_options_and_supported_algorithms(
    launch, cluster, algorithm, mode, learning_rate
):
    result = launch.run(
        cluster,
        mode,
        extra=[
            "--expected-gpu",
            "A40" if cluster == "skynet" else "H200",
            "--algorithm",
            algorithm,
            "--finetuning",
            "lora",
            "--lora-rank",
            "8",
            "--lora-alpha",
            "12.5",
            "--num-samples",
            "4",
            "--num-iterations",
            "7",
            "--warmup-iterations",
            "2",
            "--dtype",
            "bfloat16",
            "--checkpoint-every",
            "4",
        ]
        + ([] if learning_rate is None else ["--learning-rate", learning_rate]),
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads(Path(launch.env["PYTHON_CAPTURE"]).read_text())
    args = runner.parse_args(captured["argv"][1:])
    assert (
        args.algorithm,
        args.num_iterations,
        args.warmup_iterations,
        args.dtype,
        args.checkpoint_every,
    ) == (algorithm, 7, 2, "bfloat16", 4)
    assert (args.finetuning, args.lora_rank, args.lora_alpha, args.num_samples) == (
        "lora",
        8,
        12.5,
        4,
    )
    assert args.learning_rate == (
        1e-5 if learning_rate is None else float(learning_rate)
    )
    assert args.mode == mode


@pytest.mark.parametrize(
    "cluster,extra,message",
    [
        ("skynet", ["--algorithm", "ppo"], "PPO is not supported on Skynet"),
        ("skynet", ["--algorithm", "unknown"], "Unsupported algorithm"),
        ("skynet", ["--expected-gpu", "H100"], "must be A40"),
        ("pace_ice", [], "PACE requires --expected-gpu"),
        ("pace_ice", ["--expected-gpu", "A40"], "PACE requires --expected-gpu"),
        ("skynet", ["--mode", "train"], "Unsupported option"),
        ("skynet", ["--num-iterations"], "Missing value"),
        ("skynet", ["--task", "unknown"], "--task gsm8k|math is required"),
    ],
)
def test_launcher_rejects_invalid_options_before_srun(launch, cluster, extra, message):
    result = launch.run(cluster, extra=extra)
    assert result.returncode == 2 and message in result.stderr
    assert not Path(launch.env["SRUN_CAPTURE"]).exists()
    assert not launch.cache.exists() and not launch.output.exists()


@pytest.mark.parametrize(
    "option", ["--task", "--model-path", "--dataset-path", "--output"]
)
def test_explicit_paths_and_task_required(launch, option):
    result = launch.run(omit=[option])
    assert result.returncode == 2 and option in result.stderr
    assert not Path(launch.env["SRUN_CAPTURE"]).exists()


@pytest.mark.parametrize(
    "case",
    [
        "missing-model",
        "missing-dataset",
        "existing-output",
        "dangling-output",
        "missing-resume",
    ],
)
def test_local_path_checks(launch, tmp_path, case):
    extra = []
    if case == "missing-model":
        launch.model.rmdir()
    elif case == "missing-dataset":
        launch.dataset.rmdir()
    elif case == "existing-output":
        launch.output.mkdir()
    elif case == "dangling-output":
        launch.output.symlink_to(tmp_path / "nonexistent")
    else:
        extra = ["--resume", str(tmp_path / "missing.pt")]
    result = launch.run(extra=extra)
    assert result.returncode == 2
    assert not Path(launch.env["SRUN_CAPTURE"]).exists()


@pytest.mark.parametrize(
    "case",
    ["unset", "relative", "same-output", "inside-output", "symlink-inside-output"],
)
def test_cache_is_explicit_absolute_and_outside_output(launch, tmp_path, case):
    if case == "unset":
        del launch.env["TRAINING_CACHE"]
    elif case == "relative":
        launch.env["TRAINING_CACHE"] = "relative/cache"
    elif case == "same-output":
        launch.env["TRAINING_CACHE"] = str(launch.output)
    elif case == "inside-output":
        launch.env["TRAINING_CACHE"] = str(launch.output / "cache")
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        launch.env["TRAINING_CACHE"] = str(alias / launch.output.name / "cache")
    result = launch.run()
    assert result.returncode != 0 and "TRAINING_CACHE" in result.stderr
    assert not Path(launch.env["SRUN_CAPTURE"]).exists()
    assert not launch.output.exists()


@pytest.mark.parametrize("cluster", ["skynet", "pace_ice"])
@pytest.mark.parametrize("mode", ["train", "benchmark"])
def test_wrapper_resource_directives(cluster, mode):
    text = (TRAINING / cluster / f"{mode}.slurm").read_text()
    directives = {
        line.removeprefix("#SBATCH ")
        for line in text.splitlines()
        if line.startswith("#SBATCH ")
    }
    assert {"--nodes=1", "--ntasks=1", "--cpus-per-task=6"} <= directives
    if cluster == "skynet":
        assert {
            "--partition=kargo-lab",
            "--account=kargo-lab",
            "--gpus-per-node=a40:1",
        } <= directives
    else:
        assert {
            "--partition=ice-gpu",
            "--account=math",
            "--qos=coc-ice",
            "--gpus=1",
            "--constraint=gpu-h100|gpu-h200",
        } <= directives


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_tracking_flags_reach_runner(launch, mode):
    result = launch.run(
        extra=[
            "--wandb-mode",
            mode,
            "--wandb-project",
            "project",
            "--wandb-entity",
            "team",
            "--wandb-group",
            "group",
            "--wandb-name",
            "named-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads(Path(launch.env["PYTHON_CAPTURE"]).read_text())
    args = runner.parse_args(captured["argv"][1:])
    assert (
        args.wandb_mode,
        args.wandb_project,
        args.wandb_entity,
        args.wandb_group,
        args.wandb_name,
    ) == (mode, "project", "team", "group", "named-run")


def test_selection_options_reach_train_runner(launch):
    result = launch.run(
        mode="train",
        extra=[
            "--evaluation",
            "validation-test",
            "--validation-rounds",
            "4",
            "--eval-seed",
            "123",
            "--eval-batch-size",
            "2",
            "--eval-timeout-seconds",
            "3600",
        ],
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads(Path(launch.env["PYTHON_CAPTURE"]).read_text())
    args = runner.parse_args(captured["argv"][1:])
    assert (
        args.evaluation,
        args.validation_rounds,
        args.eval_seed,
        args.eval_batch_size,
        args.eval_timeout_seconds,
    ) == ("validation-test", 4, 123, 2, 3600)


def test_resume_selection_reaches_runner_and_missing_file_is_rejected(launch, tmp_path):
    checkpoint, manifest = tmp_path / "iteration-2.pt", tmp_path / "selection.json"
    extra = [
        "--evaluation",
        "validation-test",
        "--resume",
        str(checkpoint),
        "--resume-selection",
        str(manifest),
    ]
    checkpoint.write_bytes(b"checkpoint")
    missing = launch.run(mode="train", extra=extra)
    assert (
        missing.returncode == 2 and "Resume selection does not exist" in missing.stderr
    )
    manifest.write_text("{}")
    result = launch.run(mode="train", extra=extra)
    assert result.returncode == 0, result.stderr
    captured = json.loads(Path(launch.env["PYTHON_CAPTURE"]).read_text())
    args = runner.parse_args(captured["argv"][1:])
    assert (args.resume, args.resume_selection) == (
        checkpoint.resolve(),
        manifest.resolve(),
    )


def test_resumable_wrapper_resumes_latest_checkpoint_into_new_attempt(launch, tmp_path):
    root = tmp_path / "run"
    base = [
        "bash",
        str(TRAINING / "skynet/train_resumable.slurm"),
        str(root),
        "--task",
        "gsm8k",
        "--model-path",
        str(launch.model),
        "--dataset-path",
        str(launch.dataset),
    ]

    def run(extra=()):
        return subprocess.run(
            [*base, *extra],
            env=launch.env,
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def parsed():
        captured = json.loads(Path(launch.env["PYTHON_CAPTURE"]).read_text())
        return runner.parse_args(captured["argv"][1:])

    first = run()
    assert first.returncode == 0, first.stderr
    assert parsed().output == (root / "attempt-00").resolve()
    assert parsed().resume is None and parsed().mode == "train"
    for attempt, iterations in (("attempt-00", (2, 10)), ("attempt-01", (9, 10))):
        directory = root / attempt / "checkpoints"
        directory.mkdir(parents=True)
        for iteration in iterations:
            (directory / f"iteration-{iteration}.pt").write_bytes(b"checkpoint")
        (directory / "iteration-99.pt.tmp").write_bytes(b"partial")
    second = run()
    assert second.returncode == 0, second.stderr
    latest = (root / "attempt-01/checkpoints/iteration-10.pt").resolve()
    assert (parsed().output, parsed().resume) == (
        (root / "attempt-02").resolve(),
        latest,
    )
    assert parsed().resume_selection is None
    (root / "attempt-01/selection.json").write_text("{}")
    third = run(["--evaluation", "validation-test"])
    assert third.returncode == 0, third.stderr
    assert parsed().resume == latest
    assert parsed().resume_selection == (root / "attempt-01/selection.json").resolve()
    for option in ("--output", "--resume", "--resume-selection"):
        rejected = run([option, "value"])
        assert rejected.returncode == 2 and "chosen by this wrapper" in rejected.stderr

"""Installer isolation: never mutate the SDPA environment for inference."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "install", Path(__file__).resolve().parents[2] / "scripts/setup/install.py"
)
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)


@pytest.fixture
def run(monkeypatch):
    run = Mock()
    monkeypatch.setattr(install.subprocess, "run", run)
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(install.platform, "libc_ver", lambda: ("glibc", "2.31"))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/do/not/touch")
    return run


def test_default(run, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["install.py", "--dry-run"])
    install.main()
    run.assert_called_once()
    assert run.call_args.args[0] == ["uv", "sync", "--locked", "--dry-run"]
    assert run.call_args.kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(
        install.ROOT / ".venv"
    )


def test_inference(run, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["install.py", "--inference"])
    install.main()
    sync, verify = run.call_args_list
    command = sync.args[0]
    assert command[command.index("--no-group") + 1] == "sdpa"
    assert command[command.index("--group") + 1] == "inference"
    path = install.ROOT / ".venv-inference-glibc2.31"
    assert sync.kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(path)
    assert verify.args[0][0] == str(path / "bin/python")
    assert "from vllm import LLM, SamplingParams" in verify.args[0][-1]
    assert "flash_attn_2_cuda" in verify.args[0][-1]


def test_force_source(run, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["install.py", "--inference", "--build-flash-attn", "--dry-run"]
    )
    install.main()
    run.assert_called_once()
    assert run.call_args.kwargs["env"]["FLASH_ATTENTION_FORCE_BUILD"] == "TRUE"
    assert "--no-cache" in run.call_args.args[0]
    assert "--reinstall-package" in run.call_args.args[0]


def test_failure_propagates(run, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["install.py", "--inference"])
    run.side_effect = install.subprocess.CalledProcessError(1, "uv")
    with pytest.raises(install.subprocess.CalledProcessError):
        install.main()
    run.assert_called_once()


@pytest.mark.parametrize("args", [["--build-flash-attn"], ["--unknown"]])
def test_invalid_flags(run, monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["install.py", *args])
    with pytest.raises(SystemExit):
        install.main()
    run.assert_not_called()


def test_unsupported_platform(run, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["install.py", "--inference"])
    monkeypatch.setattr(install.platform, "system", lambda: "Darwin")
    with pytest.raises(SystemExit):
        install.main()
    run.assert_not_called()

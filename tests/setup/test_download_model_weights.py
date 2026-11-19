"""Offline downloader tests: real Arrow fixtures, mocked network boundaries."""

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/setup/download_model_weights.py"
spec = importlib.util.spec_from_file_location("download_model_weights", SCRIPT)
download = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = download
spec.loader.exec_module(download)
SHA = "a" * 40


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network access")

    for name in ("urlopen", "HfApi", "snapshot_download"):
        monkeypatch.setattr(download, name, forbidden)


@pytest.fixture
def paper_archive(tmp_path):
    archive = tmp_path / "paper.zip"
    with ZipFile(archive, "w") as outer:
        for task, member in download.DATASET_ARCHIVES.items():
            row = (
                {"question": "One plus one?", "answer": "Work\n#### 2"}
                if task == "gsm8k"
                else {"problem": "One plus one?", "solution": r"\boxed{2}"}
            )
            row["source_id"] = "original-id"
            data = DatasetDict(
                {s: Dataset.from_list([row]) for s in ("train", "validation", "test")}
            )
            saved = tmp_path / "fixtures" / task
            data.save_to_disk(saved)
            buffer = io.BytesIO()
            with ZipFile(buffer, "w") as inner:
                for path in saved.rglob("*"):
                    if path.is_file():
                        inner.write(
                            path, f"{task}/{path.relative_to(saved).as_posix()}"
                        )
            outer.writestr(f"wandb_export_root/datasets/{member}", buffer.getvalue())
    return archive


def mock_archive(monkeypatch, archive):
    mocked = Mock(side_effect=lambda target: target.write_bytes(archive.read_bytes()))
    monkeypatch.setattr(download, "download_datasets_archive", mocked)
    return mocked


def mock_models(monkeypatch):
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha=SHA)
    monkeypatch.setattr(download, "HfApi", lambda: api)

    def snapshot(**kwargs):
        kwargs["local_dir"].mkdir()
        (kwargs["local_dir"] / "config.json").write_text("{}")

    transfer = Mock(side_effect=snapshot)
    monkeypatch.setattr(download, "snapshot_download", transfer)
    return api, transfer


def test_full_download_pins_exact_models_and_preserves_datasets(
    tmp_path, monkeypatch, paper_archive
):
    fetch = mock_archive(monkeypatch, paper_archive)
    api, transfer = mock_models(monkeypatch)
    root = tmp_path / "output"
    download.main(
        download.DownloadConfig(download_root=root, gsm8k_model_revision="release")
    )
    assert fetch.call_count == 1
    assert [call.kwargs["repo_id"] for call in transfer.call_args_list] == [
        "realtreetune/rho-1b-sft-GSM8K",
        "realtreetune/rho-1b-sft-MATH",
    ]
    assert all(call.kwargs["revision"] == SHA for call in transfer.call_args_list)
    api.model_info.assert_any_call("realtreetune/rho-1b-sft-GSM8K", revision="release")
    manifest = json.loads((root / "download_manifest.json").read_text())
    assert len(manifest["artifacts"]) == 4
    for entry in manifest["artifacts"]:
        assert Path(entry["path"]).exists()
        if entry["kind"] == "model":
            assert entry["revision"] == SHA
        else:
            assert entry["sha256"] == download.DATASET_SHA256
            assert entry["release_asset_id"] == 196429828
            data = load_from_disk(entry["path"])
            assert set(data) == {"train", "validation", "test"}
            assert data["train"][0]["source_id"] == "original-id"
            assert entry["splits"]["train"]["rows"] == 1
    assert (
        load_from_disk(root / "datasets/gsm8k")["test"][0]["answer"] == "Work\n#### 2"
    )


@pytest.mark.parametrize("kind", ["models", "datasets"])
def test_selection_modes_and_paths(kind, tmp_path, monkeypatch, paper_archive):
    if kind == "models":
        _, transfer = mock_models(monkeypatch)
    else:
        mock_archive(monkeypatch, paper_archive)
    config = download.parse_args(
        [
            "--download-root",
            str(tmp_path / "root"),
            "--tasks",
            "math",
            f"--{kind}-only",
            "--model-download-root",
            str(tmp_path / "weights"),
            "--dataset-download-root",
            str(tmp_path / "data"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )
    download.main(config)
    entries = json.loads((tmp_path / "manifest.json").read_text())["artifacts"]
    assert len(entries) == 1 and entries[0]["task"] == "math"
    if kind == "models":
        assert transfer.call_count == 1
        assert Path(entries[0]["path"]) == tmp_path / "weights/rho-1b-sft-MATH"
    else:
        assert Path(entries[0]["path"]) == tmp_path / "data/math"


@pytest.mark.parametrize(
    "args", [["--models-only", "--datasets-only"], ["--tasks", "other"]]
)
def test_invalid_cli(args):
    with pytest.raises(SystemExit):
        download.parse_args(args)


def test_refuses_existing_output_before_network(tmp_path):
    destination = tmp_path / "datasets/gsm8k"
    destination.mkdir(parents=True)
    marker = destination / "keep"
    marker.write_text("untouched")
    with pytest.raises(FileExistsError):
        download.main(download.DownloadConfig(download_root=tmp_path))
    assert marker.read_text() == "untouched"


def test_model_failure_publishes_nothing(tmp_path, monkeypatch):
    _, transfer = mock_models(monkeypatch)
    transfer.side_effect = RuntimeError("network failed")
    with pytest.raises(RuntimeError, match="network failed"):
        download.main(download.DownloadConfig(download_root=tmp_path, models_only=True))
    assert not (tmp_path / "download_manifest.json").exists()
    assert list((tmp_path / "models").iterdir()) == []


def test_unresolved_revision_fails(tmp_path, monkeypatch):
    api, transfer = mock_models(monkeypatch)
    api.model_info.return_value.sha = "main"
    with pytest.raises(AssertionError, match="resolved model revision"):
        download.main(download.DownloadConfig(download_root=tmp_path, models_only=True))
    transfer.assert_not_called()


@pytest.mark.parametrize("corrupt", [False, True])
def test_archive_checksum(tmp_path, monkeypatch, corrupt):
    body = b"small offline payload"
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: io.BytesIO(body))
    monkeypatch.setattr(download, "DATASET_SIZE", len(body))
    monkeypatch.setattr(
        download,
        "DATASET_SHA256",
        "0" * 64 if corrupt else hashlib.sha256(body).hexdigest(),
    )
    if corrupt:
        with pytest.raises(ValueError, match="SHA-256"):
            download.download_datasets_archive(tmp_path / "archive.zip")
    else:
        download.download_datasets_archive(tmp_path / "archive.zip")
        assert (tmp_path / "archive.zip").read_bytes() == body


@pytest.mark.parametrize(
    "member", ["../escape", "/absolute", "math/../../escape", "wrong/file"]
)
def test_unsafe_archive_members_rejected(tmp_path, member):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as inner:
        inner.writestr(member, "bad")
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as outer:
        outer.writestr(
            "wandb_export_root/datasets/data-math-openai_split.zip", buffer.getvalue()
        )
    with pytest.raises(ValueError, match="Unsafe"):
        download.extract_dataset(archive, "math", tmp_path / "out")
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("malformed", ["splits", "columns", "empty"])
def test_malformed_saved_data_rejected(tmp_path, monkeypatch, paper_archive, malformed):
    row = {"problem": "Question", "solution": "Answer"}
    data = DatasetDict(
        {s: Dataset.from_list([row]) for s in ("train", "validation", "test")}
    )
    if malformed == "splits":
        del data["validation"]
    elif malformed == "columns":
        data["train"] = data["train"].remove_columns("solution")
    else:
        data["train"] = data["train"].select([])
    monkeypatch.setattr(download, "load_from_disk", lambda _: data)
    with pytest.raises(AssertionError):
        download.extract_dataset(paper_archive, "math", tmp_path / "out")


def test_later_download_failure_keeps_staged_models_unpublished(tmp_path, monkeypatch):
    mock_models(monkeypatch)
    monkeypatch.setattr(
        download,
        "download_datasets_archive",
        Mock(side_effect=ValueError("bad checksum")),
    )
    with pytest.raises(ValueError, match="checksum"):
        download.main(download.DownloadConfig(download_root=tmp_path))
    assert not (tmp_path / "download_manifest.json").exists()
    assert list((tmp_path / "models").iterdir()) == []


def test_missing_saved_dataset_fails(tmp_path):
    archive = tmp_path / "empty.zip"
    with ZipFile(archive, "w"):
        pass
    with pytest.raises(KeyError):
        download.extract_dataset(archive, "math", tmp_path / "out")

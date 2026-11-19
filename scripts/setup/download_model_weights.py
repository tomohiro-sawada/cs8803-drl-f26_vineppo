"""Download the two paper Rho checkpoints and original saved paper datasets.

No dataset reconstruction, replacement repositories, or remote code execution.
See README.md in this directory for provenance and CLI examples.
"""

import argparse
import hashlib
import io
import json
import re
import shutil
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

from datasets import DatasetDict, load_from_disk
from huggingface_hub import HfApi, snapshot_download
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "gsm8k": "realtreetune/rho-1b-sft-GSM8K",
    "math": "realtreetune/rho-1b-sft-MATH",
}
DATASET_ARCHIVES = {
    "gsm8k": "data-gsm8k-w_valid_split.zip",
    "math": "data-math-openai_split.zip",
}
UPSTREAM_COMMIT = "9f716e68bb934e71da86c226c1699c36b5553faf"
DATASET_URL = (
    "https://github.com/McGill-NLP/VinePPO/releases/download/latest/"
    "wandb_export_root.zip"
)
DATASET_ASSET_ID = 196429828
DATASET_SIZE = 8127691
DATASET_SHA256 = "c25d3977febe21260a1d6fe484ef368774e3d99ef10ef3f07a4531f4cd02a503"


@dataclass(frozen=True)
class DownloadConfig:
    download_root: Path = PROJECT_ROOT / ".cache"
    model_download_root: Path | None = None
    dataset_download_root: Path | None = None
    manifest: Path | None = None
    tasks: tuple[str, ...] = ("gsm8k", "math")
    models_only: bool = False
    datasets_only: bool = False
    gsm8k_model_revision: str = "main"
    math_model_revision: str = "main"


def download_datasets_archive(destination: Path) -> None:
    """Stream and verify the exact published asset; never trust the mutable tag."""
    digest = hashlib.sha256()
    size = 0
    with (
        urlopen(DATASET_URL, timeout=60) as response,
        destination.open("xb") as output,
        tqdm(
            total=DATASET_SIZE, unit="B", unit_scale=True, desc="Paper datasets"
        ) as bar,
    ):
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > DATASET_SIZE:
                raise ValueError("Paper dataset archive exceeds pinned size")
            digest.update(chunk)
            output.write(chunk)
            bar.update(len(chunk))
    if size != DATASET_SIZE or digest.hexdigest() != DATASET_SHA256:
        raise ValueError("Paper dataset archive does not match pinned size/SHA-256")


def extract_dataset(archive: Path, task: str, destination: Path) -> dict:
    """Preserve the upstream save_to_disk layout and validate it with datasets."""
    member = f"wandb_export_root/datasets/{DATASET_ARCHIVES[task]}"
    with ZipFile(archive) as outer, ZipFile(io.BytesIO(outer.read(member))) as inner:
        seen = set()
        for info in inner.infolist():
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in info.filename
                or not path.parts
                or path.parts[0] != task
                or stat.S_ISLNK(mode)
                or info.filename in seen
            ):
                raise ValueError(f"Unsafe or duplicate archive member: {info.filename}")
            seen.add(info.filename)
            target = destination.joinpath(*path.parts[1:])
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with inner.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    dataset = load_from_disk(str(destination))
    assert isinstance(dataset, DatasetDict), "Expected a saved DatasetDict"
    assert set(dataset) == {"train", "validation", "test"}, "Unexpected paper splits"
    required = {"question", "answer"} if task == "gsm8k" else {"problem", "solution"}
    splits = {}
    for name, split in dataset.items():
        assert len(split) > 0, f"Empty {task}/{name}"
        assert required <= set(split.column_names), f"Missing columns in {task}/{name}"
        splits[name] = {"rows": len(split), "columns": split.column_names}
    return {"archive_member": member, "splits": splits}


def main(config: DownloadConfig) -> None:
    assert config.tasks and len(set(config.tasks)) == len(config.tasks), (
        "Select unique tasks"
    )
    assert set(config.tasks) <= MODELS.keys(), "Unsupported task"
    assert not (config.models_only and config.datasets_only), "Conflicting modes"
    root = config.download_root.expanduser().resolve()
    models = (config.model_download_root or root / "models").expanduser().resolve()
    datasets = (
        (config.dataset_download_root or root / "datasets").expanduser().resolve()
    )
    manifest = (
        (config.manifest or root / "download_manifest.json").expanduser().resolve()
    )
    jobs = []
    for task in config.tasks:
        if not config.datasets_only:
            jobs.append(("model", task, models / MODELS[task].split("/")[1]))
        if not config.models_only:
            jobs.append(("dataset", task, datasets / task))
    paths = [destination for _, _, destination in jobs] + [manifest]
    for i, path in enumerate(paths):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite {path}; choose a fresh destination"
            )
        for other in paths[i + 1 :]:
            assert (
                path != other
                and path not in other.parents
                and other not in path.parents
            ), f"Overlapping output paths: {path}, {other}"

    entries = []
    with ExitStack() as stack:
        archive = None
        staged = []
        # All payloads are staged and validated before any final output is published.
        for kind, task, destination in tqdm(jobs, desc="Artifacts", unit="artifact"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                stack.enter_context(TemporaryDirectory(dir=destination.parent))
            )
            payload = temporary / "payload"
            entry = {"kind": kind, "task": task, "path": str(destination)}
            if kind == "model":
                repo = MODELS[task]
                requested = getattr(config, f"{task}_model_revision")
                revision = HfApi().model_info(repo, revision=requested).sha
                assert isinstance(revision, str) and re.fullmatch(
                    r"[0-9a-f]{40}", revision
                ), f"Invalid resolved model revision: {revision!r}"
                tqdm.write(f"Downloading {repo}@{revision}")
                snapshot_download(repo_id=repo, revision=revision, local_dir=payload)
                entry.update(
                    repo_id=repo, requested_revision=requested, revision=revision
                )
            else:
                if archive is None:
                    archive = temporary / "paper-datasets.zip"
                    download_datasets_archive(archive)
                entry.update(extract_dataset(archive, task, payload))
                entry.update(
                    url=DATASET_URL,
                    release_asset_id=DATASET_ASSET_ID,
                    sha256=DATASET_SHA256,
                    size_bytes=DATASET_SIZE,
                )
            staged.append((payload, destination))
            entries.append(entry)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        for payload, destination in staged:
            if destination.exists():
                raise FileExistsError(f"Output appeared during download: {destination}")
            payload.rename(destination)
        # Publish provenance only after every payload has reached its final path.
        with manifest.open("x") as output:
            json.dump(
                {
                    "schema_version": 1,
                    "upstream_commit": UPSTREAM_COMMIT,
                    "artifacts": entries,
                },
                output,
                indent=2,
            )
            output.write("\n")
    print(f"Downloads complete. Manifest: {manifest}")


def parse_args(argv: list[str] | None = None) -> DownloadConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, default=PROJECT_ROOT / ".cache")
    parser.add_argument("--model-download-root", type=Path)
    parser.add_argument("--dataset-download-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(MODELS), default=list(MODELS)
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--models-only", action="store_true")
    mode.add_argument("--datasets-only", action="store_true")
    for task in MODELS:
        parser.add_argument(f"--{task}-model-revision", default="main")
    args = vars(parser.parse_args(argv))
    args["tasks"] = tuple(args["tasks"])
    return DownloadConfig(**args)


if __name__ == "__main__":
    main(parse_args())

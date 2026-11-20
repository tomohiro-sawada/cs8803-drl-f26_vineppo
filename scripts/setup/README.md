# Experiment prerequisites

From the repository root (after `uv sync`):

```sh
uv run python scripts/setup/download_model_weights.py
```

This downloads **only these model repositories**, including tokenizer/config files:

- `realtreetune/rho-1b-sft-GSM8K`
- `realtreetune/rho-1b-sft-MATH`

It also downloads the original VinePPO paper dataset export and extracts GSM8K
and MATH without transforming rows, selecting subsets, or recreating splits.
Model weights are large; allow several GB of storage and network transfer. No
GPU is needed. Downloads use `huggingface_hub` (the same implementation as
`hf download`), already provided by the project's dependencies. Native Hub
progress bars, a byte-progress bar for the dataset archive, and an artifact
progress bar are displayed. No dependency-file changes are needed.

## Paths and selection

Defaults are rooted at the repository's `.cache`, independent of working directory:

```text
.cache/
  models/rho-1b-sft-GSM8K/
  models/rho-1b-sft-MATH/
  datasets/gsm8k/
  datasets/math/
  download_manifest.json
```

Choose an output root or override the model and dataset roots independently:

```sh
uv run python scripts/setup/download_model_weights.py --download-root /scratch/vineppo
uv run python scripts/setup/download_model_weights.py \
  --tasks math --datasets-only \
  --dataset-download-root /scratch/paper-data \
  --manifest /scratch/math-data-manifest.json
uv run python scripts/setup/download_model_weights.py \
  --tasks gsm8k --models-only \
  --model-download-root /scratch/paper-models \
  --manifest /scratch/gsm8k-model-manifest.json
```

`--tasks` accepts `gsm8k`, `math`, or both; both are selected by default.
`--models-only` and `--datasets-only` are mutually exclusive. Supplied relative
paths are relative to the current working directory; `~` is expanded. Separate
invocations should use separate manifests and nonexisting artifact destinations.

Public downloads normally require no token. If authentication is needed, use
`uv run hf auth login` or the standard `HF_TOKEN` environment variable. Tokens
are not recorded in the manifest. Network requests go to Hugging Face and GitHub.

## Exact paper dataset format and provenance

Source inspection is anchored at VinePPO commit
`9f716e68bb934e71da86c226c1699c36b5553faf`:

- [Download script](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/scripts/download_and_prepare_dataset.sh)
  points to release `latest`, asset `wandb_export_root.zip` (asset ID `196429828`).
- [Task loader](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/tasks/base_task.py)
  calls `DatasetDict.load_from_disk`, not `load_dataset` on raw Hub data.
- The nested archives are `data-gsm8k-w_valid_split.zip` and
  `data-math-openai_split.zip`, inside `wandb_export_root/datasets/`.
  Both contain `train`, `validation`, and `test` saved Arrow splits.
- The export also contains CollegeMath and OlympiadBench archives; those are not
  extracted by this two-task setup script. The optional MATH few-shot dataset is
  not required by the paper's `prepend_in_context_few_shot: false` configuration.

The 8,127,691-byte release archive is pinned by SHA-256:
`c25d3977febe21260a1d6fe484ef368774e3d99ef10ef3f07a4531f4cd02a503`.
The download fails if this mutable release URL serves different bytes. It never
substitutes a Hub dataset, random validation split, or alternative model.

Local `src/core/data.py` currently defines `Example` records but no disk loader.
The setup preserves the upstream saved format for subsequent normalization:

```python
from datasets import load_from_disk

paper = load_from_disk(".cache/datasets/gsm8k")
train = paper["train"]
```

GSM8K retains `question` and the full worked `answer` (including `####`). MATH
retains `problem` and `solution`. All additional columns and source row order are
preserved. Converting these to `Example.reference_answer` and formatting prompts
remain the runtime loader's responsibility, not this downloader's.

## Pinning, failures, and reruns

Each model revision defaults to `main`, which is resolved to a full commit SHA
**before** downloading. The download uses that SHA, not the moving branch.
`download_manifest.json` records requested/resolved revisions, absolute output
paths, dataset URL, asset ID, checksum, archive members, split sizes, and columns.
To reproduce a model download, pass the recorded SHA via
`--gsm8k-model-revision` and/or `--math-model-revision` and use fresh paths.
The dataset bytes are always pinned to the checksum above.

Existing outputs or a manifest cause an error before network access. All
payloads are staged alongside their destinations, and datasets are validated
with `load_from_disk` before publication. Download/validation failures remove
staging directories and propagate the original error. The manifest is written
only after all artifacts are published. Publication across multiple paths is
not a filesystem transaction: an interruption during final renames/manifest
writing can leave final artifacts without a complete manifest. Inspect those
outputs and choose fresh paths; there is no silent repair, overwrite, fallback,
or resume mode. Do not run concurrent downloads targeting the same paths.

## Focused offline tests

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  uv run pytest tests/setup/test_download_model_weights.py -q
```

Tests use tiny real saved Arrow datasets, nested ZIP fixtures, and mocked network
boundaries. No model downloads, GPU, or credentials are required.

Use `scripts/setup/install.py --inference` for the pinned vLLM environment.
Use `flash_attention_installation.sh` for the PACE FlashAttention build.

# Paper artifact replications

`evaluate_sft.py` evaluates the **initial RhoMath 1.1B SFT** checkpoints underlying
Figure 4 of [VinePPO, arXiv:2410.01679v2](https://arxiv.org/html/2410.01679v2#S4.F4).
The figure's initial-SFT labels are **GSM8K 40.3%** and **MATH 15.5%** (verified
against the v2 PDF). These are comparison targets, not thresholds asserted by the
script and not newly measured results. No training code is changed.

## Prepare and run

Run from the repository root with the project environment installed. The runner
requires CUDA and BF16 support; it uses only logical `cuda:0`. Under Slurm, leave
`CUDA_VISIBLE_DEVICES` as assigned by the scheduler. There is no CPU/MPS fallback. A40 is optional
hardware validation, **not** the paper's original hardware requirement.

Prepare artifacts with the existing downloader, on a network-enabled machine:

```sh
uv run python scripts/setup/download_model_weights.py \
  --gsm8k-model-revision b28fda3216f8178c1f37c3c764196e3ca2439a30 \
  --math-model-revision f4d0f86bfce159f0042cca9116a52c0b972f34aa
```

This defaults to `.cache/download_manifest.json`, models under
`.cache/models/rho-1b-sft-GSM8K` and `.cache/models/rho-1b-sft-MATH`, and saved
DatasetDicts under `.cache/datasets/gsm8k` and `.cache/datasets/math`.
**Do not rerun the downloader into an existing destination.** If artifacts already
exist, use their manifest. Custom downloader paths work through `--manifest`;
model/data paths are taken from its absolute `path` entries. A manifest must
contain both artifacts for the selected task. If moving artifacts between
machines, update their absolute paths explicitly and preserve revision/checksum
fields; the run records the resulting manifest hash.

Full evaluation commands (run within a Slurm GPU allocation):

```sh
uv run --offline python scripts/paper_artifact_replications/evaluate_sft.py \
  --task gsm8k --output artifacts/sft-gsm8k --batch-size 8 --rounds 16 \
  --seed 42 --expected-gpu A40

uv run --offline python scripts/paper_artifact_replications/evaluate_sft.py \
  --task math --output artifacts/sft-math --batch-size 8 --rounds 16 \
  --seed 42 --expected-gpu A40
```

Remove `--expected-gpu A40` to allow another BF16-capable CUDA GPU. Change
`--manifest` for a nondefault download location. Output directories must not
exist. For a small **GPU smoke run**, explicitly opt in:

```sh
uv run --offline python scripts/paper_artifact_replications/evaluate_sft.py \
  --task gsm8k --output artifacts/sft-gsm8k-smoke \
  --batch-size 2 --rounds 1 --smoke --limit 4 --seed 42
```

`--limit` selects the first N test rows in stored order, retaining their original
row-index IDs; it is rejected without `--smoke`. Limits must be positive and cannot
exceed the split. Smoke metrics are not paper scores. Changing `--rounds` from 16
also sets `paper_protocol_size=false`. Do not run with Python `-O`: validation uses
assertions intentionally. OOM, missing data, malformed rows, incompatible models,
and verifier failures propagate; there is no automatic batch-size reduction,
replacement dataset, download, or fabricated score.

The runner sets `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` before importing HF
libraries, uses `local_files_only=True`, disables remote model code, and reads
only `load_from_disk` datasets. `uv --offline` additionally prevents dependency
resolution from using the network; install dependencies beforehand.

## SkyNet Slurm launcher

`evaluate_sft.slurm` lives beside this runner, not in `tmp` or the training code.
Submit **from the repository root**, after creating the log directory:

```sh
mkdir -p artifacts/evaluation/slurm
sbatch scripts/paper_artifact_replications/evaluate_sft.slurm \
  --task gsm8k --output artifacts/evaluation/rho-sft-gsm8k-16rounds --batch-size 32
sbatch scripts/paper_artifact_replications/evaluate_sft.slurm \
  --task math --output artifacts/evaluation/rho-sft-math-16rounds --batch-size 32
```

The launcher requests one A40, six CPUs, 32 GB host memory, and four hours on
`kargo-lab`/`debug`. It uses the existing `.venv`, not dependency installation in
a GPU job. Logs go to `artifacts/evaluation/slurm/`; results go to the explicit
`--output`. Override account/partition/QOS on `sbatch` for another allocation.
For overcap, specify **both** `--partition=overcap --account=overcap`; jobs may be
preempted, and this runner preserves completed records but does not resume.

Follow the [SkyNet wiki](https://github.com/GeorgiaTechSkynet/userWiki): downloads,
archive extraction, and tests also belong in CPU Slurm allocations, not the login
node. Keep large caches on project storage. Do not copy the PACE-specific account,
partition, or GPU constraints from `tmp/standard_lora.slurm` to SkyNet.

## Verified prompting and decoding

Upstream source is pinned to
[`9f716e68bb934e71da86c226c1699c36b5553faf`](https://github.com/McGill-NLP/VinePPO/tree/9f716e68bb934e71da86c226c1699c36b5553faf).
Relevant original files:

- [`configs/sft_rho1b_for_gsm8k_eval.jsonnet`](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/configs/sft_rho1b_for_gsm8k_eval.jsonnet)
- [`configs/sft_rho1b_for_MATH_eval.jsonnet`](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/configs/sft_rho1b_for_MATH_eval.jsonnet)
- [`configs/prompt_library/generic_GSM8K_step_by_step.jsonnet`](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/configs/prompt_library/generic_GSM8K_step_by_step.jsonnet)
  and [`generic_MATH_step_by_step.jsonnet`](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/configs/prompt_library/generic_MATH_step_by_step.jsonnet)
- [`EfficientIIDExpander`](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/inference_strategies/tree_inference/expansion.py)
  for context-dependent token budgets.

Both tasks use exactly this zero-shot template (no newline after `Solution:`):

```text
[MATH_TASK] Problem:
{question}

Solution:
```

No chat template, demonstrations, gold solution, or extra instruction is added.
GSM8K questions are stripped as in upstream preprocessing; MATH problems are
preserved. The checkpoint tokenizer adds BOS. Decoding samples one response per
question per round with temperature **0.35**, at most **1024 new tokens**,
**top_p=0.9**, no top-k filter, EOS stopping, and stop text `\n\n\nProblem:`.
[Appendix C.4](https://arxiv.org/html/2410.01679v2#A3.SS4) specifies temperature,
length, and averaging 16 seeded rounds; the configs additionally specify nucleus
sampling and stop text. These are evaluation settings, not the training recipe.

The runner reuses `core.inference.HFInference` for token generation, left padding,
attention masks, and finish reasons. A small evaluation-only model facade changes
only its explicit `top_p=1.0` argument to `0.9`; the core training behavior remains
untouched. `runtime.evaluate.evaluate` owns verification and sample records, using
`core.data.Example`. Accuracy is the mean of per-round correct fractions, also
pooled sample accuracy because every round uses the same test set. It is **not
pass@16, any-hit accuracy, or majority voting**. Length-exhausted responses are
scored from their text, without a training truncation penalty.

## Local provenance and dataset selection

Pinned model cards:

- [GSM8K, `b28fda3216f8178c1f37c3c764196e3ca2439a30`](https://huggingface.co/realtreetune/rho-1b-sft-GSM8K/blob/b28fda3216f8178c1f37c3c764196e3ca2439a30/README.md)
- [MATH, `f4d0f86bfce159f0042cca9116a52c0b972f34aa`](https://huggingface.co/realtreetune/rho-1b-sft-MATH/blob/f4d0f86bfce159f0042cca9116a52c0b972f34aa/README.md)

The cards identify the Rho SFT checkpoints/base model and link the paper, but are
largely placeholders: **they do not establish the prompt or sampling recipe**.
Those come from the pinned original configs above. Other model revisions are
rejected rather than silently treated as the same experiment.

The downloader's original release asset is pinned by SHA-256
`c25d3977febe21260a1d6fe484ef368774e3d99ef10ef3f07a4531f4cd02a503`.
The runner validates that checksum in the manifest and the actual saved test split
sizes: GSM8K **1,319**, OpenAI-split MATH **500** (not canonical 5,000-row MATH).
No split reconstruction or shuffling occurs. GSM8K references are extracted from
exactly one `####`; MATH preserves the released `solution` and passes
`reference_is_solution=True` through evaluation to the shared verifier. This
extracts all reference answers, retaining upstream multi-answer matching rather
than reducing to the final `answer` column. The default verifier/evaluator flag
remains false for existing final-answer callers. IDs are `task:test:original-row-index`.

## Output and failure semantics

Each output directory contains:

- `config.json`: resolved run settings, prompt, decoding, hardware selection, seed
  policy, and smoke flag.
- `provenance.json`: selected download-manifest entries, manifest and normalized
  selection SHA-256, runner SHA-256, local git commit, package/CUDA versions, GPU
  name, pinned upstream/verifier provenance, `reference_is_solution`, and explicit deviations.
- `round-00/records.jsonl`, etc.: exact question/prompt/response, extracted and
  references (full solutions for MATH), correctness, finish reason, original example ID, round, and
  seed. `sample_index` is always zero; round distinguishes repeated attempts.
- `round-00/metrics.json`, etc.: cumulative counts and accuracy, seed, and completion
  flag; atomically replaced after every successful batch. Records are flushed
  **before** updating metrics or starting the next batch.
- `aggregate.json`: atomically replaced after every completed round; includes
  completed/requested rounds, mean accuracy, population standard deviation of round
  accuracies, totals, target and percentage-point difference, and completion flag.

Progress bars show example counts and running round accuracy; each completed round
prints the cumulative mean. On failure, completed batches/rounds remain readable;
aggregate includes **only completed rounds**, not a partial current round. Before
round one completes there is no aggregate. Exceptions close the inference wrapper
and propagate. Existing output is never overwritten; there is no resume mode.
JSON replacement/stream flushing protects normal interruption, not a guarantee of
power-loss durability (`fsync` is not used). A crash between flushing records and
publishing metrics can leave records ahead of the metric snapshot.

## Deviations and uncertainties

This is a paper-informed **HF backend replication**, not a bit-for-bit recreation:

1. Upstream uses vLLM; this runner uses HF `generate`, BF16, and SDPA. Kernel and
   sampling implementations differ. Seeds are `base_seed + round_index`, not the
   original batched `n=16` RNG arrangement. Batch size can change sampled outputs;
   report it along with versions/hardware. Sixteen-round accuracy matching is an
   empirical check, not a guarantee.
2. `HFInference` stops at EOS/budget only. The adapter strips the first upstream
   stop string and everything after it **after generation**. This preserves the
   scored prefix but spends extra compute and draws, changing later RNG state.
3. Long prompts get per-prompt `min(1024, 2047 - tokenized_prefix_length)` budgets;
   adjacent equal budgets are batched. No prompt is truncated or shortened because
   of another prompt in its batch. The runner counts BOS; upstream's budget
   calculation calls `tokenizer.tokenize`, excluding special tokens. This can make
   a long-prompt budget one token smaller. Oversized prompts fail explicitly.
4. The shared safe symbolic verifier is not unrestricted upstream symbolic eval.
   Its policy/provenance is recorded. Infrastructure/time-limit errors abort rather
   than becoming incorrect answers.
5. Manifest validation establishes declared downloaded revisions, not cryptographic
   integrity of local weights after download. Do not modify the downloaded files.
   Selection hashes identify the normalized rows used, not all raw dataset bytes.

The parent reports completing an eight-example GSM8K smoke run on A40. This is
not a full benchmark or validation of MATH scoring. Upstream
source/model cards and official [HF offline documentation](https://huggingface.co/docs/transformers/installation#offline-mode)
were inspected directly; the docs skill's Context7 CLI could not run because `npx`
was unavailable.

## Focused CPU tests

Run inside a Slurm CPU allocation, **not on the login node**:

```sh
CUDA_VISIBLE_DEVICES= uv run --offline pytest tests/core/test_verifier.py \
  tests/runtime/test_evaluate.py tests/test_paper_evaluate_sft.py -q
```

Tests use local Arrow fixtures and CPU fake models through the real HFInference
wrapper. They cover CLI validation, manifest pins, test-row selection, answer and
prompt formatting, nucleus override, left padding, stop-text removal, long-prompt
budgets, seeded rounds, partial final batches, incremental writes, aggregates,
cleanup on backend failure, CUDA checks, and refusal to overwrite output.

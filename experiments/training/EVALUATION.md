# Validation-selected RhoMath evaluation

This workflow supports trained **full-parameter and LoRA RhoMath actors** on the
paper GSM8K/MATH splits. It does not change the original SFT-only evaluator or
establish that the training recipe reproduces Figure 4. Numerical/segmentation
acceptance must still pass before training.

## Enable during a fresh training run

Append to an otherwise valid training submission:

```sh
--evaluation validation-test --validation-rounds 16 \
--eval-batch-size 4 --eval-seed 42 --eval-timeout-seconds 7200
```

`--evaluation disabled` is the default. `validation` validates and freezes the
winner without running final test; `validation-test` also runs final test.
Benchmark mode is rejected. To continue an interrupted selection run, see
[resuming](#resuming-an-interrupted-selection-run); do not merge validation
histories manually.

Every scheduled checkpoint (`--checkpoint-every`) and the final checkpoint is a
candidate. Choose the cadence, validation rounds, seeds, and batch size before
starting. These are recorded in `selection.json` before the first update;
validation defaults are our explicit protocol, not a claim about the paper's
exact validation cadence or seeds. A smaller validation round count reduces
cost but changes selection noise; it is recorded rather than called equivalent.

Each checkpoint is evaluated synchronously **before pruning**. Failures or the
per-child timeout abort the training invocation; no candidate is skipped or
silently retried. Logs, invocation arguments and incomplete evaluation records
remain available. Completed training checkpoints remain recoverable even if
validation fails, but failed selection runs cannot proceed to final test.

## Resuming an interrupted selection run

Pass the last completed checkpoint and the interrupted run's selection manifest
together, with a **new** output directory:

```sh
--resume artifacts/training/skynet/my-run/checkpoints/iteration-N.pt \
--resume-selection artifacts/training/skynet/my-run/selection.json \
--evaluation validation-test --output artifacts/training/skynet/my-run-resumed
```

Both options are required together. The iteration target, checkpoint cadence,
validation/test protocols and task/model/data identities must equal the source
run's; a changed plan is a new experiment, not a resume. Completed validation
candidates are carried forward and the previous best checkpoint stays protected
in its original location; source outputs are never modified. A checkpoint whose
validation was interrupted is validated **before any new update**. Source
candidates later than the resumed checkpoint are rejected rather than reused,
and a frozen selection cannot be resumed. Each invocation is a new W&B run that
records its parent run. Checkpoints are written only at the checkpoint cadence:
work after the last checkpoint is lost when a job hits its time limit, and no
walltime signal handling is implemented.

## Separate process and resource costs

Evaluation runs as a child Python process in the same Slurm allocation, using
the same visible GPU. This isolates Python/NumPy/Torch sampling RNG and model
mode from training. No extra job or GPU is implicitly requested. Training waits;
its live actor/reference/optimizers/critic remain resident during validation.
Unused parent CUDA cache is released first, but **an additional evaluation actor
and its generation memory must fit**. An OOM fails, never changes precision or
batch size. GPU fit for the training update alone is insufficient.

Before automatic final test, training model/optimizer references are released.
Checkpoint actor dtype is preserved exactly; LoRA adapters stay FP32 over their
saved FP32/BF16 base. No adapter merging, AMP, or precision conversion occurs.
Parent `peak_*_bytes` describe its allocator only, not the child or total device
memory. `validation_seconds` is separated from checkpoint I/O and is included
in iteration wall time; final-test time is recorded in `final-evaluation.json`.
Thus throughput with evaluation enabled is not pure training throughput.

Retention keeps the **latest N plus the best** checkpoint if it is older.
Peak storage can reach **N+2 full checkpoints** during a save (four for N=2),
plus evaluation records. Older losing checkpoints can be deleted; their complete
validation evidence and hashes remain in the selection manifest.

## Frozen selection and final test

The winner is highest validation mean Pass@1; exact ties select the earliest
iteration. All candidates must use identical validation data/protocol and cover
the entire declared checkpoint schedule. `selection-frozen.json` is written
once only after training and all scheduled validation complete. Test evaluation
requires this manifest, verifies the checkpoint SHA256, and cannot select a new
winner. Test CLI settings must match the **test protocol already frozen** there.

Final test covers all 1,319 GSM8K or 500 MATH examples, in **16 seeded rounds**:
21,104 or 8,000 generated responses respectively. Each round contributes one
answer per question. Aggregate accuracy is the mean of round accuracies—not
pass@16, majority voting, best-of-16, or 16 independent training seeds.

Protocol reuses the existing paper evaluator: task prompt, temperature 0.35,
top-p 0.9, top-k 0, 1,024 new tokens, 2,047 total-context budget, and stop text
`\n\n\nProblem:`. Seeds are `eval_seed + round_index`. These local seed values
are not established as the original experiment's seeds. HF/SDPA, sequential
rounds, and trimming stop text after generation retain the existing documented
deviations from upstream vLLM. Malformed answers count as wrong; truncated
answers are verified from available text without the training truncation penalty.
Backend/verifier failures abort instead of being counted as incorrect answers.

To run final test later, on an allocated GPU, use the existing training output
and a **new** evaluation output directory; seed/batch-size must match its manifest:

```sh
.venv/bin/python -m experiments.training.evaluate \
  --split test --training-output artifacts/training/skynet/my-run \
  --selection artifacts/training/skynet/my-run/selection-frozen.json \
  --output artifacts/training/skynet/my-run-final-test \
  --rounds 16 --seed 42 --batch-size 4 --expected-gpu A40 \
  --wandb-mode online --wandb-project vineppo
```

`my-run` is illustrative: substitute an actual completed run. Final test does not
accept a direct `--checkpoint`. For diagnostic validation of an existing new-format
checkpoint, the same command uses `--split validation --checkpoint <path>` instead
of `--selection`. Standalone validation does not mutate or select a training run's
winner. No reduced test subsets or partial-round "final" scores are supported.

## Provenance and output

The loader reconstructs only the actor from local config, then strictly restores
weights, architecture, LoRA configuration/scaling and dtypes. mmap avoids loading
unused optimizer/reference/critic tensor storage. Checkpoints must use modern
Torch ZIP serialization. Local model/tokenizer files and `metadata.json` are
required. New checkpoints embed task/prompt/tokenizer/train-dataset provenance;
it must match the metadata. Legacy checkpoints without that binding are rejected
explicitly—no guessed metadata migration. No model or dataset downloads occur.

Under the training output:
- `selection.json`: fixed protocols/schedule and completed validation candidates.
- `selection-frozen.json`: immutable winner and full validation evidence.
- `evaluations/validation-N/`: each candidate's config, provenance, round records,
  progressive metrics, and final `result.json`.
- `evaluations/test-selected/`: automatic final test, if requested.
- `evaluations/*-command.json` and `*.log`: child commands and output.
- `final-evaluation.json`: final test result path and duration.

`result.json` exists only after all rounds and cleanup succeed. `progress.json`
and per-round metrics are intermediate evidence; they must not be mistaken for
an accepted completed result. Failure records contain an exception type, while
full tracebacks remain in local child logs.

W&B parent history includes `eval/validation/pass_at_1`, dispersion, sample/round
counts and best score/iteration at each candidate. At completion its summary
records the selected checkpoint and final `eval/test/*` metrics. Child evaluations
have separate W&B run IDs with the parent run identity in config. Older winning
iterations do not cause backward parent history writes: final test updates only
summary. Samples, responses, checkpoint weights and `.env` are never uploaded by
this integration. Offline and disabled modes are supported unchanged.

# PACE-ICE handoff scripts from November 20, 2026

These scripts record the [VinePPO handoff](../../../../prose/workflow/tasks/agent-tasks/20261120-212505_remote-agent-handoff.md). The scheduler and cleanup scripts contain historical job IDs, scratch paths, and commit pins. Some submit or cancel jobs, update watcher state, or prune verified checkpoints.

From the repository root on PACE-ICE, regenerate the report tables from the saved final Slurm audit and current Slurm accounting:

```sh
python3 experiments/training/pace_ice/handoff_20261120/vp-format-report-20261121.py \
  artifacts/training/pace_ice/slurm/vp26-final-results-audit-7251060.out
```

`vp-training-analysis.py RUN_ROOT` produces the JSON payload used in that audit from the training and held-out records. Run it in the project Python environment under Slurm. `vp-finalize-reports.py AUDIT STAMP COMMIT DETAILED_DRAFT BRIEF_DRAFT` applies the historical finalization edits to the two supplied draft files in place.

`vp-olmo-splitter-replay.py WORKTREE` (added 2026-11-24, job 7283244) replays the stored OLMo segmentation samples through the production step splitter with the OLMo chat tokens allowlisted. It is a what-if diagnostic; production rejects those samples.

`vp-bucket-triage.py BACKUP_RECEIPT.json PLAN.json` (added 2026-11-24) lists the bucket and writes a per-file plan for thinning it: four deletion tiers and a keep list. It is read-only and deletes nothing. `scripts/remote_sync/delete_planned.py` executes a plan.

`vp-review-frozen-specs.py [FILE ...]` (added 2026-11-24, job 7286729) prints the frozen-base path and SHA256 that the newest local compact checkpoint of each reported run expects, then the SHA256 of each FILE. It is read-only.

`vp-review-olmo-frozen-xet.py FILE` (added 2026-11-24, job 7286740) prints the bucket content hash of a local file without uploading it, to compare with a bucket listing.

`vp-check-kept-bases.py` (added 2026-11-24, job 7287784) downloads the two frozen bases the thinned bucket keeps to node-local disk and asserts their SHA256. Run it with `PYTHONPATH` set to the repo root.

`vp-check-restore.py` (added 2026-11-24, job 7288364) loads a real compact checkpoint with its frozen base reached through a hard link and through a symlink. The hard link loads; the symlink is refused.

`vp-olmo-baseline-tables.py RUNS_DIR [TASK:ARM:RUN_NAME ...]` (added 2026-11-24) prints the OLMo baseline tables of the 2026-11-24 report from the frozen test records and training metrics: Pass@1 with question-clustered intervals, paired gains, the strict and lenient GSM8K readings, and training health. Extra arguments add arms such as `gsm8k:VinePPO:handoff-olmo3-gsm8k-vineppo-50-s42-30`.

`vp-link-frozen-base.py RUN BASE SHA256` (added 2026-11-24) re-hashes a completed run's frozen bases and hard-links them to the verified canonical copy, so that `hf_bucket_sync.py --linked-frozen-base` can upload the run without them.

`vp-critic-rank-results.py RUN_ROOT [RUN_ROOT ...]` (added 2026-11-24) prints one JSON line per completed PPO run: validation and test Pass@1, mean critic loss, phase times, selected-checkpoint size and GPU wall time. Given the three seed-42 RhoMath LoRA-critic runs, it first prints the rank with the highest validation Pass@1, the smaller rank on a tie.

`vp-vineppo-mc-rate.py RUN_ROOT` (added 2026-11-24) prints a completed VinePPO run's Monte Carlo tokens per primary token and Monte Carlo tokens per generation second.

`vp-math-splitter-{failures,cause,whatif}.py` (added 2026-11-24, jobs 7289868, 7289925 and 7289968) are the read-only evidence for the pending MATH splitter change. They replay the 512 stored OLMo MATH responses: which ones the vendored splitter rejects, why (nested placeholders restored in insertion order), and what restoring in reverse order would do. Run them with `PYTHONPATH=src:.`. Nothing in `src` is changed by them.

`vp-full-finetuning-grpo.slurm NAME SEED ITERATIONS CHECKPOINT_EVERY WANDB_MODE ARCHIVE` (added 2026-11-28) runs the full fine-tuning GRPO control on node-local disk, because full-weight checkpoints do not fit under the scratch quota: HF training, export of the selected actor, the 16-round vLLM test, a weight-free copy on scratch and a verified bucket mirror.

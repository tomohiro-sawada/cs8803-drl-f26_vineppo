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

`vp-olmo-baseline-tables.py RUNS_DIR` (added 2026-11-24) prints the OLMo baseline tables of the 2026-11-24 report from the frozen test records and training metrics: Pass@1 with question-clustered intervals, paired gains, the strict and lenient GSM8K readings, and training health.

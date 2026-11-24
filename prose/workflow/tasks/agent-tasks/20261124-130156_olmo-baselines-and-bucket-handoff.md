# Handoff: finish the OLMo baselines and write their report

20261124-130156 EST · `7267b98`

Written by the reviewing agent (Zed session) for the next coding agent. Tom is away and cannot answer questions.

One OLMo run is still training. When it finishes, its upload has to be confirmed and a report written. The bucket deletion Tom asked for is done and must not be repeated or extended.

## Read first

- `AGENTS.md`.
- Execution log entries `20261124-082337` to `20261124-115611` in [the earlier handoff](20261120-212505_remote-agent-handoff.md#execution-log). That file is the running log; append to it.
- [The GSM8K format note](../../../agent-reports/20261124-080354_olmo-gsm8k-format-vs-accuracy.md). The report extends it.

## Names

- `S=/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo` (scratch)
- `T=$S/training/pace_ice` (run directories)
- `CTL=$S/handoff-20261120/control-21` (plan, receipt, deletion record)
- Bucket: `tsawada/cs8803-drl-f26_vineppo`
- Handoff scripts: `experiments/training/pace_ice/handoff_20261120/`

## State

### Git

- `origin/tom/work` is at `7267b98`. Five commits were pushed today, `a811d0c` to `7267b98`.
- Uncommitted: this file, `vp-check-kept-bases.py` and `vp-check-restore.py` in the handoff scripts folder, and `artifacts/training/pace_ice/bucket-thinning-20261124/deleted.jsonl`. Commit them in step 1.
- Never stage `bash.sh`, `pyproject.toml`, `uv.lock`, `prose/reflections/*`, `wandb-metric-keys.txt`, `wandb-panels.tsv` or Slurm logs.

### Bucket: thinned, do not delete more

- **Done:** tiers 1, 2 and 3 of `$CTL/triage-plan.json` are deleted: 266 files, 584,689,066,899 bytes. The record is `$CTL/deleted.jsonl`, with a copy in `artifacts/training/pace_ice/bucket-thinning-20261124/`.
- **Now:** 11,241 files and 411,095,296,439 bytes, with 588.9 GB free under the 1 TB allowance.
- **Audited at 12:49:**
  - no deleted path is present;
  - every planned survivor has its planned size and content hash;
  - the 399 paths added since are the two OLMo uploads below;
  - `SYNC_RECEIPT.json` and all 20 tier-4 critics are present.
- **Tier 4 is Tom's decision.** It is 20 intermediate PPO critics, 248.3 GB. Leave it.
- **Skynet-only content:** the 165 tier 2–3 files now exist only at `sky1.cc.gatech.edu:/coc/flash13/tsawada7/data/hf_buckets/cs8803-drl-f26_vineppo`. A later sync there must not use `--delete`.
- **Bucket-only content:** 90 checkpoint files (39.5 GB) exist nowhere else, including the selected checkpoints of GRPO 650, RLOO 650 and PPO 650. A second Skynet sync would give them a second copy.

### OLMo baselines (seed 42, H200, launched at `4c924f8`)

| Run directory under `$T` | State | Test Pass@1 | Bucket copy without the frozen base |
|---|---|---|---|
| `handoff-olmo3-gsm8k-rloo-50-s42-20` | complete | 88.01% | verified, 157 files (job 7284130) |
| `handoff-olmo3-math-rloo-50-s42-20` | complete | 71.41% | verified, 157 files (job 7284133) |
| `handoff-olmo3-gsm8k-restem-match-s42-20` | complete | 84.49% | verified, 242 files (job 7284132) |
| `handoff-olmo3-math-restem-match-s42-20` | round 7 of 10 at 13:01 | pending | pending |

- **Running job:** 7284850, 15-hour limit, started 09:33, about 27 minutes per round. I expect training to end around 14:25 and the 16-round test to follow (estimate).
- **Its chain:**
  - `vineppo-after-7284850` (7285818) resumes the run if the allocation ends.
  - 7284851 re-hashes the run's frozen base and hard-links it to the canonical copy.
  - 7284852 uploads everything else and verifies it. It runs only if 7284851 succeeds.
- **If the run needs a successor allocation,** 7284851 fires when 7284850 ends, finds no `COMPLETE`, and fails; 7284852 is then cancelled. Resubmit both after `COMPLETE` exists. Take the commands from `sacct -X -P -j 7284851,7284852 -o SubmitLine%8000`, change the dependency, and run them from `$S/worktrees/4c924f8`.

### Numbers already computed

Job 7288969 printed tables for seven of the eight runs with `vp-olmo-baseline-tables.py`. Its output is `artifacts/training/pace_ice/slurm/vp29-olmo-tables-partial-7288969.out`. Step 4 regenerates all eight.

| Arm | MATH, strict | GSM8K, strict | GSM8K, lenient |
|---|---|---|---|
| SFT | 70.59% | 61.65% | 83.29% |
| GRPO | 71.86% | 87.87% | 89.77% |
| RLOO | 71.41% | 88.01% | 89.26% |
| RestEM | pending | 84.49% | 89.31% |

Re-grading with the current verifier changed no grade in any finished OLMo evaluation (jobs 7283551, 7284876, 7287977, 7288220).

## Remaining work

1. **Commit and push** the uncommitted files listed above, after the CPU suite passes under sbatch. The last full run passed 1,628 tests (job 7287646). Commit `7267b98` added one test to `tests/test_regrade_evaluation.py`, and only that file and `tests/test_bucket_delete_planned.py` were rerun after it (20 passed, job 7288969).
2. **Watch the last run** to `COMPLETE`. Fix a `HALTED` run by diagnosing its Slurm log, fixing the cause, and relaunching under suffix `-21`.
3. **Confirm its upload:** `artifacts/training/pace_ice/slurm/vp27-arc-olmo3-math-restem-match-s42-20-7284852.out` must contain a `verified training/pace_ice/handoff-olmo3-math-restem-match-s42-20: …` line. Then re-grade its test with `scripts/diagnostics/regrade_evaluation.py` against `$T/handoff-olmo3-math-sft-04/test`.
4. **Regenerate the tables** under sbatch from the repo root: `python experiments/training/pace_ice/handoff_20261120/vp-olmo-baseline-tables.py $T`. It asserts that every run is complete with 16 rounds.
5. **Write the report** at `prose/agent-reports/<timestamp>_olmo-baselines-and-bucket.md`. Do not edit the 20261122 reports; link them. Its contents:
   - the tables from step 4, with both GSM8K readings;
   - that OLMo VinePPO and PPO were not run, with the reasons from log entry `20261124-082337`;
   - the bucket after thinning, from this file and log entry `20261124-114344`;
   - how to restore a checkpoint whose frozen base was a deleted duplicate (below);
   - the corrected archive status (below).
6. **Close out:** append a log entry, commit and push, and remove the worktrees `$S/worktrees/4c924f8` and `$S/worktrees/0389489` once no job uses them.

## Facts the report needs

### Restoring a checkpoint

A compact checkpoint records the absolute path and SHA256 of its run's `frozen-lora.pt`. After thinning, the bucket holds one copy of each base:

| Model | Bucket path under `training/pace_ice/` | SHA256 |
|---|---|---|
| RhoMath | `handoff-rho-vineppo-650-s42-03/attempt-0000/checkpoints/frozen-lora.pt` | `086cc6d318d3aeadf44ab916dd960f5dceb56c14dcd0d1e90f5bdfa45d0b9fbf` |
| OLMo | `handoff-olmo3-math-grpo-50-s42-10/attempt-0000/checkpoints/frozen-lora.pt` | `7a574b6997d1acb19ec12204c10522f106b94ea5ee303417a78c0faa3c1c646a` |

- **Verified:** job 7287784 downloaded both from the bucket after tier 1 and matched these hashes.
- **Method:** download the run and the base for its model, then place a copy or hard link of the base at the path the checkpoint records. Job 7288364 loaded a real checkpoint through a hard link. A symlink is refused with "Checkpoint blob path must be canonical".
- **Scope:** 47 run prefixes lost their own base copy in tier 1, 25 of them reported runs. The four OLMo `-20` runs and OLMo GSM8K GRPO never had one in the bucket.
- **Full-format exceptions:** the selected checkpoints of `long-grpo-rhomath-gsm8k-seed42-vllm-h100-200iter-01` and `long-vineppo-rhomath-gsm8k-seed42-vllm-h100-200iter-01` contain their own weights and need no base.

### Archive status, correcting the 20261122 reports

Those reports say 13 of 21 required lineages had full bucket verification. All 21 now have every run directory hash-verified, with these qualifications:

- OLMo GSM8K GRPO is verified without its frozen base, which is byte-identical to the verified OLMo MATH GRPO copy.
- The four new OLMo runs are outside the 21 and are archived the same way.
- After thinning, a "verified" run in the bucket is complete except for a frozen base that was a duplicate.

### Stale local receipts

69 `*.archive.json` files on scratch name `checkpoint-history/` objects that were deleted. The content of 53 is still in the bucket at the run-mirror path, and the other 16 are on Skynet. `archive_checkpoint` trusts an existing receipt, so a run resumed with `--archive-checkpoints yes` would not re-upload those files. All affected runs are finished.

## Known gaps

- **W&B:** each allocation and each evaluation is its own run. The table script links each run's training run, not its test run.
- **OLMo VinePPO and PPO:** not run. Both need Tom's decisions; see log entry `20261124-082337`.
- **Skynet:** I rely on the receipt's manifest and cannot see that machine.
- **Deletion tool:** `delete_planned.py` records a deletion after it happens. A job killed between the two would stop the re-run on that path. It did not happen.

## Rules

- No bucket deletion of any kind, and no `sync --delete`.
- Do not delete checkpoint files or run directories on scratch, and do not run `archive_run.slurm` or `prune_verified.py`.
- Run tests, GPU code and bucket jobs under sbatch; never edit a tree that running jobs launched from.
- Do not launch OLMo VinePPO or PPO, change hyperparameters, or change segmentation.
- Follow `AGENTS.md`. Report numbers only from result files and mark estimates.

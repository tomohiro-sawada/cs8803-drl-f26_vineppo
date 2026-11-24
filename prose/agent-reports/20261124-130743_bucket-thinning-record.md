# Bucket thinning record: what the Hugging Face bucket holds after 2026-11-24

20261124-130743 EST · `7267b98`

Responds to Tom's request of 2026-11-24 to record the bucket deletion; re: bucket `tsawada/cs8803-drl-f26_vineppo` and [the final report](20261122-110919_final-report-detailed.md).

On 2026-11-24 I deleted 266 weight files (584.7 GB) from the bucket. No result, record, metric or log file was deleted, and every validation-selected checkpoint behind a reported number is still there. The cost is that most checkpoints now need one extra step before they load, and some deleted content exists only on Skynet.

## What can surprise you

1. **Most checkpoints do not load as downloaded.** The bucket keeps one copy of each model's frozen base weights. Before evaluating a checkpoint, place that base at the path the checkpoint records. See [Restoring a checkpoint](#restoring-a-checkpoint).
2. **Smoke, dry-run, benchmark, pilot and probe weights are on Skynet only.** Their metrics and logs are still in the bucket. This includes the `bench-*` runs behind the 52× iteration-time comparison and the `pilot-*` runs.
3. **Three non-selected checkpoints of the two earliest seed-42 runs are on Skynet only.** Their selected checkpoints are in the bucket.
4. **PPO critics are still complete.** All 20 intermediate critics of the three PPO 200 seeds are in the bucket (tier 4, not deleted). PPO 650 saved only its final critic.
5. **90 checkpoint files (39.5 GB) have no second copy anywhere.** They are in the bucket only. See [Single copies](#single-copies).
6. **The 20261122 reports' archive status is out of date.** They say 13 of 21 lineages were fully verified. All 21 now are; see [the handoff](../workflow/tasks/agent-tasks/20261124-130156_olmo-baselines-and-bucket-handoff.md#archive-status-correcting-the-20261122-reports).

## What was deleted

| Tier | What | Files | GB | Where the content is now |
|---|---|---:|---:|---|
| 1 | Byte-identical copies of a kept file | 101 | 321.2 | In the bucket, at the kept copy's path |
| 2 | Weights of smoke, dry-run, benchmark, pilot and probe runs | 162 | 249.9 | Skynet only |
| 3 | Non-selected full-size checkpoints of the two earliest seed-42 runs | 3 | 13.7 | Skynet only |
| Total | | 266 | 584.7 | |

- **Tier 1** is 69 extra copies of the RhoMath frozen base (303.6 GB), 31 checkpoints stored twice (5.1 GB) and one critic stored twice (12.4 GB).
- **Tier 2** covers 50 run prefixes: PPO smokes (100.9 GB), other smokes and dry runs (79.7 GB), pilots (36.7 GB), benchmarks and HF controls (32.2 GB) and probes (0.4 GB).
- **Tier 3** is `iteration-175.pt` and `iteration-200.pt` of `long-grpo-rhomath-gsm8k-seed42-vllm-h100-200iter-01`, and `iteration-200.pt` of `long-vineppo-rhomath-gsm8k-seed42-vllm-h100-200iter-01`.
- **Skynet copy:** `sky1.cc.gatech.edu:/coc/flash13/tsawada7/data/hf_buckets/cs8803-drl-f26_vineppo`, same relative paths. Its receipt lists every tier 2 and tier 3 file with the same size and content hash. I cannot see that machine; I rely on the receipt.

## What the bucket holds now

11,241 files, 411.1 GB, measured at 13:07. 588.9 GB is free under the 1 TB allowance.

| Contents | Files | GB |
|---|---:|---:|
| Results, records, selections, metrics and logs of every run | 10,517 | 2.1 |
| Validation-selected checkpoints, one per reported run | 31 | 14.5 |
| Other validation checkpoints of reported runs | 204 | 35.0 |
| Last-iteration checkpoints | 17 | 3.0 |
| Served adapters of evaluated checkpoints | 38 | 2.2 |
| Resume slots of finished runs | 7 | 1.1 |
| PPO critics: selected-checkpoint and final | 5 | 62.1 |
| PPO critics: intermediate (tier 4, not deleted) | 20 | 248.3 |
| Frozen bases: one RhoMath, one OLMo | 2 | 33.6 |
| OLMo MATH RLOO and GSM8K RestEM, uploaded after thinning | 399 | 9.2 |
| `SYNC_RECEIPT.json` | 1 | 0.0 |

The five kept critics are `critic-200.pt` for PPO 200 seeds 42 and 43, `critic-175.pt` and `critic-200.pt` for seed 44, and `critic-650.pt` for PPO 650.

The OLMo MATH RestEM run is still training and will be uploaded when it finishes.

## Restoring a checkpoint

A compact checkpoint stores only the trained adapter. It records the absolute path and SHA256 of a `frozen-lora.pt` file holding the frozen base, and refuses to load without it.

| Model | Kept base, under `training/pace_ice/` | SHA256 |
|---|---|---|
| RhoMath | `handoff-rho-vineppo-650-s42-03/attempt-0000/checkpoints/frozen-lora.pt` | `086cc6d318d3aeadf44ab916dd960f5dceb56c14dcd0d1e90f5bdfa45d0b9fbf` |
| OLMo | `handoff-olmo3-math-grpo-50-s42-10/attempt-0000/checkpoints/frozen-lora.pt` | `7a574b6997d1acb19ec12204c10522f106b94ea5ee303417a78c0faa3c1c646a` |

- **Steps:** download the run directory and the base for its model. Copy or hard-link the base to `attempt-0000/checkpoints/frozen-lora.pt` inside that run, at the same absolute path the run had on PACE scratch.
- **A symlink does not work.** The loader refuses it with "Checkpoint blob path must be canonical". A hard link loads (job 7288364).
- **Who needs this:** 47 RhoMath run prefixes, 25 of them reported runs, and the five OLMo runs other than OLMo MATH GRPO.
- **Who does not:**
  - `handoff-rho-vineppo-650-s42-03` and `handoff-olmo3-math-grpo-50-s42-10` keep their own base.
  - The selected checkpoints of the two earliest seed-42 runs (`long-*-seed42-vllm-h100-200iter-01`) contain full weights.
- **On PACE scratch nothing changed.** Runs whose files are still local load as before.

## Single copies

90 checkpoint files, 39.5 GB, exist only in the bucket. Their PACE copies were removed after the upload was hash-verified, and the Skynet sync had already listed the bucket.

- **Selected checkpoints among them:** GRPO 650 `iteration-625.pt`, RLOO 650 `iteration-525.pt`, PPO 650 `iteration-575.pt`.
- **Critics among them:** `critic-200.pt` of PPO 200 seed 44 and `critic-650.pt` of PPO 650.
- **The rest:** other validation checkpoints of the four 650 runs and OLMo MATH GRPO.
- **Fix:** a second Skynet sync without `--delete`. With `--delete`, it would erase Skynet's only copy of tiers 2 and 3.

## How it was done and checked

- **Plan:** `vp-bucket-triage.py` assigned every bucket file a tier. A tier 2 or 3 path had to be in the Skynet receipt with the same size and content hash.
- **Deletion:** `scripts/remote_sync/delete_planned.py` deleted single named paths, one tier per job (7287731, 7287869, 7287942). Each job first ran a preview that had to print the expected counts.
- **Between tiers 1 and 2,** job 7287784 downloaded both kept bases from the bucket and matched the SHA256 values above.
- **Audit at 12:49 and again at 13:07:**
  - no deleted path is in the bucket;
  - every surviving planned path has its planned size and content hash;
  - the only additions are the two OLMo uploads.

## Records

- **Deleted paths**, with tier, size and content hash: [`deleted.jsonl`](../../artifacts/training/pace_ice/bucket-thinning-20261124/deleted.jsonl), SHA256 `277ed398247a5ad2f529c939af5d5d41543480e35261e13882da4da1ee801d76`.
- **Plan and Skynet receipt:** `triage-plan.json` and `SYNC_RECEIPT.json` in `/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo/handoff-20261120/control-21/`. They are too large for the repo's 500 KiB limit. The receipt is also at the bucket root.
- **Scripts:** `scripts/remote_sync/delete_planned.py` and `experiments/training/pace_ice/handoff_20261120/vp-bucket-triage.py`.
- **Log:** entries `20261124-095625` to `20261124-114344` in [the execution log](../workflow/tasks/agent-tasks/20261120-212505_remote-agent-handoff.md#execution-log).

## Not verified

- I have not downloaded a whole run from the thinned bucket and evaluated it end to end. The restore step was tested on PACE scratch with a real checkpoint.
- PACE scratch is purged storage. Files that are "still local" today may not stay.

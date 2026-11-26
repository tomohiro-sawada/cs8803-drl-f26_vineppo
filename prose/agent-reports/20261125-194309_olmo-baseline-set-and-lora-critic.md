# OLMo 3 baselines and the LoRA critic

20261125-194309 EST · `71b4ce5`

All six OLMo-3-7B-Instruct-SFT arms finished on GSM8K. Five production arms finished on MATH. The MATH VinePPO smoke passed its end-to-end test, and the production arm remains pending because the measured cost exceeded the approximately 24-GPU-hour limit. The RhoMath sweep selected rank 16 for both OLMo PPO runs.

## OLMo held-out results

All test rows average 16 draws per question. Intervals cluster by question. Paired differences are in percentage points. GPU wall includes setup, training and final evaluation within the allocations. For SFT it covers evaluation. The trained production arms each drew 6,400 primary continuations. The sampled-token column also counts VinePPO's MC continuations. Audit 7321042 read the frozen selections, test records and Slurm accounting. [Table generator](../../experiments/training/pace_ice/handoff_20261120/vp-olmo-baseline-tables.py) · [Earlier baseline report](20261124-151031_olmo-baselines-and-bucket.md)

### GSM8K, 1,319 questions

| Arm | Selected | Validation | Pass@1, strict [95% CI] | Gain over SFT, strict | Pass@1, lenient [95% CI] | Gain over SFT, lenient | Samples | Sampled tokens | GPU wall | W&B |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT | SFT | — | 61.65% [59.80, 63.50] | — | 83.29% [81.78, 84.80] | — | — | — | 0:17:55 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/5755e93304ba4608935fa20e210084c0) |
| GRPO | 50 | 93.83% | 87.87% [86.45, 89.30] | +26.23 [+24.61, +27.84] | 89.77% [88.47, 91.07] | +6.48 [+5.39, +7.56] | 6,400 | 1,017,333 | 1:28:20 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/0ddc996b3b4d428691c0613f72eed7f7) |
| RLOO | 40 | 94.84% | 88.01% [86.59, 89.43] | +26.36 [+24.74, +27.98] | 89.26% [87.93, 90.59] | +5.97 [+4.99, +6.95] | 6,400 | 947,308 | 1:27:01 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/4218622785f8450eb6d582b9e99c9769) |
| RestEM | 10 | 92.43% | 84.49% [82.85, 86.12] | +22.84 [+21.31, +24.37] | 89.31% [87.96, 90.65] | +6.01 [+5.00, +7.02] | 6,400 | 874,590 | 4:17:52 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/54f505cdeb5a4a369d39b5ab230a09fa) |
| VinePPO | 50 | 95.17% | 89.18% [87.78, 90.58] | +27.53 [+25.85, +29.21] | 90.08% [88.76, 91.40] | +6.79 [+5.72, +7.85] | 6,400 | 75,670,198 | 4:21:03 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/a1e83cb10c8042dab7a8b1ca800a8eaa) |
| PPO | 50 | 94.84% | 88.83% [87.44, 90.21] | +27.18 [+25.52, +28.83] | 90.01% [88.72, 91.31] | +6.72 [+5.65, +7.79] | 6,400 | 941,205 | 2:03:01 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/ed6aa7cae74e4ab2942d7b5db541a179) |

VinePPO exceeded PPO by 0.35 strict points with a paired 95% interval of −0.17 to +0.88. Under the lenient reading the gap was 0.07 points [−0.43, +0.56]. Both reached about 90% lenient Pass@1. VinePPO took 4:21:03 on H200 and PPO took 2:03:01. Differences between strict and lenient scores partly reflect answer formatting. [GSM8K grading and answer format](20261124-080354_olmo-gsm8k-format-vs-accuracy.md)

### MATH, 500 questions

| Arm | Selected | Validation | Pass@1, strict [95% CI] | Gain over SFT, strict | Samples | Sampled tokens | GPU wall | W&B |
|---|---|---|---|---|---|---|---|---|
| SFT | SFT | — | 70.59% [67.32, 73.85] | — | — | — | 0:33:31 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/e4ccb98121eb4d9b86842ea569828660) |
| GRPO | 50 | 67.90% | 71.86% [68.57, 75.16] | +1.27 [+0.38, +2.17] | 6,400 | 2,789,254 | 3:16:15 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/64370195038c4e568bc0f69f97ffae3f) |
| RLOO | 40 | 68.00% | 71.41% [68.11, 74.72] | +0.83 [-0.04, +1.69] | 6,400 | 2,733,705 | 3:04:06 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/1b3716b7dedc4571826d0dfaeb0bb423) |
| RestEM | 6 | 68.00% | 71.00% [67.69, 74.31] | +0.41 [-0.31, +1.14] | 6,400 | 2,809,220 | 5:13:42 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/5ccdddee1775458b8c0a7bd2322e6713) |
| VinePPO | Pending | — | — | — | — | — | — | — |
| PPO | 30 | 66.90% | 71.12% [67.83, 74.42] | +0.54 [-0.20, +1.28] | 6,400 | 2,902,682 | 4:01:11 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/fc606b322413426b8239e69a2598e07a) |

PPO gained 0.54 points over SFT [−0.20, +1.28]. GRPO had the highest held-out point estimate. Its paired difference with PPO was +0.74 points [−0.09, +1.57]. MATH RestEM took 5:13:42. Re-grading all 8,000 RestEM and 8,000 SFT answers reproduced all stored grades.

## RhoMath PPO critic ranks

| Critic | Seed | Selected | Val. Pass@1 | Test Pass@1 | Loss mean / last 25 | Seconds per update | Critic seconds | Checkpoint GB | GPU wall | W&B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Full FP32 | 42 | 200 | 61.13% | 45.57% | 0.097 / 0.085 | 42.1 | 19.2 | 12.566 | 2:28:34 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/9a3d956d3eaa479f90624a25c7653d32) |
| LoRA 16 | 42 | 175 | 63.00% | 48.85% | 0.141 / 0.091 | 20.3 | 6.7 | 0.304 | 1:17:28 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/4a6e9bc74810491fab4712c0d2c2cc0e) |
| LoRA 64 | 42 | 200 | 62.53% | 47.68% | 0.117 / 0.089 | 20.2 | 6.7 | 0.758 | 1:17:21 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/f6135776b0d443c1bfc540b2b695ce62) |
| LoRA 256 | 42 | 125 | 60.52% | 47.53% | 0.102 / 0.081 | 21.9 | 8.4 | 2.575 | 1:19:22 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/8162e486d47a4fe8855415674fcca646) |
| Full FP32 | 43 | 200 | 64.75% | 48.53% | 0.096 / 0.079 | 42.2 | 20.8 | 12.566 | 2:34:47 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/a02dde927f9344e59ac17ff2666f9cc8) |
| LoRA 16 | 43 | 200 | 62.27% | 46.82% | 0.133 / 0.093 | 20.3 | 6.8 | 0.304 | 1:13:32 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/0a193cdade8c4c659623e82fcf7b9501) |
| Full FP32 | 44 | 175 | 61.19% | 47.80% | 0.095 / 0.081 | 44.7 | 21.9 | 12.566 | 2:44:17 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/0544e7b723c94354bb42cdb237942712) |
| LoRA 16 | 44 | 175 | 61.86% | 48.06% | 0.137 / 0.094 | 20.3 | 6.8 | 0.304 | 1:14:54 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/edf3aa9b370b43bf849a9c3b06995840) |

The seed-42 sweep used 200 updates with 128 primary continuations per update. Rank selection used the highest validation Pass@1, with smaller rank breaking ties. Rank 16 won and was carried to seeds 43 and 44 and to both OLMo PPO workloads.

Across three training seeds, rank 16 averaged 47.91% test Pass@1 with a 1.02-point sample SD. The full critic averaged 47.30% with a 1.54-point SD. The rank-16 changes on seeds 42, 43 and 44 were +3.28, −1.71 and +0.27 points. Three seeds and mixed directions leave the accuracy effect uncertain. At seed 42, more critic rank lowered training value loss while validation Pass@1 was highest at rank 16. Explained variance was unlogged. The last-25 mean token spread of return minus rollout value was 0.429, 0.423, 0.410 and 0.413 for full, 16, 64 and 256 at seed 42.

Rank 16 reduced the selected training checkpoint from 12.566 GB to 0.304 GB per boundary, a 41.4× reduction. Both sizes exclude the shared actor/reference snapshot. Full-critic size includes the actor checkpoint and companion critic file. Mean critic time fell from 19.2 to 6.7 seconds per update at seed 42. Full-critic training spent 2,065 seconds on checkpoint work including archival uploads, against 34 seconds in rank 16, which mirrored after training. The GPU wall comparison therefore includes both compute and checkpoint I/O. Historical full-critic seeds 43 and 44 each lack one iteration metric row after an archive failure, so their phase averages use 199 rows. Their GPU wall times count all started links, including the stopped 1:20 seed-44 replay.

On OLMo, the rank-16 checkpoint was 0.961 GB on both tasks. GSM8K averaged 119.7 seconds per update and 37.4 seconds in the critic phase. MATH averaged 233.9 and 46.9 seconds, respectively. Mean critic loss was 0.155 on GSM8K and 0.193 on MATH. The four-update OLMo rank-256 sizing smoke finished in 0:37:49, scored 70.56% strict GSM8K Pass@1 at iteration 4, and produced an 8.157 GB checkpoint. Audit jobs 7324782 and 7321042 generated the rank comparisons. [Rank audit script](../../experiments/training/pace_ice/handoff_20261120/vp-critic-rank-results.py)

## VinePPO sampling cost and MATH decision

| Run | Updates | Primary tokens | MC tokens | MC / primary | Primary generation | MC generation | GPU wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K smoke | 4 | 74,971 | 4,689,384 | 62.55 | 22.18 s | 603.42 s | 0:42:50 |
| GSM8K production | 50 | 966,981 | 74,703,217 | 77.25 | 276.25 s | 10,466.92 s | 4:21:03 |
| MATH smoke | 2 | 115,884 | 15,816,415 | 136.48 | 29.79 s | 4,569.47 s | 2:54:57 |

The GSM8K smoke scored 75.99% strict and 88.95% lenient at iteration 4. The two-update MATH smoke scored 70.28% [66.98, 73.57] at validation-selected iteration 2. It took 3:50:58 from first start to final completion including its queue gap. Both MATH updates logged zero splitter rejections. The smoke ran with the reverse-order placeholder change approved in the separate submission handoff. [VinePPO sampling audit](../../experiments/training/pace_ice/handoff_20261120/vp-vineppo-mc-rate.py) · [Approved MATH splitter change](../workflow/tasks/agent-tasks/20261124-190704_path-to-submission-handoff.md#toms-decisions-20261124-192730-final)

The GSM8K production rate of 7,137 MC tokens per second initially suggested about 13.0 hours just to generate the estimated 335M MATH MC tokens. The direct MATH smoke averaged 3,461 MC tokens per second and 6,706.36 seconds for the entire MC phase across both updates. The MATH rate projects 26.9 GPU-hours of MC generation alone. Scaling full update time to 50 gives 51.9 GPU-hours before setup and testing. Both are estimates from two recorded updates. The measured cost leaves the production MATH VinePPO arm pending.

## Departures from VinePPO and its upstream implementation

- **Models and tuning.** OLMo 3 is an extension to the paper's model comparison. The actor uses a BF16 base, rank-16 all-linear LoRA with alpha 32, and an independent BF16 reference. Actor adapters, gradients and AdamW moments use FP32. The paper trains the full actor. This experiment uses constant actor LR 1e-5 and smaller primary budgets than the paper's full RhoMath experiments. RhoMath uses the paper's problem format. OLMo uses its ChatML prompt and task-specific final answer instructions.
- **PPO critic.** The LoRA variant retains a separate backbone, freezes it in BF16, and trains all-linear adapters and a scalar value head in FP32. Alpha is twice the rank. Critic LR stays at 1e-5. Both PPO variants use terminal rewards, fixed rollout values and GAE targets, gamma and lambda equal to 1, and value clipping at 0.2. The full-critic reference trains the independent FP32 backbone and head.
- **Rollout and evaluation.** Training samples at temperature 1.0 and top-p 1.0 without top-k filtering. Upstream training uses temperature 0.6 and top-p 0.9. Evaluation uses temperature 0.35 and top-p 0.9. Training allows a 2,048-token context and up to 1,024 new tokens, with reward −1 for truncation. Checkpoint selection uses held-out validation and the final test uses 16 draws per question. GSM8K's strict grader depends on the `####` answer form. Its 98.94% ceiling comes from 14 test references containing commas. The lenient reading extracts the final number and may credit a coincidentally correct number.
- **Exact-token VinePPO.** Learner scoring, advantage assignment and branching use the original sampled IDs. Upstream reconstructs sequences and branches from decoded text. A character cut inside an original sampled token belongs to the earlier step, so both MATH and GSM8K advance the cut to the next complete token boundary and coalesce coincident cuts. GSM8K merges the suffix beginning at the first answer marker. The optional terminal EOS belongs to the final step. The splitter recognizes both registered specials and added tokens marked special, and only the prompt protocol's explicitly allowed controls may appear in the prompt. The MATH smoke uses the separately approved reverse-order placeholder restoration and counts residual splitter rejections treated as one step. Each distinct sampled token prefix is expanded once with nine continuations and its estimate is reused across siblings. Upstream deduplicates text prefixes.
- **Objectives and budget.** PPO and VinePPO use masked token-level advantage whitening and a token-mean policy objective. They clamp each reference k3 term to 10 before summing by response. GRPO standardizes rewards within prompts and averages token ratios and unclamped k3 within each response. RLOO uses leave-one-out rewards, sequence clipping and summed k3. The pinned upstream code lacks GRPO and RLOO. The policy clip is 0.2, with KL 1e-4 for PPO and VinePPO and 3e-3 for GRPO and RLOO. Matched OLMo RestEM samples 640 continuations per round for ten rounds, resets the adapters and optimizer before each eight-epoch MLE phase, trains at constant 1e-5 with zero KL, and chooses the validation-best of the ten final checkpoints. Upstream samples the full training set per round and uses warmup, a lower LR and within-round validation and early stopping.

[Earlier RhoMath design choices](20261122-110919_final-report-detailed.md#design-choices) · [Brief November 22 report](20261122-110919_final-report-brief.md) · [Training and method documentation](../../experiments/training/README.md)

## Archive and restore

New completed runs were mirrored to the private bucket with SHA256 verification of every uploaded file. The immutable actor/reference snapshot is stored once per pretrained model. Each new run links its identical frozen file to the local canonical copy before upload. The remote mirror omits that path. To restore, obtain and check the canonical snapshot hash, then copy or hard-link it to the exact recorded path. The compact loader requires a canonical path.

A LoRA-critic checkpoint carries the trainable critic and optimizer, together with the hash of its original frozen backbone. Recreate the critic from the same pinned pretrained model at the same dtype. CPU job 7301847 restored the selected RhoMath rank-16 checkpoint and the OLMo rank-256 smoke checkpoint, checked both frozen fingerprints, and evaluated both restored value networks. Full optimizer and RNG resume are covered by the CPU suite. [Baseline bucket and restore instructions](20261124-151031_olmo-baselines-and-bucket.md#restore-a-compact-checkpoint) · [Execution log and receipts](../workflow/tasks/agent-tasks/20261120-212505_remote-agent-handoff.md#execution-log)


| Run | Verification job | Files | Bytes |
|---|---:|---:|---:|
| OLMo MATH RestEM | 7284852 | 242 | 6,657,866,161 |
| OLMo GSM8K VinePPO smoke | 7292021 | 88 | 1,157,840,831 |
| OLMo GSM8K VinePPO | 7292546 | 157 | 2,611,756,178 |
| OLMo GSM8K PPO smoke, rank 256 | 7293836 | 88 | 16,509,925,044 |
| RhoMath PPO rank 16, seed 42 | 7293228 | 208 | 2,520,192,927 |
| RhoMath PPO rank 64, seed 42 | 7293230 | 208 | 6,153,392,990 |
| RhoMath PPO rank 256, seed 42 | 7293232 | 208 | 20,687,045,421 |
| RhoMath PPO rank 16, seed 43 | 7298730 | 208 | 2,519,713,991 |
| RhoMath PPO rank 16, seed 44 | 7298732 | 208 | 2,519,932,387 |
| OLMo MATH VinePPO smoke | 7309563 | 89 | 1,307,436,911 |
| OLMo GSM8K PPO, rank 16 | 7300740 | 157 | 5,013,821,201 |
| OLMo MATH PPO, rank 16 | 7301435 | 157 | 5,025,610,106 |

Inventory 7321042 counted 14,853 files and 496,813,613,168 logical bytes and found the second Skynet receipt. The earlier baseline report lists the corresponding receipts for the seven pre-existing OLMo arms. The full CPU suite passed 1,637 tests on code pin `04a670e` in job 7324786.

# OLMo 3 baselines and bucket state

20261124-151031 EST · `ca57462`

The four baseline arms are complete on MATH and GSM8K. MATH RestEM scored 71.00% Pass@1 at validation-selected iteration 6 after 5:13:42 on H200. Re-grading all 8,000 RestEM responses and all 8,000 SFT responses reproduced every stored grade. The paired gain was +0.41 percentage points, with a question-clustered 95% interval of −0.31 to +1.14.

## Baseline results

Each score averages 16 draws per held-out question. Intervals cluster by question. The paired differences are in percentage points and cover one training seed. The trained arms each sampled 6,400 primary continuations. GRPO and RLOO used 50 updates and KL 3e-3. RestEM used ten rounds of 80 prompts with eight samples per prompt, eight epochs, minibatch 64, constant LR 1e-5 and no KL. GPU wall time is the sum of the completed allocations counted by the table script. Forge links identify the training allocation or the SFT evaluation. Re-grading and table job 7290075 produced the results below.

### MATH (500 questions)

| Arm | Selected | Validation | Pass@1, strict [95% CI] | Gain over SFT, strict | Samples | Sampled tokens | GPU wall | W&B |
|---|---|---|---|---|---|---|---|---|
| SFT | SFT | — | 70.59% [67.32, 73.85] | — | — | — | 0:33:31 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/e4ccb98121eb4d9b86842ea569828660) |
| GRPO | 50 | 67.90% | 71.86% [68.57, 75.16] | +1.27 [+0.38, +2.17] | 6,400 | 2,789,254 | 3:16:15 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/64370195038c4e568bc0f69f97ffae3f) |
| RLOO | 40 | 68.00% | 71.41% [68.11, 74.72] | +0.83 [-0.04, +1.69] | 6,400 | 2,733,705 | 3:04:06 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/1b3716b7dedc4571826d0dfaeb0bb423) |
| RestEM | 6 | 68.00% | 71.00% [67.69, 74.31] | +0.41 [-0.31, +1.14] | 6,400 | 2,809,220 | 5:13:42 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/5ccdddee1775458b8c0a7bd2322e6713) |

#### Paired differences between trained arms, MATH

| Comparison | strict |
|---|---|
| GRPO minus RLOO | +0.45 [-0.31, +1.21] |
| GRPO minus RestEM | +0.86 [+0.05, +1.67] |
| RLOO minus RestEM | +0.41 [-0.40, +1.22] |

#### Training health, MATH (mean over all iterations)

| Arm | Iterations | Reward | Clip (%) | Grad norm | Truncated (%) | Length | Accepted per round | Validation by iteration |
|---|---|---|---|---|---|---|---|---|
| GRPO | 50 | 0.627 | 0.026 | 0.034 | 1.891 | 435.821 | — | 10 → 66.25, 20 → 66.90, 30 → 67.30, 40 → 67.05, 50 → 67.90 |
| RLOO | 50 | 0.632 | 49.453 | 5.742 | 1.594 | 427.141 | — | 10 → 66.05, 20 → 66.30, 30 → 66.10, 40 → 68.00, 50 → 67.60 |
| RestEM | 10 | 0.622 | — | 0.031 | 2.156 | 438.941 | 412.200 | 1 → 67.35, 2 → 66.90, 3 → 66.30, 4 → 67.25, 5 → 67.50, 6 → 68.00, 7 → 67.85, 8 → 66.85, 9 → 67.25, 10 → 67.50 |

### GSM8K (1319 questions)

| Arm | Selected | Validation | Pass@1, strict [95% CI] | Gain over SFT, strict | Pass@1, lenient [95% CI] | Gain over SFT, lenient | Samples | Sampled tokens | GPU wall | W&B |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT | SFT | — | 61.65% [59.80, 63.50] | — | 83.29% [81.78, 84.80] | — | — | — | 0:17:55 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/5755e93304ba4608935fa20e210084c0) |
| GRPO | 50 | 93.83% | 87.87% [86.45, 89.30] | +26.23 [+24.61, +27.84] | 89.77% [88.47, 91.07] | +6.48 [+5.39, +7.56] | 6,400 | 1,017,333 | 1:28:20 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/0ddc996b3b4d428691c0613f72eed7f7) |
| RLOO | 40 | 94.84% | 88.01% [86.59, 89.43] | +26.36 [+24.74, +27.98] | 89.26% [87.93, 90.59] | +5.97 [+4.99, +6.95] | 6,400 | 947,308 | 1:27:01 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/4218622785f8450eb6d582b9e99c9769) |
| RestEM | 10 | 92.43% | 84.49% [82.85, 86.12] | +22.84 [+21.31, +24.37] | 89.31% [87.96, 90.65] | +6.01 [+5.00, +7.02] | 6,400 | 874,590 | 4:17:52 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/54f505cdeb5a4a369d39b5ab230a09fa) |

#### Paired differences between trained arms, GSM8K

| Comparison | strict | lenient |
|---|---|---|
| GRPO minus RLOO | -0.13 [-0.82, +0.56] | +0.51 [-0.11, +1.13] |
| GRPO minus RestEM | +3.39 [+2.43, +4.34] | +0.46 [-0.19, +1.11] |
| RLOO minus RestEM | +3.52 [+2.58, +4.46] | -0.04 [-0.48, +0.39] |

#### Training health, GSM8K (mean over all iterations)

| Arm | Iterations | Reward | Clip (%) | Grad norm | Truncated (%) | Length | Accepted per round | Validation by iteration |
|---|---|---|---|---|---|---|---|---|
| GRPO | 50 | 0.790 | 0.121 | 0.080 | 0.422 | 158.958 | — | 10 → 84.05, 20 → 92.43, 30 → 93.10, 40 → 93.36, 50 → 93.83 |
| RLOO | 50 | 0.811 | 39.578 | 3.749 | 0.328 | 148.017 | — | 10 → 91.15, 20 → 94.24, 30 → 94.57, 40 → 94.84, 50 → 94.44 |
| RestEM | 10 | 0.778 | — | 0.068 | 0.250 | 136.655 | 499.500 | 1 → 83.11, 2 → 88.07, 3 → 90.08, 4 → 90.55, 5 → 91.69, 6 → 91.55, 7 → 91.35, 8 → 92.09, 9 → 92.02, 10 → 92.43 |

#### Answer format, GSM8K

| Arm | Responses | With a `####` answer | Truncated | Right final number, marked wrong |
|---|---:|---:|---:|---:|
| SFT | 21,104 | 17,319 (82.07%) | 69 | 4,568 |
| GRPO | 21,104 | 21,040 (99.70%) | 63 | 400 |
| RLOO | 21,104 | 21,012 (99.56%) | 85 | 265 |
| RestEM | 21,104 | 20,990 (99.46%) | 78 | 1,017 |

On GSM8K, the strict grader expects the number after `####`. Its ceiling is 98.94% because 14 references contain separators that the grader strips only from responses. The lenient reading looks for the final number and ignores formatting such as commas and trailing decimal zeros. It may credit an accidentally correct last number. The lenient intervals for the trained arms' pairwise differences all include zero. [GSM8K grading and format analysis](20261124-080354_olmo-gsm8k-format-vs-accuracy.md)

## VinePPO and PPO at the baseline cutoff

OLMo VinePPO was pending because the tokenizer's chat markers were absent from the splitter's special-ID set and GSM8K cuts could fall inside sampled tokens. The approved fixes are to recognize added special tokens and advance cuts to the next sampled-token boundary. MATH still requires approval for its upstream splitter change. The full FP32 PPO critic had an estimated 117 GB for weights, gradients and optimizer moments alone. The actor and reference added an estimated 29 GB before activations and the sampling replica. The approved smaller critic uses LoRA plus a trainable value head. [OLMo implementation handoff](../workflow/tasks/agent-tasks/20261124-132948_olmo-vineppo-lora-critic-handoff.md)

## Private bucket

After MATH RestEM verification, inventory job 7291562 counted 11,483 files and 417,753,162,600 logical bytes, leaving about 582.2 GB under the recorded 1 TB allowance. Archive job 7284852 hash-verified all 242 non-frozen files for MATH RestEM (6,657,866,161 bytes). The three other new runs have their own full non-frozen verification receipts in jobs 7284130, 7284132 and 7284133.

All 21 lineages covered by the November 22 reports now have verified run mirrors and the required model base content in the bucket. The shared base copies supply paths removed during the approved tier 1 thinning. The 20 tier 4 critics remain in the bucket. At the inventory check, the second Skynet receipt was absent. The thinning record identifies the tier 2–3 files held on Skynet and the 90 checkpoint files whose only recorded copy is in the bucket. [Thinning and archive evidence](20261124-130743_bucket-thinning-record.md)

## Restore a compact checkpoint

1. Download the run and identify the absolute frozen-blob path and SHA256 recorded in the checkpoint. Download its model's canonical base from the bucket location below.
2. Verify the downloaded base with `sha256sum`. Copy or hard-link it at the exact recorded path, preserving the checkpoint's original directory layout. The loader accepts a hard link, as verified in job 7288364. It rejects a symlink.
3. Load the compact checkpoint with its original model and evaluation configuration. Keep any companion critic blob referenced by a PPO training checkpoint at its recorded path as well.

| Model | Base path under `training/pace_ice/` | SHA256 |
|---|---|---|
| RhoMath | `handoff-rho-vineppo-650-s42-03/attempt-0000/checkpoints/frozen-lora.pt` | `086cc6d318d3aeadf44ab916dd960f5dceb56c14dcd0d1e90f5bdfa45d0b9fbf` |
| OLMo | `handoff-olmo3-math-grpo-50-s42-10/attempt-0000/checkpoints/frozen-lora.pt` | `7a574b6997d1acb19ec12204c10522f106b94ea5ee303417a78c0faa3c1c646a` |

The early seed-42 full-format checkpoints already contain their weights. Some scratch upload receipts still refer to removed `checkpoint-history/` objects. Use the run mirror and canonical base paths when restoring. [Restore and receipt notes](../workflow/tasks/agent-tasks/20261124-130156_olmo-baselines-and-bucket-handoff.md#restoring-a-checkpoint)

Previous results and timings are in the November 22 reports. [Detailed report](20261122-110919_final-report-detailed.md) · [Brief report](20261122-110919_final-report-brief.md)

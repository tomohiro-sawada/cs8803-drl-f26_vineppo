# VinePPO and RLVR on PACE-ICE

20261123-045707 EST · `7f2ad45`

All 21 required training and baseline runs have a completion marker, a 16-round held-out result and a verified online Forge run. 13 required runs have whole-lineage bucket verification. The other 8 retained all required local result files, with incomplete bucket coverage under the Hugging Face storage restriction. Optional long seeds 43 and 44 were dropped first to protect required storage.

## Reading the results

Every trained row uses the checkpoint with the highest mean validation Pass@1, with the earliest iteration breaking ties. The SFT rows evaluate the unadapted model. Each result averages one draw per held-out question across 16 seeded rounds. The 95% intervals in the results table cluster by question, averaging those 16 outcomes before computing the standard error. Paired intervals use differences on the same questions. The three-seed t intervals measure dispersion across training runs. The 650-iteration and OLMo comparisons each have one training seed.

The token columns count only sampled primary and MC continuations used in the recorded training lineage. Two short PPO trials missed one metric row after saving a validated checkpoint and before a failed upload. Long VinePPO missed row 625 for the same reason. The corresponding token counts are measured lower bounds. Times are hours, minutes and seconds. GPU wall time sums the started one-GPU Slurm allocations, including setup, evaluation and unsuccessful links. Start→end includes gaps between those allocations. Initial queue time runs from the first submission to the first start. Separate CPU archive jobs are outside those timings.

“Full” means every file in the exact mirror was downloaded and SHA256-compared before pruning any local checkpoint. The number in parentheses is the Slurm verification job. “Winner and test” and “Winner” denote partial coverage. “Pending” denotes a complete training result with the full bucket copy still unverified.

## Results

| Run | Seed | Winner | Pass@1 [95% CI] | Primary / MC training tokens | GPU wall | Start→end | Initial queue | W&B | Verified bucket copy |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| RhoMath SFT | 42 | SFT | 40.15% [38.05, 42.26] | — | 0:04:35 | 0:04:35 | 0:51:02 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/6e2b00733ec34779ae801c00a320b29d) | Full (7208659) |
| GRPO 200 (1e-4 ablation) | 42 | 200 | 48.26% [45.86, 50.66] | 2,666,859 / 0 | 0:47:30 | 0:47:30 | 0:23:16 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/028dcbe7769a4311b85e940246232edf) | Full (7208659) |
| GRPO 200 (1e-4 ablation) | 43 | 100 | 45.60% [43.28, 47.93] | 2,699,047 / 0 | 0:52:17 | 0:52:17 | 1:00:00 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/ae64c7a874d54513a06ab4f3dc8a2d96) | Full (7208659) |
| GRPO 200 (1e-4 ablation) | 44 | 200 | 47.50% [45.10, 49.90] | 2,808,774 / 0 | 0:58:38 | 0:58:38 | 1:52:20 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/f7a655bd2937437da7f2831ae4e3066e) | Full (7208659) |
| VinePPO 200 (no dedup) | 42 | 150 | 50.47% [48.15, 52.79] | 2,765,239 / 77,554,868 | 1:44:35 | 1:44:35 | 0:22:16 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/799d68000e574e4a8678610f565df53b) | Full (7208659) |
| VinePPO 200 (no dedup) | 43 | 125 | 49.88% [47.54, 52.21] | 2,778,522 / 72,833,734 | 1:42:58 | 1:42:58 | 0:33:50 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/d9966bb4f1b149259f36200cabf1044b) | Full (7208659) |
| VinePPO 200 (no dedup) | 44 | 125 | 48.48% [46.17, 50.80] | 2,695,926 / 72,923,477 | 1:44:47 | 1:44:47 | 1:11:05 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/a46b774c4d5d496894d85c82a76c92b5) | Full (7208659) |
| GRPO 500 (1e-4 ablation) | 42 | 450 | 52.08% [49.72, 54.44] | 7,744,277 / 0 | 1:53:23 | 1:53:23 | 0:42:29 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/4d0549e9034642bd9372147eff790d34) | Full (7208659) |
| GRPO 500 (1e-4 ablation) | 43 | 350 | 50.88% [48.54, 53.22] | 7,627,512 / 0 | 1:56:07 | 1:56:07 | 1:29:46 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/014ace656f404ad5a2a4fa445e02f2a3) | Full (7209032) |
| GRPO 500 (1e-4 ablation) | 44 | 425 | 52.25% [49.87, 54.62] | 7,423,966 / 0 | 1:58:53 | 1:58:53 | 1:29:46 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/f85ac506e433499ba0cae2e07906c2cb) | Full (7209032) |
| GRPO 200 (1e-4 ablation), earlier replicate | 42 | 150 | 47.09% [44.72, 49.46] | 2,690,397 / 0 | 0:42:05 | 0:42:05 | 0:15:32 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/36a30617c8464529800a0ea847efd6b5) | Full (7208659) |
| VinePPO 200 (no dedup), earlier replicate | 42 | 175 | 50.93% [48.58, 53.29] | 2,789,203 / 78,072,615 | 1:50:34 | 1:50:34 | 0:17:44 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/05cc2b2ba8854f3e871594b260135bcf) | Full (7208659) |
| RLOO 200 | 42 | 175 | 46.46% [44.19, 48.72] | 2,605,118 / 0 | 0:48:23 | 0:48:23 | 0:08:36 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/031c6a4410204436ab23d5ddca785050) | Full (7227707) |
| RLOO 200 | 43 | 200 | 46.77% [44.50, 49.04] | 2,565,044 / 0 | 0:45:34 | 0:45:34 | 0:12:37 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/a47fa97417624902ba00cf5f6ec21f13) | Full (7227734) |
| RLOO 200 | 44 | 175 | 45.40% [43.13, 47.67] | 2,633,247 / 0 | 0:49:49 | 0:49:49 | 0:30:43 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/2da621a5f42c41d08705eafde3721f40) | Full (7228207) |
| GRPO 200 | 42 | 150 | 47.37% [45.09, 49.65] | 2,711,588 / 0 | 0:51:05 | 0:51:05 | 0:30:43 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/154b72d1fa604d5d8df0a0df1bd6e82b) | Full (7228227) |
| GRPO 200 | 43 | 200 | 48.53% [46.21, 50.85] | 2,704,787 / 0 | 0:50:57 | 0:50:57 | 0:57:20 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/15e2401a8c204478a9472e9302060331) | Full (7228663) |
| GRPO 200 | 44 | 200 | 46.89% [44.56, 49.21] | 2,686,032 / 0 | 0:53:32 | 0:53:32 | 0:58:21 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/a00706756a784e42931bb8e42f747640) | Full (7228746) |
| PPO 200 | 42 | 200 | 45.57% [43.25, 47.89] | 2,541,225 / 0 | 2:28:34 | 2:28:34 | 1:50:41 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/9a3d956d3eaa479f90624a25c7653d32) | Full (7231508) |
| PPO 200 | 43 | 200 | 48.53% [46.19, 50.87] | ≥2,735,678 / 0 | 2:34:47 | 2:44:26 | 1:48:35 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/a02dde927f9344e59ac17ff2666f9cc8) | Winner and test |
| PPO 200 | 44 | 175 | 47.80% [45.45, 50.14] | ≥2,593,354 / 0 | 2:44:17 | 3:05:31 | 1:21:53 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/0544e7b723c94354bb42cdb237942712) | Winner and test |
| VinePPO 200 (dedup) | 42 | 200 | 51.11% [48.75, 53.47] | 2,764,012 / 52,120,683 | 1:33:43 | 1:33:43 | 1:22:58 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/a4f52111cac94f24a6c2f2986452aa34) | Full (7230112) |
| VinePPO 200 (dedup) | 43 | 200 | 51.40% [49.05, 53.75] | 2,818,999 / 56,020,071 | 1:34:58 | 1:34:58 | 1:23:59 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/fcdd745e1fd14763a1289ad55c2af64a) | Full (7230164) |
| VinePPO 200 (dedup) | 44 | 200 | 52.67% [50.30, 55.03] | 2,750,509 / 48,957,038 | 1:28:51 | 1:28:51 | 1:35:02 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/1dcfccce20dd463c9ca441c62bc2fae9) | Full (7230236) |
| RestEM matched | 42 | 1 | 40.37% [38.22, 42.53] | 2,627,443 / 0 | 0:58:03 | 0:58:03 | 0:33:57 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/75607342034443738694f6863678b0ee) | Full (7230787) |
| RLOO 650 | 42 | 525 | 51.48% [49.14, 53.82] | 34,870,707 / 0 | 7:37:08 | 7:37:08 | 0:47:07 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/1cb4bf0fbcb54428aaae194194ec69b1) | Pending |
| GRPO 650 | 42 | 625 | 53.63% [51.26, 55.99] | 40,727,435 / 0 | 10:33:54 | 10:42:58 | 1:29:18 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/3745648e28cc4077a7426b7de64db222) | Pending |
| PPO 650 | 42 | 575 | 53.62% [51.23, 56.00] | 36,855,940 / 0 | 27:19:46 | 27:50:04 | 2:54:41 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/116ea5f7d5b54879b0a4ecab61e817ae) | Pending |
| VinePPO 650 (dedup) | 42 | 550 | 56.90% [54.50, 59.30] | ≥38,736,542 / ≥629,448,585 | 15:37:05 | 16:27:08 | 0:30:43 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/1738284b47af4e3390aa8b79494403cd) | Winner |
| OLMo MATH SFT | 42 | SFT | 70.59% [67.32, 73.85] | — | 0:33:31 | 0:33:31 | 1:22:51 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/e4ccb98121eb4d9b86842ea569828660) | Full (7230727) |
| OLMo MATH GRPO 50 | 42 | 50 | 71.86% [68.57, 75.16] | 2,789,254 / 0 | 3:57:09 | 4:46:37 | 0:00:02 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/64370195038c4e568bc0f69f97ffae3f) | Winner and test |
| OLMo GSM8K SFT | 42 | SFT | 61.65% [59.80, 63.50] | — | 0:17:55 | 0:17:55 | 0:11:39 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/5755e93304ba4608935fa20e210084c0) | Full (7236496) |
| OLMo GSM8K GRPO 50 | 42 | 50 | 87.87% [86.45, 89.30] | 1,017,333 / 0 | 1:28:20 | 1:28:20 | 2:04:07 | [Run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/0ddc996b3b4d428691c0613f72eed7f7) | Pending |

- GRPO 200 (1e-4 ablation) had n=3, mean 47.12%, sample SD 1.37 pp.
- VinePPO 200 (no dedup) had n=3, mean 49.61%, sample SD 1.02 pp.
- GRPO 500 (1e-4 ablation) had n=3, mean 51.74%, sample SD 0.74 pp.
- RLOO 200 had n=3, mean 46.21%, sample SD 0.72 pp.
- GRPO 200 had n=3, mean 47.60%, sample SD 0.85 pp.
- PPO 200 had n=3, mean 47.30%, sample SD 1.54 pp.
- VinePPO 200 (dedup) had n=3, mean 51.73%, sample SD 0.83 pp.

The paired differences below are in percentage points. Seed columns give question-clustered 95% intervals. The final column measures training-seed variation using a t interval where all three seeds exist.

| Comparison | Seed 42 | Seed 43 | Seed 44 | Mean | 3-seed t interval |
|---|---:|---:|---:|---:|---:|
| VinePPO 200 (dedup) minus GRPO 200 | +3.74 [+2.12, +5.37] | +2.87 [+1.15, +4.59] | +5.78 [+4.16, +7.41] | +4.13 | [+0.42, +7.84] |
| VinePPO 200 (dedup) minus RLOO 200 | +4.66 [+3.02, +6.30] | +4.63 [+2.95, +6.30] | +7.27 [+5.61, +8.92] | +5.52 | [+1.75, +9.28] |
| VinePPO 200 (dedup) minus PPO 200 | +5.54 [+3.84, +7.24] | +2.87 [+1.15, +4.58] | +4.87 [+3.20, +6.54] | +4.43 | [+0.97, +7.89] |
| VinePPO 200 (dedup) minus VinePPO 200 (no dedup) | +0.64 [-0.99, +2.28] | +1.52 [-0.07, +3.11] | +4.18 [+2.65, +5.71] | +2.12 | [-2.46, +6.70] |
| GRPO 200 minus GRPO 200 (1e-4 ablation) | -0.89 [-2.56, +0.77] | +2.93 [+1.37, +4.49] | -0.61 [-2.25, +1.02] | +0.48 | [-4.81, +5.76] |
| VinePPO 200 (no dedup) minus GRPO 500 (1e-4 ablation) | -1.61 [-3.36, +0.14] | -1.00 [-2.78, +0.77] | -3.76 [-5.48, -2.05] | -2.13 | [-5.73, +1.47] |
| VinePPO 200 (dedup) minus GRPO 500 (1e-4 ablation) | -0.97 [-2.81, +0.88] | +0.52 [-1.29, +2.33] | +0.42 [-1.28, +2.13] | -0.01 | [-2.07, +2.05] |
| RestEM matched minus RhoMath SFT | +0.22 [-0.53, +0.97] | — | — | +0.22 | — |
| VinePPO 650 (dedup) minus GRPO 650 | +3.27 [+1.60, +4.94] | — | — | +3.27 | — |
| VinePPO 650 (dedup) minus RLOO 650 | +5.42 [+3.69, +7.16] | — | — | +5.42 | — |
| VinePPO 650 (dedup) minus PPO 650 | +3.28 [+1.54, +5.03] | — | — | +3.28 | — |
| OLMo MATH GRPO 50 minus OLMo MATH SFT | +1.27 [+0.38, +2.17] | — | — | +1.27 | — |
| OLMo GSM8K GRPO 50 minus OLMo GSM8K SFT | +26.23 [+24.61, +27.84] | — | — | +26.23 | — |

### RhoMath at 25,600 primary continuations

Deduplicated VinePPO reached 51.73 ± 0.83% across seeds 42–44. The seed means were 47.60 ± 0.85% for GRPO, 47.30 ± 1.54% for PPO and 46.21 ± 0.72% for RLOO. The dispersion terms are sample standard deviations over training seeds. VinePPO beat each other RL arm in each same-seed comparison. Relative to GRPO, the paired mean gap was +4.13 percentage points, with a three-seed t interval of +0.42 to +7.84. The held-out data distinguish each of the three VinePPO versus GRPO checkpoints at the question level. Three training seeds leave the size of that advantage uncertain.

The deduplicated runs used 67,269–68,661 MC requests and 49.0–56.0M MC tokens each. The three historical no-dedup runs used 95,231–97,278 MC requests and 72.8–77.6M MC tokens each. The new trials ran in 1:28:51–1:34:58, versus 1:42:58–1:44:47 for the historical three-seed group. The improvement in requests and tokens is measured across different sampled trajectories. The four-iteration smoke measured the computation reduction holding the observed prefixes fixed. The +2.12-point mean accuracy change from the historical trials has a three-seed t interval of -2.46 to +6.70 points.

The historical 500-iteration GRPO ablation reached 51.74 ± 0.74% in 1:53:23–1:58:53 with 7.42–7.74M training tokens. Its mean differs from the new VinePPO 200 mean by 0.01 points, with seedwise gaps on either side of zero. It had more primary episodes and more validation candidates. The old no-dedup equal-wall-clock observation therefore remains a comparison specific to those historical arms. A single baseline or historical replica at the same fixed seed also cannot establish reproducibility, since vLLM runs differed before their first update.

Matched RestEM used exactly 25,600 primary continuations. Its held-out result was 40.37%, +0.22 points above the SFT baseline with a paired question interval of -0.53 to +0.97. Its first MLE phase was the validation winner. Validation fell from 51.61% after phase 1 to 45.24% after phase 10. The unadapted validation baseline was 53.62%, and the trained winner was selected from the trained candidates.

### OLMo at 6,400 primary continuations

MATH GRPO improved held-out Pass@1 from 70.59% to 71.86% at one training seed. The paired question-level gain was 1.28 points [0.38, 2.17]. GSM8K GRPO improved Pass@1 from 61.65% to 87.87%. The paired gain was 26.23 points [24.61, 27.84]. The GSM8K grader extracted an answer in 17,319 of 21,104 SFT responses and 21,040 of 21,104 GRPO responses. Accuracy among extracted answers was 75.12% and 88.14%, respectively. The gain includes improved answer-format adherence and more correct responses among the answers the verifier could read. The two checkpoints also changed which responses were parseable, so these rates cannot isolate a reasoning-only effect.

### Paper comparison

The paper's RhoMath GSM8K Figure 4 reports 40.3% SFT, 42.8% RestEM, 44.5% RLOO, 44.6% GRPO, 50.1% PPO and 53.4% VinePPO. Our SFT pipeline scored 40.15%. The short matched 200-iteration results reproduce VinePPO's lead over PPO, GRPO and RLOO at equal primary episodes. The GRPO–RLOO gap is smaller than either method’s gap to VinePPO. PPO 200 and matched RestEM fall below the paper's values, while both group methods score higher. Training decoding, LoRA, training budgets and the RestEM recipe differ, so gaps against Figure 4 measure the difference between protocols as well as any implementation differences. [VinePPO Figure 4 and Table 2](https://proceedings.mlr.press/v267/kazemnejad25a.html)

### RhoMath at 332,800 primary continuations

VinePPO reached 56.90% Pass@1 on iteration 550, GRPO 53.63% on iteration 625, RLOO 51.48% on iteration 525 and PPO 53.62% on iteration 575. The paired gaps from VinePPO were +3.27 points against GRPO, +5.42 against RLOO and +3.28 [+1.54, +5.03] against PPO. These results use the same training-seed label and same questions. Allocated GPU wall was 7:37:08 for RLOO, 10:33:54 for GRPO, 15:37:05 for VinePPO and 27:19:46 for PPO. Their primary sample budgets match. Their token budgets and GPU hours differ.

## Design choices

- Sampled token IDs define the actions and states. Learner-side scoring uses the sampled IDs, and vLLM receives the exact prefix IDs for every Monte Carlo (MC) branch. Behavior probabilities are recomputed by the learner with fixed rollout actor weights before updating. Upstream reconstructs the sequences and the MC prefixes from text. Re-encoding can change the action sequence even when the decoded text is identical. [Upstream episode generator](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/episode_generators/math_episode_generator_with_mc_advantages.py)
- The MATH splitter assigns a token to the step containing its first character. A cut inside a sampled token advances through that token. Coincident boundaries merge, and MC branching and advantage assignment both use the same resulting token boundary. Alignment errors and genuine splitter failures raise. GSM8K still requires a token-aligned cut and treats the text after the first answer marker as a single step. A terminal stop token belongs to the final step. The exact-token replay of the original 512 RhoMath MATH responses converted all 42 inside-token failures to successes and preserved the 463 prior successes. Seven genuine upstream splitter assertions remain. The reported OLMo MATH arm uses GRPO and does not call the reasoning-step splitter.
- Deduplicated VinePPO samples nine continuations per distinct token prefix per iteration. Episodes sharing a prefix reuse that one MC estimate. Upstream deduplicates prefix text. The earlier RhoMath arms requested each prefix occurrence separately. Reuse changes the covariance of estimates across episodes and avoids repeated requests. As in the paper, rewards are terminal and VinePPO's step credit is a difference between successive values.
- The actor uses a BF16 base and rank-16, alpha-32 all-linear LoRA with FP32 trainable parameters. The paper reports full-model fine-tuning. LoRA keeps a frozen actor and reference plus the vLLM replica within one GPU and makes actor candidates small enough to retain individually. Training sampling uses temperature 1.0, top-p 1.0 and no top-k cutoff so learner-side old log-probabilities target the unfiltered rollout distribution. The paper's training uses temperature 0.6 and top-p 0.9. The evaluation protocol uses temperature 0.35 and top-p 0.9 for one sample per question per round. Training sets a 2,048-token context. Evaluation sets 2,047 for RhoMath and 2,048 for OLMo. Generation allows up to 1,024 new tokens. The training reward is -1 for truncation.
- VinePPO and PPO whiten advantages over the masked response tokens in each actor minibatch, use a token-mean policy objective and clamp each nonnegative reference k3 term to 10 before summing by response. GRPO standardizes rewards within the prompt group using population standard deviation with epsilon 1e-6, then computes PPO-style token ratios and averages the loss and unclamped k3 within each response before averaging responses. RLOO subtracts the mean of the other rollouts, skips whitening and standard-deviation scaling, clips the sequence importance ratio, and sums unclamped token k3 per response. Whitening spans the logical minibatch with sample variance, so memory-only microbatch splits preserve the statistics and loss denominators. The policy clip is 0.2. The KL coefficient is 1e-4 for PPO and VinePPO and 3e-3 for the new GRPO and RLOO arms. Historical GRPO at 1e-4 is a coefficient ablation. Raw reference KL and clipping metrics use different units across arms. Dividing summed KL by mean response length gives a rough length normalization whose weighting differs from GRPO's mean of sequence means. The paper gives the coefficients but leaves this reduction unspecified. Its pinned repository does not implement GRPO or RLOO, so the corresponding loss definitions in this report describe this implementation.
- Actor LR is 1e-5. The short and historical arms used a constant LR. The 650-iteration runs use 3% linear warmup by optimizer step and degree-one polynomial decay to zero, including the PPO critic. Saved learner state includes schedule position, optimizer moments, and Torch and data RNG states. The 650-iteration design has 64 prompts with eight responses and minibatch 64. Batch scale and LR schedule both change from the 200-iteration design.
- PPO has a separate fully trainable FP32 backbone and value head, a 1e-5 AdamW critic LR, gamma 1, GAE lambda 1 and value clipping 0.2. Both the critic's rollout values and its returns are fixed for the update. Truncations are terminal. Compact actor candidates stay available for evaluation while the critic and optimizer rotate through at most two locally generated large blobs during a save.
- Matched RestEM draws 320 prompts with eight samples for each of ten policy iterations, exactly 25,600 primary samples. It keeps at most ten correct continuations per prompt, resets the LoRA adapter and optimizer before each MLE phase, and applies eight token-NLL epochs in batches of 64 at constant 1e-5, with no KL. The last model from each MLE phase becomes the next sampler. Four-round validation over the ten final checkpoints chooses the held-out candidate. Upstream starts each phase from the SFT model, samples the full training set every round, uses its filtered temperature/top-p recipe, trains at 1e-6 with warmup, and does within-iteration validation and early stopping. The matched run tests RestEM within the same primary sample budget as the short RL arms. [Matched recipe and upstream references](../workflow/tasks/agent-tasks/20261120-180926_baselines-long-runs-olmo-guide.md#5-restem)
- OLMo uses `allenai/Olmo-3-7B-Instruct-SFT` at revision `e1452fc572d51966ff4aaeb25118b891eb93e549` on H200. The ChatML user prompt asks for a boxed MATH answer or a separate GSM8K line beginning with `#### `. Decode ends on the tokenizer EOS or the registered `<|im_end|>` ID. RhoMath keeps its `[MATH_TASK]` prompt and next-problem stop string. The OLMo runtime probe compares adapter-induced score changes within each backend. The verified adapter update, retirement check and restored-checkpoint evaluation allow the requested OLMo model to run.

## Training health

The validation trajectories are the recorded seed-42 checkpoints, in percent. Iteration 0 is the validation baseline. The table above gives the validation-selected held-out evaluation for every finished seed.

| Seed-42 validation checkpoints | Iteration → Pass@1 (%) |
|---|---|
| GRPO 200 (1e-4 ablation) | 0 → 53.75, 25 → 54.42, 50 → 55.83, 75 → 58.24, 100 → 58.78, 125 → 59.58, 150 → 58.85, 175 → 59.05, 200 → 60.79 |
| VinePPO 200 (no dedup) | 0 → 53.62, 25 → 57.04, 50 → 58.51, 75 → 59.52, 100 → 62.80, 125 → 61.39, 150 → 66.76, 175 → 65.08, 200 → 66.09 |
| GRPO 500 (1e-4 ablation) | 0 → 53.75, 25 → 55.23, 50 → 58.51, 75 → 60.39, 100 → 62.73, 125 → 62.47, 150 → 61.93, 175 → 62.40, 200 → 64.34, 225 → 64.41, 250 → 64.61, 275 → 63.94, 300 → 63.34, 325 → 63.47, 350 → 64.54, 375 → 63.67, 400 → 64.88, 425 → 66.15, 450 → 67.69, 475 → 66.29, 500 → 67.29 |
| GRPO 200 (1e-4 ablation), earlier replicate | 0 → 53.62, 25 → 54.42, 50 → 57.37, 75 → 59.32, 100 → 58.45, 125 → 60.52, 150 → 61.93, 175 → 61.13, 200 → 59.79 |
| VinePPO 200 (no dedup), earlier replicate | 0 → 53.75, 25 → 54.89, 50 → 60.32, 75 → 61.26, 100 → 61.93, 125 → 62.00, 150 → 64.34, 175 → 67.23, 200 → 66.35 |
| RLOO 200 | 0 → 53.62, 25 → 54.36, 50 → 56.50, 75 → 55.29, 100 → 57.37, 125 → 58.71, 150 → 59.58, 175 → 60.72, 200 → 60.59 |
| GRPO 200 | 0 → 53.62, 25 → 55.36, 50 → 58.91, 75 → 58.58, 100 → 58.45, 125 → 60.66, 150 → 62.94, 175 → 62.80, 200 → 62.73 |
| PPO 200 | 0 → 53.55, 25 → 53.75, 50 → 52.41, 75 → 54.29, 100 → 56.64, 125 → 58.98, 150 → 56.50, 175 → 60.25, 200 → 61.13 |
| VinePPO 200 (dedup) | 0 → 53.69, 25 → 55.76, 50 → 59.18, 75 → 61.66, 100 → 63.67, 125 → 64.28, 150 → 64.54, 175 → 64.95, 200 → 65.75 |
| RestEM matched | 0 → 53.62, 1 → 51.61, 2 → 50.80, 3 → 51.34, 4 → 48.73, 5 → 49.73, 6 → 48.59, 7 → 47.86, 8 → 45.71, 9 → 46.11, 10 → 45.24 |
| RLOO 650 | 0 → 53.69, 25 → 56.37, 50 → 58.71, 75 → 59.72, 100 → 59.12, 125 → 62.53, 150 → 63.07, 175 → 64.61, 200 → 63.34, 225 → 64.01, 250 → 65.75, 275 → 64.48, 300 → 64.81, 325 → 66.49, 350 → 66.76, 375 → 66.69, 400 → 66.76, 425 → 66.42, 450 → 66.49, 475 → 67.63, 500 → 66.76, 525 → 67.76, 550 → 66.35, 575 → 66.82, 600 → 66.89, 625 → 67.23, 650 → 66.76 |
| GRPO 650 | 0 → 53.62, 25 → 54.42, 50 → 56.84, 75 → 60.12, 100 → 61.73, 125 → 63.61, 150 → 63.61, 175 → 64.28, 200 → 63.67, 225 → 63.61, 250 → 65.42, 275 → 65.21, 300 → 66.89, 325 → 65.55, 350 → 65.15, 375 → 66.22, 400 → 66.09, 425 → 65.88, 450 → 65.01, 475 → 65.28, 500 → 65.88, 525 → 66.62, 550 → 67.02, 575 → 66.42, 600 → 66.89, 625 → 67.63, 650 → 67.36 |
| PPO 650 | 0 → 53.75, 25 → 56.77, 50 → 58.78, 75 → 63.00, 100 → 63.81, 125 → 64.08, 150 → 63.67, 175 → 64.81, 200 → 65.62, 225 → 67.90, 250 → 66.96, 275 → 69.24, 300 → 67.69, 325 → 68.70, 350 → 67.16, 375 → 67.29, 400 → 68.43, 425 → 68.50, 450 → 69.77, 475 → 68.10, 500 → 70.78, 525 → 70.17, 550 → 70.51, 575 → 71.18, 600 → 70.38, 625 → 70.78, 650 → 70.24 |
| VinePPO 650 (dedup) | 0 → 53.69, 25 → 58.38, 50 → 62.53, 75 → 66.55, 100 → 66.96, 125 → 66.69, 150 → 68.70, 175 → 70.44, 200 → 68.63, 225 → 68.43, 250 → 67.49, 275 → 69.57, 300 → 68.10, 325 → 69.91, 350 → 69.91, 375 → 70.78, 400 → 70.31, 425 → 71.25, 450 → 70.58, 475 → 70.84, 500 → 70.98, 525 → 70.64, 550 → 72.59, 575 → 71.45, 600 → 71.58, 625 → 71.92, 650 → 71.78 |
| OLMo MATH GRPO 50 | 0 → 67.15, 10 → 66.25, 20 → 66.90, 30 → 67.30, 40 → 67.05, 50 → 67.90 |
| OLMo GSM8K GRPO 50 | 0 → 68.57, 10 → 84.05, 20 → 92.43, 30 → 93.10, 40 → 93.36, 50 → 93.83 |

The training summaries average each run's last 25 recorded iterations, or all ten RestEM iterations, then average seeds by arm. Clip rates refer to token ratios for PPO, VinePPO and GRPO, and sequence ratios for RLOO. Gradient norms are measured before clipping. Ref. KL uses the arm-specific units documented above. MC value SD is the mean logged per-iteration standard deviation.

| Arm | Reward | Ref. KL | Clip (%) | Grad norm | Truncated (%) | Length | Zero adv. (%) | Zero-variance groups (%) | Value loss | MC value mean / SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GRPO 200 (1e-4 ablation) | 0.574 | 0.122 | 0.624 | 0.513 | 0.646 | 111.576 | 37.244 | 41.417 | — | — / — |
| VinePPO 200 (no dedup) | 0.610 | 9.333 | 0.461 | 0.667 | 0.062 | 113.542 | 58.967 | — | — | 0.537 / 0.418 |
| GRPO 500 (1e-4 ablation) | 0.625 | 2.831 | 0.593 | 0.570 | 0.594 | 134.270 | 39.657 | 45.833 | — | — / — |
| GRPO 200 (1e-4 ablation), earlier replicate | 0.563 | 0.653 | 0.665 | 0.528 | 0.156 | 109.958 | 40.790 | 43.500 | — | — / — |
| VinePPO 200 (no dedup), earlier replicate | 0.580 | 9.763 | 0.477 | 0.652 | 0.031 | 114.414 | 59.725 | — | — | 0.506 / 0.417 |
| RLOO 200 | 0.544 | 3.469 | 35.490 | 14.757 | 0.115 | 101.266 | 33.503 | 35.083 | — | — / — |
| GRPO 200 | 0.570 | 0.084 | 0.453 | 0.475 | 0.156 | 110.002 | 35.543 | 38.750 | — | — / — |
| PPO 200 | 0.539 | 6.439 | 0.449 | 0.588 | 0.083 | 107.849 | 0.000 | — | 0.082 | — / — |
| VinePPO 200 (dedup) | 0.610 | 9.719 | 0.467 | 0.662 | 0.062 | 114.977 | 58.133 | — | — | 0.544 / 0.418 |
| RestEM matched | 0.422 | — | — | 0.478 | 0.316 | 102.634 | — | — | — | — / — |
| RLOO 650 | 0.644 | 7.134 | 28.617 | 17.191 | 0.039 | 106.658 | 40.514 | 44.125 | — | — / — |
| GRPO 650 | 0.668 | 0.114 | 0.180 | 0.424 | 2.156 | 127.393 | 36.273 | 47.812 | — | — / — |
| PPO 650 | 0.721 | 10.641 | 0.233 | 0.625 | 0.016 | 114.825 | 0.000 | — | 0.045 | — / — |
| VinePPO 650 (dedup) | 0.753 | 13.452 | 0.258 | 0.807 | 0.039 | 119.728 | 67.426 | — | — | 0.693 / 0.403 |
| OLMo MATH GRPO 50 | 0.631 | 0.003 | 0.028 | 0.027 | 1.906 | 438.685 | 46.410 | 54.500 | — | — / — |
| OLMo GSM8K GRPO 50 | 0.870 | 0.028 | 0.036 | 0.048 | 0.469 | 159.446 | 59.607 | 68.750 | — | — / — |

### Resume continuity

| Run | Resume boundary and next observed update | Evidence |
|---|---|---|
| PPO 200, seed 43 | 200 → held-out test | The checkpoint and winning validation were durable. Row 200 was suppressed by upper-layer archival failure, and the new allocation ran the frozen test. |
| PPO 200, seed 44 | 175 → 176 | The iteration-175 candidate and validation were durable. Row 175 is missing. The recovery wrote its rollover at 176 and trained through 200. |
| VinePPO 650, seed 42 | 328 → 329 and 625 → 626 | After the scheduled handoff, actor LR moved from 5.1071e-6 to 5.0912e-6 at consecutive schedule positions, with reward 0.719→0.705 and summed KL 12.605→12.543. The bucket interruption at 625 left a validated checkpoint with a matching SHA256, and the next link published rollover at 626. The missing row prevents an adjacent metrics check at that boundary. |
| GRPO 650, seed 42 | 494 → 495 | Consecutive LR 2.4742e-6→2.4584e-6, reward 0.674→0.713, mean token KL 0.117→0.096 and 512 samples in each row. |
| PPO 650, seed 42 | 3 restores | 170 → 171 (reward +0.064, value loss -0.0149), 365 → 366 (reward -0.096, value loss +0.0061), 561 → 562 (reward +0.012, value loss -0.0017). All PPO rollover positions used the expected actor and critic LRs after restore. |
| OLMo MATH GRPO | Two restarted attempts | The eventual 50-update run started fresh. The prior attempts failed at the first batch and before the iteration-2 checkpoint, respectively. |

The other completed training runs in the results table used a single allocation. The original smoke, fault-injection and probe lineages have their individual attempts and wall times in the handoff log.

### Allocation and iteration timing

This timing table averages within each arm over completed runs. Iteration time covers primary and secondary sampling, learning and checkpoint validation. It excludes startup and the separate frozen held-out test. The `other` phase includes task handling such as answer verification and segmentation. MC generation is already part of MC estimation. For the complete end-to-end wall time, use the results table.

| Arm | Minutes in recorded iterations / mean per run | Main phases (% of measured iteration time) |
|---|---:|---|
| GRPO 200 (1e-4 ablation) | 43.5 | primary generation 10.1%, score cache 12.6%, actor refresh 2.5%, actor update 59.2%, checkpoint 0.4%, validation 14.8%, other 0.4% |
| VinePPO 200 (no dedup) | 96.4 | primary generation 3.7%, mc estimation 56.3%, score cache 5.5%, actor refresh 1.5%, actor update 23.0%, checkpoint 0.2%, validation 6.4%, other 3.4% |
| GRPO 500 (1e-4 ablation) | 105.2 | primary generation 10.9%, score cache 12.4%, actor refresh 2.3%, actor update 58.4%, checkpoint 0.2%, validation 15.5%, other 0.4% |
| GRPO 200 (1e-4 ablation), earlier replicate | 36.2 | primary generation 10.6%, score cache 11.6%, actor refresh 2.0%, actor update 56.5%, checkpoint 1.7%, validation 17.3%, other 0.3% |
| VinePPO 200 (no dedup), earlier replicate | 100.3 | primary generation 3.5%, mc estimation 56.5%, score cache 5.4%, actor refresh 1.0%, actor update 22.1%, checkpoint 0.7%, validation 7.4%, other 3.4% |
| RLOO 200 | 39.1 | primary generation 9.4%, score cache 13.3%, actor refresh 2.1%, actor update 55.6%, checkpoint 4.4%, validation 14.9%, other 0.2% |
| GRPO 200 | 41.4 | primary generation 9.6%, score cache 12.8%, actor refresh 2.0%, actor update 55.7%, checkpoint 5.0%, validation 14.6%, other 0.3% |
| PPO 200 | 142.8 | primary generation 2.6%, ppo estimation 4.0%, score cache 3.7%, actor refresh 0.6%, actor update 15.2%, critic update 48.0%, checkpoint 22.1%, validation 3.8% |
| VinePPO 200 (dedup) | 83.4 | primary generation 4.3%, mc estimation 47.6%, score cache 6.3%, actor refresh 1.1%, actor update 26.9%, checkpoint 2.4%, validation 7.2%, other 4.1% |
| RestEM matched | 46.5 | primary generation 4.5%, actor refresh 0.2%, actor update 73.7%, checkpoint 4.8%, validation 16.8% |
| RLOO 650 | 446.4 | primary generation 6.5%, score cache 14.5%, actor refresh 0.6%, actor update 72.8%, checkpoint 0.2%, validation 5.0%, other 0.4% |
| GRPO 650 | 614.8 | primary generation 6.8%, score cache 11.1%, actor refresh 0.5%, actor update 77.4%, checkpoint 0.1%, validation 3.9%, other 0.2% |
| PPO 650 | 1610.8 | primary generation 2.1%, ppo estimation 4.6%, score cache 4.0%, actor refresh 0.4%, actor update 18.0%, critic update 66.1%, checkpoint 3.4%, validation 1.3% |
| VinePPO 650 (dedup) | 906.8 | primary generation 3.1%, mc estimation 49.8%, score cache 6.9%, actor refresh 0.3%, actor update 31.7%, checkpoint 0.8%, validation 2.1%, other 5.1% |
| OLMo MATH GRPO 50 | 147.5 | primary generation 6.1%, score cache 6.5%, actor refresh 0.5%, actor update 28.1%, checkpoint 1.0%, validation 45.3%, other 12.5% |
| OLMo GSM8K GRPO 50 | 65.0 | primary generation 6.8%, score cache 11.9%, actor refresh 0.8%, actor update 50.8%, checkpoint 2.3%, validation 27.5% |

## Execution and archive evidence

The watch controller now checks Slurm accounting for completed prerequisites. The original controller stopped when it submitted an `afterok` dependency on a job the Slurm controller had already purged. PACE's accounting retained its successful exit. The replacement controller records dependency state, admission and disk waits. A rejected submit backs off independently. Running jobs own their after-any successors and their progress guard. The first production VinePPO rollover released the other three 650-iteration arms as soon as the new link recorded a completed update.

The pinned RhoMath production smoke passed a real interruption and final evaluation. CPU suites and the OLMo adapter and restoration probes are in the handoff log. The HF control matched uninterrupted learner state bitwise. The resumed vLLM control reproduced the expected prompt IDs, seeds and sampling parameters. vLLM sampling on accelerators can vary on identical seed/configuration, so equivalence to an uninterrupted real vLLM run is a different test. At the RhoMath 1+3 worker check, vLLM and HF greedily matched 6/8 prompts. The logged minimum top-two logit gap was zero across all eight prompts and did not establish the cause of the two divergences.

The cleanup receipt `handoff-20261120/pruned-verified-checkpoints-20261121.jsonl` records 101 previously verified checkpoint paths across 32 runs and about 213.8 GiB reclaimed. Later verifications have separate per-run receipts. Downloads went to node-local `/tmp`. The short-run PPO storage path separates the compact actor from the 12,414,435,295-byte critic and optimizer blob. The rotating PPO recovery slots use up to two large files during a save. Full-archive pruning requires an exact directory verification receipt. The archive mirror excludes W&B's local cache.

The private bucket refused new writes with HTTP 403 requesting automatic credit recharge. Whole-run verification for PPO 200 seeds 43 and 44, the two OLMo GRPO runs and all four RhoMath 650 runs is incomplete. Earlier successful partial-copy jobs separately hash-verified their candidate and held-out files. New whole-run archives, and even the GSM8K GRPO test-only copy, failed at the later restriction. The RhoMath 650 VinePPO run reached the same restriction during iteration-625 archival. Its new chain used the exact saved learner state and locally retained its checkpoints. The other long arms initially used local preservation. See the execution log for the exact roots and bytes verified for each lineage. [Hugging Face storage limits](https://huggingface.co/docs/hub/main/en/storage-limits)

The OLMo MATH recovery distinguishes infrastructure and data failures. Job 7210784 applied the adapter but failed a threshold measuring cross-backend disagreement. The paired probe saw an adapter-effect projection of 0.970, relative squared error 0.069 across 1,499 scored tokens and no drift in the base route. The corrected production run uses the original requested model. The subsequent retries corrected added-token stop identity, malformed parser boundaries, compact checkpoint provenance, a startup wall-time allowance with an unchanged hard CPU budget, and six malformed gold answer spans in the official training split. All 11,500 training examples were retained. The full-solution extraction rules and held-out references remained the same. [Handoff and allocation log](../workflow/tasks/agent-tasks/20261120-212505_remote-agent-handoff.md#execution-log)

## Bottlenecks

1. The private bucket is the remaining release gate. Peak scratch pressure came from separate OLMo frozen-state copies and PPO critic rotation. Deduplicating immutable model payloads by exact hash or provisioning verified storage would release more storage than shortening retention of the compact actors.
2. PPO's FP32 critic accounted for 66.1% of the long run's recorded iteration time, with 3.4% in checkpointing. Profile critic forward/backward and optimizer kernels separately from the checkpoint writes before changing the precision or optimizer. The short, fully archived seed-42 run also spent 34.4 minutes in checkpoints. Later local chains demonstrate storage rotation independently of the optimizer.
3. MC estimation accounts for about half the VinePPO 650 iteration time. Exact-prefix dedup already reduces the sampled work. The recorded MC generation is nested inside MC estimation. Improving batching, request scheduling and reuse across identical states within the same iteration targets the existing bottleneck.
4. The learner update dominates long GRPO and RLOO iteration time, at 77.4% and 72.8%, respectively. Same-sample budget comparisons therefore differ from the GPU-hour comparisons. The long-run time comparison uses the actual Slurm allocations in the results table.
5. Held-out and validation decoding have different costs. Validation takes 45.3% of measured MATH GRPO iteration time, against 27.5% for GSM8K OLMo. Shared immutable model staging and worker startup are candidates for reducing evaluator overhead. Size allocation reservations from observed peaks and durations so independently ready jobs can backfill. Initial GPU queue time peaked at 2:54:41 for long PPO in this matrix.

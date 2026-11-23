# VinePPO and RLVR on PACE-ICE

20261123-045707 EST · `7f2ad45`

**Training.** All 21 required 16-round tests are complete and verified in Forge. The complete per-run table includes validation-selected checkpoint, token costs, intervals, GPU wall time, queue time and bucket evidence. [Detailed report](20261122-110919_final-report-detailed.md#results)

**RhoMath GSM8K, 25,600 primary samples.** Deduplicated VinePPO scored 51.73 ± 0.83% across seeds 42–44. GRPO scored 47.60 ± 0.85%, PPO 47.30 ± 1.54%, and RLOO 46.21 ± 0.72%. VinePPO beat GRPO by 4.13 points on average (+0.42 to +7.84 across training seeds). The historical 1e-4 GRPO results are a KL ablation. Matched RestEM scored 40.37% on 25,600 samples, close to the 40.15% SFT baseline.

**RhoMath GSM8K, 332,800 primary samples.** At seed 42, VinePPO reached 56.90% at iteration 550 in 15:37:05 of GPU time. GRPO reached 53.63% at iteration 625 in 10:33:54. RLOO reached 51.48% at iteration 525 in 7:37:08. PPO reached 53.62% at iteration 575 in 27:19:46 of allocated GPU time. The optional two seeds were dropped to protect mandatory storage.

**OLMo 3 7B SFT and GRPO.** MATH improved from 70.59% to 71.86%, with a 1.28-point paired gain [0.38, 2.17]. The successful training and test took 3:16:15. GSM8K improved from 61.65% to 87.87%, with a 26.23-point paired gain [24.61, 27.84]. Training and test took 1:28:20. Extractable GSM8K answers increased from 17,319 to 21,040 of 21,104, while accuracy among extracted answers increased from 75.12% to 88.14%.

**PACE and storage.** The watcher recovered from the Slurm accounting dependency failure. The first production VinePPO rollover opened three additional long arms, and all four trained concurrently on H100. PPO rotation bounded large critic storage. Hugging Face returned HTTP 403 requiring automatic credit recharge. 13 full production mirrors passed SHA256 verification. 8 required lineages have incomplete bucket coverage, including two short PPO runs, both OLMo GRPO runs and the long RhoMath arms. The detailed report captures the measured compute bottlenecks and the handoff log records each interruption. [Execution log](../workflow/tasks/agent-tasks/20261120-212505_remote-agent-handoff.md#execution-log)

# Handoff: finish baselines, OLMo, unattended long runs, reports

Written 20261120-212505 at commit `ca1ced8`; results and job states updated 20261120-212749. **The working tree has uncommitted changes** (section 2); nothing below is committed.

You are taking over from the agent that wrote this. Tom is away for several days and will not answer questions. Read this file first; it supersedes the two companion documents wherever they disagree:

- `20261120-184322_unattended-training-plan.md`: resume design, test tables, cluster dry runs, report outline. Still the reference for those.
- `20261120-180926_baselines-long-runs-olmo-guide.md`: algorithm detail for dedup, RLOO, PPO, RestEM, long runs, OLMo. Its section 6 targets the wrong model and its "decisions needed" list is resolved here.
- Earlier history: `20261119-225338_vineppo-grpo-pace-ice-handoff-2.md` ("How to submit" still applies).

Statements marked **(estimate)** or **(inference)** were not measured or verified.

`OUT=/home/hice1/tsawada7/scratch/cs8803-drl-f26_vineppo/training/pace_ice`

## 1. Goal

1. Implement the remaining baselines: PPO, RLOO, RestEM.
2. Integrate `allenai/Olmo-3-7B-Instruct-SFT`.
3. Launch training for 1 and 2 and the long RhoMath runs, unattended, so every run survives the 16-hour job limit, crashes and node failures. Budget 4–5 days, 9 days maximum.
4. Write two reports (detailed and dense). **Tom said these can wait until he is back**; do them last.

Tom's priority order if time runs short: resume mechanism, then RLOO and the long run, then PPO, then RestEM, then OLMo.

## 2. State at handoff

### Jobs

| Job | What | State at 21:27 |
|---|---|---|
| 7208659 | Bucket archive: push, then verify, of 46 run directories | Still running. Push finished for all 46 (191 GB in the bucket); verify (re-download and SHA256 of every file) had not yet reported a run. Log: `artifacts/training/pace_ice/slurm/bucket-archive-push-verify-7208659.out`; a verified run prints a `verified ...` line. |

No training job is running, so the main checkout is free to edit. First action: `squeue -u tsawada7`, then read that job's log. If it failed, its scratch directory `bucket-verify-archive` under the project's scratch root may be left behind; the script refuses to reuse it.

### Results so far

RhoMath 1.1B, GSM8K, LoRA rank 16, LR 1e-5, 128 primary episodes per iteration, four validation rounds every 25 iterations, 16-round test on the validation-selected checkpoint. All at KL coefficient 1e-4 and without prefix dedup.

| Arm | Seed 42 | Seed 43 | Seed 44 | Mean | Tokens per run | Wall clock |
|---|---:|---:|---:|---:|---:|---:|
| SFT (pipeline evaluator) | 40.15% | | | | | |
| GRPO, 200 iterations | 48.26% | 45.60% | 47.50% | 47.12% | 2.7M | 0:48–0:59 |
| VinePPO, 200 iterations | 50.47% | 49.88% | 48.48% | 49.61% | 76–80M | 1:43–1:45 |
| GRPO, 500 iterations | 52.08% | 50.88% | 52.25% | 51.74% | 7.4–7.7M | 1:53–1:59 |

- **Equal episodes:** VinePPO leads GRPO by 2.5 points on average, ahead in all three seeds; a three-seed t-interval on the gap is −1.6 to +6.6.
- **Equal wall-clock:** GRPO at 500 iterations beat VinePPO at 200 in all three seeds (by 1.6, 1.0 and 3.8 points; mean 2.1). Only the seed-44 gap excludes zero at the question level, and a three-seed t-interval on VinePPO minus GRPO is −5.7 to +1.5, so GRPO's lead is consistent but not established. GRPO had 20 selection candidates against VinePPO's 8.
- **A fixed seed does not reproduce a run** on the vLLM backend: two seed-42 runs differ in iteration 1, before any update. The 500-iteration GRPO runs scored 2–5 points higher on validation at iterations 150 and 200 than the 200-iteration runs with identical settings, which is run-to-run variation.
- **Not established:** whether VinePPO at 200 iterations had converged. Its validation peaked at iteration 125–150 in all three seeds of this set and was flat or lower by 200, while its per-response reference KL grew to 8–10. Tom assumes it is still improving; the long runs are what test this.
- Outputs: `$OUT/long-{grpo,vineppo}-rhomath-gsm8k-seed*-vllm-h100-{200,500}iter-*`, `$OUT/sft-baseline-test-rhomath-gsm8k-vllm-h100-01`. W&B groups `seeds-20261120` and `walltime-20261120` at `https://forge.coreweave.com/wandb/tsawada/vineppo`.
- The older seed-42 pair (`...-200iter-01`, 47.09% and 50.93%) is a separate replicate on earlier code.

### Uncommitted changes in the main checkout

| Files | Change | Tested |
|---|---|---|
| `src/algorithms/vineppo.py`, `experiments/training/run.py`, `experiments/training/README.md`, `tests/algorithms/test_vineppo.py`, `tests/core/test_vllm_inference.py`, `tests/runtime/test_train.py` | Prefix dedup (section 4a) | Full CPU suite, 1,563 passed (job 7208412). No GPU run yet. |
| `experiments/training/evaluate.py`, `experiments/training/EVALUATION.md`, `tests/test_checkpoint_evaluation.py` | `--split test --baseline` scores the unadapted model on test | Same suite; used by job 7204554 |
| `scripts/remote_sync/hf_bucket_sync.py`, `scripts/README.md` | Bucket mirror with SHA256 verification | Run end to end on one run directory; no unit tests |
| `bash.sh`, `pyproject.toml`, `uv.lock`, `prose/reflections/*`, `wandb-metric-keys.txt`, `wandb-panels.tsv` | Tom's own work | Leave alone |

Tom has not asked for a commit. Section 3 covers why you need pinned code anyway and what to do about it.

## 3. Rules and hazards

**Code and jobs**

- **A running job re-reads code from the checkout it was launched from** at every validation (the evaluator is a child process). Editing that tree mid-run mixes versions. Never launch a multi-hour job from a tree you will keep editing.
- **Pinned launches need a commit and a worktree.** Tom's standing rule is no commits or branches without his request, and he is unreachable. My recommendation: commit only the paths in the first three rows of the table above to `tom/work` locally, with a clear message, do not push, and create a detached worktree on scratch for launches. Record the commit hash here. If you would rather not commit, use the fallback already used twice: launch from the main checkout, save a snapshot tarball beside the runs (`$OUT/*-code-*.tgz` shows the pattern), and do not touch `src/` or `experiments/` until those jobs end.
- **Never stage Tom's files** (last table row). Use explicit paths with `git add`.
- **Tests and GPU code run under `sbatch`**, never on the login node. The login node caps each user at one CPU-second per second and 4 GB of memory.
- **Smoke before seeds.** Gate long jobs with `--dependency=afterok:<id> --kill-on-invalid-dep=yes`.
- **Do not cancel jobs you did not submit.** Check job names against this file first.
- **Do not change settings to make a failing run pass.** A run that halts stays halted; record why.

**Cluster limits (measured; probes under `$OUT/slurm-lifecycle-probe-20261120/`)**

- 16-hour job limit. A job that needs longer must chain.
- **Per-job cap of 16 GPU-hours, enforced.** A 2-GPU, 16-hour job sat pending with reason `QOSMaxGRESMinutesPerJob` and I cancelled it. One GPU for 16 hours is the most a job can ask for; two GPUs get 8 hours.
- At the time limit the job gets SIGTERM about 17 s late, then about 30 s before SIGKILL. `--signal=B:USR1@60` reached the batch shell 90 s early; the variant without `B:` was not delivered to steps.
- `sbatch` works from inside a job. Requeue works (`SLURM_RESTART_COUNT` increments; use `--open-mode=append`). `SLURM_JOB_END_TIME` is set.
- 50 submitted jobs per user. No cron or scrontab.
- Compute nodes reach huggingface.co and the W&B host.
- H100s were fully allocated this evening; waits ranged from 5 minutes to over an hour.

**Storage**

- Scratch is at 245 of 300 GB. PPO and OLMo checkpoints do not fit until space is freed.
- **Delete nothing locally until `hf_bucket_sync.py verify` has passed for that exact run directory.** Push success is not verification. The script never deletes; deletion is a separate, deliberate step.
- Keep locally, even after verification: every run in the results table above, and anything a running job reads.
- Secrets are in `.env` (`HF_TOKEN`, `WANDB_API_KEY`). Never print them, pass them on a command line, or write them to a log. The bucket script redacts the token from its own errors.

**Repository conventions (`AGENTS.md`)**

- Fail loudly with asserts; no fallbacks or silent repair. Minimal code, no duplication.
- Obey `AGENT INSTRUCTIONS` notes in docstrings.
- When you add to `prose` or update a log, include the timestamp (`source ./bash.sh; timestamp`) and `git rev-parse --short HEAD`.

## 4. Decisions already made by Tom

Do not reopen these.

### a. Prefix dedup: implemented, uncommitted

Each distinct token prefix is sampled once per iteration and its continuations are shared by every episode with that prefix. Upstream deduplicates on prefix text; we match token IDs. Pooling is dropped. The rationale is in the `estimate_vineppo_batch` docstring and `experiments/training/README.md`; new runs record `mc_prefix_policy` in `metadata.json`. All existing VinePPO runs are the no-dedup arm.

Remaining: a four-iteration GPU smoke, then VinePPO reruns. Expect about 26% fewer MC requests **(estimate)**; `mc_expansions` should drop by roughly the number of prompts times seven per iteration.

### b. KL coefficient: follow the paper

GRPO and RLOO use 3e-3; PPO and VinePPO keep 1e-4. All existing GRPO runs used 1e-4.

- **Why 1e-4 was used:** `kl_coef` is a single default in `src/runtime/config.py` with no CLI option, set when the VinePPO loop was written. I verified the PPO/VinePPO value against upstream and missed that the paper's Table 2 gives a different value for GRPO and RLOO. Upstream's repository has no GRPO or RLOO code or configs, so the paper table is the only source.
- **Do:** add `--kl-coef` to `run.py` and the `launch.sh` allowlist, record it, then rerun GRPO alongside RLOO at 3e-3. Keep the 1e-4 GRPO runs as a labelled ablation.
- **Watch out:** the KL term is reduced differently per loss. VinePPO/PPO (`actor.py`) sums over tokens per response and clamps each token at 10; GRPO (`grpo.py`) averages over tokens; RLOO sums per response, unclamped. So the logged `reference_kl` is not comparable across arms (convert with response length), and at the same coefficient GRPO's penalty is about 100 times weaker than a summed one. The paper does not say which reduction its 3e-3 assumes. Use 3e-3 with the existing reductions and state this in the report. Do not change the reductions without Tom.

### c. Hugging Face bucket

- Bucket: `tsawada/cs8803-drl-f26_vineppo` (private). Tool: `scripts/remote_sync/hf_bucket_sync.py push|verify RUN_DIR...`. Paths mirror scratch (`training/pace_ice/<run>`); W&B local folders are skipped.
- Measured: 4.45 GB uploaded in 43 s; verifying that run took 3 minutes.
- **During training:** Tom wants checkpoint state pushed as runs progress so scratch does not fill. Add this to the resumable wrapper or the trainer's checkpoint step: push after each candidate checkpoint, verify, and only then let pruning remove local copies that are neither the best nor the latest. A failed push must fail loudly and must not block or corrupt training state; decide and document which.
- The 46 archived directories exclude the two GRPO-500 runs that were still running when the list was made (seeds 43 and 44, both finished now). Push and verify those two.

### d. RestEM: matched to our RL budget, not paper scale

Same primary-sample budget as the RL arms (iterations × episodes), not the paper's ten passes over the full training set. Design and upstream recipe are in the guide, section 5. State the deviation in the report.

### e. Long runs: paper scale, paper schedule

- 650 iterations, 512 episodes per iteration (`--prompts-per-iteration 64 --num-samples 8 --minibatch-size 64`).
- LR warmup over 3% of steps and polynomial decay, as in the paper. **This is new code**: the scheduler state must be saved in the checkpoint, restored on resume, covered by the resume identity check and by the bitwise-resume tests.
- **Seeds were not settled.** I asked "one seed or three" and Tom answered "yes". Run one seed of each arm first; add seeds 43 and 44 if the budget allows.
- Estimates per seed at this scale: VinePPO about 15 hours after dedup, GRPO and RLOO about 7, PPO about 8 **(all estimates)**. VinePPO is at the 16-hour limit, so it will chain.

### f. Model: Olmo-3-7B-Instruct-SFT, MATH first

- `allenai/Olmo-3-7B-Instruct-SFT`, revision `e1452fc572d51966ff4aaeb25118b891eb93e549`, not gated, 7.30B parameters in BF16 (about 14.6 GB), `Olmo3ForCausalLM`. Tom proposed OLMo 2 7B SFT or Llama 3.2 3B Instruct as fallbacks.
- **Tasks:** both, MATH first (more headroom). Instruct an explicit answer format in the prompt.
- **vLLM support:** Tom is right that `Olmo3ForCausalLM` is officially supported with LoRA. It is served through vLLM's Transformers modeling backend (the docs list it in that table with LoRA checked, and the installed 0.34.0 backend class declares LoRA support). My earlier "go/no-go risk" overstated it. What is still unmeasured on our stack is throughput on that backend and runtime adapter loading through our `publish_adapter` path; run `scripts/diagnostics/check_remote_sampler.py` once on this model before building on it.
- **Unverified:** the SFT variant's MATH accuracy (the card I read was for the final Instruct model, 87.3). Measure it first; if there is little headroom, record that and continue with the next step anyway.
- **Answer format:** the vendored MATH grader extracts `\boxed{...}`, "final answer is $...$", or "the answer is"; the GSM8K grader expects a `#### ` line. Pick the instruction to match and keep the grader unchanged.
- **Code that assumes RhoMath** (guide section 6 has line numbers): the `[MATH_TASK]` prompt defined twice and asserted by the evaluator; the `\n\n\nProblem:` stop string and 2,047-token context; an assert that every prompt starts with BOS; segmentation rejecting special tokens in the prompt, which a chat template contains; step splitters tuned to SFT output. This tokenizer's BOS and EOS are the same token, and generation also stops on `<|im_end|>`.
- **Resources (estimates):** actor and reference are 14.6 GB each plus the replica on one 80 GB GPU; expect a smaller microbatch. Compact checkpoints store about 29 GB of frozen weights per run, so the bucket path in (c) must work first.

### g. Resume fidelity

Tom accepted the definition in the plan, section 2e: learner state round-trips bitwise; with a deterministic sampler an interrupted run equals the uninterrupted one bitwise; on the real backend the resumed trainer starts byte-identical to the checkpoint and issues the same sampling requests. Bitwise identity to an uninterrupted vLLM run is not achievable and is not required.

### h. Watcher

Tom suggested a persistent tmux session on the login node that watches and manages jobs.

- Feasible: tmux 3.2a is installed, and a monitoring script of Tom's has run on this login node for 22 hours. It does not survive a login-node reboot and is tied to one node (`login-ice-gnr-2`).
- **Design it as a supplement.** Correctness must come from the self-resubmitting job chain (plan, section 2). The watcher only does what the chain cannot: refresh a status file, push finished runs to the bucket, and launch later waves when their gate conditions are met.
- Keep it light (login-node limits above) and make every action it takes idempotent and logged.
- Whether you yourself keep running for days is outside what I can see. If you may not, make the watcher sufficient to launch the remaining waves alone, and say in this file which waves depend on it.

## 5. Order of work

Each step launches as soon as it is ready; nothing waits for a later step.

1. **Housekeeping.** Confirm job 7208659 ended cleanly with 46 `verified` lines; push and verify the two unarchived runs; decide commit versus snapshot (section 3).
2. **Free scratch.** After verification, remove checkpoints of smoke, benchmark and diagnostic runs only (about 100 GB; `metrics.jsonl` and results stay). List what you deleted in the execution log.
3. **Resume mechanism** (plan, sections 2–4): PACE resumable wrapper with self-resubmission and a crash-loop guard, deadline and signal checkpointing, a separate resume slot, tests T1–T9, GPU tests G1–G3, cluster dry runs D1–D10. Keep jobs within one GPU and 16 hours.
4. **Dedup GPU smoke, `--kl-coef`, RLOO smoke.** Then launch RLOO and GRPO at 3e-3 and VinePPO with dedup, three seeds each at the 200-iteration settings, through the new wrapper.
5. **LR schedule, then the long runs** (4e).
6. **PPO** (guide section 3): set λ to 1.0, decide value clipping, load the critic in FP32 independently of `--dtype`, allow the vLLM sampler, and deal with 13 GB checkpoints **(estimate)** via the bucket.
7. **RestEM** (4d).
8. **OLMo** (4f).
9. **Reports** (plan, section 7), when Tom is back or everything else is done.

## 6. Unresolved items to carry in the report

- GRPO beat VinePPO at equal wall-clock in all three seeds, at a tenth of the sampled tokens. The long runs and the 3e-3 reruns decide whether that holds.
- At 1+3 workers, greedy output matched the HF learner on 6 of 8 prompts; the near-tie diagnostic (`learner_min_top2_logit_gap`) is broken and reports 0.0.
- The MATH step splitter fails on about 1% of RhoMath samples; long MATH VinePPO runs will abort on these. No policy is approved for dropping them.
- Validation is not reproducible across runs at a fixed seed.

## 7. Keep this file current

Append an execution log below with timestamp and commit hash for each entry: job IDs, what passed, what was deleted, and any departure from this plan with its reason.

## Execution log

### 20261120-214939 · `fa78a89`

- Slurm initially showed only archive job 7208659. Left it running. Its output had 46 pushes and no printed verification lines at this check. Deleted no scratch files.
- Job 7209032 independently pushed and SHA256-verified both remaining 500-iteration GRPO directories, seeds 43 and 44. Both stay local.
- Used the recommended commit-and-worktree path. Local `4a16b8e` pins the inherited dedup, baseline evaluator and bucket mirror. `d3dbf16` adds PACE chaining, checkpoint history archival, an independent rolling resume slot and the KL CLI. `fa78a89` adds the per-optimizer-step polynomial schedule and PPO BF16 LoRA, FP32 critic, λ = 1 and value clip 0.2. No push or staging of Tom's files.
- The PACE warning is relayed by a per-attempt file from the Slurm batch process to the learner. The learner checks it at the same boundary as its deadline. Archive failures are fatal to the link after the checkpoint is durable and before any candidate pruning. Successors retry the boundary without altering hyperparameters.
- CPU 7209143 exposed fixture and import-path failures. CPU 7209229 passed 1,579 and exposed one remaining scratch symlink issue. CPU 7209318 passed all 1,580 in 2m00s from the physical worktree path. Earlier dependent jobs remained gated. H100 7209319–7209323 are the gated VinePPO, RLOO, PPO, deadline/signal and crash-loop tests.
- OLMo weights are staged at the pinned revision. Job 7209038 used the live submit directory and exited in 14s. Job 7209066 used the pinned directory and exited in 15s because the download omitted tokenizer/config files. CPU job 7209103 staged the missing exact filenames. Probe 7209105 awaits H100 priority.

### 20261120-224905 · `55774a4`

- Archive 7208659 completed in 1:13:51 with 46 verified directories. The two GRPO-500 archives in 7209032 both verified in 16:39. Cleanup 7209612 took 3s and removed 25 checkpoint files totaling 86,950,966,927 bytes. The exact path and size for every deletion is in `$SCRATCH_PROJECT/handoff-20261120/cleanup-receipt.json` and the Slurm output. Eligible directories were `bench-grpo-hf-01`, `bench-grpo-vllm-colocated-01`, `bench-grpo-vllm-colocated-h100-02`, `bench-vineppo-hf-h100-02`, `bench-vineppo-vllm-colocated-02`, `checkpoint-review-compact-lora-01`, `checkpoint-review-full-01`, `math-vineppo-strict-smoke-01`, `math-vineppo-strict-smoke-02`, `smoke-e2e-grpo-01`, `smoke-e2e-vineppo-01`, `smoke-e2e-vineppo-gsm8k-compact-h100-01`, `smoke-resume-grpo-01`, `smoke-selection-resume-rhomath-h100-01`, `smoke-selection-resume-rhomath-h100-02`, `smoke-vllm-1p1-grpo-a40-01`, and `smoke-vllm-1p1-vineppo-a40-01`. The comparison runs and all non-checkpoint results were retained.
- RestEM and the OLMo chat/stop protocols are in `638f3c5`. The matched RestEM configuration uses ten rounds of 320 distinct prompts per round, eight samples per prompt, eight update epochs, LoRA reset before the updates, and constant 1e-5 LR. The long RL schedule uses degree-one decay to zero. The authors specify polynomial decay and 3% warmup without specifying the degree in the paper table.
- CPU jobs 7209501 and 7209603 verified 1,585 tests each. Their test runtimes were 1:55 and 2:29. Commit `d167344` adds bitwise restore and actual sampler-request audits. Commit `fa4f00f` retains the newest candidates and best across attempts only after archive verification and excludes nested W&B directories from finished-run mirrors.
- D1 GRPO jobs 7209322 and 7210182 exited normally on the deadline at iterations 5 and 12 after 7:29 and 7:15. The third link 7210340 resumed at 12. D8 job 7209323 deliberately failed on the GPU assertion after 57s. Its successor 7210188 wrote `HALTED` and cancelled its own successor.
- VinePPO 7209319 and RLOO 7209320 validated iteration 2 but failed in candidate verification after 6:19 and 5:37. They passed a string to the SHA256 reader and halted as designed on repeated failure. OLMo 7209105 served the pinned model on vLLM but exited after 6:38 because the acceptance diagnostic incorrectly required separate GPUs. Local `55774a4` fixes both errors and adds a test for download verification and receipt publication.
- The login-node tmux session did not persist. Slurm-owned watcher 7210467 replaces it, gated on CPU job 7210466. Its plan, launch receipts and progress are in `$SCRATCH_PROJECT/handoff-20261120/control-02/`, where `SCRATCH_PROJECT` is the scratch root above `training/`. It schedules the fixed smoke and dry runs, then the comparison and long waves when their proofs are present. Disk-heavy training runs enter serially after quota checks. The first 650-iteration VinePPO run requests 8-hour links to exercise production-scale rollover within the 16 GPU-hour cap. The subsequent long arms request 16-hour links. No production result is available at this point.

### 20261120-230234 · `a1bb081`

- Watcher 7210467 submitted its first GPU allocations, but the CPU allocation's Slurm memory variable reached those jobs and conflicted with their GPU memory directive. The manifest also still pointed its job entries to `d167344` rather than the corrected archive and acceptance code. GPU jobs 7210518–7210520 stopped in 1s each before training. D3–D5 jobs 7210523–7210525 stopped in 1s each because the predecessor-time query used the same separator as the disjunctive GPU constraint. Cancelled only our superseded pending jobs and watcher. The independent GRPO deadline chain kept running.
- Commit `a1bb081` removes inherited scheduler variables on submission and preserves the entire disjunctive constraint. CPU 7210727 passed 1,597 tests in 1:47 and the Slurm allocation took 1:50. This includes split-at-every-boundary tests for PPO, exact resume with the deterministic fake vLLM replica, selection cadence and concurrent ownership, and the checkpoint receipt test.
- Replacement watcher 7210728 is active with afterany successor 7210770. Every command in `$SCRATCH_PROJECT/handoff-20261120/control-03/wave-plan.json` was checked against the pinned `a1bb081` worktree before submission. Jobs 7210771–7210773 are the replacement VinePPO, RLOO and PPO smokes. Jobs 7210774–7210781 cover G1 and D2–D7 plus the dedicated 600-second automatic warning. RestEM smoke is 7210782 and OLMo runtime LoRA acceptance is 7210784. The OLMo MATH segmentation diagnostic plans 512 sampled responses after acceptance. Production remains gated on real completion and interruption evidence.
- Existing comparison wall clocks from Slurm, excluding queue time. GRPO 200, seeds 42–44 respectively, took 47:30, 52:17 and 58:38. VinePPO 200 took 1:44:35, 1:42:58 and 1:44:47. GRPO 500 took 1:53:23, 1:56:07 and 1:58:53. The RhoMath SFT test evaluation took 4:35.

### 20261120-231101 · `7cddd63`

- D1 GRPO completed iteration 24, froze iteration 16 as the winner, finished all 16 held-out rounds at 41.51%, and published a verified test record and `COMPLETE`. Jobs 7209322, 7210182, 7210340 and 7210475 took 7:29, 7:15, 7:29 and 10:07. Total allocated wall clock was 32:20 and first start to final end was 32:50. The last allocation reached Slurm's time limit after publishing completion and cancelling 7210616. It was recorded as `TIMEOUT`, so this is a successful artifact/resume test with an unclean final scheduler exit.
- `7cddd63` makes the learner defer a frozen test when the allocation is ending. The successor handles it from the frozen manifest. CPU 7210895 passed 1,598 tests in 1:46. The patch has its own GPU probe 7210896 and Slurm warning injector 7210897. The other active waves remain on their pinned `a1bb081` worktree.
- The watcher sent real Slurm batch USR1 to the VinePPO and RLOO smokes at iteration 1. Both learners finished iteration 2, downloaded and SHA256-verified their frozen weights and candidate, and exited 0 with a completed resume boundary. First-link wall times were 7:17 (7210771, successor 7210789) and 7:18 (7210772, successor 7210790). VinePPO used 31 and 39 MC requests for 44 and 53 per-episode values, respectively, saving 27 of 97 requests (27.8%) over separate sampling of those same prefixes. RLOO recorded KL coefficient 0.003. The successor evaluations are pending.

### 20261121-122202 · `7e0a2ea`

Watcher 7210728 stopped at 23:47 after Slurm rejected an `afterok` dependency on finished HF control 7210775. PACE's controller purges completed jobs after 300 seconds, while `sacct` retained the successful allocation. None of the production waves launched. The patch checks each planned dependency's durable `COMPLETED 0:0` accounting before submitting, records waiting dependencies and missing evidence, and isolates and backs off a rejected submission while processing other ready jobs.

| Run | Slurm job IDs | Allocated wall clock | First allocation start to last end | Result |
|---|---|---:|---:|---|
| VinePPO dedup smoke | 7210771, 7210789 | 14:10 | 47:50 | 4 iterations and 16 test rounds, 40.44%, winner at 2. Dedup saved 52 of 218 MC requests (23.9%). |
| RLOO smoke | 7210772, 7210790 | 18:14 | 52:23 | 4 iterations and 16 test rounds, 40.41%, winner at 2. |
| RestEM smoke | 7210782, 7211431, 7211893 | 20:07 | 41:41 | 4 iterations and 16 test rounds, 40.27%, winner at 2. Recovered its final test after quota exhaustion in the middle allocation. |
| PPO smoke | 7210773, 7211444, 7211937, 7212026 | 13:06 | 36:22 | Reached the iteration-4 checkpoint. Candidate verification exhausted quota and the chain halted. |
| D2 learner kill | 7210774, 7211120 | 2:30 | 30:11 | Resumed at 1 and finished 6 iterations. |
| G1 HF control | 7210775 | 1:23 | 1:23 | Finished 6 iterations. |
| D3 checkpoint-write kill | 7210777, 7211338 | 14:59 | 31:02 | Used previous complete boundary at 1 and finished the test. |
| D4 validation kill | 7210778, 7211353 | 9:35 | 27:47 | Revalidated iteration 2 and finished with candidates 2, 4 and 6 exactly once. |
| D5 final-test kill | 7210779, 7211383 | 14:30 | 33:25 | Restarted the test from its frozen selection and completed all 16 rounds. |
| D6 node-requeue path | 7210780 | 8:40 | 32:25 | Same job ID, restart count 1, recovered iteration 1 and completed. |
| D7 worker kill | 7210781, 7211423 | 10:10 | 27:13 | Bitwise restore at iteration 1 and completed test. |
| D9 automatic warning | 7210776, 7211204 | 7:39 | 22:27 | Slurm delivered the warning before the first update finished. Successor halted at 0. |
| Frozen-test deferral | 7210896, 7211515, 7211938 | 7:53 | 31:19 | Deferral marker and frozen selection were written. Successor started its test and halted after hitting quota. |
| OLMo vLLM LoRA acceptance | 7210784 | 2:31 | 2:31 | Loaded the model and reached the adapter-effect assertion. The acceptance gate remains closed. |

- G1 and G2 audits 7224731 and 7224732 both passed in 52s each. The two real HF final checkpoints match bitwise, and the actual resumed vLLM prompts, seeds and sampling parameters match their expected values. The full test suite on the watcher patch passed 1,600 tests in 1:47 under 7224823.
- Watcher 7224824 resumed the existing plan from `$SCRATCH_PROJECT/handoff-20261120/control-04/`, adopted both audit receipts, submitted D9 retry 7224875, and has successor 7224874. The retry allows more time before the warning on H100/H200. Subsequent checkpoint-heavy admissions are serialized with disk thresholds. Scratch had 10.2 GiB free. Job 7224877 is pushing and verifying 11 completed smoke and recovery directories in the private bucket. No cleanup was performed on those runs.
- W&B API audit 7224853 scanned 210 runs in `tsawada/vineppo` and found zero PPO, RLOO or DPO+ matches by run name or algorithm config. The RhoMath PPO and RLOO allocations were smoke tests with W&B disabled. The online production runs stayed gated. DPO+ had no implementation or launch plan in this checkout.

### 20261121-124219 · `0b08bd3`

- Stopped watcher 7224824, successor 7224874, and pending D9 retry 7224875 before replacing the serial admission plan. Before stopping verifier 7224877, collected its three completed verification lines. Together with the 46 verified archive lines from 7208659 and two from 7209032, those authorized deletion of 62 checkpoint files across 20 run directories. Reclaimed 126,406,535,319 bytes (117.7 GiB). Kept metrics, results, selections, and logs. Every deleted path and byte count is in `$SCRATCH_PROJECT/handoff-20261120/pruned-verified-checkpoints-20261121.jsonl`. Scratch was approximately 172.3 of 300 GiB used afterward.
- `9637623` puts archived PPO critic and optimizer state in separate SHA-addressed blobs so the learner retains at most two during rotation. Whole-run verification and checkpoint verification now download to node-local storage. CPU 7225178 and verifier 7225180 exposed duplicate filesystem rows from PACE `findmnt`. `88bbacf` accepts repeated rows only when all report a local filesystem. Resubmitted the remaining eight probe verifications as 7225330.
- In 7210784, vLLM reported loading both LoRA snapshots through the Olmo3 Transformers backend. The old check compared the adapter run to the HF baseline and used the HF–vLLM baseline gap as its threshold. `b08430c` compares the paired changes within each backend, and checks their agreement against repeated vLLM baseline measurements. H200 probe 7225195 will determine whether the actual adapter effect agrees with HF. The OLMo gate is still closed.
- `0b08bd3` contains the production wave generator, retains 20 GiB of headroom after accounting for the peak reservations of concurrent jobs, and opens the remaining 650-iteration arms when VinePPO publishes a successful rollover. Whole-run archival verifies the remote copy before deleting local checkpoints and records the path of each deletion. CPU 7225324 and GRPO chain smoke 7225325 gate training at this commit. Fresh online PPO smoke 7225326 gates PPO. Injectors 7225327 and 7225328 send the actual Slurm warning after the first update. Watcher 7225329 uses `$SCRATCH_PROJECT/handoff-20261120/control-05/`. Each production run will log online to `tsawada/vineppo`.

### 20261121-125241 · `8726b44`

- CPU 7225324 passed all 1,602 tests in 2:01 and used 2:06 of Slurm wall clock. Watcher 7225329 started and refreshed its gate status. Production stayed gated while Slurm lacked GPUs for both smokes.
- Review found two ways PPO could use space beyond the published two-blob limit. A killed writer could leave an unfinished critic file, and AdamW CPU step counters could keep a deleted critic file mapped after resume. The learner now discards its abandoned critic staging files under the run lock and loads critic state into memory without retaining a mapping. The retention regression test covers the abandoned staging file.
- Replaced the two queued, unstarted GPU allocations and their injectors. CPU 7225616, GRPO chain smoke 7225617 (`handoff-production-chain-smoke-02`), online PPO smoke 7225618 (`handoff-smoke-ppo-05`), and injectors 7225619–7225620 are pinned to this commit. Watcher 7225621 uses `control-06` and keeps the original 9-day deadline. The plan retains the production names and settings. OLMo acceptance 7225195 remains queued on H200, and legacy bucket re-verification 7225330 continues from node-local storage.
- Removed 13 clean detached worktrees after confirming no queued or running allocations used them. Historical result audit 7225553 checked 16 test rounds and the frozen selection where applicable, then calculated question-level intervals from the per-example results. Slurm retained accounting for all 12 historical rows, including the earlier seed-42 replicates.

### 20261121-125513 · `1c7ea5d`

- The project uses Forge for W&B, and the launch environment had no `WANDB_BASE_URL`. This commit pins `https://api.forge.coreweave.com` and the `tsawada` entity. The online PPO smoke uses the same project. CPU 7225616 passed 1,602 tests in 2:02 on the preceding PPO fix.
- Replaced the unstarted validation jobs and watcher with CPU 7225658, GRPO chain smoke 7225659 (`handoff-production-chain-smoke-03`), online PPO smoke 7225660 (`handoff-smoke-ppo-06`), warning injectors 7225661–7225662, and watcher 7225663 in `control-07`. All 48 production and archive job entries pin `1c7ea5d`, and all 24 GPU entries specify the Forge entity. The inputs and original deadline are unchanged.
- Node-local verifier 7225330 printed verification lines for D3–D6 before cleanup. Reclaimed another 15 checkpoint files totaling 28,225,210,179 bytes (26.3 GiB) from those four directories. The exact paths were appended to `pruned-verified-checkpoints-20261121.jsonl`. The verifier continued through the remaining probes.

### 20261121-130211 · `1c7ea5d`

- CPU 7225658 passed all 1,602 tests in 1:42 and used 1:45 of Slurm wall clock. Watcher 7225663 is refreshing `control-07/watch-status.json` and has successor 7225697. Slurm lists GPU preflights 7225659, 7225660 and 7225195 as pending for resources. Every production gate still has a live upstream allocation.
- Relaunched D9 as 7225740 (`handoff-d9-auto-warning-04`) with the previously planned 22-minute link and the Slurm warning 600 seconds before its limit. The initial D9 failed because its warning preceded the first completed update. Relaunched frozen-test deferral as 7225741 (`handoff-frozen-test-defer-02`), with injector 7225742 during final validation. Its earlier run successfully deferred the test and exhausted scratch quota during the successor. These probes have no production gates.
- After the first two verified cleanup passes, Lustre reported 145.3 GiB used of 300 GiB. Verifier 7225330 subsequently printed the D7, VinePPO and RLOO probe verifications. The final RestEM archive verification was still running at this check.

### 20261121-130406 · `1c7ea5d`

- Legacy archive verifier 7225330 finished all eight exact-run checks in 22:50. Pruned 12 checkpoint files totaling 18,917,639,972 bytes from D7 and the VinePPO, RLOO and RestEM smokes on the strength of their verification lines. The three passes in this handoff recovered 173,549,385,470 bytes (161.6 GiB) from 89 checkpoint paths across 28 run directories. All paths and byte counts are in `pruned-verified-checkpoints-20261121.jsonl`.
- Removed the two superseded detached worktrees for `8726b44` and `0b08bd3` after their last allocations ended. The active watcher and training preflights use `1c7ea5d`. The queued paired OLMo acceptance probe uses `b08430c`.

### 20261121-131135 · `1c7ea5d`

- Slurm still showed the unstarted preflights waiting for resources. Broadened the GRPO chain smoke 7225659 to A40, L40S, A100 80 GB, H100 or H200. The existing colocated A40 benchmark used the same prompts, samples and microbatch. Broadened PPO smoke 7225660 to H100 or H200. Updated only the two queued resource constraints. The production H100 and H200 settings and the model and training settings are unchanged.

### 20261121-131952 · `1c7ea5d`

- H200 OLMo acceptance 7225195 completed in 2:42. Over 1,499 token scores, vLLM's within-backend adapter delta had a projection of 0.970 onto the learner delta and relative squared error of 0.069. Mean absolute change was 0.0562 in vLLM and 0.0557 in HF. Repeated vLLM base scores were identical. The retired adapter was rejected, and the unadapted, zero-adapter and HF greedy sequences agreed on all eight prompts.
- The old test required a change larger than ten times the zero-adapter HF–vLLM gap. The measured gap of 0.246 would require more than 2.457, while the triangle inequality bounds the old comparison by 0.246 + 1.204 = 1.450 even though the adapter worked. The probe confirmed OLMo 3 applies runtime LoRA through the Transformers backend. `acceptance.json` is present, so OLMo 3 remains the production model. Its first launch still awaits the pinned chain smoke.
- The pinned GRPO chain preflight 7225659 started on A40 at 13:12:30. Its successor 7225968 and warning injector 7225661 are present in Slurm.

### 20261121-132355 · `1c7ea5d`

- GRPO preflight 7225659 received USR1 from 7225661 and yielded with a durably saved boundary at iteration 2 in 8:27. Successor 7225968 resumed at 2 and published `ROLLOVER` after completing iteration 3. Its eight-iteration training and 16-round selected test are in progress.
- PPO 7225660 ran on H200, received USR1 from 7225662 and yielded at iteration 2 in 5:37. It uploaded and SHA256-verified the 4,400,347,131-byte frozen file, the 12,414,435,295-byte critic and optimizer file, and the 151,914,948-byte compact iteration file. Only `critic-2.pt` remained locally at the boundary. Its successor is 7226121.
- W&B API audit 7226191 finished in 3s and confirmed `[handoff-smoke-ppo-06](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/b9d24e0eb84e4bf188a8925c06e3b23e)` in `tsawada/vineppo` with algorithm `ppo`. RLOO production still waits for the complete same-commit GPU chain proof.
- Frozen test 7225741 wrote its selection and deferred the held-out test during USR1 in 2:35. Successor 7226150 started with frozen position 2 and is running the held-out test. D9 7225740 has started on H200 and passed its first update.

### 20261121-133445 · `1c7ea5d`

- D9 automatic Slurm warning 7225740 wrote its USR1 stop file and yielded at iteration 14. Successor 7226122 completed iterations 15 and 16 and wrote `ROLLOVER` and `COMPLETE`. The links took 12:32 and 3:58, for 16:30 allocated and 16:33 from first start to final end.
- Frozen test successor 7226150 completed all 16 rounds on the frozen iteration-1 winner at 40.11% Pass@1 in 5:17. The two links used 7:52 allocated, with 8:18 from first start to final end.
- PPO successor 7226121 restored iteration 2 and wrote its rollover after iteration 3. After writing the iteration-4 critic, only that critic file remained local. The iteration-2 critic's archive receipt and small actor candidate remained available for earlier-winner evaluation. The PPO test and the GRPO chain's 16-round test are still running.

### 20261121-133842 · `1c7ea5d`

- Same-commit GRPO chain 7225659 and 7225968 completed all eight iterations and 16 test rounds. Validation selected iteration 4 and test Pass@1 was 40.31%. The links used 8:27 and 16:53, for 25:20 allocated and 25:26 from first start to final end. `ROLLOVER`, `test-complete.json` and `COMPLETE` are present.
- Watcher 7225663 submitted the first 12 jobs at the same commit. RLOO 200 seeds 42–44 are 7226473–7226475. VinePPO 650 seed 42 is 7226476. GRPO 200 seeds 42–44 are 7226477–7226479, and VinePPO 200 seeds 42–44 are 7226480–7226482. RestEM's one-iteration budget check is 7226483, and the OLMo MATH SFT baseline is 7226484. The remaining 650 arms wait for the first VinePPO rollover. The queued jobs use separate GPU allocations and share the scratch headroom calculation.
- Scheduled 7226512 to mirror and verify the chain smoke and 7226513 to do the same for the PPO smoke after its final test. Their archival wrappers will add per-path pruning receipts under `control-07`. Lustre reported about 142.6 GiB free immediately after the first wave was submitted.

### 20261121-134206 · `1c7ea5d`

- PPO smoke completed all four iterations and 16 test rounds at 40.32% Pass@1 on the iteration-2 winner. The two links took 5:37 and 14:01, for 19:38 allocated and 23:35 from first start to final end. The online run, rollover, test record and `COMPLETE` are present. PPO's smoke gate is open.
- The watcher held PPO production at a 164 GiB admission threshold while reserving space for the first 12 jobs. Chain smoke archive 7226512 verified and pruned its checkpoint files in 2:40. PPO smoke archive 7226513 is running. Archive jobs 7226604 and 7226605 verify and then reclaim only eligible `.pt` files from the D9 and frozen-test preflights. This releases additional space without reducing the 20 GiB headroom.

### 20261121-134646 · `1c7ea5d`

- The fresh PPO smoke and both recovered diagnostics passed whole-run SHA256 verification. The watcher admitted PPO 200 seed 42 as 7226686 with about 171.8 GiB free before its reservation. Seeds 43–44 need 196 GiB free with the current queued reservations. The first production wave is queued in Slurm.
- Found three old 12.57 GB PPO files plus its frozen weights under halted `handoff-smoke-ppo-02`. Job 7226701 mirrors and verifies that exact directory. Job 7226702 does the same for the superseded chain RLOO, chain VinePPO and frozen-test probes. These directories still have their original `HALTED` markers. Their `.pt` files will be eligible for pruning only after the corresponding verification lines appear.

### 20261121-135432 · `1c7ea5d`

- RLOO seeds 42 and 43 are training concurrently. Jobs 7226473 and 7226474 had completed iterations 6 and 8 at the status check. Forge API audit 7226845 confirmed online `[RLOO seed 42](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/031c6a4410204436ab23d5ddca785050)` with algorithm `rloo` and the earlier PPO smoke with state `finished`. DPO+ has no run in this handoff's production plan.
- Verifier 7226702 printed all three superseded-probe verification lines. Verifier 7226701 completed the 42.2 GB halted PPO directory push and is still checking its downloaded contents. Cleanup remains gated on the PPO verification line.

### 20261121-135550 · `1c7ea5d`

- The old halted PPO directory verified in job 7226701 after 7:15. The other three superseded probes verified in job 7226702 after 5:01. Pruned 12 `.pt` checkpoint paths totaling 56,059,879,972 bytes (52.2 GiB) across the four directories. Their logs, selections, metrics and original `HALTED` markers remain. The same JSONL deletion receipt now contains 101 unique paths across 32 old run directories.
- The watcher admitted PPO seed 43 as job 7226940. Seed 44 remains governed by the combined peak reservations and 20 GiB of headroom.

### 20261121-140817 · `1c7ea5d`

- Both running RLOO seeds reached iteration 74 and have SHA256 receipts through iteration 50. Their metadata records the pinned clean commit and KL coefficient 0.003. No active production run had halted. Slurm's other admitted GPU jobs were pending for priority. The watcher heartbeat was current and all outstanding gates had a live producer or a current disk wait.

### 20261121-141055 · `1c7ea5d`

- Slurm started three additional H100 production allocations at 14:08:49. They are RLOO 200 seed 44 (7226475), VinePPO 650 seed 42 (7226476), and GRPO 200 seed 42 (7226477). The two earlier RLOO seeds were still running, so five production GPU allocations were active concurrently. The other 650 arms wait on 7226476's first successful post-resume update.

### 20261121-143540 · `1c7ea5d`

- RLOO 200 seed 42 (7226473) wrote `COMPLETE` after 48:23 on H100. Its 16-round test was 46.46% Pass@1 on the validation-selected iteration 175. It sampled 2,605,118 training tokens in 25,600 primary continuations, with no MC tokens or allocation resumes. The full-run archive verifier 7227707 is running.
- The newly available capacity admitted PPO seed 44 as job 7227708. All three 200-iteration PPO run specifications now have Slurm receipts. GRPO seed 43 (7226478) started on H100. VinePPO 650 seed 42 has recorded its first ten updates. The actual run metadata confirms 64 × 8 primary continuations per iteration, KL coefficient 0.0001, exact-token-prefix deduplication and a degree-one schedule with 312 warmup optimizer steps out of 10,400.

### 20261121-144210 · `1c7ea5d`

- RLOO 200 seed 43 (7226474) completed in 45:34 on H100, including all 16 test rounds. It scored 46.77% Pass@1 on validation-selected iteration 200 and sampled 2,565,044 training tokens. Neither of the first two production RLOO trials resumed.
- Whole-run archives 7227707 and 7227734 finished in 4:50 and 5:28 for RLOO seeds 42 and 43. Each verified 212 files and then pruned only its checkpoint `.pt` files. The two production trials have `COMPLETE`, online W&B records, the held-out results and exact-path prune receipts in `control-07`.
- CPU analysis job 7227733 took 16s to verify the selection and test hashes for both runs, calculate question-clustered intervals, and extract per-iteration health and phase metrics. GRPO 200 seed 44 (7226479) has started alongside the other two GRPO seeds.

### 20261121-150024 · `1c7ea5d`

- RLOO 200 seed 44 (7226475) completed in 49:49 with 45.40% Pass@1 across all 16 test rounds. Validation selected iteration 175. It sampled 2,633,247 training tokens. All three RLOO 200 training allocations completed without resuming. The third archive is 7228207.
- GRPO 200 seed 42 (7226477, KL 0.003) completed in 51:05 with 47.37% Pass@1 across 16 test rounds. Validation selected iteration 150. It sampled 2,711,588 training tokens and did not resume. The watcher is handling its archive.
- The 650-iteration VinePPO run completed its first 25 training updates and still owns its 8-hour first allocation. Rollover gates remain correctly closed.

### 20261121-151302 · `8052fab`

- OLMo MATH SFT attempt 7226484 failed after 4:19 at the first generation request. The pinned tokenizer registers `<|im_start|>` and `<|im_end|>` as special added tokens, while Transformers' named `all_special_ids` list omits them. `8052fab` checks the chat markers in the added-token registry and keeps the model's configured stop pair `[100265, 100257]`. A regression test covers both recognition and rejection of a non-special marker.
- OLMo-specific CPU allocation 7228433 checks the exact tokenizer against the generation config on both tasks, then runs the full suite. Watcher 7228434 uses `control-08`, gates fresh suffix-04 OLMo jobs on that suite, and preserves all RhoMath job receipts and gates. The older failed OLMo root remains available for diagnosis. The stop-token fix required a separate OLMo pin after production had already started. RhoMath and RestEM retain `1c7ea5d` and their existing worktree. No model or training setting changed.
- RLOO seed 44 and GRPO seed 42 archives, 7228207 and 7228227, verified their complete run directories. All three RLOO 200 runs and the first GRPO 200 run now have their full completion evidence. Two VinePPO 200 trials have started alongside the VinePPO 650 rehearsal.

### 20261121-151557 · `8052fab`

- CPU 7228433 confirmed the pinned tokenizer's chat start ID 100264 and generation stop IDs 100265 and 100257 for both MATH and GSM8K, then passed all 1,603 tests in 1:41. Slurm wall clock was 1:54. Watcher 7228434 submitted OLMo MATH SFT retry 7228515 and retained the original model revision and evaluation protocol. All three VinePPO 200 jobs are running alongside the 650-iteration rehearsal.

### 20261121-153007 · `8052fab`

- The last two GRPO 200 runs at KL 0.003 completed all 16 test rounds without resuming. Seed 43 (7226478) scored 48.53% Pass@1 in 50:57, with 2,704,787 sampled tokens and winner iteration 200. Seed 44 (7226479) scored 46.89% in 53:32, with 2,686,032 tokens and winner iteration 200. The replacement watcher is archiving both.

### 20261121-154213 · `8052fab`

- All three corrected-KL GRPO 200 archives are now verified. PPO seed 42 (7226686) started training and opened [its online W&B run](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/9a3d956d3eaa479f90624a25c7653d32). Its iteration-2 metric records 128 primary continuations in 26.0s. The RestEM probe 7226483 is running and has an online W&B receipt.
- Slurm briefly reported `MaxGRESRunMinsPerUser` for the other PPO seeds and the OLMo retry. The cluster's `coc-ice` QoS limits each user's total remaining allocated GPU time to 1,920 minutes. The watcher has submitted ready jobs within scratch reservations, and Slurm controls their actual GPU allocations under that cap.

### 20261121-154946 · `2d27896`

- RestEM's full-budget probe (7226483) completed in 10:05 allocated. Its measured iteration used 2,560 primary samples and 271,342 sampled tokens, with 921 accepted episodes, and took 223.7s including checkpoint archival. The whole probe archive 7229003 verified in 1:20. The watcher submitted the matched 10-round trial 7229004. Its queued link was reduced to 3h, based on the measured round and a separate allowance for validation and test. The 25,600-sample budget and all optimization settings are unchanged.
- PPO seed 43 started. Slurm reported the 1,920 GPU-minute cap for queued PPO seed 44 (7227708). The first PPO seed averaged 37.9s through iteration 13. Set the still-queued seed-44 link to 3h, with its existing successor mechanism if another link is needed. Models, update counts, validation and checkpoint cadence are unchanged.
- `2d27896` gives the other 650 arms 8-hour links so four one-GPU links can fit the cluster's 32 GPU-hour concurrent allowance. The original VinePPO rehearsal was already set to 8h. Rebuilt watcher 7229086 in `control-09` with the carried receipts and only those link limits changed. The RhoMath training pin remains `1c7ea5d`, and the OLMo pin remains `8052fab`. The generator also produced eight valid optional long-seed specifications in a staging check. None entered the active plan.

### 20261121-160136 · `2d27896`

- All three PPO 200 allocations are running. Seed 42 completed its first validation boundary and verified its 12,414,435,295-byte iteration-25 critic in checkpoint history. Only that one critic blob was present locally at the check. Seed 43 has an online W&B run. Seed 44 started in a 3-hour allocation. Watcher 7229086 is healthy, and the matched RestEM and corrected OLMo baseline receipts remain queued in Slurm.

### 20261121-164242 · `2d27896`

- All three deduplicated VinePPO 200 runs completed 16 test rounds without resuming, each on the validation-selected iteration 200. Seed 42 (7226480) scored 51.11% Pass@1 in 1:33:43 with 54,884,695 sampled tokens. Seed 43 (7226481) scored 51.40% in 1:34:58 with 58,839,070 tokens. Seed 44 (7226482) scored 52.67% in 1:28:51 with 51,707,547 tokens. Whole-run archives 7230112 and 7230164 verified in 4:42 and 4:47. Seed 44 archive 7230236 is running.
- The matched RestEM trial 7229004 started on H100 at 16:18:12 and has finished three rounds. The corrected OLMo MATH SFT baseline 7228515 started on H200 at 16:38:18. The first 650 VinePPO allocation is still running and has completed 100 iterations.

### 20261121-170033 · `2d27896`

- Archive 7230236 verified the third VinePPO 200 trial in 4:41. All nine RLOO, GRPO and VinePPO 200 trials now have their tested results, online runs and hash-verified bucket copies.
- The corrected OLMo MATH SFT baseline passed the previous failure point, evaluated all 500 examples in each of its first seven rounds, and is logging to [W&B](https://forge.coreweave.com/wandb/tsawada/vineppo/runs/e4ccb98121eb4d9b86842ea569828660). Matched RestEM has completed seven of ten training rounds. The PPO seeds and first long VinePPO link are still progressing.

### 20261121-171721 · `2d27896`

- OLMo 3 MATH SFT retry 7228515 finished in 33:31 on H200 with 70.59% Pass@1 over 16 rounds and 8,000 judged responses from the pinned, unadapted SFT model. It has online W&B tracking. Archive 7230727 verified in 51s. The MATH GRPO preflight 7230728 started at 17:14:29.
- Matched RestEM seed 42 (7229004) completed in 58:03 on H100 with no allocation resume. Its 16-round test scored 40.37% Pass@1 on validation-selected round 1. It sampled exactly 25,600 primary continuations and 2,627,443 tokens across ten training rounds. Whole-run archive 7230787 is in progress.

### 20261121-172831 · `6c839a7`

- RestEM archive 7230787 verified in 5:19. Its completion marker, 16-round test, online run and bucket receipt are all present.
- OLMo MATH preflight 7230728 failed during initial validation in 9:32. The bounded MATH worker exited 1 while grading the first batch, before the evaluator had written any answers for that batch. The existing evaluation log retained the exception type and hid the worker's captured stderr. Successor 7230760 halted immediately at iteration 0 as designed.
- `6c839a7` preserves bounded stderr and the failing extracted inputs on a nonzero worker exit, with a CPU regression check. Scoring, worker limits, schedule and model settings are unchanged. CPU suite 7230951 gates the fresh `-05` OLMo smoke and downstream steps in `control-10`. Watcher 7230952 carries the completed `-04` SFT result and archive receipt and preserves all RhoMath jobs. The retry will expose the actual grader failure if it recurs.

### 20261121-173135 · `6c839a7`

- CPU 7230951 passed all 1,603 tests in 1:50 and used 1:53 of Slurm wall clock. Watcher 7230952 submitted OLMo MATH smoke retry 7230979. The superseded `8052fab` and `2d27896` worktrees were removed after checking that no allocation used them.

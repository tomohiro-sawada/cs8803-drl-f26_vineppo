# Plan: unattended multi-day training with crash-safe resume

Written 20261120-184322 at commit `ca1ced8` (working tree has uncommitted changes; see the companion guide).
Companion: `20261120-180926_baselines-long-runs-olmo-guide.md` holds the algorithm and model detail for the baselines, the long runs and OLMo. This plan adds the resume mechanism, its tests, the cluster dry runs, the launch gates and the reports. Where the two disagree, this plan wins.

Statements marked **(estimate)** or **(inference)** were not measured. Everything else was read from code, run outputs, or the scheduler probes in section 1.

`OUT=/home/hice1/tsawada7/scratch/cs8803-drl-f26_vineppo/training/pace_ice`

## Goal

1. Implement the remaining baselines: PPO, RLOO, RestEM.
2. Integrate OLMo-3-7B-Instruct.
3. Launch training for 1 and 2 and the longer RhoMath run, unattended, so that every run survives job time limits, crashes and node failures. Budget: 4–5 days expected, 9 days maximum.
4. Deliver two reports, one detailed and one dense, covering algorithmically significant design choices, results with evidence of training health, and pipeline bottlenecks.

**Nothing in this plan is implemented yet.** No multi-day run is launched until the gates in section 4 pass.

## 1. What the cluster does (measured 20261120, jobs 7205051–7205175)

Probe scripts and logs: `$OUT/slurm-lifecycle-probe-20261120/`.

| Question | Result |
|---|---|
| Longest job | 16 hours (`ice-gpu` MaxTime). No QOS available to us allows more. |
| What a job gets at its time limit | SIGTERM to the batch shell and to every step, about 17 s after the limit. The step then has about 30 s (`KillWait`) before SIGKILL. |
| Earlier warning | `--signal=B:USR1@60` delivered SIGUSR1 to the batch shell 90 s before the limit (Slurm rounds up to its 60 s tick). |
| Warning sent to steps (no `B:`) | Not delivered: the step saw only SIGCONT. Do not rely on it. |
| `sbatch` from inside a running job | Works. A successor submitted with `--dependency=afterany` started 44 s after its parent timed out. |
| Requeue | Works. `scontrol requeue` restarted the same job ID 2 min 19 s later with `SLURM_RESTART_COUNT=1`; `--open-mode=append` kept the first run's log. |
| Time left, from inside the job | `SLURM_JOB_END_TIME` is set (epoch seconds). |
| Preemption | Configured as requeue, but no QOS lists a QOS it can preempt, so jobs under `coc-ice` should not be preempted **(inference)**. Node failure still requeues a `--requeue` job. |
| Queue limits | 50 submitted jobs per user under `coc-ice`; 500 on the association. |
| Scheduling | Priority is fair-share only. The backfill window is 4 hours, so a 4-hour job can fill gaps that a 16-hour job cannot **(inference from `bf_window=14400`)**. |
| `cron` / `scrontab` | Not permitted. Nothing can watch the jobs from outside. |

**Unresolved:** the QOS shows a per-job cap of 960 GPU-minutes (16 GPU-hours). A 2-GPU, 16-hour request (1,920) was accepted at submission; probe 7205053 will show whether it is allowed to start. Until then, keep every job at or under 16 GPU-hours. One-GPU colocated jobs are exactly at the cap.

**Availability right now is the main schedule risk.** All 48 H100s were allocated when this was written. Jobs submitted at 17:59 waited 42 minutes for the first GPU, and the scheduler's estimate for the rest was about 22 hours. Earlier today waits were 5–110 minutes.

## 2. Resume design

### What exists

- **Trainer checkpoints** restore the actor, the reference, the optimizer (and the critic and its optimizer for PPO), the prompt-order generator, and the CPU and CUDA RNG states, after asserting config, dataset, model and optimizer identity. Writes are atomic (temp file, fsync, hard link).
- **Selection state** resumes through `--resume-selection`; a checkpoint whose validation was interrupted is validated before any new update.
- **Final test** restarts through `experiments/training/finalize.py`: an incomplete 16-round test is rerun from scratch and never merged with partial rounds.
- **A resume wrapper** exists for Skynet only (`experiments/training/skynet/train_resumable.slurm`): each start trains into a new `attempt-NN` under one run root from the highest completed checkpoint.

### What is missing

1. **No PACE wrapper.** Nothing resubmits a job that timed out or failed.
2. **No signal handling.** A job killed between checkpoints loses everything since the last one. At `--checkpoint-every 25` on the 512-episode VinePPO run that is up to 35 minutes **(estimate)**.
3. **Checkpoints are tied to validation.** With selection enabled, every checkpoint must be validated, so cheap frequent resume points are not possible today.
4. **Planned features have no checkpoint state yet:** an LR schedule, RestEM's per-iteration adapter reset, and any dedup bookkeeping.

### Proposal

**a. Self-resubmitting chain, not in-place requeue.** Add `experiments/training/pace_ice/train_resumable.slurm`, a port of the Skynet wrapper that:

- submits its own successor at start, held with `--dependency=afterany:$SLURM_JOB_ID`, running the identical command;
- on entry, exits successfully without resubmitting if the run root already holds a completion record (test complete, or selection frozen for validation-only runs);
- carries `--requeue --open-mode=append` so a node failure restarts the same link.

A chain is preferred over requeue-on-signal because the successor exists from the first second: it covers out-of-memory kills, Python exceptions and node failures, which a signal handler never sees.

**b. Stop a crash loop.** A run that fails deterministically would otherwise resubmit forever. Each link records the highest completed iteration in the run root; a link that finds no progress since the previous link cancels its successor and writes `HALTED` with the reason. One bad run stops itself and does not block the others.

**c. Checkpoint on a wall-clock deadline (recommended) and on the warning signal (backstop).**

- **Deadline:** at each iteration boundary the trainer compares the time left (`SLURM_JOB_END_TIME`) with the longest iteration seen so far plus the measured checkpoint time and a margin. If the next iteration might not fit, it writes a resume checkpoint and exits 0. This is deterministic and needs no signal.
- **Signal:** `--signal=B:USR1@600`. The batch shell forwards it to the learner step; the learner sets a flag and takes the same exit path at the next boundary. Ten minutes covers the longest iteration expected **(estimate: under 4 minutes)**.
- Mid-iteration state is never saved. An interrupted iteration is rerun from its start.

**d. Separate resume checkpoints from selection candidates.** Add `--resume-checkpoint-every N`: a `resume-latest.pt` slot overwritten every N iterations and on the deadline exit, never validated, never a candidate, pruned by the next one. Candidates stay on the `--checkpoint-every` schedule, so the selection contract is unchanged. Compact LoRA resume checkpoints are 152 MB and take about 0.3 s for RhoMath.

**e. Faithfulness: what "identical to a non-preempted run" can mean here.**

- **HF backend:** resume is bitwise exact, and CPU tests already assert it for every algorithm.
- **vLLM backend:** two uninterrupted runs with the same seed already differ in iteration 1, before any update, because the replica batches concurrent requests. So bitwise identity to an uninterrupted run cannot be tested on the production path, with or without a resume.
- **What will be proven instead:**
  1. Every piece of learner state round-trips bitwise (weights, optimizer moments, schedule position, RNG states).
  2. With a deterministic sampler, an interrupted-and-resumed run equals the uninterrupted run bitwise, through the real launcher and wrapper.
  3. On the real backend, the resumed trainer starts from state byte-identical to the checkpoint, and issues the same sampling requests (same prompts, same per-request seeds) as the uninterrupted run would.
- Per-request sampling seeds are derived from the run seed, the iteration and a call counter that resets each iteration; no sampler-side state needs to persist.

## 3. Test plan

Tests and GPU code run under `sbatch`, never on the login node.

### CPU tests

| # | Test | Asserts |
|---|---|---|
| T1 | State coverage | After a resume, every tensor in actor, reference, critic, both optimizers (including step counts and moments), LR-schedule state, the prompt generator and the CPU/CUDA RNG states equals the saved value bitwise. A deliberately added unsaved attribute makes the test fail. |
| T2 | Split anywhere | For each algorithm (VinePPO, GRPO, RLOO, PPO, RestEM): train N iterations straight through; separately, interrupt at k and resume, for every k from 1 to N−1. Metrics history and final state are bitwise equal. Extends `test_checkpoint_resume_is_bitwise_equal_to_uninterrupted` and its LoRA, grouped and PPO siblings. |
| T3 | Chained resume | Three interruptions in one run give the same result as none. |
| T4 | Trainer on the vLLM sampler | Split-anywhere through `VLLMInference` with the existing fake replica. No test does this today. Also asserts the request bodies and seeds after resume equal the uninterrupted run's. |
| T5 | Deadline and signal exit | With an injected clock or flag: exactly one resume checkpoint, exit code 0, no partial iteration in `metrics.jsonl`, no selection candidate added. |
| T6 | Resume slot versus candidates | Resume checkpoints never enter selection; candidate schedule and best-checkpoint protection are unchanged by any number of interruptions. |
| T7 | Corrupt or truncated checkpoint | Load fails loudly. The wrapper does not fall back to an older file. |
| T8 | Wrapper | With the existing fake-Slurm launcher harness: picks the highest completed iteration; exits without resubmitting when complete; halts after a no-progress link; refuses a second concurrent link on one run root. |
| T9 | Final-test restart | Existing finalize tests, run through the PACE wrapper. |

### GPU tests (smoke scale, RhoMath, about 10 minutes each)

| # | Test | Asserts |
|---|---|---|
| G1 | HF backend, interrupted versus straight | Final checkpoints bitwise equal. This repeats the Skynet check whose comparison output was never saved (only exit status 0 is recorded). |
| G2 | vLLM backend, state round trip | The resumed trainer's state equals the checkpoint bitwise before its first update, and the first post-resume requests match the uninterrupted run's. |
| G3 | vLLM backend, each algorithm | Interrupt and resume VinePPO, GRPO, RLOO, PPO and RestEM once each; all complete, selection intact, final test runs. |

## 4. Cluster dry runs (gates before any multi-day launch)

All use the real wrapper, the real launcher and a real GPU, at smoke scale.

| # | Dry run | Pass condition |
|---|---|---|
| D1 | Time limit | 10-minute limit; the run needs three links. Each link exits 0 on the deadline, the next resumes at the right iteration, the last one freezes selection and completes the test, and nothing is resubmitted afterwards. |
| D2 | Hard kill | `scancel --signal=KILL` on the learner mid-iteration. The successor resumes from the last resume checkpoint; no partial metrics. |
| D3 | Kill during checkpoint write | The successor uses the previous complete checkpoint; no temp file is ever picked up. |
| D4 | Kill during validation | That checkpoint is revalidated before any update; candidates are neither lost nor duplicated. |
| D5 | Kill during final test | A fresh 16-round attempt runs; partial rounds are not merged; the completion record is written once. |
| D6 | Node-failure path | `scontrol requeue` mid-run. Same job ID restarts with restart count 1, a fresh worker directory, and resumes. |
| D7 | Worker death | Kill the vLLM replica. The job fails, the successor resumes. |
| D8 | Crash loop | A run rigged to fail at start halts after one no-progress link and leaves `HALTED`. |
| D9 | Signal path | `--signal=B:USR1@600` reaches the learner and triggers the deadline exit. |
| D10 | Full-length rehearsal | One real run of at least two 16-hour links, before Tom leaves. This is the only test of a real time-limit crossing at production scale. |

Record each result with job IDs in this file's execution log.

## 5. Order of work and launch waves

Each wave launches only after its own smoke run and the gates above pass. Runs go out as soon as they are ready; nothing waits for a later wave.

| Wave | Content | Needs |
|---|---|---|
| 0 | Resume mechanism, T1–T9, G1–G3, D1–D9 on GRPO and VinePPO | Nothing |
| 1 | Prefix dedup; RLOO; D10 rehearsal doubling as the first long run | Wave 0; decisions 1, 3 |
| 2 | PPO | Wave 0; decision 2 |
| 3 | Long RhoMath runs (512 episodes per iteration) | Wave 1 |
| 4 | RestEM | Wave 0; decision 4 |
| 5 | OLMo-3-7B-Instruct | Go/no-go probe; decisions 2, 5 |

Start the OLMo go/no-go probe (does vLLM's generic backend load a LoRA adapter at runtime?) during wave 0. It needs only a download and one short job, and a "no" removes wave 5.

**Budget (all estimates).** At the paper's 512 episodes per iteration and 650 iterations: VinePPO about 15 hours per seed after dedup, GRPO and RLOO about 7, PPO about 8. One seed of each is under 2 GPU-days; three seeds about 5. RestEM at paper scale adds about 18 hours per seed. OLMo cannot be costed before its benchmark. Queue time is not included and is currently the larger number.

## 6. Unattended operation

- **Pinned code.** Every job re-imports the evaluator at each validation, so the launch tree must not change. Commit, then launch from a detached worktree on scratch. The editor could not create that directory earlier (outside the project); Tom runs once: `git worktree add --detach /home/hice1/tsawada7/scratch/cs8803-drl-f26_vineppo/worktrees/<commit> <commit>`. Development continues in the main checkout.
- **Per-run status file** in each run root, rewritten at every boundary: state, iteration, link number, last error. With no cron, this and W&B are the only views.
- **Notification.** Add `--mail-type=FAIL,REQUEUE` with Tom's address if he wants it. A mail program is configured; delivery is untested. Add a test mail to the dry runs.
- **Disk.** Scratch is at 231 of 300 GB. Resume slots are overwritten in place and old attempts' non-best candidates are pruned. PPO (about 13 GB per checkpoint) and OLMo (about 29 GB of frozen weights per run) do not fit without decision 2.
- **Queue.** At most 50 submitted jobs. Each run holds two (the running link and its successor), so at most 25 concurrent runs.
- **Failure policy.** A run that halts stays halted. No automatic change to batch size, precision or settings.

## 7. Reports

Written to `prose/agent-reports/`, named `<timestamp>_final-report-detailed.md` and `<timestamp>_final-report-brief.md`, each with the commit hash. The detailed report is written first; the brief is cut from it, about one page.

| Section | Content |
|---|---|
| Design choices | Every algorithmically significant decision, with what upstream does and why ours differs: exact sampled tokens; step segmentation and the MATH boundary rule; prefix dedup; LoRA in place of full fine-tuning; sampling temperature 1.0 with no top-p; KL coefficient and clamp; advantage normalization; constant LR versus the paper's schedule; PPO critic precision, λ and value clipping; RestEM deviations; the OLMo prompt and stop rule. |
| Results | Test Pass@1 per method, per seed, with the seed mean and spread; question-clustered intervals for single checkpoints; paired differences; the SFT baseline; cost in sampled tokens and GPU-hours; comparison with the paper's Figure 4. |
| Training health | Validation curves; reward; reference KL; clip fraction; gradient norm; truncation and response length; zero-advantage and zero-variance-group fractions; critic loss for PPO; MC value statistics for VinePPO; and the count and position of every resume, with a check that curves are continuous across them. |
| Bottlenecks | Per-phase time breakdown per method and model; what limits throughput now; queue time against run time; disk; and ranked suggestions. |

Every interrupted, halted or unrun arm is labelled as such.

## 8. Decisions needed before Tom leaves

1. **KL coefficient for GRPO and RLOO:** paper 3e-3, ours 1e-4.
2. **Scratch:** permission to delete about 90 GB of smoke and benchmark checkpoints (results are kept in `metrics.jsonl`), or another location for PPO and OLMo checkpoints.
3. **Long-run definition and seeds:** 650 iterations at 512 episodes, one seed or three?
4. **RestEM scale:** paper scale or matched to the RL budget.
5. **OLMo task and prompt:** MATH or GSM8K, and the instructed answer format.
6. **LR schedule:** keep constant LR (as in all runs so far) or add the paper's warmup and decay for the long runs. Adding it means new checkpoint state.
7. **Resume fidelity:** confirm the definition in section 2e is acceptable, given that bitwise identity to an uninterrupted vLLM run is not achievable.
8. **Priority if time runs short.** Proposed: resume mechanism, then RLOO and the long run, then PPO, then RestEM, then OLMo.
9. **Who operates while Tom is away.** The chain runs itself, but halted runs, launching later waves and writing the reports need an agent session with a terminal on the cluster. Without one, only what is launched before departure will run.

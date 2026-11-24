# Handoff: OLMo VinePPO, LoRA-critic PPO, and the OLMo baseline report

Written 20261124-132948 at commit `1e31719` (branch `tom/work`, pushed).
Supersedes the open items in `20261124-130156_olmo-baselines-and-bucket-handoff.md`; that file's recipes and paths still hold.

## Goal

Complete the OLMo-3-7B-Instruct-SFT baseline set (SFT, GRPO, RLOO, RestEM, VinePPO, PPO on GSM8K and MATH) and report it.

## Names

- `S=/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo`, `T=$S/training/pace_ice`, `CTL=$S/handoff-20261120/control-21`
- Bucket: `tsawada/cs8803-drl-f26_vineppo`
- Launch recipe, link job and upload job: see the 130156 handoff and `sacct -X -P -j 7282894 -o SubmitLine%8000`.

## Tom's decisions (do not reopen)

- **Bucket:** tier 4 (20 intermediate PPO critics, 248.3 GB) stays. No further deletion of any kind. Record: `prose/agent-reports/20261124-130743_bucket-thinning-record.md`.
- **OLMo VinePPO:** GSM8K first, then decide MATH from the measured MC throughput.
- **OLMo PPO:** run it with a smaller critic (LoRA adapters plus a value head on a frozen base). This departs from the paper and must be labelled so.
- **RhoMath LoRA-critic ablation:** run it, because RhoMath is the only model with a full FP32 critic as reference.
- **Second Skynet sync:** Tom's Skynet agent does it, without `--delete`, writing `SYNC_RECEIPT_2.json`.
- **Unchanged:** KL 3e-3 for GRPO and RLOO, 1e-4 for PPO and VinePPO; RestEM matched to the RL sample budget; dedup on.
- **Undecided:** how to fix W&B run fragmentation. Do not change `tracking.py` without asking.

## Segmentation changes (three, each a labelled departure from upstream)

1. **Special-token allowlist (bug).** `src/core/segmentation.py:83,88` use `tokenizer.all_special_ids`, which for OLMo is `[100257, 100277]` and omits `<|im_start|>` (100264) and `<|im_end|>` (100265). Build the set from the special entries of `added_tokens_decoder`. Add a test.
2. **GSM8K rounding.** `gsm8k_step_boundaries` calls `_token_step_boundaries(..., round_forward=True)`, as MATH does. Tom approved the plan that includes this. Update the docstring's `AGENT INSTRUCTIONS` and the pinned tests. Replay: 512 of 512 OLMo responses segment, 9.4 steps on average.
3. **MATH nested placeholders.** 7 of 512 OLMo MATH responses fail `assert text == reasoning_latex` at `src/core/vendor/vineppo/math_extract_steps_inplace.py:207`.
   - Cause: `_recover` restores placeholders in insertion order, so a math span nested inside a later-matched span is never restored. Triggers: `\( … \)` inside `\text{(…)}` inside `\[ … \]`, and currency (`$10,000 … $920.25`) read as inline math that swallows `\( … \)` spans.
   - The responses are ordinary: all ended with `stop`, 329–689 tokens, 5 of 7 correct, 4 of 7 from one currency prompt (`math:train:10647`).
   - Fix: restore in reverse insertion order. What-if run (jobs 7289925, 7289968): all 7 split into 22–47 steps, and all 505 accepted responses keep identical boundaries.
   - Residual failures: treat the response as one step (advantage `reward − V(prompt)`), counted in a logged metric. Do not drop: dropping removes mostly correct responses and biases against currency and nested-math prompts. Upstream drops.
   - **Needs Tom's yes before it is committed**, because the vendored file's `AGENT INSTRUCTIONS` forbid silent repairs and `tests/core/test_segmentation.py` pins upstream outcomes with a corpus fingerprint. Only MATH VinePPO depends on it.
   - Not checked: whether RhoMath's MATH failures share the cause.
   - Evidence: `$CTL/math-splitter-{failures,cause,whatif}.py`; outputs in `artifacts/training/pace_ice/slurm/vp31-math-splitter-*.out`. Move the scripts into the repo if they are cited in the report.

## LoRA critic

- **Where:** `TokenCritic` in `src/algorithms/ppo.py` wraps a caller-supplied backbone; the critic is built in `experiments/training/run.py` near line 811.
- **Design:** frozen base, LoRA adapters, trainable value head. One new option for the critic rank; no rank means the existing full critic.
- **RhoMath sweep:** ranks 16, 64, 256 at 200 iterations, seed 42, against the existing full-critic PPO 200 runs (seeds 42–44). Add seeds 43 and 44 for the best rank.
- **Reference (RhoMath PPO 200 seed 42):** 42 s per iteration, critic update 19 s, actor update 6 s, checkpoints 2,065 s in total, 12.4 GB per critic checkpoint.
- **Report:** validation and test Pass@1, critic loss, explained variance if logged, time per iteration, checkpoint size, per rank.
- **Then:** OLMo PPO on GSM8K with the chosen rank; MATH if time allows.

## State at handoff

- **MATH RestEM** (`handoff-olmo3-math-restem-match-s42-20`, job 7284850): running, round 8 of 10 at 13:27. Successor 7285818, link job 7284851, upload job 7284852. If it rolls to the successor, the link job fails for lack of `COMPLETE`; resubmit link and upload from `sacct -X -P -j 7284851,7284852 -o SubmitLine%8000`, run from `$S/worktrees/4c924f8`.
- **Other seven OLMo runs:** complete, bucket copies verified. Seven-run tables: `artifacts/training/pace_ice/slurm/vp29-olmo-tables-partial-7288969.out`.
- **Tests:** 1,629 pass at `1e31719` (job 7289648).
- **Scratch:** about 191 of 300 GiB. Each OLMo run writes a 29 GB frozen base until linked. Worktree `0389489` is free to remove; `4c924f8` is in use.

## OLMo VinePPO cost (estimates from replay job 7283244)

| | GSM8K with rounding | MATH |
|---|---|---|
| MC tokens per primary token | about 70 | about 120 |
| Total MC tokens | about 70M | about 335M |
| Sampling time | about 5 h | about 18 h, plus slow grading |

Measure the real rate in the GSM8K run before committing to MATH.

## Order of work

1. Watch 7284850 to `COMPLETE`, confirm the upload job's `verified` line, re-grade against `$T/handoff-olmo3-math-sft-04/test`, regenerate tables with `vp-olmo-baseline-tables.py $T`.
2. Write `prose/agent-reports/<timestamp>_olmo-baselines-and-bucket.md`.
3. Segmentation changes 1 and 2 with tests; smoke-test; launch OLMo VinePPO on GSM8K.
4. LoRA critic with tests; RhoMath sweep; OLMo PPO on GSM8K.
5. MATH VinePPO, if Tom approved change 3 and the measured rate fits.
6. Final report covering all arms; append to the execution log in `20261120-212505_remote-agent-handoff.md`.

## Rules and pitfalls

- **Git:** never stage `bash.sh`, `pyproject.toml`, `uv.lock`, `prose/reflections/*`, `wandb-metric-keys.txt`, `wandb-panels.tsv` or Slurm logs. Never print `.env` secrets.
- **Code (`AGENTS.md`):** asserts, no fallbacks, minimal code, no duplication. Obey `AGENT INSTRUCTIONS` docstrings.
- **Where things run:** tests, GPU code and bucket jobs under sbatch (`--partition=ice-cpu --account=math --qos=coc-ice` for CPU). One GPU and at most 16 hours per job; OLMo needs `--constraint=gpu-h200`.
- **Launch trees:** submit from a clean detached worktree at the new commit, with `.env` symlinked. Never edit a tree that running jobs launched from.
- **Scratch:** do not delete checkpoints or run directories; do not run `archive_run.slurm` or `prune_verified.py`. 90 checkpoint files (39.5 GB) exist only in the bucket.
- **Compute nodes cannot see the login node's `/tmp`.** Put shared files on scratch.
- **GSM8K grader:** strict Pass@1 is capped at 98.94% (14 references contain commas). Report strict and lenient for OLMo GSM8K.
- **W&B:** each allocation and each evaluation is a separate run.
- **Reports:** numbers only from result files; mark estimates; link the 20261122 reports, do not edit them.
- **`hw5_evaluate` in the queue is not ours.**

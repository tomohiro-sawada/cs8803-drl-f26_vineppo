# Handoff: from results to the December 7 submission

Written 20261124-190704 at commit `cc4308f` (branch `tom/work`).
It starts where [the OLMo and LoRA-critic handoff](20261124-132948_olmo-vineppo-lora-critic-handoff.md) ends. That file's recipes, paths and rules still hold.

## Goal

Turn the results into the course submission due Monday, December 7: a report that reproduces the paper's main result, describes one modification, and analyses it.

## What is due

From [the course page](../../../knowledge-bases/course-instructions/cs8803-drl-f26__project_html.md#final-project-due-december-7):

| Deliverable | State |
|---|---|
| Code: repository link, README with setup and run instructions, scripts to reproduce; a notebook is recommended | Code exists. Root `README.md` has no reproduction section. `main.ipynb` is a stub with empty sections |
| Report: PDF, about 2–4 pages, with introduction, modification, setup, results and discussion, video link | Not started |
| Final video, 12–15 minutes, updated with implementation, modification and findings | Tom's |
| Team contribution survey | Tom's |

The grade does not depend on the modification beating the baseline. It does require an analysis of why it worked or failed.

## State at handoff

- **Results in hand:** [results summary](../../../agent-reports/20261124-174645_results-summary.md), with four figures and their data in `artifacts/training/pace_ice/results-summary-20261124/`.
- **Running, owned by the Claude Code session that resumed the earlier `/goal` at 16:30:**
  - OLMo GSM8K VinePPO, job 7292545: iteration 29 of 50 at 18:55, about 278 s per iteration, validation 92.36% and 93.90% at iterations 10 and 20.
  - RhoMath PPO with a LoRA critic, seed 42: rank 16 is complete (48.85% test at iteration 175, in 1:17:28, against 45.57% in 2:28:34 for the full critic at the same seed); ranks 64 and 256 are running (jobs 7293229, 7293231); selector job 7293376 waits on them.
  - Then, under that `/goal`: seeds 43 and 44 for the chosen rank, and OLMo GSM8K PPO at that rank.
- **Do not duplicate that session's work.** Read the tail of [the execution log](20261120-212505_remote-agent-handoff.md#execution-log) first. If its last entry is more than a few hours old and jobs are idle, take over from the log.

## What the evidence supports

Use this when drafting the report. Numbers are test Pass@1 unless marked.

| Claim in the paper | Our result | Verdict |
|---|---|---|
| VinePPO is the best method (RhoMath, GSM8K) | Best at both budgets: 56.90% at 650 × 512 (one seed), 51.73% at 200 × 128 (three seeds) | Replicates |
| VinePPO beats PPO by 3.3 points | +3.28 [+1.54, +5.03] at 650 × 512 | Replicates, one seed |
| VinePPO beats GRPO and RLOO by 8.8 and 8.9 | +3.27 and +5.42. Our GRPO and RLOO score 9.0 and 7.0 above the paper's | Same sign, much smaller |
| PPO beats GRPO by 5.5 | −0.01 [−1.57, +1.54] | Does not replicate |
| RestEM gains 2.5 over SFT | +0.22 [−0.53, +0.97] at our matched 25,600-sample budget | Not comparable; different recipe |
| VinePPO passes PPO's peak in 3.0× less time and 9× fewer steps (RhoMath; on MATH, by the axis range of Figure 6) | Validation, one seed: 2.4× less time (10.0 h against 23.8 h), 1.35× fewer iterations (425 against 575) | Time replicates for a different reason: our PPO is the slow arm |
| The gap widens on MATH | RhoMath was not run on MATH | Untested |
| Holds for a 7B model | OLMo 3 has SFT, GRPO, RLOO and RestEM only | Untested until VinePPO and PPO finish |

- **Why the time result differs in kind:** in the paper a VinePPO step is 4.75× slower than a PPO step (380 s against 80 s on MATH, Table 5). In ours VinePPO takes 15.1 h and PPO 26.8 h for the same 650 iterations, because the full FP32 critic update is 66% of PPO's time.
- **Why our GRPO and RLOO are stronger than the paper's is not established.** Candidates: training temperature 1.0 with no top-p (upstream configs use 0.6 and 0.9), LoRA with LR 1e-5, the truncation penalty, and our own loss definitions (the paper's code has neither method). The trainer rejects any temperature but 1.0, so that one cannot be ablated cheaply.
- **Support for LoRA from outside:** Schulman et al., [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (2025), report that LoRA matches full fine-tuning for a GRPO-like policy gradient on GSM8K and MATH, with an optimal learning rate about 10× higher. Ours is 1e-5 against the paper's 1e-6. We have not tested the claim ourselves; item 6 below would.

## Decisions Tom owes

In blocking order. My recommendation follows each.

1. **What is "the modification"?** The report needs one, compared against the original method.

   | Option | In hand | Missing | Cost |
   |---|---|---|---|
   | A. Parameter-efficient VinePPO: LoRA policy for every method, LoRA critic for PPO | All RhoMath runs; rank sweep running | A full fine-tuning control (item 6); critic-accuracy measurement (item 5) | About 10–15 GPU-hours (estimate) |
   | B. New model: OLMo 3 7B | Four of six arms | VinePPO and PPO on both tasks | GSM8K is running; MATH about 15 GPU-hours per arm (estimate) |
   | C. Adaptive MC budget for VinePPO: fewer rollouts where the outcome is already decided | Nothing | Design, code, runs | Item 5 for the evidence, then 3 seeds × about 1.5 h |
   | D. Prefix caching (Tom's earlier idea) | vLLM 0.34's prefix cache is on by default, and each prefix is sent once with `n = 9` | A measurement with caching off | One replay job |

   - **Recommendation: A as the modification, B as the extension, C only if Tom wants the inference exercise.** A reframes the largest deviation (LoRA) as the thing under test, and most of its evidence exists.
   - **D probably has little headroom (inference, not measured).** The stock server already caches prefixes. I have not measured how much prefill the nine siblings share.
   - **Evidence for C:** 48% of VinePPO response tokens have exactly zero raw advantage early in the 650 run and 67% at the end. Those tokens sit in steps whose start and end states had the same estimated value.

2. **Approve the MATH splitter fix** (reverse-order placeholder restore; a response that still fails becomes one step). It has been pending since this morning and blocks OLMo MATH VinePPO. Evidence: `experiments/training/pace_ice/handoff_20261120/vp-math-splitter-{failures,cause,whatif}.py`. Recommendation: approve.

3. **More seeds at the paper's budget?** The headline numbers have one seed. GPU wall per seed: RLOO 7:37, GRPO 10:34, VinePPO 15:37, PPO 27:20. Recommendation: seeds 43 and 44 for VinePPO and GRPO (about 52 GPU-hours), and PPO only with a LoRA critic if the rank sweep supports it.

4. **RLOO: footnote or fix?** It clips 29–49% of sequences. A 200 × 128 rerun is about 50 minutes per seed, but needs a tuning choice first. Recommendation: footnote.

5. **RhoMath on MATH?** The paper's budget is 1,000 × 512; the cost here is unmeasured. Recommendation: skip unless items 1–3 finish early.

6. **W&B fragmentation.** Still undecided. The figures are drawn from run directories, so nothing waits on it. Recommendation: leave it.

7. **Merge `tom/work` into `main`.** `main` is about 60 commits behind, the repository is public, and the submission is a link. Recommendation: one pull request once the report is drafted.

## Work that needs no decision

1. **Close out the running jobs** when the other session ends: test, link the frozen base, upload, verify.
2. **Refresh the summary.** Add the new arms to `RUNS` in `scripts/figures/collect_results_summary.py`, rerun both scripts, and write a new dated report that links the old one. Do not edit the old one.
3. **Generalization-slope figure** (the paper's Figure 5): validation Pass@1 against training accuracy (`stopped_correctness_mean`) per checkpoint, for the four 650 runs. It needs no GPU.
4. **Train, validation and test check.** GSM8K validation runs far above test: 53.6% against 40.15% for RhoMath SFT, 68.57% against 61.65% for OLMo SFT. OLMo MATH does not show it (67.15% validation, 70.59% test). The cause is not established. The 373 validation questions were carved from the original GSM8K training split. Check: evaluate SFT on 373 random training questions with the validation protocol, to see whether validation behaves like training data or like test data. Report the result either way.
5. **Value-estimate replay** (new diagnostic, about 30M tokens and 40 minutes on one H100, estimate).
   - **What:** from one PPO actor checkpoint and its critic, sample 64 prompts × 8 responses, segment them, and draw 256 continuations from every distinct prefix. Store per prefix: depth, remaining tokens, and each continuation's reward and length. Record the critic's value at the same states.
   - **Why one dataset:** it gives the critic's error against ground truth for the full critic and each LoRA rank (the paper's Section 7), the error of K = 1, 3, 9 by subsampling, and the tokens any allocation rule would spend (option C).
   - **Hooks that exist:** `record_prefix_samples` in `src/algorithms/vineppo.py` and `record_value_estimate` in `src/runtime/train.py`. `scripts/diagnostics/benchmark_sampler_replay.py` already records prefixes.
   - **Checkpoints:** the PPO 200 critics are in the bucket (12.4 GB each); the LoRA-critic checkpoints are on scratch. The critic must be paired with the actor of the same iteration.
   - **Motivation:** from logged aggregates I get a rough explained variance of 0.5 for the PPO 650 critic at the end of training and 0.2–0.3 for the 200-iteration critics. The paper reports 0.7–0.9. My figure is a back-of-envelope estimate; do not report it.
6. **Full fine-tuning control for GRPO** (200 × 128, seeds 42–44). This tests the claim that LoRA is enough.
   - Full fine-tuning needs `--backend hf --finetuning full --dtype float32`; the vLLM backend requires a LoRA actor. The HF GRPO benchmark took 52.7 s per iteration, so expect 3–6 hours per seed (estimate).
   - **Check first:** whether the 16-round test can run through the vLLM evaluator on a full-weights checkpoint. HF and vLLM evaluations are different protocols and their numbers are not comparable.
   - VinePPO cannot have this control: at HF speed it is 2,034 s per iteration.
7. **Documentation.** Root `README.md`, "Step segmentation", still says GSM8K uses exact newline cuts; `8f2c355` changed that. Add a short reproduction section pointing to `experiments/training/README.md` and `scripts/figures/`.
8. **Report draft.** Sections as the course page lists them. Take every number from `results.json` or a report table.

## Gated work

| Work | Gate |
|---|---|
| OLMo MATH VinePPO | Decision 2, and the measured GSM8K rate |
| Seeds 43 and 44 at 650 × 512 | Decision 3 |
| Adaptive MC budget | Decision 1, and item 5's data |
| RhoMath on MATH | Decision 5 |
| Pull request to `main` | Decision 7 |

## Compute and storage

- **Scratch:** 213.5 of 300 GB by `pace-quota`. The uv cache is 32 GB and sits on scratch (`~/.cache` links there).
- **Bucket:** 417.8 GB at 14:56 by listing. About 438 GB now by my sum of the uploads logged since; not re-listed.
- **Per-run footprint after archiving:** the 650 runs hold 0.1–0.3 GB each, VinePPO 650 holds 5 GB. A full-critic PPO run needs up to two 12.4 GB critic files while it runs.
- **Limits:** one GPU and at most 16 hours per job; long runs chain through `train_resumable.slurm`.

## Rules and pitfalls

- **All rules in the 132948 handoff apply:** no bucket deletion, no scratch deletion, heavy work under sbatch, clean detached worktrees for launches, the git exclusions.
- **Two sessions share this checkout.** Stage only your own files, by name.
- **`prose/reflections/` is Tom's.** Never write or stage it. His messages there contain W&B report links with access tokens.
- **Reports are not revised in place.** A follow-up is a new dated file. Dated one-line corrections have been tolerated.
- **Vendored code and `AGENT INSTRUCTIONS` docstrings** need Tom's approval to change. That covers the MATH splitter and the MC-prefix policy in `estimate_vineppo_batch`.
- **matplotlib is not a project dependency.** Plot with `uv run --no-project --with matplotlib`; do not edit `pyproject.toml`.
- **A seed is a replicate, not a control.** vLLM sampling differs between identical runs.

## Not checked

- Whether full fine-tuning still runs end to end; it has not been exercised since the LoRA switch, as far as I know.
- Whether RhoMath's 7 MATH splitter failures share OLMo's cause.
- Prefix caching's actual contribution to MC time.
- The current bucket listing.

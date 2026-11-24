# Implementation guide: prefix dedup, more baselines, longer runs, OLMo-3-7B-Instruct

Written 20261120-180926 at commit `ca1ced8`; the working tree has uncommitted changes (see "State").
Continues `20261119-225338_vineppo-grpo-pace-ice-handoff-2.md`. Its "How to submit" section still applies and is not repeated here.

Statements marked **(estimate)** or **(inference)** were not measured or verified. Everything else was read from the code, from run outputs, or from the pinned upstream source (McGill-NLP/VinePPO at `9f716e6`).

`OUT=/home/hice1/tsawada7/scratch/cs8803-drl-f26_vineppo/training/pace_ice`

## State

### Three-seed result (jobs 7199697–7199702, W&B group `seeds-20261120`)

RhoMath 1.1B, GSM8K, 200 iterations, 128 primary episodes per iteration, LoRA rank 16, LR 1e-5, four validation rounds every 25 iterations, 16-round test on the validation-selected checkpoint. SFT scores 40.4% on the same test split (standalone script).

| Seed | GRPO test | GRPO selected iteration | VinePPO test | VinePPO selected iteration | VinePPO − GRPO |
|---|---:|---:|---:|---:|---:|
| 42 | 48.26% | 200 | 50.47% | 150 | +2.21 |
| 43 | 45.60% | 100 | 49.88% | 125 | +4.27 |
| 44 | 47.50% | 200 | 48.48% | 125 | +0.99 |
| Mean | 47.12% | | 49.61% | | +2.49 |

- **Uncertainty:** seed standard deviation is 1.4 points (GRPO) and 1.0 (VinePPO). A t-interval on the mean gap over three seeds is −1.6 to +6.6. Every VinePPO seed beat every GRPO seed; the closest pair is 0.2 points apart.
- **Cost:** VinePPO sampled 75.6–80.3M tokens per run against GRPO's 2.67–2.81M, and took 1:43–1:45 against 0:48–0:59.
- **Code:** these ran on the tree later committed as `2f78567` (`src/` and `experiments/` are byte-identical to the launch snapshot). Their `metadata.json` records `418fec2`, `dirty: true`.
- **Outputs:** `$OUT/long-{grpo,vineppo}-rhomath-gsm8k-seed42-vllm-h100-200iter-02` and `...-seed{43,44}-vllm-h100-200iter-01`.
- **The earlier seed-42 pair (`...-200iter-01`) is a separate replicate.** A fixed seed does not reproduce a run: the two seed-42 runs already differ in iteration 1, before any update.

### Queued when this was written (all pending on priority)

| Job | Run | Output |
|---|---|---|
| 7204554 | SFT on the test split through the pipeline evaluator, 16 rounds | `$OUT/sft-baseline-test-rhomath-gsm8k-vllm-h100-01` |
| 7204551–7204553 | GRPO, 500 iterations, seeds 42–44 (wall-clock match for VinePPO 200) | `$OUT/long-grpo-rhomath-gsm8k-seed{42,43,44}-vllm-h100-500iter-01` |

- The 500-iteration GRPO runs have 20 selection candidates against VinePPO's 8, a small selection advantage for GRPO. Report it with the result.
- These jobs run the tree fingerprinted `af7aee309dda`, saved at `$OUT/walltime-20261120-code-ca1ced8-dirty-af7aee309dda.tgz`. Do not edit `src/` or `experiments/` in the main checkout until they finish (see "Rules for every launch").

### Uncommitted

- `experiments/training/evaluate.py`, `experiments/training/EVALUATION.md`, `tests/test_checkpoint_evaluation.py`: `--split test --baseline` is now accepted, so the unadapted model can be scored on test without a selection. Full CPU suite passed (1,562 tests, job 7204327).
- `bash.sh`, `pyproject.toml`, `uv.lock`: Tom's W&B export helper and its dependency.

## Decisions needed from Tom

1. **KL coefficient for GRPO and RLOO.** The paper uses 3e-3 for both (Table 2) and 1e-4 for PPO and VinePPO. Every run of ours used 1e-4; it is not a CLI option. Rerunning GRPO at 3e-3 costs three 50-minute jobs.
2. **Scratch space.** Scratch is at 231 of 300 GB. PPO and OLMo checkpoints will not fit (sections 3 and 6). About 90 GB is smoke and benchmark checkpoints whose results are already in `metrics.jsonl`; nothing has been deleted.
3. **RestEM scale.** The paper recipe is about 18 H100-hours per seed **(estimate)**. Paper scale, or a budget matched to our RL runs?
4. **What "longer" means.** More iterations at 128 episodes, or the paper's 512 episodes per iteration? Section 5 argues for 512.
5. **OLMo task and prompt.** GSM8K may have little headroom for this model (section 6). MATH or GSM8K, and what answer format do we instruct?

## Order of work

1. Prefix dedup (changes every later VinePPO run, so it goes first).
2. RLOO (launch only).
3. PPO.
4. Longer RhoMath runs.
5. RestEM.
6. OLMo-3-7B-Instruct (largest; start its go/no-go probe early, in parallel).

## 1. Deduplicate identical MC prefixes

**Why.** Upstream samples each distinct prefix once (`_create_value_estimation_requests` deduplicates on query text). Ours samples the bare prompt separately for each of the eight responses to it. Bare-prompt requests are 26% of MC requests and about 28% of MC tokens **(estimate)**.

**Change.** In `estimate_vineppo_batch` (`src/algorithms/vineppo.py`), sample each distinct `(prefix token IDs, budget)` once and fan the result out. Untested sketch:

```python
unique = {}
for key in zip(map(tuple, prefixes), budgets, strict=True):
    unique.setdefault(key, len(unique))
sampled = inference.generate_each(
    [list(p) for p, _ in unique], [b for _, b in unique],
    num_samples=num_samples, temperature=temperature,
)
groups = [sampled[unique[key]] for key in zip(map(tuple, prefixes), budgets, strict=True)]
```

- Match on exact token IDs, not text. This is stricter than upstream and consistent with the exact-sampled-token policy.
- Keep scoring per episode: each episode's scorer still grades the shared continuations against its own reference.
- Make it the only behaviour, with no flag. The three-seed runs above are the no-dedup arm.
- Record the policy in `metadata.json` so old and new runs are distinguishable.

**Acceptance** (`tests/algorithms/test_vineppo.py`): eight episodes on one prompt produce one bare-prompt request; their first prefix value is identical; distinct prefixes are unaffected; result order is unchanged; `record_prefix_samples` still fires once per episode and prefix.

**Then** rerun VinePPO, 200 iterations, seeds 42–44, same settings as the table above. Expect about 1.3 hours each **(estimate)**.

## 2. RLOO

- **Exists:** `src/algorithms/rloo.py`, `--algorithm rloo`, and the vLLM backend accepts it.
- **GPU evidence:** only HF-backend resume checks on Skynet. It has never run on PACE or through vLLM.
- **Do:** a four-iteration smoke (copy job 7199658's submit line, `--algorithm rloo`, drop `--mc-num-samples`), then seeds 42–44 at the GRPO settings. About 50 minutes each.
- **Blocked on decision 1** for the KL coefficient. If 3e-3 is chosen, add `--kl-coef` to `experiments/training/run.py` and the `launch.sh` allowlist, and rerun GRPO with it.
- **Paper target:** 44.5% (GRPO 44.6%).

## 3. PPO

**Exists:** `src/algorithms/ppo.py` (`TokenCritic`, `masked_gae`, `update_critic`), trainer support, CPU tests in `tests/runtime/test_ppo_train.py`. No PPO run exists on any GPU.

**Gaps against the paper:**

| Setting | Paper | Ours | Action |
|---|---|---|---|
| GAE λ | 1.0 | 0.95, not a CLI option | Set 1.0 |
| Value clipping | 0.2 | None | Add, or record as a deviation |
| Critic LR | 1e-6, same as actor **(inference)** | `--critic-learning-rate` 1e-5 | Tom's call; LoRA actor already uses 1e-5 |

**Blockers in the current code:**

- **Precision.** `run.py` rejects PPO with `--dtype bfloat16` because the critic is fully trainable. Running everything in FP32 would give PPO a different actor precision from the other arms. Load the critic backbone in FP32 independently of `--dtype`, keep the actor BF16 with FP32 LoRA, and narrow the guard to full fine-tuning.
- **Sampling backend.** `run.py` rejects PPO with `--backend vllm`. The reason is not recorded. The critic lives on the learner and PPO only needs primary samples, so there is no structural obstacle **(inference)**. Allow it; on HF, primary sampling alone is about 39 s per iteration.
- **Checkpoint size.** The critic and its AdamW moments are saved in every checkpoint, compact or not: about 13 GB each **(estimate)**. With best plus two latest plus one in flight, that is about 50 GB per run. Use `--keep-checkpoints 1`, run seeds one at a time, and resolve decision 2 first.

**Do:** fix the three blockers, add tests for the dtype split, run a four-iteration GPU smoke including resume, then seeds 42–44. About 1.2 hours each on vLLM **(estimate)**.

**Paper target:** 50.1%.

## 4. Longer RhoMath runs

**What the 200-iteration runs show.** VinePPO validation peaked at iteration 125–175 in all four runs and was flat or lower by 200. Its reference KL is near the per-token clamp. LR is constant; the code has no scheduler. More iterations at 128 episodes will mostly add drift **(inference)**.

**Paper recipe:** 650 iterations × 512 episodes, minibatch 64, 16 optimizer steps per iteration, polynomial LR decay with 3% warmup. Ours so far: 200 × 128, minibatch 32, 8 steps per iteration, which is 8% of the paper's episodes.

**Stage A (no new code beyond section 1):** 200 iterations at `--prompts-per-iteration 64 --num-samples 8 --minibatch-size 64`, both arms, seed 42.

| Arm | Per iteration **(estimate)** | 200 iterations |
|---|---:|---:|
| GRPO | 40 s | 2.2 h |
| VinePPO, after dedup | 85 s | 4.7 h |

**Stage B (only if stage A is still improving at 200):** 650 iterations. VinePPO needs about 15 hours **(estimate)**, against a 16-hour job limit, so plan a resume chain.

- PACE has no resumable wrapper; only Skynet has `train_resumable.slurm`. Resume by hand: new output directory, `--resume` and `--resume-selection` together.
- Work after the last checkpoint is lost at the time limit. Keep `--checkpoint-every 25` or lower.
- VinePPO resume has run on a GPU only at smoke scale (MATH, jobs 7198213 and 7198214).
- An LR schedule is a new feature: the scheduler state must go into the checkpoint and the resume identity check. Ask before adding it.

## 5. RestEM

Not implemented. Upstream recipe for RhoMath on GSM8K (`configs/polIter_rho1bSft2_restem_GSM8K.jsonnet`, `trainers/restem_trainer.py`, `episode_generators/math_restem_episode_generator.py`):

| Step | Upstream |
|---|---|
| Iterations | 10 |
| Sampling per iteration | Every training question, 8 responses, temperature 0.6, top-p 0.9 |
| Filter | Keep correct responses, at most 10 per question |
| Model at the start of each iteration | Re-initialised from the SFT model, with a fresh optimizer |
| Update | Token-mean negative log-likelihood, 8 epochs, batch 64, LR 1e-6, 3% warmup, no KL term in the loss |
| Within-iteration selection | Checkpoint every 500 steps; walk checkpoints in order and stop at the first validation drop |
| Next iteration's sampler | The checkpoint chosen above |

**Scale.** 7,100 questions × 8 = 56,800 samples per iteration and about 3,500 optimizer steps **(estimate)**; ten iterations is about 20 times the optimizer steps of our 200-iteration RL runs.

**Suggested design.** Add `RestEMConfig` as a fifth algorithm in the existing trainer so that checkpointing, selection and the evaluator are reused:

- Per iteration: sample, keep correct, reset the LoRA adapter and optimizer to their initial state, run `--update-epochs 8` of NLL, checkpoint, validate.
- New code: `src/algorithms/restem.py` (filter and loss), a loss branch in `update_actor` (`src/algorithms/actor.py`), the adapter reset, and `restem` in the algorithm-identity check at `experiments/training/evaluate.py:548`.
- I have not traced every place `PPOTrainer.train` assumes PPO-style batches (old and reference log-probabilities, advantages). Expect to touch `_cache_scores`.
- Check how prompts are drawn per iteration; RestEM needs every question exactly once.

**Deviations to record:** LoRA instead of full fine-tuning; temperature 1.0 with no top-p, as in our other arms (the trainer rejects any other temperature); each iteration's final model is the candidate and the next sampler, with no within-iteration early stopping.

**Do:** implement, run one iteration to measure cost, then settle decision 3 before launching seeds.

**Paper target:** 42.8%.

## 6. OLMo-3-7B-Instruct

**Verified facts** (`allenai/Olmo-3-7B-Instruct`, revision `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`):

- Not gated, Apache-2.0, 7.30B parameters in BF16 (about 14.6 GB), `Olmo3ForCausalLM`, 32 layers, vocabulary 100,278.
- Tokenizer: BOS and EOS are both `<|endoftext|>` (100257); pad is `<|pad|>` (100277). The chat template is ChatML-style with a default system prompt. Generation stops on `<|im_end|>` (100265) or `<|endoftext|>`.
- Both local environments ship `transformers/models/olmo3`.
- vLLM 0.34.0 maps `Olmo3ForCausalLM` to its generic `TransformersForCausalLM` backend, not a native implementation.
- It is an RL-post-trained instruct model, not a task SFT model. Its card reports 87.3 on MATH and gives no GSM8K figure.

**Step 0, go/no-go probe (do first).** The pipeline depends on vLLM loading a LoRA adapter at runtime. On the generic backend that is unverified, as is throughput. Download the pinned revision on the login node (extend `scripts/setup/download_model_weights.py`; compute nodes are offline), then run `scripts/diagnostics/check_remote_sampler.py` against it through `TRAINING_ENTRY`. If adapters do not load, stop and report; the alternatives (republishing merged weights each iteration, or HF sampling) are too slow for VinePPO **(inference)**.

**Step 1, measure before designing.** With a chat-template prompt: 16-round test accuracy on GSM8K and MATH, response-length quantiles, and the truncation rate at the current 1,024-token cap. If GSM8K is above roughly 90%, use MATH (decision 5).

**RhoMath-specific code that must become model-aware:**

| Location | Assumption |
|---|---|
| `experiments/training/run.py:76` and `scripts/paper_artifact_replications/evaluate_sft.py:30` | The `[MATH_TASK] Problem:` prompt, defined twice |
| `experiments/training/evaluate.py:427` | Asserts the training prompt equals that constant |
| `experiments/training/evaluate.py:55-72` | Stop string `\n\n\nProblem:`, 2,047-token context |
| `experiments/training/evaluate.py:93` | Asserts every prompt starts with BOS; this tokenizer may not add one |
| `src/core/segmentation.py:86` | Rejects special tokens in the prompt; a chat template contains them |
| `src/core/segmentation.py` splitters | Tuned to SFT output. The MATH splitter already fails on about 1% of RhoMath samples; the rate on free-form chat output is unknown |
| `src/core/verifier.py` | GSM8K grading expects a `#### ` answer line |

Put the prompt, stop rule and context budget in one place keyed by model, record the key in `metadata.json`, and have the evaluator assert on it. Do not add a second copy of the constants.

**Resources (all estimates):**

- **GPU memory, colocated on one 80 GB H100:** actor base 14.6 GB, reference 14.6 GB, replica capped at 32 GB, leaving about 18 GB for activations with no gradient checkpointing. Expect to lower `--microbatch-size`. Run `benchmark.slurm` before any pilot.
- **Disk:** the compact format stores the frozen actor base and the reference once per run, about 29 GB, plus the 14.6 GB download. Resolve decision 2 first. Referencing the local model files by hash instead of copying them would remove the 29 GB.
- **LoRA:** `all-linear` at rank 16 gives about 40M trainable parameters.

**Then:** `scripts/diagnostics/sample_segmentation.py` for the rejection rate, a four-iteration smoke for each of GRPO and VinePPO, a three-iteration benchmark, and only then a pilot sized from the benchmark.

The first handoff said to keep PACE model restrictions in force. I did not find where they are written down; check before staging the weights.

## Rules for every launch

- **Pin the code.** Each job re-imports the evaluator at every validation, so an edit to the launch tree mid-run mixes versions. Commit, then launch from a detached worktree of that commit and keep developing in the main checkout. Untested: pass `PYTHON` and `VLLM_PYTHON` as absolute paths to the main checkout's environments, pass absolute model and dataset paths, and export `WANDB_API_KEY` (the tracker otherwise reads `.env` beside the code).
- **Smoke before seeds.** Gate long jobs on a short end-to-end job with `--dependency=afterok:<id> --kill-on-invalid-dep=yes`.
- **Tests and GPU code run under `sbatch`**, never on the login node.
- **Use `--checkpoint-format compact-lora`** for LoRA runs, a fresh `--output`, and a new W&B group per batch of runs.
- **Report the uncertainty that matches the claim.** Question-clustered intervals describe one checkpoint; seed spread describes the method.

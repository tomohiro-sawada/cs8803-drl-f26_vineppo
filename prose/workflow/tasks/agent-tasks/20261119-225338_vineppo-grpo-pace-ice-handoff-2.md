# Handoff 2: batched VinePPO and GRPO on PACE-ICE

Written 20261119-225338 at commit `fefee7e`; updated 20261119-230622 at commit `4050b3d` with Tom's decisions (see "Decisions from Tom"). Everything is committed and pushed to `origin/tom/work`.
Continues `20261119T173853Z_vineppo_grpo_cluster_handoff.md`; read that for the full spec.
Tom's target: RhoMath 1.1B results within about 12 hours. The small model (MobileLLM, SmolLM) is dropped.

## State in one paragraph

The distributed sampler, the single-GPU colocated layout and evaluation through the replica all work end to end for **GRPO**. The 50-iteration GRPO pilot reached iteration 39 and then **failed** at its iteration-40 validation because of a bug in my retry code. **VinePPO cannot run a pilot yet**: 2.3% of sampled responses are rejected by the strict segmentation check, which aborts the run. Tom has now decided how to fix that (below); it is not implemented.

## Decisions from Tom (20261119-230622)

- **Segmentation: approved.** Keep the exact sampled token IDs and compute character spans by decoding them incrementally, instead of re-encoding the decoded text. Update the regression tests in `tests/core/test_segmentation.py` and the `AGENT INSTRUCTIONS` note in `src/core/segmentation.py` to record the new policy.
- **Not approved yet:** what to do when a step boundary falls inside a token. Upstream assigns the token to the step containing its first character. Keep raising until Tom signs off; no such case occurred in 512 samples.
- **Learning signal: approved order of work.** Before any hyperparameter tuning: (1) more validation rounds per checkpoint, (2) a longer run, (3) upstream's 512 episodes per iteration if still flat, (4) only then a learning-rate sweep.

## Do these first

1. **Fix the retry bug, then resume the GRPO pilot** (details below).
2. **Implement incremental-decoding segmentation**, rerun `scripts.diagnostics.sample_segmentation` to confirm the rejection rate, then run the VinePPO benchmark and pilot.
3. **Rerun the pilots with a better measurement** (see "Learning signal").

Job 7184580 (`smoke-resume-grpo`) **completed, exit 0**: plain resume on the vLLM backend works. I did not inspect its output beyond the exit status (`$OUT/smoke-resume-grpo-01`).

Job 7184517 (`flash-attn-build`) is Tom's, not part of this work. Leave it alone.

## Bug that killed the GRPO pilot (job 7184690)

- **Symptom:** `AssertionError: Workers stopped before completing every request` in `VLLMClient.complete` (`src/core/vllm_inference.py`), raised inside the iteration-40 validation child.
- **Mechanism (from reading the code, not reproduced):** on an `httpx.TransportError` a drain thread requeues the request and then returns, retiring its slot. If every other thread has already seen an empty queue and exited, the requeued request is orphaned. With `--eval-batch-size 256` there is one thread per request, so any late transport error orphans it.
- **Trigger unknown.** The child's error counters were lost with the process. Validations 0, 10, 20 and 30 passed with the same settings, so it is intermittent. One lead: 256 threads share one `httpx.Client`, whose default pool is 100 connections.
- **Suggested fix:** after requeueing, keep the thread draining instead of returning, keep the bounded attempt count, and log the exception text. Add a test in `tests/core/test_vllm_inference.py` where the failure happens after the other slots have drained; the existing retry test does not cover that ordering.

### Resuming the pilot

`iteration-40.pt` was saved before validation failed; `selection.json` holds candidates 10, 20, 30. Resume needs a **new** output directory and both resume flags:

```bash
P=$OUT/pilot-grpo-rhomath-gsm8k-seed42-vllm-colocated-01
sbatch --cpus-per-task=12 --mem-per-cpu=8G --time=04:00:00 --job-name=pilot-grpo-rhomath \
  experiments/training/pace_ice/train.slurm \
  --task gsm8k --model-path .cache/models/rho-1b-sft-GSM8K --dataset-path .cache/datasets/gsm8k \
  --output $OUT/pilot-grpo-rhomath-gsm8k-seed42-vllm-colocated-02 \
  --resume $P/checkpoints/iteration-40.pt --resume-selection $P/selection.json \
  --algorithm grpo --backend vllm --num-workers 1 --worker-placement colocated \
  --prompts-per-iteration 16 --num-samples 8 --num-iterations 50 --update-epochs 2 \
  --minibatch-size 32 --microbatch-size 4 --max-sequence-length 2048 --max-new-tokens 1024 \
  --seed 42 --dtype bfloat16 --finetuning lora --lora-rank 16 --lora-alpha 32 \
  --learning-rate 0.00001 --checkpoint-every 10 --keep-checkpoints 2 \
  --evaluation validation-test --validation-rounds 1 --eval-batch-size 256 --eval-seed 42 \
  --wandb-mode online --wandb-group pilot-20261119 --wandb-name pilot-grpo-rhomath-seed42-resumed
```

Resume **with selection** on the vLLM backend has never run on a GPU. It should validate iteration 40 before training. If it fails, a fresh 50-iteration run costs only about 15 minutes of GPU time.

## VinePPO blocker: segmentation rejects 2.3% of samples

- **Evidence:** job 7184578, `$OUT/segmentation-rate-01`. 512 samples at T=1 from the SFT actor: 500 accepted, 12 rejected.
  - 10 are `roundtrip`: the sampled token IDs are not the tokenizer's canonical encoding of their own text (for example sampled `+`,`$` versus re-encoded `+$`).
  - 2 are `upstream_or_span`: length-truncated 1024-token responses, `ValueError: too many values to unpack`.
- **Consequence:** a 128-episode iteration is clean with probability about 5%; a 50-iteration run essentially never. Job 7184402 (`smoke-e2e-vineppo`) died this way at iteration 4. The earlier 3-iteration VinePPO smoke passed by luck.
- **Cause of the 10:** at T=1 the model sometimes emits a token sequence that is not the canonical encoding of its text. Both decode to the same string, but re-encoding gives different IDs, so the check fails.
- **The 2 truncated-response failures are a separate bug** in the GSM8K step splitter on a 1024-token response. Incremental decoding will not fix them. Not diagnosed; the records are in `$OUT/segmentation-rate-01/records.jsonl`.
- **What upstream does** (read at `9f716e6`, `episode_generator_with_reward_function.py` and `math_episode_generator_with_mc_advantages.py`): it keeps only text, re-tokenizes `query + response` with offsets, checks that the new IDs decode back to the same text, and assigns each token the advantage of the step containing its first character. Responses whose step splitting raises are logged, counted in `parse_failed`, and dropped.
- **Why we do not copy that:** the training sequence would then be the canonical encoding and not the sampled one, so for about 2% of episodes the gradient is taken on a path the policy did not sample. Old log-probabilities are rescored on the training sequence, so the PPO ratio is unaffected; the bias is in which sequence is trained on, and is probably small (not measured). Re-tokenizing remains a defensible alternative if matching upstream matters more.
- **Decided:** incremental decoding (see "Decisions from Tom"). The other options considered were treating a rejected response as one step, and dropping rejected episodes.

Do not launch VinePPO benchmarks or the VinePPO pilot until the fix is in and the rejection rate is remeasured.

## Results so far

All outputs are under `OUT=/home/hice1/tsawada7/scratch/cs8803-drl-f26_vineppo/training/pace_ice`.

| Run | Job | GPU | Outcome |
|---|---|---|---|
| `smoke-vllm-1p1-{vineppo,grpo}-a40-01` | 7179619, 7179620 | 2x A40, dedicated | 3 iterations each, zero worker errors |
| `sampler-acceptance-1p1-a40-01` | 7179833 | 2x A40 | passed; see numbers below |
| `smoke-e2e-vineppo-01` | 7184402 | 1x A40, colocated | failed at iteration 4, segmentation |
| `smoke-e2e-grpo-01` | 7184579 | 1x A40, colocated | completed: baseline, 2 validations, freeze, 16-round test |
| `bench-grpo-vllm-colocated-01` | 7184581 | 1x A40 | 51.5 s/iteration (3 measured) |
| `bench-grpo-hf-01` | 7184583 | 1x H100 | 52.7 s/iteration (3 measured) |
| `pilot-grpo-rhomath-gsm8k-seed42-vllm-colocated-01` | 7184690 | 1x H100, colocated | failed at iteration-40 validation |

- **Sampler acceptance:** greedy output of the untrained adapter matched the base model and the HF learner on 8/8 prompts. Replica versus learner log-probabilities differ by 0.010 mean, 0.12 max per token with the untrained adapter (BF16 and backend), and 0.014 mean, 0.23 max after a deliberate adapter change that moved learner scores by up to 8.3. The retired actor was rejected. The `learner_min_top2_logit_gap` field is 0.0 everywhere and is broken; ignore it.
- **End-to-end smoke (GRPO, 4 tiny iterations):** final test 40.36% over 16 rounds (std 0.63 points), consistent with the 40.4% SFT baseline. The 16-round test took 650 s on an A40.
- **Benchmarks are not matched:** the two GRPO benchmarks ran on different GPU types, so they do not compare HF with vLLM. On the A40 the update dominated (about 40 s of 51 s); on the H100 HF generation dominated (about 39 s of 53 s).
- **Pilot timing (H100, colocated):** about 11 s per iteration without validation, about 45 s per one-round validation.
- **Pilot learning signal: none yet.** Validation was 54.69% at iteration 0, then 53.89%, 54.16%, 54.16% at iterations 10, 20, 30. Two smoke runs with the same seed gave baselines of 53.89% and 54.42%, so run-to-run noise is at least half a point and these differences are inside it. `approx_kl` rounds to 0.000 and `clip_frac` is 0.001 throughout: at LR 1e-5 and 8 optimizer steps per iteration the policy is barely moving. Report this as it is; do not tune it without Tom.
- **Validation is not reproducible across runs** even at a fixed seed, because requests are batched concurrently on the replica.

## Learning signal

The flat curve is expected at this scale and is not yet evidence of a problem.

- **The pilot is small next to upstream's recipe** (`configs/trainers/ppo_MATH.jsonnet` and the Rho configs at `9f716e6`):

| | Upstream | Our pilot |
|---|---|---|
| Episodes per iteration | 512 | 128 |
| Iterations | 650 (GSM8K) | 50, reached 39 |
| Batch per optimizer step | 64 | 32 |
| Optimizer steps per iteration | 16 | 8 |
| Learning rate | 1e-6, full fine-tune | 1e-5, LoRA rank 16 |
| Sampling temperature | 0.6, top-p 0.9 | 1.0, no top-p |

  That is about 3% of upstream's optimizer steps and 1.5% of its episodes.
- **One validation round cannot resolve a small gain:** on 373 questions the standard error is about 2.6 points.
- **The policy is moving slowly:** KL to the reference grew from 0.0002 to 0.016 over 39 iterations; gradient norm 0.35-0.47 (clip is 1.0); clip fraction 0.1-0.2%. Mean training reward was 0.39 over the first ten iterations and 0.45 over the last ten, which is noisy because each iteration draws different prompts.
- **Approved next steps, in order:** raise `--validation-rounds` (each round costs about 45 s; four rounds halves the standard error), run longer (650 iterations at this batch size is roughly two hours plus validation), then try 512 episodes per iteration, and only then sweep the learning rate.
- **Not verified:** how upstream uses its sampling temperature in the loss, and upstream's KL coefficient (the Rho configs import `refKl0.0001.jsonnet`, which I did not open; ours is 1e-4).

## Not done

- Incremental-decoding segmentation (approved, not implemented).
- VinePPO benchmark and pilot (blocked on the above). Jobs 7184582 and 7184584 were cancelled before starting.
- Frozen sampler replay (`scripts/diagnostics/benchmark_sampler_replay.py`): never executed.
- 1 learner + 3 workers: never run. It needs `--ntasks=4 --nodes=2` for A40s; untested.
- Budget-gate write-up and the matched benchmark table.
- Step 3 GPU checks from the first handoff (forced early winner, best checkpoint surviving pruning after resume): only the existing CPU tests cover these.
- Full test suite has not been rerun since the launcher changes; `tests/test_training_launchers.py` and `tests/test_training_runner.py` pass (185).

## How to submit

```bash
cd /home/hice1/tsawada7/Files/projects/cs8803-drl-f26_vineppo
export TRAINING_CACHE=/home/hice1/tsawada7/scratch/cs8803-drl-f26_vineppo/training-cache
export VLLM_PYTHON=$PWD/.venv-inference-glibc2.34/bin/python
OUT=/home/hice1/tsawada7/scratch/cs8803-drl-f26_vineppo/training/pace_ice
```

- **Use single-GPU colocated jobs** (`--backend vllm --num-workers 1 --worker-placement colocated`, default `--ntasks=1`). They started within seconds to a few minutes tonight; two-GPU single-node requests waited 3h45m.
- **GPU type:** the PACE wrappers now accept A40, L40S, A100, H100 or H200. Omit `--expected-gpu` to take whichever is free; the actual GPU is recorded in `metadata.json`.
- **Per-job overrides used tonight:** `--cpus-per-task=12 --mem-per-cpu=8G`.
- **Replica startup** takes 3-5 minutes before the first iteration.
- **Logs:** Slurm output in `artifacts/training/pace_ice/slurm/`, replica logs in `$TRAINING_CACHE/workers/<jobid>/worker-0.log`, validation child logs in `<run>/evaluations/*.log`.
- **W&B:** online logging works from compute nodes using `WANDB_API_KEY` in `.env`. Runs go to `https://forge.coreweave.com/wandb/tsawada/vineppo` (that host, not wandb.ai), in groups `smoke-20261119`, `bench-20261119`, `pilot-20261119`. The failed pilot is run `eb0059e909154ab88da5da8816a538a2`.
- **Disk:** `$OUT` is 52 GB against a 300 GB scratch quota; checkpoints are 4.55 GB each. Nothing has been deleted.
- Do not run the test suite or GPU code on the login node; use `sbatch` on `ice-cpu` for tests.

## Commits from this session

Both are on `origin/tom/work`; the working tree was clean at `4050b3d`.

- `0224577`: `--worker-placement dedicated|colocated` in `launch.sh` and `run.py`; wider PACE GPU allowlist with optional `--expected-gpu`; `--gpu-memory-utilization` required in `vllm_worker.py` (0.4 colocated, 0.85 dedicated); launcher tests; tonight's Slurm logs; this document.
- `4050b3d`: Tom's concurrent work, committed at his request (PACE-ICE scheduler logger and test, FlashAttention build scripts, `.gitignore` and `scripts/README.md` edits).

## Design decisions made so far

- **Colocated layout** departs from the first handoff's separate-GPU requirement. Learner and replica alternate, so they share one GPU; the replica is capped at 40% of memory.
- **Replica adapters are BF16; learner adapters are FP32.** Sampling and rescoring agree only to the differences measured above.
- **Evaluation through the replica is a distinct protocol** (server-side stop string, per-request seeds). Its numbers are not directly comparable with the earlier HF evaluations, and validation accuracy (about 54%) is on a different split from the 40.4% test figure.
- **The iteration-0 baseline** is stored in `selection.json` under `baseline` and is never a selection candidate.
- **A worker exiting ends the whole job.**

## Execution log (20261119-233556, HEAD `46d3041`)

- vLLM drain slots now keep consuming the queue after bounded transport retries. The regression covers a single remaining slot retrying the same request and seed.
- Segmentation decodes the original sampled prefixes in prompt context, including byte-split Unicode. GSM8K uses the first exact answer marker to begin its terminal answer suffix, which also covers any repeated markers. The strict inside-token assertion still applies.
- CPU preflight `7185504` accepted all 512 saved responses. The original 500 successes retained identical boundaries. All ten noncanonical token sequences and both truncated responses with repeated answer markers now pass. The fresh GPU rerun is `7185468`, with the original settings of 64 prompts, 8 draws, batch size 16, seed 42 and a 1,024-token cap.
- The local GSM8K dataset has 7,100 train, 373 validation and 1,319 test rows. Pairwise overlap was zero after whitespace normalization and case folding. Split fingerprints and normalized question SHA256s are in `artifacts/training/pace_ice/slurm/vineppo-preflight-7185504.out`.
- GRPO resumed at iteration 40 on an A40 in `7185453`, validated the recovered checkpoint before updating and trained through iteration 50. One-round validation was 54.69% at baseline and 53.89%, 54.16%, 54.16%, 58.98%, 59.79% at iterations 10–50. Baseline through iteration 30 used the H100 allocation. The frozen selection is `$OUT/pilot-grpo-rhomath-gsm8k-seed42-vllm-colocated-02/selection-frozen.json`; it points to iteration 50.
- The GRPO final test stalled after 6 of 16 rounds. Logs showed three `ReadError('[Errno 9] Bad file descriptor')` retries, and the replica was idle with no established evaluator sockets. The evaluator still had 284 threads. The client now closes sockets after their responses, uses the drain slots as its concurrency limit and bounds the transport timeouts. Standalone A40 retry `7185849` reads the same frozen selection and uses 16 rounds, seed 42 and batch size 256. The stalled allocation was cancelled after checking that its checkpoint and selection were intact.
- `7185621` passed 1,533 tests on `ice-cpu`. The previous full run timed out in the existing fake distributed launcher test, which passed in the rerun. `7185841` reran the suite after the HTTP connection changes.
- Matched H100 GRPO vLLM benchmark `7185497` repeats HF job `7184583` at 16 × 8, one warm-up and three measured iterations. Fresh four-round H100 GRPO pilot `7185731` uses the original training settings, evaluates baseline and every ten iterations through 50, and reserves the test split for the frozen winner.

## Results and follow-up runs (20261120-003240, HEAD `46d3041`)

The fresh A40 diagnostic `7185468` accepted 512/512 responses, including all four length-truncated responses. The first four-round VinePPO pilot `7186263` then stopped before update 4 when a tokenizer prefix changed after another byte arrived. RhoMath's byte-fallback decoder can temporarily replace a whole run of bytes, including an earlier complete character. The revised alignment selects exact decoded prefixes against the full decoded response and retains their original token indices. CPU job `7186741` passed 1,536 tests and replayed 1,024 saved responses without changing any previously accepted boundaries. Fresh VinePPO retry `7186748` uses the verified code.

| H100 benchmark | Slurm job | Wall clock | Queue | Mean s/iteration | Primary generation s/iteration | MC estimation s/iteration | Actor update s/iteration | Whole-iteration sampled tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GRPO HF | 7184583 | 0:03:33 | 0:07:22 | 52.71 | 39.14 | | 9.39 | 289 |
| GRPO colocated vLLM | 7185497 | 0:05:02 | 0:26:29 | 15.49 | 2.11 | | 9.69 | 982 |
| VinePPO HF | 7186265 | 2:08:01 | 0:11:08 | 2,033.89 | 10.82 | 2,012.31 | 5.49 | 203 |
| VinePPO colocated vLLM | 7186013 | 0:03:30 | 0:06:15 | 38.99 | 2.05 | 21.75 | 9.49 | 13,642 |

Each benchmark uses 16 × 8 primary draws, one warm-up and three measured iterations. VinePPO uses nine siblings per MC prefix. The colocated VinePPO run sampled 45,729 primary and 1,549,981 MC tokens across 1,553 prefixes during the measured iterations. Both colocated benchmarks recorded zero worker errors. Raw `metrics.jsonl`, `summary.json` and `metadata.json` are in `$OUT/bench-grpo-hf-01`, `$OUT/bench-grpo-vllm-colocated-h100-02`, `$OUT/bench-vineppo-vllm-colocated-02`, and `$OUT/bench-vineppo-hf-h100-02`.

| GRPO validation iteration | H100 four-round accuracy | Paired change vs SFT, percentage points | Approximate 95% paired interval |
|---:|---:|---:|---:|
| 0 | 53.55% | | |
| 10 | 54.16% | +0.60 | −0.76 to +1.96 |
| 20 | 54.49% | +0.94 | −0.73 to +2.61 |
| 30 | 54.02% | +0.47 | −1.70 to +2.64 |
| 40 | 55.56% | +2.01 | −0.66 to +4.69 |
| 50 | 56.57% | +3.02 | +0.29 to +5.74 |

- GRPO job `7185731` selected iteration 50 and scored **43.77%** on the 16-round test, with a 95% question-clustered interval of 41.56–45.97%. The resumed original pilot's frozen iteration 50 scored **44.12%** in retry `7185849`, with a 95% question-clustered interval of 41.90–46.34%. Both tests account for all 21,104 predictions. Raw per-round results are in each evaluation's `result.json`, under `evaluations/test-selected` for the H100 run and `evaluations/test-selected-retry-01` for the A40 retry.
- The 50-iteration H100 budget, with 20% headroom, is about 63 minutes for GRPO and 92 minutes for VinePPO, or 92 minutes elapsed and 2.6 GPU-hours for the pair on two cards. Inputs are 7 minutes startup, 16 or 45 seconds per steady iteration, five 5-second checkpoint writes, six 195-second four-round validations, and a 12-minute final test. This projection counts checkpoint and evaluation time separately from iteration time. GRPO's observed Slurm runtime was 19:19 after 15:56 in the queue. The two completed colocated benchmark queue times were 26:29 (GRPO) and 6:15 (VinePPO).
- Use 200 iterations with four validation rounds every 25 iterations for the longer stage. The same conservative projection gives about 123 minutes GRPO and 239 minutes VinePPO, plus queue, including headroom. At the VinePPO steady-time bound, 650 iterations would take about 8.1 hours for rollouts and updates alone before headroom and evaluation. GRPO job `7186783` starts the 200-iteration stage from SFT with the same rollout and optimizer settings.
- Frozen sampler replay `7186293` requests one learner and one worker on H100s, using eight prompts, two primary draws, nine siblings and in-flight bounds 32, 128, 256, and 512. Its recorded exact prefixes are the input for the later three-worker repeat.

## Matched pilot results (20261120-014733, HEAD `46d3041`)

VinePPO retry `7186748` completed 50 iterations and the 16-round test on one H100 in 37:20 after 10:04 in the queue. It sampled 6,400 primary responses and 217,008 MC continuations, with zero worker errors. The [validation plot](../../../../artifacts/training/pace_ice/rhomath-gsm8k-seed42-validation.png) shows both arms against iteration, cumulative primary plus MC tokens, and allocated H100 hours through each validation.

| Iteration | VinePPO four-round accuracy | Paired change vs SFT, percentage points | Approximate 95% paired interval |
|---:|---:|---:|---:|
| 0 | 53.69% | | |
| 10 | 54.09% | +0.40 | −1.07 to +1.88 |
| 20 | 54.22% | +0.54 | −1.45 to +2.52 |
| 30 | 56.30% | +2.61 | +0.05 to +5.18 |
| 40 | 57.17% | +3.49 | +0.32 to +6.66 |
| 50 | 58.24% | +4.56 | +1.15 to +7.97 |

| Run | Wall clock | Queue | Validation-selected iteration | Validation Pass@1 | Test Pass@1, 16 rounds | Approximate 95% question-clustered interval |
|---|---:|---:|---:|---:|---:|---:|
| GRPO 50, H100, 7185731 | 0:19:19 | 0:15:56 | 50 | 56.57% | 43.77% | 41.56–45.97% |
| VinePPO 50, H100, 7186748 | 0:37:20 | 0:10:04 | 50 | 58.24% | 46.04% | 43.77–48.31% |
| GRPO 200, H100, 7186783 | 0:42:05 | 0:15:32 | 150 | 61.93% | 47.09% | 44.72–49.46% |
| VinePPO 200, H100, 7187902 | 1:50:34 | 0:17:44 | 175 | 67.23% | 50.93% | 48.58–53.29% |

- At 50 iterations, the paired VinePPO minus GRPO test difference was +2.27 percentage points (95% question-clustered interval +0.97 to +3.57). This is a single seed, with 20.76M sampled tokens and 0.557 allocated H100 hours through validation for VinePPO versus 0.666M tokens and 0.258 hours for GRPO.
- The longer GRPO run finished in 42:05 on one H100. Its iteration-150 winner improved +8.31 percentage points over its own SFT validation (paired interval +4.60 to +12.02). Pruning retained iterations 150, 175 and 200, and final test reloaded iteration 150. The matched 200-iteration VinePPO run is `7187902`. Training uses 128 primary episodes per iteration in every run here.
- Raw studies are under `$OUT/pilot-vineppo-rhomath-gsm8k-seed42-vllm-h100-4rounds-02` and `$OUT/long-grpo-rhomath-gsm8k-seed42-vllm-h100-200iter-01`. Each has a frozen selection, per-round question records, and a test result.
- One-worker frozen replay `7186293` completed in 7:25 after 65:43 in the queue on two H100s. It replayed 52 prefixes with nine siblings each. The fastest tested bound was 128 sequences, at 17,098 completion tokens/s versus 249 tokens/s for sequential HF (31,885 versus 33,475 output tokens). Exact inputs and the complete sweep are in `$OUT/sampler-replay-rhomath-h100-1plus1-01`. Three-worker repeat `7187904` reads those prefixes and waits for the HF benchmark and long VinePPO jobs to release their GPUs.

## Completion (20261120-113927, HEAD `46d3041`)

Both 200-iteration runs improved over their SFT validation baselines. VinePPO selected iteration 175 at 67.23%. The 16-round test difference versus GRPO 200 was +3.84 percentage points (95% question-clustered interval +2.20 to +5.48). The runs use one seed and equal counts of primary episodes. VinePPO sampled 80.86M tokens in total, including 78.07M MC tokens, versus GRPO's 2.69M. These trajectories show gains at 128 primary episodes per iteration, so the conditional 512-episode experiment and learning-rate sweep were unnecessary.

| Iteration | GRPO validation | GRPO change vs SFT, pp (95% interval) | VinePPO validation | VinePPO change vs SFT, pp (95% interval) |
|---:|---:|---:|---:|---:|
| 0 | 53.62% | | 53.75% | |
| 25 | 54.42% | +0.80 (−1.35, +2.96) | 54.89% | +1.14 (−1.25, +3.53) |
| 50 | 57.37% | +3.75 (+0.75, +6.75) | 60.32% | +6.57 (+3.49, +9.65) |
| 75 | 59.32% | +5.70 (+2.27, +9.12) | 61.26% | +7.51 (+4.15, +10.87) |
| 100 | 58.45% | +4.83 (+1.20, +8.45) | 61.93% | +8.18 (+4.64, +11.72) |
| 125 | 60.52% | +6.90 (+3.16, +10.65) | 62.00% | +8.24 (+4.70, +11.78) |
| 150 | 61.93% | +8.31 (+4.60, +12.02) | 64.34% | +10.59 (+6.58, +14.60) |
| 175 | 61.13% | +7.51 (+3.53, +11.48) | 67.23% | +13.47 (+9.70, +17.24) |
| 200 | 59.79% | +6.17 (+2.20, +10.13) | 66.35% | +12.60 (+8.67, +16.53) |

- **Wall clock** is Slurm start to end, including startup, checkpoint I/O, validation, and final test where applicable. The original GRPO pilot allocated 13:39 (`7184690`, stopped at validation 40), 22:25 (`7185453`, trained to iteration 50 and stalled during final test), and 14:03 (`7185849`, completed the standalone final test). Total allocated wall time was 50:07, and 1:15:56 elapsed from the original allocation start to the final result. The interrupted VinePPO attempt `7186263` allocated 7:39 and completed three updates.
- **GPU selection fixture:** `7187992` ran 11:09 and exited with the intentional status 42 after validating checkpoint 2. `7187993` resumed and completed in 6:32. It retained the iteration-1 winner through iteration 4, pruned iteration 3, and verified the frozen selection still pointed to the source checkpoint. The fixture uses single-request validation and a near-zero learning rate to keep the actor stable during the retention check.
- **Matched timing:** Colocated vLLM reduced average iteration time by 3.40× for GRPO and 52.17× for VinePPO at the matched H100 rollout settings. The VinePPO HF arm sampled 1.20M MC tokens in the timed iterations, and vLLM sampled 1.55M. The completed two-card colocated pilot pair used 0.94 GPU-hours and had a 37:20 critical-path runtime excluding queue. The matched 200-iteration pair used 2.54 GPU-hours and had a 1:50:34 critical-path runtime on separate cards.
- **Three-worker replay:** `7187904` used four H100s across two nodes for 9:00 (0.60 GPU-hours) and read the byte-identical 52-prefix workload. At 128 sequences per worker it achieved 36,110 completion tokens/s versus 17,098 for one worker. The one-worker replay used 0.247 GPU-hours. The four-card replay waited 6:20:10 after its last dependency finished. The single-card colocated layout met the pilot runtime target with two cards for the pair, and the replay timing isolates generation on a small fixed workload.
- **Three-worker acceptance:** `7187967` ran 1:28 on four H100s across two nodes. Worker greedy outputs agreed with the base replica on 8/8 prompts and with the HF learner on 6/8. Before the adapter change, all workers differed from the learner by 0.0129 mean absolute log probability per scored token. After the change, each worker's max score change exceeded 8.1, and all rejected the retired actor. The minimum-logit-gap diagnostic cannot explain the two greedy divergences.
- The final experiment is in `$OUT/long-vineppo-rhomath-gsm8k-seed42-vllm-h100-200iter-01`. Slurm stores the full submissions via `sacct -X -P -j <job-id> -o JobID,State,Submit,Start,End,Elapsed,AllocTRES,SubmitLine` and the inline scripts via `sacct --batch-script -j <job-id>`. The replay jobs use `TRAINING_ENTRY=scripts/diagnostics/benchmark_sampler_replay.py` and `REPLAY_MAX_SEQUENCES=32,128,256,512`. The three-worker replay uses the first replay's `prefixes.json` as `REPLAY_PREFIXES`. The acceptance job uses `TRAINING_ENTRY=scripts/diagnostics/check_remote_sampler.py`.
- The HF VinePPO benchmark's stderr is compressed losslessly as `artifacts/training/pace_ice/slurm/bench-vineppo-hf-7186265.err.gz`.

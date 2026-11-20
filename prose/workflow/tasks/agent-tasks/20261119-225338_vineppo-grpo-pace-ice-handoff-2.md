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

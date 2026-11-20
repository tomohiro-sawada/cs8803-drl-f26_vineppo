# Handoff 2: batched VinePPO and GRPO on PACE-ICE

Written 20261119-225338 at commit `fefee7e` (working tree dirty, see "Uncommitted changes").
Continues `20261119T173853Z_vineppo_grpo_cluster_handoff.md`; read that for the full spec.
Tom's target: RhoMath 1.1B results within about 12 hours. The small model (MobileLLM, SmolLM) is dropped.

## State in one paragraph

The distributed sampler, the single-GPU colocated layout and evaluation through the replica all work end to end for **GRPO**. The 50-iteration GRPO pilot reached iteration 39 and then **failed** at its iteration-40 validation because of a bug in my retry code. **VinePPO cannot run a pilot yet**: 2.3% of sampled responses are rejected by the strict segmentation check, which aborts the run. That needs a policy decision from Tom, not a silent fix.

## Do these first

1. **Check job 7184580 (`smoke-resume-grpo`).** It was running when this was written. It is the first GPU test of resume on the vLLM backend (plain resume, no selection). Output: `$OUT/smoke-resume-grpo-01`. It should reach iteration 6 from `smoke-e2e-grpo-01/checkpoints/iteration-4.pt`.
2. **Fix the retry bug, then resume the GRPO pilot** (details below).
3. **Ask Tom for the segmentation decision** before touching the VinePPO arm.

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
- **Why I did not fix it:** `src/core/segmentation.py` carries `AGENT INSTRUCTIONS` forbidding silent recovery; changing the policy needs an explicit decision and updated tests in `tests/core/test_segmentation.py`.
- **Options to put to Tom:**
  1. Compute character spans by decoding the sampled tokens incrementally, so no re-encoding is needed. Keeps every sample and its exact tokens. Does not address the two truncated-response failures.
  2. Treat a rejected response as a single step (one advantage for the whole response).
  3. Drop or resample rejected episodes. Simple, but it biases the behaviour distribution and breaks the fixed 16x8 batch shape.

Until this is decided, do not launch VinePPO benchmarks or the VinePPO pilot.

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

## Not done

- VinePPO benchmark and pilot (blocked, above). Jobs 7184582 and 7184584 were cancelled before starting.
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

## Uncommitted changes

Mine, not yet committed:

- `experiments/training/launch.sh`: `--worker-placement dedicated|colocated`, wider GPU allowlist, optional `--expected-gpu`.
- `experiments/training/run.py`: the same option and the colocated GPU check.
- `experiments/training/vllm_worker.py`: `--gpu-memory-utilization` is now required (0.4 colocated, 0.85 dedicated).
- `experiments/training/pace_ice/{train,benchmark}.slurm`: constraint covers all five GPU types.
- `tests/test_training_launchers.py`: tests for the above.

Not mine (Tom's concurrent work; do not revert or commit on his behalf): `.gitignore`, `scripts/README.md`, `scripts/setup/flash_attention_installation.sh`, `scripts/setup/build_flash_attention_local.sh`, `scripts/diagnostics/log_pace_ice.py`, `tests/test_log_pace_ice.py`.

## Design decisions made so far

- **Colocated layout** departs from the first handoff's separate-GPU requirement. Learner and replica alternate, so they share one GPU; the replica is capped at 40% of memory.
- **Replica adapters are BF16; learner adapters are FP32.** Sampling and rescoring agree only to the differences measured above.
- **Evaluation through the replica is a distinct protocol** (server-side stop string, per-request seeds). Its numbers are not directly comparable with the earlier HF evaluations, and validation accuracy (about 54%) is on a different split from the 40.4% test figure.
- **The iteration-0 baseline** is stored in `selection.json` under `baseline` and is never a selection candidate.
- **A worker exiting ends the whole job.**

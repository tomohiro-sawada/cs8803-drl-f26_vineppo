# Cluster handoff — batched VinePPO and GRPO on GSM8K

## Assignment and order of operations

You own implementation, cluster verification, launch, monitoring, recovery, and reporting. Start with the RhoMath 1.1B GSM8K SFT checkpoint. Implement token-exact distributed sampling and arbitrary-actor evaluation, benchmark at matched rollout size, then launch a bounded learning pilot. Run the 140M benchmark when its approved weights are available. Reserve additional seeds and larger allocations for demonstrated learning. The one-day target is a design constraint to measure, not an established runtime.

The cluster checkout is authoritative. The source snapshot used for this handoff was supplied on November 19, 2026. No changes or jobs described below have been executed for you. Proposed switches below are design requirements until you wire them into the CLI.

## 1. Establish the baseline on the cluster

The previous Skynet run lived under /coc/flash13/tsawada7/Files/projects/cs8803-drl-f26_vineppo/artifacts/training/skynet/prelaunch-20261119-031148/{grpo-scale,vineppo-smoke}. The local model and dataset were .cache/models/rho-1b-sft-GSM8K and .cache/datasets/gsm8k under that checkout. Verify paths, repository instructions, active jobs, free space, and current branch before editing. Keep existing runs intact and use a fresh worktree if the checkout contains unrelated changes.

From the root run:
01. Use the shell commands below to locate current entry points, including the evaluator and validation split, which were not included in the snapshot.
02. Read the matched wrappers, trainer tests, checkpoint selection implementation, and dependency pins. Record commit, dirty diff, Python, torch/CUDA, transformers, PEFT, vLLM, driver, GPU model/UUID, tokenizer/model revision and split fingerprints in each run.
03. Run the existing relevant CPU tests and a small single-GPU HF control. Record command, result and any pre-existing failures.

~~~bash
git status --short --branch
git rev-parse HEAD
rg --files | rg 'AGENTS.md|slurm|sbatch|launch.sh|selection.py|inference.py|evaluate|vllm|test_'
rg -n 'class CheckpointSelection|class MeasuredInference|class PPOTrainer|def estimate_vineppo|SLURM|373' experiments src
squeue -u "$USER"
df -h .
~~~

Historical measured iteration (one warm-up and one measured iteration per job, BF16, LoRA rank 16 / alpha 32, 2048 context, 1024 output cap, two update epochs):

| Measurement | GRPO scale | VinePPO smoke |
|---|---:|---:|
| Prompts × primary samples | 64 × 8 | 4 × 4 |
| Primary output tokens / seconds | 55,492 / 318.91 | 1,599 / 7.66 |
| Branch prefixes × siblings | 0 | 56 × 9 |
| Branch output tokens / generation seconds | 0 | 31,813 / 173.28 |
| Score cache / actor update seconds | 13.42 / 192.35 | 0.42 / 1.62 |
| Checkpoint seconds | 49.98 | 49.65 |
| Entire iteration seconds | 574.78 | 232.68 |

Generation rates were 174 primary tokens/s for GRPO and 184 MC tokens/s for VinePPO. The summary rates 96.5 and 143.6 include updates and checkpoint I/O. Branching consumed 95.2% of sampled VinePPO tokens. DDP has little upside for that smoke job. The 512-episode GRPO update took substantial time. Re-measure at 128 episodes. Never infer comparative learning speed from these two jobs.

## 2. Implement sampling on dedicated GPUs

**Process layout.** Start single-node with one learner plus one vLLM replica, each on its own GPU. Then test one learner plus three independent replicas on four A40s. One full model per replica, tensor parallel size 1, one coordinator and one optimizer. Learner hosts actor, frozen reference, rescoring and updates. Use worker count as configuration, so a future eight-GPU or A100 allocation requires no algorithm changes. Validate actual devices and memory per job. Avoid allocating eight cards or introducing DDP for the initial pilot. Separate GPUs provide memory isolation; speedup depends on occupancy and work distribution.

**Launcher.** Current launch.sh expects submission from repo root, a unique output, absolute TRAINING_CACHE outside outputs, explicit local assets and valued options. It launches one srun task; run.py currently insists on one visible device. Add backend and worker-count options end to end (Slurm wrapper, launch.sh allowlist, argparse, metadata). Preserve HF defaults. In distributed mode have the single Slurm job step acquire all requested local GPUs and start a coordinator/learner and persistent child workers with disjoint device visibility. Use subprocess or spawn with visibility set before CUDA initialization. Respect scheduler-provided CUDA_VISIBLE_DEVICES tokens, including remapped IDs; do not assume host GPU 0 belongs to you. Verify ranks, GPU UUIDs and allocation cardinality at startup. Give workers bounded CPU threads and distinct endpoints when needed. Cleanly stop children on normal exit, exceptions, job termination and timeouts. Current Skynet launcher accepts A40; PACE wrapper expects H100/H200. Extend a wrapper for A100 only after verifying actual scheduler resources. Never pass A100 as A40. Keep PACE model restrictions in force.

**Interfaces.** Locate core/inference.py, PPOTrainer.collect_completions, _validate_training, train, estimate_vineppo, _cache_scores and the measured hooks in experiments/training/run.py. The trainer currently asserts inference.model is policy. Preserve that check for local HF; replace it for remote workers with a verified actor version and base/adapter identity. Retain optimizer ownership and frozen-reference checks. Keep gradient-bearing scoring on the learner. Audit other inference.model accesses and close/context manager callers. Keep the fixed-checkpoint text evaluator usable.

**Collect → queue → reduce → update.**

1. Freeze behavior actor v. Sample all primary completions from v. Keep ordered prompt/sample indices for grouped GRPO rewards; never rely on worker completion order. Segment VinePPO primary response token IDs using the existing task-specific tokenizer logic, including EOS/context handling.
2. Build one MC job per nonterminal prefix for all episodes before waiting. Include version, iteration, example/episode ID, boundary index, exact prompt + response-prefix IDs, required number of siblings, sampling config and remaining budget. Use the existing continuation_budget rule (minimum of output cap and remaining context). Keep access to the correct example/reference per request; the old scorer closes over one example.
3. Dispatch many eligible requests across workers. A worker must receive multiple prefixes in one engine batch or enqueue them concurrently in a persistent engine. Keep bounded in-flight work, refill as slots free up and route new work to available workers. Group siblings on one worker when practical for prefix reuse. Independent sibling draws need distinct reproducible random streams. One synchronous call for each prefix misses the main optimization. Preserve exact per-request budgets and response ordering.
4. Return continuation-only token IDs, finish reason, sample key and version. Match the HF contract for stop vs length, including EOS on a stopped continuation and no padded tokens. Test EOS on the final allowed token, multiple EOS IDs, variable prefix lengths and an exhausted budget. Disable unintended generation defaults; translate HF sampling semantics to the installed backend explicitly (e.g. unrestricted top-k/top-p, no added penalties, same temperature). Avoid decode/re-tokenize, duplicated BOS, new chat templates, string stops or hidden truncation. Verify supported vLLM APIs from its installed version.
5. Verify exactly K unique successful completions per prefix. Score response prefix + sampled suffix; exclude prompt tokens. Preserve truncation reward, verification, mean sibling reward, adjacent prefix-value differences, terminal reward minus last value, and step-to-token mapping from vineppo.py. Keep callbacks, sufficient diagnostics and metric cardinalities. Only primary response tokens enter the existing actor update. A deterministic fake sampler should give identical advantages and recorded samples under shuffled arrival order. Deduplicate retries by logical ID; abort an incomplete iteration if full recovery at the same version is impossible.
6. Cache old and reference scores at the existing pre-update point. Perform existing updates. Publish actor v+1 only at the completed iteration boundary. Drain requests from v before switching, await acknowledgments from every worker, then allow new rollouts. On resume, load learner checkpoint first and initialize every worker from that restored actor.

**Weight publication.** For LoRA use the immutable base revision plus a versioned adapter if the installed engine supports that checkpoint's architecture, rank and target modules. Publish to a new directory and atomically expose a manifest after all writes finish. Use unique adapter identities per version. Include base/tokenizer fingerprint, actor/adapter hash, iteration and expected worker acknowledgments. Ensure cached prefix KVs cannot cross actor versions. An acknowledged rebuild from a fresh merged actor export is a correctness fallback; time export, load and cache invalidation separately. Avoid mutating the live learner or merging into the frozen reference. Full resumable trainer checkpoints and inference-only exports serve different purposes. Benchmark refresh frequency of once per iteration even on the fallback path.

**Sampler acceptance.** Exercise 1 HF GPU, 1 learner + 1 worker, then 1 + 3 with the same fixtures. Run fixed-prefix greedy checks, compare teacher-forced sampled-token logprobs with the frozen learner when supported, and inspect discrepancies relative to a same-backend BF16 control. Log absolute differences and near-tied logits before diagnosing a mismatch. For stochastic checks use multiple draws on a fixed prefix; identical strings across engines are unnecessary. Verify a deliberate adapter change affects fixed-prefix scores on every worker, a stale result is rejected, and a fresh process resumed from checkpoint emits the expected version. Check masks, finite losses and pre-update ratios near one on identical learner weights. Learner-recomputed old/current ratios alone cannot detect stale remote sampling.

## 3. Evaluate arbitrary trained actors

Trace CheckpointSelection.validate/freeze/test and the existing vLLM fixed-SFT script. Parameterize the latter or its loader so a checkpoint selection resolves to the exact base + adapter or standalone exported actor. Reuse the existing prompt formatter, verifier, EOS handling, sampling settings and split loader. Record the evaluated actor hash next to each score. Evaluation may reuse a verified, quiescent worker or run in a separate allocation. If it changes worker weights, restore and verify the training version before sampling resumes.

Run end-to-end tests with the initial SFT actor, a visibly changed trained actor, and that actor reloaded from a saved checkpoint. Compare exported-vs-live scores on fixed token sequences and verify every validation prediction carries the expected identity. Force an early checkpoint to win on a fixture and ensure the final test reloads that checkpoint even after newer ones exist. Resume a short run and confirm best-so-far survives pruning. Save prediction records with question ID, round, seed, answer, correctness and finish reason. Keep validation/test RNG isolated from training.

## 4. Experiment 1 — benchmark before training

Run RhoMath first. Repeat the matched systems comparison on facebook/MobileLLM-R1-140M-base for fast feedback after the pending access request is approved. Stage its approved pinned assets through the authorized account before launching offline jobs. If access is pending or the architecture is unsupported, document the exact blocker and continue RhoMath. Keep model-specific tokenizers and record prompt format, base vs SFT status, GSM8K training exposure where documented, initial accuracy and truncation. Interpret within-model systems results. Small-model speed or reward does not establish RhoMath learning.

| Stage | Workload | Purpose |
|---|---|---|
| Integration smoke | Tiny fixed prompts, both algorithms, two iterations with a real update/export between them | Correctness, version refresh, validation and resume |
| Frozen sampler replay | Same recorded RhoMath prefixes, K=9, HF vs vLLM; one vs three workers | Isolate routing/batching gains. Vary concurrent prefixes while preserving the workload |
| Matched training benchmark | Each available model × GRPO/VinePPO, 16 prompts × 8 responses. Compare HF and 1+1; measure 1+3 where it could improve the deadline | Budget realistic iteration, checkpoint, validation, and startup costs |

Use at least one warm-up and three measured iterations per timed configuration if feasible. Separate job startup/compilation, warm-up, learner update, score cache, MC sampling, refresh, checkpoint and evaluation. Use coordinator elapsed time to last result for parallel phases; summing worker times inflates wall time. Retain per-worker counters for imbalance. Record primary/MC tokens and completions separately, prefix lengths, output-length quantiles/max, request queue and generation time, memory per GPU and checkpoint bytes. Report useful output tokens/s for each generation phase, whole-iteration tokens/s and allocated GPU-hours (all allocated cards × elapsed hours). For replay compare workload lengths across sampled draws. Benchmark at least one real 373-question validation round and estimate final-test cost from actual per-round throughput. Save commands, raw JSONL and a small comparison table.

## 5. Experiment 2 — bounded RhoMath pilot

Freeze the following before inspecting learning curves. Suggested operational defaults below resolve choices left open by Tom.

| Setting | Pilot contract |
|---|---|
| Data | GSM8K first. Confirm exactly 373 fixed validation IDs and no overlap with policy-training or test examples. run.py loads every row of local train; establish that it is already filtered, or exclude holdout IDs explicitly. Record normalized question hashes as well as split hashes. Upstream SFT exposure is a separate provenance issue. |
| Matched rollout | Same initial weights, prompt order/seed (start at 42), formatter, decoder, context 2048 / new tokens 1024, verifier and truncation handling. 16 distinct prompts × 8 responses = 128 primary episodes per iteration per algorithm. GRPO groups within prompt. Initial VinePPO K=9 per nonterminal prefix. |
| Update | Start from the measured BF16 LoRA rank 16, alpha 32, LR 1e-5, two epochs. Proposed common minibatch 32, microbatch 4 (8 optimizer steps per iteration for 128 episodes). Verify effective accumulation and log actual steps. Preserve repo loss/KL defaults and record them. |
| Duration | Start with 50 iterations. Choose 100 only before launch and only if the benchmark budget supports it for both arms. One initial seed per arm. |
| Checkpoints and validation | Evaluate iteration 0 SFT once, then checkpoint and validate every 10 iterations, including the final iteration. Exactly one sampled pass across all 373 fixed validation questions with the same decoding settings and seed schedule. Use mean question accuracy. Predeclare best eligible trained checkpoint by validation accuracy, ties to earliest, reporting baseline separately. Keep best + latest resumable checkpoints. |
| Final test | Freeze one selected checkpoint per algorithm. Evaluate each on the official GSM8K test split for 16 independent seeded rounds using the same question set and decoder. Report mean single-sample correctness across rounds, per-round values and question-clustered uncertainty. Keep test results out of selection. |
| Learning signal | Plot validation accuracy against iterations, all sampled tokens and allocated GPU-hours. Report paired question-level changes vs iteration 0 for each scheduled checkpoint. Training reward, KL, clip fraction, zero-advantage fraction, group-zero-variance rate, completion lengths and truncation help diagnose results. An isolated one-round validation maximum is weak evidence. |

**Budget gate.** Calculate elapsed time separately per arm and for the entire pair under a maximum of four concurrently allocated A40s. Estimate startup + N × steady iteration time (including refresh) + scheduled checkpoints + validation rounds + final test and add 20% headroom. Count each cost once. Use conservative observed timings and report assumptions and Slurm queue time separately. Fifty iterations in 24 hours gives 1,728 s/iteration before overhead for one arm; two serial arms share that budget. Compare concurrent 1+1 jobs against sequential 1+3 jobs using measured times. For concurrent jobs use the slower arm for pair elapsed time; account for final eval contention. Choose the smallest layout that meets the target. If training fits but test needs a follow-on allocation, reserve it and state separate training and final-result ETAs. If neither layout fits 50 iterations, report the projected overrun and run a labeled shorter diagnostic within budget. Keep the planned 50-iteration result pending. Never silently change rollout count, K or eval protocol to claim the target.

**Launch procedure.** Discover the actual sbatch wrapper on the cluster. Submit from repo root with the site's correct GPU request, time limit, account, CPU/RAM, and an absolute shared TRAINING_CACHE. Verify GPU ownership inside the job. Use unique outputs per model, algorithm, seed, backend and layout. Example runner argument fragment after the wrapper and resource request are verified (append actual backend/layout flags implemented in step 2):

~~~bash
--task gsm8k --model-path "$model_dir" --dataset-path "$dataset_dir" \
--output "$new_output" --expected-gpu A40 --algorithm vineppo \
--prompts-per-iteration 16 --num-samples 8 --mc-num-samples 9 \
--num-iterations 50 --update-epochs 2 --minibatch-size 32 --microbatch-size 4 \
--max-sequence-length 2048 --max-new-tokens 1024 --seed 42 \
--dtype bfloat16 --finetuning lora --lora-rank 16 --lora-alpha 32 \
--learning-rate 0.00001 --checkpoint-every 10 --keep-checkpoints 2 \
--evaluation validation-test --validation-rounds 1 --eval-seed 42 --wandb-mode disabled
~~~

For GRPO change algorithm and output. Verify final test stays at 16 rounds independently of validation-rounds. For a separate final-test job use validation-only training and the frozen selection manifest. For benchmarks use benchmark mode with automatic evaluation disabled. Record the exact expanded submission command and Slurm job ID. Keep monitoring until completion or a concrete external blocker.

## 6. Diagnose by symptom

| Symptom | First discriminating check | Recovery |
|---|---|---|
| Pending job or wrong device count | squeue reason, scontrol show job, step-visible GPU list and UUIDs | Fix request/wrapper mismatch. Keep single-node layout. Do not launch workers on login nodes. |
| Multiple engines allocate on learner GPU | Log PID, visibility and UUID from each process before model load | Fix spawn environment and process-local ordinal mapping. Verify disjoint live GPU processes. |
| Worker startup fails, offline model missing, gated 403 | Check actual local config, tokenizer, shards, installed vLLM support and worker stderr | Stage authorized artifacts, pin compatible runtime and smoke-test one model. Continue RhoMath while MobileLLM is pending. |
| Sampler OOM / worker killed | Separate GPU KV/cache peak from host MaxRSS and learner activation peak | Reduce in-flight sequences or batched tokens and tune KV memory budget on workers. Reduce learner microbatch and preserve effective minibatch if learner OOMs. Preserve caps and K for matched comparisons. |
| Throughput unchanged after vLLM integration | Count pending requests, simultaneous distinct prefixes per worker, startup/reload time, long-tail lengths | Remove blocking waits inside the prefix loop; keep engines alive, refill queues, measure 1+1 before 1+3. One busy worker and two idle ones indicate routing trouble. |
| Samples look reasonable but rewards or KL diverge | Inspect token IDs, EOS, prompt duplication, finish reason, verifier input, adapter/version acknowledgments and behavior logprobs | Replay exact examples on HF; check sampling defaults, stale cache, reference identity and benchmark labels before tuning RL. |
| Missing siblings, hanging barrier, worker dies mid-iteration | Compare submitted/completed unique keys by version, check heartbeat, timeouts and stderr | Retry bounded work at the same version. If integrity is uncertain restart from the last complete checkpoint; discard partial metrics and do not apply partial updates. |
| Validation tracks SFT forever or best file disappears | Compare evaluated hashes/outputs after a deliberate weight change; inspect selection/pruning | Correct actor export/load and protect best_path. Re-run affected validation from retained checkpoints before selecting. |
| Checkpoint slow, disk full, timeout/preemption | Inspect shared filesystem space/quota and sacct job state. Historical checkpoints were 4.55 GB each | Budget best + latest retained + new in-flight checkpoint and exports per arm. Resume into a new output with --resume and matching --resume-selection when automatic selection is enabled. Verify iteration, optimizer, RNG, actor version and prompt stream. |
| No learning | Inspect baseline accuracy, per-prompt rewards, length stops, effective steps, advantage signs/scale, gradients, ratio/clip fraction and adapter deltas | Verify a tiny known-answer batch and a deliberately changed actor. If code is correct and validation remains flat, report it; defer seed expansion and DDP. |

For live diagnostics use squeue -j JOBID, scontrol show job JOBID, sacct -j JOBID --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode and the job's stdout/stderr plus metrics.jsonl. Sample worker GPU memory/utilization inside the allocation. Avoid duplicate resubmission while a job is pending or recovering.

## Completion evidence

Deliver the code diff and tests, exact commands, allocated GPU layouts and version-sync evidence, fresh and resumed smoke results, matched benchmark table, chosen budget projection, job IDs, validation trajectories, selection manifests and final per-round test results. Label unrun MobileLLM work or interrupted pilot arms explicitly. Recommend additional independent seeds only after improvement versus SFT recurs across scheduled validation measurements; use spare A40s for seeds before distributing actor updates. Revisit DDP only if matched-scale profiling shows learner updates materially limit end-to-end time.

Code review by Fable 5.2 Extra:

Reviewed commit: `46d3041986c51294f55476cbc91f598f58f3d560`. The original review below is preserved; concrete proposals and qualifications follow under **Proposed fixes**.

---

I found no defects in the core training math, but five problems will stop or silently spoil long runs. Two of them are already in your handoff notes; I reproduced those and pinned down the "undiagnosed" one.

I could not run the test suite: torch, transformers, sympy and httpx aren't installed here and package installs are off for this account. An organization owner can change that network setting, and a new chat would pick it up. I reproduced items 2 and 3 and the splitter error in item 1 by running the repo's own code with stubs; item 5 is arithmetic.

## Stops or spoils runs

1. **VinePPO aborts on any unsegmentable sample, and resume replays it** (`src/runtime/train.py:521`).
   - **Undiagnosed truncated-response failure:** it is `src/core/segmentation.py:162`. The two-target unpack raises `ValueError: too many values to unpack` whenever a runaway response contains `\n#### ` twice. Incremental decoding will not fix it.
   - **MATH:** your own proxy corpus shows 2 of 16 responses with a step boundary inside a token (`segmentation.py:205`). The "none in 512" evidence was GSM8K only.
   - **Resume:** the checkpoint restores RNG state and rejects a changed seed (`train.py:873`), and `resume-check` shows resumed metrics identical to the original. So on the HF backend a resumed run hits the same sample again.
   - **Fix:** drop and count such responses, with an asserted ceiling on the rate so it still fails loudly when something is wrong.

2. **One transient HTTP error aborts a vLLM call despite `retries=2`** (`src/core/vllm_inference.py:183-190`).
   - The slot requeues the request and then exits. If it was the last slot alive, nothing picks the request up, which is always the case with one slot per request (eval batch 256).
   - Fix: keep draining after a requeue, and retire a slot only after repeated consecutive failures.
   - The test at `tests/core/test_vllm_inference.py:265` accepts `AssertionError`, so it cannot catch this.
   - As far as I know, httpx's default pool is 100 connections, so 256 threads never run concurrently. Its 5-second keep-alive also matches what I recall as vLLM's server default, a plausible trigger; I did not verify either.

3. **A requeued job with `--backend vllm` exits at startup** (`experiments/training/launch.sh:108`). The worker directory is keyed by `SLURM_JOB_ID`, which a requeue keeps, so `mkdir` fails. `vllm_worker.py:52` would also reject the stale identity files. Key it by job ID plus `SLURM_RESTART_COUNT`.

4. **MATH grading starts a fresh Python process that imports sympy for every graded answer, serially** (`src/core/verifier.py:40-56`, called per MC continuation from `train.py:494`).
   - At roughly half a second each (my estimate, not measured), a 128-episode VinePPO iteration on MATH would spend over an hour grading.
   - A cache keyed on (predictions, references) would remove most of it.
   - A grader timeout or unlisted exception also aborts the run (`verifier.py:50-53`). That is rare (16,000 gradings in your committed evals passed) but has the same replay problem as item 1.
   - No MATH training run completed in the artifacts: `rloo-math` exited 1 and its log is not committed.

5. **Full-parameter training in bfloat16 mostly cannot move the weights** (`experiments/training/run.py:552-580`).
   - Near |w| = 0.02 the BF16 spacing is 1.2e-4, so an Adam step of 1e-6 or 1e-5 rounds to zero. For weights with σ ≈ 0.02, about 2% can change at 1e-6 and 15% at 1e-5.
   - This affects `--finetuning full --dtype bfloat16` and the PPO critic under bfloat16 even with a LoRA actor, so a PPO baseline would look worse than it is.
   - Fix: assert that trainable parameters are FP32.

## Smaller

6. **Checkpoints are 4.55 GB and take 43–65 s to write** (`train.py:817-820`). With LoRA, almost all of that is the frozen base and the reference, and each validation re-hashes the file about three times.
7. **A preemption during the final test strands the resumable wrapper.** The selection is already frozen, so every restart fails at `experiments/training/selection.py:98`. This is documented, but the wrapper has no test-only path.
8. **Provenance in `experiments/training/evaluate.py` is inaccurate in two places.** `evaluation_run` is set to the training run (line 593), and the deviations list describes HF even with `--backend vllm` (lines 614-620).
9. **Code used only by tests:**
   - `main` and `EvaluatorConfig` in `src/runtime/evaluate.py`
   - single-episode `estimate_vineppo` (`vineppo.py:90`, `train.py:209`)
   - `return_logprobs` in `HFInference.generate`
   - the three classes in `algorithms/base.py`
   - three stub scripts and an empty `tests/algorithms/test_base.py`
   - an unused import at `core/data.py:22`, a stale docstring at `core/data.py:82-83`, and stale directory names at `src/README.md:7-10`
10. **Repo hygiene:**
    - 58 MB of artifacts are committed, with 35 files over your own 500 KB pre-commit limit.
    - `.gitignore` lists `bash.sh`, `.trash` and three `prose/` directories that this diff adds, and two setup scripts `source ./bash.sh`.
    - `ruff-format` is an unused dev dependency; it is Reflex's bindings package and is not needed for `ruff format`.
    - Absolute paths, Slurm accounts and a cluster node inventory are committed; check them before the repo goes public.
    - I found no credentials.

## Coverage

I read all of `src/` and `experiments/training/`. In `scripts/` I read the setup and paper-replication code and skimmed the diagnostics; I sampled the tests. Lint is clean apart from the unused import.

Sources: [[ruff-format on PyPI](https://pypi.org/project/ruff-format/0.3.0/)](https://pypi.org/project/ruff-format/0.3.0/)

---

## Proposed fixes

Written `20261120-013813` (using `timestamp` from `bash.sh`), HEAD `46d3041`. These are implementation proposals, **not implemented fixes or approval to change experiment semantics**. Paths below are repository-relative. Original line numbers refer to the reviewed commit, not necessarily the changing working tree.

### Coordination and scope

The agent implementing [handoff 2](20261119-225338_vineppo-grpo-pace-ice-handoff-2.md) owns the active segmentation/client changes and running experiments. Do not duplicate or overwrite that work. During inspection, its uncommitted changes already included incremental sampled-token decoding, first-marker GSM8K splitting, the last-slot retry fix, stricter retry tests, and explicit HTTP connection settings. Its later byte-decoding revision and cluster test results are recorded in its execution log. These observations are a snapshot, not a claim that those changes have been committed or that all long-run failures are resolved.

This follow-up changes only this note. In particular, it does not change code, dependencies, checkpoints, selection manifests, jobs, or the other agent's handoff.

### 1. Segmentation failures and deterministic replay

**Immediate fix / existing parallel work:** retain the exact sampled IDs; derive spans from decoded prefixes in prompt context. Replace the unbounded GSM8K answer-marker split with `reasoning, marker, answer = text.partition("\n#### ")`, retaining the complete suffix, including later markers. Do not change the verifier's last-answer extraction rule: segmentation and answer grading have different jobs. The parallel agent already implemented this, so review and integrate its tests rather than writing a second implementation.

**Remaining policy decision:** the handoff explicitly says inside-token handling is not approved and must continue raising. Therefore do **not** silently adopt the review's blanket drop recommendation. The 2/16 MATH observations are a re-encoded proxy corpus, not a measured rejection rate for the actual RhoMath sampled-token distribution. Measure both tasks with the intended tokenizer, backend, generation settings, and a trained checkpoint as well as SFT; a clean 512-sample GSM8K corpus does not establish MATH or long-run safety.

If Tom approves dropping known unsegmentable primary responses, implement it as an explicit training policy:

- In `src/core/segmentation.py`, distinguish recognized sample-level failures (e.g. an exact character cut inside a token or a known upstream partition failure) from invalid IDs, gapped spans, broken context alignment, and other implementation-contract violations. Use a narrowly scoped typed error with a reason code for eligible sample failures; do not catch every `AssertionError` or `Exception` in the trainer.
- In the VinePPO branch of `PPOTrainer.train`, build one ordered list of accepted `(episode_id, episode, boundaries, scorer)` records before MC generation. Use that same list to pair estimates with episodes. Filtering only the estimator inputs while zipping estimates against the original `raw_episodes` would misassign advantages.
- Add a checkpointed configuration field such as `max_segmentation_drop_fraction`, default `0.0` (strict). If enabled explicitly, assert `dropped / generated <= ceiling` on each completed primary batch **before any MC work or optimizer step**, and assert at least one episode survives. Set a nonzero ceiling only after task-specific measurement and approval; do not infer one from 16 proxy examples.
- Record generated/accepted/dropped counts and reason counts, preserve IDs and tokens of rejected samples in a bounded diagnostic artifact, and keep all generation in the token/time budget. Label reward and length metrics as accepted-only where appropriate. No resampling until a batch happens to pass; that hides both cost and selection bias. Check partial/singleton minibatches against the existing advantage-whitening requirements.
- Restrict this policy to VinePPO primary segmentation. Do not remove arbitrary siblings from GRPO/RLOO groups or drop MC continuations from a value average. Dropping primary trajectories changes the training distribution and must be disclosed in provenance.
- Preserve RNG/config identity checks on resume. A new segmentation policy is an explicit experiment change, not a reason to permit arbitrary changed seeds. If an old run needs migration, record the parent checkpoint and policy transition rather than weakening general resume validation.

**Acceptance:** repeated answer markers preserve all sampled IDs and terminal EOS; noncanonical tokenizations and byte fallback retain exact token boundaries; accepted fixtures do not change; invalid spans still abort; approved sample rejection keeps IDs/scorers/estimates aligned; ceiling and all-rejected cases abort before updates; interrupted/resumed execution reproduces the same accepted data and updates. Keep the strict inside-token regression until the policy is approved. The parallel agent's pilot `7186263` also exposed a byte-prefix-retraction failure after the initial clean diagnostic; its subsequent fix needs its own regression, not suppression as a sample rejection.

### 2. vLLM retries, concurrency, and transport settings

**Fix:** retain the parallel agent's `continue` after requeueing a transport failure. A request gets at most `retries + 1` attempts, with an unchanged body/seed and original result index. After exhaustion, raise the actual transport error. Malformed responses, stale adapters, and incorrect sibling counts must still fail immediately.

Do **not** add slot retirement just because the review suggests it: bounded per-request retries already terminate a permanently failing call. Retiring the last consumer reintroduces the bug unless a separate replacement/failover mechanism is added. Replica failover is a different feature and is unnecessary for the current fail-the-job-on-worker-death contract.

The parallel change also explicitly disables keep-alive, removes the connection-pool cap, and bounds transport timeouts. Keep that separate from the queue correctness fix and validate it under realistic long MC requests: a five-minute read timeout may be too short under heavy server queueing. Prefer a documented finite per-replica connection bound matching the maximum drain-slot concurrency, or document why the drain slots alone are sufficient. Avoid adding another hidden queue below those slots.

Installed source inspection confirmed HTTPX's default pool limit of 100 and default keep-alive expiry of five seconds, and vLLM's default HTTP keep-alive of five seconds. Those defaults **do not prove the cause** of the observed socket error/stall; keep-alive concerns idle sockets, not a five-second limit on active generation. The queue bug is independently established. HTTP status failures are also distinct from `httpx.TransportError`; do not imply that the existing retries cover every HTTP error.

**Acceptance:** one request/one slot fails once then succeeds; a failure after all other slots exit still succeeds; exhaustion raises `httpx.ConnectError`, never the orphaned-results assertion; request bodies/seeds are unchanged; each logical result appears once in input order; malformed/stale responses are not retried. Run a bounded real-server repeated-evaluation soak with recorded exceptions, latency and worker stats. The tightened exception test and single-slot regression already exist in the parallel diff; reuse them.

### 3. Requeue-safe worker identities

**Fix in `experiments/training/launch.sh`:** use `worker_dir="$TRAINING_CACHE/workers/$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"`. Zero here means the documented first launch when Slurm omits the restart variable, not recovery from an invalid identity. Validate that a supplied restart count is a nonnegative integer. Keep exclusive `mkdir` for the attempt directory and the duplicate-rank assertion in `vllm_worker.py`; do not replace them with `exist_ok` or delete old identity files.

Record job ID and restart count in worker identities and training metadata, and have discovery validate the current attempt. Retain old directories/logs for diagnosis. Ensure the launcher waits for termination of its own old steps before a retry can reuse their ports; a changed directory alone cannot make an old server disappear. Continue failing loudly on a genuine port collision. Consider append-mode Slurm logs so requeues do not overwrite evidence.

**Acceptance in `tests/test_training_launchers.py`:** launch the same fake job with restart counts absent/0 and then 1, using fresh run outputs; the directories and learner arguments differ, old identities are untouched, and a duplicate launch of the same job/restart pair fails. Test dedicated and colocated layouts. Ordinary `train.slurm` still rejects an existing output; automatic training continuation belongs to the resumable wrapper, not this directory fix.

### 4. MATH grading cost and failures

**First, add a bounded exact-result cache in `src/core/verifier.py`:** convert the already-extracted prediction/reference lists to ordered tuples and cache the boolean returned by the existing isolated worker (for example, a bounded LRU of 8,192 entries). Recreate lists for the worker, whose grader mutates its input. Preserve order, multiplicity and current extraction/last-N semantics in the key; do not cache by response text alone or by only the displayed final answer. A process-local cache needs no persistence or extra dependency; if persisted later, its key must include the verifier policy and implementation identity. Cache only successfully parsed boolean results, never exceptions or timeouts.

Measure hit rate, misses, worker wall time, and total verification time on representative **training** primary/MC calls before promising a speedup. The review's half-second estimate and hour-per-iteration projection are not measurements. Inspection of the committed MATH 16-round records found 8,000 rows, 7,985 with nonempty extracted answers, and 4,447 distinct `(reported extracted answers, reference solution)` keys. That is only a proxy for actual worker-cache keys, but it does not justify claiming that a cache removes nearly all work.

**If misses remain material:** introduce a bounded ordered batch-verification path that runs independent existing single-use workers concurrently, bounded by allocated CPUs and memory. Deduplicate identical keys before launching; expand results back to original request order and still count every MC sibling. This removes serial wall-time multiplication without weakening per-answer timeout/CPU/address-space limits. Do not simply put the current worker in a persistent loop: its CPU limit is cumulative for the process, and a hung symbolic operation must remain killable. Persistent preloaded workers are a later optimization requiring per-request supervision and resource-limit tests.

**Failure policy:** keep timeout, worker death, missing dependencies and invalid IPC as errors, not incorrect answers. `verifier.py` and `vendor/vineppo/NOTICE.md` deliberately enforce this. Attach episode/prefix/sample identity and save bounded failing-input evidence before re-raising, then fix a reproduced parser defect narrowly. A genuine compute-budget-exceeded reward would be a new grading policy needing explicit approval, versioned provenance, and evaluation-comparability tests; do not implement `except Exception: return False` or silently omit failed MC siblings. Resume should not change seed just to evade the input.

Recover the missing `rloo-math.log` from the original run if available; committed `status.json` establishes exit 1 and its metrics file is empty, **not the cause**. After fixes, require a MATH smoke with primary grading, MC grading, checkpoint/resume and held-out evaluation before a long run.

**Acceptance:** cached/uncached grading agrees for numeric, symbolic, multi-answer, reordered, duplicated and question-dependent cases; cache hits avoid process launches; capacity is bounded; timeout/crash/protocol failures remain errors and are not cached; concurrent grading preserves order, kills/reaps failed workers, and respects allocation bounds. Extend `tests/core/test_verifier.py` rather than introducing another grader implementation.

### 5. FP32 trainable parameters

**Minimal safe fix:** reject `--finetuning full --dtype bfloat16` in `run.py` argument validation. For PPO, initially reject `--dtype bfloat16` regardless of actor finetuning, because the critic backbone and head are fully trainable in that dtype. Permit BF16 frozen bases with FP32 LoRA adapters for the critic-free algorithms; `apply_lora` already establishes that arrangement. Fail with a message directing users to FP32, rather than silently casting a loaded model and changing memory use.

Add named-parameter FP32 assertions to `PPOTrainer._validate_training` for actor and critic trainable parameters so direct callers cannot bypass the CLI guard. Apply the same contract at standalone actor/critic update boundaries if those remain supported. Validate loaded optimizer moment dtypes too. Update the runner's precision metadata and training README; existing unsafe checkpoints require an explicit migration/new run rather than implicit moment casting during resume.

If BF16 compute with full tuning is later required, implement mixed precision with FP32 parameter/optimizer storage as a separate measured feature. Similarly, supporting LoRA/BF16 actor plus an FP32 PPO critic requires an explicit separate critic-loading dtype and corrected provenance, not casting only the value head.

The spacing calculation near 0.02 supports the concern, but the review's 2%/15% fractions assume a weight distribution and step magnitude; they are not measurements of this model's optimizer updates. The current LoRA pilots already report FP32 trainable actor weights, so this finding does not establish that their learning signal is invalid. Casting loss/log-probabilities to FP32 does not fix BF16 parameter storage.

**Acceptance:** unsafe CLI combinations fail before allocation/model loading; direct BF16 actor/critic updates fail by parameter name; FP32 full/PPO and BF16-base FP32-LoRA training pass; a controlled small-gradient, low-learning-rate update changes FP32 parameters; checkpoint/resume preserves parameters, optimizer moments and updates. Keep frozen BF16 inference tests valid.

### 6. Checkpoint size and repeated hashing

**Stage 1: compact LoRA checkpoints with an explicit format version.** In `runtime/train.py`, save adapter tensors, optimizer, iteration/config/data identities, RNG state and provenance per checkpoint. Save or reference immutable frozen actor-base/reference state once, with content hashes—not merely model path or `config.json` hash. Assert that omitted parameters are frozen and account explicitly for persistent buffers. If the reference differs from the base, preserve it separately. Full tuning and the full PPO critic still need their complete changing states.

Update `runtime/checkpoint_actor.py` and training restore together: recreate from the verified local immutable base, apply the exact adapter configuration, load the saved adapter state with exact expected keys/shapes/dtypes, then restore optimizer/RNG for training. Missing bases, changed content, unsupported formats, or unexpected adapter keys must fail. Do not use unrestricted `strict=False` as a way to ignore mismatches. Update the full-state-only instruction in `core/lora.py` as part of this deliberate format change. Keep current experiments on the old reader until migration is tested; support old/new formats through explicit version dispatch, not a guessed fallback.

**Stage 2: remove redundant digest passes without removing integrity checks.** Establish one immutable checkpoint identity when publishing it; pass that identity through selection/evaluation. The loading process should independently verify bytes once and reuse that verified digest for its result instead of rehashing through several helpers. Retain a before/after mutation guard and immutable/no-overwrite publication contract; do not cache hashes indefinitely by path or trust a writable checksum sidecar alone. If immutability cannot be guaranteed, retain the protective checks rather than weakening correctness for speed.

Keep atomic publication and best-checkpoint retention. Shared base/reference blobs must not be pruned while any retained or frozen-selected checkpoint depends on them. For the observed 12,615,680 trainable LoRA parameters, adapters plus two FP32 Adam moments are about 151 MB before metadata, rather than 4.55 GB of repeated state; this is a size estimate, not a benchmark. The 43–65 s writes were observed on Skynet; inspected PACE H100 pilot metrics show approximately 4.7–5.2 s for the same 4.55 GB, so do not generalize the earlier timing to every filesystem.

**Acceptance:** identical actor logits and resumed next updates versus full checkpoints; exact optimizer/RNG recovery; changed/missing base fails; malformed adapter keys fail; interrupted writes leave the prior checkpoint intact; selected earlier winners and their base dependencies survive pruning/resume; evaluation does not load optimizer/reference tensors unnecessarily. Measure bytes, save/load time and number of full-file hash passes on the target filesystem.

### 7. Resume after frozen selection / final-test preemption

**Fix:** keep `CheckpointSelection._resume` rejecting frozen training. Instead teach `experiments/training/skynet/train_resumable.slurm` to distinguish training from final testing before choosing a training checkpoint:

1. If no frozen manifest exists, use the current highest-completed-checkpoint training-resume path.
2. If a frozen manifest exists, validate its identity and declared plan. For validation-only runs, report completion without training. For validation-test runs, use the **selected winner**, which may be older than the latest checkpoint, and invoke the existing `experiments.training.evaluate` test path with that frozen manifest and its exact protocol.
3. If a complete validated final-test result already exists for that selection/protocol, return success without rerunning. Otherwise create a fresh test-attempt directory and run all 16 rounds; retain partial failed attempts but do not combine them with a new run. This avoids introducing RNG/counter restoration for partial evaluation in the initial fix.
4. Atomically publish a small completion record linking selection hash, checkpoint hash, protocol and successful result. Handle preemption after `result.json` but before that record by validating and adopting the existing complete result. A conflicting frozen manifest or changed protocol must fail, not select whichever is newest.

Add an evaluation dispatch mode to the shared launcher so the wrapper can reuse its vLLM startup/cleanup and the restart-safe identities from item 3 without constructing a trainer or optimizer. Give test attempts unique paths and preserve parent-training/checkpoint-source provenance. Do not unfreeze selection, rerun validation to choose another winner, or select test attempts by accuracy. The standalone evaluator already supports frozen-manifest testing; the missing feature is orchestration, not a second evaluator.

**Acceptance:** preempt before first test output, midway through rounds, after final result but before completion record, and after completion. Across all cases: zero additional training updates, identical frozen-manifest bytes, original winner/protocol, no partial-result aggregation, and no duplicate completed test. Test both HF and vLLM launcher dispatch, plus corrupted/conflicting manifests. Update `experiments/training/EVALUATION.md`.

### 8. Accurate evaluation provenance

**Fix in `experiments/training/evaluate.py`:** retain separate identities for (a) the selection-owning training invocation, (b) the checkpoint's source invocation, and (c) this evaluation invocation. Set `evaluation_run.path` to `args.output`. Enter `WandBTracker` before building final provenance, then obtain the child identity from its own `args.output / "wandb.json"` when enabled; use `wandb: null` when disabled. Do not copy the parent's run ID. Populate the same identity in `provenance.json` and `result.json`. Selection resume may re-home `training_run`, but must preserve the historical `evaluation_run` and `checkpoint_source` of each candidate.

Replace the unconditional deviations list with a common list plus explicit HF/vLLM additions. HF describes post-generation stop trimming and its round RNG; vLLM describes server-side stopping, per-request hashed seeds, BF16 serving/adapter conversion, and its concurrent batching differences. Only state deviations actually supported by the generator/protocol, and retain the no-bitwise-parity caveat. Derive backend labels from the same source as `protocol_for` to avoid drift. Include vLLM/server software identity for remote evaluation instead of presenting only learner-library versions.

**Acceptance in `tests/test_checkpoint_evaluation.py` and selection/tracking tests:** parent and child IDs/paths differ with tracking enabled; disabled mode uses the evaluation output with null tracking; resumed candidates keep original evaluation identity; HF-only deviations are absent under vLLM and vice versa. Do not rewrite historical frozen manifests merely to correct old descriptive provenance; use a linked correction note if needed.

### 9. Test-only APIs, unused scaffolding, and stale documentation

Make this a separate cleanup after the parallel agent's functional changes stabilize. Test-only usage is not itself proof of useless code: preserve the behavioral checks while removing redundant production entry points.

- **`runtime.evaluate.main` / `EvaluatorConfig`:** remove the unused alternate artifact-writing wrapper and config. Keep `evaluate`, `EvaluationRecord`, `ClosableTextGenerator` and `inference_context`, which real evaluators use. Move relevant artifact/lifecycle/failure coverage into tests of `experiments.training.evaluate`; retain context cleanup tests directly. Remove imports used only by the deleted wrapper.
- **Single-episode VinePPO wrappers:** migrate callers/tests to `estimate_vineppo_batch` with a one-item list, preserving explicit single-episode expected-value tests. Then remove the module-level and trainer-level `estimate_vineppo` wrappers, unused imports and stale doc references. Keep tests that grouped algorithms never perform MC estimation, now guarding the batch method. Avoid creating a replacement production wrapper purely for tests.
- **`return_logprobs`:** recommend retaining this small diagnostic capability for now. Tests use it to independently compare generation transition scores with teacher-forced `token_logprobs`, including LoRA, EOS alignment and sampling-setting overrides. Deleting it and checking `token_logprobs` against itself would weaken validation. If reducing the public API is desired, first move the independent generation-score oracle into a test helper, then remove the flag, optional `Completion.logprobs`, and README claims together. Neither choice changes the trainer's need to cache normalized, unscaled old-policy scores.
- **`algorithms/base.py`:** remove unused `RolloutGenerator`, `StepSegmenter`, `AdvantageEstimator` and their now-unused imports. Keep `AlgorithmConfig`: all four algorithm configs and `runtime/config.py` use it. The review's three unused classes are not a justification to delete the whole module.
- **Stubs:** remove `scripts/codebase_hygiene/main.py`, `scripts/remote_sync/hf_bucket_sync.py`, `scripts/setup/vllm_installation.sh`, and empty `tests/algorithms/test_base.py` only after moving their intended future work/agent instructions into a task note. Remove stale directory/helper claims from `scripts/README.md` and `scripts/setup/README.md`; point setup documentation at the implemented installer instead. Do not replace stubs with larger unused abstractions.
- **Straightforward cleanup:** delete unused `itertools.pairwise` in `core/data.py`; describe `EpisodeWithAdvantages.advantages` as one value per response token rather than nonexistent step/value fields; update `src/README.md` to `core`, `algorithms`, `runtime`, and the application entry points under `experiments/training`. Update examples/docstrings affected by wrapper removal.

**Acceptance:** repository-wide reference search has no dangling removed symbols/paths; preserved single-item and batched MC expected results agree; generation/rescoring has an independent oracle; real evaluators retain cleanup and artifact tests; lint and focused tests pass before the full suite. Run tests on an allocated CPU node, not the login node.

### 10. Repository hygiene and publication

**Artifacts / size policy:** keep small summaries, configs, provenance and minimal regression fixtures in Git. Archive bulky `records.jsonl`, numerical evidence and raw scheduler logs in a durable private artifact store, with an in-repo manifest containing immutable location/version, SHA256, byte count, producer commit and retrieval instructions. Verify retrieval and hashes before removing anything from tracking; scratch storage alone is not durable archival. Preserve small test fixtures under `tests/fixtures` so normal tests need no remote downloads. No deletion, upload, or history rewrite is authorized by this proposal.

The inspected tracked tree has about 60.0 MB (57.3 MiB) under `artifacts`; 35 artifact files exceed 500 KiB. `uv.lock` is another oversized tracked file. Keep the generated lockfile, with an exact documented hook exemption if necessary; do not broadly exempt `artifacts/` or raise the limit to accommodate raw runs. Add CI enforcement of the same policy and ensure the large-file hook checks the intended files, including already tracked files when doing a repository audit. Untracking files prevents future growth but does not shrink old Git history; any history rewrite requires coordinated approval after archiving.

**Ignore policy:**

- Keep `.trash/` ignored and remove its currently tracked entries from the index in the eventual cleanup, without deleting the user's local trash. An ignore rule does not protect already tracked contents.
- Keep `bash.sh` tracked as a required shared helper and remove its contradictory ignore entry. Retain one definition of `timestamp`, and make setup scripts locate the helper relative to the repository/script location rather than relying on an arbitrary working directory. Do not make a required helper optional with a fallback.
- Explicitly decide which prose is shared. Recommended: track shared conventions and agent reports, remove their contradictory directory ignores, and separate private cluster inventories from general knowledge-base documents with narrowly targeted ignores. Do not broadly unignore and publish private notes. Preserve the existing agent-task exception and validate intended paths with `git check-ignore`/`git ls-files -ci`.

**Dependency:** remove `ruff-format` from the dev dependency group and regenerate `uv.lock` using the project's package manager without unrelated upgrades; keep `ruff`. The package metadata identifies `ruff-format` as Reflex's Python bindings, and no repository code imports `ruff_format`. Validate the normal lint/format commands and a locked environment afterward. Coordinate dependency edits with ongoing environment builds rather than syncing over an active run's environment.

**Before publication:** review the tracked tree **and reachable history** for personal absolute paths, account/QoS names, hostnames/IPs, node inventories, reservation details, credentials and copied logs. The tracked Skynet CSV inventories under `prose/knowledge-bases/gpu-cluster-info/skynet/` need an explicit disclosure decision. Move site-specific account/QoS choices to documented submission-time configuration where practical; sanitize public examples, not provenance silently in place. If raw provenance is hash-linked, retain the private original and publish a clearly labeled redacted derivative. Keep credentials in the existing ignored secret store/environment and scan locally without printing matched secrets. The original review's “no credentials found” is not a full-history guarantee. If a real credential is found, rotate/revoke it first; deleting its current file is insufficient.

**Acceptance:** a fresh checkout has the required helper and usable setup docs; shared docs are intentionally tracked, private/trash files are not; small tests work without archived payloads; archived evidence is retrievable and hash-verifiable; CI rejects a new oversized artifact while accepting the explicit lockfile exception; dependency resolution stays locked; publication review covers history as well as HEAD.

### Suggested order and validation status

1. Let the parallel agent finish/integrate items 1–2; preserve its exact-token policy and regression coverage. Inside-token dropping remains an approval gate.
2. Add the precision guard (5) before any full-BF16/PPO run, then requeue identities (3), frozen-test orchestration (7), and provenance fixes (8).
3. Before MATH training, add measured caching/diagnostics and obtain a successful MATH smoke (4), plus task-specific segmentation evidence (1).
4. Migrate compact checkpoints (6) only after resume/evaluation parity tests; do cleanup/publication work (9–10) separately from active pilot execution.

Validation for this **note** was source/call-site inspection against the reviewed commit and the evolving working tree, plus inspection of committed artifacts and the parallel agent's recorded cluster logs. No implementation test suite was run for these proposals. Two exploratory local checks (an in-process grading comparison and a torch update experiment) timed out at 120 seconds and produced no results; they do not validate or refute the proposed fixes. Small isolated grading checks completed, but they are not representative training benchmarks. Treat timing/throughput and algorithm-policy claims above with the stated qualifications. Future code validation should use allocated `ice-cpu`/GPU jobs, as required by the handoff.

## Deferred setup tasks (20261120-120541, HEAD `418fec2`)

- Add a repository check that rejects forbidden characters such as hyphens in Python filenames. Keep repository convention checks outside the behavioral test suite.
- Plan private durable storage for large checkpoints under PACE-ICE's scratch quota. The stub mentioned a Hugging Face bucket and an ignored `HF_TOKEN` environment variable. Archival requires a storage destination and retrieval verification.
- Maintain vLLM setup for the host OS, GPU and CUDA compatibility through the existing inference installer.

## Implementation record (20261120-123712, HEAD `418fec2`)

The PACE-ICE handoff was committed and pushed to `origin/tom/work` as `418fec2` before this follow-up. Its completion table gives the matched 50- and 200-iteration GSM8K pilots, queue times, and 16-round test results. Its timing sections cover the HF and vLLM benchmarks, interrupted attempts, and the GPU selection fixture. Paths below use the same PACE-ICE scratch `$OUT`.

- **MATH segmentation.** The approved first-character rule projects each split onto the earliest complete boundary at or after the character cut, using the original sampled token IDs. The crossing token receives the previous step's credit and stays in the prefix sent to MC. Coincident cuts collapse and a terminal EOS stays with the final step. An integration test checks the exact noncanonical MC prefix, the reconstructed response fed to the grader and the expanded PPO advantages. The GSM8K policy still requires exact cuts. Historical fixed-span proxy diagnostics replay their recorded strict policy.
- **Rollout transport and requeues.** The pushed handoff contains the last-slot HTTP retry fix, exact-token GSM8K handling and vLLM evaluation/replay evidence. Worker discovery now validates the Slurm job ID and restart count, and each attempt uses an exclusive directory. The launcher receives explicit process completion notifications and joins its steps during cleanup. Tests cover dedicated and colocated startup, early worker failure, restart identity and duplicate attempts.
- **FP32 optimization.** The CLI requires FP32 full tuning and PPO. BF16 frozen bases retain FP32 LoRA adapters. Direct updates and checkpoint restore validate trainable parameter precision and floating-point optimizer state. A small-step regression shows FP32 weights move, and checkpoint tests compare resumed weights, RNG and moments.
- **Checkpoint storage.** `--checkpoint-format compact-lora` opts in to adapter and optimizer checkpoints plus a shared SHA256-verified frozen actor/reference file. Legacy and current full checkpoints remain readable. Publication is atomic and exclusive. The evaluator independently verifies checkpoint bytes once, compares the expected digest and checks file identity through loading. Tests verify identical actor logits and resumed updates, including corrupt/missing dependency and key checks.
- **Frozen test resume and provenance.** The Skynet wrapper locks each run and dispatches to finalization when it finds a frozen selection. The finalizer validates the declared plan and the selected checkpoint, adopts a completed 16-round test or starts a fresh attempt, and publishes a completion record. A failed freeze publication leaves no visible frozen manifest. Tests cover multiple interruption points and both HF and vLLM dispatch. Evaluation records distinguish selection owner, checkpoint source and evaluation invocation with its own W&B identity and backend configuration.
- **Repository hygiene.** Removed unused scaffolding and moved its setup instructions above. Retained the HF generation log-probability oracle. Regenerated `uv.lock` with only the `ruff-format` dependency removed and no other package version/source changes. CI checks added/modified Git blobs against 500 KiB with an exact `uv.lock` exemption. An isolated Git fixture verified the size check. The local trash files were preserved while their index entries were removed. The history scan covered 567 reachable blobs and found personal paths and cluster details, with zero credential-like paths in the pattern scan. A durable private store and a disclosure decision for the tracked Skynet inventory are required before archival or public release.

### Review training wall clock

Slurm elapsed time measures allocation start through exit, including model and worker startup, validation and checkpoint I/O. Queue time is separate.

| Run | Job | Wall clock | Queue | Outcome |
|---|---:|---:|---:|---|
| GSM8K GRPO, two iterations, full checkpoints | 7198163 | 7:41 | 5:03 | Completed |
| GSM8K GRPO, two iterations, compact LoRA | 7198164 | 6:55 | 5:33 | Completed |
| MATH VinePPO, train and validate iteration 1 | 7198213 | 13:30 | 5:34 | Intentional exit 42 after held-out validation |
| MATH VinePPO, resume and validate iteration 2 | 7198214 | 10:31 | 1:36 after dependency | Completed |

The matched H100 checkpoint runs used the same GSM8K validation protocol and produced the same accuracy at both iterations. A full checkpoint was 4,552,389,818 bytes. A compact checkpoint was 151,888,836 bytes plus one shared blob of 4,400,347,131 bytes. Full write-and-publish times were 4.58 and 6.29 seconds. Compact times were 8.68 seconds with the first shared-blob write and 0.33 seconds for the subsequent checkpoint. The paired validation time was 34.64 and 37.55 seconds for full versus 30.00 and 32.26 seconds for compact. These are single-job measurements under `$OUT/checkpoint-review-{full,compact-lora}-01`.

The MATH canary used seed 42, one primary per iteration and three MC siblings per prefix. It resumed the compact iteration-1 checkpoint with the approved policy and performed one further update, grading 33 MC continuations at 11 prefixes. One-round accuracy on the same 500 held-out validation questions was 19.4% for SFT, 20.0% at iteration 1 and 19.6% at iteration 2. The frozen manifest still points to the iteration-1 checkpoint in the initial attempt. Allocated wall clock was 24:01 across both jobs and the first start to final completion took 25:37. The probe verifies execution and recovery at smoke scale. Raw outputs are in `$OUT/math-vineppo-strict-smoke-{01,02}` and `$OUT/review-slurm`.

### MATH sampled-token and grading evidence

- The pinned RhoMath MATH SFT and dataset are in the MATH training cache with `download_manifest.json`. HF preflight `7198001` took 5:47 and rejected 49/512 responses at temperature 1. Replaying the exact saved IDs preserved all 463 previously valid partitions and accepted all 42 inside-token failures, 35 stopped and seven truncated. The remaining 7/512 (1.37%) are genuine upstream splitter failures. Six fail the text reconstruction assertion and one fails coverage of the final character. The splitter still raises on these failures. No response was dropped or replaced. See `$OUT/segmentation-rate-math-01/records.jsonl` and `$OUT/math-grading-profile-01/{first-character-replay,upstream-failures}.json`. The original Skynet `rloo-math.log` was unavailable, and its recorded exit status does not identify the cause.
- The existing 8,192-entry process-local cache remains keyed on ordered exact prediction/reference tuples. The cold-cache profile graded 499 of 512 primary responses and recorded 76 cache hits and 421 subprocesses over 250.28 seconds. For 819 MC siblings across 91 prefixes, it graded 789 stops, recorded 448 hits and launched 339 subprocesses over 207.84 seconds. Worker means were 0.594 and 0.612 seconds. The sampled MC diagnostic used eight previously segmentable primary episodes to obtain exact prefixes. Its 10:29 generation job `7198130` and 7:51 serial grading job `7198137` are separate from training. See `$OUT/math-grading-profile-01/metrics.json`.
- `--math-grader-workers` explicitly enables ordered batch verification of each primary group or MC sibling group. It bounds overlap by allocated CPUs and memory and deduplicates exact inputs before starting the same isolated grader processes. Every sibling contributes to the result in its original order. Worker timeout and failure propagate with bounded example and sample context. The default remains one worker.
- On the same 230 stopped responses from 12 primary and 16 MC groups, cold-cache verification took 98.46 seconds serially and 31.75 seconds with four workers (3.10×). Both paths launched 166 subprocesses and returned identical ordered `VerificationResult` values. The grader caps and pre-verification deduplication are covered by the CPU suite. See `$OUT/math-grading-profile-01/batch-comparison.json`.

The production vLLM diagnostic used one colocated H100 per task, 64 fixed prompts and eight draws per prompt at temperature 1, seed 42, 2,048 total context and 1,024 new tokens. The SFT arm used a zero LoRA update. The trained arms used saved adapters.

| Task and actor | Slurm job | Strict accepts | Current accepts | Inside-token cases recovered | Remaining splitter assertions |
|---|---:|---:|---:|---:|---:|
| GSM8K SFT | 7198550 | 512/512 | 512/512 | | 0 |
| GSM8K VinePPO iteration 175 | 7198550 | 512/512 | 512/512 | | 0 |
| MATH SFT | 7198549 | 468/512 | 508/512 | 40 | 4 |
| MATH VinePPO iteration 1 | 7198549 | 476/512 | 510/512 | 34 | 2 |

The GSM8K diagnostic completed in 1:36 and the MATH diagnostic in 5:18. CPU job `7198743` replayed the frozen vLLM MATH IDs and established that all six remaining failures violate the upstream text reconstruction assertion. It also checked that all prior successes kept their exact partitions. Raw records and provenance are under `$OUT/segmentation-vllm-{gsm8k-7198550,math-7198549}`, with MATH comparisons in `policy-comparison.json`. The observed splitter failures still block long strict MATH runs. The two MATH arms characterize different checkpoints of a small diagnostic and give no estimate of the effect of training on failure rate.

### Verification (20261120-124720, HEAD `418fec2`)

The final allocated CPU suite (`7198754`) passed all 1,558 tests. Ruff, `git diff --check` and `uv lock --check --offline --python 3.13` passed. Final-test recovery was exercised on CPU with simulated interruption and both launcher modes. The MATH canary exercised actual GPU training, checkpoint restore and held-out validation. Its continuation used the current segmentation policy, and the SFT and trained segmentation diagnostics used production vLLM.

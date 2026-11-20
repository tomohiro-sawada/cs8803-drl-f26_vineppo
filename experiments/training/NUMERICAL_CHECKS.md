# Cluster numerical acceptance checks

These are required validation steps, **not results already obtained on A40,
H100, or H200**. Local tests use tiny CPU models (including CPU BF16); they do not
establish CUDA kernel accuracy, memory fit, throughput, or learning quality.
Repeat checks when changing GPU architecture, Torch/Transformers/PEFT versions,
attention backend, dtype, adapter configuration, or optimizer.

## 1. Establish an FP32 baseline before BF16

Use the same local model revision, tokenizer, task, fixed prompt/response IDs and
seed. Record GPU name, driver/CUDA, package versions, git commit/dirty state and
all hyperparameters. The runner writes most of this to `metadata.json`; preserve
artifact revision/checksums separately. It does not hash the model weights.

Run a short LoRA benchmark in FP32 first, then a separate BF16-base benchmark.
Use a new output directory for each. Comparing independently sampled trajectories
is useful for throughput but **not** for numerical error: fixed-data comparisons
below must reuse identical token IDs, masks, old/reference scores and advantages.
Do not compare independently sampled losses as if they ought to match.

Cluster submission examples are in [README.md](README.md). Start with one update
epoch, microbatch size 1, at least two responses per prompt, and a small prompt
batch. Keep a nonconstant-reward diagnostic batch so a policy update is observable.
Use production context lengths when measuring memory, after the small smoke test.

## 2. Check zero-initialized LoRA and policy scores

Before the first update:

- Actor and reference must start from identical SFT base weights. LoRA B is zero;
  A is randomized. In eval mode the actor should reproduce the unadapted reference
  on fixed inputs to the numerical tolerance of the selected backend/dtype.
- Check every trainable actor parameter is an FP32 LoRA A/B parameter; all base
  parameters are frozen. Verify optimizer ownership by parameter identity.
  The reference must have separate storage, no trainable parameters, and no grads.
- Recompute actor scores on the cached rollout before updating: valid-response
  ratios `exp(new_logp - old_logp)` should be one within measured precision.
  Compare unpadded and right-padded copies, including EOS and real pad-token IDs.
- Compare FP32 vs BF16-base selected-token log-probabilities on those same inputs.
  Record maximum/mean absolute differences and ratio differences, rather than
  only the final scalar loss. FP32 log-softmax does not undo BF16 forward error.

Exact base/reference tensor equality is appropriate for frozen-state checks.
For FP32 functional comparisons, `rtol=1e-5, atol=1e-6` is an initial diagnostic
threshold, not a promise for every GPU/kernel. Establish BF16 thresholds from the
FP32 comparison and the algorithm's clip width (default 0.2); differences that
materially change clipping/signs must be investigated, not waived with broad
tolerances. Do not silently cast the entire wrapped actor to BF16 after wrapping:
that would also cast the adapters and invalidate the intended optimizer precision.

## 3. Verify a real update, not merely finite execution

Snapshot frozen base/reference weights and adapter parameters. On a fixed batch:

- Loss, gradients, gradient norm, ratios and optimizer moments must be finite.
- At least one adapter B gradient/update must be nonzero for a batch with a
  nonzero policy gradient. **A can have zero gradient on the first step because
  B starts at zero**; requiring every adapter gradient to be nonzero is wrong.
- Frozen base and reference tensors must remain bitwise unchanged, including
  after multiple update epochs. Old log-probabilities and raw/group advantages
  must remain unchanged throughout those epochs.
- Inspect trainable parameter delta norms and the fraction of elements changed.
  Nonzero gradients with no effective update may indicate precision loss.
  Check AdamW `exp_avg`/`exp_avg_sq` for actor adapters are FP32.
- Rescore the fixed batch: at least some response probabilities should move.
  A constant-reward group can legitimately have no policy update (while KL may
  still act if actor and reference differ). Do not misdiagnose that as a bug.

Full tuning and PPO require FP32. PPO's critic remains full-parameter even
with a LoRA actor. Check its parameter deltas and moment dtype separately.

## 4. Grouped-algorithm checks

On hand-constructed reward groups, with KL coefficient zero when checking policy
terms in isolation:

- GRPO: `[0, 1]` produces approximately `[-1, 1]` (population std + epsilon).
- RLOO: `[0, 1]` produces `[-1, 1]`; `[0, 0, 1]` produces `[-0.5, -0.5, 1]`.
- Tied groups, including eight copies of `-0.1`, produce **exact zero** advantages
  and policy gradients. Adding a constant to a group must not change advantages
  except for ordinary input floating-point representation error.
- Groups must not mix across prompts, even after generation-budget grouping.
- GRPO uses equal-completion token means, not a global token mean. Test unequal
  response lengths. RLOO uses the exponential of the **sum** of token log-ratios,
  not their mean; inspect sequence-ratio overflow and clipping on long responses.
- No post-group whitening, critic estimation, or MC branching should occur.
  Benchmark `mc_tokens`, `mc_continuations`, and `mc_expansions` must be zero.
- Record how often groups have zero reward variance before planning long runs:
  all-correct/all-wrong groups carry no policy signal. The standard runner does
  not retain per-example reward groups; use a bounded diagnostic capture rather
  than assuming aggregate reward means establish useful within-group variance.

The sequence-ratio RLOO variant can be sensitive to long trajectories and repeated
updates. Start with one update epoch, then inspect ratio/clip metrics before
increasing reuse. A finite sample k3 penalty on old-policy tokens is not an exact
current-policy KL measurement.

## 5. Microbatch invariance and resume

For one fixed logical minibatch and identical initial state/RNG, compare
microbatch sizes 1 and 2. Compare summed loss components, gradients and parameter
updates with dtype-appropriate tolerances. Do **not** change logical minibatch
size at the same time: PPO/VinePPO whitening statistics and the optimizer schedule
would change legitimately. Repeat for grouped losses with unequal lengths.

Compare uninterrupted two-iteration execution with one iteration, checkpoint,
fresh process, load, and a second iteration. Match configuration/data and GPU/
software. Verify adapters, optimizer states, iteration number and next scores;
compare next sampled tokens where deterministic kernels permit exact replay.
Run deterministic fixed-input checks separately from stochastic trajectory replay.

Negative checks should reject changed LoRA alpha/rank/scaling, changed objective
(GRPO versus RLOO), or full/LoRA mismatch **before restoration**. Checkpoints are
full trainer states, not adapter-only exports. Test load on the intended cluster
filesystem, and budget transient space for a new checkpoint before pruning.

## 6. Only then measure throughput and learning

After warmup (including optimizer initialization), inspect allocated/reserved GPU
peaks and process memory reported by `nvidia-smi`. Keep operating headroom for
longer sampled outputs; microbatch size does not limit generation KV caches.
Check checkpoint I/O separately from compute. Initial three-iteration benchmarks
have only two measured iterations and are not robust runtime forecasts.

Run a short held-out before/after evaluation with a fixed protocol. Sweep LoRA
learning rate independently of full tuning; rank 16/alpha 32 and the 10x LR default
are starting hypotheses, not validated RhoMath optima. For a fair GRPO/RLOO/
VinePPO comparison, report both primary-rollout and MC-token budgets, tuning mode,
loss reduction, KL coefficient and observed KL scale. Equal iteration counts or
identical KL coefficients do not imply equal compute or regularization strength.

Do not launch a long run solely because CPU tests or the first GPU step passed.

# Exact sampled tokens vs re-tokenized text in VinePPO training

Written 20261119-231049 at commit `58d4c84`. Ours = this repo at
`58d4c846719d79368b88610c7e236fd5e3c0eef6`; upstream = McGill-NLP/VinePPO at
`9f716e68bb934e71da86c226c1699c36b5553faf`. Line numbers refer to those commits.

## The problem

A sampled token sequence `y` is not always the tokenizer's canonical encoding of its
own text: the model can emit `+`, `$` where the tokenizer would encode `+$` as one
token. Measured at T=1 on the RhoMath GSM8K SFT actor (job 7184578,
`segmentation-rate-01`): 12 of 512 responses rejected, 10 for this reason and 2 from an
unrelated step-splitter failure on truncated 1024-token responses.

## Our decision (approved by Tom, not yet implemented)

Train on the exact sampled token IDs. Obtain per-token character spans by decoding the
sampled tokens incrementally, instead of re-encoding the decoded text.

- **What changes:** [`src/core/segmentation.py` L93-L110](https://github.com/tomohiro-sawada/cs8803-drl-f26_vineppo/blob/58d4c846719d79368b88610c7e236fd5e3c0eef6/src/core/segmentation.py#L93-L110)
  currently decodes, re-encodes with offsets (L101-L107) and asserts the IDs round-trip
  (L108-L110). That assert is what rejects the 2.3%. The policy note at L63-L76 and the
  tests in `tests/core/test_segmentation.py` must be updated with it.
- **What stays:** a step boundary that falls inside a token still raises
  ([L205-L208](https://github.com/tomohiro-sawada/cs8803-drl-f26_vineppo/blob/58d4c846719d79368b88610c7e236fd5e3c0eef6/src/core/segmentation.py#L205-L208)).
  Upstream's first-character rule is not approved; no such case occurred in 512 samples.
- **Already exact elsewhere:** MC prefixes are sent as token IDs with no added special
  tokens ([`src/core/vllm_inference.py` L374, L386](https://github.com/tomohiro-sawada/cs8803-drl-f26_vineppo/blob/58d4c846719d79368b88610c7e236fd5e3c0eef6/src/core/vllm_inference.py#L374-L386)),
  and the replica must echo the prefix unchanged (L181).
- **Why:** the update is then taken on the sequence the policy actually sampled, and
  every sample is kept. It costs nothing extra over re-tokenizing.

## What upstream does

It keeps only text and re-tokenizes.

- **Re-tokenize:** `_tokenize_trajectory` encodes `query + response` with offsets
  ([`episode_generator_with_reward_function.py` L107-L158](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/episode_generators/episode_generator_with_reward_function.py#L107-L158), encoding at L122-L125).
- **Check is on text, not IDs:** it asserts the new IDs decode back to the same text
  ([L160-L177](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/episode_generators/episode_generator_with_reward_function.py#L160-L177)),
  which a canonical encoding always satisfies.
- **Token advantages:** each token takes the advantage of the step containing its first
  character ([`math_episode_generator_with_mc_advantages.py` L329-L369](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/episode_generators/math_episode_generator_with_mc_advantages.py#L329-L369), rule at L367-L369).
- **MC prefixes are text:** `query_text + response_text[:boundary]`
  ([L601-L616](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/episode_generators/math_episode_generator_with_mc_advantages.py#L601-L616)), so the server re-tokenizes them too.
- **Parse failures are dropped:** a response whose step splitting raises is logged,
  counted in `parse_failed`, and skipped
  ([L506-L516](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/episode_generators/math_episode_generator_with_mc_advantages.py#L506-L516)).

## Off-policy details

Let `d(y)` be the text of token sequence `y` and `c(t)` the canonical encoding of text `t`.
Upstream trains on `y' = c(d(y))`; we train on `y`.

- **The PPO ratio is unaffected in both.** Old log-probabilities are rescored by the
  actor on the training sequence before the update (ours:
  [`src/runtime/train.py` L685-L700](https://github.com/tomohiro-sawada/cs8803-drl-f26_vineppo/blob/58d4c846719d79368b88610c7e236fd5e3c0eef6/src/runtime/train.py#L685-L700);
  upstream: [`ppo_trainer.py` L1571-L1587](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/trainers/ppo_trainer.py#L1571-L1587)).
  The ratio starts at 1 either way.
- **The bias is in which sequence is trained on.** `y'` is drawn with the probability of
  its *text*, `sum over {y : d(y) = d(y')} of pi_old(y)`, but the loss uses
  `grad log pi(y')`, the probability of one tokenization. Whenever `y' != y` the gradient
  is on a path the policy did not sample. Not measured; it touches a few tokens in the
  affected ~2% of episodes.
- **Values follow the same choice.** Upstream's MC states are canonical prefixes, so its
  values and its update agree with each other, though not with the sampled path. Ours
  branch from the exact sampled prefix.
- **Dropping parse failures is a selection effect,** separate from tokenization: the
  batch is no longer a plain sample from the policy. Ours currently aborts instead.

Two other behaviour-vs-loss gaps are worth separating from the above:

- **Upstream samples with top-p 0.9 at temperature 0.6**
  ([`polIter_rho1bSft2_ppo_MATH.jsonnet`](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/configs/polIter_rho1bSft2_ppo_MATH.jsonnet)).
  Its loss divides logits by the temperature
  ([`ppo_trainer.py` L1812](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/trainers/ppo_trainer.py#L1812))
  and I found no nucleus term in that function, so the scored distribution is not the
  truncated one that was sampled. Ours samples at T=1 with no top-p
  ([`src/runtime/config.py` L80](https://github.com/tomohiro-sawada/cs8803-drl-f26_vineppo/blob/58d4c846719d79368b88610c7e236fd5e3c0eef6/src/runtime/config.py#L80)), so the two match.
- **Our vLLM replica holds the adapter in BF16; the learner's is FP32.** Measured
  per-token log-probability gap with an untrained adapter: 0.010 mean, 0.12 max
  (`sampler-acceptance-1p1-a40-01`). This is a real, small behaviour mismatch of ours.

## Ablations this sets up

| Ablation | A | B |
|---|---|---|
| Training sequence | exact sampled tokens (ours) | re-tokenized text (upstream) |
| Unsegmentable responses | abort (ours now) | drop and count (upstream) |
| Sampling | T=1, no top-p (ours) | T=0.6, top-p 0.9 (upstream) |
| Sampler precision | HF learner samples itself | BF16 vLLM replica |

The rejection rate should be remeasured under T=0.6 with top-p 0.9: I expect fewer
non-canonical samples there, but have not measured it.

Also confirmed while reading: upstream's Rho configs set the reference-KL coefficient to
1e-4 ([`refKl0.0001.jsonnet`](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/configs/trainers/refKl0.0001.jsonnet)),
matching ours, with loss type `control_variate`
([`klLoss.jsonnet`](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/configs/trainers/klLoss.jsonnet)).
I have not compared that loss type with our KL term.

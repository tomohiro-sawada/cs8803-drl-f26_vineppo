# Coding-agent handoff: audit and analyze the September 30 hindcast cycle

## 1. Objective and scientific context

We are building a sports forecasting dataset linking binary Polymarket markets to time-indexed sports commentary. The broader scientific goal is to study whether language models interpret new evidence and update beliefs usefully, potentially supporting later supervised or RLVR experiments. Agreement with market movements is an intermediate diagnostic; outcome forecasting quality and improvement over market probabilities are separate questions.

The latest bounded hindcast cycle is complete. Your immediate task is to make its interpretation defensible, identify implementation or data problems, and prepare a small audited evaluation set. Do not start by expanding model inference or training a policy.

Please work through the tasks below autonomously using existing artifacts. If evidence is missing, record the missing evidence and its consequence rather than manufacturing certification. Deliver implemented analysis, reproducible outputs, and a concise scientific interpretation—not just a proposed plan.

This handoff was written from `report.md`, `findings.md`, and `README(1).md`. The repository source, prompts, and raw response artifacts were not available to the author. Treat proposed schemas and command names below as interface requirements to adapt to the actual repository, not as descriptions of code already present.

## 2. Locate the existing implementation before editing

Read the repository's applicable AGENTS.md instructions and inspect the working tree. Preserve unrelated changes. Locate:

- `experiments/llm_vs_crowd/2026-09-30-190000_hindcast-qa-jev/`.
- Its `run.py`, README, findings, report, `cycle_*.sh`, and `cycle_launch.sh`.
- Parent run: `experiments/dataset/2026-09-30-010000_temporal-inspection/`.
- The implementation of `artifact_dir(__file__)`, request hashing, QA support, endpoint pairing, sampling, scoring, and control construction.
- Canonical games, markets, market ticks, commentary, and provider crosswalks, wherever they actually live.

The run's baseline was `2552a0dedffea7c877885314d6f18a6457867e05`; the findings say its implementation was first committed at `e42dd7d`. Do not reset the repository to these commits. Use them to understand provenance.

Existing workflow examples:

```bash
source bash.sh
uv run python experiments/llm_vs_crowd/2026-09-30-190000_hindcast-qa-jev/run.py preflight
uv run python experiments/llm_vs_crowd/2026-09-30-190000_hindcast-qa-jev/run.py report
```

Inspect the actual CLI before adding commands. Reuse existing modules and artifact conventions. Put new analysis in a versioned follow-up artifact directory, preserving the original run as an immutable reference. Avoid schema migrations unless they are necessary for the requested analysis.

## 3. Established facts from the completed cycle

### Coverage and execution

- Deterministic QA covered 10,018 games.
- Frozen chat workload: 150 windows × 2 information arms × 2 endpoints = 600 requests per model.
- Current-request-hash successes: Gemma 3 4B IT 412; Gemma 3 27B IT 587; Gemini 2.5 Pro 574.
- Gemma 4B: 13 invalid formats, 22 API failures, 2 oversized omissions, 151 unattempted requests. Repeated HTTP 429 circuits stopped its run.
- Gemma 27B: 11 invalid formats, no API failures, 2 oversized omissions.
- Gemini: 24 invalid formats, no API failures, 2 oversized omissions.
- Jev: 976 successful decisions from 1,000 windows, spanning 817 games; 24 API failures.
- No full-corpus neural extension was run. No durable cycle workers remain.
- Ledger: $16.50656033 spent, $0.366 reserved against a $50 cap. Earlier probes were outside this ledger, so this is not an exact all-in provider charge.

### Certification gaps

- Strict support: 0/10,018 games and 0/150 chat windows.
- All fixture identities remain `needs_review`.
- 9,883 games have retrospective event timestamps, not verified historical provider-publication timestamps; 135 have unknown timestamp semantics.
- Supported independent settlement labels: 0/150 windows. Brier score and outcome log loss are unavailable.
- Terminal market prices were deliberately not converted into labels.

These are separate evidence gaps. An unreviewed identity is not necessarily incorrect. A retrospective timestamp may locate a play correctly without proving when the text became available. A correct game mapping does not supply a settlement label or establish live availability.

### Preliminary model findings

Across provisional strata, Gemini evidence has update MAE 0.0440 and correlation 0.477; Gemini lagged-anchor has MAE 0.0541 and correlation 0.447. Both Gemma models have near-zero or negative pooled correlations. These aggregate numbers combine target-enriched strata and different success subsets.

For Gemini evidence, the more informative stratified comparison is:

| Stratum | Complete endpoint pairs | Model update MAE | Zero-update MAE | Sign agreement outside the 0.01 flat band |
| --- | ---: | ---: | ---: | ---: |
| Large change | 46 | 0.0706304 | 0.0883478 | 0.695652 |
| Random | 91 | 0.0315165 | 0.0199560 | 0.555556 |

The model's descriptive MAE improvement is approximately 20% on large-change windows, but its MAE is approximately 58% worse on random windows. Existing individual MAE bootstrap intervals are not confidence intervals for the model-minus-baseline difference.

The report already includes `pairwise_common_success`: Gemini–Gemma27B has 135 common windows per arm; comparisons involving Gemma4B have 92–96. Inspect and reuse this implementation. Do not present matched model comparison as entirely missing. Verify denominators and summarize its results before adding duplicate machinery.

### Jev findings

Core same-window direction agreement is 0.496, with a game-bootstrap 90% interval of 0.472–0.523. It includes increase/decrease/flat categories and is not binary accuracy. Mean semantic score is -0.0683; abstention rate is about 0.0031.

Lag-scan agreement/coverage: -120 s 0.408/516; -60 s 0.406/394; 0 s 0.483/925; +60 s 0.476/403; +120 s 0.418/514. Between-game shuffle: 0.349/915. Displaced controls: -600 s 0.392/503, +600 s 0.414/508.

These use different eligible subsets. They suggest association but do not establish a precise lag optimum. Jev's training cutoff is unverified. Minute archive resolution cannot establish second-level synchronization.

## 4. Priority A: verify definitions and construct an auditable analysis table

Before computing new metrics, write a short metric/prompt contract from the actual implementation:

1. Exact information supplied in EVIDENCE and LAGGED_ANCHOR at each endpoint.
2. Whether endpoint forecasts are independent calls, whether they share conversational context, and exactly which historical price each anchored endpoint receives.
3. Definition of the predicted event, including Yes/No orientation and team mapping.
4. Definition of market price: token, normalization if any, observation time, as-of lookup policy, and staleness threshold.
5. Endpoint sampling, large-change selection rule, random sampling population, overlap/deduplication, and random-fill policy.
6. Timing: event time, publication time, retrieval time, and actual market endpoint spacing.
7. Formula and denominator for every metric, including sign agreement and treatment of predicted flat/abstention.

Use notation consistently:

- `q_left`, `q_right`: model probabilities at the two endpoints.
- `m_left`, `m_right`: market values at those endpoints.
- `delta_q = q_right - q_left`.
- `delta_m = m_right - m_left`.
- Update error: `abs(delta_q - delta_m)`.
- Zero-update error: `abs(delta_m)`.
- Market reconstruction error: `abs(q_right - m_right)`.
- Anchor adjustment: `q_right - m_left`.

The last quantity is not interchangeable with `delta_q`. An anchored prompt does not guarantee `q_left == m_left`. Different price anchors at the endpoints can mechanically contribute to `delta_q`; document this before attributing arm differences to commentary interpretation.

Build or extend a normalized per-window analysis table with stable game/market/window keys, model, arm, stratum, hashes, endpoint statuses, probabilities, market values, timing/staleness, QA dimensions, and exclusion reasons. Include invalid/unattempted endpoints as statuses, not as fabricated forecasts. Ensure unique join keys and reject duplicate endpoint records instead of allowing many-to-many joins.

Acceptance: every reported metric is reproducible from this table, denominators reconcile with the original run, and a selected row can be traced to its exact prompt, response, and underlying observations.

## 5. Priority B: paired comparisons against persistence and between arms

### B1. Model versus zero update

For each complete window, calculate:

```text
model_error = abs(delta_q - delta_m)
zero_error = abs(delta_m)
paired_difference = model_error - zero_error
```

Negative differences favor the model. Report mean paired difference, model MAE, baseline MAE, relative MAE change when baseline MAE is nonzero, windows, games, and confidence intervals for the paired difference.

Use a game-cluster bootstrap: sample game IDs with replacement; retain all eligible rows from each sampled game and preserve multiplicity; compute the paired statistic in each replicate. Resampling the two methods independently is incorrect. Fixed seed; configurable replicates; default 5,000. Use 95% intervals for new comparisons and retain legacy 90% intervals under explicit labels. Report window-weighted estimates plus a game-balanced sensitivity estimate because repeated windows vary by game.

Separate random, large-change, and random-fill strata. Keep random-fill supplemental. Do not treat the pooled target-enriched sample as population performance. A game-cluster interval addresses within-game dependence, not selection bias or population representativeness.

### B2. Information-arm comparison

For each model, require all four endpoint forecasts to succeed for a matched arm comparison. Compare EVIDENCE versus LAGGED_ANCHOR errors on identical windows, with game-cluster paired intervals. Also compare each arm against persistence on that same subset.

Report the full-success estimate and matched-subset estimate distinctly. Different numerical MAEs on different subsets do not establish an arm effect. Interpret any difference in light of the prompt contract; anchoring may introduce information advantages or mechanical movement tracking.

### B3. Model comparisons and missingness

Reuse existing pairwise common-success analysis, adding paired error-difference intervals if absent. Do not require all models to succeed for every pairwise comparison. Optional all-model intersection is a sensitivity analysis, not the only table.

Report execution/format success by model, arm, stratum, sport, game, and request order where available. Distinguish conditional-on-success predictive performance from end-to-end reliability. Never silently score missing forecasts as zero updates. A persistence fallback can be evaluated as a separately named operational policy, with its own explicit definition.

Acceptance: a small synthetic example with unequal game sizes verifies cluster multiplicity, matched subsets, interval reproducibility, and the sign convention. Bootstrap code must not use an `isin` filter that loses repeated sampled games.

## 6. Priority C: Jev controls on shared eligible windows

### C1. Definitions and baselines

Verify how Jev probabilities become labels and how abstention is defined. The report defines semantic score as `P(increase) - P(decrease)` and uses market flat band 0.01. Do not assume the abstention rule from the prose without checking code.

Report confusion matrices, true/predicted class proportions, per-class recall, balanced accuracy where all required classes are present, majority-class accuracy, and always-flat accuracy. Report how abstentions enter each denominator. A 50% reference is inappropriate for three classes; 1/3 is also not a universal empirical chance baseline under class imbalance.

Check what `mean_signed_response` actually computes before interpreting it; its name alone is insufficient. Preserve its formula in the metric contract.

### C2. Matched lag and displaced comparisons

Persist per-window outputs for each shift. For each nonzero shift, compare it to zero shift on the intersection of windows eligible under both conditions. Optionally show the all-shifts intersection if sufficiently large. Report paired differences with game-cluster intervals, not just separate marginal intervals.

For each comparison, disclose eligibility loss, games, sports, market movement distribution, timestamp spacing, and exclusion reasons. Shared eligibility solves one denominator problem but does not make the retained sample representative.

Retain the existing as-of rules unless a separate versioned sensitivity is justified: endpoints no more than 90 seconds old, actual spacing 45–90 seconds, same observed event segment; gaps over 30 minutes split coverage heuristically. Do not use future prices to fill gaps. Positive lag means market later than commentary. Do not automatically apply a lag correction selected from these exploratory results.

### C3. Shuffle comparison

Compare original and shuffled outputs on the same eligible target windows. Audit donor matching, target orientation, within-game exclusions, and reproducibility. Preserve the original shuffle as a fixed reference. Add a configurable repeated-permutation sensitivity, preferably 100 permutations when compute permits, reusing cached semantic outputs if mathematically equivalent and valid. If new neural calls would be required, document that and leave the expanded permutation run pending.

Distinguish bootstrap uncertainty over games from variation across randomized shuffles. Neither provides independent ground truth or certification of causal alignment.

Acceptance: synthetic controls catch orientation flips, incorrect shift signs, future-price lookups, and unequal subset comparisons. A genuine lag effect should survive matched evaluation before motivating a frozen correction assessed on held-out games.

## 7. Priority D: inspect responses and create a small audit set

Create a deterministic audit packet from the completed pilot. Target 30–50 unique windows, with a reproducible random component and clearly labeled diagnostic oversampling. Cover sports, random/large-change strata, large model errors, disagreement between arms, flat predictions, and invalid formats. Selection using observed errors is acceptable for debugging; it is not an unbiased evaluation sample.

Each case should show game/market IDs, target definition, team orientation, provider evidence, both timestamped commentary prefixes, relevant structured score/state, as-of market observations with staleness, exact prompts/responses, parsed forecasts, model/request hashes, and QA reasons. HTML or Markdown is sufficient; reuse audit.html components if present.

Audit dimensions must remain separate:

| Dimension | Required evidence |
| --- | --- |
| Fixture identity | Teams, competition, scheduled date/time, stable provider IDs, source provenance |
| Market target | Actual market wording/rules and Yes/No token mapping; distinguish winner from spread/total/other targets |
| Event-time alignment | Timestamp semantics, timezone, commentary ordering and game-clock conversion evidence |
| Historical availability | Publication/delivery timestamps or archived evidence showing text was available by the forecast cutoff |
| Settlement label | Independent official result or documented market resolution, including void/cancelled/ambiguous states |

Produce a review template containing evidence references, status, reviewer, review date, and reason. Programmatic evidence may resolve factual fields; subjective/manual decisions without sufficient evidence remain pending. Never bulk-promote `needs_review`, label absent publication data as verified, or infer certainty from a plausible text match.

For response debugging, check reversed teams, misunderstood scores, incorrect elapsed time, target confusion, identical/flat endpoint outputs, invented priors, endpoint inconsistency, parsing errors, and truncation. The current negative Gemma correlations could reflect model limitations or implementation/data problems. Do not assume either explanation until inspection supports it.

Acceptance: every audit judgment has traceable evidence; unresolved dimensions remain explicit; any discovered implementation correction produces a new versioned analysis and an explanation of which prior metrics changed.

## 8. Priority E: separate retrospective, availability-certified, and outcome evaluation

Use orthogonal QA columns and derived cohorts rather than one ambiguous pass flag. At minimum distinguish identity status, target status, event-time status, publication-availability status, settlement status, and inference completeness. If already present, reuse them.

A useful retrospective diagnostic cohort may have verified identities and event alignment but unknown publication availability. Label it accordingly. Do not discard all such data solely because strict live support is zero, and do not describe it as live-certified.

If supported settlement evidence exists in accessible source artifacts, ingest it with provenance. Otherwise produce an explicit acquisition specification. Define the label for the actual binary market; do not assume every market is straight game winner. Cancelled/void markets need a declared exclusion policy. Soccer A-wins versus not-A is valid: not-A includes draws and B wins; No-on-A is not B-win.

Once labels exist, report model, contemporaneous market, lagged market, and declared baselines at matching cutoffs using Brier score and clipped log loss, with game-cluster paired differences. Keep these separate from market-update imitation metrics. Market prices are comparison predictors, not outcome labels. Do not infer labels from near-terminal prices.

Availability-certified data alone do not establish a leakage-free model: weight revision and training cutoff are additional provenance requirements. Pinned OpenRouter IDs are not verified weight revisions; Jev has no verified cutoff in this run.

## 9. Optional follow-up: shrinkage and structured baselines

Only after the audits and paired analysis, prepare a small experiment specification for reducing noisy updates. A shrinkage predictor `alpha * delta_q` tests whether the model contains useful directional information but miscalibrates magnitude. If fitting alpha, use training games and evaluate on held-out games; never fit on the reported test windows. Respect any probability constraints if reconstructing endpoint probabilities.

A structured-state baseline can test whether commentary contributes beyond sport, score, time remaining, possession, and lagged market information. Match information cutoffs and split by game, with temporal splits when feasible. These are separate hypotheses from whether commentary predicts market movement.

Do not run RLVR or a broad model sweep during this handoff. Market-update rewards could train imitation of market dynamics rather than improve outcome forecasting. The eventual reward target must follow the scientific question and the available labels.

## 10. Deliverables and verification

Produce:

1. Metric/prompt contract and source/artifact inventory.
2. Normalized per-window analysis output with provenance and exclusions.
3. Stratified model-versus-persistence comparisons with paired game-cluster intervals.
4. Matched arm and existing common-success model comparisons, plus reliability tables.
5. Jev class baselines, matched controls, and explicit shift conventions.
6. Deterministic audit packet and evidence-based QA review template.
7. Settlement/availability acquisition specification for unresolved evidence.
8. Updated findings describing supported claims, unsupported claims, sample selection, and the next decision.
9. Reproducible commands and focused tests for the new analysis.

Use offline analysis by default. Do not automatically resume old inference, expand the neural workload, or consume the remaining spend cap. If necessary data are unavailable, complete everything independent of them and state the precise blocker. Preserve raw archives and original outputs. Hash configs, manifests, and inputs; record code revision, seeds, schema versions, and dirty-tree provenance under existing conventions.

Run focused tests, lint, and end-to-end report generation. Important tests include endpoint key uniqueness, missing forecast handling, flat-band boundaries, orientation, bootstrap multiplicity, join cardinality, shared control eligibility, shift sign, and as-of lookup. Exact regression numbers should use tolerances and frozen fixtures, not network calls.

Final scientific report must answer:

- Does Gemini beat persistence on random windows? On large-change windows? With what uncertainty?
- Do apparent arm differences survive identical-window comparisons, and what information could explain them?
- Are Gemma failures attributable to identifiable implementation/data problems or unresolved model behavior?
- Does Jev's association survive shared-window controls, and how much lag precision is justified?
- What fraction is identity-verified, event-aligned, availability-certified, and outcome-labeled?
- What can we claim about evidence interpretation, market association, and outcome forecasting separately?

Acceptable conclusions include negative or inconclusive results. The successful outcome is a trustworthy experiment and a clear next decision, not a favorable benchmark number.

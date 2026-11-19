# Two-hour handoff: explain and verify the completed baseline

Prepared October 2, 2026, following the completed Inkling Small run.

## Objective and scope

Use the next approximately two hours to turn the existing results into a trustworthy, inspectable scientific finding. Prioritize cached analysis over new inference. The additional baseline now exists; the highest-return work is matched comparisons, diagnosing its errors, and resolving known measurement ambiguities.

Start a hard 120-minute work budget when you begin. Reserve the final 20 minutes for integration, focused verification, and a written review. Do not start another paid cohort, a broad model search, a prompt sweep, ingestion/backfill, or a GUI rewrite. This handoff supersedes the previous four-hour inference instructions. New API spending is not needed for these tasks.

Read applicable AGENTS.md and the latest agent-produced review before changing code. Some earlier tasks may already be complete: verify their actual artifacts and reuse them. Do not repeat work because this handoff mentions it. Report blocked or unfinished items explicitly.

## Evidence to establish at startup

Locate the actual run directory containing shard `inkling_small_json_instruction`, model `thinkingmachines/inkling-small`, and the latest `coverage.json`. Do not guess its directory name. Record its request/config/cohort/parser identities and parent runs in your new analysis manifest.

Reported latest results, to reproduce from cached endpoint records:

- 596 planned logical endpoints: 556 strict successes, 37 invalid-format responses, 3 oversized inputs not sent. All endpoint statuses resolved.
- Strict complete window-arm pairs: 261 of 298; probability-usable pairs: 274. These are window-arm pairs, not 261 distinct windows.
- Random EVIDENCE: 92 windows / 50 games; model update MAE 0.0416304, persistence 0.0194130, paired difference +0.0222174, 95% game-cluster interval [0.0127415, 0.0320731].
- Random LAGGED_ANCHOR: 89 windows / 49 games; model MAE 0.0397416, persistence 0.0205169, difference +0.0192247, interval [0.0118505, 0.0265379].
- Large-change EVIDENCE: 38 windows; difference −0.0026316, interval [−0.0160526, 0.0113224]. Large-change LAGGED_ANCHOR: 39 windows; difference +0.0071795, interval [−0.0125641, 0.0300000]. Neither supports a clear advantage.
- Probability-usable random results also lose to persistence: EVIDENCE n=96, difference +0.0226458 [0.0135414, 0.0320431]; LAGGED_ANCHOR n=91, difference +0.0204505 [0.0130000, 0.0278153].
- Existing intervals use 5,000 game-cluster draws, seed 20260930. Preserve compatible conventions.

Previous artifact roots include:

```
artifacts/experiments/llm_vs_crowd/2026-10-01-020000_overnight-hindcast/
artifacts/experiments/llm_vs_crowd/2026-10-02-001200_hindcast-followup/
```

Find the latest `root_cause.md`, `eligibility_manifest.json`, `chronology_scope.json`, `jev_stage_table.md`, request/forecast/attempt records, and prior cached comparison outputs. Use repository artifact resolvers where available. All new analysis goes into a new directory referencing immutable parents. No edits to archived requests, responses, timestamps, or frozen cohorts.

## Scientific contract

Let Δp = p_right − p_left and Δm = m_right − m_left. Model error is |Δp − Δm|; persistence error is |Δm|. A positive paired error difference means the model is worse.

This measures retrospective association with observed market updates. It is not outcome forecasting accuracy or prediction of subsequent prices. EVIDENCE and LAGGED_ANCHOR both contain structured state and commentary; the latter supplies the same stale m_left at both endpoints. Their contrast does not identify commentary's incremental value.

The frozen pilot has 150 windows from 50 games. The applicable-check-screened cohort excludes one contradictory Hawaii window, leaving 149. Unsupported checks remain unknown; the 149 are not certified historically available or semantically correct.

## Execution and parallel ownership

Parallel agents are authorized for this handoff if available. Use at most three bounded workers plus a coordinator. Agree on exclusive file ownership before edits. Existing shared parser/bootstrap/GUI modules have one owner; other workers consume their outputs. If workers are unavailable, perform A, then B, then the highest-priority portion of C.

| Owner | Deliverable | Suggested working time |
|---|---|---|
| A | Reproduced metrics and matched model comparison | Minutes 10–75 |
| B | Matched arm contrast and compact error review | Minutes 10–80 |
| C | Verify prior audit completion; Jev zero-shift trace and Hawaii sensitivity | Minutes 10–80 |
| Coordinator | Resolve artifacts/interfaces, integrate, verify, write review | Throughout; final 20 minutes reserved |

At minute 10, freeze interfaces: normalized endpoint table, window metadata, eligibility manifest, and validity modes. At minute 60, workers report concrete findings/blockers. At minute 90, stop expanding scope. Prefer a completed small analysis over unfinished infrastructure.

## A. Reproduce results and compare models on identical windows

### A1. Reproduction and coverage

Run the existing analysis against the actual Inkling cached records. Confirm the counts and estimates above within numeric tolerance. If they differ, first explain source/version/eligibility differences rather than editing expected values.

Produce coverage by model, arm, stratum, and validity rule: planned endpoints, resolved statuses, complete pairs, distinct windows, and games. Show missing-pair reasons derived from their endpoints. Preserve both historical strict validity and the existing versioned probability-usable sensitivity. Do not invent another extraction heuristic this cycle.

Audit any disagreement between these validity modes using a few exact examples. A citation failure with a usable probability remains a citation failure. Truncation, ambiguous probabilities, or absent fields must not become default predictions.

### A2. Common-success comparisons

Compare Inkling Small, Gemini, and GPT-OSS using the 149-window manifest and exact shared complete pairs. Include Gemma only if readily available and compatible; do not delay the primary comparison.

For each arm and stratum, compute:

1. Each model versus persistence on its own eligible complete pairs, clearly marked as different cohorts.
2. Pairwise model comparisons on the intersection for that pair of models.
3. A three-model intersection table if enough pairs exist; report its coverage explicitly.

Join by stable window identity and verify market, endpoint timestamps, target orientation, and stratum. Never join by file position or use a request hash as the cross-model window key. Detect duplicate keys instead of silently dropping them. Persistence must use the same rows as each comparison.

Report windows/games, model MAE, same-cohort persistence MAE, paired error difference, and 95% game-cluster interval. Reuse existing bootstrap code; sample games with replacement and preserve sampled multiplicity and all their windows. Include the established game-balanced sensitivity. Keep random, target-enriched large-change, and random-fill separate.

Inventory material prompt/config differences: instruction version, JSON schema or instruction mode, evidence compression, structured state, citations, model/provider, output cap, and parser version. If prompts differ, describe the result as a comparison of implemented model configurations. Common-success filtering does not remove prompt confounding or failure-selection bias.

Deliver `matched_comparisons.csv`, `coverage_by_model.csv`, and a concise methods note. No model ranking from unmatched pooled MAEs.

## B. Explain the negative result and measure the anchor contrast

### B1. Matched anchor analysis

Within each model, retain only windows with complete endpoints in both arms under the same validity rule. For each stratum, calculate the paired arm difference:

```
anchor effect on error = error_LAGGED_ANCHOR − error_EVIDENCE
```

Negative favors the anchor. Report sample size and game-cluster uncertainty. Do not infer improvement from the separate 92-window and 89-window random MAEs.

Also report median and p90 |Δp| and signed mean Δp for each arm on those same windows. Smaller updates alone are not better forecasts. Identify whether the prompt's evidence/state differs between arms due to token budgeting or compression. Add an evidence-matched sensitivity only where deterministic saved evidence equality can be established. Do not use a new LLM judgment to declare equivalence.

### B2. Error concentration and manual review packet

For Inkling random EVIDENCE, quantify median/p90/max absolute update, median/p90 error, and the largest five windows' share of total absolute error. Add a deterministic diagnostic table removing the top five only as a sensitivity, with its selection rule visible. Do not replace the primary result or present the selected subset as unbiased.

Create at most eight review cases: top five error windows from distinct games where feasible, two near-median error cases selected deterministically, and one useful validity-disagreement case. Preserve actual IDs and selection reasons.

For each case include market target/team orientation, sport, endpoint timestamps, state/score/period, exact archived prompts and raw responses or links, p_left/p_right, m_left/m_right, Δp/Δm, and validation status. Inspect the records to label only directly supported issues: target reversal, period/score misunderstanding, inconsistent input, citation failure, or no evident explanation. Mark ambiguous interpretations as uncertain. Do not attribute errors to a mechanism based on probability magnitude alone.

Use the existing GUI's exact-prompt view and shared game/window selection. Supply working resolved deep links if supported. If prompt/raw response links already suffice, do not add UI infrastructure. Fix only a demonstrated small selection/display defect with a focused check. Screenshots are useful if readily available; do not block the analysis on screenshot tooling.

Deliver `anchor_comparisons.csv`, `error_diagnostics.csv`, and `review_cases.md` with an appropriately bounded interpretation.

## C. Close the most consequential existing audit gaps

### C1. Hawaii exclusion sensitivity

First look for an already completed analysis. If absent, compute original 150-window versus screened 149-window results for GPT-OSS and Gemini, using cached responses and identical validity rules. Show the change in random-window MAE and their common-success comparison. Record which model/arm/stratum rows actually contain Hawaii. Do not assume every table loses one row.

Known contradictory window: game 50016141, 2025-11-02T06:28:06Z–06:29:07Z; request hash supplied by the user:

```
70ea6891dcb21e6dc520f28471d3eefac2e2a945860f627b8dc838875291004c
```

Snapshot: period 3, 1:57, San Jose State leads 38–21. UTC-ordered evidence includes period-4 event `401760403855` at 06:18:09 and end-fourth event `401760403873` at 06:25:09, then period-3 event `401760403639` at 06:27:22. Sorting by sporting period would conceal a contradiction without repairing information inclusion or historical availability.

The recovered full stream reportedly has 13 period regressions clustered from 06:03:49Z. Recovery copies CFBD `wallclock` into UTC timestamp fields; rendering preserves UTC order. This locates the contradiction upstream of rendering but does not settle its underlying cause. Record any newer provenance evidence; do not launch broad provider research or reconstruct times.

### C2. Jev zero-shift replay

The previous report has 976 successful targets but only 925 valid zero-shift market comparisons; 51 are outside supported segments. The five-shift intersection has only 67 windows. First check whether a complete replay explanation now exists. If so, verify representative records and summarize it; do not redo it.

If unresolved, produce a cached trace for all 51 exclusions if straightforward, plus three to five detailed examples. Join original and replay window/market IDs, timestamps, source series/version, selected observation IDs/times/prices, segment boundaries, and failed predicate. Check retained zero-shift comparisons for equality too.

Classify supported explanations: source/version difference, intentional stricter support rule, endpoint selection discrepancy, missing archived input, or implementation defect. Zero shift should reproduce the original comparison when inputs and rules are identical. A stricter control-support rule does not automatically invalidate the original result.

If a bug is demonstrated and small, implement a focused fix, add a meaningful reproducer, and regenerate a versioned cached table. Otherwise provide the exact failing trace and next implementation step. Do not relax support checks merely to recover coverage; do not interpolate prices or tune shift values. First-failure exclusion counts depend on shift order and are not independent prevalence estimates.

Do not spend this cycle estimating an optimum lag or rerunning inference. Deliver `hawaii_sensitivity.csv` and `jev_zero_shift_replay.csv` or links to verified existing equivalents, with limitations.

## Engineering, scripts, and verification

Reuse coherent existing modules. Keep joins/eligibility explicit, arithmetic separate from rendering, and provenance attached to every derived table. Avoid layers of thin wrappers, duplicated bootstrap implementations, broad refactors, and unexplained booleans. Comments should explain invariants and decisions.

Long cached jobs should use one self-contained resumable script, run inside tmux. It must accept explicit source/output paths and deadline, write progress and structured stop status, avoid concurrent writers, and generate summaries automatically. Document actual launch, status, and resume commands. No shell loop with unconditional success after failures. Small analyses need not be artificially put in tmux.

Focused verification:

- Reproduce reported Inkling coverage and primary metrics.
- Check one manually computed window against table arithmetic.
- Assert uniqueness and exact metadata alignment before paired joins.
- Verify persistence and both models use identical comparison rows.
- Verify strict and probability-usable analyses retain their own provenance.
- Test any actual eligibility/replay/GUI bug fix with its failing example.
- Suppress inferential intervals for the one- or two-window random-fill results; show descriptive values and counts.

Run relevant repository checks for changed code once integration is stable. Record actual commands/results. Do not spend the time budget writing tests that merely mirror report formatting. Commit coherent changes if repository convention allows; no push/deploy is needed.

## Final review, due by minute 120

Write one short `review.md` linking tables, case packet, parent artifacts, manifest, commits, and checks. Answer:

1. Does Inkling's negative random-window finding reproduce under both validity rules?
2. What do common-success comparisons say, and which configuration differences limit interpretation?
3. Does the matched stale-anchor contrast improve error, change update magnitude, or remain inconclusive?
4. Is error spread across windows or concentrated in a few extremes? Which inspected issues are directly supported?
5. Does Hawaii exclusion materially change prior conclusions?
6. Why do the 51 Jev zero-shift exclusions occur, or what exact evidence is still missing?

Use probabilities or percentage points consistently; identify units and the sign convention. Include denominators and uncertainty. Separate completed work, verified prior work, and blocked work. Choose one next experiment only after these findings; do not launch it in this cycle.

The current defensible finding is that Inkling Small loses to persistence on random windows under this retrospective protocol, with a consistent extraction sensitivity. Additional models extend a protocol-specific finding; they do not establish a universal model limitation, explain a causal mechanism, or certify the underlying chronology.

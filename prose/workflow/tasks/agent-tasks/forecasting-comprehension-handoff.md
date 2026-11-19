# Forecasting: controlled state-comprehension experiment

## Objective and scope

Implement and run a small, predeclared comparison of the archived presentation against an explicit sport-state presentation. Determine whether naming teams, stating target-team score margins, and explaining verified clock/period semantics reduces factual interpretation errors. Measure market-update error separately. Preserve every previous experiment.

Proceed autonomously with offline implementation and audit. Paid execution remains within the previously authorized $10 aggregate ceiling, inclusive of all earlier runs, retries, and pending reservations. Reconcile the ledger first; neither $3.82 nor any older subtotal is the current balance. Do not add models, expand to a full cohort, or increase the ceiling. If no funded diagnostic fits, deliver the frozen requests and runnable script, mark execution budget-blocked, and finish offline work.

This guide uses the supplied conversation's reported findings as starting evidence. Independently verify them against current artifacts before implementing case-specific judgments. Repository paths and APIs below are proposed contracts: map them to the actual repository rather than inventing existing modules.

## Context that must survive the handoff

The existing experiment independently elicits probabilities at two endpoints, then compares their difference against the contemporaneous market-price difference. Persistence predicts zero price change. This measures retrospective market-update agreement; it does not measure outcome calibration.

Several configurations lose to persistence on random windows. The full-Inkling probability-usable analysis reportedly has EVIDENCE MAE 5.51 percentage points versus 1.88 for persistence, and LAGGED_ANCHOR 5.94 versus 1.79. These are historical results, not targets to optimize against.

Confirmed examples motivate this diagnostic:

- Kings–Jazz, game `20023251`, window `2026-02-12T03:40:20Z–03:41:18Z`, EVIDENCE: left reasoning reads Jazz 85, Kings 59 but concludes Kings lead by 26 and emits 0.99; right emits 0.01. The emitted and normalized probabilities agree. Verify target identity and the original prompt before using this case.
- Rams–Cardinals: a response reverses the leading team, interprets Q2 remaining time as near game end, and infers no remaining timeouts from one observed timeout.
- Florida State–Alabama: anchored full-Inkling responses confuse Q2 with game end and reverse the leading team; EVIDENCE responses interpret the period correctly.
- Ciurnelli–Kisimov: the old reducer carried a first-set 3–6 score into a scoreless second-set-start event. The repair separates completed-set information from unknown current-set information. The reported repair changes interpretation but does not improve this window's update error.

One tennis window is a confirmed defect; 36 baseball/handball windows with unresolved score semantics are an uncertainty category. Hawaii remains quarantined for chronology contradictions. Do not silently repair wallclock chronology or certify unknown sport/provider semantics.

## Work allocation and dependencies

The coordinator may assign these work packages to parallel coding agents. Each worker owns its modules and outputs; only the coordinator modifies shared CLI registration, dependencies, and final report. Agree on contracts before editing. Use isolated worktrees if available; otherwise enforce exclusive file ownership. No concurrent writes to an attempts file or ledger.

| Work package | Owns | Can start when | Completion evidence |
|---|---|---|---|
| A: state and renderer | reducer review, canonical facts, renderer, tests | immediately | verified fixtures, parity check, renderer diff |
| B: roster and factual rubric | case manifest, annotations, scoring | immediately | frozen roster and gold facts with provenance |
| C: execution | resumable runner, budget/deadline integration | immediately using fixture requests | dry run, stop/resume checks, actual run status |
| D: analysis | joins, factual summaries, paired diagnostic tables | immediately using fixture outputs | tested analysis and final tables |
| Coordinator | integration, freeze, launch, review | continuously | immutable manifest, decisions, final report |

A and B agree on fact identifiers. C consumes frozen request bytes; it must not construct sport states. D consumes normalized responses and annotations; it must not change eligibility. If only one worker is available, implement A → B → C → D. Do not build a new orchestration framework.

## First 20 minutes: inspect and reconcile

1. Read applicable AGENTS.md. Record git HEAD, status, environment/tool versions, and relevant existing uncommitted changes. Never clean or overwrite another worker's edits.
2. Locate the completed full-Inkling manifest, original requests/responses, `extreme_updates.json`, `rerun_summary.json`, repair review, `state_repair.py`, `repaired_rerun.py`, and the actual `sport_state_v1` reducer/tests. Use `rg --files` and targeted `rg` searches.
3. Verify the examples above and identify exact window, endpoint, arm, target entity, source artifact, and request hashes. Missing artifacts produce an explicit blocker, not reconstructed quotes.
4. Reconcile actual spend and outstanding reservations across previous shards. Record available dollars and estimate the proposed matrix using verified model pricing and a conservative token allowance. If provider capabilities/pricing are needed, check current authoritative documentation.
5. Fix the discrepancy where impact previews replace `state` without synchronizing `missing_state_fields`, while reruns update both. Route both through one canonical builder. Assert byte-identical rendered requests for equal inputs/configuration.
6. Inspect reducer code and tests. Report whether aliases, score scope, field provenance, cumulative-score preservation, and unknown semantics are covered. A reported test count alone is insufficient evidence.

## A. Canonical facts and explicit presentation

Keep one pure pipeline: validated source events → canonical sport state → presentation → request. The renderer must not repair source events or infer hidden facts. Reuse the existing reducer, parser, and provider adapter; make the new renderer a small, deep module with a clear contract.

Suggested interfaces, adapted to repository conventions:

- `build_snapshot(events, cutoff, sport_profile) -> Snapshot`
- `derive_prompt_facts(snapshot, target) -> PromptFacts`
- `render_state(facts, presentation_version) -> str`
- `build_endpoint_request(frozen_input, presentation_version, config) -> Request`

`PromptFacts` contains stable entity IDs and names, target identity, verified score scope, score values, target margin when defined, period, clock value/direction/scope, supported status, unknown fields, and provenance. Each derived fact stores its source event ID/timestamp and derivation rule. Do not introduce dozens of flags or duplicate facts in several mutable dictionaries.

The new presentation must make these facts explicit:

- Target: named team/player and exact binary proposition. Probability 1 refers to this proposition.
- Home/away mapping: both named entities, where applicable.
- Scores: named values and scope. For cumulative team scores, compute `target score − opponent score` and express lead, tie, or deficit explicitly. Fail on ambiguous mappings; do not guess.
- Period: verified sport-specific meaning. For regulation American football, Q2 is the second quarter of four regulation quarters; for regulation basketball, use the verified league's quarter structure. Do not assume all basketball games use four quarters.
- Clock: state whether it counts down or up and whether it applies to the current period. Explain regulation structure only when verified. A period clock is not total match time remaining. Do not claim the game can end after that period; overtime possibility remains distinct.
- Tennis: completed observed sets and current-set games are separate; points are separate if supported. Unknown current-set values stay unknown. Do not infer unobserved completed sets, match format, or winner.
- Timeouts/possession: unknown unless supported. An observed timeout does not imply zero remaining timeouts.

Example format, conditional on verified facts:

> Target: Kings win the game. Jazz: 85 points; Kings: 59 points. Kings trail by 26 points. Score scope: cumulative game score. Current period: [verified value]. Clock: [verified value and semantics]. Unverified state fields: [list].

Keep event IDs, commentary, cutoff, market anchor, probability schema, and inference settings identical between presentation variants. Replace only the state block. Preserve the exact original archived prompt in the control. If the old prompt has a confirmed defective state, treat it as a separate repair case, not a clean representation comparison.

Tests must catch meaningful failures: target swapped home/away; target deficit sign; tie; Q2 versus final period; unknown clock direction; tennis transition with null scores; cumulative basketball/football scores across a boundary; unknown timeout count; aliases; no post-cutoff facts; preview/rerun parity. Inspect golden prompt diffs. Do not write tests that merely restate renderer strings without checking semantics.

## B. Freeze the diagnostic before inference

Use full Inkling with the documented successful execution configuration from its completed run. Do not silently change effort, cap, provider routing, sampling settings, or probability validation. Record requested and served identities separately. This is a representation experiment, not another model sweep.

Default roster: six screened windows from distinct games, if available:

1. Kings–Jazz reversal.
2. Rams–Cardinals reversal/time confusion.
3. Florida State–Alabama period confusion.
4. Ordinary basketball case.
5. Ordinary football case.
6. Ordinary verified-clock case from a supported sport.

Select ordinary cases deterministically from existing screened windows without inspecting their prediction errors: sort stable IDs, exclude known-failure games, choose the first eligible case for each predeclared slot. Freeze selection rules and hashes. Keep Hawaii and unresolved semantics out of the clean comparison. If a required case lacks verifiable facts, record it as audit-only and do not substitute based on model performance.

For these six windows: 2 presentations × 2 arms × 2 endpoints × 2 independent fresh replicates = 96 logical calls. Both variants get the same number of replicates. Archived calls are historical comparisons and do not substitute for fresh controls. Record replicate IDs; even identical prompt hashes must not cause replicate collapse through cache deduplication. Cache reuse is permitted only to resume that same logical replicate after a verified completed response.

Optional separate tennis repair diagnostic: original defective versus repaired presentation, both arms/endpoints, two replicates = 16 calls. Do not pool it with clean six-window results. If funds are tight, omit tennis first; then freeze a reduced clean matrix before any calls, retaining balanced variants and replicates. Do not drop cells after seeing errors.

Create `cases.jsonl`, `gold_facts.jsonl`, `requests.jsonl`, `config.json`, and `manifest.json`. The manifest records schema/presentation/parser versions, source hashes, selected roster, exclusions, matrix size, budget decision, UTC deadline, and code state. Freeze them before launch. Any revision gets a new directory/version and a documented reason.

### Factual rubric

Gold facts must come from verified inputs, not model rationales or the final outcome. Annotate target identity, each named score/scope, leading entity, period meaning, clock semantics, and unsupported timeout/game-end assertions. Use per-fact labels:

- correct;
- contradicted;
- unsupported assertion;
- not stated;
- unscorable input.

A valid probability does not demonstrate comprehension. Absence of a claim is `not stated`, not correct or wrong. Retain raw final content and any returned reasoning even when citations/schema fail. If no rationale is available, probability usability remains assessable, but factual comprehension may be unobservable. Do not request hidden reasoning or assume an absent reasoning field means no reasoning occurred.

Score existing returned explanations manually against the rubric, with exact short evidence spans and gold provenance. Prepare anonymized review records with variant hidden where practical; note that wording may reveal it. Have a second reviewer check all contradictions and a subset of correct labels if workers are available. Disagreements stay visible. No additional LLM judge is needed.

Do not add a comprehension questionnaire only to the explicit variant. If a structured factual readback is later needed, it must be a separate balanced experiment; adding it changes the forecasting task.

## C. Resumable execution in tmux

Write a self-contained `run_comprehension.sh` using the existing project environment and runner. It accepts an experiment directory, validates frozen hashes, performs preflight, resumes missing logical replicates, and always emits coverage/analysis on graceful stop. It must run without an interactive prompt. Never put secrets in manifests, logs, or command lines.

The budget manager owns atomic reservations and settlement. The deadline manager owns cancellation. The provider adapter owns response capture. Keep these policies out of renderers and analysis.

Requirements:

- One shared account-slot pool and existing aggregate spending guard, including outstanding requests and retries.
- Conservative concurrency; bounded physical retries using existing policy, at most three attempts per logical call. Do not retry a completed usable call merely to obtain a better explanation or forecast.
- Persist raw response, actual provider/model, finish reason, reasoning/final usage, billed cost, validation components, and logical replicate ID.
- Deadline/cancellation apply to HTTP timeouts, slot waits, backoff, and account cooldown. A transient account circuit gets a bounded recovery decision; exhausted quota or budget produces a terminal blocked status.
- Append-only attempts; atomic status snapshots; lock against two runners writing one experiment. Do not duplicate completed replicates after restart.
- A heartbeat reports completed logical cells, probability-usable cells, spend/reservations, deadline, current phase, and stop reason.
- SIGTERM requests graceful stop. Stop only this experiment's owned child processes; do not use broad `pkill python` or kill unrelated tmux sessions.

If no current deadline is specified, set an immutable launch + 120-minute deadline and stop dispatching at least 20 minutes before it for annotation/analysis. Resumes preserve that deadline. Longer inference runs still belong in tmux.

Deliver tested launch, status, resume, and stop commands using actual paths/session names. For example, after the script exists:

```bash
tmux new-session -d -s forecast-comprehension 'bash /absolute/repo/path/run_comprehension.sh /absolute/experiment/path'
tmux capture-pane -p -t forecast-comprehension -S -80
```

Use an explicit stop file or the supervisor PID recorded by this run for graceful stopping. Validate stop/resume using a fake provider fixture before paid execution: completed cells stay complete, unattempted cells remain identifiable, no post-stop dispatch occurs, reservations reconcile, and analysis still runs. Wrapper exit zero is not evidence that all cells completed.

## D. Analysis and interpretation

Produce endpoint coverage by variant/arm/replicate and all validation components. Keep strict-format validity, probability usability, citation compliance, and factual correctness separate. Reuse existing probability eligibility rules unchanged; any new rule is separately versioned.

Join using window ID, arm, endpoint, target orientation, configuration identity, and replicate. Assert uniqueness and equal market values/timestamps across variants. Compare only matched successful cells, while showing how many cells were excluded and why.

Primary diagnostic: counts of observable factual contradictions and unsupported assertions by category, plus not-stated/unscorable counts. Show paired original-versus-explicit transitions and case-level evidence. Report window/game denominators; do not treat correlated endpoints or claims as independent observations.

Secondary: within-presentation replicate disagreement in probability; absolute predicted updates; update MAE versus persistence; matched presentation difference; arm-specific results. Keep random and selected failure cases separate. Report all cases, not only improvements. With roughly six deliberately selected games, emphasize descriptive paired values; avoid broad prevalence estimates, rankings, or decorative bootstrap confidence intervals.

Two fresh repeats give only a weak diagnostic of variability. A deterministic sampling setting can make repeats identical, and provider execution need not be deterministic. Record both facts rather than interpreting repeat agreement as proof of stability.

Possible interpretations:

- Factual errors decline, MAE does not: presentation helps interpretation; the market-update objective remains unresolved.
- MAE declines, factual errors do not: no demonstrated comprehension mechanism.
- Known failures improve but ordinary cases degrade: revision is not ready for broad evaluation.
- No improvement: inspect faithful rendering, response observability, and variability before broad claims.

The diagnostic is selected partly on known failures. Never report it as representative sports forecasting performance. Freeze any chosen prompt revision before a later evaluation on new games disjoint from all cases used for inspection, repair, or prompt choice. That evaluation is not authorized by this guide's execution matrix. Do not train/select prompts against held-out market outcomes.

## Final deliverables and acceptance

Deliver one experiment directory containing frozen inputs/configuration; source/code provenance; raw responses and attempts; coverage; annotated comprehension records; paired diagnostic tables; tests and exact commands; and `review.md` with relative links to each artifact.

The review must answer:

1. Is the reducer repair correct, and do impact previews exactly match rerun rendering?
2. Which facts changed in each presentation, and were any unverified facts introduced?
3. Did the experiment execute the frozen matrix? If partial, give exact missing cells, spend, and stop reason.
4. Did clarification reduce observable score/period/clock errors on the diagnostic cases?
5. How much repeat variation remains, and how does update error behave separately?
6. What is justified next, and what remains unverified?

Record each work package as completed, partial, blocked, failed, or unattempted. Distinguish implemented, tested, launched, and completed. Archive the actual runnable code state with relevant diff/commit; do not commit unrelated edits. Provide the core reducer and focused tests for review rather than citing only a total pass count.

Priority order if time runs short: canonical facts and parity fix → frozen roster/rubric → runnable bounded execution → factual annotation → descriptive analysis → report. Finish offline deliverables even when inference is blocked. Do not spend the cycle adding more models or rebuilding the GUI.

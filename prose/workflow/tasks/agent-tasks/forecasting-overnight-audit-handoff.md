# Forecasting overnight handoff: make existing results independently auditable

## Mission

Tom will be away for approximately 12 hours and will manually inspect the pipeline afterward. Use this interval to establish what the existing results mean and make the inspection efficient. Do not run another model sweep or new paid inference. Prepare a disjoint new-game evaluation, but do not launch it.

Proceed autonomously with read-only investigation, local implementation, focused tests, cached analyses, and reversible audit-interface changes. Do not overwrite historical experiments, reconstruct uncertain timestamps, change eligibility to improve scores, or commit unrelated edits. This assignment supersedes earlier suggestions to spend the remaining budget on another immediate run.

The deadline is launch + 12 hours, persisted in UTC. Reserve the final hour for integration, inspection-packet verification, and reporting. Completion earlier is preferable to filling the interval with low-value work.

## Scientific context

The protocol independently elicits probabilities at two endpoints and evaluates their difference against a contemporaneous market-price difference. Persistence predicts zero change. This is retrospective market-update agreement, not outcome calibration.

Several configurations lose to persistence on random windows, including matched-model comparisons. Excluding Hawaii and the confirmed tennis defect has not reversed those reported findings. Nevertheless, chronology, input representation, independent elicitation variability, and objective mismatch remain possible contributors.

The completed six-game state-presentation diagnostic has 96 probability-usable calls. Explicit presentation appears to remove selected score/period errors, but its factual summaries need annotation repair. It does not establish representative forecasting improvement.

Known examples:

- Kings–Jazz, game `20023251`, window `2026-02-12T03:40:20Z–03:41:18Z`: responses reverse Jazz 85/Kings 59 and predict Kings near 0.99. Fresh archived controls consistently reverse the leader at both endpoints; therefore they can obtain a small update error despite incorrect interpretation. Explicit responses put Kings around 0.01–0.03.
- Rams–Cardinals: leader reversals, Q2 interpreted as near game end, unsupported timeout assertions.
- Florida State–Alabama: period/game-end and leader errors, particularly in some anchored calls.
- Ciurnelli–Kisimov: old state construction carried a first-set score into a new set whose score was unknown. This is a confirmed construction defect; repair results must stay separate.
- Hawaii: sporting-order contradictions exist in archived parsed records; source-array order is also unreliable. Underlying wallclock semantics remain unresolved. Keep quarantine.
- Jev zero-shift: 51 exclusions were attributed to commentary-support rules, rather than different selected market prices. Extreme terminal/gap timestamps still warrant source tracing.

Treat these as reported findings to verify against current artifacts. Do not invent file paths, raw evidence, or existing APIs if an artifact is missing.

## Priorities and work ownership

The coordinator may delegate these packages to parallel agents. Use exclusive file ownership or isolated worktrees, agree on contracts first, and have only the coordinator modify shared CLI registration/dependencies. Workers write separate outputs; no concurrent writers to ledgers or experiment manifests.

| Package | Responsibility | Dependencies | Priority |
|---|---|---|---|
| A | request provenance and deadline repair verification | none | highest |
| B | reusable end-to-end window trace | none; contract shared with C/D | highest |
| C | offline cohort invariants and impact census | B's record contract, not its full implementation | high |
| D | claim-level annotation repair and review queue | current responses/gold facts | high |
| E | ergonomic manual-inspection packet or GUI integration | B/C/D outputs | high |
| F | frozen proposal for disjoint new-game evaluation | audited renderer/cohort knowledge | secondary |
| Coordinator | source inventory, integration, final report | continuous | highest |

If parallel workers are unavailable, do A → B → C → D → E → F. Do not create an orchestration platform to perform this task.

## First 30 minutes: establish the current state

1. Read applicable AGENTS.md. Record repository root, HEAD, status, environment versions, and relevant uncommitted changes. Preserve other workers' edits.
2. Locate original pilot inputs, model manifests, requests, attempts, raw responses, normalized predictions, market selections, state repair artifacts, Jev controls/replay, and the current GUI. Use `rg --files` and targeted searches.
3. Identify the actual sport-state reducer and tests. The uploaded review names `sport_state_v1`, but the core module/test review remains incomplete.
4. Locate the six-game diagnostic and compare `review.md`, manifest, requests, config, endpoint rows, annotations, and summaries. Record actual paths in an inventory; retain immutable source hashes.
5. Create one new audit output directory with a manifest. Record launch time, absolute UTC deadline, input hashes, commands, code state, and per-package statuses.
6. Record the current reported budget ($6.482200775 spent, $3.517799225 remaining) as an unverified historical subtotal until reconciled. No new API inference is authorized by this handoff. Existing background jobs, if any, must be identified without killing unrelated work.

## A. Resolve provenance and verify stopping behavior

### Request hash mismatch

An external review found matching SHA-256 hashes for uploaded cases, config, and gold facts, but not for the uploaded request file versus `manifest.source_hashes.requests`.

Do not update the manifest hash to make the check pass. Determine:

1. What exact bytes were frozen before dispatch?
2. What exact request bodies were actually sent?
3. Does the upload differ only in whitespace/key order/newline conventions, or do semantic fields differ?
4. Were replicate IDs, configuration, messages, deadlines, or preparation metadata changed after freeze?
5. Can each response be joined to the dispatched request and attempt independently of row order?

Compare byte hashes and canonical JSON hashes separately. Define canonicalization explicitly; preserve array order and every semantically relevant field. Distinguish a full-file hash from body, configuration, semantic-request, and logical-request identities. Never normalize away meaningful differences.

Deliver `provenance_report.md`, machine-readable request reconciliation, and any conflicting versions. State whether provenance is established, partially established, or unresolved. If the uploaded copy alone explains the mismatch, document evidence; if dispatched bodies cannot be recovered, say so.

### Deadline path

The completed diagnostic reportedly wrote `deadline.json` while the runner expected `logs/shared_deadline.json`, so its deadline was not enforced. A subsequent correction is reported.

Using a fake provider, verify the actual integrated path now enforces:

- expiry before dispatch sends zero calls;
- expiry during slot acquisition or retry prevents another dispatch;
- running requests use bounded timeouts/cancellation consistent with deadline policy;
- resumes preserve the original deadline;
- graceful stop produces coverage/analysis and settles reservations;
- completed logical replicates are not resent;
- no next shard starts after cancellation.

Separate dispatch cutoff from final-analysis deadline. Test observable behavior, not merely that a deadline file exists. Do not claim the historical run enforced a repaired policy retroactively.

## B. One-command end-to-end window trace

Build a small reusable command using current modules, proposed shape:

`audit-window --window-id ID --output DIR`

Provide real commands after implementation; this is not an assertion that this CLI already exists. It must operate entirely on cached artifacts.

Export these linked stages:

1. Game/provider identities, crosswalk, market question/settlement, binary outcomes, target identity, home/away mapping, and orientation.
2. Raw archived provider events around the window, including preceding state suppliers and relevant transitions; original values and source order.
3. Recovery transformations, timestamp parsing, ordering, and cutoff inclusion decisions.
4. Canonical endpoint snapshots: value, scope, supplying event, event time, and derivation for every field. Include unknowns and carry-forward decisions.
5. Evidence selection/compression: selected and omitted event IDs, reasons, limits, hashes, and earliest/latest included timestamps.
6. Exact dispatched messages and inference configuration for every requested model/arm/endpoint/replicate. Include served identity and actual finish/usage when available.
7. Raw response references, final content, returned reasoning if retained, extraction output, validation components, normalized probabilities.
8. Market as-of selections: chosen observations, timestamps, age, orientation, value, support predicates, and price differences.
9. Analysis joins: stable keys, eligibility, exclusion reasons, predictions, market update, absolute error, and persistence error.

Output `trace.json` plus a readable `trace.md` or static HTML with links to local copies. Bundle the relevant raw bytes; do not leave all evidence accessible only through paths on another machine. Use relative links inside the packet. Keep source IDs and paths in provenance; never expose credentials.

Each stage reports verified, unresolved, missing, or not applicable. If exact dispatched bytes are missing, distinguish a reconstruction from an archived request. Never label reconstruction as original.

Deep-module contract: one resolver finds source artifacts, one trace builder produces a typed record, and renderers present it. UI code must not redo joins, derive sports rules, or repair data. Prefer existing parsing/state/market-selection modules; avoid duplicate implementations that can drift.

Trace at least:

- Kings–Jazz;
- Rams–Cardinals;
- Florida State–Alabama;
- confirmed tennis transition;
- affected Hawaii window plus an earlier retained Hawaii window;
- two ordinary windows selected deterministically without inspecting errors;
- representative extreme Jev terminal/gap cases and a normal zero-shift case.

Record the selection rule and IDs. Missing cases remain visible in the inventory.

## C. Offline invariant census and affected-results sensitivity

Audit the complete existing pilot cohort and its archived model requests/results; do not broaden claims to the entire dataset unless the scan actually covers it. For each check, report cohort denominator, applicable denominator, failed count, unknown count, and concrete examples. Screening is not certification.

Checks:

- Unique keys for game/market/window/arm/endpoint/configuration/replicate; no silent many-to-many joins or duplicate loss through dict overwrites.
- Target identity and orientation agree across request, response normalization, selected market token/outcome, and paired analysis.
- Endpoint timestamps are ordered; market observations obey implemented as-of rules; repeated observations and sparse coverage are distinguished from unsorted archives.
- No supplied evidence beyond cutoff under the declared timestamp field. This cannot establish publication-time availability when only retrospective event time exists.
- Period/clock progression under applicable sport/provider semantics. Do not impose countdown on hockey or numeric period ordering on unsupported labels.
- Score scope, carry-forward across transitions, score provenance, and missing-field synchronization. Preserve cumulative basketball/football scores; tennis current-set state is separate from completed sets.
- Header/commentary conflicts, ambiguous home/away aliases, missing target rules, and unsupported state assertions introduced by the renderer.
- Anchor timestamp/value/orientation and staleness; commentary/state fields otherwise matched across arms/presentations where the experiment requires it.
- Probability extraction agrees with emitted numbers, domain checks, validation versions, pair eligibility, and analysis signs.
- Recompute cached update MAE and persistence comparisons from exported endpoint/market rows. Reproduce headline results or explain discrepancies.

Classify findings as confirmed defect, suspected defect, unknown semantics, verified model-response error, or protocol limitation. A heuristic flag is not a confirmed defect. Keep overlapping reason counts explicit; also report distinct affected windows/games.

Produce a defect-to-results impact table: affected windows, models/arms, original scores, and cached exclusion sensitivities where justified. Preserve primary historical results. Exclusions discovered using errors are exploratory; never silently replace the primary cohort. Do not turn unknown semantics into demonstrated corruption or infer correctness from a flat market.

For Jev: independently rerun the cached zero-shift replay if its code/inputs are available. Verify the reported 33 terminal, 13 early-support, and five gap cases. Trace extreme timestamp values back to identities and raw records. Do not repair a 359-day gap by clipping or guessing units.

## D. Repair factual annotations without another judge model

The current `score_analysis.py` records contradiction indices but assigns other correctness labels through regex keywords; period and clock share a contradiction list, and target identity is universally `not stated`. Replace this with aggregation of explicit annotations keyed by stable response ID and fact/claim ID.

Annotation schema:

- logical response identity and content hash;
- fact category and specific claim;
- exact short evidence span with offsets where practical;
- expected fact and source provenance;
- label: correct / contradicted / unsupported assertion / not stated / unscorable input;
- reviewer, review method, confidence/ambiguity note, and timestamp.

Presence of `Q2`, `lead`, or `remaining` does not establish correctness. A response may correctly state Q2 but incorrectly infer that the game is nearly over; separate period identity from game-end inference. One response can contain conflicting claims—record both rather than forcing a blanket correct label.

Possession can be supported by supplied commentary even when a snapshot field is unknown. Check the full available input before declaring a possession statement unsupported. Conversely, one observed timeout does not establish remaining timeout count. Unknown state fields alone are not a gold answer to every factual question.

Prepare all 96 diagnostic responses for review. Preserve available final rationale even when citations are invalid. If only raw files contain the explanation, include them. If unavailable, mark unobservable, not correct. Do not request hidden reasoning or assume reasoning settings from absence of a returned field.

An agent may review and propose claim-level annotations, but label these explicitly as agent review pending Tom's confirmation. If two independent workers review, preserve disagreement and reviewer identity. Do not claim human manual review or reviewer independence merely because separate agent passes occurred. No paid LLM judge calls.

Aggregation code must only validate and summarize annotations, not manufacture labels. Show response-level and distinct-game counts alongside claim counts. Keep correlation and selection limitations visible. Retain the old annotation version for comparison.

## E. Make Tom's inspection efficient

Reuse the existing GUI if practical. Prefer a reliable static audit packet over a risky interface rewrite. Required workflow:

- Select a case and endpoint.
- See named target/home/away scores and period/clock beside raw events.
- Open the exact original versus explicit prompts and a focused diff.
- Compare emitted versus normalized probabilities and selected market observations.
- Follow every flag/derived field to its source record.
- Record a review label and note without modifying experimental inputs.

Preserve case selection when toggling overlays or moving between views. Make model/market overlays optional. Keep the factual source view available without predictions to reduce outcome-driven inspection. Show unknown semantics prominently; never replace missing values with zero.

Create a queue ordered by diagnostic category, not only absolute model error: confirmed defects, identity/orientation flags, chronology anomalies, representation failures, ordinary controls, and extraction/join checks. Deduplicate cases while preserving multiple reasons.

Deliver `START_HERE.md` with actual commands, packet entry point, a 30–60 minute suggested inspection sequence, and remaining questions. Verify links/files resolve and representative pages actually display. If using a GUI, test the existing selection-preservation behavior. Include portable files for review even when the app requires the original repository.

## F. Prepare the next evaluation, without dispatch

Only after recording the current renderer version and unresolved profile limits:

- Define a game-disjoint roster excluding every game used for failure inspection, repair, diagnostic selection, or prompt tuning.
- Freeze selection without inspecting prediction errors or resolved outcomes. Use supported score/clock profiles and show coverage/exclusion counts.
- Keep random and target-enriched strata separate. Choose a modest balanced original/explicit comparison; provide the matrix size and expected cost rather than maximizing remaining spend.
- Specify identical commentary/cutoff/anchor/settings, same eligibility rules, factual audit sample, primary paired metrics, and missingness reporting.
- Record any historical knowledge-contamination limitation and retrospective timestamp limitation.
- Prepare immutable manifests/requests only if inputs are ready; otherwise deliver a precise blocker list.

Do not launch inference. The remaining budget is not a directive to spend it. Tom will decide after inspecting the packet.

## Execution, engineering, and verification

Use one self-contained resumable `run_overnight_audit.sh` for long scans, running in tmux. It discovers actual repository paths, checks inputs, executes task phases, writes checkpoints, and generates the final report on normal completion or graceful stop. Give actual launch/status/resume/stop commands. A command template must be clearly labeled until tested.

Persist one launch epoch and UTC deadline. Resumes preserve it. Bound CPU/memory and worker concurrency for the Framework server; do not parallelize expensive full-cohort scans without considering its 16 GB RAM. Avoid repeated whole-archive parsing by building one immutable index reused by tracing/census. Cache keys include source hashes and code/schema versions. No broad `pkill`, no unrelated tmux deletion.

Ousterhout-oriented requirements:

- Clear modules with small interfaces and substantial internal responsibility.
- One source of truth for target mapping, state rebuilding, request identity, and eligibility.
- Pure deterministic transformations where possible; IO at boundaries.
- Structured errors and explicit unknowns instead of silent fallback.
- No index-based semantic labels, mutable global experiment state, or chains of special-case patches.
- Document invariants and why a decision exists, not merely what each line does.
- Keep changes scoped; use existing dependencies unless a new one is clearly necessary.

Meaningful tests: duplicate/many-to-many joins rejected; swapped orientation detected; post-cutoff event flagged; sport-specific score scope; missing source fields remain unknown; exact request versus reconstruction distinction; canonical hash detects semantic changes; annotation aggregation never guesses labels; stop/resume/deadline behavior. Reproduce cached headline arithmetic independently.

Record exact commands, outcomes, relevant diff/commit, and code state. Do not claim passing tests solely from an old review. Provide the core reducer and relevant tests in the review bundle.

## Final deliverables and acceptance

One portable audit directory containing:

- `START_HERE.md`;
- concise `overnight_review.md`;
- source inventory and immutable audit manifest;
- request provenance reconciliation;
- deadline integration evidence;
- end-to-end case traces with relevant raw bytes;
- cohort invariant census and impact/sensitivity tables;
- claim-level annotations and review queue;
- runnable script, code provenance, and verification log;
- proposed disjoint evaluation and cost estimate, marked not launched.

The final review answers:

1. Which results can be reproduced from source bytes?
2. Which confirmed defects affect which results, and how much?
3. Which issues remain ambiguous because semantics or artifacts are missing?
4. Which model-response errors are directly evidenced despite valid input/extraction?
5. What should Tom inspect first, and which records resolve each question?
6. Is the explicit renderer ready for a held-out comparison, and what must be fixed first?

Every package gets completed/partial/blocked/failed/unattempted status. Distinguish implemented, tested, launched, and completed. Never mark the overnight task complete because a wrapper exits zero. If time runs short, prioritize provenance, traceability, and honest missing-status reporting over interface polish or preparing a larger next experiment.

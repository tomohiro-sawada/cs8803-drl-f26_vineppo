# Next-cycle handoff: reliable measurement before more model breadth

Date: 2026-10-02, America/New_York.

## 1. Objective and highest-ROI choices

The last cycle improved inspection and supervision, but no new model completed an evaluation. This cycle must produce useful results even if API access fails again.

Prioritize four outcomes:

1. Recompute existing results on a precisely defined input-screened cohort, using cached responses. Establish how much the known Hawaii contradiction changes earlier conclusions.
2. Establish reliable probability extraction and one consistent operational gate. Inspect cached formatting failures before spending more on them.
3. Explain the 51 zero-shift Jev exclusions and the severe loss of coverage under shifted controls.
4. Complete one additional baseline through inference, analysis, and GUI inspection if a short operational pilot succeeds.

In parallel, make the existing GUI usable for the user's manual review and investigate the unresolved CFBD timestamp semantics. No broad GUI rewrite, new model search, RLVR, learned lag correction, or full-dataset regeneration in this cycle. The clarified-prompt experiment is deferred until these priorities are satisfied; a working manifest is not an experimental result.

Do not restart the expired overnight run in place. Use a new experiment directory, parent references, and explicitly recorded new runtime deadline. Reuse exact compatible cached requests; preserve every previous run and failure record.

## 2. Known evidence and remaining uncertainty

Read the actual artifacts and applicable AGENTS.md/conventions before editing. These are reported observations, not a substitute for verifying the code and joins.

Parent artifacts:

- `artifacts/experiments/llm_vs_crowd/2026-10-02-001200_hindcast-followup/`
- `artifacts/experiments/llm_vs_crowd/2026-10-01-020000_overnight-hindcast/`
- Original pilot and offline audit: `2026-09-30-190000_hindcast-qa-jev`, `2026-09-30-214900_hindcast-jev-audit`.

Important reports: `morning_review.md`, `root_cause.md`, `ledger.md`, `jev_stage_table.md`, `chronology_scope.json`, `eligibility_manifest.json`, `jobs/mistral_small_4/coverage.md`, `inkling_small_routing.json`, and request/forecast/attempt files.

Current understanding:

- The pilot comprises 150 windows from 50 games. One Hawaii window is quarantined; 149 remain eligible under applicable checks. Handball/tennis checks include unknown applicability. Remaining windows are not certified valid.
- The recovered Hawaii CFBD game stream has 13 period regressions clustered from `2025-11-02T06:03:49Z`. Two earlier pilot windows end at `03:07:07Z` and `03:38:07Z`, before that cluster.
- Recovery copies `play['wallclock']` into both `event_time_utc` and `provider_timestamp_utc`; hindcast sorting preserves the inversion. This locates the contradiction upstream of rendering. It does not prove the underlying cause or timestamp semantics.
- Inkling Small's route was excluded by `allow_fallbacks: false` and an unavailable pin. The user's minimal paid request worked. This is not evidence the model lacks an endpoint.
- Llama 4096 repaired three previously truncated prompts. Larger-cohort reliability remains unknown.
- Inkling full-model smoke produced 8 successes and 4 truncations.
- Mistral smoke passed with only 3/12 valid responses while Llama was blocked with 19/24. These gates must be reconciled.
- Mistral bulk has 160 successful endpoints, 137 invalid-format endpoints (129 malformed JSON, 8 unknown evidence IDs), and 291 unattempted endpoints, plus other statuses; only 48 complete window-arm pairs and eight windows complete in both arms. Pooled MAE 0.123 is not a cohort or population estimate.
- The shared account circuit stopped work around 01:27 EDT, well before the deadline, after repeated 429s. Account-wide versus provider-specific limiting is not yet established by the attached reports.
- Jev: 1000 targets, 976 inference successes, 925 valid zero-shift market pairs, 67 windows/63 games valid across five shifts. The intersection's majority baseline is 27/67, approximately 0.403; Jev agreement is approximately 0.493. This does not establish a lag optimum.
- `ledger.md` incorrectly labels all 20 truncation IDs as Llama evidence: the per-model totals show three Llama and 17 Qwen. Repair the report generator, not just the generated heading.

Scientific contract:

- Model output is event-win probability; update error against market movement is a market-association diagnostic.
- EVIDENCE and LAGGED_ANCHOR both contain structured state and commentary. The contrast adds the same stale `m_left` at both endpoints. It does not isolate commentary value.
- Identity, occurrence time, historical availability, settlement rules, and outcome labels are separate evidence dimensions.
- No new outcome-scoring claims without independently supported rule-mapped labels. No alignment certification from absence of detected violations.

## 3. Parallel modules and coordinator responsibilities

Deploy bounded workers if available; the user authorizes parallel agent work. Use exclusive file ownership/worktrees. First map actual files and agree on small interfaces; do not assume names below require new modules if equivalents exist.

| Owner | Scope | Dependency |
|---|---|---|
| A: cached measurement | Versioned probability extraction, failed-output audit, cached sensitivity analyses | Existing artifacts; no network |
| B: Jev replay | Zero-shift trace and coverage explanation | Existing artifacts; no network |
| C: chronology and review | CFBD provenance/semantics, targeted review packet, GUI verification | Existing artifacts; documentation/source lookup if necessary |
| D: inference reliability | Consistent gates, route/cap diagnostics, bounded single-baseline completion | Coordinate parsing contract with A; input eligibility with C |
| Coordinator | Integration, shared status/deadline/budget, reports and commits | Review worker diffs and verify actual execution |

Assign ownership before editing shared parsing, overnight-runner, or GUI modules. Workers do not concurrently edit those files. Reuse existing pure parsers, joins, selection helpers and bootstrap code where correct. Keep provider decisions in the execution module and eligibility decisions in explicit manifests, rather than sprinkling them throughout scripts/UI.

## 4. Workstream A — recover information from cached outputs first

### A1. Audit raw formatting failures

Inspect all 129 Mistral malformed-JSON responses programmatically. Generate a deterministic human-readable sample across failure shapes (at least 12, or all if fewer), linking raw response, exact request, finish reason, validation message, usage, and hash.

Determine whether failures are:

- JSON surrounded by code fences or explanation;
- a complete object with wrong fields/types;
- reasoning text stored separately from final content;
- actual malformed or incomplete JSON;
- incompatible provider output/schema behavior;
- parser/adapter errors;
- absent numeric probabilities.

Do not assume malformed JSON means the model produced no usable probability. Conversely, do not recover numbers from prose by taking the first number or accepting a guessed target.

### A2. Separate measurement from auxiliary validity

Define one versioned parser/validator contract for all models:

- `probability_status`: unambiguous numeric target probability, finite, in [0,1], from the designated response field/object.
- `format_status`: JSON syntax and schema compliance.
- `citation_status`: evidence-ID validity, missing citations, or not requested.
- `termination_status`: stop, length, refusal, transport failure, etc.
- `extraction_method` and parser version, plus original status/raw reference.

Default reject truncated responses from the primary probability-usable analysis even if a partial number is visible. Never use a default 0.5 or 0 for invalid responses. Reject bools masquerading as numbers, nonfinite/out-of-range values, multiple conflicting forecast objects, and ambiguous target fields.

Safe deterministic normalization may remove a surrounding code fence or decode an explicitly documented wrapper. Preserve original bytes. Do not call another LLM to repair outputs or invent fields. Unknown evidence IDs remain invalid citations even if the probability is usable.

Preserve historical `success_status_only` results as the original strict analysis. Add a separately labeled, consistently applied `probability_usable_v1` sensitivity where justified. Do not change only Llama/Mistral to make their scores look better. Distinguish response-format compliance from forecasting accuracy in the report.

Verification: fixtures for fenced JSON, malformed JSON, conflicting objects, invalid citations with valid probability, truncation, bool/NaN/out-of-range values, and a missing probability. Confirm raw/source records remain unchanged.

### A3. Recompute the useful comparisons offline

Build a normalized cached table for Gemini, Gemma, GPT-OSS and any usable new-model outputs, bound to exact requests/configurations. Use the frozen 150-window cohort and the separately labeled 149-window applicable-check-screened cohort. For Mistral, label partial coverage and report strata separately.

Required comparisons:

1. Original strict results reproduced from cached data.
2. Strict results after excluding the quarantined Hawaii window.
3. Probability-usable sensitivity, full and screened, if A2 changes eligibility.
4. Prompt/evidence-matched arm sensitivity where anchor-induced compression changes evidence.

For each model/arm/stratum report windows/games, endpoint and pair coverage, model update MAE, persistence MAE on the same pairs, paired difference, directional summary, and fixed-seed 5000-draw game-cluster intervals when meaningful. Reuse sampled-game multiplicity and existing game-balanced sensitivity. Very small cohorts get descriptive results with explicit limitations.

Show how much Hawaii exclusion changes GPT-OSS random-window error and its paired comparison to Gemini. Do not assume removing one extreme fixes the broad failure; calculate it. Include median/p90/max absolute predicted update, signed-update bias, and worst-five error concentration without removing extremes from primary tables.

This work must finish regardless of API availability. Deliver `cached_comparisons.csv`, `coverage_by_validity.csv`, `failed_output_audit.md`, and `cached_findings.md` (names may follow repo conventions).

## 5. Workstream B — Jev zero-shift replay and control coverage

### B1. Explain all 51 zero-shift exclusions

For every successful original Jev target that becomes `outside_supported_segment` at zero shift, join:

- original window/game/source IDs;
- original left/right timestamps, market row IDs, prices and series/version;
- commentary boundaries;
- re-query source series, segment boundaries and eligibility predicates;
- selected as-of observation IDs/times/prices;
- exact first differing assumption.

Produce a replay table for all 51 cases and at least five detailed examples across sources where available. Also verify zero-shift row/time/price equality on the retained 925; report discrepancies separately.

A zero-shift replay should reproduce the original comparison if it uses the same source, boundaries and rules. If shifted controls deliberately impose stricter support rules, document that difference and its purpose; do not call original windows invalid by implication. If code is wrong, add a minimal reproducer and new analysis version. Do not merely relax the support filter to recover coverage.

### B2. Explain endpoint collapse and irregular spacing

For `nonincreasing_observations` and `incompatible_spacing`, inspect at least five cases per reason. Determine whether shifted endpoints select the same observation, reverse time due to a bug, or encounter legitimate archive gaps. Distinguish archive ordering from as-of selection behavior.

Report per-shift eligibility, both marginal reason flags and the existing first-failure counts. First-failure counts depend on shift order and do not estimate independent defect prevalence. Include retained/excluded sport/provider/stratum composition and observation-gap/quote-age summaries where available.

Do not interpolate missing prices or replace the observed target with a different interval without defining a new diagnostic. Do not tune shifts to maximize agreement.

### B3. Analysis after replay correctness is established

Retain pairwise zero-versus-shift comparisons on their shared eligible windows, plus the 67-window all-shift intersection sensitivity. Report baselines/class distribution for each actual cohort. Reuse paired game-cluster uncertainty, and state exploratory multiplicity.

If the 51-case investigation reveals an implementation change, rerun controls from cached decisions under a new version. Do not infer temporal precision from 49.3% agreement versus 40.3% majority accuracy alone.

Deliver `jev_zero_shift_replay.csv`, a short explanation with code references and examples, and corrected/versioned control tables if warranted.

## 6. Workstream C — targeted provenance and manual review

### C1. Finish CFBD timestamp tracing

Trace Hawaii native events `401760403855`, `401760403873`, `401760403639` and neighbors back to archived raw provider payloads. Record raw field value, source array position/sequence, provider version/retrieval metadata, parsed value, and each transformation.

Investigate documented meaning of `wallclock` using authoritative provider schema/documentation where available. A UTC-looking value does not establish event occurrence or publication time. If archived raw payloads are absent or field semantics undocumented, state that exact limitation. Do not assign blame to the provider or claim the recovery transformation is proven wrong without evidence.

Check a small predeclared source sample: the whole Hawaii stream plus three other CFBD games selected without model-error ranking. This is a targeted source diagnostic, not a prevalence estimate. Do not launch a broad ingestion/backfill job.

Rename/rewrite the root-cause conclusion to distinguish observed contradiction, first located stage, underlying cause status and scope. Keep existing source records immutable. No inferred wall-clock reconstruction from sporting clocks.

### C2. Give the user a compact review queue

Assemble at most 12 linked cases: Hawaii, several clock/duplicate reviews from distinct games, unknown-applicability handball/tennis examples, and a few large model errors from otherwise screened inputs. Record why each was selected. This is diagnostic sampling, not unbiased prevalence estimation.

Verify the existing GUI on these cases:

- game/market/window persist across temporal, hindcast, Jev and prompt panels;
- archived exact messages display original order/newlines and provenance;
- raw responses and original versus derived validity statuses are inspectable;
- unsupported checks display unknown rather than pass;
- a temporal-only game does not silently switch to another game;
- nonzero visual shifts remain labeled and leave source timestamps unchanged.

Capture screenshots if the available browser tooling supports it. Do not install a large new UI stack. Fix only demonstrated ergonomics/selection defects. Provide actual deep links with resolved IDs, not just a link shape. Reviews remain pending until the user performs them; automated AppTests are not manual evidence certification.

## 7. Workstream D — one reliable new baseline, not more breadth

### D1. Standardize gates before selection

Adopt one explicit operational pilot policy across candidates before new outputs are inspected for forecasting performance. Suggested pilot: six screened windows across sports with short/long contexts, both arms and both endpoints = 24 logical requests per candidate.

Primary operational measure is probability usability under A2, with citation/schema compliance reported separately. To proceed to the cohort, require at least 23/24 usable probabilities, no unresolved systematic truncation, and no unresolved routing/budget/limiter defect. These thresholds are operational heuristics, not inferential guarantees. Keep the original strict-format result visible. If changing the requested output schema, that is a new prompt/configuration version and cannot be pooled with the old one.

Do not pass Mistral with one success, or require perfect auxiliary citations from one model while accepting malformed responses from another. Pilot failures prompt one targeted repair and a new configuration; no endless prompt tuning or choosing by update MAE.

### D2. Candidate order and repair scope

First candidate: existing Llama 3.3 70B at 4096 output tokens, using the known working adapter. Reuse compatible prior 4096 successes. Investigate citation failures separately; do not keep the 2048 cap merely because historical requests used it.

Second candidate if needed: paid `thinkingmachines/inkling-small` with the route restriction repaired. First perform one minimal availability call, then actual forecasting pilot calls with explicit parameter support. Do not silently strip structured-output settings. If those settings exclude the route, create a separately documented JSON-instruction configuration.

If neither works, use the previously authorized fallback roster, beginning with a model other than the unresolved high-malformed-output Mistral configuration. At most one additional fallback pilot this cycle. Check the existing registry/current endpoint before launch; keep actual model/provider identities separate. Avoid treating a provider-switch alias as the same exact served checkpoint without support.

Inkling full-model truncation may be tested with a larger output budget as a bounded alternative if operationally preferable; log the reason for the choice before predictive results. Do not smoke every model or use a failing account circuit to cycle through providers blindly.

### D3. Account recovery and self-contained supervision

Read actual 429 bodies/headers and circuit state. Determine whether the limit is account-wide, provider-specific, request-rate or token-throughput, or exhausted quota. Start at a conservative global concurrency of one until evidence supports increasing it. One consistent slot pool applies across processes; model-specific concurrency is separate.

Implement bounded cooldown recovery: honor retry delay, wait without holding a provider slot, send a single probe after cooldown, and resume if successful. Cap consecutive failed recovery probes (for example three) and persist stop reason. For quota exhaustion, stop promptly. No credential rotation or unbounded retries. Slot waits, HTTP calls and backoff all respect the absolute deadline/cancellation.

Run one selected model on the 149-window applicable-check-screened cohort: two arms × two endpoints = 596 logical requests maximum. Any revised eligibility manifest gets its own cohort identity. Checkpoint per response and perform coverage/analysis automatically at completion or stop. Compare to persistence on identical success pairs by stratum; never report only pooled MAE.

Keep the previously authorized aggregate $10 ceiling inclusive of the reported $1.91 already spent: reconcile actual spend first, and use only the remainder. Do not silently grant a fresh $10 budget. Set a new explicit runtime deadline of launch + four hours, leaving time for analysis; do not modify the expired deadline in the old run. If a known existing project limit is lower, honor it.

The runner must be self-contained and resumable in tmux, with actual launch/status/resume/graceful-stop commands, progress heartbeat, request/config hashes, provider/model, attempts, finish reason, latency, usage, in-flight reservations and structured stop reasons. At most three physical attempts per new request. Single writer per shard. Safe atomic summaries and append-only attempts.

Full inference success means planned endpoint statuses are resolved, pair coverage is reported, and analysis exists. Wrapper exit success is not sufficient. If no candidate passes or the circuit cannot recover, finish the offline work and return a concrete blocker instead of claiming completed experiments.

## 8. Engineering and verification

Use deep, cohesive modules: parser owns extraction; control replay owns market selection; runner owns provider/retry/deadline behavior; GUI consumes saved records. Prefer explicit configuration and schemas over boolean switches scattered across files. Comments explain decisions and invariants. Avoid broad refactors and duplicated implementations.

Read repository conventions, use established artifact resolution, and commit coherent changes without pushing/deploying unless separately authorized. Do not expose credentials or modify raw/frozen artifacts. Preserve original statuses when generating derived tables. Fix the ledger's model-filtering error in its generator with a regression check.

Required focused checks:

- Parser rejection/normalization cases from A2.
- Cached metric reproduction and aligned model/persistence pairs.
- Zero-shift replay and duplicate/as-of endpoint cases.
- Uniform smoke decision across candidates.
- Circuit cooldown/probe/cancellation and global deadline behavior.
- Resume hashing and compatible-cache reuse.
- GUI selection preservation on actual cases, with relevant AppTests and visual inspection where feasible.

Do not repeatedly run the entire suite after unchanged code. Run relevant suites once integration is stable; rerun only for new changes/failures. Log what was actually executed and passed.

## 9. Final deliverables and decision rules

One concise `review.md` must answer:

1. Does excluding Hawaii materially change the prior Gemini/GPT-OSS conclusions?
2. Which apparent format failures contain valid probabilities, and how does a consistently applied extraction rule change coverage/results?
3. Why are 51 original comparisons excluded at zero shift? Is there a bug or a deliberate estimand/support difference?
4. What is known versus unknown about CFBD wallclock semantics?
5. Did one baseline finish? If not, what exact operational blocker remains?
6. Can the user navigate the 12-case queue without losing the selected game/window?

Link source/config/cohort/parser versions, numeric tables, raw examples, screenshots/deep links, commits, tests and exact commands. Include strict and probability-usable results separately; original and input-screened results separately; random and target-enriched strata separately. Attach counts and uncertainty to claims. Do not describe pilot/template development as held-out evidence.

Choose the single next scientific experiment only after reviewing these results. If cached screening/extraction materially changes the negative result, repair measurement and validate on new games before broader conclusions. If it does not, the negative market-update finding becomes more robust to those particular explanations, while historical availability/outcome validity remain unresolved. If zero-shift replay is defective, repair controls before discussing temporal precision. A new baseline that completes and loses to persistence extends the protocol-specific finding; it does not establish a universal model limitation.

The highest-value success is a trustworthy, inspectable comparison and a clear next decision—not a longer model list.

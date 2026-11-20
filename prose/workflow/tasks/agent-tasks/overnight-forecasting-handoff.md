# Overnight handoff: chronology audit, unified inspection GUI, and baseline recovery

Date: 2026-10-02, America/New_York. The user will be offline for approximately nine hours, until roughly 09:00 EDT. Set actual deadlines relative to launch time; leave the final hour for integration, verification, and reporting.

## 1. Objective and execution order

By morning, the user should be able to inspect a single game and window across market observations, commentary, Jev, hindcast predictions, and the exact LLM requests without losing their selection. We also need to know whether the Hawaii–San Jose State contradiction is isolated or reflects a systematic timeline defect, and recover the existing Inkling/Llama/Qwen baseline roster where feasible.

Success means concrete evidence, working inspection tools, and accurately reported coverage. A job exiting successfully does not mean an experiment completed successfully.

Priority:

1. Reproduce and trace the Hawaii chronology contradiction; measure the scope of similar defects.
2. Unify GUI selection and make exact prompts easy to inspect.
3. Diagnose and repair baseline access/format/rate-limit failures; run bounded smoke tests immediately.
4. Resume baseline inference only after the chronology gate below, and only on a documented eligible cohort.
5. Audit Jev exclusions and run a small controlled prompt-presentation experiment if dependencies and time permit.

Do not add additional model families, start RLVR, fit a lag correction, or rebuild the entire pipeline overnight. Preserve existing scientific results and raw artifacts.

## 2. Context you must understand before coding

The project pairs sports commentary and structured game states with Polymarket observations. The current hindcast harness makes independent left/right endpoint calls asking for event-win probability. The model update is p_right - p_left; the observed market update is m_right - m_left. Update MAE measures market-update association under this protocol. It does not establish outcome forecasting skill or superior probabilities relative to the market.

The pilot has 150 windows, three strata, and six sports. Previous runs included Gemini, Gemma 27B/4B and GPT-OSS. Random-window updates from Gemini and GPT-OSS lose to persistence under the reported paired comparisons. GPT-OSS has some extreme changes; these include genuine response-comprehension failures and at least one contradictory input. Do not attribute all errors to model noise.

The current arms are:

| Input | EVIDENCE | LAGGED_ANCHOR |
|---|---|---|
| Structured state | Present | Present |
| Cumulative commentary | Present | Present |
| Market anchor | Absent | Same m_left supplied at both endpoints |

This contrast tests adding the stale market anchor. It does not isolate commentary value. Anchor-induced context compression can change the evidence as well; inspect and report this separately.

Jev produces interval-level expected market direction, compared with observed direction. Existing cached output may not contain event-level citations. Do not invent event-to-tick causal attribution from interval membership or an empty citation field.

Identity verification, historical publication availability, and rule-mapped settlement evidence are separate dimensions. An event-occurrence timestamp is not a certified publication timestamp. Current outcome metrics remain unsupported without valid labels.

Known paths, relative to the forecasting repository:

- `artifacts/experiments/llm_vs_crowd/2026-10-01-020000_overnight-hindcast/requests/gpt_oss.jsonl`
- `artifacts/experiments/llm_vs_crowd/2026-10-01-020000_overnight-hindcast/jev_strict_lag.json`
- Prior experiment directories: `2026-09-30-190000_hindcast-qa-jev` and `2026-09-30-214900_hindcast-jev-audit`.
- Reports: `morning_review.md`, `alignment_audit.md`, `operations.md`, `coverage.csv`, `comparisons.csv`, `magnitude_summary.csv`, `magnitude_extremes.csv`, and `strict_jev_lag.csv`.

Locate actual source locations and artifact conventions; these names are discovery clues, not permission to assume files or schemas exist. Reuse `artifact_dir(__file__)` if that is the repository convention. Read applicable `AGENTS.md` before editing. Do not print credentials or `.env` contents.

### The concrete Hawaii reproducer

Request hash supplied by user:

`70ea6891dcb21e6dc520f28471d3eefac2e2a945860f627b8dc838875291004c`

Window begins `2025-11-02T06:28:06Z` and ends approximately `06:29:07Z`. Target is Hawaii winning against San Jose State. Snapshot reports period 3, 1:57 remaining, San Jose State ahead 38–21. Supplied history includes:

| Event ID | Assigned UTC | Sporting state |
|---|---|---|
| `401760403855` | 06:18:09 | Period 4; field goal; 45–38 |
| `401760403873` | 06:25:09 | End of fourth quarter; 45–38 |
| `401760403639` | 06:27:22 | Period 3; incomplete pass; 38–21 |

The assigned timeline puts fourth-quarter completion before a third-quarter play. Prior inspection also found endpoint prompts identical apart from cutoff timestamp; verify this from saved messages rather than assuming it.

Sorting by period could improve presentation while retaining future sporting evidence at a third-quarter cutoff. Sorting IDs lexicographically is not a valid chronology repair without a provider guarantee. Find the first incorrect transformation.

## 3. Parallel ownership and integration

Use parallel agents if your environment supports them; otherwise execute the same workstreams as independent modules. The user explicitly requests modular parallel work. Assign one coordinator and at most five bounded workers. Use isolated worktrees/branches if supported, or explicit file ownership in a shared checkout. Do not have two agents edit the same file concurrently.

| Workstream | Ownership | Dependencies | Required output |
|---|---|---|---|
| A: chronology | Raw-source trace, pure semantic checks, cohort flags | Repository discovery | Reproducer, root-cause evidence, scope tables, eligible-cohort manifest |
| B: GUI | Shared selection controller, exact-prompt inspector, coordinated panels | Agree on IDs with coordinator; consume A flags | Working unified inspection flow, regression checks, screenshots |
| C: baseline recovery | Provider adapter/configuration, error classification, resumable runners | Smoke tests independent; full inference gated by A | Inkling/Llama/Qwen diagnostics and bounded results |
| D: Jev eligibility | Existing exclusion logic and retained/excluded analysis | Frozen artifacts | Reason audit, stage denominators, fixed-cohort controls where feasible |
| E: prompt experiment | Clarified template and fixed development experiment | A clean inputs; C usable adapter | Versioned template, bounded matched comparison |
| Coordinator | Interfaces, integration, verification, morning report | All workers | Integrated commit(s), command inventory, honest status ledger |

Agree on contracts before parallel editing. Suggested contracts, adapted to existing types:

- Inspection selection: canonical game ID, market ID, optional window ID, run ID, model, arm, endpoint, request hash.
- Integrity finding: game/window/request scope, check name, severity, source references, evidence, applicability, unresolved cause if applicable.
- Request result: request/config hashes, attempt ID, provider/model, status, finish reason, usage, raw output reference, parsed fields, validation failures.
- Eligibility manifest: parent run/version, rules/version, eligible IDs, exclusions with explicit reasons, input hashes.

Do not invent a large generic framework. Prefer a few deep modules that own their decisions: chronology validation, request execution, saved-request loading, and selection resolution. UI code should consume these interfaces without duplicating joins, parsing, or provider rules.

## 4. Workstream A — chronology and evidence selection

### A1. Reproduce and trace the exact request

Find the saved request by hash. Confirm target mapping, endpoint, raw source identity, and any compression. Persist a compact replay bundle containing saved messages, source references, snapshot, relevant raw events, and transformation versions. Do not modify the original file.

Build a trace table for all Hawaii events involved in the regression and a few neighbors:

- Raw event ID, source array position, and provider sequence/order field if available.
- All relevant raw timestamp fields, exact values, and evidence for their meaning.
- Period, clock, home/away scores, description, terminal status.
- Parsed timestamp and normalized UTC, plus the function and rule producing it.
- Snapshot selection rule and selected source event.
- Inclusion at left/right endpoint and the predicate that admitted each event.
- Prompt serialization position.

Trace the snapshot independently: determine whether its construction orders events differently from history selection. Inspect raw source fields before normalized tables. If documentation is insufficient to identify timestamp semantics, mark them unknown rather than inferring occurrence time from a plausible field name.

Classify where inconsistency first appears: raw source, parsing, UTC normalization, ordering, snapshot selection, evidence selection, or prompt rendering. Provide exact code/source references and a minimal reproducer. A hypothesis is not a confirmed root cause.

### A2. Implement sport-aware semantic checks

Use deterministic pure functions over existing records; separate detection from repair. Each check must return applicability and evidence. Unsupported sports/fields are unknown, not passes.

Minimum checks:

1. Period regression along the assigned UTC sequence for sports with ordered periods.
2. Clock increase within a period where the clock counts down, allowing documented corrections, tied timestamps, and boundary exceptions.
3. Terminal-game event followed by a nonterminal state. End of a quarter alone is not game termination; overtime and correction records need explicit handling.
4. History state later than the endpoint snapshot in sporting progression.
5. Snapshot score/team ownership inconsistent with its referenced event or canonical target mapping.
6. Duplicate event IDs with conflicting values and ambiguous same-time ordering.
7. Endpoint sporting progression inconsistent with endpoint wall-clock progression.

Do not apply football clock assumptions to baseball/tennis. For baseball use inning and half-inning when present; for tennis distinguish set number from provider period conventions. Keep sport/provider adapters explicit and small.

Score decreases can indicate corrections and are review candidates, not automatically invalid games. Likewise, later commentary can reflect a stale snapshot rather than demonstrable future inclusion. Preserve those distinctions.

### A3. Measure scope and decide the inference gate

Run on all frozen 150 pilot windows and the Jev cohort where inputs permit. If inexpensive, scan the larger normalized dataset too, with progress/checkpointing. Report unique games, windows, endpoints, and requests separately; six affected requests are not six independent games.

Output counts by sport/provider/check/severity and a deterministic review sample. Eligibility rules must depend on inputs, not model error or whether filtering improves metrics.

Gate for full inference:

- If the contradiction is an isolated input defect, quarantine demonstrably inconsistent windows in a versioned manifest and resume the existing roster on the objectively screened subset. Preserve the original cohort and report lost coverage.
- If a reproducible upstream repair is justified, implement it narrowly with regression checks; regenerate affected inputs into a new version and document changed IDs/hashes. Smoke-test repaired requests before running them.
- If timestamps are broadly untrustworthy or the cause remains unresolved, stop full inference on affected sources. Continue API diagnostics, GUI work, and offline analyses. Report this as a data gate, not an API failure.
- Unknown historical publication availability remains a separate status even when semantic checks pass.

Never repair alignment by simply sorting periods, swapping scores to match responses, inventing wall-clock times, or shifting commentary to maximize market agreement.

### A4. Required verification

Regression fixtures must catch the Hawaii period-4-to-3 sequence and distinguish a valid period transition/overtime case. Include an ambiguous duplicate-timestamp case and at least one non-countdown sport. Verify that flagging does not mutate source records. If a repair exists, verify both expected sporting chronology and wall-clock selection against independently supported source semantics.

## 5. Workstream B — unified GUI and exact prompt audit

The screenshots show two inspection workflows with separate selectors. Temporal inspection uses canonical game selection; hindcast uses a case selection. Switching changes the inspected game. A Prompt browser tab and timeline-layer controls already exist: extend them rather than creating another app.

### B1. One shared selection state

Create a single selection resolver keyed by canonical game and market, with optional window/run/model/arm/endpoint/request. Keep this state in the existing session mechanism and deep links where supported.

Required behavior:

- Switching temporal/hindcast/prompt/Jev panels preserves game, market, interval and zoom.
- Choosing a hindcast case updates the shared game/market/window once; all panels follow.
- Changing model or arm preserves the window.
- Selecting a game without a hindcast case shows “No hindcast case for this game” and retains the game.
- A filter that hides the active case reports that fact; it does not silently switch games.
- Restore old URLs where feasible through explicit ID mapping; never join by display names or array positions.
- Preserve game-level and request-level review records separately. Existing reviews must remain intact.

Avoid a top-level architecture rewrite. Put selection resolution in one module; panels receive resolved objects. Fetch/load selected detail lazily rather than loading every prompt on each rerender.

### B2. Exact saved-prompt inspection

Provide two clearly labeled views:

1. **Exact request:** messages in saved role/order, content rendered with actual newlines, copyable raw JSON, request hash and configuration. This must load archived messages, not regenerate a prompt with the current template.
2. **Structured inspection:** parsed events with original prompt position, source sequence, UTC, period/clock, named score ownership, event IDs and integrity flags. Alternative sorting is allowed only with an explicit label and an easy return to original order.

Render text as text, not executable HTML. Preserve message boundaries and all content. Do not turn literal backslash-n sequences into newlines unless the stored JSON actually decodes them that way. Verify the exact request against its archived representation/hash convention.

Include left/right message comparison showing added/removed/changed evidence, snapshot and anchor fields; distinguish metadata-only cutoff changes from sporting-evidence changes. Show compression/truncation metadata where available. Add raw response, parsed forecast, finish reason, validation errors, and provider/configuration provenance.

For Hawaii, the user must be able to see third-quarter snapshot versus fourth-quarter history immediately and navigate to the source trace.

### B3. Coordinated timeline and sensible defaults

Show market observations and commentary markers by default when data exist. LLM predictions/Jev judgments remain toggleable; include a hide-judgments option for less biased manual review. Use a shared time axis and target orientation. Forecasts are discrete endpoint markers; do not interpolate a continuous forecast trajectory across independently sampled windows.

Keep original timestamps visible. A user-selected visual shift is a display setting, not a source-data correction. Indicate it clearly in the plot and review record.

Jev displays interval scope and the commentary included in that interval. If archived Jev output has no citations, show “Event citations were not recorded.” Do not label interval events as attributed causes.

### B4. Acceptance checks

- Open Hawaii by request hash/deep link; inspect exact left/right requests and raw responses.
- Switch every panel; game/market/window stay fixed.
- Change model; window stays fixed.
- Select a temporal-only game; no automatic substitution occurs.
- Filter away the active case; retained selection and warning are correct.
- A nonzero display shift is visibly labeled and leaves source timestamps unchanged.
- Saved reviews and existing URLs are preserved or explicitly migrated with backup.
- No endpoint causes inference or network calls merely by browsing.

Run meaningful selection/join regression tests and one end-to-end manual browser check. Provide screenshots of unified Hawaii inspection and a game without hindcast coverage.

## 6. Workstream C — recover existing baseline shards

### C1. Diagnose before retrying

Produce one error ledger with every prior failure classified as provider/API, rate limit, quota, truncation, JSON syntax, schema/type, evidence-ID validation, refusal, or unattempted. Save raw outputs and precise validator errors. Distinguish missing forecasts from attempts that never happened.

Smoke results previously reported: Inkling 0 usable endpoints; Llama 7/24 smoke requests failed `invalid_format`; Qwen rate-limited. Verify against current artifacts.

### C2. Inkling paid endpoint

Replace `thinkingmachines/inkling-small:free` with `thinkingmachines/inkling-small` in explicit run configuration. The user demonstrated a successful basic curl completion with the paid ID. Credentials already reside in `.env`; never include them in logs or artifacts.

Run actual forecasting requests, not only a coin prompt. Start with 12 representative endpoints covering all six sports, both arms, and short/long prompts. Record actual resolved model/provider, finish reason, usage, schema validation and latency. Diagnose failures; do not route back to the free agentic endpoint.

User authorizes this paid endpoint for the overnight task. Set a conservative total OpenRouter spending guard of $10 for newly launched requests, or honor any lower existing project limit. This is a run limit, not a price assertion. Estimate prompt/output cost from current route pricing and actual token counts, persist the estimate, and include bounded retries and outstanding requests in the guard. If cost cannot be bounded reliably, complete smoke diagnostics and leave full paid inference blocked with a concrete reason. Do not request another spending confirmation overnight.

### C3. Llama format recovery

Inspect all seven failed smoke outputs. For each save finish reason, content length, token usage, raw body, parse/schema error and citation failures.

Raise output cap only where truncation is supported by evidence. If appropriate, test 4096 tokens against 2048 on the same failed prompts; do not change unrelated settings simultaneously. If the actual model/provider supports JSON schema, use it with required parameter support rather than accepting silent parameter omission. Otherwise use explicit JSON instructions and a small schema.

Keep probability extraction status separate from auxiliary citation/rationale validity. A numeric probability may be parseable when evidence IDs are invalid; preserve both facts and define metric eligibility consistently across models. Do not count missing/invalid numbers as zero probability. Do not silently change validation policy for Llama alone.

Any deterministic parser repair must be narrow, documented and uniformly applied. Do not call another LLM to repair the answer or manually salvage probabilities without recording a separate status.

Rerun a 24-endpoint representative smoke set, including the prior failures and long prompts. State coverage, not just a pass/fail gate. Configuration changes get new hashes; retained unchanged successes may be reused only when configuration/input hashes match.

### C4. Qwen rate-limit recovery

Read the actual 429 response and headers. Identify request-rate/token-throughput exhaustion versus quota. Begin at concurrency one and use one shared provider limiter across workers. Honor retry delays where supplied; use bounded exponential backoff with jitter otherwise. Persist retry state and stop reason. Do not rotate credentials to evade limits.

Smoke at most 12 representative endpoints before deciding whether a full shard can finish. Quota exhaustion should stop instead of retrying for hours. Provider switches require a distinct recorded serving configuration and separate results.

### C5. Full-run limits and analysis

After A's gate, resume at most the existing 150 windows per model, two arms, two endpoints: 600 logical endpoints per model, across Inkling, Llama and Qwen. Smoke calls overlap/reuse these requests when hashes match. Set maximum attempts per request to three, including the original new-run attempt. Do not create an unbounded retry queue.

Use a new run directory and immutable parent references. Original-template clean-cohort results, repaired-input results, and clarified-template development results must remain separate configurations. Report pair coverage as well as endpoint coverage. Never combine model/provider aliases merely because their display names look similar.

For completed data, reuse existing paired game-cluster analysis with fixed seed and 5,000 draws if inexpensive. Compare to persistence by stratum on identical eligible windows; include window-weighted and game-balanced sensitivity, games/windows, usable-pair coverage, and model comparisons on common-success pairs. Tiny cohorts get descriptive summaries, not misleading zero-width bootstrap claims.

## 7. Workstream D — explain Jev exclusions

Read `jev_strict_lag.json` and the functions producing `incompatible_spacing`, `nonincreasing_observations`, and every other reason. Do not infer definitions from names.

Report a stage table: original targets → valid original comparison → eligible per shift → fixed shared-shift intersection. For each stage include input denominator, retained windows/games, excluded count, and reason definitions. Specify whether reason counts overlap or have a precedence rule.

For each major reason inspect five deterministic examples, or all if fewer than five. Include raw observation sequence and expected eligibility behavior.

- Distinguish unsorted observations from duplicate timestamps and genuinely decreasing observation time.
- Do not average/drop conflicting duplicate observations without a documented semantic rule.
- Explain required spacing and whether it concerns base data or shifted-control construction.
- Keep base-data failure separate from lack of coverage for all shifted controls.

Compare retained/excluded cohorts by sport/provider/stratum and available market-activity or quote-age summaries. Record unavailable dimensions rather than inventing proxies.

If feasible from cached data, add empirical majority/always-flat baselines and class distribution on the existing 67-window fixed intersection. Run different-game permutations only if valid assignments can be built while preserving required coverage/constraints; document those constraints. Do not reuse the full cohort's majority baseline for this subset.

Changing an objectively wrong eligibility implementation requires a reproducer and a new analysis version. Do not relax rules to improve Jev agreement or pick the most favorable lag. Existing lag results are exploratory; multiple comparisons and varying coverage limit precision.

## 8. Workstream E — bounded prompt-presentation experiment

This is secondary to evidence repair and the GUI. Do not launch it on contradictory inputs. If time is short, deliver a tested template and frozen experiment manifest without claiming experimental results.

Build a clarified template with:

- Explicit forecast objective and target-team/home-away mapping.
- Named team scores, period semantics and unambiguous clock wording.
- Deterministically derived target leading/trailing statement, checked against named scores.
- Clearly separated structured snapshot, history, optional anchor and output schema.
- Anchor timestamp and explicit stale-at-right-endpoint meaning.
- Original validated evidence order and stable evidence IDs.

Example for the snapshot only: “Target: Hawaii wins. Hawaii is away; San Jose State is home. Period 3 of 4 regulation quarters; 1 minute 57 seconds remaining in this quarter. Hawaii 21; San Jose State 38. Hawaii trails by 17.” Never attach this header to unresolved period-4 history and call the input repaired.

Choose 24 semantically screened development windows before viewing new outputs, balanced across six sports where eligible. Use Gemini and GPT-OSS if already configured and usable. Test EVIDENCE only initially to avoid mixing anchor effects. Same source evidence, cutoff, compression policy, output schema and decoding configuration; only presentation changes. Use paired original/clarified templates at both endpoints: at most 192 logical requests (24 windows × 2 models × 2 templates × 2 endpoints), excluding exact hash-matched cached reuse.

Changing informational content, chronology or evidence compression invalidates a presentation-only claim; either hold those fixed or label a separate repair experiment. Save the exact input difference and context lengths.

Report orientation/period interpretation errors with linked raw responses, parse/schema coverage, paired update-MAE differences, predicted-update magnitude summaries, and games/windows. Do not auto-label rationales correct using superficial keyword matches; use deterministic contradictions where possible and mark remaining semantic judgments pending human review.

This is development evidence on inspected pilot data. Do not describe it as held-out improvement, choose among many templates by test score, or overwrite prior baseline numbers. A commentary ablation is outside this overnight scope.

## 9. Long-running scripts, tmux, and resource hygiene

Every task with waiting must have a self-contained resumable command and run inside tmux. Reuse the existing harness behind a thin entrypoint; do not create several independent retry/serialization implementations.

Required runner properties:

- Explicit config/manifest input, absolute artifact resolution, deadline, spend/attempt/concurrency limits.
- Checkpoint after each completed request; append-only attempt records and atomic summary writes.
- Resume by exact input/config hash; no duplicate work or concurrent writers to the same shard.
- Bounded network timeouts, retries and rate limiting; graceful SIGINT/SIGTERM handling.
- Heartbeat/progress with attempted, successful, invalid, failed, unattempted, in-flight and remaining counts, plus last error and next retry time.
- Final structured stop reason: complete, deadline, quota, budget, data gate, provider unavailable, or crash.
- Coordinator status separate from scientific coverage. Export child PID/session ownership and stop only processes this run created.

Provide actual commands after implementing the entrypoint. A template, to adapt rather than copy blindly:

```bash
tmux new-session -d -s forecast-overnight-20261002 \
  'cd /actual/repository && ./scripts/run_overnight_20261002.sh'
tmux attach -t forecast-overnight-20261002
```

The wrapper must establish the project environment, load credentials privately, launch bounded tasks, and write logs/status. It should not print secrets or rely on an interactive shell remaining open. Retain the tmux session/logs for inspection. Supply status, resume and graceful-stop commands in the morning report.

Set inference deadline at approximately launch + 7.5 hours; leave time to aggregate/report. If work finishes early, do not fill the remaining hours with extra models or repeated tests. Continue the authorized audit/integration tasks instead.

The host is a Framework 13 with limited RAM and a prior runaway/high-temperature process incident. These are API experiments: do not launch local model inference, unbounded worker pools, busy polling, or CPU-heavy whole-dataset jobs without limits. Resource checks should be lightweight. Do not kill unrelated preexisting processes. A venv is an environment, not an independently running process.

## 10. Engineering and artifact rules

Apply Ousterhout-style design concretely:

- Keep complexity inside small, cohesive modules; do not distribute timestamp/provider decisions throughout UI and scripts.
- Define stable interfaces and errors at boundaries; UI panels should not know provider response formats.
- Reuse existing joins, hashing, schema validation and statistical code where correct.
- Write comments explaining timestamp semantics, cohort decisions and invariants, not narrating obvious code.
- Avoid broad refactors, speculative abstractions, duplicated compatibility wrappers and silent fallbacks.
- Add meaningful regression tests for observed failures and integration boundaries; avoid tests that simply mirror code.

Do not overwrite raw source, frozen request/response files, existing reviews or previous reports. Store new outputs in a versioned experiment directory consistent with the repo. Record source commit, dirty-tree state when relevant, input hashes, configuration, seeds and cohort rules. Keep generated outputs in established artifact locations rather than accidentally committing large files or secrets.

Every repair needs: failing reproducer, expected behavior, implemented change, affected IDs, and validation result. Every unresolved issue needs a specific blocker and reproducible next action. Commit coherent changes with an integration summary; do not push/deploy/send messages unless separately authorized by repo instructions/user scope.

## 11. Morning deliverables and acceptance

Produce one entrypoint report `morning_review.md` linking all new artifacts and deep links. Its first section must answer:

1. What caused the Hawaii contradiction, or what precise step remains unresolved?
2. How widespread are semantic violations, and which sources/cohorts are affected?
3. Can the user inspect one game across all GUI panels without losing selection?
4. Which baselines actually ran, with endpoint/pair coverage and exact stop reasons?
5. What do Jev exclusions mean, and do they preferentially retain particular sources/windows?
6. Did clarified presentation change errors on the bounded development cohort, if run?

Required artifact groups:

| Artifact group | Contents |
|---|---|
| Chronology | Hawaii replay/trace, invariant definitions, scope tables, review cases, repair or unresolved-root-cause note |
| Cohorts | Frozen parent and new eligibility manifests, explicit exclusion reasons, changed-input hashes |
| GUI | Working launch/deep-link instructions, verification results, two screenshots |
| Models | Error ledger, smoke outcomes, run configurations, raw attempts, coverage, cost/usage, stop reasons |
| Jev | Stage denominators, exclusion examples, retained/excluded composition, subset baselines where feasible |
| Prompt | Template version and input-difference example, experiment manifest/results or explicit not-run status |
| Integration | Commit(s), commands to reproduce/analyze/resume/stop, test outcomes and remaining issues |

Include these five scientific hypotheses with supported/weakened/unresolved status and evidence:

- A pipeline defect materially contaminates endpoint evidence.
- Response comprehension/elicitation contributes to large update errors independently of contradictory inputs.
- Failure against persistence extends to additional model families on screened random windows.
- Adding the stale market anchor changes update tracking; commentary's incremental value remains separately untested.
- Jev association persists under justified shared-window controls; temporal precision remains separately assessed.

For model summaries, include median/p90/max absolute predicted update, mean signed update error (predicted minus observed update), and worst-five share of total absolute error by model/arm/stratum. Keep extremes in the main results. For n < 5 label the concentration statistic accordingly; if total error is zero, the fraction is undefined.

Show original full-cohort results and consistency-screened sensitivity results separately. Input repairs and presentation changes are separate versions, not interchangeable filtering. Report uncertainty and retained counts; a broad interval is unresolved evidence, not equivalence.

End with the strongest defensible paper claim and one next experiment. Do not claim certified alignment merely because no check fired; certification requires evidence about identity and timestamp semantics. Do not claim outcome forecasting skill without rule-mapped settlement labels. The overnight work is successful even if it establishes a concrete data blocker and stops further inference, provided it makes that blocker inspectable and reproducible.

# Project instructions

## Purpose and scope
<!-- Fill in 1-3 sentences: what the project does, its primary success criterion,
and any important non-goal. Avoid a repository inventory or aspirational roadmap. -->
<PROJECT_PURPOSE_AND_SUCCESS_CRITERION>

This is an educational project for the Georgia Tech Deep Reinforcement Learning course. 
The goal of the project is to: 

- Replicate the VinePPO paper, and 
- TODO: ablate it against an interesting variant of said paper 

The final deliverable is a 2~4 page conference paper style write up, evaluating the proposed methods and baselines. 

## Working commands
<!-- Supply exact commands from the repository root, including the environment
runner. Include a cheap, representative smoke test. Remove unused rows; never
invent a command to complete the table. State external prerequisites if needed. -->
| Action | Command |
| --- | --- |
| Set up environment | `<SETUP_COMMAND>` |
| Format / lint / type-check | `<STATIC_CHECK_COMMAND>` |
| Run focused tests | `<FOCUSED_TEST_COMMAND>` |
| Exercise one real path | `<SMOKE_COMMAND>` |

## Sources of truth and boundaries
<!-- Identify maintained sources versus generated/installed copies. Name only
boundaries that affect edits. Link one representative implementation if useful.
Document actual enforcement separately from behavioral instructions. -->
- Edit <MAINTAINED_SOURCE>; regenerate <DERIVED_OUTPUT> with <COMMAND>.
- Keep <PROVIDER_OR_IO_MECHANICS> inside <BOUNDARY>; callers use <INTERFACE>.
- Follow <REPRESENTATIVE_IMPLEMENTATION> for <SPECIFIC_KIND_OF_CHANGE>.
- Treat <CANONICAL_SPEC_OR_CONFIG> as authoritative for <DECISION>.

## Domain invariants
<!-- Replace these example-shaped bullets with 2-5 concrete project rules.
Useful subjects: units, tensor shapes, timestamp meanings, evidence cutoffs,
identity keys, missing-value semantics, numerical tolerances, data splits.
For each invariant, name a check or say explicitly that verification is manual.
Delete this section if there are no non-obvious domain rules. -->
- <INVARIANT>; verify with <CHECK_OR_MANUAL_PROCEDURE>.
- When <AMBIGUOUS_OR_INVALID_INPUT>, <REJECT_OR_MARK_EXPLICITLY>; preserve
  <SOURCE_EVIDENCE> so the decision can be inspected.
- Derive expected test results from <INDEPENDENT_SPEC_OR_VERIFIED_EXAMPLES>.

## Change style 

I value robustness, minimizing total lines of code, avoiding bloat, avoiding duplication.
I hate fallbacks, useless if-else statements (things like "elif code is broken then repair it" should not exist, it should fail loudly), and I love asserts, which allow to catch early bugs. Fallbacks hide bugs and bloat the code.

## Change discipline
<!-- Customize only where the project's needs differ. Keep this section short;
formatting preferences belong in tool configuration when enforceable. -->
- Inspect the relevant implementation and existing diff before editing; preserve
  unrelated work. Keep each change centered on one reviewable behavior.
- For changes to interfaces, data semantics, dependencies, or execution flow,
  state the proposed approach and validation briefly before implementation.
  Proceed within the authorized scope; routine edits need no approval checkpoint.
- Reuse existing mechanisms. Add an abstraction, dependency, configuration option,
  or fallback only when a current requirement justifies it; explain that need.
- Diagnose the owning cause before adding another special case. If evidence
  contradicts the approach, revise it rather than accumulating speculative fixes.
- Comment non-obvious constraints and reasons; keep descriptions consistent with
  the code. Update the canonical documentation when behavior or contracts change.

## Validation and completion
<!-- Name acceptance criteria specific to this project. Keep fast checks above;
link expensive validation here with an explicit trigger. -->
- For a bug fix, reproduce the failure before changing behavior when feasible,
  and retain an appropriate regression check. Test observable requirements.
- Run checks appropriate to the affected behavior. For cross-boundary changes,
  exercise <REPRESENTATIVE_END_TO_END_CASE> and inspect <IMPORTANT_OUTPUTS>.
- Preserve acceptance criteria while fixing code. If a test, fixture, tolerance,
  or validator is wrong, explain the independent evidence for changing it.
- Report changed behavior, validation performed, and unresolved limitations.
  Distinguish checks that passed from checks not run and explain omissions.

## Execution and data permissions
<!-- Fill in actual paths and numerical budgets. Authorization can be given in
the task or standing policy; do not require repeated approval already granted.
For important boundaries, configure enforcement outside the agent's control. -->
- Ordinary development may write <EDITABLE_PATHS> and run <ALLOWED_VALIDATION>.
- Preserve <PROTECTED_DATA_RESULTS_OR_FIXTURES>; write trial outputs to
  <SCRATCH_OUTPUT_LOCATION>. Enforcement: <READ_ONLY_MOUNT_OR_OTHER_CONTROL>.
- Routine runs are limited to <TIME_MEMORY_WORKER_GPU_OR_API_COST_BUDGET>.
  Work beyond this budget requires authorization covering the additional use.
- Deletion or overwriting of retained artifacts, stopping unrelated jobs,
  deployment, and system changes require authorization covering those actions.
- Before an authorized cleanup, inspect current targets and verify
  <DISPOSABILITY_CRITERIA>. Prior successful cleanups do not establish eligibility.
- If completion requires crossing an unauthorized boundary, report the evidence
  and proposed action. Continue useful work that stays within scope.

## Task-specific references
<!-- Keep only current documents. Use explicit triggers, not "read all docs".
Replace the examples below; delete the section if no references are needed. -->
- When changing <DATA_OR_SCHEMA>: read <PATH> for <CONTRACT_AND_LINEAGE>.
- When running <EXPENSIVE_EXPERIMENT>: read <PATH> for <PROCEDURE_AND_BUDGET>.

<!-- MAINTENANCE: Review this file when workflows change. After a failure,
prefer a code fix or automated check where possible; add a permanent instruction
only for a durable lesson. Check for conflicts with global and scoped rules.
Keep large logs in files and retrieve relevant excerpts rather than pasting them
into every session. Remove this comment after adopting any relevant guidance. -->

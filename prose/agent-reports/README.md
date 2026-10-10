# Agent reports.

Deliverables from agent runs — verification notes, ad-hoc analysis, responses to task
patches. Provisional by construction: the human reads, triages, and promotes substance to
where it belongs; the report stays as an audit trail.

**Agents: write here by default** when a task patch or prompt asks for a report and does
not name a more specific path. Read and link freely.

This is not [`knowledge-bases/dump/`](../knowledge-bases/dump/) — that tree holds unfiled
*claims about the world* that may graduate to the knowledge base. Agent reports are
*task output*, tied to a prompt and a moment in the repo.

## Conventions

- One file per report: `YYYY-MM-DD-HHMMSS_<topic>.md`. The date in the name is when the report
  was written — no `Status:` line is needed, and the task-note check deliberately does
  not cover this tree.
  NOTE: older reports do not contain the HHMMSS part of the filename. Please enforce this new convention. We have several agent reports produced on the same day sometimes; we want to keep things organized.
- Open with provenance, anchored backwards like everything else:
  `Responds to prose/workflow/tasks/patches/2026-09-02-1930_verifying-details.md at `f90b07e`; re: experiments/llm_vs_crowd/2026-08-31_multi-model-next-tick/.`
- Do not revise in place. A follow-up is a new dated file that links the old one.

## Where reports go after triage

| If the substance is… | Promote it to… |
| --- | --- |
| The canonical analysis of one experiment run | `experiments/<class>/<run>/findings.md` |
| What the human thinks the results mean | [`reflections/`](../reflections/) — **human writes only** |
| A durable claim about the world or the data | [`knowledge-bases/`](../knowledge-bases/) |

When a report is *about* one run and you already know it is the official record, write
directly to that run's `findings.md` instead of here.

Where this sits in the pipeline: [`workflow/tasks/`](../workflow/tasks/) (the brief) →
here (agent deliverable) → `experiments/*/findings.md` (filed per-run record) →
[`reflections/`](../reflections/) (human synthesis) → the knowledge base (established).
See [`prose/README.md`](../README.md) for the full routing table.

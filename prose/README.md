# README.

Everything written in words rather than code. Nothing here is hidden behind a dot prefix,
because notes you cannot `rg` for are notes you will not find.

## Which subdirectory?

| I want to write down… | It goes in |
| --- | --- |
| An error I just debugged, and how | [`debugging-log.md`](debugging-log.md) — newest entry at the top, dated, with the commit hash |
| Why a capability exists, and what it was like before | [`change-rationale.md`](change-rationale.md) — past tense, anchored to a commit *range* |
| Something I now believe about a paper, an API, or the data, that will still be true next month | [`knowledge-bases/`](knowledge-bases/) |
| The same, but I don't know where it belongs yet | [`knowledge-bases/dump/`](knowledge-bases/dump/) — file it away later |
| What I currently think the results mean | [`reflections/`](reflections/) — hand-written by the human only; agents read and link, never write |
| An agent's report on a task or verification | [`agent-reports/`](agent-reports/) — agents write by default; human triages and promotes |
| How code in this repo should be written | [`conventions/`](conventions/) — canonical; `CLAUDE.md` and `.cursor/rules/` import from here |
| A prompt I want to reuse | [`workflow/prompt/`](workflow/prompt/) |
| Files I am unlikely to change due to external constraints (usually derived from group meetings and course instructions) | [`workflow/stable/`](workflow/stable/) |
| Something to come back to | [`workflow/tasks/`](workflow/tasks/) |
| Task decompositions and rough approaches. (Subject to change) Compare with `workflow/stable/deadlines.md`. | [`workflow/tasks/plans`](workflow/tasks/plans) |
| Final submission: this is the main notebook to be submitted | [`submission`]|



When in doubt, append to `knowledge-bases/dump/` — deciding costs more than filing does,
and an unfiled note is worth more than an unwritten one.

Where *files* go, as opposed to prose, is in the [root README](../README.md).

# README.

Everything written in words rather than code. Nothing here is hidden behind a dot prefix,
because notes you cannot `rg` for are notes you will not find.

## Which subdirectory?

| I want to write down… | It goes in |
| --- | --- |
| An error I just debugged, and how | [`log.md`](log.md) — newest entry at the top, dated, with the commit hash |
| Why a capability exists, and what it was like before | [`changelog.md`](changelog.md) — past tense, anchored to a commit *range* |
| Something I now believe about a paper, an API, or the data, that will still be true next month | [`knowledge-bases/`](knowledge-bases/) |
| The same, but I don't know where it belongs yet | [`knowledge-bases/dump/`](knowledge-bases/dump/) — file it away later |
| What I currently think the results mean | [`reflections/`](reflections/) — hand-written by the human only; agents read and link, never write |
| An agent's report on a task or verification | [`agent-reports/`](agent-reports/) — agents write by default; human triages and promotes |
| How code in this repo should be written | [`conventions/`](conventions/) — canonical; `CLAUDE.md` and `.cursor/rules/` import from here |
| A prompt I want to reuse | [`workflow/prompt/`](workflow/prompt/) |
| Something to come back to | [`workflow/tasks/`](workflow/tasks/) |
| Files I am unlikely to change due to external constraints (usually derived from group meetings and course instructions) | [`workflow/stable/`](workflow/stable/) |


When in doubt, append to `knowledge-bases/dump/` — deciding costs more than filing does,
and an unfiled note is worth more than an unwritten one.

Where *files* go, as opposed to prose, is in the [root README](../README.md).

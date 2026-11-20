# README.

Directory containing *established* knowledge pertaining to the project.
That is, this should only contain knowledge which are likely to remain constant throughout the project.
This includes notes about papers, library documentations, etc.

This is knowledge about *the world and the problem*, not about how to write code — those
conventions live in [`../conventions/`](../conventions/). Errors and their debugging go
in [`../log.md`](../log.md).


## Here or upstream? The boundary.

Two things here look alike and are not, so the rule is worth stating once:

- **The vendored subdirectory** (`gtr2608_forecasting-kb/`) holds knowledge about *the
  world and the literature* — true regardless of this repo, and outliving it. It is
  read-only here; see below.
- **Everything else in this directory** holds knowledge about *this repo's dependencies*
  — the API quirk, the schema surprise, the undocumented rate limit. It lives here rather
  than upstream because it must stay editable **in the same commit as the code it
  constrains**, and because its claims are regression-tested by `tests/` in this repo.
  Splitting a claim from its test across two repos means the test can change and the note
  cannot notice.

The practical test: *if I discovered this while debugging `src/`, could I fix the note
without leaving this repo?* If the answer has to be yes, it goes here.
[`polymarket_api.md`](polymarket_api.md) is the worked example.

## Subdirs.

- `dump`: Notes to be filed away later to the appropriate parts of the knowledge base.
- `gtr2608_forecasting-kb`: **Read-only here — generated, do not hand-edit.** A vendored
  snapshot of the
  [gtr2608_forecasting-kb](https://github.com/tomohiro-sawada/gtr2608_forecasting-kb)
  notes repo. Refresh it by running
  [`scripts/bin/sync_knowledge_base.sh`](../../scripts/bin/sync_knowledge_base.sh), which
  mirrors upstream with `--delete` and records the source commit in `.upstream`.

  Editing it here does not work, and fails quietly. The sync refuses only on
  *uncommitted* edits; commit them and the guard passes, `--delete` reverts them, and the
  sync commit lands as a normal-looking "Sync … from upstream" diff. Nothing warns you.
  The snapshot also carries its own `CLAUDE.md` and `.cursor/rules/`, written for the
  upstream repo, which invite an agent to edit `wiki/` — the root
  [`CLAUDE.md`](../../CLAUDE.md) overrides them for this repo.

  Changes belong upstream, in a clone of that repo where its `bin/wiki lint` hook and
  `.tests/` suite run. Then sync the result back here.

Everything outside that subdirectory is written by hand. External knowledge bases each get
their own dedicated subdirectory, so a sync never touches the hand-written notes.

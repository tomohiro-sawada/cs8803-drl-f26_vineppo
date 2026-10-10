# Conventions.

The canonical, tool-neutral statement of how this codebase is written. Everything else
that states a convention — `CLAUDE.md`, `.cursor/rules/*.mdc`, `tests/README.md` — is a
pointer into this directory, so there is exactly one copy to keep true.

These are repo-invariant: they describe how *I* write code, not what *this project*
studies. Project knowledge (papers, API behaviour, dataset quirks) belongs in
[`../knowledge-bases/`](../knowledge-bases/); the running error log is [`../log.md`](../log.md).

| File | Covers |
| --- | --- |
| [`code_style.md`](code_style.md) | Ousterhout's *A Philosophy of Software Design*: deep modules, information hiding, pulling complexity downwards. |
| [`naming.md`](naming.md) | How to name things, and using naming difficulty as a design smell. |
| [`testing.md`](testing.md) | Test structure, designing for testability, and testing impure/ML code. |
| [`workflow.md`](workflow.md) | Committing, the error log, running experiments, and things never to do (`rm`, hyphens under `src/`). |
| [`system_specs.md`](system_specs.md) | Hardware and shell environments this code has to run on. |

For where files go — which directory owns what — see the [root README](../../README.md).
That is deliberately not duplicated here.

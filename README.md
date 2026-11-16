# README. 

CS8803 DLR Fall 2026 VinePPO 


## Setup

From the repository root, run:

```sh
uv sync
uv run python -c "import baselines; import benchmarking"
```

The root project installs the packages under `src` in editable mode, so source
changes are immediately available to imports. Do not run `uv add --editable
src/baselines`: `baselines` is part of this project, not a separate dependency.

## Repo structure.

Writing: 
- The main report is in `main.ipynb`. 
- `prose`: writing and dev logs written by Tom or by the agent

Code:
- Backend is in `src`. Subdirectories are meant to be imported as libraries. 
- Tests for `src`. Directory and test structures mirror the structures in `src`. 
- `scripts` include single use utility scripts, e.g. for setting up codebase, generating figures, etc. 














## Rules. 

- We are NOT allowed to use Qwen and DeepSeek in our experiments per cluster restrictions.

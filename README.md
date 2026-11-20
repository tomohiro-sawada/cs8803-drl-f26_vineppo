# README. 

CS8803 DLR Fall 2026 VinePPO 


## Setup

From the repository root, run:

```sh
uv sync
uv run python -c "from core.inference import HFInference, token_logprobs"
```

The root project installs the packages under `src` in editable mode, so source
changes are immediately available to imports.

## Optional accelerated-inference environment

The default `uv sync` selects `dev` + `sdpa` (Torch >=2.15), with no vLLM or
FlashAttention. To opt in without modifying `.venv`, run from the repository root:

```sh
.venv/bin/python scripts/setup/install.py --inference
# Preview resolution without installing:
.venv/bin/python scripts/setup/install.py --inference --dry-run
```

The installer uses `.venv-inference-glibc<VERSION>` for the **current host's**
GLIBC, Python 3.13, and the mutually exclusive `inference` group. vLLM 0.34.0
requires Torch 2.14.0; the common dependency floor allows that version, while the
`sdpa` group preserves the default Torch >=2.15 requirement. `uv.lock` records
both stacks. The flag belongs to the setup script, not to uv itself.

For example, on GLIBC 2.39, use the isolated environment explicitly:

```sh
.venv-inference-glibc2.39/bin/python -c 'import torch, vllm, vllm._C_stable_libtorch, flash_attn, flash_attn_2_cuda; print(torch.__version__)'
# If using uv run, select the same groups (otherwise uv selects SDPA):
UV_PROJECT_ENVIRONMENT=.venv-inference-glibc2.39 uv run --locked --no-group sdpa --group inference python your_script.py
```

Return to SDPA by using the original `.venv` / ordinary `uv run`; no runtime
backend fallback is added. Installing these packages does not replace
`HFInference` with a vLLM backend or change the model's attention implementation.

FlashAttention is pinned to upstream commit
`68bd095cdd338c5171bccb9c200529f3620b85c9` (version 2.8.4), which includes the
C++20 build fix required by Torch >=2.13. PyPI 2.8.3.post2 failed to compile
against Torch 2.14 because it explicitly selects C++17. Build prerequisites are
installed first, against the selected Torch. The upstream build may fetch a
GitHub wheel or compile locally; this can require substantial downloads, disk,
RAM, and time.
If the wheel has an incompatible GLIBC/ABI, explicitly request a source build:

```sh
# Run on the target compute-node OS with a matching CUDA toolkit/nvcc loaded.
.venv/bin/python scripts/setup/install.py --inference --build-flash-attn
```

This sets `FLASH_ATTENTION_FORCE_BUILD=TRUE`, bypasses the cache, and reinstalls
FlashAttention; `MAX_JOBS` defaults to 4 and can be set lower. A GLIBC error is
**not** caused by the absence of a GPU and must not be ignored. Separate paths
prevent cross-OS reuse but do not prove compatibility: check native imports and
GPU execution on each target node. Do not assume PyPI supplies GLIBC-specific
FlashAttention wheels; PyPI 2.8.3.post2 publishes only a source distribution.
A source build on a newer GLIBC host does not make it portable to older nodes.

Validation on `dave` in the existing `slurm_2` allocation (GLIBC 2.39, A40):
vLLM 0.34.0 and Torch 2.14.0+cu130 installed; vLLM's native
`_C_stable_libtorch` extension imported successfully, and a Torch CUDA tensor
operation passed. This is **not** an end-to-end vLLM generation test.
FlashAttention's pinned source build started but the Slurm step was cancelled
before completion; FlashAttention is **not installed/validated yet**. No GLIBC
error occurred. The base `.venv` was not modified by these installation attempts.

Resume on an allocated A40 with a CUDA toolkit loaded (allow sufficient build
time). The architecture setting below is A40-focused, not a universal GPU build:

```sh
FLASH_ATTN_CUDA_ARCHS=80 NVCC_THREADS=1 .venv/bin/python scripts/setup/install.py --inference
```

Then validate FlashAttention forward/backward kernels and vLLM generation before
using the stack for experiments. Do not run another installer concurrently in
the same environment.

## Step segmentation

VinePPO calls `step_boundaries_from_tokens()` in `src/core/segmentation.py`.
It decodes successive prefixes in prompt context and maps each character cut
to an index in the original sampled response. MATH moves cuts that fall
inside tokens to the next complete token boundary and merges coincident cuts.
The crossing token belongs to the earlier step. MC branching and PPO credit
use the same response-relative offsets. GSM8K uses exact newline cuts and
keeps the suffix from the first answer marker together. Sampled EOS belongs
to the final step.

## HF inference

`src/core/inference.py` uses the loaded decoder-only training model directly;
there is no tokenizer loading, model copy, backend fallback, or weight sync.
Given a loaded `model` and `tokenizer` (for example,
`realtreetune/rho-1b-sft-GSM8K`), pass exact, unpadded token prefixes:

```python
from core.inference import HFInference, token_logprobs

# Padding is explicit; EOS can also serve as padding. EOS termination IDs are
# taken from model.generation_config.eos_token_id.
inference = HFInference(model, pad_token_id=tokenizer.eos_token_id)
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
completions = inference.generate([prompt_ids], num_samples=3, max_new_tokens=256)[0]

# Branch without decoding/retokenizing the trajectory.
branch_ids = prompt_ids + completions[0].token_ids[:step_boundary]
branches = inference.generate([branch_ids], num_samples=3, max_new_tokens=128)[0]
```

Results are nested `[prefix][sample]`. Each `Completion` contains only generated
tokens, including the first EOS but excluding subsequent batch padding, and a
`"stop"` or `"length"` finish reason. Generated non-EOS padding IDs remain tokens.
The longest prefix plus `max_new_tokens` must fit the model context: generation
does not truncate inputs or silently shrink the budget. Callers choose the
remaining continuation budget when branching.

The default is temperature-1 sampling without top-k/top-p filtering, deliberately
different from the paper's temperature-0.6/top-p-0.9 recipe. Use `temperature=0`
with one sample for greedy evaluation. `return_logprobs=True` returns aligned
sampling-distribution log-probabilities (after temperature), including EOS.
For greedy decoding these are raw actor scores, not deterministic-selection
probabilities. This option retains vocabulary scores during decoding, so leave
it off when not needed.

For PPO, `token_logprobs(model, input_ids, attention_mask)` returns differentiable
float32 raw actor scores of shape `[B, T-1]`, aligned with `input_ids[:, 1:]`.
Apply a separate response/loss mask to exclude prompt and padding targets. Cache
old-policy scores under `torch.no_grad()` after **all** trajectory and MC sampling
and before any optimizer update; keep those scores fixed across update epochs.
Both APIs leave the model in evaluation mode to disable dropout; gradients still
work for scoring. Generation uses KV caching, whereas scoring disables it.
This initial scoring path does not use HF gradient checkpointing, which is
inactive in evaluation mode.

Run the offline, tiny-model inference tests with:

```sh
uv run pytest tests/core/test_inference.py -q
```

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

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
completions = inference.generate(
    [prompt_ids], num_samples=3, max_new_tokens=256
)[0]

# Branch without decoding/retokenizing the trajectory.
branch_ids = prompt_ids + completions[0].token_ids[:step_boundary]
branches = inference.generate(
    [branch_ids], num_samples=3, max_new_tokens=128
)[0]
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

# Fixed OpenRouter segmentation proxy

Generated 2026-11-18 using `meta-llama/llama-3.2-1b-instruct` via OpenRouter's
Cloudflare provider. This is a general instruction-tuned model of similar size,
**not RhoMath** and not a math-specialized equivalent. Outputs may be wrong or
degenerate; they are test data, not instructions or reference solutions.

`openrouter_proxy.json` retains six complete outputs from the 32-output corpus:
four successes (including a long GSM8K response) and both observed MATH failures.
Expected outcomes are snapshots of the current strict segmentation policy, not
human judgments of ideal reasoning steps. No outputs were edited to make them
pass. Original API request IDs are included for traceability.

Text was locally encoded using the pinned task-specific RhoMath tokenizer.
The stored token IDs/spans are **not sampled IDs from either model**. The offline
test exercises character splitting and exact token-cut validation with those
fixed spans; it does not validate the tokenizer itself or sampled-token
roundtripping. It requires no API key, network access, GPU, or model weights.
No tokenizer vocabulary or model weights are redistributed in the fixture.

The two failures demonstrate character boundaries inside the RhoMath tokens
`=\` and `$,`. They deliberately assert rejection rather than adding snapping,
step merging, or another implicit recovery policy.

Corpus metadata/provenance: `artifacts/segmentation/openrouter-llama32-1b-proxy/`.
The pulled directory contains aggregate metrics and provenance, not the full
`gsm8k.jsonl` / `math.jsonl` outputs. Only these six selected completions can be
replayed from the committed fixture. Generator:
`scripts/diagnostics/sample_openrouter_segmentation.py`.

## Offline replay checks

From the repository root:

```sh
.venv/bin/python -m scripts.diagnostics.check_segmentation \
  --input tests/fixtures/segmentation/openrouter_proxy.json
```

This prints JSON with per-record observed/expected outcomes, per-task totals,
input SHA256s, and offending token spans and surrounding text for inside-token
cuts. It exits **1** for any rejection, changed expectation, or result that cannot
be compared. The two known MATH rejections therefore make this strict check fail.
For a regression gate that accepts only unchanged recorded outcomes:

```sh
.venv/bin/python -m scripts.diagnostics.check_segmentation \
  --input tests/fixtures/segmentation/openrouter_proxy.json \
  --allow-known-rejections
```

This does not repair or suppress failures in the report. Any drift or
non-comparable result still exits nonzero. `--output report.json` exclusively
creates a report file instead of printing it; existing files are never overwritten.

When the full producer JSONL files are copied locally, pass them explicitly:

```sh
.venv/bin/python -m scripts.diagnostics.check_segmentation \
  --input artifacts/segmentation/openrouter-llama32-1b-proxy/gsm8k.jsonl \
          artifacts/segmentation/openrouter-llama32-1b-proxy/math.jsonl
```

Missing files, duplicate task/IDs, malformed records, and invalid span layouts
fail loudly. No aggregate-metrics or fixture fallback is used. Saved spans cannot
validate tokenizer correspondence, roundtrip failures, or special-token checks;
those expected outcomes are marked non-comparable. A recorded decoded/original
text mismatch is not replayed against the wrong text. This checker performs no
network/API requests and requires no tokenizer, model weights, or GPU.

Keep the live API generator outside normal tests. Review applicable dataset/model
and provider redistribution terms before publishing fixtures. No API credentials
are stored here. This deliberately selected fixture set is not suitable for
estimating failure rates; those counts belong to the full corpus report.

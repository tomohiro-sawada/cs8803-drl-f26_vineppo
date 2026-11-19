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

Full local corpus/provenance: `artifacts/segmentation/openrouter-llama32-1b-proxy/`.
Generator: `scripts/diagnostics/sample_openrouter_segmentation.py`.

Keep the live API generator outside normal tests. Review applicable dataset/model
and provider redistribution terms before publishing fixtures. No API credentials
are stored here. This deliberately selected fixture set is not suitable for
estimating failure rates; those counts belong to the full corpus report.

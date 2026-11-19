The shared abstraction is **a sequence of examples**, not a loader that knows whether it is training or evaluating.

```text
Dataset-specific loader                  e.g. load_gsm8k(...)
└── Sequence[Example]                     id, question, reference_answer
    │
    ├── Evaluation selection              fixed examples, stable order
    │   └── Generation batches            small slices of examples
    │       └── Format questions → TextGenerator
    │           └── Verify completions
    │               └── Evaluation records and metrics
    │
    └── Training selection                training partition
        └── Rollout sampling              owns shuffle/RNG/cursor
            └── Format questions → tokenize → generate
                └── Trajectories          generated tokens, rewards, etc.
                    └── Compute advantages and cache scores
                        └── Optimization minibatches
                            └── Collate   padding, tensors, masks
                                └── PPO updates
```

Loading and prompt formatting can be shared functions. **Selection, sampling, and collation are separate operations**, rather than modes of a universal loader.

## How evaluation batching should work

For now, let `evaluate(...)` slice its input sequence directly:

```python
def evaluate(
    examples: Sequence[Example],
    generator: TextGenerator,
    *,
    batch_size: int,
    # Other explicit evaluation settings...
):
    assert examples, "Evaluation requires at least one example"
    assert batch_size > 0, "batch_size must be positive"

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        # Format questions.
        # Generate completions.
        # Check output cardinality.
        # Verify and record results.
```

This naturally:
- Preserves order.
- Includes the final partial batch.
- Leaves the input reusable for another checkpoint.
- Requires no sampler, custom `DataLoader`, or `EvalBatch` class.

Here, `batch_size` means **number of questions submitted to one generation call**, not number of completions. With `num_samples=4`, a batch of eight questions yields eight groups of four completions. The generator adapter owns backend-specific tokenization and padding.

If another caller needs identical sequence batching later, extract the slicing into a small `iter_batches(...)` function. There is no need to introduce that abstraction before it removes actual duplication.

## Training has two different batching needs

**1. Rollout batches: which questions to generate from**

The trainer selects questions using its own sampling policy and RNG state. It can then split that selection into smaller generation calls to fit memory.

The number of questions collected before a policy update need not equal the generation-call batch size.

**2. Optimization minibatches: which generated trajectories to learn from**

After generation, verification, advantage estimation, and score caching, the trainer selects optimization minibatches. A training collator converts their variable-length data into aligned tensors and masks.

That collator does not load questions or choose the sampling order.

**Practical starting point:** use inline slicing in `evaluate.py`, explicit sampling in the training runtime, and a separate training `collate(...)` function when implementing optimization. Keep `data.py` focused on the records those operations exchange.


---

I’ll read the contract and existing data-loading code, then recommend where the evaluation loader should live and what it should share with training.

**Share dataset loading and example types; keep evaluation batching separate from training’s rollout sampling and tensor collation.** You do not need two `DataLoader` classes—or a universal one with a `train/eval` switch.

The file currently lives at [`src/core/data.py`](cs8803-drl-f26_vineppo/src/core/data.py), rather than `src/code/data.py`. It is a good home for shared data types, but I would put GSM8K-specific loading in `core/gsm8k.py`, consistent with the report.

## 1. Distinguish three responsibilities

“Training dataloader” can mean two different things in this project:

- **Dataset loading:** read questions and reference answers. Evaluation and training can share this.
- **Prompt sampling/batching:** choose questions for evaluation or rollout collection. Share basic batching mechanics, but not selection policy or mutable iterator state.
- **Optimization batching:** turn generated trajectories, cached log-probabilities, and advantages into tensors. This is training-only.

Evaluation should never need an `Episode` or `TrainBatch`. Those represent downstream training stages, not dataset examples.

## 2. Start with examples, not a loader class

In `core/data.py`, introduce a small shared record:

```python
@dataclass(frozen=True)
class Example:
    id: str
    question: str
    reference_answer: str
```

The contract should be:

- `id` identifies the **original source row**, not its position in a selected subset or batch.
- `question` is unformatted dataset text.
- `reference_answer` is the target answer expected by the verifier—not a gold solution appended to the prompt.
- Records are valid immediately after construction.

For GSM8K, extracting the reference answer from the source solution belongs in dataset normalization. Invalid reference data should fail loading; that is different from a generated answer failing to parse, which is an evaluation outcome.

Keep dataset revision and split in run metadata. If row indices supply IDs, their identity is meaningful only relative to that pinned dataset revision/configuration/split.

## 3. Give the evaluator a sequence

For the current scale, I would use this conceptual boundary:

```python
# core/gsm8k.py
def load_gsm8k(*, split: str, revision: str) -> list[Example]: ...
def format_prompt(question: str) -> str: ...

# runtime/evaluate.py
def evaluate(
    examples: Sequence[Example],
    generator: TextGenerator,
    *,
    batch_size: int,
    # Explicit generation and scoring settings.
) -> EvaluationResult: ...
```

These are proposed interfaces, not existing implementations.

A sequence is sufficient here: it has a known length, can be reused for repeated checkpoint evaluations, and is trivial to construct in tests. A custom loader object adds little unless it actually owns meaningful state or resources.

In your current [`Evaluator`](cs8803-drl-f26_vineppo/src/runtime/evaluate.py), I would replace the proposed `dataloader: DataLoader` dependency with `examples: Sequence[Example]`. Keeping the class is fine; its input contract matters more than class versus function.

**Let the evaluator own generation batching**, because batch size is an execution setting, not a property of the dataset.

Within each batch it should:

1. Format questions into prompts.
2. Call `TextGenerator.generate`.
3. Assert one output group per example and the requested number of samples per group.
4. Verify each completion against its corresponding reference.
5. Record the example ID and sample index.

Yes—this answers the batching TODO in `evaluate.py`: formatting, count checking, and verification can all happen batch by batch.

## 4. Separate policies, not duplicated implementations

For **evaluation**:

- Use an explicitly selected, fixed set of examples.
- Preserve order initially; avoid length sorting until there is a demonstrated need.
- Visit each selected example once per evaluation invocation.
- Include the final partial batch.
- Keep multiple generated samples attached to one example; do not duplicate examples in the loader.
- Reject an empty selection rather than report misleading accuracy.

For **training rollout collection**:

- Select from the training partition.
- Own shuffle order, RNG state, and any resume cursor separately.
- Reuse the same example representation and appropriate prompt formatter.

For **optimization**:

- Use a separate training collator for padding, masks, cached scores, and advantages.

A validation subset used during training should be selected once and reused—not resampled at every checkpoint. Final test evaluation should remain a separate explicit selection.

## 5. Minimal module boundary

I would keep:

- **`core/data.py`** — `Example`, trajectory/training types, and genuinely shared batching helpers if needed.
- **`core/gsm8k.py`** — source loading/normalization and a separate pure prompt-formatting function.
- **`runtime/evaluate.py`** — evaluation batching, generation, verification, aggregation.
- **`main(...)`** — load/select examples, construct the backend, call evaluation, save results and provenance.

No base dataset hierarchy, custom evaluation sampler, or `EvalBatch` class is necessary yet.

The most useful initial tests are stable IDs after selection, final partial-batch inclusion, repeated evaluation of the same sequence, exact question-to-result alignment, and rejection of malformed source records. Prompt-formatting tests should also establish that reference answers never enter the prompt.

**Bottom line:** keep the shared *data model* in `core/data.py`; share the *source loader* between training and evaluation; keep their *iteration policies* independent. Evaluation can simply consume `Sequence[Example]`.

No files changed or tests run; this is a design recommendation based on the current code and linked contract.

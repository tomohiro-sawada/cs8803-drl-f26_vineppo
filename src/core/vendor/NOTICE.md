# VinePPO verification sources

Source repository: https://github.com/McGill-NLP/VinePPO
Pinned revision: **9f716e68bb934e71da86c226c1699c36b5553faf**.
All VinePPO source downloads used this full revision, not a moving branch.

Paper: https://arxiv.org/html/2410.01679v2#A3 (Appendix C.1–C.4).
C.1 credits Minerva (Lewkowycz et al.), PRM800K (Lightman et al.), and
DeepSeekMath (Shao et al.) for MATH extraction/comparison scripts. The actual
scripts used here were obtained from the official VinePPO release:

- `src/treetune/tasks/math_answer_exctraction.py` (upstream spelling): entire
  extraction/normalization module. The answer-splitting loop is factored into
  `split_math_answers` without changing its body; two bare exception handlers
  are narrowed to expected normalization failures. Tests reconstruct and hash
  the original source after undoing these documented mechanical changes.
- `src/treetune/tasks/math_grader_minerva.py`: `parse_digits`, `is_digit`,
  `symbolic_equal`, `math_equal`, `is_correct`, `eval_math`.
- `src/treetune/tasks/math_grader.py`: PRM800K-derived `BAD_SUBSTRINGS`,
  `BAD_REGEXES`, `_parse_latex`, `_inject_implicit_mixed_number`,
  `count_unknown_letters_in_expr`, `should_allow_eval`.
- `src/treetune/tasks/gsm8k.py`: extraction and grading methods, detached from
  the heavyweight Task/datasets framework, fixed to `use_original_format=True`;
  the unused last-number branch is removed.

License: MIT, copyright (c) 2024 McGill NLP (`LICENSE.VinePPO`). Also included
are the upstream PRM800K MIT and DeepSeekMath code MIT notices
(`LICENSE.prm800k`, `LICENSE.DeepSeekMath`). VinePPO explicitly identifies
`math_grader.py` as copied from OpenAI PRM800K. Original source byte hashes are
recorded in `sources.json`. `safe_symbolic.py` and `worker.py` are local code.

## Selected official configuration

The experiment configs import `tasks/math_inplace_no_answer_prefix.jsonnet`
(`answer_prefix: null`) and `tasks/gsm8k_orig_format.jsonnet`
(`use_original_format: true`). Confirmed in
`polIter_rho1bSft2_ppo_MATH.jsonnet`, `polIter_rho1bSft2_ppo_GSM8K.jsonnet`
and the corresponding `sft_rho1b_for_*_eval.jsonnet` files. VinePPO inherits
these PPO task settings. This is **not** the alternative MATH `# Answer` task
or the alternative GSM8K last-number task.

The precise MATH task call chain at this revision is:

1. `MATH.extract_predicted_answer_from_text` (`math.py:319–347`) dispatches
   to `extract_math_minerva_few_shot_cot_answer(problem, text)` because
   `answer_prefix is None`. This calls `extract_math_answer(problem, text)`.
2. `MATH.grade_answer` (`math.py:449–484`) selects
   `grade_answer_minerva_format`, **not** PRM800K's standalone `grade_answer`.
3. `grade_answer_minerva_format` (`math.py:486–496`) ignores the scalar
   `ground_truth` argument. It re-extracts the reference list using
   `extract_math_answer(item["problem"], item["solution"])`, assigns the
   already-extracted prediction list, and calls `eval_math(item)`.
4. `eval_math` deduplicates both lists in first-occurrence order, keeps the
   last N distinct predictions (N = distinct reference count), then calls
   `is_correct`. List comparison requires every prediction and every reference
   to have a match; it is unordered coverage, **not** positional or bijective
   matching. The numeric tolerance can let one prediction match multiple
   distinct references. Those official semantics are preserved.

Source: https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/tasks/math.py#L319-L496

## Extraction and equality policy

MATH preserves the official precedence: the Minerva final-answer phrase,
boxed answers, `The answer is`/`the answer is`, fenced program output, last
number. The last-number and program-output paths are upstream behavior, not
new scoring fallbacks. A malformed `boxed` occurrence suppresses those later
paths; unclosed boxes are omitted, while earlier closed boxes remain usable.
Normalization includes fractions, square roots, whitespace, units, dollar
signs, and trailing punctuation, as implemented in the vendored module.

The API accepts an already-extracted, normalized final reference answer and
`question: str = ""`. The response goes through `extract_math_answer(question,
response)`. The reference goes through `strip_string`, then the **same official
splitting loop** with that question; it is never sent through reasoning/last-
number extraction and is not wrapped in synthetic solution markup.

The official loop first checks for the literal, case-sensitive substring
`"separated by commas"` in the question and absence of **any** `()[]` character
in the answer. If true, it splits on every comma. Otherwise, it splits on
LaTeX `\\text{and}` (allowing whitespace inside braces). Otherwise, it retains
one answer. Comma handling takes precedence over `\\text{and}`, even when no
comma occurs. Tuples/intervals therefore stay intact. No additional list,
English "and", or multi-box heuristics are introduced.

`eval_math` receives both full lists. `extracted_answers: tuple[str, ...]`
reports the deduplicated, last-N selected prediction list; `extracted_answer`
remains its final member for compatibility. Multiple answers are not silently
joined into an ambiguous string. Missing/single empty extraction gives `None`,
`False`, and `()`. For a single distinct reference, boxes `1, 2, 1` still select
`2`, not `1`. For two distinct references, they select `(1, 2)`.

**Parity contract:** the scalar-reference API is equivalent to the official
full-solution path when its normalized reference preserves the gold extracted
answer list under this splitting rule. Tests compare the adapter against the
actual `extract_math_answer(question, gold_solution)` + `eval_math` path for
single boxes, separate boxes represented by a comma-delimited reference,
LaTeX-and lists, duplicates, distractors, tuples and case-sensitive questions.
For example, reference `1,2` plus a "separated by commas" question preserves
separate gold boxes `1` and `2`; an unsplit scalar with a question lacking that
instruction does not. Original box boundaries or earlier gold answers already
lost by data normalization cannot be reconstructed from a scalar. The adapter
does not claim parity for such lossy data. Callers must pass the original
question and preserve the normalized reference content. The empty question
default exists for compatibility only and disables question-dependent splitting.
GSM8K accepts but ignores the question.

MATH equality retains numeric tolerance (absolute 1e-3), the upstream
percentage equivalence (reference / 100, reference, reference * 100),
recursive tuple/matrix comparison, equation handling, union handling, and
symbolic simplification. It is intentionally not stricter mathematical
identity: e.g. 50 can match 0.5, and tuple bracket types can differ.

GSM8K takes the text after the last `####`, strips outer whitespace, removes
commas from the prediction, lowercases both sides, and compares strings.
The normalized reference is **not** comma-stripped by the official grader.
There is no numeric/symbolic equivalence: `42.0`, `042`, `42.` and `42 dollars`
do not match reference `42`. No `####` means no answer, even with a final number.

Finish reason is deliberately outside this API. Both stop- and length-ended
text can be verified. C.4 describes final-answer matching without a separate
length exclusion, and task `evaluate_predictions` grades response strings.
The **training** episode generator (`math_episode_generator.py`, lines 136–144)
instead assigns its configured unfinished-response penalty to length-ended
responses. C.2 discusses truncation and says a negative (-1) penalty did not
improve performance. Do not equate this verifier's correctness with training
reward or discard finish reasons in the caller.

## Security changes (not bit-for-bit unrestricted upstream parsing)

Upstream `parse_expr` executes Python via eval. Its character-count/large-power
heuristics are not a security boundary. Upstream `N(string)` also sympifies
strings, and its thread timeouts can wait on executor shutdown. We do not run
these eval-based entry points on model output or references.

- The original symbolic parser calls are replaced with restricted AST
  construction after the same SymPy token transformations. Only arithmetic,
  explicit mathematical functions/constants, and validated constructor
  literals are allowed. No attributes, indexing, keywords, comprehensions,
  arbitrary Python calls, or input string literals. Unsupported expressions
  are nonmatches through upstream's parse-failure handling, not another grader.
- `_sympy_parse` retains implicit multiplication and caret-to-power conversion.
  Numeric approximation goes through the safe parser too. The unused ANTLR
  parser path is removed: upstream's active fast-LaTeX path uses pylatexenc.
- The SympySimplifier process pool and LaTeX conversion thread timeout are
  replaced by synchronous operations inside a single-use subprocess. The
  parent kills and reaps it after 15 seconds; the child has 10/11-second CPU
  soft/hard limits and no core dumps. On Linux it also has a 1 GiB address-space
  cap. macOS does not enforce that memory cap (its setrlimit rejects AS/DATA);
  CPU/wall bounds still apply. Use Linux for hard memory containment.
- The worker uses Python isolated mode, an empty environment, JSON IPC (no
  pickle), and creates no descendants. It is a resource boundary, **not** an OS
  sandbox. The restricted parser is what prevents source-code execution.
- All worker timeouts and abnormal terminations now **raise**, including
  `SIGKILL` and `SIGXCPU`. A SIGKILL cannot identify external termination versus
  OOM or a hard CPU limit. A whole-worker timeout can include startup/import
  delay, so it is not labeled an incorrect mathematical answer either.
  `subprocess.run` still kills and reaps a timed-out worker. This intentionally
  differs from upstream's simplify/task timeout-as-False policy.
- Removed the worker's memory/recursion-as-False catch. Narrowed vendored bare
  and broad exception handlers: numeric conversion catches expected numeric
  errors; symbolic branches catch an explicit set of malformed/unsupported
  math exceptions (ValueError, TypeError, SyntaxError, TokenError,
  PolynomialError, NotImplementedError, ArithmeticError). MemoryError,
  RecursionError, RuntimeError, import/name/attribute faults and interrupts
  propagate. As with upstream parsing, ValueError/TypeError cannot always be
  distinguished from implementation bugs inside a math dependency; the
  restricted handlers are not a claim of perfect fault attribution.
- Removed upstream's diagnostic stdout prints (including the special `2,3,4`
  print) because they corrupt the worker's JSON protocol; scoring is unchanged.
  Dependency/import/protocol failures propagate. Inputs, including question,
  above 65,536 characters fail with ValueError; they are not silently truncated.
  Empty references fail explicitly.

Dependencies: pylatexenc 2.10 (upstream version), SymPy 1.14.0 (pinned for
parser reproducibility), regex >= 2023.12.25 (resolved in uv.lock). No ANTLR,
latex2sympy, timeout package, model framework, or network is needed at runtime.

# OLMo 3: how much of the GSM8K gain is answer format

20261124-080354 EST · `4c924f8`

Responds to Tom's request of 2026-11-24 to record this finding; re: [final report, “OLMo at 6,400 primary continuations”](20261122-110919_final-report-detailed.md#olmo-at-6400-primary-continuations).

## Finding

GRPO raised OLMo 3's GSM8K test Pass@1 from 61.65% to 87.87%. About three quarters of that 26-point gain is the model learning to write its answer where the grader looks. Counting a response as correct when its final number equals the reference, the same two checkpoints score 83.29% and 89.77%.

| Reading | SFT baseline | GRPO, iteration 50 | Paired gain [95% CI] |
|---|---:|---:|---:|
| Official grader (strict) | 61.65% | 87.87% | +26.23 [24.61, 27.84] |
| Final number equals the reference (lenient) | 83.29% | 89.77% | +6.48 [5.39, 7.56] |
| Responses with a `####` answer | 17,319 (82.07%) | 21,040 (99.70%) | |
| Truncated responses | 69 | 63 | |

Each column is 21,104 graded responses: 1,319 test questions over 16 rounds. Intervals cluster by question. Both arms have one training seed.

## Where the baseline loses points

The official grader reads the text after the last `####` and compares it with the reference as a string. The baseline has 4,568 responses (21.65 points) that end in the right number and are marked wrong:

- **No answer after `####`:** 3,134 responses. Of the 3,785 baseline responses with no extractable `####` answer, 3,650 put the answer in `\boxed{}`, as the model does for MATH.
- **A `####` line that is not the bare number:** 1,434 responses, such as `#### $70,000`, `#### $20.00`, or the number followed by a code fence or words.

After GRPO, 400 such responses remain (1.90 points). Of these, 186 are the separator cases in the next section. The four others I read end in trailing zeros, such as `#### 6.00`.

## A ceiling in the official grader

Fourteen of the 1,319 test references contain a thousands separator, such as `2,125`. The grader removes separators from the response and not from the reference, so no response to those questions can be marked correct. All 224 graded responses to them were marked wrong in each of the four evaluations I checked. The upstream grader at the pinned commit behaves the same way, so this comes with the paper's code. [Upstream `gsm8k.py`](https://github.com/McGill-NLP/VinePPO/blob/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/tasks/gsm8k.py)

- **Cap:** strict Pass@1 cannot exceed 98.94% (1,305 of 1,319 questions).
- **Cost grows with accuracy:** by the lenient reading, the 224 responses were right 186 times for OLMo GRPO (0.88 points lost), 165 for OLMo SFT, 79 for RhoMath VinePPO 650 (0.37 points) and 29 for RhoMath SFT (0.14 points).

## RhoMath is barely affected

RhoMath writes `####` natively: 21,092 of 21,104 responses for VinePPO 650 and 21,012 for SFT. VinePPO 650 beats SFT by 16.75 points on the strict reading and 16.96 on the lenient one. The RhoMath comparisons in the final report need no caveat beyond the ceiling above.

## MATH is not a format effect

OLMo's MATH gain stands: 70.59% to 71.86%, a paired gain of 1.27 points [0.38, 2.17]. An answer was extracted from 99.84% of baseline responses and 99.88% of GRPO responses.

The baseline had been graded before two grader fixes (`2f545ce`, `75af343`) and the GRPO run after them. Re-grading all 16,000 stored responses with the current grader changed no grade.

## RLOO shows the same pattern

The OLMo GSM8K RLOO run launched today (`handoff-olmo3-gsm8k-rloo-50-s42-20`, selected iteration 40) scores 88.01% strict and 89.26% lenient. Against the baseline that is +26.36 [24.74, 27.98] strict and +5.97 [4.99, 6.95] lenient. RLOO and GRPO are indistinguishable on these 1,319 questions: RLOO minus GRPO is +0.13 [−0.56, 0.82] strict and −0.51 [−1.13, 0.11] lenient (job 7284876).

## What to report

- Give both readings for every OLMo GSM8K row, including the RestEM run still training.
- Describe the 26-point gain as format compliance plus a 6.5-point accuracy gain.

## Method and limits

- **Script:** `scripts/diagnostics/regrade_evaluation.py`, CPU jobs 7282389 (GSM8K) and 7283551 (MATH). It re-grades the stored responses and reproduced every stored grade: 4 × 21,104 for GSM8K and 2 × 8,000 for MATH. The strict numbers above are therefore the final report's numbers.
- **The lenient reading is mine, not the paper's.** It takes the last number in the response and ignores `$`, separators and trailing zeros. The script asserts that every response the official grader accepts is also lenient-correct.
- **Over-credit is possible** when the last number is right by accident. The two readings disagree on 1,829 OLMo responses that have a `####` answer. I read 12 of them, and all 12 were right answers in another format.

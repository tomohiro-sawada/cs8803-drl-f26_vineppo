"""
Benchmark an LLM policy on MATH/GSM8K.
Backend is in `src/benchmarking`.
See details from Appx C.1 of original paper.

Use cases I have in mind at this point:
- Sanity check on the downloaded RhoMath 1.1B model (i.e. model weights prior to using RL) to make sure it matches the numbers reported in the paper; Figure 4 of the paper reports Initial SFT.
- Maybe: validation suite - benchmark intermediate checkpoints on validation set to make sure they are performing as expected
- Evaluate the final policy checkpoints and measure their accuracy.
"""

from benchmarking.base import Evaluator

# Run the evaluator on the model ...?

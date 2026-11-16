# README.

We assume everything in `scripts/setup` has been run correctly. 

## Directories.
- `core`: logic shared between training and evaluation. 
- `baselines`: Implementation of baselines.
- `benchmarking`: Infra for benchmarking trained policies on MATH and GSM8K. 

We only have two scripts at the root of `src` because there are only two types of experiments we run: training or evaluation.
This corresponds to `src/treetune/runtime` in [the original implementation](https://github.com/McGill-NLP/VinePPO/tree/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/runtime). 





<!---->

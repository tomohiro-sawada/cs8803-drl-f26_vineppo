I'm trying to think of a good "additional modification" part of the project that will genuinely make me learn the basics of inference. Any ideas? Constraints: https://www.pair.toronto.edu/cs8803-drl-f26/project.html

---

I’d **implement a single-GPU sampler that forks a reasoning prefix into \(K\) continuations**, then use it to answer an RL question: **at a fixed GPU-time budget, how many branches should VinePPO spend on each value estimate?** This gives you a reason to learn prefill, KV caching, and batched decoding, plus a substantive experimental extension beyond integrating vLLM.

I couldn’t verify the Fall 2026 project page. The Fall 2025 version asks for a substantive extension and gives an algorithmic improvement as one example. If the current rubric is similar, a compute-budgeted change to VinePPO’s estimator makes a clearer course contribution than a backend swap alone. [CS 8803 - DRL](https://www.pair.toronto.edu/cs8803-drl-f25/project.html?utm_source=chatgpt.com)

### What I’d implement personally

Use RhoMath’s existing forward pass on one A40. Write the generation loop yourself. Start with **one actual GSM8K reasoning prefix** and implement three versions:

| Version | What you implement | What the comparison teaches |
|---|---|---|
| **A. Full recomputation** | Append each sampled token and forward the entire sequence again. Use short examples. | A simple correctness oracle for next-token logits. |
| **B. Cached decoding** | Prefill once, save the KV cache, then forward one new token at a time. | What the cache contains, how positions advance, and how prefill differs from decode. |
| **C. Forked decoding** | Prefill one prefix, draw \(K\) independent first tokens from its final logits, copy the prefix cache into \(K\) rows, and decode the branches as a batch. | Which work siblings can share, when their state diverges, and what happens as rows hit EOS at different times. |

The subtle implementation detail in C is that **the final prefill logits already predict the first branch token**. Feeding the last prefix token again introduces an off-by-one error. Test B and C by forcing a known suffix and comparing next-token logits at each position against A. Copying the prefix cache is sufficient for this exercise: it saves repeated prefill computation while still consuming \(K\) copies of prefix KV memory. [How To Scale Your Model](https://jax-ml.github.io/scaling-book/inference/?utm_source=chatgpt.com)

### The experiment that makes it a project

Your smoke run spent **173 seconds generating 504 MC continuations from 56 prefixes**, versus **1.6 seconds on the actor update**. That establishes where to experiment. It leaves open whether repeated prefill or suffix decoding accounts for most of those 173 seconds.

1. **Measure the mechanism.** Replay frozen prefixes at \(K\in\{1,3,9\}\). Compare C against a control that **batches the same \(K\) branches but prefills \(K\) copies of the prefix**. That control isolates prefix reuse from ordinary batching. Report prefill time, decode time, total time, and peak memory. A small end-to-end gain is informative if suffix decoding dominates.
2. **Measure the RL tradeoff.** On those same prefixes, estimate \(V(s)\) with each \(K\). Compare against an independent, larger-sample reference on a subset. Plot **value-estimation error versus measured GPU-seconds**. For binary rewards and independent continuations, \(\operatorname{Var}[\hat V_K(s)\mid s]=p_s(1-p_s)/K\). Nine branches buy three times lower standard error than one, but the measured cost need not be nine times higher.
3. **Make one algorithmic change.** Compare fixed \(K=9\) with a simple **depth-dependent branch budget**: use a frozen calibration set to estimate continuation cost and reward variance by reasoning-step depth, then choose \(K\) per depth before training. This tests whether redistributing rollouts improves value estimates at the same GPU cost. If time permits, compare held-out GSM8K learning curves at matched GPU-hours.

**My scope recommendation:** own A–C and the frozen-prefix experiment yourself. Let the cluster agent handle multi-GPU orchestration. The most educational deliverable is a plot you can explain from first principles: **when nine continuations share a reasoning prefix, how much time do you save, how much memory do you spend, and how much more precise is the resulting value estimate?**

---

Can I conceivably make this a well-scoped <1000 line script for which I can iterate quickly (I.e. load model, run inference, profile, and check efficiency)? Also, can you provide seminal and/or technically deep papers that are directly relevant for this?

---

Yes. **I’d aim for 250–400 lines** for a single-GPU benchmark that answers one question: **how much does prefilling a reasoning prefix once save when VinePPO samples nine continuations from it?** Use RhoMath’s existing `forward()` and attention kernels. Write the decode loop, cache manipulation, correctness check, and timing yourself. That scope lets you learn the mechanics and iterate independently of training or Slurm orchestration.

### The smallest useful script

Load the model once and run three modes in the same process. Start with one unpadded prefix so every branch has the same initial position.

| Mode | Implementation | Purpose |
|---|---|---|
| `recompute` | Forward the entire prefix plus suffix at each step. | Correctness oracle on short examples. |
| `repeat_prefill` | Put \(K\) copies of the prefix in a batch, prefill all rows, then decode with a KV cache. | Batched baseline. |
| `fork_prefill` | Prefill one row, use its final logits for all \(K\) first-token draws, copy its cache into \(K\) rows, then run the **same decode loop**. | Isolates the benefit of avoiding repeated prefill, including the cost of copying the cache. |

I’d give the script three entry points:

- **`check`** — Force two **different** short suffixes from the same prefix. At every position, compare each branch’s logits against `recompute` on that branch’s exact token history. Different suffixes catch accidental cache aliasing. Comparing the *first* token catches the common mistake of feeding the last prefix token twice.
- **`bench`** — On one A40, warm up, then sweep \(K\in\{1,3,9\}\) over short, median, and long prefixes from your actual workload. First replay fixed suffixes of identical length across modes. Report median **prefill, cache-copy, decode, and end-to-end milliseconds**, plus peak allocated memory. Synchronize at timing boundaries and exclude model loading. Then run a smaller EOS-aware sampling benchmark to measure useful output tokens/s under realistic branch lengths.
- **`profile`** — Capture one representative \(K=9\) trace with named `prefill`, `fork_cache`, and `decode` regions. Keep profiling separate from benchmark timing. For a quick edit–run cycle, one prefix, two branches, and eight generated tokens exercise the whole path.

**The first plot I’d make** is a stacked bar chart of `repeat_prefill` versus `fork_prefill` on real prefixes. If repeated prefill takes fraction \(f\) of baseline time, even eliminating it entirely caps speedup at \(1/(1-f)\). Your 173-second MC measurement makes this worth investigating, but it doesn’t tell you \(f\). Measuring that fraction gives you a useful result even if the optimization barely moves end-to-end time.

### Three papers, in reading order

| Paper | What to take into the script |
|---|---|
| **Pope et al., *Efficiently Scaling Transformer Inference* (MLSys 2023)** | **Start here for fundamentals.** Read the treatment of prompt processing, incremental generation, batch size, and memory bandwidth. Before running your sweep, predict what increasing \(K\) should do to decode latency per step and aggregate tokens/s. The single-device reasoning is enough for this experiment. [proceedings.mlsys.org](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html?utm_source=chatgpt.com) |
| **Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)** | **Read the parallel-sampling example.** Your cache copy avoids repeated prefill computation but still allocates \(K\) copies of the prefix KV tensors. PagedAttention explains how branches can share physical KV blocks and acquire private storage as they diverge. This gives you a concrete interpretation of your peak-memory plot. [doi.org](https://doi.org/10.1145/3600006.3613165?utm_source=chatgpt.com) |
| **Juravsky et al., *Hydragen: High-Throughput LLM Inference with Shared Prefixes* (2024)** | **Closest match to VinePPO branching.** Even after a prefix has been computed once, every sibling query attends to it at every decode step. Hydragen separates attention over the common prefix from attention over private suffixes, allowing sibling queries to reuse prefix reads. Read it after profiling so you can judge whether your prefix lengths and \(K=9\) make that optimization interesting. [arxiv.org](https://arxiv.org/abs/2402.05099?utm_source=chatgpt.com) |

The distinction worth learning is **three meanings of “sharing a prefix”**: compute its KV tensors once, store them once, and reuse their memory reads across sibling queries during decoding. Your script implements the first. PagedAttention and Hydragen explain the next two. For a course project, I’d stop the systems implementation there and use the measured cost curve for the RL extension: **how much value-estimate precision does each additional sibling buy per GPU-second?**

---

Actually, how is your idea different from KV caching?? Aren't you just saving the KV across the prefixes?

---

Yes. **The proposed optimization is prefix KV reuse.** I made it sound more distinct from KV caching than it is.

| Mechanism | What gets reused |
|---|---|
| Ordinary KV caching | **Across token steps in one continuation.** When generating the next token, reuse the KVs for all earlier tokens in that sequence. |
| The proposed fork | **Across sibling continuations.** Prefill one reasoning prefix, then use its KVs to initialize all nine branches. Each branch then uses ordinary KV caching as it generates. |
| Reuse across VinePPO step boundaries | **Across nested prefixes.** If one reasoning prefix extends another, reuse the shorter prefix’s KVs and compute only the extension. This is a separate experiment. |

For a 200-token prefix and nine siblings, the benchmark compares **prefilling nine identical rows** with **prefilling one row and copying its cache eight times**. Both paths use KV caching during decoding. Both end up with nine copies of the prefix KVs. The experiment measures whether the prefill computation you save outweighs the cache-copy cost.

So the script is an **inference learning exercise** built around a standard technique. The substantive VinePPO extension would be testing whether a different allocation of branches improves **step-advantage estimates per GPU-second**.

---

Concretely, will this make me learn vllm under the hood since I have to mess around with the KV cache itself?

---

**Yes for the model side of inference. Partly for vLLM itself.** Your PyTorch script would teach you what a KV cache contains, how prefill produces it, how decode appends to it, and how to initialize nine independent continuations from one prefix. You’d manipulate the actual tensors and verify correctness by comparing next-token logits.

vLLM adds a different layer of machinery:

| Your script | What vLLM adds |
|---|---|
| Copy the prefix tensors into nine batch rows. | Map logical token positions to physical KV blocks so requests can share storage. |
| Explicitly tell the script which prefix to reuse. | Discover reusable full blocks by hashing token IDs and their prefix history. |
| Run a fixed batch until it finishes. | Schedule requests across steps as sequences arrive and finish, under a shared token and cache budget. |

Those first two vLLM mechanisms are documented in its [prefix-caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/). [vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/?utm_source=chatgpt.com)

**For a concrete under-the-hood exercise, do one experiment in both implementations.** Take two VinePPO step prefixes from the *same* trajectory, where the second extends the first. In PyTorch, prefill the shorter prefix, then feed only the extra tokens and verify that the resulting logits match a fresh prefill of the longer prefix. In vLLM, submit the same token IDs sequentially with prefix caching on and off. Trace which blocks the second request reuses, then vary the shared length around a block boundary to see when an extra block becomes reusable. This follows the same operation from **tensor state** to **block lookup and scheduling**. [vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/?utm_source=chatgpt.com)
"""What-if replay of stored OLMo samples through the production step splitter.

Production rejects every OLMo sample at the control-token allowlist, because the
tokenizer lists its chat tokens as added special tokens and not in
all_special_ids. The proxy below adds them, and nothing else changes.
"""
import collections, json, re, statistics, sys

W = sys.argv[1]
sys.path[:0] = [W, W + "/src"]
from transformers import AutoTokenizer
from core import segmentation as seg
from core.prompt_protocol import prompt_protocol

S = "/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo"
tok = AutoTokenizer.from_pretrained(f"{S}/cache/math/models/Olmo-3-7B-Instruct-SFT-e1452fc", local_files_only=True)


class Proxy:
    is_fast = tok.is_fast
    bos_token_id = tok.bos_token_id
    all_special_ids = sorted(set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items() if t.special})
    decode = staticmethod(tok.decode)


strict_gsm8k = seg.gsm8k_step_boundaries


def rounded_gsm8k(text, ids, spans):
    return seg._token_step_boundaries(text, ids, spans, seg._gsm8k_character_boundaries, round_forward=True)


def replay(task, policy):
    seg.gsm8k_step_boundaries = rounded_gsm8k if policy == "round-forward" else strict_gsm8k
    out = f"{S}/training/pace_ice/olmo3-sft-{task}-segmentation-hf-20"
    eos = json.load(open(f"{out}/tokenizer.json"))["eos_token_ids"]
    controls = prompt_protocol("olmo3-sft", task).context_special_ids(tok)
    rows = [json.loads(line) for line in open(f"{out}/records.jsonl")]
    outcomes, steps, lengths, mc = collections.Counter(), [], [], []
    for row in rows:
        ids = row["token_ids"]
        try:
            cuts = seg.step_boundaries_from_tokens(ids, Proxy, dataset=task, eos_token_ids=eos, context_ids=row["prompt_ids"], context_special_ids=controls)
        except (AssertionError, ValueError) as error:
            message = re.sub(r"[0-9]+", "N", str(error).splitlines()[0] if str(error) else type(error).__name__)[:80]
            outcomes[f"{row['finish_reason']}: {message}"] += 1
            continue
        outcomes[f"{row['finish_reason']}: success"] += 1
        steps.append(len(cuts) - 1)
        lengths.append(len(ids))
        # Nine continuations per intermediate state, each as long as the sampled remainder,
        # plus the bare prompt shared by eight siblings.
        mc.append(9 * (sum(len(ids) - cut for cut in cuts[1:-1]) + len(ids) / 8))
    summary = {"task": task, "policy": policy, "samples": len(rows), "outcomes": dict(outcomes.most_common())}
    if steps:
        summary |= {"steps_mean": round(statistics.mean(steps), 1), "steps_median": statistics.median(steps), "steps_max": max(steps),
                    "tokens_mean": round(statistics.mean(lengths), 1), "mc_tokens_per_primary_token_estimate": round(sum(mc) / sum(lengths), 1)}
    print(json.dumps(summary), flush=True)


print(json.dumps({"all_special_ids": tok.all_special_ids, "proxy_special_ids": Proxy.all_special_ids}))
replay("math", "production")
replay("gsm8k", "production")
replay("gsm8k", "round-forward")

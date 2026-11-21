"""Check the HF exact-resume pair and replay the expected vLLM resume requests."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from transformers import AutoTokenizer

from core.prompt_protocol import prompt_protocol
from core.vllm_inference import stream_seed
from experiments.training.pace_ice.chain import latest_checkpoint
from experiments.training.resume import atomic_record
from experiments.training.run import load_examples
from runtime.checkpoints import assert_state_equal


def state(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def last(root):
    assert (root / "COMPLETE").is_file()
    return latest_checkpoint(sorted(root.glob("attempt-*")))


def compare_hf(control, interrupted):
    left, right = last(control), last(interrupted)
    a, b = state(left), state(right)
    for ckpt in (a, b):
        del ckpt["config"]["checkpoint_dir"]
        if "frozen" in ckpt:
            del ckpt["frozen"]["path"]
    assert_state_equal(a, b)
    return {"control": str(left), "resumed": str(right), "bitwise": True}


def compare_requests(root):
    assert (root / "COMPLETE").is_file()
    audits = sorted(root.glob("attempt-*/resume-audit.json"))
    assert audits, "No vLLM resume audit was recorded"
    for audit in audits:
        loaded = json.loads(audit.read_text())
        assert loaded["state_bitwise_equal"] is True
        iteration = loaded["iteration"]
        checkpoint = state(loaded["checkpoint"])
        assert checkpoint["iteration"] == iteration
        metadata = json.loads((audit.parent / "metadata.json").read_text())
        cli = metadata["cli"]
        records = audit.parent / "sampling-requests.jsonl"
        with records.open() as stream:
            first = json.loads(next(stream))
        assert first["iteration"] == iteration and first["call"] == 0
        profile = prompt_protocol(cli["model_protocol"], cli["task"])
        tokenizer = AutoTokenizer.from_pretrained(
            cli["model_path"], local_files_only=True, trust_remote_code=False
        )
        examples, _ = load_examples(Path(cli["dataset_path"]), cli["task"])
        generator = torch.Generator().set_state(checkpoint["generator"])
        indices = torch.randperm(len(examples), generator=generator)[
            : cli["prompts_per_iteration"]
        ]
        prompts = profile.encode(
            tokenizer,
            [profile.format(tokenizer, examples[i].question) for i in indices.tolist()],
        )
        assert all(
            len(p) + cli["max_new_tokens"] <= cli["max_sequence_length"]
            for p in prompts
        )
        assert len(first["requests"]) == len(prompts)
        for index, (request, prompt) in enumerate(
            zip(first["requests"], prompts, strict=True)
        ):
            assert request["prompt"] == prompt
            assert request["seed"] == stream_seed(cli["seed"], iteration, 0, index)
            assert request["n"] == cli["num_samples"]
            assert request["max_tokens"] == cli["max_new_tokens"]
    return {
        "run": str(root),
        "audits": [str(path) for path in audits],
        "requests_equal": True,
    }


if __name__ == "__main__":
    mode, flag, *paths = sys.argv[1:]
    result = {"hf": compare_hf, "vllm": compare_requests}[mode](*map(Path, paths))
    atomic_record(Path(flag), result)
    print(json.dumps(result))

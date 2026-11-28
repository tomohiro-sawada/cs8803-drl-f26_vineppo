"""Write a validation-selected, fully fine-tuned actor as a local model directory.

    python scripts/diagnostics/export_checkpoint_model.py SELECTION.json MODEL_PATH OUT

The vLLM replicas serve a base model plus LoRA adapters, so the vLLM evaluator
cannot publish a full-weights checkpoint. OUT holds the selected checkpoint's
actor weights in their own dtype and MODEL_PATH's tokenizer files, so
evaluate_model_baseline.py can serve it as a base model and run the same
16-round test as every LoRA run. export.json records the source checkpoint.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from runtime.checkpoint_actor import load_checkpoint_actor


def main(selection, model_path, out):
    model_path, out = Path(model_path), Path(out)
    best = json.loads(Path(selection).read_text())["best"]["checkpoint"]
    model, _, info = load_checkpoint_actor(
        Path(best["path"]), model_path, device="cpu", expected_sha256=best["sha256"]
    )
    assert info["finetuning"] == "full", info["finetuning"]
    out.mkdir(parents=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="64GB")
    assert (out / "model.safetensors").is_file()
    for path in model_path.glob("*token*"):
        shutil.copy2(path, out / path.name)
    (out / "export.json").write_text(
        json.dumps({"checkpoint": best, "dtype": info["dtype"]}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])

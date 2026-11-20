"""Token-exact sampling from persistent vLLM replicas serving the learner's LoRA actor.

Each replica holds one full base model on its own GPU. The learner publishes a
uniquely named adapter per actor version; every replica must acknowledge it
before sampling, and every response must echo that name. vLLM keys its prefix
cache on adapter name and path, so cached KV cannot cross actor versions.
Requests carry exact token IDs: no decode/retokenize, added BOS, chat template,
string stop, penalty or truncation. Sibling draws use seed + sample index, with
one seed per logical request, so retries and arrival order cannot change which
random stream a request receives. Replica adapters use the base dtype (BF16),
unlike the learner's FP32 adapters: behaviour and rescored log-probabilities
agree only up to that rounding, which the sampler acceptance check measures.
"""

import json
import logging
import os
import re
import shutil
import subprocess
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from math import isfinite
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from types import TracebackType
from typing import Self

import httpx
import torch

from core.inference import Completion
from core.lora import lora_identity

BASE_MODEL = "base"  # Served name of the unadapted model on every replica.
logger = logging.getLogger(__name__)


def slurm_attempt() -> dict:
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    assert re.fullmatch(r"0|[1-9][0-9]*", restart), "Invalid SLURM_RESTART_COUNT"
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": int(restart),
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_seed(*key) -> int:
    """A reproducible 48-bit sampling seed for one logical request."""
    return int.from_bytes(sha256(":".join(map(str, key)).encode()).digest()[:6], "big")


def gpu_identity() -> dict:
    """The single CUDA device this process may use, as assigned by the scheduler."""
    assert torch.cuda.is_available(), "CUDA required; no CPU/MPS fallback"
    assert torch.cuda.device_count() == 1, "Expose exactly one GPU per process"
    gpu = torch.cuda.get_device_properties(0)
    return {
        "name": gpu.name,
        "uuid": f"GPU-{gpu.uuid}",
        "total_memory": gpu.total_memory,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).split()[0],
    }


class VLLMClient:
    """Bounded concurrent requests over replicas; results keep request order."""

    def __init__(
        self,
        urls: Sequence[str],
        *,
        api_key: str,
        max_sequences: int = 256,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ):
        assert urls and len(set(urls)) == len(urls), "Worker URLs must be unique"
        assert api_key, "Workers require an API key"
        assert max_sequences > 0 and retries >= 0
        self.clients = [
            httpx.Client(
                base_url=url,
                # Close each socket after its response. The drain slots bound
                # concurrency; a second queue in the HTTP pool can stall them.
                headers={"Authorization": f"Bearer {api_key}", "Connection": "close"},
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=0),
                timeout=httpx.Timeout(300.0, connect=30.0, pool=30.0),
                transport=transport,
            )
            for url in urls
        ]
        self.max_sequences = max_sequences
        self.retries = retries
        self.stats = [{"requests": 0, "tokens": 0, "errors": 0} for _ in urls]

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        *,
        count: int,
        api_key: str,
        model_path: Path,
        other_gpu_uuids: Sequence[str] = (),
        timeout: float = 2400.0,
        **options,
    ) -> tuple[Self, list[dict]]:
        """Wait for `count` worker identity files and healthy servers.

        Every process must own a distinct physical GPU and serve `model_path`.
        """
        deadline = monotonic() + timeout
        attempt = slurm_attempt()
        while True:
            assert monotonic() < deadline, f"Workers not ready within {timeout}s"
            files = sorted(directory.glob("worker-*.json"))
            assert len(files) <= count, "More workers than requested"
            workers = [json.loads(path.read_text()) for path in files]
            assert all(
                all(worker[key] == value for key, value in attempt.items())
                for worker in workers
            ), "Worker belongs to a different Slurm attempt"
            if len(files) == count and all(_healthy(w["url"]) for w in workers):
                break
            sleep(5)
        uuids = [worker["gpu"]["uuid"] for worker in workers] + list(other_gpu_uuids)
        assert len(set(uuids)) == len(uuids), f"Processes share a GPU: {uuids}"
        client = cls([worker["url"] for worker in workers], api_key=api_key, **options)
        for served in client.get_all("/v1/models"):
            base = [m for m in served["data"] if m["id"] == BASE_MODEL]
            assert len(base) == 1 and Path(base[0]["root"]).resolve() == model_path, (
                "Worker serves a different base model"
            )
        return client, workers

    def close(self) -> None:
        for client in self.clients:
            client.close()

    def get_all(self, path: str) -> list[dict]:
        responses = [client.get(path) for client in self.clients]
        assert all(r.status_code == 200 for r in responses), [r.text for r in responses]
        return [response.json() for response in responses]

    def post_all(self, path: str, body: dict) -> None:
        for client in self.clients:
            response = client.post(path, json=body)
            assert response.status_code == 200, f"{path} failed: {response.text}"

    def served(self) -> list[set[str]]:
        return [
            {m["id"] for m in models["data"]} for models in self.get_all("/v1/models")
        ]

    def complete(self, bodies: Sequence[dict]) -> list[list[dict]]:
        """Return each request's choices in sample-index order.

        Every replica drains one shared queue through max_sequences // n slots,
        refilling as requests finish. A transport failure requeues that request
        (same logical sample and seed) up to `retries` times. The slot keeps
        draining so a late retry always has a consumer.
        """
        assert bodies and len({body["n"] for body in bodies}) == 1
        n = bodies[0]["n"]
        pending = deque((index, 0) for index in range(len(bodies)))
        results: list[list[dict] | None] = [None] * len(bodies)
        failures: list[BaseException] = []
        lock = Lock()

        def drain(worker: int) -> None:
            while True:
                with lock:
                    if failures or not pending:
                        return
                    index, attempt = pending.popleft()
                body = bodies[index]
                try:
                    response = self.clients[worker].post("/v1/completions", json=body)
                    assert response.status_code == 200, response.text
                    payload = response.json()
                    assert payload["model"] == body["model"], "Stale actor response"
                    choices = sorted(payload["choices"], key=lambda c: c["index"])
                    assert [c["index"] for c in choices] == list(range(n)), (
                        "Missing or duplicate sibling"
                    )
                    assert choices[0]["prompt_token_ids"] == body["prompt"], (
                        "Worker altered the prefix"
                    )
                except httpx.TransportError as error:
                    logger.warning(
                        "vLLM replica %s request %s failed on attempt %s: %r",
                        worker,
                        index,
                        attempt + 1,
                        error,
                    )
                    with lock:
                        self.stats[worker]["errors"] += 1
                        if attempt < self.retries:
                            pending.append((index, attempt + 1))
                        else:
                            failures.append(error)
                    continue
                except BaseException as error:
                    logger.exception(
                        "vLLM replica %s request %s aborted", worker, index
                    )
                    with lock:
                        failures.append(error)
                    return
                with lock:
                    assert results[index] is None, "Duplicate logical request"
                    results[index] = choices
                    self.stats[worker]["requests"] += 1
                    self.stats[worker]["tokens"] += payload["usage"][
                        "completion_tokens"
                    ]

        slots = max(1, self.max_sequences // n)
        with ThreadPoolExecutor(len(self.clients) * slots) as pool:
            for future in [
                pool.submit(drain, worker)
                for _ in range(slots)
                for worker in range(len(self.clients))
            ]:
                future.result()
        if failures:
            raise failures[0]
        assert all(result is not None for result in results), (
            "Workers stopped before completing every request"
        )
        return results


def _healthy(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=5.0).status_code == 200
    except httpx.TransportError:
        return False


def publish_adapter(
    client: VLLMClient, model, directory: Path, prefix: str, manifest: dict
) -> str:
    """Export the adapter into a new directory and load it on every replica.

    The directory appears atomically, after the adapter and its manifest are
    complete. The live model is neither merged nor mutated. Returns the unique
    served name, which embeds the adapter content hash.
    """
    staging = directory.with_name(f".{directory.name}.staging")
    model.save_pretrained(str(staging), save_embedding_layers=False)
    digest = file_sha256(staging / "adapter_model.safetensors")
    name = f"{prefix}-{digest[:16]}"
    manifest = manifest | {
        "name": name,
        "adapter_sha256": digest,
        "expected_acknowledgments": len(client.clients),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(directory)
    client.post_all(
        "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(directory)}
    )
    assert all(name in served for served in client.served()), (
        "A worker did not acknowledge the published adapter"
    )
    with (directory.parent / "published.jsonl").open("a") as stream:
        stream.write(json.dumps(manifest) + "\n")
    return name


def retire_adapter(client: VLLMClient, name: str, directory: Path) -> None:
    """Unload a drained adapter everywhere, then delete its export."""
    client.post_all("/v1/unload_lora_adapter", {"lora_name": name})
    assert not any(name in served for served in client.served()), (
        "A worker still serves a retired adapter"
    )
    shutil.rmtree(directory)


class VLLMInference:
    """HFInference's generation contract, sampled by replicas of `model`.

    `model` is the borrowed learner actor: gradient scoring and updates stay on
    the learner, and `sync` publishes its adapter as actor version N. Calls are
    synchronous, so every request of version N has drained before `sync(N + 1)`.
    """

    def __init__(
        self,
        model,
        client: VLLMClient,
        *,
        pad_token_id: int,
        export_dir: Path,
        seed: int,
        manifest: dict,
    ):
        assert lora_identity(model) is not None, (
            "Remote sampling publishes LoRA adapters"
        )
        assert type(pad_token_id) is int and 0 <= pad_token_id < model.config.vocab_size
        self.model = model
        self.client = client
        self.pad_token_id = pad_token_id
        self.export_dir = export_dir
        self.seed = seed
        self.manifest = manifest
        self.version: int | None = None
        self.adapter: str | None = None
        self.calls = 0

    def close(self) -> None:
        self.client.close()
        self.model = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def sync(self, version: int) -> None:
        """Publish the learner's current adapter as `version`; retire the previous."""
        self.export_dir.mkdir(exist_ok=True)
        previous, previous_version = self.adapter, self.version
        self.adapter = publish_adapter(
            self.client,
            self.model,
            self.export_dir / f"v{version}",
            f"actor-v{version}",
            self.manifest | {"version": version},
        )
        if previous is not None:
            retire_adapter(
                self.client, previous, self.export_dir / f"v{previous_version}"
            )
        self.version = version
        self.calls = 0

    def generate(
        self,
        prefixes: Sequence[Sequence[int]],
        *,
        num_samples: int = 1,
        max_new_tokens: int | Sequence[int],
        temperature: float = 1.0,
    ) -> list[list[Completion]]:
        """Return continuation-only completions in [prefix][sample] order.

        max_new_tokens is one budget, or one budget per prefix. Sampling is
        unfiltered (no top-k/top-p/min-p) and unpenalized, as in HFInference.
        """
        assert self.adapter is not None, "sync() must publish an actor before sampling"
        vocab = self.model.config.vocab_size
        budgets = (
            [max_new_tokens] * len(prefixes)
            if type(max_new_tokens) is int
            else list(max_new_tokens)
        )
        assert type(num_samples) is int and num_samples > 0
        assert isinstance(temperature, (int, float)) and isfinite(temperature)
        assert temperature >= 0
        assert temperature > 0 or num_samples == 1, "Greedy decoding needs one sample"
        assert len(prefixes) == len(budgets) > 0
        assert all(type(budget) is int and budget > 0 for budget in budgets)
        assert all(
            len(prefix) > 0 and all(type(t) is int and 0 <= t < vocab for t in prefix)
            for prefix in prefixes
        ), "Prefixes must contain valid integer token IDs"
        assert all(
            len(prefix) + budget <= self.model.config.max_position_embeddings
            for prefix, budget in zip(prefixes, budgets, strict=True)
        ), "Prefix + max_new_tokens exceeds the model context"
        eos = self.model.generation_config.eos_token_id
        eos_ids = [eos] if type(eos) is int else eos
        assert isinstance(eos_ids, list) and eos_ids, "Configure EOS token IDs"

        call, self.calls = self.calls, self.calls + 1
        results = self.client.complete(
            [
                {
                    "model": self.adapter,
                    "prompt": list(prefix),
                    "n": num_samples,
                    "seed": stream_seed(self.seed, self.version, call, index),
                    "max_tokens": budget,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": 0,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "stop_token_ids": eos_ids,
                    "add_special_tokens": False,
                    "skip_special_tokens": False,
                    "return_token_ids": True,
                }
                for index, (prefix, budget) in enumerate(
                    zip(prefixes, budgets, strict=True)
                )
            ]
        )
        completions = []
        for choices, budget in zip(results, budgets, strict=True):
            group = []
            for choice in choices:
                tokens, reason = choice["token_ids"], choice["finish_reason"]
                assert reason in ("stop", "length"), f"Unsupported finish: {reason!r}"
                ends = [i for i, token in enumerate(tokens) if token in eos_ids]
                assert ends == ([len(tokens) - 1] if reason == "stop" else []), (
                    "EOS must end exactly the stopped continuations"
                )
                assert reason == "stop" or len(tokens) == budget, (
                    "Generation stopped without EOS before exhausting its budget"
                )
                assert 0 < len(tokens) <= budget
                group.append(Completion(tokens, reason))
            completions.append(group)
        return completions

    def generate_each(
        self,
        prefixes: Sequence[Sequence[int]],
        budgets: Sequence[int],
        *,
        num_samples: int,
        temperature: float,
    ) -> list[list[Completion]]:
        """All prefixes are in flight together, each with its own exact budget."""
        return self.generate(
            prefixes,
            num_samples=num_samples,
            max_new_tokens=budgets,
            temperature=temperature,
        )

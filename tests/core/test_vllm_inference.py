"""Remote-sampler contracts against a deterministic fake vLLM replica; no GPU."""

import json
import random
import threading
from types import SimpleNamespace

import httpx
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from algorithms.vineppo import estimate_vineppo_batch
from core.data import Episode
from core.lora import apply_lora
from core.vllm_inference import BASE_MODEL, VLLMClient, VLLMInference

EOS = 2


def sample(body, index):
    """Deterministic in (prefix, seed + sibling); independent of arrival order."""
    rng = random.Random(f"{body['prompt']}:{body['seed'] + index}")
    tokens = []
    while len(tokens) < body["max_tokens"]:
        tokens.append(rng.choice([EOS, 5, 6, 7, 8]))
        if tokens[-1] == EOS:
            return tokens, "stop"
    return tokens, "length"


class FakeReplica:
    """The subset of vLLM's OpenAI server the sampler uses."""

    def __init__(self, *, shuffle_seed=0, fail=0, mutate=None):
        self.served = {BASE_MODEL}
        self.requests = []
        self.active = self.peak = 0
        self.rng = random.Random(shuffle_seed)
        self.fail = fail
        self.mutate = mutate
        self.lock = threading.Lock()

    def __call__(self, request):
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in self.served]})
        body = json.loads(request.content)
        if path == "/v1/load_lora_adapter":
            assert body["lora_name"] not in self.served
            self.served.add(body["lora_name"])
            return httpx.Response(200, text="ok")
        if path == "/v1/unload_lora_adapter":
            if body["lora_name"] not in self.served:
                return httpx.Response(404, text="unknown adapter")
            self.served.remove(body["lora_name"])
            return httpx.Response(200, text="ok")
        assert path == "/v1/completions"
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            fail, self.fail = self.fail > 0, self.fail - 1
            delay = self.rng.random() * 0.01
        threading.Event().wait(delay)  # Scramble completion order across threads.
        with self.lock:
            self.active -= 1
            self.requests.append(body)
        if fail:
            raise httpx.ConnectError("worker died")
        assert body["model"] in self.served, "Unknown adapter"
        choices = []
        for index in range(body["n"]):
            tokens, reason = sample(body, index)
            choices.append(
                {
                    "index": index,
                    "token_ids": tokens,
                    "finish_reason": reason,
                    "prompt_token_ids": body["prompt"],
                }
            )
        self.rng.shuffle(choices)
        payload = {
            "model": body["model"],
            "choices": choices,
            "usage": {"completion_tokens": sum(len(c["token_ids"]) for c in choices)},
        }
        if self.mutate is not None:
            self.mutate(payload)
        return httpx.Response(200, json=payload)


@pytest.fixture
def actor():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        return apply_lora(
            GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=16,
                    n_positions=32,
                    n_embd=8,
                    n_layer=1,
                    n_head=1,
                    eos_token_id=EOS,
                )
            ),
            rank=2,
            alpha=4.0,
        )


def make(actor, tmp_path, replicas, name="actors", **options):
    client = VLLMClient(
        [f"http://worker-{i}" for i in range(len(replicas))],
        api_key="key",
        transport=httpx.MockTransport(
            lambda request: replicas[int(request.url.host.split("-")[1])](request)
        ),
        **options,
    )
    inference = VLLMInference(
        actor,
        client,
        pad_token_id=0,
        export_dir=tmp_path / name,
        seed=42,
        manifest={"base_model": "local"},
    )
    return inference


PREFIXES = [[1, 5], [1, 5, 6, 7], [1], [1, 8, 8, 8, 8, 8], [1, 5]]
BUDGETS = [6, 3, 1, 9, 6]


def test_token_contract_order_budgets_and_explicit_sampling(actor, tmp_path):
    replica = FakeReplica()
    inference = make(actor, tmp_path, [replica])
    inference.sync(0)
    groups = inference.generate_each(PREFIXES, BUDGETS, num_samples=9, temperature=1.0)
    assert [len(group) for group in groups] == [9] * 5
    for prefix, budget, group in zip(PREFIXES, BUDGETS, groups, strict=True):
        got = [(c.token_ids, c.finish_reason) for c in group]
        # Exactly this prefix's request produced the group, in sibling order.
        assert got in [
            [sample(body, index) for index in range(9)]
            for body in replica.requests
            if body["prompt"] == prefix
        ]
        for tokens, reason in got:
            assert len(tokens) <= budget and EOS not in tokens[:-1]
            assert (tokens[-1] == EOS) == (reason == "stop")
    assert groups[0] != groups[4], "Duplicate prefixes need independent draws"
    assert {body["max_tokens"] for body in replica.requests} == set(BUDGETS)
    assert any(
        c.finish_reason == "stop" and len(c.token_ids) == b
        for b, g in zip(BUDGETS, groups)
        for c in g
    ), "EOS on the final token"
    # Duplicate prefixes are distinct logical requests with distinct streams.
    assert len({body["seed"] for body in replica.requests}) == 5
    for body in replica.requests:
        assert body["model"] == inference.adapter != BASE_MODEL
        assert (body["top_p"], body["top_k"], body["min_p"]) == (1.0, 0, 0.0)
        assert body["repetition_penalty"] == 1.0
        assert body["presence_penalty"] == body["frequency_penalty"] == 0.0
        assert body["add_special_tokens"] is False and "stop" not in body
        assert body["stop_token_ids"] == [EOS] and body["temperature"] == 1.0


def test_arrival_order_and_layout_cannot_change_samples_or_advantages(actor, tmp_path):
    episodes = [
        ("a", Episode([1, 5], [6, 7, 8, 5, EOS], 1.0, "stop"), (0, 2, 5)),
        ("b", Episode([1], [5, 5, 5, 5], -1.0, "length"), (0, 1, 3, 4)),
    ]
    outcomes = []
    for name, replicas in (
        ("one", [FakeReplica(shuffle_seed=1)]),
        ("three", [FakeReplica(shuffle_seed=s) for s in (2, 3, 4)]),
    ):
        inference = make(actor, tmp_path, replicas, name)
        inference.sync(0)
        records = []
        estimates = estimate_vineppo_batch(
            [
                (key, episode, bounds, lambda ids, reason: float(sum(ids) % 3))
                for key, episode, bounds in episodes
            ],
            inference=inference,
            num_samples=4,
            max_new_tokens=6,
            max_sequence_length=32,
            temperature=1.0,
            record_prefix_samples=records.append,
        )
        outcomes.append(([e.raw_step_advantages for e in estimates], records))
        assert sum(len(r.requests) for r in replicas) == 5
        if len(replicas) == 3:
            assert all(replica.requests for replica in replicas), "Idle worker"
    assert outcomes[0] == outcomes[1]


def test_in_flight_work_is_bounded_and_concurrent(actor, tmp_path):
    replica = FakeReplica()
    inference = make(actor, tmp_path, [replica], max_sequences=12)
    inference.sync(0)
    inference.generate([[1, 5]] * 20, num_samples=4, max_new_tokens=4)
    assert 1 < replica.peak <= 3


def test_refresh_requires_every_acknowledgment_and_retires_old_actor(actor, tmp_path):
    replicas = [FakeReplica(), FakeReplica()]
    inference = make(actor, tmp_path, replicas)
    with pytest.raises(AssertionError, match="sync"):
        inference.generate([[1]], max_new_tokens=1)
    inference.sync(0)
    first = inference.adapter
    before = inference.generate([[1, 5]], num_samples=2, max_new_tokens=4)
    with torch.no_grad():
        next(p for p in actor.parameters() if p.requires_grad).add_(1.0)
    inference.sync(1)
    assert (
        inference.adapter != first
        and first.split("-")[-1] != inference.adapter.split("-")[-1]
    )
    assert all(r.served == {BASE_MODEL, inference.adapter} for r in replicas)
    assert not (tmp_path / "actors/v0").exists()
    manifest = json.loads((tmp_path / "actors/v1/manifest.json").read_text())
    assert manifest["version"] == 1 and manifest["expected_acknowledgments"] == 2
    assert manifest["name"] == inference.adapter
    assert len((tmp_path / "actors/published.jsonl").read_text().splitlines()) == 2
    after = inference.generate([[1, 5]], num_samples=2, max_new_tokens=4)
    assert before != after, "A new version must draw from new streams"
    replicas[1].served.clear()  # A worker that lost the adapter must fail loudly.
    with pytest.raises(AssertionError):
        inference.sync(2)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda p: p.update(model="actor-v0-stale"), "Stale actor"),
        (lambda p: p["choices"].pop(), "sibling"),
        (lambda p: p["choices"][0].update(index=1), "sibling"),
        (
            lambda p: [c.update(prompt_token_ids=[1, 1, 5]) for c in p["choices"]],
            "prefix",
        ),
        (lambda p: p["choices"][0].update(finish_reason="abort"), "Unsupported finish"),
        (lambda p: p["choices"][0].update(token_ids=[5, EOS, 6]), "EOS"),
        (
            lambda p: p["choices"][0].update(token_ids=[5], finish_reason="length"),
            "budget",
        ),
    ],
)
def test_stale_or_malformed_results_are_rejected(actor, tmp_path, mutate, message):
    inference = make(actor, tmp_path, [FakeReplica(mutate=mutate)])
    inference.sync(0)
    with pytest.raises(AssertionError, match=message):
        inference.generate([[1, 5]], num_samples=3, max_new_tokens=4)


def test_bounded_retry_reuses_the_logical_request_then_aborts(actor, tmp_path):
    healthy = make(actor, tmp_path, [FakeReplica()], "healthy")
    healthy.sync(0)
    expected = healthy.generate(PREFIXES, num_samples=3, max_new_tokens=5)
    flaky = [FakeReplica(fail=2), FakeReplica()]
    inference = make(actor, tmp_path, flaky, "flaky", max_sequences=6)
    inference.sync(0)
    assert inference.generate(PREFIXES, num_samples=3, max_new_tokens=5) == expected
    assert inference.client.stats[0]["errors"] == 2
    dead = make(actor, tmp_path, [FakeReplica(fail=10**6)], "dead")
    dead.sync(0)
    with pytest.raises(httpx.ConnectError, match="worker died"):
        dead.generate(PREFIXES, num_samples=3, max_new_tokens=5)


def test_last_active_slot_drains_its_own_retry(actor, tmp_path, caplog):
    replica = FakeReplica(fail=1)
    inference = make(actor, tmp_path, [replica], max_sequences=1)
    inference.sync(0)

    result = inference.generate(PREFIXES[:1], num_samples=1, max_new_tokens=5)

    first, retry = replica.requests
    assert first == retry
    assert result[0][0].token_ids == sample(first, 0)[0]
    assert inference.client.stats[0]["errors"] == 1
    assert "worker died" in caplog.text


@pytest.mark.parametrize(
    "prefixes,options",
    [
        ([], {"max_new_tokens": 1}),
        ([[]], {"max_new_tokens": 1}),
        ([[99]], {"max_new_tokens": 1}),
        ([[1]], {"max_new_tokens": 0}),
        ([[1]], {"max_new_tokens": 32}),
        ([[1]], {"max_new_tokens": [1, 2]}),
        ([[1]], {"max_new_tokens": 1, "temperature": 0.0, "num_samples": 2}),
    ],
)
def test_invalid_requests_never_reach_workers(actor, tmp_path, prefixes, options):
    replica = FakeReplica()
    inference = make(actor, tmp_path, [replica])
    inference.sync(0)
    with pytest.raises(AssertionError):
        inference.generate(prefixes, **options)
    assert not replica.requests


def test_only_lora_actors_and_distinct_workers(actor, tmp_path):
    client = SimpleNamespace(clients=[])
    with pytest.raises(AssertionError, match="LoRA"):
        VLLMInference(
            actor.get_base_model().__class__(actor.config),
            client,
            pad_token_id=0,
            export_dir=tmp_path,
            seed=0,
            manifest={},
        )
    with pytest.raises(AssertionError, match="unique"):
        VLLMClient(["http://a", "http://a"], api_key="key")

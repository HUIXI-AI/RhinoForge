"""Pi0.5 prepare_graphs(language_token_counts=...): one denoise Graph per valid count."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api.policy import Pi05Policy

TOKENS = "observation.language.tokens"
MASK = "observation.language.attention_mask"


class _Cache:
    """Graph double keyed like the real caches: denoise by count, others by shape."""

    def __init__(self, capacity=192, arenas=0):
        self.capacity = capacity
        self.runtime_policy = SimpleNamespace(graph_arena_count=arenas)
        self.frozen = False
        self.clear()

    def begin_warmup(self):
        self.frozen = False

    def clear(self):
        self.entries = {}

    def call(self, key):
        if key in self.entries:
            self.entries[key] += 1
        else:
            assert not self.frozen, f"READY miss for {key}"
            self.entries[key] = 0

    def freeze(self):
        self.frozen = True

    def size(self):
        return len(self.entries)

    def max_entries(self):
        return self.capacity

    def snapshot(self):
        return [SimpleNamespace(replay_count=count, recapture_count=0)
                for count in self.entries.values()]

    def cache_invariant_ok(self):
        return True


def _batch(valid=5, width=8):
    tokens = torch.zeros(1, width, dtype=torch.int64)
    tokens[0, :valid] = torch.arange(2, 2 + valid)
    mask = torch.zeros(1, width, dtype=torch.bool)
    mask[0, :valid] = True
    return {"observation.state": torch.ones(1, 32), TOKENS: tokens, MASK: mask}


def _prepared_policy(monkeypatch, *, capacity=192, arenas=0, unstable_count=None):
    from rpu_backend.adapters.pi05 import Pi05Adapter, runtime

    monkeypatch.setattr(runtime, "_pi05_graph_runtime_profile", lambda *_: {
        "siglip_graph": True, "gemma_graph": True, "fused_denoise": True,
        "denoise_graph": True, "adarms_graph": False,
    })
    caches = {"vision": _Cache(), "prefill": _Cache(), "denoise": _Cache(capacity, arenas)}
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_inference_steps=10)
    model._rpu_fuse_validated = True
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(model=SimpleNamespace(
            vision_tower=SimpleNamespace(_rpu_siglip_graph_cache=caches["vision"]),
            language_model=SimpleNamespace(_rpu_gemma_graph_cache=caches["prefill"]),
        )),
        gemma_expert=SimpleNamespace(model=SimpleNamespace(_rpu_adarms_graph_cache=_Cache())),
    )
    model._rpu_fused_denoise_graph_cache = caches["denoise"]
    requests = []

    def predict(batch, **kwargs):
        count = int(batch[MASK].sum())
        requests.append((count, batch[TOKENS].clone(), batch[MASK].clone()))
        caches["vision"].call("images")
        caches["prefill"].call("prefix")
        caches["denoise"].call(count)
        value = float(count)
        if count == unstable_count and caches["denoise"].entries[count] > 0:
            value += 1.0
        return torch.full((1, 50, 7), value)

    lerobot = SimpleNamespace(config=SimpleNamespace(chunk_size=50, max_action_dim=32),
                              model=model, predict_action_chunk=predict)
    adapter = object.__new__(Pi05Adapter)
    adapter._rpu_is_ready = True
    adapter._lerobot_policy = lerobot
    policy = object.__new__(Pi05Policy)
    policy._rpu_ready = True
    policy._lerobot_policy = lerobot
    policy._adapter = adapter
    return policy, model, caches, requests, runtime


def test_each_count_builds_replays_and_reaches_ready(monkeypatch):
    policy, model, caches, requests, runtime = _prepared_policy(monkeypatch)
    try:
        result = policy.prepare_graphs(_batch(), language_token_counts=[7, 3, 5, 3])
        assert result["phase"] == "READY" and result["ready_replay_validated"]
        assert result["language_token_counts"] == [3, 5, 7]
        assert result["graph_counts"] == {"vision": 1, "prefill": 1, "denoise": 3}
        # Representative RECORD, each extra count RECORD + WARMING, representative
        # WARMING (after the others), then READY for every count.
        assert [count for count, _, _ in requests] == [5, 3, 3, 7, 7, 5, 5, 3, 7]
        assert all(cache.frozen for cache in caches.values())
        assert min(caches["denoise"].entries.values()) >= 2
        # Probe tokens repeat the representative's valid tokens; the mask stays right-padded.
        _, tokens, mask = requests[3]
        assert tokens[0].tolist() == [2, 3, 4, 5, 6, 2, 3, 0]
        assert mask[0].tolist() == [True] * 7 + [False]
        _, tokens, _ = requests[1]
        assert tokens[0].tolist() == [2, 3, 4, 0, 0, 0, 0, 0]
        runtime._validate_prepared_language_count(model, _batch(7)[MASK])
        with pytest.raises(RuntimeError, match="prepared for"):
            runtime._validate_prepared_language_count(model, _batch(4)[MASK])
    finally:
        runtime._PI05_PREPARED_GRAPH_PROFILES.pop(model, None)


def test_default_prepare_keeps_one_denoise_graph_and_no_count_check(monkeypatch):
    policy, model, caches, requests, runtime = _prepared_policy(monkeypatch)
    try:
        result = policy.prepare_graphs(_batch())
        assert result["language_token_counts"] is None
        assert result["graph_counts"] == {"vision": 1, "prefill": 1, "denoise": 1}
        assert len(requests) == 3
        # Without prepared counts the request is not refused here (READY lookup decides).
        runtime._validate_prepared_language_count(model, _batch(4)[MASK])
    finally:
        runtime._PI05_PREPARED_GRAPH_PROFILES.pop(model, None)


@pytest.mark.parametrize("counts, error, match", [
    ([3, 7], ValueError, "representative"),
    ([0, 5], ValueError, r"\[1, 8\]"),
    ([5, 9], ValueError, r"\[1, 8\]"),
    ([5, True], TypeError, "ints"),
    ([], TypeError, "nonempty"),
    ("5", TypeError, "iterable"),
])
def test_invalid_counts_fail_before_any_graph(monkeypatch, counts, error, match):
    policy, model, caches, requests, runtime = _prepared_policy(monkeypatch)
    with pytest.raises(error, match=match):
        policy.prepare_graphs(_batch(), language_token_counts=counts)
    assert not requests and model not in runtime._PI05_PREPARED_GRAPH_PROFILES


def test_capacity_and_left_padding_fail_before_any_graph(monkeypatch):
    policy, model, caches, requests, runtime = _prepared_policy(monkeypatch, capacity=2)
    with pytest.raises(ValueError, match="holds 2"):
        policy.prepare_graphs(_batch(), language_token_counts=[3, 5, 7])
    batch = _batch()
    batch[MASK] = batch[MASK].flip(1)
    with pytest.raises(ValueError, match="right-padded"):
        policy.prepare_graphs(batch, language_token_counts=[5])
    assert not requests


def test_an_undersized_cold_arena_plan_fails_before_any_graph(monkeypatch):
    # Vision, prefill, AdaRMS safety, one transient queue, one denoise graph per count.
    assert Pi05Policy.graph_arena_count() == 5 and Pi05Policy.graph_arena_count(41) == 45
    policy, model, caches, requests, runtime = _prepared_policy(monkeypatch, arenas=6)
    with pytest.raises(ValueError, match=r"needs 7 Graph arenas.*graph_arena_count=7"):
        policy.prepare_graphs(_batch(), language_token_counts=[3, 5, 7])
    assert not requests and model not in runtime._PI05_PREPARED_GRAPH_PROFILES
    policy, model, caches, requests, runtime = _prepared_policy(monkeypatch, arenas=7)
    try:
        assert policy.prepare_graphs(_batch(), language_token_counts=[3, 5, 7])["phase"] == "READY"
    finally:
        runtime._PI05_PREPARED_GRAPH_PROFILES.pop(model, None)


def test_count_replay_mismatch_unwinds_to_warming(monkeypatch):
    policy, model, caches, requests, runtime = _prepared_policy(monkeypatch, unstable_count=7)
    with pytest.raises(RuntimeError, match="language count 7 WARMING replay differs"):
        policy.prepare_graphs(_batch(), language_token_counts=[3, 5, 7])
    assert not any(cache.frozen for cache in caches.values())
    assert model not in runtime._PI05_PREPARED_GRAPH_PROFILES

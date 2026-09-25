"""Check Pi05 runtime boundaries, diagnostic metrics and Graph admission.

The production numerical helper and prepare_graphs transaction run here. Only
native Graph caches and the policy's three tiny action results are doubled.
"""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.pi05 import Pi05Adapter, runtime


class _Owner:
    pass


class _GraphCache:
    def __init__(self):
        self.frozen = False
        self.clear()

    def begin_warmup(self):
        self.frozen = False

    def clear(self):
        self.entries = 0
        self.replays = 0
        self.recaptures = 0
        self.invariant = True

    def call(self):
        if self.entries:
            self.replays += 1
        else:
            assert not self.frozen
            self.entries = 1

    def freeze(self):
        self.frozen = True

    def size(self):
        return self.entries

    def snapshot(self):
        return [SimpleNamespace(replay_count=self.replays, recapture_count=self.recaptures)]

    def cache_invariant_ok(self):
        return self.invariant


@pytest.fixture
def prepare_owner(monkeypatch):
    owners = []
    monkeypatch.setattr(runtime, "_pi05_graph_runtime_profile", lambda *_: {
        "siglip_graph": True, "gemma_graph": True, "fused_denoise": True,
        "denoise_graph": True, "adarms_graph": False,
    })

    def make(actions, graph_fault=None):
        model = _Owner()
        owners.append(model)
        model.config = SimpleNamespace(num_inference_steps=10)
        model._rpu_fuse_validated = True
        caches = [_GraphCache() for _ in range(3)]
        model.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(model=SimpleNamespace(
                vision_tower=SimpleNamespace(_rpu_siglip_graph_cache=caches[0]),
                language_model=SimpleNamespace(_rpu_gemma_graph_cache=caches[1]),
            )),
            gemma_expert=SimpleNamespace(model=SimpleNamespace(_rpu_adarms_graph_cache=_GraphCache())),
        )
        model._rpu_fused_denoise_graph_cache = caches[2]
        state = {"calls": 0, "actions": actions}

        def predict(_batch, *, num_steps, **kwargs):
            state.setdefault("noise_inputs", []).append(kwargs.get("noise"))
            assert num_steps == 10
            index = state["calls"]
            state["calls"] += 1
            for cache in caches:
                cache.call()
            if index == 2:
                if graph_fault == "recapture":
                    caches[0].recaptures += 1
                elif graph_fault == "inventory":
                    caches[0].entries += 1
                elif graph_fault == "invariant":
                    caches[0].invariant = False
                elif graph_fault == "replay_count":
                    caches[0].replays = 0
            return state["actions"][index]

        adapter = object.__new__(Pi05Adapter)
        adapter._rpu_is_ready = True
        adapter._lerobot_policy = SimpleNamespace(model=model, predict_action_chunk=predict)
        return adapter, model, caches, state

    yield make
    for model in owners:
        runtime._PI05_PREPARED_GRAPH_PROFILES.pop(model, None)


def test_prepare_admits_existing_microdifference_and_reports_actual_equality(prepare_owner):
    build = torch.ones(1, 50, 7)
    warm = build + 0.001953125
    ready = warm + 0.001953125
    adapter, model, caches, _ = prepare_owner([build, warm, ready])
    result = adapter.prepare_graphs({})
    assert result["phase"] == "READY" and result["ready_replay_validated"]
    assert result["build_replay_bit_equal"] is False
    assert result["action_bit_equal"] is False
    assert result["build_replay_max_abs"] == 0.001953125
    assert result["replay_parity"]["contract"] == "pi05-runtime-boundary-and-numeric-diagnostics-v2"
    for comparison in ("build_vs_warming", "warming_vs_ready"):
        assert result["replay_parity"][comparison]["max_abs"] == 0.001953125
        assert result["replay_parity"][comparison]["row_single_zero_count"] == 0
    assert runtime._PI05_PREPARED_GRAPH_PROFILES[model]["phase"] == "READY"
    assert all(cache.frozen and cache.replays == 2 for cache in caches)


def test_prepare_keeps_exact_equality_diagnostics_when_equal(prepare_owner):
    action = torch.ones(1, 50, 7)
    adapter, _, _, _ = prepare_owner([action] * 3)
    result = adapter.prepare_graphs({})
    assert result["action_bit_equal"] is True
    assert result["build_replay_bit_equal"] is True
    assert result["build_replay_max_abs"] == 0


@pytest.mark.parametrize("failure_stage", [1, 2], ids=["WARMING", "READY"])
@pytest.mark.parametrize("kind,error", [
    ("shape", "shape mismatch"),
    ("dtype", "torch.float32 boundary"), ("nan", "non-finite"),
])
def test_prepare_rejects_invalid_action_and_allows_clean_retry(
    prepare_owner, failure_stage, kind, error,
):
    base = torch.ones(1, 50, 7)
    bad = base.clone()
    if kind == "shape":
        bad = torch.ones(1, 50, 8)
    elif kind == "dtype":
        bad = bad.half()
    elif kind == "nan":
        bad[0, 0, 0] = float("nan")
    actions = [base, base, base]
    actions[failure_stage] = bad
    adapter, model, caches, state = prepare_owner(actions)
    with pytest.raises(RuntimeError, match=error):
        adapter.prepare_graphs({})
    assert model not in runtime._PI05_PREPARED_GRAPH_PROFILES
    assert all(not cache.frozen for cache in caches)
    state.update(actions=[base] * 3, calls=0)
    assert adapter.prepare_graphs({})["phase"] == "READY"


@pytest.mark.parametrize("fault,error", [
    ("recapture", "grew or recaptured"), ("inventory", "grew or recaptured"),
    ("invariant", "invariant failed"), ("replay_count", "two replays"),
])
def test_numeric_diagnostics_do_not_bypass_graph_hard_gates(prepare_owner, fault, error):
    action = torch.ones(1, 50, 7)
    adapter, model, _, _ = prepare_owner([action] * 3, graph_fault=fault)
    with pytest.raises(RuntimeError, match=error):
        adapter.prepare_graphs({})
    assert model not in runtime._PI05_PREPARED_GRAPH_PROFILES


def _validate(actual, reference):
    return runtime._validate_fused_parity(actual, reference, expected_shape=None)


def test_default_internal_shape_is_preserved_and_public_shape_must_match():
    cropped = torch.ones(1, 50, 7)
    with pytest.raises(RuntimeError, match="expected shape"):
        runtime._validate_fused_parity(cropped, cropped)
    assert _validate(cropped, cropped)["mse"] == 0
    full = torch.ones(1, 50, 32)
    assert runtime._validate_fused_parity(full, full)["mse"] == 0
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _validate(full, cropped)


@pytest.mark.parametrize("side", ["actual", "reference"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_on_either_side_is_a_hard_failure(side, bad):
    tensors = {"actual": torch.ones(1, 50, 7), "reference": torch.ones(1, 50, 7)}
    tensors[side][0, 0, 0] = bad
    with pytest.raises(RuntimeError, match="non-finite"):
        _validate(**tensors)


@pytest.mark.parametrize("side", ["actual", "reference"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32])
def test_cpu_fp32_contract_is_not_implicitly_cast(side, dtype):
    tensors = {"actual": torch.ones(1, 50, 7), "reference": torch.ones(1, 50, 7)}
    tensors[side] = tensors[side].to(dtype)
    with pytest.raises(RuntimeError, match="torch.float32 boundary"):
        _validate(**tensors)


def test_non_cpu_boundary_is_rejected_before_reading_values():
    with pytest.raises(RuntimeError, match="CPU boundary"):
        _validate(torch.empty(1, 50, 7, device="meta"), torch.ones(1, 50, 7))


def test_zero_row_differences_are_reported():
    zeros = torch.zeros(1, 50, 7)
    result = _validate(zeros, zeros)
    assert result["row_both_zero_count"] == 50 and result["row_cosine_min"] == 1
    tiny = zeros.clone()
    tiny[0, 0, 0] = 1e-7
    for actual, reference in ((tiny, zeros), (zeros, tiny)):
        assert _validate(actual, reference)["row_single_zero_count"] == 1


def test_finite_differences_are_measured_without_an_arbitrary_ceiling(prepare_owner):
    reference = torch.ones(1, 50, 7)
    actual = reference + 0.5
    metrics = _validate(actual, reference)
    assert metrics["mse"] == 0.25 and metrics["max_abs"] == 0.5
    adapter, _, _, _ = prepare_owner([reference, actual, actual])
    result = adapter.prepare_graphs({})
    assert result["replay_parity"]["numerical_status"] == "DIAGNOSTIC"
    assert result["replay_parity"]["task_quality"] == "NOT_EVALUATED"
    assert result["replay_parity"]["build_vs_warming"]["mse"] == 0.25


def test_prepare_forwards_explicit_noise_through_build_warm_and_ready(prepare_owner):
    action = torch.ones(1, 50, 7)
    adapter, _, _, state = prepare_owner([action] * 3)
    noise = torch.randn(1, 50, 32)
    original = noise.clone()
    assert adapter.prepare_graphs({}, noise=noise)["phase"] == "READY"
    assert len(state["noise_inputs"]) == 3
    assert all(value is noise for value in state["noise_inputs"])
    assert torch.equal(noise, original)

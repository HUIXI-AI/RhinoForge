"""Board-free transaction and support-boundary checks for Gemma4-E4B."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import pytest
import torch

import rpu_backend
from rpu_backend.adapters import gemma4
from rpu_backend.adapters.gemma4 import cache as cache_module
from rpu_backend.adapters.gemma4 import runtime, weights
from rpu_backend.api import causal_lm
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.runtime.execution_planner import _physical_manifest_fingerprint


class _FakeTensor:
    def float(self):
        return self

    def to(self, *args, **kwargs):
        return self


@pytest.mark.parametrize("length,position,stage,canonical,chunk", [
    (33, 16, "prefill", 16, 32), (1, 520, "decode", 0, 0),
])
def test_cold_and_forward_planner_bind_the_actual_owner_and_window(
    monkeypatch, length, position, stage, canonical, chunk,
):
    from rpu_backend.runtime import decoder
    owner = SimpleNamespace(_rpu_execution={"prefill": {"chunk_size": 32}},
                            config=SimpleNamespace(text_config=SimpleNamespace(sliding_window=512)),
                            _execution_session=SimpleNamespace(generation=7))
    marker = object()
    native_calls = []
    monkeypatch.setattr(torch.ops.rpu, "gemma4_resolve_stage_domain",
                        lambda *args: native_calls.append(args) or (), raising=False)
    def planner(real_len, physical_limit, budget, **options):
        assert (real_len, physical_limit, budget) == (length, 1 if length == 1 else 1024 - position, 0)
        assert options["execution_owner"] is owner
        assert options["execution_native"] == ("gemma4", 77)
        assert options["execution_stage"] == stage
        assert options["position"] == canonical
        assert options["physical_metadata"] == (("execution_generation", 7),)
        options["resolve_stage_domain"](length)
        options["plan_result_sink"](marker)
        return length, chunk
    monkeypatch.setattr(decoder, "plan_bounded_prefill_execution", planner)
    assert gemma4._execution_plan(owner, 77, length, position, 1024) is marker
    assert native_calls == [(77, length, canonical, 528 if length == 1 else position + length, chunk)]


class _FakeActivation(_FakeTensor):
    def __mul__(self, other):
        return self


class _FakeCache:
    def __init__(self, geom, max_seq_len, device):
        self.geom = geom
        self.max_seq_len = max_seq_len
        self.device = device
        self.position = 0
        self._seen_tokens = 0
        self.k_caches = []
        self.v_caches = []


def _text_config():
    return SimpleNamespace(
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=42,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        hidden_size_per_layer_input=256,
        num_kv_shared_layers=18,
        vocab_size=262144,
        vocab_size_per_layer_input=262144,
        max_position_embeddings=131072,
        layer_types=list(gemma4._SUPPORTED_LAYER_TYPES),
        attention_k_eq_v=False,
        enable_moe_block=False,
        num_experts=None,
        use_double_wide_mlp=False,
        use_bidirectional_attention=None,
        hidden_activation="gelu_pytorch_tanh",
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        final_logit_softcapping=30.0,
        sliding_window=512,
    )


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(text_config=_text_config())

    def to(self, *args, **kwargs):
        return self

    def forward(self, *args, **kwargs):
        return "original-forward"

    def state_dict(self):
        tensor = _FakeTensor()
        prefix = runtime.PFX
        return {
            prefix + "norm.weight": tensor,
            prefix + "embed_tokens.weight": tensor,
            prefix + "embed_tokens_per_layer.weight": tensor,
            prefix + "per_layer_model_projection.weight": tensor,
            prefix + "per_layer_projection_norm.weight": tensor,
        }


@pytest.fixture
def install_runtime(monkeypatch):
    import gc
    gc.collect()
    from rpu_backend.api import _execution
    from rpu_backend.runtime import _native_retirement
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    state = {
        "next_handle": 1,
        "alive": set(),
        "events": [],
        "set_failure": None,
        "destroy_fails": False,
        "claims": [],
        "releases": [],
        "graph_caches": [],
        "forward_chunks": [],
        "forward_descriptors": [],
        "resolver_calls": [],
        "forward_output_seq_len": 1,
        "captures": [],
    }

    def claim(model):
        state["claims"].append(model)

    def release(model):
        state["releases"].append(model)

    def create():
        handle = state["next_handle"]
        state["next_handle"] += 1
        state["alive"].add(handle)
        state["events"].append(("create", handle))
        return handle

    def destroy(handle):
        state["events"].append(("destroy", handle))
        if state["destroy_fails"]:
            raise RuntimeError("synthetic destroy failure")
        if handle not in state["alive"]:
            raise RuntimeError(f"double destroy for {handle}")
        state["alive"].remove(handle)

    def set_weights(handle, *args, **kwargs):
        state["events"].append(("set_weights", handle))
        failure = state["set_failure"]
        if failure is not None:
            raise failure

    def native_forward(handle, *args, **kwargs):
        state["events"].append(("forward", handle))
        state["forward_chunks"].append(args[6])
        state["forward_descriptors"].append(tuple(args[8]))
        return torch.zeros(1, state["forward_output_seq_len"], 2560)

    def resolve_stage_domain(
        handle, execution_len, position, max_kv_len, requested_chunk_size
    ):
        del handle
        state["resolver_calls"].append((
            execution_len, position, max_kv_len, requested_chunk_size,
        ))
        capacity = (
            requested_chunk_size
            if requested_chunk_size
            else max(16, ((execution_len + 15) // 16) * 16)
        )
        schedule = []
        offset = 0
        while offset < execution_len:
            length = min(capacity, execution_len - offset)
            schedule.extend((
                len(schedule) // 4,
                offset,
                length,
                position + offset + length,
            ))
            offset += length
        count = len(schedule) // 4
        route = (1, 101, 2, 0, (), 0)
        manifest_values = [
            1, execution_len, execution_len, 0,
            max(max_kv_len, position + execution_len),
            execution_len, 1, 1, 1,
            route[0], route[1], route[2], route[3], 0,
        ]
        manifest_fingerprint = _physical_manifest_fingerprint(
            manifest_values
        )
        descriptor = [
            3, 0, capacity, capacity, capacity,
            0, 1, 0, 0, 0, 0,
            count, count, count, 1,
            *schedule, *schedule, *schedule,
            0, execution_len, -1,
            1, execution_len, execution_len, 0,
            max(max_kv_len, position + execution_len),
            execution_len, 1, 1, 1,
            manifest_fingerprint >> 32,
            manifest_fingerprint & 0xFFFFFFFF,
            route[0], route[1], route[2], route[3], 0, route[5],
        ]
        descriptor[1] = len(descriptor)
        return [1, 1, len(descriptor), *descriptor]

    class GraphCache:
        def __init__(self, *, runtime_policy=None):
            from rpu_backend.runtime.execution_planner import PreparedExecutionPlans
            self._execution_plans = PreparedExecutionPlans(32)
            self.runtime_policy = runtime_policy
            self.cleared = False
            self.ready = False
            self.signatures = []
            state["graph_caches"].append(self)

        def is_frozen(self):
            return self.ready

        def prepare_plan(self, key, factory):
            return self._execution_plans.prepare(key, factory)

        def begin_warmup(self):
            self._execution_plans.begin_warmup()
            self.ready = False
            state["events"].append(("graph_warmup", None))

        def freeze(self):
            assert self.signatures
            self._execution_plans.freeze()
            self.ready = True

        def clear(self):
            self._execution_plans.clear()
            self.cleared = True
            self.signatures.clear()
            state["events"].append(("graph_clear", None))

        def cache_invariant_ok(self):
            return True

        @contextmanager
        def capture(self, signature):
            key = signature.kwargs
            if key not in self.signatures:
                if self.ready:
                    raise RuntimeError("READY miss")
                self.signatures.append(key)
            state["events"].append(("capture", None))
            state["captures"].append(signature)
            yield

    class GraphSignature:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(causal_lm, "_claim_live_instance", claim)
    monkeypatch.setattr(causal_lm, "_release_live_instance", release)
    monkeypatch.setattr(
        runtime,
        "swizzle_decoder_and_ple",
        lambda raw, geom: ({"q": []}, {}),
    )
    monkeypatch.setattr(runtime, "compute_side_input", lambda *a, **k: _FakeTensor())
    monkeypatch.setattr(
        weights,
        "make_rope_tables",
        lambda *args, **kwargs: (_FakeTensor(),) * 4,
    )
    monkeypatch.setattr(weights, "set_gemma4_weights", set_weights)
    monkeypatch.setattr(cache_module, "Gemma4KVCache", _FakeCache)
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", GraphCache)
    monkeypatch.setattr(rpu_backend.graph, "GraphSignature", GraphSignature)
    monkeypatch.setattr(torch.ops.rpu, "gemma4_create", create, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "gemma4_planner_cache_identity", lambda handle: (handle, 1), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "gemma4_set_chunk_envelope",
                        lambda handle, max_kv_len, chunk: None, raising=False)
    monkeypatch.setattr(
        torch.ops.rpu, "gemma4_destroy", destroy, raising=False
    )
    monkeypatch.setattr(
        torch.ops.rpu, "gemma4_forward", native_forward, raising=False
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "gemma4_resolve_stage_domain",
        resolve_stage_domain,
        raising=False,
    )
    return state


def _adapter(model=None, *, rpu_execution=None):
    return gemma4.Gemma4Adapter(
        model or _Model(), rpu_execution=rpu_execution
    )


def _finalize(model):
    resource = getattr(model, "_rpu_gemma4_retirement_state", None)
    if resource is not None:
        resource.retire_owned(model)


def test_profile_preflight_accepts_only_certified_e4b():
    config = SimpleNamespace(text_config=_text_config())
    gemma4.Gemma4Adapter.preflight(config)

    config.text_config.hidden_size = 2048
    with pytest.raises(UnsupportedModelError, match="only the certified E4B"):
        gemma4.Gemma4Adapter.preflight(config)

    config = SimpleNamespace(text_config=_text_config())
    config.text_config.hidden_activation = "silu"
    with pytest.raises(UnsupportedModelError, match="hidden_activation"):
        gemma4.Gemma4Adapter.preflight(config)

    config = SimpleNamespace(text_config=_text_config())
    config.text_config.num_experts = "invalid"
    with pytest.raises(UnsupportedModelError, match="num_experts='invalid'"):
        gemma4.Gemma4Adapter.preflight(config)


def test_execution_config_accepts_only_prefill_chunk_size():
    adapter = _adapter(
        rpu_execution={"prefill": {"chunk_size": 32}}
    )
    assert adapter.model._rpu_execution == {
        "prefill": {"chunk_size": 32}
    }

    with pytest.raises(ValueError, match="not supported"):
        _adapter(rpu_execution={"prefill": {"padding_rows": 0}})

    with pytest.raises(ValueError, match="not supported"):
        _adapter(rpu_execution={"vision": {"chunk_size": 32}})


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "4096"])
def test_invalid_max_seq_fails_before_claim_or_native_create(
    install_runtime, value
):
    adapter = _adapter()
    adapter.model._rpu_max_seq = value

    with pytest.raises(ValueError, match="positive integer"):
        adapter.to_rpu()

    assert install_runtime["claims"] == []
    assert install_runtime["events"] == []


def test_requested_capacity_above_checkpoint_default_is_installed(install_runtime):
    adapter = _adapter()
    adapter.model._rpu_max_seq = 131073

    model = adapter.to_rpu()
    assert model.build_rpu_cache().max_seq_len == 131073
    assert install_runtime["claims"]


def test_prepare_failure_releases_owner_without_creating_handle(
    install_runtime, monkeypatch
):
    adapter = _adapter()

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("synthetic prepare failure")

    monkeypatch.setattr(
        runtime, "swizzle_decoder_and_ple", fail_prepare
    )

    with pytest.raises(RuntimeError, match="prepare failure"):
        adapter.to_rpu()

    assert install_runtime["claims"] == [adapter.model]
    assert install_runtime["releases"] == [adapter.model]
    assert install_runtime["events"] == []
    assert not any(hasattr(adapter.model, name) for name in gemma4._INSTALL_ATTRS)


def test_graph_prepare_failure_releases_owner_before_native_create(
    install_runtime, monkeypatch
):
    adapter = _adapter()

    class FailingGraphCache:
        def __init__(self, *, runtime_policy=None):
            raise RuntimeError("synthetic graph prepare failure")

    monkeypatch.setattr(rpu_backend.graph, "GraphCache", FailingGraphCache)
    with pytest.raises(RuntimeError, match="graph prepare failure"):
        adapter.to_rpu()

    assert install_runtime["releases"] == [adapter.model]
    assert install_runtime["events"] == []


def test_finalizer_registration_failure_precedes_native_create(
    install_runtime, monkeypatch
):
    adapter = _adapter()

    def fail_finalizer(*args, **kwargs):
        raise RuntimeError("synthetic finalizer failure")

    monkeypatch.setattr(weakref, "finalize", fail_finalizer)
    with pytest.raises(RuntimeError, match="finalizer failure"):
        adapter.to_rpu()

    assert install_runtime["events"] == []
    assert install_runtime["alive"] == set()
    assert install_runtime["releases"] == [adapter.model]


def test_set_weights_failure_cleans_graph_then_handle_and_releases_owner(
    install_runtime,
):
    adapter = _adapter()
    model = adapter.model
    install_runtime["set_failure"] = RuntimeError("synthetic setter failure")

    with pytest.raises(RuntimeError, match="setter failure"):
        adapter.to_rpu()

    assert install_runtime["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("graph_clear", None),
        ("destroy", 1),
    ]
    assert install_runtime["alive"] == set()
    assert install_runtime["releases"] == [model]
    assert not any(hasattr(model, name) for name in gemma4._INSTALL_ATTRS)
    assert "forward" not in vars(model)
    assert model.forward() == "original-forward"


def test_base_exception_uses_the_same_cleanup_path(install_runtime):
    class InstallInterrupted(BaseException):
        pass

    adapter = _adapter()
    install_runtime["set_failure"] = InstallInterrupted()

    with pytest.raises(InstallInterrupted):
        adapter.to_rpu()

    assert install_runtime["events"][-2:] == [
        ("graph_clear", None),
        ("destroy", 1),
    ]
    assert install_runtime["releases"] == [adapter.model]


def test_publication_failure_rollback_bypasses_attribute_hooks(install_runtime):
    class InstallInterrupted(BaseException):
        pass

    interruption = InstallInterrupted("synthetic publication interruption")

    class RejectingModel(_Model):
        def __init__(self):
            object.__setattr__(self, "_reject_publication", False)
            super().__init__()

        def __setattr__(self, name, value):
            if (
                self.__dict__.get("_reject_publication", False)
                and name == "_rpu_gemma4_handle_finalizer"
            ):
                raise interruption
            super().__setattr__(name, value)

        def __delattr__(self, name):
            if (
                self.__dict__.get("_reject_publication", False)
                and name.startswith("_rpu_gemma4_")
            ):
                raise AssertionError("rollback re-entered __delattr__")
            super().__delattr__(name)

    model = RejectingModel()
    adapter = _adapter(model)
    object.__setattr__(model, "_reject_publication", True)

    with pytest.raises(InstallInterrupted) as exc_info:
        adapter.to_rpu()

    assert exc_info.value is interruption
    assert install_runtime["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("graph_clear", None),
        ("destroy", 1),
    ]
    assert install_runtime["alive"] == set()
    assert install_runtime["releases"] == [model]
    assert not any(hasattr(model, name) for name in gemma4._INSTALL_ATTRS)
    assert "forward" not in vars(model)
    assert model.forward() == "original-forward"


def test_destroy_failure_poison_keeps_process_owner(install_runtime):
    adapter = _adapter()
    install_runtime["set_failure"] = RuntimeError("synthetic setter failure")
    install_runtime["destroy_fails"] = True

    with pytest.raises(RuntimeError, match="setter failure"):
        adapter.to_rpu()

    assert install_runtime["releases"] == []
    assert adapter.model._rpu_gemma4_install_poisoned is True
    with pytest.raises(RPUBackendError, match="could not be fully cleaned"):
        adapter.to_rpu()
    assert len(install_runtime["claims"]) == 1


def test_success_publishes_finalizer_and_reentry_is_noop(install_runtime):
    adapter = _adapter()
    model = adapter.to_rpu()

    assert model._rpu_gemma4_handle == 1
    assert model._rpu_gemma4_handle_finalizer.alive
    assert model._rpu_gemma4_ready is True
    assert adapter.to_rpu() is model
    assert install_runtime["events"] == [
        ("create", 1),
        ("set_weights", 1),
    ]

    with pytest.raises(ValueError, match="positive integer"):
        model.build_rpu_cache(0)
    with pytest.raises(ValueError, match="exceeds the installed RoPE horizon"):
        model.build_rpu_cache(gemma4.DEFAULT_MAX_SEQ + 1)
    cache = model.build_rpu_cache()
    assert isinstance(cache, _FakeCache)
    assert cache.max_seq_len == gemma4.DEFAULT_MAX_SEQ

    _finalize(model)
    assert install_runtime["alive"] == set()


@pytest.mark.parametrize(
    ("configured_chunk", "expected_resolved_chunk"),
    [("auto", 32), (32, 32)],
)
def test_prefill_chunk_config_reaches_native_descriptor_and_graph_signature(
    install_runtime, monkeypatch, configured_chunk, expected_resolved_chunk
):
    model = _adapter(
        rpu_execution={"prefill": {"chunk_size": configured_chunk}}
    ).to_rpu()
    cache = model.build_rpu_cache()
    monkeypatch.setattr(
        torch.nn.functional, "embedding", lambda *a, **k: _FakeActivation()
    )
    monkeypatch.setattr(
        runtime,
        "softcap_lm_logits",
        lambda hidden, *_args: torch.zeros(
            hidden.shape[0], hidden.shape[1], 8
        ),
    )

    install_runtime["forward_output_seq_len"] = 32
    model.forward(
        input_ids=torch.arange(32, dtype=torch.long).reshape(1, 32),
        past_key_values=cache,
    )
    assert install_runtime["forward_chunks"] == [0]
    assert install_runtime["forward_descriptors"][-1]
    assert install_runtime["resolver_calls"][-1][-1] == (
        0 if configured_chunk == "auto" else configured_chunk
    )
    receipt = model._rpu_last_execution_plan
    assert receipt["resolved_chunk_size"] == expected_resolved_chunk
    graph_words = tuple(
        int(receipt["physical_plan_digest"][offset:offset + 8], 16)
        for offset in range(0, 64, 8)
    )
    dyn_dims = install_runtime["captures"][-1].kwargs["dyn_dims"]
    assert dyn_dims[-9] == expected_resolved_chunk
    assert tuple(dyn_dims[-8:]) == graph_words

    install_runtime["forward_output_seq_len"] = 1
    model.forward(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        past_key_values=cache,
    )
    assert install_runtime["forward_chunks"][-1] == 0
    assert install_runtime["forward_descriptors"][-1]
    assert install_runtime["resolver_calls"][-1][1] == 0
    assert cache.position == 33
    _finalize(model)


def test_concurrent_build_is_rejected_before_claim(install_runtime):
    adapter = _adapter()
    assert gemma4._BUILD_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(RPUBackendError, match="another Gemma4 build"):
            adapter.to_rpu()
    finally:
        gemma4._BUILD_LOCK.release()

    assert install_runtime["claims"] == []
    assert install_runtime["events"] == []


def test_forward_rejects_unsupported_semantics_before_graph_dispatch(
    install_runtime,
):
    model = _adapter().to_rpu()
    ids = torch.tensor([[1]], dtype=torch.long)
    cache = model.build_rpu_cache()

    calls = [
        ({"input_ids": ids, "inputs_embeds": torch.ones(1, 1, 1)}, ValueError),
        ({"input_ids": ids.float()}, TypeError),
        ({"input_ids": ids, "labels": ids}, NotImplementedError),
        ({"input_ids": ids, "use_cache": False}, NotImplementedError),
        ({"input_ids": ids, "return_dict": False}, NotImplementedError),
        ({"input_ids": ids, "logits_to_keep": -1}, ValueError),
        ({"input_ids": ids, "past_key_values": object()}, TypeError),
        ({
            "input_ids": ids,
            "past_key_values": cache,
            "position_ids": torch.tensor([[0.0]]),
        }, TypeError),
    ]
    for kwargs, error_type in calls:
        with pytest.raises(error_type):
            model.forward(**kwargs)

    assert all(event[0] != "forward" for event in install_runtime["events"])
    _finalize(model)


def test_same_config_calibration_retires_actual_gemma4_graph(install_runtime):
    from rpu_backend.api._execution import record_planner_native
    from rpu_backend.runtime.execution_planner import PlannerCalibrationSelection, PlannerCostScope
    model = _adapter().to_rpu()
    session = model._execution_session
    record_planner_native(model, "prefill", "", ("gemma4", model._rpu_gemma4_handle))
    selection = PlannerCalibrationSelection(
        PlannerCostScope("a" * 64, "b" * 64, "c" * 64), "d" * 64, "e" * 64)
    old = session.config
    install_runtime["events"].clear()
    session._select_planner_calibration(model, "prefill", selection, native_prefix="gemma4")
    assert session.config == old and session.generation == 1
    assert install_runtime["events"] == [("graph_warmup", None), ("graph_clear", None)]
    _finalize(model)


@pytest.mark.parametrize("rollback", [False, True])
def test_hot_import_after_ready_retires_receipt_and_can_build(install_runtime, monkeypatch, rollback):
    model = _adapter().to_rpu()
    cache, session = model.build_rpu_cache(), model._execution_session
    graph = model._rpu_gemma4_graph_cache
    monkeypatch.setattr(torch.nn.functional, "embedding", lambda *a, **k: _FakeActivation())
    monkeypatch.setattr(runtime, "softcap_lm_logits", lambda *a, **k: torch.ones(1, 1, 8))
    ids = torch.tensor([[1]], dtype=torch.long)
    model.forward(input_ids=ids, past_key_values=cache)
    old_receipt = model._rpu_last_execution_plan
    graph.freeze()
    old_config, old_generation = session.config, session.generation
    if rollback:
        apply = session._apply
        def interrupted(*args, **kwargs):
            apply(*args, **kwargs)
            raise RuntimeError("after graph retirement")
        session._apply = interrupted
        with pytest.raises(RuntimeError, match="after graph retirement"):
            session.reconfigure({"prefill": {"chunk_size": 16}})
        assert session.config == old_config and session.generation == old_generation
        assert session.stats()["state"] == "QUIESCENT"
    else:
        session.reconfigure({"prefill": {"chunk_size": 16}})
        assert session.generation == old_generation + 1
    assert not graph.ready and not graph.signatures
    assert not hasattr(model, "_rpu_last_execution_plan")
    cache.position = cache._seen_tokens = 0
    model.forward(input_ids=ids, past_key_values=cache)
    graph.freeze()
    assert model._rpu_last_execution_plan is not old_receipt
    assert model._rpu_last_execution_plan["selected"]["physical_metadata"]["execution_generation"] == session.generation
    cache.position = cache._seen_tokens = 0
    model.forward(input_ids=ids, past_key_values=cache)
    assert len(graph.signatures) == 1
    _finalize(model)


def test_cache_position_advances_only_after_logits_succeed(
    install_runtime, monkeypatch
):
    model = _adapter().to_rpu()
    cache = model.build_rpu_cache()
    monkeypatch.setattr(
        torch.nn.functional, "embedding", lambda *a, **k: _FakeActivation()
    )

    def fail_logits(*args, **kwargs):
        raise RuntimeError("synthetic logits failure")

    monkeypatch.setattr(runtime, "softcap_lm_logits", fail_logits)
    with pytest.raises(RuntimeError, match="logits failure"):
        model.forward(
            input_ids=torch.tensor([[1]], dtype=torch.long),
            past_key_values=cache,
        )

    assert ("forward", 1) in install_runtime["events"]
    assert cache.position == 0
    assert cache._seen_tokens == 0
    _finalize(model)


def test_decode_planning_reuses_graph_mask_bucket_and_checks_real_horizon(monkeypatch):
    from rpu_backend.runtime import decoder
    from test_graph_cache_freeze import _cache

    graph = _cache()
    owner = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(sliding_window=512)),
                            _rpu_execution={}, _execution_session=SimpleNamespace(generation=0))
    native_calls = []
    monkeypatch.setattr(torch.ops.rpu, "gemma4_resolve_stage_domain",
                        lambda *args: native_calls.append(args), raising=False)
    def planner(real, limit, budget, **kw):
        assert (real, limit, budget) == (1, 1, 0)
        assert kw["graph_cache"] is graph
        def resolve():
            kw["resolve_stage_domain"](1)
            return object()
        marker = graph.prepare_plan((limit, kw["position"], kw["plan_signature"]), resolve)
        kw["plan_result_sink"](marker)
        return 1, 16
    monkeypatch.setattr(decoder, "plan_bounded_prefill_execution", planner)
    plans = [gemma4._execution_plan(owner, 77, 1, position, 1024, graph_cache=graph)
             for position in (100, 511, 512, 519, 527, 528)]
    assert plans[0] is plans[1]
    assert plans[2] is plans[3] is plans[4]
    assert plans[1] is not plans[2] and plans[4] is not plans[5]
    assert native_calls == [(77, 1, 0, kv, 0) for kv in (1, 528, 544)]
    gemma4._execution_plan(owner, 77, 1, 519, 520, graph_cache=graph)
    assert native_calls[-1] == (77, 1, 0, 520, 0)
    count = len(native_calls)
    with pytest.raises(ValueError, match="RoPE/KV horizon"):
        gemma4._execution_plan(owner, 77, 1, 520, 520, graph_cache=graph)
    assert len(native_calls) == count

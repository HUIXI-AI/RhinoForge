"""Board-free support and composite-install contracts for Qwen3-VL."""
from __future__ import annotations

import inspect
import weakref
from types import SimpleNamespace

import pytest
import torch

import rpu_backend
from rpu_backend.adapters import qwen3_vl
from rpu_backend.adapters.qwen3_vl import text as qwen3_vl_text
from rpu_backend.runtime import decoder as shared_decoder
from rpu_backend.api._execution import (
    reconfigure_rpu_execution,
    rpu_execution_stats,
)
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError


def _config():
    return SimpleNamespace(
        model_type="qwen3_vl",
        text_config=SimpleNamespace(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
        ),
        vision_config=SimpleNamespace(
            hidden_size=1024,
            intermediate_size=4096,
            depth=24,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[5, 11, 17],
            hidden_act="gelu_pytorch_tanh",
            out_hidden_size=2048,
        ),
    )


class _TextModel:
    def __init__(self):
        self.layers = []
        self.embed_tokens = SimpleNamespace(
            weight=torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        )

    def to(self, *args, **kwargs):
        return self

    def forward(self, *args, **kwargs):
        return "original-text-forward"


class _VisionModel:
    def forward(self, *args, **kwargs):
        return "original-vision-forward"


class _LmHead:
    def __init__(self):
        self.weight = torch.nn.Parameter(
            torch.full((2, 2), 2.0), requires_grad=False
        )

    def to(self, *args, **kwargs):
        return self


class _Model:
    def get_submodule(self, path):
        value = self
        for component in path.split("."):
            value = getattr(value, component)
        return value

    def __init__(self):
        self.config = _config()
        self.model = SimpleNamespace(
            language_model=_TextModel(),
            visual=_VisionModel(),
        )
        self.lm_head = _LmHead()

    def to(self, *args, **kwargs):
        return self

    def forward(self, *args, **kwargs):
        return "original-top-forward"


class _Cache:
    def __init__(self, state, name):
        self._state = state
        self._name = name

    def clear(self):
        self._state["events"].append(("clear", self._name))

    def begin_warmup(self):
        self._state["events"].append(("warmup", self._name))

    def cache_invariant_ok(self):
        return True

    def prepare_plan(self, key, prepare):
        return prepare()

    def is_frozen(self):
        return False


def _decode_descriptor(has_lm_head):
    """Small valid native wire with the actual lm-head topology site."""
    from rpu_backend.runtime.execution_planner import _physical_manifest_fingerprint

    route = (9, 3145128204432384004, 1, 0, 2, int(has_lm_head), 151936 if has_lm_head else 0)
    manifest = (1, 1, 1, 0, 1, 1, 1, 3, 1)
    fingerprint = _physical_manifest_fingerprint((*manifest, *route))
    descriptor = [
        3, 0, 16, 16, 16, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1,
        *([0, 0, 1, 1] * 3), 0, 1, -1,
        *manifest, fingerprint >> 32, fingerprint & 0xFFFFFFFF,
        *route[:5], 0, *route[5:],
    ]
    descriptor[1] = len(descriptor)
    return tuple(descriptor)


class _Finalizer:
    def __init__(self, state, name):
        self._state = state
        self._name = name
        self.alive = True

    def __call__(self):
        if self.alive:
            self.alive = False
            self._state["events"].append(("destroy", self._name))
        return True

    def detach(self):
        self.alive = False


@pytest.fixture
def install_runtime(monkeypatch):
    from rpu_backend.api import _execution
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(qwen3_vl, "_FAILED_RETIREMENTS", [])
    children = []
    state = {
        "events": [], "claims": [], "set_error": None,
        "vision_kwargs": None, "text_kwargs": None,
        "lm_head_set": False, "decode_plans": [], "plan_error": None,
        "allocator_enabled": False, "allocator_error": None,
        "claim_error": None,
    }

    def set_caching_allocator(enabled):
        state["events"].append(("set_caching_allocator", enabled))
        if state["allocator_error"] is not None:
            raise state["allocator_error"]
        state["allocator_enabled"] = enabled

    def move_to_rpu(model, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        if device is not None and torch.device(device).type == "rpu":
            assert state["allocator_enabled"] is True
            state["events"].append(("rpu_allocation", type(model).__name__))
        return model

    monkeypatch.setattr(torch.rpu, "set_caching_allocator", set_caching_allocator)
    monkeypatch.setattr(_TextModel, "to", move_to_rpu)
    monkeypatch.setattr(_LmHead, "to", move_to_rpu)

    def claim(model):
        if state["claim_error"] is not None:
            raise state["claim_error"]
        if not state["claims"]:
            assert state["allocator_enabled"] is False
        state["claims"].append(model)
        state["events"].append(("claim", "model"))

    def install_vision(model, **kwargs):
        children.append(weakref.ref(model))
        state["events"].append(("install", "vision"))
        state["vision_kwargs"] = kwargs
        model._rpu_vision_handle = 11
        model._rpu_vision_graph_cache = _Cache(state, "vision")
        model._rpu_vision_handle_finalizer = _Finalizer(state, "vision")
        model.forward = lambda *args, **kwargs: "rpu-vision-forward"
        return 11

    def install_text(model, **kwargs):
        children.append(weakref.ref(model))
        state["events"].append(("install", "text"))
        state["text_kwargs"] = kwargs
        model._rpu_decoder_handle = 22
        model._rpu_text_graph_cache = _Cache(state, "text")
        model._rpu_decoder_graph_cache = _Cache(state, "decoder")
        model._rpu_decoder_handle_finalizer = _Finalizer(state, "text")
        model._rpu_decode_stage_descriptor = resolve_decode(22)
        model.forward = lambda *args, **kwargs: "rpu-text-forward"
        return 22

    def set_lm_head(handle, weight):
        state["events"].append(("set_lm_head", handle))
        if state["set_error"] is not None:
            raise state["set_error"]
        state["lm_head_set"] = True

    def resolve_decode(handle):
        state["decode_plans"].append(state["lm_head_set"])
        if state["lm_head_set"] and state["plan_error"] is not None:
            raise state["plan_error"]
        return _decode_descriptor(state["lm_head_set"])

    def set_linear_acc32(handle, enabled):
        state["events"].append(("set_linear_acc32", handle, enabled))

    def set_vision_linear_acc32(handle, enabled):
        state["events"].append(("set_vision_linear_acc32", handle, enabled))

    def enable_vision_mergers(model):
        state["events"].append(("enable_vision_mergers", model._rpu_vision_handle))
        model._rpu_vision_merger_on_device = True
        model._rpu_patch_embed_on_device = True

    monkeypatch.setattr(qwen3_vl, "_claim_live_instance", claim)
    monkeypatch.setattr(
        qwen3_vl,
        "convert_linear_weights_inplace",
        lambda *args, **kwargs: state["events"].append(("swizzle", "text")),
    )
    monkeypatch.setattr(
        qwen3_vl,
        "transform_linear_weight",
        lambda weight, partition: weight.clone(),
    )
    monkeypatch.setattr(
        qwen3_vl, "install_qwen3_vl_vision_for_rpu", install_vision
    )
    monkeypatch.setattr(
        qwen3_vl, "install_qwen3_vl_text_for_rpu", install_text
    )
    monkeypatch.setattr(
        qwen3_vl, "enable_qwen3_vl_vision_merger_on_device", enable_vision_mergers
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "causal_decoder_set_lm_head",
        set_lm_head,
        raising=False,
    )
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor", resolve_decode, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_planner_cache_identity",
                        lambda handle: (handle, int(state["lm_head_set"])), raising=False)
    for op, name in (("causal_decoder_destroy", "text"), ("qwen3vl_vision_destroy", "vision")):
        monkeypatch.setattr(torch.ops.rpu, op,
                            lambda handle, name=name: state["events"].append(("destroy", name)), raising=False)
    monkeypatch.setattr(
        torch.ops.rpu,
        "causal_decoder_enable_execution_reconfigure",
        lambda _handle: None,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "causal_decoder_set_linear_acc32",
        set_linear_acc32,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3vl_vision_set_linear_acc32",
        set_vision_linear_acc32,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "execution_reconfigure_begin",
        lambda attempt: state["events"].append(("begin", attempt)) or 91,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "execution_reconfigure_commit",
        lambda token: state["events"].append(("commit", token)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "execution_reconfigure_abort",
        lambda token: state["events"].append(("abort", token)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "execution_reconfigure_abort_attempt",
        lambda attempt: state["events"].append(("abort_attempt", attempt)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "causal_decoder_stage_chunk_size_override",
        lambda handle, token, chunk: state["events"].append(
            ("stage_text", handle, token, chunk)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3vl_vision_stage_chunk_size",
        lambda handle, token, chunk: state["events"].append(
            ("stage_vision", handle, token, chunk)
        ),
        raising=False,
    )
    yield state
    # Do not run simulated native handles after these CPU doubles are removed.
    # Weak references leave actual cyclic-GC tests free to collect in the body.
    for reference in children:
        parent = getattr(reference(), "_qwen3_vl_retirement_owner", None)
        if parent is not None:
            parent._gc_retirement_enabled = False


def test_profile_requires_the_certified_vision_tower():
    config = _config()
    qwen3_vl.Qwen3VLAdapter.preflight(config)

    config.vision_config.intermediate_size = 4304
    with pytest.raises(UnsupportedModelError, match="unsupported vision profile"):
        qwen3_vl.Qwen3VLAdapter.preflight(config)


def test_exact_8b_profile_selects_padding_only_at_the_top_level(
    install_runtime,
):
    model = _Model()
    text = model.config.text_config
    text.hidden_size = 4096
    text.intermediate_size = 12288
    text.num_hidden_layers = 36
    text.num_attention_heads = 32
    text.num_key_value_heads = 8
    text.head_dim = 128
    text.model_type = "qwen3_vl_text"
    text.hidden_act = "silu"
    text.rms_norm_eps = 1e-6
    text.vocab_size = 151936
    text.attention_bias = False
    text.use_cache = True
    text.rope_scaling = {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
        "rope_theta": 5_000_000,
    }
    model.config.tie_word_embeddings = False
    vision = model.config.vision_config
    vision.hidden_size = 1152
    vision.intermediate_size = 4304
    vision.depth = 27
    vision.num_heads = 16
    vision.deepstack_visual_indexes = [8, 16, 24]
    vision.out_hidden_size = 4096
    vision.model_type = "qwen3_vl"
    vision.in_channels = 3
    vision.num_position_embeddings = 2304

    adapter = qwen3_vl.Qwen3VLAdapter(model)
    assert install_runtime["claims"] == []
    assert install_runtime["events"] == []
    assert not hasattr(model, "_rpu_swizzle_started")

    assert adapter.to_rpu() is model
    assert install_runtime["vision_kwargs"]["_allow_padded_8b"] is True
    assert install_runtime["vision_kwargs"]["_allow_graph_blocked_32b"] is False


def test_broken_ready_marker_is_rejected_instead_of_noop(install_runtime):
    model = _Model()
    model._rpu_swizzled = True
    adapter = qwen3_vl.Qwen3VLAdapter(model)

    with pytest.raises(RPUBackendError, match="runtime is incomplete"):
        adapter.to_rpu()

    assert install_runtime["claims"] == []
    assert install_runtime["events"] == []


@pytest.mark.parametrize("error_type", [RuntimeError, BaseException])
def test_post_install_failure_cleans_graphs_and_handles_and_poison_reentry(
    install_runtime, error_type
):
    adapter = qwen3_vl.Qwen3VLAdapter(_Model())
    model = adapter.model
    install_runtime["set_error"] = error_type("synthetic lm-head failure")

    with pytest.raises(error_type, match="lm-head failure"):
        adapter.to_rpu()

    assert install_runtime["events"][-5:] == [
        ("clear", "text"),
        ("clear", "decoder"),
        ("clear", "vision"),
        ("destroy", "text"),
        ("destroy", "vision"),
    ]
    assert model._rpu_swizzle_started is True
    assert model._rpu_swizzled is False
    assert not hasattr(model, "_rpu_lm_head_w_keepalive")
    assert "forward" not in vars(model)
    assert "forward" not in vars(model.model.language_model)
    assert "forward" not in vars(model.model.visual)
    assert not any(
        name.startswith("_rpu_decoder_") or name.startswith("_rpu_text_")
        for name in vars(model.model.language_model)
    )
    assert not any(
        name.startswith("_rpu_vision_")
        for name in vars(model.model.visual)
    )
    with pytest.raises(RPUBackendError, match="prior swizzle attempt"):
        adapter.to_rpu()
    assert len(install_runtime["claims"]) == 1


def test_ready_publication_rollback_bypasses_all_model_attribute_hooks(
    install_runtime,
):
    class InstallInterrupted(BaseException):
        pass

    interruption = InstallInterrupted("synthetic ready publication interruption")
    fault = {"fired": False}

    class RejectingCleanupMixin:
        def __setattr__(self, name, value):
            if fault["fired"]:
                raise AssertionError("rollback re-entered __setattr__")
            super().__setattr__(name, value)

        def __delattr__(self, name):
            if fault["fired"]:
                raise AssertionError("rollback re-entered __delattr__")
            super().__delattr__(name)

    class RejectingText(RejectingCleanupMixin, _TextModel):
        pass

    class RejectingVision(RejectingCleanupMixin, _VisionModel):
        pass

    class RejectingTop(_Model):
        def __setattr__(self, name, value):
            if fault["fired"]:
                raise AssertionError("rollback re-entered top-level __setattr__")
            if (
                self.__dict__.get("_interrupt_ready_publication", False)
                and name == "_rpu_swizzled"
            ):
                fault["fired"] = True
                raise interruption
            super().__setattr__(name, value)

        def __delattr__(self, name):
            if fault["fired"]:
                raise AssertionError("rollback re-entered top-level __delattr__")
            super().__delattr__(name)

    model = RejectingTop()
    model.model.language_model = RejectingText()
    model.model.visual = RejectingVision()
    adapter = qwen3_vl.Qwen3VLAdapter(model)
    vars(model)["_interrupt_ready_publication"] = True

    with pytest.raises(InstallInterrupted) as exc_info:
        adapter.to_rpu()

    assert exc_info.value is interruption
    assert install_runtime["events"][-5:] == [
        ("clear", "text"),
        ("clear", "decoder"),
        ("clear", "vision"),
        ("destroy", "text"),
        ("destroy", "vision"),
    ]
    assert model._rpu_swizzle_started is True
    assert model._rpu_swizzled is False
    assert "forward" not in vars(model)
    assert "forward" not in vars(model.model.language_model)
    assert "forward" not in vars(model.model.visual)
    assert not any(
        name.startswith("_rpu_decoder_") or name.startswith("_rpu_text_")
        for name in vars(model.model.language_model)
    )
    assert not any(
        name.startswith("_rpu_vision_")
        for name in vars(model.model.visual)
    )


def test_success_is_published_last_and_reentry_is_a_noop(install_runtime):
    adapter = qwen3_vl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()

    assert model._rpu_swizzled is True
    assert adapter._rpu_is_ready is True
    assert model.model.language_model._rpu_decoder_handle == 22
    assert model.model.visual._rpu_vision_handle == 11
    assert install_runtime["text_kwargs"]["deepstack_lang_layers"] == [0, 1, 2]
    assert install_runtime["events"][:2] == [
        ("claim", "model"),
        ("set_caching_allocator", True),
    ]
    assert install_runtime["events"].count(("set_caching_allocator", True)) == 1
    claim_index = install_runtime["events"].index(("claim", "model"))
    allocator_index = install_runtime["events"].index(
        ("set_caching_allocator", True)
    )
    allocations = [
        index for index, event in enumerate(install_runtime["events"])
        if event[0] == "rpu_allocation"
    ]
    assert allocations and all(
        claim_index < allocator_index < index for index in allocations
    )
    assert install_runtime["events"][-3:] == [
        ("set_vision_linear_acc32", 11, False),
        ("set_linear_acc32", 22, False),
        ("set_lm_head", 22),
    ]
    events = list(install_runtime["events"])
    assert adapter.to_rpu() is model
    assert install_runtime["events"] == events


def test_dense_install_streams_each_layer_after_claim_and_allocator(
    install_runtime, monkeypatch,
):
    model = _Model()
    events = install_runtime["events"]

    class Layer:
        def __init__(self, name):
            self.name = name
            self.projection = SimpleNamespace(
                weight=torch.empty(1, dtype=torch.float16)
            )

        def get_submodule(self, _name):
            return self.projection

        def to(self, device):
            assert str(device) == "rpu"
            assert install_runtime["allocator_enabled"] is True
            events.append(("transfer", self.name))
            return self

    layers = [Layer("layer-0"), Layer("layer-1")]
    model.model.language_model.layers = layers

    def swizzle(layer, *, skip_names):
        assert layer in layers
        assert skip_names == set()
        events.append(("swizzle", layer.name))

    monkeypatch.setattr(qwen3_vl, "convert_linear_weights_inplace", swizzle)
    monkeypatch.setattr(
        qwen3_vl.gc,
        "collect",
        lambda: events.append(("gc", "install")) or 0,
    )

    assert qwen3_vl.Qwen3VLAdapter(model).to_rpu() is model
    assert events[:8] == [
        ("claim", "model"),
        ("set_caching_allocator", True),
        ("swizzle", "layer-0"),
        ("transfer", "layer-0"),
        ("gc", "install"),
        ("swizzle", "layer-1"),
        ("transfer", "layer-1"),
        ("gc", "install"),
    ]


def test_changed_profile_rejects_before_allocator_or_model_mutation(install_runtime):
    model = _Model()
    adapter = qwen3_vl.Qwen3VLAdapter(model)
    original_head = model.lm_head.weight
    model.config.vision_config.intermediate_size = 4304

    with pytest.raises(UnsupportedModelError, match="unsupported vision profile"):
        adapter.to_rpu()

    assert install_runtime["events"] == []
    assert install_runtime["claims"] == []
    assert install_runtime["allocator_enabled"] is False
    assert not getattr(model, "_rpu_swizzle_started", False)
    assert model.lm_head.weight is original_head


def test_late_delivery_rejection_precedes_allocator_mutation(
    install_runtime, monkeypatch
):
    adapter = qwen3_vl.Qwen3VLAdapter(_Model())

    def reject_delivery(_model):
        raise RPUBackendError("synthetic invalid delivery weights")

    monkeypatch.setattr(qwen3_vl, "_validate_delivery_weights", reject_delivery)
    with pytest.raises(RPUBackendError, match="invalid delivery weights"):
        adapter.to_rpu()

    assert install_runtime["events"] == []
    assert install_runtime["claims"] == []
    assert install_runtime["allocator_enabled"] is False
    assert not getattr(adapter.model, "_rpu_swizzle_started", False)


def test_allocator_initialization_failure_releases_claim_and_swizzle_lock(
    install_runtime, monkeypatch,
):
    adapter = qwen3_vl.Qwen3VLAdapter(_Model())
    install_runtime["allocator_error"] = RuntimeError("synthetic allocator init failure")
    releases = []
    monkeypatch.setattr(qwen3_vl, "_release_live_instance", releases.append)

    with pytest.raises(RuntimeError, match="allocator init failure"):
        adapter.to_rpu()

    assert install_runtime["events"] == [
        ("claim", "model"),
        ("set_caching_allocator", True),
    ]
    assert install_runtime["claims"] == [adapter.model]
    assert releases == [adapter.model]
    assert install_runtime["allocator_enabled"] is False
    assert not getattr(adapter.model, "_rpu_swizzle_started", False)
    # A failed cold initializer has not poisoned a partially transformed model.
    install_runtime["allocator_error"] = None
    assert adapter.to_rpu() is adapter.model


def test_pre_mutation_publication_failure_releases_claim_and_allows_retry(
    install_runtime, monkeypatch,
):
    class PublicationInterrupted(BaseException):
        pass

    interruption = PublicationInterrupted("synthetic owner publication failure")

    class InterruptingAdapter(qwen3_vl.Qwen3VLAdapter):
        def __setattr__(self, name, value):
            if (
                name == "_gc_retirement_enabled"
                and value is True
                and self.__dict__.get("_interrupt_publication", False)
            ):
                raise interruption
            super().__setattr__(name, value)

    adapter = InterruptingAdapter(_Model())
    adapter._interrupt_publication = True
    releases = []
    monkeypatch.setattr(qwen3_vl, "_release_live_instance", releases.append)

    with pytest.raises(PublicationInterrupted) as exc_info:
        adapter.to_rpu()

    assert exc_info.value is interruption
    assert install_runtime["events"] == [
        ("claim", "model"),
        ("set_caching_allocator", True),
    ]
    assert releases == [adapter.model]
    assert not adapter._gc_retirement_enabled
    assert not hasattr(adapter.model, "_qwen3_vl_retirement_owner")
    assert not hasattr(adapter.model, "_rpu_swizzle_started")

    adapter._interrupt_publication = False
    assert adapter.to_rpu() is adapter.model


def test_live_claim_rejection_does_not_change_allocator_policy(install_runtime):
    adapter = qwen3_vl.Qwen3VLAdapter(_Model())
    install_runtime["claim_error"] = RuntimeError("synthetic live owner")

    with pytest.raises(RuntimeError, match="live owner"):
        adapter.to_rpu()

    assert install_runtime["events"] == []
    assert install_runtime["claims"] == []
    assert install_runtime["allocator_enabled"] is False
    assert not getattr(adapter.model, "_rpu_swizzle_started", False)


def test_public_vision_chunk_reaches_the_native_installer(install_runtime):
    model = _Model()
    model._rpu_execution = {"vision": {"chunk_size": 512}}
    qwen3_vl.Qwen3VLAdapter(model).to_rpu()

    assert install_runtime["vision_kwargs"]["execution_chunk_size"] == 512


def test_component_overrides_win_over_top_level_defaults(install_runtime):
    model = _Model()
    model._rpu_execution = {
        "prefill": {"chunk_size": 64},
        "vision": {"chunk_size": 144},
        "components": {
            "language_model": {"prefill": {"chunk_size": 32}},
            "vision_encoder": {"vision": {"chunk_size": 256}},
        },
    }
    adapter = qwen3_vl.Qwen3VLAdapter(model)
    adapter.to_rpu()

    assert adapter._text_execution["prefill"]["chunk_size"] == 32
    assert adapter._vision_execution["vision"]["chunk_size"] == 256
    assert install_runtime["text_kwargs"]["execution_config"] == {
        "prefill": {"chunk_size": 32}
    }
    assert install_runtime["vision_kwargs"]["execution_chunk_size"] == 256


def test_execution_session_reconfigures_children_and_invalidates_precisely(
    install_runtime,
):
    adapter = qwen3_vl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    install_runtime["events"].clear()

    reconfigure_rpu_execution(model, {
        "components": {
            "language_model": {"prefill": {"chunk_size": 64}},
        },
    })
    events = list(install_runtime["events"])
    assert any(event[:2] == ("stage_text", 22) for event in events)
    assert not any(event[0] == "stage_vision" for event in events)
    assert ("warmup", "text") in events
    assert ("warmup", "decoder") in events
    assert ("warmup", "vision") not in events
    assert model._qwen3_vl_execution_component_generations == {
        "language_model": 1,
        "vision_encoder": 0,
    }

    install_runtime["events"].clear()
    reconfigure_rpu_execution(model, {
        "components": {
            "vision_encoder": {"vision": {"chunk_size": 256}},
        },
    })
    events = list(install_runtime["events"])
    assert any(event[:2] == ("stage_vision", 11) for event in events)
    assert not any(event[0] == "stage_text" for event in events)
    assert ("warmup", "vision") in events
    assert ("warmup", "text") not in events
    assert model.model.visual._rpu_execution_graph_key_words == (1,)
    assert model._qwen3_vl_execution_component_generations == {
        "language_model": 1,
        "vision_encoder": 1,
    }
    stats = rpu_execution_stats(model)
    assert stats["generation"] == 2
    assert stats["commit_count"] == 2


def test_execution_session_rolls_back_native_and_python_state(install_runtime):
    adapter = qwen3_vl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    install_runtime["events"].clear()
    original_reset = adapter._reset_component_graphs
    calls = 0

    def fail_once(changed):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic graph retirement failure")
        return original_reset(changed)

    adapter._reset_component_graphs = fail_once
    with pytest.raises(RuntimeError, match="graph retirement failure"):
        reconfigure_rpu_execution(model, {
            "components": {
                "vision_encoder": {"vision": {"chunk_size": 256}},
            },
        })

    staged = [
        event[-1]
        for event in install_runtime["events"]
        if event[0] == "stage_vision"
    ]
    assert staged == [256, 0]
    assert model.model.visual._rpu_vision_execution_chunk_size == "auto"
    assert model._qwen3_vl_execution_component_generations == {
        "language_model": 0,
        "vision_encoder": 0,
    }
    stats = rpu_execution_stats(model)
    assert stats["state"] == "QUIESCENT"
    assert stats["generation"] == 0
    assert stats["failure_count"] == 1


def test_execution_session_begin_failure_does_not_rollback_or_poison(
    install_runtime, monkeypatch,
):
    adapter = qwen3_vl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    install_runtime["events"].clear()
    attempts = []
    monkeypatch.setattr(
        torch.ops.rpu,
        "execution_reconfigure_begin",
        lambda _attempt: (_ for _ in ()).throw(RuntimeError("begin failed")),
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "execution_reconfigure_abort_attempt",
        attempts.append,
    )

    with pytest.raises(RuntimeError, match="begin failed"):
        reconfigure_rpu_execution(model, {
            "components": {
                "vision_encoder": {"vision": {"chunk_size": 256}},
            },
        })

    assert len(attempts) == 1
    assert not any(
        event[0] in {"stage_text", "stage_vision"}
        for event in install_runtime["events"]
    )
    assert getattr(
        model.model.visual, "_rpu_vision_execution_chunk_size", "auto"
    ) == "auto"
    stats = rpu_execution_stats(model)
    assert stats["state"] == "QUIESCENT"
    assert stats["generation"] == 0


def test_execution_session_poisoned_when_rollback_cannot_restore(
    install_runtime,
):
    adapter = qwen3_vl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    adapter._reset_component_graphs = lambda _changed: (_ for _ in ()).throw(
        RuntimeError("persistent graph retirement failure")
    )

    with pytest.raises(RuntimeError, match="persistent graph retirement failure"):
        reconfigure_rpu_execution(model, {
            "components": {
                "vision_encoder": {"vision": {"chunk_size": 256}},
            },
        })
    assert rpu_execution_stats(model)["state"] == "POISONED"
    with pytest.raises(RuntimeError, match="poisoned"):
        reconfigure_rpu_execution(model, {})


def test_close_uses_execution_session_shutdown(install_runtime, monkeypatch):
    released = []
    monkeypatch.setattr(
        qwen3_vl, "_release_live_instance", lambda model: released.append(model)
    )
    adapter = qwen3_vl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()

    adapter.close()

    assert released == [model]
    assert adapter._rpu_is_ready is False
    assert rpu_execution_stats(model)["state"] == "CLOSED"
    assert not hasattr(model.model.language_model, "_rpu_decoder_handle")
    assert not hasattr(model.model.visual, "_rpu_vision_handle")
    adapter.close()


def test_staged_32b_streams_after_claim_without_second_swizzle(
    monkeypatch, install_runtime
):
    from rpu_backend.quant import load as w8a16_load

    model = _Model()
    # A staged install must remain owned entirely by its materializer; this
    # sentinel makes any accidental fall-through into ordinary layer streaming
    # fail instead of passing because the default fake decoder has no layers.
    model.model.language_model.layers = [object()]
    plan = object()
    vars(model)[w8a16_load._W8A16_IMAGETEXT_STAGE_ATTR] = plan
    monkeypatch.setenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED", "1")
    monkeypatch.setattr(
        qwen3_vl, "_is_qwen3_vl_32b_w8a16_config", lambda _config: True
    )
    monkeypatch.setattr(
        w8a16_load, "_validate_staged_w8a16_imagetext", lambda seen: plan
    )
    monkeypatch.setattr(
        qwen3_vl,
        "_validate_qwen3_vl_32b_w8a16_model",
        lambda _model: pytest.fail("staged model reached full CPU inventory validation"),
    )

    def materialize(seen):
        assert seen is model
        assert seen._rpu_swizzle_started is True
        assert install_runtime["allocator_enabled"] is True
        assert install_runtime["events"][-2:] == [
            ("claim", "model"),
            ("set_caching_allocator", True),
        ]
        install_runtime["events"].append(("stream", "decoder"))

    monkeypatch.setattr(
        w8a16_load, "_materialize_staged_w8a16_imagetext_for_rpu", materialize
    )
    monkeypatch.setattr(qwen3_vl, "_qwen3_vl_text_scale_lists", lambda _text: ())

    adapter = qwen3_vl.Qwen3VLAdapter(model)
    assert adapter._rpu_w8a16_staged_plan is plan
    assert adapter.to_rpu() is model
    assert adapter._rpu_w8a16_staged_plan is None
    assert ("stream", "decoder") in install_runtime["events"]
    assert ("swizzle", "text") not in install_runtime["events"]
    assert [event for event in install_runtime["events"]
            if event[0] in {"set_linear_acc32", "set_vision_linear_acc32"}] == [
        ("set_vision_linear_acc32", 11, False),
        ("set_linear_acc32", 22, False),
    ]


def test_qwen3vl_text_graph_signature_builds_once_then_stably_replays(
    monkeypatch,
):
    class Signature(tuple):
        def __new__(cls, *, op_id, shapes, dyn_dims, dtypes):
            return tuple.__new__(cls, (op_id, tuple(shapes), tuple(dyn_dims), tuple(dtypes)))

    class Cache:
        def __init__(self):
            self.entries = set()
            self.builds = 0
            self.replays = 0

        def capture(self, signature):
            cache = self

            class Scope:
                def __enter__(self):
                    if signature in cache.entries:
                        cache.replays += 1
                    else:
                        cache.entries.add(signature)
                        cache.builds += 1

                def __exit__(self, *args):
                    return False

            return Scope()

    monkeypatch.setattr(rpu_backend.graph, "GraphSignature", Signature)
    monkeypatch.setattr(qwen3_vl, "chunk_policy_key", lambda _handle: 11)
    text_model = SimpleNamespace(
        _rpu_text_hidden_size=5120,
        _rpu_text_num_layers=64,
        _rpu_text_deepstack_hash=123,
        _rpu_decoder_handle=7,
    )
    cache = Cache()

    for position in (17, 18, 19):
        signature = qwen3_vl._qwen3_vl_text_graph_signature(
            text_model, 1, 1, position, 0
        )
        with cache.capture(signature):
            pass

    assert (cache.builds, cache.replays, len(cache.entries)) == (1, 2, 1)
    continuation_a = qwen3_vl._qwen3_vl_text_graph_signature(
        text_model, 8, 16, 17, 8
    )
    continuation_b = qwen3_vl._qwen3_vl_text_graph_signature(
        text_model, 8, 16, 25, 8
    )
    assert continuation_a != signature
    assert continuation_a != continuation_b

    source = inspect.getsource(qwen3_vl._qwen3_vl_text_graph_signature)
    assert "int(logical_len)" in source
    assert "int(execution_len)" in source
    assert "text_model._rpu_text_num_layers" in source
    assert "text_model._rpu_text_deepstack_hash" in source
    assert "chunk_policy_key(text_model._rpu_decoder_handle)" in source
    assert "if logical_len != 1" in source
    assert "dyn_dims.append(int(position))" in source


def test_forward_rejects_unsupported_semantics_before_dispatch():
    ids = torch.tensor([[1]], dtype=torch.long)
    calls = [
        ({"input_ids": ids, "use_cache": False}, NotImplementedError),
        ({"input_ids": ids, "return_dict": False}, NotImplementedError),
        ({"input_ids": ids.float()}, TypeError),
        ({"input_ids": ids, "attention_mask": torch.zeros_like(ids)},
         NotImplementedError),
        ({"input_ids": ids, "attention_mask": torch.ones(1, 1, 1, 1)},
         NotImplementedError),
        ({"input_ids": ids, "attention_mask": torch.ones(1, 2)}, ValueError),
        ({"input_ids": ids, "past_key_values": object()}, TypeError),
    ]
    for kwargs, error_type in calls:
        with pytest.raises(error_type):
            qwen3_vl._rpu_qwen3vl_forward(SimpleNamespace(), **kwargs)


def test_text_wrapper_restores_old_state_on_common_install_failure(monkeypatch):
    class InstallInterrupted(BaseException):
        pass

    state = {"events": []}
    old_cache = _Cache(state, "old-text")
    model = SimpleNamespace(
        config=SimpleNamespace(
            rope_scaling={"mrope_section": [24, 20, 20]},
            num_hidden_layers=28,
            hidden_size=2048,
        ),
        _rpu_text_graph_cache=old_cache,
        _rpu_text_num_layers=7,
        _rpu_text_hidden_size=8,
        _rpu_text_deepstack_hash=9,
    )

    class GraphCache:
        def clear(self):
            state["events"].append(("clear", "new-text"))

    monkeypatch.setattr(
        shared_decoder,
        "build_mrope_cos_sin_tables",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", GraphCache)

    def fail_install(*args, **kwargs):
        assert kwargs["deepstack_lang_layers"] == [0, 1, 2]
        raise InstallInterrupted("synthetic common install interruption")

    monkeypatch.setattr(
        shared_decoder, "_install_causal_decoder_forward", fail_install
    )

    with pytest.raises(InstallInterrupted, match="common install interruption"):
        qwen3_vl_text.install_qwen3_vl_text_for_rpu(
            model,
            vision_config=SimpleNamespace(deepstack_visual_indexes=[5, 11, 17]),
        )

    # The shared installer owns graph cleanup; this double fails before any
    # native install. The M-RoPE wrapper must not retry that cleanup.
    assert state["events"] == []
    assert model._rpu_text_graph_cache is old_cache
    assert model._rpu_text_num_layers == 7
    assert model._rpu_text_hidden_size == 8
    assert model._rpu_text_deepstack_hash == 9


def test_text_wrapper_rollback_bypasses_rejecting_attribute_hooks(monkeypatch):
    class InstallInterrupted(BaseException):
        pass

    interruption = InstallInterrupted("synthetic text publication interruption")

    class RejectingModel(SimpleNamespace):
        def __setattr__(self, name, value):
            if self.__dict__.get("_reject_runtime_attrs", False) and name.startswith(
                "_rpu_text_"
            ):
                raise interruption
            super().__setattr__(name, value)

        def __delattr__(self, name):
            if self.__dict__.get("_reject_runtime_attrs", False) and name.startswith(
                "_rpu_text_"
            ):
                raise AssertionError("rollback re-entered __delattr__")
            super().__delattr__(name)

    old_cache = object()
    model = RejectingModel(
        config=SimpleNamespace(
            rope_scaling={"mrope_section": [24, 20, 20]},
            num_hidden_layers=28,
            hidden_size=2048,
        ),
        _rpu_text_graph_cache=old_cache,
        _rpu_text_num_layers=7,
        _rpu_text_hidden_size=8,
        _rpu_text_deepstack_hash=9,
    )
    object.__setattr__(model, "_reject_runtime_attrs", True)

    class GraphCache:
        def clear(self):
            pass

    monkeypatch.setattr(
        shared_decoder,
        "build_mrope_cos_sin_tables",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", GraphCache)
    monkeypatch.setattr(
        shared_decoder,
        "_install_causal_decoder_forward",
        lambda *args, **kwargs: pytest.fail("common installer must not run"),
    )

    with pytest.raises(InstallInterrupted) as exc_info:
        qwen3_vl_text.install_qwen3_vl_text_for_rpu(model)

    assert exc_info.value is interruption
    assert model._rpu_text_graph_cache is old_cache
    assert model._rpu_text_num_layers == 7
    assert model._rpu_text_hidden_size == 8
    assert model._rpu_text_deepstack_hash == 9

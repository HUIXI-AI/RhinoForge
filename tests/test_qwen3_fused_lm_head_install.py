"""Actual Qwen3 adapter/session/fused-head installation with CPU native doubles."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import qwen3
from rpu_backend.api import _execution, causal_lm
from rpu_backend.runtime import _native_retirement, decoder, hw_attrs
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import _decode_descriptor


def _linear(int8):
    layer = torch.nn.Linear(2, 2, bias=False).half()
    if int8:
        layer.weight = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.int8), requires_grad=False)
        layer.register_buffer("weight_scale", torch.ones(2, dtype=torch.float16))
    return layer


class _Inner(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return "original-inner"


class _Model(torch.nn.Module):
    def __init__(self, size, *, int8_head=True, int8_decoder=True):
        super().__init__()
        hidden, intermediate, layers = (4096, 12288, 36) if size == 8 else (5120, 17408, 40)
        self.config = SimpleNamespace(
            hidden_size=hidden, intermediate_size=intermediate,
            num_hidden_layers=layers, num_key_value_heads=8, vocab_size=2,
            quant_config=dict(method="w8a16", quantized_lm_head=int8_head,
                              lm_head_untied=True, quantized_embed_tokens=False),
        )
        self.model = _Inner()
        self.model.embed_tokens = torch.nn.Embedding(2, 2).half()
        block = torch.nn.Module()
        block.self_attn = torch.nn.Module()
        block.mlp = torch.nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(block.self_attn, name, _linear(int8_decoder))
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(block.mlp, name, _linear(int8_decoder))
        self.model.layers = torch.nn.ModuleList([block])
        self.lm_head = _linear(int8_head)

    def to(self, *args, **kwargs):
        return self  # Device migration is the only model method doubled.

    def forward(self, *args, **kwargs):
        return "original-outer"


class _Cache:
    def __init__(self, events):
        self.events = events

    def clear(self):
        self.events.append("clear")

    def cache_invariant_ok(self):
        return True

    def prepare_plan(self, key, prepare):
        return prepare()

    def is_frozen(self):
        return False


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    state = dict(events=[], resources=[], set_error=None, post_error=None, destroy_error=None,
                 lm_head_set=False, decode_plans=[], plan_error=None)
    events = state["events"]
    original_to = torch.Tensor.to

    def to_cpu_device(tensor, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)) and str(args[0]).split(":")[0] == "rpu":
            args = ("cpu", *args[1:])
        if str(kwargs.get("device")).split(":")[0] == "rpu":
            kwargs = dict(kwargs, device="cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_cpu_device)
    monkeypatch.setattr(qwen3, "transform_linear_weight", lambda weight, partition: weight.clone())
    def swizzle(module, **kwargs):
        events.append("swizzle")
        for name, child in module.named_modules():
            if isinstance(child, torch.nn.Linear):
                child._rpu_linear_partition = 0 if name.endswith(("o_proj", "down_proj")) else 1
                child._rpu_linear_num_cores = 8
    monkeypatch.setattr(qwen3, "convert_linear_weights_inplace", swizzle)

    def validate_postinstall(model):
        events.append("postinstall")
        if state["post_error"] is not None:
            raise state["post_error"]

    monkeypatch.setattr(hw_attrs, "validate_postinstall", validate_postinstall)
    monkeypatch.setattr(hw_attrs, "install_hw_attr_validator", lambda model: None)

    def destroy(handle):
        events.append("destroy")
        if state["destroy_error"] is not None:
            raise state["destroy_error"]

    def install(inner, **kwargs):
        events.append("install")
        state["install_kwargs"] = kwargs
        state["scale_lists"] = kwargs["scale_lists"]
        inner._rpu_decoder_handle = 17
        inner._rpu_decoder_graph_cache = _Cache(events)
        resource = _native_retirement._InstalledNativeResource(
            inner, 17, destroy, graphs=(inner._rpu_decoder_graph_cache,),
            keepalive=(), label="decoder", handle_name="_rpu_decoder_handle",
        )
        state["resources"].append(resource)
        inner._rpu_decoder_retirement_state = resource
        inner._rpu_decoder_handle_finalizer = resource.finalizer
        inner._rpu_deepstack_lang_layers = ()
        inner._rpu_batch_decode_enabled = False
        inner._rpu_decoder_num_layers = 1
        inner._rpu_decoder_hidden_size = 2
        inner._rpu_decoder_deepstack_hash = 0
        inner._rpu_decode_stage_descriptor = resolve_decode(17)

        def rpu_decoder_model_forward(self, *args, **kwargs):
            return "installed-inner"

        inner.forward = MethodType(rpu_decoder_model_forward, inner)
        return 17

    def set_head(handle, weight, scale=None):
        assert handle == 17 and weight.dtype == torch.int8 and scale.dtype == torch.float16
        assert state["resources"][0].keepalive[-2:] == (weight, scale)
        events.append("head")
        if state["set_error"] is not None:
            raise state["set_error"]
        state["lm_head_set"] = True

    def resolve_decode(handle):
        state["decode_plans"].append(state["lm_head_set"])
        if state["lm_head_set"] and state["plan_error"] is not None:
            raise state["plan_error"]
        return _decode_descriptor(state["lm_head_set"])

    monkeypatch.setattr(qwen3, "_install_causal_decoder_forward", install)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head", set_head, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor", resolve_decode, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_planner_cache_identity",
                        lambda handle: (handle, int(state["lm_head_set"])), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_enable_execution_reconfigure",
                        lambda handle: events.append("reconfigure"), raising=False)
    yield state
    for resource in state["resources"]:
        resource.finalizer.detach()


@pytest.mark.parametrize("size,int8_decoder", [(8, True), (14, True), (8, False)])
def test_int8_head_installs_by_default_and_uses_the_shared_session(runtime, monkeypatch, size, int8_decoder):
    model = _Model(size, int8_decoder=int8_decoder)
    adapter = qwen3.Qwen3Adapter(model)
    assert adapter.to_rpu() is model
    assert runtime["events"] == ["swizzle", "swizzle", "install", "head", "reconfigure", "postinstall"]
    assert (runtime["scale_lists"] is not None) is int8_decoder
    assert runtime["decode_plans"] == [False, True]
    assert model.model._rpu_decode_stage_descriptor == _decode_descriptor(True)
    assert model.forward.__func__.__name__ == "fused_forward"
    assert hasattr(model.forward.__func__, "__wrapped__")
    assert adapter._execution_session is model._execution_session is model.model._execution_session
    assert model._rpu_execution_generation == model.model._rpu_execution_generation == 0
    assert decoder._causal_lm_runtime_complete(model)
    before = list(runtime["events"])
    assert adapter.to_rpu() is model and runtime["events"] == before

    def run(inner, handle, **kwargs):
        assert inner._execution_session._active
        return torch.tensor([[[0., 1.]]]), kwargs["past_key_values"]

    monkeypatch.setattr(qwen3, "_run_causal_decoder_forward", run)
    result = model(input_ids=torch.ones(1, 1, dtype=torch.long))
    assert result.logits.tolist() == [[[0., 1.]]]
    assert not model._execution_session._active
    decoder._close_causal_lm_model(model)
    assert runtime["events"][-2:] == ["clear", "destroy"]
    with pytest.raises(RuntimeError, match="closed"):
        model(input_ids=torch.ones(1, 1, dtype=torch.long))


def test_fp16_head_wrapper_delegates_ordinary_requests_to_existing_forward(runtime):
    model = _Model(8, int8_head=False, int8_decoder=False)
    qwen3.Qwen3Adapter(model).to_rpu()
    assert model.forward.__func__.__name__ == "segmented_forward"
    assert model(input_ids=torch.ones(1, 1, dtype=torch.long)) == "original-outer"
    assert "head" not in runtime["events"]
    decoder._close_causal_lm_model(model)


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("hidden,intermediate,layers,ceiling", [
    (1024, 3072, 28, 512),
    (2048, 6144, 28, 512),
    (2560, 9728, 36, 256),
    (4096, 12288, 36, 128),
])
def test_installed_fp16_and_w8_decoder_share_long_prefill_envelope(
    runtime, int8, hidden, intermediate, layers, ceiling,
):
    # Reuse the small installation model: metadata selects the real adapter
    # profile while native/device operations remain CPU doubles.
    model = _Model(8, int8_head=int8, int8_decoder=int8)
    model.config.hidden_size = hidden
    model.config.intermediate_size = intermediate
    model.config.num_hidden_layers = layers
    adapter = qwen3.Qwen3Adapter(model)
    adapter.to_rpu()

    installed = runtime["install_kwargs"]
    assert installed["arch"] == "qwen3"
    assert installed["chunk_envelope_for"]("qwen3", layers, hidden) == (4096, ceiling)
    assert (installed["scale_lists"] is not None) is int8
    qwen3.Qwen3Adapter.preflight_execution(
        model.config, {"prefill": {"chunk_size": ceiling}},
    )
    with pytest.raises(qwen3.UnsupportedModelError, match="exceeds this profile"):
        qwen3.Qwen3Adapter.preflight_execution(
            model.config, {"prefill": {"chunk_size": ceiling + 16}},
        )
    decoder._close_causal_lm_model(model)


def test_14b_and_unknown_geometry_do_not_inherit_long_prefill(runtime):
    model = _Model(14)
    qwen3.Qwen3Adapter(model).to_rpu()
    lookup = runtime["install_kwargs"]["chunk_envelope_for"]
    assert lookup("qwen3", 40, 5120) == (192, 32)
    with pytest.raises(RuntimeError, match="no certified chunk envelope"):
        lookup("qwen3", 28, 1536)
    decoder._close_causal_lm_model(model)


@pytest.mark.parametrize("int8", [False, True])
def test_cold_session_passes_real_preinstall_without_runtime_generation(runtime, int8):
    model = _Model(8, int8_head=int8, int8_decoder=int8)
    adapter = qwen3.Qwen3Adapter(model)
    assert adapter._execution_session.generation == 0
    assert not hasattr(model, "_rpu_execution_generation")
    assert not hasattr(model.model, "_rpu_execution_generation")
    hw_attrs.validate_preinstall(model)
    assert runtime["events"] == []


@pytest.mark.parametrize("target", ["outer", "inner"])
def test_injected_cold_generation_still_rejects_before_weight_mutation(runtime, target):
    model = _Model(8)
    owner = model if target == "outer" else model.model
    owner._rpu_execution_generation = 9
    adapter = qwen3.Qwen3Adapter(model)
    from rpu_backend.runtime import RPUConfigError
    with pytest.raises(RPUConfigError, match="BEFORE"):
        adapter.to_rpu()
    assert owner._rpu_execution_generation == 9
    assert not hasattr(model, "_rpu_swizzle_started")
    assert runtime["events"] == []


@pytest.mark.parametrize("failure", ["set_error", "plan_error", "post_error"])
@pytest.mark.parametrize("instance_forward", [False, True])
def test_failed_head_install_restores_both_entry_points_and_retires_once(runtime, failure, instance_forward):
    model = _Model(8)
    if instance_forward:
        model.forward = lambda *a, **k: "custom-outer"
        model.model.forward = lambda *a, **k: "custom-inner"
    original_outer, original_inner = model.forward, model.model.forward
    error = KeyboardInterrupt("head install failed")
    runtime[failure] = error
    adapter = qwen3.Qwen3Adapter(model)
    with pytest.raises(KeyboardInterrupt) as caught:
        adapter.to_rpu()
    assert caught.value is error
    assert model.forward == original_outer and model.model.forward == original_inner
    assert ("forward" in vars(model)) is instance_forward
    assert ("forward" in vars(model.model)) is instance_forward
    assert runtime["events"][-2:] == ["clear", "destroy"]
    assert not hasattr(model.model, "_rpu_decoder_handle")
    assert not hasattr(model, "_rpu_lm_head_w_keepalive")
    assert not adapter._rpu_is_ready and adapter._all_layers_once_handle is None
    assert model._execution_session.stats()["state"] == "CLOSED"
    before = list(runtime["events"])
    with pytest.raises(qwen3.RPUBackendError, match="prior swizzle attempt"):
        adapter.to_rpu()
    assert runtime["events"] == before


def test_failed_native_retirement_preserves_poisoned_fused_owner(runtime):
    model = _Model(8)
    runtime["post_error"] = error = RuntimeError("post install failed")
    runtime["destroy_error"] = RuntimeError("native retirement failed")
    adapter = qwen3.Qwen3Adapter(model)
    with pytest.raises(RuntimeError) as caught:
        adapter.to_rpu()
    assert caught.value is error
    assert "native retirement failed" in error.__notes__[0]
    assert model.model._rpu_decoder_handle == 17
    assert model.forward.__func__.__name__ == "fused_forward"
    assert model._rpu_lm_head_w_keepalive is not None
    assert model._execution_session.stats()["state"] == "POISONED"
    before = list(runtime["events"])
    with pytest.raises(RuntimeError):
        model(input_ids=torch.ones(1, 1, dtype=torch.long))
    assert runtime["events"] == before


def test_bad_int8_head_scale_rejects_before_session_or_mutation(runtime):
    model = _Model(8, int8_decoder=False)
    model.lm_head.weight_scale = torch.ones(1, dtype=torch.float16)
    with pytest.raises(qwen3.RPUBackendError, match="vocab_size"):
        qwen3.Qwen3Adapter(model)
    assert runtime["events"] == []
    assert not hasattr(model, "_rpu_swizzle_started")
    assert not hasattr(model, "_execution_session")

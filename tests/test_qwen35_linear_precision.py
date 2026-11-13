"""Cold arithmetic selection through the public schema and actual installers."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import qwen3_5 as adapter
from rpu_backend.adapters.qwen3_5 import text, vision
from rpu_backend.api._execution import normalize_rpu_execution
from rpu_backend.api.errors import UnsupportedModelError
from test_qwen35_core_profiles import _config


@pytest.mark.parametrize("device", [None, "rpu"])
@pytest.mark.parametrize("cores,quantized,text_acc32,vision_acc32", [
    (cores, False, text_acc32, vision_acc32)
    for cores in (4, 6, 8)
    for text_acc32, vision_acc32 in ((False, False), (False, True), (True, False), (True, True))
] + [(cores, True, text_acc32, vision_acc32)
     for cores in (4, 6, 8)
     for text_acc32, vision_acc32 in ((False, False), (False, True), (True, False), (True, True))])
def test_public_from_pretrained_passes_arithmetic_to_real_adapter(
        monkeypatch, device, cores, quantized, text_acc32, vision_acc32):
    from rpu_backend.api import conditional_generation as public

    config = _config()
    if quantized:
        from test_qwen35_w8_core_profiles import _w8_config
        config = _w8_config()
        monkeypatch.setenv("QWEN3_5_QUANT_ALLOW_UNCERTIFIED", "1")
    model = torch.nn.Module()
    model.config = config
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.visual = torch.nn.Module()
    if quantized:
        # The real model-specific loader publishes this after preserving the
        # checkpoint's integer projections. No weights are needed by preflight.
        model._qwen3_5_quant_config = config.quant_config
    seen = []
    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *_, **__: config)

    def load(path, *, config, dtype, **kwargs):
        assert path == "/mock-q35-weights" and config is model.config
        assert dtype is torch.float16
        seen.append("load")
        return model
    monkeypatch.setattr(adapter.Qwen3_5Adapter, "load_pretrained", staticmethod(load))
    original_init = adapter.Qwen3_5Adapter.__init__
    def install_wrapper(instance, loaded):
        original_init(instance, loaded)
        assert instance._rpu_execution["prefill"]["linear_acc32"] is text_acc32
        assert instance._rpu_execution["vision"]["linear_acc32"] is vision_acc32
        seen.append("adapter")
    monkeypatch.setattr(adapter.Qwen3_5Adapter, "__init__", install_wrapper)
    def to_rpu(instance):
        assert instance.model is model
        assert instance._rpu_execution == model._execution_session.config
        seen.append("to_rpu")
        return model
    monkeypatch.setattr(adapter.Qwen3_5Adapter, "to_rpu", to_rpu)
    requested = {"model": {"num_cores": cores},
                 "prefill": {"linear_acc32": text_acc32},
                 "vision": {"linear_acc32": vision_acc32}}
    result = public.RPUModelForConditionalGeneration.from_pretrained(
        "/mock-q35-weights", device=device, rpu_execution=requested)
    assert result is model and not model.training
    assert seen == ["load", "adapter", *(["to_rpu"] if device else [])]
    assert model._execution_session.config == requested
    assert getattr(model.to, "__rpu_wrapped__", False)


@pytest.mark.parametrize("stage", ["prefill", "vision"])
@pytest.mark.parametrize("enabled", [False, True])
def test_public_from_pretrained_admits_qwen3vl_arithmetic_before_weights(
        monkeypatch, stage, enabled):
    from rpu_backend.api import conditional_generation as public
    from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import _config as q3_config
    config = q3_config()
    config.architectures = ["Qwen3VLForConditionalGeneration"]
    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *_, **__: config)
    loads = []

    class ReachedWeights(RuntimeError):
        pass

    def stop_at_weights(path, **kwargs):
        loads.append(path)
        raise ReachedWeights

    monkeypatch.setattr(public.AutoModelForImageTextToText, "from_pretrained", stop_at_weights)
    with pytest.raises(ReachedWeights):
        public.RPUModelForConditionalGeneration.from_pretrained(
            "/mock-q3vl", rpu_execution={stage: {"linear_acc32": enabled}})
    assert loads == ["/mock-q3vl"]

    # A valid arithmetic field must not weaken the exact geometry preflight.
    config.vision_config.depth -= 1
    with pytest.raises(UnsupportedModelError, match="unsupported vision profile"):
        public.RPUModelForConditionalGeneration.from_pretrained(
            "/mock-q3vl", rpu_execution={stage: {"linear_acc32": enabled}})
    assert loads == ["/mock-q3vl"]


@pytest.mark.parametrize("size,bits", [(8, 8), (8, 4), (32, 8), (32, 4)])
@pytest.mark.parametrize("stage", ["prefill", "vision"])
def test_public_from_pretrained_admits_large_quantized_qwen3vl_acc32_before_weights(
        monkeypatch, size, bits, stage):
    from rpu_backend.api import conditional_generation as public
    from rpu_backend.quant import load_qwen3_vl_runtime
    from test_qwen3_vl_runtime_quant_loading import config as runtime_config

    config = runtime_config(size, bits)
    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *_, **__: config)
    class ReachedQuantizedWeights(RuntimeError):
        pass

    def stop_at_quantized_weights(*args, **kwargs):
        raise ReachedQuantizedWeights

    monkeypatch.setattr(load_qwen3_vl_runtime, "load_qwen3_vl_runtime_imagetext",
                        stop_at_quantized_weights)
    monkeypatch.setattr(public.AutoModelForImageTextToText, "from_pretrained",
                        lambda *_, **__: pytest.fail("quantized profile fell through to HF weights"))
    with pytest.raises(ReachedQuantizedWeights):
        public.RPUModelForConditionalGeneration.from_pretrained(
            "/mock-q3vl-quantized", rpu_execution={stage: {"linear_acc32": True}})



def _normalize(value):
    return normalize_rpu_execution(value, entry_point="Q35 arithmetic test",
                                   supported=adapter.Qwen3_5Adapter._EXECUTION_SUPPORTED)


@pytest.mark.parametrize("text_acc32,vision_acc32", [(False, False), (False, True), (True, False), (True, True)])
def test_public_arithmetic_choices_are_independent_and_frozen(text_acc32, vision_acc32):
    config = _normalize({"prefill": {"linear_acc32": text_acc32},
                         "vision": {"linear_acc32": vision_acc32}})
    adapter.Qwen3_5Adapter.preflight_execution(_config(), config)
    assert text._validate_qwen35_linear_accumulation(_config(), config) is text_acc32
    assert vision._qwen35_vision_precision(config) == (vision_acc32, vision_acc32)
    with pytest.raises(TypeError):
        config["vision"]["linear_acc32"] = not vision_acc32


@pytest.mark.parametrize("stage", ["prefill", "vision"])
@pytest.mark.parametrize("value", [0, 1, "true", "auto", None])
def test_arithmetic_requires_an_explicit_boolean(stage, value):
    with pytest.raises(TypeError, match="must be bool"):
        _normalize({stage: {"linear_acc32": value}})


@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("enabled", [False, True])
def test_reduced_vision_accumulation_preserves_requested_path(cores, enabled):
    from rpu_backend.adapters.qwen3_5.cores import vision_core_count
    config = _normalize({"model": {"num_cores": cores}, "vision": {"linear_acc32": enabled}})
    adapter.Qwen3_5Adapter.preflight_execution(_config(), config)
    assert vision_core_count(config) == 4
    assert vision._qwen35_vision_precision(config) == (enabled, enabled)


@pytest.mark.parametrize("method", ["w8a16", "w4a16_pgrp"])
@pytest.mark.parametrize("enabled", [False, True])
def test_quantized_text_accumulation_is_an_independent_cold_choice(method, enabled):
    config = _config()
    config.quant_config = {"default": {"method": method}}
    if method == "w4a16_pgrp":
        config.quant_config["default"]["group_size"] = 32
    execution = _normalize({"prefill": {"linear_acc32": enabled}})
    adapter.Qwen3_5Adapter.preflight_execution(config, execution)
    assert text._validate_qwen35_linear_accumulation(config, execution) is enabled


@pytest.mark.parametrize("stage", ["prefill", "vision"])
def test_hot_precision_change_rejects_before_graph_or_native_mutation(monkeypatch, stage):
    config = _config()
    owner = SimpleNamespace(config=config)
    instance = SimpleNamespace(model=owner, _rpu_execution=_normalize({}),
        preflight_execution=adapter.Qwen3_5Adapter.preflight_execution)
    monkeypatch.setattr(adapter, "_qwen3_5_text_runtime_state", lambda _: SimpleNamespace())
    monkeypatch.setattr(torch.ops.rpu, "execution_reconfigure_begin",
                        lambda *_: pytest.fail("native mutation"), raising=False)
    with pytest.raises(ValueError, match="immutable after installation"):
        adapter._validate_qwen3_5_execution_reconfigure(instance,
            _normalize({stage: {"linear_acc32": True}}))


def test_unset_and_internal_g05_precision_keep_their_existing_paths():
    assert vision._qwen35_vision_precision({}) == (False, False)
    assert vision._qwen35_vision_precision({}, certified_g05=True) == (True, False)
    assert vision._qwen35_vision_precision({"vision": {"linear_acc32": True}}, certified_g05=True) == (True, False)
    with pytest.raises(ValueError, match="requires ACC32"):
        vision._qwen35_vision_precision({"vision": {"linear_acc32": False}}, certified_g05=True)


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("choice", [None, False, True])
def test_real_vision_installer_binds_arithmetic_before_weight_conversion(monkeypatch, cores, choice):
    from test_qwen3_5_vision_helpers import _runtime_capable_vision
    from rpu_backend.api import causal_lm
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    events = []
    for name, callback in (
        ("create", lambda: events.append("create") or 17),
        ("destroy", lambda _: events.append("destroy")),
        ("set_linear_acc32", lambda handle, enabled: events.append(("linear", handle, enabled))),
        ("set_gelu_erf_ultra", lambda handle, enabled: events.append(("gelu", handle, enabled))),
        ("set_execution_cores", lambda handle, count: events.append(("cores", handle, count))),
        ("get_execution_topology", lambda *_: (4, 4, 4, 8)),
        ("set_chunk_size_cap", lambda *_: None),
        ("set_cache_capacity", lambda *_: None),
        ("set_prefill_chunk_size", lambda *_: None),
        ("resolve_prefill_domain", lambda *_: None),
    ):
        monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_" + name, callback, raising=False)
    def convert(*_, **__):
        events.append("convert")
        raise RuntimeError("stop before weights")
    monkeypatch.setattr(vision, "_convert_vision_block_weights_for_rpu", convert)
    execution = {"model": {"num_cores": cores}}
    if choice is not None:
        execution["vision"] = {"linear_acc32": choice}
    model = _runtime_capable_vision()
    model.config = _config().vision_config
    with pytest.raises(RuntimeError, match="stop before weights"):
        vision.install_qwen3_5_vision_for_rpu(model, execution_config=execution)
    expected = ["create", *([("cores", 17, 4)] if cores != 8 else []),
                ("linear", 17, choice is True)]
    if choice is True:
        expected.append(("gelu", 17, True))
    assert events == [*expected, "convert", "destroy"]

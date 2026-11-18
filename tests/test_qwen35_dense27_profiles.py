"""Official 27B cold metadata and raw-storage guards; no device or checkpoint."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import cores, quant_scope as qs, weights
from test_qwen35_core_profiles import _config as dense_config
from test_qwen35_legacy_finalized_upload import (
    _fixture, _install, _StopAtNativeWeights, host_device,
)


def config27(method=qs.W8A16, version=1):
    config = dense_config(9)
    tc = config.text_config
    for name, value in dict(hidden_size=5120, intermediate_size=17408,
            num_hidden_layers=64, num_attention_heads=24, linear_num_value_heads=48).items():
        setattr(tc, name, value)
    tc.layer_types = ([qs.LINEAR_ATTENTION] * 3 + [qs.FULL_ATTENTION]) * 16
    config.vision_config.out_hidden_size = 5120
    spec = {"method": method}
    if method == qs.W4A16_PGRP:
        spec["group_size"] = 32
    config.quant_config = ({"format_version": 1, **spec} if version == 1 else
                           {"format_version": 2, "default": spec,
                            "projection_overrides": {"linear_attn.in_proj_b": spec}})
    return config


def runtime_config27(recipe):
    config = config27()
    if recipe == qs.FP16:
        del config.quant_config
    elif recipe in (qs.W8A16, qs.W4A16_PGRP):
        return config27(recipe)
    elif recipe in qs.AWQ_PROFILES:
        config.quant_config = qs._quanttrio_policy(
            qs.is_full_from_layer_types(config.text_config.layer_types), recipe)
    else:
        from rpu_backend.quant.convert_qwen3_5 import _quant_config
        config.quant_config = _quant_config(
            qs.is_full_from_layer_types(config.text_config.layer_types), None, 32,
            profile=recipe)
    return config


@pytest.mark.parametrize("recipe", [qs.FP16, qs.W8A16, qs.W4A16_PGRP,
    "w4_mlp_only", "w4_mlp_w8_attn_gdn", "w4_main_w8_gdn", "w8_main_w8_gdn",
    *qs.AWQ_PROFILES])
@pytest.mark.parametrize("count", [4, 6, 8])
def test_public_preflight_keeps_official_27b_named_and_dense_recipes(recipe, count, monkeypatch):
    from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    config = runtime_config27(recipe)
    Qwen3_5Adapter.preflight(config)
    Qwen3_5Adapter.preflight_execution(config, {"model": {"num_cores": count}})
    assert cores.validate_official_27b_runtime_profile(config)


@pytest.mark.parametrize("recipe", [qs.FP16, "w4_mlp_only", "w4_mlp_w8_attn_gdn", *qs.AWQ_PROFILES])
@pytest.mark.parametrize("invalid", [False, True])
def test_public_loader_reaches_weight_loading_only_after_official_27b_checks(
    recipe, invalid, monkeypatch,
):
    from rpu_backend.adapters import qwen3_5 as adapter
    from rpu_backend.api import conditional_generation
    config = runtime_config27(recipe)
    if invalid:
        config.text_config.num_attention_heads = 16
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    monkeypatch.setattr(conditional_generation.AutoConfig, "from_pretrained", lambda *_a, **_kw: config)
    monkeypatch.setattr(conditional_generation, "get_adapter", lambda _: adapter.Qwen3_5Adapter)
    reached = []

    class ReachedWeights(Exception):
        pass

    def load(*_args, **kwargs):
        reached.append(kwargs["config"])
        raise ReachedWeights

    monkeypatch.setattr(adapter.Qwen3_5Adapter, "load_pretrained", load)
    with pytest.raises((ValueError, adapter.UnsupportedModelError) if invalid else ReachedWeights):
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
            "unused-test-checkpoint", rpu_execution={"model": {"num_cores": 6}})
    assert reached == ([] if invalid else [config])


@pytest.mark.parametrize("method", [qs.W8A16, qs.W4A16_PGRP])
@pytest.mark.parametrize("version", [1, 2])
def test_modern_profile_accepts_equivalent_raw_and_hf_metadata(method, version):
    config = config27(method, version)
    for candidate in (config, config.to_dict()):
        assert qs.validate_modern_27b_profile(candidate)
        assert not qs.validate_legacy_text_profile(candidate)
    for count in (4, 6, 8):
        assert cores.validate_core_profile(config, {"model": {"num_cores": count}}).num_cores == count


@pytest.mark.parametrize("field,value", [("head_dim", 128), ("num_attention_heads", 16),
    ("num_hidden_layers", True), ("hidden_act", "gelu"), ("linear_num_value_heads", 32),
    ("attention_bias", True), ("attn_output_gate", False)])
def test_modern_metadata_near_neighbours_reject_before_allocation(field, value):
    config = config27()
    # Inject malformed metadata past the HF constructor to exercise our guard.
    object.__setattr__(config.text_config, field, value)
    with pytest.raises(ValueError, match="exact dense semantics"):
        qs.validate_modern_27b_profile(config)


def test_fp16_source_recognition_and_core_admission():
    config = config27()
    del config.quant_config
    assert qs.validate_modern_27b_profile(config, allow_fp16=True)
    for count in (4, 6, 8):
        topology = cores.validate_core_profile(config, {"model": {"num_cores": count}})
        assert topology.num_cores == count


@pytest.mark.parametrize("fault", ["mixed", "group64", "tied", "vision", "nested"])
def test_modern_profile_rejects_semantic_and_precision_conflicts(fault):
    config = config27(qs.W4A16_PGRP, 2)
    if fault == "mixed":
        config.quant_config["projection_overrides"]["linear_attn.in_proj_b"] = {"method": qs.FP16}
    elif fault == "group64":
        config.quant_config = {"method": qs.W4A16_PGRP, "group_size": 64}
    elif fault == "tied":
        config.tie_word_embeddings = True
    elif fault == "vision":
        config.vision_config.out_hidden_size = 4096
    else:
        config.text_config.quant_config = deepcopy(config.quant_config)
    with pytest.raises(ValueError):
        qs.validate_modern_27b_profile(config)


def _raw_uniform(method):
    inner, config = _fixture()
    config.quant_config = {"method": method}
    if method == qs.W4A16_PGRP:
        config.quant_config["group_size"] = 32
    paths = qs.quant_relpaths(qs.is_full_from_layer_types(config.layer_types))
    for name in paths:
        module = inner.get_submodule(name)
        module.weight = torch.nn.Parameter(torch.zeros_like(module.weight, dtype=torch.int8), requires_grad=False)
        shape = ((module.out_features,) if method == qs.W8A16 else
                 (module.in_features // 32, module.out_features))
        scale = torch.ones(shape, dtype=torch.float16)
        if "weight_scale" in module._buffers:
            module.weight_scale = scale
        else:
            module.register_buffer("weight_scale", scale)
    return inner, config


@pytest.mark.parametrize("method", [qs.W8A16, qs.W4A16_PGRP])
@pytest.mark.parametrize("fault", [None, "shape", "zero", "nan", "floating_weight"])
def test_all_raw_projection_scales_are_checked_including_last_layer(method, fault):
    inner, config = _raw_uniform(method)
    last = inner.layers[-1].self_attn.o_proj
    if fault == "shape":
        last.weight_scale = torch.ones(1, dtype=torch.float16)
    elif fault == "zero":
        last.weight_scale.reshape(-1)[-1] = 0
    elif fault == "nan":
        last.weight_scale.reshape(-1)[-1] = float("nan")
    elif fault == "floating_weight":
        last.weight = torch.nn.Parameter(last.weight.half(), requires_grad=False)
    if fault:
        with pytest.raises(ValueError):
            weights.validate_modern_27b_cold_weights(inner, config, cold_cpu=True)
    else:
        weights.validate_modern_27b_cold_weights(inner, config, cold_cpu=True)


def test_existing_9b_vl_does_not_inherit_modern_w4_admission():
    config = dense_config(9)
    config.quant_config = {"method": qs.W4A16_PGRP, "group_size": 32}
    with pytest.raises(ValueError, match="9B VL"):
        cores.validate_dense_large_vl_profile(config)


def test_w4_raw_out_of_range_values_are_rejected_before_packing():
    inner, config = _raw_uniform(qs.W4A16_PGRP)
    inner.layers[-1].self_attn.o_proj.weight[-1, -1] = 8
    with pytest.raises(ValueError, match="signed INT4"):
        weights.validate_modern_27b_cold_weights(inner, config, cold_cpu=True)


def test_modern_uniform_upload_moves_only_final_projections(monkeypatch, host_device):
    """Exercise the real text install/transform loop with small tensor storage."""
    from rpu_backend.adapters.qwen3_5 import text

    inner, config = _raw_uniform(qs.W8A16)
    monkeypatch.setattr(cores, "validate_official_27b_runtime_profile", lambda *_a, **_kw: True)
    originals = [m.weight.detach() for m in inner.modules()
                 if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d))]
    pointers = {tensor.data_ptr() for tensor in originals}
    original_to = torch.Tensor.to

    def finalized_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, (str, torch.device)) and torch.device(device).type == "rpu":
            assert tensor.data_ptr() not in pointers
        return original_to(tensor, *args, **kwargs)

    payloads = []

    def native_boundary(*args):
        payloads.append(args)
        raise _StopAtNativeWeights

    monkeypatch.setattr(torch.Tensor, "to", finalized_to)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", native_boundary, raising=False)
    with pytest.raises(_StopAtNativeWeights):
        _install(inner, config)
    args, = payloads
    # GDN Q/K/V/Z/out/b/a now use ordinary W8; companion scalars stay FP16.
    assert all(args[index][0].dtype == torch.int8 for index in (25, 26, 35, 36, 37, 41, 42))
    assert args[27][0].dtype == args[28][0].dtype == torch.float16
    assert inner.layers[0].linear_attn.in_proj_qkv.weight.numel() == 0


@pytest.mark.parametrize("recipe", ["fp16", "mlp_only", "mixed", "awq"])
def test_six_core_install_pads_real_fp16_and_quantized_mlp_sources(
    monkeypatch, host_device, recipe,
):
    """Exercise the production packing caller with the real 27B MLP width.

    The small hidden width keeps this a host test.  Only cold model geometry
    is replaced; padding, packing, dtype checks and native argument assembly
    run unchanged.  In particular, FP16 must not enter the integer-only pad.
    """
    from rpu_backend.adapters.qwen3_5 import text
    inner, config = _fixture()
    config.intermediate_size = 17408
    for layer in inner.layers:
        for name, n, k in (("gate_proj", 17408, 128),
                           ("up_proj", 17408, 128),
                           ("down_proj", 128, 17408)):
            setattr(layer.mlp, name, torch.nn.Linear(k, n, bias=False, dtype=torch.float16))
    config.quant_config = None
    for name in qs.quant_relpaths(qs.is_full_from_layer_types(config.layer_types)):
        module = inner.get_submodule(name)
        module.weight = torch.nn.Parameter(module.weight.detach().half(), requires_grad=False)
        module._buffers.pop("weight_scale", None)
    if recipe != "fp16":
        config.quant_config = ({"format_version": 2,
            "default": {"method": qs.W8A16 if recipe == "mixed" else qs.FP16},
            "projection_overrides": {"mlp." + role: {"method": qs.W4A16_PGRP, "group_size": 32}
                for role in ("gate_proj", "up_proj", "down_proj")}} if recipe != "awq" else
            qs._quanttrio_policy(qs.is_full_from_layer_types(config.layer_types), qs.ACTIVATION_AWQ_PROFILE))
        specs = qs.resolve_quant_specs(qs.is_full_from_layer_types(config.layer_types), config.quant_config)
        for name, spec in specs.items():
            if spec.method == qs.FP16:
                continue
            module = inner.get_submodule(name)
            module.weight = torch.nn.Parameter(torch.zeros_like(module.weight,
                dtype=torch.uint8 if spec.method == qs.W4A16_AWQ else torch.int8), requires_grad=False)
            shape = ((module.out_features,) if spec.method == qs.W8A16 else
                     (module.in_features // spec.group_size, module.out_features))
            module.register_buffer("weight_scale", torch.ones(shape, dtype=torch.float16))
            if spec.method == qs.W4A16_AWQ:
                module.register_buffer("weight_zero", torch.zeros(shape, dtype=torch.uint8))
    if recipe == "awq":
        monkeypatch.setattr(text, "resolve_quant_config", lambda _: config.quant_config)
    weights.validate_modern_27b_cold_weights(inner, config, cold_cpu=True)
    monkeypatch.setattr(cores, "validate_core_profile", lambda *_: cores.core_topology({"model": {"num_cores": 6}}))
    monkeypatch.setattr(cores, "validate_official_27b_runtime_profile", lambda *_: True)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_execution_cores", lambda *_: None, raising=False)
    payloads = []

    def capture(*args, **kwargs):
        payloads.append(args)
        if recipe == "awq":
            assert kwargs["gate_negative_zero"][1].numel() > 0
        raise _StopAtNativeWeights

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", capture, raising=False)
    with pytest.raises(_StopAtNativeWeights):
        _install(inner, config)
    args, = payloads
    physical = 17664 if recipe == "awq" else 17472
    expected_dtype = torch.float16 if recipe in ("fp16", "awq") else torch.uint8
    # Gate/Up/Down lists are the actual finalized payloads handed to native.
    assert args[10][0].dtype == args[11][0].dtype == args[12][0].dtype == expected_dtype
    expected_elements = physical * 128 // (1 if recipe in ("fp16", "awq") else 2)
    assert args[10][0].numel() == args[12][0].numel() == expected_elements


@pytest.mark.parametrize("method", [qs.FP16, qs.W8A16, qs.W4A16_PGRP,
    "w4_mlp_only", "w4_mlp_w8_attn_gdn", "w4_main_w8_gdn", "w8_main_w8_gdn", *qs.AWQ_PROFILES])
@pytest.mark.parametrize("count", [4, 6, 8])
@pytest.mark.parametrize("fault", ["raw", "allocator", "late_pool", "no_device", None])
def test_official_cold_owner_reserves_before_mutation_and_never_bulk_uploads(monkeypatch, method, count, fault):
    from rpu_backend.adapters import qwen3_5 as adapter
    from rpu_backend.api import causal_lm
    from rpu_backend.api._execution import bind_execution_session
    from rpu_backend.graph import GraphRuntimePolicy

    config = runtime_config27(method)
    events = []
    if method in qs.AWQ_PROFILES:
        # Raw AWQ source ownership and packing are exercised separately with
        # real shards/tensors. This fixture covers the public resource owner.
        from rpu_backend.adapters.qwen3_5 import loader, text
        monkeypatch.setattr(loader, "validate_awq_loaded_model", lambda _: True)
        monkeypatch.setattr(text, "require_quanttrio_awq_native", lambda: None)
        monkeypatch.setattr(text, "_validate_projection_specs", lambda *_: None)

    class Inner:
        # Keep the real unstamped HF child config: quantization belongs to root.
        def __init__(self):
            self.config = config.text_config
            self.embed_tokens = object()  # CPU head alias owner; no device storage.

        def to(self, *_args, **_kwargs):
            pytest.fail("modern 27B must not upload the raw text stack")

    class Model:
        def __init__(self):
            self.config = config
            self.model = SimpleNamespace(language_model=Inner())
            self._rpu_execution = {"model": {"num_cores": count}}

        def half(self):
            events.append("half")
            return self

        def get_output_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(shape=(248320, 5120)))

    model = Model()
    early_head = object()
    original_cleanup = adapter._cleanup_failed_text_install
    owner = adapter.Qwen3_5Adapter.__new__(adapter.Qwen3_5Adapter)
    owner.model, owner._rpu_is_ready = model, False
    owner._execution_session = bind_execution_session(
        model, model._rpu_execution, entry_point="modern27 cold test", graph_mode="BOUNDED_ONESHOT",
        supported={"model": {"num_cores"}})
    owner._execution_session._bind_planner_owner(model.model.language_model)
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)

    def claim(value):
        assert value is model
        causal_lm._claim_live_instance(value)
        events.append("claim")

    def release(value):
        assert value is model
        causal_lm._release_live_instance(value)
        events.append("release")

    def cleanup(*args):
        if fault is not None:
            # The early resource handler releases before outer cleanup runs.
            assert causal_lm._LIVE_REF is None
        head_expected = method == qs.FP16 and fault is None
        assert getattr(model, "_lm_head_w_rpu_keepalive", None) is (early_head if head_expected else None)
        original_cleanup(*args)
        assert not hasattr(model, "_lm_head_w_rpu_keepalive")
        assert not owner._rpu_is_ready and owner._handle is None
        causal_lm._release_live_instance(model)
        events.append("cleanup")

    monkeypatch.setattr(adapter, "_claim_live_instance", claim)
    monkeypatch.setattr(adapter, "_release_live_instance", release)
    monkeypatch.setattr(adapter, "_cleanup_failed_text_install", cleanup)
    for name in ("qwen3_5_set_execution_cores", "qwen3_5_get_execution_topology",
                 "qwen3_5_vision_set_execution_cores", "qwen3_5_vision_get_execution_topology",
                 "linear_with_accumulation"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *_: None, raising=False)

    def raw_check(value):
        assert value is model
        events.append("raw-check")
        if fault == "raw":
            raise RuntimeError("bad last raw tensor")

    monkeypatch.setattr(cores, "validate_dense_large_vl_weights", raw_check)
    monkeypatch.setattr(cores, "validate_cold_weight_dtype", lambda *_a, **_kw: None)
    monkeypatch.setattr(cores, "validate_text_weight_geometry", lambda *_: None)
    monkeypatch.setattr(cores, "validate_text_mlp_padding_storage", lambda *_a, **_kw: None)

    def allocator(enabled):
        assert enabled is True
        assert causal_lm._LIVE_REF() is model
        events.append("caching-allocator")
        if fault == "allocator":
            raise RuntimeError("allocator already frozen to direct")

    def prepare(policy):
        assert policy.execution_core_count == count
        assert policy.qwen35_dense_27b_sdk_budget and not policy.qwen35_legacy_27b_sdk_budget
        assert policy.graph_arena_count == 4
        events.append("four-arenas")
        if fault == "late_pool":
            raise RuntimeError("pool already frozen")
        return fault != "no_device"

    def install(inner, passed_config, *_args, **kwargs):
        assert inner is model.model.language_model and passed_config is config
        assert kwargs.get("_cpu_stage_weights", False) is False
        assert kwargs["_public_prefill_graph_mode"] == "BOUNDED_ONESHOT"
        assert kwargs["_graph_runtime_policy"].qwen35_dense_27b_sdk_budget
        assert getattr(model, "_lm_head_w_rpu_keepalive", None) is (early_head if method == qs.FP16 else None)
        events.append("finalized-install")
        raise _StopAtNativeWeights

    monkeypatch.setattr(torch.rpu, "set_caching_allocator", allocator)
    monkeypatch.setattr(torch.rpu, "get_caching_allocator",
                        lambda: pytest.fail("cold install must claim allocator atomically"))
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", prepare)
    monkeypatch.setattr(adapter, "install_qwen3_5_text_for_rpu", install)
    def upload_head(output, embedding, num_cores, *, release_source):
        assert method == qs.FP16 and fault is None
        assert tuple(output.weight.shape) == (248320, 5120)
        assert embedding is model.model.language_model.embed_tokens
        assert num_cores == cores.core_topology(model._rpu_execution).lm_head_tp
        assert release_source is True
        assert events[-1] == "half"
        events.append("head-upload")
        return early_head

    monkeypatch.setenv("RPU_QWEN3_5_FREE_HF_WEIGHTS", "1")
    monkeypatch.setattr(adapter, "_upload_lm_head_weight", upload_head)
    with pytest.raises((_StopAtNativeWeights, RuntimeError, adapter.RPUBackendError)):
        owner.to_rpu(max_seq_len=128)
    if fault is None:
        expected_head = ["head-upload"] if method == qs.FP16 else []
        assert events == ["raw-check", "claim", "caching-allocator", "four-arenas",
                          "half", *expected_head, "finalized-install", "cleanup"]
    else:
        assert "half" not in events and "finalized-install" not in events
        assert not hasattr(model, "_rpu_swizzle_started")
        expected = ["raw-check"]
        if fault != "raw":
            expected += ["claim", "caching-allocator"]
            if fault != "allocator":
                expected.append("four-arenas")
            expected += ["release", "cleanup"]
        assert events == expected
    assert causal_lm._LIVE_REF is None
    # A resource conflict must not strand the process-wide model owner.
    replacement = Model()
    causal_lm._claim_live_instance(replacement)
    causal_lm._release_live_instance(replacement)


@pytest.mark.parametrize("method", [qs.FP16, qs.W8A16, qs.W4A16_PGRP,
    "w4_mlp_only", "w4_mlp_w8_attn_gdn", "w4_main_w8_gdn", "w8_main_w8_gdn",
    *qs.AWQ_PROFILES])
@pytest.mark.parametrize("count", [4, 6, 8])
def test_real_cache_factory_keeps_top_level_quantization_and_exact_physical_heads(method, count):
    from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache

    class CaptureCache(Qwen3_5Cache):
        def __init__(self, **kwargs):
            self.geometry = kwargs

    config = runtime_config27(method)
    original_child = config.text_config.to_dict()
    model = SimpleNamespace(config=config, _rpu_execution={"model": {"num_cores": count}})
    model.model = SimpleNamespace(language_model=SimpleNamespace(
        _rpu_qwen3_5=SimpleNamespace(execution_topology=cores.core_topology(model._rpu_execution))))
    cache = CaptureCache.from_model(model, max_seq_len=8192, device="cpu")
    assert cache.geometry["num_layers"] == 64
    assert cache.geometry["num_kv_heads"] == 24
    assert cache.geometry["num_v_heads"] == 48
    assert cache.geometry["conv_dim"] == 10240
    assert cache.geometry["attn_tp"] == (8 if count == 8 else 4)
    assert cache.geometry["num_cores"] == count
    assert config.text_config.to_dict() == original_child
    assert getattr(config.text_config, "quant_config", None) is None
    reduced = CaptureCache.from_config(config, max_seq_len=8192, device="cpu", num_cores=6)
    assert reduced.geometry["attn_tp"] == 4


# Head allocation order, actual packing, and failure retirement. Tiny CPU
# tensors exercise the real upload/cleanup code; no device or checkpoint.
from rpu_backend.adapters import qwen3_5 as adapter
from rpu_backend.api import causal_lm
from rpu_backend.runtime import weights as runtime_weights, hw_attrs
from rpu_backend.graph import GraphRuntimePolicy
from test_qwen3_5_failed_install_retirement import resources


@pytest.fixture
def small_install(resources, monkeypatch):
    c = resources
    del c.inner._rpu_qwen3_5
    del c.root._lm_head_w_rpu_keepalive
    causal_lm._release_live_instance(c.root)
    c.root._rpu_execution = {"model": {"num_cores": 8}}
    c.root.config = runtime_config27(qs.FP16)
    c.inner.embed_tokens = torch.nn.Embedding(512, 128, dtype=torch.float16)
    c.output = torch.nn.Linear(128, 512, bias=False, dtype=torch.float16)
    c.root.get_output_embeddings = lambda: c.output
    c.root.model = c.inner
    c.inner.to = lambda *_a, **_k: (c.events.append("bulk"), c.inner)[1]
    c.root.half = lambda: (c.events.append("half"), c.root)[1]
    c.adapter = adapter.Qwen3_5Adapter.__new__(adapter.Qwen3_5Adapter)
    vars(c.adapter).update(model=c.root, _rpu_is_ready=False, _handle=None,
                          _graph_cache=None, _execution_session=c.session)
    monkeypatch.setattr(cores, "validate_dense_large_vl_weights", lambda *_: None)
    monkeypatch.setattr(cores, "validate_cold_weight_dtype", lambda *_a, **_k: None)
    for name in ("validate_preinstall", "validate_postinstall", "install_hw_attr_validator"):
        monkeypatch.setattr(hw_attrs, name, lambda *_: None)
    monkeypatch.setattr(torch.rpu, "set_caching_allocator", lambda enabled:
                        c.events.append("caching" if enabled else "direct"))
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", lambda policy:
                        (c.events.append("arenas"), True)[1])
    monkeypatch.setattr(runtime_weights, "_release_cpu_weight_pages", lambda: c.events.append("free_cpu"))
    monkeypatch.setenv("RPU_QWEN3_5_FREE_HF_WEIGHTS", "1")
    original_to = torch.Tensor.to
    def tensor_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, (str, torch.device)) and torch.device(device).type == "rpu":
            c.events.append("head_upload")
            if "head" in c.faults:
                raise RuntimeError("head allocation failed")
            return tensor.clone()
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    def install(*args, **kwargs):
        c.events.append("text_install")
        if "before_publish" in c.faults:
            raise RuntimeError("text before publish failed")
        c.inner._rpu_qwen3_5 = c.state
        if "after_publish" in c.faults:
            raise RuntimeError("text after publish failed")
        return 7
    monkeypatch.setattr(adapter, "install_qwen3_5_text_for_rpu", install)
    return c



@pytest.mark.parametrize("kind", ["official_fp16", "official_w8", "dense9_fp16"])
def test_order_once_and_original_payload(small_install, kind):
    c = small_install
    if kind == "official_w8": c.root.config = runtime_config27(qs.W8A16)
    if kind == "dense9_fp16":
        c.root.config = dense_config(9)
        c.session._graph_mode = "RETAINED_CACHE"  # actual 9B public cold contract
    raw = c.output.weight.detach().clone()
    expected = adapter.transform_linear_weight(raw, 1, 8)
    result = c.adapter.to_rpu(max_seq_len=64)
    assert result is c.root
    assert c.events.count("head_upload") == c.events.count("text_install") == 1
    assert c.events.index("arenas") < c.events.index("half")
    if kind == "official_fp16":
        assert c.events.index("half") < c.events.index("head_upload") < c.events.index("text_install")
        assert c.output.weight.numel() == 0
    else:
        assert c.events.index("text_install") < c.events.index("head_upload")
    assert torch.equal(c.root._lm_head_w_rpu_keepalive, expected)
    assert ("bulk" in c.events) == (kind == "dense9_fp16")
    adapter._cleanup_failed_text_install(c.adapter, c.root, c.inner, False, None)
    assert not hasattr(c.root, "_lm_head_w_rpu_keepalive")
    assert c.events[-3:] == ["clear", "discard", "destroy"]



@pytest.mark.parametrize("fault", ["head", "before_publish", "after_publish"])
def test_primary_failure_cleanup_and_no_retry(small_install, fault):
    c = small_install
    c.faults.add(fault)
    with pytest.raises(RuntimeError, match="failed") as first:
        c.adapter.to_rpu(max_seq_len=64)
    assert not hasattr(c.root, "_lm_head_w_rpu_keepalive")
    assert not hasattr(c.inner, "_rpu_qwen3_5")
    assert c.root._rpu_swizzle_started and not c.adapter._rpu_is_ready
    before = list(c.events)
    with pytest.raises(adapter.RPUBackendError, match="prior install failed"):
        c.adapter.to_rpu(max_seq_len=64)
    assert c.events == before
    assert ("text_install" in c.events) == (fault != "head")
    assert ("destroy" in c.events) == (fault == "after_publish")
    assert not getattr(first.value, "__notes__", ())



@pytest.mark.parametrize("cleanup_fault", ["clear", "discard", "destroy"])
def test_failed_retirement_keeps_early_head_and_poison(small_install, cleanup_fault):
    c = small_install
    c.faults.update({"after_publish", cleanup_fault})
    with pytest.raises(RuntimeError, match="text after publish failed") as primary:
        c.adapter.to_rpu(max_seq_len=64)
    assert c.root._lm_head_w_rpu_keepalive is not None
    assert c.inner._rpu_qwen3_5 is c.state
    assert c.state.retirement.failed and c.state.handle == 7
    assert c.session.stats()["state"] == "POISONED"
    assert any(cleanup_fault + " failed" in note for note in primary.value.__notes__)
    calls = list(c.events)
    with pytest.raises(Exception): c.adapter.to_rpu(max_seq_len=64)
    assert c.events == calls
    causal_lm._release_live_instance(c.root)
    with pytest.raises(RuntimeError, match="process is unsafe"):
        causal_lm._claim_live_instance(SimpleNamespace())



@pytest.mark.parametrize("alias", ["untied", "same_parameter", "shared_storage", "none_output"])
@pytest.mark.parametrize("release_source", [False, True])
def test_real_upload_alias_and_release(small_install, alias, release_source):
    c = small_install
    embedding = c.inner.embed_tokens
    if alias == "same_parameter": c.output.weight = embedding.weight
    elif alias == "shared_storage": c.output.weight = torch.nn.Parameter(embedding.weight.detach())
    output = None if alias == "none_output" else c.output
    raw = (output.weight if output is not None else embedding.weight).detach().clone()
    embed_before = embedding.weight.detach().clone()
    expected = adapter.transform_linear_weight(raw, 1, 8)
    actual = adapter._upload_lm_head_weight(output, embedding, 8, release_source=release_source)
    assert torch.equal(actual, expected)
    assert torch.equal(embedding.weight, embed_before)
    freed = release_source and alias == "untied"
    assert c.events.count("free_cpu") == int(freed)
    if output is not None: assert (output.weight.numel() == 0) == freed
    assert c.events.count("head_upload") == 1

"""Large runtime-quantized profile admission and composite installation, without a board.

Use real checkpoint config predicates, adapter policy and installation order.
Only materialized tensor validation, device transfer and native components are
doubled; checkpoint arithmetic and physical owners have separate loader tests.
"""
import copy

import pytest
import torch

from rpu_backend.adapters import qwen3_vl
from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.quant import load, load_qwen3_vl_runtime as runtime
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import (
    _Model,
    install_runtime,  # noqa: F401
)
from test_qwen3_vl_runtime_quant_loading import config


PROFILES = [(8, 8), (8, 4), (32, 8), (32, 4)]


@pytest.mark.parametrize("size,bits", PROFILES)
def test_large_runtime_config_enters_real_adapter_without_legacy_opt_in(monkeypatch, size, bits):
    monkeypatch.delenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED", raising=False)
    cfg = config(size, bits)
    assert runtime.is_qwen3_vl_runtime_quant_config(cfg)
    assert not load.is_qwen3_vl_32b_w8a16_config(cfg)
    qwen3_vl.Qwen3VLAdapter.preflight(cfg)
    qwen3_vl.Qwen3VLAdapter.preflight_execution(cfg, {
        "prefill": {"chunk_size": "auto", "linear_acc32": False, "fast_replay": True},
        "vision": {"linear_acc32": False},
    })


@pytest.mark.parametrize("size,bits", PROFILES)
@pytest.mark.parametrize("fault", ["head_scope", "text_geometry", "vision_padding", "tap_order"])
def test_large_adapter_rejects_nearby_runtime_profiles(size, bits, fault):
    cfg = config(size, bits)
    if fault == "head_scope":
        cfg.quant_config["quantized_lm_head"] = False
    elif fault == "text_geometry":
        cfg.text_config.num_hidden_layers -= 1
    elif fault == "vision_padding":
        cfg.vision_config.intermediate_size = 4352
    else:
        cfg.vision_config.deepstack_visual_indexes = [8, 24, 16]
    with pytest.raises(UnsupportedModelError):
        qwen3_vl.Qwen3VLAdapter.preflight(cfg)


def test_8b_w4_without_preserved_awq_recipe_is_rejected():
    cfg = config(8, 4)
    cfg.quant_config.pop("source_quantization")
    with pytest.raises(UnsupportedModelError):
        qwen3_vl.Qwen3VLAdapter.preflight(cfg)


@pytest.mark.parametrize("opt_in", [None, "1"])
def test_32b_fp16_remains_excluded_even_with_legacy_opt_in(monkeypatch, opt_in):
    cfg = config(32, 8, source=True)
    if opt_in is None:
        monkeypatch.delenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED", raising=False)
    else:
        monkeypatch.setenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED", opt_in)
    with pytest.raises(UnsupportedModelError):
        qwen3_vl.Qwen3VLAdapter.preflight(cfg)


@pytest.mark.parametrize("opt_in", [None, "true", "1"])
def test_legacy_32b_config_needs_no_opt_in_and_keeps_bounded_envelope(monkeypatch, opt_in):
    cfg = config(32, 8)
    cfg.quant_config = copy.deepcopy(load._QWEN3_VL_32B_W8A16_QUANT_CONFIG)
    if opt_in is None:
        monkeypatch.delenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED", raising=False)
    else:
        monkeypatch.setenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED", opt_in)
    qwen3_vl.Qwen3VLAdapter.preflight(cfg)
    for enabled in (False, True):
        qwen3_vl.Qwen3VLAdapter.preflight_execution(cfg, {
            "prefill": {"linear_acc32": enabled, "chunk_size": 64}})
        with pytest.raises(UnsupportedModelError, match="ceiling 64"):
            qwen3_vl.Qwen3VLAdapter.preflight_execution(cfg, {
                "prefill": {"linear_acc32": enabled, "chunk_size": 128}})


@pytest.mark.parametrize("size,bits", PROFILES)
@pytest.mark.parametrize("cores", [4, 6])
def test_large_quantized_profiles_bind_component_topology(size, bits, cores):
    cfg = config(size, bits)
    execution = {"model": {"num_cores": cores}}
    qwen3_vl.Qwen3VLAdapter.preflight_execution(cfg, execution)
    topology = qwen3_vl._qwen3_vl_core_topology(cfg, execution)
    assert topology.num_cores == cores
    assert topology.attn_tp == topology.lm_head_tp == 4
    assert topology.mlp_tp == cores  # packed W4 pads its MLP6 width instead of MLP4
    assert topology.physical_kv_cores == 8


@pytest.mark.parametrize("size,bits", PROFILES)
@pytest.mark.parametrize("stage", ["prefill", "vision"])
@pytest.mark.parametrize("component_override", [False, True])
def test_large_quantized_acc32_is_admitted_with_component_priority(size, bits, stage, component_override):
    execution = {stage: {"linear_acc32": True}}
    if component_override:
        component = "language_model" if stage == "prefill" else "vision_encoder"
        execution = {stage: {"linear_acc32": False}, "components": {component: execution}}
    qwen3_vl.Qwen3VLAdapter.preflight_execution(config(size, bits), execution)


@pytest.mark.parametrize("stage", ["prefill", "vision"])
def test_acc16_component_override_wins_over_root_acc32_default(stage):
    component = "language_model" if stage == "prefill" else "vision_encoder"
    qwen3_vl.Qwen3VLAdapter.preflight_execution(config(32, 8), {
        stage: {"linear_acc32": True},
        "components": {component: {stage: {"linear_acc32": False}}},
    })


@pytest.mark.parametrize("size,bits,source", [(2, 8, False), (2, 4, False),
                                             (4, 8, False), (4, 4, False), (8, 8, True)])
def test_large_quantized_acc16_boundary_does_not_change_existing_acc32_selection(size, bits, source):
    qwen3_vl.Qwen3VLAdapter.preflight_execution(config(size, bits, source=source), {
        "prefill": {"linear_acc32": True}, "vision": {"linear_acc32": True}})


@pytest.fixture
def quantized_install(monkeypatch, install_runtime):
    """Use small tensor owners to observe real top-level routing and policy."""
    state = install_runtime
    from rpu_backend.graph import GraphRuntimePolicy

    # Cold arena allocation is a device boundary; dedicated arena-plan tests
    # exercise the real owner selection, capacity and failure lifecycle.
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", lambda self: True)

    def validate(model):
        assert runtime.is_qwen3_vl_runtime_quant_config(model.config)
        state["events"].append(("validate_runtime_inventory",))
        return True

    def move(model):
        assert model._rpu_swizzle_started and state["allocator_enabled"]
        state["events"].append(("move_runtime_decoder",))

    monkeypatch.setattr(runtime, "_validate_qwen3_vl_runtime_model", validate)
    monkeypatch.setattr(runtime, "_move_materialized_runtime_decoder_for_rpu", move)
    ordinary_transform = qwen3_vl.transform_linear_weight

    def transform(weight, partition):
        state["events"].append(("head_swizzle", weight.dtype, partition))
        return ordinary_transform(weight, partition)

    monkeypatch.setattr(qwen3_vl, "transform_linear_weight", transform)
    native_head = torch.ops.rpu.causal_decoder_set_lm_head

    def set_head(handle, weight, scale):
        state["head_payload"] = (weight, scale)
        native_head(handle, weight)

    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head", set_head)

    def merger(model, *, w8a16=False):
        assert w8a16
        state["events"].append(("enable_w8_vision_mergers", model._rpu_vision_handle))
        model._rpu_vision_merger_on_device = True
        model._rpu_patch_embed_on_device = True

    monkeypatch.setattr(qwen3_vl, "enable_qwen3_vl_vision_merger_on_device", merger)
    monkeypatch.setattr(qwen3_vl, "register_qwen3vl_vision_fused_merger",
                        lambda *a, **k: pytest.fail("large profile selected a small-model fused merger"))
    for name in ("qwen3vl_vision_set_n1200_tm160", "qwen3vl_vision_set_w8_compact_encoder"):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda *a, name=name: pytest.fail(f"large profile selected {name}"),
                            raising=False)
    for name in ("causal_decoder_set_fast_replay", "causal_decoder_set_preload_replay_skip"):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda handle, enabled, name=name: state["events"].append((name, handle, enabled)),
                            raising=False)
    return state


def small_quantized_model(size, bits):
    model = _Model()
    model.config = config(size, bits)
    dtype = torch.int8 if bits == 8 else torch.uint8
    layer = torch.nn.Module()
    layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
    for role in qwen3_vl._QWEN3_VL_TEXT_PROJECTIONS:
        owner, name = role.split(".")
        projection = torch.nn.Module()
        projection.register_parameter("weight", torch.nn.Parameter(
            torch.ones((2, 2), dtype=dtype), requires_grad=False))
        projection.register_buffer("weight_scale", torch.ones(2, dtype=torch.float16))
        getattr(layer, owner).add_module(name, projection)
    text = model.model.language_model
    text.config, text.layers = model.config.text_config, [layer]
    model.model.visual.config = model.config.vision_config
    model.lm_head.weight = torch.nn.Parameter(
        torch.ones((2, 2), dtype=dtype), requires_grad=False)
    model.lm_head.weight_scale = torch.ones(2, dtype=torch.float16)
    return model


@pytest.mark.parametrize("size,bits", PROFILES)
@pytest.mark.parametrize("fast_replay", [None, True])
def test_large_runtime_to_rpu_uses_private_installers_and_generic_replay(
    quantized_install, size, bits, fast_replay,
):
    state = quantized_install
    model = small_quantized_model(size, bits)
    if fast_replay is not None:
        model._rpu_execution = {"prefill": {"fast_replay": fast_replay, "linear_acc32": False,
                                            "chunk_size": 64},
                                "vision": {"linear_acc32": False}}
    owner = qwen3_vl.Qwen3VLAdapter(model)
    assert owner.to_rpu() is model
    vision_args, text_args = state["vision_kwargs"], state["text_kwargs"]
    assert vision_args["_allow_runtime_quantized_large"] is True
    assert vision_args["_allow_padded_8b"] is False
    assert vision_args["_allow_graph_blocked_32b"] is False
    assert vision_args["w8a16"] is True
    assert not vision_args.get("_pack_fp16_block_weights", False)
    assert text_args.get("_runtime_quantized_32b", False) is (size == 32)
    assert not text_args.get("runtime_align_w8a16", False)
    assert text_args["deepstack_lang_layers"] == [0, 1, 2]
    assert text_args.get("logical_vocab_size") == (model.config.text_config.vocab_size if bits == 4 else None)
    for role, scales in zip(qwen3_vl._QWEN3_VL_TEXT_PROJECTIONS, text_args["scale_lists"], strict=True):
        assert len(scales) == 1
        assert scales[0] is model.model.language_model.layers[0].get_submodule(role).weight_scale
    events = state["events"]
    assert events.count(("move_runtime_decoder",)) == 1
    assert events.index(("set_caching_allocator", True)) < events.index(("move_runtime_decoder",))
    assert ("swizzle", "text") not in events
    if bits == 8:
        assert ("head_swizzle", torch.int8, 1) in events
    else:
        assert not any(event[0] == "head_swizzle" for event in events)
    assert ("set_linear_acc32", 22, False) in events
    assert ("set_vision_linear_acc32", 11, False) in events
    assert owner._linear_acc32_policy == {"prefill": False, "vision": False}
    assert ("enable_w8_vision_mergers", 11) in events
    for name in ("qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip"):
        assert (name, 11, True) in events
    for name in ("causal_decoder_set_fast_replay", "causal_decoder_set_preload_replay_skip"):
        assert ((name, 22, True) in events) is (fast_replay is True)
    visual = model.model.visual
    assert not getattr(visual, "_rpu_vision_chunked_merger_w8a16", False)
    assert not getattr(visual, "_rpu_vision_compact_encoder_w8a16", False)
    weight, scale = state["head_payload"]
    assert weight.dtype == (torch.int8 if bits == 8 else torch.uint8)
    assert scale.data_ptr() == model.lm_head.weight_scale.data_ptr()
    assert weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert owner.to_rpu() is model
    assert events.count(("move_runtime_decoder",)) == 1
    owner.close()


@pytest.mark.parametrize("size,bits", PROFILES)
def test_large_config_change_fails_before_claim_and_weight_mutation(quantized_install, size, bits):
    model = small_quantized_model(size, bits)
    owner = qwen3_vl.Qwen3VLAdapter(model)
    quantized_install["events"].clear()
    model.config.vision_config.deepstack_visual_indexes = [8, 24, 16]
    with pytest.raises(UnsupportedModelError):
        owner.to_rpu()
    assert quantized_install["claims"] == []
    assert quantized_install["events"] == []
    assert not getattr(model, "_rpu_swizzle_started", False)


@pytest.mark.parametrize("bits", [8, 4])
def test_runtime_32b_public_explicit_chunk_uses_private_ceiling(bits):
    cfg = config(32, bits)
    qwen3_vl.Qwen3VLAdapter.preflight_execution(cfg, {"prefill": {"chunk_size": 64}})
    with pytest.raises(UnsupportedModelError, match="ceiling 64"):
        qwen3_vl.Qwen3VLAdapter.preflight_execution(cfg, {"prefill": {"chunk_size": 80}})

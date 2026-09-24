"""CPU doubles exercise eager-merger installation, output and retirement ownership."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import qwen3_vl as qvl
from rpu_backend.adapters.qwen3_vl import vision
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import (  # noqa: F401
    install_runtime, _Model,
)


def _merger(postshuffle=False):
    return SimpleNamespace(
        use_postshuffle_norm=postshuffle,
        norm=torch.nn.LayerNorm(8 if postshuffle else 2),
        linear_fc1=torch.nn.Linear(8, 8),
        linear_fc2=torch.nn.Linear(8, 3),
    )


def _visual():
    return SimpleNamespace(
        merger=_merger(), deepstack_merger_list=[_merger(True)],
        _rpu_vision_handle=11, _rpu_vision_has_dispatched=False,
        _rpu_vision_host_fp32_patch=False,
        _rpu_vision_patch_embed_w=torch.ones(3, 2, dtype=torch.float16),
        _rpu_vision_patch_embed_b=torch.ones(3, dtype=torch.float16),
    )


@pytest.fixture
def cpu_transfers(monkeypatch):
    original = torch.Tensor.to
    state = {"calls": 0, "fail_at": None, "error": RuntimeError}
    def to(tensor, *args, **kwargs):
        positional_rpu = bool(args) and isinstance(args[0], str) and args[0] == "rpu"
        if kwargs.get("device") == "rpu" or positional_rpu:
            state["calls"] += 1
            if state["calls"] == state["fail_at"]:
                raise state["error"]("transfer interrupted")
            if positional_rpu:
                args = ("cpu", *args[1:])
            else:
                kwargs["device"] = "cpu"
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(vision, "tp_col_swizzle_mc_weight", lambda value, cores: value.clone())
    return state


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_at", [2, 6, 10])
def test_partial_aux_install_rolls_back_every_scope_and_can_retry(cpu_transfers, error, fail_at):
    visual = _visual()
    originals = [merger.linear_fc1.weight for merger in (visual.merger, *visual.deepstack_merger_list)]
    cpu_transfers.update(fail_at=fail_at, error=error)
    with pytest.raises(error, match="transfer interrupted"):
        vision.enable_qwen3_vl_vision_merger_on_device(visual)
    for merger, original in zip((visual.merger, *visual.deepstack_merger_list), originals):
        assert merger.linear_fc1.weight is original
        assert not any(name in vars(merger) for name in vision._VISION_MERGER_AUX_ATTRS)
    assert not any(name in vars(visual) for name in (
        "_rpu_patch_embed_w_rpu", "_rpu_patch_embed_b_rpu",
        "_rpu_patch_embed_on_device", "_rpu_vision_merger_on_device"))
    cpu_transfers["fail_at"] = None
    vision.enable_qwen3_vl_vision_merger_on_device(visual)
    assert visual._rpu_patch_embed_on_device and visual._rpu_vision_merger_on_device
    calls = cpu_transfers["calls"]
    visual._rpu_vision_has_dispatched = True
    vision.enable_qwen3_vl_vision_merger_on_device(visual)
    assert cpu_transfers["calls"] == calls


def test_failed_aux_completion_preserves_already_installed_merger(cpu_transfers):
    visual = _visual()
    vision._convert_vision_merger_for_rpu(visual.merger)
    old = {name: vars(visual.merger)[name] for name in vision._VISION_MERGER_AUX_ATTRS}
    cpu_transfers["fail_at"] = cpu_transfers["calls"] + 2
    with pytest.raises(RuntimeError, match="transfer interrupted"):
        vision.enable_qwen3_vl_vision_merger_on_device(visual)
    assert all(vars(visual.merger)[name] is value for name, value in old.items())
    assert not hasattr(visual.deepstack_merger_list[0], "_rpu_merger_on_device")


@pytest.mark.parametrize("attribute,value,error", [
    ("_rpu_vision_handle", None, RuntimeError),
    ("_rpu_vision_has_dispatched", True, RuntimeError),
    ("_rpu_vision_host_fp32_patch", True, ValueError),
])
def test_unsafe_precision_change_fails_before_any_transfer(cpu_transfers, attribute, value, error):
    visual = _visual()
    setattr(visual, attribute, value)
    with pytest.raises(error):
        vision.enable_qwen3_vl_vision_merger_on_device(visual)
    assert cpu_transfers["calls"] == 0


@pytest.mark.parametrize("postshuffle", [False, True])
def test_merger_keeps_cpu_fp32_norm_and_returns_independent_cpu_snapshot(cpu_transfers, monkeypatch, postshuffle):
    merger = _merger(postshuffle)
    vision._convert_vision_merger_for_rpu(merger)
    seen = []
    merger.norm.register_forward_pre_hook(lambda module, args: seen.append((args[0].dtype, args[0].shape)))
    final = torch.arange(3, dtype=torch.float16).reshape(1, 3)
    calls = []
    linear = torch.nn.functional.linear
    class DeviceResult:
        def float(self):
            pytest.fail("FP32 expansion must happen after transferring to CPU")
        def cpu(self):
            return final
    def measured_linear(*args, **kwargs):
        calls.append(args[0].dtype)
        return linear(*args, **kwargs) if len(calls) == 1 else DeviceResult()
    def native_linear(x, weight, bias, partition, cores, acc32, scale=None):
        assert partition == 1 and cores == 8 and acc32 is False and scale is None
        return measured_linear(x, weight, bias)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", native_linear, raising=False)
    output = vision._merger_forward_on_device(merger, torch.arange(8, dtype=torch.float32).reshape(4, 2))
    assert seen == [(torch.float32, torch.Size((1, 8) if postshuffle else (4, 2)))]
    assert calls == [torch.float16, torch.float16]
    assert output.dtype == torch.float32 and output.device.type == "cpu"
    saved = output.clone()
    final.add_(100)
    assert torch.equal(output, saved)


@pytest.mark.parametrize("fail_destroy", [False, True])
def test_composite_close_clears_child_aux_only_after_native_retirement(install_runtime, monkeypatch, fail_destroy):
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    visual = model.model.visual
    visual.merger = _merger()
    visual.deepstack_merger_list = [_merger(True)]
    children = (visual.merger, *visual.deepstack_merger_list)
    for merger in children:
        for name in vision._VISION_MERGER_AUX_ATTRS:
            vars(merger)[name] = torch.ones(1)
    def destroy(handle):
        assert all(hasattr(merger, "_rpu_fc1_w") for merger in children)
        if fail_destroy:
            raise RuntimeError("vision destroy failed")
        install_runtime["events"].append(("destroy", "vision"))
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_destroy", destroy, raising=False)
    if fail_destroy:
        with pytest.raises(RuntimeError, match="vision destroy failed"):
            adapter.close()
        assert all(hasattr(merger, "_rpu_fc1_w") for merger in children)
    else:
        adapter.close()
        assert all(not any(name in vars(merger) for name in vision._VISION_MERGER_AUX_ATTRS)
                   for merger in children)
        assert all(merger.linear_fc1.weight.dtype == torch.float32 for merger in children)


def _model_2b():
    """Exact ordinary 2B metadata, retaining the tiny installation tensors."""
    model = _Model()
    model.config.architectures = ["Qwen3VLForConditionalGeneration"]
    model.config.tie_word_embeddings = True
    model.config.image_token_id, model.config.video_token_id = 151655, 151656
    model.config.vision_start_token_id, model.config.vision_end_token_id = 151652, 151653
    model.config.text_config = SimpleNamespace(
        model_type="qwen3_vl_text", hidden_size=2048,
        intermediate_size=6144, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        hidden_act="silu", rms_norm_eps=1e-6, vocab_size=151936,
        attention_bias=False, use_cache=True, tie_word_embeddings=True,
        rope_scaling={"rope_type": "default", "rope_theta": 5_000_000,
                      "mrope_interleaved": True, "mrope_section": [24, 20, 20]},
    )
    model.config.vision_config = SimpleNamespace(
        model_type="qwen3_vl", hidden_size=1024, intermediate_size=4096,
        depth=24, num_heads=16, patch_size=16, temporal_patch_size=2,
        spatial_merge_size=2, deepstack_visual_indexes=[5, 11, 17],
        hidden_act="gelu_pytorch_tanh", out_hidden_size=2048,
        in_channels=3, num_position_embeddings=2304,
    )
    return model


def _model_4b():
    model = _model_2b()
    text = model.config.text_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 2560, 9728, 36
    text.num_attention_heads = 32
    model.config.vision_config.out_hidden_size = 2560
    return model


def _model_8b():
    """Exact public config with the small installation doubles above."""
    model = _Model()
    model.config.tie_word_embeddings = False
    model.config.text_config = SimpleNamespace(
        model_type="qwen3_vl_text", hidden_size=4096,
        intermediate_size=12288, num_hidden_layers=36,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        hidden_act="silu", rms_norm_eps=1e-6, vocab_size=151936,
        attention_bias=False, use_cache=True,
        rope_scaling={"rope_type": "default", "rope_theta": 5_000_000,
                      "mrope_interleaved": True, "mrope_section": [24, 20, 20]},
    )
    model.config.vision_config = SimpleNamespace(
        model_type="qwen3_vl", hidden_size=1152, intermediate_size=4304,
        depth=27, num_heads=16, patch_size=16, temporal_patch_size=2,
        spatial_merge_size=2, deepstack_visual_indexes=[8, 16, 24],
        hidden_act="gelu_pytorch_tanh", out_hidden_size=4096,
        in_channels=3, num_position_embeddings=2304,
    )
    return model


@pytest.mark.parametrize(
    "factory,legacy_2b",
    [(_model_2b, True), (_model_4b, False)],
)
def test_dense_eager_profile_accepts_only_the_exact_ordinary_configs(
    factory, legacy_2b,
):
    model = factory()
    assert qvl._is_qwen3_vl_dense_eager_profile(model)
    assert qvl._is_qwen3_vl_2b_eager_profile(model) is legacy_2b


@pytest.mark.parametrize("factory", [_model_2b, _model_4b])
@pytest.mark.parametrize("scope", ["root", "text", "vision"])
@pytest.mark.parametrize("attribute", ["quant_config", "quantization_config"])
def test_dense_eager_profile_rejects_quant_metadata_at_every_config_scope(
    factory, scope, attribute,
):
    model = factory()
    owner = {
        "root": model.config,
        "text": model.config.text_config,
        "vision": model.config.vision_config,
    }[scope]
    setattr(owner, attribute, {})
    assert not qvl._is_qwen3_vl_dense_eager_profile(model)


@pytest.mark.parametrize("factory", [_model_2b, _model_4b])
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model: setattr(model.config, "architectures", ["Other"]),
            id="architecture",
        ),
        pytest.param(
            lambda model: setattr(model.config, "tie_word_embeddings", False),
            id="root-tie",
        ),
        pytest.param(
            lambda model: setattr(model.config, "image_token_id", 151654),
            id="image-token",
        ),
        pytest.param(
            lambda model: setattr(model.config.text_config, "rms_norm_eps", 1e-5),
            id="text-rms-eps",
        ),
        pytest.param(
            lambda model: model.config.text_config.rope_scaling.update(
                rope_type="linear"
            ),
            id="text-rope",
        ),
        pytest.param(
            lambda model: setattr(
                model.config.text_config, "tie_word_embeddings", False
            ),
            id="text-tie",
        ),
        pytest.param(
            lambda model: setattr(model.config.vision_config, "model_type", "other"),
            id="vision-model-type",
        ),
        pytest.param(
            lambda model: setattr(
                model.config.vision_config, "num_position_embeddings", 2303
            ),
            id="vision-position-embeddings",
        ),
        pytest.param(
            lambda model: setattr(
                model.config.vision_config,
                "deepstack_visual_indexes",
                [5, 11, 16],
            ),
            id="vision-deepstack",
        ),
    ],
)
def test_dense_eager_profile_rejects_semantic_drift(factory, mutate):
    model = factory()
    mutate(model)
    assert not qvl._is_qwen3_vl_dense_eager_profile(model)


@pytest.fixture
def eager_install_runtime(install_runtime, cpu_transfers, monkeypatch):
    """Run the real auxiliary converter from the real top-level install."""
    installer = qvl.install_qwen3_vl_vision_for_rpu
    state = install_runtime
    state["eager_calls"] = []
    state["bank_moves"] = []
    state["vision_routes"] = []

    # This lifecycle double has no decoder layers. Bank allocation itself is
    # covered separately; retain the selected precision/per-layer contract.
    from rpu_backend.quant import load as quant_load
    def move_banks(model, *, dtype, per_layer=False):
        state["bank_moves"].append((model.config.text_config.hidden_size, dtype, per_layer))
    monkeypatch.setattr(quant_load, "_move_materialized_decoder_for_rpu", move_banks)

    def install_visual(model, **kwargs):
        handle = installer(model, **kwargs)
        # The native tower installer is mocked, but auxiliary tensors/conversion
        # and retirement run through production code with tiny CPU transfers.
        for name, value in vars(_visual()).items():
            if name != "_rpu_vision_handle":
                vars(model)[name] = value
        model._rpu_vision_host_fp32_patch = qvl.rpu_env_bool("RPU_QWEN3VL_VISION_HOST_FP32_PATCH")
        return handle

    def enable(visual):
        state["eager_calls"].append(visual)
        assert ("install", "vision") in state["events"]
        assert ("install", "text") in state["events"]
        assert ("set_vision_linear_acc32", 11, visual._rpu_vision_linear_acc32) in state["events"]
        owner = visual._qwen3_vl_retirement_owner
        assert not owner._rpu_is_ready
        assert not visual._rpu_vision_has_dispatched
        return vision.enable_qwen3_vl_vision_merger_on_device(visual)

    monkeypatch.setattr(qvl, "install_qwen3_vl_vision_for_rpu", install_visual)
    monkeypatch.setattr(qvl, "enable_qwen3_vl_vision_merger_on_device", enable)
    def register_merger(visual):
        assert state["bank_moves"], "ordinary VL8 must not register a fused merger"
        visual._rpu_vision_fused_merger = True
        state["vision_routes"].append("fused_merger")

    def vision_route(handle, enabled, *, name):
        assert state["bank_moves"], "ordinary VL8 must not enable compact Vision routes"
        assert (handle, enabled) == (11, True)
        state["vision_routes"].append(name)

    monkeypatch.setattr(qvl, "register_qwen3vl_vision_fused_merger", register_merger)
    for name in (
        "qwen3vl_vision_set_large_image_auto_chunk",
        "qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip",
        "qwen3vl_vision_set_fp16_chunked_merger", "qwen3vl_vision_set_fp16_compact_encoder",
    ):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda *args, name=name: vision_route(*args, name=name),
                            raising=False)
    for name in (
        "qwen3vl_vision_set_bake_merger", "causal_decoder_set_fast_replay",
        "causal_decoder_set_preload_replay_skip",
    ):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda *args: pytest.fail("ordinary VL must not enable delivery-only routes"),
                            raising=False)
    return state


@pytest.mark.parametrize("factory", [_model_2b, _model_4b, _model_8b])
@pytest.mark.parametrize("acc32", [False, True])
def test_public_defaults_to_eager_aux_once_and_retires_it(eager_install_runtime, cpu_transfers, factory, acc32):
    source = factory()
    if acc32:
        source._rpu_execution = {"vision": {"linear_acc32": True}}
    adapter = qvl.Qwen3VLAdapter(source)
    model = adapter.to_rpu()
    visual = model.model.visual
    assert eager_install_runtime["eager_calls"] == [visual]
    assert visual._rpu_vision_merger_on_device and visual._rpu_patch_embed_on_device
    assert all(merger._rpu_merger_on_device
               for merger in (visual.merger, *visual.deepstack_merger_list))
    fp16_vision_banks = factory in (_model_2b, _model_4b)
    compact_vision = fp16_vision_banks and acc32
    assert bool(getattr(visual, "_rpu_vision_fused_merger", False)) is compact_vision
    assert bool(getattr(visual, "_rpu_vision_chunked_merger_fp16", False)) is compact_vision
    assert bool(getattr(visual, "_rpu_vision_compact_encoder_fp16", False)) is compact_vision
    assert eager_install_runtime["bank_moves"] == [
        (model.config.text_config.hidden_size, torch.float16, factory is not _model_2b)]
    expected_routes = []
    if fp16_vision_banks:
        if acc32:
            expected_routes.append("qwen3vl_vision_set_large_image_auto_chunk")
        expected_routes += ["qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip"]
        if acc32:
            expected_routes += ["fused_merger", "qwen3vl_vision_set_fp16_chunked_merger",
                                "qwen3vl_vision_set_fp16_compact_encoder"]
    assert eager_install_runtime["vision_routes"] == expected_routes
    calls = cpu_transfers["calls"]
    assert calls > 0
    assert adapter.to_rpu() is model
    assert cpu_transfers["calls"] == calls
    assert eager_install_runtime["eager_calls"] == [visual]
    adapter.close()
    assert ("destroy", "vision") in eager_install_runtime["events"]
    assert ("destroy", "text") in eager_install_runtime["events"]
    assert not hasattr(visual, "_rpu_patch_embed_w_rpu")
    assert all(not any(name in vars(merger) for name in vision._VISION_MERGER_AUX_ATTRS)
               for merger in (visual.merger, *visual.deepstack_merger_list))


@pytest.mark.parametrize("profile", [
    "2b_unknown_semantics", "4b_unknown_semantics",
    "2b_root_quant_metadata", "2b_text_quant_metadata", "2b_vision_quant_metadata",
    "4b_root_quant_metadata", "4b_text_quant_metadata", "4b_vision_quant_metadata",
    "2b_quantized_text", "4b_quantized_text", "8b_quantized_text",
    "2b_cpu_fp32", "4b_cpu_fp32", "8b_cpu_fp32",
    "2b_w4", "4b_w4", "8b_w4",
])
def test_public_eager_default_preserves_other_profiles_and_explicit_fp32(
    eager_install_runtime, cpu_transfers, monkeypatch, profile,
):
    model = _model_2b() if profile.startswith("2b") else (_model_4b() if profile.startswith("4b") else _model_8b())
    if profile.endswith("unknown_semantics"):
        model.config.text_config.rms_norm_eps = 1e-5
    if "_quant_metadata" in profile:
        scope = profile.split("_", 2)[1]
        owner = {
            "root": model.config,
            "text": model.config.text_config,
            "vision": model.config.vision_config,
        }[scope]
        owner.quantization_config = {}
    if profile.endswith("cpu_fp32"):
        monkeypatch.setenv("RPU_QWEN3VL_VISION_HOST_FP32_PATCH", "1")
    if profile.endswith("quantized_text"):
        monkeypatch.setattr(qvl, "_qwen3_vl_text_projection_weights_are_fp16", lambda model: False)
    if profile.endswith("w4"):
        monkeypatch.setattr(qvl, "_lm_head_optimization_config", lambda model: (True, 0, False, 32))
        monkeypatch.setattr(qvl, "_quantize_pack_lm_head_w4_groupwise",
                            lambda *args, **kwargs: (torch.ones(2, dtype=torch.uint8), torch.ones(2).half()))
        def set_lm_head(*args):
            eager_install_runtime["lm_head_set"] = True
        monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head", set_lm_head, raising=False)
    adapter = qvl.Qwen3VLAdapter(model)
    adapter.to_rpu()
    assert eager_install_runtime["eager_calls"] == []
    # W4 transfers its lm-head tensors; none of these paths convert vision aux.
    if not profile.endswith("w4"):
        assert cpu_transfers["calls"] == 0
    assert not hasattr(model.model.visual, "_rpu_vision_merger_on_device")
    assert not hasattr(model.model.visual.merger, "_rpu_fc1_w")
    if profile.endswith("cpu_fp32"):
        assert model.model.visual._rpu_vision_host_fp32_patch
    adapter.close()


@pytest.mark.parametrize("scope", ["root", "text", "vision"])
@pytest.mark.parametrize("attribute", ["quant_config", "quantization_config"])
def test_public_8b_rejects_nested_quant_metadata_before_claim_or_transfer(
    eager_install_runtime, cpu_transfers, scope, attribute,
):
    model = _model_8b()
    owner = {
        "root": model.config,
        "text": model.config.text_config,
        "vision": model.config.vision_config,
    }[scope]
    setattr(owner, attribute, {})

    with pytest.raises(
        qvl.UnsupportedModelError,
        match="requires an FP16 checkpoint or a registered runtime quantization profile",
    ):
        qvl.Qwen3VLAdapter(model).to_rpu()

    assert eager_install_runtime["events"] == []
    assert eager_install_runtime["eager_calls"] == []
    assert cpu_transfers["calls"] == 0


@pytest.mark.parametrize("failure", ["aux_transfer", "later_lm_head"])
@pytest.mark.parametrize("factory", [_model_2b, _model_4b, _model_8b])
def test_public_eager_install_failure_retires_handles_and_aux(
    eager_install_runtime, cpu_transfers, failure, factory,
):
    model = factory()
    if failure == "aux_transfer":
        cpu_transfers["fail_at"] = 6
        match = "transfer interrupted"
    else:
        eager_install_runtime["set_error"] = RuntimeError("lm-head registration failed")
        match = "lm-head registration failed"
    adapter = qvl.Qwen3VLAdapter(model)
    with pytest.raises(RuntimeError, match=match):
        adapter.to_rpu()
    visual = model.model.visual
    assert eager_install_runtime["eager_calls"] == [visual]
    assert ("destroy", "vision") in eager_install_runtime["events"]
    assert ("destroy", "text") in eager_install_runtime["events"]
    assert not adapter._rpu_is_ready
    assert not hasattr(visual, "_rpu_patch_embed_w_rpu")
    assert all(not any(name in vars(merger) for name in vision._VISION_MERGER_AUX_ATTRS)
               for merger in (visual.merger, *visual.deepstack_merger_list))
    transfers = cpu_transfers["calls"]
    with pytest.raises(qvl.RPUBackendError, match="prior swizzle attempt|closed"):
        adapter.to_rpu()
    assert cpu_transfers["calls"] == transfers


@pytest.mark.parametrize("field,value", [
    ("hidden_size", 1024), ("intermediate_size", 4352), ("depth", 26),
    ("num_heads", 8), ("patch_size", 14), ("temporal_patch_size", 1),
    ("spatial_merge_size", 4), ("deepstack_visual_indexes", [5, 11, 17]),
    ("hidden_act", "gelu"), ("out_hidden_size", 2048),
    ("model_type", "other"), ("in_channels", 1), ("num_position_embeddings", 1024),
])
def test_8b_vision_mismatch_is_rejected_before_default_aux_or_weight_mutation(
    eager_install_runtime, cpu_transfers, field, value,
):
    model = _model_8b()
    setattr(model.config.vision_config, field, value)
    with pytest.raises(qvl.UnsupportedModelError):
        qvl.Qwen3VLAdapter(model).to_rpu()
    assert eager_install_runtime["claims"] == []
    assert eager_install_runtime["events"] == []
    assert eager_install_runtime["eager_calls"] == []
    assert cpu_transfers["calls"] == 0
    assert not hasattr(model, "_rpu_swizzle_started")


def test_existing_2b_delivery_keeps_its_separate_fused_replay_install(install_runtime, monkeypatch):
    # Delivery config/weight admission is outside this tiny routing double.
    # Exercise the real installation branch after those validators accept it.
    monkeypatch.setattr(qvl, "_validate_delivery_weights", lambda model: True)
    monkeypatch.setattr(qvl, "_normalize_delivery_execution", lambda value: value)
    monkeypatch.setattr(qvl, "_qwen3_vl_text_scale_lists", lambda model: {})
    monkeypatch.setattr(qvl, "_text_decode_execution_plan", lambda *args, **kwargs:
                        SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(1,)))))
    calls = []
    monkeypatch.setattr(qvl, "register_qwen3vl_vision_fused_merger", lambda visual: calls.append("fused"))
    enable = qvl.enable_qwen3_vl_vision_merger_on_device
    def eager(visual):
        calls.append("eager")
        enable(visual)
    monkeypatch.setattr(qvl, "enable_qwen3_vl_vision_merger_on_device", eager)
    for name in (
        "causal_decoder_set_chunk_envelope", "qwen3vl_vision_set_delivery_compatibility",
        "qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip",
        "qwen3vl_vision_set_bake_merger", "causal_decoder_set_fast_replay",
        "causal_decoder_set_preload_replay_skip",
    ):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda *args, name=name: calls.append(name), raising=False)
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    assert calls == [
        "causal_decoder_set_chunk_envelope", "qwen3vl_vision_set_delivery_compatibility",
        "fused", "eager", "qwen3vl_vision_set_fast_replay",
        "qwen3vl_vision_set_preload_replay_skip", "qwen3vl_vision_set_bake_merger",
        "causal_decoder_set_fast_replay", "causal_decoder_set_preload_replay_skip",
    ]
    assert model.model.language_model._rpu_text_fast_replay
    assert model.model.language_model._rpu_text_fused_decode_position
    adapter.close()

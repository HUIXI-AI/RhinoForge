"""Host checks for the new precision recipe and bounded codec admission."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_tts import codec_transformer as codec


def _core():
    config = SimpleNamespace(num_hidden_layers=8, hidden_size=512, intermediate_size=1024,
        num_attention_heads=16, num_key_value_heads=16, head_dim=64,
        sliding_window=72, hidden_act="silu", attention_bias=False)
    layers = [SimpleNamespace(
        self_attn=SimpleNamespace(q_norm=torch.nn.Identity(), k_norm=torch.nn.Identity(),
                                 o_proj=SimpleNamespace(bias=None)),
        mlp=SimpleNamespace(down_proj=SimpleNamespace(bias=None)),
        attention_type="sliding_attention",
        input_layernorm=SimpleNamespace(variance_epsilon=1e-5),
        post_attention_layernorm=SimpleNamespace(variance_epsilon=1e-5),
        self_attn_layer_scale=SimpleNamespace(scale=torch.ones(512)),
        mlp_layer_scale=SimpleNamespace(scale=torch.ones(512))) for _ in range(8)]
    return SimpleNamespace(config=config, layers=layers, norm=SimpleNamespace(variance_epsilon=1e-5),
        input_proj=SimpleNamespace(weight=SimpleNamespace(shape=(512, 1024))),
        output_proj=SimpleNamespace(weight=SimpleNamespace(shape=(1024, 512))),
        parameters=lambda: iter([torch.zeros(1)]))


@pytest.mark.parametrize("field,value", [
    ("num_hidden_layers", 7), ("hidden_size", 1024), ("intermediate_size", 2048),
    ("num_attention_heads", 8), ("num_key_value_heads", 8), ("head_dim", 128),
    ("sliding_window", 32), ("hidden_act", "gelu"), ("attention_bias", True),
])
def test_changed_geometry_rejected(field, value):
    core = _core()
    codec._validate_core(core)
    setattr(core.config, field, value)
    with pytest.raises(ValueError, match="geometry"):
        codec._validate_core(core)


def test_norm_bias_and_transformed_weight_rejection():
    for change in (
        lambda core: setattr(core.layers[0].self_attn, "k_norm", torch.nn.LayerNorm(64)),
        lambda core: setattr(core.layers[0].mlp.down_proj, "bias", torch.ones(512)),
        lambda core: setattr(core.layers[0].post_attention_layernorm, "variance_epsilon", 1e-6),
        lambda core: setattr(core, "parameters", lambda: iter([torch.ones(1).half()])),
    ):
        core = _core()
        change(core)
        with pytest.raises(ValueError):
            codec._validate_core(core)


def test_layerscale_fold_is_fp32_before_half_and_keeps_source():
    torch.manual_seed(9)
    linear = torch.nn.Linear(7, 5, bias=False)
    scale = torch.randn(5)
    weight_before, scale_before = linear.weight.clone(), scale.clone()
    actual = codec._folded_weight(linear, scale)
    assert torch.equal(actual, (weight_before * scale[:, None]).half())
    assert torch.equal(linear.weight, weight_before) and torch.equal(scale, scale_before)
    assert actual.dtype == torch.float16 and actual.data_ptr() != linear.weight.data_ptr()


@pytest.mark.parametrize("capability", [None, lambda: False, lambda: True])
def test_old_native_falls_back_without_allocation(monkeypatch, capability):
    native = SimpleNamespace()
    if capability is not None:
        native.causal_decoder_has_per_head_k_rope = capability
    monkeypatch.setattr(torch.ops, "rpu", native)
    owner = SimpleNamespace(_components={}, _restore=[])
    assert codec.install_codec_transformer(owner, _core()) is False
    assert owner._components == {} and owner._restore == []


@pytest.mark.parametrize("hidden", [None, torch.zeros(1, 0, 1024), torch.zeros(1, 256, 1024),
    torch.zeros(2, 3, 1024), torch.zeros(1, 3, 512), torch.zeros(1, 3, 1024).half(),
    torch.full((1, 3, 1024), float("nan"))])
def test_input_rejection_precedes_native_access(hidden):
    with pytest.raises(ValueError, match="1–255"):
        codec._forward({}, hidden)


@pytest.mark.parametrize("count", [63, 64])
def test_device_boundary_uses_the_shared_planner(monkeypatch, count):
    from rpu_backend.runtime import decoder
    def plan(planner, handle, cache, length):
        assert length == count
        raise RuntimeError("shared planner reached")
    monkeypatch.setattr(decoder, "_text_prefill_execution_plan", plan)
    part = dict(cache=SimpleNamespace(reset_to_position=lambda value: None),
                graph=None, handle=1, planner=None)
    with pytest.raises(RuntimeError, match="shared planner reached"):
        codec._forward(part, torch.zeros(1, count, 1024))
    assert part["last_device"] == "rpu"


@pytest.mark.parametrize("count", [65, 73, 255])
def test_long_codec_plans_explicit_mask_single_chunk(monkeypatch, count):
    from rpu_backend.runtime import decoder
    def resolve(*args):
        assert args == (7, count, 0, False, count, 0, 1, count)
        raise RuntimeError("masked planner reached")
    def plan(real, limit, padding, **kwargs):
        assert (real, limit, padding) == (count, 256, 15)
        assert kwargs["plan_signature"] == (72, count)
        kwargs["resolve_stage_domain"](count)
    monkeypatch.setattr(decoder, "plan_bounded_prefill_execution", plan)
    monkeypatch.setattr(torch.ops, "rpu", SimpleNamespace(causal_decoder_resolve_prefill_stage_domain=resolve))
    part = dict(cache=SimpleNamespace(reset_to_position=lambda value: None),
                graph=None, handle=7, planner=None)
    with pytest.raises(RuntimeError, match="masked planner reached"):
        codec._forward(part, torch.zeros(1, count, 1024))


@pytest.mark.parametrize("length", [65, 73, 80, 255, 256])
def test_sliding_mask_keeps_exactly_72_causal_positions(length):
    with torch.inference_mode():
        mask = codec._sliding_mask(length)
    assert mask.device.type == "cpu" and mask.dtype == torch.float16
    assert mask.is_contiguous() and mask._version >= 0
    for q in range(length):
        for k in range(length):
            assert float(mask[q, k]) == (0 if max(0, q - 71) <= k <= q else float("-inf"))
    assert torch.equal((mask == 0).sum(1), torch.arange(1, length + 1).clamp_max(72))


def test_failed_length_switch_does_not_reuse_previous_signature(monkeypatch):
    from rpu_backend import graph
    from rpu_backend.runtime import decoder

    plan = SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(
        physical_descriptor=[1])), graph_key_words=lambda: [1])
    def bounded(count, *args, **kwargs):
        kwargs["plan_result_sink"](plan)
        return count, count
    monkeypatch.setattr(decoder, "plan_bounded_prefill_execution", bounded)
    monkeypatch.setattr(decoder, "_text_prefill_execution_plan",
                        lambda planner, handle, cache, count: (count, count, plan))
    original_to = torch.Tensor.to
    def fail_upload(self, *args, **kwargs):
        if args and args[0] == "rpu":
            raise RuntimeError("upload failed")
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", fail_upload)
    monkeypatch.setattr(torch.ops, "rpu", SimpleNamespace(linear_with_accumulation=lambda *args: None))
    evicted = []
    monkeypatch.setattr(graph, "GraphSignature", lambda **kwargs: SimpleNamespace(**kwargs))
    signature = graph.GraphSignature(op_id="qwen3_tts_codec_frontend", shapes=[1, 65, 65, 512],
                               dyn_dims=[7, 72, 1, 1], dtypes=[torch.float16])
    part = dict(cache=SimpleNamespace(reset_to_position=lambda value: None),
                graph=SimpleNamespace(evict=evicted.append), handle=7, planner=None,
                signature=signature, mask=codec._sliding_mask(65))
    for count in (64, 65):
        with pytest.raises(RuntimeError, match="upload failed"):
            codec._forward(part, torch.zeros(1, count, 1024))
        assert "signature" not in part
        assert (part["mask"] is None) if count == 64 else part["mask"].shape == (65, 65)
    assert evicted == [signature]

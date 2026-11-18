"""Host checks for the new precision recipe and bounded codec admission."""
from types import SimpleNamespace

import pytest
import torch


def test_cpu_fallback_records_request_length_and_actual_mode():
    from rpu_backend.adapters.qwen3_tts import codec_transformer as codec

    hidden = torch.zeros(1, 64, 1024, dtype=torch.float32)
    seen = []
    part = {"cpu_forward": lambda **kwargs: seen.append(kwargs["inputs_embeds"]) or "cpu-output"}
    with torch.inference_mode():
        assert codec._forward(part, hidden) == "cpu-output"
        assert codec._forward(part, hidden) == "cpu-output"
    assert seen == [hidden, hidden]
    assert part["request_input_lengths"] == {"64": 2}
    assert part["last_device"] == "cpu"
    assert part["last_execution_plan"]["logical_len"] == 64
    assert part["last_execution_plan"]["inference_mode"] is True

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


@pytest.mark.parametrize("count", [64, 72, 255])
def test_long_codec_uses_original_sliding_window_without_native_access(count):
    hidden = torch.randn(1, count, 1024)
    result = object()
    def cpu_forward(*, inputs_embeds):
        assert inputs_embeds is hidden
        return result
    part = {"cpu_forward": cpu_forward}
    assert codec._forward(part, hidden) is result
    assert part["last_device"] == "cpu"


@pytest.mark.parametrize("budget", [4, 6, 8])
def test_codec_installer_and_io_keep_the_component_core_layout(monkeypatch, budget):
    from contextlib import nullcontext
    from rpu_backend import graph
    from rpu_backend.api import cache as cache_module
    from rpu_backend.runtime import weights

    cores = 8 if budget == 8 else 4
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    for name, n, k in (("q_proj", 1024, 512), ("k_proj", 1024, 512),
                        ("v_proj", 1024, 512), ("o_proj", 512, 1024)):
        setattr(layer.self_attn, name, torch.nn.Linear(k, n, bias=False))
    layer.self_attn.q_norm = layer.self_attn.k_norm = torch.nn.Identity()
    layer.mlp = torch.nn.Module()
    for name, n, k in (("gate_proj", 1024, 512), ("up_proj", 1024, 512),
                       ("down_proj", 512, 1024)):
        setattr(layer.mlp, name, torch.nn.Linear(k, n, bias=False))
    layer.attention_type = "sliding_attention"
    layer.input_layernorm = layer.post_attention_layernorm = torch.nn.Module()
    layer.input_layernorm.variance_epsilon = 1e-5
    layer.input_layernorm.weight = torch.ones(512)
    layer.self_attn_layer_scale = layer.mlp_layer_scale = torch.nn.Module()
    layer.self_attn_layer_scale.scale = torch.ones(512)
    core = _core()
    core.layers = [layer] * 8
    core.input_proj = torch.nn.Linear(1024, 512)
    core.output_proj = torch.nn.Linear(512, 1024)
    core.norm.weight = torch.ones(512)
    core.forward = lambda **kw: pytest.fail("unexpected CPU fallback")
    core.rotary_emb = lambda h, p: (torch.ones(1, 64, 64), torch.zeros(1, 64, 64))
    class Owner:
        pass
    owner = Owner()
    owner._components, owner._restore, owner._num_cores = {}, [], budget
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda value, *a, **kw:
        value if a == ("rpu",) else original_to(value, *a, **kw))
    packed = []
    def pack(weight, partition, active_cores):
        packed.append((tuple(weight.shape), partition, active_cores))
        assert active_cores == cores
        return weight
    monkeypatch.setattr(weights, "transform_linear_weight", pack)
    monkeypatch.setattr(graph, "GraphCache", lambda **kw: SimpleNamespace(
        capture=lambda sig: nullcontext(), clear=lambda: None, evict=lambda sig: None,
        cache_invariant_ok=lambda: True))
    class Cache:
        def __init__(self, *args, attn_tp):
            assert args == (8, 1, 64, 16, 64) and attn_tp == cores
            self.k_caches = self.v_caches = []
        def reset_to_position(self, pos):
            assert pos == 0
    monkeypatch.setattr(cache_module, "RPUCache", Cache)
    created, installed, linear_cores = [], [], []
    def create(**kwargs):
        created.append(kwargs)
        return 41
    def bind(*args):
        installed.append(args)
    def linear(hidden, weight, bias, partition, active_cores, acc32):
        linear_cores.append(active_cores)
        assert (partition, active_cores, acc32) == (1, cores, True)
        return torch.nn.functional.linear(hidden, weight, bias)
    native = SimpleNamespace(causal_decoder_create=create,
        causal_decoder_set_weights=bind, causal_decoder_destroy=lambda handle: None,
        causal_decoder_set_tts_codec_mode=lambda handle: None,
        causal_decoder_topology=lambda handle: (cores, cores, cores, cores, 8),
        causal_decoder_set_linear_acc32=lambda handle, acc: None,
        causal_decoder_set_chunk_envelope=lambda *args: None,
        causal_decoder_resolve_decode_stage_descriptor=lambda handle: [41],
        causal_decoder_forward=lambda handle, hidden, *a, **kw: hidden,
        linear_with_accumulation=linear)
    monkeypatch.setattr(torch.ops, "rpu", native)
    try:
        assert codec.install_codec_transformer(owner, core)
        assert created == [{"num_cores": cores, "vocab_size": 0}]
        assert len(installed) == 1 and len(packed) == 58
        assert installed[0][-7:] == (16, 16, 64, 512, 1024, 1e-5, True)
        part = owner._components["codec_transformer"]
        assert part["native_topology"] == (cores, cores, cores, cores, 8)
        result = core.forward(inputs_embeds=torch.zeros(1, 1, 1024))
        assert result.last_hidden_state.shape == (1, 1, 1024)
        assert torch.isfinite(result.last_hidden_state).all() and linear_cores == [cores, cores]
    finally:
        for part in owner._components.values():
            part["resource"].retire()

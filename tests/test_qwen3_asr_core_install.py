"""Exercise resident ASR installation with real tensor shapes and inert device calls."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api import _qwen3_asr_audio as audio_module


def _audio_model(width, layers, heads, output):
    intermediate = 4 * width
    with torch.device("meta"):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        for role in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(layer.self_attn, role, torch.nn.Linear(width, width, dtype=torch.float16))
        layer.fc1 = torch.nn.Linear(width, intermediate, dtype=torch.float16)
        layer.fc2 = torch.nn.Linear(intermediate, width, dtype=torch.float16)
        layer.self_attn_layer_norm = torch.nn.LayerNorm(width, eps=1e-5, dtype=torch.float16)
        layer.final_layer_norm = torch.nn.LayerNorm(width, eps=1e-5, dtype=torch.float16)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([layer] * layers)
        model.ln_post = torch.nn.LayerNorm(width, eps=1e-5, dtype=torch.float16)
        model.proj1 = torch.nn.Linear(width, width, dtype=torch.float16)
        model.proj2 = torch.nn.Linear(width, output, dtype=torch.float16)
        model.positional_embedding = SimpleNamespace(
            positional_embedding=torch.empty(32, width, dtype=torch.float16))
    model.config = SimpleNamespace(d_model=width, encoder_layers=layers,
        encoder_attention_heads=heads, encoder_ffn_dim=intermediate,
        activation_function="gelu", n_window=50, n_window_infer=800, output_dim=output)
    return model


@pytest.mark.parametrize("width,layers,heads,output", [(896, 18, 14, 1024), (1024, 24, 16, 2048)])
@pytest.mark.parametrize("cores", [4, 8])
@pytest.mark.parametrize("topology_mismatch", [False, True])
def test_resident_install_keeps_packing_graph_native_and_cache_cores(
        monkeypatch, width, layers, heads, output, cores, topology_mismatch):
    from rpu_backend import graph

    model = _audio_model(width, layers, heads, output)
    original_shapes = [tuple(parameter.shape) for parameter in model.parameters()]
    events, packed, installed, projections, caches = [], [], [], [], []
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda value, *a, **kw:
        value if a == ("rpu",) else original_to(value, *a, **kw))
    def transform(weight, partition, active_cores):
        assert weight.dtype == torch.float16 and weight.is_contiguous()
        assert active_cores == cores
        events.append("pack")
        packed.append((tuple(weight.shape), partition, active_cores))
        return weight
    monkeypatch.setattr(audio_module, "transform_linear_weight", transform)
    def policy(**kwargs):
        assert kwargs == {"execution_core_count": cores}
        return SimpleNamespace(execution_core_count=cores)
    monkeypatch.setattr(graph, "GraphRuntimePolicy", SimpleNamespace(from_environment=policy))
    def graph_cache(*, max_entries, runtime_policy):
        assert max_entries == 1 and runtime_policy.execution_core_count == cores
        return SimpleNamespace(runtime_policy=runtime_policy,
            clear=lambda: events.append("clear"), cache_invariant_ok=lambda: True)
    monkeypatch.setattr(torch, "rpu", SimpleNamespace(GraphCache=graph_cache), raising=False)
    class Cache:
        def __init__(self, *args, dtype, attn_tp):
            assert args == (layers, 1, 128, 16, 64)
            assert dtype == torch.float16 and attn_tp == cores
            self.attn_tp, self.physical_kv_cores = attn_tp, 8
            caches.append(self)
    monkeypatch.setattr(audio_module, "RPUCache", Cache)
    def create(acc32):
        assert acc32 is True
        events.append("create")
        return 71
    def set_cores(handle, count):
        assert handle == 71 and count == cores
        events.append("cores")
    def set_weights(*args):
        assert args[0] == 71
        assert args[-6:] == (16, 64, width, 4 * width, width, 1e-5)
        installed.append(args)
        events.append("weights")
    def projection(handle, weight, bias):
        assert handle == 71 and tuple(weight.shape) == (output, width)
        assert tuple(bias.shape) == (output,)
        projections.append(weight)
        events.append("projection")
    actual = (1, 8 if cores == 4 else 4, cores, cores, 1, 8, 4 * width, 4 * width) if topology_mismatch else (
        1, cores, cores, cores, 1, 8, 4 * width, 4 * width)
    native = SimpleNamespace(siglip_create=create, siglip_set_execution_core_count=set_cores,
        siglip_set_weights=set_weights, siglip_set_audio_mode=lambda handle: events.append("audio"),
        siglip_set_audio_projection=projection, siglip_get_execution_topology=lambda handle: actual,
        siglip_destroy=lambda handle: events.append("destroy"))
    monkeypatch.setattr(torch.ops, "rpu", native)
    encoder = audio_module.ResidentAudioEncoder(model, {}, num_cores=cores)
    try:
        if topology_mismatch:
            with pytest.raises(RuntimeError, match="native topology"):
                encoder.install()
            assert caches == [] and encoder.positions is None
        else:
            encoder.install()
            assert len(caches) == 1 and encoder.cache.attn_tp == cores
            assert encoder.cache.physical_kv_cores == 8
            # Eight is the maximum number of 100-frame audio blocks, not cores.
            assert tuple(encoder.positions.shape) == (8, 13, width)
        assert events[0] == "create"
        if cores == 4:
            assert events.index("cores") < events.index("pack")
        else:
            assert "cores" not in events
        expected_layer = [((1024, width), 1, cores)] * 3 + [((width, 1024), 0, cores),
            ((4 * width, width), 1, cores), ((width, 4 * width), 0, cores)]
        assert packed == expected_layer * layers + [((width, width), 1, cores), ((output, width), 1, cores)]
        assert len(installed) == len(projections) == 1
        assert all(len(group) == layers for group in installed[0][1:17])
        assert [tuple(parameter.shape) for parameter in model.parameters()] == original_shapes
    finally:
        encoder.close()
    assert events[-2:] == ["clear", "destroy"]
    assert encoder.handle is None and encoder.cache is None and encoder.graph is None

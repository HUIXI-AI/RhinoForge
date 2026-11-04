"""Host-only private 32B Text admission; no allocation of model data or board claim."""
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.qwen3_vl import text


SHAPES = (("self_attn.q_proj", 8192, 5120),
          ("self_attn.k_proj", 1024, 5120),
          ("self_attn.v_proj", 1024, 5120),
          ("self_attn.o_proj", 5120, 8192),
          ("mlp.gate_proj", 25600, 5120),
          ("mlp.up_proj", 25600, 5120),
          ("mlp.down_proj", 5120, 25600))


def model_and_scales(dtype):
    model = torch.nn.Module()
    model.config = NS(model_type="qwen3_vl_text", num_hidden_layers=64,
        hidden_size=5120, intermediate_size=25600, num_attention_heads=64,
        num_key_value_heads=8, head_dim=128)
    model.layers = torch.nn.ModuleList()
    packed = dtype == torch.uint8
    for _ in range(64):
        layer = torch.nn.Module()
        layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
        for role, n, k in SHAPES:
            owner, name = role.split(".")
            projection = torch.nn.Linear(k, n, bias=False, device="meta")
            projection.weight = torch.nn.Parameter(
                torch.empty((n, k // 2 if packed else k), dtype=dtype, device="meta"),
                requires_grad=False)
            projection.register_buffer("weight_scale", torch.empty(
                (32, n * k // 1024) if packed else (n,),
                dtype=torch.float16, device="meta"))
            setattr(getattr(layer, owner), name, projection)
        model.layers.append(layer)
    scales = tuple([layer.get_submodule(role).weight_scale for layer in model.layers]
                   for role, _, _ in SHAPES)
    return model, scales


@pytest.fixture
def shared_installer(monkeypatch):
    calls = []
    def install(model, **kwargs):
        calls.append((model, kwargs))
        return 73
    monkeypatch.setattr(text, "install_mrope_text_decoder_for_rpu", install)
    return calls


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
def test_private_32b_binds_actual_quantized_owners_and_bounded_envelope(dtype, shared_installer):
    model, scales = model_and_scales(dtype)
    execution = {"model": {"num_cores": 8}, "prefill": {"chunk_size": "auto"}}
    assert text.install_qwen3_vl_text_for_rpu(model, scale_lists=scales,
        execution_config=execution, _runtime_quantized_32b=True) == 73
    bound_model, kwargs = shared_installer[0]
    assert bound_model is model and kwargs["scale_lists"] is scales
    assert kwargs["execution_config"] is execution and kwargs["topology"] is None
    assert kwargs["_kv_cache_layer_bank_size"] == 8
    envelope = kwargs["chunk_envelope_for"]
    assert envelope("qwen3_vl_text", 64, 5120) == text.ChunkEnvelope(0, 64)
    with pytest.raises(RuntimeError, match="no certified chunk envelope"):
        envelope("qwen3_vl_text", 63, 5120)
    assert not hasattr(model, "_rpu_decoder_handle")


@pytest.mark.parametrize("fault", [
    "fp16", "mixed", "scale_replaced", "missing_lists", "missing_role", "short_list",
    "scale_dtype", "scale_shape", "scale_stride", "scale_device", "weight_shape",
    "layer_count", "geometry", "float_geometry", "live_config", "legacy", "topology", "cores",
])
def test_private_32b_rejects_before_shared_installer(fault, shared_installer):
    model, scales = model_and_scales(torch.int8)
    kwargs = {"scale_lists": scales, "_runtime_quantized_32b": True}
    projection = model.layers[0].self_attn.q_proj
    if fault in ("fp16", "mixed"):
        projection.weight = torch.nn.Parameter(projection.weight.to(
            torch.float16 if fault == "fp16" else torch.uint8), requires_grad=False)
    elif fault == "scale_replaced":
        projection.weight_scale = projection.weight_scale.clone()
    elif fault == "missing_lists":
        kwargs["scale_lists"] = None
    elif fault == "missing_role":
        del model.layers[-1].mlp.down_proj
    elif fault == "short_list":
        scales[-1].pop()
    elif fault in ("scale_dtype", "scale_shape", "scale_stride", "scale_device"):
        if fault == "scale_dtype":
            scale = torch.empty((8192,), dtype=torch.float32, device="meta")
        elif fault == "scale_shape":
            scale = torch.empty((4096,), dtype=torch.float16, device="meta")
        elif fault == "scale_stride":
            scale = torch.empty((16384,), dtype=torch.float16, device="meta")[::2]
        else:
            scale = torch.empty((8192,), dtype=torch.float16, device="cpu")
        projection.weight_scale = scales[0][0] = scale
    elif fault == "weight_shape":
        projection.weight = torch.nn.Parameter(projection.weight[:, :-1], requires_grad=False)
    elif fault == "layer_count":
        model.layers = model.layers[:-1]
    elif fault == "geometry":
        model.config.num_attention_heads = 40
    elif fault == "float_geometry":
        model.config.num_hidden_layers = 64.0
    elif fault == "live_config":
        kwargs["text_config"] = NS(**vars(model.config))
        model.config.hidden_size = 4096
    elif fault == "legacy":
        kwargs["runtime_align_w8a16"] = True
    elif fault == "topology":
        kwargs["topology"] = NS(num_cores=8)
    else:
        kwargs["execution_config"] = {"model": {"num_cores": 4}}
    with pytest.raises(ValueError, match="runtime quantized 32B"):
        text.install_qwen3_vl_text_for_rpu(model, **kwargs)
    assert shared_installer == []
    assert not hasattr(model, "_rpu_decoder_handle")
    assert not hasattr(model, "_rpu_text_graph_cache")


@pytest.mark.parametrize("flag", [None, 0, 1, "true"])
def test_private_flag_requires_actual_bool(flag, shared_installer):
    with pytest.raises(ValueError, match="must be a bool"):
        text.install_qwen3_vl_text_for_rpu(object(), _runtime_quantized_32b=flag)
    assert shared_installer == []


def test_ordinary_32b_remains_without_envelope_and_existing_profiles_unchanged(shared_installer):
    with pytest.raises(RuntimeError, match="no certified chunk envelope"):
        text.lookup_causal_decoder("qwen3_vl_text", 64, 5120)
    text.install_qwen3_vl_text_for_rpu(NS(config=NS()))
    assert shared_installer[0][1]["_kv_cache_layer_bank_size"] == 1
    envelope = shared_installer[0][1]["chunk_envelope_for"]
    with pytest.raises(RuntimeError, match="no certified chunk envelope"):
        envelope("qwen3_vl_text", 64, 5120)
    assert envelope("qwen3_vl_text", 36, 4096) == text.ChunkEnvelope(0, 128)
    assert envelope("qwen3_vl_text", 36, 2560) == text.ChunkEnvelope(0, 256)

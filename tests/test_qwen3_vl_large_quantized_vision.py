"""Host gates for exact padded 8B/32B W8 Vision; no hardware certificate."""
import gc
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.qwen3_vl import vision
from rpu_backend.quant._common import quantize_linear_per_channel
from test_qwen3_vl_vision_weight_banks import transfers  # noqa: F401


def _config(out_hidden=4096):
    return NS(model_type="qwen3_vl", depth=27, hidden_size=1152,
              intermediate_size=4304, num_heads=16, patch_size=16,
              temporal_patch_size=2, spatial_merge_size=2, in_channels=3,
              num_position_embeddings=2304, deepstack_visual_indexes=[8, 16, 24],
              hidden_act="gelu_pytorch_tanh", out_hidden_size=out_hidden)


def _block():
    def linear(k, n, salt):
        layer = torch.nn.Linear(k, n, dtype=torch.float16)
        with torch.no_grad():
            rows = torch.arange(n)[:, None]
            cols = torch.arange(k)[None, :]
            layer.weight.copy_(((rows * 7 + cols * 3 + salt) % 61 - 30).float() / 256)
            layer.bias.copy_((torch.arange(n) % 13 - 6).float() / 128)
        return layer
    return NS(attn=NS(qkv=linear(1152, 3456, 1), proj=linear(1152, 1152, 2)),
              mlp=NS(linear_fc1=linear(1152, 4304, 3), linear_fc2=linear(4304, 1152, 4)))


def _convert(block, **overrides):
    kwargs = dict(w8a16=True, physical_head_dim=80, physical_intermediate_size=4352)
    vision._convert_vision_block_weights_for_rpu(block, 16, 1152, **(kwargs | overrides))


def _weights(block):
    return (block._rpu_q_w, block._rpu_k_w, block._rpu_v_w,
            block.attn.proj.weight.detach(), block.mlp.linear_fc1.weight.detach(),
            block.mlp.linear_fc2.weight.detach())


def _unswizzle(weight, col):
    """Invert the documented eight-core, 32-byte W8 physical layout."""
    n, k = weight.shape
    if col:
        return weight.view(n // 128, k // 32, 8, 16, 32).permute(2, 0, 3, 1, 4).reshape(n, k)
    return weight.view(n // 16, k // 256, 8, 16, 32).permute(0, 3, 2, 1, 4).reshape(n, k)


def test_lightnav_encoder_range_calibration_keeps_symmetric_w8():
    weight = torch.tensor([
        [-2.0, -0.7, 0.4, 1.8],
        [-0.6, -0.1, 0.3, 0.5],
    ], dtype=torch.float16)
    _, base_scale = vision._quantize_vision_w8_weight(weight)
    quantized, scale = vision._quantize_vision_w8_weight(
        weight, range_scale=1.04)
    assert torch.equal(scale, (base_scale.float() * 1.04).half())
    assert torch.equal(
        quantized,
        torch.round(weight.float() / scale.float()[:, None])
        .clamp(-127, 127).to(torch.int8),
    )
    for value in (1, True, 1.03):
        with pytest.raises(ValueError, match="range_scale"):
            vision._quantize_vision_w8_weight(weight, range_scale=value)


def test_real_w8_padding_quantization_swizzle_and_bank_preserve_logical_math(transfers):
    block = _block()
    raw = (*block.attn.qkv.weight.detach().chunk(3), block.attn.proj.weight.detach(),
           block.mlp.linear_fc1.weight.detach(), block.mlp.linear_fc2.weight.detach())
    reference = tuple(quantize_linear_per_channel(w) for w in raw)
    biases = (*block.attn.qkv.bias.detach().chunk(3), block.mlp.linear_fc1.bias.detach())
    _convert(block)
    weights = _weights(block)
    roles = ("q", "k", "v", "o", "fc1", "fc2")
    assert [tuple(w.shape) for w in weights] == [(1280, 1152)] * 3 + [
        (1152, 1280), (4352, 1152), (1152, 4352)]
    for index, (role, weight, (expected, expected_scale)) in enumerate(zip(roles, weights, reference)):
        assert weight.dtype == torch.int8 and weight.is_contiguous()
        scale = getattr(block, f"_rpu_{role}_ws")
        assert scale.dtype == torch.float16 and scale.shape == (weight.shape[0],)
        assert bool(torch.isfinite(scale).all()) and bool((scale > 0).all())
        logical = _unswizzle(weight, col=role in ("q", "k", "v", "fc1"))
        logical_scale = scale
        if index < 3:
            physical = logical.view(16, 80, 1152)
            assert torch.count_nonzero(physical[:, 72:]) == 0
            logical = physical[:, :72].reshape(1152, 1152)
            logical_scale = scale.view(16, 80)[:, :72].reshape(1152)
            bias = getattr(block, f"_rpu_{role}_b").view(16, 80)
            assert torch.count_nonzero(bias[:, 72:]) == 0
            assert torch.equal(bias[:, :72].reshape(1152), biases[index])
        elif role == "o":
            physical = logical.view(1152, 16, 80)
            assert torch.count_nonzero(physical[:, :, 72:]) == 0
            logical = physical[:, :, :72].reshape(1152, 1152)
        elif role == "fc1":
            assert torch.count_nonzero(logical[4304:]) == 0
            assert torch.count_nonzero(block._rpu_fc1_b[4304:]) == 0
            assert torch.equal(block._rpu_fc1_b[:4304], biases[-1])
            logical, logical_scale = logical[:4304], scale[:4304]
        else:
            assert torch.count_nonzero(logical[:, 4304:]) == 0
            logical = logical[:, :4304]
        # Per-channel quantization of padding must not alter any real value/scale.
        assert torch.equal(logical, expected) and torch.equal(logical_scale, expected_scale)
        assert torch.equal(logical.float() * logical_scale.float()[:, None],
                           expected.float() * expected_scale.float()[:, None])

    snapshots = tuple(w.clone() for w in weights)
    identities = tuple(w.data_ptr() for w in weights)
    _convert(block)
    assert tuple(w.data_ptr() for w in _weights(block)) == identities
    assert all(torch.equal(a, b) for a, b in zip(_weights(block), snapshots))
    with pytest.raises(RuntimeError, match="already converted"):
        _convert(block, w8a16=False)

    views = vision._vision_block_weights_to_rpu(weights, w8a16=True, pack_weights=True)
    assert transfers == [((15925248,), torch.int8)]
    assert len({view.untyped_storage().data_ptr() for view in views}) == 1
    offset = 0
    for view, expected in zip(views, snapshots):
        assert torch.equal(view, expected) and view.stride() == expected.stride()
        assert view.storage_offset() == offset and offset % 256 == 0
        offset += view.numel()
    assert offset == views[0].untyped_storage().nbytes() == 15925248
    retained = views[-1]
    del views, weights, block
    gc.collect()
    assert torch.equal(retained, snapshots[-1])


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_partial_padded_w8_conversion_cannot_be_retried(monkeypatch, error):
    block = _block()
    real = vision.tp_row_swizzle_mc_weight
    calls = []
    def fail_fc2(weight, *args, **kwargs):
        calls.append(tuple(weight.shape))
        if len(calls) == 2:
            raise error("interrupted padded FC2 swizzle")
        return real(weight, *args, **kwargs)
    monkeypatch.setattr(vision, "tp_row_swizzle_mc_weight", fail_fc2)
    with pytest.raises(error, match="interrupted padded FC2"):
        _convert(block)
    assert block.attn.proj.weight.dtype == torch.int8
    assert block.mlp.linear_fc1.weight.dtype == torch.int8
    assert block._rpu_qwen3vl_vision_weights_converted == vision._VISION_CONVERSION_IN_PROGRESS
    with pytest.raises(RuntimeError, match="partial prior weight conversion"):
        _convert(block)
    assert calls == [(1152, 1280), (1152, 4352)]


@pytest.mark.parametrize("out_hidden", [4096, 5120])
@pytest.mark.parametrize("fault", [None, "flag_type", "missing_flag", "fp16", "cores", "legacy8b", "legacy32b",
                                  "depth", "heads", "width", "fractional_width", "mlp", "type", "channels",
                                  "positions", "taps", "fractional_tap", "output", "blocks"])
def test_large_w8_installer_admission_precedes_any_mutation(monkeypatch, out_hidden, fault):
    cfg = _config(out_hidden)
    visual = NS(config=cfg, blocks=[None] * 27)
    kwargs = dict(w8a16=True, _allow_runtime_quantized_large=True)
    if fault == "flag_type": kwargs["_allow_runtime_quantized_large"] = 1
    elif fault == "missing_flag": kwargs["_allow_runtime_quantized_large"] = False
    elif fault == "fp16": kwargs["w8a16"] = False
    elif fault == "cores": kwargs["num_cores"] = 4
    elif fault == "legacy8b": kwargs["_allow_padded_8b"] = True
    elif fault == "legacy32b": kwargs["_allow_graph_blocked_32b"] = True
    elif fault == "blocks": visual.blocks.pop()
    elif fault is not None:
        field, value = {"depth": ("depth", 26), "heads": ("num_heads", 18),
                        "width": ("hidden_size", 1168), "fractional_width": ("hidden_size", 1152.5),
                        "mlp": ("intermediate_size", 4352),
                        "type": ("model_type", "qwen3_5_vl"), "channels": ("in_channels", 4),
                        "positions": ("num_position_embeddings", 2305),
                        "taps": ("deepstack_visual_indexes", [8, 16, 25]),
                        "fractional_tap": ("deepstack_visual_indexes", [8, 16.5, 24]),
                        "output": ("out_hidden_size", 4097)}[fault]
        setattr(cfg, field, value)
    before = dict(vars(visual))
    calls = []
    monkeypatch.setattr(vision, "_install_qwen3_vl_vision_for_rpu_impl",
                        lambda *args, **kw: calls.append(kw) or 17)
    if fault is None:
        assert vision.install_qwen3_vl_vision_for_rpu(visual, **kwargs) == 17
        assert len(calls) == 1 and calls[0]["_allow_runtime_quantized_large"]
        assert calls[0]["w8a16"] and not calls[0]["_allow_padded_8b"]
    else:
        with pytest.raises(TypeError if fault == "flag_type" else ValueError):
            vision.install_qwen3_vl_vision_for_rpu(visual, **kwargs)
        assert calls == [] and vars(visual) == before


def test_old_private_fp16_flag_still_rejects_w8(monkeypatch):
    visual = NS(config=_config(), blocks=[None] * 27)
    monkeypatch.setattr(vision, "_install_qwen3_vl_vision_for_rpu_impl",
                        lambda *args, **kwargs: pytest.fail("legacy FP16 flag admitted W8"))
    with pytest.raises(ValueError, match="FP16"):
        vision.install_qwen3_vl_vision_for_rpu(visual, w8a16=True, _allow_padded_8b=True)


@pytest.mark.parametrize("out_hidden", [4096, 5120])
def test_actual_large_w8_install_banks_each_block_without_default_pairing(monkeypatch, transfers, out_hidden):
    """Only device/native boundaries are doubled; the actual installer owns routing."""
    model = torch.nn.Module()
    model.config = _config(out_hidden)
    # Real weight conversion and bank contents are covered above. Tiny preconverted
    # operands here isolate installer lifecycle/routing without allocating 27 towers.
    def tiny_block():
        block = NS(attn=NS(proj=torch.nn.Linear(32, 16, dtype=torch.float16)),
                   mlp=NS(linear_fc1=torch.nn.Linear(32, 16, dtype=torch.float16),
                          linear_fc2=torch.nn.Linear(32, 16, dtype=torch.float16)),
                   norm1=torch.nn.LayerNorm(16), norm2=torch.nn.LayerNorm(16),
                   _rpu_qwen3vl_vision_weights_converted=True)
        for role in ("q", "k", "v", "o", "fc1", "fc2"):
            setattr(block, f"_rpu_{role}_ws", torch.ones(16).half())
            setattr(block, f"_rpu_{role}_b", torch.zeros(16).half())
            if role in ("q", "k", "v"):
                setattr(block, f"_rpu_{role}_w", torch.ones(16, 32, dtype=torch.int8))
        for layer in (block.attn.proj, block.mlp.linear_fc1, block.mlp.linear_fc2):
            layer.weight = torch.nn.Parameter(torch.ones(16, 32, dtype=torch.int8), requires_grad=False)
        return block
    model.blocks = [tiny_block() for _ in range(27)]
    model.patch_embed = NS(proj=torch.nn.Conv3d(1, 1, 1))
    model.pos_embed = torch.nn.Embedding(1, 1)
    model.merger = torch.nn.Linear(1, 1)
    model.deepstack_merger_list = torch.nn.ModuleList()
    conversion = []
    monkeypatch.setattr(vision, "_convert_vision_block_weights_for_rpu",
                        lambda *args, **kw: conversion.append(kw))
    cache_calls = []
    monkeypatch.setattr(vision, "_make_dummy_vision_kv_caches",
                        lambda *a, **kw: cache_calls.append((a, kw)) or NS(max_seq_len=1200))
    class GraphCache:
        def clear(self): pass
        def cache_invariant_ok(self): return True
    monkeypatch.setattr(vision.rpu_backend.graph, "GraphCache", GraphCache)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_create", lambda: 17, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_destroy", lambda handle: None, raising=False)
    weights = []
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_set_weights_w8a16",
                        lambda *a: weights.append(a), raising=False)
    for name in ("set_chunk_envelope", "set_rope_route", "set_chunk_size"):
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_" + name, lambda *a: None, raising=False)
    for name in ("set_w8_compact_encoder", "set_n1200_tm160", "set_fast_replay"):
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_" + name,
                            lambda *a: pytest.fail("large installer enabled a small-profile schedule"), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_position_idx_keepalive",
                        lambda h: torch.zeros(4096, 2, dtype=torch.int16), raising=False)
    monkeypatch.delenv("RPU_QWEN3VL_VISION_BATCH", raising=False)
    vision.install_qwen3_vl_vision_for_rpu(
        model, w8a16=True, _allow_runtime_quantized_large=True, max_seq_len=1200)
    try:
        assert len(conversion) == 27
        assert all(x["physical_head_dim"] == 80 and x["physical_intermediate_size"] == 4352
                   and x["w8a16"] for x in conversion)
        assert cache_calls == [((27, 1200, 16, 80), {})]
        assert weights[0][17:21] == (16, 80, 1152, 4352)
        bank_transfers = [x for x in transfers if x[1] == torch.int8]
        assert bank_transfers == [((3072,), torch.int8)] * 27
        assert not model._rpu_vision_default_pair_n256
        assert model._rpu_vision_freq_cos.shape[-1] == 18
    finally:
        model._rpu_vision_retirement_state.retire()

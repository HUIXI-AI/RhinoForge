"""Admitted 2B/4B FP16 and W8 Vision banks preserve bytes and owners."""

import gc
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_vl import vision


@pytest.fixture
def transfers(monkeypatch):
    calls = []
    original = torch.Tensor.to
    def to(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            calls.append((tuple(tensor.shape), tensor.dtype))
            kwargs["device"] = "cpu"
            return original(tensor, *args, **kwargs).clone()
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to)
    return calls


@pytest.mark.parametrize("shapes", [
    [(1024, 1024)] * 4 + [(4096, 1024), (1024, 4096)],
    [(3, 17), (7, 13), (2, 5), (5, 19), (11, 23), (13, 29)],
])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16])
def test_weight_bank_preserves_bytes_shapes_strides_and_aligned_views(transfers, shapes, dtype):
    generator = torch.Generator().manual_seed(41)
    weights = tuple(torch.randint(-128, 128, shape, generator=generator).to(dtype)
                    for shape in shapes)
    originals = tuple(weight.clone() for weight in weights)
    views = vision._vision_block_weights_to_rpu(weights, w8a16=dtype == torch.int8, pack_weights=True)
    assert len(transfers) == 1 and len(transfers[0][0]) == 1
    assert transfers[0][1] == dtype
    base = views[0].untyped_storage().data_ptr()
    end = 0
    for view, original in zip(views, originals, strict=True):
        offset = (end + 255) // 256 * 256
        assert torch.equal(view, original)
        assert tuple(view.shape) == tuple(original.shape)
        assert view.stride() == (original.shape[1], 1) and view.is_contiguous()
        assert view.storage_offset() * view.element_size() == offset and offset % 256 == 0
        assert view.untyped_storage().data_ptr() == base
        assert view.data_ptr() == base + offset
        assert view.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
        end = offset + view.numel() * view.element_size()
    assert views[0].untyped_storage().nbytes() == (end + 255) // 256 * 256
    assert all(torch.equal(weight, original) for weight, original in zip(weights, originals))
    # There is no explicit bank attr: any surviving tensor view owns its storage.
    retained = views[-1]
    del views, weights
    gc.collect()
    assert torch.equal(retained, originals[-1])


def _config():
    return SimpleNamespace(depth=24, hidden_size=1024, intermediate_size=4096,
                           num_heads=16, spatial_merge_size=2, out_hidden_size=2560)


@pytest.mark.parametrize("field,value", [
    ("depth", 23), ("hidden_size", 1152), ("intermediate_size", 4352),
    ("num_heads", 32), ("spatial_merge_size", 1), ("out_hidden_size", 2049),
    ("out_hidden_size", 4096), ("w8a16", False), ("num_cores", 4), ("num_layers", 23),
])
def test_neighboring_profiles_keep_independent_transfers(transfers, field, value):
    cfg = _config()
    arguments = dict(w8a16=True, num_cores=8, num_layers=24)
    assert vision._use_w8_vision_weight_banks(cfg, **arguments)
    if field in arguments:
        arguments[field] = value
    else:
        setattr(cfg, field, value)
    pack = vision._use_w8_vision_weight_banks(cfg, **arguments)
    assert not pack
    dtype = torch.int8 if arguments["w8a16"] else torch.float32
    weights = tuple(torch.full((16, 32), i, dtype=dtype) for i in range(6))
    result = vision._vision_block_weights_to_rpu(
        weights, w8a16=arguments["w8a16"], pack_weights=pack)
    assert len(transfers) == 6
    assert len({x.untyped_storage().data_ptr() for x in result}) == 6
    expected_dtype = torch.int8 if arguments["w8a16"] else torch.float16
    assert all(x.dtype == expected_dtype for x in result)
    assert all(torch.equal(actual, source.to(expected_dtype))
               for actual, source in zip(result, weights))


@pytest.mark.parametrize("out_hidden", [2048, 2560])
def test_fp16_storage_admission_does_not_enable_w8_default_pairing(out_hidden):
    cfg = SimpleNamespace(**vars(_config()), model_type="qwen3_vl", patch_size=16,
        temporal_patch_size=2, in_channels=3, num_position_embeddings=2304,
        hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[5, 11, 17])
    cfg.out_hidden_size = out_hidden
    args = dict(w8a16=False, num_cores=8, num_layers=24)
    assert vision._use_fp16_vision_weight_banks(cfg, **args)
    assert not vision._use_w8_vision_weight_banks(cfg, **args)
    for override in ({"w8a16": True}, {"num_cores": 4}, {"num_layers": 23}):
        assert not vision._use_fp16_vision_weight_banks(cfg, **(args | override))


@pytest.mark.parametrize("field,value", [
    ("model_type", "qwen3_5_vl"), ("depth", 23), ("hidden_size", 1152),
    ("intermediate_size", 4304), ("num_heads", 32), ("patch_size", 14),
    ("temporal_patch_size", 1), ("spatial_merge_size", 1),
    ("out_hidden_size", 4096), ("in_channels", 4),
    ("num_position_embeddings", 2305), ("hidden_act", "gelu"),
    ("deepstack_visual_indexes", [5, 11, 23]),
])
def test_fp16_neighboring_geometry_rejected_before_installer_mutation(transfers, monkeypatch, field, value):
    cfg = SimpleNamespace(**vars(_config()), model_type="qwen3_vl", patch_size=16,
        temporal_patch_size=2, in_channels=3, num_position_embeddings=2304,
        hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[5, 11, 17])
    setattr(cfg, field, value)
    visual = SimpleNamespace(config=cfg, blocks=[None] * 24)
    before = dict(vars(visual))
    monkeypatch.setattr(vision, "_install_qwen3_vl_vision_for_rpu_impl",
                        lambda *a, **k: pytest.fail("installation started before geometry rejection"))
    with pytest.raises(ValueError, match="exact eight-core 2B/4B FP16 tower"):
        vision.install_qwen3_vl_vision_for_rpu(visual, _pack_fp16_block_weights=True)
    assert vars(visual) == before and transfers == []


@pytest.mark.parametrize("fault", ["dtype", "noncontiguous", "count"])
def test_bad_fp16_bank_rejects_before_transfer(transfers, fault):
    weights = [torch.ones(16, 32, dtype=torch.float16) for _ in range(6)]
    if fault == "dtype":
        weights[-1] = weights[-1].float()
    elif fault == "noncontiguous":
        weights[-1] = weights[-1].t()
    else:
        weights.pop()
    with pytest.raises(ValueError, match="six contiguous CPU"):
        vision._vision_block_weights_to_rpu(weights, w8a16=False, pack_weights=True)
    assert transfers == []


@pytest.mark.parametrize("out_hidden", [2048, 2560])
def test_runtime_w8_vision_bank_applies_to_both_exact_model_widths(transfers, out_hidden):
    cfg = _config()
    cfg.out_hidden_size = out_hidden
    pack = vision._use_w8_vision_weight_banks(cfg, w8a16=True, num_cores=8, num_layers=24)
    weights = tuple(torch.full((16, 64), i, dtype=torch.int8) for i in range(6))
    output = vision._vision_block_weights_to_rpu(weights, w8a16=True, pack_weights=pack)
    assert len(transfers) == 1
    assert all(torch.equal(a, b) for a, b in zip(output, weights))

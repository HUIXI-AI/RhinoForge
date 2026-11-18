"""CPU checks for Pi05 N32K32 packing and rejection of old packed owners.

The package's device preflight keeps native loading disabled on a host without
RPU devices. These checks exercise weight preparation, not kernel execution.
"""

import pytest
import torch
import torch.nn as nn

N, K = 128, 512


def _int8_linear(n=N, k=K):
    mod = nn.Linear(k, n, bias=False)
    vals = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    mod.weight = nn.Parameter(vals, requires_grad=False)
    mod.register_buffer("weight_scale", torch.rand(n, dtype=torch.float16))
    return mod


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = _int8_linear()
        self.self_attn.k_proj = _int8_linear()
        self.self_attn.v_proj = _int8_linear()
        self.self_attn.o_proj = _int8_linear()
        self.mlp = nn.Module()
        self.mlp.gate_proj = _int8_linear()
        self.mlp.up_proj = _int8_linear()
        self.mlp.down_proj = _int8_linear()


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


def test_int4_child_names_excludes_kv_short_suffixes():
    from rpu_backend.adapters.pi05.w4pack import int4_child_names
    cfg = {"method": "w4a16", "mixed_int8_suffixes": ["k_proj.weight", "v_proj.weight"]}
    names = int4_child_names(cfg)
    assert names == {"q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert "k_proj" not in names and "v_proj" not in names


def test_int4_child_names_excludes_kv_full_suffixes():
    from rpu_backend.adapters.pi05.w4pack import int4_child_names
    cfg = {
        "method": "w4a16",
        "mixed_int8_suffixes": [
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        ],
    }
    names = int4_child_names(cfg)
    assert names == {"q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def test_int4_child_names_empty_for_fake():
    from rpu_backend.adapters.pi05.w4pack import int4_child_names
    assert int4_child_names({"method": "w4a16_fake_int8"}) == set()


def test_pack_helper_preserves_logical_dimensions_and_padded_physical_owners():
    from rpu_backend.adapters.pi05.w4pack import pack_int4_projections_inplace
    torch.manual_seed(0)
    decoder = _Decoder()
    int4 = {"q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    n_packed = pack_int4_projections_inplace(decoder, int4, attn_num_cores=8)
    assert n_packed == 5

    layer = decoder.layers[0]
    for parent_name, proj_name in (
        ("self_attn", "q_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
    ):
        module = getattr(getattr(layer, parent_name), proj_name)
        w = module.weight
        assert w.dtype == torch.uint8
        # TP8 column shards have 16 logical N lanes and 32 physical lanes.
        # Row shards retain all 128 N lanes. Logical Linear dimensions stay put.
        physical_n = 128 if proj_name in {"o_proj", "down_proj"} else 256
        assert tuple(w.shape) == (physical_n, K // 2)
        assert (module.out_features, module.in_features) == (N, K)
        assert module._rpu_w4_layout == "n32k32"
        assert w.is_contiguous()
        assert module.weight_scale.dtype == torch.float16
        assert module.weight_scale.dim() == 2
        assert module.weight_scale.size(0) == 32
        assert module.weight_scale.is_contiguous()

    for proj_name in ("k_proj", "v_proj"):
        w = getattr(layer.self_attn, proj_name).weight
        assert w.dtype == torch.int8
        assert tuple(w.shape) == (N, K)
        assert tuple(getattr(layer.self_attn, proj_name).weight_scale.shape) == (N,)

    owners = [(module, module.weight, module.weight_scale)
              for parent in (layer.self_attn, layer.mlp)
              for name, module in parent.named_children() if name in int4]
    assert pack_int4_projections_inplace(decoder, int4, attn_num_cores=8) == 0
    for module, weight, scale in owners:
        assert module.weight is weight and module.weight_scale is scale


@pytest.mark.parametrize("layout", [None, "n16k64"])
def test_prepacked_owner_without_current_layout_is_rejected(layout):
    from rpu_backend.adapters.pi05.w4pack import pack_int4_projections_inplace

    decoder = _Decoder()
    projection = decoder.layers[0].self_attn.q_proj
    projection.weight = nn.Parameter(torch.zeros(N, K // 2, dtype=torch.uint8),
                                     requires_grad=False)
    projection.weight_scale = torch.ones(32, N, dtype=torch.float16)
    if layout is not None:
        projection._rpu_w4_layout = layout
    with pytest.raises(RuntimeError, match="installed N32K32"):
        pack_int4_projections_inplace(decoder, {"q_proj"}, attn_num_cores=8)

"""Rhino full-W8 vision conversions and actual native installation arguments."""
from types import SimpleNamespace as NS

import pytest
import torch

import rpu_backend
from rpu_backend.adapters.rhinovla import vision
from rpu_backend.quant._common import quantize_linear_per_channel
from test_qwen3_vl_vision_weight_banks import transfers  # noqa: F401


def _block():
    return NS(
        attn=NS(qkv=torch.nn.Linear(256, 768), proj=torch.nn.Linear(256, 256)),
        mlp=NS(linear_fc1=torch.nn.Linear(256, 512), linear_fc2=torch.nn.Linear(512, 256)),
        norm1=torch.nn.LayerNorm(256), norm2=torch.nn.LayerNorm(256),
    )


def _merger():
    return NS(hidden_size=1024, use_postshuffle_norm=False,
              norm=torch.nn.LayerNorm(256),
              linear_fc1=torch.nn.Linear(1024, 1024),
              linear_fc2=torch.nn.Linear(1024, 256))


def _unswizzle(weight, *, col):
    n, k = weight.shape
    if col:
        return weight.view(n // 128, k // 32, 8, 16, 32).permute(2, 0, 3, 1, 4).reshape(n, k)
    return weight.view(n // 16, k // 256, 8, 16, 32).permute(0, 3, 2, 1, 4).reshape(n, k)


def test_block_w8_preserves_quantized_math_and_refuses_precision_switch():
    block = _block()
    raw = (*block.attn.qkv.weight.detach().chunk(3), block.attn.proj.weight.detach(),
           block.mlp.linear_fc1.weight.detach(), block.mlp.linear_fc2.weight.detach())
    expected = [quantize_linear_per_channel(w) for w in raw]
    vision._convert_vision_block_weights_for_rpu(block, 4, 256, w8a16=True)
    weights = (block._rpu_q_w, block._rpu_k_w, block._rpu_v_w,
               block.attn.proj.weight, block.mlp.linear_fc1.weight, block.mlp.linear_fc2.weight)
    for name, weight, (q, scale) in zip(("q", "k", "v", "o", "fc1", "fc2"), weights, expected):
        assert weight.dtype == torch.int8 and weight.is_contiguous()
        assert torch.equal(_unswizzle(weight, col=name in ("q", "k", "v", "fc1")), q)
        assert torch.equal(getattr(block, f"_rpu_{name}_ws"), scale)
    before = tuple(w.data_ptr() for w in weights)
    vision._convert_vision_block_weights_for_rpu(block, 4, 256, w8a16=True)
    assert tuple(w.data_ptr() for w in weights) == before
    with pytest.raises(RuntimeError, match="precision cannot change"):
        vision._convert_vision_block_weights_for_rpu(block, 4, 256)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 100000.0])
def test_invalid_block_fails_before_any_swizzle(value):
    block = _block()
    with torch.no_grad():
        block.mlp.linear_fc2.weight[0, 0] = value
    pointers = (block.attn.proj.weight.data_ptr(), block.mlp.linear_fc1.weight.data_ptr())
    with pytest.raises(ValueError, match="finite in FP16"):
        vision._convert_vision_block_weights_for_rpu(block, 4, 256, w8a16=True)
    assert not hasattr(block, "_rpu_qwen3vl_vision_weights_converted")
    assert pointers == (block.attn.proj.weight.data_ptr(), block.mlp.linear_fc1.weight.data_ptr())


def test_w8_eager_merger_uses_integer_payloads_and_matching_scales(transfers, monkeypatch):
    merger = _merger()
    raw = [merger.linear_fc1.weight.detach().clone(), merger.linear_fc2.weight.detach().clone()]
    expected = [quantize_linear_per_channel(w) for w in raw]
    vision._prepare_vision_patch_merger_for_rpu(merger, w8a16=True)
    calls = []
    def linear(value, weight, scale, bias):
        assert weight.dtype == torch.int8 and scale.dtype == torch.float16
        calls.append((weight, scale))
        dequantized = (_unswizzle(weight, col=True).float() * scale.float()[:, None]).half()
        return torch.nn.functional.linear(value, dequantized, bias)
    monkeypatch.setattr(torch.ops.rpu, "linear_w8a16", linear, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    source = torch.randn(8, 256)
    actual = vision._run_vision_patch_merger_on_rpu(merger, source)
    reference = merger.norm(source).view(-1, 1024).half()
    for index, ((q, scale), layer) in enumerate(zip(expected, (merger.linear_fc1, merger.linear_fc2))):
        reference = torch.nn.functional.linear(reference, (q.float() * scale.float()[:, None]).half(), layer.bias.half())
        if index == 0:
            reference = torch.nn.functional.gelu(reference)
    assert torch.equal(actual, reference.float()) and len(calls) == 2
    assert all(torch.equal(a, b) for a, b in zip(raw, (merger.linear_fc1.weight, merger.linear_fc2.weight)))
    with pytest.raises(RuntimeError, match="precision cannot change"):
        vision._prepare_vision_patch_merger_for_rpu(merger)


@pytest.mark.parametrize("patch,mergers", [(False, False), (False, True), (True, False)])
def test_full_w8_rejects_cpu_projection_fallback_before_install(patch, mergers):
    with pytest.raises(ValueError, match="requires device patch embedding and mergers"):
        vision.install_qwen3_vl_vision_for_rpu(object(), w8a16=True,
            rpu_patch_embed=patch, rpu_mergers=mergers)


class _Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = NS(num_heads=4, hidden_size=256, intermediate_size=512,
                         spatial_merge_size=2, deepstack_visual_indexes=[0, 1, 2],
                         hidden_act="gelu_pytorch_tanh")
        self.blocks = [_block()]
        self.patch_embed = NS(proj=torch.nn.Conv3d(1, 256, (1, 16, 16)))
        self.pos_embed = torch.nn.Embedding(16, 256)
        self.merger = _merger()
        self.deepstack_merger_list = [_merger() for _ in range(3)]


def test_full_w8_installs_actual_encoder_patch_and_all_merger_owners(transfers, monkeypatch):
    from rpu_backend.api import _execution
    from rpu_backend.graph import lazy_init_guard
    monkeypatch.setattr(_execution, "_require_execution_process_safe", lambda: None)
    monkeypatch.setattr(lazy_init_guard, "_verify_lazy_init", lambda _: None)
    monkeypatch.setattr(vision, "build_vision_rope_tables", lambda *a, **kw: (torch.zeros(16, 16), torch.zeros(16, 16)))
    monkeypatch.setattr(vision, "_make_dummy_vision_kv_caches", lambda *a: NS(max_seq_len=1024))
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", lambda: NS(clear=lambda: None, cache_invariant_ok=lambda: True))
    events = []
    native = {}
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_create", lambda: events.append("create") or 77, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_destroy", lambda _: events.append("destroy"), raising=False)
    for method in ("set_weights_w8a16", "set_merger_weights", "set_deepstack_merger_weights"):
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_" + method,
                            lambda *a, method=method: native.__setitem__(method, a), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_set_weights",
                        lambda *a: pytest.fail("FP16 setter used for full W8"), raising=False)
    for method in ("set_chunk_envelope", "set_rope_route", "set_chunk_size"):
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_" + method, lambda *a: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_position_idx_keepalive", lambda _: torch.zeros(1024, 2), raising=False)
    model = _Vision()
    patch_raw = vision._fold_conv3d_to_linear_weight(model.patch_embed.proj.weight.detach())
    try:
        assert vision.install_qwen3_vl_vision_for_rpu(model, w8a16=True, rpu_patch_embed=True, rpu_mergers=True) == 77
        vision.register_qwen3vl_vision_fused_merger(model, w8a16=True)
        args = native["set_weights_w8a16"]
        assert len(args) == 29
        assert all(w.dtype == torch.int8 for group in args[1:7] for w in group)
        assert all(s.dtype == torch.float16 for group in args[-6:] for s in group)
        assert model._rpu_vision_projection_weights == args[1:7]
        assert model._rpu_vision_projection_scales == args[-6:]
        q, scale = quantize_linear_per_channel(patch_raw)
        assert torch.equal(_unswizzle(model._rpu_vision_patch_embed_w_rpu, col=True), q)
        assert torch.equal(model._rpu_vision_patch_embed_scale, scale)
        patch_merger = native["set_merger_weights"]
        assert patch_merger[3].dtype == patch_merger[5].dtype == torch.int8
        assert patch_merger[-2].dtype == patch_merger[-1].dtype == torch.float16
        ds = native["set_deepstack_merger_weights"]
        assert len(ds[3]) == len(ds[5]) == len(ds[-2]) == len(ds[-1]) == 3
        for index, merger in enumerate((model.merger, *model.deepstack_merger_list)):
            weight = patch_merger[5] if index == 0 else ds[5][index - 1]
            q, scale = quantize_linear_per_channel(merger.linear_fc2.weight)
            assert weight is merger._rpu_merger_fc2_fused_w_rpu
            assert torch.equal(_unswizzle(weight, col=False), q)
            assert torch.equal(merger._rpu_merger_fc2_scale_rpu, scale)
        with pytest.raises(RuntimeError, match="precision cannot change"):
            vision.install_qwen3_vl_vision_for_rpu(model)
        assert events == ["create"]
        model.blocks[0]._rpu_q_ws[0] = float("nan")
        with pytest.raises(ValueError, match="finite positive FP16 scales"):
            vision.install_qwen3_vl_vision_for_rpu(model, w8a16=True, rpu_patch_embed=True, rpu_mergers=True)
        assert events == ["create"] and model._rpu_vision_handle == 77
    finally:
        finalizer = getattr(model, "_rpu_vision_handle_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer()


@pytest.mark.parametrize("cores", [4, 8])
def test_four_core_mergers_run_batched_with_rpu_output_and_no_raw_copies(transfers, monkeypatch, cores):
    from rpu_backend.api import _execution
    from rpu_backend.graph import lazy_init_guard
    monkeypatch.setattr(_execution, "_require_execution_process_safe", lambda: None)
    monkeypatch.setattr(lazy_init_guard, "_verify_lazy_init", lambda _: None)
    monkeypatch.setattr(vision, "build_vision_rope_tables", lambda *a, **kw: (torch.zeros(16, 16), torch.zeros(16, 16)))
    monkeypatch.setattr(vision, "_make_dummy_vision_kv_caches", lambda *a, **kw: NS(max_seq_len=1024))
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", lambda **kw: NS(clear=lambda: None, cache_invariant_ok=lambda: True))
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_create", lambda *a: 77, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_destroy", lambda _: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_topology", lambda _: (4, 4, 4, 8), raising=False)
    for method in ("set_weights", "set_chunk_envelope", "set_rope_route", "set_chunk_size"):
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_" + method, lambda *a: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_position_idx_keepalive", lambda _: torch.zeros(1024, 2), raising=False)
    # Explicit opposite selections: only the eight-core install follows them.
    monkeypatch.setenv("RPU_RHINOVLA_VISION_BATCH_MERGERS", "0")
    monkeypatch.setenv("RPU_RHINOVLA_VISION_RPU_MERGER_OUTPUT_RPU", "0")
    monkeypatch.setenv("RPU_RHINOVLA_VISION_SKIP_RAW_SNAPSHOTS", "0")
    model = _Vision()
    try:
        assert vision.install_qwen3_vl_vision_for_rpu(
            model, rpu_patch_embed=True, rpu_mergers=True, num_cores=cores) == 77
        reduced = cores != 8
        assert model._rpu_vision_batch_mergers is reduced
        assert model._rpu_vision_merger_output_rpu is reduced
        assert model._rpu_vision_collect_raw_snapshots is not reduced
        assert not model._rpu_vision_merger_norm_on_rpu
    finally:
        finalizer = getattr(model, "_rpu_vision_handle_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer()

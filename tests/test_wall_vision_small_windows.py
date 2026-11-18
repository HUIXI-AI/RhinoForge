"""CPU checks for the controlled small-image window schedule and mask lifetime."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "python/rpu_backend/adapters/wall_oss/vision.py"


def _helpers():
    # Load actual pure CPU layout helpers without importing the device backend.
    tree = ast.parse(ADAPTER.read_text())
    names = {"_get_window_index", "_block_diag_mask", "_pack_multi_image",
             "_vision_graph_dyn_dims"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in names]
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "WallOssVision")
    nodes += [node for node in owner.body if isinstance(node, ast.FunctionDef)
              and node.name == "_select_native_window_boundaries"]
    constants = {"_NATIVE_WINDOW_BOUNDARIES", "_SMALL_NATIVE_WINDOW_BOUNDARIES"}
    nodes += [node for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in constants]
    namespace = {"torch": torch, "F": F, "Sequence": list}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ADAPTER), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("cameras", [2, 3])
def test_small_window_metadata_admits_only_two_or_three_actual_16x16_grids(cameras):
    h = _helpers()
    select = h["_select_native_window_boundaries"]
    owner = SimpleNamespace(_native_window_sdpa=True, _sms=2, _window_size=112, _patch_size=14)
    grid = torch.tensor([[1, 16, 16]] * cameras)
    _, _, _, boundaries = h["_pack_multi_image"](grid, 112, 2, 14)
    assert select(owner, False, boundaries, grid) is boundaries
    for bad_grid in (grid[:1], torch.tensor([[1, 16, 16]] * 4),
                     torch.tensor([[1, 18, 18]] * cameras),
                     torch.tensor([[1, 8, 32]] * cameras),
                     torch.tensor([[1, 16, 16]] * (cameras - 1) + [[1, 8, 32]])):
        with pytest.raises(ValueError, match="two/three 16x16 grids"):
            select(owner, False, boundaries, bad_grid)
    with pytest.raises(ValueError):
        select(owner, False, torch.tensor([0, 128, 256]), grid)
    with pytest.raises(ValueError):
        select(owner, False, None, grid)
    owner._window_size = 224
    with pytest.raises(ValueError):
        select(owner, False, boundaries, grid)
    owner._native_window_sdpa = False
    assert select(owner, False, boundaries, grid) is None
    owner._native_window_sdpa = True
    # Preserve the old exact M576 boundary ABI, independent of the small grid.
    exact = torch.tensor(h["_NATIVE_WINDOW_BOUNDARIES"])
    assert select(owner, True, exact) is exact
    key = h["_vision_graph_dyn_dims"]([32, 256 * cameras, 1, cameras], boundaries.tolist())
    assert key != h["_vision_graph_dyn_dims"]([32, 256 * cameras, 1, cameras], None)
    other_cameras = 5 - cameras
    assert key != h["_vision_graph_dyn_dims"]([32, 256 * other_cameras, 1, other_cameras], boundaries.tolist())
    assert key[-5:] == [0, 64, 128, 192, 256]


@pytest.mark.parametrize("cameras", [2, 3])
def test_small_window_reorder_matches_real_transformers_layout(cameras):
    spec = importlib.util.find_spec("transformers")
    if spec is None:
        pytest.skip("transformers is needed for the actual HF layout comparison")
    source = Path(spec.origin).parent / "models/qwen2_5_vl/modeling_qwen2_5_vl.py"
    tree = ast.parse(source.read_text())
    method = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and node.name == "get_window_index"), None)
    grid = torch.tensor([[1, 16, 16]] * cameras)
    if method is None:
        from transformers.vision_utils import get_vision_window_index

        hf_window, hf_cu = get_vision_window_index(
            grid, window_size=112, spatial_merge_size=2, patch_size=14,
        )
    else:
        namespace = {"torch": torch, "F": F}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        owner = SimpleNamespace(window_size=112, spatial_merge_size=2, patch_size=14,
                                spatial_merge_unit=4)
        hf_window, hf_cu = namespace["get_window_index"](owner, grid)
    window, reverse, mask, cu = _helpers()["_pack_multi_image"](grid, 112, 2, 14)
    assert torch.equal(window, hf_window)
    assert cu.tolist() == [0, 64, 128, 192, 256]
    assert torch.unique_consecutive(torch.as_tensor(hf_cu)).tolist() == list(range(0, 256 * cameras + 1, 64))
    assert torch.equal(window.sort().values, torch.arange(64 * cameras))
    for image in range(cameras):
        assert torch.equal(window[image * 64:(image + 1) * 64][reverse[image]],
                           torch.arange(image * 64, (image + 1) * 64))
    assert torch.equal(torch.isfinite(mask).sum(-1), torch.full((256,), 64))


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("cameras", [2, 3])
def test_small_window_batch_matches_dense_same_input_attention_and_isolation(full, cameras):
    h = _helpers()
    grid = torch.tensor([[1, 16, 16]] * cameras)
    window, _, mask, _ = h["_pack_multi_image"](grid, 112, 2, 14)
    generator = torch.Generator().manual_seed(2516)
    # Match native per-core storage [packed rows, local heads, d80], including
    # the real merge-unit reorder performed before all 32 encoder blocks.
    q, k, v = [torch.randn(64 * cameras, 4, 2, 80, generator=generator, dtype=torch.float64)
               [window].reshape(256 * cameras, 2, 80) for _ in range(3)]

    def attention(q, k, v, batch, seq, bias=None):
        arrays = [a.reshape(batch, seq, 2, 80).transpose(1, 2) for a in (q, k, v)]
        out = F.scaled_dot_product_attention(*arrays, attn_mask=bias)
        return out.transpose(1, 2).reshape(256 * cameras, 2, 80)

    dense = attention(q, k, v, cameras, 256, None if full else mask.double())
    native = attention(q, k, v, cameras if full else 4 * cameras, 256 if full else 64)
    torch.testing.assert_close(native, dense, rtol=1e-12, atol=1e-12)
    changed_v = v.clone()
    last_camera = 256 * (cameras - 1)
    changed_v[last_camera:] += 1.0
    changed = attention(q, k, changed_v, cameras if full else 4 * cameras, 256 if full else 64)
    assert torch.equal(native[:last_camera], changed[:last_camera])
    assert not torch.equal(native[last_camera:], changed[last_camera:])


@pytest.mark.parametrize("w8a16", [False, True])
def test_small_window_builder_preserves_exact_profile_and_cold_admission(w8a16):
    tree = ast.parse(ADAPTER.read_text())
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "build_wall_oss_vision")
    start = next(i for i, node in enumerate(builder.body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "exact_native_window")
    guard = compile(ast.Module(body=builder.body[start:start + 3], type_ignores=[]),
                    str(ADAPTER), "exec")
    cold = dict(native_window_sdpa=True, w8a16=w8a16, fp16_transfer=not w8a16,
                sorting=SimpleNamespace(vision_row=True), experimental_schedule=(0, False),
                window=True, mixed_precision=False, per_window_sdpa=False, fused_merger=True)
    exec(guard, dict(cold))
    for update in (dict(sorting=SimpleNamespace(vision_row=False)), dict(window=False),
                   dict(mixed_precision=True), dict(per_window_sdpa=True),
                   dict(fused_merger=False), dict(experimental_schedule=(32, False))):
        with pytest.raises(ValueError, match="requires the exact"):
            exec(guard, dict(cold, **update))
    if not w8a16:
        with pytest.raises(ValueError):
            exec(guard, dict(cold, fp16_transfer=False))
    exec(guard, dict(cold, experimental_schedule=(0, True), mixed_precision=True,
                     sorting=SimpleNamespace(vision_row=False)))


def test_mask_lifetime_and_window_descriptor_are_bound_to_emission():
    from test_wall_vision_small_image_routes import _body

    declarations = _body("    std::vector<BufferDecl> declare_buffers(")
    assert "resident_mask ? 1 : 3, resident_mask ? 9 : 4" in declarations
    assert "window_mask_present_ && !small_image_native_window_enabled()" in declarations
    producer = _body("    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    preload = _body("    void emit_preload_weights()")
    for body in (producer, preload):
        assert "small_image_schedule_flags()" in body
        assert "QWEN25_WALL_SORTING_SITE" in body
    assert "else if (small_image_resident_mask_enabled())" in preload
    assert 'prepared_window_mask_, addr_offset("sdpa_mask").value' in preload
    domain = _body("    physical_manifest_domain_for_candidate(")
    small = domain[domain.index("if (small_image_native_window_enabled() ||"):
                   domain.index("LayoutContext ddr_layout")]
    assert "subclass_spm_kv_by_mha_eligible(plan, raw_layout, position)" in small
    assert "!sites.empty() && std::all_of(" in small
    assert "generic_attention_site_raw_valid(plan, site.site_id)" in small
    assert "AttentionExecutionPolicy::SPM_KV_BY_MHA" in small
    assert "plan, raw_layout, physical_len, logical_len, position" in small
    consumer = _body("    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override")
    assert consumer.count("if (!small_image_resident_mask_enabled())") == 2
    small = consumer[consumer.index("if (small_image_native_window_enabled())"):
                     consumer.index("} else if (is_generic() && schedule_")]
    assert "QWEN25_MISC_RAW_ATTN_SITE : QWEN25_WINDOW_RAW_ATTN_SITE" in small
    assert "site.batch, site.seq_q, site.seq_k" in small
    assert "site.batch, site.seq_k" in small
    assert "consume_generic_attention_route(" in small


def test_raw_only_window_plan_omits_unused_ddr_insert_but_keeps_fallback():
    from test_wall_vision_small_image_routes import _body

    producer = _body("    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    assert "if (!small_image_raw_kv_only_enabled()) {\n                    append_kv(chunk);" in producer
    consumer = _body("    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override")
    branch = consumer[consumer.index("if (small_image_raw_kv_only_enabled())"):
                      consumer.index("double attn_scale")]
    raw, mirror = branch.split("} else {", 1)
    assert "has_complete_physical_manifest()" in raw
    assert "AttentionExecutionPolicy::SPM_KV_BY_MHA" in raw
    assert "rpu_launch_insert_kvcache" not in raw
    assert "QWEN25_KV_FLAG_DDR_MIRROR" in mirror
    assert "restore_kvinsert_plan(" in mirror
    assert "consume_physical_route(" in mirror
    assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in mirror

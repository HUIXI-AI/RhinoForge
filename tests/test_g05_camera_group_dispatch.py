"""Execute production K1 packing and whole Vision forwarding with CPU owners."""
import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "python/rpu_backend/adapters/qwen3_5/vision.py"
G05 = ROOT / "python/rpu_backend/adapters/g05/runtime.py"


def _actual_functions(path, names, namespace):
    """Reuse whole function bodies; only hardware dependencies are doubled."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.mark.parametrize("cameras,batch", [(2, 1), (2, 2), (3, 1), (3, 3)])
@pytest.mark.parametrize("direct_fusion", [False, True])
def test_actual_k1_pack_and_whole_forward_consume_every_camera(monkeypatch, cameras, batch, direct_fusion):
    import torch
    from rpu_backend.runtime.vision_position import interpolate_vision_positions

    # CPU doubles preserve the production Tensor operations and shape checks.
    # Only the explicit device transfer is redirected to CPU.
    tensor_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs = dict(kwargs, device="cpu")
        return tensor_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    api = ModuleType("rpu_backend.api")
    api.causal_lm = SimpleNamespace(_LIVE_LOCK=threading.RLock(), _LIVE_TERMINAL_REASON=None)
    monkeypatch.setitem(sys.modules, "rpu_backend.api", api)
    outputs_module = ModuleType("transformers.modeling_outputs")
    outputs_module.BaseModelOutputWithPooling = SimpleNamespace
    monkeypatch.setitem(sys.modules, "transformers.modeling_outputs", outputs_module)

    calls, grids, plans, positions = [], [], [], []
    live = {"output": None, "merged": None, "resolved": None, "resets": 0}

    class GraphCache:
        @contextmanager
        def capture(self, signature):
            yield
        def is_frozen(self): return False
        def lookup(self, signature): return object()

    class Cache:
        k_caches = [torch.zeros(1)]
        v_caches = [torch.ones(1)]
        def reset_to_position(self, position):
            assert position == 0

    class VisionOwner:
        def fast_pos_embed_interpolate(self, grid):
            assert torch.equal(grid, torch.tensor([[1, 16, 16]] * cameras))
            marker = torch.arange(1, cameras + 1, dtype=torch.float16).repeat_interleave(256)
            return marker[:, None].expand(-1, 1024).contiguous()

    owner = VisionOwner()
    owner.patch_size = 16
    owner.config = SimpleNamespace(temporal_patch_size=2)
    owner.spatial_merge_size = owner._rpu_vision_spatial_merge_size = 2
    owner._rpu_vision_handle = 42
    owner._rpu_vision_num_layers = 1
    owner._rpu_vision_hidden_size = 1024
    owner._rpu_vision_kv_cache = Cache()
    owner._rpu_vision_step0 = owner._rpu_vision_has_merger = True
    owner._rpu_vision_k1_camera_group_available = (True, True)
    owner._rpu_vision_exact_chunk_size = None
    owner._rpu_vision_chunk_size_cap = 0
    owner._rpu_vision_position_idx_keepalive = torch.empty(cameras * 256, 2, dtype=torch.int16)
    owner._rpu_vision_freq_cos = torch.empty(16, 32)
    owner._rpu_vision_graph_cache = GraphCache()
    owner._rpu_vision_graph_key = owner._rpu_vision_graph_sig = None
    owner._rpu_vision_step0_pos_grid = None
    owner._rpu_vision_step0_pos_temporal_num_frames = None
    owner._rpu_vision_step0_pos_camera_batch_count = None
    def installed_forward(): pass
    if batch > 1:
        installed_forward._g05_k1_camera_group_wrapper = True
        installed_forward._g05_k1_camera_batch = batch
    owner._rpu_g05_vision_forward_impl = installed_forward

    def plan(model, handle, rows, *, group_id, temporal_num_frames, camera_batch_count, compact_input):
        assert model is owner and handle == 42 and temporal_num_frames == 1 and not compact_input
        assert rows == 256 * batch and camera_batch_count == batch
        stage = SimpleNamespace(compute_chunk=rows, physical_descriptor=(rows, batch),
            physical_metadata={"stage_plan_fingerprint_hi": 1, "stage_plan_fingerprint_lo": 2,
                               "layout_hash_hi": 3, "layout_hash_lo": 4})
        plans.append((group_id, rows, batch))
        return SimpleNamespace(selected=SimpleNamespace(stage_tuple=stage), physical_plan_digest="cpu-double",
            as_dict=lambda **kwargs: {"selected": {"physical_metadata": {"camera_batch_count": batch}}})

    run_starts = [3 + 71 * camera for camera in range(cameras)]
    target = torch.full((1, run_starts[-1] + 67, 2048), -1.0, dtype=torch.float16)
    def consume(handle, embed, k, v, rows, pos, destination, starts, frames, consumed_batch,
                fingerprint_hi, fingerprint_lo, layout_hi, layout_lo, descriptor):
        assert handle == 42 and frames == 1 and consumed_batch == batch
        assert rows == 256 * batch and tuple(embed.shape) == (1, rows, 1536)
        assert k[0] is owner._rpu_vision_kv_cache.k_caches[0]
        assert v[0] is owner._rpu_vision_kv_cache.v_caches[0]
        assert (fingerprint_hi, fingerprint_lo, layout_hi, layout_lo) == (1, 2, 3, 4)
        assert descriptor == (rows, batch)
        first_camera = 0 if batch > 1 else len(calls)
        selected = list(range(first_camera + 1, first_camera + batch + 1))
        assert tuple(pos.shape) == (rows, 1024)
        for ordinal, marker in enumerate(selected):
            assert torch.all(embed[0, ordinal * 256:(ordinal + 1) * 256] == marker)
            assert torch.all(pos[ordinal * 256:(ordinal + 1) * 256] == marker)
        position = owner._rpu_vision_position_idx_keepalive[:rows].clone()
        assert tuple(position.shape) == (rows, 2)
        for ordinal in range(batch):
            coords = position[ordinal * 256:(ordinal + 1) * 256]
            assert set(map(tuple, coords.tolist())) == {(h, w) for h in range(16) for w in range(16)}
        positions.append(position)
        expected_starts = run_starts if batch > 1 else [run_starts[first_camera]]
        if direct_fusion:
            assert destination is target
            assert starts == expected_starts, "native direct fusion requires every camera run start"
            for start, marker in zip(starts, selected, strict=True):
                destination[0, start:start + 64].fill_(marker)
        else:
            assert destination is None and starts == []
        # Deliberately reuse model buffers for B1. Production must snapshot each
        # camera before the next call overwrites the same returned storage.
        if live["output"] is None:
            live["output"] = torch.empty(1, rows, 1024, dtype=torch.float16)
            live["merged"] = torch.empty(1, rows // 4, 2048, dtype=torch.float16)
        for ordinal, marker in enumerate(selected):
            live["output"][0, ordinal * 256:(ordinal + 1) * 256].fill_(marker)
            live["merged"][0, ordinal * 64:(ordinal + 1) * 64].fill_(marker)
        live["resolved"] = rows
        calls.append({"rows": rows, "starts": list(starts), "cameras": selected})
        return live["output"]

    def reset(): live["resets"] += 1
    ops = SimpleNamespace(rpu=SimpleNamespace(qwen3_5_vision_forward=consume,
        qwen3_5_vision_get_resolved_chunk_size=lambda handle: live["resolved"],
        qwen3_5_vision_get_merger_out=lambda handle: live["merged"], spm_alloc_reset_temporary=reset))
    class TorchCPU:
        rpu = SimpleNamespace(get_debug_export=lambda: False)
        def __getattr__(self, name): return ops if name == "ops" else getattr(torch, name)
    namespace = dict(torch=TorchCPU(), _plan_qwen3_5_vision_group=plan,
        interpolate_vision_positions=interpolate_vision_positions,
        _vision_group_graph_signature=lambda key, planned: key,
        _evict_vision_graph=lambda *args: None,
        _g05_packed_vision_capacity_enabled=lambda: False)
    _actual_functions(VISION, ["_validate_vision_grid", "_compute_vision_position_idx_cpu",
        "_cached_vision_position_idx_cpu", "_rpu_vision_forward"], namespace)
    actual_positions = namespace["_cached_vision_position_idx_cpu"]
    def position_group(model, grid, merge_size, **kwargs):
        grids.append(grid.clone())
        return actual_positions(model, grid, merge_size, **kwargs)
    namespace["_cached_vision_position_idx_cpu"] = position_group
    _actual_functions(G05, ["_pack_g05_vision"], namespace)
    pixels = {name: torch.full((1, 1, 3, 256, 256), float(camera + 1))
              for camera, name in enumerate(("head_rgb", "left_wrist_rgb", "right_wrist_rgb")[:cameras])}
    patches, grid, split_sizes, frames = namespace["_pack_g05_vision"](pixels, owner)
    assert tuple(patches.shape) == (256 * cameras, 1536)
    assert split_sizes == [64] * cameras and frames == 1
    assert torch.equal(grid, torch.tensor([[1, 16, 16]] * cameras))
    result = namespace["_rpu_vision_forward"](owner, patches, grid,
        _rpu_temporal_num_frames=frames, _rpu_camera_batch_count=batch,
        **({"_rpu_fusion_target": target, "_rpu_fusion_run_starts": run_starts} if direct_fusion else {}))
    assert len(calls) == len(grids) == len(plans) == cameras // batch
    assert all(tuple(value.shape) == (batch, 3) for value in grids)
    assert live["resets"] == 1
    if direct_fusion:
        assert result.last_hidden_state is None and result.pooler_output is None
        mask = torch.ones(target.shape[1], dtype=torch.bool)
        for camera, start in enumerate(run_starts):
            assert torch.all(target[0, start:start + 64] == camera + 1)
            mask[start:start + 64] = False
        assert torch.all(target[0, mask] == -1)
    else:
        assert tuple(result.last_hidden_state.shape) == (256 * cameras, 1024)
        assert tuple(result.pooler_output.shape) == (64 * cameras, 2048)
        assert result.pooler_output.dtype == torch.float32
        for camera in range(cameras):
            assert torch.all(result.last_hidden_state[camera * 256:(camera + 1) * 256] == camera + 1)
            assert torch.all(result.pooler_output[camera * 64:(camera + 1) * 64] == camera + 1)

"""Board-free contract gates for the exact Qwen3-VL outer-pipeline input."""

import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "python/rpu_backend/adapters/qwen3_vl/vision.py"


def _helper_node() -> tuple[str, ast.FunctionDef]:
    source = VISION.read_text(encoding="utf-8")
    module = ast.parse(source)
    matches = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_qwen3vl_vision_input_for_spm_pipeline"
    ]
    assert len(matches) == 1
    return source, matches[0]


def _helper_source(source: str, node: ast.FunctionDef) -> str:
    assert node.end_lineno is not None
    return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])


def test_helper_is_exact_fail_closed_and_preserves_the_normal_forward():
    source, node = _helper_node()
    body = _helper_source(source, node)

    for token in (
        'type(handle) is not int or handle <= 0',
        'grid_thw.device.type != "cpu"',
        'grid_thw.size(0) not in (1, 3)',
        'grid_thw.tolist() != [[1, 16, 16]] * image_count',
        'hidden_states.device.type != "cpu"',
        'hidden_states.dtype != torch.float16',
        'hidden_states.size(0) != image_count * 256',
        'hidden_size != 1024 or spatial_merge_size != 2',
        'patch_weight.device.type != "cpu"',
        'patch_weight.dtype != torch.float16',
        'patch_weight.size(1) != hidden_states.size(1)',
        'device_patch_embed: bool = False',
        'vision_model, "_rpu_patch_embed_w_rpu", None',
        'vision_model, "_rpu_patch_embed_b_rpu", None',
        'patch_weight_rpu.device.type != "rpu"',
        'patch_weight_rpu.dtype != torch.float16',
        '(1024, hidden_states.size(1))',
        'patch_weight_rpu.is_contiguous()',
        '(patch_bias is None) != (patch_bias_rpu is None)',
        'patch_bias_rpu.device.type != "rpu"',
        'patch_bias_rpu.dtype != torch.float16',
        'tuple(patch_bias_rpu.shape) != (1024,)',
        'patch_bias_rpu.is_contiguous()',
        'keepalive.device.type != "rpu"',
        'keepalive.dtype != torch.int16',
        'patch_rpu.device.type != "rpu"',
        'patch_rpu.dtype != torch.float16',
        'tuple(patch_rpu.shape) != (image_count * 256, 1024)',
        'patch_rpu.is_contiguous()',
        'tuple(prepared.shape) != (image_count, 256, 1024)',
        'prepared.is_contiguous()',
    ):
        assert token in body

    # The helper is opt-in only: defining it must not alter or wrap the public
    # forward/install paths.
    assert source.count("_prepare_qwen3vl_vision_input_for_spm_pipeline") == 1
    assert 'getattr(vision_model, "_rpu_patch_embed_on_device"' not in body


def test_helper_stages_exact_inputs_without_entering_graph_or_spm_lifecycle():
    source, node = _helper_node()
    body = _helper_source(source, node)

    fast_pos = body.index("interpolate_vision_positions(vision_model,")
    pos_idx = body.index("_compute_vision_position_idx_cpu(")
    keepalive = body.index("keepalive.narrow(0, 0, 256).copy_(")
    raw_to_rpu = body.index("patch_input_rpu = hidden_states.contiguous().to(")
    device_patch = body.index("torch.nn.functional.linear(", raw_to_rpu)
    cpu_patch = body.index("torch.nn.functional.linear(", device_patch + 1)
    cpu_to_rpu = body.index("patch_cpu.to(", cpu_patch)
    patch_gate = body.index('patch_rpu.device.type != "rpu"', cpu_to_rpu)
    add_pos = body.index("patch_rpu + position_rpu", patch_gate)
    result_check = body.index('prepared.device.type != "rpu"', add_pos)
    result = body.index("return prepared", result_check)
    assert (
        fast_pos
        < pos_idx
        < keepalive
        < raw_to_rpu
        < device_patch
        < cpu_patch
        < cpu_to_rpu
        < patch_gate
        < add_pos
        < result_check
        < result
    )

    for token in (
        "canonical_grid.narrow(0, 0, 1)",
        'device="rpu", dtype=torch.float16, non_blocking=False',
        "patch_input_rpu, patch_weight_rpu, patch_bias_rpu",
        "hidden_states.contiguous(), patch_weight, patch_bias",
        "patch_rpu + position_rpu",
        "image_count, 256, 1024",
    ):
        assert token in body

    assert body.count("torch.nn.functional.linear(") == 2

    for token in (
        "_rpu_vision_pos_embeds",
        "_rpu_vision_pos_embeds_cpu32",
        "_rpu_vision_pos_embeds_key",
        "_rpu_vision_pos_idx_key",
        "_rpu_vision_pos_idx_val",
        "_rpu_vision_keepalive_key",
    ):
        assert token in body

    calls = {
        ast.unparse(call.func)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }
    forbidden_calls = {
        "vision_model._rpu_vision_graph_cache.capture",
        "torch.ops.rpu.qwen3vl_vision_forward",
        "torch.ops.rpu.qwen3vl_vision_pop_merged",
        "torch.ops.rpu.qwen3vl_vision_pop_deepstack_snapshots",
        "rpu_backend.reset_spm_allocator",
    }
    assert calls.isdisjoint(forbidden_calls)
    assert not any(name.endswith(".clone") for name in calls)
    assert not any("reset_to_position" in name for name in calls)


@pytest.fixture
def helper_device_boundary(monkeypatch):
    """Run the actual helper with CPU storage and observable device copies."""
    source, helper_node = _helper_node()
    nodes = [helper_node, next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_compute_vision_position_idx_cpu"
    )]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(VISION), "exec"), scope)
    uploads, publications = [], []
    original_to = torch.Tensor.to

    class RPUTensor(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="rpu")

        def copy_(self, source, *args, **kwargs):
            # The position-copy boundary must use a CPU source and publish its
            # destination on every call, including a memoized grid.
            assert self.dtype == torch.int16 and source.device.type == "cpu"
            value = self.as_subclass(torch.Tensor)
            torch.Tensor.copy_(value, source, *args, **kwargs)
            publications.append((self.data_ptr(), self.nbytes, value.clone()))
            return self

    def device_to(value, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if target == "rpu" or isinstance(target, torch.device) and target.type == "rpu":
            uploads.append((value.dtype, tuple(value.shape)))
            if "device" in kwargs:
                kwargs["device"] = "cpu"
            else:
                args = ("cpu", *args[1:])
            return original_to(value, *args, **kwargs).as_subclass(RPUTensor)
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", device_to)
    return SimpleNamespace(scope=scope, rpu=lambda value: value.as_subclass(RPUTensor),
                           uploads=uploads, publications=publications)


def _pipeline_inputs(boundary, image_count):
    grid = torch.tensor([[1, 16, 16]] * image_count)
    weight = torch.linspace(-0.1, 0.1, 4096).reshape(1024, 4).half()
    bias = torch.linspace(-0.1, 0.1, 1024).half()
    model = SimpleNamespace(
        _rpu_vision_handle=7, _rpu_vision_hidden_size=1024,
        _rpu_vision_spatial_merge_size=2, _rpu_vision_max_hw=16,
        _rpu_vision_patch_embed_w=weight, _rpu_vision_patch_embed_b=bias,
        _rpu_patch_embed_w_rpu=boundary.rpu(weight),
        _rpu_patch_embed_b_rpu=boundary.rpu(bias),
        _rpu_vision_position_idx_keepalive=boundary.rpu(torch.full(
            (272, 2), 777, dtype=torch.int16)),
        _rpu_vision_pos_embeds_key=grid.clone(),
        _rpu_vision_pos_embeds=boundary.rpu(torch.full(
            (image_count * 256, 1024), 0.125, dtype=torch.float16)),
    )
    patches = torch.linspace(-0.5, 0.5, image_count * 1024).reshape(-1, 4).half()
    return model, patches, grid


@pytest.mark.parametrize("image_count", [1, 3])
@pytest.mark.parametrize("device_patch_embed", [False, True])
def test_position_cpu_copy_refreshes_dirty_keepalive_and_publishes_each_call(
        helper_device_boundary, image_count, device_patch_embed):
    boundary = helper_device_boundary
    model, patches, grid = _pipeline_inputs(boundary, image_count)
    helper = boundary.scope["_prepare_qwen3vl_vision_input_for_spm_pipeline"]
    keepalive = model._rpu_vision_position_idx_keepalive
    version = keepalive._version
    # Independent 2x2-block order expected by the Vision RoPE kernel.
    expected_indices = torch.tensor([
        (row, col) for block_row in range(0, 16, 2)
        for block_col in range(0, 16, 2)
        for row in range(block_row, block_row + 2)
        for col in range(block_col, block_col + 2)
    ], dtype=torch.int16)
    expected_patches = torch.nn.functional.linear(
        patches, model._rpu_vision_patch_embed_w,
        model._rpu_vision_patch_embed_b) + 0.125

    for call in range(3):
        # An untracked CPU alias dirties the same DDR address without changing
        # the tensor version. A grid/identity/version cache must not skip copy.
        ctypes.memset(keepalive.data_ptr(), 0x7F, 1024)
        assert keepalive._version == version
        result = helper(model, patches, grid, device_patch_embed=device_patch_embed)
        assert len(boundary.publications) == call + 1
        ptr, byte_count, published = boundary.publications[-1]
        assert ptr == keepalive.data_ptr() and byte_count == 1024
        assert torch.equal(published, expected_indices)
        assert torch.equal(keepalive.as_subclass(torch.Tensor)[256:],
                           torch.full((16, 2), 777, dtype=torch.int16))
        assert model._rpu_vision_pos_idx_val.device.type == "cpu"
        assert model._rpu_vision_keepalive_key == ((1, 16, 16, 1, 2), 256, False)
        assert result.device.type == "rpu" and result.is_contiguous()
        assert torch.equal(result.as_subclass(torch.Tensor),
                           expected_patches.view(image_count, 256, 1024))
        version = keepalive._version
    assert all(dtype != torch.int16 for dtype, _shape in boundary.uploads)


@pytest.mark.parametrize("fault", ["position_shape", "keepalive_shape"])
def test_wrong_position_geometry_rejects_before_keepalive_copy(
        helper_device_boundary, fault):
    boundary = helper_device_boundary
    model, patches, grid = _pipeline_inputs(boundary, 1)
    if fault == "position_shape":
        boundary.scope["_compute_vision_position_idx_cpu"] = (
            lambda *_args: torch.zeros((255, 2), dtype=torch.int16))
        error = r"position_idx must be \[256,2\]"
    else:
        model._rpu_vision_position_idx_keepalive = boundary.rpu(
            torch.full((255, 2), 777, dtype=torch.int16))
        error = "installed position_idx keepalive"
    before = model._rpu_vision_position_idx_keepalive.as_subclass(torch.Tensor).clone()
    with pytest.raises(RuntimeError, match=error):
        boundary.scope["_prepare_qwen3vl_vision_input_for_spm_pipeline"](
            model, patches, grid)
    assert not boundary.publications and not boundary.uploads
    assert torch.equal(model._rpu_vision_position_idx_keepalive.as_subclass(torch.Tensor), before)

"""CPU-only checks for Qwen3.5 vision input layout helpers."""

import ast
import ctypes
import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch


_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_E2E = os.path.join(
    _REPO,
    "python/rpu_backend/tests/model/qwen3_5/verify_e2e_single_image.py",
)
_stubs = {
    name: MagicMock()
    for name in (
        "rpu_backend",
        "rpu_backend.adapters",
        "rpu_backend.adapters.qwen3_5",
        "rpu_backend.adapters.qwen3_5.text",
        "rpu_backend.api",
        "rpu_backend.api.cache",
        "rpu_backend.runtime",
        "rpu_backend.runtime.log",
        "rpu_backend.runtime.weights",
    )
}

with patch.dict(sys.modules, _stubs):
    _spec = importlib.util.spec_from_file_location(
        "qwen3_5_vision_cpu",
        os.path.join(_REPO, "python/rpu_backend/adapters/qwen3_5/vision.py"),
    )
    _vision = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_vision)


@pytest.fixture(autouse=True)
def _planner_ops(monkeypatch):
    from rpu_backend.api import causal_lm

    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    from rpu_backend.api import _execution
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    monkeypatch.setattr(
        _vision.torch.ops.rpu, "qwen3_5_vision_set_cache_capacity", MagicMock(), raising=False
    )
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_set_chunk_size_cap",
        MagicMock(),
        raising=False,
    )
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_set_prefill_chunk_size",
        MagicMock(),
        raising=False,
    )
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_set_linear_acc32",
        MagicMock(),
        raising=False,
    )
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_resolve_prefill_domain",
        MagicMock(return_value=[[
            2, 256, 896, 128, 128, 512, 1, 0, 7,
            0, 9, 1, 1, 1, 1, 1, 0, 0, 256,
        ]]),
        raising=False,
    )


def test_position_idx_enforces_rope_table_bounds():
    boundary = _vision._compute_vision_position_idx_cpu(
        torch.tensor([[1, 2, 128]]), 2, max_hw=128
    )
    assert int(boundary[:, 1].max()) == 127

    with pytest.raises(ValueError, match="exceeds 2D RoPE table max_hw=128"):
        _vision._compute_vision_position_idx_cpu(
            torch.tensor([[1, 16, 256]]), 2, max_hw=128
        )




def test_position_idx_cache_is_exact_and_single_entry():
    model = types.SimpleNamespace()
    grid = torch.tensor([[1, 4, 4]])

    first = _vision._cached_vision_position_idx_cpu(model, grid, 2, max_hw=4)
    second = _vision._cached_vision_position_idx_cpu(
        model, grid.clone(), 2, max_hw=4
    )
    assert second is first

    changed = _vision._cached_vision_position_idx_cpu(
        model, torch.tensor([[1, 2, 4]]), 2, max_hw=4
    )
    assert changed is not first
    assert changed.shape == (8, 2)


@pytest.mark.parametrize("rope_disabled", [False, True])
def test_position_preparation_refreshes_dirty_alias_without_temporary_upload(
        monkeypatch, rope_disabled):
    """Run the actual forward preparation block with observable copy boundaries."""
    filename = os.path.join(_REPO, "python/rpu_backend/adapters/qwen3_5/vision.py")
    with open(filename, encoding="utf-8") as source:
        tree = ast.parse(source.read(), filename=filename)
    forward = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "_rpu_vision_forward")
    loop = next(node for node in ast.walk(forward)
                if isinstance(node, ast.For)
                and ast.unparse(node.iter) == "enumerate(planned_groups)")
    start = next(i for i, node in enumerate(loop.body)
                 if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) == "pos_idx_cpu")
    end = next(i for i, node in enumerate(loop.body)
               if isinstance(node, ast.Expr)
               and ast.unparse(node.value) == "cache.reset_to_position(0)")
    block = compile(ast.Module(loop.body[start:end + 1], type_ignores=[]),
                    filename, "exec")
    publications, uploads = [], []

    class PositionTensor(torch.Tensor):
        @property
        def device(self):
            return types.SimpleNamespace(type="rpu")

        def copy_(self, source, *args, **kwargs):
            assert source.device.type == "cpu"
            plain = self.as_subclass(torch.Tensor)
            torch.Tensor.copy_(plain, source, *args, **kwargs)
            publications.append((self.data_ptr(), self.nbytes, plain.clone()))
            return self

        def zero_(self):
            plain = self.as_subclass(torch.Tensor)
            torch.Tensor.zero_(plain)
            publications.append((self.data_ptr(), self.nbytes, plain.clone()))
            return self

    original_to = torch.Tensor.to

    def observe_to(value, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if getattr(target, "type", None) == "rpu" or target == "rpu":
            uploads.append(tuple(value.shape))
            return value.clone().as_subclass(PositionTensor)
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observe_to)
    keepalive = torch.full((272, 2), 777, dtype=torch.int16).as_subclass(PositionTensor)
    model = types.SimpleNamespace(
        _rpu_vision_freq_cos=torch.empty(128, 1),
        _rpu_vision_rope_disable=rope_disabled,
    )
    # Include consecutive same-grid calls and A/B/A geometry. The cached CPU
    # indices may be reused; the private DDR destination must always refresh.
    grids = [(16, 16), (16, 16), (8, 16), (16, 16)]
    for call, (height, width) in enumerate(grids):
        rows = height * width
        plain = keepalive.as_subclass(torch.Tensor)
        tail = plain[rows:].clone()
        version = keepalive._version
        ctypes.memset(keepalive.data_ptr(), 0x7F, rows * 4)
        assert keepalive._version == version
        expected = torch.tensor([
            (row, col)
            for block_row in range(0, height, 2)
            for block_col in range(0, width, 2)
            for row in range(block_row, block_row + 2)
            for col in range(block_col, block_col + 2)
        ], dtype=torch.int16)
        if rope_disabled:
            expected.zero_()

        def reset(position):
            assert position == 0 and len(publications) == call + 1
            assert torch.equal(plain[:rows], expected)

        namespace = dict(_vision.__dict__, self=model, keepalive=keepalive,
                         cache=types.SimpleNamespace(reset_to_position=reset),
                         grid_thw_cpu=torch.tensor([[1, height, width]]),
                         grid_start=0, temporal_num_frames=1, camera_batch_count=1,
                         spatial_merge_size=2, n_i=rows, i=0)
        exec(block, namespace)
        assert torch.equal(publications[-1][2], expected)
        assert publications[-1][:2] == (keepalive.data_ptr(), rows * 4)
        assert torch.equal(plain[rows:], tail)
        assert namespace["pos_idx_cpu"].device.type == "cpu"
    assert uploads == []

    before = plain.clone()
    namespace["n_i"] = 255
    with pytest.raises(RuntimeError, match="position_idx rows"):
        exec(block, namespace)
    assert len(publications) == len(grids)
    assert torch.equal(plain, before)


def test_temporal_step0_position_pack_is_core_major_and_lossless():
    pos = torch.arange(12 * 16, dtype=torch.float16).view(12, 16)
    packed = _vision._pack_temporal_step0_pos(pos, num_cores=4)

    assert packed.shape == (4, 12, 4)
    assert packed.is_contiguous()
    restored = packed.permute(1, 0, 2).reshape_as(pos)
    assert torch.equal(restored, pos)


def test_rope_tables_preserve_model_inv_freq():
    head_dim = 64
    max_hw = 30
    inv_freq = (
        1.0
        / (
            10000.0
            ** (
                torch.arange(0, head_dim // 2, 2, dtype=torch.float32)
                / (head_dim // 2)
            )
        )
    ).half()

    cos, sin = _vision.build_vision_rope_tables(
        head_dim, max_hw, device="cpu", inv_freq=inv_freq
    )
    freqs = torch.outer(torch.arange(max_hw, dtype=inv_freq.dtype), inv_freq)

    assert torch.equal(cos, freqs.cos().half())
    assert torch.equal(sin, freqs.sin().half())


@pytest.mark.parametrize(
    "grid, match",
    [
        (torch.empty((0, 3), dtype=torch.long), "at least one image"),
        (torch.tensor([[1, 3, 3]]), "must be divisible"),
        (torch.tensor([[0, 2, 2]]), "must be positive"),
        (torch.tensor([[2, 2, 2]]), "video grids are not supported"),
    ],
)
def test_vision_grid_rejects_uninitialized_or_truncated_layouts(grid, match):
    with pytest.raises(ValueError, match=match):
        _vision._compute_vision_position_idx_cpu(grid, 2)


def test_visual_run_starts_accepts_multi_image_and_rejects_fragmented_runs():
    image_token = 99
    ids = torch.tensor([[1, 99, 99, 2, 99, 99, 99, 3]])
    assert _vision._visual_run_starts(ids, image_token, [2, 3]) == [1, 4]

    fragmented = torch.tensor([[1, 99, 2, 99, 99, 3]])
    assert _vision._visual_run_starts(fragmented, image_token, [2, 1]) is None


def _runtime_capable_vision():
    class VisionStub:
        def forward(self, *args, **kwargs):
            return args, kwargs

    cfg = types.SimpleNamespace(
        num_heads=16,
        hidden_size=1024,
        intermediate_size=4096,
        spatial_merge_size=2,
    )
    vision = VisionStub()
    vision.blocks = [types.SimpleNamespace() for _ in range(24)]
    vision.merger = object()
    vision.config = cfg
    return vision


def _unsupported_08b_vision():
    class VisionStub:
        def forward(self, *args, **kwargs):
            return args, kwargs

    cfg = types.SimpleNamespace(
        num_heads=12,
        hidden_size=768,
        intermediate_size=3072,
        spatial_merge_size=2,
    )
    vision = VisionStub()
    vision.blocks = [types.SimpleNamespace() for _ in range(12)]
    vision.merger = object()
    vision.config = cfg
    return vision


def _mark_complete_vision_runtime(vision, *, finalizer_alive=True):
    from rpu_backend.adapters.qwen3_5.text import _Qwen35NativeState
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    original_forward = vision.forward
    vision._rpu_qwen3_5_had_instance_forward = False
    vision._rpu_qwen3_5_original_forward = original_forward
    values = {
        "_rpu_vision_handle": 23,
        "_rpu_vision_handle_finalizer": types.SimpleNamespace(
            alive=finalizer_alive
        ),
        "_rpu_vision_freq_cos": object(),
        "_rpu_vision_freq_sin": object(),
        "_rpu_vision_position_idx_keepalive": object(),
        "_rpu_vision_step0": False,
        "_rpu_vision_patch_embed_w": object(),
        "_rpu_vision_patch_embed_b": None,
        "_rpu_vision_has_merger": False,
        "_rpu_vision_kv_cache": object(),
        "_rpu_vision_graph_disable": False,
        "_rpu_vision_graph_cache": object(),
        "_rpu_vision_graph_key": None,
        "_rpu_vision_graph_sig": None,
        "_rpu_vision_debug_graph": object(),
        "_rpu_vision_spatial_merge_size": 2,
        "_rpu_vision_num_layers": 24,
        "_rpu_vision_hidden_size": 1024,
        "_rpu_vision_linear_acc32": False,
        "_rpu_vision_w8a16": False,
        "_rpu_vision_math_precision": "base",
        "_rpu_vision_k1_b3_available": False,
        "_rpu_vision_k1_camera_group_available": (False, False),
        "_rpu_vision_gelu_erf_mode": "standard",
        "_rpu_vision_chunk_size_cap": 0,
        "_rpu_vision_exact_chunk_size": None,
        "_rpu_vision_padding_rows": "auto",
        "_rpu_vision_padding_budget": 0,
        "_rpu_vision_control_snapshot": _vision._resolve_vision_install_options()[4],
        "_rpu_vision_last_a6_plan": None,
        "_rpu_vision_a6_plans": {},
    }
    for name, value in values.items():
        setattr(vision, name, value)
    state = _Qwen35NativeState(handle=23)
    resource = _InstalledNativeResource(state, 23, lambda _: None, graphs=(),
        keepalive=(), label="Vision fixture", handle_name="handle")
    state.retirement, state.handle_finalizer = resource, resource.finalizer
    vision._rpu_vision_retirement_state = state
    vision._rpu_vision_handle_finalizer = resource.finalizer
    if not finalizer_alive:
        resource.finalizer.detach()
    for block in vision.blocks:
        block._rpu_qwen3_5_vision_weights_converted = True
    vision.forward = types.MethodType(_vision._rpu_vision_forward, vision)
    return vision._rpu_vision_handle_finalizer


@pytest.mark.parametrize("env_value", [None, "0", "true"])
def test_vision_install_requires_exact_numeric_blocked_opt_in(
    monkeypatch, env_value
):
    if env_value is None:
        monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    else:
        monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", env_value)
    create = MagicMock(side_effect=AssertionError("must reject before create"))
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_create",
        create,
        raising=False,
    )

    with pytest.raises(NotImplementedError, match="experimental/numeric-blocked"):
        _vision.install_qwen3_5_vision_for_rpu(_runtime_capable_vision())
    create.assert_not_called()


def test_vision_install_exact_numeric_blocked_opt_in_reaches_create(monkeypatch):
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    create = MagicMock(side_effect=RuntimeError("reached-create"))
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_create",
        create,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="reached-create"):
        _vision.install_qwen3_5_vision_for_rpu(_runtime_capable_vision())
    create.assert_called_once_with()


def test_vision_install_admits_candidate_08b_only_after_controlled_opt_in(monkeypatch):
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    create = MagicMock(side_effect=RuntimeError("reached-create"))
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_create",
        create,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="reached-create"):
        _vision.install_qwen3_5_vision_for_rpu(_unsupported_08b_vision())
    create.assert_called_once_with()


def test_vision_candidate_still_fails_before_create_without_opt_in(monkeypatch):
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    create = MagicMock(side_effect=AssertionError("must reject before create"))
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_create",
        create,
        raising=False,
    )

    with pytest.raises(NotImplementedError, match="experimental/numeric-blocked"):
        _vision.install_qwen3_5_vision_for_rpu(_unsupported_08b_vision())
    create.assert_not_called()


def test_fuse_rejects_invalid_inputs_before_lazy_install(monkeypatch):
    install = MagicMock(side_effect=AssertionError("must reject before install"))
    monkeypatch.setattr(_vision, "install_qwen3_5_vision_for_rpu", install)
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(visual=types.SimpleNamespace())
    )
    grid = torch.tensor([[1, 2, 2]])

    with pytest.raises(ValueError, match="image_grid_thw"):
        _vision.fuse_visual_embeds(model, None, torch.tensor([[1]]), None, None)
    with pytest.raises(NotImplementedError, match="video inputs"):
        _vision.fuse_visual_embeds(
            model, None, torch.tensor([[1]]), None, grid,
            video_grid_thw=grid,
        )
    with pytest.raises(ValueError, match="input_ids required"):
        _vision.fuse_visual_embeds(model, None, None, None, grid)
    with pytest.raises(ValueError, match="pixel_values is required"):
        _vision.fuse_visual_embeds(
            model, None, torch.tensor([[1]]), None, grid,
            mm_token_type_ids=torch.tensor([[1]]),
        )
    with pytest.raises(ValueError, match="mm_token_type_ids is required"):
        _vision.fuse_visual_embeds(
            model, None, torch.tensor([[1]]), torch.tensor([[0.0]]), grid,
        )
    install.assert_not_called()

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(image_token_id=99),
        model=types.SimpleNamespace(
            visual=types.SimpleNamespace(
                config=types.SimpleNamespace(spatial_merge_size=2)
            )
        ),
    )
    with pytest.raises(RuntimeError, match="image embeds rows"):
        _vision.fuse_visual_embeds(
            model,
            torch.zeros(1, 1, 4),
            torch.tensor([[99]]),
            torch.zeros(16, 4),
            torch.tensor([[1, 4, 4]]),
            mm_token_type_ids=torch.ones(1, 1, dtype=torch.long),
        )
    install.assert_not_called()


def test_vision_execution_options_snapshot_and_legacy_precedence(monkeypatch):
    monkeypatch.setenv("QWEN3_5_VISION_CHUNK", "128")
    cap, exact, rows, budget, snapshot = (
        _vision._resolve_vision_install_options(
            execution_config={"vision": {"chunk_size": "auto"}}
        )
    )
    assert (cap, exact, rows, budget) == (0, None, "auto", 0)
    assert snapshot["chunk_source"] == "canonical"
    with pytest.raises(TypeError):
        snapshot["chunk_source"] = "legacy_env"

    cap, exact, _, _, _ = _vision._resolve_vision_install_options(
        execution_config={"vision": {"chunk_size": 128}}
    )
    assert (cap, exact) == (128, 128)
    with pytest.raises(ValueError, match="conflicts with canonical"):
        _vision._resolve_vision_install_options(
            execution_config={"vision": {"chunk_size": 256}}
        )


def test_vision_execution_options_reject_unsafe_padding():
    with pytest.raises(ValueError, match="padding is unsupported"):
        _vision._resolve_vision_install_options(
            execution_config={"vision": {"padding_budget": 1}}
        )


def test_vision_a6_dry_resolver_is_declared_registered_and_consumed():
    paths = {
        "decl": "src/core/rpu_kernel_decls.h",
        "dispatch": "src/core/rpu_dispatch_registrations.inc",
        "native": "src/fused/rpu_qwen3_5_vision_model.cpp",
        "adapter": "python/rpu_backend/adapters/qwen3_5/vision.py",
    }
    sources = {
        name: open(os.path.join(_REPO, path), encoding="utf-8").read()
        for name, path in paths.items()
    }
    symbol = "qwen3_5_vision_resolve_prefill_domain"
    assert all(symbol in source for source in sources.values())
    assert "resolved != planned" in sources["adapter"]
    assert "saved_compact_step0" in sources["native"]
    assert "fmb_three_stage_chunk_plan_fingerprint" in sources["native"]
    assert "merger_chunks_per_span" in sources["native"]
    assert "resolve_spm_pipeline_component_layout_for_cpu_contract" in sources["native"]
    assert "if active_sig is not None:" in sources["adapter"]

    resolver = sources["native"].split(
        "Qwen3_5VisionModel::resolve_prefill_domain(", 1
    )[1].split("// ── plan_kv_first_chunks", 1)[0]
    assert resolver.index("RpuExecutionCleanupGuard") < resolver.index(
        "saved_temporal"
    )

    tree = ast.parse(sources["adapter"])
    forward = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_rpu_vision_forward"
    )
    forward_source = ast.unparse(forward)
    planning = forward_source.index("planned_groups = tuple(")
    embedding = forward_source.index("embed_packed =")
    dispatch = forward_source.index("in enumerate(planned_groups)")
    assert planning < embedding < dispatch
    assert forward_source.count("_plan_qwen3_5_vision_group(") == 1


def test_vision_admits_all_groups_before_first_device_mutation(monkeypatch):
    mutations = []

    class Hidden:
        device = types.SimpleNamespace(type="rpu")
        dtype = torch.float16
        shape = (384, 1)

        @staticmethod
        def dim():
            return 2

        @staticmethod
        def size(dim):
            assert dim == 0
            return 384

        @staticmethod
        def contiguous():
            mutations.append("embed")
            return Hidden()

    cache = types.SimpleNamespace(
        k_caches=[], v_caches=[], reset_to_position=MagicMock()
    )
    previous_receipt = {"previous": object()}
    vision = types.SimpleNamespace(
        _rpu_vision_handle=7,
        _rpu_vision_spatial_merge_size=2,
        _rpu_vision_num_layers=0,
        _rpu_vision_kv_cache=cache,
        _rpu_vision_step0=True,
        _rpu_vision_has_merger=False,
        _rpu_vision_exact_chunk_size=256,
        _rpu_vision_padding_rows="auto",
        _rpu_vision_padding_budget=0,
        _rpu_vision_a6_plans=previous_receipt,
        _rpu_vision_last_a6_plan=object(),
    )
    logical_lengths = []

    def plan(_vision_model, _handle, logical_len, **_kwargs):
        logical_lengths.append(logical_len)
        if logical_len == 128:
            raise RuntimeError("later group rejected")
        return object()

    monkeypatch.setattr(_vision, "_validate_vision_grid", lambda *_args: None)
    monkeypatch.setattr(_vision, "_plan_qwen3_5_vision_group", plan)

    with pytest.raises(RuntimeError, match="later group rejected"):
        _vision._rpu_vision_forward(
            vision,
            Hidden(),
            torch.tensor([[1, 16, 16], [1, 8, 16]]),
        )

    assert logical_lengths == [256, 128]
    assert mutations == []
    cache.reset_to_position.assert_not_called()
    assert vision._rpu_vision_a6_plans is previous_receipt


def test_vision_graph_disable_is_nonretained_oneshot(monkeypatch):
    source = open(
        os.path.join(
            _REPO, "python/rpu_backend/adapters/qwen3_5/vision.py"
        ),
        encoding="utf-8",
    ).read()
    branch = source.split(
        'if getattr(self, "_rpu_vision_graph_disable", False):', 1
    )[1].split("else:", 1)[0]
    assert "_run_vision_diagnostic_oneshot(self, op_args)" in branch
    assert "_rpu_vision_graph_cache.capture" not in branch

    events = []

    class Graph:
        state_value = 0

        def begin(self):
            events.append("begin")
            self.state_value = 1

        def end(self):
            events.append("end")
            self.state_value = 0

        def abort(self):
            events.append("abort")
            self.state_value = 0

        def state(self):
            return self.state_value

    class GraphCache:
        entries = 2

        def size(self):
            return self.entries

        def capture(self, _signature):
            raise AssertionError("diagnostic path must not enter retained cache")

    @contextmanager
    def oneshot(graph):
        graph.begin()
        try:
            yield
        except BaseException:
            graph.abort()
            raise
        else:
            graph.end()

    graph = Graph()
    cache = GraphCache()
    model = types.SimpleNamespace(
        _rpu_vision_debug_graph=graph,
        _rpu_vision_graph_cache=cache,
    )
    output = object()
    launch = MagicMock(return_value=output)
    monkeypatch.setattr(_vision, "_oneshot_scope", oneshot)
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_forward",
        launch,
        raising=False,
    )

    assert _vision._run_vision_diagnostic_oneshot(model, (7, 11)) is output
    launch.assert_called_once_with(7, 11)
    assert events == ["begin", "end"]
    assert graph.state() == 0
    assert cache.size() == 2


def test_vision_install_requires_planner_ops_before_handle_creation(monkeypatch):
    create = MagicMock(side_effect=AssertionError("must not create handle"))
    convert = MagicMock(side_effect=AssertionError("must not mutate weights"))
    monkeypatch.setattr(
        _vision.torch.ops.rpu, "qwen3_5_vision_create", create, raising=False
    )
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_set_chunk_size_cap",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        _vision, "_convert_vision_block_weights_for_rpu", convert
    )
    with pytest.raises(RuntimeError, match="lacks required planner op"):
        _vision.install_qwen3_5_vision_for_rpu(_runtime_capable_vision())
    create.assert_not_called()
    convert.assert_not_called()


@pytest.mark.parametrize("capacity", [0, -1, True, 4096.0])
def test_vision_install_rejects_invalid_cache_length(monkeypatch, capacity):
    create = MagicMock(side_effect=AssertionError("must not create handle"))
    monkeypatch.setattr(
        _vision.torch.ops.rpu, "qwen3_5_vision_create", create, raising=False
    )
    with pytest.raises(ValueError, match="positive integer"):
        _vision.install_qwen3_5_vision_for_rpu(
            _runtime_capable_vision(), max_seq_len=capacity
        )
    create.assert_not_called()


@pytest.mark.parametrize("capacity", [4097, 16384, 32768])
def test_vision_install_binds_requested_capacity_before_weights(monkeypatch, capacity):
    vision = _runtime_capable_vision()
    destroy = MagicMock()
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_create", lambda: 17, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_destroy", destroy, raising=False)

    def stop(*args, **kwargs):
        torch.ops.rpu.qwen3_5_vision_set_cache_capacity.assert_called_once_with(17, capacity)
        raise RuntimeError("reached weight boundary")

    monkeypatch.setattr(_vision, "_convert_vision_block_weights_for_rpu", stop)
    with pytest.raises(RuntimeError, match="reached weight boundary"):
        _vision.install_qwen3_5_vision_for_rpu(vision, max_seq_len=capacity)
    destroy.assert_called_once_with(17)


def _complete_vision_descriptor():
    from rpu_backend.runtime.execution_planner import (
        _physical_manifest_fingerprint,
    )

    manifest_values = [
        1, 64, 64, 0, 64, 64, 1, 1, 1,
        1, 101, 1, 0, 0,
    ]
    manifest_fingerprint = _physical_manifest_fingerprint(manifest_values)
    stage_descriptor = [
        3, 47, 64, 64, 64, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 64, 64,
        0, 0, 64, 64,
        0, 0, 64, 64,
        0, 64, 0,
        1, 64, 64, 0, 64, 64, 1, 1, 1,
        manifest_fingerprint >> 32,
        manifest_fingerprint & 0xFFFFFFFF,
        1, 101, 1, 0, 0, 0,
    ]
    descriptor = [
        3, 64, 64, 64, 64, 0, 0,
        0, 1, 3, 4, 1, 1, 0, 1, 1, 0,
        0, 64,
        *stage_descriptor,
    ]
    return descriptor, tuple(stage_descriptor)


def test_vision_plan_decoder_preserves_native_physical_descriptor():
    descriptor, stage_descriptor = _complete_vision_descriptor()
    stage = _vision._decode_qwen3_5_vision_plan_descriptor(
        descriptor,
        logical_len=64,
        temporal_num_frames=1,
        camera_batch_count=1,
        compact_input=False,
    )
    assert stage.physical_descriptor == stage_descriptor
    metadata = dict(stage.physical_metadata)
    assert metadata["manifest_state"] == 1
    assert metadata["graph_lifecycle"] == 1


@pytest.mark.parametrize(
    "relative_word, value, match",
    [
        (36, 2, "COMPLETE native physical manifest is inconsistent"),
        (40, 0, "native physical manifest fingerprint is stale"),
    ],
)
def test_vision_plan_decoder_uses_central_manifest_validation(
    relative_word, value, match,
):
    descriptor, _ = _complete_vision_descriptor()
    descriptor[19 + relative_word] = value
    with pytest.raises(ValueError, match=match):
        _vision._decode_qwen3_5_vision_plan_descriptor(
            descriptor,
            logical_len=64,
            temporal_num_frames=1,
            camera_batch_count=1,
            compact_input=False,
        )


@pytest.mark.parametrize("num_heads, expected_tp", [(12, 4), (16, 8)])
def test_vision_attention_tp_covers_complete_heads(num_heads, expected_tp):
    assert _vision._resolve_qwen3_5_vision_attention_tp(num_heads) == expected_tp


def test_vision_preinstall_debug_flags_do_not_look_like_partial_runtime(
    monkeypatch,
):
    vision = _runtime_capable_vision()
    vision._rpu_vision_graph_disable = True
    vision._rpu_vision_rope_disable = True
    impl = MagicMock(side_effect=RuntimeError("reached-impl"))
    monkeypatch.setattr(_vision, "_install_qwen3_5_vision_for_rpu_impl", impl)

    with pytest.raises(RuntimeError, match="reached-impl"):
        _vision.install_qwen3_5_vision_for_rpu(vision)

    impl.assert_called_once()
    assert vision._rpu_vision_graph_disable is True
    assert vision._rpu_vision_rope_disable is True


def test_vision_install_failure_destroys_handle_and_allows_retry(monkeypatch):
    create = MagicMock(return_value=17)
    destroy = MagicMock()
    monkeypatch.setattr(
        _vision.torch.ops.rpu, "qwen3_5_vision_create", create, raising=False)
    monkeypatch.setattr(
        _vision.torch.ops.rpu, "qwen3_5_vision_destroy", destroy, raising=False)
    monkeypatch.setattr(
        _vision,
        "_convert_vision_block_weights_for_rpu",
        MagicMock(side_effect=RuntimeError("conversion-failed")),
    )
    vision = _runtime_capable_vision()
    original_forward = vision.forward

    with pytest.raises(RuntimeError, match="conversion-failed"):
        _vision.install_qwen3_5_vision_for_rpu(vision)

    destroy.assert_called_once_with(17)
    assert not any(name.startswith("_rpu_vision_") for name in vars(vision))
    assert vision.forward == original_forward

    create.side_effect = RuntimeError("retry-reached-create")
    with pytest.raises(RuntimeError, match="retry-reached-create"):
        _vision.install_qwen3_5_vision_for_rpu(vision)
    assert create.call_count == 2
    assert not any(name.startswith("_rpu_vision_") for name in vars(vision))


@pytest.mark.parametrize("interruption", ["pending_handle", "pending_finalizer", "finalizer_create"])
@pytest.mark.parametrize("destroy_fails", [False, True])
def test_actual_vision_pending_ownership_is_published_before_interruptions(monkeypatch, request, interruption, destroy_fails):
    from rpu_backend.api import causal_lm
    from rpu_backend.runtime import _native_retirement

    events, finalizers = [], []
    primary = RuntimeError("vision setup interrupted")
    pending_name = {
        "pending_handle": "_rpu_vision_installing_handle",
        "pending_finalizer": "_rpu_vision_installing_finalizer",
    }.get(interruption)
    base = _runtime_capable_vision()
    class Vision(type(base)):
        def __setattr__(self, name, value):
            if name == pending_name:
                pytest.fail("pending resource ownership must not call user setters")
            super().__setattr__(name, value)
    vision = Vision()
    vars(vision).update(vars(base))
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_create", lambda: 43, raising=False)
    def destroy(handle):
        events.append(handle)
        if destroy_fails:
            raise RuntimeError("vision native retirement failed")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_destroy", destroy, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_set_linear_acc32", MagicMock(side_effect=primary), raising=False)
    original_finalize = _native_retirement.weakref.finalize
    def finalize(*args, **kwargs):
        if interruption == "finalizer_create":
            raise primary
        result = original_finalize(*args, **kwargs)
        finalizers.append(result)
        request.addfinalizer(result.detach)
        return result
    monkeypatch.setattr(_native_retirement.weakref, "finalize", finalize)
    with pytest.raises(RuntimeError, match="vision setup interrupted") as caught:
        _vision.install_qwen3_5_vision_for_rpu(vision)
    assert caught.value is primary
    assert events == [43]
    if destroy_fails:
        assert vision._rpu_vision_installing_handle == 43
        assert causal_lm._LIVE_TERMINAL_REASON is not None
        assert any("vision native retirement failed" in note for note in primary.__notes__)
        assert all(item.alive for item in finalizers)
    else:
        assert not any(name.startswith("_rpu_vision_") for name in vars(vision))
        assert all(not item.alive for item in finalizers)
    for item in finalizers:
        item()
    assert events == [43]


def test_vision_install_admits_candidate_9b_only_after_controlled_opt_in(monkeypatch):
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    cfg = types.SimpleNamespace(
        num_heads=16,
        hidden_size=1152,
        intermediate_size=4304,
        spatial_merge_size=2,
    )
    vision = types.SimpleNamespace(
        blocks=[object()] * 27,
        merger=object(),
        config=cfg,
    )
    create = MagicMock(side_effect=RuntimeError("reached-create"))
    monkeypatch.setattr(
        _vision.torch.ops.rpu,
        "qwen3_5_vision_create",
        create,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="reached-create"):
        _vision.install_qwen3_5_vision_for_rpu(vision)
    create.assert_called_once_with()
    assert not hasattr(vision, "_rpu_vision_handle")


def test_vision_install_is_noop_for_a_live_published_handle(monkeypatch):
    vision = _runtime_capable_vision()
    _mark_complete_vision_runtime(vision)
    impl = MagicMock(side_effect=AssertionError("must not reinstall"))
    monkeypatch.setattr(_vision, "_install_qwen3_5_vision_for_rpu_impl", impl)

    assert _vision.install_qwen3_5_vision_for_rpu(vision) == 23
    impl.assert_not_called()


def test_vision_install_rejects_live_handle_with_incomplete_runtime(monkeypatch):
    vision = _runtime_capable_vision()
    _mark_complete_vision_runtime(vision)
    del vision._rpu_vision_graph_cache
    impl = MagicMock(side_effect=AssertionError("must not reinstall"))
    monkeypatch.setattr(_vision, "_install_qwen3_5_vision_for_rpu_impl", impl)

    with pytest.raises(RuntimeError, match="state is incomplete"):
        _vision.install_qwen3_5_vision_for_rpu(vision)
    impl.assert_not_called()


def test_vision_install_rejects_broken_published_handle(monkeypatch):
    vision = _runtime_capable_vision()
    vision._rpu_vision_handle = 23
    vision._rpu_vision_handle_finalizer = types.SimpleNamespace(alive=False)
    impl = MagicMock(side_effect=AssertionError("must not double-swizzle"))
    monkeypatch.setattr(_vision, "_install_qwen3_5_vision_for_rpu_impl", impl)

    with pytest.raises(RuntimeError, match="state is incomplete"):
        _vision.install_qwen3_5_vision_for_rpu(vision)
    impl.assert_not_called()


def test_vision_rollback_clears_graph_before_handle_finalizer(monkeypatch):
    events = []
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_destroy", lambda _h: events.append("native-destroy"), raising=False)

    class GraphCache:
        def cache_invariant_ok(self):
            return True
        def clear(self):
            events.append("graph-cache-clear")

    class Finalizer:
        alive = True

        def detach(self):
            events.append("finalizer-detach")
            self.alive = False

    vision = _runtime_capable_vision()
    original_forward = vision.forward
    vision._rpu_qwen3_5_original_forward = original_forward
    vision._rpu_qwen3_5_had_instance_forward = False
    assert "forward" not in vars(vision)
    vision.forward = object()
    vision._rpu_vision_installing = True
    vision._rpu_vision_installing_handle = 29
    vision._rpu_vision_installing_finalizer = Finalizer()
    vision._rpu_vision_graph_cache = GraphCache()

    _vision._rollback_qwen3_5_vision_install(vision)

    assert events == ["graph-cache-clear", "native-destroy", "finalizer-detach"]
    assert not any(name.startswith("_rpu_vision_") for name in vars(vision))
    assert "forward" not in vars(vision)
    assert vision.forward == original_forward


def test_vision_rollback_uses_published_finalizer_during_commit_transition(monkeypatch):
    events = []
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_destroy", lambda _h: events.append("native-destroy"), raising=False)

    class GraphCache:
        def cache_invariant_ok(self):
            return True

        def clear(self):
            events.append("graph-cache-clear")

    class Finalizer:
        alive = True

        def detach(self):
            events.append("finalizer-detach")
            self.alive = False

    vision = _runtime_capable_vision()
    original_forward = vision.forward
    vision._rpu_qwen3_5_original_forward = original_forward
    vision._rpu_qwen3_5_had_instance_forward = False
    vision._rpu_vision_installing = True
    vision._rpu_vision_handle = 31
    vision._rpu_vision_handle_finalizer = Finalizer()
    vision._rpu_vision_graph_cache = GraphCache()
    vision.forward = object()

    _vision._rollback_qwen3_5_vision_install(vision)

    assert events == ["graph-cache-clear", "native-destroy", "finalizer-detach"]
    assert not any(name.startswith("_rpu_vision_") for name in vars(vision))
    assert "forward" not in vars(vision)
    assert vision.forward == original_forward


def test_vision_rollback_does_not_double_destroy_dead_finalizer(monkeypatch):
    vision = _runtime_capable_vision()
    vision._rpu_vision_installing = True
    vision._rpu_vision_installing_handle = 37
    vision._rpu_vision_installing_finalizer = types.SimpleNamespace(alive=False)
    destroy = MagicMock()
    monkeypatch.setattr(_vision, "_qwen3_5_vision_destroy_handle", destroy)

    with pytest.raises(RuntimeError, match="dead finalizer"):
        _vision._rollback_qwen3_5_vision_install(vision)

    destroy.assert_not_called()
    assert vision._rpu_vision_installing_handle == 37


def test_failed_vision_retirement_cannot_reenter_ready_or_forward(monkeypatch, request):
    vision = _runtime_capable_vision()
    _mark_complete_vision_runtime(vision)
    finalizer = vision._rpu_vision_handle_finalizer
    request.addfinalizer(finalizer.detach)
    vision._rpu_vision_handle_finalizer = finalizer
    vision._rpu_vision_graph_cache = types.SimpleNamespace(clear=MagicMock(side_effect=RuntimeError("graph clear failed")))
    assert _vision._qwen3_5_vision_runtime_complete(vision)
    with pytest.raises(RuntimeError, match="graph clear failed"):
        _vision._rollback_qwen3_5_vision_install(vision)
    assert finalizer.alive and vision._rpu_vision_handle == 23
    assert not _vision._qwen3_5_vision_runtime_complete(vision)
    with pytest.raises(RuntimeError, match="unsafe process"):
        _vision.install_qwen3_5_vision_for_rpu(vision)
    create = MagicMock(side_effect=AssertionError("terminal process must not create a fresh handle"))
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_create", create, raising=False)
    with pytest.raises(RuntimeError, match="unsafe process"):
        _vision.install_qwen3_5_vision_for_rpu(_runtime_capable_vision())
    create.assert_not_called()
    with pytest.raises(RuntimeError, match="unsafe process"):
        vision.forward(None, None)


def test_partial_vision_weight_conversion_is_poisoned(monkeypatch):
    hidden = 4
    block = types.SimpleNamespace(
        attn=types.SimpleNamespace(
            qkv=torch.nn.Linear(hidden, 3 * hidden),
            proj=torch.nn.Linear(hidden, hidden),
        ),
        mlp=types.SimpleNamespace(
            linear_fc1=torch.nn.Linear(hidden, 2 * hidden),
            linear_fc2=torch.nn.Linear(2 * hidden, hidden),
        ),
    )
    col = MagicMock(side_effect=lambda tensor, **_kwargs: tensor)
    row_calls = []

    def row_swizzle(tensor, **_kwargs):
        row_calls.append(tensor)
        if len(row_calls) == 2:
            raise RuntimeError("fc2-swizzle-failed")
        return tensor

    monkeypatch.setattr(_vision, "tp_col_swizzle_mc_weight", col)
    monkeypatch.setattr(_vision, "tp_row_swizzle_mc_weight", row_swizzle)

    with pytest.raises(RuntimeError, match="fc2-swizzle-failed"):
        _vision._convert_vision_block_weights_for_rpu(block, 1, hidden)
    assert block._rpu_qwen3_5_vision_conversion_started is True
    assert not hasattr(block, "_rpu_qwen3_5_vision_weights_converted")

    call_counts = (col.call_count, len(row_calls))
    with pytest.raises(RuntimeError, match="reload the model"):
        _vision._convert_vision_block_weights_for_rpu(block, 1, hidden)
    assert (col.call_count, len(row_calls)) == call_counts


def test_08b_attention_swizzles_on_tp4_and_mlp_stays_tp8(monkeypatch):
    from rpu_backend.runtime.hw_attrs import (
        install_hw_attr_validator,
        mark_first_forward_done,
    )

    hidden = 12
    intermediate = 24
    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.qkv = torch.nn.Linear(hidden, 3 * hidden)
    block.attn.proj = torch.nn.Linear(hidden, hidden)
    block.mlp = torch.nn.Module()
    block.mlp.linear_fc1 = torch.nn.Linear(hidden, intermediate)
    block.mlp.linear_fc2 = torch.nn.Linear(intermediate, hidden)
    # Text installation has already stamped every Vision child when the
    # first image lazily converts the real block's weights.
    install_hw_attr_validator(block)
    mark_first_forward_done(block)
    col = MagicMock(side_effect=lambda tensor, **_kwargs: tensor)
    row = MagicMock(side_effect=lambda tensor, **_kwargs: tensor)
    monkeypatch.setattr(_vision, "tp_col_swizzle_mc_weight", col)
    monkeypatch.setattr(_vision, "tp_row_swizzle_mc_weight", row)

    _vision._convert_vision_block_weights_for_rpu(block, 12, hidden)

    assert block._rpu_vision_w8a16 is False
    assert block._rpu_qwen3_5_vision_weights_converted is True
    assert not hasattr(block, "_rpu_qwen3_5_vision_conversion_started")

    assert [call.kwargs for call in col.call_args_list] == [
        {"num_cores": 4},
        {"num_cores": 4},
        {"num_cores": 4},
        {},
    ]
    assert [call.kwargs for call in row.call_args_list] == [
        {"num_cores": 4},
        {},
    ]


def test_9b_runtime_padding_preserves_real_lanes_and_zeros_pad(monkeypatch):
    num_heads = 2
    head_dim = 72
    padded_head_dim = 80
    hidden = num_heads * head_dim
    intermediate = 270
    padded_intermediate = 512
    block = types.SimpleNamespace(
        attn=types.SimpleNamespace(
            qkv=torch.nn.Linear(hidden, 3 * hidden),
            proj=torch.nn.Linear(hidden, hidden),
        ),
        mlp=types.SimpleNamespace(
            linear_fc1=torch.nn.Linear(hidden, intermediate),
            linear_fc2=torch.nn.Linear(intermediate, hidden),
        ),
    )
    q_w = block.attn.qkv.weight[:hidden].detach().clone()
    q_b = block.attn.qkv.bias[:hidden].detach().clone()
    o_w = block.attn.proj.weight.detach().clone()
    fc1_w = block.mlp.linear_fc1.weight.detach().clone()
    fc1_b = block.mlp.linear_fc1.bias.detach().clone()
    fc2_w = block.mlp.linear_fc2.weight.detach().clone()

    monkeypatch.setattr(
        _vision, "tp_col_swizzle_mc_weight", lambda tensor, **_kwargs: tensor
    )
    monkeypatch.setattr(
        _vision, "tp_row_swizzle_mc_weight", lambda tensor, **_kwargs: tensor
    )

    _vision._convert_vision_block_weights_for_rpu(block, num_heads, hidden)

    actual_q_w = block._rpu_q_w.view(num_heads, padded_head_dim, hidden)
    actual_q_b = block._rpu_q_b.view(num_heads, padded_head_dim)
    torch.testing.assert_close(actual_q_w[:, :head_dim], q_w.view(num_heads, head_dim, hidden))
    torch.testing.assert_close(actual_q_b[:, :head_dim], q_b.view(num_heads, head_dim))
    assert torch.count_nonzero(actual_q_w[:, head_dim:]).item() == 0
    assert torch.count_nonzero(actual_q_b[:, head_dim:]).item() == 0

    actual_o_w = block.attn.proj.weight.view(hidden, num_heads, padded_head_dim)
    torch.testing.assert_close(actual_o_w[:, :, :head_dim], o_w.view(hidden, num_heads, head_dim))
    assert torch.count_nonzero(actual_o_w[:, :, head_dim:]).item() == 0

    actual_fc1_w = block.mlp.linear_fc1.weight
    actual_fc2_w = block.mlp.linear_fc2.weight
    torch.testing.assert_close(actual_fc1_w[:intermediate], fc1_w)
    torch.testing.assert_close(block._rpu_fc1_b[:intermediate], fc1_b)
    torch.testing.assert_close(actual_fc2_w[:, :intermediate], fc2_w)
    assert tuple(actual_fc1_w.shape) == (padded_intermediate, hidden)
    assert tuple(actual_fc2_w.shape) == (hidden, padded_intermediate)
    assert torch.count_nonzero(actual_fc1_w[intermediate:]).item() == 0
    assert torch.count_nonzero(block._rpu_fc1_b[intermediate:]).item() == 0
    assert torch.count_nonzero(actual_fc2_w[:, intermediate:]).item() == 0

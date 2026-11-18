import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from _module_stubs import real_rpu_env_bool, stub_modules


_REPO = Path(__file__).resolve().parents[1]


def _exec_source(mod_name, rel_path, stubs):
    """exec one repo source file with `stubs` visible in sys.modules."""
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel_path)
    module = importlib.util.module_from_spec(spec)
    with stub_modules(stubs):
        spec.loader.exec_module(module)
    return module


@lru_cache
def _load_runtime_without_rpu():
    vision = ModuleType("rpu_backend.adapters.wall_oss.vision")
    vision.build_wall_oss_vision = None
    action = ModuleType("rpu_backend.adapters.wall_oss.action")
    action.build_wall_oss_action = None
    action._dof_mask_is_row_constant = lambda mask: bool(torch.equal(
        mask[0], mask[0, :1].expand_as(mask[0])
    ))
    llm = ModuleType("rpu_backend.adapters.wall_oss.llm")
    llm.build_wall_oss_llm = None
    checkpoint_validation = ModuleType(
        "rpu_backend.adapters.wall_oss.checkpoint_validation"
    )
    checkpoint_validation.validate_rtc_w4_checkpoint_pair = None
    checkpoint_validation.validate_w8_nvfp4_checkpoint_pair = None
    sorting = ModuleType("rpu_backend.adapters.wall_oss._sorting")
    sorting.sorting_config_from_env = None
    api_execution = ModuleType("rpu_backend.api._execution")
    api_execution.bind_execution_session = None
    api_execution.execution_serialized = lambda fn: fn
    api_execution.native_execution_reconfigure = None
    api_execution.normalize_rpu_execution = None
    api_execution.resolve_component_rpu_execution = None
    execution_planner = ModuleType("rpu_backend.runtime.execution_planner")
    execution_planner.GRAPH_COMPOSITE_CHILD = "composite_child"
    # runtime.py's line 42 is `from rpu_backend.runtime import rpu_env_bool`.
    # A bare ModuleType("rpu_backend") has no __path__, so without this entry
    # that import can only resolve when some EARLIER test in the same process
    # already put the real `rpu_backend.runtime` in sys.modules — i.e. the file
    # passes or fails depending on what ran before it. Stubbing the submodule
    # (the pattern runtime/__init__.py:64-67 documents) makes it self-contained.
    runtime_pkg = ModuleType("rpu_backend.runtime")
    runtime_pkg.rpu_env_bool = real_rpu_env_bool()
    stubs = {
        name: ModuleType(name)
        for name in (
            "rpu_backend",
            "rpu_backend.api",
            "rpu_backend.adapters",
            "rpu_backend.adapters.wall_oss",
        )
    }
    for stub in (
        vision,
        action,
        llm,
        checkpoint_validation,
        sorting,
        api_execution,
        execution_planner,
        runtime_pkg,
    ):
        stubs[stub.__name__] = stub
    return _exec_source(
        "wall_oss_runtime_cpu",
        "python/rpu_backend/adapters/wall_oss/runtime.py",
        stubs,
    )


def test_wall_oss_rope_positions_match_training_contract():
    runtime = _load_runtime_without_rpu()
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2)
    )
    rope = runtime._build_rope_helper(config)

    # grid [1,4,4] / merge 2 expands to four image tokens. The text prefix
    # makes an incorrect tokens_per_second=2 temporal coordinate observable.
    input_ids = torch.tensor(
        [[101, 102, 151655, 151655, 151655, 151655, 103, 104]]
    )
    token_types = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.int32
    )
    positions, delta = rope.get_rope_index(
        input_ids,
        token_types,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        attention_mask=torch.ones_like(input_ids),
    )

    expected = torch.tensor(
        [
            [[0, 1, 2, 2, 2, 2, 4, 5]],
            [[0, 1, 2, 2, 3, 3, 4, 5]],
            [[0, 1, 2, 3, 2, 3, 4, 5]],
        ]
    )
    assert config.vision_config.tokens_per_second == 1
    assert torch.equal(positions, expected)
    assert torch.equal(delta, torch.tensor([[-2]]))


def test_wall_oss_rope_cache_distinguishes_same_length_marker_layouts():
    runtime = _load_runtime_without_rpu()
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2)
    )
    rope = runtime._build_rope_helper(config)
    vla = runtime.WallOssVLA.__new__(runtime.WallOssVLA)
    vla._rope = rope
    vla._rope_cache = {}
    vla._host_cache = True
    pad = runtime.IMAGE_PAD_ID
    ids_a = torch.tensor([[101, 102, pad, pad, pad, pad, 103, 104]])
    ids_b = torch.tensor([[101, 102, 103, pad, pad, pad, pad, 104]])
    grid = torch.tensor([[1, 4, 4]])

    cached_a, _ = vla._rope_index_cached(ids_a, grid, 8, 8)
    cached_b, cached_b_2d = vla._rope_index_cached(ids_b, grid, 8, 8)
    fresh_b, _ = rope.get_rope_index(
        ids_b,
        (ids_b == pad).int(),
        image_grid_thw=grid,
        attention_mask=torch.ones_like(ids_b),
    )

    assert not torch.equal(cached_a, fresh_b)
    assert torch.equal(cached_b, fresh_b)
    assert torch.equal(
        cached_b_2d,
        fresh_b.squeeze(1).transpose(0, 1).contiguous(),
    )


def test_wall_oss_fused_denoise_default_and_opt_out():
    runtime = _load_runtime_without_rpu()
    cases = [
        (None, True, "1"),
        ("0", False, "0"),
        ("false", False, "false"),
        ("False", False, "False"),
        ("", False, ""),
        ("1", True, "1"),
    ]
    for value, expected, persisted in cases:
        initial = {} if value is None else {"RPU_WALL_OSS_FUSED_DENOISE": value}
        with patch.dict(runtime.os.environ, initial, clear=True):
            assert runtime._fused_denoise_enabled() is expected
            assert runtime.os.environ["RPU_WALL_OSS_FUSED_DENOISE"] == persisted


if __name__ == "__main__":
    test_wall_oss_rope_positions_match_training_contract()
    test_wall_oss_fused_denoise_default_and_opt_out()
    print("PASS")

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_host_only_import_is_silent() -> None:
    site_packages = [path for path in sys.path if "site-packages" in path]
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'python')!r}); "
        f"sys.path.extend({site_packages!r}); "
        "import rpu_backend, torch; "
        "assert rpu_backend.__version__; "
        "assert hasattr(torch, 'rpu')"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("RPU_BACKEND_SO", None)
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_nofile_preflight_is_silent_and_best_effort(monkeypatch, capsys) -> None:
    from rpu_backend import _bootstrap_env

    calls = []
    monkeypatch.setattr(
        _bootstrap_env.resource,
        "getrlimit",
        lambda _kind: (1024, 4096),
    )
    monkeypatch.setattr(
        _bootstrap_env.resource,
        "setrlimit",
        lambda kind, value: calls.append((kind, value)),
    )
    _bootstrap_env.ensure_nofile_limit()
    assert calls == [
        (_bootstrap_env.resource.RLIMIT_NOFILE, (4096, 4096))
    ]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""

    def deny_limit_change(*_args):
        raise OSError("denied")

    monkeypatch.setattr(
        _bootstrap_env.resource,
        "setrlimit",
        deny_limit_change,
    )
    _bootstrap_env.ensure_nofile_limit()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_public_api_and_host_graph_lifecycle() -> None:
    import rpu_backend
    from rpu_backend.graph import GraphCache, GraphSignature

    assert rpu_backend.__all__ == [
        "__version__",
        "RPUCache",
        "RPUModelForCausalLM",
        "reset_graph_cache",
    ]

    cache = GraphCache()
    signature = GraphSignature(op_id="host-contract", shapes=(1, 16))
    assert cache.phase == "CONFIGURING"
    assert cache.size() == 0
    assert cache.cache_invariant_ok()
    cache.begin_warmup()
    with cache.capture(signature):
        pass
    assert cache.phase == "WARMING"
    with pytest.raises(RuntimeError, match=r"requires the C\+\+ backend"):
        cache.freeze()


def test_pi05_remap_cache_detects_source_replacement_with_older_mtime(
    monkeypatch, tmp_path,
) -> None:
    from rpu_backend.adapters.pi05 import loader

    source = tmp_path / "model.safetensors"
    cache = tmp_path / "model_remapped.safetensors"
    source.write_bytes(b"make")
    cache.write_bytes(b"remapped-make")
    stats = {
        source: SimpleNamespace(st_mtime_ns=10, st_ctime_ns=10),
        cache: SimpleNamespace(st_mtime_ns=20, st_ctime_ns=20),
    }
    monkeypatch.setattr(
        loader,
        "os",
        SimpleNamespace(
            path=SimpleNamespace(exists=os.path.exists),
            stat=lambda path: stats[Path(path)],
        ),
    )
    assert not loader._cache_is_stale([source], cache)

    source.write_bytes(b"pour")
    stats[source] = SimpleNamespace(st_mtime_ns=9, st_ctime_ns=30)
    assert source.read_bytes() == b"pour"
    assert loader._cache_is_stale([source], cache)


def test_graph_structure_diagnostics_are_safe_and_public() -> None:
    from rpu_backend.graph import Graph

    graph = Graph()

    class FakeGraph:
        @staticmethod
        def dump_replay_plan() -> str:
            return "GraphReplayPlan segments=1"

        @staticmethod
        def dump_tree() -> str:
            return "GraphTree segments=1"

    graph._impl = FakeGraph()
    assert graph.dump_replay_plan() == "GraphReplayPlan segments=1"
    assert graph.dump_tree() == "GraphTree segments=1"

    header = (ROOT / "src/graph/graph_runtime.h").read_text(encoding="utf-8")
    bindings = (ROOT / "src/graph/graph_pybind.cpp").read_text(encoding="utf-8")
    runtime = (ROOT / "src/graph/graph_runtime.cpp").read_text(encoding="utf-8")
    diagnostics = runtime.split("const char* graph_state_name", 1)[1].split(
        "RpuExecutionCoordinator::Claim", 1
    )[0]

    for name in ("dump_replay_plan", "dump_tree"):
        assert f"std::string {name}() const;" in header
        assert f'.def("{name}", &RpuKernelGraph::{name}' in bindings
    for forbidden in (
        "get_kernel_" + "args_instr",
        "instr_" + "data",
        "args_" + "data",
        ".src_" + "addr",
        ".dst_" + "addr",
        ".live_" + "base",
        ".src_dev_" + "addr",
        ".dst_dev_" + "addr",
        "reinterpret_cast",
    ):
        assert forbidden not in diagnostics
    assert "boundary_flush_count=" in diagnostics


def test_external_adapter_registration() -> None:
    from rpu_backend.runtime import registry

    class PublicTestAdapter:
        pass

    architecture = "RhinoForgePublicTestArchitecture"
    try:
        registry.register_adapter(architecture, PublicTestAdapter)
        assert registry.get_adapter(architecture) is PublicTestAdapter
        assert architecture in registry.list_adapters()
        registry.register_adapter(architecture, PublicTestAdapter)
        with pytest.raises(RuntimeError, match="already registered"):
            registry.register_adapter(architecture, type("OtherAdapter", (), {}))
    finally:
        registry.ADAPTERS.pop(architecture, None)


def test_registry_excludes_unsupported_profiles() -> None:
    from rpu_backend import model_registry

    required = {
        "qwen3-0.6b",
        "llama-3.2-1b",
        "qwen3-vl-2b",
        "qwen3-vl-4b",
        "qwen3-vl-8b",
        "qwen3_5-0.8b",
        "dinov3-vit-b",
        "pi05-libero-finetuned",
        "qwen3-14b-w8a16-lmhead-int8",
    }
    excluded = {
        "dinov3-vit-s",
        "dinov3-vit-l",
        "gemma2",
        "paligemma2",
        "paligemma2-3b-pt-224",
        "paligemma2-3b-mix-224",
        "lingbot-vla-4b",
        "qwen3-14b",
        "qwen3-0.6b-w8a16-lmhead-int8",
        "qwen3-1.7b-w8a16-lmhead-int8",
        "qwen3-4b-w8a16-lmhead-int8",
        "qwen3-8b-w8a16-lmhead-int8",
        "qwen3-0.6b-instruct",
        "qwen3-4b-instruct",
    }
    assert required <= model_registry.MODELS.keys()
    assert excluded.isdisjoint(model_registry.MODELS)
    for relative in model_registry.MODELS.values():
        path = Path(relative)
        assert not path.is_absolute()
        assert ".." not in path.parts


def test_qwen3_vl_deepstack_targets_first_text_layers() -> None:
    from types import SimpleNamespace

    from rpu_backend.adapters.qwen3_vl.text import (
        _deepstack_text_layer_indices,
    )

    config = SimpleNamespace(deepstack_visual_indexes=[5, 11, 17])
    assert _deepstack_text_layer_indices(config) == [0, 1, 2]


def test_wall_oss_dual_448_routes_as_two_ordered_single_image_calls(
    monkeypatch,
) -> None:
    from types import MethodType, SimpleNamespace

    import torch

    from rpu_backend.adapters.wall_oss.vision import WallOssVision

    vision = object.__new__(WallOssVision)
    vision._layer_group = 0
    vision._per_window_sdpa = False
    vision._execution_chunk_size = 768
    vision._fused_assemble = False
    vision.cache = SimpleNamespace(max_seq_len=1024)
    calls = []

    def forward_one(self, pixels, grid, fused=False):
        assert self is vision
        assert not fused
        calls.append((int(pixels.size(0)), tuple(grid.reshape(-1).tolist())))
        value = float(len(calls))
        return torch.full((256, 1), value)

    def forward_group(*_args, **_kwargs):
        raise AssertionError("public dual-448 must not use packed Vision")

    vision._forward_one = MethodType(forward_one, vision)
    vision._forward_group = MethodType(forward_group, vision)
    monkeypatch.setenv("RPU_WALL_OSS_BATCH_VISION", "0")
    pixels = torch.cat((torch.zeros(1024, 1), torch.ones(1024, 1)), dim=0)
    grid = torch.tensor([[1, 32, 32], [1, 32, 32]], dtype=torch.long)

    merged = vision.forward(pixels, grid)

    assert calls == [
        (1024, (1, 32, 32)),
        (1024, (1, 32, 32)),
    ]
    assert tuple(merged.shape) == (512, 1)
    assert torch.equal(merged[:256], torch.ones(256, 1))
    assert torch.equal(merged[256:], torch.full((256, 1), 2.0))


def test_wall_oss_multichunk_experimental_routes_stay_fail_closed() -> None:
    from types import SimpleNamespace

    import torch

    from rpu_backend.adapters.wall_oss.vision import WallOssVision

    vision = object.__new__(WallOssVision)
    vision._layer_group = 0
    vision._execution_chunk_size = 768
    vision._per_window_sdpa = True
    vision.cache = SimpleNamespace(max_seq_len=2048)
    one_grid = torch.tensor([[1, 32, 32]], dtype=torch.long)
    with pytest.raises(ValueError, match="per-window SDPA multi-chunk"):
        vision._forward_one(torch.empty(1024, 1), one_grid)

    vision._per_window_sdpa = False
    two_grid = torch.tensor(
        [[1, 32, 32], [1, 32, 32]], dtype=torch.long
    )
    with pytest.raises(ValueError, match="packed multi-image multi-chunk"):
        vision._forward_group(torch.empty(2048, 1), two_grid)


def test_lingbot2_enables_shared_allocator_before_rpu_weights() -> None:
    source = (
        ROOT
        / "python"
        / "rpu_backend"
        / "adapters"
        / "lingbot_vla_v2"
        / "runtime.py"
    ).read_text(encoding="utf-8")
    build = source.split("def _build_lingbot_vla_v2_impl(", 1)[1]
    assert build.index("torch.rpu.set_caching_allocator(True)") < build.index(
        'text_model.to("rpu")'
    )
    close = source.split("    def close(self) -> None:", 1)[1].split(
        "    # ---------------------------------------------------------------- prefix", 1
    )[0]
    assert close.index("self._retire_resources()") < close.index(
        "torch.rpu.empty_cache()"
    )


def test_qwen3_vl_enables_shared_allocator_before_rpu_ownership() -> None:
    source = (
        ROOT
        / "python"
        / "rpu_backend"
        / "adapters"
        / "qwen3_vl"
        / "__init__.py"
    ).read_text(encoding="utf-8")
    install = source.split("    def to_rpu(self):", 1)[1].split(
        "# ------------------------------------------------------------------ #\n"
        "# Top-level forward replacement",
        1,
    )[0]
    enable = install.index("torch.rpu.set_caching_allocator(True)")
    assert enable < install.index("_claim_live_instance(self.model)")
    assert enable < install.index("rpu_backend._cpp_ext.reserve_batch_buffers()")
    assert enable < install.index('text_model.to("rpu")')


def test_shutdown_invalidates_registered_graphs_before_allocator_teardown() -> None:
    source = (ROOT / "src" / "core" / "rpu_backend.cpp").read_text(
        encoding="utf-8"
    )
    shutdown = source.split("void rpu_shutdown()", 1)[1]
    assert shutdown.index("g_rpu_backend_lifecycle = RpuBackendLifecycleState::kShutdown") < shutdown.index(
        "invalidate_registered_rpu_kernel_graphs()"
    )
    assert shutdown.index("invalidate_registered_rpu_kernel_graphs()") < shutdown.index(
        "rpu::RPUCachingAllocator::get().mark_shutdown()"
    )


def test_qwen3_4b_auto_chunk_ceiling_matches_numeric_certificate() -> None:
    from rpu_backend.adapters.qwen3 import lookup_causal_decoder

    envelope = lookup_causal_decoder("qwen3", 36, 2560)
    assert (envelope.max_kv_len, envelope.chunk) == (1024, 128)


def test_qwen3_vl_32b_graph_resource_contract_is_pinned() -> None:
    from rpu_backend.adapters.qwen3_vl.text import lookup_causal_decoder

    envelope = lookup_causal_decoder("qwen3_vl_text", 64, 5120)
    assert (envelope.max_kv_len, envelope.chunk) == (128, 16)

    adapter = (
        ROOT / "python" / "rpu_backend" / "adapters" / "qwen3_vl" / "__init__.py"
    ).read_text(encoding="utf-8")
    runtime_preflight = adapter.split(
        "def _preflight_qwen3_vl_32b_rpu_runtime", 1
    )[1].split("def _check_profile", 1)[0]
    assert (
        "_QWEN3_VL_32B_REQUIRED_LKN_BATCH_CONFIG = (65_536, 8, 64) * 2"
        in adapter
    )
    assert "reserve_batch_buffers =" in runtime_preflight
    assert "get_lkn_batch_config()" in runtime_preflight
    exact_install = adapter.split("    def to_rpu(self):", 1)[1]
    assert exact_install.index("_preflight_qwen3_vl_32b_rpu_runtime(") < (
        exact_install.index("_claim_live_instance(self.model)")
    )
    assert exact_install.index("_claim_live_instance(self.model)") < (
        exact_install.index("reserve_batch_buffers()")
    ) < exact_install.index("_poison_live_instance(")
    assert "GraphCache(max_entries=2)" in exact_install
    assert "GraphCache(max_entries=1)" in exact_install
    assert (
        "text_model._rpu_decoder_graph_cache = bounded_text_cache"
        in exact_install
    )
    assert "_reserved_32b_token=(" in exact_install

    bindings = (ROOT / "src" / "core" / "rpu_pybind.inc").read_text(
        encoding="utf-8"
    )
    assert 'm.def("reserve_batch_buffers"' in bindings
    assert 'm.def("_reserve_batch_buffers"' not in bindings

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "check_cxx_source_compiles(" in cmake
    assert "rhino_lkn::ReserveBatchBuffers()" in cmake
    assert "RHINOFORGE_HAS_LAUNCH_BATCH_RESERVATION" in cmake

    example = tomllib.loads(
        (ROOT / "examples" / "configs" / "qwen3_vl_32b_w8a16.toml").read_text(
            encoding="utf-8"
        )
    )
    env = example["runner"]["env"]
    assert tuple(env[name] for name in (
        "LKN_MAX_BATCH_ENTRIES",
        "LKN_KD_BUF_MB",
        "LKN_INSTR_BUF_MB",
    )) == (65_536, 8, 64)
    assert example["request"]["max_new_tokens"] == 4


@pytest.mark.parametrize(
    ("runtime_case", "message"),
    [
        ("missing", "reserved Graph-buffer support"),
        ("mismatch", "requires unchanged load/current"),
    ],
)
@pytest.mark.parametrize("device", [None, "rpu"])
def test_qwen3_vl_32b_rpu_loader_rejects_runtime_before_weight_stage(
    monkeypatch, runtime_case: str, message: str, device: str | None
) -> None:
    from types import SimpleNamespace

    from rpu_backend.api import conditional_generation
    from rpu_backend.api.errors import RPUBackendError
    from rpu_backend.adapters import qwen3_vl
    from rpu_backend.quant import load as quant_load

    config = SimpleNamespace(
        architectures=["Qwen3VLForConditionalGeneration"]
    )
    monkeypatch.setattr(
        conditional_generation.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        conditional_generation,
        "get_adapter",
        lambda _arch: qwen3_vl.Qwen3VLAdapter,
    )
    monkeypatch.setattr(qwen3_vl, "_check_profile", lambda _config: None)
    monkeypatch.setattr(
        qwen3_vl,
        "_is_qwen3_vl_32b_w8a16_config",
        lambda _config: True,
    )

    if runtime_case == "missing":
        cpp_ext = SimpleNamespace(
            get_lkn_batch_config=lambda: (65_536, 8, 64) * 2,
        )
    else:
        cpp_ext = SimpleNamespace(
            get_lkn_batch_config=lambda: (65_536, 8, 64, 65_536, 8, 63),
            reserve_batch_buffers=lambda: None,
        )
    monkeypatch.setattr(qwen3_vl.rpu_backend, "_cpp_ext", cpp_ext)

    weight_calls = []

    def unexpected_weight_call(name):
        def fail(*_args, **_kwargs):
            weight_calls.append(name)
            raise AssertionError(f"{name} ran before runtime preflight")

        return fail

    monkeypatch.setattr(
        quant_load,
        "stage_w8a16_imagetext_for_rpu",
        unexpected_weight_call("stage_w8a16_imagetext_for_rpu"),
    )
    monkeypatch.setattr(
        quant_load,
        "load_w8a16_imagetext",
        unexpected_weight_call("load_w8a16_imagetext"),
    )
    monkeypatch.setattr(
        conditional_generation.AutoModelForImageTextToText,
        "from_pretrained",
        unexpected_weight_call("AutoModelForImageTextToText.from_pretrained"),
    )

    with pytest.raises(RPUBackendError, match=message):
        kwargs = {} if device is None else {"device": device}
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
            "unused/checkpoint", **kwargs
        )

    assert weight_calls == []


def test_qwen3_vl_32b_rejects_video_and_multiple_images_before_dispatch(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import torch

    from rpu_backend.adapters import qwen3_vl

    class FakeCache:
        pass

    monkeypatch.setattr(qwen3_vl, "RPUCache", FakeCache)
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(_rpu_text_exact_32b_w8a16=True),
            visual=object(),
        ),
        config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=5120),
            vision_config=SimpleNamespace(spatial_merge_size=2),
        ),
    )
    ids = torch.tensor([[1]], dtype=torch.long)
    cache = FakeCache()
    with pytest.raises(NotImplementedError, match="video is outside"):
        qwen3_vl._rpu_qwen3vl_forward(
            model,
            input_ids=ids,
            past_key_values=cache,
            pixel_values_videos=torch.empty(1),
            video_grid_thw=torch.tensor([[1, 16, 16]]),
        )
    with pytest.raises(ValueError, match="exactly one still-image"):
        qwen3_vl._rpu_qwen3vl_forward(
            model,
            input_ids=ids,
            past_key_values=cache,
            pixel_values=torch.empty(1),
            image_grid_thw=torch.tensor([[1, 16, 16], [1, 16, 16]]),
        )
    with pytest.raises(ValueError, match="exactly one still-image"):
        qwen3_vl._rpu_qwen3vl_forward(
            model,
            input_ids=ids,
            past_key_values=cache,
            pixel_values=torch.empty(1),
            image_grid_thw=torch.tensor([[2, 16, 16]]),
        )


def test_qwen3_vl_32b_invalid_request_cannot_consume_vision_cache(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import torch

    from rpu_backend.adapters import qwen3_vl

    class FakeCache:
        def __init__(self, position: int = 0) -> None:
            self.position = position
            self.max_seq_len = 128

    class FakeVision:
        def __init__(self) -> None:
            self.calls = 0
            self.graph_cache = {}

        def __call__(self, _pixels, *, grid_thw):
            self.calls += 1
            self.graph_cache[tuple(grid_thw.reshape(-1).tolist())] = True
            raise RuntimeError("valid request reached Vision BUILD")

    monkeypatch.setattr(qwen3_vl, "RPUCache", FakeCache)
    vision = FakeVision()
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(_rpu_text_exact_32b_w8a16=True),
            visual=vision,
        ),
        config=SimpleNamespace(
            image_token_id=99,
            text_config=SimpleNamespace(hidden_size=5120, vocab_size=128),
            vision_config=SimpleNamespace(spatial_merge_size=2),
        ),
    )
    valid_ids = torch.cat((
        torch.full((1, 64), 99, dtype=torch.long),
        torch.ones((1, 17), dtype=torch.long),
    ), dim=1)
    valid_pixels = torch.empty((256, 1536), dtype=torch.float32)
    valid_grid = torch.tensor([[1, 16, 16]], dtype=torch.long)
    valid_mask = torch.ones_like(valid_ids)
    valid_mm_types = (valid_ids == 99).long()
    valid_kwargs = {
        "input_ids": valid_ids,
        "past_key_values": FakeCache(),
        "attention_mask": valid_mask,
        "pixel_values": valid_pixels,
        "image_grid_thw": valid_grid,
        "mm_token_type_ids": valid_mm_types,
        "logits_to_keep": 1,
    }
    long_ids = torch.cat(
        (valid_ids, torch.ones((1, 48), dtype=torch.long)), dim=1
    )
    noncert_ids = torch.cat((
        torch.full((1, 16), 99, dtype=torch.long),
        torch.ones((1, 65), dtype=torch.long),
    ), dim=1)
    bad_placeholder_ids = torch.cat((
        torch.full((1, 63), 99, dtype=torch.long),
        torch.ones((1, 18), dtype=torch.long),
    ), dim=1)
    split_image_ids = torch.cat((
        torch.full((1, 32), 99, dtype=torch.long),
        torch.ones((1, 1), dtype=torch.long),
        torch.full((1, 32), 99, dtype=torch.long),
        torch.ones((1, 16), dtype=torch.long),
    ), dim=1)
    out_of_range_ids = valid_ids.clone()
    out_of_range_ids[0, -1] = 128
    bad_cases = [
        (
            {
                "input_ids": long_ids,
                "attention_mask": torch.ones_like(long_ids),
                "mm_token_type_ids": (long_ids == 99).long(),
            },
            ValueError,
            "KV-cache horizon",
        ),
        (
            {"image_grid_thw": valid_grid.to(torch.float32)},
            TypeError,
            "integer",
        ),
        (
            {"image_grid_thw": torch.tensor([[1, 0, 16]], dtype=torch.long)},
            ValueError,
            "positive height",
        ),
        (
            {"pixel_values": torch.empty((256, 1), dtype=torch.float32)},
            ValueError,
            "shape",
        ),
        (
            {
                "input_ids": noncert_ids,
                "attention_mask": torch.ones_like(noncert_ids),
                "pixel_values": torch.empty((64, 1536), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 8, 8]], dtype=torch.long),
                "mm_token_type_ids": (noncert_ids == 99).long(),
            },
            ValueError,
            "exact image_grid_thw",
        ),
        (
            {
                "input_ids": bad_placeholder_ids,
                "attention_mask": torch.ones_like(bad_placeholder_ids),
                "mm_token_type_ids": (bad_placeholder_ids == 99).long(),
            },
            ValueError,
            "image token count",
        ),
        (
            {
                "input_ids": split_image_ids,
                "attention_mask": torch.ones_like(split_image_ids),
                "mm_token_type_ids": (split_image_ids == 99).long(),
            },
            ValueError,
            "one contiguous processor group",
        ),
        (
            {
                "input_ids": out_of_range_ids,
                "attention_mask": torch.ones_like(out_of_range_ids),
                "mm_token_type_ids": (out_of_range_ids == 99).long(),
            },
            ValueError,
            "inside the model vocabulary",
        ),
        ({"mm_token_type_ids": None}, ValueError, "mm_token_type_ids"),
        (
            {"mm_token_type_ids": torch.zeros_like(valid_ids)},
            ValueError,
            "mark exactly the image tokens",
        ),
        (
            {"cache_position": torch.arange(81)},
            NotImplementedError,
            "explicit position_ids or cache_position",
        ),
        ({"logits_to_keep": 0}, ValueError, "logits_to_keep=1"),
    ]
    for overrides, error, message in bad_cases:
        kwargs = dict(valid_kwargs)
        kwargs.update(overrides)
        with pytest.raises(error, match=message):
            qwen3_vl._rpu_qwen3vl_forward(model, **kwargs)
        assert vision.calls == 0
        assert vision.graph_cache == {}

    with pytest.raises(RuntimeError, match="reached Vision BUILD"):
        qwen3_vl._rpu_qwen3vl_forward(model, **valid_kwargs)
    assert vision.calls == 1
    assert len(vision.graph_cache) == 1


def test_qwen3_vl_32b_decode_is_bound_to_the_public_four_token_request(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import torch

    from rpu_backend.adapters import qwen3_vl

    class FakeCache:
        max_seq_len = 128

        def __init__(self, position: int) -> None:
            self.position = position

    class FakeEmbedding:
        calls = 0

        def __call__(self, _input_ids):
            self.calls += 1
            raise RuntimeError("valid decode reached embedding")

    monkeypatch.setattr(qwen3_vl, "RPUCache", FakeCache)
    embedding = FakeEmbedding()
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(_rpu_text_exact_32b_w8a16=True),
            visual=object(),
        ),
        config=SimpleNamespace(
            image_token_id=99,
            text_config=SimpleNamespace(hidden_size=5120, vocab_size=128),
            vision_config=SimpleNamespace(spatial_merge_size=2),
        ),
        get_input_embeddings=lambda: embedding,
    )
    invalid = [
        ({"past_key_values": FakeCache(80)}, ValueError, "position 81, 82, or 83"),
        ({"past_key_values": FakeCache(84)}, ValueError, "position 81, 82, or 83"),
        ({"cache_position": torch.tensor([81])}, NotImplementedError, "only input_ids"),
        ({"logits_to_keep": 0}, ValueError, "logits_to_keep=1"),
    ]
    base = {
        "input_ids": torch.ones((1, 1), dtype=torch.long),
        "past_key_values": FakeCache(81),
        "logits_to_keep": 1,
    }
    for overrides, error, message in invalid:
        kwargs = dict(base)
        kwargs.update(overrides)
        with pytest.raises(error, match=message):
            qwen3_vl._rpu_qwen3vl_forward(model, **kwargs)
        assert embedding.calls == 0

    with pytest.raises(RuntimeError, match="valid decode reached embedding"):
        qwen3_vl._rpu_qwen3vl_forward(model, **base)
    assert embedding.calls == 1


def test_qwen3_vl_32b_direct_text_install_is_rejected_before_dispatch() -> None:
    from types import SimpleNamespace

    from rpu_backend.adapters.qwen3_vl import install_qwen3_vl_text_for_rpu

    config = SimpleNamespace(num_hidden_layers=64, hidden_size=5120)
    with pytest.raises(ValueError, match="gated top-level adapter"):
        install_qwen3_vl_text_for_rpu(object(), text_config=config)


def test_qwen3_vl_32b_reserve_failure_releases_claim_without_mutation(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from rpu_backend.adapters import qwen3_vl

    text_model = SimpleNamespace()
    vision_model = SimpleNamespace()
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=5120,
                intermediate_size=25600,
                num_hidden_layers=64,
                num_key_value_heads=8,
            ),
            vision_config=SimpleNamespace(),
        ),
        model=SimpleNamespace(
            language_model=text_model,
            visual=vision_model,
        ),
    )
    adapter = object.__new__(qwen3_vl.Qwen3VLAdapter)
    adapter.model = model
    adapter._rpu_is_ready = False
    adapter._rpu_w8a16_staged_plan = None
    adapter._is_graph_blocked_32b_w8a16 = True

    events = []
    monkeypatch.setattr(qwen3_vl, "_check_profile", lambda _config: None)
    monkeypatch.setattr(
        qwen3_vl,
        "_validate_qwen3_vl_32b_w8a16_model",
        lambda _model: True,
    )
    monkeypatch.setattr(
        qwen3_vl,
        "_claim_live_instance",
        lambda owner: events.append(("claim", owner)),
    )
    monkeypatch.setattr(
        qwen3_vl,
        "_release_live_instance",
        lambda owner: events.append(("release", owner)),
    )
    monkeypatch.setattr(
        qwen3_vl,
        "_poison_live_instance",
        lambda owner, _reason: events.append(("poison", owner)),
    )
    monkeypatch.setattr(
        qwen3_vl.rpu_backend._cpp_ext,
        "get_lkn_batch_config",
        lambda: (65_536, 8, 64) * 2,
        raising=False,
    )

    def fail_reserve():
        events.append(("reserve", model))
        raise RuntimeError("reservation failed")

    monkeypatch.setattr(
        qwen3_vl.rpu_backend._cpp_ext,
        "reserve_batch_buffers",
        fail_reserve,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="reservation failed"):
        adapter.to_rpu()

    assert [event for event, _owner in events] == [
        "claim",
        "reserve",
        "release",
    ]
    assert not hasattr(model, "_rpu_swizzle_started")


def test_native_loader_only_selects_packaged_extension(
    tmp_path: Path, monkeypatch
) -> None:
    from rpu_backend._native_loader import find_shared_object

    package_dir = tmp_path / "python" / "rpu_backend"
    package_dir.mkdir(parents=True)
    external = tmp_path / "build" / "rpu_backend.so"
    external.parent.mkdir()
    external.write_bytes(b"external")
    monkeypatch.setenv("RPU_BACKEND_SO", str(external))

    assert find_shared_object(package_dir=str(package_dir)) is None
    packaged = package_dir / "rpu_backend.so"
    packaged.write_bytes(b"packaged")
    assert find_shared_object(package_dir=str(package_dir)) == str(packaged)


def test_rhinovla_has_no_model_source_execution_loader() -> None:
    package = ROOT / "python" / "rpu_backend" / "adapters" / "rhinovla"
    assert not (package / "checkpoint.py").exists()
    assert "load_action_bundle" not in (package / "__init__.py").read_text(
        encoding="utf-8"
    )


def test_pi05_probe_loader_reuses_only_an_unchanged_file(
    tmp_path: Path, monkeypatch
) -> None:
    import torch

    from rpu_backend.adapters.pi05 import runtime

    path = tmp_path / "noise.pt"
    torch.save(torch.ones(1, 2), path)
    runtime._load_pi05_probe_tensor_snapshot.cache_clear()
    original_load = torch.load
    calls = []

    def counted_load(*args, **kwargs):
        calls.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    try:
        first = runtime._load_pi05_probe_tensor(
            str(path), name="noise", expected_shape=(1, 2)
        )
        second = runtime._load_pi05_probe_tensor(
            str(path), name="noise", expected_shape=(1, 2)
        )
        assert first is second
        assert len(calls) == 1

        previous_mtime = path.stat().st_mtime_ns
        torch.save(torch.full((1, 2), 2.0), path)
        current_mtime = path.stat().st_mtime_ns
        changed_mtime = max(current_mtime, previous_mtime + 1_000_000)
        os.utime(path, ns=(changed_mtime, changed_mtime))
        third = runtime._load_pi05_probe_tensor(
            str(path), name="noise", expected_shape=(1, 2)
        )
        assert len(calls) == 2
        assert torch.equal(third, torch.full((1, 2), 2.0))
    finally:
        runtime._load_pi05_probe_tensor_snapshot.cache_clear()


def test_pi05_prepared_profile_reuses_its_prefix_plan() -> None:
    from rpu_backend.adapters.pi05 import runtime

    class Model:
        pass

    model = Model()
    runtime._PI05_PREPARED_GRAPH_PROFILES[model] = {
        "runtime": {},
        "_prefix_execution_plan": (800, (800, 400)),
    }
    try:
        assert runtime._plan_pi05_prefix_execution(model, 800) == (800, 400)
    finally:
        runtime._PI05_PREPARED_GRAPH_PROFILES.pop(model, None)


def test_pi05_prepared_profile_rejects_prefix_pad16_drift(monkeypatch) -> None:
    from rpu_backend.adapters.pi05 import runtime

    class Model:
        pass

    model = Model()
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(_rpu_chunk_size=400)
            )
        )
    )
    model._rpu_fused_denoise_handle = object()
    monkeypatch.setenv("RPU_PI05_PREFIX_PAD16", "1")
    runtime._PI05_PREPARED_GRAPH_PROFILES[model] = {
        "runtime": runtime._pi05_graph_runtime_profile(model, 10),
        "_prefix_execution_plan": (801, (816, 400)),
    }
    monkeypatch.setenv("RPU_PI05_PREFIX_PAD16", "0")
    try:
        with pytest.raises(RuntimeError, match="prefix_pad16"):
            runtime._validate_prepared_graph_profile(model, 10)
    finally:
        runtime._PI05_PREPARED_GRAPH_PROFILES.pop(model, None)


@pytest.mark.parametrize(
    ("setting", "profiler_enabled", "expected"),
    [(None, False, True), (None, True, False), ("0", False, False)],
)
def test_idle_record_scope_default_is_profiler_safe(
    setting, profiler_enabled, expected, monkeypatch
) -> None:
    import torch

    from rpu_backend.graph import _runtime

    if setting is None:
        monkeypatch.delenv("RPU_SKIP_IDLE_RECORD_FUNCTION", raising=False)
    else:
        monkeypatch.setenv("RPU_SKIP_IDLE_RECORD_FUNCTION", setting)
    monkeypatch.setattr(_runtime, "_SKIP_IDLE_RF_ENV", None)
    monkeypatch.setattr(
        torch.autograd, "_profiler_enabled", lambda: profiler_enabled
    )
    assert _runtime._skip_idle_record_function() is expected

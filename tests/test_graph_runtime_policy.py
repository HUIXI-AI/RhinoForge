"""Host-only contracts for the immutable Graph runtime policy."""

from __future__ import annotations

import importlib.util
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
POLICY_ENV = (
    "RPU_FASTREPLAY_SKIP_SYNC",
    "RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY",
    "RPU_GRAPH_HOST_OP_DEFER_GATE",
    "RPU_GRAPH_DEFER_TO_COPY",
    "RPU_SIGLIP_ISOLATE_PATCH_EMBED",
    "RPU_LINGBOT2_MULTIVIEW_SPM_Z2",
    "RPU_WALL_OSS_FAST_REPLAY",
    "RPU_DEEP_FAST_REPLAY",
)


class _NativePolicy:
    def __init__(self, *values, **options):
        self.values = values
        self.options = options

    def prepare_arenas(self):
        assert self.values[10] is True or (len(self.values) >= 13 and self.values[12] > 0)
        return True


class _NativeCache:
    def __init__(self, max_entries, runtime_policy):
        self.max_entries = max_entries
        self.runtime_policy = runtime_policy


class _Backend:
    GraphHostOpDeferMode = SimpleNamespace(Auto=0, ForceOff=1, ForceOn=2)
    GraphBatchSnapshotMode = SimpleNamespace(Auto=0, Off=1)
    GraphRuntimePolicy = _NativePolicy
    GraphCache = _NativeCache

    @staticmethod
    def get_lkn_batch_config():
        return (262_144, 64, 512, 262_144, 64, 512)


@pytest.fixture
def runtime_module(monkeypatch):
    path = ROOT / "python/rpu_backend/graph/_runtime.py"
    spec = importlib.util.spec_from_file_location("_policy_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.configure_backend(_Backend())
    for name in POLICY_ENV:
        monkeypatch.delenv(name, raising=False)
    return module


def test_tts_predictor_budget_is_cold_and_requires_its_owner(runtime_module):
    policy_type = runtime_module.GraphRuntimePolicy
    policy = policy_type.from_environment(graph_arena_count=6, tts_predictor_group_15=True)
    assert policy._native().options["tts_predictor_group_15"] is True
    assert "tts_predictor_group_15" not in policy_type.from_environment()._native().options
    with pytest.raises(FrozenInstanceError):
        policy.tts_predictor_group_15 = False
    for overrides in ({"graph_arena_count": 0}, {"execution_core_count": 4},
                      {"lingbot2_multiview_spm_z2": True}):
        args = {"graph_arena_count": 6, "tts_predictor_group_15": True, **overrides}
        with pytest.raises(ValueError, match="TTS Predictor"):
            policy_type.from_environment(**args)
    with pytest.raises(TypeError, match="tts_predictor_group_15"):
        policy_type.from_environment(graph_arena_count=6, tts_predictor_group_15=1)


def test_graph_policy_is_frozen_at_cache_creation(runtime_module, monkeypatch):
    monkeypatch.setenv("RPU_FASTREPLAY_SKIP_SYNC", "1")
    monkeypatch.setenv("RPU_WALL_OSS_FAST_REPLAY", "on")
    first = runtime_module.GraphCache()

    monkeypatch.setenv("RPU_FASTREPLAY_SKIP_SYNC", "0")
    monkeypatch.setenv("RPU_WALL_OSS_FAST_REPLAY", "off")
    second = runtime_module.GraphCache()

    assert first.runtime_policy.fast_replay_skip_sync is True
    assert first.runtime_policy.fmb_fast_replay is True
    assert second.runtime_policy.fast_replay_skip_sync is False
    assert second.runtime_policy.fmb_fast_replay is False
    assert first._impl.runtime_policy.values[7:10] == (262_144, 64, 512)
    assert "LKN=extension-load-snapshot" in first.runtime_policy.provenance


def test_default_core_budget_preserves_native_policy_identity(runtime_module):
    default = runtime_module.GraphRuntimePolicy.from_environment()
    explicit = runtime_module.GraphRuntimePolicy.from_environment(
        execution_core_count=8
    )
    assert default == explicit
    assert default.execution_core_count == 8
    assert len(default._native().values) == 11
    assert default._native().values == explicit._native().values
    assert "owner:execution_core_count" not in default.provenance


def test_graph_owner_core_budget_is_frozen_and_reaches_native_cache(runtime_module):
    policy = runtime_module.GraphRuntimePolicy.from_environment(
        execution_core_count=6
    )
    reduced = runtime_module.GraphCache(runtime_policy=policy)
    default = runtime_module.GraphCache()

    assert reduced.runtime_policy is policy
    assert reduced._impl.runtime_policy.values[-5:] == (262_144, 64, 512, False, 6)
    assert len(default._impl.runtime_policy.values) == 11
    assert default.runtime_policy.execution_core_count == 8
    assert "owner:execution_core_count" in policy.provenance
    with pytest.raises(FrozenInstanceError):
        policy.execution_core_count = 8
    with pytest.raises(AttributeError):
        reduced.runtime_policy = default.runtime_policy
    assert reduced.runtime_policy is policy


@pytest.mark.parametrize("value", [True, 0, 9, "6", 6.0])
def test_graph_owner_core_budget_rejects_invalid_values(runtime_module, value):
    with pytest.raises(ValueError, match="execution_core_count"):
        runtime_module.GraphRuntimePolicy.from_environment(
            execution_core_count=value
        )


def test_graph_policy_owner_reference_cannot_be_replaced(monkeypatch):
    import rpu_backend.graph as graph

    monkeypatch.setattr(graph, "so_path", None)
    policy = graph.GraphRuntimePolicy(execution_core_count=6)
    owner = graph.Graph(runtime_policy=policy)
    with pytest.raises(AttributeError):
        owner.runtime_policy = graph.GraphRuntimePolicy()
    assert owner.runtime_policy is policy


@pytest.mark.parametrize("value", ["yes", "01", "garbage"])
def test_graph_policy_rejects_ambiguous_boolean_tokens(
    runtime_module, monkeypatch, value
):
    monkeypatch.setenv("RPU_FASTREPLAY_SKIP_SYNC", value)
    with pytest.raises(ValueError, match="RPU_FASTREPLAY_SKIP_SYNC"):
        runtime_module.GraphCache()


def test_graph_policy_rejects_dual_host_defer_authority(
    runtime_module, monkeypatch
):
    monkeypatch.setenv("RPU_GRAPH_HOST_OP_DEFER_GATE", "auto")
    monkeypatch.setenv("RPU_GRAPH_DEFER_TO_COPY", "auto")
    with pytest.raises(ValueError, match="cannot both be set"):
        runtime_module.GraphCache()


def test_lingbot_owner_override_is_typed_and_does_not_reread_z2(
    runtime_module, monkeypatch
):
    monkeypatch.setenv("RPU_LINGBOT2_MULTIVIEW_SPM_Z2", "garbage")
    policy = runtime_module.GraphRuntimePolicy.from_environment(
        lingbot2_multiview_spm_z2=True
    )
    assert policy.lingbot2_multiview_spm_z2 is True
    assert "owner:lingbot2_multiview_spm_z2" in policy.provenance


@pytest.mark.parametrize("value", [0, 1, "0", "1"])
def test_lingbot_owner_override_rejects_non_boolean_values(
    runtime_module, value
):
    with pytest.raises(TypeError, match="override must be bool"):
        runtime_module.GraphRuntimePolicy.from_environment(
            lingbot2_multiview_spm_z2=value
        )


def test_qwen27b_budget_requires_typed_owner_and_survives_environment_drift(
    runtime_module, monkeypatch
):
    monkeypatch.setenv("RPU_QWEN3_5_SDK_SEGMENT_BUDGET", "1")
    assert not runtime_module.GraphCache().runtime_policy.qwen35_legacy_27b_sdk_budget
    policy = runtime_module.GraphRuntimePolicy.from_environment(
        qwen35_legacy_27b_sdk_budget=True
    )
    cache = runtime_module.GraphCache(runtime_policy=policy)
    monkeypatch.setenv("RPU_QWEN3_5_SDK_SEGMENT_BUDGET", "garbage")
    assert cache.runtime_policy.qwen35_legacy_27b_sdk_budget is True
    assert cache._impl.runtime_policy.values[-1] is True
    assert "owner:qwen35_legacy_27b_sdk_budget=1" in policy.provenance
    assert not runtime_module.GraphCache().runtime_policy.qwen35_legacy_27b_sdk_budget


@pytest.mark.parametrize("value", [0, 1, "0", "1", None])
def test_qwen27b_budget_rejects_untyped_owner(runtime_module, value):
    with pytest.raises(TypeError, match="override must be bool"):
        runtime_module.GraphRuntimePolicy.from_environment(
            qwen35_legacy_27b_sdk_budget=value
        )


def test_qwen27b_arena_preparation_requires_owner_and_reaches_native(runtime_module):
    with pytest.raises(ValueError, match="requires an explicit owner arena plan"):
        runtime_module.GraphRuntimePolicy.from_environment().prepare_arenas()
    policy = runtime_module.GraphRuntimePolicy.from_environment(
        qwen35_legacy_27b_sdk_budget=True
    )
    assert policy.prepare_arenas() is True
    runtime_module.configure_backend(None)
    assert policy.prepare_arenas() is False


@pytest.mark.parametrize("count", [True, False, -1, 4097, 4.0, "4", None])
def test_graph_arena_count_is_a_bounded_integer(runtime_module, count):
    with pytest.raises(ValueError, match="graph_arena_count"):
        runtime_module.GraphRuntimePolicy(graph_arena_count=count)


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_explicit_arena_count_preserves_sdk_snapshot_and_native_argument_order(runtime_module, cores):
    policy = runtime_module.GraphRuntimePolicy.from_environment(
        execution_core_count=cores, graph_arena_count=4)
    cache = runtime_module.GraphCache(max_entries=1, runtime_policy=policy)
    assert cache.runtime_policy is policy
    assert policy._native().values[-6:] == (262_144, 64, 512, False, cores, 4)
    assert len(policy._native().values) == 13
    assert policy.prepare_arenas() is True
    assert "owner:graph_arena_count=4" in policy.provenance
    with pytest.raises(FrozenInstanceError):
        policy.graph_arena_count = 2
    runtime_module.configure_backend(None)
    assert policy.prepare_arenas() is False


def test_explicit_arena_plan_cannot_enable_legacy_segmentation(runtime_module):
    with pytest.raises(ValueError, match="cannot override"):
        runtime_module.GraphRuntimePolicy(qwen35_legacy_27b_sdk_budget=True, graph_arena_count=4)
    assert runtime_module.GraphRuntimePolicy(graph_arena_count=4096).graph_arena_count == 4096


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("snapshot_mode,native_mode", [("auto", 0), ("off", 1)])
def test_modern_27b_uses_a_separate_cold_owner_with_four_arenas(
    runtime_module, cores, snapshot_mode, native_mode
):
    policy = runtime_module.GraphRuntimePolicy.from_environment(
        execution_core_count=cores, graph_arena_count=4,
        qwen35_dense_27b_sdk_budget=True, batch_snapshot_mode=snapshot_mode)
    cache = runtime_module.GraphCache(max_entries=1, runtime_policy=policy)
    assert cache.runtime_policy is policy
    assert cache._impl.runtime_policy.values[-3:] == (False, cores, 4)
    assert cache._impl.runtime_policy.options == {
        "batch_snapshot_mode": native_mode,
        "qwen35_dense_27b_sdk_budget": True,
    }
    assert policy.prepare_arenas()


@pytest.mark.parametrize("arenas,legacy", [(0, False), (3, False), (4, True)])
def test_modern_27b_rejects_incompatible_arena_owners(runtime_module, arenas, legacy):
    with pytest.raises(ValueError):
        runtime_module.GraphRuntimePolicy(graph_arena_count=arenas,
            qwen35_dense_27b_sdk_budget=True, qwen35_legacy_27b_sdk_budget=legacy)


def test_native_graph_execution_has_no_runtime_selector_getenv():
    graph_sources = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "src/graph/graph_runtime.cpp",
            "src/graph/graph_runtime_execute.cpp",
            "src/core/fused_model_base.cpp",
            "src/fused/rpu_qwen3vl_multiview_spm_z2.cpp",
        )
    )
    for name in POLICY_ENV + (
        "LKN_MAX_BATCH_ENTRIES",
        "LKN_KD_BUF_MB",
        "LKN_INSTR_BUF_MB",
        "RPU_QWEN3_5_SDK_SEGMENT_BUDGET",
    ):
        assert f'std::getenv("{name}")' not in graph_sources

    startup = (ROOT / "src/core/rpu_backend.cpp").read_text(encoding="utf-8")
    assert startup.count('parse_lkn_env("LKN_MAX_BATCH_ENTRIES"') == 1
    assert startup.count('parse_lkn_env("LKN_KD_BUF_MB"') == 1
    assert startup.count('parse_lkn_env("LKN_INSTR_BUF_MB"') == 1
    getter = startup[startup.index("std::vector<int64_t> rpu_get_lkn_batch_config()") :]
    getter = getter[: getter.index("\n}\n")]
    assert "read_lkn_batch_config()" not in getter


def test_batch_snapshot_mode_is_instance_scoped_and_ignores_model_env(runtime_module, monkeypatch):
    monkeypatch.setenv("RPU_LINGBOT2_BATCH_SNAPSHOT", "1")
    auto = runtime_module.GraphRuntimePolicy(batch_snapshot_mode="auto")
    off = runtime_module.GraphRuntimePolicy.from_environment(batch_snapshot_mode="off")
    first = runtime_module.GraphCache(runtime_policy=auto)
    second = runtime_module.GraphCache(runtime_policy=off)
    monkeypatch.setenv("RPU_LINGBOT2_BATCH_SNAPSHOT", "0")
    assert first.runtime_policy.batch_snapshot_mode == "auto"
    assert second.runtime_policy.batch_snapshot_mode == "off"
    assert first._impl.runtime_policy.options == {"batch_snapshot_mode": 0}
    assert second._impl.runtime_policy.options == {"batch_snapshot_mode": 1}
    with pytest.raises(FrozenInstanceError):
        auto.batch_snapshot_mode = "off"


@pytest.mark.parametrize("mode", [True, None, 0, "on", "AUTO", "unknown"])
def test_batch_snapshot_mode_rejects_non_modes(runtime_module, mode):
    with pytest.raises(ValueError, match="batch_snapshot_mode"):
        runtime_module.GraphRuntimePolicy(batch_snapshot_mode=mode)


def test_batch_snapshot_policy_keeps_older_native_fallback(runtime_module, monkeypatch):
    legacy = SimpleNamespace(GraphHostOpDeferMode=_Backend.GraphHostOpDeferMode,
                             GraphRuntimePolicy=_NativePolicy)
    runtime_module.configure_backend(legacy)
    policy = runtime_module.GraphRuntimePolicy(batch_snapshot_mode="auto")
    assert policy._native().options == {}

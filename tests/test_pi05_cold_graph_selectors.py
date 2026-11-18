"""Host contracts for Pi0.5 cold selectors and strict installation admission."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import pi05
from rpu_backend.adapters.pi05 import runtime
from rpu_backend.api.errors import RPUConfigError
from rpu_backend.runtime import hw_attrs


ROOT = Path(__file__).resolve().parents[1]
_SELECTORS = (
    ("RPU_PI05_PREFIX_PAD16", "_legacy_prefix_pad16_request",
     "_rpu_legacy_prefix_pad16_request", runtime._cold_prefix_pad16_request),
    ("RPU_PI05_KVINSERT_PAD16", "_legacy_kvinsert_pad16_request",
     "_rpu_legacy_kvinsert_pad16_request", runtime._cold_kvinsert_pad16_request),
)


def _cold_policy():
    # A real Module tree is essential: a SimpleNamespace policy does not make
    # validate_preinstall scan its .model child and missed the board failure.
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.config = SimpleNamespace(
        chunk_size=50, image_features=("camera0", "camera1", "camera2"))
    return policy


@pytest.mark.parametrize("prefix,kvinsert", [(False, False), (False, True),
                                           (True, False), (True, True)])
def test_constructor_keeps_cold_snapshots_outside_uninstalled_module_tree(
    monkeypatch, prefix, kvinsert,
):
    for (environment, _, _, _), value in zip(_SELECTORS, (prefix, kvinsert)):
        monkeypatch.setenv(environment, str(int(value)))
    policy = _cold_policy()
    adapter = pi05.Pi05Adapter(policy)

    hw_attrs.validate_preinstall(policy)
    for (environment, local_name, model_name, _), value in zip(
        _SELECTORS, (prefix, kvinsert)
    ):
        assert getattr(adapter, local_name) is value
        assert model_name not in vars(policy.model)
        monkeypatch.setenv(environment, str(int(not value)))
    # Changing the environment after construction must not change this owner.
    hw_attrs.validate_preinstall(policy)
    assert adapter._legacy_prefix_pad16_request is prefix
    assert adapter._legacy_kvinsert_pad16_request is kvinsert


@pytest.mark.parametrize("selector", _SELECTORS, ids=("prefix", "kvinsert"))
@pytest.mark.parametrize("injection", ("before_constructor", "after_constructor"))
def test_stale_selector_is_rejected_before_claim_and_is_not_adopted(
    monkeypatch, selector, injection,
):
    environment, local_name, model_name, _ = selector
    monkeypatch.setenv(environment, "0")
    policy = _cold_policy()
    if injection == "before_constructor":
        setattr(policy.model, model_name, True)
    adapter = pi05.Pi05Adapter(policy)
    if injection == "after_constructor":
        setattr(policy.model, model_name, True)
    assert getattr(adapter, local_name) is False

    def unexpected_claim(_owner):
        raise AssertionError("stale state reached the live-model claim")

    monkeypatch.setattr(pi05, "_claim_live_instance", unexpected_claim)
    monkeypatch.delenv("RPU_PI05_KEEP_CPU", raising=False)
    with pytest.raises(RPUConfigError, match=model_name):
        adapter.to_rpu()
    assert getattr(policy.model, model_name) is True
    assert not getattr(policy, "_rpu_swizzle_started", False)


def test_selector_registration_is_postinstall_only_and_noise_is_late_state():
    policy = _cold_policy()
    pi05.Pi05Adapter(policy)
    hw_attrs.validate_preinstall(policy)
    for _, _, name, _ in _SELECTORS:
        assert name not in hw_attrs.PUBLIC_HW_ATTRS
        assert name in hw_attrs.INTERNAL_HW_ATTRS_TRANSITIONAL
        setattr(policy.model, name, True)
        with pytest.raises(RPUConfigError, match=name):
            hw_attrs.validate_preinstall(policy)
        hw_attrs.validate_postinstall(policy)
        delattr(policy.model, name)

    policy.model._rpu_noise_seed = 42
    with pytest.raises(RPUConfigError, match="_rpu_noise_seed"):
        hw_attrs.validate_preinstall(policy)
    del policy.model._rpu_noise_seed
    hw_attrs.validate_preinstall(policy)
    hw_attrs.validate_postinstall(policy)
    hw_attrs.install_hw_attr_validator(policy)
    policy.model._rpu_noise_seed = 42
    assert policy.model._rpu_noise_seed == 42
    hw_attrs.validate_postinstall(policy)


@pytest.mark.parametrize("fused", (True, False))
def test_ready_reentry_preserves_installed_snapshots_after_environment_change(
    monkeypatch, fused,
):
    from rpu_backend.tests.acceptance.test_pi05_adapter_install_lifecycle import (
        _complete_policy,
    )

    policy, model, _, _, _ = _complete_policy(fused=fused)
    model._rpu_legacy_prefix_pad16_request = False
    model._rpu_legacy_kvinsert_pad16_request = True
    monkeypatch.setenv("RPU_PI05_PREFIX_PAD16", "1")
    monkeypatch.setenv("RPU_PI05_KVINSERT_PAD16", "0")
    try:
        adapter = pi05.Pi05Adapter(policy)
        assert adapter._rpu_is_ready
        assert adapter._legacy_prefix_pad16_request is False
        assert adapter._legacy_kvinsert_pad16_request is True
        assert adapter.to_rpu() is policy
        assert runtime._cold_prefix_pad16_request(model) is False
        assert runtime._cold_kvinsert_pad16_request(model) is True
    finally:
        runtime._PI05_HANDLES.pop(model, None)


def _source(relative: str) -> str:
    return (ROOT / relative).read_text()


def _section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


@pytest.mark.parametrize("selector", ("qkv_w8a8", "mlp_w8a8"))
@pytest.mark.parametrize("enabled", (False, True))
def test_prepared_pi_graph_rejects_vision_w8a8_drift_before_cache_entry(selector, enabled):
    from rpu_backend.tests.acceptance.test_pi05_adapter_install_lifecycle import _complete_policy

    _, model, vlm, expert, vision = _complete_policy()
    vision._siglip_graph_enabled = vlm._gemma_graph_enabled = True
    expert._adarms_graph_capture_enabled = True
    model._pi05_denoise_graph_enabled = model._pi05_denoise_unroll_enabled = True
    setattr(vision, f"_siglip_{selector}", enabled)
    try:
        frozen = runtime._pi05_graph_runtime_profile(model, 10)
        assert frozen[f"siglip_{selector}"] is enabled
        runtime._PI05_PREPARED_GRAPH_PROFILES[model] = {"runtime": frozen}
        runtime._validate_prepared_graph_profile(model, 10)
        setattr(vision, f"_siglip_{selector}", not enabled)
        with pytest.raises(RuntimeError, match=f"siglip_{selector}"):
            runtime._validate_prepared_graph_profile(model, 10)
        assert runtime._PI05_PREPARED_GRAPH_PROFILES[model]["runtime"] is frozen
    finally:
        runtime._PI05_PREPARED_GRAPH_PROFILES.pop(model, None)
        runtime._PI05_HANDLES.pop(model, None)


def test_pi05_production_graph_routes_use_component_cold_snapshots():
    runtime = _source("python/rpu_backend/adapters/pi05/runtime.py")
    profile = _section(
        runtime,
        "def _pi05_graph_runtime_profile(",
        "def _validate_prepared_graph_profile(",
    )
    fused = _section(
        runtime,
        "def _run_denoise_fused(",
        "def _run_denoise_loop(",
    )
    assert "rpu_env_bool(" not in profile + fused
    prefix_plan = _section(
        runtime,
        "def _plan_pi05_prefix_execution(",
        "def _prefill_prefix_embs(",
    )
    assert "_prefix_pad16_enabled()" not in prefix_plan
    assert prefix_plan.count("_cold_prefix_pad16_request(self)") == 1
    for expression in (
        "vision._siglip_graph_enabled",
        "vlm._gemma_graph_enabled",
        "expert._adarms_graph_capture_enabled",
        "self._pi05_denoise_graph_enabled",
        "self._pi05_denoise_unroll_enabled",
    ):
        assert expression in profile + fused

    gemma = _source("python/rpu_backend/adapters/pi05/gemma.py")
    gemma_forward = _section(
        gemma, "    def _run_gemma_prefill(", "    # P7.1h L1+L2:"
    )
    assert "_gemma_graph_enabled()" not in gemma_forward
    assert "if graph_enabled:" in gemma_forward
    assert "model._gemma_graph_enabled = graph_enabled" in gemma

    adarms = _source("python/rpu_backend/adapters/pi05/adarms.py")
    adarms_forward = _section(
        adarms, "    def rpu_adarms_model_forward(", "    # P7.1h L1+L2:"
    )
    assert "_adarms_graph_enabled()" not in adarms_forward
    assert "_adarms_w8a16_graph_enabled()" not in adarms_forward
    assert "if graph_capture_enabled:" in adarms_forward
    assert "model._adarms_graph_capture_enabled = graph_capture_enabled" in adarms

    adapter = _source("python/rpu_backend/adapters/pi05/__init__.py")
    constructor = _section(
        adapter,
        "    def __init__(self, lerobot_policy: Any, *, graph_runtime_policy=None, max_seq_len=None) -> None:",
        "    def _close_without_session(",
    )
    assert "_rpu_legacy_prefix_pad16_request" in constructor
    assert "_rpu_legacy_kvinsert_pad16_request" in constructor
    assert constructor.index("_prefix_pad16_enabled()") < constructor.index(
        "bind_rpu_execution("
    )
    install = _section(
        adapter,
        "    def _install_fused_denoise_handle(",
        "    def to_rpu(",
    )
    assert "_pi05_denoise_graph_enabled = denoise_graph_enabled" in install
    assert "_pi05_denoise_unroll_enabled = denoise_unroll_enabled" in install


def test_pi05_fast_replay_policy_is_explicit_per_cache():
    adapter = _source("python/rpu_backend/adapters/pi05/__init__.py")
    for name in (
        "RPU_WALL_OSS_FAST_REPLAY",
        "RPU_FASTREPLAY_SKIP_SYNC",
        "RPU_DEEP_FAST_REPLAY",
    ):
        assert f'os.environ.setdefault("{name}"' not in adapter
    assert "graph_runtime_policy = _pi05_graph_runtime_policy(self._graph_runtime_policy_request)" in adapter
    assert adapter.count("runtime_policy=graph_runtime_policy") == 4
    policy = _section(
        adapter,
        "def _pi05_graph_runtime_policy(runtime_policy=None):",
        "def _validate_execution_geometry(",
    )
    assert 'rpu_env_bool("RPU_FASTREPLAY_SKIP_SYNC"' not in policy
    assert "owner:pi05:fast_replay_skip_sync" not in policy

    cache_sites = {
        "python/rpu_backend/adapters/pi05/gemma.py": "gemma_graph_cache",
        "python/rpu_backend/adapters/pi05/adarms.py": "adarms_graph_cache",
        "python/rpu_backend/adapters/siglip.py": "siglip_graph_cache",
    }
    for relative, cache_name in cache_sites.items():
        source = _source(relative)
        assert "*, runtime_policy=None" in source
        assert f"{cache_name} = (" in source
        assert "GraphCache(runtime_policy=runtime_policy)" in source

    denoise_install = _section(
        adapter,
        "    def _install_fused_denoise_handle(",
        "    def to_rpu(",
    )
    assert "*, runtime_policy=None" in denoise_install
    # The denoise cache also sizes one entry per valid language count.
    assert "max_entries=_PI05_DENOISE_GRAPH_ENTRIES,\n                runtime_policy=runtime_policy)" in denoise_install

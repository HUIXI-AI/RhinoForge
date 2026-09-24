"""Host-only source contracts for Pi0.5 graph selector ownership."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text()


def _section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


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
        gemma, "    def _run_gemma_prefill(", "    def rpu_gemma_model_forward("
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
        "    def __init__(self, lerobot_policy: Any) -> None:",
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
    assert "graph_runtime_policy = _pi05_graph_runtime_policy()" in adapter
    assert adapter.count("runtime_policy=graph_runtime_policy") == 4
    policy = _section(
        adapter,
        "def _pi05_graph_runtime_policy():",
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
    assert "GraphCache(runtime_policy=runtime_policy)" in denoise_install

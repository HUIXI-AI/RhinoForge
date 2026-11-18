"""Host-only ownership contract for Graph-stable SDPA mask slots."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_sdpa_mask_cache_is_typed_model_owned_and_destructible():
    decls = _read("src/core/rpu_kernel_decls.h")
    base_header = _read("src/core/fused_model_base.h")
    base_impl = _read("src/core/fused_model_base_impl.h")
    base_cpp = _read("src/core/fused_model_base.cpp")
    sdpa = _read("src/ops/rpu_sdpa.cpp")

    assert "class SdpaStableMaskCache final" in decls
    assert "std::unique_ptr<Impl> pimpl_;" in decls
    assert "SdpaStableMaskCache(const SdpaStableMaskCache&) = delete;" in decls
    assert "SdpaStableMaskCache(SdpaStableMaskCache&&) = delete;" in decls
    assert "SdpaStableMaskCache& stable_cache" in decls

    assert "SdpaStableMaskCache     sdpa_stable_mask_cache_;" in base_impl
    assert "SdpaStableMaskCache& sdpa_stable_mask_cache();" in base_header
    assert "return pimpl_->sdpa_stable_mask_cache_;" in base_cpp

    assert "struct SdpaStableMaskCache::Impl" in sdpa
    assert "std::mutex mutex;" in sdpa
    assert "std::map<Key, at::Tensor> slots;" in sdpa
    assert "SdpaStableMaskCache::~SdpaStableMaskCache() = default;" in sdpa
    assert "thread_local" not in sdpa
    assert "stable_owner" not in sdpa
    assert "new StableMaskSlots" not in sdpa
    assert "reinterpret_cast<uintptr_t>" not in sdpa


def test_every_production_mask_builder_uses_the_model_cache():
    expected_calls = {
        "src/fused/rpu_adarms_model.cpp": 1,
        "src/fused/rpu_gemma2_model.cpp": 1,
        "src/fused/rpu_gemma4_model.cpp": 1,
        "src/fused/rpu_gemma_model.cpp": 1,
        "src/fused/rpu_gr00t_dit_model.cpp": 2,
        "src/fused/rpu_halo_action_expert_model.cpp": 0,
        "src/fused/rpu_pi05_denoise_step_model.cpp": 1,
        "src/fused/rpu_qwen25vl_vision_model.cpp": 3,
        "src/fused/rpu_qwen3_model.h": 1,
        "src/fused/rpu_qwenpi05_model.cpp": 1,
        "src/fused/rpu_rhino_vla_model.cpp": 1,
    }
    for relative, count in expected_calls.items():
        source = _read(relative)
        assert source.count("sdpa_stable_mask_cache()") == count, relative

    fused_sources = "\n".join(
        path.read_text()
        for path in (ROOT / "src/fused").glob("*.cpp")
    ) + _read("src/fused/rpu_qwen3_model.h")
    assert "stable_owner" not in fused_sources

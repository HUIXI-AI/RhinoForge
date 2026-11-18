"""Actual cold envelope wrappers/profile on CPU doubles, not hardware evidence."""
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _body(source, signature):
    begin = source.index("{", source.index(signature))
    return source[begin:source.index("\n}", begin) + 2]


FAMILIES = ("siglip", "qwenpi05", "rhino_vla", "pi05_denoise_step")


def test_siglip_cold_constructor_keeps_a8_separate_from_ln_streaming(tmp_path):
    """Execute the native constructor guard without a device or model weights."""
    from test_siglip_attention_dry_live_contract import _function

    native = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    constructor = _function(native, "    explicit SigLIPModel(")
    code = r'''
#include <cassert>
#include <optional>
#include <stdexcept>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
bool stream = false;
struct SigLIPModel {
    std::optional<bool> configured_linear_acc32_;
    bool linear_acc32_, qkv_w8a8_, mlp_w8a8_;
    bool pi05_vision_owner_ln_opt_in_ = true;
    bool pi05_vision_kv_spm_only_opt_in_ = true;
    bool pi05_ring_xor3_opt_in_ = true;
    bool pi05_vision_fc_weight_outer_opt_in_ = false;
    bool pi05_vision_ln_stream_opt_in_ = stream;
''' + constructor + r'''
};
int main() {
    for (bool streaming : {false, true}) {
        stream = streaming;
        for (bool qkv : {false, true}) for (bool mlp : {false, true}) {
            bool rejected = false;
            try { SigLIPModel model(false, qkv, mlp); }
            catch (const std::runtime_error&) { rejected = true; }
            assert(rejected == (streaming && (qkv || mlp)));
        }
    }
}
'''
    source, binary = tmp_path / "siglip_cold.cpp", tmp_path / "siglip_cold"
    source.write_text(code)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
        str(source), "-o", str(binary)], capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("family", FAMILIES)
def test_real_siglip_envelope_wrapper_publishes_actual_capacity(tmp_path, family):
    native = (ROOT / f"src/fused/rpu_{family}_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
struct FusedModelBase {
    struct ChunkEnvelope { int64_t max_kv_len=0, chunk=0; };
    struct Impl {
        ChunkEnvelope chunk_envelope_;
        std::vector<int> kv_cost_candidates_;
        std::optional<int> kv_cost_planning_context_;
        int kv_cost_layout_scope_=0;
        std::string kv_cost_snapshot_reason_;
        int64_t planner_cache_epoch_=0;
        std::vector<int> prepared_stage_candidates_;
        std::vector<int> prepared_spm_costs_;
    } impl;
    Impl* pimpl_=&impl;
    int64_t last=0;
    int64_t num_layers() const { return 27; }
    int64_t hidden_size() const { return 1152; }
    int64_t intermediate_size() const { return 4352; }
    int64_t num_q_heads() const { return 16; }
    int64_t num_kv_heads() const { return 16; }
    int64_t head_dim() const { return 80; }
    int64_t attn_tp() const { return 8; }
    int64_t get_last_resolved_chunk_size() const { return last; }
    std::vector<int64_t> kvinsert_cost_weight_identity() const { return {1}; }
    const ChunkEnvelope& chunk_envelope() const { return impl.chunk_envelope_; }
    void set_chunk_envelope(int64_t max_kv_len, int64_t chunk);
    void invalidate_planner_cache();
    std::vector<int64_t> installed_profile_identity() const;
} model;
void FusedModelBase::set_chunk_envelope(int64_t max_kv_len, int64_t chunk) __SETTER__
void FusedModelBase::invalidate_planner_cache() __INVALIDATE__
std::vector<int64_t> FusedModelBase::installed_profile_identity() const __PROFILE__
struct SigLIPRegistry {
    static FusedModelBase* get(int64_t handle, const char* caller) {
        TORCH_CHECK(handle==42, "unknown handle");
        assert(std::string(caller)=="rpu_siglip_set_chunk_envelope");
        return &model;
    }
};
void rpu_siglip_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) __WRAPPER__
template<class F> void reject(F run) {
    bool rejected=false;
    try { run(); } catch (const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
int main() {
    for (int64_t capacity : {256, 768, 2048}) {
        const auto previous_epoch=model.impl.planner_cache_epoch_;
        model.impl.prepared_stage_candidates_={1};
        model.impl.prepared_spm_costs_={1};
        rpu_siglip_set_chunk_envelope(42,capacity,0);
        assert(model.impl.planner_cache_epoch_==previous_epoch+1);
        assert(model.impl.prepared_stage_candidates_.empty());
        assert(model.impl.prepared_spm_costs_.empty());
        const auto epoch=model.impl.planner_cache_epoch_;
        model.impl.prepared_stage_candidates_={2};
        model.impl.prepared_spm_costs_={2};
        rpu_siglip_set_chunk_envelope(42,capacity,0);
        assert(model.impl.planner_cache_epoch_==epoch);
        assert(model.impl.prepared_stage_candidates_==std::vector<int>{2});
        assert(model.impl.prepared_spm_costs_==std::vector<int>{2});
        const auto profile=model.installed_profile_identity();
        assert(profile[8]==capacity && profile[9]==0);
        for (auto word : profile) std::cout << word << ' ';
        std::cout << '\n';
        reject([&] { rpu_siglip_set_chunk_envelope(43,capacity,0); });
        reject([&] { rpu_siglip_set_chunk_envelope(42,0,0); });
        reject([&] { rpu_siglip_set_chunk_envelope(42,capacity,15); });
        model.last=16;
        reject([&] { rpu_siglip_set_chunk_envelope(42,capacity+16,0); });
        assert(model.installed_profile_identity()==profile);
        assert(model.impl.planner_cache_epoch_==epoch);
        model.last=0;
    }
}
'''
    for marker, source, signature in (
        ("__SETTER__", base, "void FusedModelBase::set_chunk_envelope("),
        ("__INVALIDATE__", base, "void FusedModelBase::invalidate_planner_cache("),
        ("__PROFILE__", base, "std::vector<int64_t> FusedModelBase::installed_profile_identity()"),
        ("__WRAPPER__", native, f"void rpu_{family}_set_chunk_envelope("),
    ):
        cpp = cpp.replace(marker, _body(source, signature))
    registry = {"siglip": "SigLIPRegistry", "qwenpi05": "QwenPI05Registry",
                "rhino_vla": "RhinoVLARegistry", "pi05_denoise_step": "Pi05DenoiseStepRegistry"}[family]
    cpp = cpp.replace("SigLIPRegistry", registry).replace("rpu_siglip_", f"rpu_{family}_")
    source_path, binary = tmp_path / "siglip_profile.cpp", tmp_path / "siglip_profile"
    source_path.write_text(cpp)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
        str(source_path), "-o", str(binary)], capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    profiles = [[int(word) for word in row.split()] for row in result.stdout.splitlines()]
    assert [words[8:10] for words in profiles] == [[256, 0], [768, 0], [2048, 0]]


@pytest.mark.parametrize("family", FAMILIES)
def test_envelope_op_is_registered_separately_from_hot_weight_refresh(family):
    registration = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    assert f'm.def("{family}_set_chunk_envelope", TORCH_FN(rpu_{family}_set_chunk_envelope));' in registration
    header = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    assert f"void rpu_{family}_set_chunk_envelope(" in header
    native = (ROOT / f"src/fused/rpu_{family}_model.cpp").read_text()
    assert native.count("set_chunk_envelope(") == 2  # thin op + its one FMB call
    for name in (f"rpu_{family}_set_weights(",) + (("rpu_siglip_set_weights_w8a16(",) if family == "siglip" else ()):
        assert "set_chunk_envelope" not in _body(native, "void " + name)

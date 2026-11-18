"""Preserve DINO component admission and native normalization route contracts."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_dino_debug_exports_do_not_snapshot_before_capture_executes():
    dino = _source("src/fused/rpu_dinov3_vision_model.cpp")
    begin = dino.index("if (get_debug_export() && per_layer_debug_buf_.defined())")
    end = dino.index("publish_per_layer_qkv(post_sdpa_debug_buf_", begin)
    publication = dino[begin:end]
    # DMA probes execute at the surrounding GraphCache capture boundary.
    # Cloning here returns zeros on BUILD and the previous frame on REPLAY.
    assert "g_debug_tensors[key] = per_layer_debug_buf_[L];" in publication
    assert "g_debug_tensors[key] = buf;" in publication
    assert "g_debug_tensors[key] = buf[L];" in publication
    assert ".clone()" not in publication
    assert "rpu_ddr_flush(" not in publication


def test_standalone_dino_profile_fails_before_asset_io(monkeypatch):
    from rpu_backend.adapters.internvla_n1 import rgbd_rpu
    monkeypatch.setattr(rgbd_rpu, "require_controlled_evaluation", lambda manifest: None)
    monkeypatch.setattr(rgbd_rpu, "verify_asset_manifest", lambda *a: pytest.fail("asset I/O reached"))
    with pytest.raises(ValueError, match="must be a bool"):
        rgbd_rpu.RGBDTowerRPU(None, standalone_compatibility="true", asset_paths=("source",))
    with pytest.raises(ValueError, match="RGB tower"):
        rgbd_rpu.RGBDTowerRPU(None, standalone_compatibility=True, component_id="depth_tower", asset_paths=("source",))


def test_standalone_dino_shape_fails_before_patch_or_device_access():
    import torch
    from rpu_backend.adapters.internvla_n1.rgbd_rpu import RGBDTowerRPU
    tower = RGBDTowerRPU.__new__(RGBDTowerRPU)
    tower.handle, tower._standalone_compatibility = 1, True
    with pytest.raises(ValueError, match="one 224x224"):
        tower.forward(torch.ones(2, 3, 224, 224))


def test_native_standalone_norm_profile_is_cold_and_descriptor_owned(tmp_path):
    import re
    import shutil
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = _source("src/fused/rpu_dinov3_vision_model.cpp")
    def method(name):
        start = source.index("    void " + name + "(bool enabled) {")
        end = source.index("\n    }", start) + len("\n    }")
        return source[start:end]
    routes = source[source.index("        const int64_t norm_route = standalone_compatibility_ ? 2 : 1;"):
                    source.index("        if (!identity_rope_) {", source.index("        const int64_t norm_route = standalone_compatibility_ ? 2 : 1;"))]
    consumers = re.findall(
        r"ctx\(\)\.consume_physical_route\(\s*FmbRouteFamily::NORMALIZATION,\s*"
        r"DINO_(?:LN1|LN2|FINAL_NORM)_SITE,[^;]+;", source
    )
    assert len(consumers) == 3
    base = _source("src/core/fused_model_base.h")
    signature_start = base.index("    const FmbRouteManifestEntry& consume_physical_route(")
    signature_end = base.index(") const {", signature_start) + len(") const")
    consumer_signature = base[signature_start:signature_end]
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
enum class FmbRouteFamily { NORMALIZATION };
namespace at { using IntArrayRef = std::vector<int64_t>; enum ScalarType { kHalf, kByte }; }
struct FmbRouteManifestEntry {};
struct Context {
    mutable std::vector<int64_t> consumed;
    FmbRouteManifestEntry entry;
    __CONSUMER_SIGNATURE__ {
        assert(family == FmbRouteFamily::NORMALIZATION);
        assert(resolved_flags == 0 && resolved_arguments.empty() && invocation == 0);
        consumed.push_back(site_id);
        consumed.push_back(resolved_selector);
        return entry;
    }
};
__SITE_CONSTANTS__
struct Model {
    Context context;
    Context& ctx() { return context; }
    int layers=12, hidden=384, intermediate=1536, heads=8, hd=64, resolved=0, generation=0;
    bool standalone_compatibility_=false, identity_rope_=true, use_acc32_=true, use_16b_sdpa_=true;
    bool has_final_norm_=true;
    at::ScalarType projection_type() const { return at::kHalf; }
    int get_last_resolved_chunk_size() const { return resolved; }
    int num_layers() const { return layers; }
    int hidden_size() const { return hidden; }
    int intermediate_size() const { return intermediate; }
    int num_q_heads() const { return heads; }
    int head_dim() const { return hd; }
    void invalidate_model_state() { ++generation; }
    __SETTERS__
    std::vector<int64_t> manifest() {
        std::vector<int64_t> result;
        auto append=[&](FmbRouteFamily, int64_t site, int64_t selector) { result.push_back(site); result.push_back(selector); };
        __ROUTES__
        return result;
    }
    std::vector<int64_t> consume() {
        context.consumed.clear();
        const int64_t norm_route = standalone_compatibility_ ? 2 : 1;
        __CONSUMERS__
        return context.consumed;
    }
};
template<class F> void reject(F action) {
    bool caught=false; try { action(); } catch(const std::runtime_error&) { caught=true; }
    assert(caught);
}
int main() {
    Model m; auto baseline=m.manifest();
    assert(m.consume() == baseline);
    m.set_standalone_compatibility(true);
    assert(m.generation==1 && m.manifest()!=baseline);
    assert((m.manifest()==std::vector<int64_t>{
        DINO_LN1_SITE,2,DINO_LN2_SITE,2,DINO_FINAL_NORM_SITE,2}));
    assert(m.consume() == m.manifest());
    reject([&]{m.set_acc32(false);});
    reject([&]{m.set_16b_sdpa(false);});
    reject([&]{m.set_identity_rope(false);});
    m.resolved=257;
    reject([&]{m.set_standalone_compatibility(false);});
    for (int axis=0;axis<8;++axis) {
        Model invalid;
        if(axis==0) invalid.layers=24;
        if(axis==1) invalid.hidden=768;
        if(axis==2) invalid.intermediate=1024;
        if(axis==3) invalid.heads=6;
        if(axis==4) invalid.hd=128;
        if(axis==5) invalid.identity_rope_=false;
        if(axis==6) invalid.use_acc32_=false;
        if(axis==7) invalid.use_16b_sdpa_=false;
        reject([&]{invalid.set_standalone_compatibility(true);});
        assert(invalid.generation==0 && !invalid.standalone_compatibility_);
    }
}
'''.replace("__SETTERS__", "\n".join(method(name) for name in (
        "set_standalone_compatibility", "set_acc32", "set_16b_sdpa", "set_identity_rope")))
    cpp = cpp.replace("__ROUTES__", routes)
    cpp = cpp.replace("__CONSUMER_SIGNATURE__", consumer_signature)
    cpp = cpp.replace("__CONSUMERS__", "\n".join(consumers))
    constants = re.findall(
        r"constexpr int64_t DINO_(?:LN1|LN2|FINAL_NORM)_SITE = \d+LL;", source
    )
    assert len(constants) == 3
    cpp = cpp.replace("__SITE_CONSTANTS__", "\n".join(constants))
    path = tmp_path / "norm_profile.cpp"
    path.write_text(cpp)
    binary = tmp_path / "norm_profile"
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True)
    weight_identity = source[source.index("std::vector<int64_t> kvinsert_cost_weight_identity()"):
                             source.index("int64_t hybrid_layer_count_")]
    assert "static_cast<int64_t>(standalone_compatibility_)" in weight_identity

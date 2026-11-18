"""Execute SigLIP's real planning/consumption code with host-only shape state."""
from pathlib import Path
from pi_wall_host_catalog import with_actual_linear_contracts
import re
import shutil
import subprocess

import pytest

from _native_runtime_identity import runtime_identity_include


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("qkv,mlp", [(True, False), (False, True), (True, True)])
def test_siglip_w4_rejects_a8_before_weight_validation_or_conversion(monkeypatch, qkv, mlp):
    from rpu_backend.adapters import siglip

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported W4/A8 install reached weight validation or conversion")

    monkeypatch.setattr(siglip, "_validate_siglip_w8a8", unexpected)
    monkeypatch.setattr(siglip, "_convert_siglip_weights_for_rpu", unexpected)
    with pytest.raises(ValueError, match="W8A8 requires W8A16 weights"):
        siglip.patch_siglip_model_for_rpu_all_layers_once(
            object(), object(), num_images=2, num_cores=8,
            w8a16_default=True, linear_acc32=False, w4a16=True,
            qkv_w8a8=qkv, mlp_w8a8=mlp)


def _function(source, marker):
    start = source.index(marker)
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda match: " " * len(match[0]), source)
    brace = masked.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end]


def _run_native(tmp_path, source, torch, *, extra_sources=()):
    compiler = shutil.which("c++")
    assert compiler, "native SigLIP contract requires a C++ compiler"
    cpp, binary = tmp_path / "siglip_cold.cpp", tmp_path / "siglip_cold"
    cpp.write_text(with_actual_linear_contracts(source))
    torch_root = Path(torch.__file__).resolve().parent
    command = [compiler, "-std=c++17", "-O0", "-D_GLIBCXX_USE_CXX11_ABI=" +
               str(int(torch.compiled_with_cxx11_abi())), "-I", str(ROOT / "src"), "-I", str(runtime_identity_include(tmp_path)),
               "-I", str(torch_root / "include"), "-I", str(torch_root / "include/torch/csrc/api/include"),
               str(cpp), *map(str, extra_sources),
               "-L", str(torch_root / "lib"), "-Wl,-rpath," + str(torch_root / "lib"),
               "-ltorch_cpu", "-lc10", "-o", str(binary)]
    built = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_siglip_qkv_w8a8_actual_manifest_and_layout_identity(tmp_path, monkeypatch):
    """Run native admission/identity code; only installed-payload lookup is a double."""
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    owner = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    constants = "\n".join(line for line in owner.splitlines()
                          if line.startswith("constexpr int64_t SIGLIP_"))
    functions = "\n".join(_function(owner, marker).replace(" const override", " const") for marker in (
        "    std::vector<int64_t> pi05_vision_qkv_w8a8_identity(",
        "    void require_pi05_vision_qkv_w8a8_profile(",
        "    void validate_pi05_vision_qkv_w8a8_manifest(",
        "    int64_t subclass_layout_hash(",
        "    int64_t subclass_layout_hash_without_ln_stream(",
        "    int64_t ln_stream_elems(",
    ))
    source = r'''
#include "core/fused_model_base.h"
#include "fused/rpu_siglip_w8a8_contract.h"
#include <cassert>
#include <iostream>
using namespace v3;
enum class KernelId : uint16_t { LAYER_NORM_QUANTIZE_W8A8_H1152, LINEAR };
struct KernelCache {
    bool norm=true, linear=true;
    static KernelCache& instance() { static KernelCache cache; return cache; }
    bool has_loaded(KernelId id) const {
        return id==KernelId::LAYER_NORM_QUANTIZE_W8A8_H1152 ? norm : linear;
    }
};
''' + constants + r'''
namespace v3 {
struct Owner {
    bool qkv_w8a8_=true,mlp_w8a8_=false,linear_acc32_=false,audio_mode_=false;
    int cores=8;
    int64_t image_batch_count_=2,orig_head_dim_=72,projection_dim_=2048;
    double eps_=1e-6;
    std::optional<bool> configured_linear_acc32_=false;
    std::vector<int> layer_weights_=std::vector<int>(27),layer_bias_norm_=std::vector<int>(27);
    bool pi05_vision_fc_weight_outer_opt_in_=false,pi05_vision_kv_spm_only_opt_in_=false;
    bool pi05_vision_owner_ln_opt_in_=false,pi05_ring_xor3_opt_in_=false;
    bool pi05_vision_merge_segment_opt_in_=false,pi05_vision_q_resident_opt_in_=false;
    bool pi05_vision_ln_stream_opt_in_=false;
    at::Tensor audio_proj2_w_;
    int num_cores() const {return cores;}
    int attn_tp() const {return cores;}
    int mlp_tp() const {return cores;}
    int64_t num_layers() const {return layer_weights_.size();}
    int64_t hidden_size() const {return 1152;}
    int64_t intermediate_size() const {return 4352;}
    int64_t num_q_heads() const {return 16;}
    int64_t num_kv_heads() const {return 16;}
    int64_t head_dim() const {return 80;}
    std::vector<int64_t> core_profile_arguments() const {return {cores};}
    std::vector<int64_t> pi05_vision_fc_arguments(bool) const {return {};}
    std::vector<int64_t> pi05_vision_owner_ln_arguments(bool,int64_t) const {return {};}
    std::vector<int64_t> pi05_vision_mlp_w8a8_identity() const {return {};}
    KernelId pi05_vision_qkv_w8a8_linear_id(int64_t) const {return KernelId::LINEAR;}
    void require_pi05_vision_qkv_w8a8_payloads(int64_t) const {
        TORCH_CHECK(KernelCache::instance().norm && KernelCache::instance().linear,"missing payload");
    }
''' + functions + r'''
};
} // Native owner methods retain their production namespace.
template<class F> void rejects(F run) {
    bool rejected=false;
    try {run();} catch(const c10::Error&) {rejected=true;}
    catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
}
FmbPhysicalExecutionManifest manifest(int64_t rows,bool enabled) {
    FmbPhysicalExecutionManifest m;
    m.state=FmbPhysicalManifestState::COMPLETE;
    m.logical_length=m.physical_length=m.kv_logical_length=m.kv_insert_physical_rows=rows;
    m.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
    m.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
    if(enabled)m.routes.push_back({SIGLIP_LN1_QUANT_W8A8_SITE,FmbRouteFamily::NORMALIZATION,
        SIGLIP_LN1_QUANT_W8A8_ROUTE,0,siglip_qkv_w8a8::norm_arguments(rows),0});
    for(auto site:{SIGLIP_Q_LINEAR_SITE,SIGLIP_K_LINEAR_SITE,SIGLIP_V_LINEAR_SITE})
        m.routes.push_back({site,FmbRouteFamily::LINEAR,
            enabled?SIGLIP_QKV_W8A8_LINEAR_ROUTE:static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            0,enabled?siglip_qkv_w8a8::linear_arguments(rows):std::vector<int64_t>{},0});
    return m;
}
int main() {
    for(int64_t rows:{512,768}) {
        Owner on,off;on.image_batch_count_=off.image_batch_count_=rows/256;off.qkv_w8a8_=false;
        auto correct=manifest(rows,true),legacy=manifest(rows,false);
        on.validate_pi05_vision_qkv_w8a8_manifest(correct);
        off.validate_pi05_vision_qkv_w8a8_manifest(legacy);
        rejects([&]{on.validate_pi05_vision_qkv_w8a8_manifest(legacy);});
        rejects([&]{off.validate_pi05_vision_qkv_w8a8_manifest(correct);});
        assert(on.subclass_layout_hash()!=off.subclass_layout_hash());
        const auto loaded_identity=on.subclass_layout_hash();
        KernelCache::instance().norm=false;
        assert(on.subclass_layout_hash()!=loaded_identity);
        rejects([&]{on.validate_pi05_vision_qkv_w8a8_manifest(correct);});
        KernelCache::instance().norm=true;
        // Every fused-LN/quant ABI field and every selected PL tile/flag is bound.
        for(int route:{0,1})for(size_t field=0;field<correct.routes[route].arguments.size();++field) {
            auto wrong=correct;++wrong.routes[route].arguments[field];
            rejects([&]{on.validate_pi05_vision_qkv_w8a8_manifest(wrong);});
        }
        for(int route=0;route<4;++route)for(int field=0;field<3;++field) {
            auto wrong=correct;
            if(field==0)++wrong.routes[route].selector;
            if(field==1)++wrong.routes[route].flags;
            if(field==2)++wrong.routes[route].invocation;
            rejects([&]{on.validate_pi05_vision_qkv_w8a8_manifest(wrong);});
        }
        auto missing=correct;missing.routes.pop_back();
        rejects([&]{on.validate_pi05_vision_qkv_w8a8_manifest(missing);});
        auto duplicate=correct;duplicate.routes.push_back(correct.routes.back());
        rejects([&]{on.validate_pi05_vision_qkv_w8a8_manifest(duplicate);});
        auto foreign=correct;foreign.routes.back().site_id=SIGLIP_O_LINEAR_SITE;
        rejects([&]{on.validate_pi05_vision_qkv_w8a8_manifest(foreign);});
        auto padded=correct;padded.logical_length-=16;padded.execution_padding_rows=16;
        rejects([&]{on.validate_pi05_vision_qkv_w8a8_manifest(padded);});
        auto acc32=correct;acc32.linear_accumulation=FmbLinearAccumulationPolicy::ACC32;
        rejects([&]{on.validate_pi05_vision_qkv_w8a8_manifest(acc32);});
        on.cores=4;rejects([&]{on.require_pi05_vision_qkv_w8a8_profile(rows,rows/256);});on.cores=8;
        on.eps_=1e-5;rejects([&]{on.require_pi05_vision_qkv_w8a8_profile(rows,rows/256);});on.eps_=1e-6;
        rejects([&]{on.require_pi05_vision_qkv_w8a8_profile(rows,1);});
        rejects([&]{on.require_pi05_vision_qkv_w8a8_profile(rows-16,rows/256);});
    }
    std::cout<<"SigLIP actual W8A8 manifest and identity: PASS\n";
}
'''
    assert "SigLIP actual W8A8 manifest and identity: PASS" in _run_native(tmp_path, source, torch)


def test_siglip_mlp_w8a8_workspace_preserves_complete_live_roots(tmp_path, monkeypatch):
    """Use the real byte layout/range validators, including dead-tail reuse."""
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    source = r'''
#include "fused/rpu_siglip_w8a8_contract.h"
#include <cassert>
#include <iostream>
template<class F> void rejects(F run) {
    bool rejected=false;
    try {run();} catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
}
int main() {
    for(int64_t rows:{512,768}) {
        const auto w=siglip_mlp_w8a8::workspace(rows);
        const uint64_t expected_raw=rows==512?147456:221184;
        const uint64_t expected_q_end=rows==512?425984:638976;
        const uint64_t expected_sx_end=rows==512?427008:640512;
        assert(w.raw_bytes==expected_raw && w.fc2_q_offset==expected_raw);
        assert(w.fc2_q_offset+w.fc2_q_bytes==expected_q_end);
        assert(w.sx_offset==expected_q_end && w.sx_offset+w.sx_bytes==expected_sx_end);
        assert(w.full_bytes==uint64_t(rows)*1152*2);
        assert(w.fc1_q_bytes==uint64_t(rows)*1152 && w.fc1_bytes==uint64_t(rows)*544*2);
        assert(w.sx_offset+w.sx_bytes<w.full_bytes);
        siglip_mlp_w8a8::validate_root_extents(rows,w.full_bytes,w.full_bytes,w.fc1_bytes,w.full_bytes);
        for(int bad=0;bad<4;++bad) {
            std::array<uint64_t,4> sizes{w.full_bytes,w.full_bytes,w.fc1_bytes,w.full_bytes};
            --sizes[bad];
            rejects([&]{siglip_mlp_w8a8::validate_root_extents(rows,sizes[0],sizes[1],sizes[2],sizes[3]);});
        }
        const uint64_t base=0x40000000,capacity=8387584;
        const std::array<uint64_t,4> roots{base+256,base+256+w.full_bytes,
            base+256+2*w.full_bytes,base+256+2*w.full_bytes+w.fc1_bytes};
        const auto validate=[&](const auto& r){siglip_mlp_w8a8::validate_roots(rows,base,capacity,r[0],r[1],r[2],r[3]);};
        validate(roots);
        for(int bad=0;bad<4;++bad) {
            auto wrong=roots;wrong[bad]+=1;rejects([&]{validate(wrong);});
            wrong=roots;wrong[bad]=base-256;rejects([&]{validate(wrong);});
            wrong=roots;wrong[bad]=base+capacity;rejects([&]{validate(wrong);});
            wrong=roots;wrong[bad]=roots[(bad+1)%4];rejects([&]{validate(wrong);});
        }
        // A8 uses only half the O root, but its full FP16 partial input remains
        // live through owner reduction. A second root cannot occupy that tail.
        auto partial_tail=roots;partial_tail[1]=roots[0]+w.fc1_q_bytes;
        rejects([&]{validate(partial_tail);});
        for(bool second:{false,true}) {
            const auto c=siglip_mlp_w8a8::candidate(rows,second);
            assert(c.compiled_local_k==(second?544:1152));
            const auto args=siglip_mlp_w8a8::linear_arguments(rows,second);
            assert(args[1]==rows && args[5]==(second?0:1) && args[8]==16);
        }
    }
    for(int64_t rows:{-1,0,256,511,513,769,1024})
        rejects([&]{(void)siglip_mlp_w8a8::workspace(rows);});
    std::cout<<"SigLIP MLP W8A8 bounded live workspace: PASS\n";
}
'''
    assert "SigLIP MLP W8A8 bounded live workspace: PASS" in _run_native(tmp_path, source, torch)


def test_siglip_mlp_w8a8_actual_manifest_profile_and_cold_identity(tmp_path, monkeypatch):
    """Real owner/QKV/MLP validators compose without selector collisions."""
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    owner = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    constants = "\n".join(line for line in owner.splitlines()
                          if line.startswith("constexpr int64_t SIGLIP_"))
    functions = "\n".join(_function(owner, marker).replace(" const override", " const") for marker in (
        "    KernelId pi05_vision_mlp_w8a8_owner_id(",
        "    void require_pi05_vision_mlp_w8a8_payloads(",
        "    std::vector<int64_t> pi05_vision_mlp_w8a8_identity(",
        "    void require_pi05_vision_mlp_w8a8_profile(",
        "    void validate_pi05_vision_mlp_w8a8_manifest(",
        "    std::vector<int64_t> pi05_vision_qkv_w8a8_identity(",
        "    void require_pi05_vision_qkv_w8a8_profile(",
        "    void validate_pi05_vision_qkv_w8a8_manifest(",
        "    std::vector<int64_t> pi05_vision_owner_ln_arguments(",
        "    void require_pi05_vision_owner_ln_profile(",
        "    void validate_pi05_vision_owner_ln_manifest(",
        "    int64_t subclass_layout_hash(",
        "    int64_t subclass_layout_hash_without_ln_stream(",
        "    int64_t ln_stream_elems(",
    ))
    source = r'''
#include "core/fused_model_base.h"
#include "fused/rpu_siglip_w8a8_contract.h"
#include <cassert>
#include <iostream>
#include <set>
using namespace v3;
enum class KernelId : uint16_t { LAYER_NORM_QUANTIZE_W8A8_H1152, GELU_TANH_QUANTIZE_W8A8_H544,
    PI05_VISION_OWNER_LAYERNORM_QUANTIZE_W8A8_M512N1152,
    PI05_VISION_OWNER_LAYERNORM_QUANTIZE_W8A8_M768N1152, FC1, FC2, QKV };
struct KernelCache {
    std::set<KernelId> missing;
    static KernelCache& instance() {static KernelCache cache;return cache;}
    bool has_loaded(KernelId id) const {return !missing.count(id);}
};
''' + constants + r'''
namespace v3 {
struct Owner {
    bool qkv_w8a8_=false,mlp_w8a8_=true,linear_acc32_=false,audio_mode_=false;
    int cores=8;
    int64_t image_batch_count_=2,orig_head_dim_=72,projection_dim_=2048;
    double eps_=1e-6;
    std::optional<bool> configured_linear_acc32_=false;
    std::vector<int> layer_weights_=std::vector<int>(27),layer_bias_norm_=std::vector<int>(27);
    bool pi05_vision_fc_weight_outer_opt_in_=false,pi05_vision_kv_spm_only_opt_in_=true;
    bool pi05_vision_owner_ln_opt_in_=true,pi05_ring_xor3_opt_in_=true;
    bool pi05_vision_merge_segment_opt_in_=false,pi05_vision_q_resident_opt_in_=false;
    bool pi05_vision_ln_stream_opt_in_=false;
    at::Tensor audio_proj2_w_;
    int num_cores() const {return cores;}
    int attn_tp() const {return cores;}
    int mlp_tp() const {return cores;}
    int64_t num_layers() const {return layer_weights_.size();}
    int64_t hidden_size() const {return 1152;}
    int64_t intermediate_size() const {return 4352;}
    int64_t num_q_heads() const {return 16;}
    int64_t num_kv_heads() const {return 16;}
    int64_t head_dim() const {return 80;}
    bool pi05_vision_supported_owner() const {return true;}
    void require_pi05_vision_owner_ln_payloads(int64_t) const {}
    FmbLinearAccumulationPolicy linear_accumulation_policy() const {
        return linear_acc32_?FmbLinearAccumulationPolicy::ACC32:FmbLinearAccumulationPolicy::ACC16;
    }
    std::vector<int64_t> core_profile_arguments() const {return {cores};}
    std::vector<int64_t> pi05_vision_fc_arguments(bool) const {return {};}
    KernelId pi05_vision_mlp_w8a8_linear_id(int64_t,bool second) const {return second?KernelId::FC2:KernelId::FC1;}
    KernelId pi05_vision_qkv_w8a8_linear_id(int64_t) const {return KernelId::QKV;}
    void require_pi05_vision_qkv_w8a8_payloads(int64_t) const {
        TORCH_CHECK(KernelCache::instance().has_loaded(KernelId::LAYER_NORM_QUANTIZE_W8A8_H1152) &&
                    KernelCache::instance().has_loaded(KernelId::QKV),"missing QKV payload");
    }
''' + functions + r'''
};
}
template<class F> void rejects(F run) {
    bool rejected=false;
    try {run();} catch(const c10::Error&) {rejected=true;}
    catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
}
FmbPhysicalExecutionManifest manifest(int64_t rows,bool mlp,bool qkv) {
    FmbPhysicalExecutionManifest m;
    m.state=FmbPhysicalManifestState::COMPLETE;
    m.logical_length=m.physical_length=m.kv_logical_length=m.kv_insert_physical_rows=rows;
    m.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
    m.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
    Owner owner;
    m.routes.push_back({SIGLIP_ATTN_ALL_REDUCE_SITE,FmbRouteFamily::ALL_REDUCE,
        mlp?SIGLIP_OWNER_LN_QUANT_W8A8_ROUTE:SIGLIP_OWNER_LN_ROUTE,0,
        mlp?siglip_mlp_w8a8::owner_arguments(rows):owner.pi05_vision_owner_ln_arguments(false,rows),0});
    for(bool second:{false,true}) {
        m.routes.push_back({second?SIGLIP_FC2_LINEAR_SITE:SIGLIP_FC1_LINEAR_SITE,FmbRouteFamily::LINEAR,
            mlp?SIGLIP_MLP_W8A8_LINEAR_ROUTE:static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),0,
            mlp?siglip_mlp_w8a8::linear_arguments(rows,second):std::vector<int64_t>{},0});
    }
    if(mlp)m.routes.push_back({SIGLIP_GELU_QUANT_W8A8_SITE,FmbRouteFamily::ACTIVATION,
        SIGLIP_GELU_QUANT_W8A8_ROUTE,0,siglip_mlp_w8a8::gelu_arguments(rows),0});
    m.routes.push_back({SIGLIP_MLP_ALL_REDUCE_SITE,FmbRouteFamily::ALL_REDUCE,
        SIGLIP_OWNER_LN_COMPACT_ROUTE,0,owner.pi05_vision_owner_ln_arguments(true,rows),0});
    m.routes.push_back({SIGLIP_RAW_SPM_ATTN_SITE,FmbRouteFamily::ATTENTION,
        static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA),0,{rows/256,256,rows,rows,16,80},0});
    if(qkv)m.routes.push_back({SIGLIP_LN1_QUANT_W8A8_SITE,FmbRouteFamily::NORMALIZATION,
        SIGLIP_LN1_QUANT_W8A8_ROUTE,0,siglip_qkv_w8a8::norm_arguments(rows),0});
    for(auto site:{SIGLIP_Q_LINEAR_SITE,SIGLIP_K_LINEAR_SITE,SIGLIP_V_LINEAR_SITE})
        m.routes.push_back({site,FmbRouteFamily::LINEAR,
            qkv?SIGLIP_QKV_W8A8_LINEAR_ROUTE:static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),0,
            qkv?siglip_qkv_w8a8::linear_arguments(rows):std::vector<int64_t>{},0});
    return m;
}
int main() {
    for(int64_t rows:{512,768})for(bool qkv:{false,true}) {
        Owner on,off;on.image_batch_count_=off.image_batch_count_=rows/256;
        on.qkv_w8a8_=off.qkv_w8a8_=qkv;off.mlp_w8a8_=false;
        auto correct=manifest(rows,true,qkv),legacy=manifest(rows,false,qkv);
        const auto validate=[&](const auto& m){
            on.validate_pi05_vision_mlp_w8a8_manifest(m);
            on.validate_pi05_vision_owner_ln_manifest(m);
            on.validate_pi05_vision_qkv_w8a8_manifest(m);
        };
        validate(correct);
        off.validate_pi05_vision_mlp_w8a8_manifest(legacy);
        off.validate_pi05_vision_owner_ln_manifest(legacy);
        off.validate_pi05_vision_qkv_w8a8_manifest(legacy);
        rejects([&]{validate(legacy);});
        rejects([&]{off.validate_pi05_vision_mlp_w8a8_manifest(correct);});
        assert(on.subclass_layout_hash()!=off.subclass_layout_hash());
        const auto original_hash=on.subclass_layout_hash();
        for(auto missing:{KernelId::GELU_TANH_QUANTIZE_W8A8_H544,
                          on.pi05_vision_mlp_w8a8_owner_id(rows),KernelId::FC1,KernelId::FC2}) {
            KernelCache::instance().missing.insert(missing);
            assert(on.subclass_layout_hash()!=original_hash);
            rejects([&]{validate(correct);});
            KernelCache::instance().missing.clear();
        }
        // Each owner/FC1/FC2/GELU ABI and tail-offset field binds the descriptor.
        for(int route=0;route<4;++route) {
            for(size_t field=0;field<correct.routes[route].arguments.size();++field) {
                auto wrong=correct;++wrong.routes[route].arguments[field];
                rejects([&]{validate(wrong);});
            }
            for(int field=0;field<4;++field) {
                auto wrong=correct;
                if(field==0)++wrong.routes[route].selector;
                if(field==1)++wrong.routes[route].flags;
                if(field==2)++wrong.routes[route].invocation;
                if(field==3)wrong.routes[route].family=FmbRouteFamily::GRAPH_SCHEDULE;
                rejects([&]{validate(wrong);});
            }
            auto missing=correct;missing.routes.erase(missing.routes.begin()+route);
            rejects([&]{validate(missing);});
            auto duplicate=correct;duplicate.routes.push_back(correct.routes[route]);
            rejects([&]{validate(duplicate);});
            auto foreign=correct;foreign.routes[route].site_id=SIGLIP_O_LINEAR_SITE;
            rejects([&]{validate(foreign);});
        }
        auto wrong_compact=correct;++wrong_compact.routes[4].arguments[0];
        rejects([&]{validate(wrong_compact);});
        auto padded=correct;padded.logical_length-=16;padded.execution_padding_rows=16;
        rejects([&]{validate(padded);});
        auto acc32=correct;acc32.linear_accumulation=FmbLinearAccumulationPolicy::ACC32;
        rejects([&]{validate(acc32);});
        for(int cores:{4,6}) {on.cores=cores;rejects([&]{validate(correct);});}on.cores=8;
        for(auto flag:{&on.linear_acc32_,&on.audio_mode_,&on.pi05_vision_fc_weight_outer_opt_in_}) {
            *flag=true;rejects([&]{validate(correct);});*flag=false;
        }
        for(auto flag:{&on.pi05_vision_owner_ln_opt_in_,&on.pi05_vision_kv_spm_only_opt_in_,&on.pi05_ring_xor3_opt_in_}) {
            *flag=false;rejects([&]{validate(correct);});*flag=true;
        }
        on.eps_=1e-5;rejects([&]{validate(correct);});on.eps_=1e-6;
        rejects([&]{on.require_pi05_vision_mlp_w8a8_profile(rows,1);});
        rejects([&]{on.require_pi05_vision_mlp_w8a8_profile(rows-16,rows/256);});
    }
    std::cout<<"SigLIP actual MLP/QKV composed manifest and identity: PASS\n";
}
'''
    assert "SigLIP actual MLP/QKV composed manifest and identity: PASS" in _run_native(tmp_path, source, torch)


def test_actual_image_and_pi_loop_publication_preserves_dirty_live_ranges(tmp_path, monkeypatch):
    """Real ATen views and native publication/DMA calls, host SDK/Graph doubles."""
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    siglip = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    pi = (ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    profile = (ROOT / "src/core/rpu_profile.h").read_text()
    core = _function(siglip, "    at::Tensor run_packed_patch_embed_core(")
    image_loop = _function(core[core.index("        void* image_acquire_span ="):],
                           "        for (int64_t i = 0; i < N; ++i)")
    image_copy = core[core.index("        std::vector<at::Tensor> images = input_images;"):
                      core.index("        if (!SPM_ALLOC.is_initialized())")]
    allocation = """at::empty(
                    {N, 3, 224, 224}, at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1))"""
    assert image_copy.count(allocation) == 1
    image_copy = image_copy.replace(allocation, "bank({N, 3, 224, 224})")
    copy_line = "std::memcpy(image.data_ptr(), images[i].const_data_ptr(), images[i].nbytes());"
    assert image_copy.count(copy_line) == 1
    image_copy = image_copy.replace(copy_line, copy_line + "\n                mark_full_dirty(image);")
    image_copy = image_copy.replace("images.front().device().is_cpu()", "cpu_label(images.front())")
    image_guards = _function(siglip, "    at::Tensor forward_multi(")
    image_guards = image_guards[image_guards.index("        TORCH_CHECK(num_layers() > 0"):
                                image_guards.index("        // Patch-embed + pack all N images")]
    # Only labels/allocator/cache dirtiness are host boundaries. Keep actual
    # guard, memcpy, publication, mutable source and keepalive statements.
    image_guards = image_guards.replace("images.front().device().is_cpu()", "cpu_label(images.front())")
    image_guards = image_guards.replace("im.device().is_cpu()", "cpu_label(im)")
    for tensor in ("im", "k_caches[layer]", "v_caches[layer]"):
        image_guards = image_guards.replace(
            tensor + ".device().type() == at::kPrivateUse1",
            "(!cpu_label(" + tensor + ") && " + tensor + ".device().type() == at::kCPU)")
    image_publication = image_copy + core[core.index("        void* image_acquire_span ="):
                                          core.index(image_loop)] + image_loop
    span_proof = _function(siglip, "static bool packed_image_acquire_span(").replace(
        "at::kPrivateUse1", "at::kCPU")
    patch = _function(siglip, "static void fused_patch_embedding(")
    pixel_dma = patch[patch.index("    TORCH_CHECK(img_slot >= 0"):
                      patch.index("    // ===== Step ②")]
    loop = _function(pi, "void Pi05DenoiseStepModel::denoise_loop_forward(")
    guards = loop[loop.index("    const int64_t cs  ="):
                  loop.index('    TORCH_CHECK(num_steps >= 1')]
    publication = loop[loop.index("    // A CPU x0 is copied in full"):
                       loop.index("    // ONE forward")]
    # Replace only the RPU allocation/CPU-dirty-cache boundary. The real memcpy,
    # sized publication, mutable bases and Graph keepalive execute unchanged.
    allocation = """at::empty(
            x0_rpu.sizes(), at::TensorOptions().dtype(at::kHalf).device(x_out.device()))"""
    assert publication.count(allocation) == 1
    publication = publication.replace(allocation, "private_bank(x0_rpu.sizes())")
    copy_line = "std::memcpy(private_x0.data_ptr(), x0_rpu.const_data_ptr(), x0_rpu.nbytes());"
    assert publication.count(copy_line) == 1
    publication = publication.replace(copy_line, copy_line + "\n        mark_full_dirty(private_x0);")
    pre = _function(pi, "void Pi05DenoiseStepModel::emit_pre_layers_body(")
    dma_start = pre.index("        rpu_launch_ddr_broadcast_spm_dma_mutable(",
                          pre.index("// [A1]"))
    x0_dma = pre[dma_start:pre.index(";", dma_start) + 1]
    post = _function(pi, "void Pi05DenoiseStepModel::emit_post_layers_body(")
    dma_start = post.index("            rpu_launch_spm_copy_ddr_dma_mutable(")
    output_dma = post[dma_start:post.index(";", dma_start) + 1]
    macros = "\n".join(profile[profile.index(f"#define {name}("):].split("\n\n", 1)[0]
                       for name in ("rpu_ddr_flush_force", "rpu_ddr_flush_force_sized"))
    # Substitute the device label only; real ATen owns dtype, storage, views,
    # shape/stride validation and ordinary/inference version behavior.
    # CPU ATen models both device allocations; substitute the device-label
    # query explicitly so the original public RPU branch is still exercised.
    guards = guards.replace("const bool cpu_x0 = x0_rpu.device().is_cpu();",
                            "const bool cpu_x0 = input_cpu;")
    guards = guards.replace("x0_rpu.device().type() == at::kPrivateUse1",
                            "(!input_cpu && x0_rpu.device().type() == at::kCPU)")
    guards = guards.replace("at::kPrivateUse1", "at::kCPU")
    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/intrusive_ptr.h>
#include <array>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <unordered_map>
bool g_rpu_ddr_flush_enabled=true,g_rpu_ddr_flush_force_enabled=true;
struct Store {
 at::Tensor backing;std::vector<uint8_t> ddr,dirty;
 void* pointer=backing.data_ptr();
 c10::weak_intrusive_ptr<c10::StorageImpl> owner{backing.storage().getWeakStorageImpl()};
};
std::vector<Store> stores;
std::vector<std::pair<void*,size_t>> flushes;
int batch_calls=0,owner_lookups=0,owner_fault=0,owner_fault_target=0;
bool private_misaligned=false;
std::vector<std::vector<uint8_t>> reads;
std::vector<std::function<void()>> nodes;
std::vector<uint8_t> spm;
bool recording=false,replaying=false,graph_active=false;
std::unordered_map<const c10::TensorImpl*,
 c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>> cpu_inputs;
bool cpu_label(const at::Tensor& t){
 auto it=cpu_inputs.find(t.unsafeGetTensorImpl());return it!=cpu_inputs.end()&&!it->second.expired();}
void label_cpu(const at::Tensor& t){
 cpu_inputs.insert_or_assign(t.unsafeGetTensorImpl(),
  c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>(t.getIntrusivePtr()));}
Store& locate(void* p,size_t n,size_t& offset){
 for(auto i=stores.rbegin();i!=stores.rend();++i){auto& s=*i;
  const auto b=reinterpret_cast<uintptr_t>(s.pointer),v=reinterpret_cast<uintptr_t>(p);
  if(v>=b&&v-b<=s.ddr.size()&&n<=s.ddr.size()-(v-b)&&!s.owner.expired()){
   offset=v-b;return s;}}
 TORCH_CHECK(false,"out of logical storage");
}
at::Tensor bank(at::IntArrayRef shape){auto t=at::zeros(shape,at::kHalf);stores.push_back({t,
 std::vector<uint8_t>(t.nbytes(),0xa7),std::vector<uint8_t>(t.nbytes(),0)});return t;}
at::Tensor private_bank(at::IntArrayRef shape){
 if(!private_misaligned)return bank(shape);
 int64_t elements=1;for(auto dim:shape)elements*=dim;
 return bank({elements+32}).narrow(0,16,elements).view(shape);
}
void dirty(const at::Tensor& t,uint8_t bits){size_t off;auto& s=locate(t.data_ptr(),t.nbytes(),off);
 std::memset(t.data_ptr(),bits,t.nbytes());std::fill(s.dirty.begin()+off,s.dirty.begin()+off+t.nbytes(),1);}
void mark_full_dirty(const at::Tensor& t){size_t off;auto& s=locate(t.data_ptr(),t.nbytes(),off);
 std::fill(s.dirty.begin()+off,s.dirty.begin()+off+t.nbytes(),1);}
int64_t version(const at::Tensor& t){return t.is_inference()?-1:t._version();}
namespace rhino_lkn{
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
struct HostDDR_t{void* base;size_t bytes;
 void* get_cpu_ptr(){return base;}size_t get_memory_size(){return bytes;}};
HostDDR_t* RpuGetHostddr(void* p){
 const int index=owner_lookups++;size_t off;auto& s=locate(p,0,off);
 static HostDDR_t owner;owner={s.pointer,s.ddr.size()};
 // Fault only the SDK metadata boundary; the real ATen storage/DMA remain valid.
 if(owner_fault&&index==owner_fault_target){
  const uintptr_t address=reinterpret_cast<uintptr_t>(p);
  if(owner_fault==1)return nullptr;
  if(owner_fault==2)owner.base=nullptr;
  if(owner_fault==3)owner.bytes=0;
  if(owner_fault==4)owner.base=reinterpret_cast<void*>(address+1);
  if(owner_fault==5)owner.bytes=off+3200-1;
  if(owner_fault==6)owner.bytes=std::numeric_limits<size_t>::max();
  if(owner_fault==7){owner.base=reinterpret_cast<void*>(address-16);owner.bytes-=off-16;}
  if(owner_fault==8)owner.bytes=off+3200;
 }
 return &owner;
}
void flush_range(void* p,size_t n){size_t off;auto& s=locate(p,n,off);flushes.emplace_back(p,n);
 // Same 64-byte endpoint rounding as SDK r1_cache_flush. Offset-32 cases
 // include the adjacent half-lines; line-aligned cases leave dirty edges alone.
 const size_t begin=off&~size_t(63),end=(off+n+63)&~size_t(63);assert(end<=s.ddr.size());
 auto* cpu=static_cast<uint8_t*>(s.backing.data_ptr());for(size_t i=begin;i<end;++i){
  if(s.dirty[i])s.ddr[i]=cpu[i];else cpu[i]=s.ddr[i];s.dirty[i]=0;}}
void RpuDdrFlush(void* p,size_t n){flush_range(p,n);}
void RpuDdrFlush(void* p){size_t off;auto& s=locate(p,0,off);RpuDdrFlush(p,s.ddr.size()-off);}
void RpuDdrFlushBatch(void** pointers,size_t* sizes,size_t count){
 ++batch_calls;assert(count==2||count==3);
 for(size_t i=0;i<count;++i){
  assert(sizes[i]==size_t(count==3&&i==1?10*1024*2:3200));
  flush_range(pointers[i],sizes[i]);}
}
}
__MACROS__
void submit(std::function<void()> run){if(recording)nodes.push_back(run);else if(!replaying)run();}
void rpu_launch_ddr_broadcast_spm_dma_mutable(const uint64_t* base,int64_t offset,
 int64_t elements,uint32_t,int){submit([=]{size_t off;auto* p=reinterpret_cast<void*>(*base+offset);
 auto& s=locate(p,elements*2,off);spm.assign(s.ddr.begin()+off,s.ddr.begin()+off+elements*2);reads.push_back(spm);});}
void rpu_launch_spm_copy_ddr_dma_mutable(uint32_t,const uint64_t* base,int64_t offset,int64_t elements){
 submit([=]{size_t off;auto& s=locate(reinterpret_cast<void*>(*base+offset),elements*2,off);
 assert(spm.size()==size_t(elements*2));std::copy(spm.begin(),spm.end(),s.ddr.begin()+off);});}
struct Graph{std::vector<at::Tensor> held;void keep_alive(const at::Tensor& t){held.push_back(t);}} graph;
struct RpuKernelGraph{static bool has_active(){return graph_active;}static Graph& active(){return graph;}};
__LIVE_BASES__
__SPAN_PROOF__
namespace v3{enum class FmbRouteFamily{MUTABLE_DMA};struct InferenceContext{
 void consume_physical_route(FmbRouteFamily,int64_t,int64_t,int64_t,std::initializer_list<int64_t>,int)const{}};}
using v3::FmbRouteFamily;
enum class SiglipPatchMutableDmaRoute{DDR_BROADCAST_TO_SPM_MUTABLE};
constexpr int64_t SIGLIP_PATCH_INPUT_DMA_SITE=1;
void fused_patch_embedding(const at::Tensor& input_nchw,const at::Tensor&,const at::Tensor&,
 int64_t,int64_t,int64_t cin_orig,int64_t cin_padded,int64_t,int64_t,int64_t,
 at::Tensor&,int64_t,int img_slot,const std::vector<uint32_t>&,PatchEmbeddingLiveBases& live_bases,
 const v3::InferenceContext* physical_ctx,bool){
 const int64_t HW=input_nchw.size(2)*input_nchw.size(3),num_patches=256;
 const int64_t kh=14,kw=14,strideh=14,stridew=14;const uint32_t raw_addr=0;
 __PIXEL_DMA__
}
struct Siglip{
 // This existing packed/public test never admits private reduced-core N1.
 // The new receipt is tested whole in test_pi05_serial_cpu_camera_slab_receipt.
 bool private_serial_cpu_slab_published_=false;
 bool qkv_w8a8_=false,mlp_w8a8_=false;
 void require_pi05_vision_qkv_w8a8_profile(int64_t,int64_t) const {}
 void require_pi05_vision_mlp_w8a8_profile(int64_t,int64_t) const {}
 bool prepare_private_serial_cpu_slab(const at::Tensor&,at::Tensor&){assert(false);return false;}
 at::Tensor patch_emb_weight_,patch_emb_pos_emb_;PatchEmbeddingLiveBases patch_embedding_live_bases_;
 int64_t pe_kh_=14,pe_kw_=14,pe_cin_orig_=3,pe_cin_padded_=16,pe_cout_=1152,pe_strideh_=14,pe_stridew_=14;
 bool linear_acc32_=false;v3::InferenceContext context;const v3::InferenceContext& ctx(){return context;}
 bool has_patch_emb_=true,owner_supported=true;int cores=8;
 bool pi05_vision_kv_spm_only_opt_in_=false,pi05_vision_q_resident_opt_in_=false;
 int64_t num_layers(){return 27;}int num_cores(){return cores;}
 int64_t hidden_size(){return 1152;}int64_t intermediate_size(){return 4352;}
 int64_t num_q_heads(){return 16;}int64_t num_kv_heads(){return 16;}
 int64_t head_dim(){return 80;}int64_t orig_head_dim_=72,projection_dim_=2048;
 bool pi05_vision_supported_owner(){return owner_supported;}
 void validate(at::TensorList images,std::vector<at::Tensor>& k_caches,std::vector<at::Tensor>& v_caches){
 __IMAGE_GUARDS__
 }
 void publish(const std::vector<at::Tensor>& input_images,bool consume_external_prologue,
              bool admit_private_cpu_single_image=false){
 const int64_t npp=256;at::Tensor hidden;std::vector<uint32_t> offsets;
 __IMAGE_PUBLICATION__
 }};
struct Pi{
 int64_t chunk_size_=50,max_action_dim_=32;bool loop_mode_=true;
 at::Tensor adarms_table_=at::ones({1},at::kHalf),x0_ref_,cond_all_ref_,x_out_ref_;
 uint64_t x0_src_base_=0,x_t_src_base_=0,cond_all_src_base_=0,x_out_dst_base_=0;
 int64_t hidden_size(){return 1024;}int num_cores(){return 8;}uint32_t addr(int,const char*){return 0;}
 void publish(at::Tensor x0_rpu,at::Tensor cond_all,at::Tensor x_out,int64_t num_steps=10,bool input_cpu=false){
 __GUARDS__
 __PUBLICATION__
 }
 void consume(){const int64_t cs=chunk_size_,mad=max_action_dim_;
 __X0_DMA__
 __OUTPUT_DMA__
 }};
void check_read(uint8_t bits,size_t count){assert(reads.size()==count);for(auto& row:reads)
 for(auto b:row)assert(b==bits);}
void check_unpublished_edges(){for(auto& s:stores){assert(s.ddr.front()==0xa7&&s.ddr.back()==0xa7);}}
int main(){int cases=0;
 for(bool normal:{false,true})for(bool force:{false,true})for(bool inference:{false,true}){
 c10::InferenceMode mode(inference);g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
 for(int cameras:{1,2,3})for(bool list:{false,true}){
  ++cases;stores.clear();nodes.clear();graph.held.clear();Siglip m;std::vector<at::Tensor> images;
  if(list){for(int i=0;i<cameras;++i)images.push_back(bank({3,3,224,224}).narrow(0,1,1));}
  else{auto slab=bank({cameras+2,3,224,224});for(int i=0;i<cameras;++i)images.push_back(slab.slice(0,i+1,i+2).contiguous());}
  for(auto& s:stores)dirty(s.backing,0x55);
  std::vector<int64_t> versions;for(auto& t:images){assert(t.is_contiguous());versions.push_back(version(t));}
  recording=false;replaying=false;graph_active=false;
  for(uint8_t bits:{uint8_t(7),uint8_t(19),uint8_t(7)}){
   flushes.clear();reads.clear();for(auto& t:images)dirty(t,bits);m.publish(images,false);
   check_read(force?bits:0xa7,cameras);
   const int expected=list?cameras:1;
   assert(flushes.size()==(force?expected:0));
   for(auto& f:flushes)assert(f.second==size_t((list?1:cameras)*301056));
   if(force)assert(flushes.front().first==images.front().data_ptr());check_unpublished_edges();}
  graph_active=true;recording=true;for(auto& t:images)dirty(t,7);m.publish(images,true);recording=false;
  assert(nodes.size()==size_t(cameras)&&graph.held.size()==size_t(cameras));replaying=true;
  for(uint8_t bits:{uint8_t(7),uint8_t(19),uint8_t(7)}){
   graph.held.clear();flushes.clear();reads.clear();for(auto& t:images)dirty(t,bits);m.publish(images,true);
   for(auto& node:nodes)node();check_read(force?bits:0xa7,cameras);
   assert(flushes.size()==(force?(list?cameras:1):0));
   for(auto& f:flushes)assert(f.second==size_t((list?1:cameras)*301056));
   for(int i=0;i<cameras;++i)assert(version(images[i])==versions[i]);check_unpublished_edges();}
  replaying=false;graph_active=false;
 }
 // All these real ATen geometries must keep the original per-image path.
 // A distinct StorageImpl cannot borrow union authority even at adjacent raw
 // addresses. Overlap/gap and oversized B/C views likewise remain unmerged.
 for(int layout=0;layout<5;++layout){
  ++cases;stores.clear();nodes.clear();graph.held.clear();Siglip m;
  auto slab=bank({8,3,224,224});std::vector<at::Tensor> images;
  if(layout==0)images={slab.narrow(0,1,1),slab.narrow(0,3,1)}; // gap
  if(layout==1)images={slab.narrow(0,1,1),slab.narrow(0,1,1)}; // overlap
  if(layout==2){ // contiguous pointers but independent storage owners
   for(int i=1;i<=2;++i)images.push_back(at::from_blob(slab.narrow(0,i,1).data_ptr(),
     {1,3,224,224},at::TensorOptions().dtype(at::kHalf)));}
  if(layout==3)images={slab.narrow(0,1,2),slab.narrow(0,3,2)}; // B2
  if(layout==4){ // same backing, complete actual raw DMA beyond a C1 view
   images={slab.narrow(0,1,1).narrow(1,0,1),slab.narrow(0,2,1).narrow(1,0,1)};}
  void* ptr=nullptr;size_t bytes=0;assert(!packed_image_acquire_span(images,3,ptr,bytes));
  for(auto& s:stores)dirty(s.backing,0x55);
  std::vector<int64_t> versions;for(auto& t:images)versions.push_back(version(t));
  recording=true;graph_active=true;m.publish(images,true);recording=false;replaying=true;
  assert(nodes.size()==2&&graph.held.size()==2);
  for(uint8_t bits:{uint8_t(7),uint8_t(19),uint8_t(7)}){
   graph.held.clear();flushes.clear();reads.clear();for(auto& t:images)dirty(t,bits);
   m.publish(images,true);assert(flushes.size()==(force?2:0));
   for(int i=0;force&&i<2;++i){assert(flushes[i].first==images[i].data_ptr());
    assert(flushes[i].second==(layout==4?stores[0].ddr.size()-
      (reinterpret_cast<uintptr_t>(images[i].data_ptr())-reinterpret_cast<uintptr_t>(slab.data_ptr())):
      images[i].nbytes()));}
   for(auto& node:nodes)node();assert(reads.size()==2);
   for(int i=0;i<2;++i){assert(reads[i].size()==301056);
    auto* raw=static_cast<uint8_t*>(images[i].data_ptr());
    for(size_t k=0;k<reads[i].size();++k)assert(reads[i][k]==(force?raw[k]:0xa7));
    assert(version(images[i])==versions[i]);}
   assert(stores[0].ddr.front()==0xa7);
   if(layout!=4)assert(stores[0].ddr.back()==0xa7);
  }
  replaying=false;graph_active=false;
 }
 // CPU packed ingress: real admission + every-byte copy/publication and the
 // original mutable DMA/Graph lease run for BUILD then same-signature REPLAY.
 for(int cameras:{2,3}){
  ++cases;stores.clear();nodes.clear();graph.held.clear();cpu_inputs.clear();Siglip cpu;
  std::vector<at::Tensor> keys(27,at::ones({1},at::kHalf)),values=keys;
  auto source=at::zeros({cameras+2,3,224,224},at::kHalf);
  std::vector<at::Tensor> images;for(int i=0;i<cameras;++i){
   images.push_back(source.narrow(0,i+1,1));label_cpu(images.back());}
  std::vector<int64_t> versions;for(auto& t:images)versions.push_back(version(t));
  recording=true;replaying=false;graph_active=true;
  for(int step=0;step<3;++step){
   const uint8_t bits=step==1?19:7;
   if(step==2){ // Same signature, different source addresses and camera order.
    source=at::zeros({cameras+2,3,224,224},at::kHalf);
    for(int i=0;i<cameras;++i){images[i]=source.narrow(0,cameras-i,1);label_cpu(images[i]);}
   }
   for(int i=0;i<cameras;++i)std::memset(images[i].data_ptr(),bits+i,images[i].nbytes());
   flushes.clear();reads.clear();graph.held.clear();cpu.validate(images,keys,values);cpu.publish(images,true);
   assert(graph.held.size()==size_t(cameras));
   auto* owner=graph.held.front().storage().unsafeGetStorageImpl();
   assert(owner!=images.front().storage().unsafeGetStorageImpl());
   assert(graph.held.front().data_ptr()!=images.front().data_ptr());
   for(int i=0;i<cameras;++i){
    assert(graph.held[i].storage().unsafeGetStorageImpl()==owner);
    assert(std::memcmp(graph.held[i].const_data_ptr(),images[i].const_data_ptr(),images[i].nbytes())==0);
    assert(version(images[i])==versions[i]);}
   assert(flushes.size()==(force?1:0));if(force){
    assert(flushes[0].first==graph.held.front().data_ptr());assert(flushes[0].second==size_t(cameras*301056));}
   if(step==0){recording=false;replaying=true;assert(nodes.size()==size_t(cameras));}
   c10::weak_intrusive_ptr<c10::StorageImpl> lease(graph.held.front().storage().getWeakStorageImpl());
   stores.back().backing=at::Tensor();assert(!lease.expired());
   for(auto& t:images)std::memset(t.data_ptr(),0x66,t.nbytes());
   auto churn=at::empty({cameras,3,224,224},at::kHalf);churn.fill_(1);
   for(auto& node:nodes)node();assert(reads.size()==size_t(cameras));
   for(int i=0;i<cameras;++i)for(auto b:reads[i])assert(b==(force?bits+i:0xa7));
   graph.held.clear();assert(lease.expired());
  }
  // Each malformed private input is rejected before allocation/publication/DMA.
  int rejected_case=0;
  auto rejected=[&](std::vector<at::Tensor> bad){
   ++rejected_case;
   auto count=flushes.size(),allocated=stores.size(),submitted=nodes.size();bool failed=false;
   try{cpu.validate(bad,keys,values);cpu.publish(bad,true);}catch(const c10::Error&){failed=true;}
   if(!(failed&&flushes.size()==count&&stores.size()==allocated&&nodes.size()==submitted)) {
    std::cerr<<"rejected_case="<<rejected_case<<" cameras="<<cameras<<" force="<<force
     <<" inference="<<inference<<" failed="<<failed<<" labels=";
    for(auto& im:bad)std::cerr<<cpu_label(im)<<",";
    std::cerr<<" changes="<<flushes.size()-count<<","<<stores.size()-allocated<<","<<nodes.size()-submitted<<"\n";
   }
   assert(failed&&flushes.size()==count&&stores.size()==allocated&&nodes.size()==submitted);};
  auto bad=images;bad[1]=bad[1].to(at::kFloat);label_cpu(bad[1]);rejected(bad);
  bad=images;bad[1]=at::zeros({1,3,224,448},at::kHalf).slice(3,0,448,2);label_cpu(bad[1]);rejected(bad);
  bad=images;bad[1]=at::zeros({1,3,223,224},at::kHalf);label_cpu(bad[1]);rejected(bad);
  bad=images;bad[1]=at::zeros({1,1,224,224},at::kHalf);label_cpu(bad[1]);rejected(bad);
  bad=images;bad[1]=at::zeros({2,3,224,224},at::kHalf);label_cpu(bad[1]);rejected(bad);
  bad=images;bad[1]=at::zeros({1,3,224,224},at::kHalf);rejected(bad); // mixed device labels
  bad=images;bad[1]=at::_neg_view(images[1]);label_cpu(bad[1]);rejected(bad);
  bad=images;bad[1]=at::zeros({1,3,224,224},at::kHalf);bad[1].storage().set_nbytes(64);
  label_cpu(bad[1]);rejected(bad);
  if(!inference){bad=images;bad[1]=images[1].clone().set_requires_grad(true);label_cpu(bad[1]);rejected(bad);}
  rejected({images[0]});rejected({images[0],images[0],images[0],images[0]});
  cpu.cores=4;rejected(images);cpu.cores=8;cpu.owner_supported=false;rejected(images);cpu.owner_supported=true;
  auto kv=keys[0];keys[0]=at::ones({1},at::kHalf);label_cpu(keys[0]);rejected(images);keys[0]=kv;
  replaying=false;recording=false;graph_active=false;cpu_inputs.clear();
 }
 // Deny the union fast path on real metadata-only negative boundaries without
 // invoking the old DMA on invalid geometry/address metadata.
 {auto slab=at::zeros({4,3,224,448},at::kHalf);
  auto a=slab.narrow(0,1,1).slice(3,0,448,2),b=slab.narrow(0,2,1).slice(3,0,448,2);
  assert(a.sizes()==at::IntArrayRef({1,3,224,224})&&!a.is_contiguous());
  void* ptr=nullptr;size_t bytes=0;assert(!packed_image_acquire_span({a,b},3,ptr,bytes));
  auto empty=at::empty({1,3,0,224},at::kHalf);
  assert(!packed_image_acquire_span({empty,empty},3,ptr,bytes));
  auto neg=at::_neg_view(at::zeros({2,3,224,224},at::kHalf));
  assert(!packed_image_acquire_span({neg.narrow(0,0,1),neg.narrow(0,1,1)},3,ptr,bytes));
  auto limit=std::numeric_limits<uintptr_t>::max();
  auto overflow=at::from_blob(reinterpret_cast<void*>(limit-127),{2,3,1,64},
     at::TensorOptions().dtype(at::kHalf));
  assert(!packed_image_acquire_span({overflow.narrow(0,0,1),overflow.narrow(0,1,1)},3,ptr,bytes));
 }
 // Full union and per-camera byte patterns must be acquired afresh on REPLAY;
 // raw aliases do not increment versions in ordinary or inference mode.
 ++cases;stores.clear();nodes.clear();graph.held.clear();Siglip packed;
 auto contiguous=bank({5,3,224,224});
 std::vector<at::Tensor> cameras={contiguous.narrow(0,1,1),contiguous.narrow(0,2,1),contiguous.narrow(0,3,1)};
 for(auto& s:stores)dirty(s.backing,0x55);
 std::vector<int64_t> versions;for(auto& t:cameras)versions.push_back(version(t));
 recording=true;graph_active=true;packed.publish(cameras,true);recording=false;replaying=true;
 assert(nodes.size()==3&&graph.held.size()==3);
 for(uint8_t bits:{uint8_t(7),uint8_t(19),uint8_t(7)}){
  graph.held.clear();flushes.clear();reads.clear();
  for(int i=0;i<3;++i)dirty(cameras[i],bits+i);
  packed.publish(cameras,true);assert(flushes.size()==(force?1:0));
  if(force)assert(flushes[0].first==cameras[0].data_ptr()&&flushes[0].second==903168);
  for(auto& node:nodes)node();assert(reads.size()==3);
  for(int i=0;i<3;++i){for(auto b:reads[i])assert(b==(force?bits+i:0xa7));
   assert(version(cameras[i])==versions[i]);}check_unpublished_edges();
 }
 replaying=false;graph_active=false;
 // Existing list/patch-only admission also permits a contiguous C1 view.
 // Its backing covers the actual C3 DMA; preserve the original unsized flush
 // so dirty bytes consumed beyond that logical view still reach device DDR.
 ++cases;stores.clear();nodes.clear();graph.held.clear();Siglip legacy;
 auto short_image=bank({3,3,224,224}).narrow(0,1,1).narrow(1,0,1);
 assert(short_image.is_contiguous()&&short_image.nbytes()==100352);
 const auto legacy_version=version(short_image);for(auto& s:stores)dirty(s.backing,0x55);
 recording=true;graph_active=true;legacy.publish({short_image},true);recording=false;replaying=true;
 assert(nodes.size()==1&&graph.held.size()==1);
 for(uint8_t bits:{uint8_t(7),uint8_t(19),uint8_t(7)}){
  graph.held.clear();dirty(short_image,bits);flushes.clear();reads.clear();legacy.publish({short_image},true);
  assert(flushes.size()==(force?1:0));if(force)assert(flushes[0].second==602112);
  for(auto& node:nodes)node();assert(reads.size()==1&&reads[0].size()==301056);
  for(size_t i=0;i<reads[0].size();++i)assert(reads[0][i]==(force?(i<short_image.nbytes()?bits:0x55):0xa7));
  assert(version(short_image)==legacy_version&&stores[0].ddr.front()==0xa7);
 }
 replaying=false;graph_active=false;
 ++cases;stores.clear();nodes.clear();graph.held.clear();Pi m;
 auto x0=bank({1728}).narrow(0,64,1600).view({1,50,32});
 auto out=bank({1728}).narrow(0,64,1600).view({1,50,32});auto cond=at::zeros({10,1024},at::kHalf);
 for(auto& s:stores)dirty(s.backing,0x55);const auto xv=version(x0),ov=version(out);
 recording=true;m.consume();recording=false;assert(nodes.size()==2);replaying=true;
 at::Tensor retained;
 for(uint8_t bits:{uint8_t(7),uint8_t(19),uint8_t(7)}){
  dirty(x0,bits);dirty(out,0x33);flushes.clear();reads.clear();batch_calls=owner_lookups=0;m.publish(x0,cond,out);
  assert(batch_calls==0&&owner_lookups==0);
  assert(flushes.size()==(force?2:0));for(auto& f:flushes)assert(f.second==3200);
  for(auto& node:nodes)node();check_read(force?bits:0xa7,1);
  // A later CPU acquire must invalidate the device-written output. With force
  // off the old dirty output lines remain unprotected, preserving that contract.
  rhino_lkn::RpuDdrFlush(out.data_ptr(),out.nbytes());auto saved=out.clone();
  for(auto b:std::vector<uint8_t>(static_cast<uint8_t*>(saved.data_ptr()),static_cast<uint8_t*>(saved.data_ptr())+saved.nbytes()))
   assert(b==(force?bits:0x33));
  if(bits==7&&!retained.defined())retained=saved;
  assert(version(x0)==xv&&version(out)==ov);check_unpublished_edges();}
 assert(std::memcmp(retained.data_ptr(),out.data_ptr(),out.nbytes())==0);replaying=false;
 // Actual shape/dtype/stride preflight must fail before either publication.
 for(auto bad:{x0.to(at::kFloat),at::zeros({1,50,64},at::kHalf).slice(2,0,64,2),x0.narrow(1,0,49)}){
  auto before=flushes.size();bool failed=false;try{m.publish(bad,cond,out);}catch(const c10::Error&){failed=true;}
  assert(failed&&flushes.size()==before);}
 // Private CPU ingress copies complete current bytes afresh into a new owner.
 // The registry relinquishes its strong input reference before deferred DMA;
 // only the production Graph keepalive remains after the model ref is reset.
 ++cases;stores.clear();nodes.clear();graph.held.clear();Pi cpu;
 auto raw=at::zeros({1728},at::kHalf),input=raw.narrow(0,64,1600).view({1,50,32});
 auto cpu_cond=at::zeros({10,1024},at::kHalf);const auto input_version=version(input);
 recording=true;graph_active=true;replaying=false;cpu.consume();recording=false;replaying=true;
 assert(nodes.size()==2);std::vector<at::Tensor> saved_outputs;std::vector<uint8_t> saved_first;
 for(uint8_t bits:{uint8_t(7),uint8_t(19),uint8_t(7)}){
  auto fresh_out=bank({1728}).narrow(0,64,1600).view({1,50,32});dirty(fresh_out,0x33);
  std::memset(input.data_ptr(),bits,input.nbytes());flushes.clear();reads.clear();graph.held.clear();
  batch_calls=owner_lookups=0;cpu.publish(input,cpu_cond,fresh_out,10,true);
  assert(batch_calls==(force?1:0)&&owner_lookups==(force?2:0));
  assert(cpu.x0_ref_.data_ptr()!=input.data_ptr()&&graph.held.size()==1);
  assert(graph.held.front().is_same(cpu.x0_ref_));
  assert(std::memcmp(cpu.x0_ref_.const_data_ptr(),input.const_data_ptr(),input.nbytes())==0);
  assert(flushes.size()==(force?2:0));for(auto& f:flushes)assert(f.second==3200);
  if(force)assert(flushes[0].first==cpu.x0_ref_.data_ptr());
  c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl> lease(cpu.x0_ref_.getIntrusivePtr());
  auto private_ptr=cpu.x0_ref_.data_ptr();size_t private_offset;
  locate(private_ptr,3200,private_offset).backing=at::Tensor();cpu.x0_ref_=at::Tensor();
  assert(!lease.expired());std::memset(input.data_ptr(),0x66,input.nbytes());
  for(auto& node:nodes)node();check_read(force?bits:0xa7,1);
  rhino_lkn::RpuDdrFlush(fresh_out.data_ptr(),fresh_out.nbytes());auto saved=fresh_out.clone();
  auto* raw_saved=static_cast<uint8_t*>(saved.data_ptr());
  for(size_t i=0;i<saved.nbytes();++i)assert(raw_saved[i]==(force?bits:0x33));
  if(saved_outputs.empty())saved_first.assign(raw_saved,raw_saved+saved.nbytes());
  saved_outputs.push_back(saved);
  assert(std::memcmp(saved_outputs.front().const_data_ptr(),saved_first.data(),saved_first.size())==0);
  assert(version(input)==input_version);graph.held.clear();assert(lease.expired());
 }
 assert(saved_outputs[0].data_ptr()!=saved_outputs[1].data_ptr());replaying=false;graph_active=false;
 auto valid_out=bank({1728}).narrow(0,64,1600).view({1,50,32});
 for(auto bad:{input.to(at::kFloat),at::zeros({1,50,64},at::kHalf).slice(2,0,64,2),input.narrow(1,0,49)}){
  auto before=flushes.size();bool failed=false;try{cpu.publish(bad,cpu_cond,valid_out,10,true);}
  catch(const c10::Error&){failed=true;}assert(failed&&flushes.size()==before);}
 if(!inference){auto grad=input.clone().set_requires_grad(true);auto before=flushes.size();bool failed=false;
  try{cpu.publish(grad,cpu_cond,valid_out,10,true);}catch(const c10::Error&){failed=true;}
  assert(failed&&flushes.size()==before);}
 }
 // Owner metadata failures must preserve both original scalar logical ranges.
 // Offset-32 views exercise lower/upper line rounding without invalid CPU DMA.
 g_rpu_ddr_flush_force_enabled=true;g_rpu_ddr_flush_enabled=false;
 private_misaligned=true;recording=false;replaying=false;graph_active=false;
 for(auto fault:std::vector<std::pair<int,int>>{{1,0},{1,1},{2,0},{3,0},{4,0},{5,1},{6,0},{7,0},{8,1},{0,0}}){
  ++cases;stores.clear();nodes.clear();graph.held.clear();Pi cpu;
  auto input=at::full({1,50,32},-3.5,at::kHalf),cond=at::zeros({10,1024},at::kHalf);
  auto out=bank({1632}).narrow(0,16,1600).view({1,50,32});dirty(out,0x33);
  owner_fault=fault.first;owner_fault_target=fault.second;batch_calls=owner_lookups=0;
  flushes.clear();reads.clear();cpu.publish(input,cond,out,10,true);
  assert(batch_calls==(owner_fault?0:1)&&flushes.size()==2);
  assert(flushes[0].first==cpu.x0_ref_.data_ptr()&&flushes[1].first==out.data_ptr());
  for(auto& f:flushes)assert(f.second==3200);
  cpu.consume();assert(reads.size()==1&&reads[0].size()==input.nbytes());
  assert(std::memcmp(reads[0].data(),input.const_data_ptr(),input.nbytes())==0);
  rhino_lkn::RpuDdrFlush(out.data_ptr(),out.nbytes());
  assert(std::memcmp(out.const_data_ptr(),input.const_data_ptr(),input.nbytes())==0);
 }
 owner_fault=0;private_misaligned=false;
 // Missing loop keeps scalar3; a no-table loop admits Batch3. Both retain
 // the original x0 / full current cond / out publication order and bytes.
 for(int missing=0;missing<2;++missing){
  ++cases;stores.clear();nodes.clear();graph.held.clear();Pi cpu;
  if(missing==0)cpu.loop_mode_=false;else cpu.adarms_table_=at::Tensor();
  auto input=at::full({1,50,32},-2,at::kHalf),cond=bank({10,1024}),out=bank({1,50,32});
  dirty(cond,0x44);dirty(out,0x33);flushes.clear();batch_calls=owner_lookups=0;
  cpu.publish(input,cond,out,10,true);
  assert(batch_calls==(missing==1?1:0)&&owner_lookups==(missing==1?3:0)&&flushes.size()==3);
  assert(flushes[0].first==cpu.x0_ref_.data_ptr()&&flushes[0].second==3200);
  assert(flushes[1].first==cond.data_ptr()&&flushes[1].second==cond.nbytes());
  assert(flushes[2].first==out.data_ptr()&&flushes[2].second==3200);
  size_t cond_offset;auto& cond_store=locate(cond.data_ptr(),cond.nbytes(),cond_offset);
  for(size_t i=0;i<cond.nbytes();++i)assert(cond_store.ddr[cond_offset+i]==0x44);
 }
 // Lazy negative CPU views must fail before copying, publication or Graph lease.
 // The previous ordinary negative-value tensors passed the real memcpy/DMA path.
 ++cases;stores.clear();nodes.clear();graph.held.clear();Pi cpu;
 auto plain=at::full({1,50,32},-3.5,at::kHalf),lazy=at::_neg_view(plain);
 auto cond=at::zeros({10,1024},at::kHalf),out=bank({1,50,32});
 assert(lazy.is_neg()&&lazy.is_contiguous()&&lazy.scalar_type()==at::kHalf);
 flushes.clear();batch_calls=owner_lookups=0;graph_active=true;
 const auto before_stores=stores.size(),before_nodes=nodes.size();bool failed=false;
 try{cpu.publish(lazy,cond,out,10,true);}catch(const c10::Error&){failed=true;}
 assert(failed&&stores.size()==before_stores&&nodes.size()==before_nodes);
 assert(flushes.empty()&&batch_calls==0&&owner_lookups==0&&graph.held.empty());
 assert(!cpu.x0_ref_.defined()&&cpu.x0_src_base_==0);graph_active=false;
 std::cout<<cases<<" dirty boundary cases\n";
}
'''
    replacements = {
        "__MACROS__": macros, "__LIVE_BASES__": _function(siglip, "struct PatchEmbeddingLiveBases") + ";",
        "__PIXEL_DMA__": pixel_dma, "__IMAGE_PUBLICATION__": image_publication,
        "__IMAGE_GUARDS__": image_guards,
        "__SPAN_PROOF__": span_proof, "__GUARDS__": guards,
        "__PUBLICATION__": publication, "__X0_DMA__": x0_dma, "__OUTPUT_DMA__": output_dma,
    }
    for marker, body in replacements.items():
        cpp = cpp.replace(marker, body)
    assert _run_native(tmp_path, cpp, torch).strip() == "149 dirty boundary cases"


def test_siglip_cold_and_stale_shapes_keep_exact_native_attention_routes(tmp_path, monkeypatch):
    # No owner policy is recreated here: compile its actual predicate, manifest
    # append and both DDR consume blocks, plus the real FMB planner/consumer.
    # The owner fixture supplies geometry only; it cannot allocate RPU storage.
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    owner = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    # Both real caller scopes provide the full current physical length. The
    # owner member seq_len_ is deliberately zero/stale in the native dry probe.
    dry_scope = _function(base, "FusedModelBase::resolve_prefill_stage_domain_for_shape_impl(")
    live_scope = _function(base, "at::Tensor FusedModelBase::run_all_layers_impl(")
    assert "const InferenceContext saved_ctx = pimpl_->ctx_;" in dry_scope
    assert "make_scope_exit([&] { pimpl_->ctx_ = saved_ctx; })" in dry_scope
    for scope, consumer in ((dry_scope, "physical_manifest_domain_for_candidate("),
                            (live_scope, "subclass_spm_kv_by_mha_eligible(")):
        assignment = re.search(r"pimpl_->ctx_\.seq_len\s*=\s*seq_len;", scope)
        assert assignment is not None and assignment.start() < scope.index(consumer)
    predicate = _function(owner, "    bool subclass_spm_kv_by_mha_eligible(").replace(
        " const override", " const")
    start = owner.index("            const int64_t per_image_ctx =",
                        owner.index("    FmbPhysicalExecutionManifest physical_manifest_for_candidate("))
    append = owner[start:owner.index("            append_linear(SIGLIP_O_LINEAR_SITE", start)]
    live = owner[owner.index("        } else if (image_batch_count_ > 1) {",
                             owner.index("// SDPA over the FULL bidirectional KV cache")):]
    minibatch = _function(live, "            if (ctx().has_complete_physical_manifest()) {")
    unified = _function(live[live.index("        } else {"):],
                        "            if (ctx().has_complete_physical_manifest()) {")
    constants = "\n".join(line for line in owner.splitlines()
                          if line.startswith("constexpr int64_t SIGLIP_")
                          and ("ATTN_SITE" in line or "FALLBACK_DDR_REQUIRED" in line))
    capability = _function(helpers, "inline uint32_t GetMhaAttnUnivVlmNeed(") + "\n" + _function(
        helpers, "inline bool sdpa_by_mha_spm_is_valid(")
    vlm_limit, = re.findall(r"^#define SDPA_MAX_VLM_SIZE .+$", helpers, re.M)
    source = r'''
#include "core/fused_model_base.h"
#include "core/rpu_profile.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
using namespace v3;
''' + vlm_limit + "\n" + capability + "\n" + constants + r'''
struct HostContext {
    int64_t seq_len=0, batch_size=1, position=0;
    bool is_causal=false;
    std::optional<at::Tensor> attention_mask;
    FmbThreeStageChunkPlan stage_plan;
    FmbPhysicalManifestConsumer consumer;
    bool has_complete_physical_manifest() const { return consumer.complete(); }
    const FmbRouteManifestEntry& consume_physical_route(
        FmbRouteFamily family, int64_t site, int64_t selector, int64_t flags,
        at::IntArrayRef args, int64_t invocation) const {
        return consumer.consume_route(family,site,selector,flags,args,invocation);
    }
};
struct Owner {
    int cores=4;
    int64_t seq_len_=0, image_batch_count_=1, orig_head_dim_=72;
    std::vector<int> layer_weights_=std::vector<int>(27);
    HostContext state;
    const HostContext& ctx() const { return state; }
    int num_layers() const { return 27; }
    int64_t hidden_size() const { return 1152; }
    int64_t intermediate_size() const { return 4352; }
    int64_t num_q_heads() const { return 16; }
    int64_t num_kv_heads() const { return 16; }
    int64_t head_dim() const { return 80; }
    int num_cores() const { return cores; }
''' + predicate + r'''
    FmbPhysicalExecutionManifest dry(const LayoutContext& layout) const {
        const auto& plan=ctx().stage_plan;
        const int64_t physical_len=ctx().seq_len;
        const bool raw_spm=false;
        const bool raw_spm_eligible=subclass_spm_kv_by_mha_eligible(plan,layout,0);
        FmbPhysicalExecutionManifest manifest;
        manifest.state=FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length=manifest.physical_length=physical_len;
        manifest.kv_logical_length=manifest.kv_insert_physical_rows=physical_len;
        manifest.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
        for (const ChunkInfo& chunk : plan.compute.chunks) {
''' + append + r'''
        }
        validate_fmb_physical_manifest(manifest,physical_len,0);
        return manifest;
    }
    void dispatch() const {
        const int64_t per_image_ctx=seq_len_/image_batch_count_;
        const int64_t nq=num_q_heads(), hd=head_dim();
        for (const ChunkInfo& chunk : ctx().stage_plan.compute.chunks) {
            const int64_t seq_len=chunk.len;
            if (image_batch_count_>1) {
''' + minibatch + r'''
            } else {
''' + unified + r'''
            }
        }
    }
};
FmbThreeStageChunkPlan plan(int64_t images, int64_t chunk=0) {
    const int64_t seq=images*256;
    if (!chunk) chunk=seq;
    std::vector<ChunkInfo> input,compute;
    std::vector<FmbExecutionSpan> spans;
    for (int64_t i=0;i<images;++i) {
        input.push_back({static_cast<int>(i),i*256,256,(i+1)*256});
        spans.push_back({i*256,256,i});
    }
    for (int64_t off=0,i=0;off<seq;off+=chunk,++i) {
        const auto len=std::min(chunk,seq-off);
        compute.push_back({static_cast<int>(i),off,len,off+len});
    }
    return compose_fmb_three_stage_chunk_plan(input,{{0,0,seq,seq}},compute,spans,
        ChunkMode::KV_FIRST,{FmbSpanBoundaryPolicy::KEEP_LOCAL,
        FmbSpanBoundaryPolicy::ALLOW_CROSS,FmbSpanBoundaryPolicy::ALLOW_CROSS});
}
bool rejects(const Owner& owner) {
    try { owner.dispatch(); }
    catch (const c10::Error& error) {
        return std::string(error.what_without_backtrace()).find("EXACT_MISMATCH")!=std::string::npos;
    }
    return false;
}
int main() {
    std::ifstream maps("/proc/self/maps");
    const std::string loaded((std::istreambuf_iterator<char>(maps)),{});
    assert(loaded.find("rpu_backend.so")==std::string::npos);
    assert(loaded.find("librhino")==std::string::npos);
    LayoutContext layout;
    layout.is_causal=false;
    layout.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
    // Pi05 root4/root6 both use Vision4 serial256. Existing core8 additionally
    // permits packed3; this probe does not admit core4 packed3 or Vision6.
    for (auto [cores,images] : {std::pair{4,1},std::pair{8,1},std::pair{8,3}}) {
        for (int64_t previous : {0,128,256,768,1024}) {
            Owner owner;
            owner.cores=cores; owner.image_batch_count_=images;
            owner.seq_len_=previous;
            owner.state.seq_len=images*256;
            owner.state.stage_plan=plan(images);
            assert(owner.subclass_spm_kv_by_mha_eligible(owner.state.stage_plan,layout,0));
            const auto manifest=owner.dry(layout);
            assert(manifest.routes.size()==1 && manifest.routes[0].flags==0);
            owner.seq_len_=owner.state.seq_len; // Real forward_packed publication.
            owner.state.consumer.reset(manifest);
            owner.dispatch(); // Real DDR consume block and real strict consumer.
            auto wrong=manifest;
            wrong.routes[0].flags=1;
            owner.state.consumer.reset(wrong);
            assert(rejects(owner)); // The original cold bug must remain rejected.
            wrong=manifest; ++wrong.routes[0].arguments[3];
            owner.state.consumer.reset(wrong);
            assert(rejects(owner));
            wrong=manifest; wrong.routes[0].selector=2;
            owner.state.consumer.reset(wrong);
            assert(rejects(owner));
            owner.state.consumer.reset(manifest);
            owner.seq_len_+=16;
            assert(rejects(owner)); // Wrong live physical extent never inherits a plan.
            owner.seq_len_=owner.state.seq_len;
            owner.state.seq_len+=16;
            assert(!owner.subclass_spm_kv_by_mha_eligible(owner.state.stage_plan,layout,0));
            owner.state.seq_len-=16;
            for (int kind : {0,1}) {
                auto partial=owner.state.stage_plan;
                if (kind==0) partial.input.chunks[0].len-=16;
                else partial.spans[0].len-=16;
                assert(!owner.subclass_spm_kv_by_mha_eligible(partial,layout,0));
            }
            owner.state.stage_plan=plan(images,128);
            assert(!owner.subclass_spm_kv_by_mha_eligible(owner.state.stage_plan,layout,0));
            const auto split=owner.dry(layout);
            assert(split.routes[0].flags==1);
            // Packed multi-image splits are rejected by the owner's capacity
            // predicate before a manifest can be selected (tested below).
            if(images==1) {
                owner.state.consumer.reset(split);
                owner.dispatch();
            }
            owner.state.stage_plan=plan(images);
            auto masked=layout; masked.use_attn_mask=true;
            assert(!owner.subclass_spm_kv_by_mha_eligible(owner.state.stage_plan,masked,0));
            assert(!owner.subclass_spm_kv_by_mha_eligible(owner.state.stage_plan,layout,1));
            owner.cores=6;
            assert(!owner.subclass_spm_kv_by_mha_eligible(owner.state.stage_plan,layout,0));
        }
    }
    std::cout << "SigLIP current-shape dry/live exact routes: PASS\n";
}
'''
    result = _run_native(tmp_path, source, torch,
                         extra_sources=(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",))
    assert "SigLIP current-shape dry/live exact routes: PASS" in result


def test_siglip_packed_domain_matches_actual_forward_override(tmp_path, monkeypatch):
    """Real capacity policy, SPM/SDPA bounds and FMB override guard on CPU."""
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    owner = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    predicate = _function(owner, "    bool subclass_chunk_size_valid(").replace(
        " const override", " const")
    configure = _function(owner, "    void configure_chunk_for_shape(")
    profile_guards = "\n".join(_function(owner, marker) for marker in (
        "    void require_pi05_vision_fc_profile(",
        "    void require_pi05_vision_kv_spm_only_profile("))
    # This is only a negative control: removing this one condition reproduces
    # the old candidate that the original live EXACT guard correctly rejects.
    packed_guard = "if (image_batch_count_ > 1 && cs != seq_len) return false;"
    assert predicate.count(packed_guard) == 1
    old_predicate = predicate.replace(packed_guard, "").replace(
        "subclass_chunk_size_valid", "old_valid")
    sdpa = helpers[helpers.index("inline int64_t sdpa_tile_m_v16_initial("):
                   helpers.index("// Non-throwing capability predicate")]
    # Register writers share this header region but are unrelated to the pure
    # tiling predicate; do not import or mock the hardware kernel API here.
    for marker in ("inline void rpu_set_scm_u32_checked(",
                   "inline void rpu_set_legacy_scm_u16_checked("):
        sdpa = sdpa.replace(_function(sdpa, marker), "")
    legacy_request = _function(base, "static int64_t legacy_chunk_override_for_exact_request(")
    start = base.index("        const int64_t effective_legacy_override =",
                       base.index("at::Tensor FusedModelBase::run_all_layers_impl("))
    override_guard = base[start:base.index("        // Validate only the exact immutable winner.", start)]
    # The domain invokes the same owner validator for every enumerated point;
    # the live path invokes the same configure method before run_all_layers.
    domain = _function(base, "FusedModelBase::resolve_prefill_stage_domain_for_shape_impl(")
    assert "this->subclass_chunk_size_valid(" in domain
    assert "enumerate_chunk_size_domain_impl(" in domain
    query = _function(owner, "    std::vector<int64_t> resolve_stage_domain(")
    forward = _function(owner, "    at::Tensor forward_packed(")
    assert "configure_chunk_for_shape(seq_len, packed_image_count)" in query
    assert "configure_chunk_for_shape(seq_len, image_batch_count_)" in forward
    cpp = r'''
#include "core/fused_model_base.h"
#include "core/rpu_profile.h"
#include "core/rpu_spm_allocator.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
using namespace v3;
#define DWIDTH 2
''' + sdpa + '\n' + legacy_request + r'''
struct Owner {
    int cores=8;
    int64_t image_batch_count_=1,configured_chunk_size_=0;
    bool audio_mode_=false,has_patch_emb_=true,qkv_w8a8_=false,mlp_w8a8_=false;
    bool pi05_vision_fc_weight_outer_opt_in_=false,pi05_vision_kv_spm_only_opt_in_=false,pi05_vision_q_resident_opt_in_=false;
    bool supported_owner=false;
    int64_t orig_head_dim_=72,projection_dim_=2048;
    struct Weight { at::Tensor q_w; };
    std::vector<Weight> layer_weights_;
    std::vector<int> layer_bias_norm_;
    int64_t num_layers() const { return 27; }
    int64_t num_kv_heads() const { return 16; }
    bool pi05_vision_supported_owner() const { return supported_owner; }
    void require_pi05_vision_owner_ln_profile(int64_t,int64_t) const {}
    void require_pi05_vision_qkv_w8a8_profile(int64_t,int64_t) const {}
 void require_pi05_vision_mlp_w8a8_profile(int64_t,int64_t) const {}
    struct State { int64_t chunk_size_override_=0; } state;
    State* pimpl_=&state;
    int num_cores() const { return cores; }
    int mlp_tp() const { return cores; }
    int64_t hidden_size() const { return 1152; }
    int64_t head_dim() const { return 80; }
    int64_t num_q_heads() const { return 16; }
    int64_t intermediate_size() const { return 4352; }
    void set_chunk_size_override(int64_t x) { state.chunk_size_override_=x; }
''' + profile_guards + '\n' + configure + '\n' + predicate + '\n' + old_predicate + r'''
    void actual_override_guard(int64_t seq_len,int64_t capacity) const {
        struct { int64_t compute_chunk_size; } candidate{capacity};
''' + override_guard + r'''
    }
};
template<class F> bool rejects(F run,const char* expected) {
    try {run();} catch(const c10::Error& error) {
        return std::string(error.what_without_backtrace()).find(expected)!=std::string::npos;
    }
    return false;
}
int main() {
    std::ifstream maps("/proc/self/maps");
    const std::string loaded((std::istreambuf_iterator<char>(maps)),{});
    assert(loaded.find("rpu_backend.so")==std::string::npos);
    assert(loaded.find("librhino")==std::string::npos);
    int false_admissions=0;
    for(int images:{2,3}) for(int configured:{0,1024}) {
        Owner m; m.configured_chunk_size_=configured;
        const int64_t total=images*256;
        m.configure_chunk_for_shape(total,images);
        assert(m.state.chunk_size_override_==total);
        assert(m.old_valid(256,total,0));
        assert(rejects([&]{m.actual_override_guard(total,256);},"EXACT_MISMATCH"));
        assert(!m.subclass_chunk_size_valid(256,total,0));
        assert(m.subclass_chunk_size_valid(total,total,0));
        for(int64_t cs=16;cs<=1024;cs+=16) {
            if(m.subclass_chunk_size_valid(cs,total,0)) {
                assert(cs==total);
                m.actual_override_guard(total,cs);
            }
            if(m.old_valid(cs,total,0) && !m.subclass_chunk_size_valid(cs,total,0))
                ++false_admissions;
        }
    }
    // Single-image auto/explicit capacities retain the existing policy,
    // including the long single-image profile and short tail behavior.
    for(int cores:{4,8}) for(int64_t total:{256,512,1564}) {
        if(cores==4 && total!=256) continue;
        Owner m; m.cores=cores; m.configure_chunk_for_shape(total,1);
        for(int64_t cs=16;cs<=1024;cs+=16)
            assert(m.old_valid(cs,total,0)==m.subclass_chunk_size_valid(cs,total,0));
    }
    for(int images:{2,3}) {
        Owner m; m.cores=4;
        assert(rejects([&]{m.configure_chunk_for_shape(images*256,images);},"serial image or bounded audio"));
        Owner p; p.cores=4; p.supported_owner=true;
        p.pi05_vision_kv_spm_only_opt_in_=true; p.pi05_vision_q_resident_opt_in_=true;
        p.configure_chunk_for_shape(images*256,images);
        assert(p.state.chunk_size_override_==images*256);
        assert(p.subclass_chunk_size_valid(images*256,images*256,0));
        assert(!p.subclass_chunk_size_valid(256,images*256,0));
    }
    Owner too_small; too_small.configured_chunk_size_=256;
    assert(rejects([&]{too_small.configure_chunk_for_shape(768,3);},"full sequence"));
    Owner too_large; too_large.configure_chunk_for_shape(1024,4);
    assert(!too_large.subclass_chunk_size_valid(1024,1024,0));
    assert(false_admissions>0);
    std::cout<<"SigLIP packed exact capacity and unchanged SPM gates: PASS\n";
}
'''
    result = _run_native(tmp_path, cpp, torch)
    assert "SigLIP packed exact capacity and unchanged SPM gates: PASS" in result


def test_siglip_projector_matrix_view_preserves_live_output_rows(tmp_path, monkeypatch):
    """Actual ATen views and DDR preconditions; no kernel or graph execution."""
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    owner = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    linear = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    start = owner.index("            const bool audio_tail =")
    view = owner[start:owner.index("            // MUTABLE write-back:", start)]
    geometry = _function(linear, "void validate_parallel_linear_geometry(")
    w4_operands = _function(linear, "static void check_linear_w4a16_operands(")
    launcher = _function(linear, "void rpu_launch_linear_ddr_kernel(")
    gate = launcher[launcher.index("  TORCH_CHECK(input.dim()"):launcher.index("  const bool in_graph")]
    # The gate also screens the conv-slab route. A projector never takes it, so
    # the slab handle is null here; the struct comes from the real header so a
    # changed layout reaches this contract instead of a stale copy of it.
    slab = _function((ROOT / "src/core/rpu_kernel_decls.h").read_text(),
                     "struct LinearConv1dSlab") + ";"
    row_offset, = re.findall(r"^      const int64_t row_dst_off = .+;$", launcher, re.M)
    source = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
#include <iostream>
#include <tuple>
#include <map>
namespace rhino_lkn { uint64_t RpuGetDevAddr(void* p) { return reinterpret_cast<uintptr_t>(p); } }
using at::Tensor;
constexpr int NUM_CORES=8;
namespace graph_dma {
bool captured=false;
bool active() {return captured;}
}
''' + slab + '\n' + geometry + '\n' + w4_operands + r'''
bool validate(const at::Tensor& input,const at::Tensor& weight,at::Tensor& output,
              int partition,int num_cores,const at::Tensor& weight_scale=at::Tensor(),
              const at::Tensor& bias=at::Tensor(),bool has_bias=false,
              c10::optional<bool> linear_acc32=c10::nullopt,
              const LinearConv1dSlab* conv_slab=nullptr) {
''' + gate + r'''
    return use_acc32;
}
struct Chunk {int64_t offset,len;};
at::Tensor actual_view(at::Tensor& projector_output_,int64_t seq_len_,
                       int64_t projection_dim_,const Chunk& chunk) {
    const int64_t seq_len=chunk.len;
    at::Tensor audio_proj2_w_; // Vision profiles have no audio tail.
    std::map<int64_t, std::pair<at::Tensor, uint64_t>> audio_proj1_slots_;
    const auto projector_out_live_base_=reinterpret_cast<uintptr_t>(projector_output_.data_ptr());
    auto slot=at::empty({seq_len,projection_dim_},projector_output_.options());
''' + view + r'''
    return proj_out_chunk;
}
int64_t actual_dma_offset(int64_t dst_offset_bytes,size_t row0,size_t n,size_t a_dwidth) {
''' + row_offset + r'''
    return row_dst_off;
}
template<class F> bool rejects(F function,const char* expected) {
    try {function();} catch(const c10::Error& error) {
        return std::string(error.what_without_backtrace()).find(expected)!=std::string::npos;
    }
    return false;
}
int main() {
    at::set_num_threads(1);
    const auto options=at::TensorOptions().dtype(at::kHalf).device(at::kCPU);
    // The extracted DDR gate also admits scaled W8 in eager col partition.
    // Supply its real operands and exercise the new state/scale boundaries.
    auto gate_input=at::empty({16,1152},options);
    auto gate_output=at::empty({16,2048},options);
    auto gate_weight=at::empty({2048,1152},options);
    auto gate_w8=at::empty({2048,1152},options.dtype(at::kChar));
    auto scale=at::ones({2048},options);
    assert(rejects([&]{validate(gate_input,gate_weight,gate_output,1,8,scale);},
                   "FP16 linear does not take a W8 scale"));
    assert(rejects([&]{validate(gate_input,gate_w8,gate_output,1,8);},
                   "requires contiguous device FP16 scale"));
    assert(!validate(gate_input,gate_w8,gate_output,1,8,scale));
    assert(validate(gate_input,gate_w8,gate_output,1,8,scale,{},false,true));
    // The legacy DDR API keeps its FP16 default, while configured model
    // callers explicitly select ACC16 (the model default) or ACC32.
    assert(validate(gate_input,gate_weight,gate_output,1,8));
    graph_dma::captured=true;
    assert(rejects([&]{validate(gate_input,gate_w8,gate_output,1,8,scale);},
                   "eager col partition only"));
    graph_dma::captured=false;
    // Keep the shared launcher's real W4 helper, including its device gate;
    // this CPU-only shape harness must not pretend to admit RPU W4 execution.
    auto gate_w4=at::empty({2048,576},options.dtype(at::kByte));
    assert(rejects([&]{validate(gate_input,gate_w4,gate_output,1,8);},
                   "input and weight must share an RPU device"));
    // Serial Pi05, packed baseline, and a nonaligned final query chunk. These
    // are view/launcher geometry checks, not full model admission certificates.
    for(auto [cores,rows,capacity,width]:{
            std::tuple{4,256,256,2048},std::tuple{4,256,64,2048},
            std::tuple{8,768,768,2048},std::tuple{8,1564,512,1536}}) {
        auto weight=at::empty({width,1152},options);
        at::Tensor previous;
        for(int forward=0;forward<2;++forward) {
            auto output=at::full({1,rows,width},-1,options);
            const auto base=reinterpret_cast<uintptr_t>(output.data_ptr());
            if(previous.defined()) assert(output.data_ptr()!=previous.data_ptr());
            for(int64_t offset=0;offset<rows;offset+=capacity) {
                const Chunk chunk{offset,std::min<int64_t>(capacity,rows-offset)};
                auto input=at::empty({chunk.len,1152},options);
                auto original_rank3=output.slice(1,offset,offset+chunk.len);
                assert(rejects([&]{validate(input,weight,original_rank3,1,cores);},"rank-2"));
                auto matrix=actual_view(output,rows,width,chunk);
                assert(matrix.dim()==2 && matrix.size(0)==chunk.len && matrix.size(1)==width);
                assert(matrix.is_contiguous() && matrix.is_alias_of(output));
                const auto byte_offset=offset*width*static_cast<int64_t>(sizeof(c10::Half));
                assert(reinterpret_cast<uintptr_t>(matrix.data_ptr())==base+byte_offset);
                assert(matrix.storage_offset()==offset*width);
                for(bool acc32:{false,true})
                    assert(validate(input,weight,matrix,1,cores,{}, {},false,acc32)==acc32);
                for(size_t row0:{size_t{0},static_cast<size_t>(chunk.len-1)}) {
                    const auto dma=actual_dma_offset(byte_offset,row0,width,sizeof(c10::Half));
                    assert(base+dma==reinterpret_cast<uintptr_t>(matrix.data_ptr<c10::Half>()+row0*width));
                }
                matrix.fill_(forward+2);
                assert(output.dim()==3 && output.size(0)==1 && output.size(1)==rows);
                assert(output.slice(1,0,offset+chunk.len).eq(forward+2).all().item<bool>());
                if(offset+chunk.len<rows)
                    assert(output.slice(1,offset+chunk.len,rows).eq(-1).all().item<bool>());
                if(previous.defined()) assert(previous.eq(2).all().item<bool>());
                auto wrong=at::empty({width,chunk.len},options).transpose(0,1);
                assert(rejects([&]{validate(input,weight,wrong,1,cores);},"contiguous"));
            }
            previous=output;
        }
    }
    std::cout << "SigLIP projector rank2 alias and mutable row offsets: PASS\n";
}
'''
    result = _run_native(tmp_path, source, torch)
    assert "SigLIP projector rank2 alias and mutable row offsets: PASS" in result


def _serial_siglip_forward(monkeypatch, *, captured, legacy_alias=False):
    """Execute the installed nested caller; only hardware/Graph are doubles.

    The native op returns a fresh tensor on every call, as the actual native
    allocation control below establishes. Deferred writes happen at capture
    exit. legacy_alias is a negative producer control, not a valid native route.
    """
    import ast
    from contextlib import contextmanager
    import sys
    from types import ModuleType, SimpleNamespace

    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    path = ROOT / "python/rpu_backend/adapters/siglip.py"
    tree = ast.parse(path.read_text())
    forward = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "rpu_siglip_forward")
    if legacy_alias:
        class RemoveOutputCopy(ast.NodeTransformer):
            def visit_Assign(self, node):
                if (isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "clone"
                        and isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id == "output"):
                    return None
                return self.generic_visit(node)
        forward = RemoveOutputCopy().visit(forward)

    state = SimpleNamespace(active=False, pending=None, failure=None,
                            events=[], native_outputs=[])

    class NativeOutput(torch.Tensor):
        def clone(self, *args, **kwargs):
            assert not state.active, "output copied before capture exit"
            state.events.append("clone")
            return super().clone(*args, **kwargs).as_subclass(torch.Tensor)

    def output(value):
        return torch.full((1, 2, 4), value, dtype=torch.float16).as_subclass(NativeOutput)

    shared = output(-99)

    @contextmanager
    def capture(signature):
        assert not state.active
        assert signature["op_id"] == "pi05_siglip_compute"
        state.events.append("enter")
        state.active = True
        try:
            yield
            if state.failure == "exit":
                raise RuntimeError("capture execution failed")
            target, value = state.pending
            target.fill_(value)
            state.events.append("exit")
        finally:
            state.pending = None
            state.active = False

    def native_forward(handle, image, keys, values, descriptor):
        assert handle == 7 and keys == ["K"] and values == ["V"]
        assert descriptor == [11, 12]
        assert state.active is captured
        if state.failure == "forward":
            raise RuntimeError("native forward failed")
        if captured:
            result = shared if legacy_alias else output(-99)
            state.pending = (result, image.value)
        else:
            result = output(image.value)  # Native PASSTHROUGH returns fresh storage.
        state.native_outputs.append(result)
        return result

    def reset_spm():
        assert not state.active
        state.events.append("reset")

    modeling_outputs = ModuleType("transformers.modeling_outputs")
    modeling_outputs.BaseModelOutputWithPooling = SimpleNamespace
    monkeypatch.setitem(sys.modules, "transformers.modeling_outputs", modeling_outputs)
    plan = SimpleNamespace(
        graph_key_words=lambda: (123,),
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(11, 12))))
    namespace = dict(
        torch=SimpleNamespace(float16=torch.float16,
                              ops=SimpleNamespace(rpu=SimpleNamespace(
                                  siglip_forward=native_forward,
                                  spm_alloc_reset_temporary=reset_spm))),
        rpu_backend=SimpleNamespace(graph=SimpleNamespace(
            GraphSignature=lambda **fields: fields)),
        _num_images=1, _seq_len=256, _num_layers=1, _num_heads=16,
        _padded_head_dim=80, _attn_tp=8, _siglip_sig_w8a16=0,
        qkv_w8a8=False, mlp_w8a8=False,
        _siglip_sig_qkv_w8a8=0, _siglip_sig_mlp_w8a8=0,
        _split_siglip_graph=captured, _SIGLIP_BRANCH_PATCH_EMBED=1,
        _SIGLIP_BRANCH_PRE_EMBEDDED=2,
        _siglip_component_execution_plan=lambda *args, **kwargs: plan,
        check_pre_embedded_rows=lambda *args: None,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[])),
                 str(path), "exec"), namespace)
    cache = SimpleNamespace(k_caches=["K"], v_caches=["V"],
                            reset_to_position=lambda position: None)
    model = SimpleNamespace(_rpu_vision_handle=7, _rpu_cache=cache,
                            _rpu_siglip_graph_cache=SimpleNamespace(capture=capture))

    def call(value):
        image = SimpleNamespace(device=SimpleNamespace(type="rpu"), dtype=torch.float16,
                                is_contiguous=lambda: True, shape=(1, 3, 224, 224),
                                dim=lambda: 4, value=value)
        return namespace["rpu_siglip_forward"](model, pixel_values=image).last_hidden_state

    return torch, state, call


@pytest.mark.parametrize("captured", [True, False])
def test_serial_siglip_forward_preserves_retained_camera_outputs(monkeypatch, captured):
    torch, state, call = _serial_siglip_forward(monkeypatch, captured=captured)
    retained = [call(value) for value in (1, 2, 3)]
    combined = torch.cat(retained, dim=1)
    expected = torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.float16).reshape(1, 6, 1)
    assert torch.equal(combined, expected.expand(1, 6, 4))
    assert len({tensor.data_ptr() for tensor in retained}) == 3
    if captured:
        assert state.events == ["enter", "exit", "reset"] * 3
        assert len({tensor.data_ptr() for tensor in state.native_outputs}) == 3
        assert all(retained[i] is state.native_outputs[i] for i in range(3))
    else:
        assert state.events == ["reset"] * 3
        assert all(retained[i] is state.native_outputs[i] for i in range(3))


def test_serial_siglip_shared_native_producer_reproduces_last_camera_alias(monkeypatch):
    torch, state, call = _serial_siglip_forward(
        monkeypatch, captured=True, legacy_alias=True)
    retained = [call(value) for value in (1, 2, 3)]
    assert len({tensor.data_ptr() for tensor in retained}) == 1
    assert torch.equal(torch.cat(retained, dim=1), torch.full((1, 6, 4), 3.0))
    assert state.events == ["enter", "exit", "reset"] * 3


@pytest.mark.parametrize("failure", ["forward", "exit"])
def test_serial_siglip_failed_capture_returns_no_stale_copy(monkeypatch, failure):
    torch, state, call = _serial_siglip_forward(monkeypatch, captured=True)
    first = call(1)
    state.events.clear()
    state.failure = failure
    with pytest.raises(RuntimeError, match="failed"):
        call(2)
    assert not state.active and state.pending is None
    assert "clone" not in state.events and "reset" not in state.events
    assert torch.equal(first, torch.ones((1, 2, 4)))
    state.failure = None
    third = call(3)
    assert torch.equal(first, torch.ones((1, 2, 4)))
    assert torch.equal(third, torch.full((1, 2, 4), 3.0))


def test_siglip_actual_forward_allocates_retained_mutable_projector_outputs(tmp_path, monkeypatch):
    """Actual allocation/forward/view/mutable-consumer code; Graph/SDK doubles.

    Device labels are substituted for host ATen allocation. Planning, attention,
    and Linear math are outside this ownership test. It does not claim hardware
    Graph or SPM qualification.
    """
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    owner = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    linear = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    allocate = _function(base, "at::Tensor FusedModelBase::allocate_tracked_output(")
    allocate = allocate.replace("FusedModelBase::", "").replace("at::kPrivateUse1", "at::kCPU")
    forward = _function(owner, "    at::Tensor forward_packed(").replace("at::kPrivateUse1", "at::kCPU")
    start = owner.index("            const bool audio_tail =")
    end = owner.index("configured_linear_acc32_);", start) + len("configured_linear_acc32_);")
    consumer = owner[start:end]
    launcher = _function(linear, "void rpu_launch_linear_ddr_kernel(")
    row_offset, = re.findall(r"^      const int64_t row_dst_off = .+;$", launcher, re.M)
    start = launcher.index("        rpu_launch_spm_copy_ddr_dma_mutable(g_addr, live_dst_base, row_dst_off,")
    output_dma = launcher[start:launcher.index(";", start) + 1]
    # This mutation is solely a negative producer control: retain one old
    # projector owner instead of the real per-forward allocation.
    assignment, = re.findall(r"projector_output_ = allocate_tracked_output\([^;]+\);", forward)
    assert forward.count(assignment) == 1
    old_forward = forward.replace("forward_packed(", "old_forward_packed(", 1).replace(
        assignment, "if (!projector_output_.defined()) " + assignment)
    # This installed API cannot admit a missing projector. No artificial
    # projector-free runtime fallback or support claim is introduced here.
    install = _function(owner, "    void set_weights(")
    assert "intermediate_size > 0 && projection_dim > 0" in install
    assert 'check_global_fp16_rpu(proj_w,    "proj_w")' in install
    cpp = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>
using at::Tensor;
struct LayoutContext { int64_t chunk_size=0,num_layers=0; };
struct BufferDecl {};
namespace detail { int64_t estimate_temporary_total(const std::vector<BufferDecl>&) { return 0; } }
struct SpmAllocator { static constexpr size_t SPM_PLANNING_BUDGET=8303708; };
enum class FmbRouteFamily { LINEAR };
enum class FmbLinearRouteSelector { AUTO_TILE=1 };
int decode_fmb_prefill_stage_candidate(at::IntArrayRef x) { assert(!x.empty()); return 1; }
namespace rhino_lkn { uint64_t RpuGetDevAddr(void* p) { return reinterpret_cast<uintptr_t>(p); } }
struct Chunk {int64_t idx,offset,len;};
enum class Mode { PASSTHROUGH,BUILD,REPLAY,FAST_REPLAY };
struct Graph {
  Mode mode=Mode::PASSTHROUGH;
  std::vector<at::Tensor> refs;
  void keep_alive(const at::Tensor& t) { refs.push_back(t); }
};
Graph graph;
struct RpuKernelGraph {
 static bool has_active() {return graph.mode!=Mode::PASSTHROUGH;}
 static Graph& active() {return graph;}
};
#define rpu_ddr_flush(ptr) ((void)(ptr))
struct Dma { const uint64_t* live;int64_t offset,elements; };
std::vector<Dma> nodes;
size_t cursor=0;
void rpu_launch_spm_copy_ddr_dma_mutable(uint32_t,const uint64_t* live,
                                       int64_t offset,int64_t elements) {
 if(graph.mode==Mode::REPLAY) {
   assert(cursor<nodes.size());
   const auto& old=nodes[cursor++];
   assert(old.live==live && old.offset==offset && old.elements==elements);
 } else nodes.push_back({live,offset,elements});
}
void rpu_launch_linear_ddr_kernel(const Tensor& input,const Tensor&,const Tensor& output,
    const Tensor&,bool has_bias,int partition,const uint64_t* live_dst_base,
    int64_t dst_offset_bytes,const uint64_t*,int cores,const Tensor&,
    std::optional<bool>) {
 assert(has_bias && partition==1 && live_dst_base && cores>=4);
 assert(input.dim()==2 && output.dim()==2 && output.is_contiguous());
 assert(reinterpret_cast<uintptr_t>(output.data_ptr())==*live_dst_base+dst_offset_bytes);
 const int64_t row0=0,n=output.size(1),cur_m=output.size(0),a_dwidth=2;
 const uint32_t g_addr=0;
''' + row_offset + '\n' + output_dma + r'''
}
struct Owner {
 int cores=4;
 int64_t image_batch_count_=1,seq_len_=256,projection_dim_=2048,local_q_heads_=4;
 bool pi05_vision_kv_spm_only_opt_in_=false,qkv_w8a8_=false,mlp_w8a8_=false;
 std::map<std::pair<int64_t,int64_t>,at::Tensor> q_ddr_slots_;
 at::Tensor projector_output_,proj_w_,proj_b_=at::zeros({2048},at::kHalf);
 at::Tensor audio_proj2_w_; // Vision profiles have no audio tail.
 std::map<int64_t, std::pair<at::Tensor, uint64_t>> audio_proj1_slots_;
 uint64_t projector_out_live_base_=0;
 std::optional<bool> configured_linear_acc32_=false;
 int64_t chunk_capacity=256;
 bool fail=false;
 int64_t num_layers() const {return 27;}
 int64_t hidden_size() const {return 1152;}
 int64_t head_dim() const {return 80;}
 int num_cores() const {return cores;}
 void configure_chunk_for_shape(int64_t,int64_t) {}
 void validate_pi05_vision_owner_ln_policy(int,int64_t,int64_t) {}
 void validate_pi05_vision_qkv_w8a8_policy(int,int64_t,int64_t) {}
 void validate_pi05_vision_mlp_w8a8_policy(int,int64_t,int64_t) {}
 void validate_pi05_vision_kv_spm_only_policy(int,int64_t,int64_t) {}
 std::vector<BufferDecl> declare_buffers(const LayoutContext&) {return {};}
 void consume_manifest_route(FmbRouteFamily,int,int64_t,int64_t) {}
 static constexpr int SIGLIP_PROJECTOR_LINEAR_SITE=1;
''' + allocate + r'''
 void actual_consumer(const Chunk& chunk) {
   const int64_t seq_len=chunk.len;
   auto slot=at::empty({seq_len,1152},at::kHalf);
''' + consumer + r'''
 }
 at::Tensor run_all_layers(const at::Tensor&,std::vector<at::Tensor>&,
    std::vector<at::Tensor>&,std::optional<at::Tensor>,int64_t,bool,int64_t,
    at::IntArrayRef) {
   assert(projector_output_.defined());
   assert(projector_out_live_base_==reinterpret_cast<uintptr_t>(projector_output_.data_ptr()));
   if(RpuKernelGraph::has_active()) {
     assert(!graph.refs.empty());
     assert(graph.refs.back().is_same(projector_output_));
   }
   if(fail) TORCH_CHECK(false,"deliberate FMB execution failure");
   if(graph.mode!=Mode::FAST_REPLAY) {
     cursor=0;
     for(int64_t offset=0,index=0;offset<seq_len_;offset+=chunk_capacity,++index)
       actual_consumer({index,offset,std::min(chunk_capacity,seq_len_-offset)});
     if(graph.mode==Mode::REPLAY) assert(cursor==nodes.size());
   }
   return at::empty({1,seq_len_,1152},at::kHalf); // Ignored genuine FMB return.
 }
''' + forward + '\n' + old_forward + r'''
};
void finish(uint16_t value) {
 for(const auto& dma:nodes) {
   assert(dma.live && *dma.live);
   auto* ptr=reinterpret_cast<uint16_t*>(*dma.live+dma.offset);
   for(int64_t i=0;i<dma.elements;++i)ptr[i]=value;
 }
 graph.refs.clear(); // Actual adapter owns its returned tensor after execution.
}
bool bits(const Tensor& x,uint16_t value) {
 // Keep the full raw-bit check for every retained output after every call.
 // ATen's vectorized comparison avoids repeatedly scanning hundreds of MiB
 // in the fixture's deliberately unoptimized C++ scalar loop on ARM hosts.
 return x.view(at::kShort).eq(static_cast<int16_t>(value)).all().item<bool>();
}
int main() {
 at::set_num_threads(1);
 std::vector<Tensor> retained;
 std::vector<uint16_t> expected;
 for(bool inference:{false,true}) {
  c10::InferenceMode inference_guard(inference);
  for(auto [cores,rows,capacity,width]:{
      std::tuple{4,256,256,2048},std::tuple{8,768,768,2048},
      std::tuple{8,1564,512,1536}}) {
   Owner m; m.cores=cores;m.chunk_capacity=capacity;m.projection_dim_=width;
   m.proj_b_=at::zeros({width},at::kHalf);
   auto input=at::zeros({1,rows,1152},at::kHalf);
   const bool version_enabled=input.unsafeGetTensorImpl()->version_counter().enabled();
   const auto version=version_enabled?input.unsafeGetTensorImpl()->version_counter().current_version():0;
   std::vector<Tensor> keys,values;
   const std::vector<int64_t> descriptor={1,2};
   for(Mode mode:{Mode::PASSTHROUGH,Mode::BUILD,Mode::REPLAY,Mode::FAST_REPLAY}) {
    graph.mode=mode;if(mode==Mode::PASSTHROUGH || mode==Mode::BUILD)nodes.clear();
    for(uint16_t value:{uint16_t{0x3c00},uint16_t{0x4000},uint16_t{0x3c00}}) {
     if(mode==Mode::PASSTHROUGH)nodes.clear();
     // Writes through raw CPU alias do not bump the real source version.
     reinterpret_cast<uint16_t*>(input.data_ptr())[0]=value;
     if(version_enabled)assert(input.unsafeGetTensorImpl()->version_counter().current_version()==version);
     auto output=m.forward_packed(input,rows==768?3:1,keys,values,descriptor);
     assert(output.dim()==3 && output.size(1)==rows && output.size(2)==width);
     assert(output.is_same(m.projector_output_));
     for(const auto& old:retained)assert(!old.is_alias_of(output));
     if(mode==Mode::BUILD) {
       // Three camera calls are separate Graph executions of the same entry;
       // the helper double records a fresh node list for each BUILD control.
       assert(!nodes.empty());
     }
     finish(value);
     assert(bits(output,value));
     for(size_t i=0;i<retained.size();++i)assert(bits(retained[i],expected[i]));
     retained.push_back(output);expected.push_back(value);
     if(mode==Mode::BUILD)nodes.clear();
    }
    if(mode==Mode::BUILD) {
      // Restore exactly one recorded graph for subsequent ordinary/fast REPLAY.
      graph.refs.clear();auto output=m.forward_packed(input,rows==768?3:1,keys,values,descriptor);
      finish(uint16_t{0x3c00});retained.push_back(output);expected.push_back(0x3c00);
    }
   }
   auto previous=retained.back();m.fail=true;bool failed=false;
   try {m.forward_packed(input,rows==768?3:1,keys,values,descriptor);}catch(const c10::Error&){failed=true;}
   assert(failed && bits(previous,expected.back()));graph.refs.clear();m.fail=false;
   auto recovered=m.forward_packed(input,rows==768?3:1,keys,values,descriptor);finish(0x4000);
   assert(bits(recovered,0x4000) && bits(previous,expected.back()));
   retained.push_back(recovered);expected.push_back(0x4000);
  }
 }
 // Mutating only the genuine allocation line reproduces the old producer bug.
 graph.mode=Mode::PASSTHROUGH;nodes.clear();Owner legacy;
 auto input=at::zeros({1,256,1152},at::kHalf);std::vector<Tensor> kv;
 const std::vector<int64_t> descriptor={1};
 auto first=legacy.old_forward_packed(input,1,kv,kv,descriptor);finish(0x3c00);
 nodes.clear();auto second=legacy.old_forward_packed(input,1,kv,kv,descriptor);finish(0x4000);
 assert(first.is_alias_of(second) && bits(first,0x4000));
 // Original native/model owners are gone; retained ATen owners still hold data.
 nodes.clear();legacy.projector_output_=Tensor{};
 for(size_t i=0;i<retained.size();++i)assert(bits(retained[i],expected[i]));
 std::cout<<"Actual SigLIP fresh allocation/mutable output and retained raw ABA: PASS\n";
}
'''
    result = _run_native(tmp_path, cpp, torch)
    assert "Actual SigLIP fresh allocation/mutable output and retained raw ABA: PASS" in result


def test_hyvit2_projector_uses_matrix_metadata_without_copying_output(tmp_path, monkeypatch):
    """Execute HYViT2's output view/call and the DDR launcher's actual gates."""
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    owner = (ROOT / "src/fused/rpu_hyvit2_vision_model.cpp").read_text()
    linear = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    start = owner.index("            at::Tensor proj_out_chunk =")
    end = owner.index("&projector_bias_live_base_);", start) + len("&projector_bias_live_base_);")
    consumer = owner[start:end] + "\n            }"
    geometry = _function(linear, "void validate_parallel_linear_geometry(")
    w4_operands = _function(linear, "static void check_linear_w4a16_operands(")
    launcher = _function(linear, "void rpu_launch_linear_ddr_kernel(")
    gates = launcher[launcher.index("  TORCH_CHECK(input.dim()"):launcher.index("  const bool in_graph")]
    # As in the cold-projector contract above: the projector passes no slab, and
    # the struct is taken from the header rather than restated here.
    slab = _function((ROOT / "src/core/rpu_kernel_decls.h").read_text(),
                     "struct LinearConv1dSlab") + ";"
    rank_start = launcher.index("  TORCH_CHECK(input.dim()")
    rank_guard = launcher[rank_start:launcher.index(";", rank_start) + 1]
    source = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
using at::Tensor;
constexpr int NUM_CORES=8;
namespace graph_dma { bool active() { return false; } }
''' + slab + '\n' + geometry + '\n' + w4_operands + r'''
void rank_gate(const Tensor& input,const Tensor& weight,const Tensor& output) {
''' + rank_guard + r'''
}
float value=0;
int linear_calls=0,dma_calls=0;
void rpu_launch_linear_ddr_kernel(const Tensor& input,const Tensor& weight,
    Tensor& output,const Tensor& bias,bool has_bias,int partition,
    const uint64_t* live_dst_base,int64_t dst_offset_bytes,
    const uint64_t* live_bias_base) {
    const int num_cores=8;
    const Tensor weight_scale;
    const c10::optional<bool> linear_acc32=c10::nullopt;
    const LinearConv1dSlab* conv_slab=nullptr;
''' + gates + r'''
    assert(use_acc32 && has_bias && live_dst_base && live_bias_base);
    assert(reinterpret_cast<uintptr_t>(output.data_ptr())==*live_dst_base+dst_offset_bytes);
    assert(reinterpret_cast<uintptr_t>(bias.data_ptr())==*live_bias_base);
    output.fill_(value);++linear_calls;
}
void rpu_launch_spm_copy_ddr_dma_mutable(uint32_t,const uint64_t*,int64_t,int64_t) {
    ++dma_calls;
}
enum class FmbRouteFamily { MUTABLE_DMA,LINEAR };
enum class HyViT2DmaRoute { SPM_TO_DDR_MUTABLE };
enum class FmbLinearRouteSelector { AUTO_TILE };
constexpr int HYVIT2_TAIL_OUTPUT_DMA_SITE=1,HYVIT2_PROJECTOR_LINEAR_SITE=2;
struct Context {
    bool has_complete_physical_manifest() { return false; }
    void consume_physical_route(FmbRouteFamily,int,int64_t,int64_t,
                                std::initializer_list<int64_t>,int64_t) { assert(false); }
};
struct Chunk { int64_t offset,len,idx=0; };
struct Owner {
    Tensor projector_output_,proj_w_,proj_b_;
    int64_t seq_len_=64,projection_dim_=128;
    bool proj1_in_merger_=false,merger_member_major_=false;
    uint64_t projector_out_live_base_=0,projector_bias_live_base_=0;
    Context context;
    Context& ctx() { return context; }
    Owner():projector_output_(at::zeros({1,64,128},at::kHalf)),
            proj_w_(at::zeros({128,64},at::kHalf)),proj_b_(at::zeros({128},at::kHalf)) {
        projector_out_live_base_=reinterpret_cast<uintptr_t>(projector_output_.data_ptr());
        projector_bias_live_base_=reinterpret_cast<uintptr_t>(proj_b_.data_ptr());
    }
    void project(const Chunk& chunk,bool p1m=false) {
        const int64_t seq_len=chunk.len,out_dim=projection_dim_,h=64,DWIDTH=2;
        const uint32_t tail_src=0;
        auto slot=at::zeros({seq_len,h},at::kHalf);
''' + consumer + r'''
    }
};
int main() {
    at::set_num_threads(1);
    Owner owner;
    const auto original=owner.projector_output_.data_ptr();
    // Before the fix, both whole-output and row-slice tensors are rank three.
    for (Chunk chunk : {Chunk{0,64},Chunk{0,16},Chunk{16,48}}) {
        auto old_output=owner.projector_output_.slice(1,chunk.offset,chunk.offset+chunk.len);
        auto input=at::zeros({chunk.len,64},at::kHalf);
        bool rejected=false;
        try {rank_gate(input,owner.proj_w_,old_output);}
        catch(const c10::Error& error) {
            rejected=std::string(error.what_without_backtrace()).find("rank-2")!=std::string::npos;
        }
        assert(rejected);
    }
    value=3;owner.project({0,16});
    assert(owner.projector_output_.slice(1,0,16).eq(3).all().item<bool>());
    assert(owner.projector_output_.slice(1,16,64).eq(0).all().item<bool>());
    value=5;owner.project({16,48});
    assert(owner.projector_output_.slice(1,0,16).eq(3).all().item<bool>());
    assert(owner.projector_output_.slice(1,16,64).eq(5).all().item<bool>());
    value=7;owner.project({0,64});
    assert(owner.projector_output_.eq(7).all().item<bool>());
    assert(owner.projector_output_.data_ptr()==original);
    assert(linear_calls==3 && dma_calls==0);
    owner.project({0,64},true);
    assert(linear_calls==3 && dma_calls==1);
    std::cout << "HYViT2 rank-2 projector and output alias: PASS\n";
}
'''
    assert "HYViT2 rank-2 projector and output alias: PASS" in _run_native(tmp_path, source, torch)

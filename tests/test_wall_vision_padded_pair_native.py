"""Execute the padded-pair native masks and actual aliased-SPM estimator on host."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
VISION = (ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp").read_text()


def _body(marker, source=VISION):
    begin = source.index(marker)
    brace = source.index("{", begin)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def _run_cpp(tmp_path, name, source):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for native pair metadata checks")
    path = tmp_path / f"{name}.cpp"
    path.write_text(source)
    binary = tmp_path / name
    result = subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_native_pair_mask_isolates_padding_and_preserves_finite_queries(tmp_path):
    _run_cpp(tmp_path, "pair_masks", r"""
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("reject"); } while(0)
namespace c10 { using Half = float; }
namespace at {
enum Dtype { kHalf, kOther };
struct Tensor {
    Dtype dtype=kHalf; bool cpu=true, contiguous=true, present=true;
    std::vector<int64_t> shape{336,336};
    std::vector<float> values=std::vector<float>(336*336, -std::numeric_limits<float>::infinity());
    bool defined() const { return present; }
    struct Device { bool cpu; bool is_cpu() const { return cpu; } };
    Device device() const { return {cpu}; }
    Dtype scalar_type() const { return dtype; }
    bool is_contiguous() const { return contiguous; }
    int dim() const { return shape.size(); }
    int64_t size(int i) const { return shape.at(i); }
    template<class T> const T* data_ptr() const { return values.data(); }
};
}
""" + _body("    static void validate_padded_pair_mask(") + r"""
template<class F> void rejected(at::Tensor t, bool full, F mutate) {
    mutate(t); bool failed=false;
    try {validate_padded_pair_mask(t,full);} catch(const std::runtime_error&) {failed=true;}
    assert(failed);
}
int main() {
    at::Tensor window, full;
    for(int q=0;q<336;++q) for(int k=0;k<336;++k) {
        if(q>=324 && k>=324) window.values[q*336+k]=full.values[q*336+k]=0;
        if(q<324 && k<324) {
            full.values[q*336+k]=0;
            if(q/64==k/64) window.values[q*336+k]=0;
        }
    }
    validate_padded_pair_mask(window,false); validate_padded_pair_mask(full,true);
    rejected(window,false,[](auto&t){t.values[323*336+324]=0;});
    rejected(window,false,[](auto&t){t.values[324*336+323]=0;});
    rejected(window,false,[](auto&t){t.values[335*336+335]=-INFINITY;});
    rejected(window,false,[](auto&t){t.values[8*336+8]=-INFINITY;});
    rejected(window,false,[](auto&t){t.values[8*336+9]=NAN;});
    rejected(window,false,[](auto&t){t.values[8*336+9]=INFINITY;});
    rejected(window,false,[](auto&t){t.values[8*336+9]=1;});
    rejected(full,true,[](auto&t){t.values[8*336+9]=-INFINITY;});
    rejected(full,true,[](auto&t){t.shape={324,336};});
    rejected(full,true,[](auto&t){t.shape={1,336,336};});
    rejected(full,true,[](auto&t){t.cpu=false;});
    rejected(full,true,[](auto&t){t.dtype=at::kOther;});
    rejected(full,true,[](auto&t){t.contiguous=false;});
}
""")


def _generic_spm_cpp():
    helper = (ROOT / "src/core/rpu_helpers.h").read_text()
    tiling = helper[helper.index("inline int64_t sdpa_tile_m_v16_initial("):
                    helper.index("#ifndef SCM_REG_OFFSET")]
    tiling += _body("inline bool sdpa_try_compute_tiling(", helper)
    tiling += _body("inline SdpaTiling sdpa_compute_tiling(", helper)
    declarations = _body("    std::vector<BufferDecl> declare_buffers(")
    # The new owner is Generic. Remove only the unreachable exact-MIMC branch
    # from this SDK-free compile; sizes, declaration order, lifetimes, aliases,
    # tiling and allocator placement below are the production implementations.
    declarations = declarations.replace(_body("        if (is_exact_mimc()) {", declarations), "")
    declarations = declarations.replace(" override", "")
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    return r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("reject"); } while(0)
template<class T> T CeilDiv(T n,T d){return (n+d-1)/d;}
int64_t Align(int64_t n,int64_t d){return CeilDiv(n,d)*d;}
""" + tiling + r"""
struct SpmAllocator {
    struct AllocRequest { int64_t bytes; int first_step,last_step,scope; };
    struct AliasedPlan { std::vector<uint32_t> offsets; size_t peak_bytes=0; };
    static size_t align_up(size_t n){return (n+255)/256*256;}
    static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);
};
""" + _body("SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(", allocator) + r"""
constexpr int64_t DWIDTH=2, NUM_CORES=8, QWEN25VL_VISION_SDPA_QUERY_CHUNK=144;
enum class StorageClass {Temp, Persistent, PersistentPerLayer};
enum class BufferScope {LayerWide=0,KvInsert=1,Compute=2,OutsideLayerLoop=3};
enum class FmbRopeTableResidency {UNSPECIFIED,SPM,DDR};
enum class AttentionExecutionPolicy {SPM_KV_BY_MHA,DDR_KV};
enum class VisionSchedule {MISC,SISC};
struct BufferDecl {const char* name; int64_t size; int phase_start,phase_end; StorageClass storage; int per_layer; const char* alias_of; BufferScope scope=BufferScope::LayerWide;};
struct LayoutContext {
    int64_t chunk_size=672;
    FmbRopeTableResidency rope_table_residency=FmbRopeTableResidency::SPM;
    AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
};
struct Model {
    int64_t intermediate=3584, image_batch_count_=2, max_hw_=128;
    int64_t merge_hidden_=5120,merger_out_hidden_=2048;
    bool rope_spm_enabled_=true,window_mask_present_=true,merger=true,resident_mask=true,pair=true;
    VisionSchedule schedule_=VisionSchedule::MISC;
    int64_t hidden_size() const{return 1280;}
    int64_t num_q_heads() const{return 16;}
    int64_t head_dim() const{return 80;}
    int64_t intermediate_size() const{return intermediate;}
    int64_t num_layers() const{return 32;}
    bool is_generic() const{return true;}
    bool merger_active() const{return merger;}
    bool small_image_native_window_enabled() const{return false;}
    bool small_image_resident_mask_enabled() const{return resident_mask;}
    bool padded_pair_enabled() const{return pair;}
""" + declarations + r"""
};
"""


def test_real_generic_declarations_and_allocator_fit_pair_without_budget_override(tmp_path):
    stdout = _run_cpp(tmp_path, "pair_spm", _generic_spm_cpp() + r"""
int main() {
    for(int64_t width:{3456,3584}) {
        Model m;m.intermediate=width;
        const auto decls=m.declare_buffers({});
        std::vector<SpmAllocator::AllocRequest> requests;
        int64_t persistent=0,mask_count=0;
        for(const auto& d:decls) {
            if(d.storage==StorageClass::Temp && !d.alias_of)
                requests.push_back({d.size,d.phase_start,d.phase_end,0});
            if(d.storage==StorageClass::Persistent) persistent+=d.size;
            if(d.storage==StorageClass::PersistentPerLayer) persistent+=d.size*d.per_layer;
            if(std::string(d.name)=="sdpa_mask" || std::string(d.name)=="sdpa_full_mask") {
                assert(d.size==225792 && d.phase_start==1 && d.phase_end==9);++mask_count;
            }
        }
        const auto plan=SpmAllocator::plan_temporary_aliased(requests);
        const int64_t expected_temporary=width==3456?6408192:6429696;
        assert(mask_count==2 && plan.peak_bytes==expected_temporary && persistent==460800);
        constexpr int64_t existing_full_vla_persistent=1019904;
        constexpr int64_t available=(8*1024*1024-1024)-existing_full_vla_persistent-65536;
        constexpr int64_t effective_shared_ceiling=8303616;
        assert(plan.peak_bytes<=available);
        assert(plan.peak_bytes+existing_full_vla_persistent<=effective_shared_ceiling);
        std::cout<<"I"<<width<<" temporary="<<plan.peak_bytes<<" vision_persistent="<<persistent
                 <<" full_vla_temporary_available="<<available<<" headroom="<<available-plan.peak_bytes
                 <<" full_vla_total="<<plan.peak_bytes+existing_full_vla_persistent
                 <<" shared_budget_headroom="<<effective_shared_ceiling-plan.peak_bytes-existing_full_vla_persistent<<"\n";
    }
}
""")
    print(stdout, end="")


def test_internvla_784_full_chain_spm_and_dino_zero_lifetime(tmp_path):
    """Actual declarations and first-fit planner guard the second-frame OOM.

    The archived chain floor is input evidence; this host check never loads
    weights or device kernels and supplies no numerical/hardware credit.
    """
    dino = (ROOT / "src/fused/rpu_dinov3_vision_model.cpp").read_text()
    declarations = _body("    std::vector<BufferDecl> declare_buffers(", dino)
    declarations = declarations.replace(" override", "")
    layer = _body("    void build_layer_subgraph(", dino)
    zero_init = _body("        if (layer_idx == 0) {", layer)
    # Execute the actual layer-0 branch, rather than a copy of its condition.
    assert 'addr(0, "zero_residual")' in zero_init
    source = _generic_spm_cpp() + r"""
struct DinoModel {
    bool has_final_norm_=true,has_output_mlp_=true;
    int64_t hidden_size()const{return 384;}
    int64_t num_q_heads()const{return 8;}
    int64_t head_dim()const{return 64;}
    int64_t intermediate_size()const{return 1536;}
    int64_t num_layers()const{return 12;}
    int num_cores()const{return 8;}
""" + declarations + r"""
};
struct Layout {
    std::vector<BufferDecl> declarations;
    std::vector<SpmAllocator::AllocRequest> requests;
    std::vector<std::string> names;
    SpmAllocator::AliasedPlan plan;
    int64_t persistent=0;
    explicit Layout(std::vector<BufferDecl> decls):declarations(std::move(decls)) {
        for(const auto& d:declarations) {
            if(d.storage==StorageClass::Temp && !d.alias_of) {
                requests.push_back({d.size,d.phase_start,d.phase_end,static_cast<int>(d.scope)});names.push_back(d.name);
            }
            if(d.storage==StorageClass::Persistent)persistent+=d.size;
            if(d.storage==StorageClass::PersistentPerLayer)persistent+=d.size*d.per_layer;
        }
        plan=SpmAllocator::plan_temporary_aliased(requests);
    }
    size_t index(const char* name)const {
        auto found=std::find(names.begin(),names.end(),name);assert(found!=names.end());
        return size_t(found-names.begin());
    }
};
std::vector<uint16_t> zero_storage;
int memset_calls=0;
uint32_t addr(int core,const char* name){assert(core==0&&std::string(name)=="zero_residual");return 0;}
int num_cores(){return 8;}
void rpu_launch_memset_spm_multicore(uint32_t offset,int64_t elements,int cores){
    assert(offset==0&&elements==257*384&&cores==8);std::fill(zero_storage.begin(),zero_storage.end(),0);++memset_calls;
}
void actual_layer_zero_init(int layer_idx,int64_t seq_len,int64_t h){
""" + zero_init + r"""
}
int main() {
    Model vision;vision.intermediate=3456;vision.image_batch_count_=1;vision.max_hw_=32;
    vision.merger_out_hidden_=3584;vision.rope_spm_enabled_=false;vision.window_mask_present_=false;
    vision.merger=false;vision.resident_mask=false;vision.pair=false;vision.schedule_=VisionSchedule::SISC;
    LayoutContext vctx;vctx.chunk_size=784;vctx.rope_table_residency=FmbRopeTableResidency::DDR;
    const auto fixed=vision.declare_buffers(vctx);
    const Layout current(fixed);
    assert(current.plan.peak_bytes==6949376&&current.persistent==442368);
    // Reconstruct only the historical SDPA-output placement. All sizes,
    // other lifetimes/aliases and allocator behavior remain production code.
    auto old=fixed;
    auto output=std::find_if(old.begin(),old.end(),[](const auto& d){return std::string(d.name)=="sdpa_out";});
    assert(output!=old.end()&&output->phase_start==2&&output->phase_end==4);
    auto original_output=*output;old.erase(output);original_output.phase_start=3;
    auto after_v=std::find_if(old.begin(),old.end(),[](const auto& d){return std::string(d.name)=="v";});
    assert(after_v!=old.end());old.insert(after_v+1,original_output);
    const Layout historical(old);
    assert(historical.plan.peak_bytes==7024640);

    DinoModel tower;LayoutContext dctx;dctx.chunk_size=272;
    const auto ddecls=tower.declare_buffers(dctx);
    const Layout temporary_zero(ddecls);
    const size_t z=temporary_zero.index("zero_residual");
    assert(temporary_zero.requests[z].bytes==208896);
    assert(temporary_zero.requests[z].scope==int(BufferScope::LayerWide));
    // Zeroes survive every layer's phase-1..7 work and the final MLP tail.
    // Full ranges must be disjoint from every ordinary Temp, including values
    // dead within one layer but rewritten early in the next layer.
    for(size_t i=0;i<temporary_zero.names.size();++i)if(i!=z) {
        const uint64_t zb=temporary_zero.plan.offsets[z],ze=zb+temporary_zero.requests[z].bytes;
        const uint64_t begin=temporary_zero.plan.offsets[i],end=begin+temporary_zero.requests[i].bytes;
        assert(ze<=begin||end<=zb);
    }
    auto persistent_zero=ddecls;
    for(auto& d:persistent_zero)if(std::string(d.name)=="zero_residual")d.storage=StorageClass::Persistent;
    const Layout retained_zero(persistent_zero);
    const int64_t freed_by_two_towers=2*(retained_zero.persistent-temporary_zero.persistent);
    assert(freed_by_two_towers==417792);
    constexpr int64_t old_available=6579200;
    const int64_t available=old_available+freed_by_two_towers;
    assert(available==6996992&&historical.plan.peak_bytes>old_available);
    assert(historical.plan.peak_bytes>available&&current.plan.peak_bytes<=available);
    constexpr int64_t usable=8*1024*1024-1024,reserve=64*1024,planning=usable*99/100;
    const int64_t full_chain_persistent=usable-reserve-old_available-freed_by_two_towers;
    assert(current.plan.peak_bytes+full_chain_persistent+reserve<=usable);
    assert(current.plan.peak_bytes+full_chain_persistent<=planning);
    assert(temporary_zero.plan.peak_bytes<=available);
    // Each captured dispatch replays this existing memset once at layer 0.
    zero_storage.resize(257*384);
    for(int dispatch=0;dispatch<3;++dispatch) {
        std::fill(zero_storage.begin(),zero_storage.end(),uint16_t(9));
        for(int layer_idx=0;layer_idx<12;++layer_idx) {
            actual_layer_zero_init(layer_idx,257,384);
            assert(std::all_of(zero_storage.begin(),zero_storage.end(),[](uint16_t x){return x==0;}));
        }
        assert(memset_calls==dispatch+1);
    }
    std::cout<<"Vision784 temporary="<<current.plan.peak_bytes<<" available="<<available
             <<" headroom="<<available-current.plan.peak_bytes<<" dino_zero_freed="<<freed_by_two_towers<<"\n";
}
"""
    stdout = _run_cpp(tmp_path, "internvla_784_spm", source)
    print(stdout, end="")


def test_pair_logical_identity_masks_and_consumer_share_complete_contract():
    resolve = _body("    std::vector<int64_t> resolve_stage_domain(")
    forward = _body("    at::Tensor forward_impl(")
    producer = _body("    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    preload = _body("    void emit_preload_weights()")
    consumer = _body("    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override")
    assert "image_batch_count_ * valid_patches_per_image_ : num_patches" in resolve
    assert "physical_len == 672 && logical_len == 648" in producer
    assert "sorting_schedule_arguments()" in producer and "sorting_schedule_arguments()" in preload
    assert "manifest.logical_length == 648" in preload and "manifest.execution_padding_rows == 24" in preload
    assert "prepare_padded_pair_masks(*window_mask, *full_mask)" in forward
    assert 'addr_offset(i == 0 ? "sdpa_mask" : "sdpa_full_mask")' in preload
    assert "int mask_type = padded_pair_enabled() ? 4" in consumer
    assert 'padded_pair_enabled() && is_full\n                    ? "sdpa_full_mask" : "sdpa_mask"' in consumer
    assert "mask_type == 4 ? SPM_ALLOC.addr(0, mask_off) : 0" in consumer
    assert "valid_patches_per_image_" in _body("    int64_t subclass_layout_hash() const override")
    assert "valid_patches_per_image_" in _body("    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override")

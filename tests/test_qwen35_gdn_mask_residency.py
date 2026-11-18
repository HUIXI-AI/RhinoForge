"""CPU native regression: actual GDN declarations, SDPA sizing and SPM allocator.

Only tensor/launch facades are inert. The compiled probe uses the checked-in
allocator, residency policy, model declaration arithmetic, and DMA schedule.
No backend import, model loading, or device access is involved.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def definition(source, marker):
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda match: " " * len(match[0]), source)
    start = source.index(marker)
    start = source.rfind("\n", 0, start) + 1
    brace = masked.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end]


def source_probe():
    base = (ROOT / "src/core/fused_model_base.h").read_text()
    base_cpp = (ROOT / "src/core/fused_model_base.cpp").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    policy = (ROOT / "src/core/rpu_spm_residency.h").read_text()
    policy = re.sub(r'^#include "[^"\n]+"\s*$', '', policy, flags=re.M)
    policy = policy.replace('#pragma once', '')
    kinds = definition(base, "enum class StorageClass") + ";\n"
    kinds += definition(base, "enum class BufferScope") + ";\n"
    kinds += definition(base, "struct BufferDecl") + ";\n"
    sdpa = helpers[helpers.index("inline int64_t sdpa_tile_m_v16_initial"):
                   helpers.index("// Shared utility: float32")]
    sdpa += "\n" + definition(helpers, "inline bool sdpa_try_compute_tiling")
    sdpa += "\n" + definition(helpers, "inline SdpaTiling sdpa_compute_tiling")
    sdpa += "\n" + definition(helpers, "inline int64_t sdpa_compute_tmp_v16_size")
    code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>
#include "core/rpu_spm_allocator.h"
#define TORCH_CHECK(x, ...) do {if (!(x)) throw std::runtime_error("rejected");} while(0)
#define DWIDTH 2
#define NUM_CORES 8
template<class A, class B> int64_t Align(A a, B b) {return ((a+b-1)/b)*b;}
template<class A, class B> int64_t CeilDiv(A a, B b) {return (a+b-1)/b;}
namespace c10 {struct Half {uint16_t raw;};}
struct Tensor {
    int64_t rows=512, elems=131072; bool present=false;
    bool defined() const {return present;}
    int64_t size(int) const {return rows;}
    int64_t numel() const {return elems;}
    template<class T> T* data_ptr() const {return nullptr;}
};
int dma_calls=0;
template<class... A> void rpu_launch_ddr_broadcast_spm_dma(A...) {++dma_calls;}
'''
    code += sdpa
    code += "\n" + definition(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased")
    code += "\nnamespace v3 { class FusedModelBase;\n" + kinds
    code += "\n" + definition(base, "struct LayoutContext") + ";\n"
    # LayoutContext uses this enum; place its production definition before it.
    attention = definition(base, "enum class AttentionExecutionPolicy") + ";\n"
    rope = definition(base, "enum class FmbRopeTableResidency") + ";\n"
    if "enum class FmbForwardOperandResidency" in base:
        rope += definition(base, "enum class FmbForwardOperandResidency") + ";\n"
    code = code.replace("namespace v3 { class FusedModelBase;", "namespace v3 { class FusedModelBase;\n" + attention + rope)
    # Extract the exact stage/ChunkInfo structs: field paths and aggregate
    # ordering are part of this native route's contract, not an inert facade.
    for marker in ("enum class ChunkMode", "struct ChunkInfo", "struct ChunkPlan",
                   "struct ResolvedFmbStageChunks", "enum class FmbSpanBoundaryPolicy",
                   "struct FmbStageBoundaryPolicies", "struct FmbExecutionSpan",
                   "struct FmbThreeStageChunkPlan"):
        code += "\n" + definition(base, marker) + ";\n"
    code += r'''
enum class FmbRouteFamily {MUTABLE_DMA,GRAPH_SCHEDULE};
struct Route {int64_t selector=1;};
struct Context {
    bool complete=true;Route route; FmbThreeStageChunkPlan stage_plan;
    std::vector<int64_t> expected;
    bool has_complete_physical_manifest() const {return complete;}
    const Route& find_physical_route(FmbRouteFamily,int64_t) const {return route;}
    void consume_physical_route(FmbRouteFamily,int64_t,int64_t,int64_t flags,
            const std::vector<int64_t>& args,int64_t=0) const {assert(flags==0);assert(args==expected);}
    void consume_physical_route(FmbRouteFamily,int64_t,int64_t,int64_t,
            std::initializer_list<int64_t>,int64_t=0) const {throw std::runtime_error("preload called");}
};
class FusedModelBase {};
namespace detail {int64_t estimate_temporary_total(const std::vector<BufferDecl>&);}
'''
    code += definition(base_cpp, "int64_t detail::estimate_temporary_total")
    code += "\n}\n" + policy + "\nnamespace v3 {\n"
    model_shim = r'''
class CLASS : public FusedModelBase {
public:
    int cores=8;int64_t hidden=1024, inter=3584,nq=8,nkv=8,layers=24;
    int64_t gdn_nvh_=16,gdn_dk_=128,gdn_dv_=128,gdn_conv_dim_=6144,gdn_kc_=4;
    bool has_gdn_=true,has_qk_norm_=true,action_mode_=false,z2_bound_=false;
    int64_t action_dim_pad_=32,action_len_=0;
    // These text-mask owners have no installed G05 projection/resident state.
    // Keep the current declaration interface while leaving its action-only
    // capabilities unavailable; the real buffer arithmetic below is unchanged.
    struct UnpreparedNvfp4 {
        void declare(std::vector<BufferDecl>&,int) const {}
    } action_nvfp4_;
    bool use_action_gateup_resident(int64_t) const {return false;}
    bool use_action_nvfp4_o_resident(int64_t) const {return false;}
    bool use_action_nvfp4_down_resident(int64_t,int64_t) const {return false;}
    bool use_action_o_resident(int64_t) const {return false;}
    bool use_action_mod_body() const {return false;}
    bool use_action_down_partial(int64_t,int64_t) const {return false;}
    Tensor action_input_w_,action_input_b_,action_output_b_,final_norm_w_,expert_ids_,token_ids_,gdn_tril_,gdn_strict_;
    struct Weight {Tensor input_norm_w,post_norm_w,q_norm_w,k_norm_w;};
    std::vector<Weight> layer_weights_=std::vector<Weight>(48);
    static constexpr int kRoutedUint4Mode=6;
    struct MoeWeight {int routed_weight_mode=5;};
    std::vector<MoeWeight> moe_layer_weights_=std::vector<MoeWeight>(48);
    std::vector<int> layer_is_full_={1,0,0,1};
    struct Moe {int64_t num_experts=256,top_k=8,tensor_parallel=8,routed_intermediate=512,shared_intermediate=512;
        void validate(int64_t) const {}} moe_;
    Context context;
    const Context& ctx() const {return context;}
    int64_t override_chunk=0;
    int64_t get_chunk_size_override() const {return override_chunk;}
    static constexpr int64_t kTextGdnInputWorkspaceSite=7200073926787961303LL;
    bool use_text_gdn_input_workspace() const;
    static int gdn_input_workspace_slot(const char*);
    std::vector<int64_t> gdn_input_workspace_arguments(int64_t) const;
    int64_t gdn_input_workspace_offset(const char*,int64_t) const;
    int num_cores() const {return cores;}
    int attn_tp() const {return cores==8?8:4;}
    int mlp_tp() const {return cores;}
    int gdn_tp() const {return cores==8?8:4;}
    int router_tp() const {return attn_tp();}
    int routing_tp() const {return cores;}
    int routed_expert_tp() const {return cores;}
    int shared_expert_tp() const {return cores;}
    int64_t physical_routed_intermediate() const {return cores==6?576:512;}
    int64_t physical_shared_intermediate() const {return cores==6?576:512;}
    int64_t hidden_size() const {return hidden;}
    int64_t intermediate_size() const {return inter;}
    int64_t num_q_heads() const {return nq;}
    int64_t num_kv_heads() const {return nkv;}
    int64_t head_dim() const {return 256;}
    int64_t num_layers() const {return layers;}
    SdpaConfig make_sdpa_config(int mask=1) const {return {SdpaKernelType::FLASH_ATTN_SPM_VCTXLEN,256,nq,nkv,attn_tp(),mask};}
    std::vector<BufferDecl> declare_buffers(const LayoutContext&);
    std::vector<int64_t> gdn_mask_schedule_arguments(const FmbThreeStageChunkPlan&) const;
    void emit_masks(int layer_idx,const ChunkInfo& chunk,bool decode) {
        auto D=[](const char*){return uint32_t(0);};const int nc=gdn_tp();
        SCHEDULE
    }
};
'''
    for stem, cls, prefix in (("rpu_qwen3_5_model","Qwen3_5Model","QWEN35_TEXT"),
                              ("rpu_qwen3_5_moe_model","Qwen3_5MoeModel","QWEN35_MOE")):
        source=(ROOT / f"src/fused/{stem}.cpp").read_text()
        # Evaluate the shared no-growth policy against the exact legacy
        # declarations. Actual-budget selection is tested by the focused
        # dense/MoE budget probes using the real FMB capacity predicate.
        declaration=definition(source, f"std::vector<BufferDecl> {cls}::declare_legacy_buffers(")
        declaration = declaration.replace("::declare_legacy_buffers(", "::declare_buffers(")
        declaration = declaration.replace("const LayoutContext& ctx) const", "const LayoutContext& ctx)")
        declaration = declaration.replace(
            "const LayoutContext& ctx, int64_t sdpa_workspace_bytes) const",
            "const LayoutContext& ctx, int64_t sdpa_workspace_bytes)")
        if prefix.endswith('TEXT'):
            code += "int64_t qwen35_gdn_scalar_slots(int64_t h,int64_t n){return std::max<int64_t>(8,Align(h/n,8));}\n"
            code += "\n".join(re.findall(
                r"constexpr int64_t QWEN35_ACTION_RESIDENT_\w+\s*=\s*[^;]+;",
                source)) + "\n"
        else:
            code += "\n".join(re.findall(r"constexpr (?:int kMoePhase\w+|int64_t kMoeChunkSize) = \d+;",source))+"\n"
            code += definition(source,"int64_t aligned_bytes(")+"\n"
        sites=set(re.findall(rf"{prefix}_\w+_SITE",declaration))
        sites.add(prefix+"_GDN_MASK_RESIDENCY_SITE")
        for i,site in enumerate(sorted(sites),1): code += f"constexpr int64_t {site}={i};\n"
        route_type="Qwen35TextMutableDmaRoute" if prefix.endswith('TEXT') else "Qwen35MoeMutableDmaRoute"
        route_fn="qwen35_text_dma_route" if prefix.endswith('TEXT') else "qwen35_moe_dma_route"
        code += f"enum class {route_type} {{DDR_BROADCAST_TO_SPM=1}};\nint64_t {route_fn}({route_type} v){{return static_cast<int64_t>(v);}}\n"
        start=source.index("    bool load_chunk_masks = !decode;")
        end=source.index("\n\n    // ── in_proj",start)
        schedule=source[start:end]
        shim = model_shim.replace('CLASS',cls).replace('SCHEDULE',schedule)
        if prefix.endswith("MOE"):
            shim = shim.replace(
                "declare_buffers(const LayoutContext&);",
                "declare_buffers(const LayoutContext&, int64_t sdpa_workspace_bytes = -1);")
        code += shim
        code += declaration+"\n"
        if prefix.endswith("TEXT"):
            for marker in ("bool Qwen3_5Model::use_text_gdn_input_workspace(",
                           "int Qwen3_5Model::gdn_input_workspace_slot(",
                           "std::vector<int64_t> Qwen3_5Model::gdn_input_workspace_arguments(",
                           "int64_t Qwen3_5Model::gdn_input_workspace_offset("):
                code += definition(source, marker)+"\n"
            code += definition(source,f"std::vector<int64_t> {cls}::gdn_mask_schedule_arguments(")+"\n"
    code += r'''
const BufferDecl& slot(const std::vector<BufferDecl>& rows,const char* name) {
    return *std::find_if(rows.begin(),rows.end(),[&](const auto& d){return std::string(d.name)==name;});
}
void same_declarations(const std::vector<BufferDecl>& a,const std::vector<BufferDecl>& b) {
    assert(a.size()==b.size());
    for(size_t i=0;i<a.size();++i) {
        assert(std::string(a[i].name)==b[i].name && a[i].size==b[i].size && a[i].storage==b[i].storage);
        assert(a[i].phase_start==b[i].phase_start && a[i].phase_end==b[i].phase_end && a[i].scope==b[i].scope);
        assert(a[i].alias_of==b[i].alias_of && a[i].per_layer==b[i].per_layer);
    }
}
template<class M> void schedule_test(M& model) {
    auto& c=model.context;c.stage_plan.compute.plan={64,2};
    c.stage_plan.compute.chunks={{0,0,64,64},{1,64,64,128}};
    c.expected=model.gdn_mask_schedule_arguments(c.stage_plan);
    assert(c.expected[3]==1 && c.expected[4]==2 && c.expected[5]==64 && c.expected[6]==2);
    for(int selector:{1,2}) {
        c.route.selector=selector;
        for(int execution=0;execution<2;++execution) {
            dma_calls=0;
            for(int l:{1,2}) for(const auto& chunk:c.stage_plan.compute.chunks) model.emit_masks(l,chunk,false);
            assert(dma_calls==(selector==2?2:8));
        }
    }
    c.route.selector=3;bool rejected=false;
    try {model.emit_masks(1,{0,0,64,64},false);} catch(const std::runtime_error&) {rejected=true;}
    assert(rejected);
    c.complete=false;dma_calls=0;
    for(int l:{1,2}) for(const auto& chunk:c.stage_plan.compute.chunks) model.emit_masks(l,chunk,false);
    assert(dma_calls==8);
    dma_calls=0;model.emit_masks(1,{0,0,1,1},true);assert(dma_calls==0);
}
}
int main() {
    using namespace v3;
    int retained_cases=0,fallback_cases=0;
    for(int cores:{4,6,8}) for(int64_t cs:{64,128,256,448,512,640}) {
        for(int profile=0;profile<5;++profile) {
            Qwen3_5Model dense; Qwen3_5MoeModel moe;
            auto evaluate=[&](auto& model,const std::string& name) {
                model.cores=cores;model.nkv=model.attn_tp();model.moe_.tensor_parallel=cores;
                LayoutContext lc;lc.chunk_size=cs;lc.num_layers=model.layers;lc.max_kv_seq_len=cs;
                const auto old=model.declare_buffers(lc);
                auto probe=old;const auto choice=detail::plan_forward_spm_residency(probe,{"gdn_c_tril","gdn_c_strict"});
                bool retained=choice.retained;auto next=probe;
                lc.physical_manifest_fingerprint=123;
                same_declarations(model.declare_buffers(lc),old);
                assert(detail::estimate_temporary_total(next)<=detail::estimate_temporary_total(old));
                if(retained) {
                    ++retained_cases;
                    for(const auto& d:next) if(d.storage==StorageClass::Temp && !d.alias_of) {
                        for(const char* n:{"gdn_c_tril","gdn_c_strict"}) {
                            assert(slot(next,n).phase_start<=d.phase_start && slot(next,n).phase_end>=d.phase_end);
                        }
                    }
                } else {++fallback_cases;same_declarations(old,next);}
                std::cout<<name<<" cores="<<cores<<" chunk="<<cs<<" retained="<<retained
                         <<" old="<<choice.original_peak_bytes<<" candidate="<<choice.retained_peak_bytes<<"\n";
            };
            if(profile<4) {
                const int64_t h[]={1024,2048,2560,4096},i[]={3584,6144,9216,12288};
                dense.hidden=h[profile];dense.inter=cores==6 && profile==0?3648:i[profile];
                dense.nq=profile<2?8:16;dense.layers=profile<2?24:32;
                dense.gdn_nvh_=profile<2?16:32;dense.gdn_conv_dim_=4096+dense.gdn_nvh_*128;
                evaluate(dense,"dense"+std::to_string(profile));
            } else {
                moe.hidden=2048;moe.nq=16;moe.layers=40;moe.gdn_nvh_=32;moe.gdn_conv_dim_=8192;
                evaluate(moe,"moe35b");
            }
        }
    }
    assert(retained_cases>0 && fallback_cases>0);
    Qwen3_5Model dense;schedule_test(dense);

}
'''
    return code


def test_real_gdn_declarations_and_constant_schedule(tmp_path):
    compiler=shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    cpp,exe=tmp_path/"gdn.cpp",tmp_path/"gdn"
    cpp.write_text(source_probe())
    result=subprocess.run([compiler,"-std=c++17","-O0","-I",str(ROOT/"src"),str(cpp),"-o",str(exe)],
                          text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],text=True,capture_output=True)
    print(result.stdout)
    assert result.returncode==0,result.stderr

"""Run the real MLP manifest producer and emitters against checked launch facades.

No tensor arithmetic/device is emulated: this verifies slice shapes, addresses,
TP ownership and COMPLETE route consumption, including tail and decode dispatch.
"""
from pathlib import Path
import re
import runpy
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def source_probe():
    helpers = runpy.run_path(str(ROOT / 'tests/test_qwen35_moe_core_native.py'))
    definition = helpers['_definition']
    native = (ROOT / 'src/fused/rpu_qwen3_5_moe_model.cpp').read_text()
    header = (ROOT / 'src/fused/rpu_qwen3_5_moe_model.h').read_text()
    base = (ROOT / 'src/core/fused_model_base.h').read_text()
    methods = '\n'.join(definition(native, marker) for marker in (
        'void Qwen3_5MoeModel::append_mlp_manifest_routes(',
        'void Qwen3_5MoeModel::emit_mlp_and_output(',
        'void Qwen3_5MoeModel::emit_router_and_shuffle(',
        'void Qwen3_5MoeModel::emit_router_topk(',
        'void Qwen3_5MoeModel::emit_shared_expert(',
        'void Qwen3_5MoeModel::emit_routed_experts(',
        'void Qwen3_5MoeModel::emit_moe_merge_and_residual(',
        'FmbLinearRouteSelector Qwen3_5MoeModel::linear_route_selector(',
        'void Qwen3_5MoeModel::launch_linear(',
        'void Qwen3_5MoeModel::launch_grouped_linear(',
        'void Qwen3_5MoeModel::consume_manifest_route(',
        'bool Qwen3_5MoeModel::subclass_chunk_size_valid(',
    ))
    types = '\n'.join(definition(base, marker) + ';' for marker in (
        'struct ChunkInfo', 'enum class FmbRouteFamily',
        'enum class FmbLinearRouteSelector', 'struct FmbRouteManifestEntry',
    ))
    constants = '\n'.join(re.findall(
        r'constexpr int64_t (?:QWEN35_MOE_\w+_SITE|kMoeChunkSize|kResident\w*|kRoutedUint4\w*)\s*=\s*\d+(?:LL)?;', native))
    constants += '\n' + definition(native, 'std::vector<int64_t> shared_alpha_local_arguments(')
    enums = '\n'.join(definition(native, marker) + ';' for marker in (
        'enum class Qwen35MoeMutableDmaRoute', 'enum class Qwen35MoeActivationRoute'))
    code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
__GROUPED_LAYOUT__
#define DWIDTH 2
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
#define TORCH_INTERNAL_ASSERT(ok) TORCH_CHECK(ok)
namespace c10 {struct Half {uint16_t raw; explicit Half(float = 0) : raw(0) {}};}
namespace at {
constexpr int kHalf=1;
struct Tensor {
    int id=0, dtype=kHalf; int64_t rows=512, cols=256;
    bool defined() const { return true; }
    int dim() const { return 2; }
    int64_t size(int axis) const { return axis == 0 ? rows : cols; }
    int scalar_type() const { return dtype; }
};
}
enum class RpuRmsNormSpmRoute { BASE };
enum class RpuAllGatherSchedule { BASE };
enum class ValuOpType { SIGMOID, MUL };
enum class GeluMode { NONE };
RpuAllGatherSchedule rpu_resolve_all_gather_schedule(int64_t, int64_t) {
    return RpuAllGatherSchedule::BASE;
}
namespace v3 {
__TYPES__
struct FmbPhysicalExecutionManifest {std::vector<FmbRouteManifestEntry> routes;};
bool sdpa_is_valid_chunk_size(int, int64_t, int64_t, int64_t) {return true;}
}
__CONSTANTS__
__ENUMS__
__DMA_ROUTE__
template<class Fn>
__SLICES__
namespace v3 {
using RouteKey=std::tuple<int64_t,int64_t,int64_t>;
RouteKey key(const FmbRouteManifestEntry& r) {
    return {static_cast<int64_t>(r.family),r.site_id,r.invocation};
}
std::array<int64_t,5> consumed_fp8_dispatch{};
struct Context {
    bool output_to_spm=false;
    std::map<RouteKey,FmbRouteManifestEntry> expected;
    std::set<RouteKey> consumed;
    bool has_complete_physical_manifest() const {return true;}
    void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,
            int64_t flags,const std::vector<int64_t>& args,int64_t invocation=0) {
        RouteKey k{static_cast<int64_t>(family),site,invocation};
        const auto& r=expected.at(k);
        TORCH_CHECK(r.selector==selector && r.flags==flags && r.arguments==args);
        if(site==QWEN35_MOE_GROUPED_FP8_SITE) {
            TORCH_CHECK(args.size()==17);
            std::copy_n(args.begin()+6,5,consumed_fp8_dispatch.begin());
        }
        consumed.insert(k);
    }
};
struct Record {int64_t rows; uint32_t residual,output;};
int owner=8, routed_owner=8, outer_rows=0, norm_calls=0, output_dmas=0;
int uint4_calls=0;
int64_t active_row=0, active_t=0;
bool expected_acc32=false;
std::vector<Record> reductions;
std::vector<int64_t> bridge_rows;
constexpr uint32_t input_base=0x100000,residual_base=0x400000,scratch_base=0x800000;
void rpu_launch_rmsnorm_spm_kernel(uint32_t input,uint32_t output,uint32_t,
        int64_t rows,int64_t h,double,RpuRmsNormSpmRoute,int nc) {
    assert(rows==outer_rows && h==2048 && nc==owner);
    if (norm_calls++ == 0) {assert(input==residual_base && output==input_base);}
    else {assert(input==input_base && output==input_base);assert(active_row==outer_rows);}
}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t input,const at::Tensor& w,
        uint32_t,int64_t m,int64_t,int64_t,int,int nc,uint32_t,
        const at::Tensor&,uint32_t,int,bool acc32,bool gemv) {
    assert(m>0 && m<=128 && m==std::min<int64_t>(128,outer_rows-active_row));
    assert(acc32==expected_acc32);
    assert(gemv==(m==1 && !acc32 && w.scalar_type()==at::kHalf));
    if(w.id!=4) assert(input==input_base+active_row*2048*2);
    assert(nc==(w.id==1 && owner==6?4:owner));
    active_t=m;
}
void rpu_launch_all_gather_spm_kernel(uint32_t,uint32_t,int64_t rows,int64_t,
        int64_t,int nc,RpuAllGatherSchedule) {
    assert(rows==active_t && nc==(owner==6?4:owner));
}
void rpu_launch_spm_copy_ddr_dma(uint32_t,const at::Tensor&,int64_t offset,int64_t elems) {
    assert(owner==6 && offset==0 && elems==active_t*256);bridge_rows.push_back(active_t);
}
void rpu_launch_ddr_broadcast_spm_dma(const at::Tensor&,int64_t offset,
        int64_t elems,uint32_t,int nc) {
    assert(owner==6 && nc==6 && offset==0 && elems==active_t*256);
}
void rpu_launch_gather_fp16_spm_kernel(uint32_t input,uint32_t,uint32_t,
        int64_t rows,int64_t h,int axis,int64_t routed,int nc) {
    assert(input==input_base+active_row*2048*2);
    assert(rows==active_t && routed==rows*8 && h==2048 && axis==0 && nc==routed_owner);
}
void rpu_launch_scatternd_add_u16_fp16_spm_kernel(uint32_t,uint32_t,uint32_t,
        int64_t rows,int64_t routed,int64_t h,int nc) {
    assert(rows==active_t && routed==rows*8 && h==2048 && nc==routed_owner);
}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t,uint32_t residual,
        uint32_t output,int64_t rows,int64_t h,int producers,int consumers) {
    assert(rows==active_t && h==2048 && producers==owner && consumers==owner);
    assert(residual==residual_base+active_row*h*2);
    assert(output==input_base+active_row*h*2);
    reductions.push_back({rows,residual,output});active_row+=rows;
}
void rpu_launch_slice_spm_kernel(uint32_t input,uint32_t output,std::initializer_list<int64_t> shape,
        std::initializer_list<int64_t> out,std::initializer_list<int64_t> begin,int nc,bool local) {
    assert(nc==owner && local && input==scratch_base && output==scratch_base);
    assert(std::vector<int64_t>(shape)==std::vector<int64_t>({active_t,16}));
    assert(std::vector<int64_t>(out)==std::vector<int64_t>({active_t,1}));
    assert(std::vector<int64_t>(begin)==std::vector<int64_t>({0,0}));
}
std::array<int64_t,3> rpu_select_grouped_parallel_linear_uint4_tile(
        int64_t m_total,int64_t experts,int64_t n,int64_t k,int64_t group_size,bool acc32) {
    assert(m_total>0 && m_total<=1024 && experts==256);
    assert((n==512/routed_owner && k==2048) || (n==2048 && k==512/routed_owner));
    const auto tile=rpu_grouped_tiling::select_uint4(m_total,experts,n,k,group_size,acc32);
    return {tile.n,tile.m,tile.k};
}
__GROUPED_ROUTE_ARGUMENTS__
void rpu_launch_grouped_parallel_linear_uint4_spm_kernel(
        uint32_t,const at::Tensor&,uint32_t,const at::Tensor&,
        const at::Tensor&,uint32_t,int64_t n,int64_t k,int64_t experts,
        bool col,int nc,int64_t group_size,int64_t m_total,bool acc32) {
    assert(nc==routed_owner && experts==256 && m_total==active_t*8);
    assert(n==(col?512/routed_owner:2048) && k==(col?2048:512/routed_owner));
    assert(group_size==(col?128:64));
    assert(acc32==expected_acc32);
    ++uint4_calls;
}
void rpu_launch_grouped_parallel_linear_spm_kernel(
        uint32_t,const at::Tensor&,uint32_t,const at::Tensor&,uint32_t,uint32_t,
        int64_t n,int64_t k,int64_t experts,bool col,int nc,int64_t m_total,int64_t mode,
        bool acc32) {
    assert(mode==5 && nc==owner && experts==256);
    assert(m_total==active_t*8);
    assert(acc32==expected_acc32);
    assert(n==(col?(owner==6?96:512/owner):2048));
    assert(k==(col?2048:(owner==6?96:512/owner)));
    const auto dispatch=rpu_grouped_tiling::select_dispatch(
        m_total,n,k,static_cast<int>(mode),acc32,2,0,0,experts);
    if((owner==6 || owner==8) && !acc32) {
        assert(m_total==8 || m_total==512 || m_total==1024);
        assert(dispatch.tile.n_tile==(col && owner==6?48:64));
        assert(dispatch.tile.m_tile==(col?(owner==6?256:224):448));
        assert(dispatch.tile.k_tile==(col?256:128));
        assert(dispatch.kind==rpu_grouped_tiling::DispatchKind::Static);
    }
    assert((consumed_fp8_dispatch==std::array<int64_t,5>{
        m_total,dispatch.tile.m_tile,dispatch.tile.n_tile,
        dispatch.tile.k_tile,static_cast<int64_t>(dispatch.kind)}));
}
__NOOPS__
struct Qwen3_5MoeModel {
    struct Weight {
        at::Tensor router_w{1},shared_gate_w{2},shared_up_w{3},shared_down_w{4},shared_scalar_gate_w{5};
        at::Tensor routed_gate_w,routed_up_w,routed_down_w,routed_gate_scale,routed_up_scale,routed_down_scale;
        at::Tensor routed_gate_negative_zero,routed_down_negative_zero;
        int64_t routed_group_size=0,routed_down_group_size=0;
        int64_t routed_weight_mode=5;
    };
    struct Moe {int64_t num_experts=256,top_k=8,routed_intermediate=512;} moe_;
    std::vector<Weight> moe_layer_weights_{Weight{}};
    at::Tensor expert_ids_,token_ids_,router_bridge_;
    bool linear_acc32_=false, resident_prefill_=false;
    bool resident_planning_active_=false;
    struct UnusedResidentPlan {
        struct Request {int64_t outer_capacity=0,execution_rows=0,position=0;} request;
    };
    std::optional<UnusedResidentPlan> resident_execution_plan_;
    bool plan_resident_candidate(int64_t,int64_t,int64_t) const {
        throw std::runtime_error("unexpected resident planning in legacy fixture");
    }
    bool resident_route_capacity_valid(int64_t) const {
        throw std::runtime_error("unexpected resident capacity query in legacy fixture");
    }
    void append_resident_mlp_manifest_routes(FmbPhysicalExecutionManifest&,const ChunkInfo&) const {throw std::runtime_error("resident outside legacy fixture");}
    bool has_gdn_=true;
    double eps_=1e-6;
    Context context;
    Context& ctx() {return context;}
    int num_cores() const {return owner;}
    int router_tp() const {return owner==6?4:owner;}
    int routing_tp() const {return owner;}
    __ROUTED_TP__
    int shared_expert_tp() const {return owner;}
    int64_t num_layers() const {return 1;}
    int64_t hidden_size() const {return 2048;}
    __ROUTED_WIDTH__
    int64_t physical_shared_intermediate() const {return owner==6?576:512;}
    int64_t text_ring_route(int64_t rows,int64_t) const {return rows==1?1:2;}
    int make_sdpa_config() const {return 0;}
    bool chunk_within_envelope(int64_t rows) const {return rows<=512;}
    uint32_t addr(int core,const char* name) const {
        const std::string n(name);
        return n=="residual1"?input_base:n=="input_norm"?residual_base:scratch_base+core*0x1000000;
    }
    uint32_t layer_addr(int,int,const char*) const {return scratch_base;}
    at::Tensor optional_scale(const at::Tensor& t) const {return t;}
    void emit_layer_output_dma(int,const ChunkInfo& chunk) {assert(active_row==chunk.len);++output_dmas;}
__LINEAR_INVOCATION__
__DECLARATIONS__
};
__METHODS__
void run_case(int cores, int rows, bool acc32=false, bool mixed_weights=false,
        int tail=0,bool uint4=false) {
    owner=cores;routed_owner=uint4 && cores==6?4:cores;outer_rows=rows;active_row=0;active_t=0;
    norm_calls=0;output_dmas=0;reductions.clear();bridge_rows.clear();expected_acc32=acc32;
    uint4_calls=0;
    Qwen3_5MoeModel model;model.linear_acc32_=acc32;
    if(uint4) {
        auto& w=model.moe_layer_weights_[0];w.routed_weight_mode=6;
        w.routed_group_size=128;w.routed_down_group_size=64;
    }
    model.expert_ids_.rows=model.token_ids_.rows=128;
    assert(model.subclass_chunk_size_valid(192,4096,0));
    assert(model.subclass_chunk_size_valid(256,4096,0));
    assert(model.subclass_chunk_size_valid(512,4096,0));
    assert(!model.subclass_chunk_size_valid(576,4096,0));
    assert(!model.subclass_chunk_size_valid(160,4096,0));
    model.token_ids_.rows=64;
    assert(!model.subclass_chunk_size_valid(192,4096,0));
    model.token_ids_.rows=128;
    if (mixed_weights) model.moe_layer_weights_[0].shared_up_w.dtype=2;
    // Multiple outer chunks intentionally combine token starts with chunk IDs.
    if(!tail)tail=rows;
    const std::vector<ChunkInfo> chunks={{0,0,rows,rows},{1,rows,tail,rows+tail}};
    FmbPhysicalExecutionManifest manifest;
    for(const auto& chunk:chunks) model.append_mlp_manifest_routes(manifest,chunk);
    for(const auto& route:manifest.routes) {
        assert(model.context.expected.emplace(key(route),route).second);
        const int64_t grouped_site = uint4 ? QWEN35_MOE_GROUPED_UINT4_SITE
                                           : QWEN35_MOE_GROUPED_FP8_SITE;
        const int64_t other_grouped_site = uint4 ? QWEN35_MOE_GROUPED_FP8_SITE
                                                 : QWEN35_MOE_GROUPED_UINT4_SITE;
        assert(route.site_id!=other_grouped_site);
        if(route.site_id==grouped_site) {
            assert(route.arguments.size()==(uint4?20:17));
            const auto start=uint4?14:11;
            assert(route.arguments[start]==2); // sole register ABI
            assert(route.arguments[start+4]==(uint4?2:1));
            assert(route.arguments[start+5]%256==0);
            if(uint4) {
                assert(route.arguments[start+1]==route.arguments[8]); // M
                assert(route.arguments[start+2]==route.arguments[7]); // N
                assert(route.arguments[start+3]==route.arguments[9]); // K
                assert(route.arguments[start+5]==route.arguments[13]);
            } else {
                assert(route.arguments[start+1]==route.arguments[7]);
                assert(route.arguments[start+2]==route.arguments[8]);
                assert(route.arguments[start+3]==route.arguments[9]);
                const auto width=route.arguments[0]*(route.arguments[3]?route.arguments[4]:1);
                assert(route.arguments[start+5]==(width*2+255)/256*256);
            }

            assert(route.arguments[5]==(acc32?1:0));
            if(!uint4 && cores==6 && !acc32) {
                const bool col=route.arguments[3]!=0;
                assert(route.arguments[0]==(col?96:2048));
                assert(route.arguments[1]==(col?2048:96));
                assert(route.arguments[2]==256 && route.arguments[4]==6);
                assert(route.arguments[7]==(col?256:448));
                assert(route.arguments[8]==(col?48:64));
                assert(route.arguments[9]==(col?256:128) && route.arguments[10]==0);
            }
            if(uint4) {
                const bool col=route.arguments[3]!=0;
                assert(route.arguments[11]==1); // Q4/N64 metadata ABI
                const auto metadata=rpu_grouped_tiling::uint4_metadata_layout(
                    col?512/routed_owner:2048,col?2048:512/routed_owner,
                    col?128:64,routed_owner);
                assert(route.arguments[4]==routed_owner);
                assert(route.arguments[12]==metadata.group_block_bytes);
                assert(route.arguments[13]==metadata.expert_bytes);
                if(routed_owner==8 && !acc32) {
                    const int64_t packed_m=route.arguments[10];
                    assert(packed_m==8 || packed_m==512 || packed_m==1024);
                    assert(route.arguments[7]==(col?32:128));
                    assert(route.arguments[8]==(col?64:(packed_m==1024?288:128)));
                    assert(route.arguments[9]==(col?768:(packed_m==1024?128:256)));
                }
            }
        }
        if(route.site_id==QWEN35_MOE_LINEAR_SITE) {
            for(const auto& chunk:chunks) {
                assert(route.invocation!=model.linear_invocation(chunk.idx,FmbLinearRouteSelector::AUTO_TILE));
                assert(route.invocation!=model.linear_invocation(chunk.idx,FmbLinearRouteSelector::GEMV));
            }
        }
    }
    for(const auto& chunk:chunks) {
        outer_rows=chunk.len;active_row=0;norm_calls=0;
        model.emit_mlp_and_output(0,chunk);
        assert(active_row==chunk.len && norm_calls==2);
    }
    assert(model.context.consumed.size()==manifest.routes.size());
    assert(output_dmas==2);
    assert(reductions.size()==(rows+127)/128+(tail+127)/128);
    assert(bridge_rows.size()==(cores==6?reductions.size():0));
    assert(uint4_calls==(uint4?3*reductions.size():0));
    int64_t sum=0;for(auto r:reductions)sum+=r.rows;assert(sum==rows+tail);
    if(uint4) {
        // Tile, group-size and metadata drift must fail before the first UINT4 launch.
        auto it=std::find_if(model.context.expected.begin(),model.context.expected.end(),
            [](const auto& pair){return pair.second.site_id==QWEN35_MOE_GROUPED_UINT4_SITE;});
        assert(it!=model.context.expected.end() && it->second.arguments.size()==20);
        for(int field:{6,7,8,9,10,11,12,13,14,15,16,17,18,19}) {

            ++it->second.arguments[field];
            const int launches=uint4_calls;
            outer_rows=rows;active_row=0;norm_calls=0;bool rejected=false;
            try {model.emit_mlp_and_output(0,chunks.front());}
            catch(const std::runtime_error&) {rejected=true;}
            assert(rejected && uint4_calls==launches && active_row==0);
            --it->second.arguments[field];
        }
    } else {
        // Packed assignment count and the resolved tile/dispatch are part of
        // the actual producer/consumer identity, not inferred from raw tokens.
        auto it=std::find_if(model.context.expected.begin(),model.context.expected.end(),
            [](const auto& pair){return pair.second.site_id==QWEN35_MOE_GROUPED_FP8_SITE;});
        assert(it!=model.context.expected.end() && it->second.arguments.size()==17);
        for(int field:{6,7,8,9,10,11,12,13,14,15,16}) {
            ++it->second.arguments[field];
            outer_rows=rows;active_row=0;norm_calls=0;bool rejected=false;
            try {model.emit_mlp_and_output(0,chunks.front());}
            catch(const std::runtime_error&) {rejected=true;}
            assert(rejected && active_row==0);
            --it->second.arguments[field];
        }
    }
    // A corrupted descriptor is rejected before its first MLP launcher.
    for(auto& [key, route]:model.context.expected) if(route.site_id==QWEN35_MOE_ROUTER_SCHEDULE_SITE) {
        ++route.arguments[0];break;
    }
    outer_rows=rows;active_row=0;norm_calls=0;bool rejected=false;
    try {model.emit_mlp_and_output(0,chunks.front());}
    catch(const std::runtime_error&) {rejected=true;}
    assert(rejected && active_row==0);
}
}
int main() {
    for(int cores:{4,6,8}) {
        for(int rows:{1,64,128,192,256,320,512})v3::run_case(cores,rows);
        // P288 padded to 320 with outer C192 has a final 128-row chunk.
        v3::run_case(cores,192,false,false,128);
        v3::run_case(cores,1,true);
        v3::run_case(cores,192,true,false,128);
        v3::run_case(cores,1,false,true);
    }
    for(int cores:{4,6,8})for(int rows:{1,64,128,192,512})v3::run_case(cores,rows,false,false,0,true);
}
'''
    noops = set(re.findall(r'\b(rpu_launch_\w+)\(', methods))
    specialized = set(re.findall(r'void (rpu_launch_\w+)\(', code))
    noops = '\n'.join(f'template<class... A> void {name}(A...) {{}}'
                      for name in sorted(noops - specialized))
    declarations = '\n'.join(
        definition(native, marker).split('{', 1)[0].replace('Qwen3_5MoeModel::', '').strip() + ';'
        for marker in (
            'void Qwen3_5MoeModel::append_mlp_manifest_routes(',
            'void Qwen3_5MoeModel::emit_mlp_and_output(',
            'void Qwen3_5MoeModel::emit_router_and_shuffle(',
            'void Qwen3_5MoeModel::emit_router_topk(',
            'void Qwen3_5MoeModel::emit_shared_expert(',
            'void Qwen3_5MoeModel::emit_routed_experts(',
            'void Qwen3_5MoeModel::emit_moe_merge_and_residual(',
            'FmbLinearRouteSelector Qwen3_5MoeModel::linear_route_selector(',
            'void Qwen3_5MoeModel::launch_linear(',
            'void Qwen3_5MoeModel::launch_grouped_linear(',
            'void Qwen3_5MoeModel::consume_manifest_route(',
            'bool Qwen3_5MoeModel::subclass_chunk_size_valid(',
        ))
    # Consume-manifest helper calls always explicitly identify their slice.
    for marker, value in {
        '__ROUTED_TP__': definition(header, '    int routed_expert_tp() const'),
        '__ROUTED_WIDTH__': definition(header, '    int64_t physical_routed_intermediate() const'),
        '__TYPES__': types, '__CONSTANTS__': constants, '__ENUMS__': enums,
        '__DMA_ROUTE__': definition(native, 'constexpr int64_t qwen35_moe_dma_route('),
        '__SLICES__': definition(native, 'void for_each_moe_slice('),
        '__NOOPS__': noops, '__DECLARATIONS__': declarations, '__METHODS__': methods,
        '__LINEAR_INVOCATION__': definition(header, '    static int64_t linear_invocation('),
        '__GROUPED_ROUTE_ARGUMENTS__': 'template<class T> T Align(T value,T unit) { return (value+unit-1)/unit*unit; }\n' + definition(native, 'std::vector<int64_t> qwen35_grouped_route_arguments('),
        '__GROUPED_LAYOUT__': (ROOT / 'src/core/rpu_grouped_linear_tiling.h').read_text(),
    }.items():
        code = code.replace(marker, value)
    return code


def test_real_mlp_slices_match_manifest_and_launcher_arguments(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    cpp, exe = tmp_path/'mlp.cpp', tmp_path/'mlp'
    cpp.write_text(source_probe())
    built = subprocess.run([compiler, '-std=c++17', '-O0', '-I', str(ROOT/'src/ops'), '-I', str(ROOT/'src/core'), str(cpp), '-o', str(exe)],

                           text=True, capture_output=True)
    assert built.returncode == 0, built.stderr
    run = subprocess.run([str(exe)], text=True, capture_output=True)
    assert run.returncode == 0, run.stderr

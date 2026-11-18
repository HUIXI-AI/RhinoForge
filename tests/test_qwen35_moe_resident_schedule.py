"""Execute production resident schedule/manifest methods against launch facades.

The facades check address arithmetic, ownership and ordering, not tensor math.
Mixer internals and device arithmetic have separate native/CModel coverage.
"""
from pathlib import Path
import re
import runpy
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def source_probe():
    definition = runpy.run_path(str(ROOT / 'tests/test_qwen35_moe_core_native.py'))['_definition']
    native = (ROOT / 'src/fused/rpu_qwen3_5_moe_model.cpp').read_text()
    header = (ROOT / 'src/fused/rpu_qwen3_5_moe_model.h').read_text()
    base = (ROOT / 'src/core/fused_model_base.h').read_text()
    generic = (ROOT / 'src/ops/rpu_grouped_indexed_linear.cpp').read_text()
    generic_header = (ROOT / 'src/ops/rpu_grouped_indexed_linear.h').read_text()
    shuffle = (ROOT / 'src/ops/rpu_moe_index_shuffle.cpp').read_text()
    cache = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    dispatch = '\n'.join(definition(cache, marker) + ';' for marker in (
        'enum class KernelId :', 'static constexpr const char* KERNEL_ID_NAMES[]'))
    namespace = 'namespace rpu_grouped_indexed_linear {'
    generic_types = generic_header.split(namespace, 1)[1].split('} // namespace rpu_grouped_indexed_linear', 1)[0]
    resident_helpers = namespace + generic_types + '\n' + definition(generic, 'const gil::Tile kGilTiles[]').replace('gil::', '') + ';\n'
    resident_helpers += (ROOT / 'src/core/rpu_grouped_indexed_linear_measurements.inc').read_text().replace('gil::', '') + '\n'
    resident_helpers += '\n'.join(definition(generic, marker) for marker in (
        'void validate_shape(', 'Selection canonical(', 'const Tile &selected_tile(', 'SelectionResult pick(', 'Selection resolve('))
    resident_helpers += '\n}\nnamespace gil = rpu_grouped_indexed_linear;\n'
    resident_helpers += '\n' + definition(shuffle, 'struct FullKvsortParams') + ';\n'
    resident_helpers += '\n' + '\n'.join(definition(shuffle, marker) for marker in (
        'FullKvsortParams make_full_kvsort_params(',
        'std::pair<int64_t, int64_t> rpu_moe_kvsort_fp16_spm_sizes('))
    markers = (
        'void Qwen3_5MoeModel::append_resident_mlp_manifest_routes(',
        'void Qwen3_5MoeModel::build_resident_layer(',
        'void Qwen3_5MoeModel::emit_resident_mlp(',
        'void Qwen3_5MoeModel::emit_router_topk(',
        'int64_t Qwen3_5MoeModel::resident_router_capacity(',
        'bool Qwen3_5MoeModel::resident_route_capacity_valid(',
        'FmbLinearRouteSelector Qwen3_5MoeModel::linear_route_selector(',
        'void Qwen3_5MoeModel::launch_linear(',
        'void Qwen3_5MoeModel::consume_manifest_route(',
        'std::vector<ChunkInfo> Qwen3_5MoeModel::planned_resident_mixer_chunks(',
        'std::vector<int64_t> Qwen3_5MoeModel::resident_schedule_arguments(',
    )
    methods = '\n'.join(definition(native, marker) for marker in markers)
    declarations = '\n'.join(
        definition(native, marker).split('{', 1)[0].replace('Qwen3_5MoeModel::', '').strip() + ';'
        for marker in markers)
    types = '\n'.join(definition(base, marker) + ';' for marker in (
        'struct ChunkInfo', 'enum class FmbRouteFamily',
        'enum class FmbLinearRouteSelector', 'struct FmbRouteManifestEntry'))
    constants = '\n'.join(re.findall(
        r'constexpr int64_t (?:QWEN35_MOE_\w+_SITE|kResident\w*|kRouted\w*)\s*=\s*\d+(?:LL)?;', native))
    constants += '\n' + definition(native, 'std::vector<int64_t> shared_alpha_local_arguments(')
    enums = '\n'.join(definition(native, marker) + ';' for marker in (
        'enum class Qwen35MoeMutableDmaRoute', 'enum class Qwen35MoeActivationRoute'))
    code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "fused/qwen35_moe_mixer_plan.h"
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"
#define DWIDTH 2
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
#define TORCH_INTERNAL_ASSERT(ok) TORCH_CHECK(ok)
__DISPATCH__
int availability_queries=0;
std::set<KernelId> missing_payloads;
struct KernelCache {
    static KernelCache& instance() { static KernelCache value;return value; }
    bool has_loaded(KernelId id) const { ++availability_queries;return !missing_payloads.count(id); }
};
__RESIDENT_HELPERS__
namespace c10 { constexpr auto nullopt=std::nullopt; template<class T>using optional=std::optional<T>; struct Half { uint16_t raw; explicit Half(float=0):raw(0){} }; }
namespace at {
constexpr int kHalf=1;
struct Tensor {
    int id=0;
    bool defined() const { return true; }
    int dim() const { return 2; }
    int64_t size(int axis) const { return axis==0 ? 512 : 256; }
    int64_t numel() const { return 32768; }
    int scalar_type() const { return kHalf; }
};
}
enum class RpuRmsNormSpmRoute { BASE };
enum class RpuAllGatherSchedule { BASE };
enum class ValuOpType { SIGMOID, MUL };
enum class GeluMode { NONE };
RpuAllGatherSchedule rpu_resolve_all_gather_schedule(int64_t n,int64_t bytes) {
    assert(n==32 && bytes==2); return RpuAllGatherSchedule::BASE;
}
int64_t CeilDiv(int64_t a,int64_t b) { return (a+b-1)/b; }
namespace v3 { __TYPES__ }
__CONSTANTS__
__MODEL_SHAPE__
__ENUMS__
__DMA_ROUTE__
namespace v3 {
struct FmbPhysicalExecutionManifest { std::vector<FmbRouteManifestEntry> routes; };
using RouteKey=std::tuple<int64_t,int64_t,int64_t>;
RouteKey key(const FmbRouteManifestEntry& r) {
    return {static_cast<int64_t>(r.family),r.site_id,r.invocation};
}
struct Context {
    bool input_in_spm=false, output_to_spm=false;
    std::map<RouteKey,FmbRouteManifestEntry> expected;
    std::set<RouteKey> consumed;
    bool has_complete_physical_manifest() const { return true; }
    void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,
            int64_t flags,const std::vector<int64_t>& args,int64_t invocation=0) {
        RouteKey k{static_cast<int64_t>(family),site,invocation};
        const auto& r=expected.at(k);
        TORCH_CHECK(r.selector==selector && r.flags==flags && r.arguments==args);
        if (site==QWEN35_MOE_RESIDENT_PROJECTION_SITE) {
            last_projection_arguments=args;
            projection_query_count=availability_queries;
        }
        consumed.insert(k);
    }
    static std::vector<int64_t> last_projection_arguments;
    static int projection_query_count;
};
std::vector<int64_t> Context::last_projection_arguments;
int Context::projection_query_count=0;
// Distinct symbolic addresses make missing offsets and forbidden aliases visible.
// The intentional X/partial and Gate/Act aliases match production declarations.
std::map<std::string,uint32_t> addresses={
    {"residual1",0x100000},{"input_norm",0x500000},
    {"moe_resident_input",0x600000},{"moe_partial",0x600000},
    {"moe_resident_route_keys",0xa00000},{"moe_resident_ids",0xb00000},
    {"moe_resident_coefficients",0xb10000},{"moe_resident_counts",0xb20000},
    {"moe_resident_batch_counts",0xb30000},{"moe_routed_act",0xc00000},
    {"moe_routed_up",0xd00000},{"moe_shared_gate",0xe00000},
    {"moe_shared_up",0xe20000},{"moe_shared_act",0xe00000},
    {"moe_shared_alpha_pad",0xe40000},{"moe_shared_alpha",0xe50000},
    {"moe_ring_output",0xf00000},{"final_norm_w",0xf80000},
    {"moe_router_local",0x1100000},{"moe_router_logits",0x1200000},
    {"moe_router_probs",0x1300000},{"moe_route_expert_ids",0x1400000},
    {"moe_route_token_ids",0x1500000},{"moe_m_sizes",0x1600000},
    {"moe_expert_ids",0x1700000},{"moe_route_scores",0x1800000},
    {"moe_token_ids",0x1900000},{"moe_route_logits",0x1a00000},
    {"moe_resident_route_scores",0x1b00000},{"moe_resident_sort_values",0x1b10000},
    {"moe_resident_sorted_experts",0x1b20000},{"moe_resident_sorted_ordinals",0x1b30000},
    {"moe_resident_indices_i32",0x1b40000}};
uint32_t a(const char* name,int core=0) { return addresses.at(name)+core*0x2000000; }
int fixture_weight_mode=6;
int64_t w=0,outer_offset=0,position=0,mixer_row=0,router_row=0,route_first=0,ring_row=0;
int64_t fixture_mixer_rows=384;
int norms=0,final_norms=0,router_batches=0,indexed_calls=0,shared_step=0;
int input_dmas=0,output_dmas=0,fullsorts=0,down_calls=0;
int table_loads=0,casts=0,gathers=0,bincounts=0;
bool mixer_pending=false,mlp_started=false,partial_ready=false,down_done=false;
bool ring_pending=false,router_normalized=false,final_layer=false,throw_mixer=false;
void reset(int64_t rows,int64_t offset,int64_t pos,bool final) {
    w=rows;outer_offset=offset;position=pos;final_layer=final;
    mixer_row=router_row=route_first=ring_row=0;
    norms=final_norms=router_batches=indexed_calls=shared_step=0;
    input_dmas=output_dmas=fullsorts=down_calls=0;
    table_loads=casts=gathers=bincounts=0;
    mixer_pending=mlp_started=partial_ready=down_done=ring_pending=router_normalized=false;
}
void rpu_launch_rmsnorm_spm_kernel(uint32_t input,uint32_t output,uint32_t,
        int64_t rows,int64_t h,double,RpuRmsNormSpmRoute,int nc) {
    assert(h==2048 && nc==8);
    if(output==a("input_norm")) {
        assert(!mlp_started && !mixer_pending);
        assert(rows==std::min(fixture_mixer_rows,w-mixer_row));
        assert(input==a("residual1")+mixer_row*4096); ++norms;
    } else if(output==a("moe_resident_input")) {
        assert(input==a("residual1") && rows==w && mixer_row==w && !mlp_started);
        mlp_started=true; ++norms;
    } else {
        assert(output==a("moe_ring_output") && input==output && final_layer && ring_pending);
        assert(rows==std::min<int64_t>(192,w-ring_row)); ++final_norms;
    }
}
void rpu_launch_spm_local_copy_dma(uint32_t src,int64_t elements,uint32_t dst,int nc) {
    assert(nc==8);
    if(src==a("input_norm")) {
        assert(mixer_pending && elements==std::min(fixture_mixer_rows,w-mixer_row)*2048);
        assert(dst==a("residual1")+mixer_row*4096);
        mixer_row+=elements/2048; mixer_pending=false;
    } else {
        assert(src==a("moe_ring_output") && ring_pending);
        assert(elements==std::min<int64_t>(192,w-ring_row)*2048);
        assert(dst==a("residual1")+ring_row*4096);
        if(final_layer) assert(final_norms==ring_row/192+1);
        ring_row+=elements/2048; ring_pending=false;
    }
}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t input,const at::Tensor& weight,
        uint32_t output,int64_t m,int64_t n,int64_t k,int part,int nc,uint32_t,
        const at::Tensor&,uint32_t,int,bool acc32,bool gemv) {
    assert(nc==8 && !acc32 && !gemv && mlp_started && !partial_ready);
    if(weight.id==1) {
        assert(m==std::min<int64_t>(512,w-router_row) && n==256 && k==2048 && part==1);
        assert(input==a("moe_resident_input")+router_row*4096 && output==a("moe_router_local"));
        assert(!router_normalized);
    } else {
        assert(route_first==w*8 && m==w);
        const int ids[]={2,3,5,4}; assert(weight.id==ids[shared_step++]);
        if(weight.id==4) {
            assert(k==512 && n==2048 && part==0);
            assert(input==a("moe_shared_act") && output==a("moe_partial"));
        } else {
            assert(input==a("moe_resident_input") && k==2048 && part==1);
            assert(n==(weight.id==5?128:512));
            assert(output==a(weight.id==2?"moe_shared_gate":weight.id==3?"moe_shared_up":"moe_shared_alpha_pad"));
        }
    }
}
void rpu_launch_all_gather_spm_kernel(uint32_t src,uint32_t dst,int64_t rows,
        int64_t local_n,int64_t bytes,int nc,RpuAllGatherSchedule) {
    assert(src==a("moe_router_local") && dst==a("moe_router_logits"));
    assert(rows==std::min<int64_t>(512,w-router_row) && local_n==32 && bytes==2 && nc==8);
}
void rpu_launch_moe_topk_scalar_fp16_spm_kernel(uint32_t probs,uint32_t expert_table,
        uint32_t scores,uint32_t experts,int64_t rows,int64_t e,int64_t k,
        bool descending,bool cvt,int nc) {
    assert(probs==a("moe_router_probs") && expert_table==a("moe_expert_ids"));
    assert(scores==a("moe_route_scores"));
    assert(experts==a("moe_resident_route_keys")+router_row*8*2);
    assert(rows==std::min<int64_t>(512,w-router_row) && e==256 && k==8);
    assert(descending && !cvt && nc==8 && !router_normalized);
    router_normalized=true;
}
void rpu_launch_moe_topk_prob_norm_fp16_spm_kernel(uint32_t input,uint32_t scores,
        uint32_t output,int64_t rows,int64_t e,int64_t topk,int nc) {
    assert(router_normalized && input==a("moe_route_scores") && scores==input);
    assert(output==a("moe_resident_route_scores")+router_row*8*2);
    assert(rows==std::min<int64_t>(512,w-router_row) && e==8 && topk==8 && nc==8);
    router_row+=rows;++router_batches;router_normalized=false;
}
void rpu_launch_ddr_broadcast_spm_dma(const at::Tensor& table,int64_t offset,
        int64_t elems,uint32_t output,int nc) {
    assert(router_row==w && offset==0 && elems==w*8 && nc==8);
    assert(output==a("moe_resident_sort_values"));
    assert(table.id==(table_loads==0?9:10));
    assert(table_loads==0?fullsorts==0:(fullsorts==1 && gathers==1));
    ++table_loads;
}
void rpu_launch_moe_kvsort_fp16_spm_kernel(uint32_t keys,uint32_t values,
        uint32_t outkeys,uint32_t outvalues,int64_t routes,bool descending,int nc) {
    assert(router_row==w && table_loads==1 && fullsorts==0 && routes==w*8);
    assert(keys==a("moe_resident_route_keys") && values==a("moe_resident_sort_values"));
    assert(outkeys==a("moe_resident_sorted_experts") &&
           outvalues==a("moe_resident_sorted_ordinals") && !descending && nc==8);
    ++fullsorts;
}
void rpu_launch_cast_uint16_int32_spm_kernel(uint32_t input,uint32_t output,
        int64_t routes,int nc) {
    assert(fullsorts==1 && casts==0 && routes==w*8 && nc==8);
    assert(input==a("moe_resident_sorted_ordinals") &&
           output==a("moe_resident_indices_i32"));++casts;
}
void rpu_launch_gather_fp16_spm_kernel(uint32_t input,uint32_t indices,
        uint32_t output,int64_t n,int64_t c,int axis,int64_t k,int nc) {
    assert(casts==1 && indices==a("moe_resident_indices_i32"));
    assert(n==1 && c==w*8 && axis==1 && k==w*8 && nc==8);
    assert(table_loads==gathers+1);
    assert(input==a(gathers==0?"moe_resident_route_scores":"moe_resident_sort_values"));
    assert(output==a(gathers==0?"moe_resident_coefficients":"moe_resident_ids"));
    ++gathers;
}
void rpu_launch_moe_bincount_scalar_spm_kernel(uint32_t input,uint32_t converted,
        uint32_t counts,int64_t rows,int64_t e,bool cvt,bool output_cvt,int nc) {
    assert(gathers==2 && input==converted && e==256 && nc==8);
    const int64_t first=bincounts==0?0:(bincounts-1)*2048;
    assert(input==a("moe_resident_sorted_experts")+first*2);
    assert(rows==(bincounts==0?w*8:std::min<int64_t>(2048,w*8-first)));
    assert(cvt==(bincounts==0) && output_cvt==cvt);
    assert(counts==(bincounts==0?a("moe_resident_counts"):
                   a("moe_resident_batch_counts")+(bincounts-1)*512));
    ++bincounts;
}
void rpu_launch_qwen35_moe_indexed_uint4(uint32_t input,const at::Tensor& weight,uint32_t counts,
        const at::Tensor&,const at::Tensor&,uint32_t ids,uint32_t output,int64_t tokens,int64_t routes) {
    assert(fullsorts==1 && bincounts==1+CeilDiv(w*8,2048));
    assert(input==a("moe_resident_input") && tokens==w && shared_step==0);
    assert(routes==std::min<int64_t>(2048,w*8-route_first));
    assert(ids==a("moe_resident_ids")+route_first*2);
    assert(counts==a("moe_resident_batch_counts")+(route_first/2048)*512);
    const bool gate=indexed_calls%2==0;
    assert(weight.id==(fixture_weight_mode==6?(gate?16:17):(gate?6:7)));
    assert(output==(gate?a("moe_routed_act")+route_first*128:a("moe_routed_up"))); ++indexed_calls;
}
void rpu_launch_silu_mul_spm_kernel(uint32_t gate,uint32_t up,uint32_t output,int64_t n,int nc) {
    assert(nc==8);
    if(gate==a("moe_shared_gate")) {
        assert(shared_step==2 && route_first==w*8 && up==a("moe_shared_up"));
        assert(output==a("moe_shared_act") && n==w*64);
    } else {
        assert(indexed_calls==2*(route_first/2048+1));
        assert(gate==a("moe_routed_act")+route_first*128 && output==gate && up==a("moe_routed_up"));
        assert(n==std::min<int64_t>(2048,w*8-route_first)*64); route_first+=n/64;
    }
}
void rpu_launch_slice_spm_kernel(uint32_t input,uint32_t output,std::initializer_list<int64_t> shape,
        std::initializer_list<int64_t> out,std::initializer_list<int64_t> begin,int nc,bool local) {
    assert(shared_step==3 && nc==8 && local);
    assert(std::vector<int64_t>(shape)==std::vector<int64_t>({w,16}));
    assert(std::vector<int64_t>(out)==std::vector<int64_t>({w,1}));
    assert(std::vector<int64_t>(begin)==std::vector<int64_t>({0,0}));
    assert(input==a("moe_shared_alpha_pad") && output==a("moe_shared_alpha"));
}
void rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(uint32_t alpha,uint32_t partial,uint32_t out,
        int64_t rows,int64_t h,c10::Half,ValuOpType op,bool,int nc) {
    assert(shared_step==4 && alpha==a("moe_shared_alpha") && partial==a("moe_partial") && out==partial);
    assert(rows==w && h==2048 && nc==8 && op==ValuOpType::MUL); partial_ready=true;
}
void rpu_launch_qwen35_moe_down_combine_uint4(uint32_t act,const at::Tensor& weight,uint32_t counts,
        const at::Tensor&,const at::Tensor&,uint32_t ids,uint32_t coeff,uint32_t partial,int64_t rows) {
    assert(partial_ready && shared_step==4 && route_first==w*8 && weight.id==(fixture_weight_mode==6?18:8));
    assert(act==a("moe_routed_act") && counts==a("moe_resident_counts"));
    assert(ids==a("moe_resident_ids") && coeff==a("moe_resident_coefficients"));
    assert(partial==a("moe_partial") && rows==w); down_done=true; ++down_calls;
}
void rpu_launch_grouped_indexed_linear_selected(
        const gil::Shape& shape,
        const gil::Selection& selection,
        uint32_t input,const at::Tensor& weight,uint32_t counts,const c10::optional<at::Tensor>& scale,
        const c10::optional<at::Tensor>& negative_zero,uint32_t ids,uint32_t coefficients,uint32_t output) {
    const auto& args=Context::last_projection_arguments;
    assert(args.size()==13 && args[7]==kResidentProjectionLayoutVersion+(fixture_weight_mode==5) && args[8]==selection.m && args[9]==selection.n &&
           args[10]==selection.k && args[11]==selection.expert_partitions && args[12]==shape.TP);
    assert(shape.T==args[0] && shape.R==args[1] && shape.N==args[3] && shape.K==args[4] &&
           shape.E==args[5] && shape.Q==args[6]);
    assert(!missing_payloads.count(gil::selected_tile(shape,selection).id));
    assert(availability_queries==Context::projection_query_count);
    assert(negative_zero.has_value()==(fixture_weight_mode==6));
    assert(shape.weight_type==(fixture_weight_mode==6?gil::WeightType::Uint4:gil::WeightType::Float8E4M3));
    const auto zero=negative_zero.value_or(at::Tensor{});
    if(shape.indexed) {
        assert(coefficients==0);
        rpu_launch_qwen35_moe_indexed_uint4(input,weight,counts,*scale,zero,ids,output,shape.T,shape.R);
    } else {
        assert(shape.R==shape.T*8);
        rpu_launch_qwen35_moe_down_combine_uint4(input,weight,counts,*scale,zero,ids,coefficients,output,shape.T);
    }
}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t partial,uint32_t residual,uint32_t out,
        int64_t rows,int64_t h,int producers,int consumers) {
    assert(down_done && !ring_pending && rows==std::min<int64_t>(192,w-ring_row));
    assert(partial==a("moe_partial")+ring_row*4096 && residual==a("residual1")+ring_row*4096);
    assert(out==a("moe_ring_output") && h==2048 && producers==8 && consumers==8); ring_pending=true;
}
__NOOPS__
struct Qwen3_5MoeModel {
    // The real model planning/restore boundary is covered by the allocator
    // probe. This launch fixture supplies an immutable schedule so emission
    // can independently verify dynamic mixer shapes and buffer offsets.
    using ResidentMixerPlan=qwen35_moe_planning::Plan;
    ResidentMixerPlan fixture_plan;
    const ResidentMixerPlan& resident_plan_for_outer(const ChunkInfo& c) const {
        assert(c.idx>=0 && static_cast<size_t>(c.idx)<fixture_plan.windows.size());
        const auto& window=fixture_plan.windows[c.idx];
        assert(window.offset==c.offset && window.rows==c.len);
        return fixture_plan;
    }
    struct Weight {
        int64_t routed_weight_mode=fixture_weight_mode;
        at::Tensor router_w{1},shared_gate_w{2},shared_up_w{3},shared_down_w{4},shared_scalar_gate_w{5};
        at::Tensor routed_gate_w{6},routed_up_w{7},routed_down_w{8};
        at::Tensor resident_gate_w{16},resident_up_w{17},resident_down_w{18};
        at::Tensor routed_gate_scale,routed_up_scale,routed_down_scale,routed_gate_negative_zero,routed_down_negative_zero;
    };
    struct Moe { int64_t num_experts=256,top_k=8; } moe_;
    std::vector<Weight> moe_layer_weights_=std::vector<Weight>(40);
    std::vector<uint8_t> layer_is_full_=std::vector<uint8_t>(40);
    at::Tensor router_bridge_,expert_ids_,resident_route_ordinals_{9},resident_route_tokens_{10};
    bool linear_acc32_=false;
    uint32_t mixer_residual_offset_=0;
    double eps_=1e-6;
    Context context;
    Context& ctx() { return context; }
    int num_cores() const { return 8; }
    int shared_expert_tp() const { return 8; }
    int router_tp() const { return 8; }
    int routing_tp() const { return 8; }
    int64_t num_layers() const { return 40; }
    int64_t hidden_size() const { return 2048; }
    int64_t text_ring_route(int64_t rows,int64_t h) const { assert(rows<=192 && h==2048); return 2; }
    uint32_t addr(int core,const char* name) const { return a(name,core); }
    uint32_t layer_addr(int,int,const char*) const { return 0; }
    void emit_layer_input_dma(int,const ChunkInfo& c) {
        assert(c.offset==outer_offset && c.len==w && !mlp_started); ++input_dmas;
    }
    void emit_layer_output_dma(int,const ChunkInfo& c) {
        assert(c.offset==outer_offset && c.len==w && ring_row==w && !ring_pending); ++output_dmas;
    }
    void mixer(const ChunkInfo& c) {
        assert(!mlp_started && !mixer_pending);
        assert(c.offset==outer_offset+mixer_row && c.len==std::min(fixture_mixer_rows,w-mixer_row));
        assert(c.idx==c.offset/64 && c.kv_seq_len==position+c.offset+c.len);
        assert(mixer_residual_offset_==mixer_row*4096);
        if(throw_mixer && mixer_row>0) throw std::runtime_error("mixer failure");
        mixer_pending=true;
    }
    void build_full_attention(int,const ChunkInfo& c) { mixer(c); }
    void build_gdn(int,const ChunkInfo& c,bool decode) { assert(!decode); mixer(c); }
__LINEAR_INVOCATION__
__DECLARATIONS__
};
__METHODS__
void run_case(int rows,int64_t offset,bool full,bool spm,int tail=0,int mode=6) {
    fixture_weight_mode=mode;
    const auto original_missing=missing_payloads;
    Qwen3_5MoeModel model;
    model.context.input_in_spm=model.context.output_to_spm=spm;
    std::fill(model.layer_is_full_.begin(),model.layer_is_full_.end(),full);
    std::vector<ChunkInfo> chunks={{0,offset,rows,320+offset+rows}};
    if(tail) chunks.push_back({1,offset+rows,tail,320+offset+rows+tail});
    fixture_mixer_rows=rows==704?320:384;
    auto& plan=model.fixture_plan;
    plan.request={rows,chunks.back().offset+chunks.back().len,
                  chunks.back().offset+chunks.back().len,320,64};
    plan.mixer_capacity=std::min<int64_t>(rows,fixture_mixer_rows);
    plan.workspace_bytes=262144;
    for(const auto& c:chunks) {
        qwen35_moe_planning::Window window{
            c.offset,c.len,std::min(fixture_mixer_rows,c.len),{}};
        for(int64_t local=0;local<c.len;local+=window.uniform_rows) {
            const int64_t q=std::min(window.uniform_rows,c.len-local);
            window.calls.push_back({c.offset+local,q,320+c.offset+local,q,262144});
            ++plan.total_calls;
        }
        plan.windows.push_back(std::move(window));
    }
    FmbPhysicalExecutionManifest manifest;
    for(const auto& c:chunks) model.append_resident_mlp_manifest_routes(manifest,c);
    for(const auto& route:manifest.routes) assert(model.context.expected.emplace(key(route),route).second);
    for(const auto& c:chunks) for(int layer:{0,39}) {
        reset(c.len,c.offset,320,layer==39);
        model.build_resident_layer(layer,c);
        assert(model.mixer_residual_offset_==0 && mixer_row==c.len && ring_row==c.len);
        assert(norms==CeilDiv(c.len,fixture_mixer_rows)+1 && final_norms==(layer==39?CeilDiv(c.len,192):0));
        assert(input_dmas==!spm && output_dmas==!spm && down_calls==1);
        assert(router_batches==CeilDiv(c.len,512) && indexed_calls==2*CeilDiv(c.len*8,2048));
        assert(fullsorts==1 && table_loads==2 && casts==1 && gathers==2);
        assert(bincounts==1+CeilDiv(c.len*8,2048));
    }
    assert(model.context.consumed.size()==manifest.routes.size());
    // Resident activation/ring receipts have their own stable physical sites.
    // They must neither consume a legacy-slice identity nor launch after a
    // changed selector in their exact prepared descriptor.
    for(auto sites:std::vector<std::pair<int64_t,int64_t>>{
            {QWEN35_MOE_RESIDENT_ROUTED_SILU_SITE,QWEN35_MOE_ROUTED_SILU_SITE},
            {QWEN35_MOE_RESIDENT_SHARED_SILU_SITE,QWEN35_MOE_SHARED_SILU_SITE},
            {QWEN35_MOE_RESIDENT_SHARED_SIGMOID_SITE,QWEN35_MOE_SHARED_SIGMOID_SITE},
            {QWEN35_MOE_RESIDENT_SHARED_ALPHA_LOCAL_SITE,QWEN35_MOE_SHARED_ALPHA_LOCAL_SITE},
            {QWEN35_MOE_RESIDENT_ALL_REDUCE_SITE,QWEN35_MOE_ALL_REDUCE_SITE}}) {
        assert(sites.first!=sites.second);
        assert(std::none_of(manifest.routes.begin(),manifest.routes.end(),
            [&](const auto& route){return route.site_id==sites.second;}));
        auto found=std::find_if(model.context.expected.begin(),model.context.expected.end(),
            [&](const auto& entry){return entry.second.site_id==sites.first;});
        assert(found!=model.context.expected.end());
        auto& route=found->second;++route.selector;
        const auto& c=chunks.front();reset(c.len,c.offset,320,false);
        bool rejected=false;
        try {model.build_resident_layer(0,c);} catch(const std::runtime_error&) {rejected=true;}
        assert(rejected && !ring_pending && ring_row==0);
        if(sites.first==QWEN35_MOE_RESIDENT_ROUTED_SILU_SITE) assert(route_first==0);
        if(sites.first==QWEN35_MOE_RESIDENT_SHARED_SILU_SITE) assert(shared_step==2);
        if(sites.first==QWEN35_MOE_RESIDENT_SHARED_SIGMOID_SITE ||
           sites.first==QWEN35_MOE_RESIDENT_SHARED_ALPHA_LOCAL_SITE) assert(shared_step==3);
        if(sites.first==QWEN35_MOE_RESIDENT_ALL_REDUCE_SITE) assert(down_calls==1);
        --route.selector;
    }
    // Descriptor corruption must reject before the corresponding kernel launch.
    for(auto& entry:model.context.expected) {
        auto& route=entry.second;
        if(route.site_id!=QWEN35_MOE_RESIDENT_PROJECTION_SITE || route.selector!=2) continue;
        ++route.arguments[6];
        const auto& c=chunks.front(); reset(c.len,c.offset,320,false);
        bool rejected=false;
        try { model.build_resident_layer(0,c); } catch(const std::runtime_error&) { rejected=true; }
        assert(rejected && down_calls==0 && ring_row==0 && model.mixer_residual_offset_==0);
        --route.arguments[6]; break;
    }
    // All final selector fields (including P1 versus P8 and TP)
    // are descriptor authority for both projection families.
    for(int selector:{1,2}) for(int field:{7,8,9,10,11,12}) {
        auto found=std::find_if(model.context.expected.begin(),model.context.expected.end(),
            [&](const auto& entry){return entry.second.site_id==QWEN35_MOE_RESIDENT_PROJECTION_SITE &&
                                          entry.second.selector==selector;});
        assert(found!=model.context.expected.end());
        auto& route=found->second;++route.arguments[field];
        const auto& c=chunks.front();reset(c.len,c.offset,320,false);
        bool rejected=false;
        try { model.build_resident_layer(0,c); } catch(const std::runtime_error&) { rejected=true; }
        assert(rejected && (selector==1?indexed_calls==0:down_calls==0));
        --route.arguments[field];
    }
    // Missing mandatory canonical payload rejects manifest production, before
    // any projection can launch or acquire a descriptor with an absent tile.
    for(auto id:{mode==6?KernelId::GPL_INDEXED_W4_M256N32K256:KernelId::GPL_INDEXED_W8_M256N32K256,
                 mode==6?KernelId::GPL_SCATTER_W4_M256N128K64:KernelId::GPL_SCATTER_W8_M256N128K64}) {
        missing_payloads.insert(id);FmbPhysicalExecutionManifest unavailable;
        reset(rows,offset,320,false);bool rejected=false;
        try { model.append_resident_mlp_manifest_routes(unavailable,chunks.front()); }
        catch(const std::runtime_error&) { rejected=true; }
        assert(rejected && indexed_calls==0 && down_calls==0);
        missing_payloads=original_missing;
    }
    // A changed cold payload snapshot cannot silently change a recorded tile.
    // Reject missing selected payloads before the affected projection launches.
    for(bool indexed:{true,false}) {
        const auto shape=qwen35_resident_projection_shape(indexed,rows,indexed?std::min(rows*8,2048):rows*8,mode);
        const auto preferred=gil::canonical(shape);
        const auto id=gil::selected_tile(shape,preferred).id;
        if(missing_payloads.count(id)) continue;
        missing_payloads.insert(id);reset(rows,offset,320,false);bool rejected=false;
        try { model.build_resident_layer(0,chunks.front()); }
        catch(const std::runtime_error&) { rejected=true; }
        assert(rejected && (indexed?indexed_calls==0:down_calls==0));
        missing_payloads=original_missing;
    }
    // Offset guard also restores state when a non-first mixer slice fails.
    reset(rows,offset,320,false);
    if(rows>fixture_mixer_rows) {
        throw_mixer=true; bool rejected=false;
        // Directly exercise the real whole-layer guard, with failure on its second slice.
        try { model.build_resident_layer(0,chunks.front()); } catch(const std::runtime_error&) { rejected=true; }
        throw_mixer=false;
        assert(rejected && model.mixer_residual_offset_==0);
    }
}
}
int main() {
    for(int rows:{64,320,768,1024}) v3::run_case(rows,0,false,false,64,5);
    for(int rows:{64,128,256,320,384,512,704,768,832,1024}) {
        v3::run_case(rows,0,false,false);
        v3::run_case(rows,768,true,true,64);
    }
    // The P4096 schedule's final resident window has 256 rows.
    v3::run_case(768,3072,true,false,256);
    // The new complete REF may contain only the two required canonical W4
    // entries; other precisions/tiles are not dependencies of this model.
    for(const auto& tile:gil::kGilTiles)
        if(tile.id!=KernelId::GPL_INDEXED_W4_M256N32K256 &&
           tile.id!=KernelId::GPL_SCATTER_W4_M256N128K64)
            missing_payloads.insert(tile.id);
    v3::run_case(768,3072,true,false,256);
    missing_payloads.clear();
}
'''
    # Arithmetic-only router helpers are intentionally outside this scheduler test.
    noops = set(re.findall(r'\b(rpu_launch_\w+)\(', methods))
    specialized = set(re.findall(r'void (rpu_launch_\w+)\(', code))
    replacements = {
        '__DISPATCH__': dispatch,
        '__RESIDENT_HELPERS__': resident_helpers,
        '__TYPES__': types,
        '__CONSTANTS__': constants,
        '__MODEL_SHAPE__': definition(native, 'gil::Shape qwen35_resident_projection_shape('),
        '__ENUMS__': enums,
        '__DMA_ROUTE__': definition(native, 'constexpr int64_t qwen35_moe_dma_route('),
        '__LINEAR_INVOCATION__': definition(header, '    static int64_t linear_invocation('),
        '__DECLARATIONS__': declarations,
        '__METHODS__': methods,
        '__NOOPS__': '\n'.join(f'template<class... A> void {name}(A...) {{}}' for name in sorted(noops-specialized)),
    }
    for marker, value in replacements.items():
        code = code.replace(marker, value)
    return code


def test_resident_manifest_schedule_addresses_and_order(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    source, executable = tmp_path / 'resident.cpp', tmp_path / 'resident'
    source.write_text(source_probe())
    build = subprocess.run([compiler, '-std=c++17', '-O0', '-I', str(ROOT / 'src/core'),
                            '-I', str(ROOT / 'src/ops'),
                            '-I', str(ROOT / 'src'), str(source), '-o', str(executable)],
                           text=True, capture_output=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(executable)], text=True, capture_output=True)
    assert run.returncode == 0, run.stderr

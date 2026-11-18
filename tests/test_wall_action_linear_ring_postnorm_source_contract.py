from pi_wall_host_catalog import assert_current_catalog, CATALOG_INCLUDE
from test_siglip_attention_dry_live_contract import _function
"""Board-free contract for the exact Wall Action linear/ring/postnorm path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QWEN3 = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
ACTION = (ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()
LINEAR = (ROOT / "src/ops/rpu_linear.cpp").read_text()
CACHE = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
BASE_H = (ROOT / "src/core/fused_model_base.h").read_text()


def _section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def test_exact_path_is_fail_closed_and_excludes_residual_injection_modes():
    guard = _section(
        QWEN3,
        "const bool wall_action_linear_ring_postnorm =",
        "TORCH_CHECK(!decoder_mixed_precision_",
    )
    for contract in (
        "bs == 1",
        "!ctx().is_causal",
        "ctx().attention_mask.has_value()",
        "pre_layers_residual1_ready_",
        "!adarms_",
        "!residual_is_bf16",
        "!transition_to_bf16",
        "deepstack_lang_layers_.empty()",
        "qwen3vl_pooler_z1_retained_deepstack_count_ == 0",
        "!qwen3vl_pooler_z1_dispatch_",
        "!qwen3vl_multiview_text_dry_prepared_",
        "!qwen3vl_multiview_text_composite_prepared_",
        "!qwen3vl_multiview_text_composite_dispatch_",
        "layer_weights_.size() == 36",
    ):
        assert contract in guard


def test_layer_zero_seeds_full_carry_then_later_layers_stay_compact():
    assert "if (wall_action_linear_ring_postnorm && layer_idx > 0)" in QWEN3
    assert "/*residual_local_shard=*/true" in QWEN3
    assert "const bool residual_local_shard = layer_idx > 0;" in QWEN3
    assert '? addr(0, "action_compact_residual")\n                    : addr(0, "residual2")' in QWEN3
    assert "/*skip_down=*/wall_action_linear_ring_postnorm" in QWEN3
    assert "is_last_layer && !wall_action_linear_ring_postnorm" in QWEN3
    assert "bool skip_down = false" in BASE_H


def test_exact_buffer_and_kernel_launch_contract_are_present():
    assert_current_catalog("ActionRingPostnorm", "ActionRingPostnormSpmWeight")
    assert "h == 1024 && intermediate_size() == 2048" in ACTION
    assert "num_layers() == 36 && attn_tp() == NUM_CORES" in ACTION
    launcher=_function(LINEAR,"void rpu_launch_wall_action_linear_ring_postnorm_spm_kernel(")
    assert "residual_local_shard ? 2 : 0" in launcher
    assert "queue->set_broadcast_mode(true);" in launcher
    assert "{selected.grid_x, selected.grid_y, 1}" in launcher
    assert "missing from expansion ref" in launcher


def test_manifest_covers_actual_first_and_later_action_layers(tmp_path):
    """Run native route production/consumption with host-only tensor metadata."""
    import re
    import shutil
    import subprocess

    def block(source, marker):
        begin = source.index(marker)
        brace = source.index("{", begin)
        depth, end = 1, brace + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[begin:end]

    producer_guard = _section(QWEN3, "        const int64_t rows = physical_len;",
                              "        const bool wall_base_nvfp4_o_ring_norm =")
    producer = block(QWEN3, "if (wall_action_linear_ring) {")
    compact = block(ACTION, "    if (wall_action_fused_rows_supported(ctx.chunk_size, wall_sorting_)")
    emit = QWEN3[QWEN3.index("        const bool wall_action_w8a16_silu_mul ="):]
    emit_begin = QWEN3.rindex("        const bool wall_action_fp16_transfer =", 0,
                             QWEN3.index("        const bool wall_action_w8a16_silu_mul ="))
    consumer_guard = _section(QWEN3[emit_begin:], "        const bool wall_action_fp16_transfer =",
                             "        TORCH_CHECK(!decoder_mixed_precision_")
    consumer_if = block(emit, "if (wall_action_linear_ring_postnorm && layer_idx > 0) {")
    following = emit[emit.index(consumer_if) + len(consumer_if):]
    consumer_o = consumer_if + " " + block(following, "else {")
    consumer_down = block(emit, "        if (wall_action_linear_ring_postnorm) {")
    helpers = "\n".join(block(QWEN3, marker) for marker in (
        "    bool wall_action_fp16_transfer_enabled() const",
        "    static bool wall_action_qkv_weights_compatible(",
        "    static bool wall_action_ring_weights_compatible(",
        "    void consume_causal_route(",
        "    bool consume_ordinary_o_route(",
        "    std::vector<int64_t> ordinary_o_route_arguments(",
        "    CausalDecoderLinearRoute decoder_projection_route(",
        "    bool qwen3vl_decode_gemv(",
    ))
    planner = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    consume_route = block(planner, "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(")
    consume_route = consume_route.replace("FmbPhysicalManifestConsumer::consume_route", "consume_physical_route")
    declarations = "\n".join(re.findall(
        r"(?:inline )?(?:static )?constexpr (?:int64_t|int|uint32_t) "
        r"(?:CAUSAL_DECODER_\w+_SITE|WALL_ACTION_\w+)\s*=\s*[^;]+;", QWEN3))
    enums = "\n".join((block(BASE_H, "enum class FmbLinearRouteSelector : int") + ";",
                       block(QWEN3, "enum class CausalDecoderPlanningMode") + ";",
                       block(QWEN3, "enum class CausalDecoderLinearRoute : int64_t") + ";",
                       block(QWEN3, "enum class CausalDecoderAllReduceRoute : int64_t") + ";"))
    kernel_names = set(re.findall(r"\b(rpu_\w+)\(", consumer_o + consumer_down))
    kernel_names.discard("rpu_launch_wall_action_linear_ring_postnorm_spm_kernel")
    stubs = "\n".join(f"#define {name}(...) ((void)0)" for name in sorted(kernel_names))
    program = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>
#include "src/core/rpu_wall_sorting_config.h"
#include "src/ops/rpu_linear_tiling.h"
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("route rejected"); } while (0)
#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
constexpr int NUM_CORES=8;
namespace at {
enum ScalarType { kHalf, kChar };
struct Tensor {
    ScalarType dtype=kHalf; bool present=true;
    ScalarType scalar_type() const { return dtype; }
    bool defined() const { return present; }
};
struct IntArrayRef : std::vector<int64_t> {
    using std::vector<int64_t>::vector;
    IntArrayRef(const std::vector<int64_t>& x):std::vector<int64_t>(x) {}
};
}
enum class FmbRouteFamily { LINEAR, ALL_REDUCE };
struct ChunkInfo { int64_t idx=0,offset=0,len=20; };
struct Plan { struct { std::vector<ChunkInfo> chunks{{}}; } compute; };
struct LayoutContext { int64_t batch_size=1,chunk_size=20; bool is_causal=false,use_attn_mask=true; };
struct FmbRouteManifestEntry {
    int64_t site_id; FmbRouteFamily family; int64_t selector,flags;
    std::vector<int64_t> arguments; int64_t invocation;
};
struct Manifest { std::vector<FmbRouteManifestEntry> routes; };
struct Context {
    bool complete=true,is_causal=false; int64_t batch_size=1,position=384;
    std::optional<int> attention_mask{1}; Plan stage_plan; Manifest manifest_;
    const auto& manifest() const { return manifest_; }
    mutable std::vector<int> consumed_;
    bool has_complete_physical_manifest() const { return complete; }
    const FmbRouteManifestEntry& find_route(FmbRouteFamily family,int64_t site,int64_t invocation) const {
        for(const auto& r:manifest_.routes)
            if(r.family==family && r.site_id==site && r.invocation==invocation) return r;
        throw std::runtime_error("missing route");
    }
    CONSUME_ROUTE
};
DECLARATIONS
ENUMS
STUBS
struct Launch { uint32_t weight; bool local; int64_t rows; };
std::vector<Launch> launches;
void rpu_launch_wall_action_linear_ring_postnorm_spm_kernel(
    uint32_t,const at::Tensor&,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,
    const at::Tensor&,bool local,double,uint32_t weight,const WallSortingConfig&,int64_t rows) {
    launches.push_back({weight,local,rows});
}
int64_t causal_ring_route(int64_t,int64_t) { return 1; }
struct Model {
    struct LayerWeights {
        at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,input_norm_w,post_norm_w;
        at::Tensor q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws;
    };
    std::vector<LayerWeights> layer_weights_{36};
    WallSortingConfig wall_sorting_;
    Context context; at::Tensor final_norm_w_;
    bool wall_action_fuse_silu_mul_=true,use_silu_=true,has_qkv_bias_=true;
    bool qwen3vl_4b_w8_profile_=false,qwen3vl_fp16_decode_profile_=false;
    bool decoder_mixed_precision_=false,lingbot_folded_rmsnorm_=false;
    void emit_lingbot_nogamma_norm(uint32_t,uint32_t,int64_t,int64_t)const{assert(false);}
    int64_t seq_len_=20;
    CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
    bool has_mrope_=true,has_qk_norm_=false,linear_acc32_=false,adarms_=false;
    bool pre_layers_residual1_ready_=true,partial_mrope_active_=true;
    bool wall_action_weight_residency_=false,wall_action_kv_owner_fanout_=false;
    bool qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_dry_prepared_=false;
    bool qwen3vl_multiview_text_composite_prepared_=false,qwen3vl_multiview_text_composite_dispatch_=false;
    int qwen3vl_pooler_z1_retained_deepstack_count_=0;
    std::vector<int64_t> deepstack_lang_layers_;
    int hidden_size() const { return 1024; }
    int intermediate_size() const { return 2048; }
    int num_q_heads() const { return 16; }
    int num_kv_heads() const { return 8; }
    int head_dim() const { return 128; }
    int num_layers() const { return layer_weights_.size(); }
    int attn_tp() const { return 8; }
    int num_cores() const { return 8; }
    uint32_t addr(int,const char*) const { return 4096; }
    uint32_t layer_addr(int,int,const char*) const { return 4096; }
    std::vector<int64_t> cold_topology_arguments() const { return {}; }
    const Context& ctx() const { return context; }
    bool ordinary_row_weight_reuse_plan_enabled(const Plan&,bool,int64_t) const { return false; }
    template <typename... Args> bool wall_base_prefill_owner_norm_enabled(Args...) const { return false; }
    template <typename... Args> void consume_wall_base_prefill_owner_norm(Args...) const { assert(false); }
    void emit_adarms_mut_refresh_post(int) const {}
    double eps_=1e-6;
    HELPERS
    void configure(bool fp16,bool resident) {
        wall_sorting_.fp16_transfer_mask=fp16?4:0;
        wall_action_weight_residency_=resident;
        for(auto& w:layer_weights_) {
            for(auto* tensor:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})
                tensor->dtype=fp16?at::kHalf:at::kChar;
            for(auto* tensor:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws})
                tensor->present=!fp16;
        }
    }
    int64_t compact_bytes() const {
        LayoutContext ctx; ctx.chunk_size=seq_len_;
        const int64_t h=1024,DW=2;
        enum class SC { Temp }; constexpr int ALL=1;
        struct Buffer { const char* name; int64_t size,start,end; SC storage;
                        int layer; const char* alias; int scope; };
        std::vector<Buffer> d;
        auto A=[](int64_t bytes) { return (bytes+255)/256*256; };
        COMPACT_DECL
        return d.empty()?0:d.front().size;
    }
    bool build() {
        const int64_t physical_len=seq_len_; const bool partial_mrope=true;
        Plan plan; LayoutContext layout; context.manifest_.routes.clear();
        auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,
                        int64_t flags=0,std::vector<int64_t> args={},int64_t invocation=0) {
            context.manifest_.routes.push_back({site,family,selector,flags,args,invocation});
        };
        auto append_linear=[&](int64_t site,CausalDecoderLinearRoute route) {
            append(FmbRouteFamily::LINEAR,site,static_cast<int64_t>(route));
        };
        auto append_ring=[&](int64_t site,int64_t rows,int64_t invocation) {
            append(FmbRouteFamily::ALL_REDUCE,site,causal_ring_route(rows,hidden_size()),0,{},invocation);
        };
        PRODUCER_GUARD
        PRODUCER
        context.consumed_.assign(context.manifest_.routes.size(),0);
        return wall_action_linear_ring;
    }
    bool emit(int layer_idx) const {
        const auto& lw=layer_weights_[layer_idx]; ChunkInfo chunk; chunk.len=seq_len_;
        const int rows=seq_len_,h=1024,nq=16,nkv=8,hd=128,tp=8,bs=1;
        const uint32_t input_residual=4096;
        const bool residual_is_bf16=false,transition_to_bf16=false,is_last_layer=layer_idx==35;
        CONSUMER_GUARD
        CONSUMER_O
        CONSUMER_DOWN
        return wall_action_linear_ring_postnorm;
    }
};
int main() {
    for(int rows:{20,32}) for(bool fp16:{false,true}) for(bool resident:{false,true}) {
        Model model;model.seq_len_=rows;model.wall_sorting_.silu_mul=rows==32;
        model.configure(fp16,resident);assert(model.build());
        assert(model.compact_bytes()==(rows==20?6144:8192));
        uint32_t covered=0;
        for(uint32_t core=0;core<8;++core) {
            assert(wall_action_residual_row_offset(rows,core)==covered);
            const auto count=wall_action_residual_rows(rows,core);
            assert(count==(rows==20?(core<4?3:2):4));
            assert(count*1024*2<=model.compact_bytes());
            covered+=count;
        }
        assert(covered==uint32_t(rows));
        launches.clear();assert(model.emit(0));
        assert(launches.size()==1 && !launches[0].local && launches[0].weight==0 && launches[0].rows==rows);
        for(int layer=1;layer<36;++layer) {
            launches.clear();assert(model.emit(layer));assert(launches.size()==2);
            const bool expected=resident && layer<= (fp16?2:4);
            for(const auto& launch:launches) assert(launch.local && bool(launch.weight)==expected && launch.rows==rows);
            if(expected) {
                const uint32_t base=4096+(layer-1)*WALL_ACTION_WEIGHT_BYTES_PER_LAYER*(fp16?2:1);
                assert(launches[0].weight==base+WALL_ACTION_O_OFFSET*(fp16?2:1));
                assert(launches[1].weight==base+WALL_ACTION_DOWN_OFFSET*(fp16?2:1));
                assert(launches[1].weight+0x40000*(fp16?2:1)<=4096+WALL_ACTION_WEIGHT_BYTES_PER_CORE);
            }
        }
        assert(std::all_of(model.context.consumed_.begin(),model.context.consumed_.end(),[](int x){return x==1;}));
        const auto routes=model.context.manifest_.routes;
        for(int64_t site:{CAUSAL_DECODER_PREPARE_SITE,CAUSAL_DECODER_O_SITE,CAUSAL_DECODER_REDUCE_SITE}) {
            for(bool corrupt:{false,true}) {
                model.context.manifest_.routes=routes;
                auto& bad=model.context.manifest_.routes;
                auto found=std::find_if(bad.begin(),bad.end(),[&](const auto& r){return r.site_id==site;});
                assert(found!=bad.end());
                if(corrupt) found->selector=99;else bad.erase(found);
                try { model.emit(0);assert(false); } catch(const std::runtime_error&) {}
            }
        }
        model.context.manifest_.routes=routes;model.context.complete=false;
        try { model.emit(0);assert(false); } catch(const std::runtime_error&) {}
    }
    for(bool fp16:{false,true}) {
        Model base;base.configure(fp16,false);base.seq_len_=32;
        assert(!base.build());assert(base.compact_bytes()==0);
        base.wall_sorting_.silu_mul=true;
        for(int rows:{19,21,31,33,64}) {
            base.seq_len_=rows;assert(!base.build());assert(base.compact_bytes()==0);
        }
        for(int unsupported_feature=0;unsupported_feature<2;++unsupported_feature) {
            Model invalid;invalid.configure(fp16,true);invalid.seq_len_=32;
            invalid.wall_sorting_.silu_mul=true;
            invalid.wall_action_kv_owner_fanout_=unsupported_feature==0;
            invalid.wall_sorting_.action_activation_reuse=unsupported_feature==1;
            try { invalid.build();assert(false); } catch(const std::runtime_error&) {}
        }
    }
    Model unsupported;unsupported.configure(true,false);
    unsupported.layer_weights_[7].o_ws.present=true;
    assert(!unsupported.build());
}
'''
    for name, value in {"CONSUME_ROUTE": consume_route, "DECLARATIONS": declarations,
                        "ENUMS": enums, "STUBS": stubs, "HELPERS": helpers, "COMPACT_DECL": compact,
                        "PRODUCER_GUARD": producer_guard, "PRODUCER": producer,
                        "CONSUMER_GUARD": consumer_guard, "CONSUMER_O": consumer_o,
                        "CONSUMER_DOWN": consumer_down}.items():
        program = program.replace(name, value)
    source = tmp_path / "wall_action_routes.cpp"
    source.write_text(CATALOG_INCLUDE + program)
    executable = tmp_path / "wall_action_routes"
    compiler = shutil.which("c++")
    assert compiler, "native route parity check requires a C++ compiler"
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT), str(source),
                    "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)

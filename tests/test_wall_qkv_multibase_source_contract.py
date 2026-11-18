from pi_wall_host_catalog import assert_current_catalog, CATALOG_INCLUDE
from test_siglip_attention_dry_live_contract import _function
"""Board-free contracts for the Wall fused-QKV paths."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_H = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
CACHE_INC = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
DECLS = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
LINEAR = (ROOT / "src/ops/rpu_linear.cpp").read_text()
KVCACHE = (ROOT / "src/ops/rpu_llama_kvcache.cpp").read_text()
DECODER = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
KERNEL_NAME = "qkv_parallel_w8a16_multibase_m448n64k128"
PACKED_KERNEL_NAME = "qkv_parallel_w8a16_multibase_m320x2n64k128"
OWNER_KERNEL_NAME = (
    "qkv_parallel_w8a16_multibase_tile_owner_m320x2n64k128"
)
DIRECT_KERNEL_NAME = "owner_direct_kv_m640"
ACTION_KERNEL_NAME = "qkv_parallel_w8a16_multibase_m592n32k128"
ACTION_FUSED_ROPE_KERNEL_NAME = (
    "qkv_parallel_w8a16_multibase_partial_mrope_m20n64k128"
)


def test_multibase_launcher_preserves_expansion_graph_and_multicore_contracts():
    assert_current_catalog("Qkv", "QkvTileOwner")
    assert "WALL_PREFILL_OWNER_DIRECT_KV_M640" in CACHE_H
    launcher = _function(LINEAR,"void rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(")
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 6
    assert launcher.count("GraphDdrRegisterRole::Int8ModelWeight") == 3
    assert launcher.count("GraphDdrRegisterRole::PerChannelScale") == 3
    pairs=("2, 3","20, 21","32, 33","34, 35","40, 41","42, 43")
    positions=[launcher.index("DevAddrShift8LoHi, "+pair) for pair in pairs]
    assert positions == sorted(positions)
    for contract in ("Operation::QkvTileOwner", "Operation::Qkv", "!kv_tile_owner || rows == 640",
                     "kernel->reset_regs();", "register_writer.write(*kernel);",
                     "queue->set_broadcast_mode(true);", "{selected.grid_x, selected.grid_y, 1}"):
        assert contract in launcher
    for slot,base in ((36,"k_output_spm_addr"),(44,"v_output_spm_addr"),(38,"k_bias_spm_addr"),(46,"v_bias_spm_addr")):
        assert f"set_u32({slot}, {base});" in launcher


def test_owner_direct_kv_launcher_declares_typed_operands_and_multicore_contracts():
    start = KVCACHE.index(
        "void rpu_launch_wall_prefill_owner_direct_kv_spm_kernel("
    )
    launcher = KVCACHE[
        start : KVCACHE.index(
            "void rpu_launch_insert_kcache_spm_unified_with_plan(", start
        )
    ]
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 4
    assert launcher.count("GraphDdrRegisterRole::KeyCache") == 1
    assert launcher.count("GraphDdrRegisterRole::ValueCache") == 1
    assert launcher.count("GraphDdrRegisterRole::SemanticInput") == 2
    for abi in (
        "GraphDdrRegisterEncoding::DevAddrShift8LoHi, 4, 5",
        "GraphDdrRegisterEncoding::DevAddrShift8LoHi, 6, 7",
        "GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9",
        "GraphDdrRegisterEncoding::DevAddrShift8LoHi, 10, 11",
    ):
        assert abi in launcher
    assert "const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);" in launcher
    assert "K/V SPM input ranges must not overlap" in launcher
    assert "K/V DDR cache ranges must not overlap" in launcher
    assert "kernel->set_regs(12, static_cast<uint16_t>(0));" in launcher
    assert "kernel->reset_regs();" in launcher
    assert "register_writer.write(*kernel);" in launcher
    assert "queue->set_broadcast_mode(true);" in launcher
    assert "queue->enqueu_kernel(*kernel, {40, 1, 1}, cores);" in launcher


def test_staged_prefill_writes_directly_to_unified_qkv_buffers():
    start = DECODER.index("void build_wall_p640_staged_layer_subgraph(")
    staged = DECODER[start : DECODER.index("void build_layer_subgraph(", start)]
    assert staged.count("rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(") == 2
    assert staged.count("lw.q_ws, lw.k_ws, lw.v_ws, kHalfRows,") == 1
    assert staged.count("lw.q_ws, lw.k_ws, lw.v_ws, kPackedRows,") == 1
    assert 'addr(0, "q"), addr(0, "k"), addr(0, "v")' in staged
    assert "packed_qkv ? half * kHalfQBytes : 0" in staged
    assert "packed_qkv ? half * kHalfKvBytes : 0" in staged
    assert 'addr_offset("k").value + kv_offset' in staged
    assert 'addr_offset("v").value + kv_offset' in staged
    assert 'addr_offset("q").value + q_offset' in staged

    packed_start = staged.index("// Layers 4..35 already carry")
    packed = staged[
        packed_start : staged.index("const uint32_t q_half1_off =", packed_start)
    ]
    q_rope = packed.index('"q", "q", local_q_heads, hd, kPackedRows,')
    owner_branch = packed.index("if (wall_prefill_owner_direct_kv_)")
    direct = packed.index("rpu_launch_wall_prefill_owner_direct_kv_spm_kernel(")
    fallback_k_rope = packed.index(
        '"k", "k", /*local_heads=*/1, local_kv_dim, kPackedRows,'
    )
    fallback_plan = packed.index("const KvInsertSegmentPlan kv_plan = consume_causal_kv_plan(")
    fallback_insert = packed.index("rpu_launch_insert_kvcache_spm_unified_with_plan(")
    assert q_rope < owner_branch < direct < fallback_k_rope < fallback_plan < fallback_insert
    assert 'addr_offset("k").value, addr_offset("v").value,' in packed
    assert "lw.q_ws, lw.k_ws, lw.v_ws, kPackedRows," in packed
    assert "wall_prefill_owner_direct_kv_, wall_sorting_);" in packed
    assert "partial_mrope_active_ && slot == 0" in packed
    assert "emit_qkv_cache_half(" not in packed


def test_staged0_prefill_reuses_the_same_direct_qkv_launcher():
    start = DECODER.index("void build_layer_subgraph(")
    staged0 = DECODER[start:]
    assert "const bool wall_p640_staged0_w8a16" in staged0
    assert "!linear_acc32_" in staged0
    assert "rpu_get_linear_acc32_enabled" not in staged0
    assert staged0.count("rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(") == 1
    assert "/*fuse_silu_mul=*/use_planned_silu_mul() ||" in staged0
    assert "wall_p640_staged0_w8a16 || wall_action_w8a16_silu_mul" in staged0


def test_small_prefill_native_route_rejects_incompatible_shapes_and_stale_receipts(tmp_path):
    """Run the actual native predicate and both sides of the COMPLETE QKV site."""
    from test_rmsnorm_capability_publication_host import _block, _compile_and_run

    predicate = _block(DECODER, "    bool wall_small_prefill_fusion(")
    arguments = _block(DECODER, "    std::vector<int64_t> wall_small_prefill_qkv_arguments(")
    producer = _block(DECODER, "else if (wall_small_prefill_fused) {")[5:]
    consumer = _block(DECODER, "else if (wall_p640_staged0_w8a16 || wall_small_prefill_fused ||")[5:]
    program = f'#include "{ROOT / "src/core/rpu_wall_sorting_config.h"}"\n' + r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
namespace at { enum ScalarType { kHalf, kChar, kByte }; }
struct Tensor { at::ScalarType dtype=at::kHalf; auto scalar_type() const { return dtype; } };
constexpr int NUM_CORES=8;
enum class ChunkMode { SEQUENTIAL, KV_FIRST };
enum class CausalDecoderPlanningMode { ORDINARY, WALL_Z1 };
enum class FmbRouteFamily { LINEAR };
''' + _block(DECODER, "enum class CausalDecoderLinearRoute") .replace(
        "static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE)", "1").replace(
        "static_cast<int64_t>(FmbLinearRouteSelector::GEMV)", "2").replace(
        "static_cast<int64_t>(FmbLinearRouteSelector::ROW_WEIGHT_REUSE)", "5") + r''';
constexpr int64_t CAUSAL_DECODER_Q_SITE=403971716841024546LL;
struct ChunkInfo { int64_t len=336, offset=0, idx=0; };
struct FmbThreeStageChunkPlan {
    ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;
    struct { struct { int64_t chunk_size=336; } plan;
             std::vector<ChunkInfo> chunks={ChunkInfo{}}; } compute;
};
struct Route { FmbRouteFamily family; int64_t site, selector, flags, idx; std::vector<int64_t> args; };
std::vector<Route> routes;
void append(FmbRouteFamily family, int64_t site, int64_t selector, int64_t flags,
            std::vector<int64_t> args, int64_t idx) {
    routes.push_back({family,site,selector,flags,idx,args});
}
struct Context {
    bool complete=true;
    void consume_physical_route(FmbRouteFamily family, int64_t site, int64_t selector,
            int64_t flags, std::vector<int64_t> args, int64_t idx) {
        if (!complete || routes.size()!=1) throw std::runtime_error("missing COMPLETE route");
        const auto& route=routes.front();
        if (route.family!=family || route.site!=site || route.selector!=selector ||
            route.flags!=flags || route.args!=args || route.idx!=idx)
            throw std::runtime_error("stale route");
    }
};
int64_t launched_rows=0;
void rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(uint32_t,const Tensor&,
    const Tensor&,const Tensor&,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,
    const Tensor&,const Tensor&,const Tensor&,int64_t rows,bool owner,const WallSortingConfig&) {
    assert(!owner); launched_rows=rows;
}
struct Model {
    struct LayerWeights { Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,q_ws,k_ws,v_ws; };
    std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(36);
    WallSortingConfig wall_sorting_;
    bool has_qkv_bias_=true,has_qk_norm_=false,has_mrope_=true,use_silu_=true;
    bool linear_acc32_=false,nvfp4_=false,decoder_mixed_precision_=false;
    bool wall_prefill_mixed_requested_=false,wall_prefill_staged_mlp_=false;
    bool wall_prefill_owner_direct_kv_=false,decode_gate_up_fusion_=false,adarms_=false;
    bool pre_layers_residual1_ready_=false,wall_z1_dispatch_=false;
    bool qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
    std::vector<int32_t> mrope_section_={16,24,24};
    std::vector<int64_t> deepstack_lang_layers_;
    CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
    int64_t hidden=2048,intermediate=11008,nq=16,nkv=8,hd=128,tp=8;
    int64_t hidden_size() const { return hidden; }
    int64_t intermediate_size() const { return intermediate; }
    int64_t num_layers() const { return layer_weights_.size(); }
    int64_t num_q_heads() const { return nq; }
    int64_t num_kv_heads() const { return nkv; }
    int64_t head_dim() const { return hd; }
    int attn_tp() const { return tp; }
    int mlp_tp() const { return tp; }
    int num_cores() const { return tp; }
    Context context;
    Context& ctx() { return context; }
    uint32_t addr(int,const char*) { return 256; }
    uint32_t qkv_fused_bias_addr(int,const char*) { return 512; }
''' + predicate + arguments + r'''
    void produce(const FmbThreeStageChunkPlan& plan,int64_t physical_len) {
        const bool wall_small_prefill_fused=wall_small_prefill_fusion(plan,physical_len,0,true,false,1);
''' + producer + r'''
    }
    void emit(const FmbThreeStageChunkPlan& plan,int64_t rows) {
        const bool wall_small_prefill_fused=wall_small_prefill_fusion(plan,rows,0,true,false,1);
        const bool wall_p640_staged0_w8a16=false, wall_long_prefill_qkv=false, wall_fp16_long_prefill_fused=false;
        const auto& lw=layer_weights_.front(); const auto& chunk=plan.compute.chunks.front();
        const int layer_idx=0;
''' + consumer + r'''
    }
};
int main() {
    for (int64_t rows:{336,384}) for (bool fp16:{false,true}) {
        Model model; FmbThreeStageChunkPlan plan;
        plan.compute.plan.chunk_size=rows; plan.compute.chunks[0].len=rows;
        model.wall_sorting_.silu_mul=true; model.wall_sorting_.fp16_transfer_mask=fp16?2:0;
        for(auto& lw:model.layer_weights_) {
            auto dtype=fp16?at::kHalf:at::kChar;
            lw.q_w.dtype=lw.k_w.dtype=lw.v_w.dtype=lw.o_w.dtype=dtype;
            lw.gate_w.dtype=lw.up_w.dtype=lw.down_w.dtype=dtype;
        }
        assert(model.wall_small_prefill_fusion(plan,rows,0,true,false,1));
        routes.clear();launched_rows=0;model.produce(plan,rows);model.emit(plan,rows);
        assert(routes.size()==1 && launched_rows==rows);
        const auto selected=rpu_pi_wall::select(rpu_pi_wall::Operation::Qkv,
            fp16?rpu_pl_tiling::WeightType::Fp16:rpu_pl_tiling::WeightType::Int8,false,rows,256,2048);
        assert((routes[0].args==std::vector<int64_t>{rows,2048,2048,1024,8,
            selected.tile.m_tile,selected.tile.n_tile,selected.tile.k_tile,fp16?16:8,
            selected.grid_x,selected.grid_y,2,32,40,20,34,42,4,36,44,6,38,46}));
        routes[0].args[0]+=16;launched_rows=0;
        bool rejected=false;try { model.emit(plan,rows); } catch(const std::runtime_error&) { rejected=true; }
        assert(rejected && launched_rows==0);
        routes.clear();model.produce(plan,rows);model.context.complete=false;
        rejected=false;try { model.emit(plan,rows); } catch(const std::runtime_error&) { rejected=true; }
        assert(rejected && launched_rows==0);
        for (auto mutate:std::vector<std::function<void(Model&)>>{
            [](Model& m){m.wall_sorting_.silu_mul=false;},
            [](Model& m){m.decoder_mixed_precision_=true;},
            [](Model& m){m.wall_prefill_mixed_requested_=true;},
            [](Model& m){m.wall_prefill_staged_mlp_=true;},
            [](Model& m){m.wall_prefill_owner_direct_kv_=true;},
            [](Model& m){m.linear_acc32_=true;},
            [](Model& m){m.has_qkv_bias_=false;},
            [](Model& m){m.has_qk_norm_=true;},
            [](Model& m){m.mrope_section_={24,20,20};},
            [](Model& m){m.hidden=1024;},[](Model& m){m.intermediate=2048;},
            [](Model& m){m.nkv=2;},[](Model& m){m.tp=4;},
            [](Model& m){m.layer_weights_.pop_back();},
            [](Model& m){m.layer_weights_[2].q_w.dtype=at::kByte;},
            [](Model& m){m.layer_weights_[2].down_w.dtype=at::kByte;},
            [](Model& m){m.deepstack_lang_layers_={0};},
            [](Model& m){m.pre_layers_residual1_ready_=true;}}) {
            auto bad=model;mutate(bad);assert(!bad.wall_small_prefill_fusion(plan,rows,0,true,false,1));
        }
        if(fp16) {auto bad=model;bad.wall_sorting_.fp16_transfer_mask=4;
            assert(!bad.wall_small_prefill_fusion(plan,rows,0,true,false,1));}
        assert(!model.wall_small_prefill_fusion(plan,rows,1,true,false,1));
        assert(!model.wall_small_prefill_fusion(plan,rows,0,false,false,1));
        assert(!model.wall_small_prefill_fusion(plan,rows,0,true,true,1));
        assert(!model.wall_small_prefill_fusion(plan,rows,0,true,false,2));
        for(int64_t other:{320,335,337,383,385,640,672}) {
            auto bad=plan;bad.compute.plan.chunk_size=other;bad.compute.chunks[0].len=other;
            assert(!model.wall_small_prefill_fusion(bad,other,0,true,false,1));
        }
        auto bad=plan;bad.compute.chunks.push_back(plan.compute.chunks[0]);
        assert(!model.wall_small_prefill_fusion(bad,rows,0,true,false,1));
        bad=plan;bad.compute.chunks[0].offset=16;
        assert(!model.wall_small_prefill_fusion(bad,rows,0,true,false,1));
        bad=plan;bad.compute.chunks[0].len-=16;
        assert(!model.wall_small_prefill_fusion(bad,rows,0,true,false,1));
        bad=plan;bad.compute.plan.chunk_size-=16;
        assert(!model.wall_small_prefill_fusion(bad,rows,0,true,false,1));
        bad=plan;bad.chunk_mode=ChunkMode::KV_FIRST;
        assert(!model.wall_small_prefill_fusion(bad,rows,0,true,false,1));
    }
}
'''
    _compile_and_run(tmp_path, "wall_small_prefill_route", CATALOG_INCLUDE + program)


def test_long_prefill_actual_cold_proof_and_complete_routes(tmp_path):
    """Execute production admission and COMPLETE endpoints for both chunk invocations."""
    import re

    from test_wall_base_nvfp4_qkv_bias import META
    from test_wall_base_paired_kv import block, compile_run

    helpers = "\n".join(block(DECODER, marker) for marker in (
        "    bool wall_small_prefill_fusion(",
        "    bool wall_base_prefill_shared_input_enabled(",
        "    bool wall_base_long_prefill_qkv_fusion(",
        "    std::vector<int64_t> wall_small_prefill_qkv_arguments(",
    ))
    start = DECODER.index("        wall_base_prefill_shared_input_ =")
    cold = DECODER[start:DECODER.index("        qwen3vl_4b_decode_gateup_fused_ =", start)]
    producer = block(DECODER, "else if (wall_long_prefill_qkv || wall_fp16_long_prefill_fused) {")[5:]
    consumer = block(DECODER, "else if (wall_p640_staged0_w8a16 || wall_small_prefill_fused ||")[5:]
    admission = re.findall(r"const bool wall_long_prefill_qkv =[^;]+;", DECODER)
    assert len(admission) == 2
    planner = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    find = block(planner, "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(").replace(
        "FmbPhysicalManifestConsumer::find_route", "find_route")
    consume = block(planner, "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(").replace(
        "FmbPhysicalManifestConsumer::consume_route", "consume_physical_route")
    route_enum = block(DECODER, "enum class CausalDecoderLinearRoute").replace(
        "static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE)", "1").replace(
        "static_cast<int64_t>(FmbLinearRouteSelector::GEMV)", "2").replace(
        "static_cast<int64_t>(FmbLinearRouteSelector::ROW_WEIGHT_REUSE)", "5") + ";"
    q_site = next(line for line in DECODER.splitlines()
                  if line.startswith("inline constexpr int64_t CAUSAL_DECODER_Q_SITE ="))
    source = CATALOG_INCLUDE + META + r'''
#include <tuple>
#include <functional>
enum class ChunkMode{SEQUENTIAL,KV_FIRST};
enum class CausalDecoderPlanningMode{ORDINARY,WALL_Z1};
ROUTE_ENUM
Q_SITE_DECLARATION
enum class KernelId:uint16_t{};
KernelId pi_wall_kernel_id(std::size_t ordinal){return static_cast<KernelId>(ordinal);}
struct KernelCache{
 bool base=true,silu=true,no_up=true,qkv=true;
 static auto& instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id)const{
  for(const auto& p:rpu_pi_wall::profiles) if(p.ordinal==static_cast<std::size_t>(id)) {
   switch(p.operation){case rpu_pi_wall::Operation::SharedRaw:return base;
    case rpu_pi_wall::Operation::SharedSiluUp:return silu;
    case rpu_pi_wall::Operation::SharedSilu:return no_up;
    case rpu_pi_wall::Operation::Qkv:return qkv;default:return false;}
  }return false;
 }
};
inline bool rpu_pi_wall_operation_loaded(rpu_pi_wall::Operation operation,
                                         rpu_pl_tiling::WeightType weight) {
    bool found = false;
    for (const auto& p : rpu_pi_wall::profiles) {
        if (p.operation != operation || p.weight != weight) continue;
        found = true;
        if (!KernelCache::instance().has_loaded(pi_wall_kernel_id(p.ordinal)))
            return false;
    }
    return found;
}
struct ChunkInfo{int64_t idx,offset,len,kv_seq_len;};
struct Stage{struct{int64_t chunk_size=0,num_chunks=0;}plan;std::vector<ChunkInfo>chunks;};
struct FmbThreeStageChunkPlan{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;Stage input,qkv,compute;};
enum class FmbRouteFamily{ATTENTION,LINEAR,ACTIVATION,ALL_REDUCE,COLLECTIVE};
struct FmbRouteManifestEntry{int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>arguments;int64_t invocation;};
struct Manifest{std::vector<FmbRouteManifestEntry>routes;};
struct Context{
 Manifest manifest_;mutable std::vector<int>consumed_;bool complete_flag=true,is_causal=true;
 int64_t position=0,batch_size=1;std::optional<int>attention_mask;FmbThreeStageChunkPlan stage_plan;
 bool complete()const{return complete_flag;}const Manifest& manifest()const{return manifest_;}
 FIND
 CONSUME
};
struct Launch{int layer;int64_t rows;};std::vector<Launch> launches;
void rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(uint32_t x,const at::Tensor& qw,
 const at::Tensor& kw,const at::Tensor& vw,uint32_t q,uint32_t k,uint32_t v,
 uint32_t qb,uint32_t kb,uint32_t vb,const at::Tensor& qs,const at::Tensor& ks,
 const at::Tensor& vs,int64_t rows,bool owner,const WallSortingConfig& sorting){
 int layer=qw.address-100;
 assert(layer>=0&&layer<36&&kw.address==200+layer&&vw.address==300+layer);
 assert(qs.address==400+layer&&ks.address==500+layer&&vs.address==600+layer);
 assert(qb==1000+layer&&kb==2000+layer&&vb==3000+layer);
 assert(x==1&&q==2&&k==3&&v==4&&!owner&&sorting.silu_mul);
 launches.push_back({layer,rows});
}
struct Model{
 struct LayerWeights{at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,q_ws,k_ws,v_ws;};
 std::vector<LayerWeights>layer_weights_=std::vector<LayerWeights>(36);
 WallSortingConfig wall_sorting_;
 bool has_qkv_bias_=true,has_qk_norm_=false,has_mrope_=true,use_silu_=true;
 bool linear_acc32_=false,nvfp4_=false,decoder_mixed_precision_=false;
 bool wall_prefill_mixed_requested_=false,wall_prefill_staged_mlp_=false,wall_prefill_owner_direct_kv_=false,decode_gate_up_fusion_=false,adarms_=false;
 bool pre_layers_residual1_ready_=false,wall_z1_dispatch_=false,qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
 bool wall_base_prefill_shared_input_=false,wall_base_prefill_shared_silu_=false,wall_base_prefill_no_up_store_=false,wall_base_prefill_long_qkv_=false,scale=true;
 std::vector<int32_t>mrope_section_={16,24,24};std::vector<int64_t>deepstack_lang_layers_;
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 int64_t hidden=2048,intermediate=11008,nq=16,nkv=8,hd=128,tp=8,seq_len_=448;
 int64_t hidden_size()const{return hidden;}int64_t intermediate_size()const{return intermediate;}
 int64_t num_layers()const{return layer_weights_.size();}int64_t num_q_heads()const{return nq;}
 int64_t num_kv_heads()const{return nkv;}int64_t head_dim()const{return hd;}
 int attn_tp()const{return tp;}int mlp_tp()const{return tp;}int num_cores()const{return tp;}
 Context context;Context& ctx(){return context;}
 uint32_t addr(int,std::string s){return s=="input_norm"?1:s=="q"?2:s=="k"?3:4;}
 uint32_t qkv_fused_bias_addr(int layer,std::string s){return (s=="q_bias"?1000:s=="k_bias"?2000:3000)+layer;}
 HELPERS
 void bind(){int64_t N=num_layers(),hidden_size=hidden,intermediate_size=intermediate,num_q_heads=nq,num_kv_heads=nkv,head_dim=hd;
  bool has_scale=scale,has_nvfp4=nvfp4_,use_silu=use_silu_;
  std::vector<int64_t>mrope_section(mrope_section_.begin(),mrope_section_.end()),deepstack_lang_layers=deepstack_lang_layers_;
  std::vector<at::Tensor>q_w_list,k_w_list,v_w_list,o_w_list,gate_list,up_list,down_list;
  for(auto& lw:layer_weights_){q_w_list.push_back(lw.q_w);k_w_list.push_back(lw.k_w);v_w_list.push_back(lw.v_w);o_w_list.push_back(lw.o_w);gate_list.push_back(lw.gate_w);up_list.push_back(lw.up_w);down_list.push_back(lw.down_w);}
  COLD
 }
 Model(){wall_sorting_.silu_mul=true;for(int i=0;i<36;++i){auto& lw=layer_weights_[i];
  for(auto* t:{&lw.q_w,&lw.k_w,&lw.v_w,&lw.o_w,&lw.gate_w,&lw.up_w,&lw.down_w})t->dtype=at::kChar;
  lw.q_w.address=100+i;lw.k_w.address=200+i;lw.v_w.address=300+i;
  lw.q_ws.address=400+i;lw.k_ws.address=500+i;lw.v_ws.address=600+i;}set_rows(448);bind();}
 void set_rows(int64_t p){seq_len_=p;auto& plan=context.stage_plan;
  plan.input.plan={p,1};plan.input.chunks={{0,0,p,p}};
  plan.qkv.plan=plan.compute.plan={p/2,2};plan.qkv.chunks=plan.compute.chunks={{0,0,p/2,p/2},{1,p/2,p/2,p}};}
 bool enabled(){return wall_base_long_prefill_qkv_fusion(context.stage_plan,seq_len_,context.position,context.is_causal,context.attention_mask.has_value(),context.batch_size);}
 void build(){const auto& plan=context.stage_plan;int64_t physical_len=seq_len_,position=context.position;
  struct{bool is_causal,use_attn_mask;int64_t batch_size;}layout{context.is_causal,context.attention_mask.has_value(),context.batch_size};
  const bool wall_fp16_long_prefill_fused=false;
  PRODUCER_ADMISSION
  context.manifest_.routes.clear();
  auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>args,int64_t invocation){context.manifest_.routes.push_back({site,family,selector,flags,args,invocation});};
  PRODUCER
  context.consumed_.assign(context.manifest_.routes.size(),0);
 }
 void emit(int invocation=0,int layer_idx=0){const int64_t bs=context.batch_size;
  const auto& chunk=context.stage_plan.compute.chunks.at(invocation);const int64_t rows=chunk.len;
  const auto& lw=layer_weights_.at(layer_idx);
  const bool wall_small_prefill_fused=false,wall_p640_staged0_w8a16=false,wall_fp16_long_prefill_fused=false;
  CONSUMER_ADMISSION
  CONSUMER
 }
};
template<class F>void rejects(F f){auto before=launches.size();bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed&&before==launches.size());}
void fallback(Model& m){assert(!m.enabled());m.build();assert(m.context.manifest_.routes.empty());
 auto before=launches.size();m.emit();assert(before==launches.size());}
int main(){
 for(int64_t p:{448,512,640}){
  Model m;m.set_rows(p);assert(m.enabled());m.build();const auto valid=m.context.manifest_.routes;
  assert(valid.size()==2);const int64_t rows=p/2;
  for(int c=0;c<2;++c){const auto& r=valid[c];assert(r.family==FmbRouteFamily::LINEAR&&r.site_id==CAUSAL_DECODER_Q_SITE);
   assert(r.selector==int64_t(CausalDecoderLinearRoute::WALL_QKV_MULTIBASE)&&r.flags==0&&r.invocation==c);
   const auto selected=rpu_pi_wall::select(rpu_pi_wall::Operation::Qkv,rpu_pl_tiling::WeightType::Int8,false,rows,256,2048);
   assert((r.arguments==std::vector<int64_t>{rows,2048,2048,1024,8,selected.tile.m_tile,
    selected.tile.n_tile,selected.tile.k_tile,8,selected.grid_x,selected.grid_y,2,32,40,20,34,42,4,36,44,6,38,46}));}
  launches.clear();for(int c=0;c<2;++c)for(int l=0;l<36;++l)m.emit(c,l);
  assert(launches.size()==72);for(int i=0;i<72;++i)assert(launches[i].layer==i%36&&launches[i].rows==rows);
  assert(m.context.consumed_==std::vector<int>({1,1}));
  for(int c=0;c<2;++c){
   for(size_t a=0;a<valid[c].arguments.size();++a){m.context.manifest_.routes=valid;++m.context.manifest_.routes[c].arguments[a];rejects([&]{m.emit(c,35);});}
   for(int field=0;field<7;++field){m.context.manifest_.routes=valid;auto& r=m.context.manifest_.routes[c];
    switch(field){case 0:++r.selector;break;case 1:++r.flags;break;case 2:r.invocation=2;break;
     case 3:++r.site_id;break;case 4:r.family=FmbRouteFamily::ATTENTION;break;
     case 5:r.arguments.push_back(0);break;case 6:m.context.manifest_.routes.erase(m.context.manifest_.routes.begin()+c);break;}
    rejects([&]{m.emit(c);});}
   m.context.manifest_.routes=valid;m.context.complete_flag=false;rejects([&]{m.emit(c);});m.context.complete_flag=true;
  }
  // Actual input is one full chunk; QKV and compute must agree on both equal chunks.
  for(int stage=0;stage<3;++stage)for(int bad=0;bad<13;++bad){Model x;x.set_rows(p);
   auto& plan=x.context.stage_plan;Stage& s=stage==0?plan.input:stage==1?plan.qkv:plan.compute;
   switch(bad){case 0:++s.plan.chunk_size;break;case 1:++s.plan.num_chunks;break;
    case 2:s.chunks.pop_back();break;case 3:s.chunks.push_back(s.chunks.back());break;
    case 4:++s.chunks.front().idx;break;case 5:++s.chunks.front().offset;break;
    case 6:--s.chunks.front().len;break;case 7:--s.chunks.front().kv_seq_len;break;
    case 8:++s.chunks.back().idx;break;case 9:--s.chunks.back().offset;break;
    case 10:++s.chunks.back().len;break;case 11:++s.chunks.back().kv_seq_len;break;
    case 12:plan.chunk_mode=ChunkMode::KV_FIRST;break;}
   assert(!x.enabled());x.build();assert(x.context.manifest_.routes.empty());
  }
 }
 // The actual cold binding requires every marker, including the QKV payload.
 for(int mask=0;mask<16;++mask){auto& k=KernelCache::instance();k.base=mask&1;k.silu=mask&2;k.no_up=mask&4;k.qkv=mask&8;
  for(int64_t p:{448,512,640}){Model x;x.set_rows(p);assert(x.enabled()==(mask==15));
   assert(x.wall_base_prefill_long_qkv_==(mask==15));if(mask!=15)fallback(x);}}
 KernelCache::instance()=KernelCache{};
 // Uniform W8 proof visits every projection in every layer, rather than checking layer zero.
 for(int l=0;l<36;++l)for(auto member:{&Model::LayerWeights::q_w,&Model::LayerWeights::k_w,&Model::LayerWeights::v_w,
     &Model::LayerWeights::o_w,&Model::LayerWeights::gate_w,&Model::LayerWeights::up_w,&Model::LayerWeights::down_w}){
  Model x;(x.layer_weights_[l].*member).dtype=at::kHalf;x.bind();fallback(x);
 }
 for(auto flag:{&Model::wall_base_prefill_shared_input_,&Model::wall_base_prefill_no_up_store_,&Model::wall_base_prefill_long_qkv_}){
  Model x;x.*flag=false;fallback(x);}
 for(bool Model::* flag:{&Model::linear_acc32_,&Model::nvfp4_,&Model::decoder_mixed_precision_,
     &Model::wall_prefill_mixed_requested_,&Model::wall_prefill_staged_mlp_,&Model::wall_prefill_owner_direct_kv_,
     &Model::decode_gate_up_fusion_,&Model::adarms_,&Model::pre_layers_residual1_ready_,&Model::wall_z1_dispatch_,
     &Model::qwen3vl_pooler_z1_dispatch_,&Model::qwen3vl_multiview_text_composite_dispatch_,&Model::has_qk_norm_}){
  Model x;x.*flag=true;x.bind();fallback(x);}
 for(int64_t Model::* dim:{&Model::hidden,&Model::intermediate,&Model::nq,&Model::nkv,&Model::hd,&Model::tp}){
  Model x;++(x.*dim);x.bind();fallback(x);}
 for(int bad=0;bad<15;++bad){Model x;switch(bad){
  case 0:x.context.position=1;break;case 1:x.context.attention_mask=1;break;case 2:x.context.batch_size=2;break;
  case 3:x.context.is_causal=false;break;case 4:x.wall_sorting_.silu_mul=false;break;case 5:x.has_mrope_=false;break;
  case 6:x.scale=false;break;case 7:x.use_silu_=false;break;case 8:x.has_qkv_bias_=false;break;
  case 9:x.mrope_section_[1]=16;break;case 10:x.deepstack_lang_layers_.push_back(0);break;
  case 11:x.layer_weights_.pop_back();break;case 12:x.planning_mode_=CausalDecoderPlanningMode::WALL_Z1;break;
  case 13:for(auto& l:x.layer_weights_)for(auto* t:{&l.q_w,&l.k_w,&l.v_w,&l.o_w,&l.gate_w,&l.up_w,&l.down_w})t->dtype=at::kHalf;x.wall_sorting_.fp16_transfer_mask=2;break;
  case 14:x.wall_sorting_.fp16_transfer_mask=2;x.layer_weights_[35].v_w.dtype=at::kByte;break;
 }x.bind();fallback(x);}
 for(int64_t p:{32,224,256,320,336,384,416,447,449,480,511,513,576,639,641,672}){Model x;x.set_rows(p);fallback(x);}
}
'''
    for key, value in (("HELPERS", helpers), ("COLD", cold),
                       ("PRODUCER_ADMISSION", admission[0]), ("CONSUMER_ADMISSION", admission[1]),
                       ("PRODUCER", producer), ("CONSUMER", consumer),
                       ("FIND", find), ("CONSUME", consume), ("ROUTE_ENUM", route_enum), ("Q_SITE_DECLARATION", q_site)):
        source = source.replace(key, value)
    compile_run(tmp_path, "wall_long_prefill_qkv_routes", source)


def test_actual_qkv_leaf_row_admission_preserves_fp16_and_legacy_bounds(tmp_path):
    """Compile the actual leaf's early row/precision guards, before packet setup."""
    from test_wall_base_nvfp4_qkv_bias import META
    from test_wall_base_paired_kv import block, compile_run

    leaf = block(LINEAR, "void rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(")
    # Keep the production signature and every admission statement verbatim.
    # Tensor metadata/packet ABI remain covered by the existing launcher contract.
    admission = leaf[:leaf.index("  auto check_weight =")] + "\n}\n"
    source = META + admission + r'''
int main(){
 for(bool fp16:{false,true})for(bool silu:{false,true})for(bool transfer:{false,true})
 for(bool owner:{false,true})for(int rows:{0,32,208,223,224,225,240,255,256,257,304,320,336,352,384,400,448,512,640,672}){
  at::Tensor w,s;w.dtype=fp16?at::kHalf:at::kChar;WallSortingConfig sorting;
  sorting.silu_mul=silu;sorting.fp16_transfer_mask=transfer?2:0;
  bool expected=false;
  switch(rows){case 320:case 640:expected=true;break;
   case 336:case 384:expected=silu;break;
   case 224:case 256:expected=silu;break;}
  expected=expected&&(!fp16||transfer)&&(!owner||rows==640);
  bool accepted=true;try{rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(
   1,w,w,w,2,3,4,5,6,7,s,s,s,rows,owner,sorting);}catch(const std::exception&){accepted=false;}
  assert(accepted==expected);
 }
}
'''
    compile_run(tmp_path, "wall_long_prefill_qkv_leaf_rows", source)


def test_native_qkv_fixture_forwards_cold_config_to_actual_leaf(tmp_path):
    """Connect the real conformance branches to production admission guards."""
    from test_wall_base_nvfp4_qkv_bias import META
    from test_wall_base_paired_kv import block, compile_run

    harness = (ROOT / "python/rpu_backend/tests/conformance/_cpp/"
               "test_production_linear_launchers.cpp").read_text()
    fp16 = block(harness, 'else if (profile == "wall_qkv_pair_fp16")')
    w8 = block(harness, 'else if (profile == "wall_qkv_w8")')
    dispatch = fp16.removeprefix("else ") + " " + w8
    leaf = block(LINEAR, "void rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(")
    admission = leaf[:leaf.index("  auto check_weight =")] + "\n}\n"
    source = META + "\n#include <iostream>\n" + admission + r'''
void call_fixture(const std::string& profile, int64_t rows, int mode) {
 std::vector<at::Tensor> owners(6);
 for(auto& owner:owners) owner.dtype=profile=="wall_qkv_pair_fp16"?at::kHalf:at::kChar;
 auto w=[&](int index)->const at::Tensor& {return owners.at(index);};
 auto s=[](int index){return uint32_t(0x20000000+index*4096);};
 DISPATCH
 else throw std::runtime_error("unexpected fixture profile");
}
int main() {
 struct Case {const char* name;const char* profile;int rows;int mode;};
 const Case cases[] = {
  {"wall_qkv_w8","wall_qkv_w8",320,0},
  {"wall_qkv_pair_p118","wall_qkv_pair_fp16",640,1},
  {"wall_qkv_pair_p136","wall_qkv_pair_fp16",640,0},
  {"wall_qkv_remaining_p131","wall_qkv_pair_fp16",224,0},
  {"wall_qkv_remaining_p132","wall_qkv_pair_fp16",256,0},
  {"wall_qkv_remaining_p133","wall_qkv_pair_fp16",320,0},
  {"wall_qkv_remaining_p134","wall_qkv_pair_fp16",336,0},
  {"wall_qkv_remaining_p135","wall_qkv_pair_fp16",384,0},
  {"wall_qkv_remaining_p138","wall_qkv_w8",224,0},
  {"wall_qkv_remaining_p139","wall_qkv_w8",256,0},
  {"wall_qkv_remaining_p143","wall_qkv_w8",640,0},
 };
 int failures=0;
 for(const auto& c:cases) {
  try {call_fixture(c.profile,c.rows,c.mode);}
  catch(const std::exception& error) {
   ++failures;std::cerr<<"rejected "<<c.name<<": "<<error.what()<<"\n";
  }
 }
 std::cerr<<"accepted "<<11-failures<<"/11 fixture calls\n";
 for(const auto* profile:{"wall_qkv_pair_fp16","wall_qkv_w8"})
 for(int rows:{0,32,223,225,255,257,335,337,383,385,639,641}) {
  bool rejected=false;
  try {call_fixture(profile,rows,0);} catch(const std::exception&) {rejected=true;}
  assert(rejected);
 }
 for(int rows:{224,256,320,336,384}) {
  bool rejected=false;
  try {call_fixture("wall_qkv_pair_fp16",rows,1);}
  catch(const std::exception&) {rejected=true;}
  assert(rejected);
 }
 assert(failures==0);
}
'''
    compile_run(tmp_path, "wall_qkv_fixture_cold_config", source.replace("DISPATCH", dispatch))


def test_exact_action_qkv_reuses_existing_spm_buffers_and_falls_back_safely():
    assert_current_catalog("Qkv")
    launcher=_function(LINEAR,"void rpu_launch_wall_action_qkv_w8a16_multibase_spm_kernel(")
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 8
    assert launcher.count("GraphDdrRegisterRole::Int8ModelWeight") == 3
    assert launcher.count("GraphDdrRegisterRole::PerChannelScale") == 3
    assert launcher.count("GraphDdrRegisterRole::SemanticInput") == 2
    for pair in ("2, 3","20, 21","32, 33","34, 35","40, 41","42, 43","48, 49","50, 51"):
        assert "DevAddrShift8LoHi, "+pair in launcher
    for guard in ("wall_action_fused_rows_supported(rows, sorting)","kK = 1024", "kQ = 2048",
                  "kKv = 1024", "reg = 128; reg <= 255", "rpu_wall_action_qkv::grid(kernel_id)",
                  "plain.grid_x, plain.grid_y, 1", "queue->set_broadcast_mode(true);"):
        assert guard in launcher

    start = DECODER.index("void build_layer_subgraph(")
    action = DECODER[start:]
    assert "const bool wall_action_w8a16_qkv" in action
    assert "const bool wall_action_fused_qkv_partial_mrope" in action
    assert "!linear_acc32_" in action
    assert "rpu_get_linear_acc32_enabled" not in action
    assert action.count(
        "rpu_launch_wall_action_qkv_w8a16_multibase_spm_kernel("
    ) == 1
    assert 'addr(0, "q"), addr(0, "k"), addr(0, "v")' in action
    assert "? &cos_il_keepalive_ : nullptr" in action
    assert "? &sin_il_keepalive_ : nullptr" in action
    from test_rmsnorm_capability_publication_host import _block

    rope_fallback = _block(
        action, "else if (!wall_base_nvfp4_qkv_rope && !wall_action_fused_qkv_partial_mrope) {")
    assert rope_fallback.count("emit_qk_rope(") == 2
    assert 'emit_qk_rope("q", "q"' in rope_fallback
    assert 'emit_qk_rope("k", "k"' in rope_fallback

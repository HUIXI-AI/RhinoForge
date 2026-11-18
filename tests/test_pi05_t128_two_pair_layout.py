"""Run Gemma's real buffer declaration and allocator for paired text buckets.

Only hardware callbacks and profile eligibility (covered by the pair admission
test) are replaced. Allocation, aliases, descriptor methods, and their byte
geometry are production code; this does not initialize an device or load weights.
"""
from pathlib import Path
import re

from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def _declaration_fixture_code():
    """Shared real Gemma declarations/allocator with hardware-only callbacks stubbed."""
    source = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    methods = "\n".join(_function(source, "    " + signature) for signature in (
        "std::vector<BufferDecl> declare_buffers(",
        "int64_t pi05_pair_rows(", "int64_t pi05_prefix_rows(",
        "int64_t pi05_pair_slab_bytes(", "int64_t pi05_pair_owner_bytes(",
        "int64_t pi05_pair_owner_elements(", "int64_t pi05_pair_mask_bytes(",
        "int64_t pi05_pair_tile_rows(", "bool pi05_pair_single_compact(", "bool pi05_pair_slab_compact(",
        "int64_t pi05_pair_spill_elements(",
        "int64_t pi05_pair_compact_root_bytes(", "bool pi05_pair_scratch_mask(",
        "int64_t pi05_pair_mask_selector(", "bool pi05_pair_mask_upload(",
        "std::vector<int64_t> pi05_pair_input_compact_arguments(",
        "uint32_t pi05_pair_input_compact_address(",
        "std::vector<int64_t> pi05_pair_spill_arguments(",
        "int64_t pi05_a8_bytes(", "int64_t pi05_a8_root_bytes(", "int64_t pi05_a8_s0_bytes(",
        "bool pi05_prefill_a8_short(",
        "std::vector<int64_t> pi05_prefill_gateup_pair_arguments(",
        "std::vector<int64_t> pi05_o_pair_arguments(",
        "std::vector<int64_t> pi05_o_pair_norm_arguments(",
        "std::vector<int64_t> pi05_o_pair_companion_arguments(",
        "std::vector<FmbRouteManifestEntry> pi05_o_pair_routes(",
        "std::vector<int64_t> pi05_owner_norm_arguments(",
        "bool pi05_down_a8_enabled(", "std::vector<int64_t> pi05_down_a8_arguments(",
        "std::vector<int64_t> pi05_down_pair_arguments(",
    )).replace(" override", "")
    sites = set(re.findall(r"\bPI05_\w+_SITE\b", methods))
    constants = "\n".join(re.search(rf"^constexpr int64_t {name} = .+;$", source, re.M)[0]
                          for name in sorted(sites))
    planning = "\n".join(_function(base, signature) + (";" if signature.startswith("struct ") else "")
                         for signature in (
                             "static size_t align_spm_bytes(", "struct CpuTemporaryLayoutPlan",
                             "CpuTemporaryLayoutPlan plan_cpu_temporary_layout("))
    code = r'''
#include "core/fused_model_base.h"
#include "core/rpu_spm_allocator.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <unordered_map>
using namespace v3;
''' + _function(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(") + "\n" + planning + "\n" + constants + r'''
template<class T,class U> T Align(T n,U a){return (n+a-1)/a*a;}
template<class T> T CeilDiv(T n,T a){return (n+a-1)/a;}
constexpr int64_t DWIDTH=2;
// These callbacks are retained by the real declaration, but cannot issue DMA.
enum class ValuOpType {ADD};
void rpu_launch_ddr_broadcast_spm_dma(const c10::Half*,int64_t,uint32_t,int){assert(false);}
void rpu_launch_eltwise_binary_scalar_spm_kernel(uint32_t,c10::Half,uint32_t,int64_t,ValuOpType,int){assert(false);}
struct SdpaConfig {};
int64_t sdpa_compute_tmp_v16_size(SdpaConfig,int64_t){return 0;}
struct Owner {
    int64_t pi05_pair_rows_=320;
    bool pi05_prefill_a8_panel512_=false;
    bool pi05_full_chunk_=false;
    std::vector<BufferDecl> pi05_full_chunk_decls(const LayoutContext&, std::vector<BufferDecl>) { assert(false); return {}; }
    bool pi05_prefill_a8_=false,fp16=false;
    bool pi05_prefill_o_pair_=true,pi05_prefill_gateup_pair_=true;
    bool pi05_prefill_mask_resident_=true,pi05_prefill_owner_norm_=true;
    struct Weights {at::Tensor input_norm_w,post_attn_norm_w;};
    std::vector<Weights> layer_weights_=std::vector<Weights>(18);
    at::Tensor final_norm_w_;
    std::unordered_map<std::string,uint32_t> offsets;
    int64_t hidden_size()const{return 2048;}
    int64_t num_q_heads()const{return 8;}
    int64_t num_kv_heads()const{return 8;}
    int64_t head_dim()const{return 256;}
    int64_t intermediate_size()const{return 16384;}
    int num_layers()const{return 18;}
    int num_cores()const{return 8;}
    int attn_tp()const{return 8;}
    int mlp_tp()const{return 8;}
    bool pi05_prefill_carry_requested()const{return true;}
    bool pi05_prefill_dma_layout(const LayoutContext&)const{return true;}
    bool pi05_geglu_eligible(int64_t)const{return true;}
    bool pi05_qkv_pair_layout(const LayoutContext&)const{return false;}
    bool pi05_down_pair_layout(const LayoutContext&)const{return true;}
    bool pi05_pair_fp16()const{return fp16;}
    SdpaConfig make_sdpa_config(int)const{return {};}
    uint32_t addr(int,const char* name)const{return 0x100+offsets.at(name);}
''' + methods + "\n};\n"
    return code


def test_c320_real_declaration_plan_and_resident_mask_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    code = _declaration_fixture_code() + r'''
void verify(int64_t rows,bool a8,bool fp16=false,bool panel512=false) {
    Owner owner;owner.pi05_pair_rows_=rows;owner.pi05_prefill_a8_=a8;owner.fp16=fp16;
    owner.pi05_prefill_a8_panel512_=panel512;
    const auto gate_tile=rpu_pi_wall::select(panel512?rpu_pi_wall::Operation::A8PairedGeluPanel512:a8?rpu_pi_wall::Operation::A8PairedGelu:rpu_pi_wall::Operation::PairedGelu,fp16?rpu_pl_tiling::WeightType::Fp16:rpu_pl_tiling::WeightType::Int8,a8,rows,2048,2048);
    const auto tile=rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear,fp16?rpu_pl_tiling::WeightType::Fp16:rpu_pl_tiling::WeightType::Int8,false,rows,2048,2048);
    LayoutContext ctx;ctx.chunk_size=ctx.kv_insert_chunk_size=rows;
    ctx.max_kv_seq_len=2*rows;ctx.num_layers=18;ctx.use_attn_mask=true;ctx.is_causal=false;
    const auto decls=owner.declare_buffers(ctx);
    const auto plan=plan_cpu_temporary_layout(decls);owner.offsets=plan.offsets;
    const int64_t f=rows*4096,d=rows*512;
    const bool scratch=rows==416 || rows==432 || rows==448,single=rows==448;
    int roots=0;size_t persistent=0;
    std::unordered_map<std::string,int64_t> sizes;
    for(const auto& decl:decls) {
        if(decl.storage==StorageClass::Temp && !decl.alias_of) {
            ++roots;sizes[decl.name]=decl.size;
            assert(decl.scope==BufferScope::LayerWide && decl.phase_start==0 && decl.phase_end==30);
        } else if(decl.storage!=StorageClass::Temp) {
            assert(bool(decl.preload_callback));
            persistent+=align_spm_bytes(decl.size)*(decl.storage==StorageClass::PersistentPerLayer?decl.per_layer:1);
        }
    }
    assert(roots==(scratch?5:6) && persistent==151552);
    assert(sizes.at("residual1")== (panel512?2097152:f) && sizes.at("pi05_pair_s2")==f && sizes.at("pi05_pair_s3")==f);
    assert(sizes.at("pi05_pair_residual")==(single?d:2*d));
    const int64_t packed=Align(2*rows*2080+4*rows+30,int64_t(256));
    assert(sizes.at("pi05_pair_s1")== (a8?packed:f));
    assert(plan.offsets.at("input_norm_kv")==plan.offsets.at("pi05_pair_s1"));
    assert(plan.offsets.at("q_comp")==plan.offsets.at("pi05_pair_s2"));
    assert(plan.offsets.at("output")==plan.offsets.at("pi05_pair_s3"));
    if(scratch) assert(plan.offsets.at("sdpa_mask")==plan.offsets.at("pi05_pair_s1"));
    else {
        assert(sizes.at("sdpa_mask")==rows*rows*4);
        assert(plan.offsets.at("sdpa_mask")!=plan.offsets.at("pi05_pair_s1"));
    }
    // Independent interval oracle: every declared root is live through both
    // KVIN/COMP halves, so no two roots may overlap in the actual plan.
    for(const auto& a:sizes)for(const auto& b:sizes)if(a.first<b.first) {
        auto x=plan.offsets.at(a.first),y=plan.offsets.at(b.first);
        assert(x+a.second<=y || y+b.second<=x);
    }
    for(int layer=0;layer<18;++layer)for(int half=0;half<2;++half) {
        assert(owner.pi05_pair_mask_selector(half)==(scratch?3:half+1));
        assert(owner.pi05_pair_mask_upload(layer,half)==(scratch || half==0 || layer==0));
    }
    const auto first=owner.pi05_pair_input_compact_address(0);
    const auto second=owner.pi05_pair_input_compact_address(1);
    assert(first==owner.addr(0,"pi05_pair_residual"));
    assert(second==(single?owner.addr(0,"pi05_pair_s3")+4*d:first+d));
    bool rejected=false;
    try{owner.pi05_pair_input_compact_address(2);}catch(const c10::Error&){rejected=true;}
    assert(rejected);
    const auto routes=owner.pi05_o_pair_routes();
    const auto spill=std::count_if(routes.begin(),routes.end(),[](const auto& route){return route.site_id==PI05_O_PAIR_SPILL_SITE;});
    assert(spill==(single?2:0));
    if(panel512) {
        assert(rows==272 && a8 && !fp16);
        assert(plan.required_extent==6032640);
        // Use the same complete Pi persistent reservation as the other
        // profiles, not just the Gemma-local norm tensors.
        const size_t total=plan.required_extent+656640;
        assert(total==6689280 && total<=SpmAllocator::SPM_PLANNING_BUDGET);
        const auto gate=owner.pi05_prefill_gateup_pair_arguments();
        assert(gate.at(0)==8 && gate.at(20)==512 && gate.back()==2097152);
        assert(sizes.at("residual1")>=gate.back());
        assert(owner.pi05_a8_root_bytes()==1132800);
        std::cout<<"C272 P512 total="<<total<<" headroom="
                 <<(SpmAllocator::SPM_PLANNING_BUDGET/256*256-total)<<"\n";
    }
    if(rows==304) {
        assert(plan.required_extent==(a8?5682432:5661696));
        const size_t total=plan.required_extent+656640;
        assert(total==(a8?6339072:6318336));
        assert(total<=SpmAllocator::SPM_PLANNING_BUDGET);
        assert(owner.pi05_pair_tile_rows()==160 && !owner.pi05_pair_scratch_mask() && !owner.pi05_pair_single_compact());
        assert(owner.pi05_pair_mask_bytes()==369664 && owner.pi05_pair_compact_root_bytes()==311296);
        assert((owner.pi05_down_pair_arguments()==std::vector<int64_t>{24,304,2048,2048,8,tile.tile.n_tile,tile.grid_x,4,1,1,2,2,3,0,1,tile.tile.m_tile,tile.tile.m_tile,tile.tile.k_tile,tile.grid_y,48,50,52,26,27}));
        assert((owner.pi05_owner_norm_arguments()==std::vector<int64_t>{2,304,2048,8,38,155648,1245184,1,17,1,2,1}));
        assert(owner.pi05_o_pair_arguments().at(0)==22);
        const auto gate=owner.pi05_prefill_gateup_pair_arguments();
        if(a8) {
            assert(gate.at(0)==7 && gate.at(6)==gate_tile.tile.m_tile && gate.at(20)==256);
            assert(owner.pi05_a8_bytes()==1264640 && owner.pi05_a8_root_bytes()==1265920);
            const auto norm=owner.pi05_o_pair_norm_arguments(1);
            assert(norm.at(13)==1265248 && norm.at(13)+608+32<=packed);
            assert(256*2048*2<=f);
        } else assert(gate.at(0)==(fp16?6:5) && gate.at(6)==gate_tile.tile.m_tile);
        assert(std::count_if(routes.begin(),routes.end(),[](const auto& route){return route.site_id==PI05_O_PAIR_ROTATE_SITE;})==1);
        std::cout<<"C304 "<<(a8?"A8":"A16")<<" temp="<<plan.required_extent<<" gemma_persistent="<<persistent
                 <<" composite_persistent=656640 total="<<total<<" headroom="<<(SpmAllocator::SPM_PLANNING_BUDGET/256*256-total)<<"\n";
    }
    if(rows==432) {
        assert(plan.required_extent==(a8?7549696:7520256));
        const size_t total=plan.required_extent+656640;
        assert(total==(a8?8206336:8176896));
        assert(total<=SpmAllocator::SPM_PLANNING_BUDGET);
        assert(owner.pi05_pair_tile_rows()==112 && !owner.pi05_pair_single_compact());
        assert(owner.pi05_pair_mask_bytes()==746496 && owner.pi05_pair_compact_root_bytes()==442368);
        assert((owner.pi05_down_pair_arguments()==std::vector<int64_t>{27,432,2048,2048,8,tile.tile.n_tile,tile.grid_x,4,0,1,2,2,3,0,1,tile.tile.m_tile,3,3,tile.tile.m_tile,tile.tile.k_tile,tile.grid_y,48,50,52,26,27}));
        assert((owner.pi05_owner_norm_arguments()==std::vector<int64_t>{2,432,2048,8,54,221184,1769472,1,17,1,2,1}));
        assert(owner.pi05_o_pair_arguments().at(0)==22);
        const auto gate=owner.pi05_prefill_gateup_pair_arguments();
        if(a8) {
            assert(gate.at(0)==7 && gate.at(6)==gate_tile.tile.m_tile && gate.at(20)==384);
            assert(owner.pi05_a8_bytes()==1797120 && owner.pi05_a8_root_bytes()==1798912);
            const auto norm=owner.pi05_o_pair_norm_arguments(1);
            assert(norm.at(13)==1797984 && norm.at(13)+864+32<=packed);
            assert(384*2048*2<=f);
        } else assert(gate.at(0)==(fp16?6:5) && gate.at(6)==gate_tile.tile.m_tile);
        assert(std::count_if(routes.begin(),routes.end(),[](const auto& route){return route.site_id==PI05_O_PAIR_ROTATE_SITE;})==1);
        std::cout<<"C432 "<<(a8?"A8":"A16")<<" temp="<<plan.required_extent<<" gemma_persistent="<<persistent
                 <<" composite_persistent=656640 total="<<total<<" headroom="<<(SpmAllocator::SPM_PLANNING_BUDGET/256*256-total)<<"\n";
    }
    if(rows==304 || rows==432) {
        // Export the actual production method results for the private board
        // harness checker. This is deliberately not a complete native plan.
        const auto array=[](const std::vector<int64_t>& values){
            std::cout<<"[";for(size_t i=0;i<values.size();++i){if(i)std::cout<<",";std::cout<<values[i];}std::cout<<"]";
        };
        std::cout<<"C"<<rows<<"_ROUTE_ABI {\"precision\":\""<<(a8?"w8a8w4":(fp16?"fp16":"w8a16"))
                 <<"\",\"source\":\"production_methods_not_complete_manifest\",\"down\":";
        array(owner.pi05_down_pair_arguments());std::cout<<",\"owner\":";array(owner.pi05_owner_norm_arguments());
        std::cout<<",\"gateup\":";array(owner.pi05_prefill_gateup_pair_arguments());std::cout<<",\"o\":";array(owner.pi05_o_pair_arguments());
        std::cout<<",\"mask_selectors\":["<<owner.pi05_pair_mask_selector(0)<<","<<owner.pi05_pair_mask_selector(1)
                 <<"],\"mask_bytes\":"<<owner.pi05_pair_mask_bytes()<<",\"o_routes\":[";
        for(size_t i=0;i<routes.size();++i){
            if(i)std::cout<<",";const auto& route=routes[i];
            std::cout<<"{\"site_id\":"<<route.site_id<<",\"family\":"<<static_cast<int64_t>(route.family)
                     <<",\"selector\":"<<route.selector<<",\"flags\":"<<route.flags<<",\"invocation\":"<<route.invocation<<",\"arguments\":";
            array(route.arguments);std::cout<<"}";
        }
        std::cout<<"]}\n";
    }
    if(rows==320) {
        assert(plan.required_extent==(a8?6002176:5980160));
        // The frozen composite has 656640 B of shared persistent SPM, of
        // which the real Gemma declaration above accounts for 151552 B.
        const size_t total=plan.required_extent+656640;
        assert(total==(a8?6658816:6636800));
        assert(total<=SpmAllocator::SPM_PLANNING_BUDGET);
        assert(owner.pi05_pair_tile_rows()==160 && !owner.pi05_pair_single_compact());
        assert((owner.pi05_down_pair_arguments()==std::vector<int64_t>{24,320,2048,2048,8,tile.tile.n_tile,tile.grid_x,4,1,1,2,2,3,0,1,tile.tile.m_tile,tile.tile.m_tile,tile.tile.k_tile,tile.grid_y,48,50,52,26,27}));
        assert((owner.pi05_owner_norm_arguments()==std::vector<int64_t>{2,320,2048,8,40,163840,1310720,1,17,1,2,1}));
        const auto gate=owner.pi05_prefill_gateup_pair_arguments();
        if(a8) {assert(gate.at(0)==7 && gate.at(6)==gate_tile.tile.m_tile && gate.at(20)==256);
            assert(owner.pi05_a8_bytes()==1331200 && owner.pi05_a8_root_bytes()==1332736);
            // Actual norm ABI gives SX1 start; every split tile may load
            // 16 FP16 scales, including the explicitly initialized tail.
            const auto norm=owner.pi05_o_pair_norm_arguments(1);
            assert(norm.at(13)==1331840 && norm.at(13)+640+32<=packed);
        } else assert(gate.at(0)==5 && gate.at(6)==gate_tile.tile.m_tile);
        assert(owner.pi05_o_pair_arguments().at(0)==22);
        std::cout<<"C320 "<<(a8?"A8":"A16")<<" temp="<<plan.required_extent<<" gemma_persistent="<<persistent
                 <<" composite_persistent=656640 total="<<total<<" headroom="<<(SpmAllocator::SPM_PLANNING_BUDGET/256*256-total)<<"\n";
    }
}
int main(){for(int64_t rows:{272,288,304,320,400,416,432,448})for(bool a8:{false,true})verify(rows,a8);for(int64_t rows:{304,432})verify(rows,false,true);verify(272,true,false,true);}
'''
    output = _run_native(tmp_path, code, torch)
    assert "total=6636800 headroom=1666816" in output
    assert "total=6658816 headroom=1644800" in output
    assert "total=8176896 headroom=126720" in output
    assert "total=8206336 headroom=97280" in output
    assert "total=6318336 headroom=1985280" in output
    assert "total=6339072 headroom=1964544" in output
    assert "C272 P512 total=6689280 headroom=1614336" in output
    print(output, end="")

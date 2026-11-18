"""Actual G05 Vision owner cold/descriptor paths on CPU doubles, not HW evidence."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_high_precision_cold_binding_and_original_gelu_formulas(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    signatures = (
        "void set_high_precision(bool enabled)",
        "void set_gelu_erf_ultra(bool enabled)",
        "void set_moe_precision(bool linear_acc32, bool gelu_erf_ultra)",
        "std::vector<int64_t> g05_high_precision_encoder_arguments(int64_t rows) const",
        "std::vector<int64_t> g05_high_precision_merger_arguments(int64_t rows) const",
        "void consume_g05_high_precision_encoder_activation(int64_t rows, int64_t invocation)",
        "void consume_g05_high_precision_merger_activation(int64_t rows, int64_t invocation)",
    )
    methods = []
    for signature in signatures:
        name = signature.split("(")[0].split()[-1]
        methods.append(signature + _block(source, "Qwen3_5VisionModel::" + name + "("))
    layer = _block(source, "void Qwen3_5VisionModel::build_layer_subgraph(")
    merger = _block(source, "void Qwen3_5VisionModel::emit_merger(")
    def dispatch(body, marker):
        start = body.index(marker)
        end = body.index(");", body.index("rpu_launch_eltwise_unary_spm_kernel(", start)) + 2
        return body[start:end]
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok,...) do{if(!(ok))throw std::runtime_error("rejected");}while(0)
bool graph_active=false,coordinator_busy=false,payloads=true;int payload_checks=0;
struct RpuKernelGraph{static bool has_active(){return graph_active;}};
struct RpuExecutionCoordinator{
 static void require_graph_quiescent(const char*){TORCH_CHECK(!coordinator_busy);}
 static void check_current_thread_execution_allowed(const char*){TORCH_CHECK(!coordinator_busy);}
};
void rpu_require_high_precision_math_kernels(){++payload_checks;TORCH_CHECK(payloads);}
enum class RpuUnaryPrecision{BASE,HIGH};enum class GeluMode{TANH,ERF,ERF_ULTRA};enum class ValuOpType{ADD};
enum class FmbRouteFamily{GRAPH_SCHEDULE};
constexpr int64_t G05_VISION_HIGH_ENCODER_ACTIVATION_SITE=3127033570831610403LL;
constexpr int64_t G05_VISION_HIGH_MERGER_ACTIVATION_SITE=7199989347316866274LL;
std::vector<GeluMode> formulas;std::vector<RpuUnaryPrecision> precisions;
void rpu_launch_eltwise_unary_spm_kernel(uint32_t in,uint32_t out,int64_t count,ValuOpType op,GeluMode formula,int cores,RpuUnaryPrecision precision){
 assert(in==out&&count>0&&op==ValuOpType::ADD&&cores==8);formulas.push_back(formula);precisions.push_back(precision);
}
struct Model{
 bool high_precision_=false,high_precision_bound_=false,gelu_erf_ultra_=false,moe_precision_=false,has_temporal_=false,z2_bound_=false,linear_acc32_=true,complete=true;
 int cores=8,layers=0,resolved=0,invalidations=0;int64_t merger_hidden_=4096,current_camera_batch_count_=1;
 std::vector<int> layer_weights_;std::map<int64_t,std::vector<int64_t>> routes;
 int num_cores()const{return cores;}int num_layers()const{return layers;}int get_last_resolved_chunk_size()const{return resolved;}
 int intermediate_size()const{return 4096;}void invalidate_model_state(){++invalidations;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> arguments,int64_t invocation){
  assert(family==FmbRouteFamily::GRAPH_SCHEDULE);TORCH_CHECK(selector==1&&flags==0&&invocation==0&&routes.count(site)&&routes.at(site)==arguments);
 }
 uint32_t addr(int,const char*)const{return 0x40010000;}
 __METHODS__
 void encoder(){int64_t seq_len=256,local_inter=512;struct{int idx=0;}chunk;__ENCODER__}
 void merger(){int64_t m=64,local_mh=512,invocation=0;__MERGER__}
};
template<class F>void rejected(F f){try{f();assert(false);}catch(const std::runtime_error&){}}
int main(){
 Model base;payloads=false;base.set_high_precision(false);assert(!base.high_precision_&&base.high_precision_bound_&&base.invalidations==1&&payload_checks==0);base.encoder();base.merger();assert(formulas==std::vector<GeluMode>({GeluMode::TANH,GeluMode::ERF}));assert(precisions==std::vector<RpuUnaryPrecision>(2,RpuUnaryPrecision::BASE));rejected([&]{base.set_high_precision(true);});
 Model missing;rejected([&]{missing.set_high_precision(true);});assert(!missing.high_precision_&&!missing.high_precision_bound_&&missing.invalidations==0);payloads=true;
 for(int fault=0;fault<10;++fault){Model m;if(fault==0)m.layer_weights_.push_back(1);if(fault==1)m.layers=1;if(fault==2)m.resolved=256;if(fault==3)graph_active=true;if(fault==4)coordinator_busy=true;if(fault==5)m.cores=4;if(fault==6)m.gelu_erf_ultra_=true;if(fault==7)m.moe_precision_=true;if(fault==8)m.has_temporal_=true;if(fault==9)m.z2_bound_=true;rejected([&]{m.set_high_precision(true);});assert(!m.high_precision_&&!m.high_precision_bound_&&m.invalidations==0);graph_active=coordinator_busy=false;}
 Model high;high.set_high_precision(true);assert(high.high_precision_&&high.high_precision_bound_&&high.invalidations==1);rejected([&]{high.set_high_precision(false);});rejected([&]{high.set_gelu_erf_ultra(true);});rejected([&]{high.set_moe_precision(true,false);});assert(!high.gelu_erf_ultra_&&!high.moe_precision_);
 high.routes[G05_VISION_HIGH_ENCODER_ACTIVATION_SITE]=high.g05_high_precision_encoder_arguments(256);high.routes[G05_VISION_HIGH_MERGER_ACTIVATION_SITE]=high.g05_high_precision_merger_arguments(64);
 formulas.clear();precisions.clear();high.encoder();high.merger();assert(formulas==std::vector<GeluMode>({GeluMode::TANH,GeluMode::ERF}));assert(precisions==std::vector<RpuUnaryPrecision>(2,RpuUnaryPrecision::HIGH));
 for(auto&[site,args]:high.routes)for(auto&argument:args){auto before=formulas;++argument;rejected([&]{if(site==G05_VISION_HIGH_ENCODER_ACTIVATION_SITE)high.encoder();else high.merger();});--argument;assert(before==formulas);}
 high.complete=false;rejected([&]{high.encoder();});rejected([&]{high.merger();});assert(formulas.size()==2);
}
'''
    cpp = cpp.replace("__METHODS__", "\n".join(methods)).replace(
        "__ENCODER__", dispatch(layer, "consume_g05_high_precision_encoder_activation(seq_len")).replace(
        "__MERGER__", dispatch(merger, "consume_g05_high_precision_merger_activation(m"))
    path, exe = tmp_path / "high.cpp", tmp_path / "high"
    path.write_text(cpp)
    result = subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", str(path), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _block(text, marker):
    begin = text.index("{", text.index(marker))
    depth = 0
    for end in range(begin, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:end + 1]
    raise AssertionError(marker)


@pytest.mark.parametrize("acc16,w8", [(False, False), (True, True), (True, False), (False, True)])
@pytest.mark.parametrize("high", [False, True])
def test_owner_ln_cold_reset_geometry_descriptors_and_original_fallback(tmp_path, acc16, w8, high):
    source = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    header = (ROOT / "src/fused/rpu_qwen3_5_vision_model.h").read_text()
    methods = [
        ("std::vector<int64_t> g05_high_precision_encoder_arguments(int64_t rows) const", "Qwen3_5VisionModel::g05_high_precision_encoder_arguments("),
        ("void consume_g05_high_precision_encoder_activation(int64_t rows, int64_t invocation)", "Qwen3_5VisionModel::consume_g05_high_precision_encoder_activation("),
        ("void prepare_g05_owner_ln()", "void Qwen3_5VisionModel::prepare_g05_owner_ln("),
        ("bool is_g05_k1_fast_replay_profile() const", "bool Qwen3_5VisionModel::is_g05_k1_fast_replay_profile("),
        ("bool use_g05_owner_ln(int64_t rows) const", "bool Qwen3_5VisionModel::use_g05_owner_ln("),
        ("std::vector<int64_t> g05_owner_ln_arguments() const", "std::vector<int64_t> Qwen3_5VisionModel::g05_owner_ln_arguments("),
        ("void launch_g05_owner_ln(int layer, const ChunkInfo& chunk)", "void Qwen3_5VisionModel::launch_g05_owner_ln("),
        ("void launch_g05_compact_residual(const ChunkInfo& chunk)", "void Qwen3_5VisionModel::launch_g05_compact_residual("),
        ("int64_t subclass_layout_hash() const", "int64_t Qwen3_5VisionModel::subclass_layout_hash("),
    ]
    manifest = _block(source, "Qwen3_5VisionModel::physical_manifest_for_candidate(")
    begin = manifest.index("            if (use_g05_owner_ln(chunk.len))")
    end = manifest.index("\n        }\n        for (const ChunkInfo& chunk : plan.qkv.chunks)", begin)
    plan = manifest[begin:end]
    layer = _block(source, "void Qwen3_5VisionModel::build_layer_subgraph(")
    begin = layer.index("    const bool owner_ln = use_g05_owner_ln(seq_len);")
    end = layer.index("    const bool z2_final", begin)
    emission = layer[begin:end]
    installation = _block(source, "void Qwen3_5VisionModel::set_weights(")
    # Execute the real successful installation tail, after the caller replaces
    # the simulated tensor vectors, to catch capability state leaking on reinstall.
    install_tail = installation[installation.rindex("    prepare_g05_owner_ln();"):]
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok,...) do {if(!(ok))throw std::runtime_error("rejected");}while(0)
namespace at {constexpr int kHalf=1,kChar=2,kPrivateUse1=1;
struct Tensor {
    bool present=true,contiguous=true; int dtype=kHalf,location=kPrivateUse1; int64_t owner=42;int rank=2;int64_t rows=1024,cols=1024;int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}
    bool defined()const{return present;} int scalar_type()const{return dtype;}
    struct Device{int location;int type()const{return location;}};
    Device device()const{return {location};} bool is_contiguous()const{return contiguous;}
};}
struct Weights{at::Tensor q_w,k_w,v_w,o_w,fc1_w,fc2_w;at::Tensor q_ws{false},k_ws{false},v_ws{false},o_ws{false},fc1_ws{false},fc2_ws{false};};
struct Norms{at::Tensor ln1_w,ln1_b,ln2_w,ln2_b,q_b,k_b,v_b,o_b,fc1_b,fc2_b;};
void append_kvinsert_cost_scalar_identity(std::vector<int64_t>& out,double value){int64_t bits;std::memcpy(&bits,&value,8);out.push_back(bits);}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>& out,const at::Tensor& tensor){out.push_back(tensor.owner);}
namespace detail {int64_t layout_mix(int64_t a,int64_t b){return uint64_t(a)*131+uint64_t(b);}}
enum class KernelId {G05_VISION_FC2_WEIGHT_OUTER_W8A16_ACC16_M768N1024K512,G05_VISION_OWNER_REDUCE_LAYERNORM_M256N1024,G05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M256N1024,G05_VISION_QKV_FP16_ACC32_M256N128K1024,G05_VISION_QKV_W8A16_ACC32_M256N128K1024,G05_VISION_QKV_W8A16_ACC16_M256N128K1024,PL_AUTOTILE_BEGIN=1000};
inline KernelId parallel_linear_kernel_id(rpu_pl_tiling::WeightType type,bool acc32,const rpu_pl_tiling::TilePick&tile){return static_cast<KernelId>(static_cast<size_t>(KernelId::PL_AUTOTILE_BEGIN)+rpu_pl_tiling::parallel_kernel_ordinal(type,acc32,tile));}
struct KernelCache{ bool is_initialized()const{return true;} void* get(KernelId id){return has_loaded(id)?this:nullptr;}
    bool owner=true,companion=true; int queries=0;
    static KernelCache& instance(){static KernelCache cache;return cache;}
    bool has_loaded(KernelId id){++queries;if(id==KernelId::G05_VISION_FC2_WEIGHT_OUTER_W8A16_ACC16_M768N1024K512)return false;if(id==KernelId::G05_VISION_QKV_FP16_ACC32_M256N128K1024||id==KernelId::G05_VISION_QKV_W8A16_ACC32_M256N128K1024||id==KernelId::G05_VISION_QKV_W8A16_ACC16_M256N128K1024)return false;return id==KernelId::G05_VISION_OWNER_REDUCE_LAYERNORM_M256N1024?owner:companion;}
};
enum class FmbRouteFamily{NORMALIZATION,ALL_REDUCE,LINEAR,GRAPH_SCHEDULE};
constexpr int64_t G05_VISION_HIGH_ENCODER_ACTIVATION_SITE=3127033570831610403LL;
constexpr int64_t G05_VISION_HIGH_MERGER_ACTIVATION_SITE=7199989347316866274LL;
constexpr int64_t G05_VISION_QKV_W8_LINEAR_SITE=1317639897773331476LL;
constexpr int64_t G05_VISION_QKV_LINEAR_SITE=3357339527802085618LL;
constexpr int64_t G05_VISION_QKV_SPM_TRANSPORT_SITE=8596948981803816035LL;
constexpr int64_t G05_VISION_K1_B3_SCHEDULE_SITE=1;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_SCHEDULE_SITE=2;
constexpr int64_t G05_VISION_OWNER_LN_SITE=7401236839773657641LL;
constexpr int64_t G05_VISION_COMPACT_RESIDUAL_SITE=8040294111507309046LL;
constexpr int64_t QWEN35_VISION_ATTN_ALL_REDUCE_SITE=11,QWEN35_VISION_MLP_ALL_REDUCE_SITE=12;
constexpr int64_t QWEN35_VISION_FC1_SITE=13,QWEN35_VISION_FC2_SITE=14;
int64_t qwen35_vision_ring_route(int64_t,int64_t,int){return 7;}
int64_t qwen35_vision_linear_route(){return 8;}
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t> args;int64_t invocation;};
struct ChunkInfo{int64_t idx=2,offset=0,len=256;};
std::vector<std::string> emitted;
void rpu_launch_g05_vision_owner_reduce_layernorm_m256n1024_kernel(uint32_t partial,uint32_t residual,uint32_t compact,uint32_t normalized,uint32_t gamma,uint32_t beta){
    assert(partial==0x20000&&normalized==partial&&residual==0xa0000&&compact==0x120000);
    assert(gamma>=0x1a0000&&beta==gamma+2048);emitted.push_back("owner");
}
void rpu_launch_g05_vision_compact_residual_m256n1024_kernel(uint32_t partial,uint32_t compact,uint32_t output){
    assert(partial==0x20000&&compact==0x120000&&output==0xa0000);emitted.push_back("compact");
}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t,uint32_t,uint32_t,int64_t,int64_t,int,int){emitted.push_back("ring");}
void rpu_launch_layernorm_spm_kernel(uint32_t,uint32_t,uint32_t,uint32_t,int64_t,int64_t,double,bool,int,int){emitted.push_back("ln");}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t,const at::Tensor&,uint32_t,int64_t,int64_t,int64_t,int,int,uint32_t,const at::Tensor&,int,int,bool acc32){assert(acc32==__ACC32__);emitted.push_back("linear");}
enum class ValuOpType{ADD}; enum class GeluMode{TANH};
enum class RpuUnaryPrecision{BASE,HIGH};
void rpu_launch_eltwise_unary_spm_kernel(uint32_t,uint32_t,int64_t,ValuOpType,GeluMode formula,int,RpuUnaryPrecision precision){assert(formula==GeluMode::TANH);assert(precision==(__HIGH__?RpuUnaryPrecision::HIGH:RpuUnaryPrecision::BASE));emitted.push_back("gelu");}
struct Model{
 bool is_g05_k1_camera_group_profile()const{return false;}
 std::vector<int64_t>g05_k1_camera_group_arguments()const{assert(false);return {};}
 bool g05_k1_camera_group_available(int64_t)const{return false;}
 void consume_g05_k1_camera_group_schedule(const ChunkInfo&,bool){assert(false);}
 void emit_g05_k1_camera_group_layer(int,const ChunkInfo&){assert(false);}

 bool g05_k1_b3_available_=false;
 std::array<bool,2>g05_k1_generated_group_available_{{false,false}};
 bool is_g05_k1_b3_profile()const{return false;}
 std::vector<int64_t>g05_k1_b3_arguments()const{assert(false);return {};}
 void consume_g05_k1_b3_schedule(const ChunkInfo&,bool){assert(false);}
 void emit_g05_k1_b3_layer(int,const ChunkInfo&){assert(false);}

    int cores=8,attn=8,mlp=8,layers=24,hidden=1024,intermediate=4096,q=16,kv=16,hd=64;
    int orig_head_dim_=64,patch_dim_=1536,merger_hidden_=4096,out_hidden_size_=2048;
    double eps_=1e-6;
    bool linear_acc32_=true,gelu_erf_ultra_=false,moe_precision_=false,has_temporal_=false;
    bool high_precision_=__HIGH__;
    bool has_rope_=true,has_step0_=true,has_merger_=true,z2_bound_=false,debug=false;
    bool g05_qkv_spm_available_=false,g05_qkv_linear_available_=false,g05_w8a16_=false;
    bool g05_owner_ln_available_=false,current_compact_step0_=false,complete=true;
    int current_num_patches_=256,current_output_patches_=256,current_temporal_num_frames_=1,current_camera_batch_count_=1;
    int invalidations=0;
    std::vector<Weights> layer_weights_{24};std::vector<Norms> layer_bias_norm_{24};
    at::Tensor pe_w_,pe_b_,merger_fc1_w_,merger_fc1_b_,merger_fc2_w_,merger_fc2_b_,merger_norm_w_,merger_norm_b_;
    std::vector<Route> routes;
    int num_cores()const{return cores;}int attn_tp()const{return attn;}int mlp_tp()const{return mlp;}
    int num_layers()const{return layers;}int hidden_size()const{return hidden;}int intermediate_size()const{return intermediate;}
    int num_q_heads()const{return q;}int num_kv_heads()const{return kv;}int head_dim()const{return hd;}
    bool get_debug_export()const{return debug;}Model& ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
    uint32_t addr(int core,const char* name)const{assert(core==0);std::string n(name);return n=="oproj"?0x20000:n=="residual1"?0xa0000:n=="input_norm"?0x120000:0x220000;}
    uint32_t layer_addr(int layer,int core,const char* name)const{assert(core==0);return 0x1a0000+layer*4096+(std::string(name)=="ln2_beta"?2048:0);}
    void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t invocation){
        auto it=std::find_if(routes.begin(),routes.end(),[&](const auto&r){return r.site==site&&r.invocation==invocation;});
        TORCH_CHECK(it!=routes.end()&&it->family==family&&it->selector==selector&&it->flags==flags&&it->args==args);
    }
    void consume_manifest_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t invocation){consume_physical_route(family,site,selector,0,{},invocation);}
    void invalidate_model_state(){++invalidations;}
    void installed(){__INSTALL__
    __METHODS__
    std::vector<int64_t> kvinsert_cost_weight_identity()const __IDENTITY__
    void plan(ChunkInfo chunk={}){
        routes.clear();auto append_route=[&](FmbRouteFamily family,int64_t site,int64_t selector,int64_t invocation,std::vector<int64_t> args=std::vector<int64_t>{}){routes.push_back({site,family,selector,0,args,invocation});};
        __PLAN__
    }
    void emit(int layer_idx,ChunkInfo chunk={}){
        int64_t seq_len=chunk.len,h=hidden_size(),is_=intermediate_size(),local_inter=is_/num_cores();int tp=attn_tp();const auto& lw=layer_weights_[layer_idx];
        __EMISSION__
    }
};
template<class F>void reject(F f){const auto before=emitted;try{f();assert(false);}catch(const std::runtime_error&){}assert(emitted==before);}
int main(){
    Model m;auto& cache=KernelCache::instance();
    m.linear_acc32_=__ACC32__;
    for(auto&w:m.layer_weights_){
      at::Tensor*weights[]={&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.fc1_w,&w.fc2_w};
      at::Tensor*scales[]={&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.fc1_ws,&w.fc2_ws};
      for(int i=0;i<6;++i){weights[i]->rows=i==4?4096:1024;weights[i]->cols=i==5?4096:1024;
        if(__W8__){weights[i]->dtype=at::kChar;scales[i]->present=true;scales[i]->rank=1;scales[i]->rows=weights[i]->rows;}}
    }
    for(auto&n:m.layer_bias_norm_){for(auto*t:{&n.ln1_w,&n.ln1_b,&n.ln2_w,&n.ln2_b,&n.q_b,&n.k_b,&n.v_b,&n.o_b,&n.fc1_b,&n.fc2_b}){t->rank=1;t->rows=t==&n.fc1_b?4096:1024;}}
    // Patch/RoPE/merger are installed later; tensor admission must survive that order.
    m.has_step0_=m.has_rope_=m.has_merger_=false;m.installed();assert(m.g05_owner_ln_available_&&!m.use_g05_owner_ln(256));
    m.has_step0_=m.has_rope_=m.has_merger_=true;assert(m.use_g05_owner_ln(256));
    const auto hash=m.subclass_layout_hash();const auto identity=m.kvinsert_cost_weight_identity();
    {auto other=m;other.high_precision_=!m.high_precision_;assert(other.subclass_layout_hash()!=hash&&other.kvinsert_cost_weight_identity()!=identity);assert(other.use_g05_owner_ln(256)==m.use_g05_owner_ln(256));}
    m.plan();assert(m.routes.front().site==G05_VISION_OWNER_LN_SITE&&m.routes.back().site==G05_VISION_COMPACT_RESIDUAL_SITE);
    assert(m.routes.front().args==std::vector<int64_t>({1,256,1024,8,32,65536,524288,0x1432,1,2,1}));
    const int queries=cache.queries;
    for(int layer=0;layer<24;++layer){emitted.clear();m.emit(layer);assert(emitted==std::vector<std::string>({"owner","linear","gelu","linear","compact"}));}
    assert(cache.queries==queries); // No capability queries or tensor scans during emission.
    if(m.high_precision_){auto route=std::find_if(m.routes.begin(),m.routes.end(),[](const auto&r){return r.site==G05_VISION_HIGH_ENCODER_ACTIVATION_SITE;});assert(route!=m.routes.end()&&route->args==std::vector<int64_t>({1,1,1,256,4096,512,8,1}));for(auto&arg:route->args){++arg;reject([&]{m.consume_g05_high_precision_encoder_activation(256,2);});--arg;}m.complete=false;reject([&]{m.consume_g05_high_precision_encoder_activation(256,2);});m.complete=true;}
    reject([&]{m.launch_g05_owner_ln(-1,{});});reject([&]{m.launch_g05_owner_ln(24,{});});
    for(bool complete:{false,true}){m.complete=complete;for(int which:{0,1}){
        m.plan();if(complete){auto& route=which?m.routes.back():m.routes.front();route.args[5]+=256;}
        reject([&]{if(which)m.launch_g05_compact_residual({});else m.launch_g05_owner_ln(0,{});});
    }}m.complete=true;
    m.plan();reject([&]{m.launch_g05_owner_ln(0,{3,0,256});});reject([&]{m.launch_g05_compact_residual({3,0,256});});
    auto fallback=[&](auto change){auto other=m;change(other);other.plan();assert(!other.use_g05_owner_ln(256));assert(other.routes.front().site==QWEN35_VISION_ATTN_ALL_REDUCE_SITE&&other.routes.back().site==QWEN35_VISION_MLP_ALL_REDUCE_SITE);emitted.clear();other.emit(0);assert(emitted==std::vector<std::string>({"ring","ln","linear","gelu","linear","ring"}));};
    for(int stale:{0,512}){m.current_num_patches_=stale;m.plan();assert(m.routes.front().site==G05_VISION_OWNER_LN_SITE);}m.current_num_patches_=256;
    fallback([](auto& x){x.current_output_patches_=128;});
    fallback([](auto& x){x.current_camera_batch_count_=3;});fallback([](auto& x){x.current_temporal_num_frames_=6;});
    fallback([](auto& x){x.current_compact_step0_=true;});fallback([](auto& x){x.debug=true;});fallback([](auto& x){x.z2_bound_=true;});
    m.plan({2,0,128});assert(m.routes.front().site==QWEN35_VISION_ATTN_ALL_REDUCE_SITE);reject([&]{m.launch_g05_owner_ln(0,{2,0,128});});
    for(int missing=0;missing<3;++missing){cache.owner=missing==1;cache.companion=missing==0;m.installed();assert(!m.g05_owner_ln_available_&&m.g05_qkv_spm_available_);assert(m.subclass_layout_hash()!=hash&&m.kvinsert_cost_weight_identity()!=identity);fallback([](auto&){});}
    cache.owner=cache.companion=true;m.installed();assert(m.subclass_layout_hash()==hash&&m.kvinsert_cost_weight_identity()==identity);
    auto cold_reject=[&](auto change){auto other=m;change(other);other.installed();assert(!other.g05_owner_ln_available_&&other.invalidations==m.invalidations+1);};
    cold_reject([](auto& x){x.layer_weights_.back().fc2_w.dtype=x.g05_w8a16_?at::kHalf:at::kChar;});
    cold_reject([](auto& x){x.layer_bias_norm_.back().fc2_b.dtype=2;});
    cold_reject([](auto& x){x.layer_bias_norm_.back().ln2_w.present=false;});
    cold_reject([](auto& x){x.layer_bias_norm_.back().ln2_b.location=0;});
    cold_reject([](auto& x){x.layer_weights_.back().q_w.contiguous=false;});
    cold_reject([](auto& x){x.layer_bias_norm_.pop_back();});cold_reject([](auto& x){x.layer_weights_.pop_back();});
    if(!m.g05_w8a16_){auto other=m;other.linear_acc32_=!m.linear_acc32_;other.installed();assert(other.g05_owner_ln_available_&&other.subclass_layout_hash()!=hash&&other.kvinsert_cost_weight_identity()!=identity);}
    cold_reject([](auto& x){x.eps_=1e-5;});
    cold_reject([](auto& x){x.cores=4;});cold_reject([](auto& x){x.has_temporal_=true;});
    m.layer_bias_norm_.back().ln2_w.owner=43;assert(m.kvinsert_cost_weight_identity()!=identity);m.layer_bias_norm_.back().ln2_w.owner=42;
    m.layer_weights_.back().fc2_w.owner=43;assert(m.kvinsert_cost_weight_identity()!=identity);
}
'''
    replacements = {
        "__ACC32__": "false" if acc16 else "true",
        "__W8__": "true" if w8 else "false",
        "__HIGH__": "true" if high else "false",
        "__METHODS__": "\n".join(declaration + _block(source, marker) for declaration, marker in methods),
        "__IDENTITY__": _block(header, "std::vector<int64_t> kvinsert_cost_weight_identity() const"),
        "__PLAN__": plan, "__EMISSION__": emission, "__INSTALL__": install_tail,
    }
    for marker, value in replacements.items():
        cpp = cpp.replace(marker, value)
    cpp = chr(10).join(('#include <array>', '#include "' + str(ROOT / "src/ops/rpu_linear_tiling.h") + '"', cpp))
    path, exe = tmp_path / "owner.cpp", tmp_path / "owner"
    path.write_text(cpp)
    compiled = subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "src/ops"), "-I", str(ROOT / "src/core"), str(path), "-o", str(exe)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

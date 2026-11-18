"""Board-free contract for exact G0.5 K1 activation routing."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_g05_high_precision_cold_routes_and_native_math(tmp_path):
    """Run production cold policy, receipts and scalar/vector math on CPU doubles."""
    from test_g05_q_gate_native import _compile_run
    from test_g05_vision_owner_model import _block
    import re

    source = _read("src/fused/rpu_qwen3_5_model.cpp")
    decls = _read("src/core/rpu_kernel_decls.h")
    methods = [
        ("void set_high_precision(bool enabled)", "void Qwen3_5Model::set_high_precision("),
        ("RpuUnaryPrecision unary_precision() const", "RpuUnaryPrecision Qwen3_5Model::unary_precision("),
        ("RpuRmsNormSpmRoute rmsnorm_route(int64_t rows,int64_t cols) const", "RpuRmsNormSpmRoute Qwen3_5Model::rmsnorm_route("),
        ("int64_t high_precision_site() const", "int64_t Qwen3_5Model::high_precision_site("),
        ("std::vector<int64_t> high_precision_arguments() const", "std::vector<int64_t> Qwen3_5Model::high_precision_arguments("),
        ("void consume_high_precision_math()", "void Qwen3_5Model::consume_high_precision_math("),
        ("bool use_action_mod_body() const", "bool Qwen3_5Model::use_action_mod_body("),
        ("void prepare_action_newton_norm_shift()", "void Qwen3_5Model::prepare_action_newton_norm_shift("),
        ("bool use_action_newton_norm_shift(int64_t rows) const", "bool Qwen3_5Model::use_action_newton_norm_shift("),
        ("std::vector<int64_t> action_newton_norm_shift_arguments() const", "std::vector<int64_t> Qwen3_5Model::action_newton_norm_shift_arguments("),
        ("void apply_action_newton_norm_shift(uint32_t input,uint32_t output,const ChunkInfo&chunk)", "void Qwen3_5Model::apply_action_newton_norm_shift("),
        ("int64_t action_norm_route(int64_t rows) const", "int64_t Qwen3_5Model::action_norm_route("),
        ("std::vector<int64_t> action_norm_arguments(int64_t rows) const", "std::vector<int64_t> Qwen3_5Model::action_norm_arguments("),
        ("void apply_adaptive_norm(uint32_t input,uint32_t output,const ChunkInfo&chunk)", "void Qwen3_5Model::apply_adaptive_norm("),
    ]
    admission = _block(_block(source, "void Qwen3_5Model::set_weights("), "if (high_precision_)")
    manifest = _block(source, "FmbPhysicalExecutionManifest Qwen3_5Model::physical_manifest_for_candidate(")
    producer = "if (high_precision_)" + _block(manifest, "if (high_precision_)")
    norm_start = manifest.index("            const int64_t norm_route = action_norm_route(chunk.len);")
    norm_producer = manifest[norm_start:manifest.index("            if (use_action_tail(chunk.len))", norm_start)]
    gdn = _block(source, "void Qwen3_5Model::build_gdn(")
    tail = gdn[gdn.index("    if (decode && !high_precision_)"):gdn.index("    // out_proj (row-parallel)")]
    constants = "\n".join(re.findall(
        r"constexpr int64_t QWEN35_(?:TEXT_HIGH_PRECISION_SITE|ACTION_HIGH_PRECISION_SITE|ACTION_FUSED_NORM_SHIFT_SITE|ACTION_RMS_NORM_SHIFT_SITE|ACTION_NEWTON_NORM_SHIFT_SITE)\s*=.*?;",
        source, re.S))
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define DWIDTH 2
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10 {struct Half{uint16_t x;explicit Half(float v):x(v==1e-6f?0x11:v==1.f?0x3c00:0){}};}
namespace at {using IntArrayRef=std::vector<int64_t>;}
__PRECISION__
__NORM_ENUM__
inline constexpr int64_t RPU_RMSNORM_SCM_U16_MAX=65535;
inline constexpr int64_t RPU_RMSNORM_MAX_COLS=RPU_RMSNORM_SCM_U16_MAX/2;
RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route(int64_t rows,int64_t cols)__RESOLVER__
__CONSTANTS__
bool payloads=true;int require_calls=0;
void rpu_require_high_precision_math_kernels(){++require_calls;TORCH_CHECK(payloads);}
enum class KernelId {RHINOVLA_NEWTON_NORM_SHIFT_H1024};
struct KernelCache {bool available=true;int queries=0;
 static KernelCache&instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId){++queries;return available;}
};
struct RpuExecutionCoordinator {
 static bool quiet,allowed;
 static void require_graph_quiescent(const char*){TORCH_CHECK(quiet);}
 static void check_current_thread_execution_allowed(const char*){TORCH_CHECK(allowed);}
};bool RpuExecutionCoordinator::quiet=true,RpuExecutionCoordinator::allowed=true;
enum class FmbRouteFamily {GRAPH_SCHEDULE,NORMALIZATION};
enum class ValuOpType {SILU,MUL,ADD};
struct ChunkInfo {int64_t len=32,offset=0,idx=0;};
struct Route {int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>arguments;int64_t invocation=0;};
struct Manifest {std::vector<Route>routes;};
struct Math {std::string kind;uint32_t input,output,weight;int64_t rows,cols;int mode;};
std::vector<Math>math;
void rpu_launch_rmsnorm_spm_kernel(uint32_t x,uint32_t y,uint32_t w,int64_t rows,int64_t cols,double eps,RpuRmsNormSpmRoute route,int cores){
 assert(eps==1e-6&&cores==8);math.push_back({"norm",x,y,w,rows,cols,int(route)});
}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t x,uint32_t y,int64_t elements,ValuOpType op,bool,int cores,RpuUnaryPrecision precision){
 assert(op==ValuOpType::SILU&&cores==8);math.push_back({"silu",x,y,0,elements,0,int(precision)});
}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t x,uint32_t y,uint32_t out,int64_t elements,ValuOpType op,c10::Half,int cores){
 assert(op==ValuOpType::MUL&&cores==8);math.push_back({"mul",x,out,y,elements,0,0});
}
void rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(uint32_t shift,uint32_t x,uint32_t y,int64_t rows,int64_t cols,c10::Half,ValuOpType op,bool,int cores=8){
 assert(op==ValuOpType::ADD&&cores==8);math.push_back({"shift",x,y,shift,rows,cols,0});
}
void rpu_launch_rhinovla_newton_norm_shift_spm_kernel(uint32_t x,uint32_t y,uint32_t scale,uint32_t shift,int64_t rows,double eps,int cores){
 assert(rows>0&&rows!=16&&rows!=32&&eps==1e-6&&cores==8&&shift==scale+2048);
 math.push_back({"newton-shift",x,y,scale,rows,1024,7});
}
void rpu_launch_pi05_adarms_norm_shift_spm_kernel(uint32_t,uint32_t,uint32_t,uint32_t,int64_t,double,int){TORCH_CHECK(false);}
void rpu_launch_g05_rtc_adarms_norm_shift_spm_kernel(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,int64_t,double,int){TORCH_CHECK(false);}
void rpu_launch_qwen3_5_rms_norm_gated(uint32_t x,uint32_t z,uint32_t w,uint32_t out,int64_t rows,int64_t cols,double,int){math.push_back({"fused-base",x,out,z+w,rows,cols,0});}
struct Model {
 bool high_precision_=false,high_precision_bound_=false,action_mode_=false,z2_bound_=false,complete=true;
 bool linear_acc32_=false,use_silu_=true,has_qk_norm_=true,has_gdn_=true;
 bool action_norm_shift_available_=true,action_rtc_norm_shift_available_=true,action_rtc_active_=false;
 bool action_newton_norm_shift_available_=false;
 bool action_step_active_=false,action_loop_active_=false,action_mod_body_w8_=false,action_mod_body_fp16_=false;
 int64_t action_num_steps_=1,action_len_=32;
 int64_t gdn_nvh_=16,gdn_dk_=128,gdn_dv_=128,last=0,action_rtc_prefix_len_=0;
 int cores=8,layers=24,heads=8,kv=8,hd=256,tp=8,mt=8;
 int64_t h=2048,inter=6144;int invalidations=0,consumed=0;double eps_=1e-6;uint32_t adaptive_mod_row_offset_=0;
 std::vector<int>layer_weights_;std::vector<Route>routes;
 int num_cores()const{return cores;}int num_layers()const{return layers;}int num_q_heads()const{return heads;}int num_kv_heads()const{return kv;}int head_dim()const{return hd;}int attn_tp()const{return tp;}int mlp_tp()const{return mt;}
 bool use_action_nvfp4(int64_t)const{return false;}
 int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return inter;}int64_t get_last_resolved_chunk_size()const{return last;}
 void invalidate_model_state(){++invalidations;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 uint32_t addr(int,const char*)const{return 0x600000;}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>args,int64_t invocation=0){
  if(family==FmbRouteFamily::NORMALIZATION&&site==QWEN35_ACTION_RMS_NORM_SHIFT_SITE){TORCH_CHECK(selector==1&&flags==0&&args==std::vector<int64_t>({32,h,8}));return;}
  auto p=std::find_if(routes.begin(),routes.end(),[&](const Route&r){return r.site==site&&r.invocation==invocation;});
  TORCH_CHECK(p!=routes.end()&&p->family==family&&p->selector==selector&&p->flags==flags&&p->arguments==args);++consumed;
 }
 __METHODS__
 void plan(){Manifest manifest;__PRODUCER__ routes=manifest.routes;}
 void plan_norm(ChunkInfo chunk={}){Manifest manifest;__NORM_PRODUCER__ routes=manifest.routes;}
 void admit(int64_t hidden_size,int64_t intermediate_size,int64_t num_q_heads,int64_t num_kv_heads,int64_t head_dim,
            double eps,bool use_silu,at::IntArrayRef layer_is_full,at::IntArrayRef mrope_section,
            int64_t gdn_num_v_heads,int64_t gdn_key_head_dim,int64_t gdn_value_head_dim,int64_t gdn_conv_dim,int64_t gdn_conv_kernel){
  const int64_t N=layer_is_full.size();if(high_precision_)__ADMISSION__
 }
 void gdn_tail(bool decode){const int64_t L=decode?1:64,Hc=2,Dv=128,lval=256;const int nc=8;
  const uint32_t out_buf=0x100000,gated_buf=decode?0x200000:out_buf,z_buf=0x300000;
  auto D=[](const char*){return 0x400000u;};__GDN_TAIL__
 }
};
template<class F>void rejects(F f){auto before=math.size();try{f();assert(false);}catch(const std::runtime_error&){}assert(math.size()==before);}
int main(){
 Model base;base.plan();assert(base.routes.empty()&&base.unary_precision()==RpuUnaryPrecision::BASE&&base.rmsnorm_route(32,1024)==RpuRmsNormSpmRoute::BASE);
 base.gdn_tail(true);assert(math.size()==1&&math[0].kind=="fused-base");math.clear();
 Model missing;payloads=false;rejects([&]{missing.set_high_precision(true);});assert(!missing.high_precision_&&!missing.high_precision_bound_&&missing.invalidations==0);payloads=true;
 for(int fault=0;fault<5;++fault){Model x;if(fault==0)x.layer_weights_={1};if(fault==1)x.action_mode_=true;if(fault==2)x.z2_bound_=true;if(fault==3)x.last=1;if(fault==4)x.high_precision_bound_=true;rejects([&]{x.set_high_precision(true);});assert(x.invalidations==0);}
 RpuExecutionCoordinator::quiet=false;rejects([&]{base.set_high_precision(true);});RpuExecutionCoordinator::quiet=true;
 RpuExecutionCoordinator::allowed=false;rejects([&]{base.set_high_precision(true);});RpuExecutionCoordinator::allowed=true;
 Model m;m.set_high_precision(true);assert(m.high_precision_&&m.high_precision_bound_&&m.invalidations==1);rejects([&]{m.set_high_precision(false);});
 std::vector<int64_t>text(24),action(24,1);for(int i=0;i<24;++i)text[i]=(i+1)%4==0;
 auto admit_text=[&](const std::vector<int64_t>&layers){m.admit(2048,6144,8,8,256,1e-6,true,layers,{11,11,10},16,128,128,6144,4);};
 admit_text(text);m.admit(1024,4096,8,8,256,1e-6,true,action,{11,11,10},0,0,0,0,0);
 for(int i=0;i<24;++i){auto broken=text;broken[i]=!broken[i];rejects([&]{admit_text(broken);});}
 rejects([&]{m.admit(2048,6144,8,8,256,1e-5,true,text,{11,11,10},16,128,128,6144,4);});
 rejects([&]{m.admit(2048,6144,8,8,256,1e-6,false,text,{11,11,10},16,128,128,6144,4);});
 for(bool role:{false,true})for(bool acc32:{false,true}){
  m.action_mode_=role;m.linear_acc32_=acc32;m.has_gdn_=!role;m.h=role?1024:2048;m.inter=role?4096:6144;m.gdn_nvh_=role?0:16;m.gdn_dk_=m.gdn_dv_=role?0:128;m.plan();
  assert(m.routes.size()==1&&m.routes[0].site==(role?QWEN35_ACTION_HIGH_PRECISION_SITE:QWEN35_TEXT_HIGH_PRECISION_SITE));
  assert(m.routes[0].arguments.size()==17&&m.routes[0].arguments[1]==1&&m.routes[0].arguments[2]==role&&m.routes[0].arguments[13]==(acc32?32:16));
  m.consume_high_precision_math();m.consume_high_precision_math();const auto good=m.routes;
  for(size_t i=0;i<17;++i){++m.routes[0].arguments[i];rejects([&]{m.consume_high_precision_math();});m.routes=good;}
  ++m.routes[0].selector;rejects([&]{m.consume_high_precision_math();});m.routes=good;
  m.complete=false;rejects([&]{m.consume_high_precision_math();});m.complete=true;
 }
 m.action_mode_=true;m.h=1024;m.action_rtc_active_=true;
 for(int clean:{0,7,16,32}){math.clear();m.action_rtc_prefix_len_=clean;m.apply_adaptive_norm(0x100000,0x200000,{});
  assert(m.action_norm_route(32)==1);const int segments=clean==0||clean==32?1:2;assert(math.size()==size_t(segments*2));
  int offset=0;for(int seg=0;seg<segments;++seg){const auto&n=math[2*seg];const auto&s=math[2*seg+1];assert(n.kind=="norm"&&s.kind=="shift"&&n.input==0x100000u+offset*2048&&n.output==0x200000u+offset*2048&&s.input==n.output&&s.output==n.output&&s.weight==n.weight+2048&&n.mode==int(m.rmsnorm_route(n.rows,1024)));offset+=n.rows;}
  assert(offset==32);
 }
 for(bool acc32:{false,true}){
  Model x=m;x.linear_acc32_=acc32;x.action_mod_body_fp16_=true;x.action_step_active_=x.action_loop_active_=true;x.action_num_steps_=10;
  KernelCache::instance().available=false;x.prepare_action_newton_norm_shift();assert(!x.action_newton_norm_shift_available_);
  x.plan_norm();assert(x.routes.size()==1&&x.routes[0].site==QWEN35_ACTION_RMS_NORM_SHIFT_SITE);
  KernelCache::instance().available=true;const int before=KernelCache::instance().queries;
  x.action_rtc_prefix_len_=5;math.clear();x.apply_adaptive_norm(0x100000,0x200000,{});assert(math.size()==4&&KernelCache::instance().queries==before);
  x.prepare_action_newton_norm_shift();assert(x.action_newton_norm_shift_available_);const int cold=KernelCache::instance().queries;
  x.plan_norm();assert(x.routes.size()==1&&x.routes[0].site==QWEN35_ACTION_NEWTON_NORM_SHIFT_SITE&&x.routes[0].arguments.size()==24);
  assert(x.routes[0].arguments==std::vector<int64_t>({1,32,1024,8,24,acc32?32:16,0x11,2,2048,6144,0,32,1,1,16,32,7,8,9,2,2,0x4492,0x0092,1}));
  const auto policy=x.routes;
  for(uint32_t phase_offset:{0u,12288u})for(bool inplace:{false,true})for(int clean=0;clean<=32;++clean){
   x.adaptive_mod_row_offset_=phase_offset;x.action_rtc_prefix_len_=clean;x.plan_norm();assert(x.routes[0].arguments==policy[0].arguments);
   const uint32_t input=0x100000,output=inplace?input:0x200000;math.clear();const int receipts=x.consumed;x.apply_adaptive_norm(input,output,{});assert(x.consumed==receipts+1);
   size_t call=0;auto segment=[&](int offset,int rows,uint32_t scale){if(!rows)return;const auto route=x.rmsnorm_route(rows,1024);const auto&n=math.at(call++);
    assert(n.input==input+offset*2048&&n.output==output+offset*2048&&n.rows==rows&&n.cols==1024&&n.weight==scale&&n.mode==int(route));
    if(route==RpuRmsNormSpmRoute::NEWTON)assert(n.kind=="newton-shift");
    else{assert(n.kind=="norm");const auto&s=math.at(call++);assert(s.kind=="shift"&&s.input==n.output&&s.output==n.output&&s.weight==scale+2048&&s.rows==rows);}
   };
   segment(0,clean,0x600000+phase_offset);segment(clean,32-clean,0x600000+phase_offset+6144);assert(call==math.size()&&KernelCache::instance().queries==cold&&x.linear_acc32_==acc32);
  }
  x.adaptive_mod_row_offset_=0;x.action_rtc_prefix_len_=13;x.plan_norm({32,8,4});math.clear();x.apply_adaptive_norm(0x100000,0x200000,{32,8,4});assert(math.size()==2&&math[0].rows==5&&math[1].rows==27&&math[1].input==0x100000+5*2048);
  const auto good=x.routes;for(size_t i=0;i<24;++i){++x.routes[0].arguments[i];rejects([&]{x.apply_adaptive_norm(0x100000,0x200000,{32,8,4});});x.routes=good;}
  ++x.routes[0].selector;rejects([&]{x.apply_adaptive_norm(0x100000,0x200000,{32,8,4});});x.routes=good;
  ++x.routes[0].invocation;rejects([&]{x.apply_adaptive_norm(0x100000,0x200000,{32,8,4});});x.routes=good;
  x.complete=false;rejects([&]{x.apply_adaptive_norm(0x100000,0x200000,{32,8,4});});x.complete=true;
  for(int fault=0;fault<6;++fault){auto z=x;if(fault==0)z.action_rtc_active_=false;if(fault==1)z.action_step_active_=false;if(fault==2)z.action_loop_active_=false;if(fault==3)z.action_num_steps_=1;if(fault==4)z.action_len_=16;if(fault==5)z.action_mode_=false;assert(!z.use_action_newton_norm_shift(32));rejects([&]{z.apply_action_newton_norm_shift(0x100000,0x200000,{});});}
  {auto z=x;z.action_rtc_active_=false;z.plan_norm();math.clear();z.apply_adaptive_norm(0x100000,0x200000,{});assert(math.size()==2&&math[0].kind=="norm"&&math[0].rows==32&&math[0].mode==9&&math[1].kind=="shift");}
  rejects([&]{x.apply_action_newton_norm_shift(0x100000,0x200000,{31,0,0});});
  for(int fault=0;fault<14;++fault){auto z=x;if(fault==0)z.high_precision_=false;if(fault==1)z.action_mode_=false;if(fault==2)z.has_gdn_=true;if(fault==3)z.h=2048;if(fault==4)z.inter=6144;if(fault==5)z.layers=23;if(fault==6)z.heads=16;if(fault==7)z.kv=4;if(fault==8)z.hd=128;if(fault==9)z.cores=4;if(fault==10)z.tp=4;if(fault==11)z.mt=4;if(fault==12)z.eps_=1e-5;if(fault==13)z.eps_=NAN;const int queries=KernelCache::instance().queries;z.prepare_action_newton_norm_shift();assert(!z.action_newton_norm_shift_available_&&KernelCache::instance().queries==queries);}
 }
 m.action_mode_=false;m.has_gdn_=true;m.h=2048;m.inter=6144;m.gdn_nvh_=16;m.gdn_dk_=m.gdn_dv_=128;
 math.clear();m.gdn_tail(false);assert(math.size()==3&&math[0].kind=="norm"&&math[0].mode==9&&math[1].kind=="silu"&&math[1].mode==1&&math[2].kind=="mul");
 math.clear();m.gdn_tail(true);assert(math.size()==3&&math[0].kind=="norm"&&math[0].mode==7&&math[1].kind=="silu"&&math[1].mode==1&&math[2].kind=="mul");
 assert(m.linear_acc32_); // HIGH did not rewrite the independent Linear mode.
}
'''
    for marker, value in {
        "PRECISION": "enum class RpuUnaryPrecision : uint8_t " + _block(_read("src/core/rpu_math_precision.h"), "enum class RpuUnaryPrecision") + ";",
        "NORM_ENUM": "enum class RpuRmsNormSpmRoute : int64_t " + _block(decls, "enum class RpuRmsNormSpmRoute") + ";",
        "RESOLVER": _block(decls, "inline RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route("),
        "CONSTANTS": constants,
        "METHODS": "\n".join(declaration + _block(source, stem) for declaration, stem in methods),
        "PRODUCER": producer,
        "NORM_PRODUCER": norm_producer,
        "ADMISSION": admission,
        "GDN_TAIL": tail,
    }.items():
        cpp = cpp.replace(f"__{marker}__", value)
    _compile_run(tmp_path, "g05_high_precision", cpp)


def test_exact_g05_uses_canonical_activation_kernels_without_source_policy():
    decls = _read("src/core/rpu_kernel_decls.h")
    unary = _read("src/ops/rpu_eltwise_unary.cpp")
    recurrent = _read("src/ops/rpu_gdn_recurrent.cpp")
    text_cpp = _read("src/fused/rpu_qwen3_5_model.cpp")
    vision_cpp = _read("src/fused/rpu_qwen3_5_vision_model.cpp")
    adapter = _read("python/rpu_backend/adapters/g05/runtime.py")
    action = _read("python/rpu_backend/adapters/g05/action.py")
    dispatch = _read("src/core/rpu_dispatch_registrations.inc")
    assert "UnaryKernelPolicy" not in "\n".join((decls, unary, recurrent))
    assert "GeluMode::LEGACY_TANH" not in "\n".join(
        (decls, unary, text_cpp, vision_cpp)
    )
    assert "g05_k1_source_gelu" not in "\n".join(
        (decls, unary, recurrent, text_cpp, vision_cpp, adapter, action, dispatch)
    )
    assert "RPU_G05_K1_LEGACY" not in "\n".join(
        (decls, unary, recurrent, text_cpp, vision_cpp, adapter, action)
    )

    # Text keeps the dedicated latest Softplus in decode and prefill.
    assert "KernelId::UNARY_SOFTPLUS_SPM" in unary
    assert "ValuOpType::SOFTPLUS" in text_cpp
    assert "ValuOpType::SOFTPLUS" in recurrent

    # Vision block GELU is tanh; merger exact GELU is erf.
    assert vision_cpp.count("GeluMode::TANH") >= 2
    assert "GeluMode::ERF" in vision_cpp

    text_op = "qwen3_5_set_g05_k1_source_activations"
    vision_op = "qwen3_5_vision_set_g05_k1_source_activations"
    assert text_op not in "\n".join((decls, text_cpp, adapter, action, dispatch))
    assert vision_op not in "\n".join((decls, vision_cpp, adapter, dispatch))
    assert "_rpu_g05_k1_source_activations" not in adapter
    # This flag still owns numeric admission, chunking, and reduction policy.
    assert 'vlm_install_kwargs["_allow_certified_g05_k1"] = True' in adapter

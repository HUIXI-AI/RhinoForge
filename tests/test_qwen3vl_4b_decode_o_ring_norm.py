"""Execute exact O/ring/BASE norm manifest, consumer and cold scale ownership."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest
from test_qwen3vl_prefill_gateup_launcher import definition

ROOT = Path(__file__).resolve().parents[1]


def test_o_manifest_and_dispatch(tmp_path):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('g++ required')
    source = (ROOT/'src/fused/rpu_qwen3_model.h').read_text()
    base = (ROOT/'src/core/fused_model_base.h').read_text()
    planner = (ROOT/'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    enums = '\n'.join(definition(text, f'enum class {name}')+';' for text, name in (
        (base, 'ChunkMode'), (base, 'FmbPhysicalManifestState'), (base, 'FmbGraphLifecycle'),
        (base, 'FmbLinearAccumulationPolicy'), (base, 'FmbRouteFamily'), (base, 'FmbLinearRouteSelector'),
        (base, 'FmbSharedAllReduceRouteSelector'), (base, 'AttentionExecutionPolicy'),
        (source, 'CausalDecoderLinearRoute'), (source, 'CausalDecoderAllReduceRoute'),
        (source, 'CausalDecoderPlanningMode')))
    structs = '\n'.join(definition(base, f'struct {name}')+';' for name in (
        'ChunkInfo', 'FmbRouteManifestEntry', 'FmbPhysicalExecutionManifest'))
    constants = '\n'.join(re.findall(r'inline constexpr int64_t (?:QWEN3VL_4B_DECODE_O_RING_NORM|CAUSAL_DECODER_(?:PREPARE|O|REDUCE|MIXED_REDUCE|WALL_RING_O|WALL_RING_DOWN|WALL_BASE_NVFP4_O_RING_NORM))_SITE\s*=\s*[^;]+;', source))
    manifest = source.index('FmbPhysicalExecutionManifest physical_manifest_for_candidate(')
    start = source.index('        const bool qwen3vl4b_fused_o =', manifest)
    select = source[start:source.index('        const bool qwen3vl4b_fused_qk_kv =', start)]
    start = source.index('            if (qwen3vl4b_fused_o)', manifest)
    routes = source[start:source.index('            if (!deepstack_lang_layers_', start)]
    ring = definition(source[source.index('if (!wall_action_linear_ring && !qwen3vl4b_fused_o &&', manifest):],
                      'if (!wall_action_linear_ring && !qwen3vl4b_fused_o &&')
    start = source.index('        auto append =', manifest)
    append = source[start:source.index('        auto append_attention =', start)]
    start = source.index('        auto append_linear =', manifest)
    helpers = source[start:source.index('        auto append_kv =', start)]
    dispatch_start = source.index('        if (ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&', source.index('// Layer 0 establishes a full residual2'))
    dispatch = definition(source[dispatch_start:], 'if (ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&')
    consumer = '\n'.join(definition(planner, marker).replace('FmbPhysicalManifestConsumer::', '').replace('consume_route(', 'consume_physical_route(') for marker in (
        'bool FmbPhysicalManifestConsumer::complete() const',
        'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(',
        'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route('))
    program = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>
#include "src/ops/rpu_linear_tiling.h"
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error(#c);}while(false)
#define TORCH_INTERNAL_ASSERT(c) assert(c)
namespace at {using IntArrayRef=std::vector<int64_t>;enum ScalarType{kChar,kHalf};
struct Tensor{int id;ScalarType dtype=kChar;bool defined()const{return true;}ScalarType scalar_type()const{return dtype;}};}
__ENUMS__
__STRUCTS__
__CONSTANTS__
struct Stage{std::vector<ChunkInfo> chunks;};
struct Plan{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;Stage compute;};
using FmbThreeStageChunkPlan=Plan;
enum class KernelId{W8A16_ROW_WEIGHT_REUSE_M352N80K256};
struct KernelCache{static KernelCache& instance(){static KernelCache cache;return cache;}bool has_loaded(KernelId)const{return false;}};
struct Layout{bool is_causal=true,use_attn_mask=false;int64_t batch_size=1;AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::AUTO;};
struct Context{
 FmbPhysicalExecutionManifest manifest_;mutable std::vector<uint8_t> consumed_;
 const FmbPhysicalExecutionManifest& manifest()const{return manifest_;}
 Plan stage_plan;int64_t batch_size=1;bool is_causal=true;std::optional<int> attention_mask;
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
 __CONSUMER__
 void bind(FmbPhysicalExecutionManifest m){std::sort(m.routes.begin(),m.routes.end(),[](auto&a,auto&b){return std::make_tuple(int64_t(a.family),a.site_id,a.invocation)<std::make_tuple(int64_t(b.family),b.site_id,b.invocation);});manifest_=m;consumed_.assign(m.routes.size(),0);}
};
struct LayerWeights{at::Tensor o_w{1},o_ws{2,at::kHalf};};
int calls=0,fallback=0,live_layer=0;uint32_t live_base=0x40000000;
void rpu_launch_qwen3vl_4b_decode_o_ring_norm_spm_kernel(uint32_t input,const at::Tensor& w,const at::Tensor& scale,
 uint32_t partial,uint32_t residual,uint32_t raw,uint32_t gamma,double eps){
 assert(input==live_base+0x1000&&partial==live_base+0x3000&&residual==live_base+0x5000&&raw==live_base+0x7000);
 assert(w.id==1&&scale.id==2&&gamma==live_base+0x9000+live_layer*5120&&eps==1e-6);++calls;
}
struct CausalDecoderModel{
 bool qwen3vl_fp16_decode_profile_=false;
 bool qwen3vl_4b_w8_profile_=true,qwen3vl_4b_decode_o_ring_norm_fused_=true;
 bool linear_acc32_=false,decoder_mixed_precision_=false,adarms_=false,wall_z1_dispatch_=false;
 bool qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false,partial_mrope_active_=false;
 bool has_mrope_=true,has_qkv_bias_=false,nvfp4_=false,wall_prefill_staged_mlp_=false;
 bool decode_gate_up_fusion_=false,pre_layers_residual1_ready_=false;
 std::vector<int> deepstack_lang_layers_;std::vector<LayerWeights> layer_weights_{1};
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 int h=2560,qh=32,kvh=8,hd=128,cores=8,tp=8;int64_t seq_len_=1;double eps_=1e-6;Context context;
 int hidden_size()const{return h;}int num_q_heads()const{return qh;}int num_kv_heads()const{return kvh;}
 int head_dim()const{return hd;}int num_cores()const{return cores;}int attn_tp()const{return tp;}
 Context& ctx(){return context;}
 uint32_t addr(int core,const char* name)const{assert(core==0);std::string n=name;return live_base+(n=="output"?0x1000:n=="oproj"?0x3000:n=="residual1"?0x5000:0x7000);}
 uint32_t norm_post_addr(int layer,int core)const{assert(core==0);return live_base+0x9000+layer*5120;}
 std::vector<int64_t> bind_cold_topology_arguments(std::vector<int64_t> v)const{return v;}
 std::vector<int64_t> wall_base_nvfp4_o_ring_norm_arguments(int64_t)const{throw std::runtime_error("unexpected Wall route");}
 int64_t causal_ring_route(int64_t,int64_t)const{return 1;}
 __PREDICATES__
 FmbPhysicalExecutionManifest selected(const Plan& plan,const Layout& layout,int64_t physical_len,bool partial_mrope){
  const int64_t position=0;
  FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;bool wall_base_nvfp4_o_ring_norm = false;
    bool wall_action_linear_ring=false;
  __APPEND__
  __HELPERS__
  __SELECT__
  for(const auto& chunk:plan.compute.chunks){__RING__}
  __ROUTES__
  return manifest;
 }
 __EMITTER__
 void dispatch(const LayerWeights& lw,int layer_idx,const ChunkInfo& chunk,uint32_t input_residual){
  const auto bs=ctx().batch_size;
  __DISPATCH__ else {++fallback;}
 }
};
using Model=CausalDecoderModel;
Plan plan_for(int rows){Plan p;p.compute.chunks.push_back({0,0,rows,rows});return p;}
void rejects(std::function<void()> fn){calls=0;bool bad=false;try{fn();}catch(const std::runtime_error&){bad=true;}assert(bad&&calls==0);}
int main(){
 Model m;Layout layout;Plan plan=plan_for(1);m.context.stage_plan=plan;LayerWeights w;
 for(int change:{0,1,0}){live_base=0x40000000+change*0x10000;live_layer=change;
  auto mf=m.selected(plan,layout,1,false);assert(mf.routes.size()==1&&mf.routes[0].site_id==QWEN3VL_4B_DECODE_O_RING_NORM_SITE);
  m.context.bind(mf);calls=fallback=0;m.dispatch(w,live_layer,plan.compute.chunks[0],m.addr(0,"residual1"));
  assert(calls==1&&fallback==0&&m.context.consumed_[0]==1);
 }
 const auto canonical=m.selected(plan,layout,1,false);
 for(int fault=0;fault<8;++fault){auto bad=canonical;auto& r=bad.routes[0];
  if(fault==0)r.site_id++;if(fault==1)r.selector=2;if(fault==2)r.flags=1;if(fault==3)r.invocation=1;
  if(fault==4)r.arguments[3]=1216;if(fault==5)r.family=FmbRouteFamily::ALL_REDUCE;
  if(fault==6)bad.routes.clear();if(fault==7)bad.state=FmbPhysicalManifestState::UNSPECIFIED;
  m.context.bind(bad);rejects([&]{m.dispatch(w,0,plan.compute.chunks[0],m.addr(0,"residual1"));});
 }
 m.context.bind(canonical);rejects([&]{m.emit_qwen3vl_4b_decode_o_ring_norm(w,0,plan.compute.chunks[0],m.addr(0,"residual2"));});
 auto excluded=[&](Model bad,Layout l,int physical,int rows,bool partial,ChunkMode mode){
  auto p=plan_for(rows);p.chunk_mode=mode;bad.seq_len_=physical;bad.context.stage_plan=p;bad.context.is_causal=l.is_causal;
  bad.context.batch_size=l.batch_size;bad.partial_mrope_active_=partial;
  if(l.use_attn_mask)bad.context.attention_mask=1;
  if(l.attention_policy==AttentionExecutionPolicy::SPM_KV_BY_MHA)bad.context.attention_policy=l.attention_policy;
  auto mf=bad.selected(p,l,physical,partial);assert(std::none_of(mf.routes.begin(),mf.routes.end(),[](auto&r){return r.site_id==QWEN3VL_4B_DECODE_O_RING_NORM_SITE;}));
  assert(std::any_of(mf.routes.begin(),mf.routes.end(),[](auto&r){return r.site_id==CAUSAL_DECODER_O_SITE;}));
  assert(std::any_of(mf.routes.begin(),mf.routes.end(),[](auto&r){return r.site_id==CAUSAL_DECODER_REDUCE_SITE;}));
  bad.context.bind(mf);calls=fallback=0;bad.dispatch(w,0,p.compute.chunks[0],bad.addr(0,"residual1"));assert(calls==0&&fallback==1);
  rejects([&]{bad.emit_qwen3vl_4b_decode_o_ring_norm(w,0,p.compute.chunks[0],bad.addr(0,"residual1"));});
 };
 for(auto field:{&Model::qwen3vl_4b_w8_profile_,&Model::qwen3vl_4b_decode_o_ring_norm_fused_}){Model b;b.*field=false;excluded(b,layout,1,1,false,ChunkMode::SEQUENTIAL);}
 for(auto field:{&Model::linear_acc32_,&Model::decoder_mixed_precision_,&Model::adarms_,&Model::wall_z1_dispatch_,&Model::qwen3vl_pooler_z1_dispatch_,&Model::qwen3vl_multiview_text_composite_dispatch_}){Model b;b.*field=true;excluded(b,layout,1,1,false,ChunkMode::SEQUENTIAL);}
 for(auto field:{&Model::h,&Model::qh,&Model::kvh,&Model::hd,&Model::cores,&Model::tp}){Model b;b.*field/=2;excluded(b,layout,1,1,false,ChunkMode::SEQUENTIAL);}
 for(auto shape:std::vector<std::pair<int,int>>{{16,16},{256,256},{1,2},{2,1}})excluded(Model{},layout,shape.first,shape.second,false,ChunkMode::SEQUENTIAL);
 {Model b;b.eps_=1e-5;excluded(b,layout,1,1,false,ChunkMode::SEQUENTIAL);}
 {Model b;b.planning_mode_=CausalDecoderPlanningMode::WALL_Z1;excluded(b,layout,1,1,false,ChunkMode::SEQUENTIAL);}
 {auto l=layout;l.is_causal=false;excluded(Model{},l,1,1,false,ChunkMode::SEQUENTIAL);}
 {auto l=layout;l.use_attn_mask=true;excluded(Model{},l,1,1,false,ChunkMode::SEQUENTIAL);}
 {auto l=layout;l.batch_size=2;excluded(Model{},l,1,1,false,ChunkMode::SEQUENTIAL);}
 {auto l=layout;l.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;excluded(Model{},l,1,1,false,ChunkMode::SEQUENTIAL);}
 excluded(Model{},layout,1,1,true,ChunkMode::SEQUENTIAL);excluded(Model{},layout,1,1,false,ChunkMode::KV_FIRST);
}
'''
    values = dict(ENUMS=enums, STRUCTS=structs, CONSTANTS=constants, CONSUMER=consumer,
                  PREDICATES='\n'.join(definition(source, name) for name in (
                      'bool qwen3vl_decode_gemv(', 'bool qwen3vl_decode_mlp(',
                      'bool qwen3vl_4b_decode_o_ring_norm(', 'CausalDecoderLinearRoute decoder_projection_route(',
                      'bool ordinary_row_weight_reuse_plan_enabled(', 'std::vector<int64_t> ordinary_o_route_arguments(')),
                  APPEND=append, HELPERS=helpers, SELECT=select, ROUTES=routes, RING=ring,
                  EMITTER=definition(source, 'void emit_qwen3vl_4b_decode_o_ring_norm('), DISPATCH=dispatch)
    for key, value in values.items(): program = program.replace(f'__{key}__', value)
    cpp, exe = tmp_path/'o_route.cpp', tmp_path/'o_route'
    cpp.write_text(program)
    result = subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-Wno-misleading-indentation', '-I', str(ROOT), str(cpp), '-o', str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


def test_o_scale_backing(tmp_path):
    import torch
    source = (ROOT/'src/fused/rpu_qwen3_model.h').read_text()
    method = definition(source, 'static at::Tensor retain_qwen3vl_4b_ring_norm_scale(')
    install = definition(source[source.index('            if (qwen3vl_4b_decode_o_ring_norm_fused_)'):], 'if (qwen3vl_4b_decode_o_ring_norm_fused_)')
    program = r'''
#include <ATen/ATen.h>
#include <cassert>
#include <cstring>
#include <vector>
size_t flushed=0;void rpu_ddr_flush_force_sized(void*,size_t n){flushed+=n;}
struct Model{bool qwen3vl_4b_decode_o_ring_norm_fused_=false;struct W{at::Tensor o_ws,down_ws;};std::vector<W> layer_weights_;
__METHOD__
void install(){__INSTALL__}
};
int main(){
 auto source=(at::arange(2560,at::kFloat)/4096+.01).to(at::kHalf);auto original=source.clone();
 Model old;old.layer_weights_.push_back({source,source});old.install();assert(old.layer_weights_[0].o_ws.is_same(source)&&flushed==0);
 Model m;m.qwen3vl_4b_decode_o_ring_norm_fused_=true;m.layer_weights_.push_back({source,source});m.install();auto padded=m.layer_weights_[0].o_ws;
 assert(padded.dim()==1&&padded.numel()==2560&&padded.is_contiguous());assert(at::equal(padded,source)&&at::equal(source,original));
 assert(padded.storage().unsafeGetStorageImpl()!=source.storage().unsafeGetStorageImpl());assert(padded.storage().nbytes()==2816*2&&flushed==2*2816*2);
 assert(at::equal(m.layer_weights_[0].down_ws,source)&&m.layer_weights_[0].down_ws.storage().nbytes()==2816*2);
 auto* data=padded.data_ptr<c10::Half>();for(int i=2560;i<2816;++i)assert(data[i]==c10::Half(0));
 for(const auto& bad:std::vector<at::Tensor>{source.to(at::kFloat),source.narrow(0,0,2559),source.view({1,2560}),at::empty({5120},at::kHalf).slice(0,0,5120,2)}){
  bool rejected=false;try{Model::retain_qwen3vl_4b_ring_norm_scale(bad);}catch(const c10::Error&){rejected=true;}assert(rejected);
 }
}
'''.replace('__METHOD__', method).replace('__INSTALL__', install)
    cpp, exe = tmp_path/'o_scale.cpp', tmp_path/'o_scale'; cpp.write_text(program)
    torch_root = Path(torch.__file__).resolve().parent
    subprocess.run(['g++', '-std=c++17', '-O0', f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
                    '-I', str(torch_root/'include'), str(cpp), '-L', str(torch_root/'lib'), '-ltorch_cpu', '-lc10',
                    f'-Wl,-rpath,{torch_root / "lib"}', '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)

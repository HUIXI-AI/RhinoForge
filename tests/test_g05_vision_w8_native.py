"""Execute actual cold, descriptor and callback code for the private W8 Vision route."""
import importlib.util
import json

import pytest

from test_g05_vision_owner_model import _block
from test_g05_vision_qkv_spm_native import ROOT, _compile_qkv, _qkv_native_cpp


def _w8_native_cpp():
    source = (ROOT / 'src/fused/rpu_qwen3_5_vision_model.cpp').read_text()
    cpp = _qkv_native_cpp().split('int main(){', 1)[0]
    cpp = cpp.replace('struct Weights{', 'namespace at{using TensorList=std::vector<Tensor>;}\nstruct Weights{')
    marker = ' std::vector<int64_t>kvinsert_cost_weight_identity()const'
    extra = (' int resolved=0,invalidations=0;int get_last_resolved_chunk_size()const{return resolved;}void invalidate_model_state(){++invalidations;}\n'
             'void set_weight_scales(at::TensorList q,at::TensorList k,at::TensorList v,at::TensorList o,at::TensorList fc1,at::TensorList fc2)'
             + _block(source, 'void Qwen3_5VisionModel::set_weight_scales('))
    cpp = cpp.replace(marker, extra + marker)
    return cpp


@pytest.mark.parametrize("acc32", [True, False])
def test_vision_w8_cold_scales_complete_routes_and_old_ref_fallback(tmp_path, acc32):
    cpp = _w8_native_cpp()
    cpp += r'''
int main(){
 Model m;LayoutContext layout;auto&cache=KernelCache::instance();cache.w8=cache.acc16=cache.owner=true;m.linear_acc32_=__ACC32__;const std::string triple=m.linear_acc32_?"qkv-w8":"qkv-w8-acc16";
 std::vector<at::Tensor>scales[6];
 for(int layer=0;layer<24;++layer){auto&w=m.layer_weights_[layer];at::Tensor*weights[]={&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.fc1_w,&w.fc2_w};
  for(int i=0;i<6;++i){auto&t=*weights[i];t.dtype=at::kChar;t.owner=1000+6*layer+i;t.track_owner=true;t.rows=i==4?4096:1024;t.cols=i==5?4096:1024;auto scale=t;scale.dtype=at::kHalf;scale.rank=1;scale.owner+=10000;scales[i].push_back(scale);}}
 m.has_step0_=m.has_rope_=m.has_merger_=false;
 m.prepare_g05_owner_ln();assert(!m.g05_qkv_spm_available_&&!m.g05_w8a16_); // INT8 alone is never sufficient.
 m.set_weight_scales(scales[0],scales[1],scales[2],scales[3],scales[4],scales[5]);
 assert(m.invalidations==1&&m.g05_w8a16_&&m.g05_owner_ln_available_&&m.g05_qkv_spm_available_&&m.g05_qkv_linear_available_&&!m.is_g05_qkv_spm_profile());
 m.has_step0_=m.has_rope_=m.has_merger_=true;m.plan(layout);
 auto hash=m.subclass_layout_hash();auto identity=m.kvinsert_cost_weight_identity();
 const auto&r=m.context.find_physical_route(FmbRouteFamily::LINEAR,G05_VISION_QKV_W8_LINEAR_SITE,0);
 assert(r.arguments==std::vector<int64_t>({2,256,128,1024,8,m.linear_acc32_?32:16,m.linear_acc32_?400:512,48,128,3,9,2048,256,3,256,1,8,3,1,2,32,40,20,34,42,4,36,44,6,38,46}));
 {auto other=m;other.linear_acc32_=!m.linear_acc32_;other.prepare_g05_owner_ln();assert(other.g05_qkv_linear_available_&&other.subclass_layout_hash()!=hash&&other.kvinsert_cost_weight_identity()!=identity);rejects_before_emission([&]{other.emit_kv_first_body(0,{});});}
 auto meta=at::Tensor::metadata_reads;auto queries=cache.queries;
 for(int layer=0;layer<24;++layer){emitted.clear();m.emit_kv_first_body(layer,{});m.compute(layer,{});assert(emitted==std::vector<std::string>({"ln",triple,"rope","rope"}));}
 assert(meta==at::Tensor::metadata_reads&&queries==cache.queries);
 for(size_t i=0;i<r.arguments.size();++i){auto x=m;for(auto&route:x.context.routes)if(route.site==G05_VISION_QKV_W8_LINEAR_SITE)++route.arguments[i];rejects_before_emission([&]{x.emit_kv_first_body(0,{});});}
 {auto x=m;x.context.complete=false;rejects_before_emission([&]{x.emit_kv_first_body(0,{});});}
 {auto x=m;x.context.routes.erase(x.context.routes.begin());rejects_before_emission([&]{x.emit_kv_first_body(0,{});});}
 {auto x=m;x.alias=false;rejects_before_emission([&]{x.emit_kv_first_body(0,{});});}
 // Every scale/weight on all 24 layers participates in the cached proof.
 for(int layer=0;layer<24;++layer)for(int i=0;i<6;++i)for(int fault=0;fault<4;++fault){auto x=m;auto&w=x.layer_weights_[layer];at::Tensor*ts[]={&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.fc1_ws,&w.fc2_ws};auto&t=*ts[i];if(fault==0)t.present=false;if(fault==1)t.dtype=at::kChar;if(fault==2)t.rows--;if(fault==3)t.contiguous=false;x.prepare_g05_owner_ln();assert(!x.g05_w8a16_&&!x.g05_qkv_spm_available_&&!x.g05_qkv_linear_available_&&!x.g05_owner_ln_available_);}
 {auto x=m;x.layer_weights_.back().fc2_w.dtype=at::kHalf;x.prepare_g05_owner_ln();assert(!x.g05_w8a16_&&!x.g05_qkv_spm_available_);}
 {auto x=m;x.layer_weights_.back().fc2_ws.owner++;assert(x.kvinsert_cost_weight_identity()!=identity);}
 // Failed scale publication is atomic and resolves no new execution state.
 {auto x=m;auto bad=scales[5];bad.back().rows--;auto old=x.kvinsert_cost_weight_identity();try{x.set_weight_scales(scales[0],scales[1],scales[2],scales[3],scales[4],bad);assert(false);}catch(const std::runtime_error&){}assert(x.invalidations==1&&x.kvinsert_cost_weight_identity()==old);}
 {auto x=m;x.resolved=256;try{x.set_weight_scales(scales[0],scales[1],scales[2],scales[3],scales[4],scales[5]);assert(false);}catch(const std::runtime_error&){}assert(x.invalidations==1);}
 if(m.linear_acc32_)cache.w8=false;else cache.acc16=false;m.prepare_g05_owner_ln();m.plan(layout);assert(m.g05_w8a16_&&m.g05_owner_ln_available_&&m.g05_qkv_spm_available_&&!m.g05_qkv_linear_available_&&m.subclass_layout_hash()!=hash);
 emitted.clear();m.emit_kv_first_body(23,{});m.compute(23,{});assert(emitted==std::vector<std::string>({"ln","linear","linear","linear","rope","rope"}));
 cache.w8=cache.acc16=true;m.prepare_g05_owner_ln();
 for(auto policy:{AttentionExecutionPolicy::SPM_KV_BY_MHA,AttentionExecutionPolicy::DDR_KV,AttentionExecutionPolicy::SPM_KV_BY_MHA}){m.context.attention_policy=layout.attention_policy=policy;m.plan(layout);emitted.clear();m.emit_kv_first_body(23,{});m.compute(23,{});if(policy==AttentionExecutionPolicy::DDR_KV)assert(emitted==std::vector<std::string>({"ln","linear","linear","linear","rope","rope","kv-ddr","q-store","q-load"}));else assert(emitted==std::vector<std::string>({"ln",triple,"rope","rope"}));}
 m.current_output_patches_=128;m.plan(layout);emitted.clear();m.emit_kv_first_body(23,{});assert(std::count(emitted.begin(),emitted.end(),"linear")==3);
}
'''
    # Generic fallback must receive each scale unchanged, rather than silently
    # falling back to FP16 weights/empty scales.
    cpp = cpp.replace('const at::Tensor&,uint32_t,int64_t,int64_t,int64_t,int,int,uint32_t,const at::Tensor&,int,int,bool){emitted.push_back("linear");}',
                      'const at::Tensor&w,uint32_t,int64_t,int64_t,int64_t,int,int,uint32_t,const at::Tensor&s,int,int,bool){assert(w.dtype==at::kChar&&s.owner==w.owner+10000);emitted.push_back("linear");}')
    cpp = cpp.replace("__ACC32__", "true" if acc32 else "false")
    _compile_qkv(tmp_path, cpp)


def test_vision_w8_acc16_cold_scales_and_ordinary_precision(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_5_vision_model.cpp').read_text()
    cpp = _w8_native_cpp()
    manifest = _block(source, 'Qwen3_5VisionModel::physical_manifest_for_candidate(')
    start = manifest.index('    manifest.linear_accumulation =')
    accumulation = manifest[start:manifest.index(';', start) + 1]
    layer = _block(source, 'void Qwen3_5VisionModel::build_layer_subgraph(')
    tail = layer[layer.index('    // Phase 4: O_proj'):layer.index('    const bool z2_final')]
    extra = '''
 FmbLinearAccumulationPolicy accumulation()const{
  struct {FmbLinearAccumulationPolicy linear_accumulation;}manifest;
  __ACCUMULATION__
  return manifest.linear_accumulation;
 }
 bool use_g05_owner_ln(int64_t rows)const __OWNER_PREDICATE__
 void launch_g05_owner_ln(int,const ChunkInfo&){throw std::runtime_error("ACC32 owner route");}
 void launch_g05_compact_residual(const ChunkInfo&){throw std::runtime_error("ACC32 compact route");}
 void post_attention(int layer_idx,const ChunkInfo&chunk){
  int64_t seq_len=chunk.len,h=hidden_size(),nq=num_q_heads(),hd=head_dim(),is_=intermediate_size(),local_inter=is_/num_cores();
  int tp=attn_tp();const auto&lw=layer_weights_[layer_idx];
  __TAIL__
 }
'''.replace('__ACCUMULATION__', accumulation).replace('__TAIL__', tail).replace(
        '__OWNER_PREDICATE__', _block(source, 'bool Qwen3_5VisionModel::use_g05_owner_ln('))
    cpp = cpp.replace(' std::vector<int64_t>kvinsert_cost_weight_identity()const',
                      extra + ' std::vector<int64_t>kvinsert_cost_weight_identity()const')
    cpp = cpp.replace('enum class FmbRouteFamily{', 'enum class FmbRouteFamily{ALL_REDUCE,')
    cpp = cpp.replace('struct Model{', '''
enum class FmbLinearAccumulationPolicy{ACC16,ACC32};
enum class Qwen35VisionAllReduceRoute{PREPARE_RING_INPUT};
enum class ValuOpType{ADD};enum class GeluMode{TANH};
enum class RpuUnaryPrecision{BASE,HIGH};
int64_t qwen35_vision_ring_route(int64_t,int64_t,int){return 7;}
template<class...A>void rpu_prepare_ring_all_reduce_input(A...){emitted.push_back("ring-init");}
template<class...A>void rpu_launch_all_reduce_sum_residual_kernel(A...){emitted.push_back("ring");}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t,uint32_t,int64_t,ValuOpType,GeluMode mode,int,RpuUnaryPrecision precision){assert(mode==GeluMode::TANH&&precision==RpuUnaryPrecision::BASE);emitted.push_back("gelu");}
struct Model{''')
    cpp = cpp.replace('context.consume_physical_route(f,s,select,0,{},inv);',
                      'if(context.complete)context.consume_physical_route(f,s,select,0,{},inv);')
    cpp = cpp.replace('std::vector<std::string>emitted;',
                      'std::vector<std::string>emitted;std::vector<bool>linear_modes;std::vector<int64_t>linear_owners;')
    cpp = cpp.replace(
        'const at::Tensor&,uint32_t,int64_t,int64_t,int64_t,int,int,uint32_t,const at::Tensor&,int,int,bool){emitted.push_back("linear");}',
        'const at::Tensor&w,uint32_t,int64_t,int64_t,int64_t,int,int,uint32_t,const at::Tensor&s,int,int,bool acc32){assert(w.dtype==at::kChar&&s.owner==w.owner+10000);linear_modes.push_back(acc32);linear_owners.push_back(w.owner);emitted.push_back("linear");}')
    cpp += r'''
int main(){
 Model m;LayoutContext layout;auto&cache=KernelCache::instance();cache.w8=cache.qkv=true;cache.owner=cache.acc16=false;
 assert(m.linear_acc32_&&m.accumulation()==FmbLinearAccumulationPolicy::ACC32);
 m.linear_acc32_=false;assert(m.accumulation()==FmbLinearAccumulationPolicy::ACC16);
 std::vector<at::Tensor>scales[6];
 for(int layer=0;layer<24;++layer){auto&w=m.layer_weights_[layer];at::Tensor*weights[]={&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.fc1_w,&w.fc2_w};
  for(int i=0;i<6;++i){auto&t=*weights[i];t.dtype=at::kChar;t.owner=1000+6*layer+i;t.rows=i==4?4096:1024;t.cols=i==5?4096:1024;auto scale=t;scale.dtype=at::kHalf;scale.rank=1;scale.owner+=10000;scales[i].push_back(scale);}}
 auto install=[&](Model&x){x.set_weight_scales(scales[0],scales[1],scales[2],scales[3],scales[4],scales[5]);};
 install(m);assert(m.invalidations==1&&!m.g05_owner_ln_available_&&m.g05_qkv_spm_available_&&!m.g05_qkv_linear_available_&&m.is_g05_k1_fast_replay_profile()&&!m.use_g05_owner_ln(256));
 assert(!m.gelu_erf_ultra_&&!m.moe_precision_&&m.eps_==1e-6);
 m.plan(layout);assert(m.context.find_physical_route(FmbRouteFamily::KV_INSERT,QWEN35_VISION_KV_SITE,0).flags==2);
 for(const auto&r:m.context.routes)assert(r.site!=G05_VISION_QKV_W8_LINEAR_SITE&&r.site!=G05_VISION_QKV_LINEAR_SITE);
 for(int layer=0;layer<24;++layer){
  emitted.clear();linear_modes.clear();linear_owners.clear();m.emit_kv_first_body(layer,{});m.compute(layer,{});
  assert(emitted==std::vector<std::string>({"ln","linear","linear","linear","rope","rope"}));
  // The shared harness plans QKV routes; execute the unchanged post-attention
  // body without manifest dispatch, whose routing is covered by the owner test.
  m.context.complete=false;m.post_attention(layer,{});m.context.complete=true;
  assert(linear_modes==std::vector<bool>(6,false));
  assert(linear_owners==std::vector<int64_t>({1000+6*layer,1001+6*layer,1002+6*layer,1003+6*layer,1004+6*layer,1005+6*layer}));
  assert(std::vector<std::string>(emitted.end()-8,emitted.end())==std::vector<std::string>({"ring-init","linear","ring","ln","linear","gelu","linear","ring"}));
 }
 auto rejected=[&](auto change){auto x=m;change(x);auto old=x.kvinsert_cost_weight_identity();auto n=x.invalidations;try{install(x);assert(false);}catch(const std::runtime_error&){}assert(x.invalidations==n&&x.kvinsert_cost_weight_identity()==old);};
 rejected([](auto&x){x.resolved=256;});rejected([](auto&x){x.cores=4;});rejected([](auto&x){x.eps_=1e-5;});rejected([](auto&x){x.gelu_erf_ultra_=true;});rejected([](auto&x){x.moe_precision_=true;});rejected([](auto&x){x.has_temporal_=true;});
 // Every scale and corresponding weight remains checked before publication.
 for(auto&projection:scales)for(auto&scale:projection)scale.owner+=20000;
 for(int layer=0;layer<24;++layer)for(int i=0;i<6;++i){
  for(int fault=0;fault<6;++fault){auto saved=scales[i][layer];auto&t=scales[i][layer];if(fault==0)t.present=false;if(fault==1)t.dtype=at::kChar;if(fault==2)t.location=0;if(fault==3)t.contiguous=false;if(fault==4)t.rank=2;if(fault==5)t.rows--;rejected([](auto&){});t=saved;}
  for(int fault=0;fault<5;++fault)rejected([&](auto&x){auto&w=x.layer_weights_[layer];at::Tensor*weights[]={&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.fc1_w,&w.fc2_w};auto&t=*weights[i];if(fault==0)t.present=false;if(fault==1)t.dtype=at::kHalf;if(fault==2)t.location=0;if(fault==3)t.contiguous=false;if(fault==4)t.rank=1;});
 }
 auto saved=scales[0].back();scales[0].pop_back();rejected([](auto&){});scales[0].push_back(saved);
 install(m);assert(m.invalidations==2&&!m.g05_owner_ln_available_&&!m.g05_qkv_linear_available_);
}
'''
    _compile_qkv(tmp_path, cpp)

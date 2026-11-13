"""Execute Wall Action accumulation plumbing and cold route guards on host."""
from pathlib import Path
import re

from test_wall_base_nvfp4_qkv_bias import META
from test_wall_base_paired_kv import block, compile_run

ROOT = Path(__file__).resolve().parents[1]


def _statement(source, marker, start=0):
    begin = source.index(marker, start)
    return source[begin:source.index(';', begin) + 1]


def test_actual_action_hooks_manifest_and_cold_setter(tmp_path):
    action = (ROOT / 'src/fused/rpu_wall_oss_action_step_model.cpp').read_text()
    decoder = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()
    action_header = (ROOT / 'src/fused/rpu_wall_oss_action_step_model.h').read_text()
    base = (ROOT / 'src/core/fused_model_base.h').read_text()
    planner = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    hooks = '\n'.join(block(action, marker).replace('WallOssActionStepModel::', '') for marker in (
        'void WallOssActionStepModel::emit_pre_layers_body()',
        'void WallOssActionStepModel::emit_post_layers_body()',
        'void WallOssActionStepModel::consume_action_route(',
        'WallOssActionStepModel::physical_manifest_for_candidate(',
    ))
    hooks = hooks.replace('std::vector<int64_t> arguments)', 'std::vector<int64_t> arguments = {})')
    # Out-of-class return type precedes the qualified function marker.
    hooks = hooks.replace('physical_manifest_for_candidate(\n', 'FmbPhysicalExecutionManifest physical_manifest_for_candidate(\n', 1)
    consume = '\n'.join(block(planner, marker).replace('FmbPhysicalManifestConsumer::', '') for marker in (
        'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(',
        'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(',
    ))
    constants = '\n'.join(re.findall(r'constexpr int64_t WALL_ACTION_\w+\s*=\s*\d+LL;', action))
    accumulation = _statement(decoder, 'manifest.linear_accumulation = (decoder_mixed_precision_')
    stubs = '\n'.join(f'#define {name}(...) ((void)0)' for name in sorted(
        set(re.findall(r'\b(rpu_launch_\w+)\(', hooks)) - {'rpu_launch_linear_spm_to_spm_acc16_kernel'}))
    source = META + r'''
#include <tuple>
#undef TORCH_INTERNAL_ASSERT
#define TORCH_INTERNAL_ASSERT(x) TORCH_CHECK(x)
FAMILY
SELECTOR
ACCUMULATION_ENUM
enum class FmbPhysicalManifestState { COMPLETE };
enum class FmbGraphLifecycle { COMPOSITE_CHILD };
struct FmbRouteManifestEntry {int64_t site_id;FmbRouteFamily family;int64_t selector,flags;
 std::vector<int64_t> arguments;int64_t invocation=0;};
struct FmbPhysicalExecutionManifest {std::vector<FmbRouteManifestEntry> routes;
 FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
 FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
 FmbLinearAccumulationPolicy linear_accumulation=FmbLinearAccumulationPolicy::ACC16;};
struct ChunkInfo {int64_t idx=0,offset=0,len=32,kv_seq_len=416;};
struct FmbThreeStageChunkPlan {struct{std::vector<ChunkInfo> chunks{{}};}qkv,compute;};
struct LayoutContext {bool use_attn_mask=true,is_causal=false;};
struct Context {int body_iter=0;bool ready=true;FmbPhysicalExecutionManifest value;mutable std::vector<uint8_t> consumed_;
 bool complete()const{return ready;}bool has_complete_physical_manifest()const{return ready;}
 const auto& manifest()const{return value;}
 CONSUME
 const auto& consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,at::IntArrayRef args){
  return consume_route(family,site,selector,flags,args,0);}
};
CONSTANTS
DMA_ENUM
LINEAR_ROUTE
DMA_ROUTE
enum class RpuAllGatherSchedule { SIMPLE=1 };
RpuAllGatherSchedule rpu_resolve_all_gather_schedule(int64_t n,int64_t dw){assert(n==128&&dw==2);return RpuAllGatherSchedule::SIMPLE;}
STUBS
struct Call {uintptr_t weight;uint32_t input,output,bias;int64_t m,n,k;int partition,cores;bool acc32;};
std::vector<Call> calls;
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,const at::Tensor& w,uint32_t out,
 int64_t m,int64_t n,int64_t k,int partition,int cores,uint32_t bias,const at::Tensor& scale,
 uint32_t ts,uint16_t layer,bool acc32){
 assert(w.scalar_type()==at::kHalf&&scale.scalar_type()==at::kHalf&&!ts&&!layer);
 calls.push_back({w.address,in,out,bias,m,n,k,partition,cores,acc32});}
struct CausalDecoderModel {
 bool linear_acc32_=false,decoder_mixed_precision_=false;
 std::vector<at::Tensor> tts_predictor_heads_;
 int resolved=0,invalidations=0;
 int get_last_resolved_chunk_size()const{return resolved;}
 void invalidate_model_state(){++invalidations;}
 SETTER
 FmbPhysicalExecutionManifest physical_manifest_for_candidate(const FmbThreeStageChunkPlan&,const LayoutContext&,int64_t,int64_t,int64_t)const{
  FmbPhysicalExecutionManifest manifest;MANIFEST_ACC
  return manifest;}
};
struct Model:CausalDecoderModel {
 int64_t chunk_size_=32,action_dim_pad_=32,k_comb_pad_=32,rtc_delay_=0,num_steps_=10;
 bool b_step_is_bias_=true,velocity_blend_=false,loop_mode_=true;int have_te_=1;
 bool planned_action_route_profile_valid_=true,planned_b_step_is_bias_=true,planned_have_te_=true;
 bool planned_velocity_blend_=false,planned_loop_mode_=true;int64_t planned_rtc_delay_=0,planned_num_steps_=10;
 at::Tensor w_comb_,w3_,proj_back_,w2_t_,w2_a_,resident_weight_bundle_;Context context;
 int layers=0;bool wall_action_weight_residency_=false;
 int num_layers()const{return layers;}int intermediate_size()const{return 2048;}int attn_tp()const{return 8;}
 static constexpr int WALL_ACTION_WEIGHT_BYTES_PER_CORE=0x600000;
 ACTION_SETTER
 RESIDENCY_SETTER
 int64_t hidden_size()const{return 1024;}Context& ctx(){return context;}
 uint32_t addr(int,const std::string& s)const{
  const std::vector<std::string> names={"x_t_spm","ae_spm","ct_spm","te_spm","b_step_spm","ce_spm","emb_core0_spm","residual1","v_t_core0_spm"};
  auto i=std::find(names.begin(),names.end(),s);assert(i!=names.end());return 0x100000+0x10000*(i-names.begin());}
 HOOKS
 void build(){planned_b_step_is_bias_=b_step_is_bias_;planned_have_te_=have_te_==1;
  planned_loop_mode_=loop_mode_;planned_num_steps_=loop_mode_?10:1;num_steps_=planned_num_steps_;
  context.value=physical_manifest_for_candidate({}, {},32,32,384);
  context.consumed_.assign(context.value.routes.size(),0);}
 Model(){w_comb_.address=1;w3_.address=2;proj_back_.address=3;w2_t_.address=4;w2_a_.address=5;}
};
template<class F>void rejects(F fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}assert(rejected);}
int main(){
 Model original;assert(!original.linear_acc32_);original.build();original.emit_pre_layers_body();original.emit_post_layers_body();
 assert(calls.size()==5);for(auto c:calls)assert(!c.acc32);calls.clear();
 for(bool acc32:{false,true})for(int branch:{0,1,2})for(bool loop:{false,true}){
  Model m;m.set_linear_acc32(acc32);assert(m.invalidations==1);m.b_step_is_bias_=branch!=2;m.have_te_=branch==0;m.loop_mode_=loop;m.build();
  assert(m.context.value.linear_accumulation==(acc32?FmbLinearAccumulationPolicy::ACC32:FmbLinearAccumulationPolicy::ACC16));
  for(auto& r:m.context.value.routes)if(r.family==FmbRouteFamily::LINEAR)assert(r.selector==wall_action_linear_route()&&r.flags==0);
  for(int iteration=0;iteration<(loop?10:1);++iteration){
   calls.clear();m.context.body_iter=iteration;m.emit_pre_layers_body();m.emit_post_layers_body();
   std::vector<uintptr_t> expected=branch==0?(iteration==0?std::vector<uintptr_t>{5,4,1,2,3}:std::vector<uintptr_t>{4,1,2,3}):std::vector<uintptr_t>{1,2,3};
   assert(calls.size()==expected.size());
   for(size_t i=0;i<calls.size();++i){const auto& c=calls[i];assert(c.weight==expected[i]&&c.acc32==acc32&&c.partition==1);
    assert(c.cores==(c.weight==2?8:1));assert(c.m==(c.weight>=4?1:32));
    assert(c.n==(c.weight==3?32:1024)&&c.k==(c.weight==1?32:1024));
    assert(c.bias==(c.weight==1&&branch!=2?m.addr(0,"b_step_spm"):0));}
  }
  for(auto used:m.context.consumed_)assert(used==1);
  m.resolved=32;const auto before=m.invalidations;rejects([&]{m.set_linear_acc32(!acc32);});assert(m.linear_acc32_==acc32&&m.invalidations==before);
 }
 // Actual COMPLETE consumption rejects a changed pre or post Linear receipt.
 for(auto site:{WALL_ACTION_PRE_W3_LINEAR_SITE,WALL_ACTION_POST_PROJ_LINEAR_SITE}){
  Model m;m.build();for(auto& r:m.context.value.routes)if(r.site_id==site)++r.selector;
  rejects([&]{if(site==WALL_ACTION_PRE_W3_LINEAR_SITE)m.emit_pre_layers_body();else m.emit_post_layers_body();});}
 Model cold;assert(!cold.linear_acc32_);cold.set_linear_acc32(true);assert(cold.linear_acc32_);
 at::Tensor bundle;bundle.shape={8,0x600000/2};cold.layers=36;
 rejects([&]{cold.set_weight_residency(bundle);});assert(!cold.wall_action_weight_residency_);
 for(bool mode:{false,true})rejects([&]{cold.set_linear_acc32(mode);});
 Model resident;resident.layers=36;resident.set_weight_residency(bundle);assert(resident.wall_action_weight_residency_);
 resident.layers=0;rejects([&]{resident.set_linear_acc32(true);});
 Model invalid;invalid.planned_action_route_profile_valid_=false;
 rejects([&]{invalid.physical_manifest_for_candidate({}, {},32,32,384);});
}
'''
    replacements = {
        'FAMILY': block(base, 'enum class FmbRouteFamily : int') + ';',
        'SELECTOR': block(base, 'enum class FmbLinearRouteSelector : int') + ';',
        'ACCUMULATION_ENUM': block(base, 'enum class FmbLinearAccumulationPolicy : int') + ';',
        'CONSUME': consume, 'CONSTANTS': constants,
        'DMA_ENUM': block(action, 'enum class WallActionMutableDmaRoute : int64_t') + ';',
        'LINEAR_ROUTE': block(action, 'constexpr int64_t wall_action_linear_route()'),
        'DMA_ROUTE': block(action, 'constexpr int64_t wall_action_dma_route('),
        'STUBS': stubs, 'SETTER': block(decoder, '    void set_linear_acc32(bool enabled)'),
        'MANIFEST_ACC': accumulation, 'HOOKS': hooks,
        'ACTION_SETTER': block(action_header, '    void set_linear_acc32(bool enabled)'),
        'RESIDENCY_SETTER': block(action, 'void WallOssActionStepModel::set_weight_residency(').replace('WallOssActionStepModel::', ''),
    }
    for key, value in sorted(replacements.items(), key=lambda item: -len(item[0])):
        source = source.replace(key, value)
    compile_run(tmp_path, 'wall_action_accumulation_hooks', source)


def test_actual_causal_precision_guards_and_ordinary_projection_arguments(tmp_path):
    decoder = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()
    start = decoder.index('        const int64_t rows = physical_len;')
    producer = decoder[start:decoder.index('        const bool wall_base_nvfp4_o_ring_norm =', start)]
    start = decoder.rindex('        const bool wall_action_fp16_transfer =', 0,
                          decoder.index('        const bool wall_action_w8a16_silu_mul ='))
    guard = decoder[start:decoder.index('        TORCH_CHECK(!decoder_mixed_precision_', start)]
    helpers = '\n'.join(block(decoder, marker) for marker in (
        '    bool wall_action_fp16_transfer_enabled() const',
        '    static bool wall_action_qkv_weights_compatible(',
        '    static bool wall_action_ring_weights_compatible(',
        '    bool wall_base_paired_kv_profile() const',
        '    bool wall_base_nvfp4_qkv_bias_profile() const',
    ))
    start = decoder.index('} else if (wall_base_nvfp4_qkv_bias) {')
    ordinary = decoder.index('} else {', start)
    qkv = '\n'.join(_statement(decoder, f'rpu_launch_linear_spm_to_spm_acc16_kernel(\n                addr(0, "input_norm"), lw.{name}_w', ordinary) for name in ('q', 'k', 'v'))
    o = _statement(decoder, 'rpu_launch_linear_spm_to_spm_acc16_kernel(\n                addr(0, "output"), lw.o_w', ordinary)
    start = decoder.index('            if (rhinovla_high_precision_) consume_rhinovla_high_activation(rows, chunk.idx);')
    mlp = _statement(decoder, 'emit_mlp_pipeline(', start)
    constants = '\n'.join(re.findall(r'(?:inline )?(?:static )?constexpr (?:int64_t|int|uint32_t) WALL_ACTION_\w+\s*=\s*[^;]+;', decoder))
    source = META + r'''
#include <tuple>
CONSTANTS
struct KernelCache {bool available=true;static auto& instance(){static KernelCache c;return c;}
 void* get_kernel(const char*){return available?this:nullptr;}void* get_program(const char*){return available?this:nullptr;}};
struct Context {bool is_causal=false;std::optional<int> attention_mask{1};};
enum class ActivationKind { SILU,NONE };enum class RpuUnaryPrecision { HIGH,BASE };
struct Call {at::ScalarType dtype;bool scale,acc32;uint32_t ts;int64_t n,k;uint16_t layer;};
std::vector<Call> linears;std::vector<bool> mlp_modes;int current_precision=0;bool current_acc32=false;
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t,const at::Tensor& w,uint32_t,int64_t m,int64_t n,int64_t k,
 int partition,int cores,uint32_t,const at::Tensor& scale,uint32_t ts,uint16_t layer,bool acc32,
 bool gemv=false,bool planar=false,bool reuse=false){
 assert(m==32&&cores==8&&(partition==0||partition==1)&&!gemv&&!planar&&!reuse);
 linears.push_back({w.dtype,scale.defined(),acc32,ts,n,k,layer});}
template<class... A>void emit_mlp_pipeline(A&&... args){
 auto t=std::forward_as_tuple(args...);assert(std::get<3>(t)==32&&std::get<14>(t)==current_acc32);
 for(auto* w:{&std::get<0>(t),&std::get<1>(t),&std::get<2>(t)})
  assert(w->dtype==(current_precision==0?at::kHalf:current_precision==1?at::kChar:at::kByte));
 assert(!std::get<16>(t)&&!std::get<17>(t)&&!std::get<22>(t)&&!std::get<23>(t));
 mlp_modes.push_back(std::get<14>(t));}
struct Model {
 struct LayerWeights {at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,input_norm_w,post_norm_w;
  at::Tensor q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws,q_bias_w,k_bias_w,v_bias_w;};
 std::vector<LayerWeights> layer_weights_{36};WallSortingConfig wall_sorting_;
 at::Tensor final_norm_w_,q_nvfp4_tensor_scales_,k_nvfp4_tensor_scales_,v_nvfp4_tensor_scales_;
 bool linear_acc32_=false,nvfp4_=false,wall_action_fuse_silu_mul_=true,pre_layers_residual1_ready_=true;
 bool has_qkv_bias_=true,has_qk_norm_=false,has_mrope_=true,use_silu_=true,partial_mrope_active_=true;
 bool decoder_mixed_precision_=false,adarms_=false,wall_action_weight_residency_=false,wall_action_kv_owner_fanout_=false;
 bool qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_dry_prepared_=false,qwen3vl_multiview_text_composite_prepared_=false,qwen3vl_multiview_text_composite_dispatch_=false;
 int qwen3vl_pooler_z1_retained_deepstack_count_=0;bool wall_base_nvfp4_qkv_bias_=false;
 std::vector<int64_t> deepstack_lang_layers_;int64_t seq_len_=32;Context context;
 int hidden_size()const{return 1024;}int intermediate_size()const{return 2048;}int num_layers()const{return layer_weights_.size();}
 int num_q_heads()const{return 16;}int num_kv_heads()const{return 8;}int head_dim()const{return 128;}
 int num_cores()const{return 8;}int attn_tp()const{return 8;}int mlp_tp()const{return 8;}
 const Context& ctx()const{return context;}uint32_t addr(int,const char*)const{return 0x100000;}
 uint32_t nvfp4_scale_addr(const char*)const{return nvfp4_?0x200000:0;}
 uint32_t qkv_fused_bias_addr(int,const char*)const{return nvfp4_?0:0x300000;}
 bool qwen3vl_decode_gemv(int64_t)const{return false;}bool use_planned_silu_mul()const{return false;}
 HELPERS
 void bind(){COLD}
 void install(int precision){nvfp4_=precision==2;wall_sorting_.silu_mul=true;wall_sorting_.fp16_transfer_mask=precision==0?4:0;
  for(auto* t:{&q_nvfp4_tensor_scales_,&k_nvfp4_tensor_scales_,&v_nvfp4_tensor_scales_}){t->dtype=at::kFloat;t->shape={40};}
  for(auto& w:layer_weights_){for(auto* t:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})t->dtype=precision==0?at::kHalf:precision==1?at::kChar:at::kByte;
   for(auto* t:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws}){t->present=precision!=0;t->dtype=precision==2?at::kByte:at::kHalf;}
   w.q_bias_w.shape={2048};w.k_bias_w.shape=w.v_bias_w.shape={1024};}bind();}
 bool producer(){const int64_t physical_len=32;const bool partial_mrope=true;
  struct {int batch_size=1;bool is_causal=false,use_attn_mask=true;}layout;
  PRODUCER
  return every_wall_action_qkv&&wall_action_fused_qkv_partial_mrope&&wall_action_linear_ring;}
 bool consumer(int layer_idx){const auto& lw=layer_weights_.at(layer_idx);
  const int64_t rows=32,h=1024,nq=16,nkv=8,hd=128,tp=8,bs=1;
  const bool residual_is_bf16=false,transition_to_bf16=false;
  GUARD
  return wall_action_w8a16_qkv&&wall_action_w8a16_silu_mul&&wall_action_fused_qkv_partial_mrope&&wall_action_linear_ring_postnorm;
 }
 void ordinary(int layer_idx){const auto& lw=layer_weights_.at(layer_idx);
  const int64_t rows=32,h=1024,nq=16,nkv=8,hd=128,tp=8;const bool reuse_o_weight=false;
  QKV
  O_CALL
  const bool transition_to_bf16=false,residual_is_bf16=false,wall_small_prefill_fused=false,wall_fp16_long_prefill_fused=false,wall_p640_staged0_w8a16=false;
  const bool wall_action_w8a16_silu_mul=false,rtc_prefill_mixed=false,wall_action_linear_ring_postnorm=false,wall_base_nvfp4_down_ring_norm=false,is_last_layer=layer_idx==35;
  const bool wall_action_resident_layer=false,wall_action_fp16_transfer=wall_action_fp16_transfer_enabled(),rhinovla_high_precision_=false,rhinovla_high_precision_fusions_=false;
  const uint32_t wall_action_weight_base=0;
  MLP
 }
};
template<class F>void rejects(F fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}assert(rejected);}
int main(){
 for(int precision:{0,1,2})for(bool acc32:{false,true}){
  Model m;m.linear_acc32_=acc32;m.install(precision);
  assert(m.wall_base_paired_kv_profile()==!acc32);
  assert(m.wall_base_nvfp4_qkv_bias_==(precision==2&&!acc32));
  assert(m.producer()==(precision!=2&&!acc32));
  for(int layer:{0,1,2,4,35})assert(m.consumer(layer)==(precision!=2&&!acc32));
  current_precision=precision;current_acc32=acc32;linears.clear();mlp_modes.clear();
  // Exercise the actual ordinary calls: ACC32 takes this path; ACC16 reaches
  // the same calls when its optional fusion is absent or ineligible.
  for(int layer=0;layer<36;++layer)m.ordinary(layer);
  assert(linears.size()==144&&mlp_modes.size()==36);
  for(size_t i=0;i<linears.size();++i){auto c=linears[i];assert(c.layer==i/4&&c.acc32==acc32);
   assert(c.dtype==(precision==0?at::kHalf:precision==1?at::kChar:at::kByte));
   assert(c.scale==(precision!=0)&&c.ts==(precision==2?0x200000u:0u));
   assert(c.n==(i%4==0?2048:1024)&&c.k==(i%4==3?2048:1024));}
  m.wall_action_weight_residency_=true;
  if(acc32||precision==2)rejects([&]{m.consumer(1);});
  else assert(m.consumer(1));
 }
 Model missing;KernelCache::instance().available=false;missing.install(2);assert(!missing.wall_base_nvfp4_qkv_bias_);
}
'''
    replacements = {'CONSTANTS': constants, 'HELPERS': helpers, 'PRODUCER': producer,
                    'GUARD': guard, 'QKV': qkv, 'O_CALL': o, 'MLP': mlp,
                    'COLD': _statement(decoder, 'wall_base_nvfp4_qkv_bias_ =')}
    for key, value in replacements.items():
        source = re.sub(r'(?m)^[ \t]*' + key + r'[ \t]*$', lambda _: value, source)
    source = source.replace('void bind(){COLD}', 'void bind(){' + replacements['COLD'] + '}')
    compile_run(tmp_path, 'wall_action_accumulation_causal', source)

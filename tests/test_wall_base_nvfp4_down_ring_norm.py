"""Execute real Down admission/emission and cross-layer scheduling boundaries."""
from pathlib import Path
import re

from test_wall_base_paired_kv import block, compile_run
from test_wall_base_nvfp4_qkv_bias import META

ROOT = Path(__file__).resolve().parents[1]
DECODER = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()


def test_actual_down_next_norm_complete_and_layer_boundaries(tmp_path):
    helpers = '\n'.join(block(DECODER, marker) for marker in (
        '    bool wall_base_nvfp4_o_ring_norm_enabled(',
        '    static std::vector<int64_t> wall_base_nvfp4_o_ring_norm_arguments(',
        '    bool wall_base_nvfp4_down_ring_norm_profile() const',
        '    bool wall_base_nvfp4_down_ring_norm_enabled(',
        '    static std::vector<int64_t> wall_base_nvfp4_down_ring_norm_arguments(',
        '    void emit_wall_base_nvfp4_down_ring_norm(',
    ))
    start = DECODER.index('        wall_base_nvfp4_down_ring_norm_ =')
    cold = DECODER[start:DECODER.index(';', start)+1]
    producer = block(DECODER, 'if (wall_base_nvfp4_down_ring_norm_enabled(')
    layer = DECODER[DECODER.index('    void build_layer_subgraph('):]
    start = layer.index('        TORCH_CHECK(!wall_base_nvfp4_down_ring_norm ||')
    input_guard = layer[start:layer.index(';', start)+1]
    input_condition = layer.split('(layer_idx == 0 ||', 1)[1].split(')) {', 1)[0]
    input_condition = 'layer_idx == 0 ||' + input_condition
    skip_down = layer.split('/*skip_down=*/', 1)[1].split(',', 1)[0]
    down_emit = block(layer, 'if (wall_base_nvfp4_down_ring_norm && !is_last_layer) {')
    final_condition = re.search(r'if \((is_last_layer && !wall_action_linear_ring_postnorm)\)', layer)[1]
    planner = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    consume = block(planner, 'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(')
    consume = consume.replace('FmbPhysicalManifestConsumer::consume_route', 'consume_physical_route')
    consume = consume.replace('int64_t invocation) const', 'int64_t invocation = 0) const')
    find = block(planner, 'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(')
    find = find.replace('FmbPhysicalManifestConsumer::find_route', 'find_route')
    source = META + r'''
enum class ChunkMode{SEQUENTIAL,KV_FIRST};
struct ChunkInfo{int64_t idx=0,offset=0,len=32,kv_seq_len=416;};
struct FmbThreeStageChunkPlan{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;struct{std::vector<ChunkInfo> chunks{{}};}compute;};
enum class FmbRouteFamily{ATTENTION,LINEAR,COLLECTIVE};
enum class CausalDecoderLinearRoute{WALL_BASE_NVFP4_DOWN_RING_NORM=10};
enum class RpuRmsNormSpmRoute{BASE=1};
constexpr int64_t CAUSAL_DECODER_WALL_BASE_NVFP4_DOWN_RING_NORM_SITE=2325056782993142764LL;
struct FmbRouteManifestEntry{int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t> arguments;int64_t invocation;};
struct Manifest{std::vector<FmbRouteManifestEntry> routes;};
struct Context{Manifest manifest_;mutable std::vector<int> consumed_;bool complete_flag=true,is_causal=false;
 bool input_in_spm=true,output_to_spm=true;int64_t position=384;FmbThreeStageChunkPlan plan;
 bool complete()const{return complete_flag;}const Manifest& manifest()const{return manifest_;}
 FIND
 CONSUME
};
struct KernelCache{bool kernel=true,program=true;static auto& instance(){static KernelCache c;return c;}
 void* get_kernel(std::string s){assert(s=="wall_base_nvfp4_down_ring_norm_h32");return kernel?this:nullptr;}
 void* get_program(std::string s){assert(s=="wall_base_nvfp4_down_ring_norm_h32");return program?this:nullptr;}
};
int calls=0,last_layer=-1;
void rpu_launch_wall_base_nvfp4_down_ring_norm_spm_kernel(uint32_t input,const at::Tensor&,uint32_t partial,
 uint32_t residual,uint32_t raw,uint32_t norm,uint32_t gamma,const at::Tensor&,uint32_t ts,int64_t count,uint16_t layer,double eps){
 assert(input==1&&partial==2&&residual==3&&raw==4&&norm==3&&gamma==100+layer+1&&ts==5&&count==40&&layer<35&&eps==1e-6);
 ++calls;last_layer=layer;
}
struct Model {
 struct LayerWeights{at::Tensor down_w,down_ws;} lw;
 Context context;Context& ctx(){return context;}
 bool wall_base_nvfp4_o_ring_norm_=true,wall_base_nvfp4_down_ring_norm_=false;
 bool adarms_=false,decoder_mixed_precision_=false,wall_action_weight_residency_=false,wall_action_kv_owner_fanout_=false;
 WallSortingConfig wall_sorting_;int cache_batch_slot_=0;double eps_=1e-6;bool alias=true;
 at::Tensor down_nvfp4_tensor_scales_;
 uint32_t addr(int,std::string s)const{return s=="gate"?1:s=="down"?2:s=="residual1"?4:s=="input_norm"&&!alias?6:3;}
 uint32_t norm_input_addr(int layer,int core)const{assert(core==0);return 100+layer;}
 uint32_t nvfp4_scale_addr(std::string s)const{assert(s=="nvfp4_down_ts");return 5;}
 HELPERS
 void bind(){COLD}
 Model(){down_nvfp4_tensor_scales_.dtype=at::kFloat;down_nvfp4_tensor_scales_.shape={40};bind();}
 bool enabled(){return wall_base_nvfp4_down_ring_norm_enabled(context.plan,32,32,context.position,false,true,1,true);}
 void build(){auto& plan=context.plan;int64_t physical_len=32,logical_len=32,position=context.position;
  struct{bool is_causal=false,use_attn_mask=true;int64_t batch_size=1;}layout;bool partial_mrope=true;
  context.manifest_.routes.clear();
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t invocation){
   context.manifest_.routes.push_back({site,f,selector,flags,args,invocation});};
  PRODUCER
  context.consumed_.assign(context.manifest_.routes.size(),0);
 }
 int input_norms=0,ordinary_downs=0,final_norms=0;
 void layer(int layer_idx){
  bool wall_base_nvfp4_down_ring_norm=enabled(),wall_action_linear_ring_postnorm=false,is_last_layer=layer_idx==35;
  auto& chunk=context.plan.compute.chunks.front();INPUT_GUARD
  if(INPUT_CONDITION)++input_norms;
  if(!(SKIP_DOWN))++ordinary_downs;
  DOWN_EMIT
  if(FINAL_CONDITION)++final_norms;
 }
};
template<class F>void rejects(F f){int before=calls;bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}assert(rejected&&calls==before);}
int main(){
 for(int p:{336,384,448,512,640}){Model m;m.context.position=p;m.context.plan.compute.chunks[0].kv_seq_len=p+32;m.build();
  assert(m.enabled()&&m.context.manifest_.routes.size()==1);
  auto r=m.context.manifest_.routes[0];assert(r.selector==10&&r.arguments.size()==19&&r.arguments[2]==p&&r.arguments[16]==0&&r.arguments[17]==35&&r.arguments[18]==1);
  calls=0;for(int layer=0;layer<36;++layer){m.context.input_in_spm=layer!=0;m.context.output_to_spm=layer!=35;m.layer(layer);}
  assert(calls==35&&last_layer==34&&m.input_norms==1&&m.ordinary_downs==1&&m.final_norms==1);
 }
 for(auto member:{&KernelCache::kernel,&KernelCache::program}){auto& c=KernelCache::instance();c.*member=false;
  Model m;m.build();assert(!m.enabled()&&m.context.manifest_.routes.empty());calls=0;
  for(int layer=0;layer<36;++layer)m.layer(layer);
  assert(calls==0&&m.input_norms==36&&m.ordinary_downs==36&&m.final_norms==1);c.*member=true;
 }
 Model m;auto fallback=[&](){m.bind();assert(!m.enabled());m.build();assert(m.context.manifest_.routes.empty());};
 m.wall_base_nvfp4_o_ring_norm_=false;fallback();m.wall_base_nvfp4_o_ring_norm_=true;
 m.down_nvfp4_tensor_scales_.present=false;fallback();m.down_nvfp4_tensor_scales_.present=true;
 m.down_nvfp4_tensor_scales_.contiguous=false;fallback();m.down_nvfp4_tensor_scales_.contiguous=true;
 m.down_nvfp4_tensor_scales_.dtype=at::kHalf;fallback();m.down_nvfp4_tensor_scales_.dtype=at::kFloat;
 for(int count:{36,48}){m.down_nvfp4_tensor_scales_.shape={count};fallback();}m.down_nvfp4_tensor_scales_.shape={40};m.bind();
 m.context.position=320;assert(!m.enabled());m.context.position=384;
 m.wall_sorting_.action_activation_reuse=true;assert(!m.enabled());m.wall_sorting_.action_activation_reuse=false;
 m.context.plan.compute.chunks.push_back({});assert(!m.enabled());m.context.plan.compute.chunks.pop_back();
 m.build();m.context.input_in_spm=false;rejects([&]{m.layer(1);});m.context.input_in_spm=true;
 m.context.output_to_spm=false;rejects([&]{m.layer(34);});m.context.output_to_spm=true;
 m.alias=false;rejects([&]{m.layer(0);});m.alias=true;
 rejects([&]{m.emit_wall_base_nvfp4_down_ring_norm(m.lw,35,m.context.plan.compute.chunks[0]);});
 rejects([&]{m.emit_wall_base_nvfp4_down_ring_norm(m.lw,-1,m.context.plan.compute.chunks[0]);});
 for(int v=0;v<6;++v){m.build();
  if(v==0)m.context.manifest_.routes[0].arguments[2]=336;
  if(v==1)m.context.manifest_.routes[0].arguments[17]=36;
  if(v==2)m.context.manifest_.routes[0].selector=9;
  if(v==3)m.context.manifest_.routes[0].flags=1;
  if(v==4)m.context.manifest_.routes.clear();
  if(v==5)m.context.complete_flag=false;
  rejects([&]{m.layer(0);});m.context.complete_flag=true;
 }
}
'''
    for key, value in [('HELPERS', helpers), ('COLD', cold), ('PRODUCER', producer),
                       ('INPUT_GUARD', input_guard), ('INPUT_CONDITION', input_condition),
                       ('SKIP_DOWN', skip_down), ('DOWN_EMIT', down_emit),
                       ('FINAL_CONDITION', final_condition), ('FIND', find), ('CONSUME', consume)]:
        source = re.sub(r'\b'+key+r'\b', lambda _: value, source)
    compile_run(tmp_path, 'wall_down_route', source)

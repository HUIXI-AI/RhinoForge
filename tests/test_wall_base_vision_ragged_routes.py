"""Execute real small-Vision mask proof, route production and consumption on CPU."""
from pathlib import Path
import re

from test_wall_base_paired_kv import block, compile_run


ROOT = Path(__file__).resolve().parents[1]


def _source():
    vision = (ROOT / 'src/fused/rpu_qwen25vl_vision_model.cpp').read_text()
    planner = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    cache = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    base = (ROOT / 'src/core/fused_model_base.h').read_text()
    helpers = '\n'.join(block(vision, marker) for marker in (
        '    bool padded_pair_enabled() const',
        '    bool small_image_row_weight_outer_enabled(',
        '    bool small_image_fp16_row_profile(',
        '    bool small_image_gate_up_enabled(',
        '    bool small_image_acc16_row_projection_enabled(',
        '    bool small_image_acc16_gate_up_projection_enabled(',
        '    bool small_image_native_window_shape() const',
        '    bool small_image_native_window_enabled() const',
        '    bool small_image_raw_kv_only_enabled() const',
        '    bool small_image_resident_mask_enabled() const',
        '    bool small_image_single_raw_query_enabled() const',
        '    bool small_image_qk_rope_enabled(',
        '    void validate_small_image_qk_rope_plan(',
        '    static bool is_small_ragged_mask(',
        '    void refresh_small_ragged_mask(',
        '    bool small_image_ragged_sdpa_enabled(',
        '    std::vector<int64_t> small_image_ragged_sdpa_arguments(',
        '    void consume_small_image_ragged_sdpa(',
        '    int64_t small_image_schedule_flags() const',
        '    bool generic_attention_layer_is_full(',
        '    int64_t generic_shared_attention_mask_domain() const',
        '    std::vector<int64_t> sorting_schedule_arguments() const',
        '    FmbPhysicalManifestForwardCapability\n    physical_manifest_forward_capability(',
    )).replace(' const override', ' const')
    cold = []
    for name in ('small_qk_rope_available_', 'small_ragged_sdpa_available_'):
        start = vision.index('        ' + name + ' =')
        cold.append(vision[start:vision.index(';', start) + 1])
    consumer = '\n'.join(block(planner, marker).replace(
        'FmbPhysicalManifestConsumer::', '') for marker in (
        'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(',
        'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(',
    ))
    producer = block(vision, '            if (small_image_ragged_sdpa_enabled(physical_len))')
    dispatch = block(vision, '        if (!is_full && small_image_ragged_sdpa_enabled(seq_len))')
    full_start = vision.index('        bool is_full = !(window_mask_present_')
    full = vision[full_start:vision.index(';', full_start) + 1]
    schedule_start = vision.index('        append(FmbRouteFamily::GRAPH_SCHEDULE, QWEN25_WALL_SORTING_SITE,')
    schedule = vision[schedule_start:vision.index(';', schedule_start) + 1]
    sort_start = vision.index('        std::sort(\n            manifest.routes.begin()')
    sort_routes = vision[sort_start:vision.index('        return manifest;', sort_start)]
    source = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <c10/util/ArrayRef.h>
#include <c10/util/Half.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Layout.h>
#include <c10/util/Exception.h>
#undef TORCH_CHECK
inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
inline int64_t Align(int64_t a, int64_t b) { return CeilDiv(a, b) * b; }
#define TORCH_CHECK(ok, ...) do {if(!(ok))throw std::runtime_error(#ok);}while(false)
#undef TORCH_INTERNAL_ASSERT
#define TORCH_INTERNAL_ASSERT(ok) TORCH_CHECK(ok)
constexpr int64_t NUM_CORES=8,DWIDTH=2;
namespace at {
enum ScalarType {kHalf,kChar,kFloat};
struct Device {bool cpu=true;bool is_cpu()const{return cpu;}};
struct Version {
 bool on=true;uint32_t value=0;
 bool enabled()const{return on;}uint32_t current_version()const{assert(on);return value;}
};
struct Impl {Version version;auto& version_counter(){return version;}};
struct Tensor {
 bool present=false,contiguous=true;ScalarType dtype=kHalf;Device where;
 c10::Layout storage_layout=c10::Layout::Strided;std::vector<int64_t> shape;
 std::shared_ptr<Impl> impl=std::make_shared<Impl>();
 std::shared_ptr<std::vector<c10::Half>> storage=std::make_shared<std::vector<c10::Half>>();
 bool defined()const{return present;}Device device()const{return where;}
 auto scalar_type()const{return dtype;}auto layout()const{return storage_layout;}
 bool is_contiguous()const{return contiguous;}int64_t dim()const{return shape.size();}
 int64_t size(size_t i)const{return shape.at(i);}Impl* unsafeGetTensorImpl()const{return impl.get();}
 void* data_ptr()const{return storage->data();}
 template<class T>T* data_ptr()const{return reinterpret_cast<T*>(storage->data());}
 void put(int64_t q,int64_t k,float v){storage->at(q*shape.at(1)+k)=c10::Half(v);++impl->version.value;}
};
using IntArrayRef=c10::ArrayRef<int64_t>;
}
KERNEL_IDS
AVAILABILITY
struct KernelCache {
 KernelPayloadAvailability availability;
 static auto& instance(){static KernelCache k;return k;}
 bool has_loaded(KernelId id)const{return availability.has_loaded(id);}
};
void publish(bool ragged=true,bool rope=true){
 KernelPayloadAvailability::ExtendedPayloads bits;
 bits.set(size_t(KernelId::WALL_BASE_VISION_RAGGED_SDPA)-KernelPayloadAvailability::kExtendedBegin,ragged);
 bits.set(size_t(KernelId::WALL_BASE_VISION_QK_ROPE_2D_SPM)-KernelPayloadAvailability::kExtendedBegin,rope);
 KernelCache::instance().availability.publish(false,false,false,false,false,false,bits);
}
enum class VisionSchedule {SISC,MISC,MIMC};
enum class ChunkMode {SEQUENTIAL,KV_FIRST};
ATTENTION_POLICY
ROUTE_FAMILY
enum class FmbGraphLifecycle {COMPOSITE_CHILD};
struct FmbPhysicalManifestForwardCapability {bool supported;FmbGraphLifecycle lifecycle;};
struct ChunkInfo {int64_t idx,offset,len,kv_seq_len;};
struct FmbExecutionSpan {int64_t offset,len,group_id;};
struct ResolvedFmbStageChunks {
 struct {int64_t chunk_size,num_chunks;}plan;std::vector<ChunkInfo> chunks;
};
struct FmbThreeStageChunkPlan {
 ResolvedFmbStageChunks input,qkv,compute;std::vector<FmbExecutionSpan> spans;
 ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;
};
MAKE_STAGE
struct FmbRouteManifestEntry {int64_t site_id;FmbRouteFamily family;
 int64_t selector,flags;std::vector<int64_t> arguments;int64_t invocation;};
enum class FmbLinearAccumulationPolicy {ACC16,ACC32,MIXED_BY_SITE};
struct AccumulationField {FmbLinearAccumulationPolicy linear_accumulation=FmbLinearAccumulationPolicy::ACC16;};
struct FmbPhysicalExecutionManifest : AccumulationField {std::vector<FmbRouteManifestEntry> routes;};
struct Context {
 FmbThreeStageChunkPlan stage_plan;FmbPhysicalExecutionManifest value;
 mutable std::vector<uint8_t> consumed_;bool ready=true;
 bool complete()const{return ready;}bool has_complete_physical_manifest()const{return complete();}
 const auto& manifest()const{return value;}
 CONSUMER
 auto& consume_physical_route(FmbRouteFamily f,int64_t s,int64_t r,int64_t flags,
                             at::IntArrayRef args,int64_t i){return consume_route(f,s,r,flags,args,i);}
};
enum class RpuRmsNormSpmRoute {BASE};
SITES
struct Launch {uint32_t q,k,v,out,mask;int64_t rows;};
std::vector<Launch> launches;
int transposes=0;
void rpu_launch_v_transpose_spm(uint32_t v,uint32_t tmp,int b,int64_t rows,int64_t nq,int64_t hd,int cores){
 assert(b==1&&(rows==324||rows==336)&&nq==16&&hd==80&&cores==8);
 assert(tmp==0x500000&&v==0x300000+uint32_t((transposes%2)*rows*320));++transposes;
}
void rpu_launch_wall_base_vision_ragged_sdpa_spm(uint32_t q,uint32_t k,uint32_t v,
 uint32_t out,uint32_t mask,double scale,int64_t rows,int cores){
 assert(cores==8&&scale>0&&v==0x500000&&mask==0x600000);
 launches.push_back({q,k,v,out,mask,rows});
}
struct Model {
 struct Sorting {bool vision_row_reuse=true,silu_mul=true;uint32_t fp16_transfer_mask=0;}wall_sorting_;
 bool linear_acc32_=false;
 bool generic=true,mixed_precision_=false,per_window_sdpa_enabled_=false,z1_bound_=false;
 bool debug=false,merger=true,window_mask_present_=true,rope_spm_enabled_=true,has_rope_=true;
 bool small_qk_rope_available_=false,small_ragged_sdpa_available_=false,small_ragged_mask_exact_=false;
 int64_t small_ragged_mask_rows_=0;
 std::array<at::Tensor,2> small_ragged_mask_refs_;
 std::array<uint32_t,2> small_ragged_mask_versions_{0,0};
 std::array<bool,2> small_ragged_mask_valid_{false,false};
 int64_t layers=32,hidden=1280,heads=16,kvheads=16,dim=80,intermediate=3584;
 int64_t image_batch_count_=1,current_num_patches_=324,valid_patches_per_image_=0,max_hw_=128;
 VisionSchedule schedule_=VisionSchedule::SISC;
 std::vector<int64_t> fullatt_block_indexes_{7,15,23,31},cu_window_;
 bool is_generic()const{return generic;}bool merger_active()const{return merger;}
 bool get_debug_export()const{return debug;}int64_t num_layers()const{return layers;}
 int64_t hidden_size()const{return hidden;}int64_t num_q_heads()const{return heads;}
 int64_t num_kv_heads()const{return kvheads;}int64_t head_dim()const{return dim;}
 int64_t intermediate_size()const{return intermediate;}
 LAYER_WEIGHTS
 std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(32);
 at::Tensor freq_cos_,freq_sin_;
 Context context;auto& ctx(){return context;}
 RpuRmsNormSpmRoute small_image_rmsnorm_route(int64_t)const{return RpuRmsNormSpmRoute::BASE;}
 bool small_image_owner_ring_norm_enabled(int64_t)const{return false;}
 bool small_image_down_owner_norm_enabled(int64_t)const{return false;}
 void bind(){COLD}
 HELPERS
 auto produce(const FmbThreeStageChunkPlan& plan,AttentionExecutionPolicy policy=AttentionExecutionPolicy::SPM_KV_BY_MHA){
  FmbPhysicalExecutionManifest manifest;
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t invocation=0,
                  std::vector<int64_t> args={},int64_t flags=0){
   manifest.routes.push_back({site,f,selector,flags,std::move(args),invocation});};
  SCHEDULE
  struct {AttentionExecutionPolicy attention_policy;}layout{policy};
  const auto physical_len=current_num_patches_;
  PRODUCER
  SORT_ROUTES
  return manifest;
 }
 uint32_t addr(int core,const std::string& name){assert(core==0);
  return name=="q"?0x100000:name=="k"?0x200000:name=="v"?0x300000:
         name=="sdpa_out"?0x400000:name=="sdpa_tmp"?0x500000:0x600000;}
 void execute_window_branch(int64_t layer_idx){
  const int64_t seq_len=current_num_patches_,nq=16,hd=80;double attn_scale=1/std::sqrt(80.);
  FULL
  DISPATCH
 }
};
at::Tensor canonical(int64_t rows){
 at::Tensor t;t.present=true;t.shape={rows,rows};
 t.storage->assign(rows*rows,c10::Half(-std::numeric_limits<float>::infinity()));
 // Build independent diagonal rectangles, including the isolated dummy block.
 std::vector<int64_t> lengths{64,64,16,64,64,16,16,16,4};if(rows==336)lengths.push_back(12);
 int64_t offset=0;
 for(auto n:lengths){
  for(int64_t q=offset;q<offset+n;++q)for(int64_t k=offset;k<offset+n;++k){
   t.storage->at(q*rows+k)=c10::Half(0.f);
  }
  offset+=n;
 }
 assert(offset==rows);return t;
}
FmbThreeStageChunkPlan plan_for(int64_t rows){
 FmbThreeStageChunkPlan p;p.input=p.qkv=p.compute=make_stage({{0,0,rows,rows}});
 if(rows==672)p.spans={{0,336,0},{336,336,1}};else p.spans={{0,324,0}};return p;
}
Model exact(int64_t rows,bool fp16=false){
 Model m;for(auto& w:m.layer_weights_){
  for(auto* t:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w}){t->dtype=fp16?at::kHalf:at::kChar;t->present=true;}
  for(auto* t:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws})t->present=!fp16;
 }
 m.wall_sorting_.fp16_transfer_mask=fp16?1:0;
 m.freq_cos_.shape=m.freq_sin_.shape={128,20};m.current_num_patches_=rows;
 if(rows==672){m.image_batch_count_=2;m.valid_patches_per_image_=324;m.schedule_=VisionSchedule::MISC;}
 m.context.stage_plan=plan_for(rows);m.bind();return m;
}
template<class F>void rejects(F&& f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed);}
'''
    substitutions = {
        'KERNEL_IDS': block(cache, 'enum class KernelId : uint16_t') + ';',
        'AVAILABILITY': block(cache, 'class KernelPayloadAvailability final') + ';',
        'ATTENTION_POLICY': block(base, 'enum class AttentionExecutionPolicy : int') + ';',
        'ROUTE_FAMILY': block(base, 'enum class FmbRouteFamily : int') + ';',
        'LAYER_WEIGHTS': block(vision, '    struct LayerWeights {') + ';',
        'MAKE_STAGE': block(planner, 'ResolvedFmbStageChunks make_stage('),
        'CONSUMER': consumer, 'HELPERS': helpers,
        'PRODUCER': producer, 'DISPATCH': dispatch, 'FULL': full, 'SCHEDULE': schedule,
        'SORT_ROUTES': sort_routes,
        'SITES': '\n'.join(re.search(r'constexpr int64_t ' + name + r'\s*=.*?;', vision, re.S).group()
                           for name in ('QWEN25_SMALL_RAGGED_SDPA_SITE', 'QWEN25_WALL_SORTING_SITE')),
    }
    for key, value in substitutions.items():
        source = re.sub(r'(?m)^[ \t]*' + re.escape(key) + r'[ \t]*$', lambda _: value, source)
    return source.replace('COLD', '\n'.join(cold))


def test_actual_mask_values_metadata_and_owner_version_registration(tmp_path):
    source = _source() + r'''
int main(){
 publish();
 for(bool fp16:{false,true})for(int64_t rows:{324,336}){
  const int64_t total=rows==336?672:324;auto t=canonical(rows);Model m=exact(total,fp16);
  assert(Model::is_small_ragged_mask(t,rows));
  assert(!Model::is_small_ragged_mask(t,320));
  for(auto value:{1.f,-1.f,-65504.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
   auto bad=canonical(rows);bad.put(0,64,value);assert(!Model::is_small_ragged_mask(bad,rows));
  }
  for(auto value:{1.f,-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
   auto bad=canonical(rows);bad.put(rows-1,rows-1,value);assert(!Model::is_small_ragged_mask(bad,rows));
  }
  t.put(rows-1,rows-1,-0.f);assert(Model::is_small_ragged_mask(t,rows));
  for(int64_t boundary:{64,128,144,208,272,288,304,320}){
   auto bad=canonical(rows);bad.put(boundary-1,boundary,0);assert(!Model::is_small_ragged_mask(bad,rows));
  }
  if(rows==336){auto bad=canonical(rows);bad.put(323,324,0);assert(!Model::is_small_ragged_mask(bad,rows));}
  for(int mutation=0;mutation<8;++mutation){auto bad=t;
   switch(mutation){case 0:bad.present=false;break;case 1:bad.where.cpu=false;break;
    case 2:bad.dtype=at::kFloat;break;case 3:bad.storage_layout=c10::Layout::Sparse;break;
    case 4:bad.contiguous=false;break;case 5:bad.shape.push_back(1);break;
    case 6:bad.shape[0]--;break;case 7:bad.shape[1]++;break;}
   assert(!Model::is_small_ragged_mask(bad,rows));
   m.refresh_small_ragged_mask(bad,true);assert(!m.small_image_ragged_sdpa_enabled(total));
  }
  // Forward cannot create proof; legacy dry planning without the mask stays dense.
  m.refresh_small_ragged_mask(t,false);assert(!m.small_image_ragged_sdpa_enabled(total));
  m.refresh_small_ragged_mask(t,true);assert(m.small_image_ragged_sdpa_enabled(total));
  const size_t slot=rows==336?0:1;
  std::weak_ptr<at::Impl> kept=t.impl;assert(!kept.expired());
  auto alias=t;alias.impl=std::make_shared<at::Impl>(); // same data, new TensorImpl
  m.refresh_small_ragged_mask(alias,false);assert(!m.small_image_ragged_sdpa_enabled(total));
  m.refresh_small_ragged_mask(t,false);assert(m.small_image_ragged_sdpa_enabled(total));
  // Cache false proofs as well, then recheck when the same owner's version changes.
  t.put(0,64,0);m.refresh_small_ragged_mask(t,false);assert(!m.small_image_ragged_sdpa_enabled(total));
  assert(!m.small_ragged_mask_valid_[slot]);
  m.refresh_small_ragged_mask(t,false);assert(!m.small_image_ragged_sdpa_enabled(total));
  t.put(0,64,-std::numeric_limits<float>::infinity());m.refresh_small_ragged_mask(t,false);
  assert(m.small_image_ragged_sdpa_enabled(total));
  m.refresh_small_ragged_mask(std::nullopt,true);
  assert(!m.small_ragged_mask_refs_[slot].defined()&&!m.small_image_ragged_sdpa_enabled(total));
  m.refresh_small_ragged_mask(t,false);assert(!m.small_image_ragged_sdpa_enabled(total));
  m.refresh_small_ragged_mask(t,true);assert(m.small_image_ragged_sdpa_enabled(total));
  auto unversioned=canonical(rows);unversioned.impl->version.on=false;
  m.refresh_small_ragged_mask(unversioned,true);
  assert(!m.small_ragged_mask_refs_[slot].defined()&&!m.small_image_ragged_sdpa_enabled(total));
  m.refresh_small_ragged_mask(t,false);assert(!m.small_image_ragged_sdpa_enabled(total));
  m.refresh_small_ragged_mask(alias,true);assert(m.small_image_ragged_sdpa_enabled(total));
  auto proof_owner=alias.impl;alias=at::Tensor();
  assert(m.small_ragged_mask_refs_[slot].unsafeGetTensorImpl()==proof_owner.get());
 }
 // Pair -> single -> pair uses independent owner/version slots in both precisions.
 for(bool fp16:{false,true}){
 Model m=exact(672,fp16);auto pair=canonical(336),single=canonical(324);
 m.refresh_small_ragged_mask(pair,true);auto pair_owner=pair.impl;pair=at::Tensor();
 assert(m.small_ragged_mask_refs_[0].unsafeGetTensorImpl()==pair_owner.get());
 m.current_num_patches_=324;m.image_batch_count_=1;m.valid_patches_per_image_=0;m.schedule_=VisionSchedule::SISC;
 m.refresh_small_ragged_mask(single,true);assert(m.small_image_ragged_sdpa_enabled(324));
 m.refresh_small_ragged_mask(std::nullopt,true);assert(m.small_ragged_mask_refs_[0].defined());
 m.current_num_patches_=672;m.image_batch_count_=2;m.valid_patches_per_image_=324;m.schedule_=VisionSchedule::MISC;
 m.refresh_small_ragged_mask(m.small_ragged_mask_refs_[0],false);assert(m.small_image_ragged_sdpa_enabled(672));
 }
}
'''
    compile_run(tmp_path, 'wall_vision_ragged_mask', source)


def test_actual_cold_routes_early_capability_and_all_layer_consumption(tmp_path):
    source = _source() + r'''
int main(){
 publish();
 for(bool fp16:{false,true})for(int64_t total:{324,672}){
  const int64_t rows=total==672?336:324,batch=total==672?2:1;
  Model m=exact(total,fp16);auto mask=canonical(rows);m.refresh_small_ragged_mask(mask,true);
  assert(m.small_image_ragged_sdpa_enabled(total));
  assert((m.small_image_schedule_flags()&2048)!=0);
  assert(m.generic_shared_attention_mask_domain()==(total==672?16:1));
  const auto expected=std::vector<int64_t>{1,rows,rows,16,16,80,8,16,
    int64_t(m.schedule_),0,324,336,9,64,80,256,0,64,128,144,208,272,288,304,320,324,rows};
  assert(m.small_image_ragged_sdpa_arguments(rows)==expected);
  auto manifest=m.produce(m.context.stage_plan);
  assert(manifest.routes.size()==size_t(1+batch));
  assert(m.physical_manifest_forward_capability(manifest).supported);
  for(int64_t i=0;i<batch;++i){const auto& r=manifest.routes[i];
   assert(r.family==FmbRouteFamily::ATTENTION&&r.site_id==QWEN25_SMALL_RAGGED_SDPA_SITE);
   assert(r.selector==int64_t(AttentionExecutionPolicy::SPM_KV_BY_MHA)&&r.flags==0);
   assert(r.arguments==expected&&r.invocation==i);
  }
  rejects([&]{m.produce(m.context.stage_plan,AttentionExecutionPolicy::DDR_KV);});
  m.context.value=manifest;m.context.consumed_.resize(manifest.routes.size());
  int window_layers=0,full_layers=0;
  for(int64_t layer=0;layer<32;++layer){
   std::fill(m.context.consumed_.begin(),m.context.consumed_.end(),0);
   launches.clear();transposes=0;m.execute_window_branch(layer);
   if((layer+1)%8==0){++full_layers;assert(launches.empty()&&transposes==0);}
   else {++window_layers;assert(launches.size()==size_t(batch)&&transposes==batch);
    for(int64_t image=0;image<batch;++image){const auto& x=launches[image];
     const uint32_t offset=image*rows*320;
     assert(x.q==0x100000+offset&&x.k==0x200000+offset&&x.out==0x400000+offset&&x.rows==rows);
     assert(m.context.consumed_[image]==1);
    }
   }
  }
  assert(window_layers==28&&full_layers==4);
  for(int mutation=0;mutation<9;++mutation){auto bad=manifest;
   switch(mutation){case 0:bad.routes.back().flags^=2048;break;
    case 1:bad.routes.pop_back();break;
    case 2:bad.routes[0].family=FmbRouteFamily::ROPE;break;
    case 3:bad.routes[0].selector=int64_t(AttentionExecutionPolicy::DDR_KV);break;
    case 4:bad.routes[0].flags=1;break;case 5:bad.routes[0].invocation=-1;break;
    case 6:bad.routes[0].invocation=batch;break;case 7:bad.routes.push_back(bad.routes[0]);break;
    case 8:bad.routes.erase(bad.routes.begin());break;
   }
   assert(!m.physical_manifest_forward_capability(bad).supported);
  }
  for(size_t a=0;a<expected.size();++a){auto bad=manifest;++bad.routes[0].arguments[a];
   assert(!m.physical_manifest_forward_capability(bad).supported);
   auto trial=m;trial.context.value=bad;
   rejects([&]{trial.consume_small_image_ragged_sdpa(rows,0);});
  }
  for(int stage=0;stage<3;++stage){auto trial=m;
   auto* p=stage==0?&trial.context.stage_plan.input:stage==1?&trial.context.stage_plan.qkv:&trial.context.stage_plan.compute;
   p->chunks[0].len--;rejects([&]{trial.produce(trial.context.stage_plan);});
   rejects([&]{trial.consume_small_image_ragged_sdpa(rows,0);});
  }
  {auto trial=m;trial.context.ready=false;rejects([&]{trial.consume_small_image_ragged_sdpa(rows,0);});}
  rejects([&]{m.consume_small_image_ragged_sdpa(rows+1,0);});
  rejects([&]{m.consume_small_image_ragged_sdpa(rows,-1);});
  rejects([&]{m.consume_small_image_ragged_sdpa(rows,batch);});
  for(int mutation=0;mutation<10;++mutation){auto trial=m;
   switch(mutation){case 0:trial.small_ragged_sdpa_available_=false;break;
    case 1:trial.small_qk_rope_available_=false;break;case 2:trial.debug=true;break;
    case 3:trial.wall_sorting_.silu_mul=false;break;case 4:trial.layer_weights_[17].up_w.dtype=fp16?at::kChar:at::kHalf;break;
    case 5:trial.fullatt_block_indexes_.pop_back();break;case 6:trial.mixed_precision_=true;break;
    case 7:trial.current_num_patches_=768;break;case 8:trial.max_hw_=256;break;
    case 9:trial.small_ragged_mask_rows_--;break;
   }
   assert(!trial.small_image_ragged_sdpa_enabled(total));
   assert(!trial.physical_manifest_forward_capability(manifest).supported);
  }
  // Revalidation before fast replay rejects a manifest whose formerly exact mask changed.
  mask.put(0,64,0);m.refresh_small_ragged_mask(mask,false);
  assert(!m.physical_manifest_forward_capability(manifest).supported);
  auto dense=m.produce(m.context.stage_plan);assert(dense.routes.size()==1);
  assert(m.physical_manifest_forward_capability(dense).supported);
  assert(m.generic_shared_attention_mask_domain()==(total==672?16:17));
  publish(false);auto fallback=exact(total,fp16);auto valid=canonical(rows);
  fallback.refresh_small_ragged_mask(valid,true);
  assert(!fallback.small_image_ragged_sdpa_enabled(total));
  assert(fallback.produce(fallback.context.stage_plan).routes.size()==1);
  assert(!fallback.physical_manifest_forward_capability(manifest).supported);
  publish();assert(!fallback.small_image_ragged_sdpa_enabled(total)); // cold snapshot
  fallback.bind();fallback.refresh_small_ragged_mask(valid,true);
  assert(fallback.small_image_ragged_sdpa_enabled(total));
 }
}
'''
    compile_run(tmp_path, 'wall_vision_ragged_routes', source)

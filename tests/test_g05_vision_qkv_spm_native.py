"""Execute the real G05 QKV transport admission, layout, plan and callback bodies."""
from pathlib import Path
from pi_wall_host_catalog import with_actual_linear_contracts
import re
import subprocess

from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]


def _qkv_native_cpp():
    """Shared real-native harness; optional triple payload is absent by default."""
    source = (ROOT / 'src/fused/rpu_qwen3_5_vision_model.cpp').read_text()
    header = (ROOT / 'src/fused/rpu_qwen3_5_vision_model.h').read_text()
    methods = [
        ('std::vector<int64_t> g05_high_precision_encoder_arguments(int64_t rows) const', 'Qwen3_5VisionModel::g05_high_precision_encoder_arguments('),
        ('void consume_g05_high_precision_encoder_activation(int64_t rows, int64_t invocation)', 'Qwen3_5VisionModel::consume_g05_high_precision_encoder_activation('),
        ('bool is_g05_k1_fast_replay_profile() const', 'bool Qwen3_5VisionModel::is_g05_k1_fast_replay_profile('),
        ('bool is_g05_qkv_spm_profile() const', 'bool Qwen3_5VisionModel::is_g05_qkv_spm_profile('),
        ('bool use_g05_qkv_spm_layout(const LayoutContext& layout) const', 'bool Qwen3_5VisionModel::use_g05_qkv_spm_layout('),
        ('bool g05_qkv_spm_schedule(const FmbThreeStageChunkPlan& plan) const', 'bool Qwen3_5VisionModel::g05_qkv_spm_schedule('),
        ('std::vector<int64_t> g05_qkv_spm_arguments() const', 'std::vector<int64_t> Qwen3_5VisionModel::g05_qkv_spm_arguments('),
        ('bool consume_g05_qkv_spm_transport(const ChunkInfo& chunk, bool compute)', 'bool Qwen3_5VisionModel::consume_g05_qkv_spm_transport('),
        ('std::vector<int64_t> g05_qkv_linear_arguments() const', 'std::vector<int64_t> Qwen3_5VisionModel::g05_qkv_linear_arguments('),
        ('bool consume_g05_qkv_linear(const ChunkInfo& chunk, bool carry_qkv)', 'bool Qwen3_5VisionModel::consume_g05_qkv_linear('),
        ('void prepare_g05_owner_ln()', 'void Qwen3_5VisionModel::prepare_g05_owner_ln('),
        ('int64_t subclass_layout_hash() const', 'int64_t Qwen3_5VisionModel::subclass_layout_hash('),
        ('void emit_kv_first_body(int layer_idx, const ChunkInfo& chunk)', 'void Qwen3_5VisionModel::emit_kv_first_body('),
    ]
    layout = _block(source, 'std::vector<BufferDecl> Qwen3_5VisionModel::declare_buffers(')
    layout_prefix = layout[1:layout.index('    SdpaConfig')]
    declarations = _block(layout, 'decls = {\n            // ALL')
    compute = _block(source, 'void Qwen3_5VisionModel::build_layer_subgraph(')
    compute = compute[1:compute.index('    // SDPA：')]
    manifest = _block(source, 'Qwen3_5VisionModel::physical_manifest_for_candidate(')
    carry = manifest[manifest.index('    const bool carry_qkv ='):manifest.index('    auto append_kv_route')]
    plan = _block(manifest, manifest[manifest.rindex('        for (const ChunkInfo& chunk : plan.qkv.chunks)'):][:70])
    constants = re.findall(r'constexpr int64_t ((?:G05_VISION|QWEN35_VISION)_[A-Z0-9_]+)\s*=\s*([0-9]+)(?:LL)?;', source)
    kernel_header = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    start = kernel_header.index('enum class KernelId')
    kernel_enum = kernel_header[start:kernel_header.index('};', start) + 2]
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok,...) do {if(!(ok))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;}
namespace at {constexpr int kHalf=1,kChar=2,kPrivateUse1=1;
struct Tensor{bool present=true,contiguous=true;int dtype=kHalf,location=kPrivateUse1;int64_t owner=42;int rank=2;int64_t rows=1024,cols=1024;
 bool track_owner=false;inline static int metadata_reads=0;
 bool defined()const{if(track_owner)++metadata_reads;return present;}int scalar_type()const{if(track_owner)++metadata_reads;return dtype;}bool is_contiguous()const{if(track_owner)++metadata_reads;return contiguous;}
 struct Device{int location;int type()const{return location;}};Device device()const{return {location};}
 template<class T>T* data_ptr()const{static T data[32768];return data;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}
};}
struct Weights{at::Tensor q_w,k_w,v_w,o_w,fc1_w,fc2_w;at::Tensor q_ws{false},k_ws{false},v_ws{false},o_ws{false},fc1_ws{false},fc2_ws{false};};
struct Norms{at::Tensor ln1_w,ln1_b,ln2_w,ln2_b,q_b,k_b,v_b,o_b,fc1_b,fc2_b;};
void append_kvinsert_cost_scalar_identity(std::vector<int64_t>& out,double value){int64_t bits;std::memcpy(&bits,&value,8);out.push_back(bits);}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>& out,const at::Tensor& tensor){out.push_back(tensor.owner);}
namespace detail{int64_t layout_mix(int64_t a,int64_t b){return uint64_t(a)*131+uint64_t(b);}}
__CONSTANTS__
constexpr int DWIDTH=2;constexpr int QWEN35_VISION_KV_CAPABILITIES=1;
__KERNEL_ENUM__
struct KernelCache{ bool is_initialized()const{return true;} void* get(KernelId id){return has_loaded(id)?this:nullptr;}bool qkv=false,w8=false,acc16=false,owner=false;int queries=0;static KernelCache& instance(){static KernelCache c;return c;}bool has_loaded(KernelId id){++queries;if(id==KernelId::G05_VISION_FC2_WEIGHT_OUTER_W8A16_ACC16_M768N1024K512)return false;if(id==KernelId::G05_VISION_QKV_FP16_ACC32_M256N128K1024)return qkv;if(id==KernelId::G05_VISION_QKV_W8A16_ACC32_M256N128K1024)return w8;if(id==KernelId::G05_VISION_QKV_W8A16_ACC16_M256N128K1024)return acc16;return owner;}};
enum class AttentionExecutionPolicy{DDR_KV,SPM_KV_BY_MHA};enum class ChunkMode{KV_FIRST,SEQ_FIRST};
struct ChunkInfo{int64_t idx=0,offset=0,len=256,kv_seq_len=256;};
struct Span{int64_t offset=0,len=256,group_id=0;};struct Stage{std::vector<ChunkInfo> chunks{{}};};
struct FmbThreeStageChunkPlan{ChunkMode chunk_mode=ChunkMode::KV_FIRST;std::vector<Span>spans{{}};Stage qkv,compute;};
struct LayoutContext{AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;int64_t batch_size=1,chunk_size=256,kv_cs=256,max_kv_seq_len=256;bool is_causal=false,use_attn_mask=false;int64_t effective_kv_cs()const{return kv_cs;}};
enum class FmbRouteFamily{KV_INSERT,GRAPH_SCHEDULE,LINEAR,ROPE};
struct Route{FmbRouteFamily family;int64_t site,selector,flags;std::vector<int64_t>arguments;int64_t invocation;};
struct Context{bool complete=true,input_in_spm=true;AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;FmbThreeStageChunkPlan stage_plan;int64_t position=0,seq_len=256,batch_size=1;bool is_causal=false;std::optional<int> attention_mask;
 std::vector<at::Tensor> caches=std::vector<at::Tensor>(24);std::vector<at::Tensor>*k_caches=&caches,*v_caches=&caches;std::vector<Route> routes;std::vector<int64_t>phases;
 bool has_complete_physical_manifest()const{return complete;}
 const Route& find_physical_route(FmbRouteFamily family,int64_t site,int64_t invocation)const{auto i=std::find_if(routes.begin(),routes.end(),[&](auto&r){return r.family==family&&r.site==site&&r.invocation==invocation;});TORCH_CHECK(i!=routes.end());return *i;}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>arguments,int64_t invocation){auto&r=find_physical_route(family,site,invocation);TORCH_CHECK(r.selector==selector&&r.flags==flags&&r.arguments==arguments);if(family==FmbRouteFamily::GRAPH_SCHEDULE)phases.push_back(invocation);}
};
struct KvInsertSegmentPlan{struct Segment{int64_t position=0;};int64_t logical=256,physical=256,position=0;int64_t logical_rows()const{return logical;}int64_t physical_rows()const{return physical;}Segment segment(int)const{return {position};}int route()const{return 1;}};
KvInsertSegmentPlan restore_kvinsert_plan(int64_t,const std::vector<int64_t>&a,int,int,int){return {a.at(0),a.at(1),a.at(2)};}
KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan_auto(int64_t p,int64_t l,int64_t r,int,int,int,int){return {l,r,p};}
std::vector<std::string>emitted;
template<class...A>void rpu_launch_insert_kvcache_spm_unified_with_plan(A...){emitted.push_back("kv-ddr");}
template<class...A>void rpu_launch_spm_scatter_ddr_dma(A...){emitted.push_back("q-store");}
template<class...A>void rpu_launch_ddr_scatter_spm_dma(A...){emitted.push_back("q-load");}
template<class...A>void rpu_launch_layernorm_spm_kernel(A...){emitted.push_back("ln");}
template<class...A>void rpu_launch_rope_2d_spm_kernel(A...){emitted.push_back("rope");}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t,const at::Tensor&,uint32_t,int64_t,int64_t,int64_t,int,int,uint32_t,const at::Tensor&,int,int,bool){emitted.push_back("linear");}
void rpu_launch_g05_vision_qkv_linear_fp16_acc32_kernel(uint32_t input,const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,uint32_t qo,uint32_t ko,uint32_t vo,uint32_t qb,uint32_t kb,uint32_t vb){
 assert(input==0x10000&&qo==0x90000&&ko==0xa0000&&vo==0xb0000);
 assert(q.owner!=k.owner&&q.owner!=v.owner&&k.owner!=v.owner);
 assert(qb>=0x180000&&kb==qb+512&&vb==qb+1024);emitted.push_back("qkv");
}
void rpu_launch_g05_vision_qkv_linear_w8a16_acc32_kernel(uint32_t input,const at::Tensor&q,const at::Tensor&k,const at::Tensor&v,uint32_t qo,uint32_t ko,uint32_t vo,uint32_t qb,uint32_t kb,uint32_t vb,const at::Tensor&qs,const at::Tensor&ks,const at::Tensor&vs,bool acc32=true){
 assert(input==0x10000&&qo==0x90000&&ko==0xa0000&&vo==0xb0000);
 assert(q.owner!=k.owner&&q.owner!=v.owner&&k.owner!=v.owner);
 assert(qs.owner==q.owner+10000&&ks.owner==k.owner+10000&&vs.owner==v.owner+10000);
 assert(qb>=0x180000&&kb==qb+512&&vb==qb+1024);emitted.push_back(acc32?"qkv-w8":"qkv-w8-acc16");
}
bool dbg_q_enabled(){return false;}int64_t qwen35_vision_linear_route(){return 8;}int64_t qwen35_vision_rope_route(){return 9;}
int64_t Align(int64_t x,int64_t a){return (x+a-1)/a*a;}
enum class StorageClass{Temp};enum class BufferScope{LayerWide,KvInsert,Compute};
struct BufferDecl{std::string name;int64_t size,first,last;StorageClass storage;int count;const char*alias;BufferScope scope;};
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

 bool g05_owner_ln_available_=false,g05_qkv_spm_available_=false,g05_qkv_linear_available_=false,g05_w8a16_=false;
 bool linear_acc32_=true,gelu_erf_ultra_=false,moe_precision_=false,has_temporal_=false,has_rope_=true,has_step0_=true,has_merger_=true,z2_bound_=false,debug=false,current_compact_step0_=false;
 bool high_precision_=false;
 int cores=8,attention_tp_=8,mlp=8,layers=24,hidden=1024,intermediate=4096,q=16,kv=16,hd=64,orig_head_dim_=64,patch_dim_=1536,merger_hidden_=4096,out_hidden_size_=2048;
 int current_temporal_num_frames_=1,current_camera_batch_count_=1,current_num_patches_=0,current_output_patches_=256;double eps_=1e-6;bool alias=true;
 std::vector<Weights>layer_weights_{24};std::vector<Norms>layer_bias_norm_{24};Context context;
 at::Tensor position_idx_keepalive_,q_ddr_buf_,dbg_q_,pe_w_,pe_b_,merger_fc1_w_,merger_fc1_b_,merger_fc2_w_,merger_fc2_b_,merger_norm_w_,merger_norm_b_;
 int num_cores()const{return cores;}int attn_tp()const{return attention_tp_;}int mlp_tp()const{return mlp;}int num_layers()const{return layers;}int hidden_size()const{return hidden;}int intermediate_size()const{return intermediate;}int num_q_heads()const{return q;}int num_kv_heads()const{return kv;}int head_dim()const{return hd;}
 bool get_debug_export()const{return debug;}Context&ctx(){return context;}uint32_t addr(int,const char*n)const{std::string s(n);return s=="q_comp"?(alias?0x90000:0xc0000):s=="q"?0x90000:s=="k"?0xa0000:s=="v"?0xb0000:0x10000;}uint32_t layer_addr(int layer,int,const char*n)const{std::string s(n);return 0x180000+layer*4096+(s=="k_bias"?512:s=="v_bias"?1024:0);}struct Offset{uint32_t value=0;};Offset addr_offset(const char*){return {};}
 void emit_layer_input_dma(int,const ChunkInfo&){emitted.push_back("input");}void emit_temporal_layer(int){emitted.push_back("temporal");}
 void consume_manifest_route(FmbRouteFamily f,int64_t s,int64_t select,int64_t inv){context.consume_physical_route(f,s,select,0,{},inv);}
 __METHODS__
 std::vector<int64_t>kvinsert_cost_weight_identity()const __IDENTITY__
 std::vector<BufferDecl>buffers(const LayoutContext&ctx){
  __LAYOUT_PREFIX__
  constexpr BufferScope ALL=BufferScope::LayerWide,KVIN=BufferScope::KvInsert,COMP=BufferScope::Compute;const int64_t sdpa_tmp=65536;
  return __DECLARATIONS__;
 }
 void plan(const LayoutContext&layout,int64_t physical_len=256,int64_t logical_len=256,int64_t position=0){
  const auto&plan=context.stage_plan;const bool raw_spm=layout.attention_policy==AttentionExecutionPolicy::SPM_KV_BY_MHA;
  __CARRY__
  context.routes.clear();auto append_route=[&](FmbRouteFamily f,int64_t s,int64_t sel,int64_t inv,std::vector<int64_t>a=std::vector<int64_t>{}){context.routes.push_back({f,s,sel,0,a,inv});};
  auto append_kv_route=[&](int64_t s,int64_t inv,int64_t pos,int64_t rows,int,int,int64_t flags){context.routes.push_back({FmbRouteFamily::KV_INSERT,s,1,flags,{rows,rows,pos},inv});};
  for(const ChunkInfo&chunk:plan.qkv.chunks)__PLAN__
 }
 void compute(int layer_idx,const ChunkInfo&chunk){__COMPUTE__}
};
template<class F>void rejects_before_emission(F f){const auto previous=emitted;try{f();assert(false);}catch(const std::runtime_error&){}assert(emitted==previous);}
const BufferDecl& named(const std::vector<BufferDecl>&d,std::string name){auto i=std::find_if(d.begin(),d.end(),[&](auto&b){return b.name==name;});assert(i!=d.end());return *i;}
int main(){
 Model m;LayoutContext layout;m.prepare_g05_owner_ln();assert(m.g05_qkv_spm_available_&&!m.g05_owner_ln_available_);
 const auto identity=m.kvinsert_cost_weight_identity();const auto hash=m.subclass_layout_hash();
 auto d=m.buffers(layout);assert(named(d,"q").size==65536&&named(d,"q").scope==BufferScope::LayerWide&&named(d,"q").first==2&&named(d,"q").last==4);assert(named(d,"q_comp").size==0&&std::string(named(d,"q_comp").alias)=="q"&&named(d,"q_comp").scope==BufferScope::LayerWide);
 for(std::string n:{"k","v"})assert(named(d,n).scope==BufferScope::LayerWide&&named(d,n).last==4);
 m.plan(layout);assert(m.context.find_physical_route(FmbRouteFamily::KV_INSERT,QWEN35_VISION_KV_SITE,0).flags==2);
 assert(m.g05_qkv_spm_arguments()==std::vector<int64_t>({1,256,256,1024,16,16,64,8,65536,1,2,1,1,0}));
 for(int i=0;i<3;++i){emitted.clear();m.emit_kv_first_body(23,{});m.compute(23,{});assert(emitted==std::vector<std::string>({"ln","linear","linear","linear","rope","rope"}));}
 assert(m.context.phases==std::vector<int64_t>({0,1,0,1,0,1}));
 // Raw -> DDR -> Raw changes both declarations and executed transport.
 layout.attention_policy=m.context.attention_policy=AttentionExecutionPolicy::DDR_KV;m.plan(layout);d=m.buffers(layout);
 assert(named(d,"q").scope==BufferScope::KvInsert&&named(d,"q").last==3);assert(named(d,"q_comp").size==65536&&!named(d,"q_comp").alias&&named(d,"q_comp").scope==BufferScope::Compute);
 assert(m.context.find_physical_route(FmbRouteFamily::KV_INSERT,QWEN35_VISION_KV_SITE,0).flags==1);
 emitted.clear();m.emit_kv_first_body(0,{});m.compute(0,{});assert(emitted==std::vector<std::string>({"ln","linear","linear","linear","rope","rope","kv-ddr","q-store","q-load"}));
 layout.attention_policy=m.context.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;m.plan(layout);assert(m.use_g05_qkv_spm_layout(layout));
 emitted.clear();m.emit_kv_first_body(0,{});m.compute(0,{});assert(emitted==std::vector<std::string>({"ln","linear","linear","linear","rope","rope"}));
 auto reject_both=[&](auto change){Model x=m;change(x);for(bool compute:{false,true})rejects_before_emission([&]{if(compute)x.compute(0,{});else x.emit_kv_first_body(0,{});});};
 reject_both([](auto&x){x.alias=false;});reject_both([](auto&x){x.context.complete=false;});reject_both([](auto&x){x.context.position=1;});reject_both([](auto&x){x.context.attention_mask=1;});reject_both([](auto&x){x.context.batch_size=2;});reject_both([](auto&x){x.context.is_causal=true;});reject_both([](auto&x){x.context.seq_len=255;});
 reject_both([](auto&x){for(auto&r:x.context.routes)if(r.family==FmbRouteFamily::KV_INSERT)r.flags=1;});
 reject_both([](auto&x){for(auto&r:x.context.routes)if(r.family==FmbRouteFamily::GRAPH_SCHEDULE)r.arguments[8]=0;});
 for(int phase:{0,1}){auto x=m;x.context.routes.erase(std::remove_if(x.context.routes.begin(),x.context.routes.end(),[&](auto&r){return r.family==FmbRouteFamily::GRAPH_SCHEDULE&&r.invocation==phase;}),x.context.routes.end());rejects_before_emission([&]{if(phase)x.compute(0,{});else x.emit_kv_first_body(0,{});});}
 for(int fault=0;fault<4;++fault){ChunkInfo chunk;if(fault==0)chunk.idx=1;if(fault==1)chunk.offset=1;if(fault==2)chunk.len=128;if(fault==3)chunk.kv_seq_len=512;for(bool compute:{false,true})rejects_before_emission([&]{if(compute)m.compute(0,chunk);else m.emit_kv_first_body(0,chunk);});}
 for(int fault=0;fault<7;++fault){auto x=m;auto&p=x.context.stage_plan;if(fault==0)p.chunk_mode=ChunkMode::SEQ_FIRST;if(fault==1)p.spans.push_back({});if(fault==2)p.spans[0].group_id=1;if(fault==3)p.qkv.chunks[0].kv_seq_len=512;if(fault==4)p.compute.chunks[0].offset=1;if(fault==5)p.compute.chunks[0].idx=1;if(fault==6)p.qkv.chunks[0].len=128;assert(!x.g05_qkv_spm_schedule(p));x.plan(layout);assert(x.context.find_physical_route(FmbRouteFamily::KV_INSERT,QWEN35_VISION_KV_SITE,0).flags==1);rejects_before_emission([&]{x.compute(0,{});});}
 for(int fault=0;fault<6;++fault){auto l=layout;if(fault==0)l.use_attn_mask=true;if(fault==1)l.batch_size=2;if(fault==2)l.chunk_size=128;if(fault==3)l.kv_cs=128;if(fault==4)l.max_kv_seq_len=512;if(fault==5)l.is_causal=true;assert(!m.use_g05_qkv_spm_layout(l));assert(!named(m.buffers(l),"q_comp").alias);}
 auto cold=[&](auto change){auto x=m;change(x);x.prepare_g05_owner_ln();assert(!x.g05_qkv_spm_available_&&!x.is_g05_qkv_spm_profile()&&x.subclass_layout_hash()!=hash&&x.kvinsert_cost_weight_identity()!=identity);};
 cold([](auto&x){x.layer_weights_.back().fc2_w.dtype=2;});cold([](auto&x){x.layer_bias_norm_.back().fc2_b.present=false;});cold([](auto&x){x.layer_bias_norm_.back().ln2_w.location=0;});cold([](auto&x){x.layer_weights_.back().v_w.contiguous=false;});cold([](auto&x){x.linear_acc32_=false;});
}
'''
    replacements = {'__KERNEL_ENUM__': kernel_enum, '__CONSTANTS__': '\n'.join(f'constexpr int64_t {name}={value}LL;' for name, value in constants),
                    '__METHODS__': '\n'.join(decl + _block(source, marker) for decl, marker in methods),
                    '__IDENTITY__': _block(header, 'std::vector<int64_t> kvinsert_cost_weight_identity() const'),
                    '__LAYOUT_PREFIX__': layout_prefix, '__DECLARATIONS__': declarations,
                    '__COMPUTE__': compute, '__CARRY__': carry, '__PLAN__': plan}
    for marker, value in replacements.items():
        cpp = cpp.replace(marker, value)
    cpp = chr(10).join(('#include <array>', '#include "' + str(ROOT / "src/ops/rpu_linear_tiling.h") + '"', cpp))
    return with_actual_linear_contracts(cpp)


def _compile_qkv(tmp_path, cpp):
    path, exe = tmp_path / 'qkv.cpp', tmp_path / 'qkv'
    path.write_text(cpp)
    result = subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-I', str(ROOT / 'src/core'), str(path), '-o', str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_qkv_spm_transport_native_admission_layout_callbacks_and_fallback(tmp_path):
    _compile_qkv(tmp_path, _qkv_native_cpp())


def test_fp16_acc16_transport_requires_complete_owners_and_keeps_real_acc16_linears(tmp_path):
    cpp = _qkv_native_cpp().split('int main(){', 1)[0]
    cpp = cpp.replace('std::vector<std::string>emitted;',
                      'std::vector<std::string>emitted;std::vector<bool>linear_modes;')
    cpp = cpp.replace(
        'const at::Tensor&,uint32_t,int64_t,int64_t,int64_t,int,int,uint32_t,const at::Tensor&,int,int,bool){emitted.push_back("linear");}',
        'const at::Tensor&w,uint32_t,int64_t,int64_t,int64_t,int,int,uint32_t,const at::Tensor&s,int,int,bool acc32){assert(w.dtype==at::kHalf&&!s.defined());linear_modes.push_back(acc32);emitted.push_back("linear");}')
    cpp += r'''
int main(){
 Model m;LayoutContext layout;m.linear_acc32_=false;auto&cache=KernelCache::instance();cache.qkv=cache.owner=true;
 for(auto&w:m.layer_weights_){w.fc1_w.rows=4096;w.fc2_w.cols=4096;}
 for(auto&n:m.layer_bias_norm_){for(auto*t:{&n.ln1_w,&n.ln1_b,&n.ln2_w,&n.ln2_b,&n.q_b,&n.k_b,&n.v_b,&n.o_b,&n.fc1_b,&n.fc2_b}){t->rank=1;t->rows=t==&n.fc1_b?4096:1024;}}
 m.prepare_g05_owner_ln();assert(m.g05_qkv_spm_available_&&m.g05_owner_ln_available_&&!m.g05_w8a16_&&!m.g05_qkv_linear_available_&&!m.g05_k1_b3_available_);
 assert(m.is_g05_k1_fast_replay_profile());m.plan(layout);
 const auto hash=m.subclass_layout_hash();const auto identity=m.kvinsert_cost_weight_identity();
 for(const auto&r:m.context.routes)assert(r.site!=G05_VISION_QKV_LINEAR_SITE&&r.site!=G05_VISION_QKV_W8_LINEAR_SITE);
 const auto queries=cache.queries;
 for(int layer=0;layer<24;++layer){emitted.clear();linear_modes.clear();m.emit_kv_first_body(layer,{});m.compute(layer,{});assert(emitted==std::vector<std::string>({"ln","linear","linear","linear","rope","rope"}));assert(linear_modes==std::vector<bool>(3,false));}
 assert(cache.queries==queries);
 {auto other=m;other.linear_acc32_=true;other.prepare_g05_owner_ln();assert(other.g05_qkv_linear_available_&&other.subclass_layout_hash()!=hash&&other.kvinsert_cost_weight_identity()!=identity);rejects_before_emission([&]{other.emit_kv_first_body(0,{});});}
 // Every weight, absent scale and bias/norm extent participates in the cold proof.
 auto cold=[&](auto change){auto x=m;change(x);x.prepare_g05_owner_ln();assert(!x.g05_qkv_spm_available_&&!x.g05_owner_ln_available_&&!x.is_g05_k1_fast_replay_profile());};
 for(int layer=0;layer<24;++layer){
  for(int family=0;family<6;++family)for(int fault=0;fault<5;++fault)cold([&](auto&x){auto&w=x.layer_weights_[layer];at::Tensor*weights[]={&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.fc1_w,&w.fc2_w};at::Tensor*scales[]={&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.fc1_ws,&w.fc2_ws};auto&t=*weights[family];if(fault==0)t.rank=1;if(fault==1)--t.rows;if(fault==2)--t.cols;if(fault==3)t.dtype=at::kChar;if(fault==4)scales[family]->present=true;});
  for(int family=0;family<10;++family)for(int fault=0;fault<2;++fault)cold([&](auto&x){auto&n=x.layer_bias_norm_[layer];at::Tensor*norms[]={&n.ln1_w,&n.ln1_b,&n.ln2_w,&n.ln2_b,&n.q_b,&n.k_b,&n.v_b,&n.o_b,&n.fc1_b,&n.fc2_b};if(fault==0)norms[family]->rank=2;else --norms[family]->rows;});
 }
 // Old expansion still admits transport and generated ACC16; optional owner LN falls back.
 cache.qkv=cache.owner=false;m.prepare_g05_owner_ln();assert(m.g05_qkv_spm_available_&&!m.g05_owner_ln_available_&&!m.g05_qkv_linear_available_);m.plan(layout);
 emitted.clear();m.emit_kv_first_body(0,{});m.compute(0,{});assert(emitted==std::vector<std::string>({"ln","linear","linear","linear","rope","rope"}));
 auto x=m;for(auto&r:x.context.routes)if(r.family==FmbRouteFamily::GRAPH_SCHEDULE){r.arguments[0]++;break;}rejects_before_emission([&]{x.emit_kv_first_body(0,{});});
}
'''
    _compile_qkv(tmp_path, cpp)

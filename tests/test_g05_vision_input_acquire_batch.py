"""Execute the actual whole Vision entry with host DDR/Graph boundary doubles.

The fixture preserves production guards, refs, ranges and publication order.
It tests current bytes and delayed mutable consumption, not device kernel math
or physical SDK barriers. Hardware original-profile gates remain separate.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vision_four_original_ranges_before_delayed_consumer(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    from test_siglip_attention_dry_live_contract import _function, _run_native

    source = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    body = _function(source, "at::Tensor Qwen3_5VisionModel::forward_impl(")
    # Only device labels and the host barrier are doubled. All original
    # tensor/geometry/error checks and all four acquire sites run unchanged.
    body = body.replace("at::kPrivateUse1", "at::kCPU")
    body = body.replace("at::Tensor Qwen3_5VisionModel::forward_impl(",
                        "at::Tensor forward_impl(")
    body = body.replace("#if defined(__aarch64__)", "#if 1")
    body = body.replace('__asm__ volatile("dsb sy" ::: "memory");',
                        "system_barrier();")
    allocation = """at::empty(
                {1, keepalive_capacity_ / QWEN3_5_SPATIAL_MERGE_UNIT,
                 out_hidden_size_},
                input.options())"""
    assert body.count(allocation) == 1
    body = body.replace(allocation,
                        "bank({1, keepalive_capacity_ / "
                        "QWEN3_5_SPATIAL_MERGE_UNIT, out_hidden_size_})")
    predicates = []
    for name in ("is_g05_k1_fast_replay_profile", "g05_k1_b3_available",
                 "g05_k1_camera_group_available", "is_g05_k1_camera_group_profile",
                 "is_g05_k1_b3_profile"):
        method = _function(source, "bool Qwen3_5VisionModel::" + name + "(")
        predicates.append(method.replace("Qwen3_5VisionModel::", ""))

    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include "core/rpu_profile.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <vector>
bool g_rpu_ddr_flush_enabled=false,g_rpu_ddr_flush_force_enabled=true;
constexpr int64_t keepalive_capacity_=4608;
constexpr int64_t QWEN3_5_SPATIAL_MERGE_UNIT=4,G05_MAX_CAMERA_BATCH=3;
constexpr int64_t G05_COMPACT_CHANNELS=3,G05_COMPACT_PATCH_SIDE=16;
constexpr int64_t G05_COMPACT_IMAGE_SIDE=256,G05_TEMPORAL_PATCH_GROUP=16;
struct Range{void* ptr;size_t bytes;};
struct Owner{
 at::Tensor backing;
 c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl> view;
 std::vector<uint8_t> ddr;uintptr_t base;size_t bytes;bool lookup=true;
 Owner(at::Tensor b,const at::Tensor& v):backing(std::move(b)),view(v.getIntrusivePtr()),
  ddr(backing.nbytes(),0xa7),base(reinterpret_cast<uintptr_t>(backing.data_ptr())),bytes(backing.nbytes()){}
 void* get_cpu_ptr(){return reinterpret_cast<void*>(base);}
 size_t get_memory_size(){return bytes;}
};
std::vector<std::unique_ptr<Owner>> owners;
std::vector<Range> scalars;std::vector<std::vector<Range>> batches;
std::vector<std::string> timeline;
int batch_completions=0,system_completions=0;bool debug_export=false;
Owner& locate(void* p,size_t& offset){
 const auto a=reinterpret_cast<uintptr_t>(p);
 for(auto it=owners.rbegin();it!=owners.rend();++it){auto& o=**it;
  const auto b=reinterpret_cast<uintptr_t>(o.backing.data_ptr());
  if(a>=b&&a-b<o.ddr.size()&&!o.view.expired()){offset=a-b;return o;}}
 TORCH_CHECK(false,"lost current DDR tensor lease");
}
at::Tensor bank(at::IntArrayRef shape,at::ScalarType dtype=at::kHalf,int offset=32){
 int64_t count=1;for(auto x:shape)count*=x;
 auto flat=at::zeros({((count+offset+64+31)/32)*32},dtype);
 assert(reinterpret_cast<uintptr_t>(flat.data_ptr())%64==0);
 auto t=flat.narrow(0,offset,count).view(shape);
 owners.emplace_back(std::make_unique<Owner>(flat,t));return t;
}
void publish(void* p,size_t bytes){size_t off;auto& o=locate(p,off);
 assert(bytes>0&&bytes<=o.ddr.size()-off);
 std::memcpy(o.ddr.data()+off,p,bytes);
}
void system_barrier(){++system_completions;timeline.push_back("dsb-sy");}
namespace rhino_lkn {
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
Owner* RpuGetHostddr(void* p){size_t off;auto& o=locate(p,off);return o.lookup?&o:nullptr;}
void RpuDdrFlush(void* p){size_t off;auto& o=locate(p,off);scalars.push_back({p,o.ddr.size()-off});publish(p,o.ddr.size()-off);}
void RpuDdrFlush(void* p,size_t bytes){scalars.push_back({p,bytes});timeline.push_back("scalar");publish(p,bytes);}
void RpuDdrFlushBatch(void** ptrs,size_t* sizes,size_t count){
 assert(count==4);std::vector<Range> packet;
 for(size_t i=0;i<count;++i){packet.push_back({ptrs[i],sizes[i]});publish(ptrs[i],sizes[i]);}
 batches.push_back(packet);++batch_completions;timeline.push_back("batch");
}
}
bool get_debug_export(){return debug_export;}
bool dbg_q_enabled(){return true;}
at::Tensor empty_256b_aligned_fp16_rpu(at::IntArrayRef s,const at::TensorOptions& o,const char*){return at::empty(s,o);}
struct Plan{struct Stage{std::vector<int> chunks;}input;std::vector<int> spans;int boundary_policies=0;};
uint64_t fmb_three_stage_chunk_plan_fingerprint(const Plan&){return 0;}
struct VisionExecutionPlan{int64_t input_chunk_size=0,qkv_chunk_size=0,compute_chunk_size=0;Plan stages;};
struct Candidate{int64_t input_chunk_size=0,qkv_chunk_size=0,compute_chunk_size=0;Plan stage_plan;};
struct Prepared{Candidate plan;const Candidate& candidate()const{return plan;}};
struct Context{Plan stage_plan;};
struct Qwen3_5VisionModel{
 bool z2_bound_=false,has_rope_=true,has_step0_=true,has_merger_=true,has_temporal_=false;
 bool linear_acc32_=false,g05_w8a16_=false,g05_qkv_spm_available_=true;
 bool gelu_erf_ultra_=false,moe_precision_=false,high_precision_=false;
 bool g05_k1_b3_available_=true,g05_owner_ln_available_=true,g05_qkv_linear_available_=true;
 std::array<bool,2>g05_k1_generated_group_available_{{true,true}};
 int64_t z2_prepared_num_patches_=256,attention_tp_=8,mlp_tp_value=8;
 int64_t orig_head_dim_=64,patch_dim_=1536,merger_hidden_=4096,out_hidden_size_=2048;
 double eps_=1e-6;
 int64_t current_num_patches_=0,current_output_patches_=0,current_camera_batch_count_=1;
 int64_t current_temporal_num_frames_=1;bool current_compact_step0_=false;
 uint32_t z2_slice_addr_=0;bool merger_wrote_fusion_target_=false;
 at::Tensor position_idx_keepalive_=bank({4608,2},at::kShort);
 at::Tensor q_ddr_buf_,temporal_k_cache_,temporal_v_cache_,dbg_hidden_,dbg_q_,hidden_buf_,merged_buf_;
 at::Tensor pixel_src_ref_,step0_pos_ref_,merger_dst_ref_;
 uint64_t pixel_src_base_=0,step0_pos_src_base_=0;
 std::array<uint64_t,3>merger_dst_bases_{};
 Context context;std::vector<std::function<void()>>nodes;
 std::vector<std::vector<uint8_t>>seen;int builds=0,consumes=0;
 int64_t num_layers()const{return 24;}int64_t num_cores()const{return 8;}
 int64_t attn_tp()const{return attention_tp_;}int64_t mlp_tp()const{return mlp_tp_value;}
 int64_t hidden_size()const{return 1024;}int64_t intermediate_size()const{return 4096;}
 int64_t num_q_heads()const{return 16;}int64_t num_kv_heads()const{return 16;}int64_t head_dim()const{return 64;}
 void configure_execution_geometry(int64_t rows,int64_t frames,int64_t cameras,bool compact){
  current_output_patches_=rows;current_camera_batch_count_=cameras;
  current_temporal_num_frames_=frames;current_compact_step0_=compact;
 }
 VisionExecutionPlan resolve_execution_plan(int64_t rows){VisionExecutionPlan p;p.input_chunk_size=p.qkv_chunk_size=p.compute_chunk_size=rows;return p;}
 std::shared_ptr<Prepared> prepare_stage_candidate(at::IntArrayRef){auto p=std::make_shared<Prepared>();p->plan.input_chunk_size=p->plan.qkv_chunk_size=p->plan.compute_chunk_size=current_output_patches_;return p;}
 Context& ctx(){return context;}int64_t get_last_resolved_chunk_size(){return current_output_patches_;}
 template<class...Args>at::Tensor run_all_layers(const at::Tensor&,Args&&...){
  timeline.push_back("record-consumer");assert(!timeline.empty());
  if(!batches.empty()){assert(scalars.empty()&&batch_completions==1&&system_completions==1);}
  if(nodes.empty()){
   // Match the retained graph's mutable member addresses; consumption occurs
   // after forward and caller locals are dropped, not while it is recorded.
   nodes.push_back([this]{
    assert(pixel_src_ref_.defined()&&step0_pos_ref_.defined()&&merger_dst_ref_.defined());
    assert(pixel_src_base_==reinterpret_cast<uint64_t>(pixel_src_ref_.data_ptr()));
    assert(step0_pos_src_base_==reinterpret_cast<uint64_t>(step0_pos_ref_.data_ptr()));
    for(auto* t:{&position_idx_keepalive_,&pixel_src_ref_,&step0_pos_ref_,&merger_dst_ref_}){
     size_t off;auto& o=locate(t->data_ptr(),off);const size_t n=t->nbytes();
     assert(std::memcmp(o.ddr.data()+off,t->data_ptr(),n)==0);
     seen.emplace_back(o.ddr.begin()+off,o.ddr.begin()+off+n);
    }
    for(int c=0;c<current_camera_batch_count_;++c){size_t off;
     auto& o=locate(reinterpret_cast<void*>(merger_dst_bases_[c]),off);
     assert(off+64*2048*2<=o.ddr.size());
    }
    ++consumes;
   });++builds;
  }
  return hidden_buf_;
 }
 __PREDICATES__
 __BODY__
};
void reset_calls(){scalars.clear();batches.clear();timeline.clear();batch_completions=system_completions=0;}
void exact_ranges(const Qwen3_5VisionModel& m,const at::Tensor& input,const at::Tensor& pos,const at::Tensor& target,bool batched){
 const std::array<const at::Tensor*,4>t{{&m.position_idx_keepalive_,&input,&pos,&target}};
 const auto& actual=batched?batches.at(0):scalars;
 assert(actual.size()==4);
 for(size_t i=0;i<4;++i)assert(actual[i].ptr==t[i]->data_ptr()&&actual[i].bytes==t[i]->nbytes());
 assert(m.pixel_src_ref_.data_ptr()==input.data_ptr());
 assert(m.step0_pos_ref_.data_ptr()==pos.data_ptr());
 assert(m.merger_dst_ref_.data_ptr()==target.data_ptr());
 if(batched)assert(timeline==std::vector<std::string>({"batch","dsb-sy","record-consumer"}));
 else assert(timeline==std::vector<std::string>({"scalar","scalar","scalar","scalar","record-consumer"}));
}
void forward(Qwen3_5VisionModel& m,const at::Tensor& input,const at::Tensor& pos,const at::Tensor& target,int cameras){
 std::vector<at::Tensor>k,v;std::vector<int64_t>starts;
 for(int c=0;c<cameras;++c)starts.push_back(3+c*71);
 m.forward_impl(input,k,v,cameras*256,pos,target,starts,1,cameras,0,0,0,0,std::vector<int64_t>{1},false);
}
void mutate_all(Qwen3_5VisionModel& m,at::Tensor& input,at::Tensor& pos,at::Tensor& target,int marker){
 // Public aliases change CPU bytes without changing Tensor version/address.
 std::memset(m.position_idx_keepalive_.data_ptr(),marker,m.position_idx_keepalive_.nbytes());
 std::memset(input.data_ptr(),marker+1,input.nbytes());
 std::memset(pos.data_ptr(),marker+2,pos.nbytes());
 std::memset(target.data_ptr(),marker+3,target.nbytes());
}
template<class F>void reject(F f){bool failed=false;try{f();}catch(const c10::Error&){failed=true;}assert(failed);}
int main(){
 c10::InferenceMode inference;
 int cases=0;
 for(bool w8:{false,true})for(int cameras:{1,3}){
  if(cameras==3&&!w8)continue;
  owners.clear();Qwen3_5VisionModel m;m.g05_w8a16_=w8;
  std::vector<std::vector<uint8_t>> first;
  for(int marker:{3,11,3}){
   auto input=bank({1,cameras*256,1536});auto pos=bank({cameras*256,1024});
   auto target=bank({1,cameras==1?161:225,2048});mutate_all(m,input,pos,target,marker);
   reset_calls();forward(m,input,pos,target,cameras);exact_ranges(m,input,pos,target,true);
   assert(m.builds==1&&m.nodes.size()==1);
   input=pos=target=at::Tensor();m.seen.clear();m.nodes[0]();
   assert(m.seen.size()==4);if(first.empty())first=m.seen;else if(marker==3)assert(first==m.seen);else assert(first!=m.seen);
  }
  assert(m.consumes==3);
  auto input=bank({1,cameras*256,1536});auto pos=bank({cameras*256,1024});
  auto target=bank({1,cameras==1?161:225,2048});
  const void* stable_input=input.data_ptr();const void* stable_pos=pos.data_ptr();
  first.clear();
  for(int marker:{37,41,37}){
   mutate_all(m,input,pos,target,marker);reset_calls();forward(m,input,pos,target,cameras);
   exact_ranges(m,input,pos,target,true);assert(input.data_ptr()==stable_input&&pos.data_ptr()==stable_pos);
   m.seen.clear();m.nodes[0]();
   if(first.empty())first=m.seen;else if(marker==37)assert(first==m.seen);else assert(first!=m.seen);
  }
  assert(m.builds==1&&m.consumes==6);++cases;
 }
 // Every endpoint gets each failed owner proof. Even where SDK metadata is
 // malformed, original scalar ranges/order must remain byte-for-byte exact.
 for(int endpoint=0;endpoint<4;++endpoint)for(int fault=0;fault<9;++fault){
  owners.clear();Qwen3_5VisionModel m;
  auto input=bank({1,256,1536},at::kHalf,(fault==6||fault==7)&&endpoint==1?33:32);
  auto pos=bank({256,1024},at::kHalf,(fault==6||fault==7)&&endpoint==2?33:32);
  auto target=bank({1,161,2048},at::kHalf,(fault==6||fault==7)&&endpoint==3?33:32);
  if((fault==6||fault==7)&&endpoint==0)m.position_idx_keepalive_=bank({4608,2},at::kShort,33);
  std::array<at::Tensor*,4>t{{&m.position_idx_keepalive_,&input,&pos,&target}};
  size_t off;auto& o=locate(t[endpoint]->data_ptr(),off);const size_t n=t[endpoint]->nbytes();
  if(fault==0)o.lookup=false;
  if(fault==1)o.bytes=0;
  if(fault==2)o.base=reinterpret_cast<uintptr_t>(t[endpoint]->data_ptr())+64;
  if(fault==3)o.bytes=std::numeric_limits<size_t>::max();
  if(fault==4)o.bytes=off+n-1;
  if(fault==5){o.base=0;o.bytes=reinterpret_cast<uintptr_t>(t[endpoint]->data_ptr())+n+64;}
  if(fault==6)o.bytes=off+n;
  if(fault==7){o.base=reinterpret_cast<uintptr_t>(t[endpoint]->data_ptr());o.bytes=n+64;}
  if(fault==8){o.base=std::numeric_limits<uintptr_t>::max()-31;o.bytes=64;}
  mutate_all(m,input,pos,target,17);reset_calls();forward(m,input,pos,target,1);
  assert(batches.empty()&&system_completions==0);exact_ranges(m,input,pos,target,false);m.nodes[0]();++cases;
 }
 // Same-owner slab views keep logical offsets and sizes, not owner capacity.
 for(int endpoint=0;endpoint<4;++endpoint){
  owners.clear();Qwen3_5VisionModel m;auto input=bank({1,256,1536});auto pos=bank({256,1024});auto target=bank({1,161,2048});
  std::array<at::Tensor*,4>t{{&m.position_idx_keepalive_,&input,&pos,&target}};
  auto flat=at::zeros({2*int64_t(t[endpoint]->numel())+128},t[endpoint]->scalar_type());
  auto view=flat.narrow(0,64,t[endpoint]->numel()).view(t[endpoint]->sizes());
  owners.emplace_back(std::make_unique<Owner>(flat,view));*t[endpoint]=view;
  mutate_all(m,input,pos,target,23);reset_calls();forward(m,input,pos,target,1);
  exact_ranges(m,input,pos,target,true);m.nodes[0]();++cases;
 }
 // Unsupported private owner/precision/geometry keeps original acquires.
 for(int fault=0;fault<7;++fault){
  owners.clear();Qwen3_5VisionModel m;auto input=bank({1,256,1536});auto pos=bank({256,1024});auto target=bank({1,161,2048});
  if(fault==0){m.g05_qkv_spm_available_=false;m.linear_acc32_=true;}
  if(fault==1)m.high_precision_=true;
  if(fault==2)m.mlp_tp_value=4;
  if(fault==3)m.attention_tp_=4;
  if(fault==4)m.gelu_erf_ultra_=true;
  if(fault==5)m.moe_precision_=true;
  if(fault==6)debug_export=true;
  mutate_all(m,input,pos,target,29);reset_calls();forward(m,input,pos,target,1);
  exact_ranges(m,input,pos,target,false);m.nodes[0]();debug_export=false;++cases;
 }
 // Valid B2 routing is intentionally outside the measured N1/N3 envelope.
 owners.clear();{Qwen3_5VisionModel m;m.g05_w8a16_=true;auto input=bank({1,512,1536});auto pos=bank({512,1024});auto target=bank({1,161,2048});
  mutate_all(m,input,pos,target,31);reset_calls();forward(m,input,pos,target,2);exact_ranges(m,input,pos,target,false);m.nodes[0]();++cases;}
 // Temporal, standalone merger, and Z2 keep their existing distinct ranges.
 owners.clear();{Qwen3_5VisionModel m;auto input=bank({1,256,1536});auto pos=bank({8,256,128});auto target=bank({1,161,2048});
  mutate_all(m,input,pos,target,33);reset_calls();std::vector<at::Tensor>k,v;
  m.forward_impl(input,k,v,256,pos,target,std::vector<int64_t>{3},2,1,0,0,0,0,std::vector<int64_t>{1},false);
  exact_ranges(m,input,pos,target,false);m.nodes[0]();++cases;}
 owners.clear();{Qwen3_5VisionModel m;auto input=bank({1,256,1536});auto pos=bank({256,1024});
  reset_calls();std::vector<at::Tensor>k,v;
  m.forward_impl(input,k,v,256,pos,std::nullopt,{},1,1,0,0,0,0,std::vector<int64_t>{1},false);
  assert(batches.empty()&&scalars.size()==4);
  assert(scalars[0].ptr==m.position_idx_keepalive_.data_ptr()&&scalars[0].bytes==18432);
  assert(scalars[1].ptr==input.data_ptr()&&scalars[1].bytes==input.nbytes());
  assert(scalars[2].ptr==pos.data_ptr()&&scalars[2].bytes==pos.nbytes());
  assert(scalars[3].ptr==m.merged_buf_.data_ptr()&&scalars[3].bytes==64*2048*2);++cases;}
 owners.clear();{Qwen3_5VisionModel m;m.z2_bound_=true;m.z2_slice_addr_=1;auto input=bank({1,256,1536});auto pos=bank({256,1024});
  reset_calls();std::vector<at::Tensor>k,v;
  m.forward_impl(input,k,v,256,pos,std::nullopt,{},1,1,0,0,0,0,{},true);
  assert(batches.empty()&&scalars.size()==3&&system_completions==0);++cases;}
 // ForceOFF remains zero API calls; recording is retained, no stale-data
 // correctness claim is made with coherency disabled.
 owners.clear();{Qwen3_5VisionModel m;auto input=bank({1,256,1536});auto pos=bank({256,1024});auto target=bank({1,161,2048});
  g_rpu_ddr_flush_force_enabled=false;reset_calls();forward(m,input,pos,target,1);
  assert(scalars.empty()&&batches.empty()&&system_completions==0&&m.nodes.size()==1);g_rpu_ddr_flush_force_enabled=true;++cases;}
 // Existing error order: malformed STEP0 fails after idx only; malformed
 // fusion target fails after idx/pixel/pos. No Batch or consumer is recorded.
 for(int fault=0;fault<4;++fault){
  owners.clear();Qwen3_5VisionModel m;auto input=bank({1,256,1536});auto pos=bank({256,1024});auto target=bank({1,161,2048});
  if(fault==0)pos=bank({255,1024});if(fault==1)pos=bank({256,1024},at::kFloat);
  if(fault==2)target=bank({1,161,2047});if(fault==3)target=at::Tensor();
  reset_calls();reject([&]{forward(m,input,pos,target,1);});
  assert(batches.empty()&&m.nodes.empty()&&scalars.size()==size_t(fault<2?1:3));++cases;
 }
 owners.clear();{Qwen3_5VisionModel m;auto input=bank({1,256,1536});auto pos=bank({256,1024});auto target=bank({1,161,2048});
  reset_calls();std::vector<at::Tensor>k,v;
  reject([&]{m.forward_impl(input,k,v,256,pos,target,std::vector<int64_t>{100},1,1,0,0,0,0,std::vector<int64_t>{1},false);});
  assert(batches.empty()&&scalars.size()==3&&m.nodes.empty());++cases;
  reset_calls();reject([&]{m.forward_impl(input,k,v,256,pos,target,std::vector<int64_t>{3},1,1,0,0,0,0,{},false);});
  assert(batches.empty()&&scalars.empty()&&m.nodes.empty());++cases;}
 std::cout<<"host whole Vision range/coherency/lease cells="<<cases<<"; no device math or physical barrier claim\n";
}
'''.replace("__PREDICATES__", "\n".join(predicates)).replace("__BODY__", body)
    _run_native(tmp_path, cpp, torch)

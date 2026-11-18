"""Actual SigLIP forward/CPU receipt on real ATen storage.

SDK owner/cache, RPU labels/allocation, planner/SPM, patch/encoder math and Graph
execution are explicit doubles. The complete forward/core, setter validation,
receipt comparison/copy/publication and mutable source code are actual source.
This proves host-modeled boundaries only; it is not hardware or model numerics.
"""
from pathlib import Path
from test_siglip_attention_dry_live_contract import _function, _run_native
ROOT = Path(__file__).resolve().parents[1]


def test_whole_serial_cpu_camera_slab_receipt_dirty_aba_and_graph_lifetime(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    source = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    profile = (ROOT / "src/core/rpu_profile.h").read_text()
    forward = _function(source, "    at::Tensor forward(")
    core = _function(source, "    at::Tensor run_packed_patch_embed_core(")
    packer = _function(source, "    at::Tensor run_packed_patch_embed(")
    patch_only = _function(source, "    at::Tensor patch_embed(")
    owner = "\n".join(_function(source, marker) for marker in (
        "    bool pi05_vision_supported_owner() const", "    bool pi05_vision4_w4_owner() const"))
    span = _function(source, "static bool packed_image_acquire_span(")
    live = _function(source, "struct PatchEmbeddingLiveBases") + ";"
    patch = _function(source, "static void fused_patch_embedding(")
    dma = patch[patch.index("    TORCH_CHECK(img_slot >= 0"):patch.index("    // ===== Step ②")]
    allocation = """at::empty(
                    {N, 3, 224, 224}, at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1))"""
    assert core.count(allocation) == 1
    core = core.replace(allocation, "private_bank({N, 3, 224, 224})")
    memcpy = "std::memcpy(image.data_ptr(), images[i].const_data_ptr(), images[i].nbytes());"
    assert core.count(memcpy) == 1
    core = core.replace(memcpy, memcpy + "\n                    mark_dirty(image);")
    receipt = _function(source, "    struct PrivateSerialCpuSlabSpan") + ";\n" + "\n".join(
        _function(source, marker) for marker in (
            "    void clear_private_serial_cpu_slab()", "    bool private_serial_cpu_slab_span(",
            "    bool prepare_private_serial_cpu_slab("))
    receipt = receipt.replace("input.device().is_cpu()", "cpu_label(input)").replace("at::kPrivateUse1", "at::kCPU")
    fresh = """at::empty(
            {3, 3, 224, 224}, at::TensorOptions().dtype(at::kHalf).device(at::kCPU))"""
    assert receipt.count(fresh) == 1
    receipt = receipt.replace(fresh, "private_bank({3, 3, 224, 224})")
    memcpy = "std::memcpy(fresh.data_ptr(), storage.data(), bytes);"
    assert receipt.count(memcpy) == 1
    receipt = receipt.replace(memcpy, memcpy + "\n        mark_dirty(fresh);")
    core = core.replace("images.front().device().is_cpu()", "cpu_label(images.front())")
    forward = forward.replace("input.device().is_cpu()", "cpu_label(input)")
    for label in ("input", "k_caches[layer]", "v_caches[layer]"):
        forward = forward.replace(label + ".device().type() == at::kPrivateUse1",
            "(!cpu_label(" + label + ") && " + label + ".device().type() == at::kCPU)")
    patch_only = patch_only.replace("input.device().type() == at::kPrivateUse1",
        "(!cpu_label(input) && input.device().type() == at::kCPU)")
    span = span.replace("at::kPrivateUse1", "at::kCPU")
    weights = _function(source, "    void set_weights(").replace("at::kPrivateUse1", "at::kCPU")
    scale_check = "\n".join(_function(source, marker) for marker in (
        "static void check_siglip_w8a16_scale_lists(", "static void check_siglip_w4a16_scale_lists(")
    ).replace("at::kPrivateUse1", "at::kCPU")
    patch_setter = _function(source, "    void set_patch_emb_params(").replace("at::kPrivateUse1", "at::kCPU")
    chunk_setter = _function(source, "    void set_configured_chunk_size(")
    macros = "\n".join(profile[profile.index(f"#define {name}("):].split("\n\n", 1)[0]
        for name in ("rpu_ddr_flush_force", "rpu_ddr_flush_force_sized"))
    cpp = r'''

#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/intrusive_ptr.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>
bool g_rpu_ddr_flush_enabled=false,g_rpu_ddr_flush_force_enabled=true;
struct Store {
 void* pointer;
 std::vector<uint8_t> ddr,dirty;
 struct Owner{void* base;size_t capacity;void* get_cpu_ptr(){return base;}size_t get_memory_size(){return capacity;}};
 std::unique_ptr<Owner> owner;
 c10::weak_intrusive_ptr<c10::StorageImpl> identity;
 Store(const at::Tensor& t):pointer(t.data_ptr()),ddr(t.nbytes(),0x71),dirty(t.nbytes(),0),
  identity(t.storage().getWeakStorageImpl()),owner(std::make_unique<Owner>(Owner{t.data_ptr(),t.nbytes()})){}
};
std::vector<std::unique_ptr<Store>> stores;
std::unordered_map<const c10::StorageImpl*,c10::weak_intrusive_ptr<c10::StorageImpl>> cpu_inputs;
bool cpu_label(const at::Tensor& t){if(!t.defined())return false;
 auto it=cpu_inputs.find(t.storage().unsafeGetStorageImpl());
 return it!=cpu_inputs.end()&&!it->second.expired();}
void label_cpu(const at::Tensor& t){cpu_inputs.insert_or_assign(t.storage().unsafeGetStorageImpl(),
 c10::weak_intrusive_ptr<c10::StorageImpl>(t.storage().getWeakStorageImpl()));}
Store& locate(void* p,size_t bytes,size_t& offset){const auto ptr=reinterpret_cast<uintptr_t>(p);
 for(auto it=stores.rbegin();it!=stores.rend();++it){auto& s=**it;
  const auto base=reinterpret_cast<uintptr_t>(s.pointer);
  if(!s.identity.expired()&&ptr>=base&&ptr-base<=s.ddr.size()&&bytes<=s.ddr.size()-(ptr-base)){
   offset=ptr-base;return s;}}
 TORCH_CHECK(false,"unregistered logical memory");}
at::Tensor bank(at::IntArrayRef shape){auto t=at::zeros(shape,at::kHalf);
 stores.push_back(std::make_unique<Store>(t));return t;}
int allocations=0,owner_fault=0,copy_marks=0;
bool destination_offset=false,flush_failure=false,query_failure=false;
at::Tensor allocation_alias;
at::Tensor private_bank(at::IntArrayRef shape){++allocations;
 if(allocation_alias.defined()){auto t=allocation_alias;stores.push_back(std::make_unique<Store>(t));return t;}
 if(!destination_offset)return bank(shape);
 int64_t elements=1;for(auto n:shape)elements*=n;
 return bank({elements+32}).narrow(0,16,elements).view(shape);}
void mark_dirty(const at::Tensor& t){++copy_marks;size_t offset;auto& s=locate(t.data_ptr(),t.nbytes(),offset);
 std::fill(s.dirty.begin()+offset,s.dirty.begin()+offset+t.nbytes(),1);}
void host_write(const at::Tensor& t,const at::Tensor& value){t.copy_(value);mark_dirty(t);}
std::vector<std::pair<void*,size_t>> flushes;
namespace rhino_lkn {
using Owner=Store::Owner;
Owner* RpuGetHostddr(void* p){TORCH_CHECK(!query_failure,"SDK query failure");
 size_t offset;auto& s=locate(p,0,offset);auto& owner=*s.owner;
 if(owner_fault==1)return nullptr;
 if(owner_fault==2)owner.base=nullptr;
 if(owner_fault==3)owner.capacity=0;
 if(owner_fault==4)owner.base=static_cast<uint8_t*>(s.pointer)+1;
 if(owner_fault==5)owner.capacity=s.ddr.size()-1;
 if(owner_fault==6)owner.capacity=std::numeric_limits<size_t>::max();
 if(owner_fault==7){owner.base=static_cast<uint8_t*>(s.pointer)+32;owner.capacity=s.ddr.size();}
 if(owner_fault==8)owner.capacity=s.ddr.size()-32;
 return &owner;}
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
void RpuDdrFlush(void* p,size_t bytes){size_t offset;auto& s=locate(p,bytes,offset);
 flushes.emplace_back(p,bytes);TORCH_CHECK(!flush_failure,"SDK publication failure");
 const size_t first=offset&~size_t(63),last=(offset+bytes+63)&~size_t(63);
 assert(last<=s.ddr.size());auto* cpu=static_cast<uint8_t*>(s.pointer);
 for(size_t i=first;i<last;++i){if(s.dirty[i])s.ddr[i]=cpu[i];else cpu[i]=s.ddr[i];s.dirty[i]=0;}}
void RpuDdrFlush(void* p){size_t offset;auto& s=locate(p,0,offset);RpuDdrFlush(p,s.ddr.size()-offset);}
}
__MACROS__
using rhino_lkn::RpuGetDevAddr;
struct GraphSignature{};
struct RpuKernelGraph {
 enum class State {PASSTHROUGH,RECORDING,REPLAYING};
 State current=State::PASSTHROUGH;
 std::vector<at::Tensor> held;
 static RpuKernelGraph& active(){static RpuKernelGraph graph;return graph;}
 static bool has_active(){return true;}
 State state(){return current;}
 void keep_alive(const at::Tensor& t){held.push_back(t);}
 void record_branch(uint64_t,GraphSignature,const char*){}
};
struct Node {const uint64_t* input;const uint64_t* output;int64_t elements;};
std::vector<Node> nodes;
size_t node_cursor=0;
std::vector<std::vector<uint8_t>> reads;
const uint64_t* next_output=nullptr;
void rpu_launch_ddr_broadcast_spm_dma_mutable(const uint64_t* input,int64_t offset,
 int64_t elements,uint32_t,int){assert(offset==0&&elements==150528&&next_output);
 auto& graph=RpuKernelGraph::active();Node n{input,next_output,elements};
 if(graph.current==RpuKernelGraph::State::REPLAYING){assert(node_cursor<nodes.size());
  const auto& old=nodes[node_cursor++];assert(old.input==input&&old.output==next_output&&old.elements==elements);
 }else nodes.push_back(n);}
void execute_nodes(){for(const auto& n:nodes){size_t offset;
 auto& source=locate(reinterpret_cast<void*>(*n.input),n.elements*2,offset);
 std::vector<uint8_t> value(source.ddr.begin()+offset,source.ddr.begin()+offset+n.elements*2);
 reads.push_back(value);size_t output_offset;
 auto& target=locate(reinterpret_cast<void*>(*n.output),256*1152*2,output_offset);
 std::fill(target.ddr.begin()+output_offset,target.ddr.begin()+output_offset+256*1152*2,0);
 std::copy(value.begin(),value.end(),target.ddr.begin()+output_offset);}}
namespace v3 {
enum class FmbRouteFamily {MUTABLE_DMA};
struct InferenceContext {void consume_physical_route(FmbRouteFamily,int64_t,int64_t,int64_t,
 std::initializer_list<int64_t>,int)const{}};
}
enum class SiglipPatchMutableDmaRoute {DDR_BROADCAST_TO_SPM_MUTABLE};
constexpr int64_t SIGLIP_PATCH_INPUT_DMA_SITE=1143336462482724752LL;
__LIVE__
__SPAN__
void fused_patch_embedding(const at::Tensor& input_nchw,const at::Tensor&,const at::Tensor&,
 int64_t kh,int64_t kw,int64_t cin_orig,int64_t cin_padded,int64_t cout,
 int64_t strideh,int64_t stridew,at::Tensor& out,int64_t seq_off,int img_slot,
 const std::vector<uint32_t>&,PatchEmbeddingLiveBases& live_bases,
 const v3::InferenceContext* physical_ctx,bool){
 const int64_t HW=input_nchw.size(2)*input_nchw.size(3),num_patches=256;
 const uint32_t raw_addr=0;assert(seq_off==0&&cout==1152);
 live_bases.output=rhino_lkn::RpuGetDevAddr(out.data_ptr());next_output=&live_bases.output;
 __DMA__
}
bool spm_failure=false;
struct SpmAllocator {struct AllocRequest{int64_t bytes,begin,end;};
 bool is_initialized(){return true;}void init(){}
 std::vector<uint32_t> alloc_temporary_aliased(std::initializer_list<AllocRequest> requests){
  TORCH_CHECK(!spm_failure,"modeled SPM failure");assert(requests.size()==6);return std::vector<uint32_t>(6,0);}
 void reset_temporary(){}} SPM_ALLOC;
struct Layer {at::Tensor q_w,k_w,v_w,o_w,fc1_w,fc2_w,q_ws,k_ws,v_ws,o_ws,fc1_ws,fc2_ws;};
struct Planned {int physical_manifest=0,stage_plan=0;};
struct Prepared {Planned plan;const Planned& candidate(){return plan;}};
at::Tensor rpu_retain_linear_quant_scale(const at::Tensor&,const at::Tensor& scale,int64_t){return scale;}
__SCALE_CHECK__
struct Siglip {
 // The serial slab receipt remains an FP16/W8A16 owner, with no A8 payloads.
 const bool qkv_w8a8_=false,mlp_w8a8_=false;
 int cores=4,attention=4,mlp=4,layers=27;uint64_t generation=1;
 int64_t pe_kh_=14,pe_kw_=14,pe_strideh_=14,pe_stridew_=14,pe_cin_orig_=3,pe_cin_padded_=16,pe_cout_=1152;
 int64_t orig_head_dim_=72,projection_dim_=2048,image_batch_count_=1;
 bool has_patch_emb_=true,linear_acc32_=false,pi05_vision_fc_weight_outer_opt_in_=false,audio_mode_=false;
 bool pi05_vision_merge_segment_opt_in_=false,pi05_vision_owner_ln_opt_in_=false;
 // LayerNorm streaming stays off here: set_weights leaves ln_stream_ddr_ empty.
 bool pi05_vision_ln_stream_opt_in_=false;at::Tensor ln_stream_ddr_;
 int64_t ln_stream_elems() const{assert(false);return 0;}
 bool pi05_vision_kv_spm_only_opt_in_=false,pi05_vision_q_resident_opt_in_=false,pi05_ring_xor3_opt_in_=false;
 int64_t logical_intermediate_size_=0,reduced_w8_projection_mask_=0,local_q_heads_=0;
 int64_t configured_chunk_size_=0,patch_kernel_size_=14,patch_stride_=14,last_resolved=0;
 double eps_=1e-6;at::Tensor post_ln_w_,post_ln_b_,proj_w_,proj_b_;
 struct BiasNorm{at::Tensor ln1_w,ln1_b,ln2_w,ln2_b,q_b,k_b,v_b,o_b,fc1_b,fc2_b;};
 std::vector<BiasNorm> layer_bias_norm_;
 __RECEIPT__
 at::Tensor private_serial_cpu_slab_;PrivateSerialCpuSlabSpan private_serial_cpu_slab_span_;
 uint64_t private_serial_cpu_slab_generation_=0;bool private_serial_cpu_slab_published_=false;
 void invalidate_model_state(){++generation;}
 int64_t get_last_resolved_chunk_size(){return last_resolved;}
 void set_model_params(int64_t,int64_t,int64_t,int64_t,int64_t){}
 void set_num_layers(int64_t n){layers=n;}
 void require_pi05_vision_owner_ln_payloads(int64_t){}
 void require_pi05_vision_qkv_w8a8_payloads(int64_t){assert(false);}
 void require_pi05_vision_mlp_w8a8_payloads(int64_t){assert(false);}
 __WEIGHTS__
 __PATCH_SETTER__
 __CHUNK_SETTER__
 int prologues=0;at::Tensor patch_emb_weight_,patch_emb_pos_emb_;
 std::vector<Layer> layer_weights_;
 PatchEmbeddingLiveBases patch_embedding_live_bases_;
 Siglip(){install_precision(false);}
 void install_precision(bool w8){layer_weights_.clear();for(int i=0;i<27;++i){Layer w;
  for(auto* t:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.fc1_w,&w.fc2_w})*t=at::ones({1},w8?at::kChar:at::kHalf);
  if(w8)for(auto* t:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.fc1_ws,&w.fc2_ws})*t=at::ones({1},at::kHalf);
  layer_weights_.push_back(std::move(w));}}
 int num_cores()const{return cores;}int attn_tp()const{return attention;}int mlp_tp()const{return mlp;}
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return 1152;}
 int64_t intermediate_size()const{return 4352;}int64_t num_q_heads()const{return 16;}
 int64_t num_kv_heads()const{return 16;}int64_t head_dim()const{return 80;}
 uint64_t installed_model_state_generation()const{return generation;}
 v3::InferenceContext context;const v3::InferenceContext& ctx(){return context;}
 std::shared_ptr<Prepared> prepare_stage_candidate(at::IntArrayRef){return std::make_shared<Prepared>();}
 void validate_pi05_vision_kv_spm_only_policy(const Planned&,int64_t,int64_t){}
 void validate_pi05_vision_owner_ln_policy(const Planned&,int64_t,int64_t){}
 void require_pi05_vision_fc_profile(int64_t,int64_t){}
 void require_pi05_vision_qkv_w8a8_profile(int64_t,int64_t){assert(!qkv_w8a8_);}
 void require_pi05_vision_mlp_w8a8_profile(int64_t,int64_t){assert(!mlp_w8a8_);}
 void validate_pi05_vision_qkv_w8a8_policy(const Planned&,int64_t,int64_t){assert(!qkv_w8a8_);}
 void validate_pi05_vision_mlp_w8a8_policy(const Planned&,int64_t,int64_t){assert(!mlp_w8a8_);}
 void validate_pi05_vision_fc_manifest(int){}void require_pi05_vision_fc_schedule(int){}
 void begin_external_physical_manifest_prologue(int,int64_t,int){++prologues;}
 void cancel_external_physical_manifest_prologue(){}
 at::Tensor allocate_tracked_output(at::IntArrayRef dims){auto t=bank(dims);
  RpuKernelGraph::active().keep_alive(t);return t;}
 at::Tensor forward_packed(const at::Tensor& hidden,int64_t,std::vector<at::Tensor>&,
  std::vector<at::Tensor>&,at::IntArrayRef){return hidden;}
 __OWNER__
 __FORWARD__
 __CORE__
 __PACKER__
 __PATCH_ONLY__
};
std::vector<uint8_t> bytes(const at::Tensor& t){auto dense=t.resolve_neg().resolve_conj().contiguous();
 const auto* p=static_cast<const uint8_t*>(dense.const_data_ptr());return {p,p+dense.nbytes()};}
void expect_bytes(const at::Tensor& output,const std::vector<uint8_t>& expected){size_t offset;
 auto& s=locate(output.data_ptr(),output.nbytes(),offset);
 assert(std::equal(expected.begin(),expected.end(),s.ddr.begin()+offset));
 for(size_t i=expected.size();i<output.nbytes();++i)assert(s.ddr[offset+i]==0);}
void reset_events(){flushes.clear();reads.clear();node_cursor=0;allocations=0;copy_marks=0;}
at::Tensor input_slab(){auto t=at::arange(3*3*224*224,at::kFloat).remainder(257).mul(0.001).sub(0.1)
 .to(at::kHalf).view({3,3,224,224});
 t.data_ptr<c10::Half>()[0]=c10::Half(-0.0f);t.data_ptr<c10::Half>()[1]=c10::Half(-65504.0f);
 label_cpu(t);return t;}
template<class F>void reject_before_prologue(Siglip& m,F call){reset_events();nodes.clear();
 const int before=m.prologues;bool failed=false;try{call();}catch(const c10::Error&){failed=true;}
 assert(failed&&m.prologues==before&&flushes.empty()&&nodes.empty()&&allocations==0);}

void install_real(Siglip& m,bool w8,bool empty=false){
 const std::array<std::array<int64_t,2>,6> shape{{{1280,1152},{1280,1152},{1280,1152},
 {1152,1280},{4352,1152},{1152,4352}}};
 std::array<std::vector<at::Tensor>,6> weights,scales;
 for(size_t i=0;i<6;++i){weights[i].assign(empty?0:27,at::empty({shape[i][0],shape[i][1]},w8?at::kChar:at::kHalf));
  if(w8)scales[i].assign(27,at::ones({shape[i][0]},at::kHalf));}
 std::vector<at::Tensor> norm(27,at::ones({1152},at::kHalf)),qb(27,at::zeros({1280},at::kHalf));
 std::vector<at::Tensor> fb(27,at::zeros({4352},at::kHalf));
 auto post=at::ones({1152},at::kHalf),proj=at::empty({2048,1152},at::kHalf),pb=at::zeros({2048},at::kHalf);
 m.set_weights(weights[0],weights[1],weights[2],weights[3],weights[4],weights[5],
 norm,norm,norm,norm,qb,qb,qb,norm,fb,norm,post,post,proj,pb,
 16,80,1152,4352,2048,1e-6,scales[0],scales[1],scales[2],scales[3],scales[4],scales[5]);
}
void reinstall_same_addresses(Siglip& m){
 std::array<std::vector<at::Tensor>,6> w,s;
 for(const auto& layer:m.layer_weights_){
  const std::array<at::Tensor,6> a{layer.q_w,layer.k_w,layer.v_w,layer.o_w,layer.fc1_w,layer.fc2_w};
  const std::array<at::Tensor,6> b{layer.q_ws,layer.k_ws,layer.v_ws,layer.o_ws,layer.fc1_ws,layer.fc2_ws};
  for(size_t i=0;i<6;++i){w[i].push_back(a[i]);if(b[i].defined())s[i].push_back(b[i]);}}
 std::array<std::vector<at::Tensor>,10> bn;
 for(const auto& layer:m.layer_bias_norm_){const std::array<at::Tensor,10> a{layer.ln1_w,layer.ln1_b,
  layer.ln2_w,layer.ln2_b,layer.q_b,layer.k_b,layer.v_b,layer.o_b,layer.fc1_b,layer.fc2_b};
  for(size_t i=0;i<10;++i)bn[i].push_back(a[i]);}
 const auto postw=m.post_ln_w_,postb=m.post_ln_b_,projw=m.proj_w_,projb=m.proj_b_;
 m.set_weights(w[0],w[1],w[2],w[3],w[4],w[5],bn[0],bn[1],bn[2],bn[3],bn[4],bn[5],bn[6],bn[7],bn[8],bn[9],
  postw,postb,projw,projb,16,80,1152,4352,2048,1e-6,s[0],s[1],s[2],s[3],s[4],s[5]);
}
at::Tensor rgb_patch_oracle(const at::Tensor& input){
 // Independent ATen view of 16x16 RGB 14x14 patches, not a receipt/copy mirror.
 return input.unfold(2,14,14).unfold(3,14,14).permute({0,2,3,4,5,1}).contiguous().view({256,588});}
void assert_finite(const at::Tensor& output){size_t offset;auto& s=locate(output.data_ptr(),output.nbytes(),offset);
 auto oracle=at::from_blob(s.ddr.data()+offset,output.sizes(),at::kHalf);assert(at::isfinite(oracle).all().item<bool>());}
template<class F>void fails_no_consumer(Siglip& m,F call){reset_events();nodes.clear();
 bool failed=false;try{call();}catch(const c10::Error&){failed=true;}
 assert(failed&&!m.private_serial_cpu_slab_published_&&nodes.empty());}
int main(){int scenarios=0;std::vector<at::Tensor> keys(27,at::zeros({1},at::kHalf)),values=keys;
 const std::vector<int64_t> descriptor{11,12};auto& graph=RpuKernelGraph::active();
 std::vector<at::Tensor> saved;std::vector<std::vector<uint8_t>> saved_bytes;
 auto run=[&](Siglip& m,const at::Tensor& input,bool publication,size_t published_bytes=903168){
  ++scenarios;const auto expected=bytes(input);reset_events();node_cursor=0;
  graph.current=nodes.empty()?RpuKernelGraph::State::RECORDING:RpuKernelGraph::State::REPLAYING;
  auto output=m.forward(input,keys,values,descriptor);
  assert(flushes.size()==size_t(publication));if(publication)assert(flushes[0].second==published_bytes);
  assert(allocations==int(publication)&&copy_marks==int(publication)&&nodes.size()==1);
  execute_nodes();assert(reads.size()==1&&reads[0]==expected);
  auto consumed=at::from_blob(reads[0].data(),{1,3,224,224},at::kHalf);
  assert(at::equal(rgb_patch_oracle(consumed),rgb_patch_oracle(input)));
  expect_bytes(output,expected);assert_finite(output);
  for(size_t i=0;i<saved.size();++i)expect_bytes(saved[i],saved_bytes[i]);
  saved.push_back(output);saved_bytes.push_back(expected);return output;};
 // Whole actual forward, N1 core, mutable source and private receipt execute.
 // A->B->A, all camera slices; fresh CPU addresses with identical full bytes hit.
 for(bool normal:{false,true})for(bool inference:{false,true})for(bool w8:{false,true}){
  c10::InferenceMode mode(inference);g_rpu_ddr_flush_enabled=normal;
  Siglip m;install_real(m,w8);auto a=input_slab(),b=a.flip({3});label_cpu(b);
  nodes.clear();graph.held.clear();saved.clear();saved_bytes.clear();
  auto slab=a.clone();label_cpu(slab);
  for(int request=0;request<3;++request){slab.copy_(request==1?b:a);
   for(int camera=0;camera<3;++camera){run(m,slab.narrow(0,camera,1),camera==0);
    assert(m.private_serial_cpu_slab_published_);
    assert(m.patch_embedding_live_bases_.input[0]==reinterpret_cast<uint64_t>(m.private_serial_cpu_slab_.data_ptr())+camera*301056);}}
  auto identical=slab.clone();label_cpu(identical);run(m,identical.narrow(0,2,1),false);
  // No version update: raw same-pointer write in another camera must miss.
  slab.data_ptr<c10::Half>()[150528+7]=c10::Half(0.321f);
  run(m,slab.narrow(0,0,1),true);run(m,slab.narrow(0,1,1),false);
  slab.data_ptr<c10::Half>()[2*150528+13]=c10::Half(-0.75f);
  run(m,slab.narrow(0,0,1),true);run(m,slab.narrow(0,2,1),false);
  // Different selected indices do not alter the exact CPU-origin full bank.
  assert(saved_bytes[0]==saved_bytes[6]&&saved_bytes[1]==saved_bytes[7]&&saved_bytes[2]==saved_bytes[8]);
 }
 g_rpu_ddr_flush_enabled=false;
 // Force OFF never reuses a previously published bank. Re-enable allocates anew.
 {Siglip m;auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  run(m,slab.narrow(0,0,1),true);auto old=m.private_serial_cpu_slab_;
  g_rpu_ddr_flush_force_enabled=false;reset_events();node_cursor=0;
  graph.current=RpuKernelGraph::State::REPLAYING;
  auto out=m.forward(slab.narrow(0,1,1),keys,values,descriptor);
  assert(!m.private_serial_cpu_slab_published_&&allocations==1&&copy_marks==1&&flushes.empty());
  assert(m.patch_embedding_live_bases_.input[0]!=reinterpret_cast<uint64_t>(old.data_ptr())+301056);
  execute_nodes();for(auto v:reads[0])assert(v==0x71);
  g_rpu_ddr_flush_force_enabled=true;run(m,slab.narrow(0,1,1),true);}
 // Pure full-storage admission misses preserve the original N1 publication.
 for(int kind=0;kind<3;++kind){Siglip m;auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();
  graph.held.clear();auto input=slab.narrow(0,0,1);
  if(kind==0){auto back=at::empty({3*150528+32},at::kHalf);input=back.narrow(0,0,150528).view({1,3,224,224});input.copy_(slab.narrow(0,0,1));label_cpu(input);}
  if(kind==1)input=slab.reshape({3*150528}).narrow(0,1,150528).view({1,3,224,224});
  if(kind==2)input=at::as_strided(input,{1,3,224,224},{0,50176,224,1});
  run(m,input,true,301056);assert(!m.private_serial_cpu_slab_published_);}
 // Public RPU dirty CPU aliases always publish their original nonzero N1 span.
 {Siglip m;auto slab=input_slab();auto back=bank({3*150528+64});
  nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  for(int camera=0;camera<3;++camera){auto view=back.narrow(0,32+camera*150528,150528).view({1,3,224,224});
   host_write(view,slab.narrow(0,camera,1));const auto expected=bytes(view);reset_events();node_cursor=0;
   graph.current=nodes.empty()?RpuKernelGraph::State::RECORDING:RpuKernelGraph::State::REPLAYING;
   auto output=m.forward(view,keys,values,descriptor);assert(allocations==0&&flushes.size()==1&&flushes[0].second==301056);
   execute_nodes();assert(reads[0]==expected);expect_bytes(output,expected);++scenarios;}}
 // Live SDK owner identity is re-queried; a replacement cannot authorize memcmp.
 {Siglip m;auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  run(m,slab.narrow(0,0,1),true);size_t off;auto& prior=locate(m.private_serial_cpu_slab_.data_ptr(),0,off);
  auto replacement=std::make_unique<Store::Owner>(Store::Owner{prior.pointer,prior.ddr.size()});
  prior.owner=std::move(replacement);run(m,slab.narrow(0,1,1),true);}
 // Owner/bounds/SDK errors occur before consumer; memcpy failure never commits.
 for(int fault=1;fault<=8;++fault){Siglip m;auto slab=input_slab();owner_fault=fault;
  fails_no_consumer(m,[&]{m.forward(slab.narrow(0,0,1),keys,values,descriptor);});
  assert(copy_marks==0&&flushes.empty());owner_fault=0;++scenarios;}
 {Siglip m;auto slab=input_slab();query_failure=true;
  fails_no_consumer(m,[&]{m.forward(slab.narrow(0,0,1),keys,values,descriptor);});
  assert(copy_marks==0&&flushes.empty());query_failure=false;++scenarios;}
 {Siglip m;auto slab=input_slab();destination_offset=true;
  fails_no_consumer(m,[&]{m.forward(slab.narrow(0,0,1),keys,values,descriptor);});
  assert(copy_marks==0&&flushes.empty());destination_offset=false;++scenarios;}
 {Siglip m;auto slab=input_slab();allocation_alias=slab;
  fails_no_consumer(m,[&]{m.forward(slab.narrow(0,0,1),keys,values,descriptor);});
  assert(copy_marks==0&&flushes.empty());allocation_alias=at::Tensor();++scenarios;}
 {Siglip m;auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  run(m,slab.narrow(0,0,1),true);const auto* old=m.private_serial_cpu_slab_.storage().unsafeGetStorageImpl();
  slab.data_ptr<c10::Half>()[300]=c10::Half(0.5f);flush_failure=true;
  fails_no_consumer(m,[&]{m.forward(slab.narrow(0,1,1),keys,values,descriptor);});
  assert(copy_marks==1&&flushes.size()==1&&m.private_serial_cpu_slab_.storage().unsafeGetStorageImpl()==old);
  assert(stores.back()->identity.expired());flush_failure=false;++scenarios;
  run(m,slab.narrow(0,1,1),true);}
 // A failure after a hit, or a cached-owner SDK query failure, revokes authority.
 {Siglip m;auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  run(m,slab.narrow(0,0,1),true);query_failure=true;
  fails_no_consumer(m,[&]{m.forward(slab.narrow(0,1,1),keys,values,descriptor);});
  assert(copy_marks==0&&flushes.empty());query_failure=false;run(m,slab.narrow(0,1,1),true);
  spm_failure=true;fails_no_consumer(m,[&]{m.forward(slab.narrow(0,2,1),keys,values,descriptor);});
  assert(copy_marks==0&&flushes.empty());spm_failure=false;run(m,slab.narrow(0,2,1),true);}
 // Old banks remain Graph-leased across replacement and cold invalidations.
 {auto m=std::make_unique<Siglip>();auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  run(*m,slab.narrow(0,0,1),true);auto weak=m->private_serial_cpu_slab_.storage().getWeakStorageImpl();
  slab.data_ptr<c10::Half>()[3]=c10::Half(0.25f);run(*m,slab.narrow(0,1,1),true);
  assert(!weak.expired());m->clear_private_serial_cpu_slab();assert(!weak.expired());
  m.reset();assert(!weak.expired());graph.held.clear();assert(weak.expired());++scenarios;}
 // Delay execution until after the CPU producer is changed and the model
 // member is invalidated. The recorded mutable DMA's old bank must still live.
 {Siglip m;auto slab=input_slab();auto input=slab.narrow(0,1,1);const auto expected=bytes(input);
  nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();reset_events();
  graph.current=RpuKernelGraph::State::RECORDING;auto output=m.forward(input,keys,values,descriptor);
  auto weak=m.private_serial_cpu_slab_.storage().getWeakStorageImpl();
  slab.fill_(0.75);input=at::Tensor();slab=at::Tensor();m.clear_private_serial_cpu_slab();
  assert(!weak.expired());execute_nodes();assert(reads[0]==expected);expect_bytes(output,expected);
  graph.held.clear();assert(weak.expired());++scenarios;}
 // Failed set_weights/chunk/patch calls clear authority before validation.
 for(int cold=0;cold<3;++cold){Siglip m;auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  run(m,slab.narrow(0,0,1),true);
  fails_no_consumer(m,[&]{if(cold==0)install_real(m,false,true);
   if(cold==1)m.set_configured_chunk_size(1);
   if(cold==2)m.set_patch_emb_params(at::Tensor(),at::Tensor(),14,14);});
  assert(!m.private_serial_cpu_slab_.defined());run(m,slab.narrow(0,0,1),true);}
 {Siglip m;auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  run(m,slab.narrow(0,0,1),true);install_real(m,true);run(m,slab.narrow(0,1,1),true);
  m.set_configured_chunk_size(0);run(m,slab.narrow(0,1,1),true);
  m.set_patch_emb_params(at::empty({1152,3136},at::kHalf),at::empty({1,256,1152},at::kHalf),14,14);
  run(m,slab.narrow(0,2,1),true);++m.generation;run(m,slab.narrow(0,2,1),true);}
 {Siglip m;install_real(m,false);auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  run(m,slab.narrow(0,0,1),true);const auto generation=m.generation;
  const auto* weight=m.layer_weights_.front().q_w.data_ptr();reinstall_same_addresses(m);
  assert(m.generation!=generation&&m.layer_weights_.front().q_w.data_ptr()==weight&&
   !m.private_serial_cpu_slab_published_&&!m.private_serial_cpu_slab_.defined());
  run(m,slab.narrow(0,1,1),true);}
 // Validation errors on a previously admitted entry cannot leave a receipt.
 for(int invalid=0;invalid<6;++invalid){Siglip m;auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  run(m,slab.narrow(0,0,1),true);auto input=slab.narrow(0,0,1);
  if(invalid==0){input=input.to(at::kFloat);label_cpu(input);}
  if(invalid==1)input=input.transpose(2,3);
  if(invalid==2)input=at::_neg_view(input);
  if(invalid==3){input=input.clone();label_cpu(input);input.set_requires_grad(true);}
  if(invalid==4)m.cores=8;
  if(invalid==5)m.generation=0;
  fails_no_consumer(m,[&]{m.forward(input,keys,values,descriptor);});++scenarios;}
 // An inference-created owner is read-only on hit and replaced on any change.
 {Siglip m;auto slab=input_slab();nodes.clear();saved.clear();saved_bytes.clear();graph.held.clear();
  {c10::InferenceMode mode;run(m,slab.narrow(0,0,1),true);}assert(m.private_serial_cpu_slab_.is_inference());
  run(m,slab.narrow(0,1,1),false);slab.data_ptr<c10::Half>()[5]=c10::Half(-0.1f);
  run(m,slab.narrow(0,2,1),true);assert(!m.private_serial_cpu_slab_.is_inference());}
 std::cout<<"Whole serial CPU slab receipt: "<<scenarios<<" modeled scenarios PASS\n";
}
'''
    replacements = {"__MACROS__": macros, "__LIVE__": live, "__SPAN__": span,
        "__DMA__": dma, "__OWNER__": owner, "__FORWARD__": forward,
        "__CORE__": core, "__PACKER__": packer, "__PATCH_ONLY__": patch_only,
        "__RECEIPT__": receipt, "__WEIGHTS__": weights, "__SCALE_CHECK__": scale_check,
        "__PATCH_SETTER__": patch_setter, "__CHUNK_SETTER__": chunk_setter}
    for marker, body in replacements.items():
        assert cpp.count(marker) == 1
        cpp = cpp.replace(marker, body)
    output = _run_native(tmp_path, cpp, torch)
    assert "Whole serial CPU slab receipt:" in output and "modeled scenarios PASS" in output

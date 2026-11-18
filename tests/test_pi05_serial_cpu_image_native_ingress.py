"""Whole native SigLIP forward/packer on ATen, with hardware boundaries doubled.

CPU/device labels, allocator registration, SDK dirty lines, physical planner,
SPM, patch/encoder kernels and Graph execution are explicit host boundaries.
The actual admission, complete copy, publication, mutable input DMA setup,
Graph leases and single/list packers execute from production source. This is
not a native compiler/SDK/dispatch or model-math hardware certification.
"""
from pathlib import Path

from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def test_whole_serial_cpu_image_forward_copies_publishes_and_keeps_graph_owners(tmp_path, monkeypatch):
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
    receipt = _function(source, "    struct PrivateSerialCpuSlabSpan") + ";\n" + "\n".join(
        _function(source, marker) for marker in (
            "    void clear_private_serial_cpu_slab()", "    bool private_serial_cpu_slab_span(",
            "    bool prepare_private_serial_cpu_slab("))
    receipt = receipt.replace("input.device().is_cpu()", "cpu_label(input)").replace("at::kPrivateUse1", "at::kCPU")
    fresh = """at::empty(
            {3, 3, 224, 224}, at::TensorOptions().dtype(at::kHalf).device(at::kCPU))"""
    assert receipt.count(fresh) == 1
    receipt = receipt.replace(fresh, "private_bank({3, 3, 224, 224})")
    patch = _function(source, "static void fused_patch_embedding(")
    dma = patch[patch.index("    TORCH_CHECK(img_slot >= 0"):
                patch.index("    // ===== Step ②")]
    allocation = """at::empty(
                    {N, 3, 224, 224}, at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1))"""
    assert core.count(allocation) == 1
    core = core.replace(allocation, "private_bank({N, 3, 224, 224})")
    memcpy = "std::memcpy(image.data_ptr(), images[i].const_data_ptr(), images[i].nbytes());"
    assert core.count(memcpy) == 1
    core = core.replace(memcpy, memcpy + "\n                mark_dirty(image);")
    core = core.replace("images.front().device().is_cpu()", "cpu_label(images.front())")
    forward = forward.replace("input.device().is_cpu()", "cpu_label(input)")
    for label in ("input", "k_caches[layer]", "v_caches[layer]"):
        forward = forward.replace(label + ".device().type() == at::kPrivateUse1",
            "(!cpu_label(" + label + ") && " + label + ".device().type() == at::kCPU)")
    patch_only = patch_only.replace("input.device().type() == at::kPrivateUse1",
        "(!cpu_label(input) && input.device().type() == at::kCPU)")
    span = span.replace("at::kPrivateUse1", "at::kCPU")
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
 c10::weak_intrusive_ptr<c10::StorageImpl> identity;
 Store(const at::Tensor& t):pointer(t.data_ptr()),ddr(t.nbytes(),0x71),dirty(t.nbytes(),0),
  identity(t.storage().getWeakStorageImpl()){}
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
bool destination_offset=false;
at::Tensor private_bank(at::IntArrayRef shape){++allocations;
 if(!destination_offset)return bank(shape);
 int64_t elements=1;for(auto n:shape)elements*=n;
 return bank({elements+32}).narrow(0,16,elements).view(shape);}
void mark_dirty(const at::Tensor& t){++copy_marks;size_t offset;auto& s=locate(t.data_ptr(),t.nbytes(),offset);
 std::fill(s.dirty.begin()+offset,s.dirty.begin()+offset+t.nbytes(),1);}
void host_write(const at::Tensor& t,const at::Tensor& value){t.copy_(value);mark_dirty(t);}
std::vector<std::pair<void*,size_t>> flushes;
namespace rhino_lkn {
struct Owner {void* base;size_t capacity;
 void* get_cpu_ptr(){return base;}size_t get_memory_size(){return capacity;}};
Owner* RpuGetHostddr(void* p){size_t offset;auto& s=locate(p,0,offset);static Owner owner;
 owner={s.pointer,s.ddr.size()};const auto ptr=reinterpret_cast<uintptr_t>(p);
 if(owner_fault==1)return nullptr;
 if(owner_fault==2)owner.base=nullptr;
 if(owner_fault==3)owner.capacity=0;
 if(owner_fault==4)owner.base=reinterpret_cast<void*>(ptr+1);
 if(owner_fault==5)owner.capacity=301055;
 if(owner_fault==6)owner.capacity=std::numeric_limits<size_t>::max();
 if(owner_fault==7){owner.base=reinterpret_cast<void*>(ptr+32);owner.capacity=301056;}
 if(owner_fault==8)owner.capacity=301024;
 return &owner;}
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
void RpuDdrFlush(void* p,size_t bytes){size_t offset;auto& s=locate(p,bytes,offset);
 flushes.emplace_back(p,bytes);
 const size_t first=offset&~size_t(63),last=(offset+bytes+63)&~size_t(63);
 assert(last<=s.ddr.size());auto* cpu=static_cast<uint8_t*>(s.pointer);
 for(size_t i=first;i<last;++i){if(s.dirty[i])s.ddr[i]=cpu[i];else cpu[i]=s.ddr[i];s.dirty[i]=0;}}
void RpuDdrFlush(void* p){size_t offset;auto& s=locate(p,0,offset);RpuDdrFlush(p,s.ddr.size()-offset);}
}
__MACROS__
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
struct SpmAllocator {struct AllocRequest{int64_t bytes,begin,end;};
 bool is_initialized(){return true;}void init(){}
 std::vector<uint32_t> alloc_temporary_aliased(std::initializer_list<AllocRequest> requests){
  assert(requests.size()==6);return std::vector<uint32_t>(6,0);}
 void reset_temporary(){}} SPM_ALLOC;
struct Layer {at::Tensor q_w,k_w,v_w,o_w,fc1_w,fc2_w,q_ws,k_ws,v_ws,o_ws,fc1_ws,fc2_ws;};
struct Planned {int physical_manifest=0,stage_plan=0;};
struct Prepared {Planned plan;const Planned& candidate(){return plan;}};
struct Siglip {
 // These serial-image ingress cases retain their FP16/W8A16 owner.
 const bool qkv_w8a8_=false,mlp_w8a8_=false;
 int cores=4,attention=4,mlp=4,layers=27;uint64_t generation=1;
 int64_t pe_kh_=14,pe_kw_=14,pe_strideh_=14,pe_stridew_=14,pe_cin_orig_=3,pe_cin_padded_=16,pe_cout_=1152;
 int64_t orig_head_dim_=72,projection_dim_=2048,image_batch_count_=1;
 bool has_patch_emb_=true,linear_acc32_=false,pi05_vision_fc_weight_outer_opt_in_=false;
 bool pi05_vision_merge_segment_opt_in_=false,pi05_vision_kv_spm_only_opt_in_=false,pi05_vision_q_resident_opt_in_=false;
 int prologues=0;at::Tensor patch_emb_weight_,patch_emb_pos_emb_;
 std::vector<Layer> layer_weights_;
 __RECEIPT__
 at::Tensor private_serial_cpu_slab_;
 PrivateSerialCpuSlabSpan private_serial_cpu_slab_span_;
 uint64_t private_serial_cpu_slab_generation_=0;bool private_serial_cpu_slab_published_=false;
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
 // Deliberately non-admitted storage: exercise the original per-image copy.
 auto backing=at::empty({t.numel()+32},t.options());auto view=backing.narrow(0,0,t.numel()).view(t.sizes());
 view.copy_(t);t=std::move(view);label_cpu(t);return t;}
template<class F>void reject_before_prologue(Siglip& m,F call){reset_events();nodes.clear();
 const int before=m.prologues;bool failed=false;try{call();}catch(const c10::Error&){failed=true;}
 assert(failed&&m.prologues==before&&flushes.empty()&&nodes.empty()&&allocations==0);}
int main(){int scenarios=0;
 std::vector<at::Tensor> keys(27,at::zeros({1},at::kHalf)),values=keys;
 const std::vector<int64_t> descriptor{11,12};
 for(bool force:{false,true})for(bool inference:{false,true})for(bool w8:{false,true}){
  c10::InferenceMode mode(inference);g_rpu_ddr_flush_force_enabled=force;
  Siglip m;m.install_precision(w8);auto slab=input_slab();auto a=slab.clone();auto b=a.flip({3});
  std::vector<at::Tensor> saved;std::vector<std::vector<uint8_t>> saved_bytes;
  for(int request=0;request<3;++request){slab.copy_(request==1?b:a);
   for(int camera=0;camera<3;++camera){++scenarios;reset_events();
    auto input=slab.narrow(0,camera,1);const auto expected=bytes(input);
    auto& graph=RpuKernelGraph::active();graph.held.clear();
    graph.current=request==0&&camera==0?RpuKernelGraph::State::RECORDING:RpuKernelGraph::State::REPLAYING;
    if(graph.current==RpuKernelGraph::State::RECORDING)nodes.clear();
    auto output=m.forward(input,keys,values,descriptor);
    assert(allocations==1&&copy_marks==1&&nodes.size()==1);
    assert(flushes.size()==size_t(force));if(force)assert(flushes[0].second==301056);
    assert(m.patch_embedding_live_bases_.input[0]!=reinterpret_cast<uint64_t>(input.data_ptr()));
    size_t private_offset;
    auto private_identity=locate(reinterpret_cast<void*>(m.patch_embedding_live_bases_.input[0]),0,private_offset).identity;
    input=at::Tensor();slab.fill_(0.75); // Copy must precede this producer mutation.
    assert(graph.held.size()==2&&!private_identity.expired());execute_nodes();assert(reads.size()==1);
    if(force){assert(reads[0]==expected);expect_bytes(output,expected);saved_bytes.push_back(expected);}
    else for(uint8_t value:reads[0])assert(value==0x71);
    for(size_t i=0;force&&i<saved.size();++i)expect_bytes(saved[i],saved_bytes[i]);
    saved.push_back(output);graph.held.clear();assert(private_identity.expired());
    slab.copy_(request==1?b:a);
   }
  }
  if(force){for(int i=0;i<3;++i)assert(saved_bytes[i]==saved_bytes[6+i]);}
 }
 g_rpu_ddr_flush_force_enabled=true;RpuKernelGraph::active().current=RpuKernelGraph::State::RECORDING;
 // Public dirty device aliases retain the original sized publication, with a
 // nonzero view offset and adjacent dirty regions preserved by 64 rounding.
 for(int camera=0;camera<3;++camera){++scenarios;Siglip m;auto cpu=input_slab();
  auto backing=bank({3*3*224*224+64});auto view=backing.narrow(0,32+camera*150528,150528).view({1,3,224,224});
  host_write(view,cpu.narrow(0,camera,1));const auto expected=bytes(view);
  reset_events();nodes.clear();auto output=m.forward(view,keys,values,descriptor);
  assert(allocations==0&&flushes.size()==1&&flushes[0].first==view.data_ptr()&&flushes[0].second==301056);
  execute_nodes();assert(reads[0]==expected);expect_bytes(output,expected);
  size_t off;auto& s=locate(view.data_ptr(),view.nbytes(),off);
  assert(s.ddr.front()==0x71&&s.ddr.back()==0x71);RpuKernelGraph::active().held.clear();}
 // Metadata failure rejects the new destination before memcpy/publication/DMA.
 for(int fault=1;fault<=8;++fault){++scenarios;Siglip m;auto slab=input_slab();owner_fault=fault;
  reset_events();nodes.clear();bool failed=false;
  try{m.forward(slab.narrow(0,1,1),keys,values,descriptor);}catch(const c10::Error&){failed=true;}
  assert(failed&&allocations==1&&copy_marks==0&&flushes.empty()&&nodes.empty());owner_fault=0;}
 {++scenarios;Siglip m;auto slab=input_slab();destination_offset=true;reset_events();nodes.clear();
  bool failed=false;try{m.forward(slab.narrow(0,0,1),keys,values,descriptor);}catch(const c10::Error&){failed=true;}
  assert(failed&&copy_marks==0&&flushes.empty()&&nodes.empty());destination_offset=false;}
 // Actual forward guards reject unsupported CPU inputs before prologue/publish.
 for(int fault=0;fault<12;++fault){++scenarios;Siglip m;auto slab=input_slab();auto input=slab.narrow(0,0,1);
  auto k=keys,v=values;
  if(fault==0){input=input.to(at::kFloat);label_cpu(input);}
  if(fault==1)input=input.transpose(2,3);
  if(fault==2){input=input.reshape({1,3,224*224});}
  if(fault==3)input=at::_neg_view(input);
  if(fault==4){input=input.clone();label_cpu(input);input.set_requires_grad(true);}
  if(fault==5)m.generation=0;
  if(fault==6)m.cores=8;
  if(fault==7)m.attention=8;
  if(fault==8)m.layer_weights_[12].fc2_w=at::ones({1},at::kChar);
  if(fault==9){k[0]=at::zeros({1},at::kHalf);label_cpu(k[0]);}
  if(fault==10)k.resize(26);
  if(fault==11){input=input.clone();label_cpu(input);input.storage().set_nbytes(301055);}
  reject_before_prologue(m,[&]{m.forward(input,k,v,descriptor);});}
 {++scenarios;Siglip m;auto slab=input_slab();reject_before_prologue(m,[&]{m.patch_embed(slab.narrow(0,0,1));});}
 {++scenarios;Siglip m;auto slab=input_slab();reset_events();nodes.clear();bool failed=false;
  try{m.run_packed_patch_embed_core({slab.narrow(0,0,1)},false);}catch(const c10::Error&){failed=true;}
  assert(failed&&allocations==0&&flushes.empty()&&nodes.empty());}
 std::cout<<"Whole serial CPU SigLIP forward: "<<scenarios<<" scenarios PASS\n";
}
'''
    replacements = {"__MACROS__": macros, "__LIVE__": live, "__SPAN__": span,
        "__DMA__": dma, "__RECEIPT__": receipt, "__OWNER__": owner, "__FORWARD__": forward,
        "__CORE__": core, "__PACKER__": packer, "__PATCH_ONLY__": patch_only}
    for marker, body in replacements.items():
        assert cpp.count(marker) == 1
        cpp = cpp.replace(marker, body)
    output = _run_native(tmp_path, cpp, torch)
    assert "Whole serial CPU SigLIP forward:" in output and "scenarios PASS" in output

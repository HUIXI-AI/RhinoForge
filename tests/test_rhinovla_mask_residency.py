"""Rhino Action mask ownership and actual SPM allocation; no device use."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess

import pytest
import torch

from rpu_backend.adapters.rhinovla.fused import (
    _cached_rhino_attention_bias, _aligned_rhino_action_prefix,
    _rhino_suffix_rope_start,
)
from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/fused/rpu_rhino_vla_model.cpp'


def _run_cpp(tmp_path, program):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('host C++ compiler unavailable')
    source, binary = tmp_path/'probe.cpp', tmp_path/'probe'
    source.write_text(program)
    build = subprocess.run([compiler, '-std=c++17', '-O0', str(source), '-o', str(binary)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_content_cache_preserves_physical_padding_and_mutations():
    calls = []
    def make(prefix, suffix):
        calls.append(1)
        visible = torch.cat((prefix, suffix), dim=1).bool()
        return visible[:, None, :].expand(-1, suffix.shape[1], -1)
    def bias(mask, dtype):
        return torch.where(mask, torch.tensor(0, dtype=dtype), torch.tensor(torch.finfo(dtype).min, dtype=dtype))[:, None]
    owner = SimpleNamespace()
    with torch.inference_mode():
        prefix = torch.tensor([[1]*7+[0]*3], dtype=torch.bool)
        suffix = torch.tensor([[1]*3+[0]], dtype=torch.bool)
        a = _cached_rhino_attention_bias(owner, prefix, suffix, make, bias)
        same = _cached_rhino_attention_bias(owner, prefix.clone(), suffix.clone(), make, bias)
        assert a is same and len(calls) == 1
        assert a.shape == (1,1,4,16) and not a.is_inference()
        # Logical RoPE starts at seven. Physical suffix starts at ten, and its
        # padding row plus aligned K tail must remain invisible.
        assert torch.all(a[...,7:10] < 0) and torch.all(a[...,10:13] == 0)
        assert torch.all(a[...,13:] < 0)
        before = a._version
        a.add_(0)
        assert a._version > before
        prefix[:,6] = False
        b = _cached_rhino_attention_bias(owner, prefix, suffix, make, bias)
        assert b is not a and b[...,6].max() < 0 and a[...,6].min() == 0
        suffix[:,2] = False
        c = _cached_rhino_attention_bias(owner, prefix, suffix, make, bias)
        assert c is not b and c[...,12].max() < 0
        other = _cached_rhino_attention_bias(SimpleNamespace(), prefix, suffix, make, bias)
        assert other is not c and torch.equal(other, c)
    assert len(calls) == 4


@pytest.mark.parametrize("prefix,logical", ((230,230),(240,230),(255,251)))
def test_aligned_action_prefix_preserves_visibility_rope_and_input(prefix, logical):
    mask = torch.arange(prefix)[None] < logical
    original = mask.clone()
    position, padded = _aligned_rhino_action_prefix(
        mask, prefix_len=prefix, suffix_len=31, capacity=288,
    )
    assert position == (prefix+15)//16*16
    assert torch.equal(mask, original) and torch.equal(padded[:,:prefix], original)
    assert not padded[:,prefix:].any()
    kwargs = dict(batch_size=1, prefix_rope_deltas=torch.tensor([[-168]]))
    assert _rhino_suffix_rope_start(runtime_prefix_len=prefix, prefix_mask=mask, **kwargs) == \
        _rhino_suffix_rope_start(runtime_prefix_len=position, prefix_mask=padded, **kwargs)
    # The original text mask must remain live, not a reused padded snapshot.
    mask[:,logical-1] = False
    _, changed = _aligned_rhino_action_prefix(mask, prefix_len=prefix, suffix_len=31, capacity=288)
    assert not changed[:,logical-1].any()


def test_aligned_action_prefix_rejects_capacity_and_interior_holes():
    mask = torch.ones(1,230,dtype=torch.bool)
    with pytest.raises(ValueError,match="32-row write"):
        _aligned_rhino_action_prefix(mask,prefix_len=230,suffix_len=31,capacity=271)
    mask[:,10] = False
    with pytest.raises(ValueError,match="valid rows followed"):
        _aligned_rhino_action_prefix(mask,prefix_len=230,suffix_len=31,capacity=272)


def _declaration_program():
    from test_hy_halo_mask_residency import _program
    text = SOURCE.read_text()
    # Reuse only the actual common SDPA sizing/allocator/BufferDecl preamble.
    program = _program(preamble_only=True)
    program += r'''
namespace c10 {using Half=uint16_t;}
namespace at {struct Tensor {template<class T>T* data_ptr()const{return nullptr;}};}
void rpu_launch_ddr_broadcast_spm_dma(const uint16_t*,int64_t,uint32_t,int=8){}
void rpu_launch_memset_spm_multicore(uint32_t,int64_t,int=8){}
class RhinoVLAModel {public:
 struct {bool gated_no_sub=false,adarms_resident=false,packed_qkv=false,aligned_kv=false;}cold_config_;
 int64_t num_steps_=10;
 bool denoise_loop_weights_ready_=false,direct_action_input_=false,vector_q_norm_=false,vector_k_norm_=false,rtc_mode_=false;
 int64_t action_horizon_=30,action_dim_pad_=80,state_dim_pad_=80;
 at::Tensor action_in_b_,action_time_in_b_,time_in_b_,time_out_b_,state_b_,state_mask_b_,action_mask_b_,action_out_b_;
 struct Layer {at::Tensor q_norm_w,k_norm_w;};std::vector<Layer>layer_weights_{18};
 int64_t hidden_size()const{return 1024;}int64_t intermediate_size()const{return 3072;}
 int64_t num_q_heads()const{return 16;}int64_t num_kv_heads()const{return 8;}
 int lanes=8,cores=8;
 int64_t head_dim()const{return 128;}int64_t num_layers()const{return 18;}int attn_tp()const{return lanes;}
 int num_cores()const{return cores;}int mlp_tp()const{return cores;}
'''
    for marker in ('SdpaConfig make_sdpa_config(', 'std::vector<BufferDecl> baseline_buffer_declarations(', 'std::vector<BufferDecl> declare_buffers('):
        program += _definition(text, marker).replace(' override', '') + '\n'
    return program + '};\n'


def test_actual_declarations_retain_without_growth_or_alias(tmp_path):
    program = _declaration_program() + r'''
int main(){
 RhinoVLAModel m;
 for(bool aligned:{false,true})for(int64_t rows:{31,32})for(bool packed:{false,true})for(bool resident:{false,true})for(bool direct:{false,true})for(bool loop:{false,true})for(bool gated:{false,true})for(int64_t prefix:{48,171,240}){
  m.cold_config_.aligned_kv=aligned;
  m.cold_config_.packed_qkv=packed;
  m.cold_config_.adarms_resident=resident;
  m.direct_action_input_=direct;m.action_dim_pad_=m.state_dim_pad_=direct?96:80;
  m.denoise_loop_weights_ready_=loop;m.cold_config_.gated_no_sub=gated;
  LayoutContext l;l.chunk_size=rows;l.max_kv_seq_len=Align(prefix+31,int64_t{16});l.use_attn_mask=true;l.is_causal=false;
  auto original=m.baseline_buffer_declarations(l),candidate=original;
  auto decision=detail::plan_forward_spm_residency(candidate,{"sdpa_mask"});
  auto actual=m.declare_buffers(l);
  if(aligned)for(const auto&d:actual)
   if(std::string(d.name)=="k"||std::string(d.name)=="v")assert(d.size>=32*128*2);
  if(packed){
   assert(std::string(actual[0].name)=="packed_qkv"&&actual[0].size==rows*512*2);
   assert(actual[0].storage==StorageClass::Temp&&actual[0].scope==BufferScope::LayerWide);
  }
  if(resident){
   auto index=packed?1:0;
   assert(std::string(actual[index].name)=="adarms_pair_resident"&&actual[index].size==10*18*6*1024*2);
   assert(std::string(actual[index+1].name)=="adarms_final_resident"&&actual[index+1].size==10*3*1024*2);
   assert(actual[index].storage==StorageClass::Temp&&actual[index+1].storage==StorageClass::Temp);
   int64_t persistent=0;
   for(const auto&d:actual)if(d.storage==StorageClass::Persistent||d.storage==StorageClass::PersistentPerLayer)
    persistent+=Align(d.size,int64_t{256})*(d.storage==StorageClass::PersistentPerLayer?d.per_layer:1);
   assert(persistent+decision.retained_peak_bytes<8303616);
  }
  assert(decision.retained&&decision.original_peak_bytes==decision.retained_peak_bytes);
  std::vector<SpmAllocator::AllocRequest> req;size_t mask=0;
  for(size_t i=0;i<actual.size();++i){
   assert(actual[i].size==original[i].size&&actual[i].storage==original[i].storage);
   assert(actual[i].phase_start==candidate[i].phase_start&&actual[i].phase_end==candidate[i].phase_end);
   auto&d=actual[i];if(d.storage!=StorageClass::Temp||d.alias_of)continue;
   if(std::string(d.name)=="sdpa_mask")mask=req.size();
   req.push_back({d.size,d.phase_start,d.phase_end,static_cast<int>(d.scope)});
  }
  auto plan=SpmAllocator::plan_temporary_aliased(req);
  size_t begin=plan.offsets[mask],end=begin+req[mask].bytes;
  std::vector<unsigned char>spm(plan.peak_bytes,0);
  std::fill(spm.begin()+begin,spm.begin()+end,0x5a);
  for(int body=0;body<10;++body)for(size_t i=0;i<req.size();++i)if(i!=mask){
   size_t lo=plan.offsets[i],hi=lo+req[i].bytes;
   assert(hi<=begin||end<=lo);std::fill(spm.begin()+lo,spm.begin()+hi,body+1);
  }
  assert(std::all_of(spm.begin()+begin,spm.begin()+end,[](auto c){return c==0x5a;}));
  std::cout<<direct<<","<<loop<<","<<gated<<","<<prefix<<","<<decision.original_peak_bytes<<","<<decision.retained_peak_bytes<<"\n";
 }
 return 0;
}
}
int main(){return v3::main();}
'''
    output = _run_cpp(tmp_path, program)
    assert len(output.strip().splitlines()) == 384
    print('direct_input,unroll,gated,prefix,old_peak,new_peak\n'+output)


def test_vector_q_norm_actual_declarations_add_only_two_tail_rows(tmp_path):
    program = _declaration_program() + r'''
int main(){
 RhinoVLAModel m;
 m.cold_config_.adarms_resident=m.cold_config_.gated_no_sub=true;
 m.denoise_loop_weights_ready_=m.direct_action_input_=true;
 m.action_dim_pad_=m.state_dim_pad_=96;
 LayoutContext l;l.chunk_size=31;l.max_kv_seq_len=272;l.use_attn_mask=true;l.is_causal=false;
 auto temporary=[](const std::vector<BufferDecl>& ds){
  std::vector<SpmAllocator::AllocRequest> requests;
  for(const auto& d:ds)if(d.storage==StorageClass::Temp&&!d.alias_of)
   requests.push_back({d.size,d.phase_start,d.phase_end,static_cast<int>(d.scope)});
  return requests;
 };
 // Q padding is independent of whether KV uses its own aligned storage.
 for(bool aligned:{false,true}){
  m.cold_config_.aligned_kv=aligned;m.vector_q_norm_=false;
  const auto original=m.declare_buffers(l);
  m.vector_q_norm_=true;
  const auto padded=m.declare_buffers(l);
  assert(original.size()==padded.size());
  int expanded=0,debug_checked=0;int64_t growth=0;
  for(size_t i=0;i<original.size();++i){
   const auto& a=original[i];const auto& b=padded[i];
   assert(std::string(a.name)==b.name&&a.storage==b.storage&&a.scope==b.scope);
   assert(a.phase_start==b.phase_start&&a.phase_end==b.phase_end&&a.alias_of==b.alias_of);
   assert(a.per_layer==b.per_layer);
   const std::string name=a.name;
   if(name=="q"||name=="output"){
    assert(a.size==15872&&b.size==16384&&b.storage==StorageClass::Temp&&!b.alias_of);
    assert(b.size-a.size==2*128*2);++expanded;
   }else assert(a.size==b.size);
   if(name=="dbg_q_prenorm"||name=="dbg_q_norm"||name=="dbg_sdpa_out"){
    assert(a.size==15872&&b.size==15872);++debug_checked;
   }
   if(a.storage==StorageClass::Temp)growth+=b.size-a.size;
  }
  assert(expanded==2&&debug_checked==3&&growth==1024);
  const auto old_requests=temporary(original),new_requests=temporary(padded);
  const auto before=SpmAllocator::plan_temporary_aliased(old_requests);
  const auto after=SpmAllocator::plan_temporary_aliased(new_requests);
  assert(after.peak_bytes==before.peak_bytes+1024);
  // The real allocator must place both enlarged extents without overlaps.
  for(size_t i=0;i<new_requests.size();++i){
   const size_t lo=after.offsets[i];
   const size_t hi=lo+new_requests[i].bytes;
   assert(hi<=after.peak_bytes);
   for(size_t j=0;j<i;++j)
    assert(hi<=after.offsets[j]||after.offsets[j]+new_requests[j].bytes<=lo);
  }
  int64_t persistent=0;
  for(const auto& d:padded)if(d.storage==StorageClass::Persistent||d.storage==StorageClass::PersistentPerLayer)
   persistent+=Align(d.size,int64_t{256})*(d.storage==StorageClass::PersistentPerLayer?d.per_layer:1);
  assert(after.peak_bytes+persistent<8303616);
 }
 return 0;
}
}
int main(){return v3::main();}
'''
    _run_cpp(tmp_path, program)


def test_reduced_vector_k_norm_pads_only_kv_without_aligned_storage(tmp_path):
    program = _declaration_program() + r'''
int main(){
 RhinoVLAModel m;
 m.lanes=4;m.cores=6;
 m.cold_config_.gated_no_sub=true;
 m.denoise_loop_weights_ready_=m.direct_action_input_=true;
 m.action_dim_pad_=m.state_dim_pad_=96;
 LayoutContext l;l.chunk_size=31;l.max_kv_seq_len=272;l.use_attn_mask=true;l.is_causal=false;
 const auto original=m.declare_buffers(l);
 m.vector_k_norm_=true;
 const auto padded=m.declare_buffers(l);
 assert(original.size()==padded.size());
 int expanded=0;
 for(size_t i=0;i<original.size();++i){
  const auto& a=original[i];const auto& b=padded[i];
  assert(std::string(a.name)==b.name&&a.storage==b.storage&&a.scope==b.scope&&a.alias_of==b.alias_of);
  const std::string name=a.name;
  // Four lanes hold two K/V heads each: 31 -> 32 rows of 256 halves. The
  // debug K copies share that extent, as with eight-core aligned KV.
  if(name=="k"||name=="v"||name=="dbg_k_prenorm"||name=="dbg_k_norm"){
   assert(a.size==Align(int64_t{31*256*2},int64_t{256})&&b.size==32*256*2);++expanded;
  }else assert(a.size==b.size);
 }
 assert(expanded==4);
 return 0;
}
}
int main(){return v3::main();}
'''
    _run_cpp(tmp_path, program)


def test_expert_payload_lookup_is_cold_and_planning_only_reads_values(tmp_path):
    source = SOURCE.read_text()
    program = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do {if(!(condition))throw std::runtime_error("rejected");} while(false)
bool dry=false,available=true;int lookups=0;
enum class KernelId {PI05_ADARMS_NORM_SHIFT_H1024,PI05_GATED_RESIDUAL_H1024};
void* GET_KERNEL(KernelId){assert(!dry);++lookups;return available?reinterpret_cast<void*>(1):nullptr;}
'''
    program += _definition(source, 'struct RhinoVLAColdConfig {') + ';\n'
    program += r'''
class RhinoVLAModel {public:
 RhinoVLAColdConfig cold_config_;bool cold_config_bound_=false,expert_w8a16_=false,full_action_w8a16_=false;
 bool expert_cond_w8a16_=false,high_precision_=false;
 bool denoise_loop_weights_ready_=false,loop_mode_=true,direct_action_input_=true,denoise_adarms_tables_ready_=true;
 int layers=0,num_steps_=10,action_dim_=96,state_dim_=96,action_horizon_=30,suffix_len_=31,invalidations=0;
 std::vector<int>layer_weights_;
 std::vector<int>full_io_scales_,full_cold_weights_,full_cold_scales_,full_cold_biases_;
 int num_layers()const{return layers;}int hidden_size()const{return 1024;}
 int intermediate_size()const{return 3072;}int num_q_heads()const{return 16;}
 int num_kv_heads()const{return 8;}int head_dim()const{return 128;}int attn_tp()const{return 8;}
 int num_cores()const{return 8;}
 void invalidate_model_state(){++invalidations;}
'''
    program += _definition(source, 'void set_runtime_config(') + '\n'
    program += _definition(source, 'void validate_expert_transfer(') + '\n};\n'
    program += r'''
int main(){
 RhinoVLAModel m;
 auto install=[&]{m.set_runtime_config(true,false,false,true,false,false,true,true,true,true);};
 available=false;bool rejected=false;
 try{install();}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&!m.cold_config_bound_&&m.invalidations==0);
 available=true;install();assert(m.cold_config_bound_&&m.invalidations==1);
 m.layers=18;int before=lookups;dry=true;
 m.validate_expert_transfer(31);assert(lookups==before);
 rejected=false;try{m.validate_expert_transfer(32);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&lookups==before);
 m.num_steps_=9;rejected=false;try{m.validate_expert_transfer(31);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&lookups==before);
 dry=false;RhinoVLAModel packed;
 packed.set_runtime_config(false,false,false,false,false,false,false,false,false,false,true);
 packed.layers=18;packed.denoise_adarms_tables_ready_=false;dry=true;
 packed.validate_expert_transfer(31);assert(lookups==before);
 packed.direct_action_input_=false;rejected=false;
 try{packed.validate_expert_transfer(31);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&lookups==before);
}
'''
    _run_cpp(tmp_path, program)


def test_packed_qkv_split_registers_address_every_core_and_row(tmp_path):
    # Execute the real launcher against a recording kernel/queue. Decode the
    # shipped AllGather SCM ABI independently, including nonzero core bases;
    # the original broadcast slice incorrectly addressed core zero eight times.
    program = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define NUM_CORES 8
#define TORCH_CHECK(condition, ...) do {if(!(condition))throw std::runtime_error("rejected");} while(false)
constexpr int RHINO_TP8 = 8;
namespace c10 {using Half=uint16_t;}
struct SpmAllocator {
 static constexpr uint32_t SPM_USABLE=8387584;
 uint32_t addr(int core,uint32_t offset)const{return 0x40000000u+core*0x1000000u+offset;}
} SPM_ALLOC;
enum class KernelId {ALL_GATHER_MULTI_CORE_LITTLE_CHUNK};
enum class GraphKernelNoDdrProof {AllGatherSpm};
struct Kernel_t {
 std::array<uint16_t,8192> r{};
 void reset_regs(){r.fill(0);}
 void set_regs(uint32_t i,uint16_t v){r.at(i)=v;}
} kernel;
int calls=0,stages=0;
struct RpuKernelGraph {
 static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void stage_kernel_no_ddr(KernelId,GraphKernelNoDdrProof){++stages;}
};
Kernel_t* GET_KERNEL(KernelId){return &kernel;}
void rpu_set_legacy_scm_u16_checked(Kernel_t* k,uint32_t i,uint16_t v,const char*){k->set_regs(i,v);}
struct Queue {
 bool broadcast=false;
 void set_broadcast_mode(bool value){broadcast=value;}
 void enqueu_kernel(Kernel_t&,std::array<uint16_t,3> grid,const std::vector<uint8_t>& cores){
  assert(broadcast&&grid[0]==8&&grid[1]==1&&grid[2]==1&&cores.size()==8);
  for(int c=0;c<8;++c)assert(cores[c]==c);
  ++calls;
 }
} queue;
Queue* GET_QUEUE(int n){assert(n==8);return &queue;}
'''
    program += _definition(SOURCE.read_text(), 'void rpu_launch_rhino_packed_qkv_split_spm_kernel(')
    program += r'''
uint32_t u32(int lo,int hi){return kernel.r[lo]|(uint32_t(kernel.r[hi])<<16);}
int main(){
 const uint32_t src=SPM_ALLOC.addr(0,0x10000),dst=SPM_ALLOC.addr(0,0x20000);
 for(auto part:std::vector<std::array<int,2>>{{0,256},{256,128},{384,128}}){
  int begin=part[0],width=part[1];
  rpu_launch_rhino_packed_qkv_split_spm_kernel(src,dst,31,begin,width);
  assert(kernel.r[0]==1&&kernel.r[1]==31&&kernel.r[7]>0);
  assert(u32(2,3)==uint32_t(width*2));
  for(int core=0;core<8;++core){
   uint32_t source=u32(4096+40+core,4096+48+core);
   uint32_t target=u32(4096+56+core*16+core,4096+64+core*16+core);
   uint32_t stride=u32(4096+24+core,4096+32+core);
   assert(u32(4096+core,4096+8+core)==uint32_t(width*2));
   for(int row=0;row<31;++row)for(int col=0;col<width;++col){
    uint32_t read=source+row*stride+col*2;
    uint32_t write=target+row*u32(2,3)+col*2;
    assert(read/0x1000000u==uint32_t(core)&&write/0x1000000u==uint32_t(core));
    assert(read%0x1000000u==0x10000u+row*512*2+(begin+col)*2);
    assert(write%0x1000000u==0x20000u+(row*width+col)*2);
   }
  }
 }
 assert(calls==3&&stages==3);
 const auto reject=[&](uint32_t s,uint32_t d,int rows,int start,int cols){
  bool rejected=false;try{rpu_launch_rhino_packed_qkv_split_spm_kernel(s,d,rows,start,cols);}
  catch(const std::runtime_error&){rejected=true;}
  assert(rejected&&calls==3&&stages==3);
 };
 reject(src,src+256,31,0,256);
 reject(src,dst,32,0,256);
 reject(src,dst,31,128,256);
 reject(src,dst+2,31,0,256);
 reject(src,SPM_ALLOC.addr(0,SpmAllocator::SPM_USABLE-32),31,0,256);
}
'''
    _run_cpp(tmp_path, program)


def test_actual_prepare_cache_identity_version_failure_and_untracked_input(tmp_path):
    method = _definition(SOURCE.read_text(), 'void prepare_sdpa_mask_cached(')
    program = r'''
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("guard");}while(0)
template<class T>T Align(T x,T n){return (x+n-1)/n*n;}
namespace at {
struct Version{bool tracked=true;int64_t value=0;bool enabled()const{return tracked;}int64_t current_version()const{return value;}};
struct Impl{Version version;int value=1;const Version& version_counter()const{return version;}};
struct Tensor{
 std::shared_ptr<Impl>p;bool defined()const{return bool(p);}
 bool is_same(const Tensor&o)const{return p==o.p;}Impl*unsafeGetTensorImpl()const{return p.get();}
};
}
struct PreparedMask{int value=0,mask_type=0;};
int prepares=0;bool fail=false,mutate=false;
PreparedMask sdpa_prepare_mask(std::optional<at::Tensor> t,bool causal,int64_t,int64_t,int){
 ++prepares;if(mutate)++t->p->version.value;
 if(fail)throw std::runtime_error("prepare");
 return {t&&t->defined()?t->p->value:0,causal?1:t&&t->defined()?4:0};
}
struct Model{
 at::Tensor prepared_mask_input_ref_;PreparedMask prepared_mask_;
 int64_t prepared_mask_version_=-1,prepared_mask_q_=-1,prepared_mask_k_=-1;
 bool prepared_mask_causal_=false;int sdpa_stable_mask_cache(){return 0;}
'''+method+r'''
};
int main(){
 Model m;at::Tensor a{std::make_shared<at::Impl>()};
 auto run=[&](at::Tensor x){m.prepare_sdpa_mask_cached(x,false,31,79);};
 run(a);run(a);assert(prepares==1&&m.prepared_mask_.value==1&&m.prepared_mask_k_==80);
 a.p->value=2;++a.p->version.value;run(a);assert(prepares==2&&m.prepared_mask_.value==2);
 at::Tensor b{std::make_shared<at::Impl>()};b.p->value=3;run(b);assert(prepares==3&&m.prepared_mask_.value==3);
 fail=true;bool caught=false;try{run(a);}catch(...){caught=true;}
 assert(caught&&!m.prepared_mask_input_ref_.defined());fail=false;run(b);assert(prepares==5);
 mutate=true;caught=false;try{run(a);}catch(...){caught=true;}
 assert(caught&&!m.prepared_mask_input_ref_.defined());mutate=false;run(a);assert(prepares==7);
 a.p->version.tracked=false;run(a);run(a);assert(prepares==9);
 m.prepare_sdpa_mask_cached(a,false,31,81);assert(prepares==10&&m.prepared_mask_k_==96);
 std::weak_ptr<at::Impl> kept=a.p;a={};assert(!kept.expired());
 m.prepare_sdpa_mask_cached(std::nullopt,true,31,79);assert(kept.expired()&&m.prepared_mask_.mask_type==1);
}
'''
    _run_cpp(tmp_path, program)


def test_actual_typed_emitter_uploads_each_forward_and_rejects_drift(tmp_path):
    text = SOURCE.read_text()
    program = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("guard");}while(0)
#define NUM_CORES 8
constexpr int64_t RHINO_MASK_SPM_SCHEDULE_SITE=2981318290883577335LL;
enum class FmbRouteFamily{GRAPH_SCHEDULE};struct ChunkInfo{int idx=0;};
struct PreparedMask{int mask_type=4;};
struct Route{int64_t selector=2,flags=0;std::vector<int64_t>arguments;};
struct Context{
 int body_iter=0,consumed=0;bool complete=true;Route route;
 bool has_complete_physical_manifest()const{return complete;}
 const Route&find_physical_route(FmbRouteFamily,int64_t,int)const{return route;}
 void consume_physical_route(FmbRouteFamily,int64_t,int64_t s,int64_t f,const std::vector<int64_t>&a,int){
  TORCH_CHECK(s==route.selector&&f==route.flags&&a==route.arguments);++consumed;
 }
};
int dmas=0;void sdpa_dma_mask_to_spm(PreparedMask,uint32_t,int64_t,int64_t,int){++dmas;}
'''+_definition(text, 'enum class RhinoMaskSpmSchedule')+';\n'+r'''
struct Model{Context state;bool loop_mode_=false;int64_t num_steps_=2;PreparedMask prepared_mask_;
 Context&ctx(){return state;}int64_t num_layers()const{return 2;}int num_cores()const{return 8;}
'''+_definition(text, 'int emit_sdpa_mask(')+r'''
 void forward(){for(int b=0;b<(loop_mode_?num_steps_:1);++b){state.body_iter=b;
  for(int layer=0;layer<2;++layer)assert(emit_sdpa_mask(layer,ChunkInfo{},31,79,80,8,0)==4);
 }}
};
int main(){
 for(bool loop:{false,true})for(bool retained:{false,true}){
  Model m;m.loop_mode_=loop;m.state.route={retained?2:1,0,{31,80,8,8,loop?2:1,2,10000,retained?10000:15000,79}};
  dmas=0;m.forward();m.forward();assert(dmas==2*(retained?1:(loop?4:2)));assert(m.state.consumed==2);
 }
 for(int fault=0;fault<6;++fault){
  Model m;m.state.route.arguments={31,80,8,8,1,2,10000,10000,79};
  if(fault==0)m.state.route.selector=99;if(fault==1)m.state.route.arguments[8]=78;
  if(fault==2)m.state.route.arguments[4]=2;if(fault==3)m.state.route.arguments[7]=11000;
  if(fault==4)m.state.route.arguments.pop_back();if(fault==5)m.state.route.flags=1;
  bool caught=false;dmas=0;try{m.forward();}catch(...){caught=true;}assert(caught&&dmas==0);
 }
 Model old;old.loop_mode_=true;old.state.complete=false;dmas=0;old.forward();assert(dmas==4);
}
'''
    _run_cpp(tmp_path, program)


def test_rtc_actual_declarations_add_36kib_and_retain_mask(tmp_path):
    program = _declaration_program() + r'''
int main(){
 RhinoVLAModel m;
 m.denoise_loop_weights_ready_=m.direct_action_input_=true;
 m.cold_config_.gated_no_sub=m.cold_config_.aligned_kv=true;
 m.action_dim_pad_=m.state_dim_pad_=96;
 // Full H30/M31 v3 geometry. K304 is the fixed P230 fixture's complete
 // reserved cache envelope (P256 + 31 suffix + 16 spare, rounded to 16).
 // All delays 0..6 share these maximum three-group buffer declarations.
 LayoutContext l;l.chunk_size=31;l.max_kv_seq_len=304;l.use_attn_mask=true;l.is_causal=false;
 const auto scalar=m.declare_buffers(l);
 m.rtc_mode_=true;
 const auto rtc=m.declare_buffers(l);
 assert(scalar.size()==rtc.size());
 int expanded=0;int64_t growth=0,persistent=0;
 std::vector<SpmAllocator::AllocRequest> scalar_requests,rtc_requests;
 size_t mask=0;
 for(size_t i=0;i<scalar.size();++i){
  const auto& a=scalar[i];const auto& b=rtc[i];
  assert(std::string(a.name)==b.name&&a.storage==b.storage&&a.scope==b.scope);
  assert(a.phase_start==b.phase_start&&a.phase_end==b.phase_end);
  assert(a.alias_of==b.alias_of&&a.per_layer==b.per_layer);
  const std::string name=a.name;
  if(name=="adarms_pair_gemv"){
   assert(a.size==6*1024*2&&b.size==3*a.size);++expanded;
  }else if(name=="final_gemv"){
   assert(a.size==3*1024*2&&b.size==3*a.size);++expanded;
  }else assert(a.size==b.size);
  growth+=b.size-a.size;
  if(b.storage==StorageClass::Persistent||b.storage==StorageClass::PersistentPerLayer)
   persistent+=Align(b.size,int64_t{256})*(b.storage==StorageClass::PersistentPerLayer?b.per_layer:1);
  if(b.storage!=StorageClass::Temp||b.alias_of)continue;
  if(name=="sdpa_mask")mask=rtc_requests.size();
  scalar_requests.push_back({a.size,a.phase_start,a.phase_end,static_cast<int>(a.scope)});
  rtc_requests.push_back({b.size,b.phase_start,b.phase_end,static_cast<int>(b.scope)});
 }
 assert(expanded==2&&growth==36*1024);
 const auto before=SpmAllocator::plan_temporary_aliased(scalar_requests);
 const auto after=SpmAllocator::plan_temporary_aliased(rtc_requests);
 assert(after.peak_bytes-before.peak_bytes==36*1024);
 auto declarations=m.baseline_buffer_declarations(l);
 const auto retention=detail::plan_forward_spm_residency(declarations,{"sdpa_mask"});
 assert(retention.retained&&retention.original_peak_bytes==retention.retained_peak_bytes);
 assert(after.peak_bytes+persistent<8303616);
 const size_t mask_begin=after.offsets[mask],mask_end=mask_begin+rtc_requests[mask].bytes;
 std::vector<unsigned char>spm(after.peak_bytes,0);
 std::fill(spm.begin()+mask_begin,spm.begin()+mask_end,0x5a);
 for(int step=0;step<10;++step)for(size_t i=0;i<rtc_requests.size();++i)if(i!=mask){
  const size_t begin=after.offsets[i],end=begin+rtc_requests[i].bytes;
  assert(end<=after.peak_bytes&&(end<=mask_begin||mask_end<=begin));
  std::fill(spm.begin()+begin,spm.begin()+end,step+1);
 }
 assert(std::all_of(spm.begin()+mask_begin,spm.begin()+mask_end,[](auto c){return c==0x5a;}));
 std::cout<<before.peak_bytes<<","<<after.peak_bytes<<","<<persistent<<","<<after.peak_bytes+persistent<<"\n";
 return 0;
}
}
int main(){return v3::main();}
'''
    output = _run_cpp(tmp_path, program)
    print("scalar_temp,rtc_temp,persistent,total_bytes\n" + output)

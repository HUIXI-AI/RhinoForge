"""Execute the HyVLA S51/S64 paired-KV admission and COMPLETE route with host metadata."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest
from _native_runtime_identity import runtime_identity_include

ROOT = Path(__file__).resolve().parents[1]
DECODER = (ROOT / 'src/fused/rpu_hyvla_expert_model.cpp').read_text()
LAUNCHER = (ROOT / 'src/ops/rpu_llama_kvcache.cpp').read_text()


def block(source, marker):
    start = source.index(marker)
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def compile_run(tmp_path, name, source):
    import torch
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    torch_root = Path(torch.__file__).resolve().parent
    if "rpu_specialized_linear::" in source:
        # Execute the production tile/grid contract alongside the actual
        # launcher. The host fixture supplies only the KernelId type; it does
        # not replace the contract with a hand-written fake.
        contract = (ROOT / "src/ops/rpu_specialized_linear_contracts.h").read_text()
        contract = re.sub(r"^#(?:pragma once|include .*)\n", "", contract, flags=re.M)
        ids = set(re.findall(r"\bKernelId::(\w+)", contract + source))
        match = re.search(r"enum class KernelId(?:\s*:\s*uint16_t)?\s*\{([^}]*)\};", source)
        if match:
            existing = set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", match[1]))
            body = match[1].rstrip().rstrip(",")
            declaration = "enum class KernelId : uint16_t {" + body + "," + ",".join(sorted(ids - existing)) + "};"
            source = source[:match.start()] + declaration + "\n" + contract + source[match.end():]
        else:
            assert source.startswith(COMMON), "Locate a real namespace-level contract insertion point"
            source = (COMMON + "\nenum class KernelId : uint16_t {" + ",".join(sorted(ids)) + "};\n"
                      + contract + source[len(COMMON):])
    cpp = tmp_path / (name + '.cpp')
    cpp.write_text(source)
    binary = tmp_path / name
    command = [compiler, '-std=c++17', '-O0', '-Wall', '-Wextra',
               f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
               '-I' + str(ROOT / 'src'), '-I' + str(runtime_identity_include(tmp_path)), '-I' + str(torch_root / 'include'),
               str(cpp), str(ROOT / 'src/ops/rpu_kvinsert_segment_plan.cpp'),
               '-L' + str(torch_root / 'lib'), '-Wl,-rpath,' + str(torch_root / 'lib'),
               '-lc10', '-o', str(binary)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


COMMON = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/core/DeviceType.h>
#include "ops/rpu_kvinsert_segment_plan.h"

#undef TORCH_CHECK
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
constexpr int NUM_CORES=8;
namespace at {
enum ScalarType { kHalf,kChar,kFloat,kByte };
struct Device {
 c10::DeviceType kind=c10::DeviceType::PrivateUse1; int index=0;
 c10::DeviceType type() const { return kind; }
 bool operator==(Device b) const { return kind==b.kind && index==b.index; }
};
struct Tensor {
 uintptr_t address=0x10000000; std::vector<int64_t> shape{1,27,1,8,8,16,16};
 ScalarType dtype=kHalf; bool present=true,contiguous=true; Device where;
 bool defined()const{return present;} Device device()const{return where;}
 ScalarType scalar_type()const{return dtype;} bool is_contiguous()const{return contiguous;}
 int64_t dim()const{return shape.size();} int64_t size(size_t i)const{return shape.at(i);}
 size_t nbytes()const{size_t n=(dtype==kChar||dtype==kByte)?1:(dtype==kFloat?4:2);for(auto d:shape)n*=d;return n;}
 void* data_ptr()const{return reinterpret_cast<void*>(address);}
 template<class T> T* data_ptr()const{return reinterpret_cast<T*>(address);}
};
using IntArrayRef=c10::ArrayRef<int64_t>;
}
#include "fused/hyvla_linear_precision.h"
#include "fused/hyvla_retained_schedule.h"
auto exact_plan(int64_t p=240,int64_t rows=51,KvInsertRoute route=KvInsertRoute::PAD16_V16,int64_t physical=64) {
 return rpu_resolve_kvinsert_segment_plan(p,rows,physical,8,8,128,
  KV_INSERT_CAP_V2|KV_INSERT_CAP_V16|KV_INSERT_CAP_PAD16,route);
}
'''


def test_actual_launcher_typed_registers_ranges_and_qwen_boundary(tmp_path):
    launch = block(LAUNCHER, 'void rpu_launch_hyvla_expert_paired_kv_spm_kernel(')
    qwen_guard = LAUNCHER.split('void launch_owned_paired_kv_insert(', 1)[1]
    qwen_guard = qwen_guard[qwen_guard.index('    TORCH_CHECK('):qwen_guard.index('    const KernelId kernel_id =')]
    source = COMMON + r'''
enum class KernelId { QWEN3VL_INSERT_KV_CACHE_V16 };
enum class GraphDdrRegisterRole { KeyCache,ValueCache };
enum class GraphDdrRegisterAccess { Write };
enum class GraphDdrRegisterEncoding { DevAddrShift8LoHi };
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
 GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel {std::array<uint16_t,68> regs{};int writes=0;bool reject=false;
 void reset_regs(){regs.fill(0);writes=0;}
 uint32_t set_regs(uint32_t i,uint16_t x){if(reject)return 3;regs.at(i)=x;++writes;return 0;}
 uint32_t set_regs(uint32_t,int32_t)=delete;
} kernel;
int staged=0,queued=0;
struct Writer {std::vector<GraphDdrRegisterOperandSpec> operands;
 void write(Kernel& k){for(const auto& o:operands){auto a=o.tensor.address>>8;
  assert(k.set_regs(o.abi.lo,uint16_t(a))==0);assert(k.set_regs(o.abi.hi,uint16_t(a>>16))==0);}}
};
struct RpuKernelGraph {
 static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId,std::vector<GraphDdrRegisterOperandSpec> v){
  ++staged;assert(v.size()==2);assert(v[0].abi.role==GraphDdrRegisterRole::KeyCache);
  assert(v[1].abi.role==GraphDdrRegisterRole::ValueCache);
  assert(v[0].abi.lo==8 && v[0].abi.hi==9 && v[1].abi.lo==12 && v[1].abi.hi==13);
  assert(v[0].abi.access==GraphDdrRegisterAccess::Write && v[1].abi.access==GraphDdrRegisterAccess::Write);
  return {v};}
};
struct Queue {bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && grid==std::vector<uint16_t>({4,1,1}));
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
} queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
void rpu_ddr_flush(void*){}
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;
 uint32_t base=0x20000000;bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t off)const{return base+uint32_t(c)*0x1000000u+off;}
} spm;
#define SPM_ALLOC spm
LAUNCH
void qwen_admit(at::Tensor k_cache,at::Tensor v_cache,int64_t position,int64_t seq_len,
 int64_t num_kv_heads=8,int64_t head_dim=128,int num_cores=8,bool decode=false) { const bool lingbot_m224=false; QWEN_GUARD }
int main(){
 at::Tensor k,v;v.address=0x18000000;
 auto run=[&](at::Tensor key,at::Tensor value,uint32_t ko,uint32_t vo,const KvInsertSegmentPlan& plan,
              bool expected,int rows=64){
  const int before=staged,old_queued=queued;bool accepted=true;
  try{rpu_launch_hyvla_expert_paired_kv_spm_kernel(key,value,ko,vo,rows,plan);}
  catch(const std::exception&){accepted=false;}
  assert(accepted==expected);assert(staged==before+int(expected));assert(queued==old_queued+int(expected));
 };
 for(int p:{128,144,160,176,192,208,224,240}){
  const auto plan=exact_plan(p);k.shape[1]=v.shape[1]=(p+64)/16;
  run(k,v,0,16384,plan,true);
  assert(kernel.regs[0]==p && kernel.regs[1]==0 && kernel.regs[2]==1 && kernel.regs[3]==128);
  assert(kernel.regs[6]==0 && kernel.regs[7]==0x2000);
  assert(kernel.regs[10]==16384 && kernel.regs[11]==0x2000 && kernel.regs[21]==8);
  assert(kernel.regs[8]==0 && kernel.regs[9]==0x10 && kernel.regs[12]==0 && kernel.regs[13]==0x18);
  auto bad=k;--bad.shape[1];run(bad,v,0,16384,plan,false);
  auto other=v;--other.shape[1];run(k,other,0,16384,plan,false);
  run(bad,other,0,16384,plan,false);
  other=v;++other.shape[1];run(k,other,0,16384,plan,false);
 }
 const auto plan=exact_plan();
 run(k,v,0,16384,exact_plan(240,51,KvInsertRoute::V2,51),false);
 for(int p:{0,112,256,272})run(k,v,0,16384,exact_plan(p),false);
 for(int p:{127,129}){bool rejected=false;try{exact_plan(p);}catch(const std::exception&){rejected=true;}assert(rejected);}
 run(k,v,0,16384,exact_plan(240,64,KvInsertRoute::ALIGNED_V16,64),false);
 run(k,v,0,16384,plan,false,31);run(k,v,1,16384,plan,false);
 run(k,v,0,16352,plan,false);run(k,v,16384,0,plan,true);
 run(k,v,SpmAllocator::SPM_USABLE-16384,0,plan,true);
 run(k,v,SpmAllocator::SPM_USABLE-16384+32,0,plan,false);
 run(k,v,UINT32_MAX-31,0,plan,false);
 spm.ready=false;run(k,v,0,16384,plan,false);spm.ready=true;
 spm.base=0xf8fff000;run(k,v,0,16384,plan,false);spm.base=0x20000000;
 for(int i:{0,2,3,4,5,6}){auto bad=k;++bad.shape[i];run(bad,v,0,16384,plan,false);}
 {auto bad=k;bad.shape.pop_back();run(bad,v,0,16384,plan,false);}
 {auto bad=k;bad.contiguous=false;run(bad,v,0,16384,plan,false);}
 {auto bad=k;bad.present=false;run(bad,v,0,16384,plan,false);}
 {auto bad=k;bad.dtype=at::kFloat;run(bad,v,0,16384,plan,false);}
 {auto bad=k;bad.where.kind=c10::DeviceType::CPU;run(bad,v,0,16384,plan,false);}
 {auto bad=v;bad.where.index=1;run(k,bad,0,16384,plan,false);}
 {auto bad=v;bad.address=k.address+256;run(k,bad,0,16384,plan,false);}
 {auto bad=v;bad.address=k.address+k.nbytes();run(k,bad,0,16384,plan,true);}
 {auto bad=k;bad.address=v.address+256;run(bad,v,0,16384,plan,false);}
 kernel.reject=true;bool rejected=false;const auto old=queued;
 try{rpu_launch_hyvla_expert_paired_kv_spm_kernel(k,v,0,16384,64,plan);}catch(const std::exception&){rejected=true;}
 assert(rejected && old==queued);kernel.reject=false;
 k.shape[1]=v.shape[1]=38;qwen_admit(k,v,384,192);
 rejected=false;try{qwen_admit(k,v,384,32);}catch(const std::exception&){rejected=true;}assert(rejected);
}
'''
    compile_run(tmp_path, 'paired_launch', source.replace('LAUNCH', launch).replace('QWEN_GUARD', qwen_guard))



def test_actual_owner_admission_old_ref_and_geometry(tmp_path):
    enabled = block(DECODER, 'bool HyVlaExpertModel::optimized_prefix_supported(').replace('HyVlaExpertModel::', '') + '\n' + block(DECODER, 'bool HyVlaExpertModel::paired_kv_enabled(').replace('HyVlaExpertModel::', '')
    arguments = block(DECODER, 'std::vector<int64_t> HyVlaExpertModel::paired_kv_arguments(').replace('HyVlaExpertModel::', '')
    source = COMMON + r"""
struct Model {
 struct LayerW {at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w;
  at::Tensor q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws;};
 bool cold_w8_optimizations_available_=false;
 bool nvfp4_=false,cold_nvfp4_gateup_available_=false;
 bool cold_paired_kv_available_=true,cold_kvpad16_=true;
 int64_t expert_chunk_size_=51;
 int l=32,h=1024,i=2048,q=16,kv=8,d=128,t=8;
 int num_layers()const{return l;} int hidden_size()const{return h;}
 int intermediate_size()const{return i;} int num_q_heads()const{return q;}
 int num_kv_heads()const{return kv;} int head_dim()const{return d;} int attn_tp()const{return t;}
 std::vector<LayerW> layers_{32};
 Model(){for(auto& w:layers_)for(auto* s:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws})s->present=false;}
 ENABLED
 ARGUMENTS
};
int main(){
 Model m;const auto plan=exact_plan();
 assert(m.paired_kv_enabled(plan));
 assert(m.paired_kv_arguments(plan)==std::vector<int64_t>({1,51,64,240,8,8,128}));
 m.cold_paired_kv_available_=false;assert(!m.paired_kv_enabled(plan));m.cold_paired_kv_available_=true;
 m.cold_kvpad16_=false;assert(!m.paired_kv_enabled(plan));m.cold_kvpad16_=true;
 for(int* x:{&m.l,&m.h,&m.i,&m.q,&m.kv,&m.d,&m.t}){++*x;assert(!m.paired_kv_enabled(plan));--*x;}
 m.expert_chunk_size_=50;assert(!m.paired_kv_enabled(plan));m.expert_chunk_size_=51;
 for(int p:{128,144,160,176,192,208,224,240})assert(m.paired_kv_enabled(exact_plan(p)));
 assert(!m.paired_kv_enabled(exact_plan(240,51,KvInsertRoute::V2,51)));
 assert(!m.paired_kv_enabled(exact_plan(240,64,KvInsertRoute::ALIGNED_V16,64)));
 m.layers_.back().q_w.dtype=at::kChar;assert(!m.paired_kv_enabled(plan));m.layers_.back().q_w.dtype=at::kHalf;
 m.layers_.back().down_ws.present=true;assert(!m.paired_kv_enabled(plan));m.layers_.back().down_ws.present=false;
 m.layers_.pop_back();assert(!m.paired_kv_enabled(plan));
 Model nv;nv.nvfp4_=true;
 for(auto& w:nv.layers_){
  for(auto* t:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w}){t->dtype=at::kByte;t->shape={1024,512};}
  for(auto* t:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws}){t->present=true;t->dtype=at::kByte;t->shape={64,1024};}
 }
 assert(!nv.paired_kv_enabled(plan));nv.cold_nvfp4_gateup_available_=true;
 assert(nv.paired_kv_enabled(plan) && nv.paired_kv_arguments(plan)==m.paired_kv_arguments(plan));
 for(int p:{128,144,160,176,192,208,224,240}){
  const auto pplan=exact_plan(p);
  assert(nv.paired_kv_enabled(pplan));
  assert(nv.paired_kv_arguments(pplan)==std::vector<int64_t>({1,51,64,p,8,8,128}));
 }
 for(int p:{0,112,256,272})assert(!nv.paired_kv_enabled(exact_plan(p)));
 nv.nvfp4_=false;assert(!nv.paired_kv_enabled(plan));nv.nvfp4_=true;
 nv.layers_.back().down_ws.dtype=at::kHalf;assert(!nv.paired_kv_enabled(plan));
 nv.layers_.back().down_ws.dtype=at::kByte;assert(nv.paired_kv_enabled(plan));
 nv.cold_paired_kv_available_=false;assert(!nv.paired_kv_enabled(plan));
}
"""
    compile_run(tmp_path, 'paired_owner', source.replace('ENABLED', enabled).replace('ARGUMENTS', arguments))

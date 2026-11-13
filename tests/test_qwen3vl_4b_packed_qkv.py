"""Host execution of production QKV storage/scale/admission and SPM planning."""
from pathlib import Path
import shutil
import subprocess

import pytest
import torch
from torch.utils.cpp_extension import include_paths, library_paths

ROOT = Path(__file__).resolve().parents[1]


def method(source, marker):
    start = source.index(marker)
    begin = source.index("{", start)
    depth, end = 1, begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_native_packed_qkv_storage_scales_admission_and_layout(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required for native host checks")
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    methods = "\n".join(method(source, marker) for marker in (
        "static bool qwen3vl_4b_qkv_storage_contiguous(",
        "static void copy_qwen3vl_4b_qkv_scales(",
        "bool qwen3vl_decode_gemv(",
        "bool qwen3vl_4b_packed_decode_qkv("))
    allocator = method((ROOT / "src/core/rpu_spm_allocator.cpp").read_text(),
                       "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    begin = source.index("        const int q_phase_end = 5;")
    end = source.index('        decls.push_back({"output"', begin)
    declarations = source[begin:end]
    cpp = r'''
#include <ATen/ATen.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include "rpu_spm_allocator.h"
constexpr int NUM_CORES=8;
enum class CausalDecoderPlanningMode { ORDINARY, OTHER };
enum class StorageClass { Temp };
enum class BufferScope { LayerWide, Compute, KvInsert };
struct BufferDecl {
 const char* name; int64_t size; int phase_start,phase_end;
 StorageClass storage; int per_layer; const char* alias_of; BufferScope scope;
};
struct Layout { int64_t chunk_size=1,batch_size=1; bool is_causal=true,use_attn_mask=false; };
struct Decoder {
 bool qwen3vl_fp16_decode_profile_=false;
 bool qwen3vl_4b_packed_qkv_weights_=true,qwen3vl_4b_w8_profile_=true;
 bool linear_acc32_=false,decoder_mixed_precision_=false,adarms_=false;
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
''' + methods + r'''
 std::vector<BufferDecl> declarations(int64_t chunk) const {
  Layout ctx;ctx.chunk_size=chunk;
  bool wall_p640_staged_layout=false,spm_kv_by_mha=false,kv_first_layout=false;
  int64_t q=chunk*1024,kv=chunk*256,q_kv=q;
  BufferScope comp_scope=BufferScope::LayerWide,attn_kv_scope=BufferScope::LayerWide;
  std::vector<BufferDecl> decls;
''' + declarations + r'''
  return decls;
 }
};
''' + allocator + r'''
int main() {
 Decoder d;
 // Exact shapes at a large real storage offset; meta owns no backing GBs.
 const int64_t off=(int64_t{1}<<31)+256,total=6144*2560;
 auto bank=at::empty({off+total},at::TensorOptions().dtype(at::kChar).device(at::kMeta));
 auto q=bank.narrow(0,off,4096*2560).view({4096,2560});
 auto k=bank.narrow(0,off+q.numel(),1024*2560).view({1024,2560});
 auto v=bank.narrow(0,off+q.numel()+k.numel(),1024*2560).view({1024,2560});
 assert(d.qwen3vl_4b_qkv_storage_contiguous(q,k,v));
 auto packed=q.as_strided({6144,2560},{2560,1},q.storage_offset());
 assert(packed.storage_offset()==off && packed.numel()==total);
 assert(packed.storage().unsafeGetStorageImpl()==q.storage().unsafeGetStorageImpl());
 assert(!d.qwen3vl_4b_qkv_storage_contiguous(q,v,k));
 assert(!d.qwen3vl_4b_qkv_storage_contiguous(q,k,v.clone()));
 assert(!d.qwen3vl_4b_qkv_storage_contiguous(q,k,k));
 assert(!d.qwen3vl_4b_qkv_storage_contiguous(q.transpose(0,1),k,v));
 auto gap=at::empty({total+512},at::TensorOptions().dtype(at::kChar).device(at::kMeta));
 auto gq=gap.narrow(0,0,q.numel()).view(q.sizes());
 auto gk=gap.narrow(0,q.numel()+256,k.numel()).view(k.sizes());
 auto gv=gap.narrow(0,q.numel()+k.numel()+256,v.numel()).view(v.sizes());
 assert(!d.qwen3vl_4b_qkv_storage_contiguous(gq,gk,gv));
 // All 36 scale views share one small CPU bank and preserve per-core Q/K/V.
 auto scales=at::empty({36,6144},at::TensorOptions().dtype(at::kHalf));
 auto qs=at::arange(4096,at::kFloat).div(4096).to(at::kHalf);
 auto ks=at::arange(1024,at::kFloat).div(1024).add(2).to(at::kHalf);
 auto vs=at::arange(1024,at::kFloat).div(1024).add(4).to(at::kHalf);
 auto expected=at::cat({qs.view({8,512}),ks.view({8,128}),vs.view({8,128})},1).flatten();
 for(int i=0;i<36;++i) {
  auto s=scales.select(0,i); d.copy_qwen3vl_4b_qkv_scales(s,qs,ks,vs);
  assert(s.equal(expected) && s.storage_offset()==i*6144);
 }
 assert(d.qwen3vl_4b_packed_decode_qkv(1,true,false,1));
 assert(!d.qwen3vl_4b_packed_decode_qkv(2,true,false,1));
 assert(!d.qwen3vl_4b_packed_decode_qkv(1,false,false,1));
 assert(!d.qwen3vl_4b_packed_decode_qkv(1,true,true,1));
 assert(!d.qwen3vl_4b_packed_decode_qkv(1,true,false,2));
 for(int fault=0;fault<6;++fault) {
  Decoder bad;
  if(fault==0)bad.qwen3vl_4b_packed_qkv_weights_=false;
  if(fault==1)bad.qwen3vl_4b_w8_profile_=false;
  if(fault==2)bad.linear_acc32_=true;
  if(fault==3)bad.decoder_mixed_precision_=true;
  if(fault==4)bad.adarms_=true;
  if(fault==5)bad.planning_mode_=CausalDecoderPlanningMode::OTHER;
  assert(!bad.qwen3vl_4b_packed_decode_qkv(1,true,false,1));
 }
 // Execute actual declarations and allocator with the preceding structural slots.
 auto dec=d.declarations(1);
 std::vector<SpmAllocator::AllocRequest> reqs{{5120,1,8,0},{5120,1,8,0}};
 for(auto b:dec)reqs.push_back({b.size,b.phase_start,b.phase_end,0});
 auto plan=SpmAllocator::plan_temporary_aliased(reqs);
 assert(dec.size()==3 && dec[0].size==1024 && dec[1].size==256 && dec[2].size==256);
 assert(plan.offsets[3]==plan.offsets[2]+1024 && plan.offsets[4]==plan.offsets[3]+256);
 assert(dec[0].phase_end==5 && dec[1].phase_end==5 && dec[2].phase_end==5);
 for(int chunk:{16,96,256,512}) {
  auto enabled=d.declarations(chunk); Decoder fallback;fallback.qwen3vl_4b_packed_qkv_weights_=false;
  auto unchanged=fallback.declarations(chunk);
  for(int i=0;i<3;++i)assert(enabled[i].size==unchanged[i].size &&
   enabled[i].phase_start==unchanged[i].phase_start && enabled[i].phase_end==unchanged[i].phase_end);
 }
}
'''
    path, exe = tmp_path / "qkv.cpp", tmp_path / "qkv"
    path.write_text(cpp)
    libs = library_paths()
    command = [compiler, "-std=c++17", "-O0",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               *[f"-I{x}" for x in include_paths()], f"-I{ROOT / 'src/core'}",
               str(path), *[f"-L{x}" for x in libs], *[f"-Wl,-rpath,{x}" for x in libs],
               "-ltorch_cpu", "-lc10", "-o", str(exe)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)

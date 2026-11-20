"""Execute the actual W8A8 C++ launchers with recording host SDK adapters.

The adapters model ownership, register writes and DMA ordering, not arithmetic.
Kernel numerics and RPU Graph replay still require cmodel/device validation.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import torch

from pi_wall_host_catalog import absolute_candidate_includes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def launcher_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "W8A8 launch checks require g++"
    folder = tmp_path_factory.mktemp("w8a8_launch")
    production = (ROOT / "src/ops/rpu_linear_w8a8.cpp").read_text()
    production = production.replace('#include "rpu_ops.h"', '').replace(
        '#include "rpu_spm_allocator.h"', '')
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    enum = re.search(r"enum class KernelId[^{}]*\{[\s\S]*?\};", cache)[0]
    names = re.search(r"static constexpr const char\* KERNEL_ID_NAMES\[\] = \{[\s\S]*?\};", cache)[0]
    source = folder / "launcher.cpp"
    source.write_text(absolute_candidate_includes(r'''
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"
#include "rpu_linear_w8a8_contract.h"
#include "../fused/rpu_qwen3_w8a8_contract.h"
#include <c10/util/Half.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("rejected"); } while (0)
#define RECORD_FUNCTION(...)
__ENUM__
__NAMES__
static_assert(sizeof(KERNEL_ID_NAMES)/sizeof(KERNEL_ID_NAMES[0]) == size_t(KernelId::_COUNT));
namespace c10 {
enum class DeviceType { CPU, PrivateUse1 };
enum class Layout { Strided, Sparse };
template<class T> using optional=std::optional<T>;

}
namespace graph_dma { bool capturing=false; bool active() { return capturing; } }
struct Event {
  std::string kind;
  std::vector<uint64_t> args;
  KernelId id=KernelId::_COUNT;
  std::array<uint16_t,67> regs{};
};
std::vector<Event> events;
void record(const char* kind,std::initializer_list<uint64_t> args={}) {
  events.push_back({kind,args,KernelId::_COUNT,{}});
}
namespace at {
enum ScalarType { kHalf, kChar, kFloat };
struct Device {
  c10::DeviceType kind=c10::DeviceType::PrivateUse1;
  int index=0;
  c10::DeviceType type() const { return kind; }
  bool operator==(const Device& o) const { return kind==o.kind && index==o.index; }
};
struct Options { ScalarType dtype=kHalf; Device where{}; };
struct Sizes:std::vector<int64_t> {
  using std::vector<int64_t>::vector;
  std::vector<int64_t> vec() const { return {begin(),end()}; }
};
struct Backing { uint64_t bytes,address; };
struct Storage {
  std::shared_ptr<Backing> owner;
  explicit operator bool() const { return bool(owner); }
  uint64_t nbytes() const { return owner ? owner->bytes : 0; }
};
uint64_t next_address=0x100000000;
struct Tensor {
  Sizes dims;
  Options opt;
  std::shared_ptr<Backing> owner;
  int64_t offset=0;
  bool dense=true;
  c10::Layout storage_layout=c10::Layout::Strided;
  Tensor()=default;
  Tensor(std::initializer_list<int64_t> shape,ScalarType dtype=kHalf):dims(shape) {
    opt.dtype=dtype;
    owner=std::make_shared<Backing>(Backing{nbytes(),next_address});
    next_address+=(nbytes()+4095)/4096*4096+4096;
  }
  bool defined() const { return bool(owner); }
  Device device() const { return opt.where; }
  ScalarType scalar_type() const { return opt.dtype; }
  c10::Layout layout() const { return storage_layout; }
  bool is_contiguous() const { return dense; }
  int64_t dim() const { return dims.size(); }
  int64_t size(int64_t axis) const { return dims.at(axis<0?dims.size()+axis:axis); }
  int64_t numel() const { return std::accumulate(dims.begin(),dims.end(),int64_t{1},std::multiplies<int64_t>()); }
  uint64_t element_size() const { return opt.dtype==kChar?1:opt.dtype==kHalf?2:4; }
  uint64_t nbytes() const { return numel()*element_size(); }
  int64_t storage_offset() const { return offset; }
  Storage storage() const { return {owner}; }
  Sizes sizes() const { return dims; }
  Options options() const { return opt; }
  void* data_ptr() const { return reinterpret_cast<void*>(owner->address+offset*element_size()); }
  template<class T> T* data_ptr() const { return static_cast<T*>(data_ptr()); }
};
Tensor empty(const std::vector<int64_t>& dims,Options options) {
  Tensor t; t.dims.assign(dims.begin(),dims.end()); t.opt=options;
  t.owner=std::make_shared<Backing>(Backing{t.nbytes(),next_address});
  next_address+=(t.nbytes()+4095)/4096*4096+4096;
  record("empty",{t.owner->address,t.nbytes()}); return t;
}
}
uint64_t RpuGetDevAddr(const void* p) { return reinterpret_cast<uintptr_t>(p); }
void rpu_ddr_flush(const void* p) { record("flush",{RpuGetDevAddr(p)}); }
template<class T> T Align(T n,T a) { return (n+a-1)/a*a; }
class SpmAllocator {
 public:
  static constexpr size_t SPM_USABLE=8*1024*1024-1024, ALIGN=256;
  static constexpr size_t SPM_PLANNING_BUDGET=SPM_USABLE*99/100;
  static constexpr uint32_t base=0x10000000;
  size_t cursor=0;
  bool initialized=true;
  static SpmAllocator& instance() { static SpmAllocator value; return value; }
  bool is_initialized() const { return initialized; }
  void init() { initialized=true; record("init"); }
  uint32_t addr(int core,uint32_t offset) const { assert(core==0); return base+offset; }
  size_t total_used() const { return cursor; }
  size_t free_space() const { return SPM_USABLE-cursor; }
  uint32_t alloc_temporary(size_t bytes) {
    auto old=cursor; cursor+=Align(bytes,ALIGN);
    assert(cursor<=SPM_USABLE); record("alloc",{old,bytes}); return old;
  }
  struct ScopedTemporary {
    size_t mark=instance().cursor;
    ~ScopedTemporary() { instance().cursor=mark; }
  };
};
#define SPM_ALLOC SpmAllocator::instance()
constexpr int kKernelRegOk=0;
namespace rhino_lkn { using ::kKernelRegOk; using ::RpuGetDevAddr; }
struct Kernel_t {
  KernelId id=KernelId::_COUNT;
  std::array<uint16_t,67> regs{};
  void reset_regs() { regs.fill(0); record("reset"); }
  int set_regs(uint32_t slot,uint16_t v) { assert(slot<regs.size()); regs[slot]=v; return 0; }
};
enum class GraphDdrRegisterRole { Int8ModelWeight, PerChannelScale };
enum class GraphDdrRegisterAccess { Read };
enum class GraphDdrRegisterEncoding { DevAddrShift8LoHi };
struct GraphDdrRegisterOperandSpec {
  struct Contract { GraphDdrRegisterRole role; GraphDdrRegisterAccess access;
    GraphDdrRegisterEncoding encoding; int lo,hi; } contract;
  at::Tensor tensor;
};
struct Writer {
  std::vector<GraphDdrRegisterOperandSpec> specs;
  void write(Kernel_t& kernel) {
    for (auto& s:specs) {
      assert(s.contract.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
      assert(s.contract.access==GraphDdrRegisterAccess::Read);
      auto address=RpuGetDevAddr(s.tensor.data_ptr())>>8;
      kernel.set_regs(s.contract.lo,address&0xffff);
      kernel.set_regs(s.contract.hi,address>>16);
    }
    record("typed_write");
  }
};
struct RpuKernelGraph {
  std::vector<at::Tensor> owners;
  std::vector<GraphDdrRegisterOperandSpec> last_specs;
  bool pending=false;
  static bool has_active() { return graph_dma::capturing; }
  static RpuKernelGraph& active() { static RpuKernelGraph value; return value; }
  Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> specs) {
    assert(!pending); pending=true;
    assert(specs.size()==2 && specs[0].contract.lo==2 && specs[0].contract.hi==3);
    assert(specs[0].contract.role==GraphDdrRegisterRole::Int8ModelWeight);
    assert(specs[1].contract.lo==20 && specs[1].contract.hi==21);
    assert(specs[1].contract.role==GraphDdrRegisterRole::PerChannelScale);
    last_specs=specs; record("stage",{uint64_t(id)}); return {std::move(specs)};
  }
  void keep_alive(const at::Tensor& t) { owners.push_back(t); record("keep",{RpuGetDevAddr(t.data_ptr())}); }
};
struct KernelCache {
  std::set<KernelId> missing;
  static KernelCache& instance() { static KernelCache value; return value; }
  bool has_loaded(KernelId id) const { return !missing.count(id); }
};
Kernel_t kernel;
Kernel_t* get_kernel(KernelId id) {
  auto& g=RpuKernelGraph::active();
  if (id==KernelId::DYNAMIC_QUANTIZE_W8A8 || id==KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M128 || id==KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M192 || id==KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M256 ||
      id==KernelId::MUL_QUANTIZE_W8A8_H3200_M128 || id==KernelId::MUL_QUANTIZE_W8A8_H3200_M192 ||
      id==KernelId::LAYER_NORM_QUANTIZE_W8A8_H1152 ||
      id==KernelId::GELU_TANH_QUANTIZE_W8A8_H544) {
    assert(!g.pending); record("intrinsic_no_ddr");
  } else { assert(g.pending); g.pending=false; }
  kernel.id=id; record("get",{uint64_t(id)}); return &kernel;
}
#define GET_KERNEL(id) get_kernel(id)
struct Queue {
  int count=0; bool broadcast=false;
  void set_broadcast_mode(bool b) { broadcast=b; }
  void enqueu_kernel(Kernel_t& k,std::vector<uint16_t> grid,std::vector<uint8_t> cores) {
    assert(broadcast && int(cores.size())==count);
    for (int c=0;c<count;++c) assert(cores[c]==c);
    assert(grid.size()==3 && grid[0]==k.regs[64] && grid[1]==k.regs[65] && grid[2]==k.regs[66]);
    Event e{k.id==KernelId::DYNAMIC_QUANTIZE_W8A8?"quant":
            (k.id==KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M128||k.id==KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M192||k.id==KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M256)?"rmsquant":
            (k.id==KernelId::MUL_QUANTIZE_W8A8_H3200_M128||k.id==KernelId::MUL_QUANTIZE_W8A8_H3200_M192)?"mulquant":
            k.id==KernelId::LAYER_NORM_QUANTIZE_W8A8_H1152?"lnquant":
            k.id==KernelId::GELU_TANH_QUANTIZE_W8A8_H544?"geluquant":"linear",{uint64_t(count)},k.id,k.regs};
    events.push_back(std::move(e));
  }
} queue;
Queue* get_queue(int count) { assert(count>=1 && count<=8); queue.count=count; queue.broadcast=false; return &queue; }
#define GET_QUEUE(count) get_queue(count)
void rpu_launch_fill_spm_kernel(uint32_t address,int64_t count,c10::Half value,int cores) {
  assert(value.x==0); record("fill",{address,uint64_t(count),uint64_t(cores)});
}
void rpu_launch_ddr_broadcast_spm_dma(const at::Tensor& t,int64_t first,int64_t count,uint32_t to,int cores) {
  assert(first>=0 && count>0 && first+count<=t.numel());
  record("broadcast",{RpuGetDevAddr(t.data_ptr()),uint64_t(first),uint64_t(count),to,uint64_t(cores)});
}
void rpu_launch_ddr_scatter_spm_dma(const at::Tensor& t,int64_t first,int64_t count,int64_t stride,uint32_t to,int cores) {
  assert(first>=0 && count>0 && uint64_t(2*(first+count)+(cores-1)*stride)<=t.nbytes());
  record("scatter",{RpuGetDevAddr(t.data_ptr()),uint64_t(first),uint64_t(count),uint64_t(stride),to,uint64_t(cores)});
}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t partial,uint32_t zero,uint32_t out,int64_t m,int64_t n,int producers,int consumers) {
  assert(partial!=zero && partial!=out && zero!=out && producers==consumers);
  record("ring",{partial,zero,out,uint64_t(m),uint64_t(n),uint64_t(producers)});
}
void rpu_launch_spm_copy_ddr_dma(uint32_t from,c10::Half* out,int64_t count) {
  record("copy_out",{from,RpuGetDevAddr(out),uint64_t(count)});
}
struct ScopedEagerAtomicBatch {
  bool enabled_ = false;
  ScopedEagerAtomicBatch(bool enabled, int) : enabled_(enabled) {}
  bool active() const noexcept { return enabled_; }
  size_t recorded_node_count() const noexcept { return 0; }
  static size_t default_flush_node_threshold() noexcept { return 45000; }
  void flush_and_reopen() {}
  void submit() {}
};
void rpu_launch_spm_scatter_rows_ddr_dma(uint32_t from,c10::Half* out,int64_t rows,int64_t nc,int64_t spm_stride,int64_t ddr_stride,int64_t core_stride,int cores) {
  assert(spm_stride==nc*2 && core_stride==nc*2 && ddr_stride==nc*2*cores);
  record("scatter_out",{from,RpuGetDevAddr(out),uint64_t(rows),uint64_t(nc),uint64_t(cores)});
}
__PRODUCTION__
void clear() {
  events.clear(); RpuKernelGraph::active().owners.clear();
  RpuKernelGraph::active().last_specs.clear(); RpuKernelGraph::active().pending=false;
  KernelCache::instance().missing.clear(); SPM_ALLOC.cursor=0;
  SPM_ALLOC.initialized=true; graph_dma::capturing=false;
}
uint32_t reg32(const Event& e,int p) { return uint32_t(e.regs[p])|uint32_t(e.regs[p+1])<<16; }
std::vector<Event> actions() {
  std::vector<Event> out;
  for(auto e:events) if(e.kind=="broadcast" || e.kind=="scatter" || e.kind=="quant" ||
      e.kind=="linear" || e.kind=="mulquant" || e.kind=="rmsquant" || e.kind=="lnquant" || e.kind=="geluquant" || e.kind=="ring" || e.kind=="fill" ||
      e.kind=="copy_out" || e.kind=="scatter_out") out.push_back(e);
  return out;
}
template<class F> void reject_clean(F f) {
  events.clear(); bool rejected=false;
  try { f(); } catch(const std::exception&) { rejected=true; }
  assert(rejected && events.empty());
  assert(RpuKernelGraph::active().owners.empty() && !RpuKernelGraph::active().pending);
}
void launchers() {
  for(int cores=1;cores<=8;++cores) for(int part:{0,1}) {
    clear();
    const int n=part?80*cores:80, k=part?96:96*cores;
    at::Tensor w({n,k},at::kChar),sw({n});
    const uint32_t x=SpmAllocator::base, sx=x+0x10000,y=x+0x20000,b=x+0x40000;
    graph_dma::capturing=true;
    rpu_launch_linear_w8a8_spm_kernel(x,sx,w,sw,y,17,n,k,part,cores,b);
    auto a=actions(); assert(a.size()==1 && a[0].kind=="linear" && a[0].args[0]==uint64_t(cores));
    const auto& e=a[0];
    assert(reg32(e,0)==x && reg32(e,30)==sx && reg32(e,4)==y && reg32(e,6)==b);
    assert(reg32(e,2)==RpuGetDevAddr(w.data_ptr())>>8);
    assert(reg32(e,20)==RpuGetDevAddr(sw.data_ptr())>>8);
    assert(e.regs[8]==part && e.regs[9]==17 && e.regs[12]==96 && e.regs[15]==cores);
    assert(reg32(e,16)==96 && reg32(e,18)==160 && e.regs[26]==2);
    const auto& g=RpuKernelGraph::active();
    assert(g.owners.size()==2 && g.owners[0].owner==w.owner && g.owners[1].owner==sw.owner);
    assert(g.last_specs.size()==2 && g.last_specs[0].tensor.owner==w.owner);
    assert(std::string(KERNEL_ID_NAMES[size_t(e.id)])=="parallel_linear_w8a8_acc32_m128n64k128");
    clear(); graph_dma::capturing=true;
    rpu_launch_dynamic_quantize_w8a8_spm_kernel(x,y,sx,17,96,cores);
    a=actions(); assert(a.size()==1 && a[0].kind=="quant");
    const auto& q=a[0];
    assert(reg32(q,0)==x && reg32(q,2)==sx && reg32(q,4)==y);
    assert(q.regs[6]==17 && q.regs[9]==96 && q.regs[10]==c10::Half(127).x);
    assert(q.regs[11]==c10::Half(-127).x && reg32(q,14)==192);
    assert(q.regs[10]==v3::qwen3_w8a8::kQuantClampMaxBits);
    assert(q.regs[11]==v3::qwen3_w8a8::kQuantClampMinBits);
    assert(reg32(q,16)==2 && reg32(q,18)==96 && q.regs[20]==1);
    assert(q.regs[64]==1 && q.regs[65]==2 && q.regs[66]==1);
    assert(std::string(KERNEL_ID_NAMES[size_t(q.id)])=="dynamic_quantize_w8a8");
  }
}
void rejection() {
  clear(); const uint32_t x=SpmAllocator::base,sx=x+0x10000,y=x+0x20000;
  at::Tensor w({320,96},at::kChar),sw({320});
  auto call=[&](const at::Tensor& wt,const at::Tensor& sc,uint32_t q=SpmAllocator::base,uint32_t ss=SpmAllocator::base+0x10000) {
    rpu_launch_linear_w8a8_spm_kernel(q,ss,wt,sc,y,17,320,96,1,4,0);
  };
  for (int which=0;which<9;++which) {
    auto bad=w; bad.owner=std::make_shared<at::Backing>(*w.owner);
    if(which==0) bad.opt.dtype=at::kHalf;
    if(which==1) bad.opt.where.kind=c10::DeviceType::CPU;
    if(which==2) bad.dense=false;
    if(which==3) bad.offset=-1;
    if(which==4) bad.owner->bytes--;
    if(which==5) bad.owner->address++;
    if(which==6) bad.owner->address=1ULL<<40;
    if(which==7) bad.storage_layout=c10::Layout::Sparse;
    if(which==8) bad.dims={96,320};
    reject_clean([&]{ call(bad,sw); });
  }
  auto bad=sw; bad.owner=w.owner;
  reject_clean([&]{ call(w,bad); });
  bad=sw; bad.opt.where.index=1;
  reject_clean([&]{ call(w,bad); });
  reject_clean([&]{ call(w,sw,x,x); });
  reject_clean([&]{ call(w,sw,x+1,sx); });
  KernelCache::instance().missing.insert(KernelId::PARALLEL_LINEAR_W8A8_ACC32_M128N64K128);
  reject_clean([&]{ call(w,sw); });
  KernelCache::instance().missing.clear();
  SPM_ALLOC.initialized=false;
  reject_clean([&]{ call(w,sw); });
  SPM_ALLOC.initialized=true;
  reject_clean([&]{ rpu_launch_dynamic_quantize_w8a8_spm_kernel(x,x,sx,17,96,4); });
  reject_clean([&]{ rpu_launch_dynamic_quantize_w8a8_spm_kernel(x,y,sx,17,95,4); });
  reject_clean([&]{ rpu_launch_dynamic_quantize_w8a8_spm_kernel(x,y,sx,17,96,9); });
  KernelCache::instance().missing.insert(KernelId::DYNAMIC_QUANTIZE_W8A8);
  reject_clean([&]{ rpu_launch_dynamic_quantize_w8a8_spm_kernel(x,y,sx,17,96,4); });
}
void rms_quantization() {
  const uint32_t x=SpmAllocator::base,q=x+0x300000,sx=x+0x500000,gamma=x+0x510000;
  for(int rows:{128,192,256}){
  clear(); graph_dma::capturing=true;
  rpu_launch_rms_norm_quantize_w8a8_spm_kernel(x,q,sx,gamma,rows,1e-6,8);
  const auto a=actions(); assert(a.size()==1 && a[0].kind=="rmsquant");
  const auto& e=a[0];
  std::array<uint16_t,67> expected{};
  const auto pair=[&](int at,uint32_t value){expected[at]=uint16_t(value);expected[at+1]=uint16_t(value>>16);};
  pair(0,x);pair(2,sx);pair(4,q);expected[6]=rows;expected[7]=expected[8]=1;expected[9]=5120;
  expected[10]=c10::Half(127.0f).x;expected[11]=c10::Half(-127.0f).x;
  const auto route=v3::qwen3_w8a8::quant_arguments(rows,0);
  assert(route[7]==expected[10]&&route[8]==expected[11]);
  pair(14,10240);pair(16,2);pair(18,5120);expected[20]=1;pair(22,gamma);
  expected[24]=0x11;expected[25]=0xa66;expected[64]=8;expected[65]=expected[66]=1;
  assert(e.regs==expected && e.args[0]==8);
  assert(std::string(KERNEL_ID_NAMES[size_t(e.id)])==(rows==128?"rms_norm_quantize_w8a8_h5120_m128":rows==192?"rms_norm_quantize_w8a8_h5120_m192":"rms_norm_quantize_w8a8_h5120_m256"));
  assert(RpuKernelGraph::active().owners.empty() && !RpuKernelGraph::active().pending);
  clear();
  const auto call=[&](int m=0,double eps=1e-6,int cores=8){
    rpu_launch_rms_norm_quantize_w8a8_spm_kernel(x,q,sx,gamma,m==0?rows:m,eps,cores);
  };
  for(int m:{-1,1,16,32,64,127,129,191,193,255,257}) reject_clean([&]{call(m);});
  for(int cores:{0,1,4,6,7,9}) reject_clean([&]{call(rows,1e-6,cores);});
  for(double eps:{0.0,-1.0,1e-5,std::nextafter(1e-6,1.0),INFINITY*1.0,NAN*1.0})
    reject_clean([&]{call(rows,eps);});
  const std::array<uint32_t,4> good{x,q,sx,gamma};
  for(size_t i=0;i<good.size();++i){
    for(uint32_t invalid:{good[i]+1,SpmAllocator::base-32,
                         SpmAllocator::base+uint32_t(SpmAllocator::SPM_USABLE)-32}){
      auto bad=good;bad[i]=invalid;
      reject_clean([&]{rpu_launch_rms_norm_quantize_w8a8_spm_kernel(bad[0],bad[1],bad[2],bad[3],rows,1e-6,8);});
    }
    for(size_t j=0;j<i;++j)for(uint32_t shift:{0u,32u}){
      auto bad=good;bad[i]=bad[j]+shift;
      reject_clean([&]{rpu_launch_rms_norm_quantize_w8a8_spm_kernel(bad[0],bad[1],bad[2],bad[3],rows,1e-6,8);});
    }
  }
  SPM_ALLOC.initialized=false;reject_clean([&]{call();});SPM_ALLOC.initialized=true;
  KernelCache::instance().missing.insert(rows==128?KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M128:rows==192?KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M192:KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M256);
  reject_clean([&]{call();});
  KernelCache::instance().missing.clear();
  }
}
void qwen_producers() {
  const uint32_t x=SpmAllocator::base,gate=x+0x300000,up=x+0x500000,sx=x+0x700000;
  at::Tensor w({25600,5120},at::kChar),sw({25600});
  for(const auto& mode:std::array<std::pair<bool,int>,3>{{{false,256},{true,256},{true,512}}}){
    const bool a8=mode.first;const int m=mode.second;
    clear();graph_dma::capturing=true;
    const auto call=[&](uint32_t in,uint32_t out,int rows=256){
      if(a8) rpu_launch_qwen3_up_mul_w8a8_spm_kernel(in,sx,w,sw,out,rows);
      else rpu_launch_qwen3_up_mul_w8a16_spm_kernel(in,w,sw,out,rows);
    };
    call(x,gate,m);
    auto a=actions();assert(a.size()==1&&a[0].kind=="linear");const auto&e=a[0];
    assert(e.id==(a8?(m==512?KernelId::QWEN3_UP_MUL_W8A8_M512N64K128:KernelId::QWEN3_UP_MUL_W8A8_M272N64K384):KernelId::QWEN3_UP_MUL_W8A16_M320N112K128));
    assert(reg32(e,0)==x&&reg32(e,4)==gate&&reg32(e,16)==(a8?5120:10240));
    assert(reg32(e,18)==6400&&e.regs[9]==m&&reg32(e,10)==3200&&e.regs[12]==5120);
    assert(e.regs[8]==1&&e.regs[14]==0&&e.regs[15]==8&&e.regs[26]==2&&e.regs[27]==1);
    assert(e.regs[64]==(a8?50:29)&&e.regs[65]==1&&e.args[0]==8);
    assert(reg32(e,2)==RpuGetDevAddr(w.data_ptr())/256&&reg32(e,20)==RpuGetDevAddr(sw.data_ptr())/256);
    assert(reg32(e,30)==(a8?sx:0)&&RpuKernelGraph::active().owners.size()==2);
    clear();
    if(!a8)reject_clean([&]{call(x,gate,512);});
    for(int rows:{1,128,192,255,257,511,513})reject_clean([&]{call(x,gate,rows);});
    reject_clean([&]{call(x,x);});reject_clean([&]{call(x+1,gate);});
    reject_clean([&]{call(x,SpmAllocator::base+SpmAllocator::SPM_USABLE-32);});
    KernelCache::instance().missing.insert(e.id);reject_clean([&]{call(x,gate,m);});
  }
  for(int rows:{128,192}){
    clear();graph_dma::capturing=true;
    rpu_launch_mul_quantize_w8a8_spm_kernel(gate,up,x,sx,rows,3200,8);
    auto a=actions();assert(a.size()==1&&a[0].kind=="mulquant");const auto&e=a[0];
    assert(reg32(e,0)==gate&&reg32(e,22)==up&&reg32(e,4)==x&&reg32(e,2)==sx);
    assert(e.regs[6]==rows&&e.regs[9]==3200&&reg32(e,14)==6400&&reg32(e,18)==3200);
    assert(e.regs[10]==c10::Half(127.0f).x&&e.regs[11]==c10::Half(-127.0f).x);
    assert(e.regs[64]==8&&e.regs[65]==1&&e.regs[66]==1&&e.args[0]==8);
    assert(RpuKernelGraph::active().owners.empty());clear();
    for(int m:{1,64,127,129,191,193,256})reject_clean([&]{rpu_launch_mul_quantize_w8a8_spm_kernel(gate,up,x,sx,m,3200,8);});
    reject_clean([&]{rpu_launch_mul_quantize_w8a8_spm_kernel(gate,gate,x,sx,rows,3200,8);});
    reject_clean([&]{rpu_launch_mul_quantize_w8a8_spm_kernel(gate,up,gate,sx,rows,3200,8);});
    reject_clean([&]{rpu_launch_mul_quantize_w8a8_spm_kernel(gate,up,x,sx,rows,3199,8);});
    KernelCache::instance().missing.insert(e.id);
    reject_clean([&]{rpu_launch_mul_quantize_w8a8_spm_kernel(gate,up,x,sx,rows,3200,8);});
  }
}
void norm_quantization() {
  const uint32_t x=SpmAllocator::base,q=x+0x200000,sx=x+0x400000;
  const uint32_t gamma=x+0x410000,beta=x+0x420000;
  for(int cores:{1,8}) for(int m:{1,17,512,768}) for(double eps:{1e-6,1e-5}) {
    clear(); graph_dma::capturing=true;
    rpu_launch_layer_norm_quantize_w8a8_spm_kernel(x,q,sx,gamma,beta,m,eps,cores);
    const auto a=actions(); assert(a.size()==1 && a[0].kind=="lnquant");
    const auto& e=a[0];
    assert(e.args[0]==uint64_t(cores) && reg32(e,0)==x && reg32(e,2)==q);
    assert(reg32(e,4)==gamma && reg32(e,6)==beta && reg32(e,22)==sx);
    assert(e.regs[8]==m && e.regs[9]==1152);
    float reciprocal; auto bits=reg32(e,10); std::memcpy(&reciprocal,&bits,4);
    assert(reciprocal==1.0f/1152.0f && e.regs[12]==c10::Half(float(eps)*1152.0f).x);
    assert(e.regs[13]==0 && reg32(e,14)==0 && reg32(e,16)==2304);
    assert(e.regs[18]==1 && e.regs[19]==16 && e.regs[20]==c10::Half(std::sqrt(1152.0f)).x);
    assert(e.regs[21]==1 && e.regs[64]==(m+15)/16 && e.regs[65]==1 && e.regs[66]==1);
    assert(std::string(KERNEL_ID_NAMES[size_t(e.id)])=="layer_norm_quantize_w8a8_h1152");
    assert(RpuKernelGraph::active().owners.empty() && !RpuKernelGraph::active().pending);
  }
  clear();
  auto call=[&](int m=17,double eps=1e-6,int cores=8) {
    rpu_launch_layer_norm_quantize_w8a8_spm_kernel(x,q,sx,gamma,beta,m,eps,cores);
  };
  for(int m:{0,-1,769}) reject_clean([&]{ call(m); });
  for(int cores:{0,9}) reject_clean([&]{ call(17,1e-6,cores); });
  for(double eps:{0.0,-1.0,1e-20,1e20,INFINITY*1.0,NAN*1.0})
    reject_clean([&]{ call(17,eps); });
  const std::array<uint32_t,5> good{x,q,sx,gamma,beta};
  for(size_t i=0;i<good.size();++i) {
    for(uint32_t invalid:{good[i]+1,SpmAllocator::base-32,
                         SpmAllocator::base+uint32_t(SpmAllocator::SPM_USABLE)-32}) {
      auto bad=good; bad[i]=invalid;
      reject_clean([&]{ rpu_launch_layer_norm_quantize_w8a8_spm_kernel(
          bad[0],bad[1],bad[2],bad[3],bad[4],17,1e-6,8); });
    }
    for(size_t j=0;j<i;++j) {
      auto bad=good; bad[i]=bad[j];
      reject_clean([&]{ rpu_launch_layer_norm_quantize_w8a8_spm_kernel(
          bad[0],bad[1],bad[2],bad[3],bad[4],17,1e-6,8); });
    }
  }
  SPM_ALLOC.initialized=false; reject_clean([&]{ call(); });
  SPM_ALLOC.initialized=true;
  KernelCache::instance().missing.insert(KernelId::LAYER_NORM_QUANTIZE_W8A8_H1152);
  reject_clean([&]{ call(); });
}
void gelu_quantization() {
  const uint32_t x=SpmAllocator::base,q=x+0x100000,sx=x+0x200000;
  for(int cores:{1,8}) for(int m:{1,17,512,768}) {
    clear(); graph_dma::capturing=true;
    rpu_launch_gelu_tanh_quantize_w8a8_spm_kernel(x,q,sx,m,cores);
    const auto a=actions(); assert(a.size()==1 && a[0].kind=="geluquant");
    const auto& e=a[0];
    assert(e.args[0]==uint64_t(cores) && reg32(e,0)==x && reg32(e,2)==q && reg32(e,4)==sx);
    assert(e.regs[6]==m && e.regs[7]==544);
    assert(e.regs[64]==(m+15)/16 && e.regs[65]==1 && e.regs[66]==1);
    assert(RpuKernelGraph::active().owners.empty() && !RpuKernelGraph::active().pending);
  }
  clear();
  const std::array<uint32_t,3> good{x,q,sx};
  for(size_t i=0;i<good.size();++i) {
    for(uint32_t invalid:{good[i]+1,SpmAllocator::base-32,
                         SpmAllocator::base+uint32_t(SpmAllocator::SPM_USABLE)-32}) {
      auto bad=good;bad[i]=invalid;
      reject_clean([&]{rpu_launch_gelu_tanh_quantize_w8a8_spm_kernel(bad[0],bad[1],bad[2],17,8);});
    }
    for(size_t j=0;j<i;++j) {
      auto bad=good;bad[i]=bad[j];
      reject_clean([&]{rpu_launch_gelu_tanh_quantize_w8a8_spm_kernel(bad[0],bad[1],bad[2],17,8);});
    }
  }
  for(int m:{0,-1,769}) reject_clean([&]{rpu_launch_gelu_tanh_quantize_w8a8_spm_kernel(x,q,sx,m,8);});
  for(int cores:{0,9}) reject_clean([&]{rpu_launch_gelu_tanh_quantize_w8a8_spm_kernel(x,q,sx,17,cores);});
  SPM_ALLOC.initialized=false;
  reject_clean([&]{rpu_launch_gelu_tanh_quantize_w8a8_spm_kernel(x,q,sx,17,8);});
  SPM_ALLOC.initialized=true;
  KernelCache::instance().missing.insert(KernelId::GELU_TANH_QUANTIZE_W8A8_H544);
  reject_clean([&]{rpu_launch_gelu_tanh_quantize_w8a8_spm_kernel(x,q,sx,17,8);});
}
void eager() {
  for(int part:{0,1}) for(int cores:{1,2,3,4,5,6,7,8}) for(bool use_bias:{false,true}) {
    clear(); const int n=part?80*cores:80,k=part?96:96*cores;
    at::Tensor x({1,17,k}),w({n,k},at::kChar),sw({n}),bias({n});
    c10::optional<at::Tensor> b=use_bias?c10::optional<at::Tensor>(bias):std::nullopt;
    const auto out=rpu_linear_w8a8(x,w,sw,b,part,cores);
    assert((out.dims==at::Sizes{1,17,n}) && out.owner!=x.owner && out.owner!=w.owner);
    assert(SPM_ALLOC.cursor==0);
    auto a=actions(); size_t i=0;
    if(use_bias) {
      if(part==0) {
        assert(a[i].kind=="fill" && a[i].args[1]==uint64_t(n) && a[i].args[2]==uint64_t(cores)); ++i;
        assert(a[i].kind=="broadcast" && a[i].args[0]==RpuGetDevAddr(bias.data_ptr()) && a[i].args[4]==1); ++i;
      } else {
        assert(a[i].kind=="scatter" && a[i].args[0]==RpuGetDevAddr(bias.data_ptr()));
        assert(a[i].args[2]==80 && a[i].args[3]==160 && a[i].args[5]==uint64_t(cores)); ++i;
      }
    }
    if(part==0) {
      for(int r=0;r<17;++r,++i) {
        assert(a[i].kind=="scatter" && a[i].args[1]==uint64_t(r*k));
        assert(a[i].args[2]==96 && a[i].args[3]==192 && a[i].args[5]==uint64_t(cores));
        if(r) assert(a[i].args[4]==a[i-1].args[4]+192);
      }
    } else {
      assert(a[i].kind=="broadcast" && a[i].args[1]==0 && a[i].args[2]==uint64_t(17*k)); ++i;
    }
    const auto q=a[i++],linear=a[i++];
    assert(q.kind=="quant" && linear.kind=="linear");
    assert(reg32(q,4)==reg32(linear,0) && reg32(q,2)==reg32(linear,30));
    assert(q.regs[6]==17 && linear.regs[9]==17 && linear.regs[14]==use_bias);
    if(part==0) {
      assert(a[i].kind=="fill" && a[i].args[1]==uint64_t(17*n)); ++i;
      assert(a[i].kind=="ring" && a[i].args[0]==reg32(linear,4)); ++i;
      assert(a[i].kind=="copy_out" && a[i].args[1]==RpuGetDevAddr(out.data_ptr())); ++i;
    } else {
      assert(a[i].kind=="scatter_out" && a[i].args[1]==RpuGetDevAddr(out.data_ptr()));
      assert(a[i].args[2]==17 && a[i].args[3]==80); ++i;
    }
    assert(i==a.size());
  }
}
void chunk_tail() {
  for(int part:{0,1}) {
    clear(); constexpr int m=35,n=1024,k=512,cores=4;
    const size_t ln=part?n/cores:n,lk=part?k:k/cores;
    const size_t fixed=Align(ln*2,SpmAllocator::ALIGN)+7*SpmAllocator::ALIGN;
    const size_t per_row=3*lk+2+2*ln+(part?0:4*n);
    constexpr size_t budget=SpmAllocator::SPM_PLANNING_BUDGET & ~(SpmAllocator::ALIGN-1);
    const size_t used=Align(budget-fixed-per_row*33,SpmAllocator::ALIGN);
    SPM_ALLOC.cursor=used;
    at::Tensor x({m,k}),w({n,k},at::kChar),sw({n}),bias({n});
    auto out=rpu_linear_w8a8(x,w,sw,bias,part,cores);
    assert(SPM_ALLOC.cursor==used);
    auto a=actions(); std::vector<Event> qs,ls,outs;
    for(auto e:a) {
      if(e.kind=="quant") qs.push_back(e);
      if(e.kind=="linear") ls.push_back(e);
      if(e.kind=="copy_out" || e.kind=="scatter_out") outs.push_back(e);
    }
    assert(qs.size()==2 && ls.size()==2 && outs.size()==2);
    assert(qs[0].regs[6]==32 && qs[1].regs[6]==3);
    assert(ls[0].regs[9]==32 && ls[1].regs[9]==3);
    assert(reg32(qs[0],0)==reg32(qs[1],0) && reg32(ls[0],4)==reg32(ls[1],4));
    assert(outs[0].args[1]==RpuGetDevAddr(out.data_ptr()));
    assert(outs[1].args[1]==RpuGetDevAddr(out.data_ptr())+32*n*2);
  }
}
void public_rejection() {
  clear(); at::Tensor x({17,96}),w({320,96},at::kChar),sw({320}),b({320});
  graph_dma::capturing=true;
  reject_clean([&]{ rpu_linear_w8a8(x,w,sw,b,1,4); });
  graph_dma::capturing=false;
  auto bad=x; bad.dense=false;
  reject_clean([&]{ rpu_linear_w8a8(bad,w,sw,b,1,4); });
  auto bad_bias=b; bad_bias.dims={319};
  reject_clean([&]{ rpu_linear_w8a8(x,w,sw,bad_bias,1,4); });
  reject_clean([&]{ rpu_linear_w8a8(x,w,sw,b,2,4); });
  reject_clean([&]{ rpu_linear_w8a8(x,w,sw,b,1,9); });
  KernelCache::instance().missing.insert(KernelId::DYNAMIC_QUANTIZE_W8A8);
  reject_clean([&]{ rpu_linear_w8a8(x,w,sw,b,1,4); });
  KernelCache::instance().missing.clear();
  KernelCache::instance().missing.insert(KernelId::PARALLEL_LINEAR_W8A8_ACC32_M128N64K128);
  reject_clean([&]{ rpu_linear_w8a8(x,w,sw,b,1,4); });
  KernelCache::instance().missing.clear();
  constexpr size_t budget=SpmAllocator::SPM_PLANNING_BUDGET & ~(SpmAllocator::ALIGN-1);
  SPM_ALLOC.cursor=budget;
  reject_clean([&]{ rpu_linear_w8a8(x,w,sw,b,1,4); });
}
int main(int argc,char** argv) {
  assert(argc==2); const std::string mode=argv[1];
  if(mode=="launchers") launchers();
  else if(mode=="rejection") rejection();
  else if(mode=="qwen_producers") qwen_producers();
  else if(mode=="norm_quantization") norm_quantization();
  else if(mode=="rms_quantization") rms_quantization();
  else if(mode=="gelu_quantization") gelu_quantization();
  else if(mode=="eager") eager();
  else if(mode=="chunk_tail") chunk_tail();
  else if(mode=="public_rejection") public_rejection();
  else return 2;
}
'''.replace('__ENUM__',enum).replace('__NAMES__',names).replace('__PRODUCTION__',production)))
    binary = folder / "launcher"
    subprocess.run([compiler, "-std=c++17", "-O0", "-g", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "src/ops"), "-I", str(Path(torch.__file__).parent / "include"), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("mode", ["launchers", "rejection", "rms_quantization", "qwen_producers", "norm_quantization", "gelu_quantization", "eager", "chunk_tail", "public_rejection"])
def test_actual_w8a8_launchers(launcher_binary, mode):
    subprocess.run([str(launcher_binary), mode], check=True, capture_output=True, text=True)

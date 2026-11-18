"""Execute the exact GR00T QKV launcher's native owner and register ABI on host."""
from pathlib import Path
import re

from test_fmb_standalone_layout_rebuild import _definition
from test_gr00t_ff1_resident_w8 import COMMON, _run

ROOT = Path(__file__).resolve().parents[1]
LINEAR = ROOT / "src/ops/rpu_linear.cpp"


def test_qkv_multibase_native_registers_owners_and_rejections(tmp_path):
    source = LINEAR.read_text()
    generated = (ROOT / "src/ops/rpu_specialized_linear_contracts.h").read_text()
    kernel_ids = sorted(set(re.findall(r"KernelId::(\w+)", generated)))
    generated = generated[generated.index("namespace rpu_specialized_linear {"):]
    functions = "\n".join(_definition(source, marker) for marker in (
        "void check_gr00t_qkv_multibase_spm(",
        "void rpu_launch_gr00t_qkv_multibase_w8_kernel(",
    ))
    address_check = _definition(
        (ROOT / "src/graph/graph_register_census.cpp").read_text(),
        "void validate_ddr_register_device_address(")
    device_boundary = _definition(COMMON, "namespace rhino_lkn{")
    assert COMMON.count(device_boundary) == 1
    common = COMMON.replace(
        device_boundary,
        "std::vector<std::pair<void*,uint64_t>> dev_addrs;\n"
        "namespace rhino_lkn{uint64_t RpuGetDevAddr(void*p){"
        "for(auto entry:dev_addrs)if(entry.first==p)return entry.second;"
        "throw std::runtime_error(\"unregistered owner\");}}")
    common = common.replace(
        "void rpu_ddr_flush_force(void*){}void rpu_ddr_flush(void*){}",
        "std::vector<void*> flushed;\n"
        "void rpu_ddr_flush(void*p){flushed.push_back(p);}")
    cpp = common + r'''
#include <limits>
#include "src/ops/rpu_linear_tiling.h"
__ADDRESS_CHECK__
struct SpmAllocator{static constexpr size_t SPM_USABLE=8387584;bool init=true;
 bool is_initialized(){return init;}uint32_t addr(int,int){return 0x10000000;}}SPM_ALLOC;
__CURRENT_CONTRACT__
constexpr KernelId expected_id=KernelId::GR00T_QKV_MULTIBASE_W8A16_M592N32K128;
struct Kernel_t{std::array<uint16_t,64>r{};void reset_regs(){r.fill(0);}
 void set_regs(int i,uint16_t v){r.at(i)=v;}}kernel;
bool payload_present=true;
Kernel_t*get_kernel(KernelId id){assert(id==expected_id);return payload_present?&kernel:nullptr;}
struct Queue{bool broadcast=false;int launches=0;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel_t&k,const std::vector<uint16_t>&grid,std::initializer_list<int>cores){
  assert(&k==&kernel&&broadcast&&grid==rpu_specialized_linear::grid(expected_id));
  assert(grid==std::vector<uint16_t>({18,1,1}));
  assert(std::vector<int>(cores)==std::vector<int>({0,1,2,3,4,5,6,7}));++launches;}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_KERNEL(id) get_kernel(id)
#define GET_QUEUE(n) get_queue(n)
enum class GraphDdrRegisterRole{Int8ModelWeight,PerChannelScale};
enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
 GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Writer{std::vector<GraphDdrRegisterOperandSpec>ops;
 void write(Kernel_t&k){for(const auto&o:ops)
  rpu_pl_tiling::set_u32_reg_pair(k,o.abi.lo,RpuGetDevAddr(o.tensor.data_ptr())>>8);}};
struct RpuKernelGraph{int staged=0;std::vector<GraphDdrRegisterOperandSpec>last_ops;
 static RpuKernelGraph&active(){static RpuKernelGraph graph;return graph;}
 Writer stage_kernel_ddr_registers(KernelId id,const std::vector<GraphDdrRegisterOperandSpec>&ops){
  assert(id==expected_id&&ops.size()==6);const int regs[]={2,20,32,34,40,42};
  for(int i=0;i<6;++i){const auto&a=ops[i].abi;
   assert(a.lo==regs[i]&&a.hi==regs[i]+1&&a.access==GraphDdrRegisterAccess::Read);
   assert(a.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
   assert(a.role==(i%2?GraphDdrRegisterRole::PerChannelScale:GraphDdrRegisterRole::Int8ModelWeight));
   validate_ddr_register_device_address(RpuGetDevAddr(ops[i].tensor.data_ptr()));}
  ++staged;last_ops=ops;return {ops};}};
__FUNCTIONS__
uint32_t pair_at(int reg){return uint32_t(kernel.r.at(reg))|(uint32_t(kernel.r.at(reg+1))<<16);}
int main(){
 const auto contract=rpu_specialized_linear::contract(expected_id);
 assert(contract.m==592&&contract.n==32&&contract.k==128);
 assert(contract.gx==18&&contract.gy==1&&contract.gz==1);
 std::array<at::Tensor,6>owners;
 // Independent owners deliberately use unrelated aligned DDR addresses above 4 GiB.
 const std::array<uint64_t,6>devices{0x1256789000ULL,0x2389001200ULL,0x34abcdef00ULL,
                                  0x4501234500ULL,0x5698765400ULL,0x67fedcba00ULL};
 for(int i=0;i<6;++i){owners[i]=i%2?at::Tensor({1536},at::kHalf):at::Tensor({1536,1536},at::kChar);
  dev_addrs.push_back({owners[i].data_ptr(),devices[i]});}
 std::array<uint32_t,7>spm{0x10000000,0x10020000,0x10030000,0x10040000,0x10050000,0x10050200,0x10050400};
 auto launch=[&]{rpu_launch_gr00t_qkv_multibase_w8_kernel(spm[0],owners[0],owners[2],owners[4],
                 spm[1],spm[2],spm[3],spm[4],spm[5],spm[6],owners[1],owners[3],owners[5]);};
 auto check=[&]{
  const auto&ops=RpuKernelGraph::active().last_ops;
  const int regs[]={2,20,32,34,40,42};
  for(int i=0;i<6;++i){assert(ops[i].tensor.unsafeGetTensorImpl()==owners[i].unsafeGetTensorImpl());
   assert(ops[i].tensor.data_ptr()==owners[i].data_ptr());
   assert(pair_at(regs[i])==RpuGetDevAddr(owners[i].data_ptr())>>8);
   assert(std::count(flushed.begin(),flushed.end(),owners[i].data_ptr())==1);}
  assert(flushed.size()==6);
  Kernel_t ordinary;rpu_pl_tiling::set_parallel_linear_regs(ordinary,
   {spm[0],RpuGetDevAddr(owners[0].data_ptr()),spm[1],spm[4],41,192,1536,1,8,3072,384,
    RpuGetDevAddr(owners[1].data_ptr()),1,1});
  for(int i=0;i<32;++i)assert(kernel.r[i]==ordinary.r[i]);
  assert(pair_at(0)==spm[0]&&pair_at(4)==spm[1]&&pair_at(6)==spm[4]);
  assert(pair_at(38)==spm[5]&&pair_at(36)==spm[2]&&pair_at(46)==spm[6]&&pair_at(44)==spm[3]);
  assert(kernel.r[8]==1&&kernel.r[9]==41&&pair_at(10)==192&&kernel.r[12]==1536&&kernel.r[13]==96);
  assert(kernel.r[14]==1&&kernel.r[15]==8&&pair_at(16)==3072&&pair_at(18)==384);
  assert(pair_at(22)==0&&pair_at(24)==0&&kernel.r[26]==2&&kernel.r[27]==1);
  assert(pair_at(28)==0&&pair_at(30)==0);
 };
 launch();check();assert(queue.launches==1);
 // Address and owner changes must reach the packet; unrelated Q/K/V bases cannot be inferred.
 for(int i=0;i<6;++i){owners[i]=i%2?at::Tensor({1536},at::kHalf):at::Tensor({1536,1536},at::kChar);
  dev_addrs.push_back({owners[i].data_ptr(),devices[5-i]});}
 std::swap(spm[1],spm[3]);std::swap(spm[4],spm[5]);flushed.clear();queue.broadcast=false;
 launch();check();assert(queue.launches==2);
 const auto good_spm=spm;const int staged=RpuKernelGraph::active().staged;
 const auto invalid=[&]{rejects(launch);assert(queue.launches==2);};
 for(int slot=0;slot<6;++slot){const auto good=owners[slot];
  for(int bad=0;bad<8;++bad){owners[slot]=good;auto&t=owners[slot];switch(bad){
   case 0:t=at::Tensor();break;case 1:t.dtype=slot%2?at::kChar:at::kHalf;break;
   case 2:t.dev.value=c10::DeviceType::CPU;break;case 3:t.contiguous_=false;break;
   case 4:t.shape.push_back(1);break;case 5:t.shape[0]=1535;break;
   case 6:t.shape.back()=1520;break;
   case 7:t.bytes=std::make_shared<std::vector<uint8_t>>(good.nbytes()-1);break;}
   invalid();}
  owners[slot]=good;
  auto entry=std::find_if(dev_addrs.begin(),dev_addrs.end(),[&](auto p){return p.first==good.data_ptr();});
  const auto address=entry->second;
  for(uint64_t bad:{uint64_t{0},address+1,UINT64_C(1)<<40}){entry->second=bad;invalid();}
  entry->second=address;
 }
 const std::array<uint32_t,7>sizes{41*1536*2,41*192*2,41*192*2,41*192*2,192*2,192*2,192*2};
 for(int i=0;i<7;++i){
  for(auto bad:{0u,0x10000000u-256u,good_spm[i]+1,
                0x10000000u+uint32_t(SpmAllocator::SPM_USABLE)-(sizes[i]/256)*256+256u,
                0xffffff00u}){spm=good_spm;spm[i]=bad;invalid();}
  for(int j=0;j<i;++j){spm=good_spm;spm[i]=spm[j];invalid();
   spm=good_spm;spm[i]=spm[j]+((sizes[j]-1)/256)*256;invalid();}
 }
 spm=good_spm;SPM_ALLOC.init=false;invalid();SPM_ALLOC.init=true;
 assert(RpuKernelGraph::active().staged==staged);
 payload_present=false;invalid();payload_present=true;
 // Legal adjacent intervals and the upper SPM boundary must remain admitted.
 spm[0]=0x10000000;for(int i=1;i<7;++i)spm[i]=spm[i-1]+((sizes[i-1]+255)/256)*256;
 spm[6]=0x10000000u+((uint32_t(SpmAllocator::SPM_USABLE)-sizes[6])/256)*256;
 flushed.clear();launch();check();assert(queue.launches==3);
}
'''.replace("__FUNCTIONS__", functions).replace("__ADDRESS_CHECK__", address_check)
    cpp = cpp.replace("__CURRENT_CONTRACT__",
                      "enum class KernelId{" + ",".join(kernel_ids) + "};\n" + generated)
    _run(tmp_path, cpp)


def test_qkv_multibase_append_only_current_payload():
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    implementation = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    kernel = "GR00T_QKV_MULTIBASE_W8A16_M592N32K128"
    symbol = '"gr00t_qkv_multibase_w8a16_v2"'
    retired = '"gr00t_qkv_multibase_w8a16_m592n32k128"'
    assert cache.index(kernel) > cache.index("GR00T_FF1_RESIDENT_W8A16_M512N48K128")
    assert symbol in cache
    assert retired not in cache
    optional_expansion = implementation.split("kMixedNames[] = {", 1)[1].split("};", 1)[0]
    assert symbol in optional_expansion
    assert retired not in optional_expansion
    # Mixed preload may skip the payload; the exact launcher must reject absence.
    required = implementation.split("KERNEL_LIST = {", 1)[1].split("};", 1)[0]
    assert symbol not in required
    loader = _definition(implementation, "for (const char* name : kMixedNames)")
    absent = _definition(loader, "if (mixed_symbols.count(kname) == 0)")
    assert "continue;" in absent
    assert "load_kernel_from_oplib(" not in absent
    assert loader.index(absent) < loader.index("load_kernel_from_oplib(")
    launcher = _definition(LINEAR.read_text(), "void rpu_launch_gr00t_qkv_multibase_w8_kernel(")
    assert "TORCH_CHECK(kernel != nullptr," in launcher
    assert "void rpu_launch_gr00t_qkv_multibase_w8_kernel(" in declarations

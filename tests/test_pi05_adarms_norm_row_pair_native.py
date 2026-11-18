"""Execute the actual Pi0.5 AdaRMSNorm + shift launcher's sibling selection: warp8 at M50, the row pair at 9..16 rows."""
from pathlib import Path
from test_siglip_cold_profile import _body
from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]


def test_denoise_siblings_select_by_rows_and_keep_the_register_image(tmp_path):
    source = (ROOT / 'src/ops/rpu_rmsnorm.cpp').read_text()
    marker = 'void rpu_launch_pi05_adarms_norm_shift_spm_kernel('
    start = source.index(marker)
    launcher = source[start:source.index('{', start)] + _body(source[start:], marker)
    cpp = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
template<class...A>void message(const A&...){}
#define TORCH_CHECK(ok,...) do{if(!(ok)){message(__VA_ARGS__);throw std::runtime_error("rejected");}}while(0)
// Two bytes, as the launcher's row size takes sizeof(c10::Half).
namespace c10 {struct Half{uint16_t bits;explicit Half(float v):bits(uint16_t(std::lround(v*1e6))){}};}
static uint16_t half_to_u16(c10::Half h){return h.bits;}
struct SpmAllocator{static constexpr uint64_t SPM_USABLE=8387584;
 bool is_initialized()const{return true;}
 uint64_t addr(int core,uint32_t offset)const{return 0x20000000ULL+uint64_t(core)*0x800000+offset;}}allocator;
#define SPM_ALLOC allocator
enum class KernelId{PI05_ADARMS_NORM_SHIFT_H1024,PI05_ADARMS_NORM_SHIFT_H1024_M50_WARP8,
 PI05_ADARMS_NORM_SHIFT_H1024_ROW_PAIR};
// The REF carries every payload unless a case removes one.
std::map<KernelId,bool> loaded{{KernelId::PI05_ADARMS_NORM_SHIFT_H1024,true},
 {KernelId::PI05_ADARMS_NORM_SHIFT_H1024_M50_WARP8,true},{KernelId::PI05_ADARMS_NORM_SHIFT_H1024_ROW_PAIR,true}};
struct KernelCache{static KernelCache&instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id)const{return loaded.at(id);}};
struct Kernel_t{std::map<uint32_t,uint16_t> regs;
 void reset_regs(){regs.clear();}
 int set_regs(uint32_t r,uint16_t v){assert(r<68);regs[r]=v;return 0;}};
Kernel_t kernel;KernelId used{};
Kernel_t*get_kernel(KernelId id){used=id;return loaded.at(id)?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
struct Queue{bool broadcast=false;int launches=0;std::vector<uint16_t> grid;std::vector<uint8_t> cores;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel_t&,std::vector<uint16_t> g,std::vector<uint8_t> c){assert(broadcast);grid=g;cores=c;++launches;}}queue;
Queue*get_queue(int){return &queue;}
#define GET_QUEUE(n) get_queue(n)
__LAUNCHER__
template<class F>void reject(F f){const int launches=queue.launches;
 try{f();assert(false);}catch(const std::runtime_error&){}assert(queue.launches==launches);}
int main(){
 constexpr uint32_t in=0x20010000,out=0x20030000,scale=0x20050000,shift=0x20050800;
 const auto launch=[&](int64_t rows,bool siblings){
  rpu_launch_pi05_adarms_norm_shift_spm_kernel(in,out,scale,shift,rows,1e-6,8,siblings);};
 const auto grid=[&](uint16_t x){return queue.grid==std::vector<uint16_t>({x,1,1});};
 launch(10,false);
 const auto generic=kernel.regs;
 assert(used==KernelId::PI05_ADARMS_NORM_SHIFT_H1024&&grid(10)&&!generic.count(14));
 assert(generic.at(0)==1024&&generic.at(2)==2048&&generic.at(4)==(in&0xffff)&&generic.at(13)==(shift>>16));
 // The row pair: the generic image plus register 14 = rows, ceil(rows / 2) blocks.
 for(int64_t rows:{9,10,15,16}){
  launch(rows,true);
  assert(used==KernelId::PI05_ADARMS_NORM_SHIFT_H1024_ROW_PAIR&&grid(uint16_t((rows+1)/2)));
  auto image=kernel.regs;assert(image.at(14)==rows);image.erase(14);assert(image==generic);}
 // Up to 8 rows already fit one wave; past 16 the single-row payload keeps its grid.
 for(int64_t rows:{1,8,17,64}){
  launch(rows,true);assert(used==KernelId::PI05_ADARMS_NORM_SHIFT_H1024&&grid(uint16_t(rows))&&!kernel.regs.count(14));}
 launch(50,true);
 assert(used==KernelId::PI05_ADARMS_NORM_SHIFT_H1024_M50_WARP8&&grid(8)&&!kernel.regs.count(14));
 // A REF without the sibling, or a caller that does not admit siblings, keeps the generic payload.
 loaded[KernelId::PI05_ADARMS_NORM_SHIFT_H1024_ROW_PAIR]=false;
 launch(10,true);assert(used==KernelId::PI05_ADARMS_NORM_SHIFT_H1024&&grid(10)&&kernel.regs==generic);
 loaded[KernelId::PI05_ADARMS_NORM_SHIFT_H1024_ROW_PAIR]=true;
 launch(16,false);assert(used==KernelId::PI05_ADARMS_NORM_SHIFT_H1024&&grid(16));
 // In place is admitted; a partial output alias is not.
 rpu_launch_pi05_adarms_norm_shift_spm_kernel(in,in,scale,shift,10,1e-6,8,true);
 assert(used==KernelId::PI05_ADARMS_NORM_SHIFT_H1024_ROW_PAIR&&grid(5));
 reject([&]{rpu_launch_pi05_adarms_norm_shift_spm_kernel(in,in+2048,scale,shift,10,1e-6,8,true);});
}
'''
    _compile_run(tmp_path, 'pi05_adarms_norm_row_pair_native', cpp.replace('__LAUNCHER__', launcher))

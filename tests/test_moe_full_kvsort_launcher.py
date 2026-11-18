"""Execute the full-sort launcher against a host queue recorder, without RPU."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(source, signature):
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_full_sort_runtime_abi_capacity_padding_and_fail_before_submit(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/ops/rpu_moe_index_shuffle.cpp").read_text()
    params = source[source.index("struct FullKvsortParams {"):
                    source.index("\n}  // namespace",
                                 source.index("struct FullKvsortParams {"))]
    code = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
template<class... Args> [[noreturn]] void fail(Args&&...) {
    throw std::runtime_error("rejected");
}
#define TORCH_CHECK(c, ...) do {if (!(c)) fail(__VA_ARGS__);} while (0)
namespace c10 {struct Half {
    uint16_t x=0;
    Half()=default;
    explicit Half(float v) {assert(v==0);x=0;}
};}
constexpr uint16_t ALU_MAX_FP16_WMODE=0x892, ALU_MIN_FP16_WMODE=0xa92;
constexpr uint16_t ALU_GT_IF16_OU8_WMODE=0x3a12, ALU_LT_IF16_OU8_WMODE=0x3c12;
constexpr uint16_t NEG_INF_FP16=0xfc00, POS_INF_FP16=0x7c00;
enum class KernelId {MOE_KVSORT_FP16};
struct Kernel_t {
    std::array<uint16_t,68> regs{};
    void reset_regs(){regs.fill(0);}
    void set_regs(int r,uint16_t value){regs.at(r)=value;}
    void set_regs(int,uint32_t)=delete;
    uint32_t pair(int r) const {return regs[r]|(uint32_t(regs[r+1])<<16);}
} kernel;
int fetches=0, submits=0;
bool available=true;
struct KernelCache {
    static KernelCache& instance(){static KernelCache c;return c;}
    bool has_loaded(KernelId)const{return available;}
};
Kernel_t* get_kernel(KernelId){++fetches;return available?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
struct SpmAllocator {
    static constexpr uint64_t SPM_USABLE=8*1024*1024-1024;
    bool initialized=true;
    bool is_initialized()const{return initialized;}
    uint32_t addr(int c,int off)const{assert(c==0);return 0x40000000+off;}
} spm;
#define SPM_ALLOC spm
struct Queue {
    bool broadcast=false;int expected_cores=8;
    void set_broadcast_mode(bool v){broadcast=v;}
    void enqueu_kernel(Kernel_t& k,std::vector<uint16_t> g,
                      std::vector<uint8_t> cores){
        assert(broadcast&&int(cores.size())==expected_cores);
        for(int i=0;i<expected_cores;++i)assert(cores[i]==i);
        assert(g==std::vector<uint16_t>({k.regs[64],1,1}));
        ++submits;
    }
} queue;
Queue* get_queue(int n){queue.expected_cores=n;return &queue;}
#define GET_QUEUE(n) get_queue(n)
struct Fill{uint32_t address;int64_t count;uint16_t bits;int cores;};
std::vector<Fill> fills;
void rpu_launch_fill_spm_kernel(uint32_t a,int64_t n,c10::Half h,int cores){
    fills.push_back({a,n,h.x,cores});
}
__PACK__
__PARAMS__
__SIZES__
__LAUNCH__
void launch(int64_t c,bool descending=false,int cores=8,
            uint32_t ik=0x40010000,uint32_t iv=0x40110000,
            uint32_t ok=0x40210000,uint32_t ov=0x40310000){
    rpu_launch_moe_kvsort_fp16_spm_kernel(ik,iv,ok,ov,c,descending,cores);
}
void reject(const std::function<void()>& fn){
    auto oldf=fetches,olds=submits;auto oldfill=fills.size();
    bool rejected=false;try{fn();}catch(const std::runtime_error&){rejected=true;}
    assert(rejected&&oldf==fetches&&olds==submits&&oldfill==fills.size());
}
int main(){
    // Runtime reference vectors: C6144 has only VLM padding, not SPM padding.
    launch(6144);
    assert(fills.empty()&&submits==1);
    assert(kernel.pair(0)==0x40010000/32&&kernel.pair(6)==0x40310000/32);
    assert(kernel.pair(8)==6144&&kernel.pair(10)==8192);
    const uint16_t expected[]={384,24,512,32,5,32,5,16,2,64,1,0,0,
                              0x892,0xa92,0x3a12,0x3c12,0x7c00};
    for(int i=0;i<18;++i)assert(kernel.regs[i+12]==expected[i]);
    assert((rpu_moe_kvsort_fp16_spm_sizes(6144)==std::make_pair<int64_t,int64_t>(6144,6144)));
    assert((rpu_moe_kvsort_fp16_spm_sizes(8192)==std::make_pair<int64_t,int64_t>(8192,8192)));
    // Multi-warp runtime requires 32768 output elements for 19320 logical.
    launch(19320,true,4);
    assert((rpu_moe_kvsort_fp16_spm_sizes(19320)==std::make_pair<int64_t,int64_t>(19328,32768)));
    assert(kernel.pair(8)==9664&&kernel.pair(10)==16384);
    assert(kernel.regs[12]==604&&kernel.regs[13]==38&&kernel.regs[14]==1024);
    assert(kernel.regs[15]==64&&kernel.regs[16]==6&&kernel.regs[22]==2);
    assert(kernel.regs[23]==1&&kernel.regs[24]==1&&kernel.regs[64]==2);
    assert(kernel.regs[25]==0xa92&&kernel.regs[27]==0x3c12&&kernel.regs[29]==0xfc00);
    assert(fills.size()==2&&fills[0].address==0x40010000+19320*2);
    assert(fills[0].count==8&&fills[0].bits==0xfc00&&fills[0].cores==4);
    assert(fills[1].address==0x40110000+19320*2&&fills[1].bits==0);
    // Exercise every transition, including non-v16 tails and warp boundaries.
    for(int64_t c:{1,15,16,17,4095,4096,4097,8191,8192,8193,16383,
                  16384,16385,32767,32768,32769,65535,65536,65537,131071,131072}){
        const auto p=make_full_kvsort_params(c);
        assert(p.input_elements>=c&&p.input_elements-c<16*p.warps);
        assert(p.input_elements==int64_t(p.warps)*p.c_per_warp);
        assert(p.c_per_warp%16==0&&p.c_per_warp<=p.c_pad_per_warp);
        assert(p.c_pad_per_warp<=16384&&p.output_elements>=p.input_elements);
        assert(p.warps==1||p.output_elements==int64_t(p.warps)*16384);
        launch(c);
    }
    reject([]{launch(0);});reject([]{launch(131073);});
    reject([]{launch(6144,false,0);});reject([]{launch(6144,false,9);});
    reject([]{launch(6144,false,8,0x40010002);});
    reject([]{launch(6144,false,8,0x40010000,0x40010000);});
    // A logical-size output reservation near the SPM end is insufficient
    // for the multi-warp merge extent and must fail before padding writes.
    reject([]{launch(19320,false,8,0x40010000,0x40110000,0x407f5f00);});
    available=false;reject([]{launch(19320);});available=true;
    spm.initialized=false;reject([]{launch(6144);});
}
"""
    code = (code.replace("__PACK__", _function(source, "void set_reg_u32("))
            .replace("__PARAMS__", params)
            .replace("__SIZES__", _function(
                source, "std::pair<int64_t, int64_t> rpu_moe_kvsort_fp16_spm_sizes("))
            .replace("__LAUNCH__", _function(
                source, "void rpu_launch_moe_kvsort_fp16_spm_kernel(")))
    cpp, binary = tmp_path / "kvsort.cpp", tmp_path / "kvsort"
    cpp.write_text(code)
    subprocess.run([compiler, "-std=c++17", "-O0", "-Wall", "-Wextra",
                    "-Werror", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

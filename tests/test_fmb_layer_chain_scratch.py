"""Execute the real output-side scratch DMA helpers with owned tensor doubles."""

from pathlib import Path
import re
import subprocess

from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]


def test_scratch_reload_matches_store_owner_rows_and_rejects_invalid_input(tmp_path):
    source = (ROOT / "src/core/fused_model_base.cpp").read_text()
    methods = []
    for name, declaration in (
        ("chunk_rows_", "int64_t chunk_rows_(const ChunkInfo& chunk) const"),
        ("chunk_row_offset_", "int64_t chunk_row_offset_(const ChunkInfo& chunk) const"),
        ("emit_layer_chain_scratch_store", "void store(int layer_idx, const ChunkInfo& chunk, const char* src_buf)"),
        ("emit_layer_chain_scratch_load", "void load(int layer_idx, const ChunkInfo& chunk, const char* dst_buf)"),
    ):
        methods.append(declaration + _block(source, f"FusedModelBase::{name}("))
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace at { struct Tensor { std::vector<uint16_t> data = std::vector<uint16_t>(512 * 4); }; }
enum class SpmFmbRuntimeMemberCapability { Unsealed, CurrentCallbackDenseDdrRefs };
struct ChunkInfo { int64_t len, offset; };
std::vector<uint16_t> spm(128 * 4);
int calls = 0;
void rpu_launch_spm_copy_ddr_dma(uint32_t address, const at::Tensor& owner,
                                int64_t offset, int64_t count) {
    assert(address == 0x40010000 && count == int64_t(spm.size()));
    auto& data = const_cast<at::Tensor&>(owner).data;
    std::copy(spm.begin(), spm.end(), data.begin() + offset);
}
void rpu_launch_ddr_broadcast_spm_dma(const at::Tensor& owner, int64_t offset,
                                    int64_t count, uint32_t address, int cores) {
    TORCH_CHECK(offset >= 0 && count > 0 && offset <= int64_t(owner.data.size()) &&
                count <= int64_t(owner.data.size()) - offset);
    assert(address == 0x40010000 && cores == 8);
    spm.assign(owner.data.begin() + offset, owner.data.begin() + offset + count);
    ++calls;
}
struct Model {
    struct Impl {
        struct { int64_t batch_size = 1; } ctx_;
        int num_layers_ = 64;
        int64_t hidden_size_ = 4;
        at::Tensor ddr_bufA_, ddr_bufB_;
        void* ddr_bufA_ptr_ = this;
        void* ddr_bufB_ptr_ = this;
    } data;
    Impl* pimpl_ = &data;
    SpmFmbRuntimeMemberCapability capability = SpmFmbRuntimeMemberCapability::Unsealed;
    auto spm_fmb_runtime_member_capability() const { return capability; }
    uint32_t addr(int core, const char*) const { assert(core == 0); return 0x40010000; }
    int num_cores() const { return 8; }
    __METHODS__
};
int main() {
    Model m;
    // The final layer must still reload its intermediate ping-pong side,
    // rather than the final output tensor. Every row tile is independent.
    for (int layer : {0, 1, 62, 63}) for (int row : {0, 128, 256, 384}) {
        const uint16_t value = uint16_t(1000 + 512 * layer + row);
        std::fill(spm.begin(), spm.end(), value);
        m.store(layer, {128, row}, "tile");
        const auto& owner = layer % 2 ? m.data.ddr_bufA_ : m.data.ddr_bufB_;
        assert(owner.data[row * 4] == value);
        std::fill(spm.begin(), spm.end(), 0);
        m.load(layer, {128, row}, "tile");
        assert(std::all_of(spm.begin(), spm.end(), [&](auto x) { return x == value; }));
    }
    auto rejected = [&](auto fn) {
        const int before = calls;
        bool threw = false;
        try { fn(); } catch (const std::runtime_error&) { threw = true; }
        assert(threw && calls == before);
    };
    for (int layer : {-1, 64}) rejected([&] { m.load(layer, {128, 0}, "tile"); });
    for (auto chunk : std::vector<ChunkInfo>{{0, 0}, {128, -1}, {128, 500},
             {std::numeric_limits<int64_t>::max(), 0}, {128, std::numeric_limits<int64_t>::max()}})
        rejected([&] { m.load(0, chunk, "tile"); });
    m.data.ctx_.batch_size = 0;
    rejected([&] { m.load(0, {128, 0}, "tile"); });
    m.data.ctx_.batch_size = 1;
    m.data.ddr_bufA_ptr_ = nullptr;
    rejected([&] { m.load(0, {128, 0}, "tile"); });
    m.data.ddr_bufA_ptr_ = &m;
    m.capability = SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs;
    rejected([&] { m.load(0, {128, 0}, "tile"); });
}
'''.replace("__METHODS__", "\n".join(methods))
    path, executable = tmp_path / "scratch.cpp", tmp_path / "scratch"
    path.write_text(cpp)
    compiled = subprocess.run(
        ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(executable)],
        capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_residual_shard_load_uses_live_input_and_correct_pingpong_owner(tmp_path):
    """Execute actual helper: eight distinct row shards, mutable layer0, both owners."""
    source = (ROOT / "src/core/fused_model_base.cpp").read_text()
    header = (ROOT / "src/core/fused_model_base.h").read_text()
    methods = []
    for name, declaration in (
        ("chunk_rows_", "int64_t chunk_rows_(const ChunkInfo& chunk) const"),
        ("chunk_row_offset_", "int64_t chunk_row_offset_(const ChunkInfo& chunk) const"),
        ("emit_layer_residual_shard_load", "void load(int layer_idx, const ChunkInfo& chunk, const char* dst_buf, bool scratch_output_side = false)"),
    ):
        methods.append(declaration + _block(source, f"FusedModelBase::{name}("))
    site = re.search(
        r"inline constexpr int64_t FMB_SHARED_LAYER_INPUT_SCATTER_DMA_SITE\s*=\s*[^;]+;",
        header,
    ).group(0)
    selector = ("enum class FmbSharedMutableDmaRouteSelector : int64_t " +
                _block(header, "enum class FmbSharedMutableDmaRouteSelector") + ";")
    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace c10 { using Half = uint16_t; }
namespace at { struct Tensor { std::vector<uint16_t> data = std::vector<uint16_t>(512 * 5120); }; }
enum class SpmFmbRuntimeMemberCapability { Unsealed, CurrentCallbackDenseDdrRefs };
enum class FmbRouteFamily { MUTABLE_DMA };
__SITE__
__SELECTOR__
struct ChunkInfo { int64_t len, offset; };
struct Context { int64_t batch_size = 1; const at::Tensor* hidden_states = nullptr; bool scatter_route = true; };
std::array<std::vector<uint16_t>,8> spm;
const at::Tensor* last_owner = nullptr;
uint64_t* last_live = nullptr;
int calls = 0, consumed = 0;
void consume_shared_runtime_route(Context& ctx, FmbRouteFamily family, int64_t site,
        int64_t selector, int invocation, std::initializer_list<int64_t> args, int cores, int mlp) {
    TORCH_CHECK(ctx.scatter_route);
    assert(family==FmbRouteFamily::MUTABLE_DMA && site==FMB_SHARED_LAYER_INPUT_SCATTER_DMA_SITE &&
           selector==int64_t(FmbSharedMutableDmaRouteSelector::DDR_SCATTER_TO_SPM) &&
           invocation==0 && args.size()==0 && cores==8 && mlp==8);
    ++consumed;
}
void copy(const at::Tensor& owner, int64_t offset, int64_t count, int64_t stride, uint32_t dst, int cores) {
    TORCH_CHECK(offset>=0 && count>0 && stride==count*2 && cores==8 &&
        offset<=int64_t(owner.data.size()) && cores*count<=int64_t(owner.data.size())-offset);
    assert(dst==0x40010000 && count==16*5120 && stride==163840);
    for(int core=0;core<cores;++core)
        spm[core].assign(owner.data.begin()+offset+core*count,owner.data.begin()+offset+(core+1)*count);
    last_owner=&owner; ++calls;
}
void rpu_launch_ddr_scatter_spm_dma(const at::Tensor& owner,int64_t offset,int64_t count,
        int64_t stride,uint32_t dst,int cores) {
    last_live=nullptr; copy(owner,offset,count,stride,dst,cores);
}
void rpu_launch_ddr_scatter_spm_dma_mutable(uint64_t& live,const at::Tensor& owner,
        int64_t byte_offset,int64_t count,int64_t stride,uint32_t dst,int cores) {
    assert(byte_offset%2==0); last_live=&live; copy(owner,byte_offset/2,count,stride,dst,cores);
}
struct Model {
    struct Impl {
        Context ctx_; int num_layers_=64; int64_t hidden_size_=5120;
        at::Tensor ddr_bufA_,ddr_bufB_;
        void* ddr_bufA_ptr_=this; void* ddr_bufB_ptr_=this;
    } data;
    Impl* pimpl_=&data;
    uint64_t live=0x80000000;
    int cores=8;
    SpmFmbRuntimeMemberCapability capability=SpmFmbRuntimeMemberCapability::Unsealed;
    auto spm_fmb_runtime_member_capability() const { return capability; }
    uint32_t addr(int core,const char*) const { assert(core==0);return 0x40010000; }
    int num_cores() const { return cores; }
    int mlp_tp() const { return cores; }
    uint64_t& current_hidden_in_src_base() { return live; }
    __METHODS__
};
int main() {
    Model m; at::Tensor input0,input1;
    for(auto item:std::vector<std::pair<at::Tensor*,int>>{{&input0,1},{&input1,2},{&m.data.ddr_bufA_,3},{&m.data.ddr_bufB_,4}})
        for(int row=0;row<512;++row)std::fill_n(item.first->data.begin()+row*5120,5120,uint16_t(item.second*1000+row));
    const auto verify=[&](const at::Tensor& owner,int row) {
        assert(last_owner==&owner);
        for(int core=0;core<8;++core)assert(spm[core].size()==16*5120 &&
            std::equal(spm[core].begin(),spm[core].end(),owner.data.begin()+(row+core*16)*5120));
    };
    // Four segments share the stable live-base slot; caller owner/base can change.
    for(auto* input:{&input0,&input1}) {
        m.data.ctx_.hidden_states=input; m.live+=0x100000;
        for(int row:{0,128,256,384}) {m.load(0,{128,row},"compact");verify(*input,row);assert(last_live==&m.live);}
    }
    assert(consumed==8);
    for(int layer:{0,1,62,63})for(int row:{0,128,256,384}) {
        if(layer>0) {m.load(layer,{128,row},"compact");verify(layer%2?m.data.ddr_bufB_:m.data.ddr_bufA_,row);assert(last_live==nullptr);}
        m.load(layer,{128,row},"compact",true);
        verify(layer%2?m.data.ddr_bufA_:m.data.ddr_bufB_,row);assert(last_live==nullptr);
    }
    assert(consumed==8); // Fixed scratch/input owners never consume mutable-input site.
    const auto rejected=[&](auto fn){const int before=calls;bool threw=false;try{fn();}catch(const std::runtime_error&){threw=true;}assert(threw&&calls==before);};
    for(int layer:{-1,64})rejected([&]{m.load(layer,{128,0},"compact");});
    for(auto chunk:std::vector<ChunkInfo>{{0,0},{127,0},{128,-1},{128,500},{std::numeric_limits<int64_t>::max(),0},{128,std::numeric_limits<int64_t>::max()}})
        rejected([&]{m.load(0,chunk,"compact");});
    m.data.ctx_.batch_size=0; rejected([&]{m.load(0,{128,0},"compact");});m.data.ctx_.batch_size=1;
    m.data.hidden_size_=0;rejected([&]{m.load(0,{128,0},"compact");});m.data.hidden_size_=5120;
    m.cores=0;rejected([&]{m.load(0,{128,0},"compact");});m.cores=8;
    m.data.ctx_.hidden_states=nullptr;rejected([&]{m.load(0,{128,0},"compact");});m.data.ctx_.hidden_states=&input1;
    m.data.ctx_.scatter_route=false;rejected([&]{m.load(0,{128,0},"compact");});m.data.ctx_.scatter_route=true;
    m.data.ddr_bufA_ptr_=nullptr;rejected([&]{m.load(1,{128,0},"compact");});rejected([&]{m.load(63,{128,0},"compact",true);});
    m.data.ddr_bufA_ptr_=&m;m.capability=SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs;
    rejected([&]{m.load(0,{128,0},"compact");});
}
'''.replace("__SITE__", site).replace("__SELECTOR__", selector).replace("__METHODS__", "\n".join(methods))
    path, executable = tmp_path / "shard.cpp", tmp_path / "shard"
    path.write_text(cpp)
    compiled = subprocess.run(
        ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(executable)],
        capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

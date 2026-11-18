"""Execute the actual grouped SPM DMA emitter against a byte-addressed facade."""
from pathlib import Path
import re
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[1]
definition = runpy.run_path(str(ROOT / "tests/test_qwen35_gdn_mask_residency.py"))["definition"]


def test_group_broadcast_layout_ordering_and_fail_closed(tmp_path):
    source = (ROOT / "src/ops/rpu_memcpy.cpp").read_text()
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    code = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
namespace c10 {struct Half {uint16_t value;};}
struct SpmAllocator {
    static constexpr uint64_t SPM_USABLE = 0x7ffc00;
    bool initialized = true;
    bool is_initialized() const {return initialized;}
    uint32_t addr(int c, uint32_t off) const {return 0x40000400 + c*0x1000000 + off;}
} SPM_ALLOC;
struct Event {int channel,peer; uint64_t src,dst,bytes;};
std::vector<Event> events;
namespace graph_dma {
bool enabled = true;
bool active() {return enabled;}
void emit_barrier(uint8_t channel, uint8_t peer) {events.push_back({channel,peer,0,0,0});}
void emit_fixed(uint64_t src, uint64_t dst, uint64_t bytes, int channel) {
    events.push_back({channel,-1,src,dst,bytes});
}
}
'''
    code += definition(decls, "struct DeferredDmaChannels") + ";\n"
    # The real declaration carries the default (non-deferred) argument.
    code += re.search(r"void rpu_launch_spm_group_broadcast_dma\([^;]*\);", decls).group() + "\n"
    for marker in (
        "static inline uint64_t spm_unified_to_per_core_abs(",
        "static inline void emit_scatter_pre_barriers(",
        "static inline void emit_scatter_post_barriers(",
        "void rpu_launch_spm_group_broadcast_dma(",
        "void rpu_join_deferred_dma(",
    ):
        code += "\n" + definition(source, marker)
    code += r'''
int main() {
    constexpr int elements = 32*256, bytes = elements*2;
    const uint32_t src = SPM_ALLOC.addr(0, 4096), dst = src+bytes;
    // Exercise 2 -> 8, the admitted contiguous prefixes and group sizes, and
    // every producer base that keeps the producers inside the prefix.
    for (int consumers=1; consumers<=8; ++consumers) {
        for (int producers=1; producers<=consumers; ++producers) {
          if (consumers%producers) continue;
          for (int base=0; base+producers<=consumers; ++base) {
            events.clear();
            rpu_launch_spm_group_broadcast_dma(src,elements,dst,producers,consumers,base);
            assert(events.size()==size_t(3*consumers-2));
            for (int c=1; c<consumers; ++c) {
                assert(events[c-1].channel==2*c && events[c-1].peer==0);
                auto post=events[2*consumers+c-2];
                assert(post.channel==0 && post.peer==2*c);
            }
            std::unordered_map<uint64_t,uint16_t> memory;
            for (int c=0;c<consumers;++c) for(int i=0;i<elements;++i)
                memory[SPM_ALLOC.addr(c,4096)+i*2] = (c+1)*123+i;
            // Reverse DMA order deliberately: copying onto producer core 1
            // must not clobber its original source for consumers 4..7.
            for (int c=consumers-1;c>=0;--c) {
                auto event=events[consumers-1+c];
                assert(event.channel==c && event.peer==-1 && event.bytes==bytes);
                assert(event.src==SPM_ALLOC.addr(base+c/(consumers/producers),4096));
                assert(event.dst==SPM_ALLOC.addr(c,4096+bytes));
                for(int i=0;i<elements;++i)
                    memory[event.dst+i*2]=memory.at(event.src+i*2);
            }
            for(int c=0;c<consumers;++c)for(int i=0;i<elements;++i)
                assert(memory.at(SPM_ALLOC.addr(c,4096+bytes)+i*2)==
                    uint16_t((base+c/(consumers/producers)+1)*123+i));
          }
        }
    }
    // Deferred K then V burst (Wall: 2 -> 8 from bases 0 and 2): DMAs only on
    // channels 1..7, each burst fences its own channels, no closing barrier
    // until the join, which closes exactly the channels used.
    {
        DeferredDmaChannels token;
        const uint32_t dst2 = dst + bytes;
        events.clear();
        rpu_launch_spm_group_broadcast_dma(src,elements,dst,2,8,0,&token);
        const size_t first = events.size();
        rpu_launch_spm_group_broadcast_dma(src,elements,dst2,2,8,2,&token);
        int dmas = 0, fences = 0;
        uint8_t fenced = 0;
        for (size_t i = 0; i < events.size(); ++i) {
            const auto& e = events[i];
            if (i == first) fenced = 0;
            if (e.peer == 0) {
                assert(e.channel % 2 == 0 && e.channel >= 2 && e.channel <= 14);
                assert(!(fenced & (1u << (e.channel / 2))));
                fenced |= 1u << (e.channel / 2); ++fences;
                continue;
            }
            assert(e.peer == -1 && e.channel >= 1 && e.channel <= 7);
            // A DMA runs only after its own burst fenced that channel.
            assert(fenced & (1u << e.channel));
            const int c = dmas % 8, base = dmas < 8 ? 0 : 2;
            assert(e.channel == 1 + dmas % 7);
            assert(e.src == SPM_ALLOC.addr(base + c / 4, 4096));
            assert(e.dst == SPM_ALLOC.addr(c, 4096 + bytes * (dmas < 8 ? 1 : 2)));
            ++dmas;
        }
        assert(dmas == 16 && fences == 14 && token.used == 0xfe);
        events.clear();
        rpu_join_deferred_dma(token);
        assert(events.size() == 7 && token.used == 0 && token.next == 0);
        for (int ch = 1; ch < 8; ++ch)
            assert(events[ch-1].channel == 0 && events[ch-1].peer == 2*ch);
        // An empty token joins nothing; a short burst joins only its channels.
        events.clear(); rpu_join_deferred_dma(token); assert(events.empty());
        rpu_launch_spm_group_broadcast_dma(src,elements,dst,1,2,0,&token);
        assert(token.used == 0x06);
        events.clear(); rpu_join_deferred_dma(token);
        assert(events.size() == 2 && events[0].peer == 2 && events[1].peer == 4);
    }
    auto reject=[&](uint32_t s,int64_t n,uint32_t d,int p,int c,int b=0) {
        events.clear(); bool rejected=false;
        DeferredDmaChannels token;
        try {rpu_launch_spm_group_broadcast_dma(s,n,d,p,c,b);}
        catch(const std::runtime_error&) {rejected=true;}
        assert(rejected && events.empty());
        rejected=false;
        try {rpu_launch_spm_group_broadcast_dma(s,n,d,p,c,b,&token);}
        catch(const std::runtime_error&) {rejected=true;}
        assert(rejected && events.empty() && token.used == 0);
    };
    // Producers outside the consumer prefix, or a negative base.
    reject(src,elements,dst,2,8,-1); reject(src,elements,dst,2,8,7);
    reject(src,elements,dst,2,4,3); reject(src,elements,dst,8,8,1);
    reject(src,elements,dst,0,8); reject(src,elements,dst,3,8);
    reject(src,elements,dst,2,9); reject(src,elements,dst,4,2);
    reject(src,0,dst,2,8); reject(src,-1,dst,2,8);
    reject(src,std::numeric_limits<int64_t>::max(),dst,2,8);
    reject(src,7,dst,2,8); reject(src+2,elements,dst,2,8);
    reject(src,elements,dst+2,2,8);
    reject(SPM_ALLOC.addr(0,0)-16,elements,dst,2,8);
    reject(SPM_ALLOC.addr(1,0),elements,dst,2,8);
    reject(SPM_ALLOC.addr(0,SpmAllocator::SPM_USABLE-16),elements,dst,2,8);
    reject(src,elements,SPM_ALLOC.addr(0,SpmAllocator::SPM_USABLE-16),2,8);
    reject(src,elements,src,2,8); reject(src,elements,src+16,2,8);
    reject(src,elements,src-16,2,8);
    graph_dma::enabled=false; reject(src,elements,dst,2,8);
    graph_dma::enabled=true; SPM_ALLOC.initialized=false;
    reject(src,elements,dst,2,8);
}
'''
    cpp = tmp_path / "group_dma.cpp"
    cpp.write_text(code)
    exe = tmp_path / "group_dma"
    subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_mutable_broadcast_defers_its_tail(tmp_path):
    source = (ROOT / "src/ops/rpu_memcpy.cpp").read_text()
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    code = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
namespace c10 {struct Half {uint16_t value;};}
struct SpmAllocator {
    uint32_t addr(int c, uint32_t off) const {return 0x40000400 + c*0x1000000 + off;}
} SPM_ALLOC;
struct Event {int channel,peer; const uint64_t* live; int64_t offset; uint64_t dst,bytes;};
std::vector<Event> events;
int immediate = 0;
struct Queue_t {void add_barrier(uint8_t, uint8_t) {}};
struct QueueCache {
    static QueueCache& instance() {static QueueCache cache; return cache;}
    Queue_t* get(uint8_t) {static Queue_t queue; return &queue;}
};
struct RpuKernelGraph {static void check_direct_immediate_dma_allowed(const char*) {}};
void rpu_add_dma_checked(Queue_t&, uint64_t, uint64_t, size_t, int, const char*) {}
void submit_immediate_batch(Queue_t*, const char*) {++immediate;}
namespace graph_dma {
bool enabled = true;
bool active() {return enabled;}
void emit_barrier(uint8_t channel, uint8_t peer) {events.push_back({channel,peer,nullptr,0,0,0});}
void emit_mutable_src(const uint64_t* live, int64_t offset, uint64_t dst, size_t bytes, uint8_t channel) {
    events.push_back({channel,-1,live,offset,dst,bytes});
}
}
'''
    code += definition(decls, "struct DeferredDmaChannels") + ";\n"
    code += re.search(r"void rpu_launch_ddr_broadcast_spm_dma_mutable\(\s*const uint64_t\*[^;]*\);",
                      decls).group() + "\n"
    for marker in (
        "static inline uint64_t spm_unified_to_per_core_abs(",
        "void rpu_launch_ddr_broadcast_spm_dma_mutable(const uint64_t* live_src_base,",
        "void rpu_join_deferred_dma(",
    ):
        code += "\n" + definition(source, marker)
    code += r'''
int main() {
    // Wall's AdaLN slab: 49 rows of 3H=3072 halves, layer 0 reads rows 0..1.
    const int64_t width = 3072, rows = 49, elements = rows*width, bytes = elements*2;
    const int64_t head = 2*width, head_bytes = head*2, offset = 3*bytes;
    const uint32_t dst = SPM_ALLOC.addr(0, 4096);
    uint64_t live = 0x80000000;
    auto core = [&](int c) {return uint64_t(SPM_ALLOC.addr(c, 4096));};
    // Without a token the burst is unchanged: fence, one DMA per core, join.
    rpu_launch_ddr_broadcast_spm_dma_mutable(&live, offset, elements, dst, 8);
    assert(events.size() == 22);
    for (int c = 0; c < 8; ++c) {
        const auto& e = events[7+c];
        assert(e.channel == c && e.live == &live && e.offset == offset &&
               e.dst == core(c) && e.bytes == uint64_t(bytes));
    }
    for (int ch = 1; ch < 8; ++ch)
        assert(events[ch-1].channel == 2*ch && events[ch-1].peer == 0 &&
               events[14+ch].channel == 0 && events[14+ch].peer == 2*ch);
    // With a token only the head is joined before the next kernel.
    events.clear();
    DeferredDmaChannels token;
    rpu_launch_ddr_broadcast_spm_dma_mutable(&live, offset, elements, dst, 8, &token, head);
    assert(events.size() == 22 + 7 + 8);
    std::vector<int64_t> covered(8, 0);
    for (size_t i = 0; i < events.size(); ++i) {
        const auto& e = events[i];
        if (i < 22) {
            if (i >= 7 && i < 15) assert(e.bytes == uint64_t(head_bytes) && e.channel == int(i) - 7);
        } else if (i < 29) {
            assert(e.peer == 0 && e.channel == 2*int(i-21));
        } else {
            const int c = int(i) - 29;
            assert(e.peer == -1 && e.channel == 1 + c%7 && e.live == &live);
            assert(e.offset == offset + head_bytes && e.dst == core(c) + head_bytes &&
                   e.bytes == uint64_t(bytes - head_bytes));
        }
        // Every DMA writes the slab byte it reads: head and tail tile it.
        if (e.peer == -1) {
            const int c = int((e.dst - core(0)) / 0x1000000);
            assert(int64_t(e.dst - core(c)) == e.offset - offset);
            covered[c] += int64_t(e.bytes);
        }
    }
    for (int64_t bytes_per_core : covered) assert(bytes_per_core == bytes);
    assert(token.used == 0xfe);
    events.clear(); rpu_join_deferred_dma(token);
    assert(events.size() == 7 && token.used == 0);
    // A zero head defers the whole slab.
    events.clear();
    rpu_launch_ddr_broadcast_spm_dma_mutable(&live, 0, elements, dst, 8, &token, 0);
    assert(events.size() == 15 && events[7].bytes == uint64_t(bytes) && token.used == 0xfe);
    token = DeferredDmaChannels{};
    // A head needs a token, must be 16-byte aligned, and must leave a tail.
    auto reject = [&](DeferredDmaChannels* t, int64_t sync) {
        events.clear(); bool rejected = false;
        try {rpu_launch_ddr_broadcast_spm_dma_mutable(&live, 0, elements, dst, 8, t, sync);}
        catch (const std::runtime_error&) {rejected = true;}
        assert(rejected && events.empty() && token.used == 0 && token.next == 0);
    };
    reject(nullptr, head); reject(&token, -8); reject(&token, 4);
    reject(&token, elements); reject(&token, elements + 8);
    // PASSTHROUGH copies everything in one immediate batch; nothing to join.
    graph_dma::enabled = false; events.clear();
    rpu_launch_ddr_broadcast_spm_dma_mutable(&live, 0, elements, dst, 8, &token, head);
    assert(immediate == 1 && events.empty() && token.used == 0);
}
'''
    cpp = tmp_path / "mutable_dma.cpp"
    cpp.write_text(code)
    exe = tmp_path / "mutable_dma"
    subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)

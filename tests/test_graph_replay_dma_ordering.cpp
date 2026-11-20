// Device contract tests; build with RHINOFORGE_BUILD_DEVICE_TESTS.
// Run on an exclusively locked board with the matching Release runtime.

#include <ATen/EmptyTensor.h>
#include "graph/graph_infra.h"
#include "graph/graph_runtime.h"
#include "rpu_caching_allocator.h"
#include "rpu_dma_endpoint.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rpu_spm_pool.h"

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

void rpu_release_conv2d_spm_cache();

namespace {

int g_passed = 0;
int g_failed = 0;
const char* g_section = "(uninit)";

void section(const char* name) {
    g_section = name;
    std::printf("\n--- %s ---\n", name);
}

#define EXPECT(cond, label)                                                    \
    do {                                                                       \
        if (cond) {                                                            \
            ++g_passed;                                                        \
            std::printf("  [PASS] %s\n", label);                               \
        } else {                                                               \
            ++g_failed;                                                        \
            std::fprintf(stderr,                                               \
                         "  [FAIL] %s (%s:%d in %s)\n",                        \
                         label, __FILE__, __LINE__, g_section);                \
        }                                                                      \
    } while (0)

using State = RpuKernelGraph::State;

GraphSignature make_sig(int64_t op_id, uint64_t branch_key = 0) {
    GraphSignature sig;
    sig.op_id = op_id;
    sig.shapes = {1, 1};
    sig.dyn_dims = {0};
    sig.dtypes = {2};
    sig.flags = {0u};
    sig.branch_key = branch_key;
    sig.segment_key = 0;
    return sig;
}

bool throws_with_message(const std::function<void()>& fn,
                         const char* expected) {
    try {
        fn();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

struct ManagedDdrFixture {
    static constexpr size_t kBytes = 64 * 1024;

    ManagedDdrFixture()
        : storage(rpu::RPUCachingAllocatorAdapter::get().allocate(kBytes)),
          base(::rhino_lkn::RpuGetDevAddr(storage.get())) {
        if (base == 0 || base % rpu::kAlignment != 0) {
            throw std::runtime_error("managed DDR fixture is not DMA aligned");
        }
    }

    uint64_t addr(size_t offset) const { return base + offset; }

    at::DataPtr storage;
    uint64_t base;
};

// 模拟 caller 拥有的 dev_addr (mutable DMA 的 live_base 指向这种 uint64_t):
// 每个变量代表一个 tensor 的 dev_addr 槽位,跨 forward 由 caller 重新赋值
// (record_dma_mutable_src/dst 拿到的是指针,每次 prepare deref 取最新值)。
struct LiveAddrs {
    uint64_t src_live;
    uint64_t dst_live;
    uint64_t src2_live;
};

void test_direct_allocator_dma_registry() {
    section("direct allocator participates in managed DMA resolution");
    uint64_t released_addr = 0;
    {
        void* ptr = ::rhino_lkn::RpuDdrAlloc(256);
        EXPECT(ptr != nullptr, "direct DDR allocation succeeds");
        if (ptr == nullptr) return;
        rpu_register_ddr_dma_allocation(ptr, 256);
        at::DataPtr storage(
            ptr, ptr, &rpu_free_registered_ddr,
            at::Device(c10::DeviceType::PrivateUse1, 0));
        released_addr = ::rhino_lkn::RpuGetDevAddr(ptr);
        const auto endpoint = rpu_resolve_device_dma_endpoint(
            released_addr + 32, 64, "direct allocator fixture");
        EXPECT(endpoint.owner != nullptr && endpoint.offset == 32,
               "direct DDR interior address resolves to owner+offset");
    }
    EXPECT(
        throws_with_message(
            [&] {
                (void)rpu_resolve_device_dma_endpoint(
                    released_addr, 32, "released direct allocator fixture");
            },
            "device DDR DMA range is outside every live logical allocation or SPM_ALLOC root"),
        "direct DDR release removes the reverse endpoint mapping");
}

// ----------------------------------------------------------------------
// Schema-v11 compatibility floor: pure ID-0 windows retain topology schema 1
// byte-for-byte, and legacy replay keeps its historical descriptor-overwrite
// behavior.  Nonzero endpoint strictness is covered by the FMB contract test.
// ----------------------------------------------------------------------
void test_legacy_id0_hash_and_replay_behavior(
        const ManagedDdrFixture& dma) {
    section("schema-v11 keeps legacy ID-0 hash and replay semantics");

    {
        RpuKernelGraph graph;
        const auto sig = make_sig(/*op_id=*/8090);
        graph.begin(sig);
        const GraphOpStreamStamp begin = graph.op_stream_stamp();
        graph.record_dma_fixed(
            /*src=*/0x41000000ull, /*dst=*/0x42000000ull,
            /*bytes=*/64, /*channel=*/3);
        const GraphOpStreamStamp end = graph.op_stream_stamp();
        const GraphOpStreamWindowDigest digest =
            graph.digest_op_stream_window(begin, end);
        EXPECT(digest.topology_hash == UINT64_C(0x9540f6c74d2e67c1) &&
                   digest.node_count == 1 &&
                   digest.contains(GraphNodeKind::Dma),
               "legacy one-Fixed-DMA topology remains schema-1 byte exact");
        graph.abort();
    }

    RpuGraphCache cache;
    const auto sig = make_sig(/*op_id=*/8091);
    LiveAddrs build_live{
        dma.addr(0x0800), dma.addr(0x1400), dma.addr(0x3000)};
    LiveAddrs replay_live{
        dma.addr(0x2000), dma.addr(0x2c00), dma.addr(0x3400)};
    const uint64_t build_fixed_src = dma.addr(0x0000);
    const uint64_t build_fixed_dst = dma.addr(0x0400);
    const uint64_t build_mutable_src_dst = dma.addr(0x0c00);
    const uint64_t build_mutable_dst_src = dma.addr(0x1000);
    const uint64_t replay_fixed_src = dma.addr(0x1800);
    const uint64_t replay_fixed_dst = dma.addr(0x1c00);
    const uint64_t replay_mutable_src_dst = dma.addr(0x2400);
    const uint64_t replay_mutable_dst_src = dma.addr(0x2800);
    auto& graph = cache.get_or_create(sig);
    graph.begin(sig);
    graph.record_dma_fixed(
        build_fixed_src, build_fixed_dst,
        /*bytes=*/64, /*channel=*/0);
    graph.record_dma_mutable_src(
        &build_live.src_live, /*offset=*/0,
        build_mutable_src_dst, /*bytes=*/64, /*channel=*/1);
    graph.record_dma_mutable_dst(
        build_mutable_dst_src, &build_live.dst_live,
        /*offset=*/32, /*bytes=*/64, /*channel=*/2);
    graph.end();

    graph.begin(sig);
    graph.record_dma_fixed(
        replay_fixed_src, replay_fixed_dst,
        /*bytes=*/64, /*channel=*/0);
    graph.record_dma_mutable_src(
        &replay_live.src_live, /*offset=*/32,
        replay_mutable_src_dst, /*bytes=*/64, /*channel=*/1);
    graph.record_dma_mutable_dst(
        replay_mutable_dst_src, &replay_live.dst_live,
        /*offset=*/64, /*bytes=*/64, /*channel=*/2);
    graph.end();

    const auto plan = graph.replay_plan();
    bool overwritten = plan.steps != nullptr && plan.steps->size() == 3;
    if (overwritten) {
        const auto& fixed = plan.steps->at(0).as_dma();
        const auto& mutable_src = plan.steps->at(1).as_dma();
        const auto& mutable_dst = plan.steps->at(2).as_dma();
        overwritten =
            fixed.semantic_endpoint_id == 0 &&
            fixed.src_addr == replay_fixed_src &&
            fixed.dst_addr == replay_fixed_dst &&
            mutable_src.semantic_endpoint_id == 0 &&
            mutable_src.live_base == &replay_live.src_live &&
            mutable_src.live_offset == 32 &&
            mutable_src.dst_addr == replay_mutable_src_dst &&
            mutable_dst.semantic_endpoint_id == 0 &&
            mutable_dst.live_base == &replay_live.dst_live &&
            mutable_dst.live_offset == 64 &&
            mutable_dst.src_addr == replay_mutable_dst_src;
    }
    EXPECT(overwritten,
           "legacy ID-0 replay still accepts and stores endpoint drift");
    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

// Schemas 2/3/4 are independent fixed goldens, not values learned from the
// current hasher.  The fixture is the same one-Fixed-DMA window as schema 1,
// with only the semantic endpoint/canonical/SPM-peer sidecars progressively
// enabled.  This freezes all pre-schema-5 encodings byte-for-byte.
void test_semantic_dma_topology_schema_goldens() {
    section("schema 2/3/4 topology hashes stay byte exact");

    auto digest_for = [](int64_t op_id,
                         bool canonical,
                         uint64_t spm_peer_id) {
        RpuKernelGraph graph;
        graph.begin(make_sig(op_id));
        const GraphOpStreamStamp begin = graph.op_stream_stamp();
        graph.record_dma_fixed(
            /*src=*/UINT64_C(0x41000000),
            /*dst=*/UINT64_C(0x42000000),
            /*bytes=*/64, /*channel=*/3);
        RpuKernelGraphContractTestAccess::
            set_dma_semantics_for_topology_golden(
                graph, /*node_idx=*/0,
                /*endpoint_id=*/UINT64_C(0x1234), canonical, spm_peer_id);
        const GraphOpStreamStamp end = graph.op_stream_stamp();
        const auto digest = graph.digest_op_stream_window(begin, end);
        graph.abort();
        return digest.topology_hash;
    };

    EXPECT(digest_for(/*op_id=*/8092, /*canonical=*/false,
                      /*spm_peer_id=*/0) ==
               UINT64_C(0x13ad710e5b4dc56a),
           "schema-2 semantic endpoint topology golden");
    EXPECT(digest_for(/*op_id=*/8093, /*canonical=*/true,
                      /*spm_peer_id=*/0) ==
               UINT64_C(0x89e3123270f686cc),
           "schema-3 canonical-member topology golden");
    EXPECT(digest_for(/*op_id=*/8094, /*canonical=*/true,
                      /*spm_peer_id=*/UINT64_C(0x5678)) ==
               UINT64_C(0x28ce3999ddb61ddd),
           "schema-4 typed-SPM-peer topology golden");
    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

// ----------------------------------------------------------------------
// ① + ② + ③: 多段 graph,各段 mutable_dmas 独立且 dma_id 单调
// ----------------------------------------------------------------------
//
// 用 capture_data<MemcpyNodeData> (HOST_MEMCPY) 当 segment 边界把 5 个 DMA
// 节点切成 2 segments:
//   nodes_[0]  Dma  Fixed                            seg 0 start
//   nodes_[1]  Dma  MutableSrc  → dma_id 0
//   nodes_[2]  Barrier                               (不增 dma_id)
//   nodes_[3]  Dma  Fixed                            (不增 dma_id)
//   nodes_[4]  Dma  MutableDst  → dma_id 1
//   nodes_[5]  Memcpy (HOST_MEMCPY)                  → segment 边界
//   nodes_[6]  Dma  MutableSrc  → dma_id 0 (seg 1, counter reset)
//   nodes_[7]  Dma  Fixed                            (不增 dma_id)
// ----------------------------------------------------------------------
void test_dma_id_monotone_across_segments(const ManagedDdrFixture& dma) {
    section("dma_id monotone within segment, reset per segment");

    RpuGraphCache cache;
    auto sig = make_sig(/*op_id=*/8001);
    LiveAddrs live{
        dma.addr(0x2000), dma.addr(0x2400), dma.addr(0x2800)};

    const uint64_t kFixedSrc1 = dma.addr(0x0000);
    const uint64_t kFixedDst1 = dma.addr(0x0400);
    const uint64_t kFixedSrc2 = dma.addr(0x0800);
    const uint64_t kFixedDst2 = dma.addr(0x0c00);
    const uint64_t kFixedSrc3 = dma.addr(0x1000);
    const uint64_t kFixedDst3 = dma.addr(0x1400);
    const uint64_t kFixedSrc4 = dma.addr(0x1800);
    const uint64_t kFixedDst4 = dma.addr(0x1c00);

    // MemcpyNodeData 作为 segment 边界用;HOST_MEMCPY kind 跑 ::memcpy,不需要
    // 真 device 地址,只占位推 segment 切分。
    std::vector<uint8_t> host_a(64, 0xAA);
    std::vector<uint8_t> host_b(64, 0xBB);

    auto& g = cache.get_or_create(sig);
    g.begin(sig);
    EXPECT(g.state() == State::RECORDING, "begin -> RECORDING");

    // ----- seg 0 nodes -----
    g.record_dma_fixed(kFixedSrc1, kFixedDst1, /*bytes=*/64, /*channel=*/0);
    g.record_dma_mutable_src(&live.src_live, /*offset=*/0,
                              kFixedDst2, /*bytes=*/128, /*channel=*/1);
    g.record_barrier(/*self=*/0, /*target=*/1);
    g.record_dma_fixed(kFixedSrc3, kFixedDst3, /*bytes=*/64, /*channel=*/2);
    g.record_dma_mutable_dst(kFixedSrc4, &live.dst_live, /*offset=*/32,
                              /*bytes=*/256, /*channel=*/3);

    // segment boundary
    MemcpyNodeData m;
    m.dst = host_b.data();
    m.src = host_a.data();
    m.bytes = host_a.size();
    m.kind = CopyKind::HOST_MEMCPY;
    g.capture_data(m);

    // ----- seg 1 nodes -----
    g.record_dma_mutable_src(&live.src2_live, /*offset=*/64,
                              kFixedDst4, /*bytes=*/512, /*channel=*/4);
    g.record_dma_fixed(kFixedSrc2, kFixedDst2, /*bytes=*/64, /*channel=*/5);

    g.end();
    EXPECT(g.state() == State::BUILT, "end -> BUILT");

    // ----- inspect segments_ via replay_plan() (read-only view) -----
    auto plan = g.replay_plan();
    EXPECT(plan.segments_size() == 2,
           "two segments formed (Memcpy node split them)");
    if (plan.segments_size() < 2) return;

    const Segment& s0 = plan.segment_at(0);
    EXPECT(s0.mutable_dmas.size() == 2,
           "seg 0 has 2 mutable DMA slots (MutableSrc + MutableDst); "
           "Fixed/Barrier excluded");
    if (s0.mutable_dmas.size() == 2) {
        const auto& slot0 = s0.mutable_dmas[0];
        const auto& slot1 = s0.mutable_dmas[1];
        EXPECT(slot0.dma_id == 0, "seg 0 slot[0].dma_id == 0");
        EXPECT(slot1.dma_id == 1, "seg 0 slot[1].dma_id == 1 (monotone)");
        EXPECT(slot0.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc,
               "seg 0 slot[0].kind == MutableSrc");
        EXPECT(slot1.kind == Segment::PreparedMutableDmaSlot::Kind::MutableDst,
               "seg 0 slot[1].kind == MutableDst");
        EXPECT(slot0.live_base == &live.src_live,
               "seg 0 slot[0].live_base aliases caller-owned uint64_t");
        EXPECT(slot0.live_offset == 0, "seg 0 slot[0].live_offset == 0");
        EXPECT(slot0.fixed_addr == kFixedDst2,
               "seg 0 slot[0].fixed_addr == dst (MutableSrc 把 dst 当作 baked)");
        EXPECT(slot0.fixed_allocation_id != 0 &&
                   slot0.fixed_allocation_id == rpu_resolve_device_dma_endpoint(
                       kFixedDst2, slot0.bytes, "seg0 mutable-src fixed side").allocation_id,
               "seg 0 slot[0] retains the live fixed-side allocation identity");
        EXPECT(slot0.bytes == 128, "seg 0 slot[0].bytes == 128");
        EXPECT(slot0.channel == 1, "seg 0 slot[0].channel == 1");
        EXPECT(slot1.live_base == &live.dst_live,
               "seg 0 slot[1].live_base aliases caller-owned uint64_t");
        EXPECT(slot1.live_offset == 32, "seg 0 slot[1].live_offset == 32");
        EXPECT(slot1.fixed_addr == kFixedSrc4,
               "seg 0 slot[1].fixed_addr == src (MutableDst 把 src 当作 baked)");
        EXPECT(slot1.fixed_allocation_id != 0 &&
                   slot1.fixed_allocation_id == rpu_resolve_device_dma_endpoint(
                       kFixedSrc4, slot1.bytes, "seg0 mutable-dst fixed side").allocation_id,
               "seg 0 slot[1] retains the live fixed-side allocation identity");
        EXPECT(slot1.bytes == 256, "seg 0 slot[1].bytes == 256");
        EXPECT(slot1.channel == 3, "seg 0 slot[1].channel == 3");
    }

    const Segment& s1 = plan.segment_at(1);
    EXPECT(s1.mutable_dmas.size() == 1,
           "seg 1 has 1 mutable DMA slot (counter reset across segments)");
    if (s1.mutable_dmas.size() == 1) {
        const auto& slot0 = s1.mutable_dmas[0];
        EXPECT(slot0.dma_id == 0,
               "seg 1 slot[0].dma_id == 0 (per-segment dma_id counter)");
        EXPECT(slot0.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc,
               "seg 1 slot[0].kind == MutableSrc");
        EXPECT(slot0.live_base == &live.src2_live,
               "seg 1 slot[0].live_base == &src2_live");
        EXPECT(slot0.live_offset == 64, "seg 1 slot[0].live_offset == 64");
        EXPECT(slot0.fixed_addr == kFixedDst4,
               "seg 1 slot[0].fixed_addr == kFixedDst4");
        EXPECT(slot0.fixed_allocation_id != 0 &&
                   slot0.fixed_allocation_id == rpu_resolve_device_dma_endpoint(
                       kFixedDst4, slot0.bytes, "seg1 mutable-src fixed side").allocation_id,
               "seg 1 slot[0] retains the live fixed-side allocation identity");
    }

    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

// ----------------------------------------------------------------------
// ④: 单段 sig 的 BUILD→REPLAY 触发 sync-only fast path
// ----------------------------------------------------------------------
//
// 同一 sig 连续两次 forward,单段 graph (无 Memcpy 边界,只有 Dma + Barrier):
//   forward 1: RECORDING → BUILT,prepared_segment_miss_total 累 1 (full build)
//   forward 2: REPLAYING → BUILT,prepared_segment_hit_total 累 1 (sync-only)
// 同时验证两次 mutable_dmas 槽位元数据稳定。
void test_single_segment_sync_only_hit(const ManagedDdrFixture& dma) {
    section("single-segment sig: forward 2 hits sync-only fast path");

    RpuGraphCache cache;
    auto sig = make_sig(/*op_id=*/8002);
    LiveAddrs live{
        dma.addr(0x0800), dma.addr(0x0c00), dma.addr(0x1000)};

    const uint64_t kFixedSrc = dma.addr(0x0000);
    const uint64_t kFixedDst = dma.addr(0x0400);

    // forward 1
    {
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        EXPECT(g.state() == State::RECORDING, "fwd1 begin -> RECORDING");
        g.record_dma_mutable_src(&live.src_live, /*offset=*/0,
                                  kFixedDst, /*bytes=*/64, /*channel=*/0);
        g.record_dma_fixed(kFixedSrc, kFixedDst, /*bytes=*/64, /*channel=*/1);
        g.record_dma_mutable_dst(kFixedSrc, &live.dst_live, /*offset=*/32,
                                  /*bytes=*/64, /*channel=*/2);
        g.end();
        EXPECT(g.state() == State::BUILT, "fwd1 end -> BUILT");

        const auto& stats = g.debug_stats();
        EXPECT(stats.segment_count == 1, "fwd1 single segment");
        EXPECT(stats.prepared_segment_miss_total == 0,
               "fwd1 RECORDING path doesn't bump hit/miss (only REPLAY does)");
        EXPECT(stats.prepared_segment_hit_total == 0, "fwd1 hit_total == 0");
    }

    // forward 2: caller 把 live 地址 "漂移" 到新 dev_addr;sync-only 会通过
    // update_dma_kernel 把新 dev_addr 推下去；本测只验槽位元数据和
    // fast-path 触发，不把 DMA payload 内容纳入断言。
    live.src_live = dma.addr(0x1400);
    live.dst_live = dma.addr(0x1800);
    {
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        EXPECT(g.state() == State::REPLAYING, "fwd2 begin -> REPLAYING (sig hit)");
        g.record_dma_mutable_src(&live.src_live, /*offset=*/0,
                                  kFixedDst, /*bytes=*/64, /*channel=*/0);
        g.record_dma_fixed(kFixedSrc, kFixedDst, /*bytes=*/64, /*channel=*/1);
        g.record_dma_mutable_dst(kFixedSrc, &live.dst_live, /*offset=*/32,
                                  /*bytes=*/64, /*channel=*/2);
        g.end();
        EXPECT(g.state() == State::BUILT, "fwd2 end -> BUILT");

        const auto& stats = g.debug_stats();
        EXPECT(stats.segment_count == 1, "fwd2 still single segment");
        EXPECT(stats.prepared_segment_hit_total >= 1,
               "fwd2 REPLAY hits sync-only fast path (hit_total >= 1)");
        EXPECT(stats.prepared_segment_miss_total == 0,
               "fwd2 single-seg never misses (private_queue_ kd_buf preserved)");

        // 槽位元数据稳定 (live_base 仍然是 caller-owned uint64_t 的地址)
        auto plan = g.replay_plan();
        if (plan.segments_size() == 1 &&
            plan.segment_at(0).mutable_dmas.size() == 2) {
            const auto& s = plan.segment_at(0);
            EXPECT(s.mutable_dmas[0].live_base == &live.src_live,
                   "fwd2 slot[0].live_base 跨 REPLAY 稳定");
            EXPECT(s.mutable_dmas[1].live_base == &live.dst_live,
                   "fwd2 slot[1].live_base 跨 REPLAY 稳定");
        }
    }
    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

// ----------------------------------------------------------------------
// ⑤: 多段 sig 的 BUILD→REPLAY:每段都 miss (kd_buf 只能容纳一段)
// ----------------------------------------------------------------------
void test_multi_segment_rebuild_each_replay(const ManagedDdrFixture& dma) {
    section("multi-segment sig: each replay segment misses (kd_buf rotates)");

    RpuGraphCache cache;
    auto sig = make_sig(/*op_id=*/8003);
    LiveAddrs live{
        dma.addr(0x0800), dma.addr(0x0c00), dma.addr(0x1000)};

    const uint64_t kFixedDst = dma.addr(0x0400);
    std::vector<uint8_t> host_a(32, 0x10);
    std::vector<uint8_t> host_b(32, 0x20);

    auto record_two_segments = [&](RpuKernelGraph& g) {
        g.record_dma_mutable_src(&live.src_live, 0, kFixedDst, 64, 0);
        // segment boundary
        MemcpyNodeData m{};
        m.dst = host_b.data();
        m.src = host_a.data();
        m.bytes = host_a.size();
        m.kind = CopyKind::HOST_MEMCPY;
        g.capture_data(m);
        g.record_dma_mutable_dst(kFixedDst, &live.dst_live, 0, 64, 0);
    };

    // forward 1 (BUILD)
    {
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        record_two_segments(g);
        g.end();
        EXPECT(g.state() == State::BUILT, "fwd1 BUILT");
        const auto& stats = g.debug_stats();
        EXPECT(stats.segment_count == 2, "fwd1 two segments");
    }

    // forward 2 (REPLAY) — 多段下 segment fingerprint 跟不上 kd_buf,每段都 miss
    {
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        EXPECT(g.state() == State::REPLAYING, "fwd2 -> REPLAYING");
        record_two_segments(g);
        g.end();
        EXPECT(g.state() == State::BUILT, "fwd2 BUILT");
        const auto& stats = g.debug_stats();
        EXPECT(stats.segment_count == 2, "fwd2 still two segments");
        // 两段都走 fallback rebuild;不允许 sync-only 在 stale segment 上 fire
        // (Codex Round-4 Risk #2 缓解的核心)。
        EXPECT(stats.prepared_segment_hit_total == 0,
               "fwd2 multi-segment: hit_total stays 0 (no stale sync-only)");
        EXPECT(stats.prepared_segment_miss_total >= 2,
               "fwd2 multi-segment: miss_total >= 2 (each segment rebuilt)");
    }
    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

// ----------------------------------------------------------------------
// ⑥: 自动资源切段不能切开跨 stream DMA fence
// ----------------------------------------------------------------------
void test_resource_budget_preserves_dma_fence(
        const ManagedDdrFixture& dma) {
    section("resource budget splits only after a cross-stream DMA fence");

    // Graph runtime uses the configured SDK limit directly:
    // SDK cap 8 -> automatic segment budget 8 entries.
    GraphRuntimePolicy runtime_policy;
    runtime_policy.lkn_max_batch_entries = 8;

    {
        RpuGraphCache cache(/*max_entries=*/16, runtime_policy);
        auto sig = make_sig(/*op_id=*/8004);
        auto& g = cache.get_or_create(sig);
        g.begin(sig);

        // Seven safe stream-0 entries, followed by one indivisible stream-2
        // burst: pre-barrier, DMA, post-barrier.  The burst takes segment 0 to
        // ten entries; auto-splitting at eight or nine would detach its fence.
        for (int i = 0; i < 7; ++i) {
            g.record_dma_fixed(dma.addr(0x0000), dma.addr(0x0400), 64, 0);
        }
        g.record_barrier(/*self=*/2, /*target=*/0);
        g.record_dma_fixed(dma.addr(0x0000), dma.addr(0x0800), 64, 1);
        g.record_barrier(/*self=*/0, /*target=*/2);
        g.record_dma_fixed(dma.addr(0x0000), dma.addr(0x0400), 64, 0);
        g.end();

        const auto plan = g.replay_plan();
        EXPECT(plan.segments_size() == 2,
               "entry budget creates exactly two segments");
        if (plan.segments_size() == 2) {
            EXPECT(plan.segment_at(0).start_idx == 0 &&
                       plan.segment_at(0).end_idx == 10,
                   "segment 0 retains the complete pre/DMA/post fence");
            EXPECT(plan.segment_at(1).start_idx == 10 &&
                       plan.segment_at(1).end_idx == 11,
                   "first safe node after the fence starts segment 1");
        }
        EXPECT(g.debug_stats().hw_batch_submit_total == 2,
               "BUILD submits both automatic segments");
    }

    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

void test_lingbot2_exact_graph_uses_only_sealed_sdk_budget(
        const ManagedDdrFixture& dma) {
    section("LingBot2 exact labels require typed opt-in and sealed SDK budget");

    const auto segments_for = [&](bool z2,
                                  size_t entries,
                                  size_t kd_mb,
                                  size_t instr_mb,
                                  const char* graph_label) {
        GraphRuntimePolicy runtime_policy;
        runtime_policy.lingbot2_multiview_spm_z2 = z2;
        runtime_policy.lkn_max_batch_entries = entries;
        runtime_policy.lkn_kd_buf_mb = kd_mb;
        runtime_policy.lkn_instr_buf_mb = instr_mb;
        RpuGraphCache cache(/*max_entries=*/16, runtime_policy);
        auto sig = make_sig(/*op_id=*/8006);
        sig.op_id_str = graph_label;
        auto& graph = cache.get_or_create(sig);
        graph.begin(sig);
        for (size_t index = 0; index < 65537; ++index) {
            graph.record_dma_fixed(
                dma.addr(0x0000), dma.addr(0x0400), 64, 0);
        }
        graph.end();
        return graph.replay_plan().segments_size();
    };

    for (const char* graph_label : {
             "lingbot2_multiview_spm_z2_bootstrap",
             "lingbot2_multiview_spm_z2_retained",
             "lingbot2_denoise_unroll",
         }) {
        EXPECT(segments_for(true, 262144, 64, 512, graph_label) == 1,
               graph_label);
    }

    EXPECT(segments_for(
               true, 262144, 64, 512,
               "qwen3_causal_decoder") == 2,
           "ordinary/Qwen label keeps generic 65536-entry splitter");

    EXPECT(segments_for(
               false, 262144, 64, 512,
               "lingbot2_multiview_spm_z2_retained") == 2,
           "disabled typed Z2 policy keeps generic splitter");

    struct QueueDrift {
        size_t entries;
        size_t kd_mb;
        size_t instr_mb;
        const char* label;
    };
    for (const QueueDrift& drift : {
             QueueDrift{262143, 64, 512, "entry limit drift"},
             QueueDrift{262144, 63, 512, "KD limit drift"},
             QueueDrift{262144, 64, 511, "instruction limit drift"},
         }) {
        EXPECT(segments_for(
                   true, drift.entries, drift.kd_mb, drift.instr_mb,
                   "lingbot2_multiview_spm_z2_retained") == 2,
               drift.label);
    }

    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

void test_wall_oss_wrc_exact_graph_keeps_generic_entry_limit(
        const ManagedDdrFixture& dma) {
    section("Wall-OSS WRC exact signatures keep generic entry limit");

    GraphRuntimePolicy runtime_policy;
    runtime_policy.lkn_max_batch_entries = 8192;
    const auto segments_for = [&](GraphSignature sig) {
        RpuGraphCache cache(/*max_entries=*/16, runtime_policy);
        auto& graph = cache.get_or_create(sig);
        graph.begin(sig);
        for (size_t index = 0; index < 8193; ++index) {
            graph.record_dma_fixed(
                dma.addr(0x0000), dma.addr(0x0400), 64, 0);
        }
        graph.end();
        return graph.replay_plan().segments_size();
    };
    GraphSignature vision;
    vision.op_id = INT64_C(7096842445486029970);
    vision.op_id_str = "wall_oss_vision";
    vision.shapes = {1728, 1280};
    vision.dyn_dims = {
        32, 576, 1, 3, 0, 1, 1, 0,
        1, 0, 64, 128, 192, 256, 320, 384,
        448, 512, 528, 544, 560, 576};
    vision.dtypes = {5};
    EXPECT(segments_for(vision) == 2,
           "exact WRC Vision signature keeps the generic splitter");

    GraphSignature rtc;
    rtc.op_id = INT64_C(4753078248529336008);
    rtc.op_id_str = "wall_oss_action_rtc_denoise_loop_v2";
    rtc.shapes = {20, 1024};
    rtc.dyn_dims = {36, 640, 32, 6, 1, 0, 10, 1, 1};
    rtc.dtypes = {5};
    EXPECT(segments_for(rtc) == 2,
           "exact WRC RTC D0 signature keeps the generic splitter");

    rtc.dyn_dims[4] = 0;
    rtc.dyn_dims[5] = 10;
    EXPECT(segments_for(rtc) == 2,
           "exact WRC RTC D10 signature keeps the generic splitter");

    GraphSignature action;
    action.op_id = INT64_C(-4760867079612397392);
    action.op_id_str = "wall_oss_action_denoise_loop_v2";
    action.shapes = {20, 1024};
    action.dyn_dims = {36, 640, 32, 6, 1, 1, 1};
    action.dtypes = {5};
    EXPECT(segments_for(action) == 2,
           "exact WRC non-RTC signature keeps the generic splitter");

    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

// ----------------------------------------------------------------------
// ⑦: non-replayable oneshot 必须消费 QueueCache 上的 SDK batch state
// ----------------------------------------------------------------------
//
// execute_graph_oneshot() 使用共享 QueueCache，而 replayable graph 使用私有
// queue。同步 oneshot 返回后，共享 queue 不应仍可 enqueu_batch；否则下一条
// eager enqueu_kernel 会把 stale batch 作废并打印 batch-invalidated WARN。
void test_oneshot_discards_shared_queue_batch(const ManagedDdrFixture& dma) {
    section("non-replayable oneshot discards shared QueueCache batch");

    RpuGraphCache cache;
    auto sig = make_sig(/*op_id=*/8005);
    auto& g = cache.get_or_create(sig);

    g.begin(sig);
    g.record_dma_fixed(dma.addr(0x0000), dma.addr(0x0400),
                       /*bytes=*/64, /*channel=*/0);
    g.mark_non_replayable("queue discard regression");
    g.end();

    EXPECT(g.state() == State::PASSTHROUGH,
           "non-replayable end returns graph to PASSTHROUGH");
    auto* shared_queue = QueueCache::instance().get(/*core_num=*/8);
    EXPECT(shared_queue->enqueu_batch(/*wait_finish=*/true) != 0,
           "oneshot leaves no replayable batch on shared QueueCache");
    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

void test_oneshot_discards_shared_queue_batch_after_late_failure(
        const ManagedDdrFixture& dma) {
    section("failed multi-segment oneshot discards shared QueueCache batch");

    QueueCache::instance().clear();
    RpuGraphCache cache;
    auto sig = make_sig(/*op_id=*/8006);
    auto& g = cache.get_or_create(sig);
    uint64_t invalid_src = UINT64_C(0xffffffffffff0000);
    std::vector<uint8_t> host_src(32, 0x10);
    std::vector<uint8_t> host_dst(32, 0x20);

    g.begin(sig);
    g.record_dma_fixed(dma.addr(0x0000), dma.addr(0x0400),
                       /*bytes=*/64, /*channel=*/0);
    MemcpyNodeData boundary{};
    boundary.dst = host_dst.data();
    boundary.src = host_src.data();
    boundary.bytes = host_src.size();
    boundary.kind = CopyKind::HOST_MEMCPY;
    g.capture_data(boundary);
    g.record_dma_mutable_src(&invalid_src, /*offset=*/0, dma.addr(0x0800),
                             /*bytes=*/64, /*channel=*/0);
    g.mark_non_replayable("late prepare failure cleanup regression");

    EXPECT(
        throws_with_message(
            [&] { g.end(); },
            "device DDR DMA range is outside every live logical allocation or SPM_ALLOC root"),
        "second segment fails after the first segment completed");
    EXPECT(host_dst == host_src,
           "the host boundary executes before the later segment fails");
    g.abort();

    auto* shared_queue = QueueCache::instance().get(/*core_num=*/8);
    EXPECT(shared_queue->enqueu_batch(/*wait_finish=*/true) != 0,
           "failed oneshot leaves no replayable batch on shared QueueCache");
    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

// ----------------------------------------------------------------------
// ⑧: PASSTHROUGH direct launch 不触碰 immediate DMA 的 QueueCache batch
// ----------------------------------------------------------------------
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
void test_passthrough_kernel_queue_isolation() {
    section("PASSTHROUGH kernel preserves immediate QueueCache batch");

    QueueCache::instance().clear();
    auto* immediate_q = QueueCache::instance().get(/*core_num=*/1);

    // 留一条 mutable DMA 在全局 immediate queue 的 prepared batch 中。随后
    // PASSTHROUGH kernel 若误用同一 Queue_t,无论 direct launch 还是重建
    // one-kernel batch,都会让 dma_id=0 失效。
    EXPECT(immediate_q->add_dma_kernel_mutable(
               /*src_addr=*/0x10000000ull,
               /*dst_addr=*/0x40000000ull,
               /*size=*/64,
               /*channel=*/0) == 0,
           "prepare sentinel mutable DMA");
    EXPECT(immediate_q->build_batch() == 0,
           "build sentinel immediate batch");

    std::vector<char> instr(64, 0);
    ::rhino_lkn::Kernel_t kernel(instr.data(), instr.size());
    auto& passthrough = RpuKernelGraph::active();
    EXPECT(passthrough.state() == State::PASSTHROUGH,
           "outside capture uses PASSTHROUGH graph");
    passthrough.enqueue(kernel,
                        /*grid_dims=*/{1, 1, 1},
                        /*core_ids=*/{0});

    EXPECT(immediate_q->update_dma_kernel(
               /*dma_id=*/0,
               /*src_addr=*/0x10000040ull,
               /*dst_addr=*/0x40000040ull) == 0,
           "PASSTHROUGH launch leaves immediate prepared batch intact");
    EXPECT(immediate_q->enqueu_batch(/*wait_finish=*/true) == 0,
           "preserved immediate batch remains launchable");

    QueueCache::instance().clear();
}
#endif

// ----------------------------------------------------------------------
// ⑧: Graph-invisible immediate DMA 的 active-stack 准入只保留单层
// raw-kernel 降级后的 plain PASSTHROUGH 兼容后缀。
// ----------------------------------------------------------------------
void test_direct_immediate_dma_active_stack_admission(
        const ManagedDdrFixture& dma) {
    section("direct/immediate DMA active-stack admission");

    bool outside_allowed = true;
    try {
        RpuKernelGraph::check_direct_immediate_dma_allowed("outside canary");
    } catch (const std::exception&) {
        outside_allowed = false;
    }
    EXPECT(outside_allowed, "no active Graph admits direct/immediate DMA");

    {
        RpuKernelGraph recording;
        recording.begin(make_sig(/*op_id=*/8010));
        EXPECT(
            throws_with_message(
                [&] {
                    rpu_launch_ddr_spm_bytes_immediate(
                        /*ddr_ptr=*/nullptr, /*num_bytes=*/16,
                        /*spm_addr_unified=*/0);
                },
                "direct/immediate DMA is forbidden with an active Graph"),
            "RECORDING rejects immediate DMA before pointer/address/SDK work");
        recording.abort();
    }

    {
        RpuKernelGraph replaying;
        const auto sig = make_sig(/*op_id=*/8014);
        replaying.begin(sig);
        replaying.record_dma_fixed(
            dma.addr(0x0000), dma.addr(0x0400),
            /*bytes=*/64, /*channel=*/0);
        replaying.end();
        replaying.begin(sig);
        EXPECT(replaying.state() == State::REPLAYING,
               "same signature enters REPLAYING");
        const GraphOpStreamStamp before = replaying.op_stream_stamp();
        EXPECT(
            throws_with_message(
                [&] {
                    rpu_launch_ddr_spm_bytes_immediate(
                        /*ddr_ptr=*/nullptr, /*num_bytes=*/16,
                        /*spm_addr_unified=*/0);
                },
                "direct/immediate DMA is forbidden with an active Graph"),
            "REPLAYING rejects immediate DMA before pointer/address/SDK work");
        const GraphOpStreamStamp after = replaying.op_stream_stamp();
        EXPECT(before.graph == after.graph &&
                   before.state == after.state &&
                   before.build_generation == after.build_generation &&
                   before.position == after.position &&
                   before.extent == after.extent,
               "REPLAYING rejection leaves graph cursor/topology unchanged");
        replaying.abort();
    }

    {
        RpuKernelGraph outer;
        RpuKernelGraph inner;
        outer.begin(make_sig(/*op_id=*/8011));
        inner.begin(make_sig(/*op_id=*/8012));
        EXPECT(
            throws_with_message(
                [&] {
                    rpu_launch_ddr_spm_bytes_immediate(
                        /*ddr_ptr=*/nullptr, /*num_bytes=*/16,
                        /*spm_addr_unified=*/0);
                },
                "direct/immediate DMA is forbidden with an active Graph"),
            "nested active stack rejects Graph-invisible immediate DMA");
        inner.abort();
        outer.abort();
    }

#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
    {
        RpuKernelGraph downgraded;
        downgraded.begin(make_sig(/*op_id=*/8013));
        std::vector<char> instr(64, 0);
        ::rhino_lkn::Kernel_t raw_kernel(instr.data(), instr.size());
        // No pending KernelId/name: this takes the checked raw-kernel fallback,
        // leaving the sole still-open scope in plain PASSTHROUGH. LKN_SKIP=1
        // prevents device submission; this section emits no DMA.
        downgraded.enqueue(raw_kernel,
                           /*grid_dims=*/{1, 1, 1},
                           /*core_ids=*/{0});
        EXPECT(downgraded.state() == State::PASSTHROUGH &&
                   RpuKernelGraph::has_active(),
               "raw-kernel fallback leaves one open PASSTHROUGH scope");

        bool suffix_allowed = true;
        try {
            RpuKernelGraph::check_direct_immediate_dma_allowed(
                "downgraded suffix canary");
        } catch (const std::exception&) {
            suffix_allowed = false;
        }
        EXPECT(suffix_allowed,
               "single plain downgraded PASSTHROUGH suffix is admitted");
        EXPECT(
            throws_with_message(
                [&] {
                    rpu_launch_ddr_spm_bytes_immediate(
                        /*ddr_ptr=*/nullptr, /*num_bytes=*/16,
                        /*spm_addr_unified=*/0);
                },
                "ddr_spm_bytes_immediate: null ddr_ptr"),
            "admitted suffix reaches argument validation without DMA");
        downgraded.end();
    }
#endif

    EXPECT(!RpuKernelGraph::has_active() && SPM_POOL.defer_depth() == 0,
           "direct/immediate DMA admission tests leave lifecycle balanced");
}

// ----------------------------------------------------------------------
// Process ownership is scope-bound, not a permanent BUILD-thread pin.
// Sequential BUILD/REPLAY may move threads, while every foreign operation
// during an open scope must reject before touching Graph, allocator, or SDK
// state. Cleanup may bind a different thread only after quiescence.
// ----------------------------------------------------------------------
void test_process_execution_thread_ownership() {
    section("process execution ownership across host threads");

    RpuKernelGraph sequential;
    const auto sequential_sig = make_sig(/*op_id=*/8020);
    std::exception_ptr build_error;
    std::thread build_thread([&] {
        try {
            sequential.begin(sequential_sig);
            sequential.record_branch(
                /*branch_key=*/7, GraphSignature{}, "thread handoff");
            sequential.end();
        } catch (...) {
            build_error = std::current_exception();
            try {
                sequential.abort();
            } catch (...) {
            }
        }
    });
    build_thread.join();
    EXPECT(build_error == nullptr && sequential.state() == State::BUILT,
           "BUILD completes on the first host thread");

    std::exception_ptr replay_error;
    std::thread replay_thread([&] {
        try {
            sequential.begin(sequential_sig);
            sequential.record_branch(
                /*branch_key=*/7, GraphSignature{}, "thread handoff");
            sequential.end();
        } catch (...) {
            replay_error = std::current_exception();
            try {
                sequential.abort();
            } catch (...) {
            }
        }
    });
    replay_thread.join();
    EXPECT(replay_error == nullptr && sequential.state() == State::BUILT,
           "stable REPLAY may bind a different sequential host thread");

    RpuGraphCache guarded_cache;
    guarded_cache.get_or_create(make_sig(/*op_id=*/8021));
    RpuKernelGraph owner_graph;
    owner_graph.begin(make_sig(/*op_id=*/8022));

    bool foreign_end_rejected = false;
    bool foreign_abort_rejected = false;
    bool foreign_direct_rejected = false;
    bool foreign_lookup_rejected = false;
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
    bool foreign_enqueue_rejected = false;
#endif
    bool foreign_begin_rejected = false;
    bool foreign_cleanup_rejected = false;
    bool foreign_reset_temporary_rejected = false;
    bool foreign_reset_all_rejected = false;
    bool foreign_spm_init_rejected = false;
    bool foreign_conv_cache_release_rejected = false;
    const uint64_t allocator_generation_before = SPM_ALLOC.generation();
    const uint64_t persistent_generation_before =
        SPM_ALLOC.persistent_generation();
    const size_t temporary_before = SPM_ALLOC.temporary_used();
    std::thread foreign([&] {
        foreign_end_rejected = throws_with_message(
            [&] { owner_graph.end(); }, "owned by another thread");
        foreign_abort_rejected = throws_with_message(
            [&] { owner_graph.abort(); }, "owned by another thread");
        foreign_direct_rejected = throws_with_message(
            [&] {
                RpuKernelGraph::check_direct_immediate_dma_allowed(
                    "foreign direct canary");
            },
            "owned by another thread");
        foreign_lookup_rejected = throws_with_message(
            [&] {
                (void)RpuKernelGraph::active().get_kernel_reset(
                    KernelId::BINARY_SAMESHAPE);
            },
            "owned by another thread");
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
        foreign_enqueue_rejected = throws_with_message(
            [&] {
                std::vector<char> instr(64, 0);
                ::rhino_lkn::Kernel_t kernel(instr.data(), instr.size());
                RpuKernelGraph::active().enqueue(
                    kernel, /*grid_dims=*/{1, 1, 1}, /*core_ids=*/{0});
            },
            "owned by another thread");
#endif
        foreign_begin_rejected = throws_with_message(
            [&] {
                RpuKernelGraph other;
                other.begin(make_sig(/*op_id=*/8023));
            },
            "owned by another thread");
        foreign_cleanup_rejected = throws_with_message(
            [&] { guarded_cache.clear(); },
            "requires no process-active Graph");
        foreign_reset_temporary_rejected = throws_with_message(
            [&] { graph_aware_spm_alloc_reset_temporary(); },
            "owned by another thread");
        foreign_reset_all_rejected = throws_with_message(
            [&] { process_guarded_spm_alloc_reset_all(); },
            "requires no process-active Graph");
        foreign_spm_init_rejected = throws_with_message(
            [&] { process_guarded_spm_alloc_init(); },
            "requires no process-active Graph");
        foreign_conv_cache_release_rejected = throws_with_message(
            [&] { rpu_release_conv2d_spm_cache(); },
            "requires no process-active Graph");
    });
    foreign.join();

    EXPECT(foreign_end_rejected && foreign_abort_rejected,
           "wrong-thread end/abort reject without consuming the owner claim");
    EXPECT(foreign_direct_rejected && foreign_lookup_rejected,
           "foreign direct DMA and PASSTHROUGH lookup reject");
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
    EXPECT(foreign_enqueue_rejected,
           "foreign raw-kernel enqueue rejects");
#endif
    EXPECT(foreign_begin_rejected && foreign_cleanup_rejected,
           "foreign Graph begin and destructive cleanup reject while active");
    EXPECT(foreign_reset_temporary_rejected && foreign_reset_all_rejected &&
               foreign_spm_init_rejected &&
               foreign_conv_cache_release_rejected,
           "foreign allocator/cache mutations reject before their side effects");
    EXPECT(SPM_ALLOC.generation() == allocator_generation_before &&
               SPM_ALLOC.persistent_generation() ==
                   persistent_generation_before &&
               SPM_ALLOC.temporary_used() == temporary_before,
           "foreign allocator/cache rejections leave allocator state unchanged");
    EXPECT(owner_graph.state() == State::RECORDING &&
               RpuKernelGraph::has_active() && SPM_POOL.defer_depth() == 1,
           "foreign rejections leave owner Graph/TLS/SPM state unchanged");
    owner_graph.abort();

    bool cleanup_succeeded = false;
    std::thread cleanup_thread([&] {
        try {
            guarded_cache.clear();
            cleanup_succeeded = true;
        } catch (...) {
        }
    });
    cleanup_thread.join();
    EXPECT(cleanup_succeeded && guarded_cache.size() == 0,
           "cleanup binds another thread after process quiescence");
    EXPECT(!RpuKernelGraph::has_active() && SPM_POOL.defer_depth() == 0,
           "thread ownership test leaves Graph/SPM lifecycle balanced");
}

// Must remain the final test: a shutdown claim is permanently sticky even
// after its guard exits, so no later test may touch execution/runtime state.
void test_shutdown_is_permanently_sticky() {
    section("shutdown claim is permanently sticky");

    bool nested_cleanup_allowed = false;
    {
        RpuExecutionCleanupGuard shutdown(
            "shutdown sticky canary", /*shutting_down=*/true);
        try {
            RpuExecutionCleanupGuard nested("nested shutdown cleanup");
            nested_cleanup_allowed = true;
        } catch (...) {
        }
    }
    EXPECT(nested_cleanup_allowed,
           "same-thread nested cleanup is allowed inside shutdown");

    RpuKernelGraph graph;
    EXPECT(throws_with_message(
               [&] { graph.begin(make_sig(/*op_id=*/8024)); },
               "Graph execution is forbidden during process runtime cleanup"),
           "Graph execution remains rejected after shutdown guard exits");
    EXPECT(throws_with_message(
               [&] {
                   (void)RpuExecutionCoordinator::enter_physical(
                       "post-shutdown physical canary");
               },
               "requires a process-quiescent"),
           "physical arena acquisition remains rejected after shutdown");
    EXPECT(throws_with_message(
               [&] {
                   RpuKernelGraph::check_direct_immediate_dma_allowed(
                       "post-shutdown direct canary");
               },
               "execution is forbidden during runtime cleanup"),
           "direct/PASSTHROUGH execution remains rejected after shutdown");
    EXPECT(throws_with_message(
               [&] {
                   RpuExecutionCleanupGuard cleanup(
                       "post-shutdown cleanup canary");
               },
               "runtime is permanently shut down"),
           "top-level non-shutdown cleanup cannot re-enter torn-down runtime");
    EXPECT(throws_with_message(
               [&] {
                   RpuExecutionCleanupGuard shutdown_again(
                       "post-shutdown repeat canary", /*shutting_down=*/true);
               },
               "runtime is permanently shut down"),
           "top-level shutdown claim itself cannot re-enter after teardown");
}

void test_owned_dma_roundtrip_and_mutable_replay() {
    section("Tensor-owner DMA payloads, mutable addresses and stable REPLAY");
    constexpr int cores = 8;
    constexpr int64_t n = 128, offset = 128, total = 8192;
    constexpr int64_t stride = n * sizeof(c10::Half);
    const c10::Half sentinel(-123.0f);
    const auto temporary_before = SPM_ALLOC.temporary_mark();
    SPM_ALLOC.init();
    {
        // These are real allocator-owned PrivateUse1 Tensors, not raw pointer
        // fixtures or graph.record_dma_* stand-ins for the owner overloads.
        auto tensor = [] {
            return at::Tensor(at::detail::empty_strided_generic(
                {total}, {1}, &rpu::RPUCachingAllocatorAdapter::get(),
                c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), at::kHalf));
        };
        at::Tensor sources[2] = {tensor(), tensor()};
        at::Tensor destinations[2] = {tensor(), tensor()};
        auto fixed_destination = tensor();
        EXPECT(sources[0].data_ptr() != sources[1].data_ptr() &&
                   destinations[0].data_ptr() != destinations[1].data_ptr(),
               "mutable source and destination own distinct live allocations");
        SpmAllocator::ScopedTemporary spm_scope;
        const uint32_t a = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(stride));
        const uint32_t b = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(stride));
        RpuGraphCache cache(/*max_entries=*/1);
        const auto sig = make_sig(/*op_id=*/8099);
        uint64_t live_src = 0, live_dst = 0, build_generation = 0;
        uint64_t graph_lifetime = 0;
        EXPECT(throws_with_message([&] {
                   rpu_launch_ddr_broadcast_spm_dma(at::Tensor(), offset, n, a, cores);
               }, "owner must be a contiguous FP16 device Tensor"),
               "undefined Tensor owner fails before submitting DMA");
        EXPECT(throws_with_message([&] {
                   rpu_launch_spm_copy_ddr_dma(a, fixed_destination, total, n);
               }, "exceeds Tensor bytes"),
               "fixed destination slice exceeding its owner fails closed");
        EXPECT(throws_with_message([&] {
                   rpu_launch_ddr_scatter_spm_dma_mutable(
                       live_src, sources[0], total * sizeof(c10::Half), n, stride, a, cores);
               }, "exceeds Tensor bytes"),
               "mutable source burst exceeding its owner fails closed");
        for (int iteration = 0; iteration < 3; ++iteration) {
            const int which = iteration % 2;  // A -> B -> A, same signature.
            for (int side = 0; side < 2; ++side) {
                auto* input = sources[side].data_ptr<c10::Half>();
                for (int64_t i = 0; i < total; ++i)
                    input[i] = c10::Half((i % 97 - 48) * 0.125f +
                                         (which + 2 * side) * 0.5f);
                std::fill_n(destinations[side].data_ptr<c10::Half>(), total, sentinel);
                ::rhino_lkn::RpuDdrFlush(input);
                ::rhino_lkn::RpuDdrFlush(destinations[side].data_ptr());
            }
            std::fill_n(fixed_destination.data_ptr<c10::Half>(), total, sentinel);
            ::rhino_lkn::RpuDdrFlush(fixed_destination.data_ptr());
            live_src = ::rhino_lkn::RpuGetDevAddr(sources[which].data_ptr());
            live_dst = ::rhino_lkn::RpuGetDevAddr(destinations[which].data_ptr());

            // Native RpuGraphCache has no Python warmup/freeze wrapper: after
            // the sole BUILD, use lookup only and reject any missing entry.
            auto* graph = iteration == 0 ? &cache.get_or_create(sig) : cache.lookup(sig);
            if (graph == nullptr) throw std::runtime_error("owned DMA replay cache miss");
            graph->begin(sig);
            EXPECT(graph->state() == (iteration == 0 ? State::RECORDING : State::REPLAYING),
                   "one BUILD followed by REPLAY only");
            const bool replay = graph->state() == State::REPLAYING;
            try {
                rpu_launch_ddr_broadcast_spm_dma(sources[0], offset, n, a, cores);
                rpu_launch_spm_local_copy_dma(a, n, b, cores);
                rpu_launch_spm_scatter_ddr_dma(b, fixed_destination, offset, n, stride, cores);
                rpu_launch_spm_copy_ddr_dma(b, fixed_destination, offset + cores * n, n);
                rpu_launch_ddr_scatter_spm_dma(sources[0], offset, n, stride, a, cores);
                rpu_launch_spm_scatter_ddr_dma(a, fixed_destination, offset + 2 * cores * n,
                                             n, stride, cores);
                rpu_launch_ddr_broadcast_spm_dma_mutable(
                    live_src, sources[which], offset * sizeof(c10::Half), n, a, cores);
                rpu_launch_spm_copy_ddr_dma_mutable(
                    a, live_dst, destinations[which], offset * sizeof(c10::Half), n);
                rpu_launch_spm_scatter_ddr_dma(a, fixed_destination, offset + 3 * cores * n,
                                             n, stride, cores);
                rpu_launch_ddr_scatter_spm_dma_mutable(
                    live_src, sources[which], offset * sizeof(c10::Half), n, stride, a, cores);
                rpu_launch_spm_scatter_ddr_dma(a, fixed_destination, offset + 4 * cores * n,
                                             n, stride, cores);
                rpu_launch_spm_copy_ddr_dma_mutable(
                    a, live_dst, destinations[which], (offset + 2 * n) * sizeof(c10::Half), n);
                graph->end();
                cache.touch_entry(sig, replay);
            } catch (...) {
                graph->abort();
                cache.clear();
                throw;
            }

            std::vector<c10::Half> expected_fixed(total, sentinel), expected_mutable(total, sentinel);
            const auto* fixed_source = sources[0].data_ptr<c10::Half>() + offset;
            const auto* mutable_source = sources[which].data_ptr<c10::Half>() + offset;
            for (int core = 0; core < cores; ++core) {
                std::copy_n(fixed_source, n, expected_fixed.begin() + offset + core * n);
                std::copy_n(mutable_source, n,
                            expected_fixed.begin() + offset + (3 * cores + core) * n);
            }
            std::copy_n(fixed_source, n, expected_fixed.begin() + offset + cores * n);
            std::copy_n(fixed_source, cores * n, expected_fixed.begin() + offset + 2 * cores * n);
            std::copy_n(mutable_source, cores * n, expected_fixed.begin() + offset + 4 * cores * n);
            std::copy_n(mutable_source, n, expected_mutable.begin() + offset);
            std::copy_n(mutable_source, n, expected_mutable.begin() + offset + 2 * n);
            const auto check = [&](const at::Tensor& actual, const std::vector<c10::Half>& expected) {
                ::rhino_lkn::RpuDdrFlush(actual.data_ptr());
                EXPECT(actual.dim() == 1 && actual.numel() == total && actual.scalar_type() == at::kHalf,
                       "DMA output shape/dtype/nonempty boundary");
                const auto* values = actual.data_ptr<c10::Half>();
                EXPECT(std::all_of(values, values + total,
                                  [](c10::Half value) { return std::isfinite(float(value)); }),
                       "DMA output including guard rows is finite");
                EXPECT(std::memcmp(values, expected.data(), total * sizeof(c10::Half)) == 0,
                       "DMA payload and untouched guard rows are byte-exact (direction + magnitude)");
            };
            check(fixed_destination, expected_fixed);
            check(destinations[which], expected_mutable);
            check(destinations[1 - which], std::vector<c10::Half>(total, sentinel));

            const auto snapshots = cache.snapshot();
            EXPECT(cache.size() == 1 && cache.cache_invariant_ok() && snapshots.size() == 1,
                   "single signature cache size and invariant remain fixed");
            if (snapshots.size() != 1) throw std::runtime_error("owned DMA snapshot missing");
            const auto& stats = snapshots.front();
            if (iteration == 0) {
                build_generation = stats.build_generation;
                graph_lifetime = stats.graph_lifetime_id;
            }
            EXPECT(stats.dma_count == 75 && stats.barrier_count > 0 && stats.segment_count > 0,
                   "owned DMA graph is non-trivial on BUILD and REPLAY");
            EXPECT(build_generation > 0 && graph_lifetime > 0 &&
                       stats.build_generation == build_generation && stats.graph_lifetime_id == graph_lifetime &&
                       stats.execution_ordinal == uint64_t(iteration + 1) &&
                       stats.replay_count == size_t(iteration) && stats.recapture_count == 0,
                   "same Graph identity: exact ordinal/replay increase without second BUILD or recapture");
            EXPECT(!graph_dma::active() && SPM_POOL.defer_depth() == 0,
                   "completed DMA Graph returns to PASSTHROUGH without deferred SPM");
        }
        // Release the Graph while every captured tensor/live slot/SPM allocation
        // remains alive; cleanup errors must fail, never be swallowed.
        cache.clear();
        EXPECT(cache.size() == 0 && cache.cache_invariant_ok() && !graph_dma::active(),
               "owned DMA Graph is strictly reclaimed before backing storage");
    }
    EXPECT(SPM_ALLOC.temporary_mark() == temporary_before && SPM_POOL.defer_depth() == 0,
           "owned DMA roundtrip restores its exact temporary SPM watermark");
}

}  // namespace

int main() {
    const char* lkn_skip = std::getenv("LKN_SKIP");
    if (lkn_skip == nullptr || std::strcmp(lkn_skip, "1") != 0 ||
        ::geteuid() != 0) {
        std::fprintf(
            stderr,
            "[FATAL] this managed-DDR Graph test requires root and explicit "
            "process-start LKN_SKIP=1 acknowledgement (Release still submits "
            "hardware)\n");
        return 2;
    }
    std::printf("[main] start\n"); std::fflush(stdout);
    test_direct_allocator_dma_registry();
    {
        const ManagedDdrFixture dma;
        test_legacy_id0_hash_and_replay_behavior(dma);
        test_semantic_dma_topology_schema_goldens();
        test_dma_id_monotone_across_segments(dma);
        test_single_segment_sync_only_hit(dma);
        test_multi_segment_rebuild_each_replay(dma);
        test_resource_budget_preserves_dma_fence(dma);
        test_lingbot2_exact_graph_uses_only_sealed_sdk_budget(dma);
        test_wall_oss_wrc_exact_graph_keeps_generic_entry_limit(dma);
        test_oneshot_discards_shared_queue_batch(dma);
        test_oneshot_discards_shared_queue_batch_after_late_failure(dma);
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
        test_passthrough_kernel_queue_isolation();
#endif
        test_direct_immediate_dma_active_stack_admission(dma);
        test_process_execution_thread_ownership();
        test_owned_dma_roundtrip_and_mutable_replay();
    }
    rpu::empty_cache();
    test_shutdown_is_permanently_sticky();

    std::printf("\n=== Summary ===\n");
    std::printf("Passed: %d\n", g_passed);
    std::printf("Failed: %d\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}

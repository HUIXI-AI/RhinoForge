"""Host-only regressions for 65536 SDK batch ceiling and cross-process SPM safety."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]


def test_concurrent_safe_spm_env_and_forward_residency_gate(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    runtime_state_cpp = (ROOT / "src/core/rpu_runtime_state.cpp").read_text(encoding="utf-8")
    env_fn = _definition(runtime_state_cpp, "bool rpu_concurrent_safe_spm_enabled(")
    fmb = (ROOT / "src/core/fused_model_base.h").read_text(encoding="utf-8")
    alloc_h = (ROOT / "src/core/rpu_spm_allocator.h").read_text(encoding="utf-8")
    alloc_cpp = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text(encoding="utf-8")
    fmb_cpp = (ROOT / "src/core/fused_model_base.cpp").read_text(encoding="utf-8")
    residency_h = (ROOT / "src/core/rpu_spm_residency.h").read_text(encoding="utf-8")

    source = """
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
#define RPU_RUNTIME_STATE_H_INCLUDED 1
""" + env_fn + """
class SpmAllocator { public:
static constexpr size_t ALIGN = 256;
static size_t align_up(size_t n) { return (n + ALIGN - 1) & ~(ALIGN - 1); }
"""
    for marker in ("struct AllocRequest", "struct AliasedPlan"):
        source += _definition(alloc_h, marker) + ";\n"
    source += "static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);};\n"
    source += _definition(alloc_cpp, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    source += "\nnamespace v3 { class FusedModelBase;\n"
    for marker in ("enum class StorageClass", "enum class BufferScope", "struct BufferDecl"):
        source += _definition(fmb, marker) + ";\n"
    source += "namespace detail { int64_t estimate_temporary_total(const std::vector<BufferDecl>&); }\n"
    source += _definition(fmb_cpp, "int64_t detail::estimate_temporary_total(") + "\n}\n"
    source += residency_h[residency_h.index("namespace v3::detail {"):]
    source += r"""
int main() {
    unsetenv("RPU_CONCURRENT_SAFE_SPM");
    assert(!rpu_concurrent_safe_spm_enabled());
    setenv("RPU_CONCURRENT_SAFE_SPM", "0", 1);
    assert(!rpu_concurrent_safe_spm_enabled());
    setenv("RPU_CONCURRENT_SAFE_SPM", "1", 1);
    assert(rpu_concurrent_safe_spm_enabled());
    setenv("RPU_CONCURRENT_SAFE_SPM", "invalid", 1);
    bool rejected = false;
    try { (void)rpu_concurrent_safe_spm_enabled(); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    using namespace v3;
    std::vector<BufferDecl> base{
        {"sdpa_mask", 4096, 1, 1, StorageClass::Temp, 0, nullptr, BufferScope::Compute, nullptr},
        {"scratch", 65536, 0, 2, StorageClass::Temp, 0, nullptr, BufferScope::LayerWide, nullptr},
    };
    unsetenv("RPU_CONCURRENT_SAFE_SPM");
    auto normal = base;
    assert(detail::plan_forward_spm_residency(normal, {"sdpa_mask"}).retained);
    assert(normal[0].scope == BufferScope::LayerWide);

    setenv("RPU_CONCURRENT_SAFE_SPM", "1", 1);
    auto safe = base;
    assert(!detail::plan_forward_spm_residency(safe, {"sdpa_mask"}).retained);
    assert(safe[0].scope == BufferScope::Compute);
    return 0;
}
"""
    cpp = tmp_path / "concurrent_safe_spm.cpp"
    exe = tmp_path / "concurrent_safe_spm"
    cpp.write_text(source, encoding="utf-8")
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Werror", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_denoise_unroll_batch_caps_and_eager_atomic_batches_are_wired() -> None:
    for rel_path in (
        "src/fused/rpu_lingbot_denoise_model.cpp",
        "src/fused/rpu_gr00t_dit_model.cpp",
        "src/fused/rpu_pi05_denoise_step_model.cpp",
        "src/fused/rpu_hyvla_expert_model.cpp",
        "src/fused/rpu_rhino_vla_model.cpp",
    ):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "32000" not in text, f"stale 32000 batch cap remains in {rel_path}"
        assert "65536" in text, f"65536 batch cap missing in {rel_path}"

    for rel_path in (
        "src/ops/rpu_linear.cpp",
        "src/ops/rpu_linear_w8a8.cpp",
        "src/ops/rpu_eltwise_unary.cpp",
        "src/ops/rpu_cast.cpp",
        "src/ops/rpu_embedding.cpp",
        "src/ops/rpu_gr00t_adaln.cpp",
        "src/ops/rpu_hyvla_merger.cpp",
    ):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "ScopedEagerAtomicBatch" in text, f"ScopedEagerAtomicBatch missing in {rel_path}"

    vision = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text(encoding="utf-8")
    assert "mark_segment_preload_prologue(" in vision
    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text(encoding="utf-8")
    assert "mark_safe_split_boundary()" in fmb
    assert "mark_segment_preload_prologue(" in fmb

    graph_h = (ROOT / "src/graph/graph_runtime.h").read_text(encoding="utf-8")
    assert "bool flush_icache = true;" in graph_h
    for op_cpp in (ROOT / "src/ops").glob("*.cpp"):
        txt = op_cpp.read_text(encoding="utf-8")
        for line in txt.splitlines():
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            assert "set_flush_icache(false)" not in stripped, (
                f"stale set_flush_icache(false) in {op_cpp.name}: {stripped}"
            )


def test_segment_prologue_spans_census_and_fenced_burst_rewind(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    execute_cpp = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text(encoding="utf-8")
    graph_h = (ROOT / "src/graph/graph_runtime.h").read_text(encoding="utf-8")
    graph_cpp = (ROOT / "src/graph/graph_runtime.cpp").read_text(encoding="utf-8")
    child_snippet = (ROOT / "src/graph/graph_runtime_child.cpp").read_text(encoding="utf-8")
    assert "preload_prologue_spans_" in child_snippet
    assert "safe_split_boundaries_.clear();" in execute_cpp

    seg_for_each = _definition(graph_h, "void for_each_node_index(")
    mark_prologue = _definition(graph_cpp, "void RpuKernelGraph::mark_segment_preload_prologue(")
    mark_safe = _definition(graph_cpp, "void RpuKernelGraph::mark_safe_split_boundary(")
    build_segs = _definition(execute_cpp, "void RpuKernelGraph::build_segments_from_nodes(")

    source = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
namespace rhino_lkn { struct Kernel_t {}; }
bool log_at(int) { return false; }
inline void validate_graph_dma_channel(int, size_t, const char*) {}
enum class CopyKind { HOST_MEMCPY, DDR_TO_DDR };
struct MemcpyNodeData { CopyKind kind = CopyKind::HOST_MEMCPY; int dma_channel = -1; };
struct GraphRuntimePolicy {
    bool siglip_isolate_patch_embed = false;
    int execution_core_count = 8;
};
struct GraphSignature {
    std::string op_id_str;
    std::vector<int64_t> dyn_dims;
};
struct QueueLaunchState {
    bool broadcast_mode = true;
    bool flush_icache = true;
    bool operator!=(const QueueLaunchState& rhs) const {
        return broadcast_mode != rhs.broadcast_mode || flush_icache != rhs.flush_icache;
    }
};
enum class GraphNodeKind { Kernel, Dma, Barrier, ChildGraph, Memcpy, Branch };
struct RpuKernelGraph;
struct KernelNodeData {
    std::vector<uint8_t> core_ids{0};
    QueueLaunchState queue_state;
    bool isolate = false;
};
struct DmaNodeData { uint8_t channel = 0; };
struct BarrierNodeData { uint8_t self_stream = 0, target_stream = 0; };
struct BranchNodeData {
    bool metadata = true;
    bool is_physical_manifest_metadata() const { return metadata; }
};
struct ChildGraphNodeData { std::shared_ptr<RpuKernelGraph> child; };
struct GraphNode {
    GraphNodeKind kind = GraphNodeKind::Kernel;
    KernelNodeData kn;
    DmaNodeData dma;
    BarrierNodeData bar;
    BranchNodeData br;
    ChildGraphNodeData cg;
    std::variant<MemcpyNodeData> data;
    KernelNodeData& as_kernel() { return kn; }
    const KernelNodeData& as_kernel() const { return kn; }
    const DmaNodeData& as_dma() const { return dma; }
    const BarrierNodeData& as_barrier_node() const { return bar; }
    const BranchNodeData& as_branch() const { return br; }
    const ChildGraphNodeData& as_child_graph() const { return cg; }
};
struct GraphSegmentCensus {
    size_t segment_id = 0, start_idx = 0, end_idx = 0, core_count = 0;
    size_t kernel_count = 0, dma_count = 0, barrier_count = 0;
    size_t manifest_metadata_count() const {
        return (end_idx - start_idx) - (kernel_count + dma_count + barrier_count);
    }
};
struct Segment {
    size_t segment_id = 0, start_idx = 0, end_idx = 0;
    std::vector<uint8_t> core_ids;
    QueueLaunchState queue_state;
    GraphSegmentCensus census;
    std::vector<std::pair<size_t, size_t>> prologue_spans;
    template <typename Fn>
""" + seg_for_each + r"""
};
struct GraphStats {
    size_t dma_count = 0, barrier_count = 0, child_graph_count = 0;
    std::vector<uint64_t> child_graph_lifetime_ids;
    std::vector<GraphSegmentCensus> segment_census;
};
struct SegmentResourceBudget {
    size_t entries = 6;
    size_t kd_bytes = 8u << 20;
    size_t instr_bytes = 64u << 20;
};
static size_t g_test_budget_entries = 6;
SegmentResourceBudget segment_resource_budget(const std::string&, const GraphRuntimePolicy&, bool) {
    return {g_test_budget_entries, 8u << 20, 64u << 20};
}
bool causal_14b_c4_initial_prefill_needs_small_segments(const GraphSignature*, const GraphRuntimePolicy&) {
    return false;
}
struct SegmentResourceUse {
    size_t entries = 0, kd_bytes = 0, instr_bytes = 0;
    uint32_t active_streams = 0;
};
SegmentResourceUse segment_node_resources(const GraphNode& n, const std::vector<std::unique_ptr<rhino_lkn::Kernel_t>>&) {
    if (n.kind == GraphNodeKind::Branch) return {};
    return {1, 64, n.kind == GraphNodeKind::Kernel ? 4096u : 0u, 1u};
}
size_t segment_new_completion_bytes(const SegmentResourceUse&, const SegmentResourceUse&) { return 0; }
bool segment_would_exceed(const SegmentResourceUse& cur, const SegmentResourceUse& add, const SegmentResourceBudget& b) {
    return cur.entries + add.entries > b.entries ||
           cur.kd_bytes + add.kd_bytes > b.kd_bytes ||
           cur.instr_bytes + add.instr_bytes > b.instr_bytes;
}
uint8_t validated_kernel_core_extent(const GraphRuntimePolicy&, const std::vector<uint8_t>& c) {
    return static_cast<uint8_t>(c.size());
}
bool p7_should_force_isolate_kernel(const KernelNodeData& k) { return k.isolate; }
bool siglip_should_isolate_patch_embed_kernel(const std::string&, const KernelNodeData&, bool) { return false; }
std::string kernel_label_for_segment_split(const KernelNodeData&) { return "test"; }

struct RpuKernelGraph {
    enum class State { PASSTHROUGH, RECORDING, BUILT, REPLAYING };
    State state_ = State::RECORDING;
    GraphRuntimePolicy runtime_policy_;
    bool require_single_segment_ = false;
    bool has_built_signature_ = false;
    GraphSignature built_signature_;
    std::optional<GraphSignature> pending_signature_;
    uint64_t graph_lifetime_id_ = 1;
    std::vector<std::unique_ptr<rhino_lkn::Kernel_t>> kernels_;
    std::vector<GraphNode> nodes_;
    std::vector<Segment> segments_;
    std::vector<size_t> safe_split_boundaries_;
    std::vector<std::pair<size_t, size_t>> preload_prologue_spans_;
    GraphStats last_stats_;
    void mark_safe_split_boundary();
    void mark_segment_preload_prologue(size_t begin_idx, size_t end_idx);
    void build_segments_from_nodes();
};
""" + mark_safe + "\n" + mark_prologue + "\n" + build_segs + r"""
static GraphNode make_kn() { GraphNode n{}; n.kind = GraphNodeKind::Kernel; return n; }
static GraphNode make_dma(uint8_t ch) { GraphNode n{}; n.kind = GraphNodeKind::Dma; n.dma.channel = ch; return n; }
static GraphNode make_wait(uint8_t ch) { GraphNode n{}; n.kind = GraphNodeKind::Barrier; n.bar.self_stream = 0; n.bar.target_stream = static_cast<uint8_t>(ch * 2); return n; }
static GraphNode make_meta() { GraphNode n{}; n.kind = GraphNodeKind::Branch; return n; }

int main() {
    // 1. Disjoint prologue spans + M1 census preservation:
    {
        g_test_budget_entries = 7;
        RpuKernelGraph g;
        g.nodes_.push_back(make_dma(1));   // 0: prologue span 1
        g.nodes_.push_back(make_wait(1));  // 1: prologue span 1
        g.mark_segment_preload_prologue(0, 2);
        g.nodes_.push_back(make_kn());     // 2: intermediate compute (MUST NOT be replayed in prologue!)
        g.nodes_.push_back(make_dma(2));   // 3: prologue span 2
        g.nodes_.push_back(make_wait(2));  // 4: prologue span 2
        g.mark_segment_preload_prologue(3, 5);
        assert(g.preload_prologue_spans_.size() == 2);

        g.nodes_.push_back(make_meta());   // 5: metadata branch
        g.nodes_.push_back(make_kn());     // 6: step 0
        g.mark_safe_split_boundary();      // safe split at 7
        g.nodes_.push_back(make_kn());     // 7: step 1 first kernel
        g.nodes_.push_back(make_meta());   // 8: metadata branch inside seg 1
        g.nodes_.push_back(make_kn());     // 9: step 1 second kernel (pushes past 7 entries -> splits at 7!)
        g.build_segments_from_nodes();
        assert(g.segments_.size() == 2);
        assert(g.segments_[0].prologue_spans.empty());
        assert(g.segments_[1].prologue_spans.size() == 2);
        // Verify node 2 is NOT visited in segment 1's prologue!
        std::vector<size_t> visited;
        g.segments_[1].for_each_node_index([&](size_t idx) { visited.push_back(idx); });
        assert((visited == std::vector<size_t>{0, 1, 3, 4, 7, 8, 9}));
        // Verify M1: seg 1 census strictly covers [7, 10) -> 2 kernels + 1 metadata!
        assert(g.segments_[1].census.kernel_count == 2);
        assert(g.segments_[1].census.dma_count == 0);
        assert(g.segments_[1].census.barrier_count == 0);
        assert(g.segments_[1].census.manifest_metadata_count() == 1);
    }

    // 2. C1: Fenced multi-stream DMA burst overflowing budget right before EOF / Memcpy
    // unwinds to last_safe_split / last_fence_closed_fit so NO segment exceeds budget!
    {
        g_test_budget_entries = 5;
        RpuKernelGraph g;
        g.nodes_.push_back(make_kn());     // 0
        g.nodes_.push_back(make_kn());     // 1
        g.mark_safe_split_boundary();      // safe split at 2
        g.nodes_.push_back(make_kn());     // 2 (fence-closed fit at 3)
        // Fenced burst of 4 entries [3..6] -> total [0..6] = 7 > 5, followed directly by EOF!
        g.nodes_.push_back(make_dma(1));   // 3
        g.nodes_.push_back(make_dma(2));   // 4
        g.nodes_.push_back(make_wait(1));  // 5
        g.nodes_.push_back(make_wait(2));  // 6 (closes fence right at EOF!)
        g.build_segments_from_nodes();
        assert(g.segments_.size() == 2);
        assert(g.segments_[0].start_idx == 0 && g.segments_[0].end_idx == 2);
        assert(g.segments_[1].start_idx == 2 && g.segments_[1].end_idx == 7);
    }
    return 0;
}
"""
    cpp = tmp_path / "prologue_and_c1.cpp"
    exe = tmp_path / "prologue_and_c1"
    cpp.write_text(source, encoding="utf-8")
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Werror", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_qwen3_vl_extended_vocab_row_partition_and_precomputed_video(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    import torch
    import rpu_backend.adapters.qwen3_vl as qvl
    from test_qwen3_vl_merger_lifecycle import _model_4b

    model = _model_4b()
    model.config.text_config.vocab_size = 156048
    model.config.vision_config.model_type = "qwen3_vl_vision"
    assert qvl._is_qwen3_vl_dense_eager_profile(model)

    model.config.text_config.vocab_size = 156050
    assert not qvl._is_qwen3_vl_dense_eager_profile(model)

    model.config.text_config.vocab_size = 151920
    assert not qvl._is_qwen3_vl_dense_eager_profile(model)

    recorded: list[tuple] = []
    monkeypatch.setattr(
        torch.ops,
        "rpu",
        SimpleNamespace(
            linear_with_accumulation=lambda x, w, b, part, cores, acc32, scale, **kw: (
                recorded.append((part, cores, acc32)) or torch.zeros((x.shape[0], x.shape[1], w.shape[0]), dtype=x.dtype)
            )
        ),
        raising=False,
    )
    head = SimpleNamespace(
        weight=torch.zeros((156048, 2560), dtype=torch.float16),
        bias=None,
        _rpu_linear_partition=0,
        _rpu_linear_num_cores=8,
    )
    dummy = SimpleNamespace(
        lm_head=head,
        model=SimpleNamespace(language_model=SimpleNamespace(_rpu_execution={})),
        config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=156048)),
    )
    hidden = torch.zeros((1, 2, 2560), dtype=torch.float16)
    logits = qvl._apply_prefill_lm_head(dummy, hidden, slice(-1, None))
    assert logits.shape == (1, 1, 156048)
    assert recorded == [(0, 8, False)]

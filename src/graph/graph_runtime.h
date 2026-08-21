// graph_runtime.h — RpuKernelGraph runtime and op-facing graph-aware helpers.
//
// This header contains the execution-facing graph API:
//   - RpuKernelGraph state machine surface and recorded node types
//   - RpuQueue / RpuQueueCache wrappers used by GET_QUEUE
//   - GET_KERNEL / GET_QUEUE macro overrides for op code
//   - graph-aware LocalSPM acquire/release helpers
//
// Cache keys, resource descriptors, SPM pools, and HostCallback tiering live in
// graph_infra.h so runtime depends on infra, not the other way around.
#pragma once

#include "rpu_ops.h"  // KernelId, KernelCache, QueueCache, g_rpu_ddr_flush_enabled 等
#include "graph/execution_coordinator.h"
#include "graph/graph_infra.h"

#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/stack.h>
#include <c10/core/Storage.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

// 前置声明:ChildGraphNodeData 内部要 shared_ptr<RpuKernelGraph>
class RpuKernelGraph;
class RpuPhysicalArenaAuthority;
class RpuRetainedPhysicalArenaGuard;
class GraphKernelRegisterWriter;
class GraphKernelRegisterCensusGuard;
class GraphKernelRegisterOuterFastTicket;
enum class GraphDdrRegisterRole : uint8_t;
struct GraphKernelRegisterPolicy;
// Fully defined (rather than an open-ended friend forward declaration) so no
// downstream translation unit can widen this board-free test seam with new
// private Graph mutations.  The non-mutating validation helpers exercise the
// typed-register contract without constructing an RPU tensor or touching the
// device.
class RpuKernelGraphContractTestAccess final {
public:
    RpuKernelGraphContractTestAccess() = delete;
    static void stage_pending_kernel(RpuKernelGraph& graph, KernelId id);
    static void stage_pending_kernel_name(
        RpuKernelGraph& graph, std::string name);
    static void clear_pending_kernel(RpuKernelGraph& graph);
    static void set_dma_semantics_for_topology_golden(
        RpuKernelGraph& graph, size_t node_idx, uint64_t endpoint_id,
        bool canonical_member, uint64_t spm_peer_id);
    static void validate_ddr_register_dtype(
        GraphDdrRegisterRole role, at::ScalarType dtype);
    static void validate_kernel_register_policy(
        const GraphKernelRegisterPolicy& policy);
    static uint64_t kernel_register_policy_digest(
        const GraphKernelRegisterPolicy& policy);
};
struct GraphOpStreamStamp;
namespace v3 {
class FusedModelBase;
class SpmCompositeTraceCoordinator;
class SpmPipelineLease;
struct SpmFmbYieldTargetLauncherAccess;
}

// Opaque proof minted only by a live, sealed physical SpmPipelineLease.  It
// binds the process claim to the actual allocator generation, persistent top,
// arena extent, and a lease-owned lifetime generation.  Callers can transport
// the token but cannot author or inspect its identity fields.
class RpuPhysicalArenaAuthority final {
public:
    RpuPhysicalArenaAuthority(const RpuPhysicalArenaAuthority&) = default;
    RpuPhysicalArenaAuthority& operator=(
        const RpuPhysicalArenaAuthority&) = default;

private:
    RpuPhysicalArenaAuthority(
        uint64_t physical_execution_token,
        uint64_t lease_epoch,
        uint64_t plan_hash,
        uint64_t allocator_generation,
        uint64_t persistent_generation,
        size_t arena_base,
        size_t arena_end,
        size_t reserved_top_bytes,
        const std::shared_ptr<uint64_t>& lease_generation,
        uint64_t expected_lease_generation,
        std::thread::id owner_thread)
        : physical_execution_token_(physical_execution_token),
          lease_epoch_(lease_epoch),
          plan_hash_(plan_hash),
          allocator_generation_(allocator_generation),
          persistent_generation_(persistent_generation),
          arena_base_(arena_base),
          arena_end_(arena_end),
          reserved_top_bytes_(reserved_top_bytes),
          lease_generation_(lease_generation),
          expected_lease_generation_(expected_lease_generation),
          owner_thread_(owner_thread) {}

    uint64_t physical_execution_token_ = 0;
    uint64_t lease_epoch_ = 0;
    uint64_t plan_hash_ = 0;
    uint64_t allocator_generation_ = 0;
    uint64_t persistent_generation_ = 0;
    size_t arena_base_ = 0;
    size_t arena_end_ = 0;
    size_t reserved_top_bytes_ = 0;
    std::weak_ptr<uint64_t> lease_generation_;
    uint64_t expected_lease_generation_ = 0;
    std::thread::id owner_thread_;

    friend class RpuKernelGraph;
    friend class RpuRetainedPhysicalArenaGuard;
    friend class v3::SpmPipelineLease;
};

// Move-only external owner for one BUILT Graph whose retained queue embeds a
// physical SPM arena.  The guard deliberately does not own the Graph or retire
// itself in its destructor: dropping it early leaves the process registry
// fail-closed.  A bound guard pins its physical lease.  A parked guard keeps
// only the immutable Graph/plan/placement identity, so temporary SPM may be
// reused by another subsystem before the exact plan is reacquired and rebound.
// The supported teardown is:
//
//   bound:  Graph invalidate/destruction -> guard.retire() -> lease release
//   parked: Graph invalidate/destruction -> guard.retire()
//
// No SPM address is exposed.  The registry binds stable Graph identity to the
// opaque authority minted by the live physical lease.
class RpuRetainedPhysicalArenaGuard final {
public:
    RpuRetainedPhysicalArenaGuard(
        const RpuRetainedPhysicalArenaGuard&) = delete;
    RpuRetainedPhysicalArenaGuard& operator=(
        const RpuRetainedPhysicalArenaGuard&) = delete;
    RpuRetainedPhysicalArenaGuard(
        RpuRetainedPhysicalArenaGuard&& other) noexcept;
    RpuRetainedPhysicalArenaGuard& operator=(
        RpuRetainedPhysicalArenaGuard&&) = delete;
    ~RpuRetainedPhysicalArenaGuard() = default;

    bool active() const noexcept { return active_; }
    bool bound() const noexcept {
        return active_ && physical_execution_token_ != 0;
    }
    uint64_t lease_epoch() const noexcept { return lease_epoch_; }
    uint64_t plan_hash() const noexcept { return plan_hash_; }
    uint64_t allocator_generation() const noexcept {
        return allocator_generation_;
    }
    uint64_t graph_build_generation() const noexcept {
        return graph_build_generation_;
    }

    // Temporarily detach this live BUILT Graph from its exact physical lease.
    // park() must precede lease release.  rebind() accepts only a newly minted
    // authority with the same plan, persistent generation, arena extent, and
    // reserved top; allocator generation and lease epoch may advance.  This
    // initial contract is thread-affine: park/rebind/retire must run on the
    // thread that armed the retained Graph.
    void park(const RpuPhysicalArenaAuthority& authority);
    void rebind(const RpuPhysicalArenaAuthority& authority);

    // Idempotent after success.  Before success it requires the exact bound
    // or parked Graph lifetime to have completed invalidate() or
    // destruction/eviction.
    void retire();

    // Explicit pre-side-effect hook for a physical lease owner.  The execution
    // coordinator also applies this gate to its quiescent/release paths, so the
    // current SpmPipelineLease release path is protected without a core-layer
    // dependency on this Graph type.
    static void require_release_allowed(
        uint64_t lease_epoch,
        uint64_t plan_hash,
        uint64_t allocator_generation,
        const char* operation);

private:
    RpuRetainedPhysicalArenaGuard(
        uint64_t token,
        uint64_t graph_lifetime_id,
        uint64_t graph_build_generation,
        uint64_t graph_signature_identity,
        uint64_t graph_topology_hash,
        uint64_t physical_execution_token,
        uint64_t lease_epoch,
        uint64_t plan_hash,
        uint64_t allocator_generation,
        uint64_t persistent_generation,
        size_t arena_base,
        size_t arena_end,
        size_t reserved_top_bytes,
        const std::weak_ptr<uint64_t>& lease_generation,
        uint64_t expected_lease_generation,
        std::thread::id owner_thread);

    void clear() noexcept;

    uint64_t token_ = 0;
    uint64_t graph_lifetime_id_ = 0;
    uint64_t graph_build_generation_ = 0;
    uint64_t graph_signature_identity_ = 0;
    uint64_t graph_topology_hash_ = 0;
    uint64_t physical_execution_token_ = 0;
    uint64_t lease_epoch_ = 0;
    uint64_t plan_hash_ = 0;
    uint64_t allocator_generation_ = 0;
    uint64_t persistent_generation_ = 0;
    size_t arena_base_ = 0;
    size_t arena_end_ = 0;
    size_t reserved_top_bytes_ = 0;
    std::weak_ptr<uint64_t> lease_generation_;
    uint64_t expected_lease_generation_ = 0;
    std::thread::id owner_thread_;
    bool active_ = false;

    friend class RpuKernelGraph;
};

// Only top-level graphs created by RpuGraphCache or Python's raw Graph
// constructor are registered. Child graphs are owned by their parent and are
// released by parent.invalidate(); registering them separately would make a
// process reset depend on parent/child destruction order.
std::shared_ptr<RpuKernelGraph> make_registered_rpu_kernel_graph();
void invalidate_registered_rpu_kernel_graphs();

// Public allocator bindings route through these process-coordinated helpers.
// reset_temporary may execute under the current thread's active Graph claim;
// init/reset_all require an exclusive cleanup claim.
void process_guarded_spm_alloc_init();
void graph_aware_spm_alloc_reset_temporary();
void process_guarded_spm_alloc_reset_all();

// =============================================================================
// QueueLaunchState — Op 端在 Queue 上设置的状态快照
// =============================================================================
// 用于在延迟执行（RECORDING/REPLAYING）模式下保存 Op 端通过 wq->set_xxx(...)
// 设置的 queue 配置，使其成为 replay signature 的一部分。
// 在 PASSTHROUGH 模式下用于立即写回底层 Queue_t。

struct QueueLaunchState {
    bool broadcast_mode = true;
    bool flush_icache = false;

    bool operator==(const QueueLaunchState& other) const {
        return broadcast_mode == other.broadcast_mode &&
               flush_icache == other.flush_icache;
    }
    bool operator!=(const QueueLaunchState& other) const {
        return !(*this == other);
    }
};

// =============================================================================
// Typed DDR-register provenance
// =============================================================================
//
// A kernel ID/name is not enough to prove what a packed register address means.
// The trusted launcher therefore stages one typed tensor occurrence before
// GET_KERNEL, and a move-only writer is the only object allowed to write the
// corresponding address registers.  Graph RECORDING retains the tensor owner;
// REPLAY validates the exact tensor/storage/address before get_kernel_reset can
// mutate the recorded Kernel_t.

enum class GraphDdrRegisterRole : uint8_t {
    ModelWeight = 1,
    KeyCache = 2,
    ValueCache = 3,
    SemanticInput = 4,
    // Signed-int8 model weights and their FP16 per-output-channel scales use
    // distinct roles so swapping either operand fails before address lookup.
    // Keep the legacy values above stable: frozen schema-5 policy digests
    // include the role byte.
    Int8ModelWeight = 5,
    PerChannelScale = 6,
};

enum class GraphDdrRegisterAccess : uint8_t {
    Read = 1,
    Write = 2,
};

enum class GraphDdrRegisterEncoding : uint8_t {
    DevAddrShift8LoHi = 1,
};

enum class GraphKernelRegisterRuleKind : uint8_t {
    TypedDdr = 1,
    IntrinsicNoDdr = 2,
    ExplicitNoDdr = 3,
};

enum class GraphKernelNoDdrProof : uint8_t {
    None = 0,
    AllReduceSpmResidualV2 = 1,
    AllGatherSpm = 2,
    // Launcher-staged only; a matching kernel name is not a proof.
    LegacyAllReduceSpmResidual = 3,
    FillSpm = 4,
    SliceSpm = 5,
    RingAllReduceSpmResidual = 6,
};

// DDR-facing side of one Graph-visible DMA burst.  The callable API carries
// only this direction plus a Tensor-owned byte range; callers never submit a
// device address, node index, or replay cursor.
enum class GraphOuterFastDmaSide : uint8_t {
    None = 0,
    Source = 1,
    Destination = 2,
};

struct GraphDdrRegisterAbi {
    GraphDdrRegisterRole role = GraphDdrRegisterRole::ModelWeight;
    GraphDdrRegisterAccess access = GraphDdrRegisterAccess::Read;
    GraphDdrRegisterEncoding encoding =
        GraphDdrRegisterEncoding::DevAddrShift8LoHi;
    uint16_t reg_lo = 0;
    uint16_t reg_hi = 0;

    bool operator==(const GraphDdrRegisterAbi& other) const {
        return role == other.role && access == other.access &&
               encoding == other.encoding && reg_lo == other.reg_lo &&
               reg_hi == other.reg_hi;
    }
};

struct GraphDdrRegisterOperandSpec {
    GraphDdrRegisterAbi abi;
    at::Tensor tensor;
};

struct GraphKernelRegisterRule {
    std::optional<KernelId> kernel_id;
    std::string kernel_name;
    GraphKernelRegisterRuleKind kind =
        GraphKernelRegisterRuleKind::IntrinsicNoDdr;
    GraphKernelNoDdrProof no_ddr_proof = GraphKernelNoDdrProof::None;
    std::vector<GraphDdrRegisterAbi> operands;
    size_t expected_count = 0;
};

struct GraphKernelRegisterCensusStats {
    size_t node_count = 0;
    size_t dma_count = 0;
    size_t barrier_count = 0;
    size_t kernel_count = 0;
    size_t typed_kernel_count = 0;
    size_t no_ddr_kernel_count = 0;
    size_t operand_count = 0;
    size_t weight_operand_count = 0;
    size_t key_cache_operand_count = 0;
    size_t value_cache_operand_count = 0;
    size_t semantic_input_operand_count = 0;
    uint64_t ordered_node_kind_hash = 0;
    uint64_t ordered_kernel_hash = 0;
};

struct GraphKernelRegisterPhaseExpectation {
    std::string name;
    GraphKernelRegisterCensusStats stats;
};

struct GraphKernelRegisterPolicy {
    uint64_t fingerprint = 0;
    std::string name;
    std::vector<GraphKernelRegisterRule> rules;
    size_t expected_kernel_count = 0;
    size_t expected_typed_kernel_count = 0;
    size_t expected_no_ddr_kernel_count = 0;
    size_t expected_operand_count = 0;
    size_t expected_weight_operand_count = 0;
    size_t expected_key_cache_operand_count = 0;
    size_t expected_value_cache_operand_count = 0;
    size_t expected_semantic_input_operand_count = 0;
    size_t expected_node_count = 0;
    size_t expected_dma_count = 0;
    size_t expected_barrier_count = 0;
    // FNV-1a over the exact ordered kernel name stream.  This is independent
    // from addresses and complements per-rule counts/role partitions.
    uint64_t expected_ordered_node_kind_hash = 0;
    uint64_t expected_ordered_kernel_hash = 0;
    std::vector<GraphKernelRegisterPhaseExpectation> phases;
};

struct GraphDdrRegisterOccurrence {
    GraphDdrRegisterAbi abi;
    // TensorImpl and StorageImpl have independent strong owners.  Keeping only
    // a Tensor would be insufficient: Tensor::set_ mutates that TensorImpl and
    // every Tensor copy would then observe the rebound storage.
    at::Tensor tensor_owner;
    c10::Storage storage_owner;
    const c10::TensorImpl* tensor_impl = nullptr;
    const c10::StorageImpl* storage_impl = nullptr;
    const void* data_ptr = nullptr;
    const void* storage_data_ptr = nullptr;
    at::ScalarType dtype = at::ScalarType::Undefined;
    c10::Device device{c10::DeviceType::CPU};
    c10::Layout layout = c10::Layout::Strided;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    int64_t storage_offset = 0;
    uint64_t tensor_topology_id = 0;
    size_t tensor_nbytes = 0;
    size_t storage_nbytes = 0;
    uint64_t dev_addr = 0;  // runtime location: excluded from topology hashes
};

struct GraphKernelRegisterCensus {
    enum class Kind : uint8_t { NoDdr = 1, TypedDdr = 2 };
    Kind kind = Kind::NoDdr;
    size_t policy_rule_index = 0;
    std::vector<GraphDdrRegisterOccurrence> operands;
};

class GraphKernelRegisterWriter final {
public:
    // Runtime-only encoded snapshot used for enqueue/end readback.  The
    // constructor remains private, so exposing the value type does not create
    // an alternate writer authority.
    struct RegisterPair {
        uint16_t reg_lo = 0;
        uint16_t reg_hi = 0;
        uint32_t encoded_addr = 0;
    };

    GraphKernelRegisterWriter() = default;
    GraphKernelRegisterWriter(const GraphKernelRegisterWriter&) = delete;
    GraphKernelRegisterWriter& operator=(const GraphKernelRegisterWriter&) =
        delete;
    GraphKernelRegisterWriter(GraphKernelRegisterWriter&& other) noexcept;
    GraphKernelRegisterWriter& operator=(GraphKernelRegisterWriter&&) = delete;

    void write(::rhino_lkn::Kernel_t& kernel);

private:
    GraphKernelRegisterWriter(
        RpuKernelGraph* graph, uint64_t ticket,
        std::vector<RegisterPair> pairs)
        : graph_(graph), ticket_(ticket), pairs_(std::move(pairs)) {}

    RpuKernelGraph* graph_ = nullptr;
    uint64_t ticket_ = 0;
    std::vector<RegisterPair> pairs_;
    bool written_ = false;

    friend class RpuKernelGraph;
};

// =============================================================================
// GraphNode — capture 期间录到的一个执行节点（kernel / DMA / memcpy / ...）
// =============================================================================

// Data node 承载 graph-aware copy/fill 以及子图、分支和 host fallback。
// ChildGraph 节点引用已 BUILT 的子图,REPLAYING executor 通过 patch 后的
// child replay 复用其 prepared segments。
enum class GraphNodeKind : uint8_t {
    Kernel = 0, Memcpy = 1, Memset = 2,
    // Value 3 belonged to the deleted graph-aware Flush shim. Keep the hole so
    // checked-in register-census hashes remain comparable with their captured
    // hardware traces; no node or execution path may use it.
    ChildGraph = 4, Branch = 5, HostCallback = 6,
    // RNG / data-dependent shape op 的节点级 oneshot:每轮 replay 在该节点真跑
    // host op,周围可 replay 的节点仍保留 graph 收益。
    Tier3Oneshot = 7,
    // BATCH_CTX 直填 Queue_t 的 DMA / Barrier 顶替项;Kernel / Dma / Barrier
    // 在 Segment 内连续混排,共享一个 Queue_t 批次。
    Dma = 8, Barrier = 9,
};

using GraphNodeKindMask = uint16_t;

constexpr GraphNodeKindMask graph_node_kind_bit(GraphNodeKind kind) {
    return static_cast<GraphNodeKindMask>(
        GraphNodeKindMask{1} << static_cast<uint8_t>(kind));
}

// Private BUILD observation of one non-legacy semantic DMA occurrence.  The
// payload deliberately has no public getters: endpoint/range/slot/owner data
// may only cross into the FMB seal, never become an alternate authority API.
class GraphSemanticDmaOccurrence {
private:
    size_t relative_node_index = 0;
    uint64_t endpoint_id = 0;
    uint8_t variant = 0;
    uint64_t src_addr = 0;
    uint64_t dst_addr = 0;
    size_t bytes = 0;
    uint8_t channel = 0;
    int64_t live_offset = 0;
    const uint64_t* live_base = nullptr;
    std::weak_ptr<const uint64_t> owner_generation;
    uint64_t expected_owner_generation = 0;
    bool canonical_member_emission = false;
    uint64_t spm_peer_id = 0;

    friend class RpuKernelGraph;
    friend class v3::FusedModelBase;
};

// Private BUILD observation of one terminal SPM writer.  The value is a
// digest sidecar only: it exposes no public semantic ID, writer kind, node, or
// address accessor and grants no physical SPM authority.
class GraphSemanticSpmProducerYieldOccurrence {
public:
    GraphSemanticSpmProducerYieldOccurrence() = default;

private:
    size_t relative_node_index = 0;
    uint64_t semantic_yield_id = 0;
    uint8_t writer_kind = 0;

    friend class RpuKernelGraph;
    friend class v3::FusedModelBase;
    friend class v3::SpmCompositeTraceCoordinator;
};

// Publicly pointer-free description of one contiguous node window emitted
// while a graph is RECORDING.  Its private FMB-only sidecar may retain owner /
// live-slot provenance, but callers receive no endpoint, range, slot, or
// pointer getters.  Branch/HostCallback/ChildGraph remain visible through
// kind_mask so a stricter consumer can fail closed on unsupported node kinds.
class GraphOpStreamWindowDigest {
public:
    size_t begin = 0;
    size_t end = 0;
    size_t node_count = 0;
    GraphNodeKindMask kind_mask = 0;
    uint64_t topology_hash = 0;

    bool contains(GraphNodeKind kind) const {
        return (kind_mask & graph_node_kind_bit(kind)) != 0;
    }

private:
    size_t legacy_dma_count = 0;
    std::vector<GraphSemanticDmaOccurrence> semantic_dmas;
    uint64_t semantic_spm_producer_yield_id_digest = 0;
    std::vector<GraphSemanticSpmProducerYieldOccurrence>
        semantic_spm_producer_yields;

    friend class RpuKernelGraph;
    friend class v3::FusedModelBase;
};

class KernelNodeData {
public:
    KernelNodeData() = default;
    KernelNodeData(
        std::optional<KernelId> kernel_id,
        size_t kernel_idx,
        std::string kernel_name,
        std::vector<uint16_t> grid_dims,
        std::vector<uint8_t> core_ids,
        QueueLaunchState queue_state)
        : kernel_id(std::move(kernel_id)),
          kernel_idx(kernel_idx),
          kernel_name(std::move(kernel_name)),
          grid_dims(std::move(grid_dims)),
          core_ids(std::move(core_ids)),
          queue_state(queue_state) {}

    // enum 路径 和 dynamic 路径互斥:正好一个 has_value。
    //   enum    : kernel_id 有值, kernel_name 为空 (KernelId 走 GET_KERNEL(id))
    //   dynamic : kernel_id 为空, kernel_name 非空 (string 走 get_kernel_reset(name))
    std::optional<KernelId> kernel_id;
    size_t kernel_idx = 0;                  // 对应 kernels_ 下标
    std::string kernel_name;
    std::vector<uint16_t> grid_dims;
    std::vector<uint8_t> core_ids;
    QueueLaunchState queue_state;
    // Present only for a schema-5 typed register-census Graph.  Legacy Graphs
    // retain nullopt and therefore preserve schemas 1-4 byte-for-byte.
    std::optional<GraphKernelRegisterCensus> register_census;

private:
    struct SemanticSpmProducerYieldMarker {
        uint64_t semantic_yield_id = 0;
        uint8_t writer_kind = 0;
    };
    std::optional<SemanticSpmProducerYieldMarker>
        semantic_spm_producer_yield;

    friend class RpuKernelGraph;
};

// Copy backend for a MemcpyNodeData — determined at capture time.
//   HOST_MEMCPY : fallback ::memcpy (32B 未对齐 / dev_addr 查不到 / 栈上)
//   DDR_TO_DDR  : 两端 RpuGetDevAddr 命中 → CopyMemoryChannel
//   DDR_TO_SPM  : reserved;当前 CopyToDevice* 只能整块 SPM↔DDR,不支持 slice
//   SPM_TO_DDR  : reserved;同上
//   SPM_TO_SPM  : reserved
enum class CopyKind : uint8_t {
    HOST_MEMCPY = 0,
    DDR_TO_DDR  = 1,
    DDR_TO_SPM  = 2,
    SPM_TO_DDR  = 3,
    SPM_TO_SPM  = 4,
};

struct MemcpyNodeData {
    void* dst;
    const void* src;
    size_t bytes;

    // RECORDING 期记录 copy 分类和 DMA 参数;REPLAYING 期由 capture_data 覆写
    // 本轮指针和地址,execute_data_node 按 kind 分派到 DMA 或 host fallback。
    CopyKind kind = CopyKind::HOST_MEMCPY;
    uint64_t dst_dev_addr = 0;   // RpuGetDevAddr(dst),0 = 查不到
    uint64_t src_dev_addr = 0;
    int32_t  dst_pool_slot = -1; // -1 = 非 pool-managed SPM
    int32_t  src_pool_slot = -1;
    int8_t   dma_channel = -1;   // -1 = auto;batch merger 可指派 0..7
};

struct MemsetNodeData {
    void* dst;
    uint8_t value;
    size_t bytes;
};

// =============================================================================
// DmaNodeData — graph-captured DMA (Queue_t::add_dma_kernel)
// =============================================================================
//
// 顶替原 BATCH_CTX.add_dma / add_mutable_dma / add_mutable_dst_dma 入口。
// 在 RECORDING 期入 nodes_,prepare_segment_queue 走到此节点时 deref live_*_base
// 取本轮 dev_addr,然后 wq.add_dma_kernel(src, dst, bytes, channel) 进 batch。
// 与 Kernel 节点一起放在同一 Segment 内(共享同一 Queue_t 批次),无需作为段边界,
// 从而保留 BATCH_CTX "DMA + kernel + barrier 同一 enqueu_batch" 的 perf 收益。
//
// Variant:
//   Fixed       : src + dst 都是绝对 dev_addr,RECORDING 期定死,REPLAYING 不变。
//   MutableSrc  : src 由 *live_src_base + live_src_offset 派生,prepare 每次重算。
//                 典型:输入 hidden_states / cond_emb 在每轮 forward 新 alloc,
//                 caller 保持一个跨 forward 稳定的 uint64_t* 变量记录当前 dev_addr。
//   MutableDst  : dst 同理。
//
// 设计哲学跟原 BATCH_CTX 一致:_mutable 命名是"caller 端 live deref",
// 不依赖 Queue_t::add_dma_kernel_mutable / update_dma_kernel(因为
// prepare_segment_queue 每轮 replay 都重 add,kd_buf 重建)。
class GraphDmaSemanticEndpoint {
private:
    GraphDmaSemanticEndpoint(
        uint64_t semantic_id,
        uint64_t profile_hash,
        uint64_t allocation_hash,
        size_t layer_group_ordinal,
        size_t member_ordinal,
        uint8_t role,
        uint8_t arena,
        uint8_t access,
        uint8_t address_mode,
        uint8_t expected_dma_variant,
        uint8_t expected_occurrence_count,
        size_t byte_begin,
        size_t byte_count,
        bool requires_canonical_burst,
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation)
        : semantic_id_(semantic_id),
          profile_hash_(profile_hash),
          allocation_hash_(allocation_hash),
          layer_group_ordinal_(layer_group_ordinal),
          member_ordinal_(member_ordinal),
          role_(role),
          arena_(arena),
          access_(access),
          address_mode_(address_mode),
          expected_dma_variant_(expected_dma_variant),
          expected_occurrence_count_(expected_occurrence_count),
          byte_begin_(byte_begin),
          byte_count_(byte_count),
          requires_canonical_burst_(requires_canonical_burst),
          owner_generation_(owner_generation),
          expected_owner_generation_(expected_owner_generation) {}

    uint64_t semantic_id_ = 0;
    uint64_t profile_hash_ = 0;
    uint64_t allocation_hash_ = 0;
    size_t layer_group_ordinal_ = 0;
    size_t member_ordinal_ = 0;
    uint8_t role_ = 0;
    uint8_t arena_ = 0;
    uint8_t access_ = 0;
    uint8_t address_mode_ = 0;
    uint8_t expected_dma_variant_ = 0;
    uint8_t expected_occurrence_count_ = 0;
    size_t byte_begin_ = 0;
    size_t byte_count_ = 0;
    bool requires_canonical_burst_ = false;
    std::weak_ptr<const uint64_t> owner_generation_;
    uint64_t expected_owner_generation_ = 0;

    friend class v3::FusedModelBase;
    friend class RpuKernelGraph;
};

struct DmaNodeData {
    enum class Variant : uint8_t { Fixed = 0, MutableSrc = 1, MutableDst = 2 };
    Variant  variant     = Variant::Fixed;
    uint64_t src_addr    = 0;       // Fixed / MutableDst 用
    uint64_t dst_addr    = 0;       // Fixed / MutableSrc 用
    size_t   bytes       = 0;
    uint8_t  channel     = 0;

    const uint64_t* live_base = nullptr;  // MutableSrc/Dst:caller-owned ptr,
                                          // 跨 forward 稳定;每次 prepare deref。
    int64_t  live_offset = 0;             // 加到 *live_base 上得到本轮 dev_addr。

    // Zero preserves the legacy replay contract byte-for-byte.  A nonzero ID
    // is minted only by FusedModelBase and makes every endpoint-bearing field
    // exact across REPLAY; only the value stored behind live_base may change.
    uint64_t semantic_endpoint_id = 0;
    uint8_t semantic_expected_occurrence_count = 0;
    std::weak_ptr<const uint64_t> semantic_owner_generation;
    uint64_t semantic_expected_owner_generation = 0;
    bool semantic_canonical_member_emission = false;
    // Nonzero only for schema-v13 canonical FMB DMA.  This opaque identity
    // binds the fixed SPM side to one allocation-derived dense peer without
    // exposing a raw address or slice API.
    uint64_t semantic_spm_peer_id = 0;

    // Schema-5-only owner sidecar.  These fields are populated exclusively by
    // the census-owned typed DMA burst and are ignored by topology schemas
    // 1--4.  The ordinal is internal Graph state; no public API accepts or
    // returns it.
    bool outer_fast_owner_bound = false;
    GraphOuterFastDmaSide outer_fast_ddr_side =
        GraphOuterFastDmaSide::None;
    size_t outer_fast_owner_ordinal = 0;
    uint64_t outer_fast_owner_topology_id = 0;
    size_t outer_fast_owner_byte_offset = 0;
};

// =============================================================================
// BarrierNodeData — graph-captured cross-stream barrier
// =============================================================================
//
// 顶替原 BATCH_CTX.add_barrier。Queue_t 层面的 stream barrier,跟 Dma/Kernel
// 在同一 Segment 内顺序进 batch,不作为段边界。
struct BarrierNodeData {
    uint8_t self_stream   = 0;
    uint8_t target_stream = 0;
};

// =============================================================================
// ChildGraph 节点
// =============================================================================
//
// 父图通过 register_child(child, patches) 在 RECORDING 期推一个 ChildGraph 节点;
// 父图 REPLAYING 时遇到该节点 → child->replay_prepared_child(patches) 复用已
// BUILT 的子图。ownership 主体是调用方(典型:RpuGraphCacheEntry 持 shared_ptr),
// 父图持引用确保子图至少存活到父图 REPLAY 完成。
//
// RegisterPatch 让父图在 replay 前覆写 child kernel 的某个 reg(典型:输入
// dev_addr 每轮变化时 patch 进 child 的 input pointer reg)。

// width 1/2/4 → 走 set_regs(reg_idx, scalar) 单值重载;
// width 8     → 走 set_regs(reg_idx, addr, stride) 双 reg(addr+stride)重载,
//               value 装 64-bit dev_addr,stride 是配套 stride 寄存器值。
//               典型:把 input tensor 的 dev_addr 注入 child kernel 的某个
//               address-stride pair。
struct RegisterPatch {
    size_t   kernel_idx;   // child.kernels_ 的下标 (child BUILT 后稳定)
    uint32_t reg_idx;      // Kernel_t::set_regs 的第一个参数
    uint64_t value;        // 装下 dev_addr 或 16/32/64-bit 参数
    uint8_t  width;        // 1=u8 / 2=u16 / 4=u32 / 8=addr+stride
    uint32_t stride = 0;   // 仅 width=8 使用;其它 width 忽略
};

// Child graph replay 期间对 data node 指针的显式 patch。
// field:0 = Memcpy.dst,1 = Memcpy.src。dev_addr 可选;为 0 时 replay 侧
// 尝试通过 ptr 重新查询 RpuGetDevAddr。
struct DataPatch {
    size_t   node_idx = 0;
    uint8_t  field = 0;
    uint64_t ptr = 0;
    uint64_t dev_addr = 0;
};

struct ChildGraphNodeData {
    // shared_ptr 保活:父 BUILT 后必须保证 child 生命覆盖父 replay。
    // RpuGraphCache::evict 在 use_count() > 1 时拒绝;register_child 强制 child
    // 已经 BUILT 且 replayable_。
    std::shared_ptr<RpuKernelGraph> child;
    GraphSignature child_signature;   // 仅 debug / error message;空 = 调用方未提供
    std::vector<RegisterPatch> input_patches;
    std::vector<DataPatch> data_patches;
};

// =============================================================================
// Branch marker node
// =============================================================================
//
// BranchNode 是显式控制流标记,作为 segment 边界参与 dump 和校验;执行时 no-op。
// cache 正确性由 GraphSignature.branch_key 承载。
struct BranchNodeData {
    uint64_t branch_key = 0;
    GraphSignature branch_signature;  // debug only; empty = caller omitted it
    std::string label;                 // optional human-readable debug label
};

// =============================================================================
// HostCallback 节点
// =============================================================================
//
// CPU fallback op 在 RECORDING / REPLAYING 期作为 graph 内节点录制和回放,不再
// 通过 sync_point() 打断 graph。三档分级见 graph_infra.h:
//   Tier1 (.out / in-place): output dev_addr 稳定 → cacheable
//   Tier2 (functional return): graph-owned 稳定 buffer → cacheable
//   Tier3 (RNG / dynamic shape): 由 Tier3Oneshot 节点处理
//
// RECORDING 期:HostCallbackCapture RAII 录 input IValue 骨架 + 跑 CPU op +
//   登记 output dev_addrs。
// REPLAYING op stream dispatch 期:capture_host_callback_replay_args 绑定本轮
//   live tensor + 合成 .out/in-place return + 推进 cursor (不立刻执行 CPU op,
//   因为前序 RPU kernel 还没 launch)。
// REPLAYING executor 期(到达节点时):重建 stack + run_cpu_fallback_zerocopy
//   redispatch CPU + 校验 output dev_addr 稳定性。
//
struct HostCallbackNodeData {
    // 跨 session 稳定的 op 标识。指向 ATen op registry 内的 schema,生命周期由
    // dispatcher 管理,我们这里只是引用。
    // optional 包装是因为 c10::OperatorHandle 不可默认构造(没有公开默认 ctor),
    // 但 GraphNode 内的 std::variant 在某些路径(如 capture_data 的 REPLAYING
    // 覆写 std::get<ND> + 赋值)需要 alternative 可默认构造或显式初始化。
    // 真实使用时一定 has_value();0 路径(empty optional)只在尚未填充的中间态出现。
    std::optional<c10::OperatorHandle> op_handle;
    HostCallbackTier tier = HostCallbackTier::Tier1;

    // RECORDING 期录的 stack 骨架:Tensor 槽放 at::Tensor() 占位(replay 期由
    // capture_host_callback_replay_args 用 live_tensor_args 重新填),其它 IValue
    // (int / bool / Scalar / List[int] 等)直接存值,replay 期 TORCH_CHECK
    // 当前 stack 同 slot 的值与 recorded 一致(漂移说明 op 调用参数变了)。
    std::vector<c10::IValue> args_template;
    std::vector<size_t> tensor_arg_indices;   // args_template 里的 Tensor slot 下标

    // mutable output args (.out= 形式,如 `argmax.out(*, Tensor(a!) out)`)。
    // in-place self mutation(`add_` 等)在 classify_host_op 里被 Reject,
    // 不会进 HostCallback 路径。这里仅记 .out= 形式 input arg 的下标,供
    // capture_host_callback_replay_args 重建 stack 入参时复原槽位。
    std::vector<size_t> output_arg_indices;

    // graph-owned 稳定 output buffer(Tier 1 + Tier 2 共用)。RECORDING 期
    // adopt_returns 把 CPU op 跑出的 fresh tensor copy 到 stable buffer,
    // 替换 stack slot;REPLAYING/oneshot executor 同样 copy_+替换,确保下游
    // RPU op 永远看到 stable dev_addr。
    //
    // - cache eviction / invalidate 时由 nodes_.clear() 自动释放。
    // - RpuGraphCache 不能保存用户输入 tensor / 临时返回的例外:这些 buffer 是
    //   graph-owned replay artifact,语义等价于 prepared queue / SPM slot。
    std::vector<at::Tensor> output_pool;
    std::vector<uint64_t>   output_pool_dev_addrs;   // RpuGetDevAddr(output_pool[r])
    std::vector<size_t>     output_pool_bytes;       // numel * element_size

    // 每轮 forward 当前的 input tensor(对齐 tensor_arg_indices 顺序)。
    // RECORDING ctor 填一次;REPLAYING op stream 期 capture_host_callback_replay_args
    // 覆写本轮 tensor。executor (execute_data_node) 用它 + args_template 重建
    // stack 跑 CPU op。
    std::vector<at::Tensor> live_tensor_args;
};

// =============================================================================
// Tier3OneshotNode
// =============================================================================
//
// RNG / data-dependent shape 等不能完全重放的 op 会录成节点级 oneshot:
// RECORDING 期记录 op、参数骨架、输入/输出依赖;REPLAYING executor 走到该节点时
// 真跑 CPU op 并把结果写入 graph-managed transient buffer,周围 RPU 段继续 replay。
//
struct Tier3OneshotNodeData {
    // op 元数据(replay 期重新 dispatch)。同 HostCallbackNodeData 用 optional
    // 包装(c10::OperatorHandle 不可默认构造,但 std::variant 需要 alternative
    // 可默认构造)。真实使用时一定 has_value()。
    std::optional<c10::OperatorHandle> op_handle;

    // RECORDING 期录的 stack 骨架。Tensor 槽放 at::Tensor() 占位,replay 期由
    // input dep 重填;非 Tensor IValue(int / bool / Scalar 等)直接存值,
    // replay 期 TORCH_CHECK 同槽值跟 recorded 一致(漂移说明参数变了)。
    // 跟 HostCallbackNodeData::args_template 同语义。
    std::vector<c10::IValue> args_template;
    std::vector<size_t> tensor_arg_indices;   // args_template 里 Tensor slot 下标

    // 输入依赖:对哪些前置节点的 stable buffer 有读依赖。RECORDING 期由 op args
    // 里的 RPU tensor 反查最近 writer,REPLAYING 期用于 flush / 执行排序。
    std::vector<int> input_dep_node_ids;

    // 输出依赖:后续哪些节点读它的 transient output。
    std::vector<int> output_dep_node_ids;

    // transient buffer:每次 replay 重新 alloc(挂 owner tree
    // ResourceKind::TransientTier3Buffer)。元素是 GraphOwnedResource 的 idx。
    std::vector<int> transient_resource_ids;

    // 最近一次真跑后的 output dev_addrs。RECORDING 时填一份(诊断 dump 用);
    // REPLAYING 期每次执行后覆写 — RNG output 跨 replay 不同地址,这里只是
    // last seen,跟 stable 语义无关。
    std::vector<uint64_t> last_output_dev_addrs;

    // 每轮当前的 input tensor(对齐 tensor_arg_indices 顺序)。executor 用
    // live_tensor_args + args_template 重建 stack 跑 CPU op。
    std::vector<at::Tensor> live_tensor_args;

    // Tier3 不承诺跨 replay 稳定(RNG output 每步新 dev_addr),这里只 keep-alive
    // 防当前 forward 内 GC。下游 stack 替换由 executor 完成。
    std::vector<at::Tensor> output_keepalives;
};

struct GraphNode {
    GraphNodeKind kind;
    std::variant<KernelNodeData, MemcpyNodeData, MemsetNodeData,
                 ChildGraphNodeData, BranchNodeData, HostCallbackNodeData,
                 Tier3OneshotNodeData,
                 DmaNodeData, BarrierNodeData> data;

    const KernelNodeData& as_kernel() const { return std::get<KernelNodeData>(data); }
    KernelNodeData& as_kernel() { return std::get<KernelNodeData>(data); }
    const MemcpyNodeData& as_memcpy() const {
        return std::get<MemcpyNodeData>(data);
    }
    MemcpyNodeData& as_memcpy() {
        return std::get<MemcpyNodeData>(data);
    }
    const ChildGraphNodeData& as_child_graph() const {
        return std::get<ChildGraphNodeData>(data);
    }
    ChildGraphNodeData& as_child_graph() {
        return std::get<ChildGraphNodeData>(data);
    }
    const BranchNodeData& as_branch() const {
        return std::get<BranchNodeData>(data);
    }
    BranchNodeData& as_branch() {
        return std::get<BranchNodeData>(data);
    }
    const HostCallbackNodeData& as_host_callback() const {
        return std::get<HostCallbackNodeData>(data);
    }
    HostCallbackNodeData& as_host_callback() {
        return std::get<HostCallbackNodeData>(data);
    }
    const Tier3OneshotNodeData& as_tier3_oneshot() const {
        return std::get<Tier3OneshotNodeData>(data);
    }
    Tier3OneshotNodeData& as_tier3_oneshot() {
        return std::get<Tier3OneshotNodeData>(data);
    }
    const DmaNodeData& as_dma() const { return std::get<DmaNodeData>(data); }
    DmaNodeData& as_dma() { return std::get<DmaNodeData>(data); }
    const BarrierNodeData& as_barrier_node() const {
        return std::get<BarrierNodeData>(data);
    }
    BarrierNodeData& as_barrier_node() { return std::get<BarrierNodeData>(data); }
};

// =============================================================================
// NodeKindOf — compile-time 映射 data node 结构到 GraphNodeKind 枚举
// =============================================================================
// 未特化的类型在实例化 capture_data<T> 时会触发编译错误，充当静态类型保险丝。

template <typename T> struct NodeKindOf;

template <> struct NodeKindOf<MemcpyNodeData> {
    static constexpr GraphNodeKind value = GraphNodeKind::Memcpy;
};

template <> struct NodeKindOf<MemsetNodeData> {
    static constexpr GraphNodeKind value = GraphNodeKind::Memset;
};

template <> struct NodeKindOf<ChildGraphNodeData> {
    static constexpr GraphNodeKind value = GraphNodeKind::ChildGraph;
};

template <> struct NodeKindOf<BranchNodeData> {
    static constexpr GraphNodeKind value = GraphNodeKind::Branch;
};

template <> struct NodeKindOf<HostCallbackNodeData> {
    static constexpr GraphNodeKind value = GraphNodeKind::HostCallback;
};

template <> struct NodeKindOf<Tier3OneshotNodeData> {
    static constexpr GraphNodeKind value = GraphNodeKind::Tier3Oneshot;
};

template <> struct NodeKindOf<DmaNodeData> {
    static constexpr GraphNodeKind value = GraphNodeKind::Dma;
};

template <> struct NodeKindOf<BarrierNodeData> {
    static constexpr GraphNodeKind value = GraphNodeKind::Barrier;
};

// =============================================================================
// Segment（类似子图） — 一组连续、queue_state 兼容的 Kernel/Dma/Barrier nodes。
// Kernel 可混用不同 core_ids；segment.core_ids 仅记录最大的 queue 执行域，
// prepare_segment_queue 仍把每个 KernelNodeData.core_ids 单独传给 SDK。
// =============================================================================

struct Segment {
    size_t segment_id = 0;                  // segments_ 中的顺序 id,用于 dump / replay 诊断
    size_t start_idx = 0;                   // nodes_ 中的起始下标（含）
    size_t end_idx = 0;                     // 结束下标（exclusive）；区间仅含 Kernel/Dma/Barrier
    std::vector<uint8_t> core_ids;          // Queue_t 构造用的最大 core 执行域
    QueueLaunchState queue_state;           // 本 segment 的 queue 状态
    size_t replay_count = 0;                // 当前 BUILT 期内 replay 次数
    uint64_t signature_layer_hash = 0;       // segment-level layered lookup hash
    // Queue_t 不再由每个 segment 持有。改用 QueueCache::instance()
    // .get(core_num) 在 RECORDING/REPLAYING segment loop 内复用 (按 core 数
    // 全局缓存,最多 8 个 Queue_t)。每段调用前 build_batch() 会 reset 之前
    // 的 kd_buf。原因:per-segment unique_ptr 在 segments=112 时挂 112 个
    // Queue_t,直接打爆 SDK BufferPool (pool_size=16)。
    //
    // Queue_t 由每个 cache entry 持有 (RpuKernelGraph::private_queue_),
    // segment 仍不持 Queue_t,但本段 BUILD 时通过 prepare_segment_queue 填
    // mutable_dmas 槽位（单 build-order vector，Kind 字段区分 src/dst）。
    // REPLAY 时若 RpuKernelGraph::private_queue_built_segment_idx_ ==
    // 本段 idx,可走 sync-only fast path (update_dma_kernel + sync_mutable_params
    // + enqueu_batch),否则 fallback full rebuild。多段 graph 因为 kd_buf
    // 只能容纳一个段,fallback 路径会循环触发。

    // mutable DMA 槽位在 REPLAY 时按 dma_id 顺序 update_dma_kernel。
    // dma_id 由 add_dma_kernel_mutable 在本段 prepare 调用顺序累加 (0,1,2,...);
    // Fixed DMA / Barrier / Kernel **不增** dma_id (SDK 内部计数与此一致)。
    struct PreparedMutableDmaSlot {
        size_t   node_idx;                  // 关联回 nodes_,调试/审计用
        uint32_t dma_id;                    // add_dma_kernel_mutable 顺序 (0,1,2,...)
        enum class Kind : uint8_t { MutableSrc = 0, MutableDst = 1 };
        Kind     kind;
        const uint64_t* live_base;          // caller-owned uint64_t 指针 (跨 forward 稳定)
        int64_t  live_offset;               // *live_base + live_offset = 本轮 live address
        uint64_t fixed_addr;                // MutableSrc → dst (baked);MutableDst → src (baked)
        size_t   bytes;
        uint8_t  channel;
    };
    std::vector<PreparedMutableDmaSlot> mutable_dmas;
};

// =============================================================================
// GraphStats — 上一次 execute_graph_* 的统计快照
// =============================================================================
// execute_graph_for_recording / execute_graph_for_replaying / execute_graph_oneshot
// 完成后写入;Python 通过 Graph.debug_stats() 读取。

struct GraphStats {
    size_t kernel_count = 0;            // 所有 Kernel kind 节点数
    size_t data_node_count = 0;         // Memcpy + Memset 节点数
    size_t segment_count = 0;           // build_segments_from_nodes 后 segments_ 长度 (= launch 次数)
    std::vector<size_t> per_segment_replay_count;  // mirror segments_[i].replay_count(live)
    size_t boundary_flush_count = 0;    // 上次 flush_boundary_ptrs 去重后刷的 unique ptr 数

    // HostCallback 节点按 tier 分桶,并统计 graph-owned stable output 的 DDR
    // 占用字节数。REPLAYING 不重 build segments,这里只 mirror 已录图的值。
    size_t host_callback_tier1_count = 0;
    size_t host_callback_tier2_count = 0;
    size_t host_callback_tier3_count = 0;
    size_t host_callback_output_bytes = 0;

    // Tier3OneshotNode 节点数。RNG / dyn-shape op 默认录成独立节点;
    // HostCallback tier=Tier3 计数只保留兼容语义。
    size_t tier3_oneshot_count = 0;

    // launch_segment_for_replay 累计计数器。**不在 execute_graph_for_replaying
    // / replay_stats_refresh 里 reset**,跨整个 graph 生命周期累积,反映 Queue_t
    // 复用效率 (QueueCache::get 命中率)。
    //
    // Queue_t 按 core_num 全局缓存时的计数语义:
    //   prepared_segment_hit_total : QueueCache.get(core_num) 命中已缓存 Queue_t
    //                                的次数 (segments_launched_via_cached_queue)
    //   prepared_segment_miss_total: QueueCache.get(core_num) 触发 new Queue_t
    //                                的次数 (= 实际 BufferPool::AcquireBuffer 次数)
    //                                进程生命周期内最多 8 (每个 core_num 一次)
    //
    // 稳定 replay 期预期: miss_total 维持在 [1..8] (取决于 graph 涉及多少
    // 不同的 core_num),hit_total ≈ segment_count × replay_count。
    // miss_total > 8 (单进程) 说明 QueueCache 被 clear 过(异常)。
    //
    // RECORDING 期的 launch 不计入;这两个计数只反映 REPLAYING 路径。
    size_t prepared_segment_hit_total = 0;
    size_t prepared_segment_miss_total = 0;

    // hw_batch_submit_total — 真实 /dev/rpu 批提交计数:Queue_t::enqueu_batch
    // (wait_finish=true) 每成功返回一次 +1。与 prepared_segment_* 不同,**相位
    // 无关**:RECORDING/BUILD 与 REPLAYING 的提交都计入,故单步 BUILD forward
    // 也 >0。与 last_stats_ 同生命周期(begin/invalidate 重置)。用作 native
    // dispatch 证据:>0 证明本图确有 kernel 批被同步发射到设备(非建图/走指针)。
    size_t hw_batch_submit_total = 0;

    // Schema-5 typed DDR address-register census from the most recent BUILD /
    // REPLAY invocation. Physical epoch is diagnostic-only and intentionally
    // excluded from the stable topology/plan hash.
    bool register_census_present = false;
    bool register_census_build_committed = false;
    uint64_t register_census_policy_fingerprint = 0;
    uint64_t register_census_policy_digest = 0;
    uint64_t register_census_plan_hash = 0;
    uint64_t register_census_live_epoch = 0;
    GraphKernelRegisterCensusStats register_census;
    bool register_census_outer_fast_hit = false;
    size_t register_census_outer_fast_retained_validated = 0;
    size_t register_census_outer_fast_mutable_slot_count = 0;
    size_t register_census_outer_fast_mutable_occurrence_count = 0;
    size_t register_census_host_reemitted_node_count = 0;
    size_t register_census_host_reemitted_kernel_count = 0;

    // mark_non_replayable(reason) 写入的原因字符串;跨 execute_graph_* 不变,
    // 只在 RpuKernelGraph::begin / invalidate 重置。
    std::string non_replayable_reason;

    double avg_segment_size() const {
        return segment_count > 0
            ? static_cast<double>(kernel_count) / segment_count
            : 0.0;
    }
};

// =============================================================================
// RpuKernelGraph — 执行期协调器
// =============================================================================

class RpuKernelGraph {
public:
    enum class State : uint8_t {
        PASSTHROUGH,
        RECORDING,
        BUILT,
        REPLAYING
    };

    RpuKernelGraph();
    ~RpuKernelGraph();
    RpuKernelGraph(const RpuKernelGraph&) = delete;
    RpuKernelGraph& operator=(const RpuKernelGraph&) = delete;

    // ==================== 活跃图管理 ====================

    static RpuKernelGraph& active();
    static bool has_active();

    // Direct/immediate DMA is Graph-invisible by construction.  Admit it only
    // outside every Graph scope, or for the compatibility suffix of exactly
    // one scope that has already downgraded to plain PASSTHROUGH.  The latter
    // must carry no semantic/canonical DMA authority.  Call before dereferencing
    // a live address, flushing DDR, resolving an address, or touching the SDK.
    static void check_direct_immediate_dma_allowed(const char* operation);

    // 给定 graph 是否在当前线程 active_stack_ 内(任意位置,不要求栈顶)。
    // RpuGraphCache::evict 用它避免删除仍在 RECORDING/REPLAYING 的 nested graph。
    static bool is_on_active_stack(const RpuKernelGraph& g);

    // ==================== 生命周期 ====================

    // begin() 无签名:始终走 RECORDING(兼容形态;不带 admission)。
    void begin();
    // begin(sig):signature-driven admission。
    //   BUILT + replayable + has_built_signature_ + sig == built_signature_ → REPLAYING
    //   else → invalidate()(若 BUILT)+ RECORDING + pending_signature_ = sig
    // end() 在 BUILT 完成时把 pending_signature_ 写入 built_signature_。
    void begin(const GraphSignature& sig);
    void end();
    void abort();
    void invalidate();
    void sync_point();

    // Bind this exact retained BUILT identity to one physical arena.  Must run
    // outside every Graph scope while the caller owns the physical claim.
    [[nodiscard]] RpuRetainedPhysicalArenaGuard arm_retained_physical_arena(
        const RpuPhysicalArenaAuthority& authority);
    bool has_retained_physical_arena_guard() const;
    bool has_provisional_physical_build() const;

    // Fast replay:begin(sig) 已进入 REPLAYING 后跳过 op stream,直接把
    // cursor_ 推到 nodes_.size(),让 end() 进入 execute_graph_for_replaying。
    // 适用于输入/输出 buffer 跨 replay 稳定且寄存器无需重写的路径。
    //
    // 调用方契约:
    //   - begin(sig) 已成功进入 REPLAYING(state_ == REPLAYING)。
    //   - 上一次 op stream 写入 prepared_wq 的 register 值仍然指向**当前步**
    //     真正想读写的 dev_addr。stable input/position/cache buffer + in-place
    //     mutate 是最简单的满足方式。
    //   - HostCallback / Tier3Oneshot / Memcpy data node 的 capture_*_replay_args
    //     不会被调,意味着 hc.live_tensor_args 等仍是上一次 op stream 设置的
    //     值。如果需要替 HC 入参 tensor 必须自己保持 stable Python tensor
    //     object 并 in-place 改其 DDR。
    //   - 任何不满足上述条件的 graph 在 fast path 下行为未定义;调用方应有
    //     fallback。该 API **不**做正确性诊断。
    //
    // 默认关闭,由调用方显式选择。
    void skip_op_stream_for_fast_replay();

    // Fast replay variant for graphs that carry a post_fn re-emitting
    // its own kernels after the layer body. Instead of pushing cursor_ to the
    // very end (skip_op_stream_for_fast_replay), this advances cursor_ only to
    // the recorded post_fn-start position, so the layer-body re-walk is skipped
    // (the ~10k-node win) while the post_fn nodes are left ahead for the normal
    // replay path to re-emit (only ~54 nodes; its mutable-DMA op patches the
    // fresh per-forward output addr). Caller contract is skip_op_stream_for_fast_replay's,
    // plus: mark_post_fn_cursor() must have been called during RECORDING
    // (has_post_fn_cursor() true), else TORCH_CHECK aborts.
    void skip_layer_body_for_fast_replay();

    // Record the post_fn-start cursor (= # nodes emitted before post_fn). No-op
    // unless RECORDING. Called by run_all_layers right before invoking post_fn.
    void mark_post_fn_cursor();

    // True once mark_post_fn_cursor() recorded a post_fn-start position for the
    // current BUILT graph. Reset on every fresh capture / invalidate.
    bool has_post_fn_cursor() const { return has_post_fn_cursor_; }

    // Fast replay with explicit patches:在跳过 op stream 前应用 RegisterPatch[]
    // 和 DataPatch[],语义与 replay_prepared_child_with_data_patches 共用同一组
    // patch helper。用于每步变化的寄存器和 Memcpy dst/src。
    //
    // 调用方契约同 skip_op_stream_for_fast_replay,加上:
    //   - register_patches[i].kernel_idx 必须 < kernels_.size()
    //   - data_patches[i].node_idx 必须 < nodes_.size() 且对应 Memcpy node
    //   - field ∈ {0, 1};0 = Memcpy.dst,1 = Memcpy.src
    //   - ptr 必须非零
    void skip_op_stream_with_patches(
        const std::vector<RegisterPatch>& register_patches,
        const std::vector<DataPatch>& data_patches);

    // 上一次 BUILT 时录下的签名。空 (has_built_signature_=false) 表示从未 BUILT。
    const GraphSignature& built_signature() const { return built_signature_; }
    bool has_built_signature() const { return has_built_signature_; }

    // ==================== ChildGraph ====================

    // RECORDING 期父图入口:登记一个 ChildGraph 节点引用已 BUILT 的子图。
    // 父图 REPLAYING 时,executor 走到该节点会调 child->replay_prepared_child(patches)。
    // 必要条件:child 必须已 BUILT 且 replayable_;不能传 self。
    // 默认严格契约:child 的 nodes_ 只允许包含 Kernel 节点。含 Memcpy /
    //   Memset / 嵌套 ChildGraph 的 child 一律抛错。原因:replay_prepared_child
    //   走的是 prepared_wq + apply RegisterPatch 的轻路径,不重跑 op stream,
    //   data node 的 dst/src 不会被 capture_data 覆写、kernel reg 中没被
    //   RegisterPatch 覆盖的 dev_addr 也不会刷新。RECORDING 期捕获的临时
    //   tensor 一旦 GC 即 use-after-free / stale dev_addr。
    // 传 data_patches 时 opt-in 放宽 Memcpy 节点,调用方必须描述 replay 期
    // dst/src 漂移。child_signature 仅供 debug;可空。
    void register_child(std::shared_ptr<RpuKernelGraph> child,
                        std::vector<RegisterPatch> input_patches = {},
                        GraphSignature child_signature = {},
                        std::vector<DataPatch> data_patches = {});

    // 父图 REPLAYING 期(begin(sig) 进 REPLAYING 后、首条 segment launch 之前)
    // 覆写 nodes_[node_idx] 这个 ChildGraph 节点的 input_patches。典型:输入
    // tensor 的 dev_addr 每轮变化时调用方先调此 API。
    void patch_child_inputs(size_t node_idx,
                            std::vector<RegisterPatch> patches);
    void patch_child_data(size_t node_idx,
                          std::vector<DataPatch> patches);

    // 子图入口 — 父图 executor 走到 ChildGraph 节点时调用此函数:
    //   1. 应用 patches 到自身 mutable kernels(set_regs);
    //   2. 临时切到 REPLAYING 跑一遍 prepared segments;
    //   3. 切回 BUILT。
    // 调用方:父图 RpuKernelGraph::execute_data_node 的 ChildGraph case。
    // 不会触碰 active_stack_ / SPM_POOL.defer_depth_:子图的 SPM 在 RECORDING
    // 期就锁定了,父 scope 的 defer 状态不受影响。
    void replay_prepared_child(const std::vector<RegisterPatch>& patches);
    void replay_prepared_child_with_data_patches(
            const std::vector<RegisterPatch>& register_patches,
            const std::vector<DataPatch>& data_patches);

    // 显式控制流标记:RECORDING 推入 Branch 节点,REPLAYING 消费并校验
    // branch_key。本身不执行 device-side branch dispatch。
    void record_branch(uint64_t branch_key,
                       GraphSignature branch_signature = {},
                       std::string label = {});

    // ==================== DMA / Barrier 捕获 (BATCH_CTX 顶替) ====================
    //
    // 顶替 v5 BATCH_CTX 直填 Queue_t 的入口,语义跟原 batch_context.h 一致:
    //   - RECORDING:push DmaNodeData/BarrierNodeData 进 nodes_;build_segments
    //     时跟 Kernel 节点一起留在同一 Segment,共享同一 Queue_t::enqueu_batch。
    //   - REPLAYING:推 cursor + 校验,实际 src/dst deref 在每段 prepare 时做。
    //   - PASSTHROUGH / BUILT:contract violation,直接 TORCH_CHECK 抛错(BATCH_CTX
    //     原本就只在 graph 内使用,顶替接口保持同样限制)。
    //
    // 三个 DMA 变体跟 BATCH_CTX 完全 1:1 对应:
    //   record_dma_fixed         ↔ BATCH_CTX.add_dma
    //   record_dma_mutable_src   ↔ BATCH_CTX.add_mutable_dma
    //   record_dma_mutable_dst   ↔ BATCH_CTX.add_mutable_dst_dma
    // Mutable 变体的 live_*_base 必须指向 caller-owned uint64_t,生命周期
    // 跨整个 graph BUILT/REPLAY 段稳定(典型:成员字段 / 顶层模块单例)。
    void record_dma_fixed(uint64_t src_addr, uint64_t dst_addr,
                          size_t bytes, uint8_t channel);
    void record_dma_mutable_src(const uint64_t* live_src_base,
                                int64_t          src_offset,
                                uint64_t         dst_addr,
                                size_t           bytes,
                                uint8_t          channel);
    void record_dma_mutable_dst(uint64_t         src_addr,
                                const uint64_t* live_dst_base,
                                int64_t          dst_offset,
                                size_t           bytes,
                                uint8_t          channel);
    void record_dma_fixed(const GraphDmaSemanticEndpoint& endpoint,
                          uint64_t src_addr, uint64_t dst_addr,
                          size_t bytes, uint8_t channel);
    void record_dma_mutable_src(
        const GraphDmaSemanticEndpoint& endpoint,
        const uint64_t* live_src_base, int64_t src_offset,
        uint64_t dst_addr, size_t bytes, uint8_t channel);
    void record_dma_mutable_dst(
        const GraphDmaSemanticEndpoint& endpoint,
        uint64_t src_addr, const uint64_t* live_dst_base,
        int64_t dst_offset, size_t bytes, uint8_t channel);
    void record_barrier(uint8_t self_stream, uint8_t target_stream);

    // ==================== Kernel 分发 ====================

    // 取 kernel 并立即 reset_regs()。Op 端公开的统一入口：
    //   Kernel_t* kernel = GET_KERNEL(id);
    //   kernel->set_regs(...)  // 直接 set，reset 已在 runtime 内部完成
    ::rhino_lkn::Kernel_t* get_kernel_reset(KernelId id);

    // Dynamic kernel API —— 给 rpu_bmm 这类 kernel_name 由 (M/N/K/tile) 动态
    // 拼出、无法进 KernelId 枚举的 op。语义和 KernelId 版完全对称:
    //   PASSTHROUGH : 从 KernelCache shared cache 拿,不 clone
    //   RECORDING   : get_program(name) + Kernel_t 独立 clone + push 进 kernels_
    //                 set pending_kernel_name_ + pending_kernel_idx_,后续 enqueue 消费
    //   REPLAYING   : cursor 前向扫下一个 Kernel node,校验 kernel_name 一致
    // Dynamic callers use this entry point so Graph ownership remains explicit.
    ::rhino_lkn::Kernel_t* get_kernel_reset(const std::string& name);

    // Typed DDR-register census.  The outer subsystem opens exactly one scope
    // at Graph cursor/node zero.  BUILD records an exact ordered census;
    // REPLAY re-observes it before get_kernel_reset mutates Kernel_t registers.
    void begin_kernel_register_census(
        const GraphKernelRegisterPolicy& policy,
        uint64_t physical_plan_hash,
        uint64_t live_physical_epoch);
    void checkpoint_kernel_register_census(const std::string& phase_name);
    void complete_kernel_register_census();
    void cancel_kernel_register_census() noexcept;
    bool kernel_register_census_active() const {
        return kernel_register_census_active_;
    }
    const GraphKernelRegisterCensusStats& kernel_register_census_stats() const {
        return kernel_register_census_stats_;
    }
    GraphKernelRegisterWriter stage_kernel_ddr_registers(
        KernelId id, std::vector<GraphDdrRegisterOperandSpec> operands);
    void stage_kernel_no_ddr(KernelId id, GraphKernelNoDdrProof proof);
    void stage_kernel_no_ddr(const std::string& name,
                             GraphKernelNoDdrProof proof);
    // Narrow typed-wrapper seam.  Outside an active schema-5 census these are
    // no-ops; inside it they stage exactly one owner-bound DMA burst which the
    // following record_dma_* calls must consume before another burst/kernel.
    void stage_census_fixed_dma_burst_if_active(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count);
    void stage_census_mutable_dma_burst_if_active(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count);

    // 默认 queue_state (broadcast=true, flush_icache=false) 的 enqueue
    void enqueue(::rhino_lkn::Kernel_t& kernel,
                 const std::vector<uint16_t>& grid_dims,
                 const std::vector<uint8_t>& core_ids);

    // 带 queue_state 的 enqueue —— 供 RpuQueue proxy 使用
    void enqueue(::rhino_lkn::Kernel_t& kernel,
                 const std::vector<uint16_t>& grid_dims,
                 const std::vector<uint8_t>& core_ids,
                 const QueueLaunchState& queue_state);

    // ==================== Data node 捕获 ====================

    // 统一入口：RECORDING 把 data node push 入 nodes_；REPLAYING 验证 kind +
    // bytes 后用传入 data 覆写（dst/src 指针每轮 replay 可能漂移），cursor++。
    // PASSTHROUGH / BUILT 下调用是编程错误，TORCH_CHECK 抛异常。
    // 仅对 MemcpyNodeData / MemsetNodeData 两种类型显式实例化。
    template <typename ND>
    void capture_data(ND data);

    // 只翻 replayable_ bool,不清 nodes_ / kernels_ / segments_。
    // 与 invalidate() / sync_point() 严格区分:
    //  - invalidate(): 清所有录制状态 + 翻 bool
    //  - sync_point(): 执行已录前缀 + 清 nodes_/kernels_/kernel_indices_ + 翻 bool
    //  - mark_non_replayable(): 仅翻 bool;reason 进入 GraphStats 供 Python 侧
    //    通过 RpuGraphCache.snapshot() 读出来,debug 用。end() 看到 !replayable_
    //    走 oneshot 但保留 nodes_ 顺序,让 data / host 节点仍按录制顺序执行。
    void mark_non_replayable() { mark_non_replayable(std::string{}); }
    void mark_non_replayable(std::string reason);
    const std::string& non_replayable_reason() const {
        return non_replayable_reason_;
    }

    // ==================== HostCallback REPLAYING dispatch ====================

    // REPLAYING op stream 期 custom_cpu_fallback 走到 HostCallback op 时调用。
    // 不真跑 CPU op (前序 RPU kernel 还在 prepared queue 里没 launch);只:
    //   1. 校验 cursor 对应已录的 HostCallback 节点
    //   2. 校验 op_handle 一致 + 非 tensor IValue scalar 不漂移
    //   3. 用本轮 stack tensors 覆写节点的 live_tensor_args
    //   4. 合成返回值给 op stream(让下游 Python 调用链继续)
    //      - Tier1+2: 返回 graph-owned 稳定 buffer(output_pool[r])
    //                 Dynamo 路径下入参不稳定,统一走 output_pool。
    //      - Tier3: 不应到达此路径(scope non-replayable,end 走 oneshot 而非
    //        REPLAYING),抛错防御
    //   5. cursor++
    void capture_host_callback_replay_args(const c10::OperatorHandle& op,
                                            torch::jit::Stack& stack);

    // REPLAYING op stream 期 custom_cpu_fallback 走到 Tier3 op(RNG / dyn-shape)
    // 时调。模式类似 capture_host_callback_replay_args,但路由到
    // GraphNodeKind::Tier3Oneshot 节点。同样不真跑 CPU op;真正执行发生在
    // execute_data_node 的 Tier3Oneshot case。
    //   1. 校验 cursor 对应已录的 Tier3OneshotNode
    //   2. 校验 op_handle 一致 + 非 tensor scalar slot 不漂
    //   3. 用本轮 stack tensors 覆写 t.live_tensor_args
    //   4. 合成 transient buffer 返回值给 op stream(下游 Python 拿到稳定 dev_addr
    //      的 tensor,继续 op stream 直到 end)
    //   5. cursor++
    void capture_tier3_oneshot_replay_args(const c10::OperatorHandle& op,
                                            torch::jit::Stack& stack);

    // ==================== 边界 flush ====================

    // 登记一个 DDR 地址，在下一次 kernel batch / sync_point 的边界统一 flush。
    // 调用方是 graph_runtime_capture.cpp / graph_runtime_execute.cpp 的
    // HostCallback / Tier3 stable-buffer 落地点；不进 nodes_，不推进 replay
    // cursor。PASSTHROUGH 下的 flush 走 eager (rpu_ddr_flush 宏)，不经此路径。
    void record_boundary_flush(void* ptr);

    // ==================== Tensor / SPM 生命周期 ====================

    // 纯生命周期锚点：把 t 的所有权挂到 graph 上，直到
    // 图真正执行完 —— tensor_refs_ 由 end() 在执行之后才 clear。RECORDING /
    // REPLAYING 之外是 no-op(PASSTHROUGH 下 op 返回前就同步跑完了,没有延迟窗口),
    // 这与 graph_dma::active() 的判据一致。不做 flush:相干性归调用点的
    // rpu_ddr_flush_force / boundary flush 管,这里再 dc civac 一遍既是纯开销
    // (4096-token prefill = 32 MB cache walk),又给设备刚写过的 staging buffer
    // (wall_oss / pi05 的 emb_stage_)引入一条本不需要的 clean-back 依赖。
    void keep_alive(const at::Tensor& t);

    // ==================== 查询 ====================

    State state() const { return state_; }
    size_t graph_size() const { return nodes_.size(); }
    bool replayable() const { return replayable_; }

    // O(1), allocation-free position snapshot for the active BUILD/REPLAY op
    // stream.  RECORDING position is nodes_.size(); REPLAYING position is the
    // validation cursor.  PASSTHROUGH/BUILT and a partially-consumed ChildGraph
    // reject instead of returning a misleading position.
    GraphOpStreamStamp op_stream_stamp() const;

    // A stamp remains current through successful BUILD -> REPLAY transitions,
    // but not through abort/invalidate/non-replayable fallback or a fresh BUILD.
    bool is_op_stream_stamp_current(const GraphOpStreamStamp& stamp) const;

    // Hash one BUILD-time [begin,end) window without exposing nodes_. Both
    // stamps must come from this graph's current RECORDING generation.
    GraphOpStreamWindowDigest digest_op_stream_window(
        const GraphOpStreamStamp& begin,
        const GraphOpStreamStamp& end) const;

    uint64_t build_generation() const { return build_generation_; }
    // Nonzero only after a replayable signed capture successfully reaches
    // BUILT. Ordinary unobserved graphs compute it lazily on first query;
    // RECORDING windows observed through op_stream_stamp/digest request an
    // eager commit at end(). It is retained across REPLAY and cleared by fresh
    // capture/reset.
    uint64_t build_topology_hash() const;
    // Cold full-graph node-kind summary for post-BUILD admission checks. This
    // does not expose nodes_, arm topology tracing, or mutate the BUILD seal.
    // Extracted child graphs are supported even when they carry no signature.
    GraphNodeKindMask build_node_kind_mask() const;
    // Stable identity of the equality-bearing GraphSignature fields. This is
    // available after BUILD (and while that graph is REPLAYING); op_id_str is
    // intentionally excluded just like GraphSignature::operator==.
    uint64_t built_signature_identity() const;

    // Compatibility no-op: Queue_t instances are owned by the process-wide
    // QueueCache rather than by individual segments.
    void release_prepared_queues();

    // Copy a contiguous BUILT node window into a standalone child graph. The
    // extracted child rebases node/kernel indices to child-local order and supports
    // the same explicit DataPatch replay path as hand-built children.
    std::shared_ptr<RpuKernelGraph> extract_child_window(
        size_t start_idx, size_t end_idx) const;

    // Replace a contiguous BUILT parent window with one ChildGraph node. Replay
    // still runs the original op-stream; when cursor reaches this node, the raw
    // child window is consumed against the child graph and the parent executor
    // launches the child at end().
    void replace_window_with_child(
        size_t start_idx, size_t end_idx,
        std::shared_ptr<RpuKernelGraph> child,
        std::vector<RegisterPatch> input_patches = {},
        std::vector<DataPatch> data_patches = {},
        GraphSignature child_signature = {});

    // 上一次 execute_graph_* 的统计快照
    const GraphStats& debug_stats() const { return last_stats_; }

    // Op 端通过 active().local_spm_pool() 拿到 graph-managed SPM 池:
    //   RECORDING : pool.acquire_recording(...) 创建 LocalSPM_t 并记 ptr_to_slot_
    //   REPLAYING : pool.acquire_replaying() 取 RECORDING 期同序的同一只 LocalSPM_t
    //   PASSTHROUGH: Op 不应调 pool;改走 raw `new LocalSPM_t` + delete 路径
    //                (与 graph 解耦,生命周期跟函数 scope 一致)
    GraphLocalSpmPool& local_spm_pool() { return local_spm_pool_; }
    // 对称 GlobalSPM 池。rpu_sdpa 等用 GlobalSPM_t 作临时 buffer 的 op 走这条。
    GraphGlobalSpmPool& global_spm_pool() { return global_spm_pool_; }

    // ==================== _to_copy admission ====================
    //
    // 两张 range 表配合判定 `_to_copy(src)` 的 src 是不是 HostCallback chain
    // 上的真值依赖,命中即 defer:
    //
    //   persistent_stable_ranges_:
    //     adopt_returns 创建的 graph-owned Tier1/2 stable output_pool buffer。
    //     跨 replay step 不变(stable 本身就是为跨 step 稳定 dev_addr 设计的)。
    //     生命周期:begin (PASSTHROUGH→RECORDING,即新一次 capture) / abort /
    //     invalidate 清。BUILT→REPLAYING 不清。
    //
    //   live_tier1_out_ranges_:
    //     Tier1 `.out` variant 的 fresh `.out` input args(PT functional default
    //     wrapper alloc 的 values_fresh / indices_fresh)。每步 op stream 都新
    //     alloc,所以**只对当前 op stream 有效**。生命周期:每次 begin
    //     (RECORDING / REPLAYING 入口都清)清,然后 RECORDING ctor /
    //     capture_host_callback_replay_args 在每次 host_callback dispatch
    //     时入新地址。
    //
    // 不混存的原因:fresh `.out` args 跨 step 地址会被 caching allocator 复用
    // 给无关 tensor。如果跟 persistent 混在一张表里跨 replay 累积,无关 tensor
    // 会误命中 admission → 重新出现"过度 defer"风险面。
    //
    // 都按 (host_base, dev_base, nbytes) 双 range 记录,命中任一套即算 stable。
    // 双 range 覆盖 view/slice 偏移 + host_ptr 跟 storage data 不完全等价的
    // wrapper 路径(详 register/lookup 实现)。
    // public 是为了 cpp 内 anonymous namespace helper(push_unique_range /
    // range_hit)能引用;range 本身不包含敏感状态,只是 (host_base, dev_base,
    // nbytes) 三元组。
    struct HostCallbackStableRange {
        uintptr_t host_base = 0;   // stable.data_ptr() 的 host 视角整数化
        uintptr_t dev_base = 0;    // RpuGetDevAddr(stable.data_ptr())
        size_t nbytes = 0;         // stable.numel() * stable.element_size()
    };

    void register_host_callback_persistent_stable(const at::Tensor& stable);
    void register_host_callback_live_tier1_out(const at::Tensor& fresh_out_arg);

    // 若 src 在 persistent 或 live 任一表内命中,返回 true。无 graph scope /
    // 两表都空时返回 false。被 rpu_to_copy 的 graph admission 调用。
    bool is_host_callback_stable_input(const at::Tensor& src) const;

    // 给定 src tensor,返回当前 graph scope 内是否应当把 host op 推进 deferred
    // 路径(stable input chain 下游 → 进 host_callback Tier1/2;否则 inline)。
    // 跨 dtype / 跨 device 路径若需要 stable input deferred 直接调本 API。
    //
    // env 控制(默认 unset 走 Auto):
    //   unset / ""    → Auto:仅 stable input 命中时 defer(推荐)
    //   "auto"        → 同 Auto
    //   "0"/"off"     → ForceOff:全 inline(诊断 deferred 是否引入回归)
    //   "false"       → 同 ForceOff
    //   其他非空      → ForceOn:全 defer(诊断用,可能引入回归)
    //
    // env 名:首选 RPU_GRAPH_HOST_OP_DEFER_GATE;兼容 RPU_GRAPH_DEFER_TO_COPY。
    bool should_defer_host_op_input(const at::Tensor& src) const;

    // ==================== ZeroCopy storage holder owner tree =====
    //
    // run_cpu_fallback_zerocopy 创建 zero-copy CPU view 时,view 的 storage
    // 必须额外持有到 redispatch + return 处理结束。owner tree 跟踪
    // 同一 lifeline 的资源状态。生命周期通过 ZeroCopyHolderGuard
    // 管理:guard ctor 记录当前 size,dtor 截断回原 size,
    // 避免嵌套 fallback 累积 / 跨调用泄漏。
    void register_zero_copy_holder(const at::Tensor& view);
    size_t zero_copy_holder_count() const;
    void truncate_zero_copy_holders(size_t to_size);

    // ==================== Tier3 transient buffer owner tree =====
    //
    // Tier3OneshotNode 真跑出来的 fresh return tensor 跨 replay dev_addr 漂(每
    // 步 RNG 都新 alloc),下游 RPU kernel 在 RECORDING 期 set_regs 写的是漂的
    // dev_addr → REPLAYING 会读旧地址。每个 Tensor return 对应一只
    // graph-managed transient buffer,跨 replay dev_addr 稳定;内容由
    // REPLAYING 期 op 真跑后 copy_ 进去(每步 RNG 不同)。
    //
    // 跟 host_callback_persistent_owners_ 同模式 — owner tree 描述符 + 实际
    // RPU tensor 引用(防 GC)。两个 vector 同索引一一对应。
    //
    // lifecycle:begin (PASSTHROUGH→RECORDING) / abort / invalidate 全清(跨
    // capture 不复用 buffer,因为新一轮 capture 可能 op shape / dtype 不同)。
    // BUILT→REPLAYING 不清(地址稳定语义)。
    // dev_addr 由 caller 预算后传入，作为唯一地址来源，保持 owner
    // tree 与节点 last_output_dev_addrs 一致。
    size_t register_tier3_transient_buffer(int owner_node_id,
                                            const at::Tensor& buffer,
                                            uint64_t dev_addr);
    // 反查:transient_resource_id → RPU tensor 引用。Tier3Oneshot executor 用它
    // 把 fresh return copy_ 进去并替换 stack slot。
    const at::Tensor& tier3_transient_tensor_at(size_t idx) const;

private:
    friend class RpuKernelGraphContractTestAccess;
    friend struct v3::SpmFmbYieldTargetLauncherAccess;
    friend class v3::SpmCompositeTraceCoordinator;
    friend class GraphKernelRegisterWriter;
    friend class GraphKernelRegisterCensusGuard;
    friend class GraphKernelRegisterOuterFastTicket;
    struct OuterFastTensorSnapshot;
    // begin() / begin(sig) 的公共骨架。sig.has_value() 时携带 admission key;
    // nullopt 表示无签名,直接 RECORDING(不写 pending_signature_)。
    void begin_impl(const std::optional<GraphSignature>& sig);
    // Explicitly opt this RECORDING scope into retained-arena publication.
    // Only the composite coordinator may request it, after validating the
    // live lease. Ordinary physical Graph users retain the standard
    // acquire/build/replay/release lifecycle and never create a provisional
    // registry entry.
    void request_retained_physical_arena_build();
    void invalidate_impl(bool from_abort);
    void check_foreign_graph_execution_allowed(
        const char* operation,
        bool allow_managed_semantic_yield_scope_entry = false) const;

    uint64_t topology_hash_for_window(size_t begin, size_t end) const;
    uint64_t semantic_spm_producer_yield_id_digest_for_window(
        size_t begin, size_t end) const;
    std::vector<GraphSemanticSpmProducerYieldOccurrence>
    semantic_spm_producer_yields_for_window(
        size_t begin, size_t end) const;
    GraphNodeKindMask node_kind_mask_for_window(size_t begin,
                                                size_t end) const;
    void record_dma_node(DmaNodeData data);
    bool has_semantic_dma_outside_window(size_t begin, size_t end) const;
    void bind_semantic_dma_owner_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation);
    void arm_semantic_dma_owner_replay_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation);
    void arm_semantic_spm_peer_replay_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation);
    void stage_semantic_spm_producer_yield_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation,
        uint64_t semantic_yield_id,
        uint8_t writer_kind);
    void cancel_semantic_spm_producer_yield_for_fmb() noexcept;
    void poison_semantic_spm_producer_yield_for_fmb() noexcept;
    void arm_semantic_spm_producer_yield_replay_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation,
        uint64_t semantic_yield_id_digest);
    uint64_t semantic_spm_producer_yield_id_digest_for_fmb() const;
    void begin_canonical_dma_build_trace_for_fmb(uint64_t profile_hash);
    void complete_canonical_dma_build_trace_for_fmb();
    void begin_canonical_dma_callback_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation,
        uint64_t profile_hash,
        size_t layer_group_ordinal,
        size_t member_count);
    void end_canonical_dma_callback_for_fmb();
    void cancel_canonical_dma_callback_for_fmb() noexcept;
    void begin_canonical_dma_burst_for_fmb(
        const GraphDmaSemanticEndpoint& endpoint,
        uint64_t spm_peer_id);
    void end_canonical_dma_burst_for_fmb();
    void cancel_canonical_dma_burst_for_fmb() noexcept;
    bool validate_canonical_dma_before_record(
        const GraphDmaSemanticEndpoint& endpoint,
        DmaNodeData::Variant variant,
        size_t bytes,
        uint8_t channel) const;
    void commit_canonical_dma_after_record(bool canonical);

    const KernelNodeData*
        next_replay_kernel_for_semantic_spm_producer_yield() const;
    void validate_semantic_spm_producer_yield_before_kernel_lookup() const;
    void bind_semantic_spm_producer_yield_kernel_lookup(
        ::rhino_lkn::Kernel_t& kernel);
    void attach_semantic_spm_producer_yield_to_recording_node(
        KernelNodeData& node,
        const ::rhino_lkn::Kernel_t& kernel) const;
    void validate_semantic_spm_producer_yield_replay_node(
        const KernelNodeData& node,
        const ::rhino_lkn::Kernel_t& kernel) const;
    void commit_semantic_spm_producer_yield_after_kernel_enqueue() noexcept;
    bool has_semantic_spm_producer_yield_marker_nodes() const;
    void verify_semantic_spm_producer_yield_replay_arm(
        const char* operation) const;
    void verify_semantic_spm_producer_yield_end_preflight();
    void verify_composite_fmb_end_preflight() const;

    // 取 kernel 的原始入口：PASSTHROUGH 返回共享 kernel；RECORDING 创建私有 kernel；
    // REPLAYING 返回已录制 kernel。调用者需自行调用 reset_regs()。
    ::rhino_lkn::Kernel_t* get_kernel(KernelId id);
    // Dynamic 版本,语义同上;set pending_kernel_name_ 代替 pending_kernel_id_。
    ::rhino_lkn::Kernel_t* get_kernel(const std::string& name);

    const GraphKernelRegisterRule& kernel_register_rule_for(
        KernelId id) const;
    const GraphKernelRegisterRule& kernel_register_rule_for(
        const std::string& name) const;
    void admit_kernel_register_lookup(KernelId id);
    void admit_kernel_register_lookup(const std::string& name);
    void bind_pending_kernel_register_lookup(
        ::rhino_lkn::Kernel_t& kernel);
    void write_pending_kernel_ddr_registers(
        uint64_t ticket, ::rhino_lkn::Kernel_t& kernel);
    void consume_pending_kernel_register_census(
        ::rhino_lkn::Kernel_t& kernel,
        KernelNodeData* recording_node, size_t replay_node_idx);
    void stage_kernel_register_outer_fast_physical_digest(uint64_t digest);
    static OuterFastTensorSnapshot snapshot_kernel_register_outer_fast_tensor(
        const at::Tensor& tensor);
    static void check_kernel_register_outer_fast_tensor_exact(
        const OuterFastTensorSnapshot& built,
        const OuterFastTensorSnapshot& live,
        const char* context);
    static void check_kernel_register_outer_fast_tensor_live(
        const OuterFastTensorSnapshot& snapshot,
        const char* context);
    static void check_kernel_register_outer_fast_dma_range(
        const OuterFastTensorSnapshot& owner,
        size_t byte_begin, size_t byte_count);
    void stage_kernel_register_outer_fast_component(
        std::vector<std::pair<std::shared_ptr<const uint64_t>, uint64_t>>
            owner_stamps,
        std::vector<at::Tensor> key_cache,
        std::vector<at::Tensor> value_cache);
    void stage_kernel_register_outer_fast_invocation_authority(
        std::shared_ptr<const uint64_t> prepared_owner,
        std::shared_ptr<uint64_t> consumed_owner,
        uint64_t expected_generation);
    void stage_kernel_register_outer_fast_invocation_visibility(
        const at::Tensor& tensor);
    void stage_kernel_register_fixed_dma_burst(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count);
    void stage_kernel_register_mutable_dma_burst(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count);
    void bind_kernel_register_outer_fast_fixed_dma(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count);
    void bind_kernel_register_outer_fast_mutable_dma(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count);
    GraphKernelRegisterOuterFastTicket
        validate_kernel_register_outer_fast_replay();
    void commit_kernel_register_outer_fast_replay(
        GraphKernelRegisterOuterFastTicket&& ticket);
    void cancel_kernel_register_outer_fast_ticket(uint64_t nonce) noexcept;
    void consume_kernel_register_dma_burst(DmaNodeData& data);
    void check_kernel_register_dma_burst_closed(
        const char* operation) const;
    void complete_kernel_register_outer_fast_ordinary();
    void build_kernel_register_retained_catalog();
    size_t validate_kernel_register_retained_occurrences() const;
    void verify_kernel_register_outer_fast_preflight() const;
    void verify_kernel_register_outer_fast_launch_preflight() const;
    void flush_kernel_register_outer_fast_force_visibility() const noexcept;
    void flush_kernel_register_outer_fast_force_post_visibility()
        const noexcept;
    void commit_kernel_register_outer_fast_prepared(bool fast) noexcept;
    void mark_kernel_register_outer_fast_execution_success() noexcept;
    void clear_kernel_register_invocation_state() noexcept;
    void clear_kernel_register_build_state() noexcept;
    void check_kernel_register_node_emission(const char* operation) const;
    void observe_kernel_register_node(GraphNodeKind kind,
                                      const std::string* kernel_name);
    void verify_kernel_register_end_preflight() const;
    static bool force_oneshot_on_replay_enabled();

    // 线性扫 nodes_，把连续且 queue_state 兼容的 Kernel/Dma/Barrier 合并为
    // segment；Kernel 的 core_ids 可不同，segment 取最大执行域。其它 data
    // node 作为 segment 边界。
    void build_segments_from_nodes();

    // RECORDING 收尾：build_segments_from_nodes + 段间交错执行 data nodes +
    // 每段用本 graph entry 的 private Queue_t prepare + launch。Queue_t 跨段
    // 复用，不由 segment 持有。
    void execute_graph_for_recording();

    // REPLAYING 收尾：按 segment 走 QueueCache::get + build_batch +
    // enqueu_batch。段间 data nodes 重复执行 (capture_data 已在 REPLAYING
    // 期覆写过各 data node 的 dst/src)。
    void execute_graph_for_replaying();

    // 对一个 segment 设置 Queue_t 状态并 prepare 其 kernel 列表。
    // 同时填 seg.mutable_dmas (每个 add_dma_kernel_mutable 累计 dma_id
    // + push 槽位),让后续 REPLAY 能走 sync-only 快路径。
    void prepare_segment_queue(Segment& seg, ::rhino_lkn::Queue_t& wq);

    // replay 路径发射单段:从本 entry 的 private_queue_ (经 ensure_private_queue
    // 取);若 private_queue_built_segment_idx_ 与本段 idx 命中则走 sync-only
    // (launch_segment_sync_only),否则 full rebuild (prepare_segment_queue +
    // build_batch + enqueu_batch)。
    void launch_segment_for_replay(Segment& seg);

    // sync-only fast path:walk seg.mutable_dmas + update_dma_kernel +
    // sync_mutable_params + enqueu_batch。前提:private_queue_'s kd_buf 仍持本段
    // build_batch 结果 (由 launch_segment_for_replay 的 segment-idx fingerprint
    // 判定后才调用本函数)。返回 SDK rc (0 = 成功;非 0 → caller TORCH_CHECK)。
    uint32_t launch_segment_sync_only(Segment& seg);

    // 取本 cache entry 私有的 Queue_t,按 num_cores 懒分配。
    // 与 QueueCache::instance() 共享池**完全隔离**:immediate DMA 与
    // execute_graph_oneshot 走 QueueCache，PASSTHROUGH kernel 走另一条隔离
    // direct-launch queue；本 entry 的 cacheable execute 路径
    // (execute_graph_for_recording + launch_segment_for_replay) 走这个私有
    // Queue，从而避免任一旁路作废其 prepared batch。
    //
    // 行为说明:
    //   - 首次调用按 segment 最大 core 域构造；若后续 segment 的最大域不同，
    //     销毁旧 queue + 重建。**同一 segment 内**混用 1/8-core kernel 不会触发
    //     重建：每个 kernel 的 core_ids 已独立烤入 batch。跨段重建只损失
    //     prepared-kd_buf 复用收益，不影响正确性。
    //   - 不在 PASSTHROUGH (raw kernel 走隔离 direct-launch queue) 与
    //     execute_graph_oneshot (non-replayable 一次性降级，走 QueueCache)
    //     中使用；两条路径都不会被 REPLAY。
    //   - 生命周期:unique_ptr,随 RpuKernelGraph 实例析构(RpuGraphCache::evict
    //     / clear) 自动释放底层 Queue_t,无需显式 hook。
    ::rhino_lkn::Queue_t* ensure_private_queue(size_t num_cores);

    // non-replayable 降级路径：一次性按 segment 分组提交 kernel + 段间交错执行
    // data nodes，执行完丢弃 segments_。
    void execute_graph_oneshot();

    // Parent ChildGraph replay:当 BUILT parent 含 ChildGraph 节点但 replay 仍
    // 走原始 op-stream 时,把 raw child stream 消费到 child graph 上,而不是在
    // parent cursor 处匹配 raw parent nodes。
    ChildGraphNodeData& enter_replay_child_skip(const char* caller);
    void finish_replay_child_skip_if_complete();
    template <typename ND>
    bool capture_replay_child_skip_data(ND data);

    friend class v3::FusedModelBase;

    // 按 boundary_flush_ptrs_ 顺序调用 rhino_lkn::RpuDdrFlush;若
    // g_rpu_ddr_flush_enabled 为 false 则整体跳过。不 clear 集合,clear 由
    // 调用方控制(便于在 batch 前后各刷一次后再 clear)。
    void flush_boundary_ptrs();

    // 执行单个 data node（Memcpy / Memset）；末尾统一 __sync_synchronize()。
    // 三条 execute_graph_* 路径共享此 helper。
    void execute_data_node(const GraphNode& node);

    // active 栈化:nested begin 允许;同一 graph 重入仍被拒。
    // active() 返回栈顶(空时返回 fallback_graph_,保持单图老语义)。
    static thread_local std::vector<RpuKernelGraph*> active_stack_;
    static thread_local RpuKernelGraph fallback_graph_;

    // Declared before every Graph resource so its destructor runs last.  It
    // publishes eviction only after queues, nodes, segments, and tensor
    // keepalives have completed destruction; publishing from the Graph
    // destructor body would let the physical owner release SPM too early.
    struct LifetimeRetirementSentinel {
        explicit LifetimeRetirementSentinel(uint64_t id) : lifetime_id(id) {}
        ~LifetimeRetirementSentinel();
        LifetimeRetirementSentinel(const LifetimeRetirementSentinel&) = delete;
        LifetimeRetirementSentinel& operator=(
            const LifetimeRetirementSentinel&) = delete;

        uint64_t lifetime_id = 0;
    };
    LifetimeRetirementSentinel lifetime_retirement_;

    State state_ = State::PASSTHROUGH;
    bool replayable_ = true;
    bool scope_open_ = false;
    // One move-only process claim per open scope.  BUILD is not permanently
    // pinned: after end/abort releases this token, a later sequential replay
    // may bind the same Graph from another thread.
    std::optional<RpuExecutionCoordinator::Claim> execution_claim_;
    std::thread::id execution_owner_thread_;
    // Process-unique object lifetime identity.  It closes same-address ABA for
    // external guards independently of the per-BUILD nonce.
    uint64_t graph_lifetime_id_ = 0;
    // An explicitly opted-in signed BUILD performed under a physical claim
    // first records the scope token here, then publishes a provisional
    // registry entry at commit.  The entry must be atomically converted to a
    // retained guard or invalidated before the physical claim may be released.
    uint64_t recording_physical_execution_token_ = 0;
    uint64_t built_physical_execution_token_ = 0;
    bool retained_physical_build_requested_ = false;
    // Prepared-child APIs can detach or externally embed baked register
    // values.  Once either relation has existed, this Graph lifetime is never
    // eligible to become a retained physical root.
    mutable bool ever_extracted_child_window_ = false;
    bool ever_embedded_as_child_ = false;

    // mark_non_replayable() 留下的原因字符串。GraphStats 暴露给 Python 侧调试。
    // begin_impl(RECORDING 入口) 和 invalidate() 都重置成空。
    std::string non_replayable_reason_;

    // signature-driven admission。pending_signature_ 在 begin(sig) 进 RECORDING
    // 时设置,end() BUILT 完成时写入 built_signature_;has_built_signature_ 区分
    // "从未录制"和"录了但 sig 恰好全零(empty)"两种状态。begin() 无签名走
    // RECORDING 时 pending_signature_ 仍清空,end() 完成时不写 built_signature_
    // (也不命中 admission)。
    GraphSignature built_signature_;
    bool has_built_signature_ = false;
    std::optional<GraphSignature> pending_signature_;
    std::weak_ptr<const uint64_t> semantic_dma_owner_generation_;
    uint64_t semantic_dma_expected_owner_generation_ = 0;
    bool has_semantic_dma_nodes_ = false;
    bool semantic_dma_replay_owner_armed_ = false;
    bool has_semantic_spm_peer_nodes_ = false;
    bool semantic_spm_peer_replay_armed_ = false;

    struct PendingSemanticSpmProducerYield {
        uint64_t semantic_yield_id = 0;
        uint8_t writer_kind = 0;
        std::weak_ptr<const uint64_t> owner_generation;
        uint64_t expected_owner_generation = 0;
        ::rhino_lkn::Kernel_t* lookup_kernel = nullptr;
        bool kernel_lookup_seen = false;
    };
    std::optional<PendingSemanticSpmProducerYield>
        pending_semantic_spm_producer_yield_;
    std::weak_ptr<const uint64_t>
        semantic_spm_producer_yield_owner_generation_;
    uint64_t semantic_spm_producer_yield_expected_owner_generation_ = 0;
    uint64_t semantic_spm_producer_yield_id_digest_ = 0;
    uint64_t semantic_spm_producer_yield_replay_arm_id_digest_ = 0;
    bool has_semantic_spm_producer_yield_nodes_ = false;
    bool semantic_spm_producer_yield_replay_armed_ = false;
    bool semantic_spm_producer_yield_poisoned_ = false;

    struct CompositeFmbInvocationState {
        enum class Phase : uint8_t {
            None,
            BuildOpen,
            BuildComplete,
            ReplayOpen,
            ReplayComplete,
            Poisoned,
        };
        Phase phase = Phase::None;
        const void* coordinator = nullptr;
        uint64_t identity = 0;
        uint64_t build_generation = 0;
        uint64_t signature_identity = 0;
        uint64_t signature_segment_key = 0;
        size_t expected_occurrences = 0;
        size_t completed_occurrences = 0;
        size_t expected_producer_yields = 0;
        size_t completed_producer_yields = 0;
        size_t next_outer_begin = 0;
    };
    CompositeFmbInvocationState composite_fmb_invocation_;

    struct CanonicalDmaBurstPermit {
        bool active = false;
        uint64_t endpoint_id = 0;
        uint64_t spm_peer_id = 0;
        DmaNodeData::Variant variant = DmaNodeData::Variant::Fixed;
        size_t bytes = 0;
        uint8_t occurrence_count = 0;
        uint8_t next_channel = 0;
    };
    bool canonical_dma_callback_active_ = false;
    bool canonical_dma_poisoned_ = false;
    bool canonical_dma_build_trace_active_ = false;
    bool canonical_dma_build_trace_complete_ = false;
    uint64_t canonical_dma_build_trace_profile_hash_ = 0;
    size_t canonical_dma_build_trace_node_begin_ = 0;
    size_t canonical_dma_build_trace_node_end_ = 0;
    uint64_t canonical_dma_callback_profile_hash_ = 0;
    std::weak_ptr<const uint64_t> canonical_dma_callback_owner_generation_;
    uint64_t canonical_dma_callback_expected_owner_generation_ = 0;
    size_t canonical_dma_callback_layer_group_ordinal_ = 0;
    size_t canonical_dma_callback_member_count_ = 0;
    size_t canonical_dma_callback_next_member_ = 0;
    uint8_t canonical_dma_callback_next_role_ = 1;
    CanonicalDmaBurstPermit canonical_dma_burst_permit_;

    size_t cursor_ = 0;

    // Process-unique monotonic identity of the current BUILD attempt. Fresh
    // RECORDING receives a new global nonce; BUILT/REPLAY retain it so a
    // candidate callback trace can be committed after the outer Graph scope
    // closes. Neither rebuild nor same-address object reuse can revive stamps.
    uint64_t build_generation_ = 0;
    mutable uint64_t build_topology_hash_ = 0;
    // Cold BUILD instrumentation gate. Ordinary GraphCache captures never scan
    // nodes_ merely to produce a topology hash; an explicit RECORDING stamp or
    // window digest arms the one-time end() commit instead.
    mutable bool topology_hash_requested_ = false;

    // post_fn-start cursor for skip_layer_body_for_fast_replay (post_fn
    // graphs). Recorded during RECORDING via mark_post_fn_cursor(); reset on
    // every fresh capture / invalidate so a stale value can't leak across
    // rebuilds.
    size_t post_fn_cursor_ = 0;
    bool has_post_fn_cursor_ = false;

    // Set by skip_op_stream_for_fast_replay (the FULL op-stream skip, no post_fn
    // re-emit) → this replay emitted NO kernel set_regs, so the kd_buf params are
    // unchanged from BUILD/last-sync and sync_mutable_params() is redundant work
    // because it scans every recorded kernel.
    // Consumed + reset per replay in execute_graph_for_replaying. NOT set by
    // skip_layer_body_for_fast_replay (post_fn re-emits → set_regs possible).
    bool op_stream_fully_skipped_ = false;

    // Cold per-BUILD policy. Captured on entry to RECORDING and retained for
    // every REPLAY of that graph, so a later model's environment cannot alter
    // whether this graph synchronizes its prepared kernel parameters.
    bool fast_replay_skip_sync_ = false;

    struct ReplayChildSkipState {
        bool active = false;
        size_t parent_node_idx = 0;
        size_t child_cursor = 0;
    };
    ReplayChildSkipState replay_child_skip_;

    std::optional<KernelId> pending_kernel_id_;
    // Dynamic kernel API:走 KernelCache::get_kernel(string) 的 op(rpu_bmm 等)
    // 通过 get_kernel_reset(name) 进来时 set 这里;互斥于 pending_kernel_id_。
    std::optional<std::string> pending_kernel_name_;
    size_t pending_kernel_idx_ = 0;

    struct PendingKernelRegisterCensus {
        std::optional<KernelId> kernel_id;
        std::optional<std::string> kernel_name;
        GraphKernelRegisterCensus census;
        uint64_t writer_ticket = 0;
        size_t replay_node_idx = 0;
        bool kernel_lookup_seen = false;
        bool writer_required = false;
        bool writer_committed = false;
        ::rhino_lkn::Kernel_t* lookup_kernel = nullptr;
        ::rhino_lkn::Kernel_t* written_kernel = nullptr;
        std::vector<GraphKernelRegisterWriter::RegisterPair> expected_pairs;
    };

    struct OuterFastTensorSnapshot {
        at::Tensor tensor_owner;
        c10::Storage storage_owner;
        const c10::TensorImpl* tensor_impl = nullptr;
        const c10::StorageImpl* storage_impl = nullptr;
        const void* data_ptr = nullptr;
        const void* storage_data_ptr = nullptr;
        at::ScalarType dtype = at::ScalarType::Undefined;
        c10::Device device{c10::DeviceType::CPU};
        c10::Layout layout = c10::Layout::Strided;
        std::vector<int64_t> sizes;
        std::vector<int64_t> strides;
        int64_t storage_offset = 0;
        uint64_t topology_id = 0;
        size_t tensor_nbytes = 0;
        size_t storage_nbytes = 0;
        uint64_t dev_addr = 0;
    };

    struct OuterFastOwnerStamp {
        std::shared_ptr<const uint64_t> owner;
        uint64_t expected_generation = 0;
    };

    struct OuterFastBuildOwnerStamp {
        std::weak_ptr<const uint64_t> owner;
        uint64_t expected_generation = 0;
    };

    struct OuterFastInvocationAuthority {
        std::shared_ptr<const uint64_t> prepared_owner;
        std::shared_ptr<uint64_t> consumed_owner;
        uint64_t expected_generation = 0;
    };

    struct OuterFastBuildInvocationAuthority {
        std::weak_ptr<const uint64_t> prepared_owner;
        std::weak_ptr<uint64_t> consumed_owner;
    };

    struct OuterFastComponentCandidate {
        std::vector<OuterFastOwnerStamp> owners;
        std::vector<OuterFastTensorSnapshot> key_cache;
        std::vector<OuterFastTensorSnapshot> value_cache;
    };

    struct OuterFastBuildComponent {
        std::vector<OuterFastBuildOwnerStamp> owners;
        std::vector<OuterFastTensorSnapshot> key_cache;
        std::vector<OuterFastTensorSnapshot> value_cache;
    };

    struct OuterFastDmaBurstCandidate {
        DmaNodeData::Variant variant = DmaNodeData::Variant::Fixed;
        GraphOuterFastDmaSide side = GraphOuterFastDmaSide::None;
        OuterFastTensorSnapshot owner;
        uint64_t* live_base = nullptr;
        size_t byte_begin = 0;
        size_t byte_count = 0;
        size_t expected_occurrence_count = 0;
        size_t observed_occurrence_count = 0;
        size_t owner_ordinal = 0;
    };

    struct OuterFastBuildDmaBurst {
        DmaNodeData::Variant variant = DmaNodeData::Variant::Fixed;
        GraphOuterFastDmaSide side = GraphOuterFastDmaSide::None;
        OuterFastTensorSnapshot owner;
        uint64_t* live_base = nullptr;
        size_t byte_begin = 0;
        size_t byte_count = 0;
        size_t expected_occurrence_count = 0;
        size_t owner_ordinal = 0;
    };

    struct OuterFastFixedDmaOwner {
        GraphOuterFastDmaSide side = GraphOuterFastDmaSide::None;
        OuterFastTensorSnapshot owner;
        size_t byte_begin = 0;
        size_t byte_count = 0;
        size_t occurrence_count = 0;
    };

    struct OuterFastMutableDmaSlot {
        GraphOuterFastDmaSide side = GraphOuterFastDmaSide::None;
        uint64_t* live_base = nullptr;
        uint64_t owner_topology_id = 0;
        size_t byte_begin = 0;
        size_t byte_count = 0;
        size_t occurrence_count = 0;
        std::optional<OuterFastTensorSnapshot> last_successful_destination;
    };

    struct OuterFastBuildState {
        bool staged = false;
        bool committed = false;
        uint64_t physical_digest = 0;
        uint64_t topology_digest = 0;
        std::vector<OuterFastBuildComponent> components;
        std::vector<OuterFastBuildDmaBurst> dma_bursts;
        std::vector<OuterFastFixedDmaOwner> fixed_dma_owners;
        std::vector<OuterFastMutableDmaSlot> mutable_dma_slots;
        std::vector<size_t> invocation_visibility_fixed_owner_ordinals;
        std::vector<OuterFastBuildInvocationAuthority>
            invocation_authorities;
        size_t mutable_occurrence_count = 0;
    };

    enum class OuterFastCandidateMode : uint8_t {
        Unset = 0,
        Ordinary = 1,
        Fast = 2,
    };

    struct OuterFastCandidate {
        bool engaged = false;
        OuterFastCandidateMode mode = OuterFastCandidateMode::Unset;
        std::optional<uint64_t> physical_digest;
        uint64_t build_generation = 0;
        uint64_t build_topology_hash = 0;
        uint64_t policy_digest = 0;
        uint64_t plan_hash = 0;
        uint64_t live_epoch = 0;
        std::vector<OuterFastComponentCandidate> components;
        std::vector<OuterFastInvocationAuthority> invocation_authorities;
        std::vector<OuterFastTensorSnapshot> invocation_visibility;
        std::vector<OuterFastDmaBurstCandidate> ordinary_dma_bursts;
        std::optional<size_t> open_ordinary_dma_burst;
        std::vector<OuterFastDmaBurstCandidate> fast_fixed_dma_bindings;
        std::vector<OuterFastDmaBurstCandidate> fast_mutable_dma_bindings;
    };

    struct OuterFastNodeAnnotation {
        DmaNodeData* node = nullptr;
        GraphOuterFastDmaSide side = GraphOuterFastDmaSide::None;
        size_t owner_ordinal = 0;
        uint64_t owner_topology_id = 0;
        size_t owner_byte_offset = 0;
    };

    struct OuterFastPreparedCommit {
        bool valid = false;
        bool fast = false;
        std::optional<OuterFastBuildState> build_state;
        std::vector<OuterFastNodeAnnotation> node_annotations;
        std::vector<std::pair<uint64_t*, uint64_t>> slot_writes;
        std::vector<void*> boundary_visibility;
        std::vector<void*> force_visibility;
        std::vector<void*> force_post_visibility;
        std::vector<OuterFastTensorSnapshot> invocation_owners;
        std::vector<OuterFastOwnerStamp> invocation_owner_stamps;
        std::vector<OuterFastInvocationAuthority> invocation_authorities;
        std::vector<OuterFastTensorSnapshot> normal_visibility_owners;
        std::vector<OuterFastTensorSnapshot> force_visibility_owners;
        std::vector<OuterFastTensorSnapshot> force_post_visibility_owners;
        std::vector<std::pair<size_t, OuterFastTensorSnapshot>>
            mutable_successful_owners;
        GraphKernelRegisterCensusStats canonical_census;
        size_t retained_validated = 0;
        size_t mutable_slot_count = 0;
        size_t mutable_occurrence_count = 0;
    };

    struct RetainedRegisterRef {
        size_t node_index = 0;
        size_t operand_index = 0;
        size_t kernel_index = 0;
    };

    // The policy/plan and node-side census survive BUILT→REPLAY.  Everything
    // named invocation/pending is reset at each begin/end/abort.  A BUILD is not
    // authoritative until execute_graph_for_recording returns successfully.
    bool has_kernel_register_census_nodes_ = false;
    bool kernel_register_census_build_committed_ = false;
    bool kernel_register_census_active_ = false;
    bool kernel_register_census_complete_ = false;
    bool kernel_register_census_poisoned_ = false;
    bool kernel_register_census_invocation_armed_ = false;
    uint64_t kernel_register_census_plan_hash_ = 0;
    uint64_t kernel_register_census_policy_digest_ = 0;
    uint64_t kernel_register_census_live_epoch_ = 0;
    uint64_t next_kernel_register_writer_ticket_ = 0;
    size_t kernel_register_census_node_begin_ = 0;
    size_t kernel_register_census_next_phase_ = 0;
    GraphKernelRegisterCensusStats kernel_register_census_phase_begin_stats_;
    GraphKernelRegisterPolicy kernel_register_census_policy_;
    GraphKernelRegisterCensusStats kernel_register_census_stats_;
    std::vector<size_t> kernel_register_census_rule_counts_;
    std::optional<PendingKernelRegisterCensus>
        pending_kernel_register_census_;
    std::vector<RetainedRegisterRef> kernel_register_retained_catalog_;
    OuterFastBuildState kernel_register_outer_fast_build_state_;
    OuterFastCandidate kernel_register_outer_fast_candidate_;
    std::optional<OuterFastPreparedCommit>
        kernel_register_outer_fast_prepared_;
    bool kernel_register_outer_fast_validation_armed_ = false;
    bool kernel_register_outer_fast_hit_ = false;
    uint64_t kernel_register_outer_fast_ticket_nonce_ = 0;
    uint64_t next_kernel_register_outer_fast_ticket_nonce_ = 0;
    uint64_t kernel_register_outer_fast_invocation_physical_digest_ = 0;
    std::vector<OuterFastTensorSnapshot>
        kernel_register_outer_fast_invocation_owners_;
    std::vector<OuterFastOwnerStamp>
        kernel_register_outer_fast_invocation_owner_stamps_;
    std::vector<OuterFastInvocationAuthority>
        kernel_register_outer_fast_invocation_authorities_;
    std::vector<OuterFastTensorSnapshot>
        kernel_register_outer_fast_normal_visibility_owners_;
    std::vector<OuterFastTensorSnapshot>
        kernel_register_outer_fast_force_visibility_owners_;
    std::vector<OuterFastTensorSnapshot>
        kernel_register_outer_fast_force_post_visibility_owners_;
    std::vector<std::pair<uint64_t*, uint64_t>>
        kernel_register_outer_fast_invocation_slot_values_;
    std::vector<void*>
        kernel_register_outer_fast_invocation_force_visibility_;
    std::vector<void*>
        kernel_register_outer_fast_invocation_force_post_visibility_;
    std::vector<std::pair<size_t, OuterFastTensorSnapshot>>
        kernel_register_outer_fast_mutable_successful_owners_;
    size_t kernel_register_outer_fast_retained_validated_ = 0;
    size_t kernel_register_outer_fast_mutable_slot_count_ = 0;
    size_t kernel_register_outer_fast_mutable_occurrence_count_ = 0;
    size_t kernel_register_host_reemitted_node_count_ = 0;
    size_t kernel_register_host_reemitted_kernel_count_ = 0;

    std::vector<std::unique_ptr<::rhino_lkn::Kernel_t>> kernels_;
    std::vector<GraphNode> nodes_;
    std::vector<Segment> segments_;
    std::vector<at::Tensor> tensor_refs_;

    // Op 代码调 rpu_ddr_flush(ptr) 时,RECORDING/REPLAYING 期登记于此,
    // 由 flush_boundary_ptrs() 在 batch 边界统一下发。
    std::vector<void*> boundary_flush_ptrs_;

    // execute_graph_* + flush_boundary_ptrs 写入的统计快照
    GraphStats last_stats_;

    // 本 cache entry 私有的 Queue_t (lazy-init via ensure_private_queue)。
    // 详细语义见 ensure_private_queue 注释。RAII 析构自动归还 BufferPool;
    // 不在 begin/end/invalidate 中显式 reset(REPLAY 期保留同一 Queue 才能
    // 在 sync-only fast path 中享受 kd_buf reuse 的收益)。
    std::unique_ptr<::rhino_lkn::Queue_t> private_queue_;
    uint8_t private_queue_core_num_ = 0;

    // Segment fingerprint：private_queue_ 的 kd_buf 当前对应 segments_
    // 的哪个 idx (-1 = 空 / 未 build / 已失效)。launch_segment_for_replay
    // 用此判断能否走 sync-only fast path (idx 匹配 + core_num 匹配 →
    // update_dma_kernel + sync_mutable_params + enqueu_batch;否则 full
    // rebuild prepare_segment_queue + 重设 idx)。Reset 时机:
    //   - ensure_private_queue 重建 Queue_t 时 (core_num mismatch)
    //   - invalidate / abort (segments_ 被清,所有 idx 失效)
    //   - prepare_segment_queue 写新 seg 前 (前段 kd_buf 即将被覆写)
    // 必须使用 segment fingerprint，而不能只检查 batch_built_；本字段是该
    // fingerprint 的唯一来源。
    ssize_t private_queue_built_segment_idx_ = -1;

    // Per-graph cumulative REPLAY count, incremented at
    // entry of execute_graph_for_replaying. INFO logs the first-replay-per-
    // graph transition; DEBUG logs every call. Not part of any control flow.
    size_t total_replay_count_ = 0;

    // HW perf trace BUILD-time snapshot — set by execute_graph_for_recording
    // from rpu_hw_perf_snapshot().enabled, frozen for the lifetime of this
    // graph. REPLAY only dumps when (current snapshot.enabled && this flag).
    // `set_hw_perf_trace(True, ...)` produces Chrome trace JSONs only when the
    // setting was enabled at BUILD. A toggle after BUILD does not
    // retroactively enable trace on this graph (Queue_t::set_enable_hw_perf
    // is sticky on kd_buf written at BUILD); user must reset_graph_cache().
    bool hw_perf_enabled_at_build_ = false;

    // graph-managed SPM 池。生命周期: PASSTHROUGH→RECORDING 时 clear(),
    // BUILT→REPLAYING 时 reset_cursor(), invalidate()/abort() 时 clear()。详见
    // graph_runtime.cpp::begin / invalidate。Op 端通过 local_spm_pool() /
    // global_spm_pool() 公开 getter 访问。
    GraphLocalSpmPool local_spm_pool_;
    GraphGlobalSpmPool global_spm_pool_;

    // HostCallback stable output 登记表(struct 在 public section)。
    // 详细职责 / 生命周期见 register_host_callback_persistent_stable /
    // register_host_callback_live_tier1_out 注释。
    //
    // persistent 一支用 GraphOwnedResource(kind=StableOutput) 描述符表达;
    // dedup / lookup 仍按 (host_base, dev_base, nbytes) 三元组语义。
    //
    // live_tier1_out_ranges_ 是 PT functional `.out` fresh args(每步重 alloc),
    // 跟 owner tree 的稳定语义不契合,仍保持 HostCallbackStableRange。
    std::vector<GraphOwnedResource> host_callback_persistent_owners_;
    std::vector<HostCallbackStableRange> host_callback_live_tier1_out_ranges_;

    // zero-copy CPU view storage holder owner tree。run_cpu_fallback_
    // zerocopy 通过 ZeroCopyHolderGuard 管理 RAII 生命周期;guard 退栈时截回
    // 原 size,所以 RECORDING/REPLAYING/PASSTHROUGH 任何时刻进入 fallback 之前,
    // 这个 vector 都是上次 fallback 退出后的 size(典型 0),
    // 不影响 admission 路径。
    std::vector<GraphOwnedResource> host_callback_zero_copy_holders_;

    // Tier3OneshotNode transient buffer owner tree。
    // tier3_oneshot_transient_owners_ 是 owner tree 描述符
    // (kind=TransientTier3Buffer);tier3_oneshot_transient_tensors_ 同索引持
    // RPU tensor 引用(防 GC,buffer 跨 replay 复用必须保活)。两个 vector 长度
    // 严格等长,size_t 索引 = Tier3OneshotNodeData::transient_resource_ids 元素。
    std::vector<GraphOwnedResource> tier3_oneshot_transient_owners_;
    std::vector<at::Tensor>          tier3_oneshot_transient_tensors_;

    // RECORDING 期 dev_addr → 最近 writer node id 映射,用于
    // Tier3OneshotNode 的 input/output dep 推断。begin (PASSTHROUGH→RECORDING)
    // / invalidate / abort 清空。REPLAYING 期不动 — replay 不需要再推断 dep,
    // 已经录在 nodes_ 各节点的 *_dep_node_ids 字段里。
    //
    // 哪些节点 register 自己的 output:
    //   - HostCallback Tier1/2 output_pool stable buffer (adopt_returns)
    //   - Tier3Oneshot last_output_dev_addrs (Tier3OneshotCapture::adopt_returns)
    //   - Memcpy.dst_dev_addr (capture_data<MemcpyNodeData> RECORDING 路径)
    //
    // 哪些节点 query 自己的 input(消费者侧推 dep):
    //   - HostCallbackCapture ctor (input live_tensor_args dev_addr)
    //   - Tier3OneshotCapture ctor (同上,自填 input_dep_node_ids)
    //   - Memcpy.src_dev_addr (capture_data<MemcpyNodeData> RECORDING 路径)
    //
    // 注:Kernel 节点的 input/output dev_addrs 通过 op 层 set_regs 写,
    // RECORDING 期 graph 层看不到;Kernel 消费者依赖 transient buffer 跨 replay
    // 稳定 dev_addr 间接保证正确性。
    std::unordered_map<uint64_t, int> recording_dev_addr_writer_;

    // RECORDING 期某节点写 dev_addr,后续节点查上游用。
    //   - state_ != RECORDING / dev_addr == 0 → noop
    //   - 同 dev_addr 后写覆盖前者(典型:caching allocator 复用 ptr 给新节点)
    void track_recording_writer_dev_addr(uint64_t dev_addr, int node_id);
    // RECORDING 期消费者节点扫 input dev_addrs:
    //   1. 收集去重的 input dep node id(只保留 RECORDING 期已 register writer 的)
    //   2. 给上游是 Tier3Oneshot 的节点反向 push consumer_node_id 到它们的
    //      output_dep_node_ids(去重)
    // 返回 sorted 的 input_dep_node_ids,Tier3OneshotCapture ctor 直接放进自己的
    // input_dep_node_ids;HostCallbackCapture ctor 不存返回值,只触发反查回填。
    std::vector<int> resolve_recording_input_deps(
        int consumer_node_id, const std::vector<uint64_t>& input_dev_addrs);

    // HostCallbackCapture 需要访问 nodes_[idx].as_host_callback() 来
    // 在 RECORDING 期回填 output_dev_addrs / tier2 buffer / etc.;不让所有
    // node 内部状态对外暴露,只通过 friend 给 HostCallbackCapture 后门。
    friend class HostCallbackCapture;
    friend class Tier3OneshotCapture;
};

class GraphKernelRegisterOuterFastTicket final {
public:
    GraphKernelRegisterOuterFastTicket(
        const GraphKernelRegisterOuterFastTicket&) = delete;
    GraphKernelRegisterOuterFastTicket& operator=(
        const GraphKernelRegisterOuterFastTicket&) = delete;
    GraphKernelRegisterOuterFastTicket(
        GraphKernelRegisterOuterFastTicket&& other) noexcept;
    GraphKernelRegisterOuterFastTicket& operator=(
        GraphKernelRegisterOuterFastTicket&&) = delete;
    ~GraphKernelRegisterOuterFastTicket();

private:
    GraphKernelRegisterOuterFastTicket(
        RpuKernelGraph* graph, uint64_t nonce)
        : graph_(graph), nonce_(nonce) {}
    void release() noexcept {
        graph_ = nullptr;
        nonce_ = 0;
    }

    RpuKernelGraph* graph_ = nullptr;
    uint64_t nonce_ = 0;

    friend class RpuKernelGraph;
    friend class GraphKernelRegisterCensusGuard;
};

class GraphKernelRegisterCensusGuard final {
public:
    GraphKernelRegisterCensusGuard(
        RpuKernelGraph& graph,
        const GraphKernelRegisterPolicy& policy,
        uint64_t physical_plan_hash,
        uint64_t live_physical_epoch)
        : graph_(&graph) {
        graph_->begin_kernel_register_census(
            policy, physical_plan_hash, live_physical_epoch);
    }

    ~GraphKernelRegisterCensusGuard() {
        if (graph_ != nullptr && !completed_) {
            graph_->cancel_kernel_register_census();
        }
    }

    GraphKernelRegisterCensusGuard(
        const GraphKernelRegisterCensusGuard&) = delete;
    GraphKernelRegisterCensusGuard& operator=(
        const GraphKernelRegisterCensusGuard&) = delete;
    GraphKernelRegisterCensusGuard(GraphKernelRegisterCensusGuard&&) = delete;
    GraphKernelRegisterCensusGuard& operator=(
        GraphKernelRegisterCensusGuard&&) = delete;

    const GraphKernelRegisterCensusStats& stats() const {
        return graph_->kernel_register_census_stats();
    }

    void stage_outer_fast_physical_digest(uint64_t digest);

    void stage_outer_fast_component(
        std::vector<std::pair<std::shared_ptr<const uint64_t>, uint64_t>>
            owner_stamps,
        std::vector<at::Tensor> key_cache,
        std::vector<at::Tensor> value_cache);

    // Invocation-only authority (for example a freshly prepared mask).  It is
    // strongly held and revalidated through launch, but is deliberately not
    // frozen into BUILD topology or required to keep one generation forever.
    void stage_outer_fast_invocation_authority(
        std::shared_ptr<const uint64_t> prepared_owner,
        std::shared_ptr<uint64_t> consumed_owner,
        uint64_t expected_generation);

    // Normal (non-force) per-invocation visibility for a BUILD-retained fixed
    // Tensor whose contents, but not identity/storage, were refreshed.
    void stage_outer_fast_invocation_visibility(const at::Tensor& tensor);

    void stage_fixed_dma_burst(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count);

    void stage_mutable_dma_burst(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count);

    void bind_fast_fixed_dma(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count);

    void bind_fast_mutable_dma(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count);

    GraphKernelRegisterOuterFastTicket validate_outer_fast_replay();

    void commit_outer_fast_replay(
        GraphKernelRegisterOuterFastTicket&& ticket);

    void checkpoint(const std::string& phase_name) {
        TORCH_CHECK(!completed_,
                    "typed register census checkpoint after completion");
        graph_->checkpoint_kernel_register_census(phase_name);
    }

    void complete() {
        TORCH_CHECK(!completed_,
                    "typed register census guard completed twice");
        graph_->complete_kernel_register_census();
        completed_ = true;
    }

private:
    RpuKernelGraph* graph_ = nullptr;
    bool completed_ = false;
};

inline void RpuKernelGraphContractTestAccess::stage_pending_kernel(
        RpuKernelGraph& graph, KernelId id) {
    graph.pending_kernel_idx_ = graph.kernels_.size();
    graph.pending_kernel_id_ = id;
    graph.pending_kernel_name_.reset();
}

inline void RpuKernelGraphContractTestAccess::stage_pending_kernel_name(
        RpuKernelGraph& graph, std::string name) {
    graph.pending_kernel_idx_ = graph.kernels_.size();
    graph.pending_kernel_id_.reset();
    graph.pending_kernel_name_ = std::move(name);
}

inline void RpuKernelGraphContractTestAccess::clear_pending_kernel(
        RpuKernelGraph& graph) {
    graph.pending_kernel_id_.reset();
    graph.pending_kernel_name_.reset();
}

inline void RpuKernelGraphContractTestAccess::
set_dma_semantics_for_topology_golden(
        RpuKernelGraph& graph, size_t node_idx, uint64_t endpoint_id,
        bool canonical_member, uint64_t spm_peer_id) {
    TORCH_CHECK(graph.state_ == RpuKernelGraph::State::RECORDING &&
                    node_idx < graph.nodes_.size() &&
                    graph.nodes_[node_idx].kind == GraphNodeKind::Dma,
                "topology reference requires one recorded DMA node");
    auto& dma = graph.nodes_[node_idx].as_dma();
    dma.semantic_endpoint_id = endpoint_id;
    dma.semantic_canonical_member_emission = canonical_member;
    dma.semantic_spm_peer_id = spm_peer_id;
}

// Stable, pointer-free identity fields accompany the owner pointer.  The owner
// pointer is used only for same-process provenance; it is deliberately excluded
// from every topology/signature hash.
struct GraphOpStreamStamp {
    const RpuKernelGraph* graph = nullptr;
    RpuKernelGraph::State state = RpuKernelGraph::State::PASSTHROUGH;
    uint64_t build_generation = 0;
    uint64_t signature_identity = 0;
    uint64_t signature_segment_key = 0;
    size_t position = 0;
    size_t extent = 0;
};

// =============================================================================
// ZeroCopyHolderGuard
// =============================================================================
//
// run_cpu_fallback_zerocopy 入口创建一个 guard;guard ctor 记录当前 owner
// tree 上 zero_copy_holders 的 size,dtor 把 vector 截断回原 size。fallback
// 内每次 cpu_view_lifeline.push_back(view) 同步调 register_zero_copy_holder
// 镜像到 owner tree,函数退出时由 guard 自动清理。嵌套 fallback 也安全。
class ZeroCopyHolderGuard {
public:
    explicit ZeroCopyHolderGuard(RpuKernelGraph& g)
        : g_(&g), saved_size_(g.zero_copy_holder_count()) {}
    ~ZeroCopyHolderGuard() {
        if (g_) g_->truncate_zero_copy_holders(saved_size_);
    }
    ZeroCopyHolderGuard(const ZeroCopyHolderGuard&) = delete;
    ZeroCopyHolderGuard& operator=(const ZeroCopyHolderGuard&) = delete;

private:
    RpuKernelGraph* g_;
    size_t saved_size_;
};

// =============================================================================
// HostCallbackCapture — RECORDING 期的 RAII guard
// =============================================================================
//
// 用法(由 custom_cpu_fallback 在 RECORDING + Tier1/2/3 路径调用):
//
//   {
//       HostCallbackCapture guard(g, op, *stack, tier);   // 录 input 骨架 + push 节点
//       // (此时 stack 仍是 RPU tensors,准备给 run_cpu_fallback_zerocopy)
//       run_cpu_fallback_zerocopy(op, stack, /*alias_output_args=*/false);
//
//       if (tier == Tier1 || tier == Tier2) guard.adopt_returns(*stack);
//       if (tier == Tier3) guard.adopt_dynamic_return(*stack);
//   }   // guard 析构:登记 output boundary flush
//
// 不在三档构造时(Reject)就不该走 HostCallback 路径,调用方需先用
// classify_host_op 拦在外面。
class HostCallbackCapture {
public:
    HostCallbackCapture(RpuKernelGraph& g,
                        const c10::OperatorHandle& op,
                        torch::jit::Stack& stack,
                        HostCallbackTier tier);
    ~HostCallbackCapture();

    HostCallbackCapture(const HostCallbackCapture&) = delete;
    HostCallbackCapture& operator=(const HostCallbackCapture&) = delete;

    HostCallbackTier tier() const { return tier_; }
    size_t node_idx() const { return node_idx_; }

    // Tier 1+2 共用:CPU op 跑完后调,把 stack returns 的 tensor 数据 copy 到
    // graph-owned 稳定 RPU buffer + 替换 stack slot。output_pool /
    // output_pool_dev_addrs / output_pool_bytes 在这里填。
    void adopt_returns(torch::jit::Stack& stack);

    // Tier 3:CPU op 跑完后调,只把返回 tensor keep-alive(防止本轮 forward 内
    // 被 GC,下游 op 可能立即用)。不保证跨 replay 稳定 —— scope 已 mark
    // non_replayable,下次 forward 重录。
    void adopt_dynamic_return(torch::jit::Stack& stack);

private:
    HostCallbackNodeData& node_data();   // 通过 friend 访问 g_.nodes_[node_idx_]

    RpuKernelGraph* g_;
    HostCallbackTier tier_;
    size_t node_idx_;             // g_.nodes_ 内的下标(析构 / adopt 用)
    size_t num_args_;             // schema.arguments().size()
    size_t stack_size_before_returns_;  // run_cpu_fallback_zerocopy 之前的
                                        // stack.size() - num_args
};

// =============================================================================
// Tier3OneshotCapture — RECORDING 期 RAII guard
// =============================================================================
//
// 跟 HostCallbackCapture 同构,但录的是 GraphNodeKind::Tier3Oneshot。Tier3 op
// (RNG / data-dependent shape)走这条节点级 oneshot 路径。
//
// 用法(由 custom_cpu_fallback 在 RECORDING + Tier3 路径调用):
//
//   {
//       Tier3OneshotCapture guard(g, op, *stack);          // 录 input 骨架 + push 节点
//       run_cpu_fallback_zerocopy(op, stack, /*alias=*/false);
//       guard.adopt_returns(*stack);                       // fresh return → keep-alive
//   }
//
// execute_graph_* 经过该节点时由 execute_data_node 的 Tier3Oneshot case 真跑
// CPU op。
class Tier3OneshotCapture {
public:
    Tier3OneshotCapture(RpuKernelGraph& g,
                        const c10::OperatorHandle& op,
                        torch::jit::Stack& stack);
    ~Tier3OneshotCapture();

    Tier3OneshotCapture(const Tier3OneshotCapture&) = delete;
    Tier3OneshotCapture& operator=(const Tier3OneshotCapture&) = delete;

    size_t node_idx() const { return node_idx_; }

    // CPU op 跑完后调:fresh return tensor 入 output_keepalives 防 GC + 录
    // last_output_dev_addrs(诊断 dump 用,不承诺跨 replay 稳定)。stack 不动 —
    // Tier3 returns 直接由下游消费(executor 二次跑时也是 fresh return,跟
    // RECORDING 期一致)。
    void adopt_returns(torch::jit::Stack& stack);

private:
    Tier3OneshotNodeData& node_data();

    RpuKernelGraph* g_;
    size_t node_idx_;
    size_t num_args_;
    size_t stack_size_before_returns_;
};

// =============================================================================
// RpuQueue — 对 rhino_lkn::Queue_t 的 graph-aware 包装
// =============================================================================
// Op 端通过 GET_QUEUE(core_num) 获取 RpuQueue*，然后用原始形态调用：
//   auto* wq = GET_QUEUE(8);
//   wq->set_broadcast_mode(true);
//   wq->set_flush_icache(false);
//   wq->enqueu_kernel(*kernel, grid_dims, core_ids);
//
// 内部会把 queue 状态（broadcast_mode/flush_icache）+ kernel 提交一起
// 交给 RpuKernelGraph::active().enqueue(..., queue_state)。
// Op 端不需要知道当前是 PASSTHROUGH 还是 RECORDING/REPLAYING。

class RpuQueue {
public:
    explicit RpuQueue(uint8_t core_num) : core_num_(core_num) {}

    // Queue 状态配置 —— 和原始 rhino Queue_t 的 set_* 同名同签名
    void set_broadcast_mode(bool b) { state_.broadcast_mode = b; }
    void set_flush_icache(bool f) { state_.flush_icache = f; }

    // Kernel 提交 —— 内部转入 RpuKernelGraph::active().enqueue(...)
    void enqueu_kernel(::rhino_lkn::Kernel_t& kernel,
                       const std::vector<uint16_t>& grid_dims,
                       const std::vector<uint8_t>& core_ids) {
        RpuKernelGraph::active().enqueue(kernel, grid_dims, core_ids, state_);
    }

    uint8_t core_num() const { return core_num_; }
    const QueueLaunchState& state() const { return state_; }

private:
    uint8_t core_num_;
    QueueLaunchState state_;  // 持久状态（和原 Queue_t 行为一致）
};

// =============================================================================
// RpuQueueCache — 按 core_num 缓存 RpuQueue 对象
// =============================================================================

class RpuQueueCache {
public:
    static RpuQueueCache& instance() {
        static RpuQueueCache inst;
        return inst;
    }

    // core_num 有效范围 1..8，使用 core_num-1 作为索引
    RpuQueue* get(uint8_t core_num) {
        return queues_[core_num - 1].get();
    }

private:
    RpuQueueCache() {
        for (uint8_t i = 0; i < 8; ++i) {
            queues_[i] = std::make_unique<RpuQueue>(i + 1);
        }
    }

    std::array<std::unique_ptr<RpuQueue>, 8> queues_;
};

// =============================================================================
// Op 端公开宏 —— 透明化接口
// =============================================================================
//
// Op 端只需要这两个宏。graph 系统在底层自动处理 PASSTHROUGH / RECORDING / REPLAYING，
// Op 代码不需要知道 graph 存在。
//
// 典型用法：
//   Kernel_t* kernel = GET_KERNEL(KernelId::XXX);  // 已 reset_regs
//   kernel->set_regs(0, ...);
//   kernel->set_regs(1, ...);
//   auto* wq = GET_QUEUE(8);
//   wq->set_broadcast_mode(true);
//   wq->enqueu_kernel(*kernel, grid_dims, core_ids);

// 覆盖 rpu_ops.h 里的原始 GET_KERNEL / GET_QUEUE 宏，使其 graph-aware
#ifdef GET_KERNEL
#undef GET_KERNEL
#endif
#define GET_KERNEL(id) (RpuKernelGraph::active().get_kernel_reset(id))

#ifdef GET_QUEUE
#undef GET_QUEUE
#endif
#define GET_QUEUE(core_num) (RpuQueueCache::instance().get(core_num))

// =============================================================================
// Graph-aware SPM 分配 helper —— Op 端零 graph 侵入
// =============================================================================
//
// 把 "三态 acquire + 对称 release guard" 这条重复模板封装成两个 free function,
// Op 侧不再直接接触 RpuKernelGraph::active() / state() / local_spm_pool()。
//
// 语义(对 Local/Global/HostDdr 对称,这里只实现 Local,其它类型等真正被迁入
// graph 的 op 需要时再补):
//   PASSTHROUGH : acquire → new;  release → delete   (等价旧 new/delete)
//   RECORDING   : acquire → pool.acquire_recording;  release → noop
//                 (所有权转给 graph.local_spm_pool_,graph.end()/invalidate() 释放)
//   REPLAYING   : acquire → pool.acquire_replaying;  release → noop
//                 (按 RECORDING 期同序拿回同一只 LocalSPM_t)
//
// 典型用法:
//   LocalSPM_t* spm = graph_acquire_local_spm(bytes, read_write, kStride32B, 2, core_id);
//   // ... use spm->get_cpu_ptr() / get_rpu_addr() ...
//   graph_release_local_spm(spm);

inline ::rhino_lkn::LocalSPM_t* graph_acquire_local_spm(
    size_t bytes, ::rhino_lkn::MemFlag flag, ::rhino_lkn::MemStride stride,
    uint64_t alignment, int core_id) {
    auto& g = RpuKernelGraph::active();
    switch (g.state()) {
    case RpuKernelGraph::State::RECORDING:
        return g.local_spm_pool()
            .acquire_recording(bytes, flag, stride, alignment, core_id)
            .buffer.get();
    case RpuKernelGraph::State::REPLAYING:
        return g.local_spm_pool().acquire_replaying().buffer.get();
    default:  // PASSTHROUGH
        return new ::rhino_lkn::LocalSPM_t(bytes, flag, stride, alignment, core_id);
    }
}

inline void graph_release_local_spm(::rhino_lkn::LocalSPM_t* buf) {
    // RECORDING/REPLAYING 所有权归 pool,graph 生命周期管理;只有 PASSTHROUGH
    // 是函数 scope 生命周期,由调用者 delete。
    if (RpuKernelGraph::active().state() == RpuKernelGraph::State::PASSTHROUGH) {
        delete buf;
    }
}

// 对称的 GlobalSPM 版本(rpu_sdpa 等用 GlobalSPM_t 作 SDPA 临时 buffer 的 op 走这条)。
// 语义跟 graph_acquire_local_spm 完全对称,只是底层池换成 GraphGlobalSpmPool。
inline ::rhino_lkn::GlobalSPM_t* graph_acquire_global_spm(
    size_t bytes, ::rhino_lkn::MemFlag flag, ::rhino_lkn::MemStride stride,
    uint64_t alignment, int core_id) {
    auto& g = RpuKernelGraph::active();
    switch (g.state()) {
    case RpuKernelGraph::State::RECORDING:
        return g.global_spm_pool()
            .acquire_recording(bytes, flag, stride, alignment, core_id)
            .buffer.get();
    case RpuKernelGraph::State::REPLAYING:
        return g.global_spm_pool().acquire_replaying().buffer.get();
    default:  // PASSTHROUGH
        return new ::rhino_lkn::GlobalSPM_t(bytes, flag, stride, alignment, core_id);
    }
}

inline void graph_release_global_spm(::rhino_lkn::GlobalSPM_t* buf) {
    if (RpuKernelGraph::active().state() == RpuKernelGraph::State::PASSTHROUGH) {
        delete buf;
    }
}

// =============================================================================
// graph_dma — op 端使用的 BATCH_CTX 顶替接口
// =============================================================================
//
// 直接走 RpuKernelGraph::active().record_*。Op 代码不再 include rpu_batch_context.h
// 或调用 BATCH_CTX。语义跟原 BATCH_CTX.add_{dma,mutable_dma,mutable_dst_dma,barrier}
// 等价,在 RECORDING/REPLAYING 期把 DMA / Barrier 顶替项穿插进同一 Queue_t 批次。
//
// graph_active_for_dma() 是各 op 入口处的便捷检查:回答 "现在能不能调 record_dma_*",
// 顶替 v5 的 BATCH_MODE_ACTIVE。RECORDING 和 REPLAYING 都返回 true;PASSTHROUGH
// 与 BUILT 返回 false(也意味着 op 自己得 fallback 走 immediate 路径)。
namespace graph_dma {

inline bool active() {
    auto s = RpuKernelGraph::active().state();
    return s == RpuKernelGraph::State::RECORDING ||
           s == RpuKernelGraph::State::REPLAYING;
}

inline void stage_census_fixed_burst(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count) {
    RpuKernelGraph::active().stage_census_fixed_dma_burst_if_active(
        tensor, side, byte_begin, byte_count, expected_occurrence_count);
}

inline void stage_census_mutable_burst(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count) {
    RpuKernelGraph::active().stage_census_mutable_dma_burst_if_active(
        live_base, tensor, side, byte_begin, byte_count,
        expected_occurrence_count);
}

inline void emit_fixed(uint64_t src, uint64_t dst, size_t bytes, uint8_t channel) {
    RpuKernelGraph::active().record_dma_fixed(src, dst, bytes, channel);
}

inline void emit_mutable_src(const uint64_t* live_src_base,
                             int64_t          src_offset,
                             uint64_t         dst,
                             size_t           bytes,
                             uint8_t          channel) {
    RpuKernelGraph::active().record_dma_mutable_src(
        live_src_base, src_offset, dst, bytes, channel);
}

inline void emit_mutable_dst(uint64_t         src,
                             const uint64_t* live_dst_base,
                             int64_t          dst_offset,
                             size_t           bytes,
                             uint8_t          channel) {
    RpuKernelGraph::active().record_dma_mutable_dst(
        src, live_dst_base, dst_offset, bytes, channel);
}

inline void emit_fixed(const GraphDmaSemanticEndpoint& endpoint,
                       uint64_t src, uint64_t dst, size_t bytes,
                       uint8_t channel) {
    RpuKernelGraph::active().record_dma_fixed(
        endpoint, src, dst, bytes, channel);
}

inline void emit_mutable_src(const GraphDmaSemanticEndpoint& endpoint,
                             const uint64_t* live_src_base,
                             int64_t src_offset, uint64_t dst,
                             size_t bytes, uint8_t channel) {
    RpuKernelGraph::active().record_dma_mutable_src(
        endpoint, live_src_base, src_offset, dst, bytes, channel);
}

inline void emit_mutable_dst(const GraphDmaSemanticEndpoint& endpoint,
                             uint64_t src,
                             const uint64_t* live_dst_base,
                             int64_t dst_offset, size_t bytes,
                             uint8_t channel) {
    RpuKernelGraph::active().record_dma_mutable_dst(
        endpoint, src, live_dst_base, dst_offset, bytes, channel);
}

inline void emit_barrier(uint8_t self_stream, uint8_t target_stream) {
    RpuKernelGraph::active().record_barrier(self_stream, target_stream);
}

}  // namespace graph_dma

// Process-scoped ownership for Graph execution, physical SPM arenas, and
// destructive runtime cleanup.  The mutex protects only these bookkeeping
// transitions; no claim holds it across a Graph/lease/cleanup scope.
#pragma once

#include <cstdint>
#include <functional>
#include <thread>

class RpuExecutionCoordinator final {
public:
    enum class ClaimKind : uint8_t {
        None = 0,
        Graph = 1,
        Physical = 2,
        Cleanup = 3,
        Reconfigure = 4,
    };

    class Claim {
    public:
        Claim() = default;
        Claim(const Claim&) = delete;
        Claim& operator=(const Claim&) = delete;
        Claim(Claim&& other) noexcept
            : kind_(other.kind_),
              token_(other.token_),
              owner_thread_(other.owner_thread_) {
            other.clear();
        }
        Claim& operator=(Claim&&) = delete;
        bool valid() const { return kind_ != ClaimKind::None && token_ != 0; }
        uint64_t token() const { return token_; }
        std::thread::id owner_thread() const { return owner_thread_; }

    private:
        Claim(ClaimKind kind, uint64_t token, std::thread::id owner_thread)
            : kind_(kind), token_(token), owner_thread_(owner_thread) {}

        void clear() noexcept {
            kind_ = ClaimKind::None;
            token_ = 0;
            owner_thread_ = std::thread::id{};
        }

        ClaimKind kind_ = ClaimKind::None;
        uint64_t token_ = 0;
        std::thread::id owner_thread_;

        friend class RpuExecutionCoordinator;
    };

    static Claim enter_graph(const char* operation);
    static void validate_graph(const Claim& claim, const char* operation);
    static void exit_graph(Claim& claim, const char* operation);
    static void exit_graph_noexcept(Claim& claim) noexcept;

    static Claim enter_physical(const char* operation);
    static void validate_physical(const Claim& claim, const char* operation);
    // Returns zero when no physical claim exists.  A live claim must belong
    // to the current thread; this is used to stamp Graph BUILD ownership
    // before any retained-arena authority can be minted.
    static uint64_t current_physical_token_if_owned(const char* operation);
    // ChildGraph extraction/direct replay is intentionally unsupported while
    // a physical arena is live: those paths bypass ordinary Graph begin() and
    // can otherwise detach baked SPM addresses from the retained root.
    static void require_no_physical_claim(const char* operation);
    static void require_graph_quiescent(const char* operation);
    static void exit_physical(Claim& claim, const char* operation);
    static bool physical_exit_allowed_noexcept(const Claim& claim) noexcept;
    static void exit_physical_noexcept(Claim& claim) noexcept;

    static Claim enter_cleanup(
        const char* operation,
        bool shutting_down,
        const char* owned_graph_diagnostic = nullptr);
    static void exit_cleanup(Claim& claim, const char* operation);
    static void exit_cleanup_noexcept(Claim& claim) noexcept;

    // One process-wide, same-thread transaction for live execution controls.
    // Setters register staged participants; commit applies every participant
    // before releasing the claim, while abort discards an uncommitted stage.
    using ReconfigureCallback = std::function<void()>;
    static uint64_t begin_reconfigure(const char* operation);
    static uint64_t begin_reconfigure(
        uint64_t attempt_token, const char* operation);
    static uint64_t begin_reconfigure_with_quiesce(
        ReconfigureCallback quiesce, const char* operation);
    static uint64_t begin_reconfigure_with_quiesce(
        ReconfigureCallback quiesce,
        uint64_t attempt_token,
        const char* operation);
    static void stage_reconfigure(
        uint64_t token,
        const void* participant,
        ReconfigureCallback apply,
        ReconfigureCallback rollback,
        const char* operation);
    static void check_reconfigure_participant_destroy_allowed(
        const void* participant, const char* operation);
    static void commit_reconfigure(uint64_t token, const char* operation);
    static void abort_reconfigure(uint64_t token, const char* operation);
    static bool abort_reconfigure_attempt(
        uint64_t attempt_token, const char* operation);
    static void validate_reconfigure(uint64_t token, const char* operation);

    // Allocator mutations may retain the current thread's already-open Graph
    // claim (reset_temporary during RECORDING), or take an exclusive cleanup
    // claim while the process is otherwise idle.  The decision and admission
    // are one locked transition so a foreign Graph cannot enter between an
    // ownership check and the allocator side effect.
    static Claim enter_allocator_mutation(
        const char* operation, bool& graph_claim_active);

    // Retained physical Graph teardown has one deliberately narrow exception
    // to the ordinary cleanup-vs-physical exclusion: the physical owner may
    // invalidate the bound Graph while no Graph scope is active.  The physical
    // claim remains held throughout, so this does not admit another executor
    // or an allocator mutation between Graph invalidation and guard retirement.
    static Claim enter_graph_invalidation(
        const char* operation,
        uint64_t expected_physical_token,
        bool& physical_claim_active);

    // Execution paths that do not themselves take a claim (PASSTHROUGH kernel
    // lookup/enqueue and direct DMA) use this before touching shared runtime or
    // SDK state.  It intentionally does not serialize concurrent graphless
    // PASSTHROUGH execution.
    static void check_current_thread_execution_allowed(const char* operation);
    static bool current_thread_execution_allowed_noexcept() noexcept;

    static bool has_graph_scope();
};

class RpuExecutionCleanupGuard final {
public:
    explicit RpuExecutionCleanupGuard(
        const char* operation,
        bool shutting_down = false,
        const char* owned_graph_diagnostic = nullptr);
    ~RpuExecutionCleanupGuard();

    RpuExecutionCleanupGuard(const RpuExecutionCleanupGuard&) = delete;
    RpuExecutionCleanupGuard& operator=(const RpuExecutionCleanupGuard&) = delete;
    RpuExecutionCleanupGuard(RpuExecutionCleanupGuard&&) = delete;
    RpuExecutionCleanupGuard& operator=(RpuExecutionCleanupGuard&&) = delete;

private:
    RpuExecutionCoordinator::Claim claim_;
};

class RpuExecutionAllocatorMutationGuard final {
public:
    explicit RpuExecutionAllocatorMutationGuard(const char* operation);
    ~RpuExecutionAllocatorMutationGuard();

    RpuExecutionAllocatorMutationGuard(
        const RpuExecutionAllocatorMutationGuard&) = delete;
    RpuExecutionAllocatorMutationGuard& operator=(
        const RpuExecutionAllocatorMutationGuard&) = delete;
    RpuExecutionAllocatorMutationGuard(
        RpuExecutionAllocatorMutationGuard&&) = delete;
    RpuExecutionAllocatorMutationGuard& operator=(
        RpuExecutionAllocatorMutationGuard&&) = delete;

    bool graph_claim_active() const noexcept { return graph_claim_active_; }

private:
    bool graph_claim_active_ = false;
    RpuExecutionCoordinator::Claim cleanup_claim_;
};

class RpuExecutionGraphInvalidationGuard final {
public:
    RpuExecutionGraphInvalidationGuard(
        const char* operation, uint64_t expected_physical_token);
    ~RpuExecutionGraphInvalidationGuard();

    RpuExecutionGraphInvalidationGuard(
        const RpuExecutionGraphInvalidationGuard&) = delete;
    RpuExecutionGraphInvalidationGuard& operator=(
        const RpuExecutionGraphInvalidationGuard&) = delete;
    RpuExecutionGraphInvalidationGuard(
        RpuExecutionGraphInvalidationGuard&&) = delete;
    RpuExecutionGraphInvalidationGuard& operator=(
        RpuExecutionGraphInvalidationGuard&&) = delete;

    bool physical_claim_active() const noexcept {
        return physical_claim_active_;
    }

private:
    bool physical_claim_active_ = false;
    RpuExecutionCoordinator::Claim cleanup_claim_;
};

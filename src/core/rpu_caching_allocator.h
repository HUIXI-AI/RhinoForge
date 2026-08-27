// rpu_caching_allocator.h
// RPU Caching Allocator - Inspired by CUDA CachingAllocator
//
// This allocator implements a caching strategy similar to CUDA's CachingAllocator:
// - Allocations are cached and reused when possible
// - Small (<= 1MB) and large (> 1MB) allocations use separate pools
// - Blocks can be split to satisfy smaller requests
// - Adjacent free blocks are merged to reduce fragmentation
//
// Key differences from CUDA version:
// - No stream support (RPU is single-threaded)
// - Simpler design without CUDA graph support
// - Adapted for RPU memory constraints (32-byte alignment for DMA)

#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>

#include <array>
#include <atomic>
#include <mutex>
#include <set>
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>

namespace rpu {

// ============================================================================
// Constants - Similar to CUDA CachingAllocator
// ============================================================================

// Alignment requirement for CopyMemory DMA transfers
// All allocations must be aligned to this boundary
constexpr size_t kAlignment = 32;

// Minimum block size - all allocations are rounded up to at least this size
// Must be a multiple of kAlignment to ensure all split blocks are aligned
constexpr size_t kMinBlockSize = 512;

// Static assertion to ensure kMinBlockSize is properly aligned
static_assert(kMinBlockSize >= kAlignment, "kMinBlockSize must be >= kAlignment");
static_assert(kMinBlockSize % kAlignment == 0, "kMinBlockSize must be a multiple of kAlignment");

// Maximum size for "small" allocations (1 MB)
constexpr size_t kSmallSize = 1048576;

// Small allocations are packed into 2 MB buffers
constexpr size_t kSmallBuffer = 2097152;

// Large allocations between 1-10 MB may use 20 MB buffers
constexpr size_t kLargeBuffer = 20971520;

// Threshold for large buffer usage
constexpr size_t kMinLargeAlloc = 10485760;

// Round large allocations to 2 MB
constexpr size_t kRoundLarge = 2097152;

// Maximum split size - blocks larger than this won't be split (default: no limit)
constexpr size_t kMaxSplitSize = std::numeric_limits<size_t>::max();

// ============================================================================
// Size Classes for O(1) Allocation
// ============================================================================

// Size classes: 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M
// Index 0 = 512 bytes, Index 11 = 1MB
// Sizes larger than 1MB use the fallback std::set path
constexpr size_t kNumSizeClasses = 12;
constexpr size_t kMaxSizeClassSize = 1048576;  // 1MB

// Size class boundaries (each is 2x the previous)
constexpr size_t kSizeClasses[kNumSizeClasses] = {
    512,      // 0: 512 B
    1024,     // 1: 1 KB
    2048,     // 2: 2 KB
    4096,     // 3: 4 KB
    8192,     // 4: 8 KB
    16384,    // 5: 16 KB
    32768,    // 6: 32 KB
    65536,    // 7: 64 KB
    131072,   // 8: 128 KB
    262144,   // 9: 256 KB
    524288,   // 10: 512 KB
    1048576   // 11: 1 MB
};

// Convert size to size class index (-1 if too large)
inline int size_to_class(size_t size) {
    if (size <= 512) return 0;
    if (size > kMaxSizeClassSize) return -1;
    // Find the smallest class that fits: ceil(log2(size/512))
    // Use bit manipulation for speed
    size_t shifted = (size - 1) >> 9;  // (size-1) / 512
    int cls = 0;
    while (shifted > 0) {
        shifted >>= 1;
        cls++;
    }
    return cls < (int)kNumSizeClasses ? cls : -1;
}

// Get the size for a given class
inline size_t class_to_size(int cls) {
    return kSizeClasses[cls];
}

// ============================================================================
// Block - Represents a memory block
// ============================================================================

struct Block;
struct BlockPool;

// Comparator for ordering blocks by size, then by address
struct BlockComparatorSize {
    bool operator()(const Block* a, const Block* b) const;
};

// Comparator for ordering blocks by address
struct BlockComparatorAddress {
    bool operator()(const Block* a, const Block* b) const;
};

struct Block {
    void* ptr{nullptr};         // Memory address
    size_t size{0};             // Block size in bytes
    size_t requested_size{0};   // Originally requested size
    bool allocated{false};      // In-use flag
    BlockPool* pool{nullptr};   // Owning memory pool
    Block* prev{nullptr};       // Previous block if split from larger allocation
    Block* next{nullptr};       // Next block if split from larger allocation

    Block() = default;
    Block(void* p, size_t s, BlockPool* pool = nullptr)
        : ptr(p), size(s), pool(pool) {}

    // Constructor for search key
    explicit Block(size_t s) : size(s) {}

    // Check if this block was split from a larger allocation
    bool is_split() const {
        return prev != nullptr || next != nullptr;
    }

    // Insert this block between 'before' and 'after'
    void splice(Block* before, Block* after) {
        if (before) {
            before->next = this;
        }
        prev = before;
        if (after) {
            after->prev = this;
        }
        next = after;
    }
};

// ============================================================================
// BlockPool - Pool of cached free blocks
// ============================================================================

struct BlockPool {
    std::set<Block*, BlockComparatorSize> blocks;
    const bool is_small;

    explicit BlockPool(bool small) : is_small(small) {}

    // Insert a block into the pool
    std::pair<std::set<Block*, BlockComparatorSize>::iterator, bool>
    insert(Block* block) {
        return blocks.insert(block);
    }

    // Erase a block from the pool
    size_t erase(Block* block) {
        return blocks.erase(block);
    }
};

// ============================================================================
// DeviceStats - Memory statistics
// ============================================================================

struct Stat {
    int64_t current{0};
    int64_t peak{0};
    int64_t allocated{0};
    int64_t freed{0};

    void increase(int64_t amount) {
        current += amount;
        allocated += amount;
        peak = std::max(peak, current);
    }

    void decrease(int64_t amount) {
        current -= amount;
        freed += amount;
    }

    void reset_peak() {
        peak = current;
    }

    void reset_accumulated() {
        allocated = 0;
        freed = 0;
    }
};

struct DeviceStats {
    // Allocation count
    Stat allocation;

    // Reserved bytes (from system allocator)
    Stat reserved_bytes;

    // Allocated bytes (in use by application)
    Stat allocated_bytes;

    // Active bytes (allocated + pending free)
    Stat active_bytes;

    // Number of segments allocated from system
    Stat segment;

    // Inactive split bytes (free but can't be released due to fragmentation)
    Stat inactive_split_bytes;

    // Number of system allocations
    int64_t num_device_alloc{0};

    // Number of system frees
    int64_t num_device_free{0};

    // Number of OOM events
    int64_t num_ooms{0};

    // Number of allocation retries
    int64_t num_alloc_retries{0};
};

// ============================================================================
// RPUCachingAllocator - Main allocator class
// ============================================================================

class RPUCachingAllocator {
public:
    RPUCachingAllocator();
    ~RPUCachingAllocator();

    // Non-copyable
    RPUCachingAllocator(const RPUCachingAllocator&) = delete;
    RPUCachingAllocator& operator=(const RPUCachingAllocator&) = delete;

    // Allocate memory
    void* malloc(size_t size);

    // Free memory
    void free(void* ptr);

    // Get memory statistics
    DeviceStats getStats() const;

    // Reset peak statistics
    void resetPeakStats();

    // Reset accumulated statistics
    void resetAccumulatedStats();

    // Empty the cache - release all cached blocks back to the system
    void emptyCache();

    // Mark the allocator shut down. After this, free() becomes a no-op (returns
    // silently), avoiding "Attempting to free unknown pointer" warnings when
    // Python tensor deleters can fire after rpu_shutdown has torn down the
    // underlying DDR allocator (RpuDdrShutdown). Called at shutdown entry.
    void mark_shutdown();

    // Get the largest available cached block size
    size_t getLargestAvailableBlock() const;

    // Get total allocated memory from system
    size_t getTotalAllocatedMemory() const;

    // Get total cached memory (reserved but not in use)
    size_t getTotalCachedMemory() const;

    // Get singleton instance
    static RPUCachingAllocator& get();

private:
    // Round size up to allocation granularity
    static size_t round_size(size_t size);

    // Get the allocation size for a given request size
    static size_t get_allocation_size(size_t size);

    // Get the appropriate pool for a given size
    BlockPool& get_pool(size_t size);

    // Try to get a free block from the cache
    Block* get_free_block(size_t size, BlockPool& pool);

    // Allocate a new block from the system
    Block* alloc_block(size_t size, BlockPool& pool);

    // Free a block back to the cache
    void free_block(Block* block);

    // Try to merge adjacent free blocks
    size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool);

    // Should we split this block?
    bool should_split(const Block* block, size_t size) const;

    // Release a block back to the system
    void release_block(Block* block);

    // Release all non-split blocks in a pool
    void release_blocks(BlockPool& pool);

    // Size-class fast path methods
    Block* get_free_block_fast(int size_class);
    void free_block_fast(Block* block, int size_class);

private:
    mutable std::recursive_mutex mutex_;

    // Pools for small and large allocations (fallback for large sizes)
    BlockPool small_blocks_;
    BlockPool large_blocks_;

    // Size-class free lists for O(1) allocation (sizes <= 1MB)
    // Each entry is a singly-linked list head using Block::next
    std::array<Block*, kNumSizeClasses> size_class_lists_{};
    std::array<size_t, kNumSizeClasses> size_class_counts_{};  // For debugging

    // All active (allocated) blocks
    ska::flat_hash_set<Block*> active_blocks_;

    // Map from pointer to block for fast lookup
    ska::flat_hash_map<void*, Block*> ptr_to_block_;

    // Statistics
    DeviceStats stats_;

    // Total memory allocated from system
    size_t total_allocated_memory_{0};

    // Set by mark_shutdown() (called from rpu_shutdown). After shutdown,
    // free() returns silently to avoid noisy "unknown pointer" warnings
    // when Python tensor deleters fire after the underlying DDR allocator
    // has been torn down (RpuDdrShutdown). 与 GraphCache 的 shutdown_called_
    // 同模式。
    std::atomic<bool> shutdown_called_{false};
};

// ============================================================================
// RPUCachingAllocatorAdapter - Adapter for c10::Allocator interface
// ============================================================================

struct RPUCachingAllocatorAdapter final : public c10::Allocator {
    at::DataPtr allocate(size_t nbytes) override;
    void copy_data(void* dest, const void* src, std::size_t count) const override;

    static RPUCachingAllocatorAdapter& get();
};

// Get the caching allocator instance
RPUCachingAllocator& get_caching_allocator();

// Memory management functions
void empty_cache();
DeviceStats get_memory_stats();
void reset_peak_memory_stats();
void reset_accumulated_memory_stats();

} // namespace rpu

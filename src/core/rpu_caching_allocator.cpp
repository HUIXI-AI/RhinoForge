// rpu_caching_allocator.cpp
// RPU Caching Allocator Implementation

#include "rpu_caching_allocator.h"
#include "rpu_ops.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace rpu {

// ============================================================================
// Block Comparators
// ============================================================================

bool BlockComparatorSize::operator()(const Block* a, const Block* b) const {
    // First compare by size
    if (a->size != b->size) {
        return a->size < b->size;
    }
    // Then by address for deterministic ordering
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

bool BlockComparatorAddress::operator()(const Block* a, const Block* b) const {
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

// ============================================================================
// RPUCachingAllocator Implementation
// ============================================================================

RPUCachingAllocator::RPUCachingAllocator()
    : small_blocks_(/*small=*/true),
      large_blocks_(/*small=*/false) {
}

RPUCachingAllocator::~RPUCachingAllocator() {
    // Note: We intentionally don't call emptyCache() here.
    // During program shutdown, the order of static object destruction
    // is undefined, and calling RpuDdrFree at this point may cause
    // issues if the RPU runtime has already been deinitialized.
    // The OS will reclaim all memory when the process exits anyway.
}

size_t RPUCachingAllocator::round_size(size_t size) {
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    }
    // Round up to multiple of kMinBlockSize
    // Since kMinBlockSize is a multiple of kAlignment (32 bytes),
    // all returned sizes are guaranteed to be 32-byte aligned,
    // which is required by CopyMemory DMA transfers
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
}

size_t RPUCachingAllocator::get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
        // Small allocations: use small buffer size
        return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
        // Medium allocations: use large buffer size
        return kLargeBuffer;
    } else {
        // Large allocations: round up to 2 MB
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
}

BlockPool& RPUCachingAllocator::get_pool(size_t size) {
    if (size <= kSmallSize) {
        return small_blocks_;
    }
    return large_blocks_;
}

Block* RPUCachingAllocator::get_free_block(size_t size, BlockPool& pool) {
    // Create a search key block
    Block search_key(size);

    // Find the smallest block that can satisfy the request
    auto it = pool.blocks.lower_bound(&search_key);

    if (it == pool.blocks.end()) {
        return nullptr;
    }

    Block* block = *it;

    // For large requests, don't return an oversized block if max_split_size would prevent splitting
    if (size < kMaxSplitSize && block->size >= kMaxSplitSize) {
        return nullptr;
    }

    // Remove from pool
    pool.erase(block);
    if (block->is_split()) {
        stats_.inactive_split_bytes.decrease(block->size);
    }

    return block;
}

// O(1) fast path for size-class allocations
Block* RPUCachingAllocator::get_free_block_fast(int size_class) {
    Block* block = size_class_lists_[size_class];
    if (block) {
        // Pop from head of list
        size_class_lists_[size_class] = block->next;
        block->next = nullptr;
        size_class_counts_[size_class]--;
    }
    return block;
}

// O(1) fast path for returning blocks to size-class lists
void RPUCachingAllocator::free_block_fast(Block* block, int size_class) {
    // Push to head of list
    block->next = size_class_lists_[size_class];
    block->prev = nullptr;  // Not used in size-class lists
    size_class_lists_[size_class] = block;
    size_class_counts_[size_class]++;
}

Block* RPUCachingAllocator::alloc_block(size_t size, BlockPool& pool) {
    size_t alloc_size = get_allocation_size(size);

    // Allocate from system
    void* ptr = rhino_lkn::RpuDdrAlloc(alloc_size);

    if (!ptr) {
        // Allocation failed
        stats_.num_ooms++;
        return nullptr;
    }

    // Verify that RpuDdrAlloc returns 32-byte aligned memory
    // This is required for CopyMemory DMA transfers
    if (reinterpret_cast<uintptr_t>(ptr) % kAlignment != 0) {
        if (log_at(1)) {
            std::cerr << "[RPUCachingAllocator] Error: RpuDdrAlloc returned unaligned pointer "
                      << ptr << " (alignment: " << (reinterpret_cast<uintptr_t>(ptr) % kAlignment)
                      << ", required: " << kAlignment << ")" << std::endl;
        }
        rhino_lkn::RpuDdrFree(ptr);
        stats_.num_ooms++;
        return nullptr;
    }

    // Create new block
    Block* block = new Block(ptr, alloc_size, &pool);

    // Update statistics
    total_allocated_memory_ += alloc_size;
    stats_.reserved_bytes.increase(alloc_size);
    stats_.segment.increase(1);
    stats_.num_device_alloc++;

    return block;
}

bool RPUCachingAllocator::should_split(const Block* block, size_t size) const {
    size_t remaining = block->size - size;

    if (block->pool->is_small) {
        // Small pool: split if there's at least kMinBlockSize remaining
        return remaining >= kMinBlockSize;
    } else {
        // Large pool: split if request is smaller than max_split_size
        // and remaining is larger than kSmallSize
        return (size < kMaxSplitSize) && (remaining > kSmallSize);
    }
}

void* RPUCachingAllocator::malloc(size_t orig_size) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (shutdown_called_) {
        throw std::runtime_error("RPU caching allocator is shut down");
    }

    // Handle zero-size allocation
    if (orig_size == 0) {
        return nullptr;
    }

    // Round up the size
    size_t size = round_size(orig_size);

    // Try size-class fast path first (O(1) for sizes <= 1MB)
    int size_class = size_to_class(size);
    Block* block = nullptr;

    if (size_class >= 0) {
        // Use size class - round up to class size for exact match
        size = class_to_size(size_class);
        block = get_free_block_fast(size_class);
    }

    // Fallback to pool-based allocation if fast path didn't find a block
    if (!block) {
        // Get the appropriate pool
        BlockPool& pool = get_pool(size);

        // Try to get a free block from cache
        block = get_free_block(size, pool);

        if (!block) {
            // No suitable cached block, try to allocate a new one
            block = alloc_block(size, pool);

            if (!block) {
                // First allocation attempt failed, try to release cached blocks and retry
                stats_.num_alloc_retries++;

                // Release cached blocks
                release_blocks(small_blocks_);
                release_blocks(large_blocks_);

                // Retry allocation
                block = alloc_block(size, pool);

                if (!block) {
                    // Still failed, OOM
                    throw std::bad_alloc();
                }
            }
        }

        // Split the block if needed (only for pool-allocated blocks)
        if (should_split(block, size)) {
            // Create a new block for the remainder
            Block* remaining = new Block(
                static_cast<char*>(block->ptr) + size,
                block->size - size,
                &pool
            );

            // Link the blocks
            remaining->prev = block;
            remaining->next = block->next;
            if (block->next) {
                block->next->prev = remaining;
            }
            block->next = remaining;
            block->size = size;

            // Add remainder to pool
            pool.insert(remaining);

            // Update statistics for inactive split
            stats_.inactive_split_bytes.increase(remaining->size);
        }
    }

    // Mark as allocated
    block->allocated = true;
    block->requested_size = orig_size;

    // Track the block
    active_blocks_.insert(block);
    ptr_to_block_[block->ptr] = block;

    // Update statistics
    stats_.allocation.increase(1);
    stats_.allocated_bytes.increase(block->size);
    stats_.active_bytes.increase(block->size);

    // Final alignment check - this should never fail if our logic is correct
    assert(reinterpret_cast<uintptr_t>(block->ptr) % kAlignment == 0 &&
           "Allocated block is not 32-byte aligned!");

    return block->ptr;
}

void RPUCachingAllocator::free(void* ptr) {
    if (!ptr) {
        return;
    }

    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // Shutdown 后 (rpu_shutdown 调过 mark_shutdown), 底层 DDR 已经被
    // RpuDdrShutdown 全清, 这里再 free 已没有意义。直接返回, 避免 Python
    // 解释器析构 tensor 时触发的 "Attempting to free unknown pointer" 噪音。
    if (shutdown_called_) {
        return;
    }

    // Find the block
    auto it = ptr_to_block_.find(ptr);
    if (it == ptr_to_block_.end()) {
        if (log_at(2)) {
            std::cerr << "[RPUCachingAllocator] Warning: Attempting to free unknown pointer "
                      << ptr << std::endl;
        }
        return;
    }

    Block* block = it->second;

    if (!block->allocated) {
        if (log_at(2)) {
            std::cerr << "[RPUCachingAllocator] Warning: Double free detected for pointer "
                      << ptr << std::endl;
        }
        return;
    }

    // Mark as not allocated
    block->allocated = false;

    // Remove from pointer map - block is no longer active
    ptr_to_block_.erase(it);

    // Update statistics
    stats_.allocation.decrease(1);
    stats_.allocated_bytes.decrease(block->size);
    stats_.active_bytes.decrease(block->size);

    // Try size-class fast path: if block matches a size class and is not split
    int size_class = size_to_class(block->size);
    if (size_class >= 0 && block->size == class_to_size(size_class) &&
        !block->is_split()) {
        // Fast path: O(1) return to size-class free list
        active_blocks_.erase(block);
        free_block_fast(block, size_class);
        return;
    }

    // Fallback: use pool-based free with merge logic
    free_block(block);
}

void RPUCachingAllocator::free_block(Block* block) {
    BlockPool& pool = *block->pool;

    // Try to merge with adjacent blocks
    size_t original_size = block->size;
    int64_t net_change_inactive_split = 0;

    // Try to merge with previous block
    if (block->prev && !block->prev->allocated) {
        size_t merged_size = try_merge_blocks(block, block->prev, pool);
        if (merged_size > 0) {
            net_change_inactive_split -= static_cast<int64_t>(merged_size);
        }
    }

    // Try to merge with next block
    if (block->next && !block->next->allocated) {
        size_t merged_size = try_merge_blocks(block, block->next, pool);
        if (merged_size > 0) {
            net_change_inactive_split -= static_cast<int64_t>(merged_size);
        }
    }

    // Remove from active blocks
    active_blocks_.erase(block);

    // Add to pool
    pool.insert(block);

    // Update inactive split statistics
    if (block->is_split()) {
        net_change_inactive_split += static_cast<int64_t>(block->size);
    }

    if (net_change_inactive_split > 0) {
        stats_.inactive_split_bytes.increase(static_cast<size_t>(net_change_inactive_split));
    } else if (net_change_inactive_split < 0) {
        stats_.inactive_split_bytes.decrease(static_cast<size_t>(-net_change_inactive_split));
    }
}

size_t RPUCachingAllocator::try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    // Can't merge if src is null or allocated
    if (!src || src->allocated) {
        return 0;
    }

    // Remove src from pool
    size_t erased = pool.erase(src);
    if (erased == 0) {
        // src was not in the pool (shouldn't happen for free blocks)
        return 0;
    }

    size_t merged_size = src->size;

    if (dst->prev == src) {
        // [src][dst] -> merge src into dst
        dst->ptr = src->ptr;
        dst->prev = src->prev;
        if (dst->prev) {
            dst->prev->next = dst;
        }
    } else if (dst->next == src) {
        // [dst][src] -> merge src into dst
        dst->next = src->next;
        if (dst->next) {
            dst->next->prev = dst;
        }
    } else {
        // Not adjacent, shouldn't happen
        pool.insert(src);  // Put it back
        return 0;
    }

    dst->size += merged_size;

    // Note: ptr_to_block_ entries are removed when blocks are freed,
    // so we don't need to update it here during merge

    delete src;

    return merged_size;
}

void RPUCachingAllocator::release_block(Block* block) {
    // Sanity check: only release blocks that are not split
    // This ensures we're releasing the original allocated pointer
    if (block->is_split()) {
        if (log_at(1)) {
            std::cerr << "[RPUCachingAllocator] Error: Attempting to release split block "
                      << "ptr=" << block->ptr << " size=" << block->size
                      << " prev=" << block->prev << " next=" << block->next
                      << std::endl;
        }
        return;
    }

    // Release memory back to system
    rhino_lkn::RpuDdrFree(block->ptr);

    // Update statistics
    total_allocated_memory_ -= block->size;
    stats_.reserved_bytes.decrease(block->size);
    stats_.segment.decrease(1);
    stats_.num_device_free++;

    // Remove from pool
    block->pool->erase(block);

    // Note: ptr_to_block_ entries are only for active blocks,
    // freed blocks are not in the map

    delete block;
}

void RPUCachingAllocator::release_blocks(BlockPool& pool) {
    // Release all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
        Block* block = *it;
        ++it;

        // Only release blocks that are not split
        if (!block->is_split()) {
            release_block(block);
        }
    }
}

void RPUCachingAllocator::emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // Release all blocks in size-class lists
    for (size_t i = 0; i < kNumSizeClasses; i++) {
        Block* block = size_class_lists_[i];
        while (block) {
            Block* next = block->next;
            // Release memory back to system
            rhino_lkn::RpuDdrFree(block->ptr);
            total_allocated_memory_ -= block->size;
            stats_.reserved_bytes.decrease(block->size);
            stats_.segment.decrease(1);
            stats_.num_device_free++;
            delete block;
            block = next;
        }
        size_class_lists_[i] = nullptr;
        size_class_counts_[i] = 0;
    }

    // Release all cached blocks from pools
    release_blocks(small_blocks_);
    release_blocks(large_blocks_);
}

void RPUCachingAllocator::mark_shutdown() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    shutdown_called_ = true;
}

DeviceStats RPUCachingAllocator::getStats() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return stats_;
}

void RPUCachingAllocator::resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    stats_.allocation.reset_peak();
    stats_.reserved_bytes.reset_peak();
    stats_.allocated_bytes.reset_peak();
    stats_.active_bytes.reset_peak();
    stats_.segment.reset_peak();
    stats_.inactive_split_bytes.reset_peak();
}

void RPUCachingAllocator::resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    stats_.allocation.reset_accumulated();
    stats_.reserved_bytes.reset_accumulated();
    stats_.allocated_bytes.reset_accumulated();
    stats_.active_bytes.reset_accumulated();
    stats_.segment.reset_accumulated();
    stats_.inactive_split_bytes.reset_accumulated();

    stats_.num_alloc_retries = 0;
    stats_.num_ooms = 0;
}

size_t RPUCachingAllocator::getLargestAvailableBlock() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    size_t largest = 0;

    // Check size-class lists (iterate from largest to smallest)
    for (int i = kNumSizeClasses - 1; i >= 0; i--) {
        if (size_class_lists_[i] != nullptr) {
            largest = std::max(largest, kSizeClasses[i]);
            break;  // Found the largest available size class
        }
    }

    // Check small blocks
    if (!small_blocks_.blocks.empty()) {
        auto it = small_blocks_.blocks.rbegin();
        largest = std::max(largest, (*it)->size);
    }

    // Check large blocks
    if (!large_blocks_.blocks.empty()) {
        auto it = large_blocks_.blocks.rbegin();
        largest = std::max(largest, (*it)->size);
    }

    return largest;
}

size_t RPUCachingAllocator::getTotalAllocatedMemory() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return total_allocated_memory_;
}

size_t RPUCachingAllocator::getTotalCachedMemory() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    size_t cached = 0;

    // Count size-class cached memory
    for (size_t i = 0; i < kNumSizeClasses; i++) {
        cached += size_class_counts_[i] * kSizeClasses[i];
    }

    // Count pool-based cached memory
    for (const auto* block : small_blocks_.blocks) {
        cached += block->size;
    }

    for (const auto* block : large_blocks_.blocks) {
        cached += block->size;
    }

    return cached;
}

RPUCachingAllocator& RPUCachingAllocator::get() {
    static RPUCachingAllocator instance;
    return instance;
}

// ============================================================================
// RPUCachingAllocatorAdapter Implementation
// ============================================================================

at::DataPtr RPUCachingAllocatorAdapter::allocate(size_t nbytes) {
    // Handle empty tensor allocation
    if (nbytes == 0) {
        return {nullptr, nullptr, nullptr, at::Device(c10::DeviceType::PrivateUse1, 0)};
    }

    void* ptr = RPUCachingAllocator::get().malloc(nbytes);

    if (!ptr) {
        throw std::bad_alloc();
    }

    // Create custom deleter that uses our caching allocator
    auto deleter = [](void* p) {
        RPUCachingAllocator::get().free(p);
    };

    return {ptr, ptr, deleter, at::Device(c10::DeviceType::PrivateUse1, 0)};
}

void RPUCachingAllocatorAdapter::copy_data(void* dest, const void* src, std::size_t count) const {
    memcpy(dest, src, count);
}

RPUCachingAllocatorAdapter& RPUCachingAllocatorAdapter::get() {
    static RPUCachingAllocatorAdapter instance;
    return instance;
}

// ============================================================================
// Global Functions
// ============================================================================

RPUCachingAllocator& get_caching_allocator() {
    return RPUCachingAllocator::get();
}

void empty_cache() {
    RPUCachingAllocator::get().emptyCache();
}

DeviceStats get_memory_stats() {
    return RPUCachingAllocator::get().getStats();
}

void reset_peak_memory_stats() {
    RPUCachingAllocator::get().resetPeakStats();
}

void reset_accumulated_memory_stats() {
    RPUCachingAllocator::get().resetAccumulatedStats();
}

} // namespace rpu

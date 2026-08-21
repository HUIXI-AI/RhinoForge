// rpu_bounded_shape_map.h — one bounded, LRU-ordered map for the shape-keyed
// tensor caches scattered across the fused layer.
//
// The pattern it replaces: `std::map<shape, at::Tensor>` (or the equivalent
// unordered_map) that is only ever inserted into. Every one of them grows for
// the lifetime of the process, keyed on something a long-running server varies
// without bound — a decode step's kv_seq_len, an image's patch count, a
// prefill length. None of them had a cap and none of them had a clear.
//
// TWO POLICIES, and the choice is a correctness decision, not a tuning knob:
//
//   Evict  — the entry is a pure memo. It is recomputable and NOTHING outside
//            the map holds its address past the call that looked it up.
//            Dropping the least-recently-used entry costs one recompute.
//
//   Reject — the entry's DEVICE address is baked into a built graph's DMA
//            descriptors. `rpu_launch_*_dma(..., slot.data_ptr(), ...)` resolves
//            the address at BUILD and the kd_buf keeps it; a REPLAY re-issues
//            that same address without asking the map again. Evicting such an
//            entry frees the DDR block, its device VA becomes immediately
//            re-issuable, and the next REPLAY of the still-cached graph writes
//            over whoever owns it now. A device address does not retain the
//            allocation lifetime, so a map
//            like that must NEVER evict. It fails loudly at the bound instead,
//            which is a bug report, not a crash: the bound is set above the
//            distinct-shape count a real run touches, so hitting it means the
//            caller is generating unbounded shapes and needs a real lifetime.
//
// Reference stability: entries live in a std::list, so a `Value&` handed out by
// find()/touch() survives every later insert and every eviction of ANOTHER key.
// Under Evict it does not survive eviction of its OWN key — use it before the
// next lookup, which is what all present callers do.
#pragma once

#include <ATen/ATen.h>   // TORCH_CHECK

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>
#include <utility>

namespace rpu {

enum class ShapeMapPolicy { Evict, Reject };

// std::pair has no std::hash; the shape keys here are (dim, dim) pairs.
struct PairHash {
    template <class A, class B>
    size_t operator()(const std::pair<A, B>& p) const {
        size_t h = std::hash<A>()(p.first);
        h ^= std::hash<B>()(p.second) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

template <class Key, class Value, class Hash = std::hash<Key>>
class BoundedShapeMap {
public:
    // `what` is quoted verbatim in the Reject diagnostic, so name the map and
    // its owner (e.g. "Qwen3VLVisionModel::deepstack_ddr_slots_").
    BoundedShapeMap(size_t capacity, ShapeMapPolicy policy, const char* what)
        : capacity_(capacity), policy_(policy), what_(what) {}

    // Existing entry, moved to most-recently-used. nullptr on miss; never inserts.
    Value* find(const Key& k) {
        auto it = index_.find(k);
        if (it == index_.end()) return nullptr;
        entries_.splice(entries_.begin(), entries_, it->second);
        return &it->second->second;
    }

    // Existing entry, or a loud failure. Same contract as std::map::at().
    Value& at(const Key& k) {
        Value* v = find(k);
        TORCH_CHECK(v != nullptr, what_, ": no entry for this shape");
        return *v;
    }

    // Existing entry (moved to most-recently-used), or a default-constructed
    // one. Enforces the bound BEFORE inserting. `inserted`, when non-null,
    // reports whether this call created the entry.
    Value& touch(const Key& k, bool* inserted = nullptr) {
        auto it = index_.find(k);
        if (it != index_.end()) {
            entries_.splice(entries_.begin(), entries_, it->second);
            if (inserted) *inserted = false;
            return it->second->second;
        }
        if (index_.size() >= capacity_) evict_one();
        entries_.emplace_front(k, Value{});
        index_.emplace(k, entries_.begin());
        if (inserted) *inserted = true;
        return entries_.front().second;
    }

    size_t size()     const noexcept { return index_.size(); }
    size_t capacity() const noexcept { return capacity_; }

private:
    void evict_one() {
        TORCH_CHECK(policy_ == ShapeMapPolicy::Evict,
                    what_, ": exceeded its bound of ", capacity_,
                    " distinct shapes on one handle. This map cannot evict — its "
                    "entries' device addresses are baked into built graph DMA "
                    "descriptors, so freeing one would leave a cached graph "
                    "writing to a reissued VA. A real run "
                    "touches far fewer shapes than this; if you legitimately need "
                    "more, give the entries an explicit lifetime tied to the "
                    "graph cache rather than raising the bound.");
        index_.erase(entries_.back().first);
        entries_.pop_back();
    }

    std::list<std::pair<Key, Value>>                                 entries_;
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator,
                       Hash>                                         index_;
    size_t          capacity_;
    ShapeMapPolicy  policy_;
    const char*     what_;
};

}  // namespace rpu

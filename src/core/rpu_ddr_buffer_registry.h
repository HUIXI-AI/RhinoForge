// rpu_ddr_buffer_registry.h — shape-keyed at::Tensor registry for DMA-baked
// DDR buffers in FusedModelBase.
//
// Why this exists: DMA src/dst addresses are baked into kd_buf at
// Queue_t::build_batch and have no sync_mutable_params equivalent. Any DMA
// inside a cached GraphSlot must reference a pointer that is stable for the
// slot's lifetime. The registry holds one (a, b, out) tensor triple per
// distinct ShapeKey on the FusedModelBase::Impl, indexed by
// (batch, seq, hidden, dtype, device). Same shape → same data_ptr.
// Multi-key replay (A→B→A) returns the original A pointers.
#pragma once

#include <unordered_map>
#include <ATen/ATen.h>
#include <c10/core/Device.h>

#include "rpu_bounded_shape_map.h"

struct ShapeKey {
    int64_t       batch;
    int64_t       seq;
    int64_t       hidden;
    at::ScalarType dtype;
    at::DeviceType device;

    bool operator==(const ShapeKey& o) const {
        return batch == o.batch && seq == o.seq && hidden == o.hidden
            && dtype == o.dtype && device == o.device;
    }

    struct Hash {
        size_t operator()(const ShapeKey& k) const {
            // Simple bitwise mix — collisions impact only registry probe cost.
            size_t h = std::hash<int64_t>()(k.batch);
            h ^= std::hash<int64_t>()(k.seq)    + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int64_t>()(k.hidden) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(static_cast<int>(k.dtype))  + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(static_cast<int>(k.device)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
};

class DdrBufferRegistry {
public:
    struct Entry {
        at::Tensor        a;
        at::Tensor        b;
        at::Tensor        out;
        at::TensorOptions opts;  // captured at first insert
    };

    // Returns the entry for `key`, allocating missing fields lazily.
    // Asserts caller-supplied opts match the stored entry's opts on reuse.
    //
    // Layer-0 input no longer needs a registry-owned staging slot — it uses
    // a mutable DMA referencing the caller's per-forward `hidden_states`
    // directly.
    const Entry& get(const ShapeKey& key, at::TensorOptions opts,
                     bool need_ab, bool need_out);

    // Bound on distinct shapes per handle. NOT evictable: `out` is handed to
    // the subclass as output_tensor_ and `a`/`b` are the DDR ping-pong bases
    // baked into every inner layer's DMA at BUILD, so dropping an entry would
    // leave a cached graph writing to a reissued VA (C-1).
    // The limit leaves headroom for supported profile shape sets while failing
    // early enough to prevent unbounded per-handle DDR growth.
    static constexpr size_t kMaxShapes = 64;

private:
    rpu::BoundedShapeMap<ShapeKey, Entry, ShapeKey::Hash> store_{
        kMaxShapes, rpu::ShapeMapPolicy::Reject, "DdrBufferRegistry"};
};

inline const DdrBufferRegistry::Entry&
DdrBufferRegistry::get(const ShapeKey& key, at::TensorOptions opts,
                        bool need_ab, bool need_out) {
    bool inserted = false;
    Entry& e = store_.touch(key, &inserted);
    if (inserted) {
        e.opts = opts;
    } else {
        TORCH_CHECK(e.opts.dtype()  == opts.dtype()  &&
                    e.opts.device() == opts.device() &&
                    e.opts.layout() == opts.layout(),
                    "DdrBufferRegistry: TensorOptions mismatch on reuse for "
                    "(batch=", key.batch, " seq=", key.seq,
                    " hidden=", key.hidden, ")");
    }
    std::vector<int64_t> shape_vec{key.batch, key.seq, key.hidden};

    if (need_ab) {
        if (!e.a.defined()) e.a = at::empty(shape_vec, e.opts);
        if (!e.b.defined()) e.b = at::empty(shape_vec, e.opts);
    }
    if (need_out && !e.out.defined()) {
        e.out = at::empty(shape_vec, e.opts);
    }
    return e;
}

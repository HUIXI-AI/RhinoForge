// rpu_batch_signature.h — FNV-1a 64-bit hash helpers for build-signature.
//
// Hashes complete immutable kd_buf state per BatchItem so REPLAY can detect
// any divergence from the BUILD-time emission sequence.
//
// Per-item coverage:
//   COMPUTE          kind=1, KernelId, every grid_dims entry, args_count after setup
//   DMA              kind=2, src, dst, size, channel
//   BARRIER          kind=3, self, target
//   MUTABLE_DMA      kind=4, dst, size, channel  (src excluded — rewritten per replay)
//   MUTABLE_DMA_DST  kind=5, src, size, channel  (dst excluded — rewritten per replay)
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rpu_kernel_cache.h"  // KernelId

namespace rpu_sig {

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr uint64_t FNV_PRIME  = 0x100000001b3ULL;

inline uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v;
    h *= FNV_PRIME;
    return h;
}

inline uint64_t signature_word_compute(KernelId id,
                                        const std::vector<uint16_t>& grid_dims,
                                        size_t args_count_post_setup) {
    uint64_t h = mix(FNV_OFFSET, 1ULL);
    h = mix(h, static_cast<uint64_t>(id));
    h = mix(h, static_cast<uint64_t>(grid_dims.size()));
    for (uint16_t g : grid_dims) h = mix(h, static_cast<uint64_t>(g));
    h = mix(h, static_cast<uint64_t>(args_count_post_setup));
    return h;
}

inline uint64_t signature_word_dma(uint64_t src, uint64_t dst,
                                    size_t size, int channel) {
    uint64_t h = mix(FNV_OFFSET, 2ULL);
    h = mix(h, src);
    h = mix(h, dst);
    h = mix(h, static_cast<uint64_t>(size));
    h = mix(h, static_cast<uint64_t>(static_cast<uint32_t>(channel)));
    return h;
}

inline uint64_t signature_word_mutable_dma(uint64_t dst,
                                            size_t size, int channel) {
    // src is intentionally excluded — mutable DMAs rewrite src per replay via
    // Queue_t::update_dma_kernel. dst + size + channel define the
    // kd_buf packet layout, which must NOT change across replays of the same
    // GraphSlot. Adding src would force a slot rebuild every forward.
    uint64_t h = mix(FNV_OFFSET, 4ULL);
    h = mix(h, dst);
    h = mix(h, static_cast<uint64_t>(size));
    h = mix(h, static_cast<uint64_t>(static_cast<uint32_t>(channel)));
    return h;
}

inline uint64_t signature_word_mutable_dma_dst(uint64_t src,
                                                size_t size, int channel) {
    // dst is intentionally excluded — mutable-dst DMAs rewrite dst per replay
    // via Queue_t::update_dma_kernel. src + size + channel define the kd_buf
    // packet layout. Use when DDR-side dst is caller-owned per-forward (e.g.
    // an output tensor whose data_ptr is not stable across cached forwards).
    uint64_t h = mix(FNV_OFFSET, 5ULL);
    h = mix(h, src);
    h = mix(h, static_cast<uint64_t>(size));
    h = mix(h, static_cast<uint64_t>(static_cast<uint32_t>(channel)));
    return h;
}

inline uint64_t signature_word_barrier(uint8_t self, uint8_t target) {
    uint64_t h = mix(FNV_OFFSET, 3ULL);
    h = mix(h, static_cast<uint64_t>(self));
    h = mix(h, static_cast<uint64_t>(target));
    return h;
}

}  // namespace rpu_sig

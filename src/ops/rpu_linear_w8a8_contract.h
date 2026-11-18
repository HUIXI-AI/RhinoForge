// Ordinary signed INT8 X/W -> FP32 accumulation -> FP16 Y.
// Mirrors rhino-ops 631eb4e8 generator/parallel_linear_w8a8_tiling.py.
// This ABI is distinct from W8A16 and Pi0.5's paired INT32 GeGLU ABI.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace rpu_pl_w8a8 {

enum class Partition : uint8_t { Row = 0, Col = 1 };
struct Shape {
  int64_t m, local_n, local_k;
  int cores;
  Partition partition;
  bool bias;
};
struct Tile {
  int m = 0, n = 0, k = 0;
  bool resident = false, single_k = false, keep_sw = false, keep_bias = false;
  bool sx_fused = false, hoist_valu = false, n_major = false;
  int vlm_rows = 0;
  const char* source = "explicit_w8a8";
};

inline void validate_shape(const Shape& s) {
  if (s.m <= 0 || s.m > 65535 || s.local_n <= 0 ||
      s.local_n > (1 << 20) || s.local_n % 16 || s.local_k <= 0 ||
      s.local_k > 65535 || s.local_k % 32 || s.cores < 1 || s.cores > 8 ||
      (s.partition != Partition::Row && s.partition != Partition::Col))
    throw std::invalid_argument("W8A8 requires uint16 M, physical N16/K32, row/col and 1..8 cores");
}

// Global dimensions describe already padded, packed weight storage. This helper
// never invents padding: the caller owns the physical weight/input/scale buffers.
inline Shape physical_shape(int64_t m, int64_t n, int64_t k,
                            Partition partition, int cores, bool bias) {
  if (n <= 0 || k <= 0 || cores < 1 || cores > 8 ||
      (partition != Partition::Row && partition != Partition::Col) ||
      (partition == Partition::Col ? n % cores : k % cores))
    throw std::invalid_argument("W8A8 physical global shape must divide its core partition");
  Shape s{m, partition == Partition::Col ? n / cores : n,
          partition == Partition::Row ? k / cores : k, cores, partition, bias};
  validate_shape(s);
  return s;
}

inline Tile validate_tile(Tile t, int64_t local_k, bool bias) {
  if (t.m <= 0 || t.m > 65535 || t.n <= 0 || t.n > 65535 ||
      t.k <= 0 || t.k > 65535 || t.m % 16 || t.n % 16 || t.k % 32 ||
      local_k <= 0 || local_k > 65535 || local_k % 32)
    throw std::invalid_argument("W8A8 requires positive uint16 M16/N16/K32 tiles");
  if ((!t.resident && (t.single_k || t.keep_sw || t.keep_bias ||
                       t.hoist_valu || t.n_major)) ||
      (t.single_k && t.k < local_k) ||
      (t.n_major && !(t.single_k && t.hoist_valu)) ||
      (t.keep_bias && !(t.keep_sw && bias)))
    throw std::invalid_argument("Invalid W8A8 resident allocation flags");
  const int64_t mv = t.m / 16, nv = t.n / 16, kv = t.k / 32;
  int64_t rows;
  if (t.resident) {
    const int64_t x_rows = mv * kv;
    if (nv + mv > x_rows)
      throw std::invalid_argument("W8A8 resident metadata exceeds the X allocation");
    rows = nv * (local_k / 32) + x_rows + 2 * mv * nv +
           nv * (int(t.keep_sw) + int(t.keep_bias));
  } else {
    rows = std::max((mv + nv) * kv + 2 * mv * nv,
                    2 * mv * nv + mv + nv + 4);
  }
  if (rows > 400)
    throw std::invalid_argument("W8A8 tile exceeds 400 VLM rows");
  t.vlm_rows = static_cast<int>(rows);
  return t;
}

struct Candidate {
  int compiled_local_k;  // 0: general; nonzero: exact resident allocation.
  Tile tile;
  const char* name;
  std::size_t ordinal;
};
inline constexpr std::size_t kCandidateCount = 29;
inline constexpr auto installed_candidates() {
  return std::array<Candidate, kCandidateCount>{{
#define RPU_W8A8_CANDIDATE(id, name, ordinal, m, n, k, lk, flags) \
    {lk, {m, n, k, bool(flags & 1), bool(flags & 2), bool(flags & 4), \
          bool(flags & 8), bool(flags & 16), bool(flags & 32), bool(flags & 64), \
          0, "explicit_installed"}, name, ordinal},
#include "rpu_linear_w8a8_candidates.inc"
#undef RPU_W8A8_CANDIDATE
  }};
}
inline bool same_kernel(const Tile& a, const Tile& b) {
  return a.m == b.m && a.n == b.n && a.k == b.k &&
      a.resident == b.resident && a.single_k == b.single_k &&
      a.keep_sw == b.keep_sw && a.keep_bias == b.keep_bias &&
      a.sx_fused == b.sx_fused && a.hoist_valu == b.hoist_valu &&
      a.n_major == b.n_major;
}
inline Candidate installed_candidate(const Tile& tile, int64_t local_k) {
  validate_tile(tile, local_k, tile.keep_bias);
  for (auto c : installed_candidates()) {
    if (same_kernel(tile, c.tile) &&
        (!c.compiled_local_k || c.compiled_local_k == local_k))
      return c;
  }
  throw std::invalid_argument("No fixed W8A8 ASM is installed for this tile/local K");
}
inline const char* installed_name(const Tile& tile, int64_t local_k) {
  return installed_candidate(tile, local_k).name;
}

namespace detail {
struct Exact {
  Shape shape;
  std::size_t ordinal;
  const char* source;
};
inline constexpr std::array<Exact, 23> exact{{
    {{800, 256, 2048, 8, Partition::Col, false}, 19, "pi05_compact:lang_qkv_m800"},
    {{800, 2048, 256, 8, Partition::Row, false}, 22, "pi05_compact:lang_o_m800"},
    {{800, 2048, 2048, 8, Partition::Col, false}, 27, "pi05_compact:lang_gate_up_m800"},
    {{800, 2048, 2048, 8, Partition::Row, false}, 27, "pi05_compact:lang_down_m800"},
    {{400, 256, 2048, 8, Partition::Col, false}, 10, "pi05_compact:lang_qkv_m400"},
    {{400, 2048, 256, 8, Partition::Row, false}, 23, "pi05_compact:lang_o_m400"},
    {{400, 2048, 2048, 8, Partition::Col, false}, 27, "pi05_compact:lang_gate_up_m400"},
    {{400, 2048, 2048, 8, Partition::Row, false}, 27, "pi05_compact:lang_down_m400"},
    {{768, 160, 1152, 8, Partition::Col, true}, 16, "pi05_compact:vit_qkv_m768"},
    {{768, 1152, 160, 8, Partition::Row, true}, 24, "pi05_compact:vit_o_m768"},
    {{768, 544, 1152, 8, Partition::Col, true}, 21, "pi05_compact:vit_fc1_m768"},
    {{768, 1152, 544, 8, Partition::Row, true}, 20, "pi05_compact:vit_fc2_m768"},
    {{800, 768, 2048, 8, Partition::Col, false}, 26, "pi05_compact:lang_qkv_fuse_m800"},
    {{800, 4096, 2048, 8, Partition::Col, false}, 27, "pi05_compact:lang_gateup_fuse_m800"},
    {{400, 768, 2048, 8, Partition::Col, false}, 25, "pi05_compact:lang_qkv_fuse_m400"},
    {{400, 4096, 2048, 8, Partition::Col, false}, 27, "pi05_compact:lang_gateup_fuse_m400"},
    {{768, 480, 1152, 8, Partition::Col, true}, 28, "pi05_compact:vit_qkv_fuse_m768"},
    {{800, 32, 2048, 8, Partition::Col, false}, 7, "pi05_compact:lang_kv1_m800"},
    {{800, 64, 2048, 8, Partition::Col, false}, 9, "pi05_compact:lang_kv_concat_kv1_m800"},
    {{800, 320, 2048, 8, Partition::Col, false}, 19, "pi05_compact:lang_qkv_concat_kv1_m800"},
    {{400, 32, 2048, 8, Partition::Col, false}, 4, "pi05_compact:lang_kv1_m400"},
    {{400, 64, 2048, 8, Partition::Col, false}, 14, "pi05_compact:lang_kv_concat_kv1_m400"},
    {{400, 320, 2048, 8, Partition::Col, false}, 10, "pi05_compact:lang_qkv_concat_kv1_m400"},
}};
// Bounded local trials using existing payloads at new M values. These are not
// upstream measured winners. Keep full shape keys and the general fallback;
// retain only entries supported by the local same-environment measurements.
inline constexpr std::array<Exact, 25> local_candidates{{
    {{272, 256, 2048, 8, Partition::Col, false}, 10, "local_candidate:lang_qkv_m272"},
    {{272, 2048, 2048, 8, Partition::Row, false}, 27, "local_candidate:lang_down_m272"},
    // Full-prefix Q trial: resident M144/N48/K512, same family as M800 Q.
    {{544, 256, 2048, 8, Partition::Col, false}, 19, "local_candidate:lang_q_m544"},
    // B1 Graph W2/R5, 2026-11-12: two C272 quantized halves share the
    // resident weight pass. Full M544 output remains contiguous FP16.
    {{544, 2048, 2048, 8, Partition::Row, false}, 27, "local_candidate:lang_down_m544"},
    {{512, 160, 1152, 8, Partition::Col, true}, 16, "local_candidate:vit_qkv_m512"},
    {{512, 1152, 160, 8, Partition::Row, true}, 24, "local_candidate:vit_o_m512"},
    {{512, 544, 1152, 8, Partition::Col, true}, 21, "local_candidate:vit_fc1_m512"},
    {{512, 1152, 544, 8, Partition::Row, true}, 20, "local_candidate:vit_fc2_m512"},
    // B1 retained-Graph W2/R5 device measurements, 2026-11-11. Local N/K
    // encode TP8 physical storage; these choices do not extend to C64/decode.
    {{128, 3200, 5120, 8, Partition::Col, false}, 3, "local_candidate:qwen3_32b_gate_up_c128"},
    {{128, 5120, 3200, 8, Partition::Row, false}, 1, "local_candidate:qwen3_32b_down_c128"},
    // The same B1 device comparison at C192 selects one M loop for all
    // seven projections. C128 (including the C192 plan's tail) stays above.
    {{192, 1024, 5120, 8, Partition::Col, false}, 3, "local_candidate:qwen3_32b_q_c192"},
    {{192, 128, 5120, 8, Partition::Col, false}, 3, "local_candidate:qwen3_32b_kv_c192"},
    {{192, 5120, 1024, 8, Partition::Row, false}, 3, "local_candidate:qwen3_32b_o_c192"},
    {{192, 3200, 5120, 8, Partition::Col, false}, 3, "local_candidate:qwen3_32b_gate_up_c192"},
    {{192, 5120, 3200, 8, Partition::Row, false}, 3, "local_candidate:qwen3_32b_down_c192"},
    // B1 M256 W2/R5 device traces, 2026-11-12: one M loop. O selects
    // t15; Q/K/V/Gate/Down select t4. Up uses the separate fused epilogue.
    {{256, 1024, 5120, 8, Partition::Col, false}, 4, "local_candidate:qwen3_32b_q_c256"},
    {{256, 128, 5120, 8, Partition::Col, false}, 4, "local_candidate:qwen3_32b_kv_c256"},
    {{256, 5120, 1024, 8, Partition::Row, false}, 15, "local_candidate:qwen3_32b_o_c256"},
    {{256, 3200, 5120, 8, Partition::Col, false}, 4, "local_candidate:qwen3_32b_gate_up_c256"},
    {{256, 5120, 3200, 8, Partition::Row, false}, 4, "local_candidate:qwen3_32b_down_c256"},
    // C512 adaptation candidate: the installed M512 tile keeps all rows in
    // one M loop. Device validation is required for this new shape set.
    {{512, 1024, 5120, 8, Partition::Col, false}, 15, "local_candidate:qwen3_32b_q_c512"},
    {{512, 128, 5120, 8, Partition::Col, false}, 15, "local_candidate:qwen3_32b_kv_c512"},
    {{512, 5120, 1024, 8, Partition::Row, false}, 15, "local_candidate:qwen3_32b_o_c512"},
    {{512, 3200, 5120, 8, Partition::Col, false}, 15, "local_candidate:qwen3_32b_gate_up_c512"},
    {{512, 5120, 3200, 8, Partition::Row, false}, 15, "local_candidate:qwen3_32b_down_c512"},
}};
inline bool same_shape(const Shape& a, const Shape& b) {
  return a.m == b.m && a.local_n == b.local_n && a.local_k == b.local_k &&
      a.cores == b.cores && a.partition == b.partition && a.bias == b.bias;
}
}  // namespace detail

inline Tile select_tile(const Shape& s) {
  validate_shape(s);
  for (const auto& entry : detail::exact) {
    if (detail::same_shape(s, entry.shape)) {
      auto tile = installed_candidates()[entry.ordinal].tile;
      tile.source = entry.source;
      return validate_tile(tile, s.local_k, s.bias);
    }
  }
  for (const auto& entry : detail::local_candidates) {
    if (detail::same_shape(s, entry.shape)) {
      auto tile = installed_candidates()[entry.ordinal].tile;
      tile.source = entry.source;
      return validate_tile(tile, s.local_k, s.bias);
    }
  }
  auto tile = installed_candidates()[0].tile;
  tile.source = "unmeasured_general_fallback";
  return validate_tile(tile, s.local_k, s.bias);
}

struct Spans {
  uint64_t x, sx, y, bias, weight, scale;
};
inline Spans spans(const Shape& s) {
  validate_shape(s);
  const auto m = uint64_t(s.m), n = uint64_t(s.local_n), k = uint64_t(s.local_k);
  return {m * k, ((m + 15) / 16) * 16 * 2, m * n * 2,
          s.bias ? n * 2 : 0, n * k * uint64_t(s.cores),
          n * 2 * (s.partition == Partition::Col ? uint64_t(s.cores) : 1)};
}
struct Addresses {
  uint32_t x, y, bias, sx;  // Absolute addresses in this core's SPM window.
  uint64_t w = 0, sw = 0;  // DDR byte addresses; zero defers to typed Graph writer.
};
struct Descriptor {
  std::array<uint16_t, 32> regs{};
  std::array<uint16_t, 3> grid{};
  const char* name;
  Tile tile;
  std::size_t ordinal;
};

// All SPM spans remain live for the call and must be disjoint, including SX.
// The allocator supplies its usable window rather than assuming SPM starts at 0.
// DDR register slots may be left zero for GraphDdrRegisterWriter; its caller must
// validate and bind both real tensor extents before submission.
inline Descriptor make_descriptor(const Shape& s, const Addresses& a,
                                  uint32_t spm_base, uint64_t spm_bytes,
                                  const Tile* explicit_tile = nullptr) {
  validate_shape(s);
  Tile tile = explicit_tile ? validate_tile(*explicit_tile, s.local_k, s.bias)
                            : select_tile(s);
  const auto candidate = installed_candidate(tile, s.local_k);
  const auto count = spans(s);
  constexpr uint64_t u32_space = UINT64_C(1) << 32;
  if (!spm_bytes || spm_bytes > 8 * 1024 * 1024 ||
      spm_bytes > u32_space - spm_base)
    throw std::invalid_argument("Invalid W8A8 SPM window");
  struct Interval { uint64_t begin, bytes; };
  const std::array<Interval, 4> ranges{{
      {a.x, count.x}, {a.sx, count.sx}, {a.y, count.y}, {a.bias, count.bias}}};
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const auto& range = ranges[i];
    if (!range.bytes) continue;
    if (range.begin % 32 || range.begin < spm_base || range.bytes > spm_bytes ||
        range.begin - spm_base > spm_bytes - range.bytes)
      throw std::invalid_argument("W8A8 operand exceeds aligned SPM window");
    for (std::size_t j = 0; j < i; ++j) {
      const auto& other = ranges[j];
      if (other.bytes && range.begin < other.begin + other.bytes &&
          other.begin < range.begin + range.bytes)
        throw std::invalid_argument("W8A8 live SPM operands overlap");
    }
  }
  if (!s.bias && a.bias)
    throw std::invalid_argument("W8A8 bias address supplied to a no-bias shape");
  constexpr uint64_t ddr_limit = UINT64_C(1) << 40;
  for (auto operand : {Interval{a.w, count.weight}, Interval{a.sw, count.scale}}) {
    if (operand.begin && (operand.begin % 256 || operand.begin >= ddr_limit ||
        operand.bytes > ddr_limit - operand.begin))
      throw std::invalid_argument("W8A8 DDR operand exceeds 256-byte/u40 ABI");
  }
  const auto grid_n = (s.local_n + tile.n - 1) / tile.n;
  const auto grid_m = tile.resident ? 1 : (s.m + tile.m - 1) / tile.m;
  if (grid_n > 65535 || grid_m > 65535)
    throw std::invalid_argument("W8A8 launch grid exceeds uint16");
  Descriptor d{{}, {uint16_t(grid_n), uint16_t(grid_m), 1},
               candidate.name, tile, candidate.ordinal};
  auto u32 = [&d](std::size_t slot, uint64_t value) {
    d.regs[slot] = static_cast<uint16_t>(value & 0xffff);
    d.regs[slot + 1] = static_cast<uint16_t>(value >> 16);
  };
  u32(0, a.x); u32(2, a.w >> 8); u32(4, a.y); u32(6, a.bias);
  d.regs[8] = s.partition == Partition::Col;
  d.regs[9] = static_cast<uint16_t>(s.m);
  u32(10, s.local_n);
  d.regs[12] = static_cast<uint16_t>(s.local_k);
  d.regs[13] = static_cast<uint16_t>(s.local_k / 16);
  d.regs[14] = s.bias;
  d.regs[15] = static_cast<uint16_t>(s.cores);
  u32(16, s.local_k); u32(18, 2 * s.local_n); u32(20, a.sw >> 8);
  d.regs[26] = 2;  // FP16 output/bias, despite INT8 input.
  d.regs[27] = 1;
  u32(30, a.sx);
  return d;
}

}  // namespace rpu_pl_w8a8

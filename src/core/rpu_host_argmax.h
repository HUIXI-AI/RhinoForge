#pragma once

// Host reduction shared by the existing FP16 DDR reader and CPU FP32 logits.
// Keep NaNs and signed zeros observable: this code must not use fast-math.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/Half.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rpu::host_argmax {

#if defined(__aarch64__)
inline float32x4_t load_float4(const c10::Half* values) {
  static_assert(sizeof(c10::Half) == sizeof(uint16_t));
  uint16_t bits[4];
  std::memcpy(bits, values, sizeof(bits));
  return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(bits)));
}

inline float32x4_t load_float4(const float* values) {
  return vld1q_f32(values);
}
#endif

template <typename scalar_t>
inline int64_t argmax_row(const scalar_t* row, int64_t cols) {
#if defined(__aarch64__)
  const scalar_t* values = row;
  if (cols >= 8) {
    int64_t offset = 0;
    float32x4_t max0 = vdupq_n_f32(-INFINITY);
    float32x4_t max1 = max0;
    for (; offset + 8 <= cols; offset += 8) {
      const float32x4_t value0 = load_float4(values + offset);
      const float32x4_t value1 = load_float4(values + offset + 4);
      const uint32x4_t ordered0 = vceqq_f32(value0, value0);
      const uint32x4_t ordered1 = vceqq_f32(value1, value1);
      if (vminvq_u32(vandq_u32(ordered0, ordered1)) == 0) {
        for (int64_t lane = offset; lane < offset + 8; ++lane) {
          if (std::isnan(static_cast<float>(values[lane]))) return lane;
        }
      }
      max0 = vmaxq_f32(max0, value0);
      max1 = vmaxq_f32(max1, value1);
    }
    float best = vmaxvq_f32(vmaxq_f32(max0, max1));
    for (int64_t lane = offset; lane < cols; ++lane) {
      const float value = static_cast<float>(values[lane]);
      if (std::isnan(value)) return lane;
      if (value > best) best = value;
    }

    // A second ordered scan keeps the first tie, including -0 == +0 and inf.
    const float32x4_t best4 = vdupq_n_f32(best);
    offset = 0;
    for (; offset + 8 <= cols; offset += 8) {
      const uint32x4_t equal0 = vceqq_f32(load_float4(values + offset), best4);
      const uint32x4_t equal1 = vceqq_f32(load_float4(values + offset + 4), best4);
      if (vmaxvq_u32(vorrq_u32(equal0, equal1)) != 0) {
        for (int64_t lane = offset; lane < offset + 8; ++lane) {
          if (static_cast<float>(values[lane]) == best) return lane;
        }
      }
    }
    for (; offset < cols; ++offset) {
      if (static_cast<float>(values[offset]) == best) return offset;
    }
  }
#endif

  int64_t best_index = 0;
  float best_value = static_cast<float>(row[0]);
  if (std::isnan(best_value)) return 0;
  for (int64_t col = 1; col < cols; ++col) {
    const float value = static_cast<float>(row[col]);
    if (std::isnan(value)) return col;
    if (value > best_value) {
      best_value = value;
      best_index = col;
    }
  }
  return best_index;
}

// The caller has admitted a contiguous, nonempty-last-dimension CPU alias.
inline at::Tensor reduce_cpu_rows(const at::Tensor& logits) {
  const int64_t vocab = logits.size(-1);
  const int64_t rows = logits.numel() / vocab;
  at::Tensor out = at::empty({rows}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* indices = out.data_ptr<int64_t>();
  const auto scan = [&](const auto* values) {
    at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        indices[row] = argmax_row(values + row * vocab, vocab);
      }
    });
  };
  if (logits.scalar_type() == at::kHalf) {
    scan(logits.data_ptr<c10::Half>());
  } else {
    scan(logits.data_ptr<float>());
  }
  auto sizes = logits.sizes().vec();
  sizes.pop_back();
  return out.reshape(sizes);
}

// Only this CPU compatibility entry admits FP32. The RPU top1 path stays FP16.
inline at::Tensor argmax_lastdim_cpu(const at::Tensor& logits) {
  TORCH_CHECK(logits.device().is_cpu(), "host argmax: logits must be on CPU");
  TORCH_CHECK(logits.dim() == 2,
              "argmax_lastdim_host expects 2-D [rows, cols] logits, got ",
              logits.dim(), "-D");
  TORCH_CHECK(logits.scalar_type() == at::kHalf || logits.scalar_type() == at::kFloat,
              "host argmax: CPU logits must be fp16 or fp32, got ", logits.scalar_type());
  TORCH_CHECK(logits.is_contiguous(), "host top1: logits must be contiguous");
  TORCH_CHECK(logits.size(-1) > 0, "host top1: empty last dimension");
  return reduce_cpu_rows(logits);
}

}  // namespace rpu::host_argmax

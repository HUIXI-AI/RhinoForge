#pragma once

// Host top-k draw shared by the TTS Predictor route and its compiled contract
// test, so the test checks the shipped arithmetic rather than a copy of it.
//
// The reference it reproduces is transformers' TemperatureLogitsWarper ->
// TopKLogitsWarper -> softmax -> draw. Two details of that chain are load
// bearing and easy to get wrong: the scale happens BEFORE selection, so
// selection sees scaled values and two raw logits that collapse onto one
// scaled value both survive; and the warper removes `scores < kth` rather
// than keeping a fixed-size set, so every entry tied with the k-th largest is
// kept and more than `top_k` candidates can remain.
//
// The draw itself is an inverse CDF over one uniform. aten::multinomial with
// n_sample == 1 instead races a full-width Exp(1) vector and takes an argmax,
// which is the same distribution but a different number of random draws, so
// the token drawn here differs from that call under one seed. Keep NaNs
// observable: this code must not use fast-math.

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Utils.h>
#include <ATen/core/DistributionsHelper.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <vector>

namespace rpu::host_sample {

// One row's draw. `kept` exceeds `top_k` exactly when the k-th largest scaled
// logit is tied. `bad_column` is the first non-finite column when there is
// one, and then no draw happened and `column` stays -1: a non-finite logit
// means the producing route failed, not that this row has an unusual
// distribution, so it is reported instead of being sampled from.
struct RowDraw {
    int64_t column = -1;
    int64_t kept = 0;
    int64_t bad_column = -1;
};

// `scaled` holds `cols` floats and `heap` holds `min(top_k, cols)`; both are
// caller-owned so one allocation serves a whole generation loop. `uniform` is
// a draw from [0, 1).
template <typename scalar_t>
inline RowDraw sample_topk_row(const scalar_t* row, int64_t cols, float temperature,
                               int64_t top_k, double uniform, float* scaled,
                               float* heap) {
    RowDraw draw;
    const int64_t width = std::min(top_k, cols);
    float highest = -std::numeric_limits<float>::infinity();
    // One pass scales, screens for non-finite values, and keeps a width-sized
    // min-heap whose root ends up as the k-th largest scaled value.
    for (int64_t col = 0; col < cols; ++col) {
        const float value = static_cast<float>(row[col]) / temperature;
        if (!std::isfinite(value)) {
            draw.bad_column = col;
            return draw;
        }
        scaled[col] = value;
        if (value > highest) highest = value;
        if (col < width) {
            heap[col] = value;
            if (col + 1 == width) std::make_heap(heap, heap + width, std::greater<float>());
        } else if (value > heap[0]) {
            std::pop_heap(heap, heap + width, std::greater<float>());
            heap[width - 1] = value;
            std::push_heap(heap, heap + width, std::greater<float>());
        }
    }
    const float threshold = heap[0];

    // The row maximum is never below the threshold, so one term is exp(0) and
    // the total is at least 1: the walk below cannot divide by zero.
    double total = 0.0;
    for (int64_t col = 0; col < cols; ++col) {
        if (scaled[col] >= threshold) {
            total += std::exp(static_cast<double>(scaled[col]) - highest);
            ++draw.kept;
        }
    }

    const double target = uniform * total;
    double walked = 0.0;
    for (int64_t col = 0; col < cols; ++col) {
        if (scaled[col] < threshold) continue;
        draw.column = col;
        walked += std::exp(static_cast<double>(scaled[col]) - highest);
        if (walked > target) return draw;
    }
    // Rounding can leave the walk a hair short of a target near the total; the
    // remaining mass belongs to the last kept column, already recorded above.
    return draw;
}

// The production entry for CPU rows, shared by the backend op (which hands it a
// zero-copy view of RPU memory) and by the compiled semantic test.
inline at::Tensor sample_topk_lastdim_cpu(const at::Tensor& logits, double temperature,
                                          int64_t top_k,
                                          std::optional<at::Generator> generator) {
    TORCH_CHECK(logits.device().is_cpu(), "host topk sample: logits must be on CPU");
    TORCH_CHECK(logits.scalar_type() == at::kHalf || logits.scalar_type() == at::kFloat,
                "host topk sample: logits must be fp16 or fp32, got ",
                logits.scalar_type());
    TORCH_CHECK(logits.dim() == 2,
                "sample_topk_lastdim_host expects 2-D [rows, cols] logits, got ",
                logits.dim(), "-D");
    TORCH_CHECK(logits.is_contiguous(), "host topk sample: logits must be contiguous");
    TORCH_CHECK(std::isfinite(temperature) && temperature > 0.0,
                "host topk sample: temperature must be finite and positive, got ",
                temperature);
    TORCH_CHECK(top_k >= 1, "host topk sample: top_k must be at least 1, got ", top_k);
    const int64_t cols = logits.size(-1);
    const int64_t rows = logits.size(0);
    TORCH_CHECK(cols > 0, "host topk sample: empty last dimension");

    // One uniform per row, from the caller's generator or the default CPU one,
    // so torch.manual_seed still makes a generation reproducible. Drawn up front
    // under one lock, because the scan below holds no generator state.
    std::vector<double> uniforms(static_cast<size_t>(rows));
    {
        auto* engine = at::get_generator_or_default<at::CPUGeneratorImpl>(
            generator, at::detail::getDefaultCPUGenerator());
        std::lock_guard<std::mutex> lock(engine->mutex_);
        at::uniform_real_distribution<double> uniform(0.0, 1.0);
        for (int64_t row = 0; row < rows; ++row) uniforms[row] = uniform(engine);
    }

    at::Tensor out = at::empty(
        {rows}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    std::vector<float> scaled(static_cast<size_t>(cols));
    std::vector<float> heap(static_cast<size_t>(std::min(top_k, cols)));
    const auto draw_rows = [&](const auto* values) {
        int64_t* tokens = out.data_ptr<int64_t>();
        for (int64_t row = 0; row < rows; ++row) {
            const RowDraw draw = sample_topk_row(
                values + row * cols, cols, static_cast<float>(temperature), top_k,
                uniforms[row], scaled.data(), heap.data());
            TORCH_CHECK(draw.bad_column < 0,
                        "host topk sample: non-finite logit at row ", row,
                        " column ", draw.bad_column);
            tokens[row] = draw.column;
        }
    };
    if (logits.scalar_type() == at::kHalf) {
        draw_rows(logits.data_ptr<c10::Half>());
    } else {
        draw_rows(logits.data_ptr<float>());
    }
    return out;
}

}  // namespace rpu::host_sample

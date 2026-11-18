#pragma once
#include <ATen/ATen.h>
#include <memory>
#include <string>

class TtsTalkerSampler {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    at::Tensor reset_impl(int64_t frames, const at::Tensor& reference, bool interleaved);
public:
    explicit TtsTalkerSampler(std::string path);
    ~TtsTalkerSampler();
    void reset(int64_t frames, const at::Tensor& reference);
    at::Tensor reset_interleaved(int64_t frames, const at::Tensor& reference);
    void sample(const at::Tensor& logits);
    at::Tensor result();
};

#pragma once
#include "graph/graph_runtime.h"

class TtsSamplingExperiment;

class TtsPredictorGroup {
    std::string asset_path_;
    std::unique_ptr<TtsSamplingExperiment> sampler_;
    std::array<std::deque<uint64_t>, 15> dma_slots_;
    std::optional<GraphSignature> signature_;
    std::vector<GraphSignature> groups_;
    RpuGraphCache* cache_ = nullptr;
    std::vector<at::Tensor> owners_;
    std::array<at::Tensor, 15> embedding_outputs_;
    bool failed_ = false;
public:
    explicit TtsPredictorGroup(std::string path);
    ~TtsPredictorGroup();
    void reserve(int64_t frames, const at::Tensor& reference);
    void reserve_noise(int64_t frames, const at::Tensor& reference, const at::Tensor& noise);
    void retire(RpuGraphCache& cache);
    at::Tensor generate(int64_t handle, const at::Tensor& input,
        const std::vector<at::Tensor>& embeddings, const std::vector<at::Tensor>& keys,
        const std::vector<at::Tensor>& values, RpuGraphCache& cache,
        const GraphSignature& prefill, const GraphSignature& decode,
        const std::vector<int64_t>& prefill_descriptor, const std::vector<int64_t>& decode_descriptor,
        int64_t execution_len, const at::Tensor& progress);
};

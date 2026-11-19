#include "tts_predictor_group.h"
#include "tts_sampling_experiment.h"
#include "core/rpu_kernel_decls.h"
#include <cstdlib>

TtsPredictorGroup::TtsPredictorGroup(std::string path) : asset_path_(std::move(path)) {
    TORCH_CHECK(!asset_path_.empty(), "TTS group requires explicit experimental assets");
    sampler_ = std::make_unique<TtsSamplingExperiment>(asset_path_);
}
TtsPredictorGroup::~TtsPredictorGroup() = default;

void TtsPredictorGroup::reserve(int64_t frames, const at::Tensor& reference) {
    TORCH_CHECK(!failed_ && !RpuKernelGraph::has_active(), "TTS group reservation requires an idle owner");
    sampler_->reserve(int(frames), reference);
}

void TtsPredictorGroup::reserve_noise(
        int64_t frames, const at::Tensor& reference, const at::Tensor& noise) {
    TORCH_CHECK(!failed_ && !RpuKernelGraph::has_active(), "TTS group reservation requires an idle owner");
    sampler_->reserve_noise(int(frames), reference, noise);
}

void TtsPredictorGroup::retire(RpuGraphCache& cache) {
    TORCH_CHECK(!RpuKernelGraph::has_active() && (!signature_ || cache_ == &cache),
                "TTS group retirement requires its cached Graph to be cleared first");
    for (const auto& signature : groups_)
        TORCH_CHECK(cache.lookup(signature) == nullptr, "TTS group Graph is still cached");
    failed_ = true;
    sampler_.reset();
    owners_.clear();
}

at::Tensor TtsPredictorGroup::generate(int64_t handle, const at::Tensor& input,
        const std::vector<at::Tensor>& embeddings, const std::vector<at::Tensor>& keys,
        const std::vector<at::Tensor>& values, RpuGraphCache& cache,
        const GraphSignature& prefill, const GraphSignature& decode,
        const std::vector<int64_t>& prefill_descriptor, const std::vector<int64_t>& decode_descriptor,
        int64_t execution_len, const at::Tensor& progress) {
    TORCH_CHECK(!failed_ && !RpuKernelGraph::has_active(), "TTS group requires a healthy idle owner");
    TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kFloat &&
                    input.sizes().equals({1,2,1024}) && embeddings.size() == 14 &&
                    keys.size() == 5 && values.size() == 5 &&
                    (execution_len == 2 || execution_len == 16 || execution_len == 32),
                "TTS group outside frozen input profile");
    TORCH_CHECK(progress.device().is_cpu() && progress.scalar_type() == at::kLong &&
                    progress.is_contiguous() && progress.sizes().equals({4}) &&
                    progress.eq(0).all().item<bool>(), "TTS group requires zeroed progress");
    TORCH_CHECK(prefill.op_id_str == "qwen3_tts_prefill_predictor" &&
                    decode.op_id_str == "qwen3_tts_predictor" &&
                    !prefill.dyn_dims.empty() && !decode.dyn_dims.empty() &&
                    prefill.dyn_dims[0] == handle && decode.dyn_dims[0] == handle,
                "TTS group decoder owner mismatch");
    for (const auto& table : embeddings)
        TORCH_CHECK(table.device().is_privateuseone() && table.device() == keys[0].device() &&
                        table.scalar_type() == at::kHalf && table.is_contiguous() &&
                        table.sizes().equals({2048,1024}), "invalid TTS group embedding");
    auto half = input.to(at::kHalf).contiguous();
    TORCH_CHECK(at::isfinite(half).all().item<bool>(), "TTS group input exceeds FP16 range");
    if (execution_len > 2) half = at::constant_pad_nd(half, {0,0,0,execution_len-2}, 0);
    auto hidden = half.to(keys[0].device());
    const int steps = std::getenv("RPU_TTS_SAMPLING_TWO_STEP") ? 2 : 15;
    auto signature = prefill;
    signature.op_id_str = "qwen3_tts_predictor_group";
    signature.dyn_dims.push_back(reinterpret_cast<int64_t>(this));
    signature.dyn_dims.push_back(steps);
    signature.dyn_dims.push_back(execution_len);
    signature.dyn_dims.insert(signature.dyn_dims.end(), prefill_descriptor.begin(), prefill_descriptor.end());
    signature.dyn_dims.insert(signature.dyn_dims.end(), decode_descriptor.begin(), decode_descriptor.end());
    std::vector<at::Tensor> owners(embeddings);
    owners.insert(owners.end(), keys.begin(), keys.end());
    owners.insert(owners.end(), values.begin(), values.end());
    if (!signature_) {
        // Retire the two step graphs before consuming another arena.
        cache.evict(prefill);
        cache.evict(decode);
        signature_ = signature;
        for (int start = 0; start < steps; start += 8) {
            auto group = signature;
            group.dyn_dims.push_back(start);
            groups_.push_back(std::move(group));
        }
        cache_ = &cache;
        owners_ = owners;
    } else {
        TORCH_CHECK(cache_ == &cache && *signature_ == signature, "TTS group profile changed");
        for (const auto& group : groups_)
            TORCH_CHECK(cache.lookup(group) != nullptr,
                        "TTS group cache was retired; reload the model");
        for (size_t i = 0; i < owners.size(); ++i)
            TORCH_CHECK(owners_[i].is_same(owners[i]), "TTS group weight/KV owner changed");
    }
    sampler_->reset();
    sampler_->prepare_graph(keys[0].options());
    // Keep both prepared queues below the framework's conservative entry budget.
    for (int start = 0; start < steps; start += 8) {
        const auto& group = groups_[start / 8];
        auto& graph = cache.get_or_create(group);
        const auto submissions_before = graph.debug_stats().hw_batch_submit_total;
        try {
            graph.begin(group);
            const bool replay = graph.state() == RpuKernelGraph::State::REPLAYING;
            // A group is not the single-forward stream required by checked decode replay.
            graph.record_barrier(0, 0);
            for (int index = start; index < std::min(start + 8, steps); ++index) {
                const auto begin = graph.op_stream_stamp();
                auto raw = rpu_causal_decoder_forward(handle, hidden, keys, values,
                    std::nullopt, index == 0 ? 0 : index + 1, true,
                    std::nullopt, std::nullopt, std::nullopt, std::nullopt, -1, 0, false, 0,
                    index == 0 ? prefill_descriptor : decode_descriptor);
                graph.snapshot_legacy_dma_bases(begin, dma_slots_[index]);
                graph.keep_alive(hidden);
                graph.keep_alive(raw);
                auto logits = rpu_causal_decoder_tts_predictor_logits(handle);
                graph.record_barrier(0, 0);
                hidden = sampler_->run(logits, embeddings[std::min(index, 13)], index);
                graph.record_barrier(0, 0);
            }
            graph.end();
            cache.touch_entry(group, replay);
            const auto stats = graph.debug_stats();
            const auto submitted = stats.hw_batch_submit_total - submissions_before;
            TORCH_CHECK(graph.replayable() && submitted == 1,
                        "TTS group did not execute as one hardware batch: ", submitted);
        } catch (...) {
            failed_ = true;
            try { graph.abort(); }
            catch (const std::exception& error) { TORCH_WARN("TTS group abort failed: ", error.what()); }
            throw;
        }
    }
    auto* completed = progress.data_ptr<int64_t>();
    completed[0] = completed[2] = completed[3] = steps;
    completed[1] = steps + 1;
    TORCH_CHECK(steps == 15, "TTS_SAMPLING_TWO_STEP_COMPLETE");
    return sampler_->finish();
}

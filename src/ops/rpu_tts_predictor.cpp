#include "core/rpu_kernel_decls.h"
#include "graph/graph_runtime.h"
#include <ATen/record_function.h>
#include <limits>
#include "tts_sampling_experiment.h"
#include <cstdlib>

at::Tensor rpu_tts_predictor_generate(
    int64_t handle, const at::Tensor& input, at::TensorList embeddings,
    at::TensorList keys, at::TensorList values, RpuGraphCache& cache,
    const GraphSignature& prefill_signature, const GraphSignature& decode_signature,
    at::IntArrayRef prefill_descriptor, at::IntArrayRef decode_descriptor,
    int64_t execution_len, const at::Tensor& progress) {
    TORCH_CHECK(!RpuKernelGraph::has_active(), "TTS Predictor loop requires no active Graph");
    TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kFloat &&
                    input.sizes().equals({1, 2, 1024}), "TTS Predictor requires CPU FP32 [1,2,1024]");
    TORCH_CHECK(embeddings.size() == 14 && keys.size() == 5 && values.size() == 5 &&
                    execution_len >= 2 && execution_len <= 32,
                "TTS Predictor loop outside the frozen profile");
    const char* experiment_path = std::getenv("RPU_TTS_SAMPLING_EXPERIMENT");
    const bool experiment = experiment_path && *experiment_path;
    for (const auto& weight : embeddings)
        TORCH_CHECK((experiment ? weight.device() == keys[0].device() && weight.scalar_type() == at::kHalf
                               : weight.device().is_cpu() && weight.scalar_type() == at::kFloat) &&
                        weight.sizes().equals({2048, 1024}) && (!experiment || weight.is_contiguous()),
                    "TTS Predictor requires contiguous embeddings in the selected precision/device");
    TORCH_CHECK(progress.device().is_cpu() && progress.scalar_type() == at::kLong &&
                    progress.is_contiguous() && progress.sizes().equals({4}) &&
                    progress.eq(0).all().item<bool>(), "TTS Predictor requires zeroed progress");
    TORCH_CHECK(prefill_signature.op_id_str == "qwen3_tts_prefill_predictor" &&
                    decode_signature.op_id_str == "qwen3_tts_predictor" &&
                    !prefill_signature.dyn_dims.empty() && !decode_signature.dyn_dims.empty() &&
                    prefill_signature.dyn_dims[0] == handle && decode_signature.dyn_dims[0] == handle,
                "TTS Predictor Graph owner mismatch");
    auto prefill = cache.lookup_shared(prefill_signature);
    auto decode = cache.lookup_shared(decode_signature);
    auto ready = [](const std::shared_ptr<RpuKernelGraph>& graph, const GraphSignature& sig) {
        return graph && graph->state() == RpuKernelGraph::State::BUILT && graph->replayable() &&
               graph->has_built_signature() && graph->built_signature() == sig;
    };
    TORCH_CHECK(ready(prefill, prefill_signature) && ready(decode, decode_signature) &&
                    prefill != decode, "TTS Predictor loop requires two prepared replay Graphs");
    auto* completed = progress.data_ptr<int64_t>();
    std::vector<at::Tensor> tokens;
    std::unique_ptr<TtsSamplingExperiment> sampler;
    at::Tensor device_ids;
    if (experiment) {
        sampler = std::make_unique<TtsSamplingExperiment>(experiment_path);
        device_ids = at::empty({15, 8}, keys[0].options().dtype(at::kInt));
    }
    auto hidden = input;
    for (int64_t index = 0; index < 15; ++index) {
        const int64_t position = index == 0 ? 0 : index + 1;
        const int64_t rows = index == 0 ? execution_len : 1;
        if (hidden.device().is_cpu())
            TORCH_CHECK(at::isfinite(hidden).all().item<bool>(), "non-finite TTS CPU input");
        auto half = hidden.to(at::kHalf).contiguous();
        if (half.device().is_cpu())
            TORCH_CHECK(at::isfinite(half).all().item<bool>(), "TTS input exceeds FP16 range");
        if (index == 0 && rows > 2)
            half = at::constant_pad_nd(half, {0, 0, 0, rows - 2}, 0);
        auto device_input = half.to(keys[0].device());
        const auto& graph = index == 0 ? prefill : decode;
        const auto& sig = index == 0 ? prefill_signature : decode_signature;
        at::Tensor raw;
        {
            RECORD_FUNCTION(index == 0 ? "qwen3_tts_prefill_predictor" : "qwen3_tts_predictor",
                            std::vector<c10::IValue>());
            try {
                graph->begin(sig);
                TORCH_CHECK(graph->state() == RpuKernelGraph::State::REPLAYING,
                            "TTS Predictor loop cannot BUILD a Graph");
                raw = rpu_causal_decoder_forward(handle, device_input, keys, values,
                    std::nullopt, position, true, std::nullopt, std::nullopt,
                    std::nullopt, std::nullopt, -1, 0, false, 0,
                    index == 0 ? prefill_descriptor : decode_descriptor);
                graph->end();
                cache.touch_entry(sig, true);
            } catch (...) {
                try { graph->abort(); }
                catch (const std::exception& error) { TORCH_WARN("TTS Graph abort failed: ", error.what()); }
                throw;
            }
        }
        TORCH_CHECK(raw.scalar_type() == at::kHalf && raw.sizes().equals({1, rows, 1024}),
                    "invalid TTS Predictor device hidden state");
        ++completed[0];
        completed[1] += index == 0 ? 2 : 1;
        auto logits = rpu_causal_decoder_tts_predictor_logits(handle);
        TORCH_CHECK(logits.scalar_type() == at::kHalf && logits.sizes().equals({1, 1, 2048}),
                    "invalid fused TTS Predictor logits");
        ++completed[2];
        if (sampler) {
            hidden = sampler->run(logits, embeddings[std::min<int64_t>(index, 13)], device_ids, index);
            ++completed[3];
            if (index == 1 && std::getenv("RPU_TTS_SAMPLING_TWO_STEP"))
                TORCH_CHECK(false, "TTS_SAMPLING_TWO_STEP_COMPLETE");
            continue;
        }
        auto scores = logits.cpu().to(at::kFloat);
        TORCH_CHECK(at::isfinite(scores).all().item<bool>(), "non-finite TTS device logits");
        ++completed[3];
        scores = scores.select(1, 0) / 0.9;
        auto threshold = std::get<0>(scores.topk(50, -1)).select(-1, 49).unsqueeze(-1);
        scores = scores.masked_fill(scores < threshold, -std::numeric_limits<float>::infinity());
        auto ids = at::multinomial(scores.softmax(-1), 1, false);
        tokens.push_back(ids);
        if (index < 14)
            hidden = at::embedding(embeddings[index], ids, -1, false, false);
    }
    if (sampler) {
        return sampler->finish(device_ids);
    }
    return at::cat(tokens, 1);
}

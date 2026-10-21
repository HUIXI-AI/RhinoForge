// Fixed Whisper-base decoder layers: pre-norm self-attn, cross-attn, GELU FFN.

#include "model_handle_registry.h"
#include "rpu_helpers.h"
#include "rpu_kvinsert_segment_plan.h"
#include "rpu_ops.h"
#include "rpu_runtime_state.h"
#include "rpu_spm_allocator.h"

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

namespace {

constexpr int64_t WHISPER_DECODER_HIDDEN = 512;
constexpr int64_t WHISPER_DECODER_INTERMEDIATE = 2048;
constexpr int64_t WHISPER_DECODER_HEADS = 8;
constexpr int64_t WHISPER_DECODER_HEAD_DIM = 64;
constexpr int64_t WHISPER_DECODER_CROSS_TOKENS = 1500;
constexpr int64_t WHISPER_DECODER_TEXT_TOKENS = 448;
constexpr int64_t WHISPER_DECODER_CORES = 8;
constexpr int64_t WHISPER_DECODER_DWIDTH = 2;
constexpr double WHISPER_DECODER_EPS = 1e-5;
constexpr size_t WHISPER_DECODER_SDPA_TMP_BYTES = 512 * 1024;
constexpr uint32_t WHISPER_DECODER_KV_CAPABILITIES = KV_INSERT_CAP_V2;

size_t aligned_bytes(int64_t elements) {
    return static_cast<size_t>((elements * WHISPER_DECODER_DWIDTH + 255) / 256 * 256);
}

void check_device_half(
    const at::Tensor& tensor, at::IntArrayRef shape, const char* name) {
    TORCH_CHECK(tensor.defined() && tensor.device().type() == at::kPrivateUse1 &&
                    tensor.scalar_type() == at::kHalf && tensor.is_contiguous() &&
                    tensor.sizes() == shape,
                "Whisper decoder ", name,
                " must be contiguous device FP16 with shape ", shape,
                ", got ", tensor.sizes());
}

void check_cache(const at::Tensor& cache, int64_t blocks, const char* name) {
    TORCH_CHECK(cache.defined() && cache.device().type() == at::kPrivateUse1 &&
                    cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
                    cache.dim() == 7 && cache.size(0) == 1 &&
                    cache.size(1) >= blocks && cache.size(2) == 1 &&
                    cache.size(3) == 4 && cache.size(4) == 8 &&
                    cache.size(5) == 16 && cache.size(6) == 16,
                "Whisper decoder ", name,
                " must use the batch-1 H8 D64 RPUCache layout, got ",
                cache.sizes());
}

class WhisperDecoderModel {
public:
    struct Layer {
        at::Tensor sq_w, sk_w, sv_w, so_w;
        at::Tensor cq_w, co_w, ff1_w, ff2_w;
        at::Tensor n1_w, n1_b, n2_w, n2_b, n3_w, n3_b;
        at::Tensor sq_b, sv_b, so_b, cq_b, co_b, ff1_b, ff2_b;
    };

    void set_weights(
        at::TensorList sq_w, at::TensorList sk_w,
        at::TensorList sv_w, at::TensorList so_w,
        at::TensorList cq_w, at::TensorList co_w,
        at::TensorList ff1_w, at::TensorList ff2_w,
        at::TensorList n1_w, at::TensorList n1_b,
        at::TensorList n2_w, at::TensorList n2_b,
        at::TensorList n3_w, at::TensorList n3_b,
        at::TensorList sq_b, at::TensorList sv_b,
        at::TensorList so_b, at::TensorList cq_b,
        at::TensorList co_b, at::TensorList ff1_b,
        at::TensorList ff2_b) {
        TORCH_CHECK(layers_.empty(),
                    "Whisper decoder weights may be installed only once");
        const int64_t count = static_cast<int64_t>(sq_w.size());
        TORCH_CHECK(count == 1,
                    "Whisper decoder layer probe requires exactly one layer");
        auto same_count = [count](at::TensorList values, const char* name) {
            TORCH_CHECK(static_cast<int64_t>(values.size()) == count,
                        "Whisper decoder ", name, " must contain ", count,
                        " tensors");
        };
        for (const auto& item : std::vector<std::pair<at::TensorList, const char*>>{
                 {sk_w, "sk_w"}, {sv_w, "sv_w"}, {so_w, "so_w"},
                 {cq_w, "cq_w"}, {co_w, "co_w"}, {ff1_w, "ff1_w"},
                 {ff2_w, "ff2_w"}, {n1_w, "n1_w"}, {n1_b, "n1_b"},
                 {n2_w, "n2_w"}, {n2_b, "n2_b"}, {n3_w, "n3_w"},
                 {n3_b, "n3_b"}, {sq_b, "sq_b"}, {sv_b, "sv_b"},
                 {so_b, "so_b"}, {cq_b, "cq_b"}, {co_b, "co_b"},
                 {ff1_b, "ff1_b"}, {ff2_b, "ff2_b"}}) {
            same_count(item.first, item.second);
        }
        auto check_list = [](at::TensorList values, at::IntArrayRef shape,
                             const char* name) {
            for (const auto& value : values) check_device_half(value, shape, name);
        };
        for (const auto& item : std::vector<std::pair<at::TensorList, const char*>>{
                 {sq_w, "sq_w"}, {sk_w, "sk_w"}, {sv_w, "sv_w"},
                 {so_w, "so_w"}, {cq_w, "cq_w"}, {co_w, "co_w"}}) {
            check_list(item.first, {WHISPER_DECODER_HIDDEN,
                                    WHISPER_DECODER_HIDDEN}, item.second);
        }
        check_list(ff1_w, {WHISPER_DECODER_INTERMEDIATE,
                           WHISPER_DECODER_HIDDEN}, "ff1_w");
        check_list(ff2_w, {WHISPER_DECODER_HIDDEN,
                           WHISPER_DECODER_INTERMEDIATE}, "ff2_w");
        for (const auto& item : std::vector<std::pair<at::TensorList, const char*>>{
                 {n1_w, "n1_w"}, {n1_b, "n1_b"}, {n2_w, "n2_w"},
                 {n2_b, "n2_b"}, {n3_w, "n3_w"}, {n3_b, "n3_b"},
                 {sq_b, "sq_b"}, {sv_b, "sv_b"}, {so_b, "so_b"},
                 {cq_b, "cq_b"}, {co_b, "co_b"}, {ff2_b, "ff2_b"}}) {
            check_list(item.first, {WHISPER_DECODER_HIDDEN}, item.second);
        }
        check_list(ff1_b, {WHISPER_DECODER_INTERMEDIATE}, "ff1_b");
        layers_.resize(count);
        for (int64_t i = 0; i < count; ++i) {
            layers_[i] = {
                sq_w[i], sk_w[i], sv_w[i], so_w[i], cq_w[i], co_w[i],
                ff1_w[i], ff2_w[i], n1_w[i], n1_b[i], n2_w[i], n2_b[i],
                n3_w[i], n3_b[i], sq_b[i], sv_b[i], so_b[i], cq_b[i],
                co_b[i], ff1_b[i], ff2_b[i]};
        }
    }

    void reset(int64_t position) {
        TORCH_CHECK(position == 0,
                    "Whisper decoder reset position must be zero");
        expected_position_ = 0;
    }

    at::Tensor forward(
        const at::Tensor& hidden, const at::Tensor& output,
        at::TensorList self_k, at::TensorList self_v,
        at::TensorList cross_k, at::TensorList cross_v,
        int64_t position, bool prefill) {
        TORCH_CHECK(!layers_.empty(), "Whisper decoder weights are not installed");
        TORCH_CHECK(hidden.dim() == 3,
                    "Whisper decoder hidden must be [1, seq, 512]");
        const int64_t seq = hidden.size(1);
        check_device_half(hidden, {1, seq, WHISPER_DECODER_HIDDEN}, "hidden");
        check_device_half(output, {1, seq, WHISPER_DECODER_HIDDEN}, "output");
        TORCH_CHECK(seq > 0 && position >= 0 &&
                        position + seq <= WHISPER_DECODER_TEXT_TOKENS,
                    "Whisper decoder position range exceeds 448 tokens");
        TORCH_CHECK(position == expected_position_,
                    "Whisper decoder position must append exactly at ",
                    expected_position_, ", got ", position);
        TORCH_CHECK(prefill ? position == 0 && seq > 1 : seq == 1,
                    prefill ? "Whisper decoder prefill requires position 0 and seq > 1"
                            : "Whisper decoder decode requires one token");
        TORCH_CHECK(self_k.size() == layers_.size() &&
                        self_v.size() == layers_.size() &&
                        cross_k.size() == layers_.size() &&
                        cross_v.size() == layers_.size(),
                    "Whisper decoder cache lists must match the installed layers");
        for (size_t i = 0; i < layers_.size(); ++i) {
            check_cache(self_k[i], 28, "self_k");
            check_cache(self_v[i], 28, "self_v");
            check_cache(cross_k[i], 94, "cross_k");
            check_cache(cross_v[i], 94, "cross_v");
        }
        bind_addresses(hidden, output, self_k, self_v, cross_k, cross_v, prefill);
        run(hidden, output, self_k, self_v, cross_k, cross_v, position, seq);
        expected_position_ += seq;
        return output;
    }

private:
    void bind_addresses(
        const at::Tensor& hidden, const at::Tensor& output,
        at::TensorList self_k, at::TensorList self_v,
        at::TensorList cross_k, at::TensorList cross_v, bool prefill) {
        const void*& input_ptr = prefill ? prefill_input_ptr_ : decode_input_ptr_;
        const void*& output_ptr = prefill ? prefill_output_ptr_ : decode_output_ptr_;
        TORCH_CHECK(input_ptr == nullptr || input_ptr == hidden.data_ptr(),
                    "Whisper decoder Graph requires its retained input staging tensor");
        TORCH_CHECK(output_ptr == nullptr || output_ptr == output.data_ptr(),
                    "Whisper decoder Graph requires its retained output staging tensor");
        input_ptr = hidden.data_ptr();
        output_ptr = output.data_ptr();
        std::vector<const void*> live;
        for (const auto& list : {self_k, self_v, cross_k, cross_v})
            for (const auto& value : list) live.push_back(value.data_ptr());
        TORCH_CHECK(cache_ptrs_.empty() || cache_ptrs_ == live,
                    "Whisper decoder caches changed after Graph capture");
        if (cache_ptrs_.empty()) cache_ptrs_ = std::move(live);
        rpu_ddr_flush_force(hidden.data_ptr());
    }

    void run(
        const at::Tensor& hidden, const at::Tensor& output,
        at::TensorList self_k, at::TensorList self_v,
        at::TensorList cross_k, at::TensorList cross_v,
        int64_t position, int64_t seq) {
        if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
        SpmAllocator::ScopedTemporary temporary;
        auto alloc = [](size_t bytes) {
            return SPM_ALLOC.alloc_temporary((bytes + 255) / 256 * 256);
        };
        const int64_t full = seq * WHISPER_DECODER_HIDDEN;
        const int64_t local_q = seq * WHISPER_DECODER_HEAD_DIM;
        const int64_t local_ff = seq *
            (WHISPER_DECODER_INTERMEDIATE / WHISPER_DECODER_CORES);
        const uint32_t residual_a = alloc(aligned_bytes(full));
        const uint32_t residual_b = alloc(aligned_bytes(full));
        const uint32_t normalized = alloc(aligned_bytes(full));
        const uint32_t partial = alloc(aligned_bytes(full));
        const uint32_t q = alloc(aligned_bytes(local_q));
        const uint32_t k = alloc(aligned_bytes(local_q));
        const uint32_t v = alloc(aligned_bytes(local_q));
        const uint32_t attention = alloc(aligned_bytes(local_q));
        const uint32_t ff = alloc(aligned_bytes(local_ff));
        const uint32_t sdpa_tmp = alloc(WHISPER_DECODER_SDPA_TMP_BYTES);
        const uint32_t norm_w = alloc(aligned_bytes(WHISPER_DECODER_HIDDEN));
        const uint32_t norm_b = alloc(aligned_bytes(WHISPER_DECODER_HIDDEN));
        const uint32_t col_bias = alloc(aligned_bytes(
            WHISPER_DECODER_INTERMEDIATE / WHISPER_DECODER_CORES));
        const uint32_t row_bias = alloc(aligned_bytes(WHISPER_DECODER_HIDDEN));
        auto address = [](uint32_t offset) { return SPM_ALLOC.addr(0, offset); };
        rpu_launch_ddr_broadcast_spm_dma(
            hidden, 0, full, address(residual_a), WHISPER_DECODER_CORES);

        uint32_t current = residual_a;
        uint32_t alternate = residual_b;
        const int64_t kv_len = position + seq;
        const double scale = 1.0 / std::sqrt(
            static_cast<double>(WHISPER_DECODER_HEAD_DIM));
        for (size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
            const Layer& layer = layers_[layer_index];
            layer_norm(current, normalized, norm_w, norm_b,
                       layer.n1_w, layer.n1_b, seq);
            col_projection(normalized, q, col_bias, layer.sq_w, layer.sq_b,
                           seq, WHISPER_DECODER_HIDDEN);
            zero_col_projection(normalized, k, col_bias, layer.sk_w,
                                seq, WHISPER_DECODER_HIDDEN);
            col_projection(normalized, v, col_bias, layer.sv_w, layer.sv_b,
                           seq, WHISPER_DECODER_HIDDEN);
            const KvInsertSegmentPlan plan = rpu_resolve_kvinsert_segment_plan(
                position, seq, seq, WHISPER_DECODER_CORES,
                WHISPER_DECODER_HEADS, WHISPER_DECODER_HEAD_DIM,
                WHISPER_DECODER_KV_CAPABILITIES, KvInsertRoute::V2);
            auto sk = self_k[layer_index];
            auto sv = self_v[layer_index];
            rpu_launch_insert_kvcache_spm_unified_with_plan(
                sk, sv, k, v, WHISPER_DECODER_HEADS,
                WHISPER_DECODER_HEAD_DIM, WHISPER_DECODER_CORES,
                0, 0, 0, plan);
            rpu_launch_sdpa_spm_unified_kernel_v2(
                sk, sv, seq > 1 ? 1 : 0, scale, q, attention, sdpa_tmp, 0,
                seq, WHISPER_DECODER_HEADS, WHISPER_DECODER_HEADS,
                WHISPER_DECODER_HEAD_DIM, kv_len,
                WHISPER_DECODER_CORES, WHISPER_DECODER_CORES);
            row_projection(attention, partial, row_bias, layer.so_w,
                           layer.so_b, seq, WHISPER_DECODER_HIDDEN);
            rpu_launch_all_reduce_sum_residual_kernel(
                address(partial), address(current), address(alternate),
                seq, WHISPER_DECODER_HIDDEN,
                WHISPER_DECODER_CORES, WHISPER_DECODER_CORES);

            layer_norm(alternate, normalized, norm_w, norm_b,
                       layer.n2_w, layer.n2_b, seq);
            col_projection(normalized, q, col_bias, layer.cq_w, layer.cq_b,
                           seq, WHISPER_DECODER_HIDDEN);
            auto ck = cross_k[layer_index];
            auto cv = cross_v[layer_index];
            rpu_launch_sdpa_spm_unified_kernel_v2(
                ck, cv, 0, scale, q, attention, sdpa_tmp, 0,
                seq, WHISPER_DECODER_HEADS, WHISPER_DECODER_HEADS,
                WHISPER_DECODER_HEAD_DIM, WHISPER_DECODER_CROSS_TOKENS,
                WHISPER_DECODER_CORES, WHISPER_DECODER_CORES);
            row_projection(attention, partial, row_bias, layer.co_w,
                           layer.co_b, seq, WHISPER_DECODER_HIDDEN);
            rpu_launch_all_reduce_sum_residual_kernel(
                address(partial), address(alternate), address(current),
                seq, WHISPER_DECODER_HIDDEN,
                WHISPER_DECODER_CORES, WHISPER_DECODER_CORES);

            layer_norm(current, normalized, norm_w, norm_b,
                       layer.n3_w, layer.n3_b, seq);
            col_projection(normalized, ff, col_bias, layer.ff1_w,
                           layer.ff1_b, seq, WHISPER_DECODER_INTERMEDIATE);
            rpu_launch_eltwise_unary_spm_kernel(
                address(ff), address(ff), local_ff,
                ValuOpType::ADD, GeluMode::ERF, WHISPER_DECODER_CORES);
            row_projection(ff, partial, row_bias, layer.ff2_w,
                           layer.ff2_b, seq, WHISPER_DECODER_INTERMEDIATE);
            rpu_launch_all_reduce_sum_residual_kernel(
                address(partial), address(current), address(alternate),
                seq, WHISPER_DECODER_HIDDEN,
                WHISPER_DECODER_CORES, WHISPER_DECODER_CORES);
            current = alternate;
            alternate = current == residual_a ? residual_b : residual_a;
        }
        rpu_launch_spm_copy_ddr_dma(
            address(current), output.data_ptr<c10::Half>(), full);
    }

    static void layer_norm(
        uint32_t input, uint32_t output, uint32_t gamma, uint32_t beta,
        const at::Tensor& gamma_value, const at::Tensor& beta_value,
        int64_t seq) {
        rpu_launch_ddr_broadcast_spm_dma(
            gamma_value.data_ptr<c10::Half>(), WHISPER_DECODER_HIDDEN,
            SPM_ALLOC.addr(0, gamma), WHISPER_DECODER_CORES);
        rpu_launch_ddr_broadcast_spm_dma(
            beta_value.data_ptr<c10::Half>(), WHISPER_DECODER_HIDDEN,
            SPM_ALLOC.addr(0, beta), WHISPER_DECODER_CORES);
        rpu_launch_layernorm_spm_kernel(
            SPM_ALLOC.addr(0, input), SPM_ALLOC.addr(0, output),
            SPM_ALLOC.addr(0, gamma), SPM_ALLOC.addr(0, beta),
            seq, WHISPER_DECODER_HIDDEN, WHISPER_DECODER_EPS,
            false, 0, WHISPER_DECODER_CORES);
    }

    static void col_projection(
        uint32_t input, uint32_t output, uint32_t bias,
        const at::Tensor& weight, const at::Tensor& bias_value,
        int64_t seq, int64_t out_features) {
        const int64_t local = out_features / WHISPER_DECODER_CORES;
        rpu_launch_ddr_scatter_spm_dma(
            bias_value.data_ptr<c10::Half>(), local,
            local * WHISPER_DECODER_DWIDTH, SPM_ALLOC.addr(0, bias),
            WHISPER_DECODER_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            SPM_ALLOC.addr(0, input), weight, SPM_ALLOC.addr(0, output),
            seq, out_features, WHISPER_DECODER_HIDDEN, 1,
            WHISPER_DECODER_CORES, SPM_ALLOC.addr(0, bias));
    }

    static void zero_col_projection(
        uint32_t input, uint32_t output, uint32_t bias,
        const at::Tensor& weight, int64_t seq, int64_t out_features) {
        rpu_launch_memset_spm_multicore(
            SPM_ALLOC.addr(0, bias), out_features / WHISPER_DECODER_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            SPM_ALLOC.addr(0, input), weight, SPM_ALLOC.addr(0, output),
            seq, out_features, WHISPER_DECODER_HIDDEN, 1,
            WHISPER_DECODER_CORES, SPM_ALLOC.addr(0, bias));
    }

    static void row_projection(
        uint32_t input, uint32_t output, uint32_t bias,
        const at::Tensor& weight, const at::Tensor& bias_value,
        int64_t seq, int64_t in_features) {
        rpu_launch_memset_spm_multicore(
            SPM_ALLOC.addr(0, bias), WHISPER_DECODER_HIDDEN);
        rpu_launch_ddr_broadcast_spm_dma(
            bias_value.data_ptr<c10::Half>(), WHISPER_DECODER_HIDDEN,
            SPM_ALLOC.addr(0, bias), 1);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            SPM_ALLOC.addr(0, input), weight, SPM_ALLOC.addr(0, output),
            seq, WHISPER_DECODER_HIDDEN, in_features, 0,
            WHISPER_DECODER_CORES, SPM_ALLOC.addr(0, bias));
    }

    std::vector<Layer> layers_;
    int64_t expected_position_ = 0;
    const void* prefill_input_ptr_ = nullptr;
    const void* prefill_output_ptr_ = nullptr;
    const void* decode_input_ptr_ = nullptr;
    const void* decode_output_ptr_ = nullptr;
    std::vector<const void*> cache_ptrs_;
};

using WhisperDecoderRegistry = ModelHandleRegistry<WhisperDecoderModel>;

}  // namespace

int64_t rpu_whisper_decoder_create() {
    return WhisperDecoderRegistry::create();
}

void rpu_whisper_decoder_destroy(int64_t handle) {
    WhisperDecoderRegistry::destroy(handle, "rpu_whisper_decoder_destroy");
}

void rpu_whisper_decoder_set_weights(
    int64_t handle,
    at::TensorList sq_w, at::TensorList sk_w,
    at::TensorList sv_w, at::TensorList so_w,
    at::TensorList cq_w, at::TensorList co_w,
    at::TensorList ff1_w, at::TensorList ff2_w,
    at::TensorList n1_w, at::TensorList n1_b,
    at::TensorList n2_w, at::TensorList n2_b,
    at::TensorList n3_w, at::TensorList n3_b,
    at::TensorList sq_b, at::TensorList sv_b,
    at::TensorList so_b, at::TensorList cq_b,
    at::TensorList co_b, at::TensorList ff1_b,
    at::TensorList ff2_b) {
    WhisperDecoderRegistry::get(handle, "rpu_whisper_decoder_set_weights")
        ->set_weights(
            sq_w, sk_w, sv_w, so_w, cq_w, co_w, ff1_w, ff2_w,
            n1_w, n1_b, n2_w, n2_b, n3_w, n3_b,
            sq_b, sv_b, so_b, cq_b, co_b, ff1_b, ff2_b);
}

void rpu_whisper_decoder_reset(int64_t handle, int64_t position) {
    WhisperDecoderRegistry::get(handle, "rpu_whisper_decoder_reset")
        ->reset(position);
}

at::Tensor rpu_whisper_decoder_prefill(
    int64_t handle, const at::Tensor& hidden, const at::Tensor& output,
    at::TensorList self_k, at::TensorList self_v,
    at::TensorList cross_k, at::TensorList cross_v) {
    return WhisperDecoderRegistry::get(handle, "rpu_whisper_decoder_prefill")
        ->forward(hidden, output, self_k, self_v, cross_k, cross_v,
                  /*position=*/0, /*prefill=*/true);
}

at::Tensor rpu_whisper_decoder_decode(
    int64_t handle, const at::Tensor& hidden, const at::Tensor& output,
    at::TensorList self_k, at::TensorList self_v,
    at::TensorList cross_k, at::TensorList cross_v, int64_t position) {
    return WhisperDecoderRegistry::get(handle, "rpu_whisper_decoder_decode")
        ->forward(hidden, output, self_k, self_v, cross_k, cross_v,
                  position, /*prefill=*/false);
}

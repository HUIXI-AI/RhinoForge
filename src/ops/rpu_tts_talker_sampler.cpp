#include "tts_talker_sampler.h"
#include "core/rpu_spm_allocator.h"
#include "graph/graph_runtime.h"
#include "tts_sampling_assets.h"
#include <rhino_launch_program.h>
#include <array>

namespace {
constexpr int kVocab = 3072;
constexpr int kMaxFrames = 256;
constexpr size_t kRowBytes = kVocab * sizeof(c10::Half);
constexpr uint32_t kNoise = 0, kScore = 0x1800, kToken = 0x3000, kWork = 0x3020;
constexpr uint32_t kSeen = 0x184000;
constexpr uint32_t kAllow = 0x187000, kLogits = 0x190000, kScaled = 0x192000;
constexpr uint32_t kIndices = 0x194000, kTop = 0x196000, kTopIds = 0x196080;
constexpr uint32_t kMask = 0x197000, kMasked = 0x199000, kCentered = 0x19b000;
constexpr uint32_t kSanitized = 0x19d000, kAllowed = 0x1a0000, kValid = 0x1a2000;
constexpr uint32_t kOneHot = 0x1a4000, kMerged = 0x1a6000, kAllocation = 0x1a8000;

void check(int status) {
    TORCH_CHECK(status == 0, "TTS Talker sampling Launch status ", status);
}

struct TensorBuffer {
    rhino_lkn::HostDDR_t* root;
    size_t offset;
};

TensorBuffer buffer(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_contiguous(), "TTS Talker sampling requires contiguous storage");
    auto* root = rhino_lkn::RpuGetHostddr(tensor.data_ptr());
    const auto pointer = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(root && pointer >= reinterpret_cast<uintptr_t>(root->get_cpu_ptr()),
                "TTS Talker sampling requires a managed tensor");
    const size_t offset = pointer - reinterpret_cast<uintptr_t>(root->get_cpu_ptr());
    TORCH_CHECK(offset <= root->get_memory_size() &&
                    size_t(tensor.numel() * tensor.element_size()) <= root->get_memory_size() - offset,
                "TTS Talker sampling tensor exceeds managed buffer");
    return {root, offset};
}
}

struct TtsTalkerSampler::Impl {
    using Program = rhino_lkn::Program_t;
    std::array<Program, 20> programs;
    uint16_t params[20][68], scm[20][256];
    at::Tensor init, noise, output, seen;
    int frames = 0, index = 0, pending = 0;
    uint64_t logits_base = 0, noise_base = 0;
    const char* names[20] = {"binary_scalar", "binary_scalar", "binary_sameshape",
        "where_sameshape", "binary_scalar", "where_sameshape", "binary_sameshape",
        "where_sameshape", "where_sameshape", "topk_by_kvsort_fp16",
        "binary_Nx1_NxC_v16_batch_bopa", "binary_scalar", "topk_by_kvsort_fp16",
        "binary_Nx1_NxC_v16_batch", "where_sameshape", "softmax_c256_gauto",
        "binary_sameshape", "argmax_reduceC_tileN", "one_hot_last_dim", "binary_sameshape"};

    explicit Impl(const std::string& path) {
        tts_sampling_assets::read(path, tts_sampling_assets::TableSet::Talker,
                                  params, sizeof(params), scm, sizeof(scm));
        for (int i = 0; i < 20; ++i)
            check(programs[i].create_with_binary_file(
                (path + "/tts_sampling_unified.ref").c_str(), names[i]));
    }
};

TtsTalkerSampler::TtsTalkerSampler(std::string path)
    : impl_(std::make_unique<Impl>(path)) {
    TORCH_CHECK(!path.empty(), "TTS Talker sampling requires explicit experimental assets");
}

TtsTalkerSampler::~TtsTalkerSampler() = default;

at::Tensor TtsTalkerSampler::reset_impl(
        int64_t frames, const at::Tensor& reference, bool interleaved) {
    TORCH_CHECK(!RpuKernelGraph::has_active() && frames >= 2 && frames <= kMaxFrames &&
                    reference.device().is_privateuseone(),
                "invalid TTS Talker sampling reset profile");
    auto& state = *impl_;
    if (!state.init.defined()) {
        auto host = at::empty({11264}, at::TensorOptions().dtype(at::kShort));
        auto* values = reinterpret_cast<uint16_t*>(host.data_ptr<int16_t>());
        std::fill(values, values + 4096, uint16_t(0xfc00));
        for (int i = 0; i < 4096; ++i) values[4096 + i] = uint16_t(i);
        std::fill(values + 8192, values + 11264, uint16_t(0));
        std::fill(values + 8192, values + 8192 + 2048, uint16_t(0x3c00));
        values[8192 + 2150] = uint16_t(0x3c00);
        state.init = host.to(reference.device());
        state.noise = at::empty({kMaxFrames, kVocab}, reference.options().dtype(at::kHalf));
        state.output = at::empty({kVocab + 16}, reference.options().dtype(at::kHalf));
        state.seen = at::empty({kVocab}, reference.options().dtype(at::kHalf));
    }
    auto host_noise = at::empty({frames, kVocab}, at::TensorOptions().dtype(at::kFloat));
    at::Tensor predictor_noise;
    if (interleaved) {
        predictor_noise = at::empty({frames, 15, 2048},
                                    at::TensorOptions().dtype(at::kFloat));
        host_noise.narrow(0, 0, 1).exponential_(1);
        for (int64_t index = 0; index < frames; ++index) {
            predictor_noise.select(0, index).exponential_(1);
            if (index + 1 < frames) host_noise.narrow(0, index + 1, 1).exponential_(1);
        }
    } else {
        host_noise.exponential_(1);
    }
    auto noise = host_noise
                     .clamp_min(0x1p-14).to(at::kHalf).contiguous();
    TORCH_CHECK(at::isfinite(noise).all().item<bool>(), "invalid TTS Talker sampling noise");
    state.noise.narrow(0, 0, frames).copy_(noise);
    state.seen.copy_(at::zeros({kVocab}, at::TensorOptions().dtype(at::kHalf)));
    state.frames = int(frames);
    state.index = state.pending = 0;
    if (interleaved)
        return predictor_noise;
    return {};
}

void TtsTalkerSampler::reset(int64_t frames, const at::Tensor& reference) {
    reset_impl(frames, reference, false);
}

at::Tensor TtsTalkerSampler::reset_interleaved(
        int64_t frames, const at::Tensor& reference) {
    return reset_impl(frames, reference, true);
}

void TtsTalkerSampler::sample(const at::Tensor& logits) {
    using namespace rhino_lkn;
    auto& state = *impl_;
    TORCH_CHECK(RpuKernelGraph::has_active() && state.frames && !state.pending &&
                    state.index < state.frames && logits.device().is_privateuseone() &&
                    logits.scalar_type() == at::kHalf && logits.is_contiguous() &&
                    logits.sizes().equals({1, 1, kVocab}),
                "invalid TTS Talker sampling frame");
    const auto logits_buffer = buffer(logits);
    const auto init_buffer = buffer(state.init), noise_buffer = buffer(state.noise);
    const auto seen_buffer = buffer(state.seen);
    const auto output_buffer = buffer(state.output);
    auto& allocator = SpmAllocator::instance();
    TORCH_CHECK(allocator.is_initialized() && allocator.addr(0, 0) == 0x40000000,
                "TTS Talker sampling requires the core-0 shared SPM root");
    SpmAllocator::ScopedTemporary scope;
    const uint32_t base = allocator.alloc_temporary(kAllocation);
    auto& spm = *allocator.root_buffer(0);
    auto& graph = RpuKernelGraph::active();
    auto fixed = [&](auto& src, size_t src_offset, auto& dst, size_t dst_offset, size_t bytes) {
        graph.record_dma_fixed(src.get_rpu_addr() + src_offset,
                               dst.get_rpu_addr() + dst_offset, bytes, 0);
    };
    fixed(*init_buffer.root, init_buffer.offset, spm, base + kLogits, 8192);
    state.logits_base = logits_buffer.root->get_rpu_addr() + logits_buffer.offset;
    graph.record_dma_mutable_src(&state.logits_base, 0, spm.get_rpu_addr() + base + kLogits,
                                 kRowBytes, 0);
    fixed(*init_buffer.root, init_buffer.offset + 8192, spm, base + kIndices, 8192);
    fixed(*init_buffer.root, init_buffer.offset + 16384, spm, base + kAllow, kRowBytes);
    fixed(*seen_buffer.root, seen_buffer.offset, spm, base + kSeen, kRowBytes);
    fixed(*init_buffer.root, init_buffer.offset, spm, base + kAllowed, 8192);
    fixed(*init_buffer.root, init_buffer.offset, spm, base + kCentered, 8192);
    state.noise_base = noise_buffer.root->get_rpu_addr() + noise_buffer.offset + state.index * kRowBytes;
    graph.record_dma_mutable_src(&state.noise_base, 0, spm.get_rpu_addr() + base + kNoise,
                                 kRowBytes, 0);
    const std::array<std::vector<std::pair<uint32_t, uint32_t>>, 20> offsets = {{
        {{4,kLogits},{8,kMask}}, {{4,kLogits},{8,kMasked}},
        {{4,kMask},{6,kMasked},{8,kValid}}, {{6,kValid},{8,kLogits},{12,kSanitized}},
        {{4,kSanitized},{8,kMask}}, {{6,kMask},{12,kMasked}},
        {{4,kSanitized},{6,kMasked},{8,kCentered}},
        {{6,kSeen},{8,kCentered},{10,kSanitized},{12,kLogits}},
        {{6,kAllow},{8,kLogits},{12,kAllowed}},
        {{0,kAllowed},{2,kIndices},{4,kTop},{6,kTopIds}},
        {{4,kTop},{6,kAllowed},{8,kCentered}}, {{4,kCentered},{8,kLogits}},
        {{0,kLogits},{2,kIndices},{4,kTop},{6,kTopIds}},
        {{4,kTop + 49 * 2},{6,kLogits},{8,kMask}},
        {{6,kMask},{8,kLogits},{12,kMasked}}, {{0,kMasked},{2,kCentered}},
        {{4,kCentered},{6,kNoise},{8,kScore}}, {{4,kScore},{6,kToken},{22,kWork}},
        {{0,kToken},{2,kOneHot}}, {{4,kSeen},{6,kOneHot},{8,kMerged}}
    }};
    for (int i = 0; i < 20; ++i) {
        auto* kernel = graph.get_kernel_reset(state.names[i], state.programs[i]);
        for (uint32_t j = 0; j < 68; ++j) check(kernel->set_regs(j, state.params[i][j]));
        for (uint32_t j = 0; j < 256; j += 2)
            check(kernel->set_regs(4096 + j, uint32_t(state.scm[i][j]) |
                (uint32_t(state.scm[i][j + 1]) << 16)));
        const bool words32 = i == 9 || i == 12 || i == 15;
        for (auto [reg, offset] : offsets[i])
            check(kernel->set_regs(reg, uint32_t((base + offset) / (words32 ? 32 : 1))));
        graph.enqueue(*kernel, {1, 1, 1}, {0});
    }
    fixed(spm, base + kValid, *output_buffer.root, output_buffer.offset, kRowBytes);
    fixed(spm, base + kToken, *output_buffer.root, output_buffer.offset + kRowBytes, 32);
    fixed(spm, base + kMerged, *seen_buffer.root, seen_buffer.offset, kRowBytes);
    graph.keep_alive(logits);
    graph.keep_alive(state.init);
    graph.keep_alive(state.noise);
    graph.keep_alive(state.output);
    graph.keep_alive(state.seen);
    state.pending = 1;
}

at::Tensor TtsTalkerSampler::result() {
    auto& state = *impl_;
    TORCH_CHECK(!RpuKernelGraph::has_active() && state.pending, "TTS Talker sample is not ready");
    auto cpu = state.output.cpu();
    TORCH_CHECK(cpu.narrow(0, 0, kVocab).eq(1).all().item<bool>(),
                "TTS Talker sampling logits outside validated envelope");
    const auto* bytes = reinterpret_cast<const uint8_t*>(cpu.data_ptr<c10::Half>());
    const int32_t token = *reinterpret_cast<const int32_t*>(bytes + kRowBytes);
    TORCH_CHECK((token >= 0 && token < 2048) || token == 2150,
                "TTS Talker sampled token outside allowed vocabulary");
    ++state.index;
    state.pending = 0;
    return at::full({1, 1}, token, at::TensorOptions().dtype(at::kLong));
}

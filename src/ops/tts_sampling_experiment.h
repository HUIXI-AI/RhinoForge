#pragma once
#include "core/rpu_spm_allocator.h"
#include "tts_sampling_assets.h"
#include <rhino_launch_queue.h>
#include <rhino_launch_program.h>
#include <array>
#include <cstring>

static at::Tensor tts_sampling_checked_result(const at::Tensor& valid, const at::Tensor& ids) {
    TORCH_CHECK(valid.device().is_cpu() && valid.scalar_type() == at::kHalf &&
                    valid.sizes().equals({15, 2048}) && ids.device().is_cpu() &&
                    ids.scalar_type() == at::kInt && ids.sizes().equals({15, 8}),
                "invalid TTS sampling result profile");
    TORCH_CHECK(valid.eq(1).all().item<bool>(), "TTS sampling logits outside validated envelope");
    auto result = ids.select(1, 0).to(at::kLong).reshape({1, 15}).contiguous();
    TORCH_CHECK(result.ge(0).all().item<bool>() && result.lt(2048).all().item<bool>(),
                "TTS sampled index outside vocabulary");
    return result;
}

static at::Tensor tts_sampling_checked_packed_result(const at::Tensor& packed) {
    TORCH_CHECK(packed.device().is_cpu() && packed.scalar_type() == at::kHalf &&
                    packed.sizes().equals({15, 2064}), "invalid packed TTS sampling result");
    TORCH_CHECK(packed.narrow(1, 0, 2048).eq(1).all().item<bool>(),
                "TTS sampling logits outside validated envelope");
    auto result = at::empty({1, 15}, at::TensorOptions().dtype(at::kLong));
    const auto* bytes = reinterpret_cast<const uint8_t*>(packed.data_ptr<c10::Half>());
    auto* tokens = result.data_ptr<int64_t>();
    for (int index = 0; index < 15; ++index)
        tokens[index] = *reinterpret_cast<const int32_t*>(bytes + index * 4128 + 4096);
    TORCH_CHECK(result.ge(0).all().item<bool>() && result.lt(2048).all().item<bool>(),
                "TTS sampled index outside vocabulary");
    return result;
}

// Experimental host wrapper; operator assets remain outside the repository.
class TtsSamplingExperiment {
    using Kernel = rhino_lkn::Kernel_t;
    std::array<rhino_lkn::Program_t, 14> programs_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
    rhino_lkn::HostDDR_t init_{16384, rhino_lkn::read_write, rhino_lkn::kStride32B, 256};
    rhino_lkn::HostDDR_t noise_{15 * 4096, rhino_lkn::read_write, rhino_lkn::kStride32B, 256};
    at::Tensor packed_result_;
    at::Tensor graph_init_, graph_noise_, reserved_noise_;
    bool reserved_mode_ = false;
    int reserved_frames_ = 0, reserved_index_ = 0;
    int next_index_ = 0;
    std::array<uint64_t, 15 * 4> dma_bases_{};
    std::array<uint64_t, 15> reserved_noise_bases_{};
    uint16_t params_[14][68], scm_[14][256];
    const char* names_[14] = {"binary_scalar", "binary_scalar", "binary_sameshape", "where_sameshape",
        "topk_by_kvsort_fp16", "binary_Nx1_NxC_v16_batch_bopa",
        "binary_scalar", "topk_by_kvsort_fp16", "binary_Nx1_NxC_v16_batch",
        "where_sameshape", "softmax_c16_gauto", "binary_sameshape", "argmax_reduceC_tileN", "gather_scalar"};
    static void check(int status) { TORCH_CHECK(status == 0, "TTS sampling Launch status ", status); }
    struct TensorBuffer { rhino_lkn::HostDDR_t* root; size_t offset; };
    static TensorBuffer buffer(const at::Tensor& tensor) {
        TORCH_CHECK(tensor.is_contiguous(), "TTS sampling requires contiguous storage");
        auto* b = rhino_lkn::RpuGetHostddr(tensor.data_ptr());
        const auto ptr = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        TORCH_CHECK(b && ptr >= reinterpret_cast<uintptr_t>(b->get_cpu_ptr()),
                    "TTS sampling requires a managed tensor");
        const size_t offset = ptr - reinterpret_cast<uintptr_t>(b->get_cpu_ptr());
        TORCH_CHECK(offset <= b->get_memory_size() &&
                    size_t(tensor.numel() * tensor.element_size()) <= b->get_memory_size() - offset,
                    "TTS sampling tensor exceeds managed buffer");
        return {b, offset};
    }
public:
    void prepare_graph(const at::TensorOptions& options) {
        auto init = at::from_blob(init_.get_cpu_ptr(), {8192}, at::TensorOptions().dtype(at::kShort));
        auto noise = at::from_blob(noise_.get_cpu_ptr(), {15 * 2048}, at::TensorOptions().dtype(at::kShort));
        if (!graph_init_.defined()) {
            graph_init_ = init.to(options.dtype(at::kShort));
        }
        if (!reserved_mode_ && !graph_noise_.defined()) {
            graph_noise_ = noise.to(options.dtype(at::kShort));
        } else if (!reserved_mode_) {
            graph_noise_.copy_(noise);
        }
    }
    explicit TtsSamplingExperiment(const std::string& path) {
        tts_sampling_assets::read(path, tts_sampling_assets::TableSet::Predictor,
                                  params_, sizeof(params_), scm_, sizeof(scm_));
        for (int i = 0; i < 14; ++i) {
            check(programs_[i].create_with_binary_file((path + "/tts_sampling_unified.ref").c_str(), names_[i]));
            auto k = std::make_unique<Kernel>(programs_[i], names_[i]);
            for (uint32_t j = 0; j < 68; ++j) check(k->set_regs(j, params_[i][j]));
            for (uint32_t j = 0; j < 256; j += 2)
                check(k->set_regs(4096 + j, uint32_t(scm_[i][j]) | (uint32_t(scm_[i][j + 1]) << 16)));
            kernels_.push_back(std::move(k));
        }
        auto* init = static_cast<uint16_t*>(init_.get_cpu_ptr());
        for (int i = 0; i < 4096; ++i) { init[i] = 0xfc00; init[4096 + i] = i; }
        check(init_.flush());
    }
    void reset() {
        next_index_ = 0;
        if (reserved_mode_) {
            TORCH_CHECK(reserved_index_ < reserved_frames_, "TTS sampling reservation exhausted");
            return;
        }
        // shortcut: bounded FP16 noise changes sampling; not a default production path.
        auto noise = at::empty({15, 2048}, at::TensorOptions().dtype(at::kFloat)).exponential_(1);
        noise = noise.clamp_min(0x1p-14).to(at::kHalf).contiguous();
        TORCH_CHECK(at::isfinite(noise).all().item<bool>(), "invalid TTS sampling noise");
        std::memcpy(noise_.get_cpu_ptr(), noise.data_ptr(), 15 * 4096);
        check(noise_.flush());
    }
    void reserve(int frames, const at::Tensor& reference) {
        auto noise = at::empty({frames, 15, 2048}, at::TensorOptions().dtype(at::kFloat))
                         .exponential_(1);
        reserve_noise(frames, reference, noise);
    }
    void reserve_noise(int frames, const at::Tensor& reference, const at::Tensor& source) {
        TORCH_CHECK(!RpuKernelGraph::has_active() && frames >= 2 && frames <= 256 &&
                        reference.device().is_privateuseone(),
                    "invalid TTS sampling reservation profile");
        TORCH_CHECK(source.device().is_cpu() && source.scalar_type() == at::kFloat &&
                        source.is_contiguous() && source.sizes().equals({frames, 15, 2048}) &&
                        at::isfinite(source).all().item<bool>() && source.gt(0).all().item<bool>(),
                    "invalid reserved TTS sampling noise");
        auto noise = source.clamp_min(0x1p-14).to(at::kHalf).contiguous();
        TORCH_CHECK(at::isfinite(noise).all().item<bool>(), "invalid reserved TTS sampling noise");
        if (!reserved_noise_.defined())
            reserved_noise_ = at::empty({256, 15, 2048}, reference.options().dtype(at::kHalf));
        reserved_noise_.narrow(0, 0, frames).copy_(noise);
        reserved_mode_ = true;
        reserved_frames_ = frames;
        reserved_index_ = next_index_ = 0;
    }
    at::Tensor run(const at::Tensor& logits, const at::Tensor& table, int index) {
        using namespace rhino_lkn;
        TORCH_CHECK(index == next_index_ && index >= 0 && index < 15 && logits.sizes().equals({1,1,2048}) &&
                    table.sizes().equals({2048,1024}) &&
                    logits.scalar_type() == at::kHalf && table.scalar_type() == at::kHalf &&
                    logits.device().is_privateuseone() && table.device() == logits.device(),
                    "invalid TTS sampler tensor profile");
        const auto logits_buffer = buffer(logits);
        const auto table_buffer = buffer(table);
        if (!packed_result_.defined())
            packed_result_ = at::empty({15, 2064}, logits.options());
        const auto packed_buffer = buffer(packed_result_);
        auto& alloc = SpmAllocator::instance();
        TORCH_CHECK(alloc.is_initialized() && alloc.addr(0, 0) == 0x40000000,
                    "TTS sampling requires the core-0 shared SPM root");
        SpmAllocator::ScopedTemporary scope;
        const uint32_t base = alloc.alloc_temporary(0x130000);
        auto& spm = *alloc.root_buffer(0);
        const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> offsets = {
            {{4,0x120000},{8,0x12d000}}, {{4,0x120000},{8,0x12e000}},
            {{4,0x12d000},{6,0x12e000},{8,0x12f000}},
            {{6,0x12f000},{8,0x120000},{12,0x12b000}},
            {{0,0x12b000},{2,0x124000},{4,0x126000},{6,0x126080}},
            {{4,0x126000},{6,0x12b000},{8,0x129000}},
            {{4,0x129000},{8,0x122000}}, {{0,0x122000},{2,0x124000},{4,0x126000},{6,0x126080}},
            {{4,0x126062},{6,0x122000},{8,0x127000}}, {{6,0x127000},{8,0x122000},{12,0x128000}},
            {{0,0x128000},{2,0}}, {{4,0},{6,4096},{8,8192}},
            {{4,8192},{6,12288},{22,12320}}, {{2,12288},{4,0x110000}}};
        const auto address = table_buffer.root->get_rpu_addr() + table_buffer.offset;
        TORCH_CHECK(!(address & 255) && (address >> 8) <= UINT32_MAX, "invalid embedding DDR alignment");
        auto output = at::empty({1, 1, 1024}, logits.options());
        const auto output_buffer = buffer(output);
        auto* graph = RpuKernelGraph::has_active() ? &RpuKernelGraph::active() : nullptr;
        std::unique_ptr<Queue_t> queue;
        if (!graph) queue = std::make_unique<Queue_t>(1);
        auto dma = [&](auto& src, size_t src_offset, auto& dst, size_t dst_offset,
                       size_t bytes, int slot = -1, bool mutable_dst = false) {
            if (graph && slot >= 0) {
                auto& base_slot = dma_bases_[index * 4 + slot];
                if (mutable_dst) {
                    base_slot = dst.get_rpu_addr();
                    graph->record_dma_mutable_dst(src.get_rpu_addr() + src_offset,
                                                 &base_slot, dst_offset, bytes, 0);
                } else {
                    base_slot = src.get_rpu_addr();
                    graph->record_dma_mutable_src(&base_slot, src_offset,
                                                 dst.get_rpu_addr() + dst_offset, bytes, 0);
                }
            } else if (graph) graph->record_dma_fixed(src.get_rpu_addr() + src_offset,
                                                     dst.get_rpu_addr() + dst_offset, bytes, 0);
            else check(queue->add_dma_kernel(src, src_offset, dst, dst_offset, bytes));
        };
        dma(*logits_buffer.root, logits_buffer.offset, spm, base + 0x120000, 4096, 0);
        if (graph) {
            TORCH_CHECK(graph_init_.defined() &&
                            (reserved_mode_ || graph_noise_.defined()),
                        "TTS group buffers not prepared");
            auto init = buffer(graph_init_);
            dma(*init.root, init.offset, spm, base + 0x12a000, 4096);
            dma(*init.root, init.offset, spm, base + 0x12c000, 4096);
            dma(*init.root, init.offset + 8192, spm, base + 0x124000, 8192);
            if (reserved_mode_) {
                auto noise = buffer(reserved_noise_);
                auto& base_slot = reserved_noise_bases_[index];
                base_slot = noise.root->get_rpu_addr() + noise.offset +
                            (reserved_index_ * 15 + index) * 4096;
                graph->record_dma_mutable_src(&base_slot, 0,
                                              spm.get_rpu_addr() + base + 4096, 4096, 0);
            } else {
                auto noise = buffer(graph_noise_);
                dma(*noise.root, noise.offset + index * 4096, spm, base + 4096, 4096);
            }
        } else {
            dma(init_, 0, spm, base + 0x12a000, 4096);
            dma(init_, 0, spm, base + 0x12c000, 4096);
            dma(init_, 8192, spm, base + 0x124000, 8192);
            dma(noise_, index * 4096, spm, base + 4096, 4096);
        }
        for (int i = 0; i < 14; ++i) {
            auto* k = graph ? graph->get_kernel_reset(names_[i], programs_[i]) : kernels_[i].get();
            if (graph) {
                for (uint32_t j = 0; j < 68; ++j) check(k->set_regs(j, params_[i][j]));
                for (uint32_t j = 0; j < 256; j += 2)
                    check(k->set_regs(4096 + j, uint32_t(scm_[i][j]) | (uint32_t(scm_[i][j + 1]) << 16)));
            }
            for (auto [reg, offset] : offsets[i])
                check(k->set_regs(reg, uint32_t((base + offset) / ((i == 4 || i == 7 || i == 10) ? 32 : 1))));
            if (i == 13) check(k->set_regs(uint32_t(0), uint32_t(address >> 8)));
            if (graph) graph->enqueue(*k, {1,1,1}, {0});
            else check(queue->add_kernel(*k, {1,1,1}, {0}));
        }
        dma(spm, base + 0x110000, *output_buffer.root, output_buffer.offset, 2048, 1, true);
        dma(spm, base + 12288, *packed_buffer.root,
            packed_buffer.offset + index * 4128 + 4096, 32);
        dma(spm, base + 0x12f000, *packed_buffer.root,
            packed_buffer.offset + index * 4128, 4096);
        if (graph) {
            graph->keep_alive(logits);
            graph->keep_alive(output);
            graph->keep_alive(packed_result_);
            graph->keep_alive(table);
            if (reserved_mode_) graph->keep_alive(reserved_noise_);
        } else {
            check(queue->build_batch());
            check(queue->enqueue_batch(true));
            check(queue->discard_completed_batch());
        }
        ++next_index_;
        return output;
    }
    at::Tensor finish() {
        TORCH_CHECK(next_index_ == 15, "incomplete TTS sampling group");
        // Invalid logits are replaced with zero on-device; never return their generated tokens.
        auto result = tts_sampling_checked_packed_result(packed_result_.cpu());
        if (reserved_mode_) ++reserved_index_;
        return result;
    }
};

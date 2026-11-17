#pragma once
#include "core/rpu_spm_allocator.h"
#include <rhino_launch_queue.h>
#include <rhino_launch_program.h>
#include <array>
#include <cstring>
#include <fstream>

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

// Experimental host wrapper; operator assets remain outside the repository.
class TtsSamplingExperiment {
    using Kernel = rhino_lkn::Kernel_t;
    std::array<rhino_lkn::Program_t, 14> programs_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
    rhino_lkn::HostDDR_t init_{16384, rhino_lkn::read_write, rhino_lkn::kStride32B, 256};
    rhino_lkn::HostDDR_t noise_{15 * 4096, rhino_lkn::read_write, rhino_lkn::kStride32B, 256};
    at::Tensor validity_;
    at::Tensor graph_init_, graph_noise_;
    int next_index_ = 0;
    std::array<uint64_t, 15 * 4> dma_bases_{};
    uint16_t params_[14][68], scm_[14][256];
    const char* names_[14] = {"binary_scalar", "binary_scalar", "binary_sameshape", "where_sameshape",
        "topk_by_kvsort_fp16", "binary_Nx1_NxC_v16_batch_bopa",
        "binary_scalar", "topk_by_kvsort_fp16", "binary_Nx1_NxC_v16_batch",
        "where_sameshape", "softmax_c16_gauto", "binary_sameshape", "argmax_reduceC_tileN", "gather_scalar"};
    static void check(int status) { TORCH_CHECK(status == 0, "TTS sampling Launch status ", status); }
    static void read(const std::string& path, void* dst, size_t size) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        TORCH_CHECK(file && size_t(file.tellg()) == size, "TTS sampling fixture size: ", path);
        file.seekg(0);
        TORCH_CHECK(bool(file.read(static_cast<char*>(dst), size)), "TTS sampling fixture read");
    }
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
            graph_noise_ = noise.to(options.dtype(at::kShort));
        } else {
            graph_noise_.copy_(noise);
        }
    }
    explicit TtsSamplingExperiment(const std::string& path) {
        read(path + "/params.bin", params_, sizeof(params_));
        read(path + "/scm.bin", scm_, sizeof(scm_));
        for (int i = 0; i < 14; ++i) {
            check(programs_[i].create_with_binary_file((path + "/tts_sampling_experiment.ref").c_str(), names_[i]));
            auto k = std::make_unique<Kernel>(programs_[i], names_[i]);
            for (uint32_t j = 0; j < 68; ++j) check(k->set_regs(j, params_[i][j]));
            for (uint32_t j = 0; j < 256; j += 2)
                check(k->set_regs(4096 + j, uint32_t(scm_[i][j]) | (uint32_t(scm_[i][j + 1]) << 16)));
            kernels_.push_back(std::move(k));
        }
        auto* init = static_cast<uint16_t*>(init_.get_cpu_ptr());
        for (int i = 0; i < 4096; ++i) { init[i] = 0xfc00; init[4096 + i] = i; }
        check(init_.flush());
        reset();
    }
    void reset() {
        next_index_ = 0;
        // shortcut: bounded FP16 noise changes sampling; not a default production path.
        auto noise = at::empty({15, 2048}, at::TensorOptions().dtype(at::kFloat)).exponential_(1);
        noise = noise.clamp_min(0x1p-14).to(at::kHalf).contiguous();
        TORCH_CHECK(at::isfinite(noise).all().item<bool>(), "invalid TTS sampling noise");
        std::memcpy(noise_.get_cpu_ptr(), noise.data_ptr(), 15 * 4096);
        check(noise_.flush());
    }
    at::Tensor run(const at::Tensor& logits, const at::Tensor& table, const at::Tensor& ids, int index) {
        using namespace rhino_lkn;
        TORCH_CHECK(index == next_index_ && index >= 0 && index < 15 && logits.sizes().equals({1,1,2048}) &&
                    table.sizes().equals({2048,1024}) && ids.sizes().equals({15,8}) &&
                    logits.scalar_type() == at::kHalf && table.scalar_type() == at::kHalf &&
                    ids.scalar_type() == at::kInt && logits.device().is_privateuseone() &&
                    table.device() == logits.device() && ids.device() == logits.device(),
                    "invalid TTS sampler tensor profile");
        const auto logits_buffer = buffer(logits);
        const auto table_buffer = buffer(table), ids_buffer = buffer(ids);
        if (!validity_.defined()) validity_ = at::empty({15, 2048}, logits.options());
        const auto validity_buffer = buffer(validity_);
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
            TORCH_CHECK(graph_init_.defined() && graph_noise_.defined(), "TTS group buffers not prepared");
            auto init = buffer(graph_init_), noise = buffer(graph_noise_);
            dma(*init.root, init.offset, spm, base + 0x12a000, 4096);
            dma(*init.root, init.offset, spm, base + 0x12c000, 4096);
            dma(*init.root, init.offset + 8192, spm, base + 0x124000, 8192);
            dma(*noise.root, noise.offset + index * 4096, spm, base + 4096, 4096);
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
        dma(spm, base + 12288, *ids_buffer.root, ids_buffer.offset + index * 32, 32, 2, true);
        dma(spm, base + 0x12f000, *validity_buffer.root, validity_buffer.offset + index * 4096, 4096, 3, true);
        if (graph) {
            graph->keep_alive(logits);
            graph->keep_alive(output);
            graph->keep_alive(ids);
            graph->keep_alive(validity_);
            graph->keep_alive(table);
        } else {
            check(queue->build_batch());
            check(queue->enqueue_batch(true));
            check(queue->discard_completed_batch());
        }
        ++next_index_;
        return output;
    }
    at::Tensor finish(const at::Tensor& ids) {
        TORCH_CHECK(next_index_ == 15, "incomplete TTS sampling group");
        // Invalid logits are replaced with zero on-device; never return their generated tokens.
        return tts_sampling_checked_result(validity_.cpu(), ids.cpu());
    }
};

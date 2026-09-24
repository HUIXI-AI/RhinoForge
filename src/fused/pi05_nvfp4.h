#pragma once

#include "fused_model_base.h"
#include "rpu_kernel_decls.h"
#include <array>
#include <cmath>

// Separate from the original Wall NVFP4 ABI. Five immutable FP32 tables, one
// scalar per Action layer, rounded to 32B because the generator loads eight.
namespace v3 {
struct Pi05Nvfp4Tables {
    std::vector<at::Tensor> values;
    inline static constexpr std::array<const char*, 5> names{
        "nvfp4_q_ts", "nvfp4_o_ts", "nvfp4_gate_ts", "nvfp4_up_ts", "nvfp4_down_ts"};
    bool enabled() const { return !values.empty(); }
    void install(at::TensorList tables, int64_t layers, int cores) {
        TORCH_CHECK(tables.empty() || (tables.size() == 5 && layers == 18 && cores == 8),
                    "Pi05 NVFP4 striped-v2 requires five tensor-scale tables, 18 layers and TP8");
        for (const auto& table : tables) {
            TORCH_CHECK(table.defined() && table.scalar_type() == at::kFloat &&
                            table.device().type() == at::kPrivateUse1 && table.is_contiguous() &&
                            table.dim() == 1 && table.numel() == 24,
                        "Pi05 NVFP4 tensor scales must be contiguous FP32 RPU [24]");
            auto cpu = table.cpu();
            const float* p = cpu.data_ptr<float>();
            for (int i = 0; i < layers; ++i)
                TORCH_CHECK(std::isfinite(p[i]) && p[i] > 0,
                            "Pi05 NVFP4 tensor scales must be finite and positive");
        }
        values.assign(tables.begin(), tables.end());
    }
    void declare(std::vector<BufferDecl>& decls, int cores) const {
        for (size_t i = 0; i < values.size(); ++i) {
            const auto tensor = values[i];
            decls.push_back({names[i], 256, 0, 0, StorageClass::Persistent,
                0, nullptr, BufferScope::LayerWide,
                [tensor, cores](FusedModelBase&, int, uint32_t destination) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        reinterpret_cast<c10::Half*>(tensor.data_ptr<float>()),
                        tensor.numel() * 2, destination, cores);
                }});
        }
    }
};

} // namespace v3

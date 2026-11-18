#pragma once

#include "fused_model_base.h"
#include "rpu_kernel_decls.h"
#include <array>
#include <cmath>

namespace v3 {

// Cold dense-text NVFP4 owners. One FP32 tensor scale per projection family
// and global layer; a layer without that family (GDN vs full attention) holds
// 1.0. The generic Linear launcher loads eight scales at a time from the
// family's SPM table, so each table is padded to a multiple of eight. GDN b/a
// stay FP16 and have no table.
struct Qwen35TextNvfp4Tables {
    enum Family : int {
        Q, K, V, O, ATTN_GATE, GDN_Q, GDN_K, GDN_V, GDN_Z, GDN_OUT, GATE, UP, DOWN, COUNT
    };
    std::vector<at::Tensor> values;
    inline static constexpr std::array<const char*, COUNT> names{
        "q35_nvfp4_q_ts", "q35_nvfp4_k_ts", "q35_nvfp4_v_ts", "q35_nvfp4_o_ts",
        "q35_nvfp4_attn_gate_ts", "q35_nvfp4_gdn_q_ts", "q35_nvfp4_gdn_k_ts",
        "q35_nvfp4_gdn_v_ts", "q35_nvfp4_gdn_z_ts", "q35_nvfp4_gdn_out_ts",
        "q35_nvfp4_gate_ts", "q35_nvfp4_up_ts", "q35_nvfp4_down_ts"};

    bool enabled() const { return !values.empty(); }
    static int64_t table_size(int64_t layers) { return (layers + 7) / 8 * 8; }
    void install(at::TensorList tables, int64_t layers) {
        TORCH_CHECK(tables.size() == COUNT && layers > 0 && layers <= 64,
                    "Qwen3.5 text NVFP4 requires thirteen tensor-scale tables for at most 64 layers");
        for (const auto& table : tables) {
            TORCH_CHECK(table.defined() && table.scalar_type() == at::kFloat &&
                            table.device().type() == at::kPrivateUse1 &&
                            table.is_contiguous() && table.dim() == 1 &&
                            table.numel() == table_size(layers),
                        "Qwen3.5 text NVFP4 tensor scales must be contiguous FP32 RPU [layers padded to 8]");
            const auto cpu = table.cpu();
            const auto* p = cpu.data_ptr<float>();
            for (int64_t layer = 0; layer < cpu.numel(); ++layer) {
                const float retired = static_cast<float>(c10::Half(p[layer]));
                TORCH_CHECK(std::isfinite(p[layer]) && p[layer] > 0 &&
                                std::isfinite(retired) && retired > 0,
                            "Qwen3.5 text NVFP4 tensor scale must remain finite and positive in ACC16");
            }
        }
        values.assign(tables.begin(), tables.end());
    }
    void declare(std::vector<BufferDecl>& decls, int cores) const {
        for (size_t family = 0; family < values.size(); ++family) {
            const auto table = values[family];
            decls.push_back({names[family], 256, 0, 0, StorageClass::Persistent,
                0, nullptr, BufferScope::LayerWide,
                [table, cores](FusedModelBase&, int, uint32_t destination) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        reinterpret_cast<c10::Half*>(table.data_ptr<float>()),
                        table.numel() * 2, destination, cores);
                }});
        }
    }
};

} // namespace v3

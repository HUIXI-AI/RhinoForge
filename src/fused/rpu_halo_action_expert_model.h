// rpu_halo_action_expert_model.h — HALO action expert (mode="act") fused 解码器。
//
// 模型背景: HALO 的 action expert 是 Mixture-of-Transformers (MoT) 结构 —
// 同一套 transformer 拓扑, 每层带两套权重: text 行 (cs=18 时第 0/17 行) 走基类
// CausalDecoderModel 的 text 权重, action 行 (第 1..16 行) 走 *_moe_act 孪生
// 权重。本类继承 v3::CausalDecoderModel(Wall-OSS 先例) 复用其全部 prefix-KV
// 非因果 forward(position=prefix_len, is_causal=false, MASK_2D mask)、12/2 GQA
// SDPA(attn_tp=2 自动)、KV insert(prefix append)、RoPE、QKV bias 与 SwiGLU,
// 仅 override declare_buffers / build_layer_subgraph 做双权重计算 + 行掩码合并。
//
// The input contract is host-composed fp16 hidden_states x_emb [1, cs, 1536].
// Boundary projections (action input/output and time embedding) remain on the
// host and are outside this fused decoder.
//
// 数学正确性的关键: 行掩码 merge (out = x_text*text_mask + x_act*act_mask) 与
// 双 GEMM 是逐行独立运算, 所以"先合并 norm 输出再双 GEMM"与逐行选权重等价。
// SDPA 跨行混合 KV, 但两套权重的 K/V/Q 在 SDPA 前已按行合并, 与 HALO 参考实现
// (per-token branch) 一致。
#pragma once

#include "rpu_qwen3_model.h"     // v3::CausalDecoderModel + NUM_CORES/DWIDTH
#include "rpu_kernel_decls.h"
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace v3 {

class HaloActionExpertModel : public CausalDecoderModel {
public:
    HaloActionExpertModel();
    ~HaloActionExpertModel() override;

    // 单层 act 孪生权重 (布局与基类 LayerWeights 主权重逐一对应):
    //   q/k/v/o_w     [out, in]  col/row-partition 预 swizzle fp16 (RPU DDR)
    //   q/k_norm_w    [head_dim] fp16
    //   input/post_norm_w [hidden] fp16
    //   gate/up/down_w [out, in] fp16
    //   q/k/v_bias    [out] fp16 — 与 text 侧 has_qkv_bias_ 同进退
    // 所有权: set_moe_weights 持有 at::Tensor 引用 (DDR keepalive),
    // 跨 forward 稳定; invalidate_model_state 使图缓存重建后 preload 重发。
    struct MoeLayerWeights {
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor q_norm_w, k_norm_w;
        at::Tensor input_norm_w, post_norm_w;
        at::Tensor gate_w, up_w, down_w;
        at::Tensor q_bias, k_bias, v_bias;
    };

    // text 主权重入口: 内部转基类 CausalDecoderModel::set_weights (mrope/
    // deepstack 空), 同时把 per-layer 权重 + cos/sin + eps 留一份子类影子。
    // 影子的原因: 基类把 layer_weights_/cos_/sin_/eps_/prepared_attn_mask_
    // 全部 private, override build_layer_subgraph 取不到; 影子只是 at::Tensor
    // 引用计数副本, 不复制权重数据。本入口是 fp16-only (不带 W8A16 scale 列表)。
    void set_weights_halo(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        at::TensorList input_norm_list, at::TensorList post_norm_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        const at::Tensor& cos, const at::Tensor& sin,
        const at::Tensor& final_norm_w,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps, bool use_silu,
        at::TensorList q_bias_list, at::TensorList k_bias_list,
        at::TensorList v_bias_list);

    // 绑定 act 孪生权重 + 行掩码。必须在基类 set_weights 之后调用。
    //   text_row_mask: [chunk_size] fp16 RPU contig, 取值 {0,1}; 1 = text 行
    //                  (走基类权重), 0 = action 行 (走孪生权重)。本函数在 host
    //                  侧展开成 5 个宽度的常量掩码张量 (act = 1 - text)。
    //   chunk_size:    HALO act 模式的固定 query 长度 (=18); 掩码与 SPM 槽按
    //                  此宽度固化, step_forward 校验 x_emb.size(1) 必须相等。
    void set_moe_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        at::TensorList input_norm_list, at::TensorList post_norm_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        at::TensorList q_bias_list, at::TensorList k_bias_list,
        at::TensorList v_bias_list,
        const at::Tensor& final_norm_moe_w,
        const at::Tensor& text_row_mask,
        int64_t chunk_size);

    // 一次 act step。I/O 契约:
    //   x_emb       [1, chunk_size, hidden=1536] fp16 RPU contig — host 已合成
    //               的混排 hidden (M1 boundary 在 host)。
    //   k/v_caches  基类 7-D swizzled KV cache; 本步在 prefix_len 处追加。
    //   attn_mask_4d 显式 additive mask, 末两维 [chunk_size, prefix+cs]
    //               (基类 sdpa_prepare_mask 降维 + MASK_2D)。
    //   prefix_len  KV prefix 长度 (HALO act: 4706)。
    // 返回: 末隐藏 [1, chunk_size, hidden] fp16 RPU (含双 final_norm) —— 基类
    //   output DMA 的 output_tensor_ (graph 写目标)。调用方须在 capture 作用域
    //   **退出后**读 (此时图已 end() 执行写入); 切勿在作用域内提前 copy —— 那会
    //   读到尚未执行的全 0 output_tensor_（见 FusedModelBase 输出契约）。
    // 所有权: caller 持有输入张量; 返回张量由基类 registry 持有 (跨 forward 稳定)。
    at::Tensor step_forward(
        const at::Tensor& x_emb,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cos,           // per-forward RoPE 表 (CFG 三分支各异)
        const at::Tensor& sin,
        const at::Tensor& attn_mask_4d,
        int64_t prefix_len);

protected:
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override;
    // dynamic_config: 基类已做单 chunk 断言 + 自己的 mask 准备; 这里再准备一份
    // 子类可见的 PreparedMask (基类的是 private)。forward/static_config 继承。
    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override;
    // HALO act 永远单块 (query 18 全块送 MASK_2D 非因果 SDPA)。当前 tiling
    // tile_m=32 → grid=1, gqa=6, product=6，满足严格 keeper；override 只表达
    // 行掩码/KV merge 的单块语义 (cs>=seq_len)，不是绕过 product>8 守卫。
    bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len,
                                   int64_t position) const override;

private:
    // 行掩码 merge: out = text*text_mask + act*act_mask (3 次全尺寸 eltwise)。
    // 前置: text/act 位于不同 SPM 槽; MUL 就地破坏两者, out 可与 text 同槽。
    void emit_row_merge(uint32_t text_addr, uint32_t act_addr, uint32_t out_addr,
                        uint32_t text_mask_addr, uint32_t act_mask_addr,
                        int64_t num_elements, int num_cores);

    std::vector<MoeLayerWeights> text_layers_;  // [num_layers] text 权重影子 (基类副本)
    at::Tensor cos_halo_, sin_halo_;            // RoPE 表影子 (build 期 host 指针)
    double eps_halo_ = 1e-6;                    // RMSNorm eps 影子
    std::vector<MoeLayerWeights> moe_layers_;   // [num_layers] act 孪生权重
    at::Tensor final_norm_moe_w_;               // [hidden] act 侧 final RMSNorm γ
    at::Tensor text_row_mask_;                  // [chunk_size] 原始行掩码 (keepalive)
    // 宽度 -> [chunk_size, width] fp16 RPU 常量掩码 (Persistent 槽 preload 源)。
    std::unordered_map<int64_t, at::Tensor> text_mask_ddr_, act_mask_ddr_;
    int64_t moe_chunk_size_ = 0;                // 掩码/槽固化的 chunk_size (0 = 未设)
    PreparedMask prepared_mask_halo_;           // 每 forward 在 dynamic_config 重备
};

}  // namespace v3

// C API 自由函数原型集中在 src/core/rpu_kernel_decls.h，
// rpu_backend.cpp 经 TORCH_FN 绑定, 不引入本重头文件。

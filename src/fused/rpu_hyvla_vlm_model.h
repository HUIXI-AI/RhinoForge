// rpu_hyvla_vlm_model.h — Hy-Embodied-0.5-VLA 的 VLM prefill fused 解码器。
//
// 模型背景: HunYuanVL 的 VLM 是 Mixture-of-Transformers (MoT) 结构 — 同一套
// transformer 拓扑, 每层带两套权重: text 行走基类 CausalDecoderModel 的 text
// 权重, vision 行走 *_v 孪生权重。Hy-VLA 的 prefix 里两者**高度交错**(147 个
// vision token 分散在 43 段, 每 7 个 vision 夹 1 个 text), 所以只能逐行路由,
// 不能按段切分。本类继承 v3::CausalDecoderModel 复用其 prefix-KV 非因果
// forward(position=prefix_len, is_causal=false, MASK_2D mask)、16/4 GQA
// SDPA(attn_tp=4 自动)、KV insert、RoPE 与 SwiGLU, 仅 override
// declare_buffers / build_layer_subgraph 做双权重计算 + 行掩码合并。
//
// 本文件自 rpu_halo_action_expert_model.* 派生 —— 同为 MoT 双权重 + 常量行掩码,
// 实质手术只有步 3 (见 .cpp)。凡注释提到 HALO 处均为溯源, 数值一律以 Hy-VLA 为准:
// hidden=2048, inter=6144, 32 层, 16/4 heads × hd=128, eps=1e-5, prefix 192
// (= ceil16(181 有效行); 物理 240 里 59 行是 reference padding, mask 整行 False,
// 因果 mask 保证有效行只 attend 有效行 ⇒ 截断数学等价)。
//
// 数学正确性的关键: 行掩码 merge (out = x_text*text_mask + x_vis*vis_mask) 与
// 双 GEMM 是逐行独立运算, 所以"先合并 norm 输出再双 GEMM"与逐行选权重等价。
// SDPA 跨行混合 KV, 但两套权重的 K/V/Q 在 SDPA 前已按行合并, 与参考模型的
// per-token mask_apply 一致。
#pragma once

#include "rpu_qwen3_model.h"     // v3::CausalDecoderModel + NUM_CORES/DWIDTH
#include "rpu_kernel_decls.h"
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

namespace v3 {

class HyVlaVlmModel : public CausalDecoderModel {
public:
    HyVlaVlmModel();
    ~HyVlaVlmModel() override;

    // 单层 vision (*_v) 孪生权重 (布局与基类 LayerWeights 主权重逐一对应):
    //   q/k/v/o_w     [out, in]  col/row-partition 预 swizzle fp16 (RPU DDR)
    //   q/k_norm_w    [head_dim] fp16
    //   input/post_norm_w [hidden] fp16
    //   gate/up/down_w [out, in] fp16
    //   q/k/v_bias    [out] fp16 — 与 text 侧 has_qkv_bias_ 同进退
    // 所有权: set_moe_weights 持有 at::Tensor 引用 (DDR keepalive),
    // 跨 forward 稳定; invalidate_model_state 使图缓存重建后 preload 重发。
    // ⚠️ 相对 HALO 的两点差异（字段仍全部保留，属**刻意的最小改动**）：
    //  1. q_norm_w / k_norm_w 在 Hy-VLA 里**不参与计算** —— query/key_layernorm
    //     由 VLM 与 expert 两塔共享。这两个 norm 仅在 VLM 组参与计算，故
    //     build_layer_subgraph 步 3 只用
    //     基类单份权重，不对它们做双算/merge。调用方给这两项传与 text 相同的张量即可
    //     （会被 preload DMA 但不被读取）；保留字段是为了不动 set_moe_weights 的
    //     签名与校验链，把改动面收敛到层体一处。
    //  2. q/k/v_bias 恒为空 —— Hy-VLA 的 attention_bias=false（基类已按
    //     `bias = (q_bias_list.size() > 0)` 支持可选）。
    struct MoeLayerWeights {
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor q_norm_w, k_norm_w;
        at::Tensor input_norm_w, post_norm_w;
        at::Tensor gate_w, up_w, down_w;
        at::Tensor q_bias, k_bias, v_bias;
        // W8A16: per-output-channel fp16 scale [N]。全 undefined = fp16 路径
        // (发射序列逐字节不变)。**两条孪生各自独立** —— text 与 `_v` 是两组
        // 不同的权重, 各量各的 scale。
        at::Tensor q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws;
    };

    // text 主权重入口: 内部转基类 CausalDecoderModel::set_weights (mrope/
    // deepstack 空), 同时把 per-layer 权重 + cos/sin + eps 留一份子类影子。
    // 影子的原因: 基类把 layer_weights_/cos_/sin_/eps_/prepared_attn_mask_
    // 全部 private, override build_layer_subgraph 取不到; 影子只是 at::Tensor
    // 引用计数副本, 不复制权重数据。W8A16 时七条 scale 列表全给 (要么全空)。
    void set_weights_hyvla(
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
        at::TensorList v_bias_list,
        // W8A16 (可选, text 塔): 7 条 per-output-channel fp16 scale [N]。
        // 全空 = fp16。校验由基类 CausalDecoderModel::set_weights 统一做。
        at::TensorList q_ws_list = {}, at::TensorList k_ws_list = {},
        at::TensorList v_ws_list = {}, at::TensorList o_ws_list = {},
        at::TensorList gate_ws_list = {}, at::TensorList up_ws_list = {},
        at::TensorList down_ws_list = {});

    // 绑定 vision 孪生权重 + 行掩码。必须在基类 set_weights 之后调用。
    //   text_row_mask: [chunk_size] fp16 RPU contig, 取值 {0,1}; 1 = text 行
    //                  (走基类权重), 0 = vision 行 (走 *_v 孪生权重)。本函数在
    //                  host 侧展开成 3 个宽度的常量掩码张量 (vis = 1 - text)。
    //   chunk_size:    固定 prefix 长度 (Hy-VLA UMI: 192 = ceil16(181 有效行));
    //                  掩码与 SPM 槽按此宽度固化, step_forward 校验
    //                  x_emb.size(1) 必须相等。
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
        int64_t chunk_size,
        // W8A16 (可选, `_v` 孪生): 与 text 侧**各自独立**的 7 条 scale。
        // 孪生权重不经基类, 这里自己校验 (全给/全空 + int8 + numel==N)。
        at::TensorList q_ws_list = {}, at::TensorList k_ws_list = {},
        at::TensorList v_ws_list = {}, at::TensorList o_ws_list = {},
        at::TensorList gate_ws_list = {}, at::TensorList up_ws_list = {},
        at::TensorList down_ws_list = {});

    // 一次 VLM prefill。I/O 契约:
    //   x_emb       [1, chunk_size, hidden=2048] fp16 RPU contig — host 已把
    //               vision embedding 与 text embedding 混排好的 prefix。
    //   k/v_caches  基类 7-D swizzled KV cache; 本步在 prefix_len 处追加。
    //               prefill 后即为 action expert 消费的 prefix KV。
    //   attn_mask_4d 显式 additive mask, 末两维 [chunk_size, prefix+cs]
    //               (基类 sdpa_prepare_mask 降维 + MASK_2D)。⚠️ 整行 masked 的
    //               padding 行在 CPU softmax 下退化成均匀分布, RPU kernel 未必 —
    //               调用方须给这类行开对角口 (因果 mask 保证它们流不回有效行)。
    //   prefix_len  KV prefix 起始位置 (VLM prefill = 0)。
    // 返回: 末隐藏 [1, chunk_size, hidden] fp16 RPU (含 final_norm) —— 基类
    //   output DMA 的 output_tensor_ (graph 写目标)。调用方须在 capture 作用域
    //   **退出后**读 (此时图已 end() 执行写入); 切勿在作用域内提前 copy —— 那会
    //   读到尚未执行的全 0 output_tensor_（见 FusedModelBase 输出契约）。
    // 所有权: caller 持有输入张量; 返回张量由基类 registry 持有 (跨 forward 稳定)。
    // ⚠️ capture 路径下返回的是 graph 写目标 raw 指针而非 clone
    //   按 FusedModelBase 契约，每次 forward 后**必须立刻读回**,
    //   攒着多次返回值会互相别名。PASSTHROUGH 路径才返回 .clone()。
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
    // 只为挂 RPU_HY_VLA_FAST_REPLAY 的逐座开关（默认 OFF），其余全继承。
    ModelStaticConfig static_config() override;
    // dynamic_config: 基类已做单 chunk 断言 + 自己的 mask 准备; 这里再准备一份
    // 子类可见的 PreparedMask (基类的是 private)。forward/static_config 继承。
    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override;
    // prefill 永远单块 (整个 prefix 一次送 MASK_2D 非因果 SDPA)。Hy-VLA 的
    // seq=192 > 176 ⇒ tile_m_v16 封顶 8 → tile_m=128 → grid=2；16/4 GQA 在
    // attn_tp=4 下 gqa=4 ⇒ product=8，**正好压在** keeper 的 product<=8 上限。
    // override 只表达行掩码/KV merge 的单块语义 (cs>=seq_len)，不是绕过守卫；
    // 更长的 prefix 会让 grid 变大而触上限，届时须改成分块 + 掩码按 offset 取切片。
    bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len,
                                   int64_t position) const override;

private:
    // 行掩码 merge: out = text*text_mask + vis*vis_mask。
    // 前置: text/vis 位于不同 SPM 槽; MUL 就地破坏两者, out 可与 text 同槽。
    // 掩码是 [rows,1] 的**行向量**, 两次 MUL 走 Nx1_NxC 行广播 (见 declare_buffers)。
    void emit_row_merge(uint32_t text_addr, uint32_t act_addr, uint32_t out_addr,
                        uint32_t text_mask_addr, uint32_t act_mask_addr,
                        int64_t rows, int64_t cols, int num_cores);

    std::vector<MoeLayerWeights> text_layers_;  // [num_layers] text 权重影子 (基类副本)
    at::Tensor cos_hyvla_, sin_hyvla_;          // RoPE 表影子 (build 期 host 指针)
    double eps_hyvla_ = 1e-5;                   // rms_norm_eps 影子 (HALO 是 1e-6)
    std::vector<MoeLayerWeights> moe_layers_;   // [num_layers] vision (*_v) 孪生权重
    at::Tensor final_norm_text_;                // [hidden] final RMSNorm γ 影子 (基类的是
                                                // private)。仅用于在 set_moe_weights 里
                                                // 断言孪生侧传的是**同一张量** —— 步 7 据此
                                                // 单算不双算, 传了不同的 γ 会静默算错。
    at::Tensor text_row_mask_;                  // [chunk_size] 原始行掩码 (keepalive)
    // [chunk_size, 1] fp16 RPU 常量行掩码 (Persistent 槽 preload 源)。
    at::Tensor text_mask_ddr_, act_mask_ddr_;
    int64_t moe_chunk_size_ = 0;                // 掩码/槽固化的 chunk_size (0 = 未设)
    PreparedMask prepared_mask_hyvla_;           // 每 forward 在 dynamic_config 重备
};

}  // namespace v3

// C API 自由函数原型集中在 src/core/rpu_kernel_decls.h 的 HALO 段，
// rpu_backend.cpp 经 TORCH_FN 绑定, 不引入本重头文件。

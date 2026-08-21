// rpu_halo_image_flow_model.h — HALO image flow-matching denoise step (mode="gen").
//
// 模型背景: HALO 的 image generation 用 flow-matching。一次 denoise step
// (Bagel._forward_flow in the reference model) 对
// query = **1378 行 = 2 text(start/end_of_image 边界) + 1376 vae latent** 做一次
// 非因果 forward, 输出 velocity(只取 1376 vae 行)。
//
// 与 action expert (mode="act") 的两点关系:
//
//  1. **双流 MoT（非单流）。** 输入契约要求 query 含 2 个 text 行
//     (packed_text_indexes=[0,1377], packed_text_ids=[151652,151653]=start/end_of_image)。
//     The 2 text rows use the base/und twin projections
//     (`q_proj`/`q_norm`/`input_layernorm`/`mlp`/`norm`), while the 1376 VAE
//     rows use the `*_moe_gen` twin. Per-twin projections merge into one Q/K/V
//     stream before RoPE and the non-causal full-attention SDPA.
//     故本类持**两套权重**(und=base, gen=moe_gen), 每层做"双算 + 行掩码合并", 与
//     action expert 同构 (action 是 2 text + 16 act 单块; D 是 2 text + 1376 vae 多块)。
//
//  2. **KV_FIRST 强制 (多块非因果)。** is_causal=False + query 1378。单块路径断言
//     num_chunks==1, 但 VLM SDPA tile 约束把单块 cs 压到 ≤~400 → 1378 必然多块 →
//     用 ChunkMode::KV_FIRST: Phase 1 先插全部 K/V, Phase 2 每块对满 KV SDPA
//     (1378 之间双向 + 全 prefix 可见 ⇒ MASK_NONE 全 attend)。
//
// 设计 = clone(HaloActionExpertModel 双流 MoT) ⊕ KV_FIRST(GemmaModel)。继承
// v3::CausalDecoderModel (复用 Qwen2.5+qk-norm 层体的全部 kernel 装配)。行掩码逐
// chunk 显式 body 内上传 (镜像 Gemma gemma_upload_chunk_pm; preload_callback 无
// chunk.offset)。emit_row_merge / mlp_half 与 action expert 使用相同机制。
//
// The input is host-composed x_emb [1, 1378, 1536] fp16 (vae2llm/time_embedder/
// latent_pos_embed 投影 + start/end_of_image embed_tokens 已在 host 落到 2 个 text 行;
// 输出 llm2vae + slice 到 1376 vae 行 + CFG renorm 仍在 host)。
//
// This component requires its named board-level end-to-end validation gate.
#pragma once

#include "rpu_qwen3_model.h"     // v3::CausalDecoderModel + NUM_CORES/DWIDTH
#include "rpu_kernel_decls.h"
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <unordered_map>
#include <map>
#include <utility>
#include <vector>

namespace v3 {

class HaloImageFlowModel : public CausalDecoderModel {
public:
    HaloImageFlowModel();
    ~HaloImageFlowModel() override;

    // 单层孪生权重 (布局与基类 LayerWeights 主权重逐一对应; und/gen 同结构)。
    struct GenLayerWeights {
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor q_norm_w, k_norm_w;
        at::Tensor input_norm_w, post_norm_w;
        at::Tensor gate_w, up_w, down_w;
        at::Tensor q_bias, k_bias, v_bias;
    };

    // und/base 孪生入口 (= action expert set_weights_halo 角色): 内部转基类
    // CausalDecoderModel::set_weights (mrope/deepstack 空; q/k norm + QKV bias 走
    // Qwen3+bias 路径, fp16-only), 同时留一份子类影子 und_layers_ 供 KV_FIRST 两
    // 阶段 build 取用 (基类 layer_weights_/cos_/sin_/eps_ 是 private)。喂入无后缀
    // base 孪生 (start/end_of_image 那 2 个 text 行用)。
    void set_weights_image_flow(
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

    // gen 孪生入口 (= action expert set_moe_weights 角色, 减去 chunk_size SDPA
    // override): 载 `*_moe_gen` 孪生 + 行掩码 [query_len] (1=und/text 行 {0,1377})。
    // 必须在 set_weights_image_flow 之后调用 (num_layers/has_qk_norm/has_qkv_bias
    // 由 base 提供)。
    void set_gen_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        at::TensorList input_norm_list, at::TensorList post_norm_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        at::TensorList q_bias_list, at::TensorList k_bias_list,
        at::TensorList v_bias_list,
        const at::Tensor& final_norm_gen_w,
        const at::Tensor& row_mask,        // [query_len] fp16 RPU {0,1}; 1=und(text)行
        int64_t query_len);                // = 1378 (= 行掩码长度; 非 SDPA chunk plan)

    // 一次 image flow denoise step。I/O 契约:
    //   x_emb       [1, seq=1378, hidden=1536] fp16 RPU contig — host 已合成
    //               (vae 行 = vae2llm(x_t)+time_embed+latent_pos; text 行 =
    //                embed_tokens(start/end_of_image))。
    //   k/v_caches  基类 7-D swizzled KV cache; 本步在 prefix_len 处追加 1378 行。
    //   cos/sin     per-forward RoPE 表 (CFG 三分支各异; 图像 token 共享单一
    //               rope_position → 位置广播表)。
    //   prefix_len  KV prefix 长度 (CFG 分支各异: cond 9586 / cfg_text 9579 /
    //               cfg_img 7; C++ 与分支无关, 全在调用方传参)。
    // 全 attend 非因果 mask 在内部合成 (Gemma forward 同款), 调用方不传 SDPA mask
    // (行掩码在 set_gen_weights 时绑定)。KV_FIRST 内部自动多块, 不传 chunk_size。
    // 返回: 末隐藏 [1, 1378, hidden] fp16 RPU。调用方须在 capture 作用域**退出后**
    //   读（见 FusedModelBase 输出契约）。host 随后 llm2vae → slice 1376 vae
    //   行 → velocity。
    at::Tensor step_forward(
        const at::Tensor& x_emb,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cos,
        const at::Tensor& sin,
        int64_t prefix_len);

protected:
    // KV_FIRST 两阶段派发注册 (kv_first_fn / kv_first_chunk_plan_fn)。
    ModelStaticConfig static_config() override;
    // chunk_mode=KV_FIRST (无 SDPA-mask 准备 —— image gen 全 attend, MASK_NONE;
    // 行掩码 ≠ SDPA mask, 经 declare_buffers 槽 + body 内 per-chunk DMA)。
    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override;
    // KV_FIRST 专属 SPM 布局 (KVIN/COMP 相位别名 + und/gen 双算槽 + 行掩码槽 +
    // gen 孪生 norm/bias preload 槽; 无 sdpa_mask 槽 —— MASK_NONE 全 attend)。
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
    // Phase 2 body: load-q / 满 KV SDPA(MASK_NONE) / O 双算+merge / 残差 /
    // post-norm 双算+merge / SwiGLU MLP 双算+merge / (末层) final-norm 双算+merge /
    // output DMA。
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override;
    // VLM SDPA tile 约束门, 按实际 kernel 的 mask=0 (MASK_NONE) 校验 —— 见 .cpp。
    // mask=1 的 LTM sQryAcc%16 约束不适用于非因果 CFG prefix。
    bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len,
                                   int64_t position) const override;

private:
    // 基类 make_sdpa_config 是 private (用其私有 sdpa_kernel_), 子类取不到 →
    // 自备一份 (HALO 走 FLASH_ATTN_SPM, 与 step 内 SDPA dispatch 一致)。
    SdpaConfig make_sdpa_config(int mask = 1) const {
        return {SdpaKernelType::FLASH_ATTN_SPM, head_dim(), num_q_heads(),
                num_kv_heads(), attn_tp(), mask};
    }

    // Phase 1 body (KV_FIRST kv_first_fn): input-norm 双算+merge / QKV(+bias) 双算
    // +merge / qk head-norm 双算+merge / RoPE(merged) / KV insert(merged) / 存
    // merged rope_q 到 q_ddr_buf_。
    void emit_kv_first_body(int layer_idx, const ChunkInfo& chunk);
    // kv_first_chunk_plan_fn: KV-insert 相 (无 SDPA/MLP) 用更大块。
    ChunkPlan plan_kv_first_chunks(const ChunkPlan& compute_plan);

    // emit_row_merge — out = und*und_mask + gen*gen_mask。行掩码 [n,1] 经
    // Nx1_NxC_spm 广播到 [n,c] (und/gen_mask 是单一行掩码 addr, 通吃任意宽度 c);
    // 末 ADD 仍 sameshape。MUL 就地覆写 und/gen 输入槽; out 与 und 同槽是常态。
    void emit_row_merge(uint32_t und_addr, uint32_t gen_addr, uint32_t out_addr,
                        uint32_t und_mask_addr, uint32_t gen_mask_addr,
                        int64_t n, int64_t c, int num_cores);

    // 把行掩码的 [chunk.offset : offset+len] 行切片显式 DMA 进 6 个 mask 槽
    // (und/gen × q/kv/h)。preload_callback 无 chunk.offset → 必须在 body 内做
    // (镜像 Gemma gemma_upload_chunk_pm)。h_only=true 时只传 h 宽 (Phase 2 只用 h)。
    void upload_merge_masks(const ChunkInfo& chunk, bool h_only);

    std::vector<GenLayerWeights> und_layers_;   // [num_layers] und/base 孪生影子
    std::vector<GenLayerWeights> gen_layers_;   // [num_layers] gen(*_moe_gen) 孪生
    at::Tensor cos_halo_, sin_halo_;            // RoPE 表影子 (build 期 host 指针)
    at::Tensor final_norm_w_halo_;              // [hidden] und final RMSNorm γ 影子
    at::Tensor final_norm_gen_w_;               // [hidden] gen final RMSNorm γ 影子
    double eps_halo_ = 1e-6;                    // RMSNorm eps 影子
    bool has_qkv_bias_halo_ = false;            // QKV bias 在场 (两孪生一致)

    // 行掩码常量 [query_len,1] fp16 DDR (und=行掩码, gen=1-行掩码)。Nx1_NxC_spm 广播到
    // 任意 merge 宽度 (q/kv/h) → 单一行掩码通吃, 替代旧 per-width [query_len,W] 全宽掩码
    // (SPM 掩码 6 槽→2 槽, ~3072/1536/256 B/cs → ≈0)。body 内按 chunk 行切片 DMA。
    at::Tensor row_mask_;            // [query_len] keepalive
    at::Tensor und_mask_row_;        // [query_len,1] = 行掩码
    at::Tensor gen_mask_row_;        // [query_len,1] = 1-行掩码
    int64_t merge_seq_len_ = 0;      // = query_len (=1378)

    // KV_FIRST 运行态: rope_q 跨两阶段 DDR 暂存 (Gemma 同构)。按 {seq_len, q_width =
    // (num_q_heads()/attn_tp())*head_dim()} 双键的稳定 per-shape slot —— 创建后永不
    // realloc,故已捕获图的 FIXED scatter DMA 地址跨 REPLAY 始终有效;宽度入键覆盖
    // set_weights 改变 head 维度的复用场景。镜像 q_ddr_slots_/
    // temp_ddr_slots_；per-shape slot 为 replay 稳定不可释放，128 软上限
    // 阻止形状数量无界增长。
    std::map<std::pair<int64_t, int64_t>, at::Tensor> q_ddr_slots_;
    at::Tensor& q_ddr_slot() {
        return q_ddr_slots_.at({seq_len_, (num_q_heads() / attn_tp()) * head_dim()});
    }
    int64_t seq_len_ = 0;                       // 本 forward 的 query seq_len
};

}  // namespace v3

// C API 自由函数原型集中在 src/core/rpu_kernel_decls.h (HALO image_flow 段),
// rpu_backend.cpp 经 TORCH_FN 绑定。

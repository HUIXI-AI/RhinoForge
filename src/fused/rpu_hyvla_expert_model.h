// rpu_hyvla_expert_model.h — Hy-Embodied-0.5-VLA 的 action expert denoise 解码器。
//
// 模型背景: dual-tower 的第二座塔。VLM prefill (rpu_hyvla_vlm_model) 产出 240/S 行
// prefix KV 后, expert 每个去噪步用 51 行 suffix (1 state + 50 action) 去 attend
// [prefix KV | 自身 suffix KV], 输出经 host 的 action_out_proj 得到流场 v_t。
// 10 步 Euler 由调用方驱动 (P4 再做 in-graph unroll)。
//
// expert 是单塔，不做 MoT 双算：suffix 的 modality_mask 恒 True，
// mask_apply 只选择 `_v` 分支。本类只接收这一分支的权重，
// 没有孪生槽、没有行掩码、没有 merge —— 这是它比 HyVlaVlmModel 简单得多的原因。
//
// 与基类 CausalDecoderModel 的唯一实质差异, 与 VLM 塔相同:
//   **步 3 先 RoPE 再 q/k head-norm** (基类及现有 7 个 emitter 都是 norm→rope)。
// 且该 head-norm **取自 VLM 层而非 expert 层** —— vendor modeling_dual_tower.py
// 的层循环里写死 `vlm_layer = models[0].layers[layer_idx]`, 两塔共享它的
// query_layernorm/key_layernorm; expert 自己那份同名张量不参与该路径。
// 绑定时必须喂 VLM 的那份, 喂 expert 的会静默用错权重 (形状相同, 不会报错)。
//
// 几何: hidden=1024, inter=2048, 32 层, 16 q-heads / 4 kv-heads × hd=128, eps=1e-5。
// KV 几何 (4×128) 与 VLM 塔**完全一致**, 所以两塔共用同一条 7-D RPUCache。
//
#pragma once

#include "rpu_qwen3_model.h"     // v3::CausalDecoderModel + NUM_CORES/DWIDTH
#include "rpu_kernel_decls.h"
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

namespace v3 {

class HyVlaExpertModel : public CausalDecoderModel {
public:
    HyVlaExpertModel();
    ~HyVlaExpertModel() override;

    void set_runtime_config(
        bool fast_replay, bool fast_replay_preload, bool mask_once,
        bool silu_mul, bool kvpad16, bool partial_rope,
        bool rmsnorm_pad16);

    // 权重入口。全部取 expert 的 `_v` 半边, **除了**:
    //   q_norm_list / k_norm_list —— 必须是 **VLM** 层的 query/key_layernorm
    //                                (见文件头; 传 expert 的会静默算错)。
    // 内部转基类 set_weights (mrope/deepstack 空, 无 QKV bias, fp16-only),
    // 并留一份逐层权重影子供 build_layer_subgraph 使用 (基类的是 private)。
    void set_weights_expert(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        at::TensorList input_norm_list, at::TensorList post_norm_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        const at::Tensor& cos, const at::Tensor& sin,
        const at::Tensor& final_norm_w,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps, int64_t chunk_size, int64_t action_mlp_cores,
        // 量化 (可选): 7 条 scale；W8A16 为 per-output-channel fp16 [N]，
        // W4A16 为 controller-striped pgrp fp16 [K/group_size, N]。全空 = fp16。
        // 校验由基类 CausalDecoderModel::set_weights 统一做 (要么七条全给、
        // 要么七条全空；给了就要求对应权重为 int8 或 packed uint8)。
        at::TensorList q_ws_list = {}, at::TensorList k_ws_list = {},
        at::TensorList v_ws_list = {}, at::TensorList o_ws_list = {},
        at::TensorList gate_ws_list = {}, at::TensorList up_ws_list = {},
        at::TensorList down_ws_list = {});

    // 一次 denoise step。I/O 契约:
    //   x_emb       [1, chunk_size, 1024] fp16 RPU contig — host 已合成的 suffix
    //               (state token + action-time token; boundary 投影留在 host = M1)。
    //   k/v_caches  与 VLM 塔共用的 7-D swizzled cache。调用方须在每步前
    //               reset_to_position(prefix_len) —— 本步追加的 51 行用完即弃
    //               (vendor 用 copy.deepcopy 冻结 prefix KV, 等价)。
    //   cos/sin     [rows, head_dim/2] fp16;行 i 对应 position_ids[i]
    //               (= prefix 有效长度 + i)。
    //   attn_mask_4d 显式 additive mask, 末两维 [chunk_size, prefix_len+cs]。
    //               prefix 块 == prefix_pad_masks 广播 (**不是全 True**),
    //               suffix 块 = state 只看自己 + action 看全部 51。
    //   prefix_len  KV 中 prefix 的长度 (= VLM prefill 用的 S)。
    // 返回: [1, chunk_size, 1024] fp16 RPU (含 final_norm)。capture 路径下返回的是
    //   graph 写目标 raw 指针, **每次 forward 后必须立刻读回** (基类
    //   fused_model_base.cpp:1220-1246 契约; 与 VLM 塔同)。
    at::Tensor step_forward(
        const at::Tensor& x_emb,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cos, const at::Tensor& sin,
        const at::Tensor& attn_mask_4d, int64_t prefix_len,
        at::IntArrayRef planned_stage_descriptor = {});

    // ── in-graph N 步 Euler unroll (opt-in; 不调用 = 上面的单步路径逐位不变) ──
    // 绑定融合的 suffix encoder + action_out_proj + 逐步 time bias。**必须在
    // set_weights_expert 之后**调用 (要 hidden_size())。除 mo 外全部单核
    // (partition=1) col-swizzle fp16；**mo 默认走 8 核 row(K) 切分**
    // (见本 .cpp 文件头 ACTION_MLP_MC，Python 侧同源函数 `_action_mlp_cores()`):
    //   wc       [hidden, action_dim_pad] —— 折叠后的 encoder 第一个 GEMM。
    //            host 侧恒等式 (Python 折的):
    //            `(x·W_in + b_in)·W1aᵀ + te_k = x·(W_in·W1aᵀ) + (b_in·W1aᵀ + te_k)`
    //   mo/mo_bias  [hidden, hidden] / [hidden] —— action_time_mlp_out
    //   op/op_bias  [action_dim_pad, hidden] / [action_dim_pad] —— action_out_proj
    //   time_all    [num_steps, hidden] fp16 contig —— 每步的 `b_in·W1aᵀ + te_k`
    void set_action_weights(
        const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
        const at::Tensor& op, const at::Tensor& op_bias, const at::Tensor& time_all,
        int64_t action_dim, int64_t num_steps);

    // 一次 forward 展开 num_steps 个 [encoder → 32 层 → out_proj → Euler]。
    // x 全程留在 SPM (x_t_spm), host 只上传 x0 / 读回 x_traj。
    //   x0_rpu  [1, cs, action_dim_pad] fp16 RPU —— 行 0 全 0 (state 位), 行 1: = noise
    //   state_emb [hidden] fp16 RPU —— emb_stage_ 行 0, N 步不变
    //   x_traj  [num_steps, cs, action_dim_pad] fp16 RPU —— 逐步轨迹, 终值 = [-1]
    void unroll_forward(
        const at::Tensor& x0_rpu,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& state_emb,
        const at::Tensor& cos, const at::Tensor& sin,
        const at::Tensor& attn_mask_4d,
        at::Tensor& x_traj, double dt, int64_t prefix_len, int64_t num_steps,
        at::IntArrayRef planned_stage_descriptor = {});

    std::vector<int64_t> resolve_stage_domain(
        int64_t seq_len, int64_t prefix_len, int64_t mask_kv_len);

protected:
    // 单塔层体用的槽 (residual1/2, input_norm, q/k/v, output, oproj, gate/up/down,
    // sdpa_mask/tmp) 基类已全部声明; override 只为 unroll 追加 encoder/Euler 的槽,
    // 仅 unroll_mode_ 追加 hooks；mask 保留决策在完整声明后进行。
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
    ModelStaticConfig  static_config() override;
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override;
    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override;
    ModelDynamicConfig planning_dynamic_config(
        const ChunkPlan& plan) override;
    // suffix 整块进 MASK_2D 非因果 SDPA。cs=64 (=ceil16(51)) 时 tile_m=64 ⇒
    // grid=1, gqa=4 ⇒ product=4, 远在 keeper 的 product<=8 之内。
    bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len,
                                   int64_t position) const override;
    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override;
    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& manifest) const override;

private:
    std::vector<BufferDecl> baseline_buffer_declarations(const LayoutContext&) const;
    std::vector<BufferDecl> planned_buffer_declarations(
        const LayoutContext&, bool* retained = nullptr) const;
    int64_t mask_schedule_for_layout(const LayoutContext&) const;

    void emit_pre_layers_body();      // encoder → emb_stage_ 行 1:cs
    void emit_post_layers_body();     // out_proj + 图内 Euler + 轨迹落盘

    bool cold_fast_replay_ = false;
    bool cold_fast_replay_preload_ = false;
    bool cold_mask_once_ = false;
    bool cold_silu_mul_ = false;
    bool cold_kvpad16_ = false;
    bool cold_partial_rope_ = false;
    bool cold_rmsnorm_pad16_ = false;
    bool cold_config_bound_ = false;

    struct LayerW {
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor gate_w, up_w, down_w;
        // 量化 scale：W8A16 为 per-output-channel fp16 [N]，W4A16 为
        // controller-striped pgrp fp16 [K/group_size, N]。全部 undefined = fp16
        // 路径 (发射序列逐字节不变)，由基类 CausalDecoderModel::set_weights 校验。
        at::Tensor q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws;
    };
    // 权重只保留引用；W4 scale 在 set_weights_expert 中另建 aligned 保活副本。
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        auto identity = causal_kvinsert_cost_weight_identity();
        if (identity.empty()) return {};
        append_kvinsert_cost_scalar_identity(identity, eps_expert_);
        identity.insert(identity.end(), {
            static_cast<int64_t>(cold_fast_replay_),
            static_cast<int64_t>(cold_fast_replay_preload_),
            static_cast<int64_t>(cold_mask_once_),
            static_cast<int64_t>(cold_silu_mul_),
            static_cast<int64_t>(cold_kvpad16_),
            static_cast<int64_t>(cold_partial_rope_),
            static_cast<int64_t>(cold_rmsnorm_pad16_)});
        identity.push_back(static_cast<int64_t>(layers_.size()));
        for (const auto& weights : layers_) {
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.gate_w, &weights.up_w, &weights.down_w, &weights.q_ws,
                    &weights.k_ws, &weights.v_ws, &weights.o_ws, &weights.gate_ws,
                    &weights.up_ws, &weights.down_ws}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &wc_, &mo_, &mo_bias_, &op_,
                &op_bias_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    std::vector<LayerW> layers_;      // [num_layers] 权重/scale 影子
    at::Tensor cos_ref_, sin_ref_;    // RoPE 表影子 (build 期取 raw host 指针)
    double eps_expert_ = 1e-5;
    int64_t expert_chunk_size_ = 0;   // 固化的 suffix 长度 (0 = 未设)
    int action_mlp_cores_ = NUM_CORES;
    bool production_config_bound_ = false;

    // ── unroll 专属 (unroll_mode_ = false 时全部不参与) ──
    bool     unroll_mode_ = false;
    at::Tensor wc_, mo_, mo_bias_, op_, op_bias_, time_all_;
    int64_t  action_dim_ = 0, action_dim_pad_ = 0, num_steps_ = 0;
    c10::Half dt_ = c10::Half(0.0f);
    bool     dt_pinned_ = false;
    at::Tensor emb_stage_;            // 稳定 staging DDR [1, cs, hidden]; 行 0 = state
    uint64_t x_t_src_base_ = 0;       // x0 的 mutable DMA 源基址 (仅 body_iter 0 读)
    uint64_t x_traj_dst_base_ = 0;    // 逐步轨迹的 mutable DMA 目标基址
    at::Tensor x0_ref_, x_traj_ref_, state_ref_;   // 同步 forward 期间的保活
};

}  // namespace v3

// C API 自由函数原型集中在 src/core/rpu_kernel_decls.h, rpu_backend.cpp 经
// TORCH_FN 绑定, 不引入本重头文件。

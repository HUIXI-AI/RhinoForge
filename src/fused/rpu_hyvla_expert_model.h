// rpu_hyvla_expert_model.h — Hy-Embodied-0.5-VLA 的 action expert denoise 解码器。
//
// 模型背景: dual-tower 的第二座塔。VLM prefill (rpu_hyvla_vlm_model) 产出 240/S 行
// prefix KV 后, expert 每个去噪步用 51 行 suffix (1 state + 50 action) 去 attend
// [prefix KV | 自身 suffix KV], 输出经 host 的 action_out_proj 得到流场 v_t。
// 10 步 Euler 可由调用方驱动，也可使用 in-graph unroll。
//
// **与 VLM 塔的本质区别：expert 是单塔，不做 MoT 双算。** suffix 的
// modality_mask 恒 True (51/51)，因此参考模型的 mask_apply 只选 `_v` 分支，
// 非 `_v` 的 369.2M 参数是死权重。所以本类只吃一份权重 (调用方喂 `_v` 半边),
// 没有孪生槽、没有行掩码、没有 merge —— 这是它比 HyVlaVlmModel 简单得多的原因。
//
// 与基类 CausalDecoderModel 的唯一实质差异, 与 VLM 塔相同:
//   **步 3 先 RoPE 再 q/k head-norm** (基类及现有 7 个 emitter 都是 norm→rope)。
// 且该 head-norm **取自 VLM 层而非 expert 层**；两塔共享 VLM 层的
// query_layernorm/key_layernorm，expert 自己的同名张量不参与计算。
// 绑定时必须喂 VLM 的那份, 喂 expert 的会静默用错权重 (形状相同, 不会报错)。
//
// 几何: hidden=1024, inter=2048, 32 层, 16 q-heads / 4 kv-heads × hd=128, eps=1e-5。
// KV 几何 (4×128) 与 VLM 塔**完全一致**, 所以两塔共用同一条 7-D RPUCache。
//
// ``denoise_step`` 使用 VLM 的 Q/K head-norm 权重。
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
        double eps, int64_t chunk_size,
        // W8A16 (可选): 7 条 per-output-channel fp16 scale [N]。全空 = fp16 路径。
        // 校验由基类 CausalDecoderModel::set_weights 统一做 (要么七条全给、
        // 要么七条全空; 给了就要求对应权重是 int8)。
        at::TensorList q_ws_list = {}, at::TensorList k_ws_list = {},
        at::TensorList v_ws_list = {}, at::TensorList o_ws_list = {},
        at::TensorList gate_ws_list = {}, at::TensorList up_ws_list = {},
        at::TensorList down_ws_list = {});

    // 一次 denoise step。I/O 契约:
    //   x_emb       [1, chunk_size, 1024] fp16 RPU contig — host 已合成的 suffix
    //               (state token + action-time token; boundary 投影留在 host)。
    //   k/v_caches  与 VLM 塔共用的 7-D swizzled cache。调用方须在每步前
    //               reset_to_position(prefix_len) —— 本步追加的 51 行用完即弃
    //               （等价于参考模型冻结 prefix KV）。
    //   cos/sin     [rows, head_dim/2] fp16;行 i 对应 position_ids[i]
    //               (= prefix 有效长度 + i，本配置为 181..231)。
    //   attn_mask_4d 显式 additive mask, 末两维 [chunk_size, prefix_len+cs]。
    //               prefix 块 == prefix_pad_masks 广播 (**不是全 True**, 181/240),
    //               suffix 块 = state 只看自己 + action 看全部 51。
    //   prefix_len  KV 中 prefix 的长度 (= VLM prefill 用的 S)。
    // 返回: [1, chunk_size, 1024] fp16 RPU (含 final_norm)。capture 路径下返回的是
    //   graph 写目标 raw 指针, **每次 forward 后必须立刻读回** (基类
    //   FusedModelBase 契约；与 VLM 塔同)。
    at::Tensor step_forward(
        const at::Tensor& x_emb,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cos, const at::Tensor& sin,
        const at::Tensor& attn_mask_4d, int64_t prefix_len);

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
        at::Tensor& x_traj, double dt, int64_t prefix_len, int64_t num_steps);

protected:
    // 单塔层体用的槽 (residual1/2, input_norm, q/k/v, output, oproj, gate/up/down,
    // sdpa_mask/tmp) 基类已全部声明; override 只为 unroll 追加 encoder/Euler 的槽,
    // 且**仅在 unroll_mode_ 下**追加 —— 单步路径的 SPM 布局逐字节不变。
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
    ModelStaticConfig  static_config() override;
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override;
    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override;
    // suffix 整块进 MASK_2D 非因果 SDPA。cs=64 (=ceil16(51)) 时 tile_m=64 ⇒
    // grid=1, gqa=4 ⇒ product=4, 远在 keeper 的 product<=8 之内。
    bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len,
                                   int64_t position) const override;

private:
    void emit_pre_layers_body();      // encoder → emb_stage_ 行 1:cs
    void emit_post_layers_body();     // out_proj + 图内 Euler + 轨迹落盘

    struct LayerW {
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor gate_w, up_w, down_w;
        // W8A16: per-output-channel fp16 scale [N]。全部 undefined = fp16 路径
        // (发射序列逐字节不变)。定义了则对应权重必须是 int8, 由基类
        // CausalDecoderModel::set_weights 校验。
        at::Tensor q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws;
    };
    std::vector<LayerW> layers_;      // [num_layers] 权重影子 (引用计数, 不复制数据)
    at::Tensor cos_ref_, sin_ref_;    // RoPE 表影子 (build 期取 raw host 指针)
    double eps_expert_ = 1e-5;
    int64_t expert_chunk_size_ = 0;   // 固化的 suffix 长度 (0 = 未设)
    PreparedMask prepared_mask_;      // 每 forward 在 dynamic_config 重备

    // ── unroll 专属 (unroll_mode_ = false 时全部不参与) ──
    bool     unroll_mode_ = false;
    // KVPAD16（见 .cpp 文件头）：k/v 槽被保证的 SPM 行容量；0 = 关闭。
    // declare_buffers 每次重算，KV-insert 站点原样传给 launcher。
    int64_t  kv_pad_rows_ = 0;
    // RMSNORM_PAD16（见 .cpp 文件头）：补齐后的 rmsnorm 行数。0 = 开关关，走原路。
    int64_t  rmsnorm_pad_rows_  = 0;   // input/post/k_norm 的 M（cs → Align(cs,16)）
    int64_t  rmsnorm_pad_qrows_ = 0;   // q_norm 的 M（cs*local_q → Align(...,16)）
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

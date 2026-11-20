// rpu_hyvla_expert_model.cpp — Hy-Embodied-0.5-VLA 的 action expert denoise 解码器。
// 设计与易错点见 rpu_hyvla_expert_model.h 的文件头。
//
// 层体是 rpu_hyvla_vlm_model.cpp 的**单塔版**: 同样的 rope→QK-norm 换序, 但去掉了
// 全部孪生分支 (双 GEMM / 行掩码 merge / act 侧 Temp 槽 / 掩码 Persistent 槽) ——

// 基类声明层体槽，本类追加 unroll hooks，并统一规划 mask 的跨 body 生命周期。
//
#include "rpu_hyvla_expert_model.h"
#include "hyvla_linear_precision.h"
#include "hyvla_retained_schedule.h"
#include "ops/rpu_specialized_linear_contracts.h"
#include "ops/rpu_hyvla_gateup_w8.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"
#include "rpu_spm_residency.h"
#include "rpu_hyvla_gateup_resident.h"
#include "hyvla_ring_postnorm.h"
#include "hyvla_o_ring_postnorm.h"
#include "hyvla_kv_grouped.h"
#include "hyvla_qk_rope.h"
#include "hyvla_qk_rope_norm.h"
#include "hyvla_qk_rope_norm_kv.h"
#include "hyvla_nvfp4_qkv.h"
#include "hyvla_nvfp4_q_tuning.h"
#include "hyvla_nvfp4_qkv_overlap.h"
#include "ops/rpu_hyvla_sdpa_tiling.h"
#include "hyvla_down_nextnorm.h"
#include "hyvla_w8_o_ring_postnorm.h"
#include "hyvla_w8_o_resident.h"
#include "hyvla_xor3_norm.h"
#include "ops/rpu_hyvla_gateup_nvfp4.h"
#include "hyvla_nvfp4_o_ring_postnorm.h"
#include "hyvla_nvfp4_down_n128.h"
#include "hyvla_nvfp4_down_nextnorm.h"
#include "hyvla_nvfp4_resident_swiglu.h"
#include "hyvla_nvfp4_resident_scales.h"

#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_FAST_REPLAY —— **逐座 opt-in** 的 fast-replay，默认 OFF。
//
// 打开后 REPLAY 期不再重走该座的 op stream（`fused_model_base.cpp:1126` 的
// `fast_skip`），省掉每帧把 ~700 节点/层 重新 emit 一遍的 host 开销
// （实测 ViT 4.48 / VLM 4.69 / expert(unroll ×10 body) 9.37 ms）。
//
// ⚠️ 契约（`fused_model_base.cpp:1105-1130`）：**body 不得有 kernel 会读的
// per-call host 副作用**。合法出口只有：position_ids 的稳定 keepalive、
// dynamic_config 拷进稳定 DDR 槽的显式 2D mask、`hidden_in_src_base_` 与
// op 序言里 `*_mutable(&member)` 的可变基址。
//
// ⚠️ 之所以做成**逐座**：上一次是三座 + deep + skip_sync 一起开，cos 掉到
// 0.98797891 / 套件 6 failed，无法定位是谁破的契约（账本 §11）。
// 取值：逗号分隔的座名（vit / vlm / expert），或 1/on/true 表示三座全开。
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_FAST_REPLAY_PRELOAD —— 同样**逐座 opt-in**，默认 OFF。
// 打开后在 REPLAY（且权重 clean）跳过 D-502 preload 回调与 `preload_fn` 的
// host 重走 —— 已建好的 preload DMA 节点仍由 segment launch 重放到同一块
// persistent SPM，逐位不变（框架里这条是 RhinoVLA vision 验过的，见
// `fused_model_base.cpp` 的 `fast_replay_skip_preload` 注释）。
// 依赖 `fast_replay_skip_layer_loop` 一起开（框架内部与它 AND）。
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_ACTION_MLP_MC —— unroll 的 `action_time_mlp_out`（步 A6）走几个核。
// 默认 **8**；取 0/off/false 回到单核（逐字节复现改动前的行为）。
//
// 为什么值得单独开一个开关：A6 是 `[hidden, hidden]` 的 fp16 GEMM，本模型
// hidden=1024 ⇒ 权重 **2 MB**，而它和 A4/Z2 一样被写成了 `num_cores=1`。
// R44 实测（`hwdist.py`，推荐档 `hw_r44_qv`）：这 30 枪 fp16 linear 是完美的
// 三峰周期 —— **A4 4.1 µs / A6 85.5 µs / Z2 6.6 µs**，每个 Euler 步一组。
//   A6：2 MB ÷ 85.5 µs = **24.5 GB/s**，已到**单核** DDR 读上限（R1 DDR benchmark：
//   单核 RPU 读 ~23.9、8 核 DMA ~142）⇒ 它把一个核打满了，另外 7 个核在这 0.855 ms 里全程闲着。
//   A4/Z2 权重各只有 64 KB，摊到 8 核反而被 all-reduce 吃掉 ⇒ **只动 A6**。
//
// 做法：K 方向切 8 核（`partition=0`）+ 自动调度的 fused ring all-reduce。
//   ⚠️ bias **不能**进 GEMM —— partition=0 时每核都会加一遍，all-reduce 后变 8 倍。
//      改成 all-reduce 之后用 `1xC_NxC` 按列广播加一次。
//   ⚠️ 零残差复用 `enc_h_spm`：它是 A6 的输入，A6 发完就死，就地清零当残差用，
//      省掉一个 104 KB/核 的槽（`all_reduce_sum_residual` 要求三个地址互不相同）。
//   ⚠️ **Python 侧的 swizzle 必须同源**。weights.py 解析一次并把核数随
//      set_weights 显式传入，native 不再独立读取进程环境。
//
// ⚠️ **不是逐位等价**：K 被切成 8 段分别累加再跨核求和 = 求和重排。
//    验收看 cos 分布，不能只看 wall。
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_MASK_ONCE —— 同样**逐座 opt-in**，默认 OFF。
//
// 显式 2D mask 在整个 forward 里逐层恒定（形状 `(seq_q, kv_seq_len)` 由
// dynamic_config 固定，单 chunk），但 `sdpa_mask` 槽默认是**相位混叠**的 Temp
// （基类 `rpu_qwen3_model.h:1215` 的 `[3,5]`），别的 Temp 会压在同一段 SPM 上
// ⇒ 每层 SDPA 之前都得把它从 DDR 重新广播进 SPM 一次。基类那里的注释
// （`rpu_qwen3_model.h:2025`）把这条记成了前提，其实它是**槽的相位窗口**带来的，
// 不是 mask 本身要变。
//
// 实测代价（hwperf `hw_r32/`，按 `size_bytes` 分档 —— 结论 “DMA 归因必须按字节量”）：
//   | 座 | 每层-核 | 次数 | 字节/帧 | 占该座 DMA |
//   |---|---|---|---|---|
//   | ViT | 696 KB | 27×8 | 150.4 MB | 55.3% |
//   | VLM | 113 KB | 32×8 |  29.5 MB | 62.2% |
//   | expert | 30 KB | 320×8 | **79.4 MB** | **87.4%** |
// 合计 **259 MB/帧**，DMA 时长之和 24.4 ms（多数与 kernel 重叠，segwall 口径下
// 暴露 4.9 ms）。
//
// 原路径把 mask 延长到层体 [1,8]，每个 body 的层 0 重载；pre-hook [0,0]
// 可以复用该地址。现在在追加所有 hooks 后用真实 allocator 尝试全 forward
// 存活（包括 phase 0）。精确 S51 profile 冷规划时检查完整 fixed/temp/held
// 预算，允许有界增长；其他 profile 保留不增加峰值的原规则。typed schedule
// 冻结整图一次或每 body 一次，运行时不按余量重选。MASK_ONCE 关闭时仍逐层加载。
// 每次 Graph invocation 都重新加载一次当前请求的稳定 DDR mask；这不把 SPM
// 当作跨 Graph/请求的缓存，也不跳过动态 mask 的 prepare/update。
// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_SILU_MUL —— 逐座 opt-in，默认 OFF。
//
// SwiGLU 的 `unary(SILU)` + `binary(MUL)` 两枪合成 op-lib 里**已有**的
// `llama_silu_mul` 一枪（`KernelId::LLAMA_SILU_MUL`，默认 ref 里就有，launcher
// `rpu_launch_silu_mul_spm_kernel` 也早就写好了 —— 此前只有 RhinoVLA 在用，
// 见 `rpu_rhino_vla_model.cpp::rhino_vla_fused_silu_mul_enabled`）。
// **不是新算子**，是把一个已存在但没接线到本模型的核接上。
//
// 靶子：hwperf（R37 分账 `hw_r37/`）expert 段 `unary` 0.83 ms/n=330 +
// `binary_sameshape` 0.67 ms/n=330 —— 单价 2.5 / 2.0 µs，**几乎全是 launch 开销**。
// 合并后实测（`hw_r38/`）：两者归零，新增 `llama_silu_mul` **0.50 ms/n=320
// @1.6 µs**，即 eltwise 侧净 **−1.00 ms**；两枪路径要 3 读 2 写，融合核只要
// 2 读 1 写，所以省的不止是 launch。
//
// ⚠️ 顺序变化：合并后必须先发 up 的 GEMM 再做 silu_mul（原路径是 gate GEMM →
// silu → up GEMM → mul）。两者无数据依赖，交换合法。
//
// ⚠️ **不保证逐位等价**：融合核内部先 silu 后乘，中间量是否落 fp16 由核决定，
// 与两枪路径可能差一次舍入 ⇒ 验收必须跑 cos 分布 A/B，不能只看逐位。
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_KVPAD16 —— 逐座 opt-in，默认 OFF。
//
// `RPU_KVINSERT_HYBRID_V16`（R22/R23）在 `seq % 16 != 0` 时发两枪：v16 主体 +
// v2 尾巴。本座 `suffix_len = 51 = 3×16 + 3` ⇒ 尾巴只有 **3 行**，却几乎和主体
// 一样贵（hwperf `hw_r32/`，320 层-步）：
//   k: v16(48 行) 1.03 ms + v2(3 行) 0.86 ms   v: v16 1.01 + v2 0.99
//   ⇒ 6% 的行占掉 **1.85 ms**（该座 kernel 时间的 1.9%）。
// ViT 同理（588 = 36×16 + 12，尾巴 0.34 ms）。
//
// 打开后把 k/v 的 SPM 槽行数补齐到 16 的倍数（expert 51→64、ViT 588→592，
// 每核各多 3.3 KB / 1.3 KB），并把这个**行容量**告诉 launcher，让它一次 v16 打完。
// 多写进 cache 的 `[position+seq, position+pad)` 那几行是 SPM 里的既有字节，
// **不会被读** —— SDPA 收到的 `kv_seq_len` 恒等于 `position + seq`
// （expert 291、ViT 588）。而且那一段本来就装着上一帧的陈旧数据、一直靠
// `kv_seq_len` 挡着 ⇒ **逐位等价**，不是"新引入了脏数据"。
// ─────────────────────────────────────────────────────────────────────────────
// 把 k/v 槽撑到 `rows` 行（只增不减）。返回实际保证的行容量。
static int64_t hyvla_grow_kv_slots(std::vector<v3::BufferDecl>& decls,
                                   int64_t rows, int64_t local_kv_elems) {
    const int64_t need = Align(rows * local_kv_elems * 2, 256);
    for (auto& b : decls)
        if (b.name && (std::strcmp(b.name, "k") == 0 || std::strcmp(b.name, "v") == 0))
            b.size = std::max(b.size, need);
    return rows;
}

static bool hyvla_grow_slot(std::vector<v3::BufferDecl>& decls,
                            const char* name, int64_t need) {
    for (auto& b : decls)
        if (b.name && std::strcmp(b.name, name) == 0) {
            b.size = std::max(b.size, Align(need, (int64_t)256));
            return true;
        }
    return false;
}

static void hyvla_widen_sdpa_mask_slot(std::vector<v3::BufferDecl>& decls) {
    for (auto& b : decls) {
        if (b.name && std::strcmp(b.name, "sdpa_mask") == 0) {
            b.phase_start = 1;
            b.phase_end   = 8;
            b.scope       = v3::BufferScope::LayerWide;
        }
    }
}

namespace v3 {

// HYVLA_EXPERT_FIXED_KERNEL_BASIS: launchers outside the selector-controlled
// sites below are invariant model-dataflow glue.  Every cold/per-handle route
// that changes the graph, kernel, DMA, collective, RoPE, or layout is bound to
// an explicit physical-manifest entry before its launcher is emitted.

constexpr int64_t HYVLA_EXPERT_ATTN_SITE = 8372732983358876378LL;
constexpr int64_t HYVLA_EXPERT_KV_SITE = 2502630319608950519LL;
constexpr int64_t HYVLA_EXPERT_GATEUP_RESIDENT_SITE = 623508127183406950LL;
constexpr int64_t HYVLA_EXPERT_GATEUP_PRELOAD_SITE = 623508127183406951LL;
constexpr int64_t HYVLA_EXPERT_GATEUP_SWIGLU_SITE = 623508127183406970LL;
constexpr int64_t HYVLA_EXPERT_DOWN_NEXTNORM_SITE = 623508127183407040LL;
constexpr int64_t HYVLA_EXPERT_QK_ROPE_SITE = 623508127183407030LL;
constexpr int64_t HYVLA_EXPERT_QK_ROPE_NORM_SITE = 623508127183407080LL;
constexpr int64_t HYVLA_EXPERT_QK_ROPE_NORM_KV_SITE = 623508127183407100LL;
constexpr int64_t HYVLA_EXPERT_NVFP4_QKV_SITE = 623508127183407090LL;
constexpr int64_t HYVLA_EXPERT_NVFP4_Q_N32_SITE = 623508127183407120LL;
constexpr int64_t HYVLA_EXPERT_SDPA_K128_SITE = 623508127183407150LL;
constexpr int64_t HYVLA_EXPERT_SDPA_M16_SITE = 623508127183407130LL;
constexpr int64_t HYVLA_EXPERT_O_RING_POSTNORM_SITE = 623508127183407020LL;
constexpr int64_t HYVLA_EXPERT_O_RESIDENT_SITE = 623508127183407060LL;
constexpr int64_t HYVLA_EXPERT_RING_POSTNORM_SITE = 623508127183407010LL;
constexpr int64_t HYVLA_EXPERT_RESIDENT_SWIGLU_SITE = 623508127183406990LL;
constexpr int64_t HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE = 623508127183407110LL;
constexpr int64_t HYVLA_EXPERT_NVFP4_RESIDENT_SCALES_SITE = 623508127183407140LL;
constexpr int64_t HYVLA_EXPERT_KV_GROUPED_SITE = 623508127183407180LL;
constexpr int64_t HYVLA_EXPERT_PAIRED_KV_SITE = 623508127183406930LL;
constexpr int64_t HYVLA_EXPERT_KV_PADDING_SITE = 3353793944280539666LL;
constexpr int64_t HYVLA_EXPERT_GRAPH_SCHEDULE_SITE = 5602663956511155425LL;
constexpr int64_t HYVLA_EXPERT_MASK_SCHEDULE_SITE = 1728596774874433151LL;

constexpr int64_t HYVLA_EXPERT_PRE_X0_DMA_SITE = 8468013054253886391LL;
constexpr int64_t HYVLA_EXPERT_PRE_X_FEEDBACK_DMA_SITE = 8068344894213309215LL;
constexpr int64_t HYVLA_EXPERT_PRE_TIME_BROADCAST_DMA_SITE = 6202700778193353884LL;
constexpr int64_t HYVLA_EXPERT_PRE_TIME_SCATTER_DMA_SITE = 5681167721722131756LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_BIAS_DMA_SITE = 3234409676337683315LL;
constexpr int64_t HYVLA_EXPERT_PRE_WC_LINEAR_SITE = 94006821360779226LL;
constexpr int64_t HYVLA_EXPERT_PRE_ENCODER_ACTIVATION_SITE = 4663734415968545282LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_SINGLE_LINEAR_SITE = 8695816214428940237LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_MULTI_LINEAR_SITE = 2981602481703720578LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_ZERO_SITE = 4262731488614715412LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_ALL_REDUCE_SITE = 414612096590535714LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_BIAS_ADD_SITE = 2288102790066046320LL;
constexpr int64_t HYVLA_EXPERT_POST_OP_LINEAR_SITE = 2567201997024428744LL;
constexpr int64_t HYVLA_EXPERT_POST_TRAJECTORY_DMA_SITE = 601652272865429711LL;

// Separate actual kernel receipts from the historical row-padding receipts.
constexpr int64_t HYVLA_EXPERT_FULL_RMSNORM_ROUTE_SITE = 623508127183406911LL;
constexpr int64_t HYVLA_EXPERT_Q_RMSNORM_ROUTE_SITE = 623508127183406912LL;
constexpr int64_t HYVLA_EXPERT_K_RMSNORM_ROUTE_SITE = 623508127183406913LL;

constexpr int64_t HYVLA_EXPERT_INPUT_NORM_SITE = 2943148670015766034LL;
constexpr int64_t HYVLA_EXPERT_QKV_LINEAR_SITE = 1236943476379916654LL;
constexpr int64_t HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE = 4314852815466214180LL;
constexpr int64_t HYVLA_EXPERT_Q_ROPE_SITE = 2913243967564739575LL;
constexpr int64_t HYVLA_EXPERT_Q_NORM_SITE = 2998501871001866367LL;
constexpr int64_t HYVLA_EXPERT_K_PARTIAL_ROPE_SITE = 6204149761964354754LL;
constexpr int64_t HYVLA_EXPERT_K_ROPE_SITE = 2703116739699322018LL;
constexpr int64_t HYVLA_EXPERT_K_NORM_SITE = 739473353641483894LL;
constexpr int64_t HYVLA_EXPERT_PREPARE_ATTN_ALL_REDUCE_SITE = 7991151858475231951LL;
constexpr int64_t HYVLA_EXPERT_O_LINEAR_SITE = 36160785207660705LL;
constexpr int64_t HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE = 5454728429937755845LL;
constexpr int64_t HYVLA_EXPERT_POST_NORM_SITE = 48200961442401552LL;
constexpr int64_t HYVLA_EXPERT_GATE_LINEAR_SITE = 2105943201978379215LL;
constexpr int64_t HYVLA_EXPERT_MLP_SILU_SITE = 1848219361374021523LL;
constexpr int64_t HYVLA_EXPERT_UP_LINEAR_SITE = 4699339857693172978LL;
constexpr int64_t HYVLA_EXPERT_MLP_SILU_MUL_SITE = 82400951984648727LL;
constexpr int64_t HYVLA_EXPERT_MLP_MUL_SITE = 6313036385062334532LL;
constexpr int64_t HYVLA_EXPERT_DOWN_LINEAR_SITE = 2699481217144264799LL;
constexpr int64_t HYVLA_EXPERT_DOWN_NVFP4_N128_SITE = 623508127183407050LL;
constexpr int64_t HYVLA_EXPERT_DOWN_NVFP4_NEXTNORM_SITE = 623508127183407070LL;
constexpr int64_t HYVLA_EXPERT_MLP_ALL_REDUCE_SITE = 2337931566137374851LL;
constexpr int64_t HYVLA_EXPERT_FINAL_NORM_SITE = 6930260388322325380LL;

constexpr uint32_t HYVLA_EXPERT_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 | KV_INSERT_CAP_PAD16 |
    KV_INSERT_CAP_HYBRID2 | KV_INSERT_CAP_NON8_TP_V16;
constexpr int64_t HYVLA_EXPERT_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED = 2;

enum class HyVlaExpertGraphScheduleRoute : int64_t {
    REEMIT_LAYER_LOOP = 1,
    FAST_REPLAY_SKIP_LAYER_LOOP = 2,
    MASK_EACH_LAYER = 3,
    MASK_FIRST_LAYER_ONLY = 4,
    ZERO_A6_RESIDUAL = 5,
    ADD_A6_BIAS_AFTER_REDUCE = 6,
    MASK_FIRST_BODY_FIRST_LAYER = 7,
};

enum class HyVlaExpertNormalizationRoute : int64_t {
    RMSNORM_SCALAR_ROWS = 1,
    RMSNORM_V16_ROWS = 2,
};

enum class HyVlaExpertActivationRoute : int64_t {
    SILU_UNARY = 1,
    MUL = 2,
    SILU_MUL_FUSED = 3,
};

enum class HyVlaExpertRopeRoute : int64_t {
    ROPE_1D = 1,
    PARTIAL_MROPE = 2,
};

enum class HyVlaExpertKvPaddingRoute : int64_t {
    LOGICAL_ROWS = 1,
    PAD16_ROWS = 2,
};

enum class HyVlaExpertDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM_MUTABLE = 1,
    DDR_BROADCAST_TO_SPM_FIXED = 2,
    DDR_SCATTER_TO_SPM_FIXED = 3,
    SPM_TO_DDR_MUTABLE = 4,
};

enum class HyVlaExpertAllReduceRoute : int64_t {
    PREPARE_RING_INPUT = 3,
};

HyVlaExpertModel::HyVlaExpertModel() = default;
HyVlaExpertModel::~HyVlaExpertModel() = default;

void HyVlaExpertModel::set_execution_cores(int64_t num_cores) {
    TORCH_CHECK(num_cores == 8 || num_cores == 4,
                "HyVLA action expert admits eight cores or the reduced four-core child");
    TORCH_CHECK(!cold_config_bound_ && !production_config_bound_ && layers_.empty(),
                "HyVLA action expert core count must be bound before runtime config and weights");
    if (num_cores != 8) {
        configure_cold_routes(/*qwen3_spm_kv_by_mha=*/true, /*adarms_fused_bcast=*/false,
                              static_cast<int>(num_cores), /*vocab_size=*/0);
    }
}

void HyVlaExpertModel::set_runtime_config(
    bool fast_replay, bool fast_replay_preload, bool mask_once,
    bool silu_mul, bool kvpad16, bool partial_rope,
    bool rmsnorm_pad16) {
    TORCH_CHECK(
        !cold_config_bound_ && !production_config_bound_,
        "hyvla_expert_set_runtime_config must run exactly once before "
        "set_weights");
    // The V16 RMSNorm payloads launch an eight-core grid.
    TORCH_CHECK(num_cores() == 8 || !rmsnorm_pad16,
                "reduced-core HyVLA action expert requires the BASE RMSNorm route");
    // PAD16 is the existing opt-in. The generic launcher's default is BASE;
    // padding alone does not select a vector kernel. Bind availability before
    // committing the one-shot configuration. Older V16 payloads corrupt BF16
    // Inf/NaN partials through an FP16 arithmetic copy; select the expansion's
    // actual corrected payload through a distinct physical route.
    const bool vector_available = rmsnorm_pad16 &&
        KernelCache::instance().get_kernel("hyvla_expert_rmsnorm_v16_bitcopy") != nullptr;
    auto rmsnorm_contract = rpu_snapshot_rmsnorm_spm_contract(
        RPU_RMSNORM_CAP_BASE | (vector_available ? RPU_RMSNORM_CAP_V16 : 0u));
    rmsnorm_contract.v16_bitcopy = vector_available;
    const bool gateup_swiglu_available =
        KernelCache::instance().get_kernel("hyvla_expert_gateup_swiglu_fp16_v2") != nullptr;
    const bool resident_swiglu_available =
        KernelCache::instance().get_kernel("hyvla_resident_swiglu_fp16_v2") != nullptr;
    const bool w8_optimizations_available =
        KernelCache::instance().get_kernel("hyvla_gateup_resident_pair_w8_v2") != nullptr &&
        KernelCache::instance().get_kernel("hyvla_expert_gateup_swiglu_w8_v2") != nullptr;
    const bool w8_gateup_n48_available =
        KernelCache::instance().get_kernel("hyvla_expert_gateup_swiglu_w8_v2") != nullptr;
    invalidate_model_state();
    cold_rmsnorm_spm_contract_ = rmsnorm_contract;
    cold_gateup_swiglu_available_ = gateup_swiglu_available;
    cold_w8_optimizations_available_ = w8_optimizations_available;
    cold_nvfp4_gateup_available_ =
        KernelCache::instance().get_kernel("hyvla_nvfp4_gateup_swiglu_s51") != nullptr &&
        KernelCache::instance().get_kernel("hyvla_nvfp4_gateup_resident_s51") != nullptr;
    cold_nvfp4_gateup_input_reuse_available_ = hyvla_retained_schedule::gateup_input_reuse &&
        KernelCache::instance().get_kernel("hyvla_nvfp4_gateup_swiglu_input_reuse_s51") != nullptr;
    cold_nvfp4_o_ring_postnorm_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_o_ring_postnorm_nvfp4_s51") != nullptr;
    cold_nvfp4_o_n128_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_o_ring_postnorm_nvfp4_s51_n128") != nullptr;
    cold_nvfp4_down_nextnorm_n128_available_ = hyvla_retained_schedule::down_n128 &&
        KernelCache::instance().get_kernel("hyvla_expert_down_nextnorm_nvfp4_s51_n128") != nullptr;
    cold_nvfp4_down_nextnorm_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_down_nextnorm_nvfp4_s51") != nullptr;
    cold_nvfp4_down_n128_available_ = hyvla_retained_schedule::down_n128 &&
        KernelCache::instance().get_kernel("hyvla_expert_down_nvfp4_s51_n128") != nullptr;
    cold_w8_gateup_n48_available_ = w8_gateup_n48_available;
    cold_resident_swiglu_available_ = resident_swiglu_available;
    cold_w8_o_resident_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_o_resident_w8_v2") != nullptr;
    cold_w8_o_ring_postnorm_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_o_ring_postnorm_w8_s51_v2") != nullptr;
    cold_xor3_norm_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_xor3_norm_s51") != nullptr;
    cold_w8_resident_swiglu_available_ =
        KernelCache::instance().get_kernel("hyvla_resident_swiglu_w8_v2") != nullptr;
    cold_nvfp4_resident_swiglu_available_ =
        KernelCache::instance().get_kernel("hyvla_nvfp4_resident_swiglu_s51") != nullptr;
    cold_nvfp4_resident_scales_available_ =
        KernelCache::instance().get_kernel("hyvla_nvfp4_resident_scales_s51") != nullptr;
    cold_nvfp4_resident_scales_swiglu_available_ =
        KernelCache::instance().get_kernel("hyvla_nvfp4_resident_scales_swiglu_s51") != nullptr;
    cold_down_nextnorm_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_down_nextnorm_s51") != nullptr;
    cold_nvfp4_kv_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_kv_nvfp4_s51") != nullptr;
    cold_fp16_kv_grouped_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_kv_fp16_s51_v2") != nullptr;
    cold_w8_kv_grouped_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_kv_w8_s51_v2") != nullptr;
    cold_nvfp4_qkv_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_qkv_nvfp4_s51") != nullptr;
    cold_nvfp4_q_n32_available_ = hyvla_retained_schedule::q_n32 &&
        KernelCache::instance().get_kernel("hyvla_expert_q_nvfp4_s51_n32") != nullptr;
    cold_nvfp4_qkv_overlap_available_ = hyvla_retained_schedule::qkv_overlap &&
        KernelCache::instance().get_kernel("hyvla_expert_qkv_nvfp4_s51_overlap") != nullptr;
    cold_nvfp4_o_overlap_available_ = hyvla_retained_schedule::o_overlap &&
        KernelCache::instance().get_kernel("hyvla_expert_o_ring_postnorm_nvfp4_s51_n128_overlap") != nullptr;
    cold_nvfp4_down_overlap_available_ = hyvla_retained_schedule::down_overlap &&
        KernelCache::instance().get_kernel("hyvla_expert_down_nextnorm_nvfp4_s51_overlap") != nullptr;
    cold_nvfp4_gateup_overlap_available_ = hyvla_retained_schedule::gateup_overlap &&
        KernelCache::instance().get_kernel("hyvla_nvfp4_gateup_swiglu_s51_overlap") != nullptr;
    cold_nvfp4_resident_overlap_available_ = hyvla_retained_schedule::resident_overlap &&
        KernelCache::instance().get_kernel("hyvla_nvfp4_resident_swiglu_s51_overlap") != nullptr;
    cold_sdpa_k128_available_ = hyvla_retained_schedule::sdpa_k128 &&
        KernelCache::instance().get_kernel("hyvla_expert_sdpa_k128_s51") != nullptr;
    cold_sdpa_m16_available_ = hyvla_retained_schedule::sdpa_m16 &&
        KernelCache::instance().get_kernel("hyvla_expert_sdpa_m16_s51") != nullptr;
    cold_qk_rope_norm_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_qk_rope_norm_s51") != nullptr;
    cold_qk_rope_norm_kv_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_qk_rope_norm_kv_s51") != nullptr;
    cold_qk_rope_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_qk_rope_s51") != nullptr;
    cold_o_ring_postnorm_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_o_ring_postnorm_fp16_s51_v2") != nullptr;
    cold_ring_postnorm_available_ =
        KernelCache::instance().get_kernel("hyvla_expert_ring_postnorm_s51") != nullptr;
    // Task-private expansion opts in; ordinary REFs retain separate K/V writes.
    cold_paired_kv_available_ = kvpad16 &&
        KernelCache::instance().get_kernel("hyvla_expert_paired_kv_s64") != nullptr &&
        KernelCache::instance().get_program("llama_insert_kv_cache_multiwarp_v16") != nullptr;
    cold_fast_replay_ = fast_replay;
    cold_fast_replay_preload_ = fast_replay_preload;
    cold_mask_once_ = mask_once;
    cold_silu_mul_ = silu_mul;
    cold_kvpad16_ = kvpad16;
    cold_partial_rope_ = partial_rope;
    cold_rmsnorm_pad16_ = rmsnorm_pad16;
    cold_config_bound_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// set_weights_expert — 基类委托 + 子类影子
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::set_weights_expert(
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps,
    int64_t chunk_size, int64_t action_mlp_cores,
    at::TensorList q_ws_list, at::TensorList k_ws_list,
    at::TensorList v_ws_list, at::TensorList o_ws_list,
    at::TensorList gate_ws_list, at::TensorList up_ws_list,
    at::TensorList down_ws_list,
        const at::Tensor& q_ts,
        const at::Tensor& k_ts,
        const at::Tensor& v_ts,
        const at::Tensor& o_ts,
        const at::Tensor& gate_ts,
        const at::Tensor& up_ts,
        const at::Tensor& down_ts)
{
    TORCH_CHECK(
        cold_config_bound_,
        "hyvla_expert set_weights requires an explicit immutable runtime "
        "config snapshot");
    TORCH_CHECK(chunk_size == 51,
                "hyvla_expert set_weights: the certified Hy-VLA suffix has "
                "exactly 51 rows, got ", chunk_size);
    TORCH_CHECK(action_mlp_cores == 1 || action_mlp_cores == num_cores(),
                "hyvla_expert set_weights: action_mlp_cores must be 1 or ",
                num_cores(), ", got ", action_mlp_cores);
    if (production_config_bound_) {
        TORCH_CHECK(action_mlp_cores_ == action_mlp_cores,
                    "hyvla_expert set_weights: action MLP core count is "
                    "immutable for a native handle");
    }
    TORCH_CHECK(!gateup_resident_bundle_.defined(),
                "HyVLA resident weights require a fresh owner before replacing expert weights");
    // Fail the cold-only check before the base setter retains any weights.
    set_chunk_envelope(/*max_kv_len=*/291, /*chunk=*/64);
    // 校验/状态提交全部委托基类 (含 q/k norm 全套检查 + quant scale 的
    // 全给/全空、dtype、rank、numel==N 校验)。Hy-VLA attention_bias=false
    // ⇒ QKV bias 三个列表恒空; MLP 固定 SwiGLU ⇒ use_silu=true。
    CausalDecoderModel::set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list, q_norm_list, k_norm_list,
        input_norm_list, post_norm_list, gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
        eps, /*use_silu=*/true,
        /*mrope_section=*/{}, /*deepstack_lang_layers=*/{},
        /*q_bias=*/{}, /*k_bias=*/{}, /*v_bias=*/{},
        q_ws_list, k_ws_list, v_ws_list, o_ws_list,
        gate_ws_list, up_ws_list, down_ws_list,
        q_ts, k_ts, v_ts, o_ts, gate_ts, up_ts, down_ts);

    TORCH_CHECK(has_qk_norm_,
                "hyvla_expert set_weights: q/k_norm lists must be non-empty "
                "(Hy-VLA is Qwen3-style). ⚠️ 且必须是 **VLM 层** 的 "
                "query/key_layernorm —— 两塔共享 VLM 那一份, expert 自己的同名 "
                "张量是死权重 (P0 hook calls=0)。");

    // 影子副本 — build_layer_subgraph 要逐层权重, 而基类 layer_weights_ 是 private。
    // W4 scale 必须按实际 attn_tp()/action_mlp_cores retain 对齐视图。
    const int64_t N = static_cast<int64_t>(q_w_list.size());
    const bool quantized = !q_ws_list.empty();
    layers_.clear();
    layers_.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
        layers_.push_back({q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                           gate_list[i], up_list[i], down_list[i],
                           quantized ? rpu_retain_linear_quant_scale(q_w_list[i], q_ws_list[i], attn_tp()) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(k_w_list[i], k_ws_list[i], attn_tp()) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(v_w_list[i], v_ws_list[i], attn_tp()) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(o_w_list[i], o_ws_list[i], attn_tp()) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(gate_list[i], gate_ws_list[i], action_mlp_cores) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(up_list[i], up_ws_list[i], action_mlp_cores) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(down_list[i], down_ws_list[i], action_mlp_cores) : at::Tensor()});
    }
    cos_ref_ = cos;
    sin_ref_ = sin;
    eps_expert_ = eps;
    expert_chunk_size_ = chunk_size;
    action_mlp_cores_ = static_cast<int>(action_mlp_cores);
    production_config_bound_ = true;

    invalidate_model_state();   // D-503: set_weights 的末条非空语句
}

// ─────────────────────────────────────────────────────────────────────────────
// set_action_weights — 打开 in-graph unroll 模式
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::set_action_weights(
    const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
    const at::Tensor& op, const at::Tensor& op_bias, const at::Tensor& time_all,
    int64_t action_dim, int64_t num_steps)
{
    TORCH_CHECK(!gateup_resident_bundle_.defined(),
                "HyVLA resident weights require a fresh owner before replacing action weights");
    TORCH_CHECK(hidden_size() > 0 && expert_chunk_size_ > 0,
                "hyvla_expert set_action_weights 必须在 set_weights_expert 之后调用");
    TORCH_CHECK(action_dim > 0 && num_steps > 0,
                "action_dim/num_steps 必须为正, 得到 ", action_dim, "/", num_steps);
    const int64_t h = hidden_size();
    action_dim_     = action_dim;
    action_dim_pad_ = ((action_dim + 15) / 16) * 16;
    // 图内 Euler 是 `x_t_spm[cs,pad] += dt·v_t[cs,pad]` 的逐元素加 —— x 与 v 的
    // 行宽必须相同, 所以 encoder 的 K 与 out_proj 的 N 都取 action_dim_pad。
    TORCH_CHECK(action_dim_pad_ % 16 == 0, "action_dim_pad 必须是 16 的倍数");
    TORCH_CHECK(mo_bias.dim() == 1 && mo_bias.size(0) == h
             && mo_bias.scalar_type() == at::kHalf
             && mo_bias.device().type() == at::kPrivateUse1,
                "mo_bias 必须是 [hidden] fp16 device");
    TORCH_CHECK(op_bias.dim() == 1 && op_bias.size(0) == action_dim_pad_
             && op_bias.scalar_type() == at::kHalf
             && op_bias.device().type() == at::kPrivateUse1,
                "op_bias 必须是 [action_dim_pad] fp16 device");
    TORCH_CHECK(time_all.dim() == 2 && time_all.size(0) == num_steps
             && time_all.size(1) == h && time_all.scalar_type() == at::kHalf
             && time_all.is_contiguous()
             && time_all.device().type() == at::kPrivateUse1,
                "time_all 必须是 [num_steps, hidden] fp16 contig device");
    wc_ = wc; mo_ = mo; mo_bias_ = mo_bias;
    op_ = op; op_bias_ = op_bias; time_all_ = time_all;
    num_steps_ = num_steps;
    // 稳定 staging DDR: pre-hook 把 encoder 的行 1:cs 写这里, 层 0 的
    // emit_layer_input_dma 经 FMB hidden_in_src_base_ 读回 (固定 src —— data_ptr 稳定)。
    emb_stage_ = at::empty({1, expert_chunk_size_, h},
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
    unroll_mode_ = true;
    invalidate_model_state();
}

// ─────────────────────────────────────────────────────────────────────────────
// declare_buffers / static_config — 只在 unroll 下追加
// ─────────────────────────────────────────────────────────────────────────────
bool HyVlaExpertModel::gateup_resident_profile() const {
    if (!unroll_mode_ || num_steps_ != 10 || !cold_silu_mul_ ||
        num_layers() != 32 || hidden_size() != 1024 || intermediate_size() != 2048 ||
        num_q_heads() != 16 || num_kv_heads() != 8 || head_dim() != 128 ||
        attn_tp() != NUM_CORES || expert_chunk_size_ != 51 ||
        action_mlp_cores_ != NUM_CORES || layers_.size() != 32) return false;
    for (const auto& w : layers_) {
        if (w.q_w.scalar_type() != layers_.front().q_w.scalar_type() ||
            !hyvla_linear_precision(w, cold_w8_optimizations_available_,
                                     nvfp4_ && cold_nvfp4_gateup_available_)) return false;
    }
    return true;
}

std::vector<int64_t> HyVlaExpertModel::gateup_resident_arguments() const {
    auto args = std::vector<int64_t>{nvfp4_ ? 3 : (gateup_weight_bytes() == 262144 ? 2 : 1),
            gateup_resident_layers(), num_layers(), expert_chunk_size_,
            hidden_size(), intermediate_size(), NUM_CORES, num_steps_,
            gateup_layer_bytes(), gateup_bundle_bytes()};
    if (nvfp4_) args.insert(args.end(), {320, 64, 128, 8, 24, 27, 32});
    else {
        const auto c = rpu_specialized_linear::contract(gateup_weight_bytes() == 262144
            ? KernelId::HYVLA_GATEUP_RESIDENT_PAIR_W8_V2
            : KernelId::HYVLA_GATEUP_RESIDENT_PAIR_FP16_V2);
        args[0] += 3;
        args.insert(args.end(), {c.m, c.n, c.k, c.gx, 2, 32, 20, 34, 4, 36});
    }
    return args;
}

bool HyVlaExpertModel::optimized_prefix_supported(int64_t position) const {
    // Two or three cameras plus up to 64 text tokens. Resident projections
    // operate on the same S51 suffix; QK/KV helpers bind the actual prefix
    // position and validate the complete backing extent independently.
    return position >= 128 && position <= 240 && position % 16 == 0;
}

bool HyVlaExpertModel::gateup_swiglu_enabled(int64_t rows, int64_t position) const {
    return (nvfp4_ ? cold_nvfp4_gateup_available_ : cold_gateup_swiglu_available_) &&
           gateup_resident_bundle_.defined() &&
           rows == 51 && optimized_prefix_supported(position) && gateup_resident_profile();
}

std::vector<int64_t> HyVlaExpertModel::gateup_swiglu_arguments(int64_t position) const {
    // Version, layer interval, shape/TP/body count, payload tile and grid.
    if (nvfp4_) {
        std::vector<int64_t> args{cold_nvfp4_gateup_input_reuse_available_ ? 5 : 4,
            32, gateup_resident_layers(), 32 - gateup_resident_layers(),
            51, position, 1024, 2048, 256, 8, 10, 480, 32, 128, 8, 24, 36, 27, 128, 130};
        // Whole-input VLM cache base/extent/row stride, result/tensor-scale
        // bases and projection reuse count; the old REF keeps its ABI4 recipe.
        if (cold_nvfp4_gateup_input_reuse_available_)
            args.insert(args.end(), {28, 256, 64, 296, 356, 2});
        if (cold_nvfp4_gateup_overlap_available_) {
            TORCH_CHECK(!cold_nvfp4_gateup_input_reuse_available_,
                        "HyVLA DDR GateUp overlap cannot combine with whole-input reuse");
            args[0] = 6;
            args.insert(args.end(), {106, 105, 0, 1, 1});
        }
        return args;
    }
    const bool w8 = gateup_weight_bytes() == 262144;
    const auto c = rpu_specialized_linear::contract(w8
        ? KernelId::HYVLA_EXPERT_GATEUP_W8_S51_N48
        : KernelId::HYVLA_EXPERT_GATEUP_SWIGLU_FP16_V2);
    return {w8 ? 5 : 4, 32, gateup_resident_layers(), 32 - gateup_resident_layers(),
            51, position, 1024, 2048, 256, NUM_CORES, 10, c.m, c.n, c.k, c.gx,
            2, 32, 20, 34, 26, 27, 28, 29};
}

bool HyVlaExpertModel::resident_swiglu_enabled(int64_t rows, int64_t position) const {
    // The remaining DDR layers also use Gate/Up fusion, leaving no separate activation.
    if (nvfp4_)
        return cold_nvfp4_resident_swiglu_available_ &&
               (gateup_resident_layers() == 12 || gateup_resident_layers() == 24) &&
               gateup_swiglu_enabled(rows, position);
    return (gateup_weight_bytes() == 524288 ? cold_resident_swiglu_available_
                                          : cold_w8_resident_swiglu_available_) &&
           gateup_swiglu_enabled(rows, position);
}

std::vector<int64_t> HyVlaExpertModel::resident_swiglu_arguments(int64_t position) const {
    // ABI/layers/start/count/M/P/H/I/localN/cores/bodies/tileM/N/K/grid/counter/rounding.
    if (nvfp4_) {
        std::vector<int64_t> args{3, 32, 0, gateup_resident_layers(),
                51, position, 1024, 2048, 256, 8, 10, 320, 64, 128, 8, 128, 1,
                // Activation virtual grid/elements/normal/last/parameter base;
                // independent table/layer and producer weight/scale/output registers.
                1, 13056, 32512, 13056, 48, 24, 36, 27, 32, 2, 32, 20, 34, 38};
        if (cold_nvfp4_resident_overlap_available_) {
            TORCH_CHECK(resident_fp8_scale_layers_ == 0,
                        "HyVLA resident overlap requires original DDR block scales");
            args[0] = 4;
            args.insert(args.end(), {106, 105, 0, 1, 1});
        }
        return args;
    }
    const bool w8 = gateup_weight_bytes() == 262144;
    const auto c = rpu_specialized_linear::contract(w8
        ? KernelId::HYVLA_W8_RESIDENT_SWIGLU_S51
        : KernelId::HYVLA_RESIDENT_SWIGLU_FP16_V2);
    return {w8 ? 5 : 4, 32, 0, gateup_resident_layers(),
            51, position, 1024, 2048, 256, 8, 10, c.m, c.n, c.k, c.gx, 0, 1,
            2, 32, 20, 34, 26, 27, 28, 29};
}

bool HyVlaExpertModel::resident_fp8_scales_enabled(int64_t rows, int64_t position) const {
    return (resident_fp8_scale_layers_ == 4 || resident_fp8_scale_layers_ == 8) &&
           nvfp4_ && o_resident_layers_ == 0 && gateup_resident_layers() == 24 &&
           resident_storage_bytes() == 6291456 + resident_fp8_scale_layers_ * 32768 &&
           (cold_nvfp4_resident_swiglu_available_ ? cold_nvfp4_resident_scales_swiglu_available_
                                                 : cold_nvfp4_resident_scales_available_) &&
           gateup_swiglu_enabled(rows, position);
}

std::vector<int64_t> HyVlaExpertModel::resident_fp8_scales_arguments(int64_t position) const {
    // ABI/layer interval/M/P/H/I/localN/TP/bodies; code and scale extents;
    // canonical tile/grid; SPM scale ABI, FP32 tables/layer and activation mode.
    return {1, 32, 0, resident_fp8_scale_layers_, 51, position, 1024, 2048, 256, 8, 10,
            24, 6291456, resident_fp8_scale_layers_ * 32768, resident_storage_bytes(), 32768, 16384,
            320, 64, 128, 8, 20, 34, 512, 24, 36, 27,
            cold_nvfp4_resident_swiglu_available_ ? 1 : 0};
}

bool HyVlaExpertModel::ring_postnorm_enabled(int64_t rows, int64_t position) const {
    return cold_ring_postnorm_available_ && gateup_swiglu_enabled(rows, position) &&
           cold_rmsnorm_pad16_ && eps_expert_ == 1e-5 &&
           cold_rmsnorm_spm_contract_.resolve(64, 1024) == RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8;
}

bool HyVlaExpertModel::xor3_norm_enabled() const {
    return cold_xor3_norm_available_ && gateup_weight_bytes() == 262144;
}

std::vector<int64_t> HyVlaExpertModel::ring_postnorm_arguments(int64_t position) const {
    // ABI/L/M/R/P/H/cores/steps/chunk/roundmax/rounds/grid/core-counter/ring-counters/pace/bitcopy.
    auto args = std::vector<int64_t>{1, 32, 51, 64, position, 1024, 8, 10, 6528, 63488, 1, 8, 0, 24, 31, 1, 1};
    if (xor3_norm_enabled()) { args[0] = 2; args.push_back(3); }
    return args;
}

bool HyVlaExpertModel::down_nextnorm_enabled(int64_t rows, int64_t position) const {
    return cold_down_nextnorm_available_ && qk_rope_enabled(rows, position);
}

bool HyVlaExpertModel::down_nvfp4_nextnorm_enabled(int64_t rows, int64_t position) const {
    return nvfp4_ && cold_nvfp4_down_nextnorm_available_ &&
           down_nextnorm_enabled(rows, position) && !xor3_norm_enabled();
}

std::vector<int64_t> HyVlaExpertModel::down_nvfp4_nextnorm_arguments(int64_t position) const {
    auto args = down_nextnorm_arguments(position);
    const bool n128 = cold_nvfp4_down_nextnorm_n128_available_;
    args[0] = n128 ? 3 : 2;
    // Producer tile/waves/grid/localK/stride, typed DDR/table registers,
    // remapped tail register range and replay-initialized core counter pair.
    args.insert(args.end(), {n128 ? 176 : 320, n128 ? 128 : 64, 128, n128 ? 1 : 2, 8, 256, 2048,
                            2, 3, 20, 21, 24, 27, 32, 32, 50, 128});
    if (cold_nvfp4_down_overlap_available_) {
        TORCH_CHECK(!n128, "HyVLA Down overlap requires the retained N64 producer");
        args[0] = 4;
        args.insert(args.end(), {106, 105, 0, 1, 1});
    }
    return args;
}

bool HyVlaExpertModel::down_nvfp4_n128_enabled(int64_t rows, int64_t position) const {
    return nvfp4_ && cold_nvfp4_down_n128_available_ && rows == 51 &&
           optimized_prefix_supported(position) && gateup_resident_profile() &&
           !down_nvfp4_nextnorm_enabled(rows, position);
}

std::vector<int64_t> HyVlaExpertModel::down_nvfp4_n128_arguments(int64_t position) const {
    // ABI/L/M/P/H/globalK/localK/TP/steps/tileM/N/K/waves/grid/partition,
    // FP32 table register/layer register/table length, typed weight/scale pairs.
    return {1, 32, 51, position, 1024, 2048, 256, 8, 10, 176, 128, 128,
            1, 8, 0, 24, 27, 32, 2, 3, 20, 21};
}

std::vector<int64_t> HyVlaExpertModel::down_nextnorm_arguments(int64_t position) const {
    auto args = ring_postnorm_arguments(position);
    // First Down layer, cross-layer count, gamma offset, final layer/inplace, initial norm layer.
    args.insert(args.end(), {0, 31, 1, 31, 1, 0});
    return args;
}

bool HyVlaExpertModel::qk_rope_enabled(int64_t rows, int64_t position) const {
    return cold_qk_rope_available_ && cold_partial_rope_ && ring_postnorm_enabled(rows, position);
}

std::vector<int64_t> HyVlaExpertModel::qk_rope_arguments(int64_t position) const {
    // ABI/L/M/P/local Q/K/D/rotary/TP/steps/grid/shared frequency loads/DDR registers.
    return {1, 32, 51, position, 2, 1, 128, 128, 8, 10, 1, 1, 1, 1, 0, 1, 2, 3};
}

bool HyVlaExpertModel::kv_grouped_enabled(int64_t rows, int64_t position) const {
    return !nvfp4_ && gateup_swiglu_enabled(rows, position) &&
           (gateup_weight_bytes() == 524288 ? cold_fp16_kv_grouped_available_
                                          : cold_w8_kv_grouped_available_);
}

std::vector<int64_t> HyVlaExpertModel::kv_grouped_arguments(int64_t position) const {
    const bool w8 = gateup_weight_bytes() == 262144;
    const auto c = rpu_specialized_linear::contract(w8
        ? KernelId::HYVLA_EXPERT_KV_W8_S51 : KernelId::HYVLA_EXPERT_KV_FP16_S51);
    return {2, w8 ? 8 : 16, 32, 51, position, 1024, 128, 1024, 8, 10, 2,
            c.m, c.n, c.k, c.gx, (128 + c.n - 1) / c.n,
            2, 3, 32, 33, w8 ? 20 : 0, w8 ? 21 : 0,
            w8 ? 34 : 0, w8 ? 35 : 0, 4, 5, 36, 37, 1, w8 ? 1 : 2};
}

int HyVlaExpertModel::nvfp4_qkv_mode(int64_t rows, int64_t position) const {
    if (!nvfp4_ || !gateup_swiglu_enabled(rows, position)) return 0;
    TORCH_CHECK(!(cold_nvfp4_q_n32_available_ && cold_nvfp4_qkv_overlap_available_),
                "HyVLA NVFP4 Q N32 and QKV overlap require separate candidate libraries");
    if (cold_nvfp4_q_n32_available_) {
        TORCH_CHECK(cold_nvfp4_kv_available_, "HyVLA NVFP4 Q N32 requires paired KV payload");
        return 2;
    }
    TORCH_CHECK(!cold_nvfp4_qkv_overlap_available_ || cold_nvfp4_qkv_available_,
                "HyVLA NVFP4 QKV overlap requires canonical QKV payload");
    return cold_nvfp4_qkv_available_ ? 3 : cold_nvfp4_kv_available_ ? 2 : 0;
}

bool HyVlaExpertModel::nvfp4_q_n32_enabled(int64_t rows, int64_t position) const {
    return nvfp4_qkv_mode(rows, position) == 2 && cold_nvfp4_q_n32_available_;
}

std::vector<int64_t> HyVlaExpertModel::nvfp4_q_n32_arguments(int64_t position) const {
    // ABI/layers/rows/position/K/global N/local N/TP/steps/M/N/K/grid/ACC16;
    // typed DDR weight/FP8 scale registers, FP32 table/layer/count and packing.
    return {1, 32, 51, position, 1024, 2048, 256, 8, 10, 480, 32, 128, 8, 1,
            2, 3, 20, 21, 24, 27, 32, 16, 32};
}

bool HyVlaExpertModel::sdpa_k128_enabled(int64_t rows, int64_t position) const {
    TORCH_CHECK(!(nvfp4_ && cold_sdpa_k128_available_ && cold_sdpa_m16_available_),
                "HyVLA attention K128 and M16 are independent exclusive schedules");
    return nvfp4_ && cold_sdpa_k128_available_ && gateup_swiglu_enabled(rows, position) &&
           gateup_resident_layers() == 24;
}

std::vector<int64_t> HyVlaExpertModel::sdpa_k128_arguments(int64_t position) const {
    // Independent K128 recipe; online softmax block order differs from K256.
    return {1, 32, 51, position, position + 51, 16, 8, 128, 8, 2, 4, 10, 24,
            64, 128, 128, 1, 2, 8, 32768, Align(position + 51, int64_t{16}), 4, 8, 8, 1, 128};
}

bool HyVlaExpertModel::sdpa_m16_enabled(int64_t rows, int64_t position) const {
    return nvfp4_ && cold_sdpa_m16_available_ && gateup_swiglu_enabled(rows, position) &&
           gateup_resident_layers() == 24;
}

std::vector<int64_t> HyVlaExpertModel::sdpa_m16_arguments(int64_t position) const {
    // ABI/layers/Q/P/KV/Qheads/KVheads/D/TP/GQA/mask/steps/resident layers;
    // tile M/N/K, grid X/Y/cores, scratch bytes, mask pitch, vector tiling,
    // attention scale 1/sqrt(D). Payload arithmetic and K order are unchanged.
    return {1, 32, 51, position, position + 51, 16, 8, 128, 8, 2, 4, 10, 24,
            16, 128, 256, 4, 2, 8, 262144, Align(position + 51, int64_t{16}), 1, 8, 16, 1, 128};
}

std::vector<int64_t> HyVlaExpertModel::nvfp4_qkv_arguments(int mode, int64_t position) const {
    TORCH_CHECK(mode == 2 || mode == 3, "HyVLA NVFP4 grouped projections require KV or QKV");
    // ABI/layers/rows/position/K/TP/steps/projections/local N/tile/grid;
    // each weight, block scale, FP32 table and output remains independently bound.
    std::vector<int64_t> args{1, 32, 51, position, 1024, 8, 10, mode,
            mode == 3 ? 256 : 128, 128, mode == 3 ? 128 : 0,
            mode == 3 ? 320 : 480, mode == 3 ? 64 : 32, 128, 8,
            2, 20, 24, 4, 32, 34, 36, 38,
            mode == 3 ? 40 : 0, mode == 3 ? 42 : 0,
            mode == 3 ? 44 : 0, mode == 3 ? 46 : 0, 27, 16, 32};
    if (mode == 3 && cold_nvfp4_qkv_overlap_available_) {
        args[0] = 2;
        // Original/new K-loop instructions; tail VMAT fence removed, next
        // input-load fence and final output-store fence retained.
        args.insert(args.end(), {106, 105, 0, 1, 1});
    }
    return args;
}

bool HyVlaExpertModel::qk_rope_norm_enabled(int64_t rows, int64_t position) const {
    // Q/K and gamma are FP16 for every admitted projection precision. Keep
    // the original V16 arithmetic and cold shape/weight/REF admission below.
    return cold_qk_rope_norm_available_ && qk_rope_enabled(rows, position) &&
           cold_rmsnorm_spm_contract_.resolve(112, 128) == RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8 &&
           cold_rmsnorm_spm_contract_.resolve(64, 128) == RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8;
}

std::vector<int64_t> HyVlaExpertModel::qk_rope_norm_arguments(int64_t position) const {
    auto args = qk_rope_arguments(position);
    // Full padded head-norm rows, exact V16 route, params, barriers and eps.
    args.insert(args.end(), {112, 64, 128, 14, 8, 8, 32, 44, 128, 130, 2, 5, 1, 100000, 1});
    return args;
}

bool HyVlaExpertModel::qk_rope_norm_kv_enabled(
    int64_t rows, int64_t position, const KvInsertSegmentPlan& plan) const {
    return cold_qk_rope_norm_kv_available_ && qk_rope_norm_enabled(rows, position) &&
           paired_kv_enabled(plan) && plan.segment(0).position == position;
}

std::vector<int64_t> HyVlaExpertModel::qk_rope_norm_kv_arguments(
    const KvInsertSegmentPlan& plan, int64_t position) const {
    TORCH_CHECK(plan.segment_count() == 1 && plan.segment(0).position == position,
                "HyVLA fused QK norm+KV recipe requires the actual KV position");
    auto args = qk_rope_norm_arguments(position);
    const auto paired = paired_kv_arguments(plan);
    args.insert(args.end(), paired.begin(), paired.end());
    // Writer ABI2, grid, third phase counter, mutex and thirteen old/new
    // parameter pairs. Cache pairs20/21 and24/25 are inside SDK launch ABI0..63.
    args.insert(args.end(), {2, 4, 132, 3, 134, 13,
                            0, 8, 1, 9, 2, 14, 3, 10, 6, 16, 7, 17,
                            8, 20, 9, 21, 10, 22, 11, 23, 12, 24, 13, 25, 21, 26});
    return args;
}

bool HyVlaExpertModel::o_resident_enabled(int64_t rows, int64_t position) const {
    return o_resident_layers_ == 12 && cold_w8_o_resident_available_ &&
           gateup_weight_bytes() == 262144 && gateup_swiglu_enabled(rows, position);
}

std::vector<int64_t> HyVlaExpertModel::o_resident_arguments(int64_t position) const {
    const auto c = rpu_specialized_linear::contract(KernelId::HYVLA_EXPERT_O_RESIDENT_W8_S51);
    return {2, 32, 51, position, 1024, 2048, 8, 10, 0, 12, 3145728, 3145728,
            c.m, c.n, c.k, c.gx, 20, 21};
}

bool HyVlaExpertModel::o_ring_postnorm_enabled(int64_t rows, int64_t position) const {
    return o_resident_layers_ == 0 && !xor3_norm_enabled() &&
           (nvfp4_ ? (cold_nvfp4_o_ring_postnorm_available_ || cold_nvfp4_o_n128_available_) :
            (gateup_weight_bytes() == 524288 ? cold_o_ring_postnorm_available_
                                           : cold_w8_o_ring_postnorm_available_)) &&
           ring_postnorm_enabled(rows, position);
}

std::vector<int64_t> HyVlaExpertModel::o_ring_postnorm_arguments(int64_t position) const {
    auto args = ring_postnorm_arguments(position);
    // Original producer tile M/N/K, waves/tiles, local/global K, DDR registers.
    if (nvfp4_) {
        args[0] = cold_nvfp4_o_n128_available_ ? 4 : 3;
        if (cold_nvfp4_o_n128_available_)
            args.insert(args.end(), {176, 128, 128, 1, 8, 256, 2048, 2, 3, 20, 21, 24, 27, 32});
        else
            args.insert(args.end(), {320, 64, 128, 2, 16, 256, 2048, 2, 3, 20, 21, 24, 27, 32});
        if (cold_nvfp4_o_overlap_available_) {
            TORCH_CHECK(cold_nvfp4_o_n128_available_, "HyVLA O overlap requires the retained N128 producer");
            args[0] = 5;
            args.insert(args.end(), {106, 105, 0, 1, 1});
        }
    } else {
        const bool w8 = gateup_weight_bytes() == 262144;
        const auto c = rpu_specialized_linear::contract(w8
            ? KernelId::HYVLA_EXPERT_O_RING_POSTNORM_W8_S51
            : KernelId::HYVLA_EXPERT_O_RING_POSTNORM_S51);
        args[0] = w8 ? 7 : 6;
        args.insert(args.end(), {c.m, c.n, c.k, 1, c.gx, 256, 2048, 2, 3,
                                w8 ? 20 : 0, w8 ? 21 : 0, 32, 38, 40, 50});
    }
    return args;
}

void HyVlaExpertModel::set_gateup_resident_weights(const at::Tensor& bundle, int64_t o_layers) {
    TORCH_CHECK(get_last_resolved_chunk_size() == 0 && !gateup_resident_bundle_.defined(),
                "HyVLA Gate/Up residency must be installed once before planning/forward");
    TORCH_CHECK(gateup_resident_profile(),
                "HyVLA Gate/Up residency requires FP16/W8A16/NVFP4 L32/H1024/I2048/S51/TP8/ten-step SwiGLU");
    TORCH_CHECK(bundle.defined() && bundle.device().type() == at::kPrivateUse1 &&
                    bundle.scalar_type() == at::kHalf && bundle.is_contiguous() &&
                    bundle.dim() == 2 && bundle.size(0) == NUM_CORES &&
                    ((nvfp4_ && (bundle.size(1) == 12 * gateup_layer_bytes() / sizeof(c10::Half) ||
                                bundle.size(1) == 24 * gateup_layer_bytes() / sizeof(c10::Half) ||
                                bundle.size(1) == 3211264 || bundle.size(1) == 3276800)) ||
                     (!nvfp4_ && (bundle.size(1) == 6 * gateup_layer_bytes() / sizeof(c10::Half) ||
                      (gateup_weight_bytes() == 262144 &&
                       bundle.size(1) == 12 * gateup_layer_bytes() / sizeof(c10::Half))))) &&
                    bundle.device() == layers_.front().gate_w.device(),
                "HyVLA Gate/Up residency requires an owned 16-bit DMA carrier: FP16 six, W8 six/twelve, or NVFP4 twelve/twenty-four with optional GU4/GU8 FP8 scales");
    TORCH_CHECK(KernelCache::instance().get_program(
                    nvfp4_ ? "hyvla_nvfp4_gateup_resident_s51" :
                    (gateup_weight_bytes() == 262144
                        ? "hyvla_gateup_resident_pair_w8_v2"
                        : "hyvla_gateup_resident_pair_fp16_v2")) != nullptr,
                "HyVLA Gate/Up resident payload unavailable");
    TORCH_CHECK(o_layers == 0 || o_layers == 12, "HyVLA resident O count must be zero or twelve");
    TORCH_CHECK(o_layers == 0 || (gateup_weight_bytes() == 262144 &&
                    cold_w8_o_resident_available_ && cold_gateup_swiglu_available_ &&
                    bundle.size(1) == 3145728),
                "HyVLA resident O requires W8, six GateUp plus twelve O, exactly 6MiB/core and its cold payload");
    const int64_t scale_layers = !nvfp4_ ? 0 :
        (bundle.size(1) == 3211264 ? 4 : (bundle.size(1) == 3276800 ? 8 : 0));
    TORCH_CHECK(scale_layers == 0 || (o_layers == 0 && cold_nvfp4_gateup_available_ &&
                    (cold_nvfp4_resident_swiglu_available_ ? cold_nvfp4_resident_scales_swiglu_available_
                                                          : cold_nvfp4_resident_scales_available_)),
                "HyVLA GU4/GU8 FP8 scales require NVFP4 GU24, no resident O and the matching cold scale-SPM payload");
    TORCH_CHECK(!scale_layers || !cold_nvfp4_resident_overlap_available_,
                "HyVLA resident overlap cannot combine with scale-SPM without a matching payload");
    if (nvfp4_) {
        TORCH_CHECK(!cold_nvfp4_o_overlap_available_ || cold_nvfp4_o_n128_available_,
                    "HyVLA O overlap requires its N128 fusion base");
        TORCH_CHECK(!cold_nvfp4_down_overlap_available_ ||
                        (cold_nvfp4_down_nextnorm_available_ && !cold_nvfp4_down_nextnorm_n128_available_),
                    "HyVLA Down overlap requires its N64 fusion base");
        TORCH_CHECK(!cold_nvfp4_gateup_overlap_available_ ||
                        (cold_nvfp4_gateup_available_ && !cold_nvfp4_gateup_input_reuse_available_),
                    "HyVLA DDR GateUp overlap requires the original producer");
        TORCH_CHECK(!cold_nvfp4_resident_overlap_available_ || cold_nvfp4_resident_swiglu_available_,
                    "HyVLA resident overlap requires its original SwiGLU fusion");
    }
    o_resident_layers_ = o_layers;
    resident_fp8_scale_layers_ = scale_layers;
    gateup_resident_bundle_ = bundle;
    invalidate_model_state();
}

std::vector<BufferDecl> HyVlaExpertModel::baseline_buffer_declarations(
    const LayoutContext& ctx) const {
    // The base only constructs preload callbacks here; none executes during
    // this pure layout probe. Keep the non-const bridge at this one boundary.
    auto d = const_cast<HyVlaExpertModel*>(this)
        ->causal_baseline_buffer_declarations(ctx);
    if (sdpa_m16_enabled(ctx.chunk_size, ctx.max_kv_seq_len - ctx.chunk_size)) {
        TORCH_CHECK(ctx.use_attn_mask && ctx.batch_size == 1 &&
                        ctx.attention_policy == AttentionExecutionPolicy::DDR_KV,
                    "HyVLA M16 attention requires explicit mask, single batch and DDR KV");
        TORCH_CHECK(hyvla_grow_slot(d, "sdpa_tmp", HYVLA_S51_M16_SDPA_TMP_BYTES),
                    "HyVLA M16 attention scratch slot missing from baseline layout");
    }
    if (sdpa_k128_enabled(ctx.chunk_size, ctx.max_kv_seq_len - ctx.chunk_size)) {
        TORCH_CHECK(ctx.use_attn_mask && ctx.batch_size == 1 &&
                        ctx.attention_policy == AttentionExecutionPolicy::DDR_KV,
                    "HyVLA K128 attention requires explicit mask, single batch and DDR KV");
        bool found = false;
        for (auto& slot : d) {
            if (slot.name && std::strcmp(slot.name, "sdpa_tmp") == 0) {
                TORCH_CHECK(slot.size >= HYVLA_S51_K128_SDPA_TMP_BYTES,
                            "HyVLA K128 scratch baseline is shorter than the exact schedule");
                slot.size = HYVLA_S51_K128_SDPA_TMP_BYTES;
                found = true;
            }
        }
        TORCH_CHECK(found, "HyVLA K128 attention scratch slot missing from baseline layout");
    }
    if (cold_mask_once_) hyvla_widen_sdpa_mask_slot(d);
    // KVPAD16（见文件头）：把 k/v 槽撑到 16 的倍数行，KV-insert 一次 v16 打完。
    if (cold_kvpad16_ || cold_rmsnorm_pad16_) {
        const int64_t local_kv = num_kv_heads() * head_dim() / attn_tp();
        hyvla_grow_kv_slots(
            d, Align(ctx.chunk_size, (int64_t)16), local_kv);
    }
    // RMSNORM_PAD16（见文件头）：把 rmsnorm 读写的四个槽撑到 16 的倍数行。
    if (cold_rmsnorm_pad16_) {
        const int64_t hh = hidden_size(), hd_ = head_dim();
        const int64_t lq = num_q_heads() / attn_tp();
        const int64_t rmsnorm_pad_rows_ = Align(ctx.chunk_size, (int64_t)16);
        const int64_t rmsnorm_pad_qrows_ = Align(ctx.chunk_size * lq, (int64_t)16);
        const bool ok =
            hyvla_grow_slot(d, "residual1",  rmsnorm_pad_rows_  * hh  * 2) &&
            hyvla_grow_slot(d, "input_norm", rmsnorm_pad_rows_  * hh  * 2) &&
            hyvla_grow_slot(d, "q",          rmsnorm_pad_qrows_ * hd_ * 2) &&
            hyvla_grow_slot(d, "output",     rmsnorm_pad_qrows_ * hd_ * 2);
        // 四个槽名一个都不能漏 —— 漏了就是**写越界**，不是慢一点。
        TORCH_CHECK(ok, "hyvla RMSNORM_PAD16: 基类槽名对不上（residual1/input_norm/q/output）");
    }
    if (!unroll_mode_) return d;
    const int64_t cs = ctx.chunk_size, h = hidden_size(), np = action_dim_pad_;
    constexpr int DW = 2;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
    using SC = StorageClass;
    constexpr BufferScope ALL = BufferScope::LayerWide;
    // x_t_spm 要横跨整个 body (相 0..9) 才能被 post-hook 的图内 Euler 读到 ⇒ 独立槽,
    // 不与 residual1 (1..8) 混叠。pre-hook 的临时量活在相 0..0, 层 0 之前就死了,
    // 分配器可以安全复用它们的地址。
    d.push_back({"x_t_spm",       A(cs * np * DW), 0, 9, SC::Temp, 0, nullptr, ALL});
    d.push_back({"enc_h_spm",     A(cs * h * DW),  0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"emb_core0_spm", A(cs * h * DW),  0, 0, SC::Temp, 0, nullptr, ALL});
    // A6 走多核时的部分和槽（见文件头 ACTION_MLP_MC）。相位窗口同为 [0,0] ⇒ 与
    // 层体的 Temp（相 1..8）不重叠，分配器可以复用同一段地址，净 SPM 代价 ~0。
    if (action_mlp_cores_ > 1)
        d.push_back({"mo_part_spm", A(cs * h * DW), 0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"time_spm",      A(h * DW),       0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"mo_b_spm",      A(h * DW),       0, 0, SC::Temp, 0, nullptr, ALL});
    // post-hook 的产物落在相 8..9 —— **不得**与 residual1 (1..8) 混叠, post-hook 要读它。
    d.push_back({"v_t_core0_spm", A(cs * np * DW), 8, 9, SC::Temp, 0, nullptr, ALL});
    d.push_back({"op_b_spm",      A(np * DW),      8, 9, SC::Temp, 0, nullptr, ALL});
    if (gateup_resident_bundle_.defined()) {
        TORCH_CHECK(gateup_resident_profile(), "HyVLA Gate/Up resident profile drift");
        d.push_back({"gateup_resident_weights", resident_storage_bytes(),
                     0, 9, SC::Temp, 0, nullptr, ALL});
    }
    if (resident_fp8_scale_layers_ == 8) {
        // GU8 keeps 256 KiB/core of FP8 scales alongside the 6 MiB Gate/Up
        // bundle. SDPA finishes using its scratch before the following ring
        // preparation/O projection writes oproj, on the same ordered queue.
        // Share only those two phase-5 slots; Q, attention output, residuals
        // and all eight scale layers remain separate live owners.
        TORCH_CHECK(nvfp4_ && gateup_resident_profile() &&
                        gateup_resident_layers() == 24 && o_resident_layers_ == 0,
                    "HyVLA GU8 scratch reuse requires its exact resident profile");
        auto find_slot = [&](const char* name) {
            return std::find_if(d.begin(), d.end(), [&](const BufferDecl& slot) {
                return slot.name && std::strcmp(slot.name, name) == 0;
            });
        };
        auto scratch = find_slot("sdpa_tmp");
        auto projection = find_slot("oproj");
        TORCH_CHECK(scratch != d.end() && projection != d.end() &&
                        !scratch->alias_of && !projection->alias_of &&
                        scratch->size > 0 && projection->size > 0 &&
                        scratch->size % SpmAllocator::ALIGN == 0 &&
                        projection->size % SpmAllocator::ALIGN == 0 &&
                        scratch->storage == SC::Temp && projection->storage == SC::Temp &&
                        scratch->scope == ALL && projection->scope == ALL &&
                        scratch->phase_start == 5 && scratch->phase_end == 5 &&
                        projection->phase_start == 5 && projection->phase_end == 5,
                    "HyVLA GU8 scratch/O lifetime contract drift");
        projection->size = std::max(projection->size, scratch->size);
        scratch->size = 0;
        scratch->alias_of = "oproj";
    }
    return d;
}

std::vector<BufferDecl> HyVlaExpertModel::planned_buffer_declarations(
    const LayoutContext& layout, bool* retained) const {
    auto declarations = baseline_buffer_declarations(layout);
    // The first action can run after cold constants retire the vision/text
    // owners. Its free space is not the later three-owner steady-state budget.
    // GU8 therefore fixes the existing once-per-body mask route, rather than
    // admitting a once-per-forward mask which may not fit on warm REPLAY.
    TORCH_CHECK(resident_fp8_scale_layers_ != 8 ||
                    layout.forward_operand_residency != FmbForwardOperandResidency::FORWARD,
                "HyVLA GU8 requires the fixed per-body mask residency");
    bool keep = false;
    switch (layout.forward_operand_residency) {
        case FmbForwardOperandResidency::UNSPECIFIED:
            // Legacy/unbound layout probes keep their original no-growth
            // behavior. Only cold manifest selection may spend extra capacity.
            keep = resident_fp8_scale_layers_ != 8 &&
                cold_mask_once_ && layout.use_attn_mask &&
                detail::plan_forward_spm_residency(declarations, {"sdpa_mask"}).retained;
            break;
        case FmbForwardOperandResidency::PER_LAYER:
            break;
        case FmbForwardOperandResidency::FORWARD: {
            TORCH_CHECK(cold_mask_once_ && layout.use_attn_mask,
                        "HyVLA retained mask requires its cold explicit-mask owner");
            auto trial = detail::make_forward_spm_residency_candidate(
                declarations, {"sdpa_mask"});
            TORCH_CHECK(trial.valid, "HyVLA retained mask lifetime is invalid");
            declarations = std::move(trial.declarations);
            keep = true;
            break;
        }
        default:
            TORCH_CHECK(false, "HyVLA unknown bound mask residency");
    }
    if (retained) *retained = keep;
    return declarations;
}

FmbForwardOperandResidency HyVlaExpertModel::mask_residency_for_candidate(
    const LayoutContext& layout) const {
    // Reconstruction/BUILD/REPLAY consumes the descriptor's frozen choice.
    // Live free space is only consulted by the cold unbound planning oracle.
    if (resident_fp8_scale_layers_ == 8) {
        TORCH_CHECK(layout.forward_operand_residency == FmbForwardOperandResidency::UNSPECIFIED ||
                        layout.forward_operand_residency == FmbForwardOperandResidency::PER_LAYER,
                    "HyVLA GU8 requires the fixed per-body mask residency");
        return FmbForwardOperandResidency::PER_LAYER;
    }
    if (layout.forward_operand_residency != FmbForwardOperandResidency::UNSPECIFIED) {
        TORCH_CHECK(layout.forward_operand_residency == FmbForwardOperandResidency::PER_LAYER ||
                        layout.forward_operand_residency == FmbForwardOperandResidency::FORWARD,
                    "HyVLA unknown bound mask residency");
        return layout.forward_operand_residency;
    }
    if (!cold_mask_once_ || !layout.use_attn_mask)
        return FmbForwardOperandResidency::PER_LAYER;
    if (layout.chunk_size == 51 && layout.batch_size == 1 &&
        layout.attention_policy == AttentionExecutionPolicy::DDR_KV &&
        optimized_prefix_supported(layout.max_kv_seq_len - 51) &&
        gateup_resident_profile()) {
        const auto fits = [&](const LayoutContext& candidate_layout) {
            const auto trial = detail::make_forward_spm_residency_candidate(
                baseline_buffer_declarations(candidate_layout), {"sdpa_mask"});
            return trial.valid && declared_spm_layout_fits(trial.declarations);
        };
        if (!fits(layout)) return FmbForwardOperandResidency::PER_LAYER;
        if (layout.planning_chunk_capacity > 0 &&
            layout.planning_chunk_capacity != layout.chunk_size) {
            auto capacity_layout = layout;
            capacity_layout.chunk_size = layout.planning_chunk_capacity;
            if (!fits(capacity_layout)) return FmbForwardOperandResidency::PER_LAYER;
        }
        return FmbForwardOperandResidency::FORWARD;
    }
    bool retained = false;
    planned_buffer_declarations(layout, &retained);
    return retained ? FmbForwardOperandResidency::FORWARD
                    : FmbForwardOperandResidency::PER_LAYER;
}

std::vector<BufferDecl> HyVlaExpertModel::declare_buffers(
    const LayoutContext& layout) {
    return planned_buffer_declarations(layout);
}

int64_t HyVlaExpertModel::mask_schedule_for_layout(
    const LayoutContext& layout) const {
    const bool retained = mask_residency_for_candidate(layout) ==
        FmbForwardOperandResidency::FORWARD;
    return static_cast<int64_t>(retained
        ? HyVlaExpertGraphScheduleRoute::MASK_FIRST_BODY_FIRST_LAYER
        : cold_mask_once_ ? HyVlaExpertGraphScheduleRoute::MASK_FIRST_LAYER_ONLY
                          : HyVlaExpertGraphScheduleRoute::MASK_EACH_LAYER);
}

FmbForwardOperandResidency HyVlaExpertModel::physical_forward_operand_residency(
    const FmbPhysicalExecutionManifest& manifest) const {
    if (manifest.state != FmbPhysicalManifestState::COMPLETE)
        return FmbForwardOperandResidency::UNSPECIFIED;
    const FmbRouteManifestEntry* mask = nullptr;
    for (const auto& route : manifest.routes) {
        if (route.site_id != HYVLA_EXPERT_MASK_SCHEDULE_SITE) continue;
        TORCH_CHECK(mask == nullptr, "HyVLA requires exactly one mask schedule");
        mask = &route;
    }
    TORCH_CHECK(mask && mask->family == FmbRouteFamily::GRAPH_SCHEDULE &&
                    mask->flags == 0 && mask->invocation == 0 &&
                    manifest.physical_length > 0 &&
                    manifest.kv_logical_length >= manifest.physical_length &&
                    mask->arguments == std::vector<int64_t>({
                        cold_mask_once_ ? 1 : 0, manifest.physical_length,
                        manifest.kv_logical_length, attn_tp(), num_layers()}),
                "HyVLA mask schedule geometry differs from its owner");
    const auto selector = static_cast<HyVlaExpertGraphScheduleRoute>(mask->selector);
    TORCH_CHECK(cold_mask_once_
                    ? (selector == HyVlaExpertGraphScheduleRoute::MASK_FIRST_LAYER_ONLY ||
                       selector == HyVlaExpertGraphScheduleRoute::MASK_FIRST_BODY_FIRST_LAYER)
                    : selector == HyVlaExpertGraphScheduleRoute::MASK_EACH_LAYER,
                "HyVLA mask schedule differs from its cold configuration");
    return selector == HyVlaExpertGraphScheduleRoute::MASK_FIRST_BODY_FIRST_LAYER
        ? FmbForwardOperandResidency::FORWARD : FmbForwardOperandResidency::PER_LAYER;
}

ModelStaticConfig HyVlaExpertModel::static_config() {
    ModelStaticConfig cfg = CausalDecoderModel::static_config();
    cfg.fast_replay_skip_layer_loop = cold_fast_replay_;
    cfg.fast_replay_skip_preload = cold_fast_replay_preload_;
    if (!unroll_mode_) return cfg;
    cfg.pre_layers_fn  = reinterpret_cast<void (FusedModelBase::*)()>(
        &HyVlaExpertModel::emit_pre_layers_body);
    cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
        &HyVlaExpertModel::emit_post_layers_body);
    cfg.body_iterations = num_steps_;
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// unroll 的 pre/post hook
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::emit_pre_layers_body() {
    const int64_t h = hidden_size(), cs = expert_chunk_size_, np = action_dim_pad_;
    const int64_t bit = ctx().body_iter;
    if (gateup_resident_bundle_.defined() && bit == 0) {
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA, HYVLA_EXPERT_GATEUP_PRELOAD_SITE,
            static_cast<int64_t>(HyVlaExpertDmaRoute::DDR_SCATTER_TO_SPM_FIXED), 0,
            {gateup_resident_layers(), resident_storage_bytes() / 2, resident_storage_bytes(),
             NUM_CORES, 0});
        // This DMA is part of every replay, once before all ten bodies. Other
        // model graphs may have overwritten this temporary arena between calls.
        rpu_launch_ddr_scatter_spm_dma(
            gateup_resident_bundle_, 0, resident_storage_bytes() / 2, resident_storage_bytes(),
            addr(0, "gateup_resident_weights"), NUM_CORES);
    }

    // [A1] x0 DDR → x_t_spm (MUTABLE), **只在第 0 步**。第 1..N-1 步直接读上一步
    //      post-hook 留在 x_t_spm 里的 Euler 结果 (槽的生命期是 [0,9], 跨相存活)。
    if (bit == 0) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                HYVLA_EXPERT_PRE_X0_DMA_SITE,
                static_cast<int64_t>(
                    HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_MUTABLE),
                /*resolved_flags=*/0,
                {unroll_mode_ ? 1 : 0, cs * np, num_cores()});
        }
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &x_t_src_base_, /*src_offset_bytes=*/0, /*num_elements=*/cs * np,
            addr(0, "x_t_spm"), /*num_cores=*/num_cores());
    } else if (action_mlp_cores_ > 1) {
        // 多核档：A4 是列并行 ⇒ **每个核都要完整的 x_t**。但图内 Euler [Z3] 只在
        // 核 0 更新 x_t_spm（v_t 只有核 0 有），第 1 步起核 1..7 就是陈旧值
        // —— 这是第二次接错的地方（cos 0.487）。
        // 修法用本图**已经存在**的模式：[Z4] 每步都把 x_t 写进 x_traj[bit]，
        // 这里从 x_traj[bit-1] 广播回全部核（3.3 KB，~1 µs）。
        // 与 A7→emb_stage_→层 0 那条 DDR 往返同型，不引入新的存活假设。
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                HYVLA_EXPERT_PRE_X_FEEDBACK_DMA_SITE,
                static_cast<int64_t>(
                    HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_MUTABLE),
                /*resolved_flags=*/0,
                {action_mlp_cores_, num_steps_, cs * np, num_cores()});
        }
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &x_traj_dst_base_, /*src_offset_bytes=*/(bit - 1) * cs * np * DWIDTH,
            /*num_elements=*/cs * np, addr(0, "x_t_spm"), /*num_cores=*/num_cores());
    }
    const int amc = action_mlp_cores_;
    // [A2] time_all[bit] → time_spm, 作为 encoder 第一个 GEMM (A4) 的 bias。
    //      FIXED src: time_all_ 是稳定的注册权重, bit 偏移在 BUILD 期烘死
    //      (每个 body 单独 emit 一次, body_iter 是编译期常量)。
    //      ⚠️ 多核档 A4 走**列并行** ⇒ 核 c 只负责输出列 [c*h/amc, (c+1)*h/amc)，
    //         bias 必须**按核切片**（scatter），不能广播 —— 广播会让每个核都加
    //         bias[0:h/amc]，是静默算错。
    if (amc == 1) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                HYVLA_EXPERT_PRE_TIME_BROADCAST_DMA_SITE,
                static_cast<int64_t>(
                    HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_FIXED),
                /*resolved_flags=*/0,
                {action_mlp_cores_, h, h / action_mlp_cores_, DWIDTH});
        }
        rpu_launch_ddr_broadcast_spm_dma(
            time_all_.data_ptr<c10::Half>() + bit * h, h, addr(0, "time_spm"),
            /*num_cores=*/1);
    } else {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                HYVLA_EXPERT_PRE_TIME_SCATTER_DMA_SITE,
                static_cast<int64_t>(
                    HyVlaExpertDmaRoute::DDR_SCATTER_TO_SPM_FIXED),
                /*resolved_flags=*/0,
                {action_mlp_cores_, h, h / action_mlp_cores_, DWIDTH});
        }
        rpu_launch_ddr_scatter_spm_dma(
            time_all_.data_ptr<c10::Half>() + bit * h,
            /*elements_per_core=*/h / amc,
            /*core_stride_bytes=*/(h / amc) * DWIDTH,
            addr(0, "time_spm"), /*num_cores=*/amc);
    }
    // [A3] mo_bias → mo_b_spm (内容恒定, FIXED src)。
    //      单核档只有核 0 用得到；多核档 all-reduce 后 amc 个核都要按列加一次
    //      bias（A6-4 在这 amc 个核上齐发），所以要广播满 amc 个核。
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA,
            HYVLA_EXPERT_PRE_MO_BIAS_DMA_SITE,
            static_cast<int64_t>(
                HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_FIXED),
            /*resolved_flags=*/0, {action_mlp_cores_, h});
    }
    rpu_launch_ddr_broadcast_spm_dma(
        mo_bias_.data_ptr<c10::Half>(), h, addr(0, "mo_b_spm"), /*num_cores=*/amc);

    // [A4] enc_h = x_t_spm · wcᵀ + time_bias  (M=cs, N=h, K=np)。
    //      ⚠️ 多核档必须**列并行**：A6 是行(K)并行，它要求核 c 手里正好是输入的
    //      第 c 段 K —— 而列并行 A4 的输出天然就是那一段。两者相接**零通信**，
    //      就是层体里 gate/up(列) → down(行) 的同一套 TP 模式。
    //      单核 A4 仅初始化核 0；多核 A6 需要每个参与核的输入。
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, HYVLA_EXPERT_PRE_WC_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            {cs, h, np, 1, action_mlp_cores_, 1});
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "x_t_spm"), wc_, addr(0, "enc_h_spm"),
        /*M=*/cs, /*N=*/h, /*K=*/np, /*partition=*/1, /*num_cores=*/amc,
        /*bias_spm_addr=*/addr(0, "time_spm"));
    // [A5] SiLU 原地。逐元素 ⇒ 每核只处理自己那 h/amc 列。
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ACTIVATION,
            HYVLA_EXPERT_PRE_ENCODER_ACTIVATION_SITE,
            static_cast<int64_t>(HyVlaExpertActivationRoute::SILU_UNARY),
            /*resolved_flags=*/0,
            {action_mlp_cores_, cs * (h / action_mlp_cores_)});
    }
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "enc_h_spm"), addr(0, "enc_h_spm"), cs * (h / amc), ValuOpType::SILU,
        /*is_gelu=*/false, /*num_cores=*/amc);
    if (amc == 1) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR,
                HYVLA_EXPERT_PRE_MO_SINGLE_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {cs, h, h, 1, action_mlp_cores_, 1});
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "enc_h_spm"), mo_, addr(0, "emb_core0_spm"),
            /*M=*/cs, /*N=*/h, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
            /*bias_spm_addr=*/addr(0, "mo_b_spm"));
    } else {
        // A6-1 K 方向切核 ⇒ 每核一份 [cs,h] 部分和。**bias 必须留到 all-reduce 之后**。
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR,
                HYVLA_EXPERT_PRE_MO_MULTI_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {cs, h, h, 0, action_mlp_cores_, 0});
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "enc_h_spm"), mo_, addr(0, "mo_part_spm"),
            /*M=*/cs, /*N=*/h, /*K=*/h, /*partition=*/0, /*num_cores=*/amc,
            /*bias_spm_addr=*/0u);
        // A6-2 enc_h_spm 到这里已死 ⇒ 就地清零，当 all-reduce 的零残差
        //      （该 kernel 要求 input/residual/output 三地址互不相同）。
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_PRE_MO_ZERO_SITE,
                static_cast<int64_t>(
                    HyVlaExpertGraphScheduleRoute::ZERO_A6_RESIDUAL),
                /*resolved_flags=*/0,
                {action_mlp_cores_, cs * h});
        }
        rpu_launch_fill_spm_kernel(
            addr(0, "enc_h_spm"), cs * h, c10::Half(0.0f), /*num_cores=*/amc);
        // A6-3 fused ring all-reduce；调度器按 per-core chunk 自动选 pace/nopace。
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                HYVLA_EXPERT_PRE_MO_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(cs, h),
                /*resolved_flags=*/0,
                {cs, h, action_mlp_cores_, action_mlp_cores_});
        }
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "mo_part_spm"), addr(0, "enc_h_spm"), addr(0, "emb_core0_spm"),
            /*M=*/cs, /*N=*/h, /*input_num_cores=*/amc, /*output_num_cores=*/amc);
        // A6-4 bias 按列广播加一次（amc 核齐发，各核结果一致；A7 只读核 0）。
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_PRE_MO_BIAS_ADD_SITE,
                static_cast<int64_t>(
                    HyVlaExpertGraphScheduleRoute::ADD_A6_BIAS_AFTER_REDUCE),
                /*resolved_flags=*/0,
                {action_mlp_cores_, cs, h});
        }
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            addr(0, "mo_b_spm"), addr(0, "emb_core0_spm"), addr(0, "emb_core0_spm"),
            /*n=*/cs, /*c=*/h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false,
            /*num_cores=*/amc);
    }
    // [A7] emb_core0_spm[1:cs] → emb_stage_[1:cs] (src/dst 都偏一行)。行 0 保持
    //      unroll_forward 写进去的 state token 不动 —— encoder 的行 0 是 x_t 行 0
    //      (恒为 0 且不被任何东西消费) 的产物, 直接丢弃。
    rpu_launch_spm_copy_ddr_dma(
        addr(0, "emb_core0_spm") + h * DWIDTH,
        emb_stage_.data_ptr<c10::Half>() + h, (cs - 1) * h);
}

void HyVlaExpertModel::emit_post_layers_body() {
    const int64_t h = hidden_size(), cs = expert_chunk_size_, np = action_dim_pad_;
    const int64_t bit = ctx().body_iter;
    // [Z1] op_bias → op_b_spm (核 0, 内容恒定, FIXED src)。
    rpu_launch_ddr_broadcast_spm_dma(
        op_bias_.data_ptr<c10::Half>(), np, addr(0, "op_b_spm"), /*num_cores=*/1);
    // [Z2] v_t = residual1 · opᵀ + op_bias (单核)。residual1 已被末层的 final
    //      RMSNorm 原地归一化过 —— **不要再归一化一次**。
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, HYVLA_EXPERT_POST_OP_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {cs, np, h, 1, 1, 1});
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), op_, addr(0, "v_t_core0_spm"),
        /*M=*/cs, /*N=*/np, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
        /*bias_spm_addr=*/addr(0, "op_b_spm"));
    // [Z3] 图内 Euler: x_t_spm = x_t_spm + dt·v_t (SPM 原地, 核 0)。dt 在 BUILD 烘死。
    //      行 0 (state 位) 也会被更新, 但它的 encoder 输出从不进 emb_stage_ ⇒ 死路。
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "x_t_spm"), addr(0, "v_t_core0_spm"), addr(0, "x_t_spm"),
        cs * np, ValuOpType::ADD, dt_, /*num_cores=*/1);
    // [Z4] x_t_spm → x_traj[bit] DDR (MUTABLE)。终值 = x_traj[-1]。
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA,
            HYVLA_EXPERT_POST_TRAJECTORY_DMA_SITE,
            static_cast<int64_t>(
                HyVlaExpertDmaRoute::SPM_TO_DDR_MUTABLE),
            /*resolved_flags=*/0, {num_steps_, cs * np});
    }
    rpu_launch_spm_copy_ddr_dma_mutable(
        addr(0, "x_t_spm"), &x_traj_dst_base_,
        /*dst_offset_bytes=*/bit * cs * np * 2, cs * np);
}

// ─────────────────────────────────────────────────────────────────────────────
// dynamic_config — 基类断言单 chunk，并统一准备显式 mask；本类直接复用
// CausalDecoderModel 的稳定 PreparedMask。
// ─────────────────────────────────────────────────────────────────────────────
ModelDynamicConfig HyVlaExpertModel::dynamic_config(const ChunkPlan& plan) {
    ModelDynamicConfig cfg = CausalDecoderModel::dynamic_config(plan);
    // Prefix history is shared with the VLM tower through DDR KV.  Keep this
    // owner on that certified path; raw-SPM KV attention is not admissible.
    cfg.attention_policy = AttentionExecutionPolicy::DDR_KV;
    TORCH_CHECK(ctx().attention_mask.has_value(),
                "hyvla_expert dynamic_config: explicit attention mask required "
                "(denoise 是非因果 prefix attention, prefix 块还要屏蔽 padding 行)");
    return cfg;
}

ModelDynamicConfig HyVlaExpertModel::planning_dynamic_config(
    const ChunkPlan& /*plan*/) {
    ModelDynamicConfig cfg;
    cfg.chunk_mode = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    cfg.attention_policy = AttentionExecutionPolicy::DDR_KV;
    return cfg;
}

bool HyVlaExpertModel::paired_kv_enabled(const KvInsertSegmentPlan& plan) const {
    if (!cold_paired_kv_available_ || !cold_kvpad16_ ||
        num_layers() != 32 || hidden_size() != 1024 || intermediate_size() != 2048 ||
        num_q_heads() != 16 || num_kv_heads() != 8 || head_dim() != 128 ||
        attn_tp() != 8 || expert_chunk_size_ != 51 || layers_.size() != 32 ||
        plan.route() != KvInsertRoute::PAD16_V16 || plan.logical_rows() != 51 ||
        plan.physical_rows() != 64 || plan.segment_count() != 1) return false;
    const auto& segment = plan.segment(0);
    if (!optimized_prefix_supported(segment.position) || segment.token_offset != 0 || segment.rows != 64 ||
        segment.kernel != KvInsertKernel::V16) return false;
    for (const auto& w : layers_) {
        if (w.q_w.scalar_type() != layers_.front().q_w.scalar_type() ||
            !hyvla_linear_precision(w, cold_w8_optimizations_available_,
                                     nvfp4_ && cold_nvfp4_gateup_available_)) return false;
    }
    return true;
}

std::vector<int64_t> HyVlaExpertModel::paired_kv_arguments(
    const KvInsertSegmentPlan& plan) {
    return {1, plan.logical_rows(), plan.physical_rows(),
            plan.segment(0).position, 8, 8, 128};
}

FmbPhysicalExecutionManifest HyVlaExpertModel::physical_manifest_for_candidate(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len,
    int64_t position) const {
    FmbPhysicalExecutionManifest manifest;
    manifest.state = FmbPhysicalManifestState::COMPLETE;
    manifest.logical_length = logical_len;
    manifest.physical_length = physical_len;
    manifest.execution_padding_rows = physical_len - logical_len;
    manifest.kv_logical_length = position + logical_len;
    TORCH_CHECK(
        plan.qkv.chunks.size() == 1 && plan.compute.chunks.size() == 1,
        "HyVlaExpertModel COMPLETE descriptor requires one action chunk");
    const ChunkInfo& kv_chunk = plan.qkv.chunks.front();
    const ChunkInfo& chunk = plan.compute.chunks.front();
    TORCH_CHECK(
        kv_chunk.idx == chunk.idx && kv_chunk.offset == chunk.offset &&
            kv_chunk.len == chunk.len,
        "HyVlaExpertModel COMPLETE descriptor requires identical QKV and "
        "compute action chunks");
    const int64_t invocation = chunk.idx;
    const int64_t h = hidden_size();
    const int64_t nq = num_q_heads();
    const int64_t nkv = num_kv_heads();
    const int64_t hd = head_dim();
    const int64_t tp = attn_tp();
    const int64_t local_q = nq / tp;
    const int64_t local_kv_heads = nkv / tp;
    const int64_t padded_rows = Align(layout.chunk_size, int64_t{16});
    const int64_t norm_rows = cold_rmsnorm_pad16_
        ? padded_rows : chunk.len;
    const int64_t q_norm_rows = cold_rmsnorm_pad16_
        ? Align(layout.chunk_size * local_q, int64_t{16})
        : chunk.len * local_q;
    const int64_t k_norm_rows = (cold_rmsnorm_pad16_
        ? padded_rows : chunk.len) * local_kv_heads;
    const int64_t kv_physical_rows = cold_kvpad16_
        ? Align(layout.chunk_size, int64_t{16}) : kv_chunk.len;
    manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
    const KvInsertSegmentPlan kv_plan =
        resolve_kvinsert_plan_auto(
            HYVLA_EXPERT_KV_SITE, manifest.graph_lifecycle,
            position + kv_chunk.offset, kv_chunk.len, kv_physical_rows,
            attn_tp(), num_kv_heads(), head_dim(),
            HYVLA_EXPERT_KV_CAPABILITIES);
    const KvInsertRouteArguments kv_arguments =
        rpu_kvinsert_route_arguments(
            kv_plan, attn_tp(), num_kv_heads(), head_dim());
    manifest.kv_insert_physical_rows = kv_plan.physical_rows();
    manifest.linear_accumulation = FmbLinearAccumulationPolicy::ACC16;

    const auto append = [&](FmbRouteFamily family, int64_t site_id,
                            int64_t selector,
                            std::vector<int64_t> arguments = {},
                            int64_t flags = 0) {
        manifest.routes.push_back({site_id, family, selector, flags,
                                   std::move(arguments), invocation});
    };
    const auto append_linear = [&](int64_t site_id,
                                   std::vector<int64_t> arguments = {}) {
        append(FmbRouteFamily::LINEAR, site_id,
               static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
               std::move(arguments));
    };
    const auto append_norm = [&](int64_t site_id, int64_t rows,
                                 int64_t cols) {
        append(
            FmbRouteFamily::NORMALIZATION, site_id,
            static_cast<int64_t>(
                cold_rmsnorm_pad16_
                    ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                    : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
            {cold_rmsnorm_pad16_ ? 1 : 0, rows, cols});
    };

    append(
        FmbRouteFamily::GRAPH_SCHEDULE,
        HYVLA_EXPERT_GRAPH_SCHEDULE_SITE,
        static_cast<int64_t>(
            cold_fast_replay_
                ? HyVlaExpertGraphScheduleRoute::FAST_REPLAY_SKIP_LAYER_LOOP
                : HyVlaExpertGraphScheduleRoute::REEMIT_LAYER_LOOP),
        {cold_fast_replay_ ? 1 : 0,
         cold_fast_replay_preload_ ? 1 : 0,
         unroll_mode_ ? 1 : 0,
         unroll_mode_ ? num_steps_ : 1});
    append(
        FmbRouteFamily::GRAPH_SCHEDULE,
        HYVLA_EXPERT_MASK_SCHEDULE_SITE,
        mask_schedule_for_layout(layout),
        {cold_mask_once_ ? 1 : 0, chunk.len, kv_chunk.kv_seq_len,
         tp, num_layers()});
    append(
        FmbRouteFamily::KV_INSERT, HYVLA_EXPERT_KV_PADDING_SITE,
        static_cast<int64_t>(
            cold_kvpad16_ ? HyVlaExpertKvPaddingRoute::PAD16_ROWS
                          : HyVlaExpertKvPaddingRoute::LOGICAL_ROWS),
        {cold_kvpad16_ ? 1 : 0, kv_chunk.len,
         cold_kvpad16_ ? kv_physical_rows : 0,
         kv_physical_rows, tp, nkv, hd});

    const auto append_rmsnorm_route = [&](int64_t site, int64_t rows, int64_t cols) {
        append(FmbRouteFamily::NORMALIZATION, site,
               static_cast<int64_t>(cold_rmsnorm_spm_contract_.resolve(rows, cols)),
               cold_rmsnorm_spm_contract_.manifest_arguments(rows, cols));
    };
    append_rmsnorm_route(HYVLA_EXPERT_FULL_RMSNORM_ROUTE_SITE, norm_rows, h);
    append_rmsnorm_route(HYVLA_EXPERT_Q_RMSNORM_ROUTE_SITE, q_norm_rows, hd);
    append_rmsnorm_route(HYVLA_EXPERT_K_RMSNORM_ROUTE_SITE, k_norm_rows, hd);

    append_norm(HYVLA_EXPERT_INPUT_NORM_SITE, norm_rows, h);
    const int qkv_mode = nvfp4_qkv_mode(chunk.len, position + chunk.offset);
    if (nvfp4_q_n32_enabled(chunk.len, position + chunk.offset)) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_NVFP4_Q_N32_SITE,
               1, nvfp4_q_n32_arguments(position + chunk.offset));
    } else if (qkv_mode != 3) append_linear(HYVLA_EXPERT_QKV_LINEAR_SITE);
    if (qkv_mode) append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_NVFP4_QKV_SITE,
                         1, nvfp4_qkv_arguments(qkv_mode, position + chunk.offset));
    if (kv_grouped_enabled(chunk.len, position + chunk.offset))
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_KV_GROUPED_SITE,
               1, kv_grouped_arguments(position + chunk.offset));
    const bool fused_qk_norm = qk_rope_norm_enabled(chunk.len, position + chunk.offset);
    const bool fused_qk_norm_kv = qk_rope_norm_kv_enabled(
        chunk.len, position + chunk.offset, kv_plan);
    if (fused_qk_norm_kv) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_QK_ROPE_NORM_KV_SITE,
               1, qk_rope_norm_kv_arguments(kv_plan, position + chunk.offset));
    } else if (fused_qk_norm) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_QK_ROPE_NORM_SITE,
               1, qk_rope_norm_arguments(position + chunk.offset));
    } else if (qk_rope_enabled(chunk.len, position + chunk.offset)) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_QK_ROPE_SITE,
               1, qk_rope_arguments(position + chunk.offset));
    } else {
        append(
            FmbRouteFamily::ROPE,
            cold_partial_rope_ ? HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE
                               : HYVLA_EXPERT_Q_ROPE_SITE,
            static_cast<int64_t>(
                cold_partial_rope_ ? HyVlaExpertRopeRoute::PARTIAL_MROPE
                                   : HyVlaExpertRopeRoute::ROPE_1D),
            {cold_partial_rope_ ? 1 : 0, position + chunk.offset,
             chunk.len, local_q, hd, tp});
        append(
            FmbRouteFamily::ROPE,
            cold_partial_rope_ ? HYVLA_EXPERT_K_PARTIAL_ROPE_SITE
                               : HYVLA_EXPERT_K_ROPE_SITE,
            static_cast<int64_t>(
                cold_partial_rope_ ? HyVlaExpertRopeRoute::PARTIAL_MROPE
                                   : HyVlaExpertRopeRoute::ROPE_1D),
            {cold_partial_rope_ ? 1 : 0, position + chunk.offset,
             chunk.len, local_kv_heads, hd, tp});
    }
    if (!fused_qk_norm) {
        append_norm(HYVLA_EXPERT_Q_NORM_SITE, q_norm_rows, hd);
        append_norm(HYVLA_EXPERT_K_NORM_SITE, k_norm_rows, hd);
    }
    // The shared VLM prefix remains DDR_REQUIRED across every denoise step.
    // Freeze the exact pure-resolver segment plan; no legacy selector is read
    // while emitting the graph.
    append(
        FmbRouteFamily::KV_INSERT, HYVLA_EXPERT_KV_SITE,
        static_cast<int64_t>(kv_plan.route()),
        std::vector<int64_t>(kv_arguments.begin(), kv_arguments.end()),
        HYVLA_EXPERT_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED);
    if (!fused_qk_norm_kv && paired_kv_enabled(kv_plan)) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_PAIRED_KV_SITE,
               1, paired_kv_arguments(kv_plan));
    }
    append(
        FmbRouteFamily::ATTENTION, HYVLA_EXPERT_ATTN_SITE,
        static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV));
    if (sdpa_k128_enabled(chunk.len, position + chunk.offset)) {
        TORCH_CHECK(layout.chunk_size == 51 && layout.max_kv_seq_len == position + chunk.offset + chunk.len &&
                        layout.use_attn_mask && layout.batch_size == 1 &&
                        layout.attention_policy == AttentionExecutionPolicy::DDR_KV,
                    "HyVLA K128 attention layout differs from the declared schedule");
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_SDPA_K128_SITE,
               1, sdpa_k128_arguments(position + chunk.offset));
    }
    if (sdpa_m16_enabled(chunk.len, position + chunk.offset)) {
        TORCH_CHECK(layout.chunk_size == 51 && layout.max_kv_seq_len == position + chunk.offset + chunk.len &&
                        layout.use_attn_mask && layout.batch_size == 1 &&
                        layout.attention_policy == AttentionExecutionPolicy::DDR_KV,
                    "HyVLA M16 attention layout differs from the declared schedule");
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_SDPA_M16_SITE,
               1, sdpa_m16_arguments(position + chunk.offset));
    }
    append(
        FmbRouteFamily::ALL_REDUCE,
        HYVLA_EXPERT_PREPARE_ATTN_ALL_REDUCE_SITE,
        static_cast<int64_t>(
            HyVlaExpertAllReduceRoute::PREPARE_RING_INPUT),
        {chunk.len, h, tp});
    if (o_ring_postnorm_enabled(chunk.len, position + chunk.offset)) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_O_RING_POSTNORM_SITE,
               1, o_ring_postnorm_arguments(position + chunk.offset));
    } else {
        if (o_resident_enabled(chunk.len, position + chunk.offset))
            append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_O_RESIDENT_SITE,
                   1, o_resident_arguments(position + chunk.offset));
        // The remaining twenty layers retain the ordinary DDR O projection.
        append_linear(
            HYVLA_EXPERT_O_LINEAR_SITE,
            {chunk.len, h, nq * hd, 0, tp});
        if (ring_postnorm_enabled(chunk.len, position + chunk.offset)) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_RING_POSTNORM_SITE,
                   1, ring_postnorm_arguments(position + chunk.offset));
        } else {
            append(
                FmbRouteFamily::ALL_REDUCE,
                HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(chunk.len, h),
                {chunk.len, h, tp, num_cores()});
            append_norm(HYVLA_EXPERT_POST_NORM_SITE, norm_rows, h);
        }
    }
    if (gateup_resident_bundle_.defined()) {
        TORCH_CHECK(gateup_resident_profile() && chunk.len == 51 &&
                        optimized_prefix_supported(position + chunk.offset),
                    "HyVLA Gate/Up resident descriptor requires its admitted S51 prefix");
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_GATEUP_RESIDENT_SITE,
               1, gateup_resident_arguments());
        append(FmbRouteFamily::MUTABLE_DMA, HYVLA_EXPERT_GATEUP_PRELOAD_SITE,
               static_cast<int64_t>(HyVlaExpertDmaRoute::DDR_SCATTER_TO_SPM_FIXED),
               {gateup_resident_layers(), resident_storage_bytes() / 2, resident_storage_bytes(),
                NUM_CORES, 0});
    }
    if (gateup_swiglu_enabled(chunk.len, position + chunk.offset)) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_GATEUP_SWIGLU_SITE,
               1, gateup_swiglu_arguments(position + chunk.offset));
    } else {
        append_linear(
            HYVLA_EXPERT_GATE_LINEAR_SITE,
            {chunk.len, intermediate_size(), h, 1, mlp_tp()});
        append_linear(
            HYVLA_EXPERT_UP_LINEAR_SITE,
            {chunk.len, intermediate_size(), h, 1, mlp_tp()});
    }
    if (resident_fp8_scale_layers_) {
        TORCH_CHECK(resident_fp8_scales_enabled(chunk.len, position + chunk.offset),
                    "HyVLA resident FP8 scale descriptor profile drift");
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_NVFP4_RESIDENT_SCALES_SITE,
               1, resident_fp8_scales_arguments(position + chunk.offset));
    }
    // Resident producer fusion consumes activation for its installed layer interval.
    const int64_t elems_mlp = chunk.len * (intermediate_size() / mlp_tp());
    if (resident_swiglu_enabled(chunk.len, position + chunk.offset)) {
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               nvfp4_ ? HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE : HYVLA_EXPERT_RESIDENT_SWIGLU_SITE,
               1, resident_swiglu_arguments(position + chunk.offset));
    } else if (cold_silu_mul_) {
        append(
            FmbRouteFamily::ACTIVATION,
            HYVLA_EXPERT_MLP_SILU_MUL_SITE,
            static_cast<int64_t>(
                HyVlaExpertActivationRoute::SILU_MUL_FUSED),
            {cold_silu_mul_ ? 1 : 0, elems_mlp, mlp_tp()});
    } else {
        append(
            FmbRouteFamily::ACTIVATION, HYVLA_EXPERT_MLP_SILU_SITE,
            static_cast<int64_t>(
                HyVlaExpertActivationRoute::SILU_UNARY),
            {cold_silu_mul_ ? 1 : 0, elems_mlp, mlp_tp()});
        append(
            FmbRouteFamily::ACTIVATION, HYVLA_EXPERT_MLP_MUL_SITE,
            static_cast<int64_t>(HyVlaExpertActivationRoute::MUL),
            {cold_silu_mul_ ? 1 : 0, elems_mlp, mlp_tp()});
    }
    const bool fused_nvfp4_down = down_nvfp4_nextnorm_enabled(chunk.len, position + chunk.offset);
    if (fused_nvfp4_down)
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_DOWN_NVFP4_NEXTNORM_SITE,
               1, down_nvfp4_nextnorm_arguments(position + chunk.offset));
    else if (down_nvfp4_n128_enabled(chunk.len, position + chunk.offset))
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_DOWN_NVFP4_N128_SITE,
               1, down_nvfp4_n128_arguments(position + chunk.offset));
    else
        append_linear(
            HYVLA_EXPERT_DOWN_LINEAR_SITE,
            {chunk.len, h, intermediate_size(), 0, mlp_tp()});
    if (fused_nvfp4_down) {
        // The single producer/tail recipe above owns the complete Down schedule.
    } else if (down_nextnorm_enabled(chunk.len, position + chunk.offset)) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_DOWN_NEXTNORM_SITE,
               1, down_nextnorm_arguments(position + chunk.offset));
    } else {
        append(
            FmbRouteFamily::ALL_REDUCE,
            HYVLA_EXPERT_MLP_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(chunk.len, h),
            {chunk.len, h, mlp_tp(), num_cores()});
        append_norm(HYVLA_EXPERT_FINAL_NORM_SITE, norm_rows, h);
    }

    if (unroll_mode_) {
        TORCH_CHECK(
            action_dim_pad_ > 0 && num_steps_ > 0 &&
                (action_mlp_cores_ == 1 || action_mlp_cores_ == num_cores()),
            "HyVlaExpertModel COMPLETE unroll descriptor requires bound "
            "action weights and a supported core count");
        const int64_t amc = action_mlp_cores_;
        const int64_t cs = expert_chunk_size_;
        const int64_t np = action_dim_pad_;
        append(
            FmbRouteFamily::MUTABLE_DMA, HYVLA_EXPERT_PRE_X0_DMA_SITE,
            static_cast<int64_t>(
                HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_MUTABLE),
            {unroll_mode_ ? 1 : 0, cs * np, num_cores()});
        if (amc > 1 && num_steps_ > 1) {
            append(
                FmbRouteFamily::MUTABLE_DMA,
                HYVLA_EXPERT_PRE_X_FEEDBACK_DMA_SITE,
                static_cast<int64_t>(
                    HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_MUTABLE),
                {amc, num_steps_, cs * np, num_cores()});
        }
        append(
            FmbRouteFamily::MUTABLE_DMA,
            amc == 1 ? HYVLA_EXPERT_PRE_TIME_BROADCAST_DMA_SITE
                     : HYVLA_EXPERT_PRE_TIME_SCATTER_DMA_SITE,
            static_cast<int64_t>(
                amc == 1
                    ? HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_FIXED
                    : HyVlaExpertDmaRoute::DDR_SCATTER_TO_SPM_FIXED),
            {amc, h, h / amc, DWIDTH});
        append(
            FmbRouteFamily::MUTABLE_DMA,
            HYVLA_EXPERT_PRE_MO_BIAS_DMA_SITE,
            static_cast<int64_t>(
                HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_FIXED),
            {amc, h});
        append_linear(
            HYVLA_EXPERT_PRE_WC_LINEAR_SITE,
            {cs, h, np, 1, amc, 1});
        append(
            FmbRouteFamily::ACTIVATION,
            HYVLA_EXPERT_PRE_ENCODER_ACTIVATION_SITE,
            static_cast<int64_t>(HyVlaExpertActivationRoute::SILU_UNARY),
            {amc, cs * (h / amc)});
        append_linear(
            amc == 1 ? HYVLA_EXPERT_PRE_MO_SINGLE_LINEAR_SITE
                     : HYVLA_EXPERT_PRE_MO_MULTI_LINEAR_SITE,
            {cs, h, h, amc == 1 ? 1 : 0, amc, amc == 1 ? 1 : 0});
        if (amc > 1) {
            append(
                FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_PRE_MO_ZERO_SITE,
                static_cast<int64_t>(
                    HyVlaExpertGraphScheduleRoute::ZERO_A6_RESIDUAL),
                {amc, cs * h});
            append(
                FmbRouteFamily::ALL_REDUCE,
                HYVLA_EXPERT_PRE_MO_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(cs, h),
                {cs, h, amc, amc});
            append(
                FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_PRE_MO_BIAS_ADD_SITE,
                static_cast<int64_t>(
                    HyVlaExpertGraphScheduleRoute::ADD_A6_BIAS_AFTER_REDUCE),
                {amc, cs, h});
        }
        append_linear(
            HYVLA_EXPERT_POST_OP_LINEAR_SITE,
            {cs, np, h, 1, 1, 1});
        append(
            FmbRouteFamily::MUTABLE_DMA,
            HYVLA_EXPERT_POST_TRAJECTORY_DMA_SITE,
            static_cast<int64_t>(
                HyVlaExpertDmaRoute::SPM_TO_DDR_MUTABLE),
            {num_steps_, cs * np});
    }
    append_causal_decoder_preload_manifest_routes(manifest);
    append_fmb_shared_runtime_routes(
        manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA,
        /*compute_row_multiplier=*/1, num_cores(), mlp_tp());
    return manifest;
}

FmbPhysicalManifestForwardCapability
HyVlaExpertModel::physical_manifest_forward_capability(
    const FmbPhysicalExecutionManifest& manifest) const {
    if (manifest.logical_length < 0 || manifest.kv_logical_length < manifest.logical_length) return {};
    const int64_t position = manifest.kv_logical_length - manifest.logical_length;
    std::array<bool, 7> nvfp4_scale_roles{};
    for (const auto& route : manifest.routes) {
        if (route.site_id != CAUSAL_DECODER_NVFP4_SCALE_PRELOAD_SITE) continue;
        const int64_t role = route.invocation;
        if (!nvfp4_ || role < 0 || role >= 7 || nvfp4_scale_roles[role] ||
            route.family != FmbRouteFamily::MUTABLE_DMA || route.flags != 0 ||
            route.selector != static_cast<int64_t>(CausalDecoderMutableDmaRoute::DDR_BROADCAST_TO_SPM) ||
            route.arguments != bind_cold_topology_arguments({role, 32, 64, 1})) return {};
        nvfp4_scale_roles[role] = true;
    }
    if (nvfp4_ && !std::all_of(nvfp4_scale_roles.begin(), nvfp4_scale_roles.end(), [](bool x) { return x; })) return {};
    const FmbRouteManifestEntry* kv = nullptr;
    const FmbRouteManifestEntry* kv_padding = nullptr;
    for (const auto& route : manifest.routes) {
        if (route.site_id == HYVLA_EXPERT_KV_SITE) {
            if (kv) return {};
            kv = &route;
        }
        if (route.site_id == HYVLA_EXPERT_KV_PADDING_SITE) {
            if (kv_padding) return {};
            kv_padding = &route;
        }
    }
    if (!kv) return {};
    const auto plan = restore_kvinsert_plan_from_manifest(
        manifest, HYVLA_EXPERT_KV_SITE, kv->invocation,
        attn_tp(), num_kv_heads(), head_dim());
    if (plan.segment_count() == 0 || plan.segment(0).position != position) return {};
    const bool fused_qk_norm_kv = qk_rope_norm_kv_enabled(
        manifest.logical_length, position, plan);
    if (fused_qk_norm_kv && (manifest.logical_length != 51 || manifest.physical_length != 51 ||
            !optimized_prefix_supported(position) || kv->family != FmbRouteFamily::KV_INSERT ||
            kv->selector != static_cast<int64_t>(KvInsertRoute::PAD16_V16) ||
            kv->flags != HYVLA_EXPERT_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED || kv->invocation != 0 ||
            !kv_padding || kv_padding->family != FmbRouteFamily::KV_INSERT ||
            kv_padding->selector != static_cast<int64_t>(HyVlaExpertKvPaddingRoute::PAD16_ROWS) ||
            kv_padding->flags != 0 || kv_padding->invocation != 0 ||
            kv_padding->arguments != std::vector<int64_t>{1, 51, 64, 64, 8, 8, 128})) return {};
    const FmbRouteManifestEntry* pair = nullptr;
    const FmbRouteManifestEntry* resident = nullptr;
    const FmbRouteManifestEntry* preload = nullptr;
    const FmbRouteManifestEntry* swiglu = nullptr;
    const FmbRouteManifestEntry* resident_swiglu = nullptr;
    const FmbRouteManifestEntry* resident_scales = nullptr;
    const FmbRouteManifestEntry* ring_postnorm = nullptr;
    const FmbRouteManifestEntry* o_ring_postnorm = nullptr;
    const FmbRouteManifestEntry* o_resident = nullptr;
    const FmbRouteManifestEntry* qk_rope = nullptr;
    const FmbRouteManifestEntry* qk_rope_norm = nullptr;
    const FmbRouteManifestEntry* qk_rope_norm_kv = nullptr;
    const FmbRouteManifestEntry* grouped_qkv = nullptr;
    const FmbRouteManifestEntry* grouped_kv = nullptr;
    const FmbRouteManifestEntry* qkv_linear = nullptr;
    const FmbRouteManifestEntry* q_n32 = nullptr;
    const FmbRouteManifestEntry* sdpa_m16 = nullptr;
    const FmbRouteManifestEntry* sdpa_k128 = nullptr;
    const bool k128_attention = sdpa_k128_enabled(manifest.logical_length, position);
    const FmbRouteManifestEntry* sdpa_attention = nullptr;
    const bool m16_attention = sdpa_m16_enabled(
        manifest.logical_length, position);
    const int qkv_mode = nvfp4_qkv_mode(
        manifest.logical_length, position);
    const bool kv_grouped = kv_grouped_enabled(manifest.logical_length, position);
    const bool n32_q = nvfp4_q_n32_enabled(
        manifest.logical_length, position);
    const FmbRouteManifestEntry* down_nextnorm = nullptr;
    const FmbRouteManifestEntry* down_nvfp4_n128 = nullptr;
    const FmbRouteManifestEntry* down_nvfp4_nextnorm = nullptr;
    const bool fused_nvfp4_down = down_nvfp4_nextnorm_enabled(
        manifest.logical_length, position);
    const bool n128_down = down_nvfp4_n128_enabled(
        manifest.logical_length, position);
    const bool fused_down_nextnorm = down_nextnorm_enabled(
        manifest.logical_length, position);
    const bool fused_qk_norm = qk_rope_norm_enabled(
        manifest.logical_length, position);
    const bool fused_qk_rope = qk_rope_enabled(
        manifest.logical_length, position);
    const bool resident_o = o_resident_enabled(
        manifest.logical_length, position);
    const bool fused_o_ring_norm = o_ring_postnorm_enabled(
        manifest.logical_length, position);
    const bool fused_ring_norm = !fused_o_ring_norm && ring_postnorm_enabled(
        manifest.logical_length, position);
    const bool fused_resident = resident_swiglu_enabled(
        manifest.logical_length, position);
    const bool spm_scales = resident_fp8_scales_enabled(
        manifest.logical_length, position);
    if (resident_fp8_scale_layers_ && !spm_scales) return {};
    const bool fused_gateup = gateup_swiglu_enabled(
        manifest.logical_length, position);
    for (const auto& route : manifest.routes) {
        if (route.site_id == HYVLA_EXPERT_NVFP4_RESIDENT_SCALES_SITE) {
            if (resident_scales) return {};
            resident_scales = &route;
        }
        if (route.site_id == HYVLA_EXPERT_SDPA_M16_SITE) {
            if (sdpa_m16) return {};
            sdpa_m16 = &route;
        }
        if (route.site_id == HYVLA_EXPERT_SDPA_K128_SITE) {
            if (sdpa_k128) return {};
            sdpa_k128 = &route;
        }
        if ((m16_attention || k128_attention) && route.site_id == HYVLA_EXPERT_ATTN_SITE) {
            if (sdpa_attention) return {};
            sdpa_attention = &route;
        }
        if (route.site_id == HYVLA_EXPERT_QK_ROPE_NORM_KV_SITE) {
            if (qk_rope_norm_kv) return {};
            qk_rope_norm_kv = &route;
        }
        if (fused_qk_norm_kv && (route.site_id == HYVLA_EXPERT_QK_ROPE_NORM_SITE ||
                                 route.site_id == HYVLA_EXPERT_PAIRED_KV_SITE)) return {};
        if (route.site_id == HYVLA_EXPERT_DOWN_NVFP4_NEXTNORM_SITE) {
            if (down_nvfp4_nextnorm) return {};
            down_nvfp4_nextnorm = &route;
        }
        if (fused_nvfp4_down && (route.site_id == HYVLA_EXPERT_DOWN_LINEAR_SITE ||
                                 route.site_id == HYVLA_EXPERT_DOWN_NEXTNORM_SITE)) return {};
        if (route.site_id == HYVLA_EXPERT_DOWN_NVFP4_N128_SITE) {
            if (down_nvfp4_n128) return {};
            down_nvfp4_n128 = &route;
        }
        if (n128_down && route.site_id == HYVLA_EXPERT_DOWN_LINEAR_SITE) return {};
        if (route.site_id == HYVLA_EXPERT_DOWN_NEXTNORM_SITE) {
            if (down_nextnorm) return {};
            down_nextnorm = &route;
        }
        if (fused_down_nextnorm && (route.site_id == HYVLA_EXPERT_MLP_ALL_REDUCE_SITE ||
                                    route.site_id == HYVLA_EXPERT_FINAL_NORM_SITE)) return {};
        if (route.site_id == HYVLA_EXPERT_NVFP4_QKV_SITE) {
            if (grouped_qkv) return {};
            grouped_qkv = &route;
        }
        if (route.site_id == HYVLA_EXPERT_KV_GROUPED_SITE) {
            if (grouped_kv) return {};
            grouped_kv = &route;
        }
        if (route.site_id == HYVLA_EXPERT_NVFP4_Q_N32_SITE) {
            if (q_n32) return {};
            q_n32 = &route;
        }
        if ((qkv_mode || kv_grouped) && route.site_id == HYVLA_EXPERT_QKV_LINEAR_SITE) {
            if (qkv_mode == 3 || n32_q || qkv_linear) return {};
            qkv_linear = &route;
        }
        if (route.site_id == HYVLA_EXPERT_QK_ROPE_NORM_SITE) {
            if (qk_rope_norm) return {};
            qk_rope_norm = &route;
        }
        if (fused_qk_norm && (route.site_id == HYVLA_EXPERT_QK_ROPE_SITE ||
                              route.site_id == HYVLA_EXPERT_Q_NORM_SITE ||
                              route.site_id == HYVLA_EXPERT_K_NORM_SITE)) return {};
        if (route.site_id == HYVLA_EXPERT_QK_ROPE_SITE) {
            if (qk_rope) return {};
            qk_rope = &route;
        }
        if (fused_qk_rope && (route.site_id == HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE ||
                             route.site_id == HYVLA_EXPERT_K_PARTIAL_ROPE_SITE ||
                             route.site_id == HYVLA_EXPERT_Q_ROPE_SITE ||
                             route.site_id == HYVLA_EXPERT_K_ROPE_SITE)) return {};
        if (route.site_id == HYVLA_EXPERT_O_RESIDENT_SITE) {
            if (o_resident) return {};
            o_resident = &route;
        }
        if (route.site_id == HYVLA_EXPERT_O_RING_POSTNORM_SITE) {
            if (o_ring_postnorm) return {};
            o_ring_postnorm = &route;
        }
        if (fused_o_ring_norm && (route.site_id == HYVLA_EXPERT_O_LINEAR_SITE ||
                                 route.site_id == HYVLA_EXPERT_RING_POSTNORM_SITE)) return {};
        if (route.site_id == HYVLA_EXPERT_RING_POSTNORM_SITE) {
            if (ring_postnorm) return {};
            ring_postnorm = &route;
        }
        if ((fused_ring_norm || fused_o_ring_norm) && (route.site_id == HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE ||
                               route.site_id == HYVLA_EXPERT_POST_NORM_SITE)) return {};
        if (route.site_id == HYVLA_EXPERT_RESIDENT_SWIGLU_SITE ||
            route.site_id == HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE) {
            if (resident_swiglu || route.site_id != (nvfp4_
                    ? HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE
                    : HYVLA_EXPERT_RESIDENT_SWIGLU_SITE)) return {};
            resident_swiglu = &route;
        }
        if (fused_resident && (route.site_id == HYVLA_EXPERT_MLP_SILU_MUL_SITE ||
                              route.site_id == HYVLA_EXPERT_MLP_SILU_SITE ||
                              route.site_id == HYVLA_EXPERT_MLP_MUL_SITE)) return {};
        if (route.site_id == HYVLA_EXPERT_GATEUP_SWIGLU_SITE) {
            if (swiglu) return {};
            swiglu = &route;
        }
        if (fused_gateup && (route.site_id == HYVLA_EXPERT_GATE_LINEAR_SITE ||
                            route.site_id == HYVLA_EXPERT_UP_LINEAR_SITE)) return {};
        if (route.site_id == HYVLA_EXPERT_GATEUP_RESIDENT_SITE) {
            if (resident) return {};
            resident = &route;
        }
        if (route.site_id == HYVLA_EXPERT_GATEUP_PRELOAD_SITE) {
            if (preload) return {};
            preload = &route;
        }
        if (route.site_id == HYVLA_EXPERT_PAIRED_KV_SITE) {
            if (pair != nullptr) return {};
            pair = &route;
        }
    }
    if (spm_scales != (resident_scales != nullptr)) return {};
    if (resident_scales && (manifest.physical_length != 51 ||
            resident_scales->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            resident_scales->selector != 1 || resident_scales->flags != 0 ||
            resident_scales->invocation != 0 || resident_scales->arguments !=
                resident_fp8_scales_arguments(position))) return {};
    if (fused_nvfp4_down != (down_nvfp4_nextnorm != nullptr)) return {};
    if (down_nvfp4_nextnorm && (manifest.physical_length != 51 ||
            down_nvfp4_nextnorm->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            down_nvfp4_nextnorm->selector != 1 || down_nvfp4_nextnorm->flags != 0 ||
            down_nvfp4_nextnorm->invocation != 0 ||
            down_nvfp4_nextnorm->arguments != down_nvfp4_nextnorm_arguments(position))) return {};
    if (n128_down != (down_nvfp4_n128 != nullptr)) return {};
    if (down_nvfp4_n128 && (manifest.physical_length != 51 ||
            down_nvfp4_n128->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            down_nvfp4_n128->selector != 1 || down_nvfp4_n128->flags != 0 ||
            down_nvfp4_n128->invocation != 0 ||
            down_nvfp4_n128->arguments != down_nvfp4_n128_arguments(position))) return {};
    if ((fused_down_nextnorm && !fused_nvfp4_down) != (down_nextnorm != nullptr)) return {};
    if (down_nextnorm && (manifest.physical_length != 51 ||
            down_nextnorm->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            down_nextnorm->selector != 1 || down_nextnorm->flags != 0 ||
            down_nextnorm->invocation != 0 ||
            down_nextnorm->arguments != down_nextnorm_arguments(position))) return {};
    if (bool(qkv_mode) != (grouped_qkv != nullptr)) return {};
    if (kv_grouped != (grouped_kv != nullptr)) return {};
    if (grouped_kv && (manifest.physical_length != 51 ||
            grouped_kv->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            grouped_kv->selector != 1 || grouped_kv->flags != 0 ||
            grouped_kv->invocation != 0 ||
            grouped_kv->arguments != kv_grouped_arguments(position))) return {};
    if (k128_attention != (sdpa_k128 != nullptr)) return {};
    if (sdpa_k128 && (manifest.physical_length != 51 ||
            sdpa_k128->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            sdpa_k128->selector != 1 || sdpa_k128->flags != 0 || sdpa_k128->invocation != 0 ||
            sdpa_k128->arguments != sdpa_k128_arguments(position) || !sdpa_attention ||
            sdpa_attention->family != FmbRouteFamily::ATTENTION ||
            sdpa_attention->selector != static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV) ||
            sdpa_attention->flags != 0 || sdpa_attention->invocation != 0 ||
            !sdpa_attention->arguments.empty())) return {};
    if (m16_attention != (sdpa_m16 != nullptr)) return {};
    if (sdpa_m16 && (manifest.physical_length != 51 ||
            sdpa_m16->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            sdpa_m16->selector != 1 || sdpa_m16->flags != 0 || sdpa_m16->invocation != 0 ||
            sdpa_m16->arguments != sdpa_m16_arguments(position) || !sdpa_attention ||
            sdpa_attention->family != FmbRouteFamily::ATTENTION ||
            sdpa_attention->selector != static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV) ||
            sdpa_attention->flags != 0 || sdpa_attention->invocation != 0 ||
            !sdpa_attention->arguments.empty())) return {};
    if (grouped_qkv && (manifest.physical_length != 51 ||
            grouped_qkv->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            grouped_qkv->selector != 1 || grouped_qkv->flags != 0 ||
            grouped_qkv->invocation != 0 ||
            grouped_qkv->arguments != nvfp4_qkv_arguments(qkv_mode, position))) return {};
    if (n32_q != (q_n32 != nullptr)) return {};
    if (q_n32 && (manifest.physical_length != 51 ||
            q_n32->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            q_n32->selector != 1 || q_n32->flags != 0 || q_n32->invocation != 0 ||
            q_n32->arguments != nvfp4_q_n32_arguments(position))) return {};
    if (((qkv_mode == 2 && !n32_q) || kv_grouped) && (!qkv_linear || qkv_linear->family != FmbRouteFamily::LINEAR ||
            qkv_linear->selector != static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE) ||
            qkv_linear->flags != 0 || qkv_linear->invocation != 0 ||
            !qkv_linear->arguments.empty())) return {};
    if (fused_qk_norm_kv != (qk_rope_norm_kv != nullptr)) return {};
    if (qk_rope_norm_kv && (qk_rope_norm_kv->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            qk_rope_norm_kv->selector != 1 || qk_rope_norm_kv->flags != 0 ||
            qk_rope_norm_kv->invocation != 0 ||
            qk_rope_norm_kv->arguments != qk_rope_norm_kv_arguments(plan, position))) return {};
    if ((fused_qk_norm && !fused_qk_norm_kv) != (qk_rope_norm != nullptr)) return {};
    if (qk_rope_norm && (manifest.physical_length != 51 ||
            qk_rope_norm->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            qk_rope_norm->selector != 1 || qk_rope_norm->flags != 0 ||
            qk_rope_norm->invocation != 0 ||
            qk_rope_norm->arguments != qk_rope_norm_arguments(position))) return {};
    if ((fused_qk_rope && !fused_qk_norm) != (qk_rope != nullptr)) return {};
    if (qk_rope && (manifest.physical_length != 51 ||
            qk_rope->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            qk_rope->selector != 1 || qk_rope->flags != 0 ||
            qk_rope->invocation != 0 ||
            qk_rope->arguments != qk_rope_arguments(position))) return {};
    if (resident_o != (o_resident != nullptr)) return {};
    if (o_resident && (manifest.physical_length != 51 ||
            o_resident->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            o_resident->selector != 1 || o_resident->flags != 0 ||
            o_resident->invocation != 0 ||
            o_resident->arguments != o_resident_arguments(position))) return {};
    if (fused_o_ring_norm != (o_ring_postnorm != nullptr)) return {};
    if (o_ring_postnorm && (manifest.physical_length != 51 ||
            o_ring_postnorm->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            o_ring_postnorm->selector != 1 || o_ring_postnorm->flags != 0 ||
            o_ring_postnorm->invocation != 0 ||
            o_ring_postnorm->arguments != o_ring_postnorm_arguments(position))) return {};
    if (fused_ring_norm != (ring_postnorm != nullptr)) return {};
    if (ring_postnorm && (manifest.physical_length != 51 ||
            ring_postnorm->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            ring_postnorm->selector != 1 || ring_postnorm->flags != 0 ||
            ring_postnorm->invocation != 0 ||
            ring_postnorm->arguments != ring_postnorm_arguments(position))) return {};
    if (fused_resident != (resident_swiglu != nullptr)) return {};
    if (resident_swiglu && (manifest.physical_length != 51 ||
            resident_swiglu->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            resident_swiglu->selector != 1 || resident_swiglu->flags != 0 ||
            resident_swiglu->invocation != 0 ||
            resident_swiglu->arguments != resident_swiglu_arguments(position))) return {};
    if (fused_gateup != (swiglu != nullptr)) return {};
    if (swiglu && (manifest.physical_length != 51 ||
                   swiglu->family != FmbRouteFamily::GRAPH_SCHEDULE ||
                   swiglu->selector != 1 || swiglu->flags != 0 || swiglu->invocation != 0 ||
                   swiglu->arguments != gateup_swiglu_arguments(position))) return {};
    if (gateup_resident_bundle_.defined()) {
        if (!gateup_resident_profile() || manifest.logical_length != 51 ||
            !optimized_prefix_supported(position) || !resident || !preload ||
            resident->family != FmbRouteFamily::GRAPH_SCHEDULE ||
            resident->selector != 1 || resident->flags != 0 || resident->invocation != 0 ||
            resident->arguments != gateup_resident_arguments() ||
            preload->family != FmbRouteFamily::MUTABLE_DMA || preload->flags != 0 ||
            preload->invocation != 0 || preload->selector != static_cast<int64_t>(
                HyVlaExpertDmaRoute::DDR_SCATTER_TO_SPM_FIXED) ||
            preload->arguments != std::vector<int64_t>{gateup_resident_layers(),
                resident_storage_bytes() / 2, resident_storage_bytes(), NUM_CORES, 0}) return {};
    } else if (resident || preload) return {};
    const bool enabled = !fused_qk_norm_kv && paired_kv_enabled(plan);
    if (enabled != (pair != nullptr)) return {};
    if (pair && (pair->family != FmbRouteFamily::GRAPH_SCHEDULE ||
                 pair->selector != 1 || pair->flags != 0 ||
                 pair->invocation != kv->invocation ||
                 pair->arguments != paired_kv_arguments(plan))) return {};
    return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
}

// ─────────────────────────────────────────────────────────────────────────────
// subclass_chunk_size_valid — suffix 整块进 SDPA
// ─────────────────────────────────────────────────────────────────────────────
bool HyVlaExpertModel::subclass_chunk_size_valid(
    int64_t cs, int64_t seq_len, int64_t /*position*/) const {
    return cs >= seq_len && chunk_within_envelope(cs);
}

void HyVlaExpertModel::prepare_persistent_spm_for_shape(
    int64_t seq_len, int64_t prefix_len, int64_t mask_kv_len) {
    TORCH_CHECK(seq_len == expert_chunk_size_,
                "hyvla_expert_prepare_persistent_spm: seq_len must match the "
                "fixed action suffix geometry");
    TORCH_CHECK(prefix_len >= 0 && mask_kv_len == prefix_len + seq_len,
                "hyvla_expert_prepare_persistent_spm: mask width must equal "
                "prefix_len + seq_len");
    prepare_persistent_spm(seq_len, prefix_len, /*is_causal=*/false, mask_kv_len);
}

std::vector<int64_t> HyVlaExpertModel::resolve_stage_domain(
    int64_t seq_len, int64_t prefix_len, int64_t mask_kv_len) {
    TORCH_CHECK(seq_len == expert_chunk_size_,
                "hyvla_expert_resolve_stage_domain: seq_len must match the "
                "fixed action suffix geometry");
    TORCH_CHECK(prefix_len >= 0 && mask_kv_len == prefix_len + seq_len,
                "hyvla_expert_resolve_stage_domain: mask width must equal "
                "prefix_len + seq_len");
    const at::Tensor mask = at::empty(
        {1, mask_kv_len},
        at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
    return encode_fmb_prefill_stage_domain(
        resolve_prefill_stage_domain_for_shape(
            seq_len, prefix_len, std::optional<at::Tensor>(mask),
            /*is_causal=*/false));
}

// ─────────────────────────────────────────────────────────────────────────────
// build_layer_subgraph — 单塔层体 (基类 8 相, 步 3 换序)
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    TORCH_CHECK(
        ctx().has_complete_physical_manifest(),
        "HyVlaExpertModel requires a COMPLETE physical descriptor");
    TORCH_CHECK(!layers_.empty(),
                "hyvla_expert build_layer_subgraph: set_weights_expert must "
                "have been called");
    TORCH_CHECK(ctx().attention_mask.has_value(),
                "hyvla_expert build_layer_subgraph: explicit 2D mask required");

    const auto& lw = layers_[layer_idx];
    const int64_t seq_len = chunk.len;
    const int64_t cos_sin_start = ctx().position + chunk.offset;
    const bool is_last_layer = (layer_idx == num_layers() - 1);
    const int64_t h   = hidden_size();
    const int64_t nq  = num_q_heads();
    const int64_t nkv = num_kv_heads();
    const int64_t hd  = head_dim();
    const int tp = attn_tp();
    const int64_t local_q  = nq / tp;
    const int64_t elems_mlp = seq_len * (intermediate_size() / mlp_tp());
    const int64_t allocation_rows = ctx().stage_plan.compute.chunks.front().len;
    const int64_t kv_pad_rows_ = cold_kvpad16_
        ? Align(allocation_rows, int64_t{16}) : 0;
    const int64_t rmsnorm_pad_rows_ = cold_rmsnorm_pad16_
        ? Align(allocation_rows, int64_t{16}) : 0;
    const int64_t rmsnorm_pad_qrows_ = cold_rmsnorm_pad16_
        ? Align(allocation_rows * local_q, int64_t{16}) : 0;


    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_EXPERT_GRAPH_SCHEDULE_SITE,
            static_cast<int64_t>(
                cold_fast_replay_
                    ? HyVlaExpertGraphScheduleRoute::FAST_REPLAY_SKIP_LAYER_LOOP
                    : HyVlaExpertGraphScheduleRoute::REEMIT_LAYER_LOOP),
            /*resolved_flags=*/0,
            {cold_fast_replay_ ? 1 : 0,
             cold_fast_replay_preload_ ? 1 : 0,
             unroll_mode_ ? 1 : 0,
             unroll_mode_ ? num_steps_ : 1},
            chunk.idx);
    }

    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }

    const auto rmsnorm_route = [&](int64_t site, int64_t rows, int64_t cols) {
        const auto route = cold_rmsnorm_spm_contract_.resolve(rows, cols);
        TORCH_CHECK(
            !cold_rmsnorm_spm_contract_.has_vector_payload() ||
                ctx().has_complete_physical_manifest(),
            "HyVlaExpertModel non-BASE RMSNorm route requires a COMPLETE "
            "physical descriptor");
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::NORMALIZATION, site, static_cast<int64_t>(route),
                /*resolved_flags=*/0,
                cold_rmsnorm_spm_contract_.manifest_arguments(rows, cols), chunk.idx);
        }
        return route;
    };
    const auto full_norm_route = rmsnorm_route(
        HYVLA_EXPERT_FULL_RMSNORM_ROUTE_SITE,
        rmsnorm_pad_rows_ ? rmsnorm_pad_rows_ : seq_len, h);
    const auto q_head_norm_route = rmsnorm_route(
        HYVLA_EXPERT_Q_RMSNORM_ROUTE_SITE,
        rmsnorm_pad_qrows_ ? rmsnorm_pad_qrows_ : seq_len * local_q, hd);
    const auto k_head_norm_route = rmsnorm_route(
        HYVLA_EXPERT_K_RMSNORM_ROUTE_SITE,
        (rmsnorm_pad_rows_ ? rmsnorm_pad_rows_ : seq_len) * (nkv / tp), hd);

    // ── 步 1: input RMSNorm ──
    // RMSNORM_PAD16：补齐到 16 的倍数行才能走 `llama_rms_norm_v16`（见文件头）。
    // 多算的 [seq_len, pad) 行下游按 M=seq_len 消费 ⇒ 永不被读。
    const int64_t nrows  = rmsnorm_pad_rows_  ? rmsnorm_pad_rows_  : seq_len;
    const bool fused_down_nextnorm = down_nextnorm_enabled(seq_len, cos_sin_start);
    if (!fused_down_nextnorm || layer_idx == 0) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::NORMALIZATION,
                HYVLA_EXPERT_INPUT_NORM_SITE,
                static_cast<int64_t>(
                    cold_rmsnorm_pad16_
                        ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                        : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
                /*resolved_flags=*/0,
                {cold_rmsnorm_pad16_ ? 1 : 0, nrows, h}, chunk.idx);
        }
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "norm_w"), nrows, h, eps_expert_, full_norm_route, num_cores());
    }

    // ── 步 2: QKV (无 bias — Hy-VLA attention_bias=false) ──
    auto qkv = [&](const at::Tensor& w, const at::Tensor& ws,
                   const char* out, const char* ts, int64_t n) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVLA_EXPERT_QKV_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), w, addr(0, out),
            seq_len, n, h, /*partition=*/1, /*num_cores=*/tp,
            /*bias_spm_addr=*/0u, /*scale=*/ws, nvfp4_scale_addr(ts), uint16_t(layer_idx));
    };
    const int qkv_mode = nvfp4_qkv_mode(seq_len, cos_sin_start);
    if (nvfp4_q_n32_enabled(seq_len, cos_sin_start)) {
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_EXPERT_NVFP4_Q_N32_SITE, 1, 0, nvfp4_q_n32_arguments(cos_sin_start), chunk.idx);
        launch_hyvla_nvfp4_q_tuning_spm(addr(0, "input_norm"), lw.q_w, lw.q_ws,
            nvfp4_scale_addr("nvfp4_q_ts"), addr(0, "q"), uint16_t(layer_idx),
            /*overlap=*/false, seq_len, tp);
    } else if (qkv_mode != 3) qkv(lw.q_w, lw.q_ws, "q", "nvfp4_q_ts", nq * hd);
    if (qkv_mode) {
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_EXPERT_NVFP4_QKV_SITE, 1, 0, nvfp4_qkv_arguments(qkv_mode, cos_sin_start), chunk.idx);
        std::vector<at::Tensor> weights{lw.k_w, lw.v_w};
        std::vector<at::Tensor> scales{lw.k_ws, lw.v_ws};
        std::vector<uint32_t> tables{nvfp4_scale_addr("nvfp4_k_ts"), nvfp4_scale_addr("nvfp4_v_ts")};
        std::vector<uint32_t> outputs{addr(0, "k"), addr(0, "v")};
        if (qkv_mode == 3) {
            weights.insert(weights.begin(), lw.q_w);
            scales.insert(scales.begin(), lw.q_ws);
            tables.insert(tables.begin(), nvfp4_scale_addr("nvfp4_q_ts"));
            outputs.insert(outputs.begin(), addr(0, "q"));
        }
        if (qkv_mode == 3 && cold_nvfp4_qkv_overlap_available_) {
            launch_hyvla_nvfp4_qkv_overlap_spm(addr(0, "input_norm"), weights, scales,
                tables, outputs, uint16_t(layer_idx), seq_len, tp);
        } else {
            launch_hyvla_nvfp4_qkv_spm(addr(0, "input_norm"), weights, scales, tables,
                                      outputs, uint16_t(layer_idx), qkv_mode == 3, seq_len, tp);
        }
    } else if (kv_grouped_enabled(seq_len, cos_sin_start)) {
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "HyVLA grouped KV requires a COMPLETE descriptor");
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_EXPERT_KV_GROUPED_SITE, 1, 0, kv_grouped_arguments(cos_sin_start), chunk.idx);
        launch_hyvla_kv_grouped_spm(addr(0, "input_norm"), {lw.k_w, lw.v_w},
            {lw.k_ws, lw.v_ws}, {addr(0, "k"), addr(0, "v")}, seq_len, tp);
    } else {
        qkv(lw.k_w, lw.k_ws, "k", "nvfp4_k_ts", nkv * hd);
        qkv(lw.v_w, lw.v_ws, "v", "nvfp4_v_ts", nkv * hd);
    }

    // ── 步 3: **先 RoPE, 再 q/k head-norm**(与基类及其余 7 个 emitter 相反) ──
    // 依据 vendor modeling_dual_tower.py: apply_rotary_pos_emb 之后才调
    // query_layernorm/key_layernorm, 且两塔共用 VLM 层的那一份 (绑定方负责喂对)。
    // 与 VLM 塔 (rpu_hyvla_vlm_model.cpp 步 3) 完全同构, 只是这里没有孪生分支。
    // 换序可行的前提: 两个 kernel 都是 src/dst 分离、无隐藏状态的独立 SPM launch,
    // 且 QK-norm 的 framing (M=seq*heads, C=head_dim) 不受 RoPE 影响 —— RoPE 不改

    c10::Half* cos_ptr = cos_ref_.data_ptr<c10::Half>();
    c10::Half* sin_ptr = sin_ref_.data_ptr<c10::Half>();

    // PARTIAL_ROPE（见文件头）：与 llama_rope 逐位相等，但 grid 从 (seq,1,1) 塌成
    // (⌈hd/2/64⌉, ⌈seq/64⌉)。表和偏移原样复用。
    const bool prope = cold_partial_rope_;
    const int64_t local_kv_heads = nkv / tp;

    // Resolve and consume the actual KV receipts before any combined QK/cache
    // launch. The cache owners and exact plan are shared with the old fallback.
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    const int64_t expected_kv_physical_rows = cold_kvpad16_
        ? kv_pad_rows_ : seq_len;
    ctx().consume_physical_route(
        FmbRouteFamily::KV_INSERT,
        HYVLA_EXPERT_KV_PADDING_SITE,
        static_cast<int64_t>(
            cold_kvpad16_ ? HyVlaExpertKvPaddingRoute::PAD16_ROWS
                          : HyVlaExpertKvPaddingRoute::LOGICAL_ROWS),
        /*resolved_flags=*/0,
        {cold_kvpad16_ ? 1 : 0, seq_len, kv_pad_rows_,
         expected_kv_physical_rows, tp, nkv, hd}, chunk.idx);
    const FmbRouteManifestEntry& route = ctx().find_physical_route(
        FmbRouteFamily::KV_INSERT, HYVLA_EXPERT_KV_SITE, chunk.idx);
    const KvInsertSegmentPlan kv_plan =
        restore_kvinsert_plan(
            HYVLA_EXPERT_KV_SITE, route.arguments, tp, nkv, hd);
    TORCH_CHECK(
        kv_plan.logical_rows() == seq_len &&
            kv_plan.physical_rows() == expected_kv_physical_rows &&
            kv_plan.segment(0).position == cos_sin_start,
        "HyVlaExpertModel KV descriptor geometry drift at invocation ",
        chunk.idx);
    ctx().consume_physical_route(
        FmbRouteFamily::KV_INSERT, HYVLA_EXPERT_KV_SITE,
        static_cast<int64_t>(kv_plan.route()),
        HYVLA_EXPERT_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED,
        route.arguments, chunk.idx);
    const bool fused_qk_norm_kv = qk_rope_norm_kv_enabled(seq_len, cos_sin_start, kv_plan);
    if (fused_qk_norm_kv) {
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "HyVLA QK RoPE+norm+KV requires a COMPLETE descriptor");
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_EXPERT_QK_ROPE_NORM_KV_SITE, 1, 0,
            qk_rope_norm_kv_arguments(kv_plan, cos_sin_start), chunk.idx);
        launch_hyvla_qk_rope_norm_kv_spm(addr(0, "q"), addr(0, "output"),
            addr(0, "k"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "q_norm_w"), layer_addr(layer_idx, 0, "k_norm_w"),
            addr(0, "v"), cos_ref_, sin_ref_, k_cache, v_cache, seq_len, cos_sin_start,
            rmsnorm_pad_qrows_ ? rmsnorm_pad_qrows_ : seq_len * local_q,
            (rmsnorm_pad_rows_ ? rmsnorm_pad_rows_ : seq_len) * local_kv_heads,
            eps_expert_, tp, kv_plan);
    } else if (qk_rope_norm_enabled(seq_len, cos_sin_start)) {
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "HyVLA QK RoPE+norm requires a COMPLETE descriptor");
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_EXPERT_QK_ROPE_NORM_SITE, 1, 0, qk_rope_norm_arguments(cos_sin_start), chunk.idx);
        launch_hyvla_qk_rope_norm_spm(addr(0, "q"), addr(0, "output"),
            addr(0, "k"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "q_norm_w"), layer_addr(layer_idx, 0, "k_norm_w"),
            cos_ref_, sin_ref_, seq_len, cos_sin_start,
            rmsnorm_pad_qrows_ ? rmsnorm_pad_qrows_ : seq_len * local_q,
            (rmsnorm_pad_rows_ ? rmsnorm_pad_rows_ : seq_len) * local_kv_heads,
            eps_expert_, tp);
    } else {
        const bool fused_qk = qk_rope_enabled(seq_len, cos_sin_start);
        if (fused_qk) {
            TORCH_CHECK(ctx().has_complete_physical_manifest(),
                        "HyVLA shared-frequency QK RoPE requires a COMPLETE descriptor");
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_QK_ROPE_SITE, 1, 0, qk_rope_arguments(cos_sin_start), chunk.idx);
            // K RoPE may precede Q norm: their four live SPM owners are disjoint.
            launch_hyvla_qk_rope_spm(addr(0, "q"), addr(0, "output"),
                addr(0, "k"), addr(0, "input_norm"), cos_ref_, sin_ref_,
                seq_len, cos_sin_start, tp);
        } else if (prope) {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ROPE,
                    HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE,
                    static_cast<int64_t>(HyVlaExpertRopeRoute::PARTIAL_MROPE),
                    /*resolved_flags=*/0,
                    {cold_partial_rope_ ? 1 : 0, cos_sin_start,
                     seq_len, local_q, hd, tp}, chunk.idx);
            }
            rpu_launch_partial_mrope_spm_kernel(
                addr(0, "q"), addr(0, "output"), cos_ptr, sin_ptr,
                cos_sin_start, seq_len, local_q, hd, /*rotary_dim=*/hd, tp);
        } else {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ROPE, HYVLA_EXPERT_Q_ROPE_SITE,
                    static_cast<int64_t>(HyVlaExpertRopeRoute::ROPE_1D),
                    /*resolved_flags=*/0,
                    {cold_partial_rope_ ? 1 : 0, cos_sin_start,
                     seq_len, local_q, hd, tp}, chunk.idx);
            }
            rpu_launch_rope_spm_kernel(addr(0, "q"), addr(0, "output"),
                                       cos_ptr, sin_ptr, seq_len, local_q, hd,
                                       cos_sin_start, tp);
        }
        const int64_t q_norm_rows = rmsnorm_pad_qrows_
            ? rmsnorm_pad_qrows_ : seq_len * local_q;
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::NORMALIZATION, HYVLA_EXPERT_Q_NORM_SITE,
                static_cast<int64_t>(
                    cold_rmsnorm_pad16_
                        ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                        : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
                /*resolved_flags=*/0,
                {cold_rmsnorm_pad16_ ? 1 : 0, q_norm_rows, hd}, chunk.idx);
        }
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "output"), addr(0, "q"),
            layer_addr(layer_idx, 0, "q_norm_w"),
            q_norm_rows, hd, eps_expert_, q_head_norm_route, num_cores());

        if (!fused_qk) {
            if (prope) {
                if (ctx().has_complete_physical_manifest()) {
                    ctx().consume_physical_route(
                        FmbRouteFamily::ROPE,
                        HYVLA_EXPERT_K_PARTIAL_ROPE_SITE,
                        static_cast<int64_t>(HyVlaExpertRopeRoute::PARTIAL_MROPE),
                        /*resolved_flags=*/0,
                        {cold_partial_rope_ ? 1 : 0, cos_sin_start,
                         seq_len, local_kv_heads, hd, tp}, chunk.idx);
                }
                rpu_launch_partial_mrope_spm_kernel(
                    addr(0, "k"), addr(0, "input_norm"), cos_ptr, sin_ptr,
                    cos_sin_start, seq_len, local_kv_heads, hd, /*rotary_dim=*/hd, tp);
            } else {
                if (ctx().has_complete_physical_manifest()) {
                    ctx().consume_physical_route(
                        FmbRouteFamily::ROPE, HYVLA_EXPERT_K_ROPE_SITE,
                        static_cast<int64_t>(HyVlaExpertRopeRoute::ROPE_1D),
                        /*resolved_flags=*/0,
                        {cold_partial_rope_ ? 1 : 0, cos_sin_start,
                         seq_len, local_kv_heads, hd, tp}, chunk.idx);
                }
                rpu_launch_rope_spm_kernel(addr(0, "k"), addr(0, "input_norm"),
                                           cos_ptr, sin_ptr, seq_len, local_kv_heads,
                                           hd, cos_sin_start, tp);
            }
        }
        // k 槽已被 KVPAD16 撑到 Align(cs,16) 行 × local_kv 元素 ⇒ 这里补齐不需要额外撑槽。
        const int64_t k_norm_rows =
            (rmsnorm_pad_rows_ ? rmsnorm_pad_rows_ : seq_len) * local_kv_heads;
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::NORMALIZATION, HYVLA_EXPERT_K_NORM_SITE,
                static_cast<int64_t>(
                    cold_rmsnorm_pad16_
                        ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                        : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
                /*resolved_flags=*/0,
                {cold_rmsnorm_pad16_ ? 1 : 0, k_norm_rows, hd}, chunk.idx);
        }
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "input_norm"), addr(0, "k"),
            layer_addr(layer_idx, 0, "k_norm_w"),
            k_norm_rows, hd, eps_expert_, k_head_norm_route, num_cores());

    }

    // ── 步 4: KV insert (追加在 prefix 之后) + SDPA + o_proj + 残差归约 ──
    // 调用方每步前 reset_to_position(prefix_len), 所以这 51 行每步覆写同一段,
    // 等价于 vendor 的 copy.deepcopy 冻结 prefix、suffix KV 用完即弃。
    if (!fused_qk_norm_kv) {
        if (paired_kv_enabled(kv_plan)) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_PAIRED_KV_SITE,
                1, 0, paired_kv_arguments(kv_plan), chunk.idx);
            rpu_launch_hyvla_expert_paired_kv_spm_kernel(
                k_cache, v_cache, addr_offset("k").value, addr_offset("v").value,
                kv_pad_rows_, kv_plan);
        } else {
            rpu_launch_insert_kvcache_spm_unified_with_plan(
                k_cache, v_cache,
                addr_offset("k").value, addr_offset("v").value,
                nkv, hd, tp,
                /*k_cache_batch_offset_elems=*/0,
                /*v_cache_batch_offset_elems=*/0,
                kv_pad_rows_, kv_plan);
        }
    }
    const uint32_t mask_off = addr_offset("sdpa_mask").value;
    const auto& mask_route = ctx().find_physical_route(
        FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_MASK_SCHEDULE_SITE,
        chunk.idx);
    if (ctx().body_iter == 0 && layer_idx == 0) {
        // Revalidate the minted schedule against exactly the complete layout
        // used for allocation, once before any mask reuse (not per layer).
        LayoutContext layout;
        layout.chunk_size = allocation_rows;
        layout.max_kv_seq_len = ctx().attention_mask->size(-1);
        layout.num_layers = num_layers();
        layout.use_attn_mask = true;
        layout.is_causal = ctx().is_causal;
        layout.batch_size = ctx().batch_size;
        layout.attention_policy = ctx().attention_policy;
        layout.forward_operand_residency =
            physical_forward_operand_residency(ctx().physical_manifest());
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_MASK_SCHEDULE_SITE,
            mask_schedule_for_layout(layout), /*resolved_flags=*/0,
            {cold_mask_once_ ? 1 : 0, seq_len, chunk.kv_seq_len,
             tp, num_layers()}, chunk.idx);
    }
    const auto mask_schedule =
        static_cast<HyVlaExpertGraphScheduleRoute>(mask_route.selector);
    const bool upload_mask =
        mask_schedule == HyVlaExpertGraphScheduleRoute::MASK_EACH_LAYER ||
        (layer_idx == 0 &&
         (mask_schedule == HyVlaExpertGraphScheduleRoute::MASK_FIRST_LAYER_ONLY ||
          ctx().body_iter == 0));
    if (upload_mask) {
        sdpa_dma_mask_to_spm(prepared_attn_mask_, mask_off, seq_len, chunk.kv_seq_len, tp);
    }
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_attention_route(
            HYVLA_EXPERT_ATTN_SITE, AttentionExecutionPolicy::DDR_KV,
            chunk.idx);
    }
    if (sdpa_k128_enabled(seq_len, cos_sin_start)) {
        TORCH_CHECK(prepared_attn_mask_.mask_type == 4 && ctx().batch_size == 1 &&
                        ctx().attention_policy == AttentionExecutionPolicy::DDR_KV,
                    "HyVLA K128 attention requires MASK2D and single-batch DDR KV");
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_EXPERT_SDPA_K128_SITE, 1, 0, sdpa_k128_arguments(cos_sin_start), chunk.idx);
        rpu_launch_sdpa_hyvla_s51_k128(k_cache, v_cache,
            addr_offset("q").value, addr_offset("output").value,
            addr_offset("sdpa_tmp").value, mask_off,
            seq_len, chunk.kv_seq_len, HYVLA_S51_K128_SDPA_TMP_BYTES);
    } else if (sdpa_m16_enabled(seq_len, cos_sin_start)) {
        TORCH_CHECK(prepared_attn_mask_.mask_type == 4 && ctx().batch_size == 1 &&
                        ctx().attention_policy == AttentionExecutionPolicy::DDR_KV,
                    "HyVLA M16 attention requires MASK2D and single-batch DDR KV");
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_EXPERT_SDPA_M16_SITE, 1, 0, sdpa_m16_arguments(cos_sin_start), chunk.idx);
        rpu_launch_sdpa_hyvla_s51_m16(k_cache, v_cache,
            addr_offset("q").value, addr_offset("output").value,
            addr_offset("sdpa_tmp").value, mask_off,
            seq_len, chunk.kv_seq_len, HYVLA_S51_M16_SDPA_TMP_BYTES);
    } else {
        rpu_launch_sdpa_spm_dispatch(
            SdpaKernelType::FLASH_ATTN_SPM,
            k_cache, v_cache, prepared_attn_mask_.mask_type, c10::nullopt,
            addr_offset("q").value, addr_offset("output").value,
            addr_offset("sdpa_tmp").value, mask_off,
            seq_len, nq, nkv, hd, chunk.kv_seq_len, tp, NUM_CORES);
    }
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            HYVLA_EXPERT_PREPARE_ATTN_ALL_REDUCE_SITE,
            static_cast<int64_t>(
                HyVlaExpertAllReduceRoute::PREPARE_RING_INPUT),
            /*resolved_flags=*/0, {seq_len, h, tp}, chunk.idx);
    }
    rpu_prepare_ring_all_reduce_input(addr(0, "oproj"), seq_len, h, tp, num_cores());
    if (o_ring_postnorm_enabled(seq_len, cos_sin_start)) {
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "HyVLA O ring postnorm requires a COMPLETE descriptor");
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_EXPERT_O_RING_POSTNORM_SITE, 1, 0, o_ring_postnorm_arguments(cos_sin_start), chunk.idx);
        if (nvfp4_)
            launch_hyvla_nvfp4_o_ring_postnorm_spm(addr(0, "output"), lw.o_w, lw.o_ws,
                nvfp4_scale_addr("nvfp4_o_ts"), uint16_t(layer_idx),
                addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
                layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, nrows, eps_expert_, tp,
                cold_nvfp4_o_n128_available_, cold_nvfp4_o_overlap_available_);
        else if (lw.o_w.scalar_type() == at::kChar)
            launch_hyvla_w8_o_ring_postnorm_spm(addr(0, "output"), lw.o_w, lw.o_ws,
                addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
                layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, nrows, eps_expert_, tp);
        else
        launch_hyvla_o_ring_postnorm_spm(addr(0, "output"), lw.o_w,
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
            layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, nrows, eps_expert_, tp);
    } else {
        if (o_resident_enabled(seq_len, cos_sin_start) && layer_idx < o_resident_layers_) {
            TORCH_CHECK(ctx().has_complete_physical_manifest(),
                        "HyVLA resident O requires a COMPLETE descriptor");
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_O_RESIDENT_SITE, 1, 0, o_resident_arguments(cos_sin_start), chunk.idx);
            launch_hyvla_w8_o_resident_spm(addr(0, "output"), lw.o_ws, addr(0, "oproj"),
                addr(0, "gateup_resident_weights") + gateup_bundle_bytes() + layer_idx * 262144,
                seq_len);
        } else {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::LINEAR, HYVLA_EXPERT_O_LINEAR_SITE,
                    static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                    /*resolved_flags=*/0,
                    {seq_len, h, nq * hd, 0, tp}, chunk.idx);
            }
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "output"), lw.o_w, addr(0, "oproj"),
                seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
                /*bias_spm_addr=*/0u, /*scale=*/lw.o_ws,
            nvfp4_scale_addr("nvfp4_o_ts"), uint16_t(layer_idx));
        }
        if (ring_postnorm_enabled(seq_len, cos_sin_start)) {
            TORCH_CHECK(ctx().has_complete_physical_manifest(),
                        "HyVLA ring postnorm requires a COMPLETE descriptor");
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_RING_POSTNORM_SITE, 1, 0, ring_postnorm_arguments(cos_sin_start), chunk.idx);
            if (xor3_norm_enabled())
                launch_hyvla_xor3_norm_spm(addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
                    layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, nrows, eps_expert_, tp, false);
            else
            launch_hyvla_ring_postnorm_spm(
                addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
                layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, nrows, eps_expert_, tp);
        } else {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ALL_REDUCE,
                    HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE,
                    fmb_ring_all_reduce_route_selector(seq_len, h),
                    /*resolved_flags=*/0,
                    {seq_len, h, tp, num_cores()}, chunk.idx);
            }
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
                seq_len, h, tp, num_cores());

            // ── 步 5: post-attention RMSNorm ──
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::NORMALIZATION,
                    HYVLA_EXPERT_POST_NORM_SITE,
                    static_cast<int64_t>(
                        cold_rmsnorm_pad16_
                            ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                            : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
                    /*resolved_flags=*/0,
                    {cold_rmsnorm_pad16_ ? 1 : 0, nrows, h}, chunk.idx);
            }
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual2"), addr(0, "residual1"),
                layer_addr(layer_idx, 0, "post_norm_w"), nrows, h, eps_expert_, full_norm_route, num_cores());
        }
    }

    // ── 步 6: SwiGLU MLP + 残差归约 ──
    // 手写而非用基类 emit_mlp_pipeline: 保持与 VLM 塔同构的可读性, 且这里的
    // 槽名/核数组合与基类默认一致, 没有额外分支。
    const bool resident_gateup = gateup_resident_bundle_.defined();
    if (resident_gateup) {
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_GATEUP_RESIDENT_SITE,
            1, 0, gateup_resident_arguments(), chunk.idx);
        TORCH_CHECK(seq_len == 51 && optimized_prefix_supported(cos_sin_start) && num_steps_ == 10 &&
                        cold_silu_mul_, "HyVLA resident Gate/Up invocation drift");
    }
    const bool fused_resident = layer_idx < gateup_resident_layers() &&
        resident_swiglu_enabled(seq_len, cos_sin_start);
    const bool fused_gateup = layer_idx >= gateup_resident_layers() &&
        gateup_swiglu_enabled(seq_len, cos_sin_start);
    if (fused_gateup) {
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_GATEUP_SWIGLU_SITE,
            1, 0, gateup_swiglu_arguments(cos_sin_start), chunk.idx);
        if (nvfp4_)
            launch_hyvla_gateup_nvfp4(addr(0, "residual1"), addr(0, "gate"), 0,
                0, 0, seq_len, lw.gate_w, lw.up_w, lw.gate_ws, lw.up_ws,
                nvfp4_scale_addr("nvfp4_gate_ts"), nvfp4_scale_addr("nvfp4_up_ts"),
                uint16_t(layer_idx), false, cold_nvfp4_gateup_input_reuse_available_,
                cold_nvfp4_gateup_overlap_available_);
        else if (lw.gate_w.scalar_type() == at::kChar)
            launch_hyvla_gateup_w8(addr(0, "residual1"), addr(0, "gate"), 0,
                0, 0, seq_len, lw.gate_w, lw.up_w, lw.gate_ws, lw.up_ws, false,
                cold_w8_gateup_n48_available_);
        else
            rpu_launch_hyvla_expert_gateup_swiglu_fp16_spm_kernel(
                addr(0, "residual1"), lw.gate_w, lw.up_w, addr(0, "gate"), seq_len);
    } else if (resident_gateup && layer_idx < gateup_resident_layers()) {
        const uint32_t weights = addr(0, "gateup_resident_weights") +
            layer_idx * gateup_layer_bytes();
        if (fused_resident) {
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                nvfp4_ ? HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE : HYVLA_EXPERT_RESIDENT_SWIGLU_SITE,
                1, 0, resident_swiglu_arguments(cos_sin_start), chunk.idx);
        }
        if (layer_idx < resident_fp8_scale_layers_) {
            TORCH_CHECK(resident_fp8_scales_enabled(seq_len, cos_sin_start),
                        "HyVLA resident FP8 scale invocation drift");
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_NVFP4_RESIDENT_SCALES_SITE, 1, 0,
                resident_fp8_scales_arguments(cos_sin_start), chunk.idx);
            const uint32_t scales = addr(0, "gateup_resident_weights") +
                gateup_bundle_bytes() + layer_idx * 32768;
            launch_hyvla_nvfp4_resident_scales_spm(
                addr(0, "residual1"), addr(0, "gate"), addr(0, "up"),
                weights, weights + gateup_weight_bytes(), scales, scales + 16384,
                nvfp4_scale_addr("nvfp4_gate_ts"), nvfp4_scale_addr("nvfp4_up_ts"),
                uint16_t(layer_idx), fused_resident, seq_len, NUM_CORES, resident_fp8_scale_layers_);
        } else if (fused_resident) {
            if (nvfp4_)
                launch_hyvla_nvfp4_resident_swiglu_spm(
                    addr(0, "residual1"), addr(0, "gate"), addr(0, "up"),
                    weights, weights + gateup_weight_bytes(), lw.gate_ws, lw.up_ws,
                    nvfp4_scale_addr("nvfp4_gate_ts"), nvfp4_scale_addr("nvfp4_up_ts"),
                    uint16_t(layer_idx), seq_len, NUM_CORES, cold_nvfp4_resident_overlap_available_);
            else if (lw.gate_w.scalar_type() == at::kChar)
                launch_hyvla_gateup_w8(addr(0, "residual1"), addr(0, "gate"), addr(0, "up"),
                    weights, weights + gateup_weight_bytes(), seq_len,
                    lw.gate_w, lw.up_w, lw.gate_ws, lw.up_ws, true, false, true);
            else
                launch_hyvla_gateup_resident_spm(
                    addr(0, "residual1"), addr(0, "gate"), addr(0, "up"),
                    weights, weights + gateup_weight_bytes(), seq_len, true);
        } else if (nvfp4_) {
            launch_hyvla_gateup_nvfp4(addr(0, "residual1"), addr(0, "gate"), addr(0, "up"),
                weights, weights + gateup_weight_bytes(), seq_len,
                lw.gate_w, lw.up_w, lw.gate_ws, lw.up_ws,
                nvfp4_scale_addr("nvfp4_gate_ts"), nvfp4_scale_addr("nvfp4_up_ts"),
                uint16_t(layer_idx), true);
        } else if (lw.gate_w.scalar_type() == at::kChar) {
            launch_hyvla_gateup_w8(addr(0, "residual1"), addr(0, "gate"), addr(0, "up"),
                weights, weights + gateup_weight_bytes(), seq_len,
                lw.gate_w, lw.up_w, lw.gate_ws, lw.up_ws, true);
        } else {
            rpu_launch_hyvla_gateup_resident_spm_kernel(
                addr(0, "residual1"), addr(0, "gate"), addr(0, "up"),
                weights, weights + gateup_weight_bytes(), seq_len);
        }
    } else {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVLA_EXPERT_GATE_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, intermediate_size(), h, 1, mlp_tp()}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), lw.gate_w, addr(0, "gate"),
            seq_len, intermediate_size(), h, /*partition=*/1, mlp_tp(),
            /*bias_spm_addr=*/0u, /*scale=*/lw.gate_ws,
            nvfp4_scale_addr("nvfp4_gate_ts"), uint16_t(layer_idx));
        if (!cold_silu_mul_) {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ACTIVATION,
                    HYVLA_EXPERT_MLP_SILU_SITE,
                    static_cast<int64_t>(
                        HyVlaExpertActivationRoute::SILU_UNARY),
                    /*resolved_flags=*/0,
                    {cold_silu_mul_ ? 1 : 0, elems_mlp, mlp_tp()}, chunk.idx);
            }
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "gate"), addr(0, "gate"), elems_mlp, ValuOpType::SILU,
                /*is_gelu=*/false, mlp_tp());
        }
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVLA_EXPERT_UP_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, intermediate_size(), h, 1, mlp_tp()}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), lw.up_w, addr(0, "up"),
            seq_len, intermediate_size(), h, /*partition=*/1, mlp_tp(),
            /*bias_spm_addr=*/0u, /*scale=*/lw.up_ws,
            nvfp4_scale_addr("nvfp4_up_ts"), uint16_t(layer_idx));
    }
    if (!fused_gateup && !fused_resident && cold_silu_mul_) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ACTIVATION,
                HYVLA_EXPERT_MLP_SILU_MUL_SITE,
                static_cast<int64_t>(
                    HyVlaExpertActivationRoute::SILU_MUL_FUSED),
                /*resolved_flags=*/0,
                {cold_silu_mul_ ? 1 : 0, elems_mlp, mlp_tp()}, chunk.idx);
        }
        rpu_launch_silu_mul_spm_kernel(
            addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
            elems_mlp, mlp_tp());
    } else if (!fused_gateup && !fused_resident) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ACTIVATION,
                HYVLA_EXPERT_MLP_MUL_SITE,
                static_cast<int64_t>(HyVlaExpertActivationRoute::MUL),
                /*resolved_flags=*/0,
                {cold_silu_mul_ ? 1 : 0, elems_mlp, mlp_tp()}, chunk.idx);
        }
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
            elems_mlp, ValuOpType::MUL, c10::Half(1.0f), mlp_tp());
    }
    const bool fused_nvfp4_down = down_nvfp4_nextnorm_enabled(seq_len, cos_sin_start);
    if (fused_nvfp4_down) {
        // Producer is emitted together with the R16 tail below.
    } else if (down_nvfp4_n128_enabled(seq_len, cos_sin_start)) {
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "HyVLA NVFP4 Down N128 requires a COMPLETE descriptor");
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_DOWN_NVFP4_N128_SITE,
            1, 0, down_nvfp4_n128_arguments(cos_sin_start), chunk.idx);
        launch_hyvla_nvfp4_down_n128_spm(addr(0, "gate"), lw.down_w, lw.down_ws,
            nvfp4_scale_addr("nvfp4_down_ts"), uint16_t(layer_idx),
            addr(0, "down"), seq_len, h, NUM_CORES);
    } else {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVLA_EXPERT_DOWN_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, h, intermediate_size(), 0, mlp_tp()}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "gate"), lw.down_w, addr(0, "down"),
            seq_len, h, intermediate_size(), /*partition=*/0, mlp_tp(),
            /*bias_spm_addr=*/0u, /*scale=*/lw.down_ws,
                nvfp4_scale_addr("nvfp4_down_ts"), uint16_t(layer_idx));
    }
    if (fused_down_nextnorm) {
        TORCH_CHECK(ctx().has_complete_physical_manifest() &&
                        layer_idx >= 0 && layer_idx < 32 &&
                        is_last_layer == (layer_idx == 31) &&
                        addr(0, "input_norm") == addr(0, "residual2"),
                    "HyVLA Down nextnorm requires the exact COMPLETE layer/owner schedule");
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            fused_nvfp4_down ? HYVLA_EXPERT_DOWN_NVFP4_NEXTNORM_SITE : HYVLA_EXPERT_DOWN_NEXTNORM_SITE,
            1, 0, fused_nvfp4_down ? down_nvfp4_nextnorm_arguments(cos_sin_start) : down_nextnorm_arguments(cos_sin_start), chunk.idx);
        // All per-layer norm weights are persistent and preloaded before the body.
        const uint32_t next_gamma = is_last_layer ? addr(0, "final_norm_w")
            : layer_addr(layer_idx + 1, 0, "norm_w");
        if (fused_nvfp4_down)
            launch_hyvla_nvfp4_down_nextnorm_spm(addr(0, "gate"), lw.down_w, lw.down_ws,
                nvfp4_scale_addr("nvfp4_down_ts"), uint16_t(layer_idx),
                addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
                next_gamma, seq_len, h, nrows, eps_expert_, NUM_CORES, is_last_layer,
                cold_nvfp4_down_nextnorm_n128_available_, cold_nvfp4_down_overlap_available_);
        else if (xor3_norm_enabled())
            launch_hyvla_xor3_norm_spm(addr(0, "down"), addr(0, "residual2"),
                addr(0, "residual1"), next_gamma, seq_len, h, nrows, eps_expert_,
                NUM_CORES, is_last_layer);
        else
        launch_hyvla_down_nextnorm_spm(addr(0, "down"), addr(0, "residual2"),
            addr(0, "residual1"), next_gamma, seq_len, h, nrows, eps_expert_,
            NUM_CORES, is_last_layer);
    } else {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                HYVLA_EXPERT_MLP_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(seq_len, h),
                /*resolved_flags=*/0,
                {seq_len, h, mlp_tp(), num_cores()}, chunk.idx);
        }
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
            seq_len, h, mlp_tp(), num_cores());

        // ── 步 7 (末层): final RMSNorm (原地, 单份 γ) ──
        if (is_last_layer) {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::NORMALIZATION,
                    HYVLA_EXPERT_FINAL_NORM_SITE,
                    static_cast<int64_t>(
                        cold_rmsnorm_pad16_
                            ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                            : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
                    /*resolved_flags=*/0,
                    {cold_rmsnorm_pad16_ ? 1 : 0, nrows, h}, chunk.idx);
            }
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual1"), addr(0, "residual1"),
                addr(0, "final_norm_w"), nrows, h, eps_expert_, full_norm_route, num_cores());
        }

    }

    if (!ctx().output_to_spm) {
        emit_layer_output_dma(layer_idx, chunk);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// step_forward — 一次 denoise step
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor HyVlaExpertModel::step_forward(
    const at::Tensor& x_emb, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor)
{
    TORCH_CHECK(!layers_.empty() && expert_chunk_size_ > 0,
                "hyvla_expert step_forward before set_weights_expert");
    const int64_t cs = expert_chunk_size_;
    const int64_t h  = hidden_size();
    TORCH_CHECK(x_emb.dim() == 3 && x_emb.size(0) == 1 && x_emb.size(1) == cs
                && x_emb.size(2) == h && x_emb.scalar_type() == at::kHalf
                && x_emb.is_contiguous()
                && x_emb.device().type() == at::kPrivateUse1,
                "hyvla_expert step_forward: x_emb must be [1,", cs, ",", h,
                "] fp16 contig device");
    TORCH_CHECK(attn_mask_4d.defined(),
                "hyvla_expert step_forward: explicit attention mask required");
    TORCH_CHECK(prefix_len >= 0,
                "hyvla_expert step_forward: prefix_len must be >= 0");

    // per-forward cos/sin: 覆盖影子成员, 赋值即保活过 graph capture;
    // build_layer_subgraph 读它们的 raw c10::Half*。
    TORCH_CHECK(cos.defined() && sin.defined()
                && cos.device().type() == at::kPrivateUse1
                && sin.device().type() == at::kPrivateUse1
                && cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf
                && cos.is_contiguous() && sin.is_contiguous(),
                "hyvla_expert step_forward: cos/sin must be fp16 contig device");
    TORCH_CHECK(cos.dim() == 2 && sin.sizes() == cos.sizes()
                && cos.size(1) == head_dim() / 2,
                "hyvla_expert step_forward: cos/sin must be [rows, head_dim/2=",
                head_dim() / 2, "], got ", cos.sizes());
    TORCH_CHECK(cos.size(0) >= prefix_len + cs,
                "hyvla_expert step_forward: cos rows (", cos.size(0),
                ") < prefix_len+cs (", prefix_len + cs, ")");
    cos_ref_ = cos;
    sin_ref_ = sin;

    // KV headroom: 7-D K 布局的 max_seq = size(1)*size(5)。
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + cs <= kc.size(1) * kc.size(5),
                    "hyvla_expert step_forward: prefix_len(", prefix_len, ")+cs(",
                    cs, ") exceeds k_cache capacity ", kc.size(1) * kc.size(5));
    }

    rpu_ddr_flush_force(x_emb.data_ptr<c10::Half>());

    return CausalDecoderModel::forward(
        x_emb, k_caches, v_caches,
        std::optional<at::Tensor>(attn_mask_4d),
        /*position=*/prefix_len, /*is_causal=*/false,
        /*position_ids=*/std::nullopt, /*deepstack=*/std::nullopt,
        /*rope_cos_il=*/std::nullopt, /*rope_sin_il=*/std::nullopt,
        /*cos_sin_offset=*/-1, /*batch_slot=*/0,
        /*allow_batch_decode=*/false, /*planned_chunk_size=*/0,
        planned_stage_descriptor);
}

// ─────────────────────────────────────────────────────────────────────────────
// unroll_forward — 一次 forward 跑完 num_steps 步 Euler
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::unroll_forward(
    const at::Tensor& x0_rpu, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches, const at::Tensor& state_emb,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& attn_mask_4d,
    at::Tensor& x_traj, double dt, int64_t prefix_len, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor, bool inputs_published,
    const std::optional<at::Tensor>& noise_cpu)
{
    TORCH_CHECK(unroll_mode_, "hyvla_expert unroll_forward 之前必须 set_action_weights");
    TORCH_CHECK(num_steps == num_steps_, "num_steps(", num_steps,
                ") 必须等于 set_action_weights 的 num_steps(", num_steps_, ")");
    const int64_t cs = expert_chunk_size_, np = action_dim_pad_, h = hidden_size();

    // 节点数护栏: 控制单段 unroll 远低于 65536 单 batch 节点上限。

    TORCH_CHECK(num_steps <= (65536 - 1) / 1500,
                "denoise unroll (", num_steps, " 个 body) 会逼近 65536 的 batch 上限");

    TORCH_CHECK(x0_rpu.dim() == 3 && x0_rpu.size(0) == 1 && x0_rpu.size(1) == cs
             && x0_rpu.size(2) == np && x0_rpu.scalar_type() == at::kHalf
             && x0_rpu.is_contiguous() && x0_rpu.device().type() == at::kPrivateUse1,
                "x0_rpu 必须是 [1,", cs, ",", np, "] fp16 contig device");
    TORCH_CHECK(x_traj.dim() == 3 && x_traj.size(0) == num_steps && x_traj.size(1) == cs
             && x_traj.size(2) == np && x_traj.scalar_type() == at::kHalf
             && x_traj.is_contiguous() && x_traj.device().type() == at::kPrivateUse1,
                "x_traj 必须是 [", num_steps, ",", cs, ",", np, "] fp16 contig device");
    TORCH_CHECK(state_emb.dim() == 1 && state_emb.size(0) == h
             && state_emb.scalar_type() == at::kHalf
             && (state_emb.device().is_cpu()
                 || state_emb.device().type() == at::kPrivateUse1),
                "state_emb 必须是 [hidden] fp16 CPU 或 device");
    if (noise_cpu.has_value()) {
        TORCH_CHECK(noise_cpu->defined() && noise_cpu->device().is_cpu()
                 && noise_cpu->scalar_type() == at::kHalf
                 && noise_cpu->is_contiguous() && noise_cpu->dim() == 3
                 && noise_cpu->size(0) == 1 && noise_cpu->size(1) == cs - 1
                 && noise_cpu->size(2) == np,
                    "noise_cpu must be contiguous CPU FP16 [1,", cs - 1, ",", np, "]");
        TORCH_CHECK(state_emb.device().is_cpu() && state_emb.is_contiguous(),
                    "noise_cpu requires contiguous CPU FP16 state_emb");
        const auto overlaps = [](const at::Tensor& source, const at::Tensor& target) {
            const auto a = reinterpret_cast<uintptr_t>(source.data_ptr());
            const auto b = reinterpret_cast<uintptr_t>(target.data_ptr());
            return a < b + target.nbytes() && b < a + source.nbytes();
        };
        TORCH_CHECK(!overlaps(*noise_cpu, x0_rpu) && !overlaps(*noise_cpu, emb_stage_)
                 && !overlaps(state_emb, x0_rpu) && !overlaps(state_emb, emb_stage_),
                    "CPU action inputs must not alias their destination buffers");
    }
    TORCH_CHECK(attn_mask_4d.defined(), "hyvla_expert unroll_forward: 必须显式给 mask");

    // dt 是烘进图的 Euler 系数 (x += dt·v), N 步相同。Hy-VLA 的 t: 1→0 ⇒ dt<0。
    TORCH_CHECK(dt != 0.0 && std::isfinite(dt), "dt 必须是非零有限值");
    const c10::Half dt_half = c10::Half(static_cast<float>(dt));
    if (!dt_pinned_) { dt_ = dt_half; dt_pinned_ = true; }
    else TORCH_CHECK(dt_ == dt_half, "dt 在 replay 之间变了 (它已烘进图)");

    // cos/sin 影子 (build 期取 raw 指针); 与 step_forward 同一套校验。
    TORCH_CHECK(cos.defined() && sin.defined()
             && cos.device().type() == at::kPrivateUse1
             && sin.device().type() == at::kPrivateUse1
             && cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf
             && cos.is_contiguous() && sin.is_contiguous()
             && cos.dim() == 2 && sin.sizes() == cos.sizes()
             && cos.size(1) == head_dim() / 2 && cos.size(0) >= prefix_len + cs,
                "hyvla_expert unroll_forward: cos/sin 必须是 [>=prefix_len+cs, head_dim/2] "
                "fp16 contig device");
    cos_ref_ = cos;
    sin_ref_ = sin;

    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + cs <= kc.size(1) * kc.size(5),
                    "prefix_len(", prefix_len, ")+cs(", cs, ") 超出 k_cache 容量 ",
                    kc.size(1) * kc.size(5));
    }

    // emb_stage_ 行 0 = state token, N 步不变 (pre-hook 只碰行 1:cs)。每次调用写一次
    // Only the state row is host-written; remaining rows come from the device
    // pre-hook. CPU noise/state share one SDK call over their two live ranges.
    state_ref_ = state_emb;
    if (noise_cpu.has_value()) {
        void* noise_dst = x0_rpu.data_ptr<c10::Half>() + np;
        void* state_dst = emb_stage_.data_ptr<c10::Half>();
        std::memcpy(noise_dst, noise_cpu->data_ptr(), noise_cpu->nbytes());
        std::memcpy(state_dst, state_emb.data_ptr(), state_emb.nbytes());
        if (g_rpu_ddr_flush_force_enabled) {
            // Public callers may also have dirtied x0's state row through a
            // CPU alias. Include it without adding another SDK call.
            void* destinations[2] = {
                inputs_published ? noise_dst : x0_rpu.data_ptr(), state_dst};
            size_t sizes[2] = {
                inputs_published ? noise_cpu->nbytes() : x0_rpu.nbytes(),
                state_emb.nbytes()};
            ::rhino_lkn::RpuDdrFlushBatch(destinations, sizes, 2);
#if defined(__aarch64__)
            // Retain the former scalar large-range publication barrier.
            __asm__ volatile("dsb sy" ::: "memory");
#endif
        }
    } else {
        emb_stage_[0][0].copy_(state_emb);
    }

    // Adapter-owned x0 is published by copy_; ordinary callers may have
    // mutated a CPU alias and retain the conservative boundary below.
    x0_ref_ = x0_rpu; x_traj_ref_ = x_traj;
    x_t_src_base_    = ::rhino_lkn::RpuGetDevAddr(x0_rpu.data_ptr());
    x_traj_dst_base_ = ::rhino_lkn::RpuGetDevAddr(x_traj.data_ptr());
    if (!inputs_published && !noise_cpu.has_value()) {
        rpu_ddr_flush_force(x0_rpu.data_ptr<c10::Half>());
    }

    // 一次 forward —— run_all_layers 内部循环 body_iterations(=num_steps_) 次,
    // pre/post hook 每次以递增的 ctx().body_iter 重新 emit。
    (void) CausalDecoderModel::forward(
        emb_stage_, k_caches, v_caches,
        std::optional<at::Tensor>(attn_mask_4d),
        /*position=*/prefix_len, /*is_causal=*/false,
        /*position_ids=*/std::nullopt, /*deepstack=*/std::nullopt,
        /*rope_cos_il=*/std::nullopt, /*rope_sin_il=*/std::nullopt,
        /*cos_sin_offset=*/-1, /*batch_slot=*/0,
        /*allow_batch_decode=*/false, /*planned_chunk_size=*/0,
        planned_stage_descriptor);
}

}  // namespace v3

// =============================================================================
// Instance registry + C API (decls in rpu_kernel_decls.h)
// =============================================================================
using HyVlaExpertRegistry = ModelHandleRegistry<v3::HyVlaExpertModel>;

std::vector<int64_t> rpu_hyvla_expert_planner_cache_identity(int64_t handle) {
    return HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_hyvla_expert_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_hyvla_expert_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_hyvla_expert_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_kvinsert_cost_domain")
        ->kvinsert_cost_domain("hyvla_expert", descriptor);
}

std::string rpu_hyvla_expert_kvinsert_cost_catalog_sha256(int64_t handle) {
    return HyVlaExpertRegistry::get(
        handle, "rpu_hyvla_expert_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

int64_t rpu_hyvla_expert_create() {
    return HyVlaExpertRegistry::create();
}

void rpu_hyvla_expert_set_execution_core_count(int64_t handle, int64_t num_cores) {
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_execution_core_count")
        ->set_execution_cores(num_cores);
}

std::vector<int64_t> rpu_hyvla_expert_get_execution_topology(int64_t handle) {
    return HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_get_execution_topology")
        ->execution_topology();
}

void rpu_hyvla_expert_set_runtime_config(
    int64_t handle,
    bool fast_replay, bool fast_replay_preload, bool mask_once,
    bool silu_mul, bool kvpad16, bool partial_rope,
    bool rmsnorm_pad16) {
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_runtime_config")
        ->set_runtime_config(
            fast_replay, fast_replay_preload, mask_once, silu_mul,
            kvpad16, partial_rope, rmsnorm_pad16);
}

void rpu_hyvla_expert_destroy(int64_t handle) {
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_destroy")
        ->check_execution_reconfigure_destroy_allowed(
            "rpu_hyvla_expert_destroy");
    HyVlaExpertRegistry::destroy(handle, "rpu_hyvla_expert_destroy");
}

void rpu_hyvla_expert_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps,
    int64_t chunk_size, int64_t action_mlp_cores)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_weights")
        ->set_weights_expert(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, chunk_size, action_mlp_cores);
}

void rpu_hyvla_expert_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps,
    int64_t chunk_size, int64_t action_mlp_cores,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_weights_w8a16")
        ->set_weights_expert(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, chunk_size, action_mlp_cores,
            q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws);
}

void rpu_hyvla_expert_set_weights_nvfp4(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps,
    int64_t chunk_size, int64_t action_mlp_cores,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws,
    const at::Tensor& q_ts, const at::Tensor& k_ts, const at::Tensor& v_ts,
    const at::Tensor& o_ts, const at::Tensor& gate_ts, const at::Tensor& up_ts,
    const at::Tensor& down_ts)
{
    for (const auto* table : {&q_ts, &k_ts, &v_ts, &o_ts, &gate_ts, &up_ts, &down_ts})
        TORCH_CHECK(table->defined() && table->scalar_type() == at::kFloat &&
                    table->dim() == 1 && table->size(0) == 32 && table->is_contiguous() &&
                    table->device().type() == at::kPrivateUse1,
                    "HyVLA NVFP4 requires seven owned contiguous FP32[32] device tensor-scale tables");
    TORCH_CHECK(q_w.size() == 32 && q_ws.size() == 32 && q_w[0].scalar_type() == at::kByte &&
                q_ws[0].scalar_type() == at::kByte &&
                ((action_mlp_cores == v3::NUM_CORES && num_kv_heads == 8) ||
                 (action_mlp_cores == 4 && num_kv_heads == 4)) &&
                num_q_heads == 16 && head_dim == 128 &&
                hidden_size == 1024 && intermediate_size == 2048 && chunk_size == 51 && eps == 1e-5,
                "HyVLA NVFP4 requires the exact TP8 (or reduced four-core KV4) 32-layer "
                "packed-Byte/FP8-scale Action profile");
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_weights_nvfp4")
        ->set_weights_expert(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, chunk_size, action_mlp_cores,
            q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws,
            q_ts, k_ts, v_ts, o_ts, gate_ts, up_ts, down_ts);
}

at::Tensor rpu_hyvla_expert_step_forward(
    int64_t handle, const at::Tensor& x_emb,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor)
{
    return HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_step_forward")
        ->step_forward(x_emb, k_caches, v_caches, cos, sin, attn_mask_4d,
                       prefix_len, planned_stage_descriptor);
}

void rpu_hyvla_expert_set_gateup_resident_weights(int64_t handle, const at::Tensor& bundle, int64_t o_layers) {
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_gateup_resident_weights")
        ->set_gateup_resident_weights(bundle, o_layers);
}

void rpu_hyvla_expert_set_action_weights(
    int64_t handle,
    const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
    const at::Tensor& op, const at::Tensor& op_bias, const at::Tensor& time_all,
    int64_t action_dim, int64_t num_steps)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_action_weights")
        ->set_action_weights(wc, mo, mo_bias, op, op_bias, time_all,
                             action_dim, num_steps);
}

void rpu_hyvla_expert_unroll_forward(
    int64_t handle, const at::Tensor& x0_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& state_emb, const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, at::Tensor x_traj,
    double dt, int64_t prefix_len, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor, bool inputs_published,
    const std::optional<at::Tensor>& noise_cpu)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_unroll_forward")
        ->unroll_forward(x0_rpu, k_caches, v_caches, state_emb, cos, sin,
                         attn_mask_4d, x_traj, dt, prefix_len, num_steps,
                         planned_stage_descriptor, inputs_published, noise_cpu);
}

void rpu_hyvla_expert_prepare_persistent_spm(
    int64_t handle, int64_t seq_len, int64_t prefix_len,
    int64_t mask_kv_len) {
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_prepare_persistent_spm")
        ->prepare_persistent_spm_for_shape(seq_len, prefix_len, mask_kv_len);
}

std::vector<int64_t> rpu_hyvla_expert_resolve_stage_domain(
    int64_t handle, int64_t seq_len, int64_t prefix_len,
    int64_t mask_kv_len) {
    return HyVlaExpertRegistry::get(
               handle, "rpu_hyvla_expert_resolve_stage_domain")
        ->resolve_stage_domain(seq_len, prefix_len, mask_kv_len);
}

int64_t rpu_hyvla_expert_get_resolved_chunk_size(int64_t handle) {
    return HyVlaExpertRegistry::get(
               handle, "rpu_hyvla_expert_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

void rpu_hyvla_expert_set_chunk_size_override(
    int64_t handle, int64_t chunk_size) {
    HyVlaExpertRegistry::get(
        handle, "rpu_hyvla_expert_set_chunk_size_override")
        ->set_control_chunk_size_override(
            chunk_size, "rpu_hyvla_expert_set_chunk_size_override");
}

void rpu_hyvla_expert_enable_execution_reconfigure(int64_t handle) {
    HyVlaExpertRegistry::get(
        handle, "rpu_hyvla_expert_enable_execution_reconfigure")
        ->enable_execution_reconfigure_guard();
}

void rpu_hyvla_expert_stage_chunk_size_override(
    int64_t handle, int64_t token, int64_t chunk_size) {
    TORCH_CHECK(token > 0,
                "Hy-VLA action expert hot-reconfigure token must be positive");
    HyVlaExpertRegistry::get(
        handle, "rpu_hyvla_expert_stage_chunk_size_override")
        ->stage_control_chunk_size_override(
            static_cast<uint64_t>(token), chunk_size,
            "rpu_hyvla_expert_stage_chunk_size_override");
}

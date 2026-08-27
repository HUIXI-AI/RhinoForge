// rpu_hyvla_merger.cpp — Hy-VLA merger 的组轴（DwPooler）算子，两个独立 op。
//
// 背景：merger 的 proj1 已经 fuse 在 ViT 的 fused 图尾部，余下的 DwPooler
// （4 成员分组 → pooled → predictor → 组内 softmax → 加权求和）若整段在 host 上跑，
// `mean.dim` / `softmax.int` 对**非最后一维**会静默
// 回落 CPU（见 `rpu_reduce.cpp` / `rpu_backend.cpp`），放在 RPU 上反而更慢。
//
// 这里把组轴那两段搬回 device。能搬的**前提是 member-major 布局**
// `[4, G, D]`（G = 相机数 × 49 组）：这样"同一成员的全部组"在内存里是一整块，
// 组轴的三件事（pooled / softmax over 4 / 加权求和）全都退化成**扁平算子**，
// 可以直接按元素区间在 8 核间等分 —— 既不需要跨步 kernel，也不需要把数据复制到每核。
// member-major 由 ViT 尾部的两次 permute3d 产出（见 rpu_hyvit2_vision_model.cpp）。
//
// 分片：每核 N/8 个元素，N = G*D。**允许切在行中间** —— 这里全是逐元素运算，
// 行边界没有语义。`ddr_scatter_spm_dma` 正好是"按固定 core_stride 等分一段连续
// DDR"，与这个分片一一对应。
//
// ⚠️ 这**两个** op 都在 `GraphCache.capture` **之外**调用（merger 在 ViT 的 capture
// 结束之后、VLM 的 capture 之前），所以走 immediate 变体；峰值约 830 KB/核。
//
// 本文件末尾的 `merger_fused` 把这两个 op 连同 host 侧的 4 次
// `F.linear` 一起收进**一个图内 op**（`RPU_HY_VLA_MERGER_IN_GRAPH`，默认 ON）。
// 开关 ON 时不调用这两个 immediate op；它们保留为回退路径。

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rpu_eltwise.h"
#include <c10/util/Half.h>
#include <vector>

namespace {

constexpr int64_t kMembers = 4;   // 2×2 的组内成员数
constexpr int     kCores   = 8;

inline uint32_t align256(uint32_t v) { return (v + 255u) & ~255u; }

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_PROJ1_IN_MERGER —— **C++ 侧默认 ON**（`0`/`off`/`false` 回旧路径）。
//
// 把 merger 的 `proj1 [2048,1152]` 从 ViT 尾部搬进本 op。DDR 路径在
// partition=1 下要求每个核重读完整的 [588,1152] 输入，并产生额外的
// SPM→DDR→SPM 往返。
//
// 搬进来之后：ViT 尾部直接把 [S,1152] 落到 tracked output；本 op 广播的是
// [S,1152] 而不是 [S,2048]，proj1 变成和另外 4 个投影同型的
// col-partition acc32 GEMM + 一次 all_gather。
//
// ⭐ **不新增任何 SPM**：proj1 的输入借 `a_sc`（它第一次被写是在步 ③ 的 gather，
// 此前是空的；需要 M4*K = 677376 元素 ≤ N4 = 1204224），输出借 `a_cs`（尺寸恰好
// 就是 col-shard M4*ln），只多一个 ln 大小的 bias 槽（512 B）。
//
// ⚠️ **Python 与 C++ 必须同开同关**（`runtime.py::_proj1_in_merger`）——
// 错配 = ViT 的输出维与本 op 期望的输入维对不上，会在 check 里报错而不是静默算错，
// 但 ViT 侧若单独关掉则会把已经 proj1 过的 [S,2048] 再 proj1 一次 ⇒ **静默算错**。
// ─────────────────────────────────────────────────────────────────────────────
bool proj1_in_merger_on() {
    static const bool on = [] {
        const char* e = std::getenv("RPU_HY_VLA_PROJ1_IN_MERGER");
        if (!e || !*e) return true;
        const std::string v(e);
        return !(v == "0" || v == "off" || v == "false" || v == "False");
    }();
    return on;
}

// 校验 + 取形状。nxc/sc 都是 [4, G, D] fp16 RPU contiguous（member-major）。
struct MergerDims { int64_t G, D, N, per_core; };

MergerDims check_member_major(const at::Tensor& t, const char* who,
                              int64_t D_override = 0) {
    TORCH_CHECK(t.dim() == 3 && t.size(0) == kMembers,
                who, ": expected member-major [4, G, D], got ", t.sizes());
    TORCH_CHECK(t.scalar_type() == at::kHalf, who, ": must be FP16");
    TORCH_CHECK(t.is_contiguous(), who, ": must be contiguous");
    TORCH_CHECK(t.device().type() == at::kPrivateUse1, who, ": must be on the RPU device");
    // D_override != 0 ⇒ 输入是 proj1 **之前**的 [4,G,K]，链路其余部分按 D 走。
    const int64_t G = t.size(1), D = D_override ? D_override : t.size(2), N = G * D;
    // 分片要求能被核数整除。G*D = 147*2048 = 301056 ⇒ 37632/核。
    TORCH_CHECK(N % kCores == 0,
                who, ": G*D (", N, ") must be divisible by ", kCores);
    // core_stride_bytes 要 16B 对齐（add_dma 的 bytes%16 检查）。
    TORCH_CHECK(((N / kCores) * (int64_t)sizeof(c10::Half)) % 16 == 0,
                who, ": per-core byte stride must be 16B-aligned");
    return {G, D, N, N / kCores};
}

// 把 [4,G,D] 的第 m 个成员切片按核等分装进 SPM，返回各成员在 SPM 的基址。
std::vector<uint32_t> stage_members(const at::Tensor& t, const MergerDims& d,
                                    uint32_t base_off) {
    c10::Half* p = const_cast<c10::Half*>(t.data_ptr<c10::Half>());
    rpu_ddr_flush(p);
    const uint32_t stride = align256((uint32_t)(d.per_core * sizeof(c10::Half)));
    std::vector<uint32_t> addrs(kMembers);
    for (int64_t m = 0; m < kMembers; ++m) {
        const uint32_t off = base_off + (uint32_t)m * stride;
        rpu_launch_ddr_scatter_spm_dma_immediate(
            p + m * d.N, d.per_core,
            d.per_core * (int64_t)sizeof(c10::Half),
            SPM_ALLOC.addr(0, off), kCores);
        addrs[m] = SPM_ALLOC.addr(0, off);
    }
    return addrs;
}

}  // namespace

// =============================================================================
// merger_pool — pooled = Σ_m nxc[m]，**不除 4**，并平铺成 [4,G,D]
//
// 不除 4：pooled 的唯一消费者是 host 侧 `F.linear(pooled, Wb)`（predictor.0 按 K
// 拆分的后一半），所以 1/4 折进 Wb 即可。fp16 里乘 0.25 是精确的 2 的幂缩放
// （Wb 的量级 ~1e-2，缩放后 ~2.5e-3，远离非规格化），⇒ 少一次 kernel 且不损精度。
//
// 平铺成 [4,G,D]：这样 host 的两个 K 拆分 GEMM 输出形状相同，直接相加即可，
// **避开首轴隐式广播**（该路径在 RPU 上会静默产 NaN）。
// 额外的三份写入只改变存储量，不改变数值。
// =============================================================================
at::Tensor rpu_hyvla_merger_pool(const at::Tensor& nxc) {
    const auto d = check_member_major(nxc, "hyvla_merger_pool");

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    const uint32_t stride = align256((uint32_t)(d.per_core * sizeof(c10::Half)));
    const uint32_t off_m   = SPM_ALLOC.alloc_temporary((size_t)stride * kMembers);
    const uint32_t off_acc = SPM_ALLOC.alloc_temporary((size_t)stride);
    const uint32_t acc = SPM_ALLOC.addr(0, off_acc);

    const auto a = stage_members(nxc, d, off_m);

    // acc = ((m0 + m1) + m2) + m3
    rpu_launch_eltwise_binary_spm_kernel(a[0], a[1], acc, d.per_core,
                                         ValuOpType::ADD, c10::Half(1.0), kCores);
    rpu_launch_eltwise_binary_spm_kernel(acc, a[2], acc, d.per_core,
                                         ValuOpType::ADD, c10::Half(1.0), kCores);
    rpu_launch_eltwise_binary_spm_kernel(acc, a[3], acc, d.per_core,
                                         ValuOpType::ADD, c10::Half(1.0), kCores);

    at::Tensor out = at::empty({kMembers, d.G, d.D},
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
    c10::Half* op = out.data_ptr<c10::Half>();
    for (int64_t m = 0; m < kMembers; ++m) {
        rpu_launch_spm_scatter_ddr_dma_immediate(
            acc, op + m * d.N, d.per_core,
            d.per_core * (int64_t)sizeof(c10::Half), kCores);
    }
    rpu_ddr_flush(op);
    return out;
}

// =============================================================================
// merger_combine — 组内 softmax（成员轴）+ 加权求和
//
//   e_m = exp(sc[m] - max_m sc[m])
//   out = (Σ_m nxc[m] · e_m) / (Σ_m e_m)
//
// 写成"先加权求和再除一次"而不是"先 softmax 再加权求和"：数学等价，但把 4 次
// DIV 省成 1 次。max 减法是标准的 softmax 数值稳定化，与 host 的 `F.softmax` 同法。
// =============================================================================
at::Tensor rpu_hyvla_merger_combine(const at::Tensor& nxc, const at::Tensor& sc) {
    const auto d = check_member_major(nxc, "hyvla_merger_combine(nxc)");
    const auto ds = check_member_major(sc, "hyvla_merger_combine(sc)");
    TORCH_CHECK(d.G == ds.G && d.D == ds.D,
                "hyvla_merger_combine: nxc ", nxc.sizes(), " vs sc ", sc.sizes());

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    const uint32_t stride = align256((uint32_t)(d.per_core * sizeof(c10::Half)));
    const uint32_t off_x   = SPM_ALLOC.alloc_temporary((size_t)stride * kMembers);
    const uint32_t off_e   = SPM_ALLOC.alloc_temporary((size_t)stride * kMembers);
    const uint32_t off_mx  = SPM_ALLOC.alloc_temporary((size_t)stride);
    const uint32_t off_sum = SPM_ALLOC.alloc_temporary((size_t)stride);
    const uint32_t off_out = SPM_ALLOC.alloc_temporary((size_t)stride);
    const uint32_t mx  = SPM_ALLOC.addr(0, off_mx);
    const uint32_t sum = SPM_ALLOC.addr(0, off_sum);
    const uint32_t o   = SPM_ALLOC.addr(0, off_out);

    const auto x = stage_members(nxc, d, off_x);
    const auto e = stage_members(sc,  d, off_e);

    // mx = max over the 4 members
    rpu_launch_eltwise_binary_spm_kernel(e[0], e[1], mx, d.per_core,
                                         ValuOpType::MAX, c10::Half(1.0), kCores);
    rpu_launch_eltwise_binary_spm_kernel(mx, e[2], mx, d.per_core,
                                         ValuOpType::MAX, c10::Half(1.0), kCores);
    rpu_launch_eltwise_binary_spm_kernel(mx, e[3], mx, d.per_core,
                                         ValuOpType::MAX, c10::Half(1.0), kCores);

    // e_m = exp(sc_m - mx)   （原地覆盖 e[]）
    for (int64_t m = 0; m < kMembers; ++m) {
        rpu_launch_eltwise_binary_spm_kernel(e[m], mx, e[m], d.per_core,
                                             ValuOpType::SUB, c10::Half(1.0), kCores);
        rpu_launch_eltwise_unary_spm_kernel(e[m], e[m], d.per_core,
                                            ValuOpType::EXP, /*is_gelu=*/false, kCores);
    }

    // sum = Σ e_m ；out = Σ nxc_m·e_m（mx 用完，借它当乘积暂存）
    rpu_launch_eltwise_binary_spm_kernel(e[0], e[1], sum, d.per_core,
                                         ValuOpType::ADD, c10::Half(1.0), kCores);
    rpu_launch_eltwise_binary_spm_kernel(sum, e[2], sum, d.per_core,
                                         ValuOpType::ADD, c10::Half(1.0), kCores);
    rpu_launch_eltwise_binary_spm_kernel(sum, e[3], sum, d.per_core,
                                         ValuOpType::ADD, c10::Half(1.0), kCores);

    rpu_launch_eltwise_binary_spm_kernel(x[0], e[0], o, d.per_core,
                                         ValuOpType::MUL, c10::Half(1.0), kCores);
    for (int64_t m = 1; m < kMembers; ++m) {
        rpu_launch_eltwise_binary_spm_kernel(x[m], e[m], mx, d.per_core,
                                             ValuOpType::MUL, c10::Half(1.0), kCores);
        rpu_launch_eltwise_binary_spm_kernel(o, mx, o, d.per_core,
                                             ValuOpType::ADD, c10::Half(1.0), kCores);
    }
    rpu_launch_eltwise_binary_spm_kernel(o, sum, o, d.per_core,
                                         ValuOpType::DIV, c10::Half(1.0), kCores);

    at::Tensor out = at::empty({d.G, d.D},
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
    c10::Half* op = out.data_ptr<c10::Half>();
    rpu_launch_spm_scatter_ddr_dma_immediate(
        o, op, d.per_core, d.per_core * (int64_t)sizeof(c10::Half), kCores);
    rpu_ddr_flush(op);
    return out;
}

// =============================================================================
// merger_fused — 整条 merger 余部（pool → predictor → 组内 softmax 加权 → proj2）
// 收进**一个图内 op**。取代 host 侧的 `4×F.linear + 2×F.gelu + pool + combine`。
//
// 图内执行把多次同步派发合并为一次批量提交，并让线性层使用 SPM 路径，
// 避免中间结果在 DDR 间往返。
//
// ⚠️ **必须在 `graph_cache.capture` 之内调用**（用的全是 graph-only wrapper）。
// 调用点在 ViT 的 capture **之后**、且要先 `spm_alloc_reset_temporary()` ——
// 本 op 峰值约 7.07 MB/核，调用前必须回收 ViT 的 temporary 才能满足 SPM 预算。
//
// ── 布局：为什么是这样切的（这段是本 op 唯一的设计难点）──────────────────
// 三种分片在这条链上轮流出现，而**只有两种转换是有原语的**：
//   full（每核一份完整副本） --col-partition GEMM--> col-shard（每核 N/8 列）
//   col-shard --all_gather--> full
// 反过来（full → shard）**没有原语**：所有 kernel 收的是一个 SPM 地址、8 核共用，
// 无法让核 c 读自己那 1/8。⇒ 设计约束是"**永远不要产生一个之后还要再切片的 full**"。
//
// 由此定下：4 个 GEMM 全走 col-partition（输入 full、输出 col-shard），每个之后
// all_gather 回 full；而 pool / softmax / 加权求和这些**逐元素**运算直接在 full 上
// 做 —— 8 个核各算一遍同样的结果。冗余 8×，但它们是 SPM 内的 eltwise，
// 相比省下的 DDR 往返可以忽略；换取的是整条链一次 DDR 中转都没有。
//
// 数值上与 host 版逐项对应，只有一处**代数重排**：host 版把 `pooled` 平铺成
// [4,G,D] 再做 M=588 的 GEMM，这里改成 M=147 算一次、再广播加到 4 个成员上
// （`pooled` 对 4 个成员本来就相同），因此只计算一个代表值再广播。
//
// ⚠️⚠️ **GEMM 必须用 `linear_spm_to_spm`（acc32），不能用 `_acc16`。**
// 这不是性能选择，而是数值契约：`F.linear` 对应的 DDR 路径使用 fp32 累加；
// acc16 会改变 K=2048 时的累加结果。这里必须保持 acc32。
// =============================================================================
at::Tensor rpu_hyvla_merger_fused(
    const at::Tensor& nxc,
    const at::Tensor& wa,  const at::Tensor& b0,
    const at::Tensor& wb,
    const at::Tensor& w2,  const at::Tensor& b2,
    const at::Tensor& wp2, const at::Tensor& bp2,
    const c10::optional<at::Tensor>& wp1,
    const c10::optional<at::Tensor>& bp1) {
    // proj1 进图 ⇒ nxc 是 proj1 **之前**的 [4,G,K]，输出维 D 由 wp1 决定。
    const bool proj1 = wp1.has_value() && wp1->defined();
    TORCH_CHECK(proj1 == (bp1.has_value() && bp1->defined()),
                "hyvla_merger_fused: wp1/bp1 必须同时给或同时不给");
    const int64_t K = nxc.size(2);
    const auto d = check_member_major(nxc, "hyvla_merger_fused",
                                      proj1 ? wp1->size(0) : 0);
    const int64_t G = d.G, D = d.D;
    if (proj1) {
        TORCH_CHECK(wp1->dim() == 2 && wp1->size(1) == K,
                    "hyvla_merger_fused: wp1 应为 [D, K]=[", D, ",", K,
                    "]，实得 ", wp1->sizes());
        TORCH_CHECK(K <= D, "hyvla_merger_fused: proj1 借 a_sc 暂存输入，"
                    "要求 K(", K, ") <= D(", D, ")");
    }
    TORCH_CHECK(D % (16 * kCores) == 0,
                "hyvla_merger_fused: D (", D, ") must be divisible by ", 16 * kCores);
    const int64_t ln  = D / kCores;      // col-partition 的每核宽度
    const int64_t NG  = G * D;           // 一个成员的元素数
    const int64_t N4  = kMembers * NG;   // 全部 4 个成员
    const int64_t M4  = kMembers * G;    // GEMM 的行数

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    const size_t B = sizeof(c10::Half);
    auto alloc = [&](int64_t elems) {
        return SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(align256((uint32_t)(elems * B))));
    };
    // 峰值 = 下面这些之和 ≈ 7.07 MB/核（硬门槛 7.5）。`sc` 被复用三次：
    // gather(sc) → GEMM3 的输入 → gather(sc2) → combine 的 e[]。
    const uint32_t a_nxc  = alloc(N4);      // 2.41 MB  nxc，full
    const uint32_t a_sc   = alloc(N4);      // 2.41 MB  sc / sc2 / e[]，full
    const uint32_t a_pool = alloc(NG);      // 0.60 MB  pooled，之后复用为 mx / 乘积暂存
    const uint32_t a_sum  = alloc(NG);      // 0.60 MB  Σe，之后复用为末次 gather 的 dst
    const uint32_t a_out  = alloc(NG);      // 0.60 MB  加权和
    const uint32_t a_cs   = alloc(M4 * ln); // 0.30 MB  col-shard（GEMM 输出）
    const uint32_t a_csb  = alloc(G * ln);  // 0.075 MB pooled 那半的 col-shard
    const uint32_t a_cso  = alloc(G * ln);  // 0.075 MB proj2 的 col-shard
    const uint32_t a_ba   = alloc(ln);
    const uint32_t a_b2   = alloc(ln);
    const uint32_t a_bp   = alloc(ln);
    const uint32_t a_bp1  = proj1 ? alloc(ln) : 0u;   // proj1 的 bias（512 B）

    // ── 输入：nxc 每帧是新的 at::empty ⇒ 必须用 mutable 变体 ──────────────
    c10::Half* nxc_p = const_cast<c10::Half*>(nxc.data_ptr<c10::Half>());
    rpu_ddr_flush(nxc_p);
    static thread_local uint64_t merger_src_base = 0;
    merger_src_base = ::rhino_lkn::RpuGetDevAddr(nxc_p);
    // proj1 进图时广播的是 [4,G,K]（1.35 MB）而不是 [4,G,D]（2.41 MB），
    // 落在 a_sc（步 ③ 的 gather 之前它是空的）；否则照旧直接落 a_nxc。
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &merger_src_base, /*src_offset_bytes=*/0,
        proj1 ? (kMembers * G * K) : N4, proj1 ? a_sc : a_nxc, kCores);

    // bias：col-partition ⇒ 核 c 要的是 bias[c*ln : (c+1)*ln]，正好是 scatter 的
    // 语义（连续块按核切）。权重是注册张量、地址稳定 ⇒ 用 fixed 变体。
    auto stage_bias = [&](const at::Tensor& b, uint32_t dst) {
        TORCH_CHECK(b.numel() == D, "hyvla_merger_fused: bias numel ", b.numel(),
                    " != D ", D);
        c10::Half* p = const_cast<c10::Half*>(b.data_ptr<c10::Half>());
        rpu_ddr_flush(p);
        rpu_launch_ddr_scatter_spm_dma(p, ln, ln * (int64_t)B, dst, kCores);
    };
    stage_bias(b0, a_ba);
    stage_bias(b2, a_b2);
    stage_bias(bp2, a_bp);

    // ── ⓪ proj1（避免 ViT 尾部的 linear_ddr 路径）─────────────────────────
    // 与另外 4 枪同型：col-partition ACC32 GEMM（输入 full → 输出 col-shard）
    // + all_gather 回 full。proj1 必须使用 ACC32。
    if (proj1) {
        stage_bias(*bp1, a_bp1);
        rpu_launch_linear_spm_to_spm_kernel(a_sc, *wp1, a_cs, M4, D, K,
                                            /*partition=*/1, kCores, a_bp1);
        rpu_launch_all_gather_spm_kernel(a_cs, a_nxc, M4, ln, (int64_t)B, kCores);
    }

    // ── ① pooled = Σ_m nxc[m]（**不除 4**，1/4 折在 Wb 里，与 host 版同）──
    auto member = [&](uint32_t base, int64_t m) {
        return (uint32_t)(base + (uint32_t)(m * NG * (int64_t)B));
    };
    rpu_launch_eltwise_binary_spm_kernel(member(a_nxc, 0), member(a_nxc, 1),
                                         a_pool, NG, ValuOpType::ADD,
                                         c10::Half(1.0), kCores);
    for (int64_t m = 2; m < kMembers; ++m)
        rpu_launch_eltwise_binary_spm_kernel(a_pool, member(a_nxc, m), a_pool, NG,
                                             ValuOpType::ADD, c10::Half(1.0), kCores);

    // ── ② predictor.0（K 拆两半）+ GELU ─────────────────────────────────
    rpu_launch_linear_spm_to_spm_kernel(a_nxc, wa, a_cs, M4, D, D,
                                              /*partition=*/1, kCores, a_ba);
    rpu_launch_linear_spm_to_spm_kernel(a_pool, wb, a_csb, G, D, D,
                                              /*partition=*/1, kCores, /*bias=*/0);
    // pooled 那半对 4 个成员相同 ⇒ 广播加（host 版是算 4 遍，同值）。
    for (int64_t m = 0; m < kMembers; ++m) {
        const uint32_t dst = (uint32_t)(a_cs + (uint32_t)(m * G * ln * (int64_t)B));
        rpu_launch_eltwise_binary_spm_kernel(dst, a_csb, dst, G * ln,
                                             ValuOpType::ADD, c10::Half(1.0), kCores);
    }
    rpu_launch_eltwise_unary_spm_kernel(a_cs, a_cs, M4 * ln, ValuOpType::ADD,
                                        /*is_gelu=*/true, kCores);

    // ── ③ predictor.2 ────────────────────────────────────────────────────
    rpu_launch_all_gather_spm_kernel(a_cs, a_sc, M4, ln, (int64_t)B, kCores);
    rpu_launch_linear_spm_to_spm_kernel(a_sc, w2, a_cs, M4, D, D,
                                              /*partition=*/1, kCores, a_b2);
    rpu_launch_all_gather_spm_kernel(a_cs, a_sc, M4, ln, (int64_t)B, kCores);

    // ── ④ 组内 softmax + 加权求和（算法与 merger_combine 逐行相同）──────
    //   e_m = exp(sc_m - max_m sc_m)；out = (Σ_m nxc_m·e_m) / (Σ_m e_m)
    const uint32_t mx = a_pool;   // pooled 已经用完，借它当 max / 乘积暂存
    rpu_launch_eltwise_binary_spm_kernel(member(a_sc, 0), member(a_sc, 1), mx, NG,
                                         ValuOpType::MAX, c10::Half(1.0), kCores);
    for (int64_t m = 2; m < kMembers; ++m)
        rpu_launch_eltwise_binary_spm_kernel(mx, member(a_sc, m), mx, NG,
                                             ValuOpType::MAX, c10::Half(1.0), kCores);
    for (int64_t m = 0; m < kMembers; ++m) {
        const uint32_t e = member(a_sc, m);
        rpu_launch_eltwise_binary_spm_kernel(e, mx, e, NG, ValuOpType::SUB,
                                             c10::Half(1.0), kCores);
        rpu_launch_eltwise_unary_spm_kernel(e, e, NG, ValuOpType::EXP,
                                            /*is_gelu=*/false, kCores);
    }
    rpu_launch_eltwise_binary_spm_kernel(member(a_sc, 0), member(a_sc, 1), a_sum, NG,
                                         ValuOpType::ADD, c10::Half(1.0), kCores);
    for (int64_t m = 2; m < kMembers; ++m)
        rpu_launch_eltwise_binary_spm_kernel(a_sum, member(a_sc, m), a_sum, NG,
                                             ValuOpType::ADD, c10::Half(1.0), kCores);
    rpu_launch_eltwise_binary_spm_kernel(member(a_nxc, 0), member(a_sc, 0), a_out, NG,
                                         ValuOpType::MUL, c10::Half(1.0), kCores);
    for (int64_t m = 1; m < kMembers; ++m) {
        rpu_launch_eltwise_binary_spm_kernel(member(a_nxc, m), member(a_sc, m), mx, NG,
                                             ValuOpType::MUL, c10::Half(1.0), kCores);
        rpu_launch_eltwise_binary_spm_kernel(a_out, mx, a_out, NG, ValuOpType::ADD,
                                             c10::Half(1.0), kCores);
    }
    rpu_launch_eltwise_binary_spm_kernel(a_out, a_sum, a_out, NG, ValuOpType::DIV,
                                         c10::Half(1.0), kCores);

    // ── ⑤ GELU + proj2 ──────────────────────────────────────────────────
    rpu_launch_eltwise_unary_spm_kernel(a_out, a_out, NG, ValuOpType::ADD,
                                        /*is_gelu=*/true, kCores);
    rpu_launch_linear_spm_to_spm_kernel(a_out, wp2, a_cso, G, D, D,
                                              /*partition=*/1, kCores, a_bp);
    rpu_launch_all_gather_spm_kernel(a_cso, a_sum, G, ln, (int64_t)B, kCores);

    at::Tensor out = at::empty({G, D},
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
    c10::Half* op = out.data_ptr<c10::Half>();
    static thread_local uint64_t merger_dst_base = 0;
    merger_dst_base = ::rhino_lkn::RpuGetDevAddr(op);
    rpu_launch_spm_copy_ddr_dma_mutable(a_sum, &merger_dst_base,
                                        /*dst_offset_bytes=*/0, NG);
    rpu_ddr_flush(op);
    return out;
}

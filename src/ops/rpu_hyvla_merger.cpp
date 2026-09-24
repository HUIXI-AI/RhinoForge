// rpu_hyvla_merger.cpp — Hy-VLA DwPooler merger operations.
//
// member-major [4,G,D] layout makes each member's groups contiguous, turning
// pooling, four-member softmax and weighted sum into flat elementwise work.
// The immediate pool/combine variants split N=G*D evenly across eight cores;
// row boundaries do not matter for those elementwise slices.
// The ViT produces this ordering with permute3d. merger_fused additionally
// captures the linear and activation stages in one graph. The immediate
// variants remain available when that graph route is disabled.

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rpu_eltwise.h"
#include <c10/util/Half.h>
#include <vector>

namespace {

constexpr int64_t kMembers = 4;   // 2×2 的组内成员数
constexpr int     kCores   = 8;

inline uint32_t align256(uint32_t v) { return (v + 255u) & ~255u; }

// Move proj1 [2048,1152] into the merger when optional wp1/bp1 are supplied.
// ViT then emits [S,1152]; this op broadcasts it and runs column-partitioned
// ACC32 GEMM followed by all_gather.
// Reuse a_sc for the input before its later gather use, and a_cs for the
// column-sharded output; their declared capacities must cover both roles.
// Python binds the same cold choice to ViT set_weights and these optional
// tensors. Their presence is this launcher's sole authority.

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

// merger_pool computes sum over four members and tiles it as [4,G,D].
// The consumer folds the factor 1/4 into Wb. Equal-shaped predictor outputs
// can then be added without an implicit leading-axis broadcast.
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

// merger_fused captures pool, predictor, member softmax, weighted sum and proj2
// in one graph. Call it inside graph_cache.capture after the ViT capture has
// completed and its temporary SPM allocations have been reset.
//
// Column-partitioned GEMMs consume full inputs and produce column shards;
// all_gather restores full outputs. Elementwise stages operate redundantly on
// each full copy, avoiding a full-to-shard conversion that the wrappers cannot
// express with one shared SPM address.
// The pooled predictor input is common to all four members, so compute it once
// per group and broadcast its contribution to the member results.
//
// Use ACC32 linear_spm_to_spm to preserve the accumulation policy of the
// original F.linear path; changing to ACC16 changes the numerical operation.
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

    // proj1 uses column-partitioned ACC32 GEMM (full input, column-sharded output)
    // followed by all_gather. Keep the merger's ACC32 accumulation policy.
    if (proj1) {
        stage_bias(*bp1, a_bp1);
        rpu_launch_linear_spm_to_spm_kernel(a_sc, *wp1, a_cs, M4, D, K,
                                            /*partition=*/1, kCores, a_bp1);
        rpu_launch_all_gather_spm_kernel(
            a_cs, a_nxc, M4, ln, (int64_t)B, kCores,
            rpu_resolve_all_gather_schedule(ln, (int64_t)B));
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
    rpu_launch_all_gather_spm_kernel(
        a_cs, a_sc, M4, ln, (int64_t)B, kCores,
        rpu_resolve_all_gather_schedule(ln, (int64_t)B));
    rpu_launch_linear_spm_to_spm_kernel(a_sc, w2, a_cs, M4, D, D,
                                              /*partition=*/1, kCores, a_b2);
    rpu_launch_all_gather_spm_kernel(
        a_cs, a_sc, M4, ln, (int64_t)B, kCores,
        rpu_resolve_all_gather_schedule(ln, (int64_t)B));

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
    rpu_launch_all_gather_spm_kernel(
        a_cso, a_sum, G, ln, (int64_t)B, kCores,
        rpu_resolve_all_gather_schedule(ln, (int64_t)B));

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

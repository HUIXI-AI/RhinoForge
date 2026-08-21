// rpu_permute.cpp - Permute kernel implementation for RPU
#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"  // for SPM_ALLOC (isolated SPM tests only)
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ::rhino_lkn;

// Multi-core 3D SPM permutation over contiguous FP16 tensors.
void rpu_launch_permute3d_spm_kernel(uint32_t input_spm_addr, uint32_t output_spm_addr,
                                     int64_t b, int64_t n, int64_t c,
                                     const std::vector<int64_t>& perm, int num_cores) {
    TORCH_CHECK(perm.size() == 3, "permute3d_spm: perm must have 3 elements");
    const std::string order =
        std::to_string(perm[0]) + std::to_string(perm[1]) + std::to_string(perm[2]);

    std::string kernel_name;
    if (order == "102")      kernel_name = "transpose_nbc_c16";
    else if (order == "201") kernel_name = "transpose_cbn_c16";
    else if (order == "021") kernel_name = "transpose_bcn_c16";
    else if (order == "210") kernel_name = "transpose_cnb_c16";
    else if (order == "120") kernel_name = "transpose_ncb_c16";
    else TORCH_CHECK(false, "permute3d_spm: unsupported order ", order);

    Kernel_t* kernel = RpuKernelGraph::active().get_kernel_reset(kernel_name);
    TORCH_CHECK(kernel != nullptr, "permute3d_spm: failed to get ", kernel_name, " kernel");

    const int64_t c_v16 = CeilDiv(c, V16_NUM);
    const int64_t n_v16 = CeilDiv(n, V16_NUM);
    const int64_t b_v16 = CeilDiv(b, V16_NUM);
    const int64_t c_v16_per_wrp = std::min((int64_t)MAX_FP16_VLM_V16_PER_THD, c_v16);
    const int64_t c_v256_per_wrp = c_v16_per_wrp / V16_NUM;
    const int64_t c_per_thd = c_v16_per_wrp * V16_NUM;
    const int64_t c_per_wrp = c_v16_per_wrp * V16_NUM;
    const int64_t c_tail = c, n_tail = n, b_tail = b;  // contiguous -> no tail padding

    // common addr + c regs
    kernel->set_regs(0, (uint16_t)(input_spm_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(input_spm_addr >> 16));
    kernel->set_regs(2, (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(output_spm_addr >> 16));
    kernel->set_regs(4, (uint16_t)1);  // data_type=1 (fp16)
    kernel->set_regs(6, (uint16_t)c);
    kernel->set_regs(7, (uint16_t)c_tail);  // c_in_tail

    int64_t gx, gy, gz;
    const char last = order[2];
    if (last == '2') {
        // get_tailc_config (e.g. 102): output v16 tail on c
        const int64_t n_v16_per_wrp =
            std::min(n_v16, (int64_t)MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);
        const int64_t n_per_thd = n_v16_per_wrp * V16_NUM;
        const int64_t n_per_wrp = n_v16_per_wrp * V16_NUM;
        gx = CeilDiv(n_v16, n_v16_per_wrp);
        gy = CeilDiv(c_v16, c_v16_per_wrp);
        gz = b;
        kernel->set_regs(8,  (uint16_t)c_tail);  // c_out_tail
        kernel->set_regs(9,  (uint16_t)n);
        kernel->set_regs(10, (uint16_t)n_v16);
        kernel->set_regs(12, (uint16_t)c_per_thd);
        kernel->set_regs(13, (uint16_t)c_per_wrp);
        kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
        kernel->set_regs(15, (uint16_t)c_v256_per_wrp);
        kernel->set_regs(16, (uint16_t)n_per_thd);
        kernel->set_regs(17, (uint16_t)n_per_wrp);
        kernel->set_regs(18, (uint16_t)n_v16_per_wrp);
        kernel->set_regs(19, (uint16_t)b);
    } else if (last == '1') {
        // get_tailn_config (e.g. 201 / 021): output v16 tail on n
        const int64_t n_v16_per_wrp =
            std::min(n_v16, (int64_t)MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);
        const int64_t n_v16_tail = n_v16 + 1;
        const int64_t n_per_thd = n_v16_per_wrp * V16_NUM;
        const int64_t n_per_wrp = n_v16_per_wrp * V16_NUM;
        gx = CeilDiv(n_v16, n_v16_per_wrp);
        gy = CeilDiv(c_v16, c_v16_per_wrp);
        gz = b;
        kernel->set_regs(8,  (uint16_t)n);
        kernel->set_regs(9,  (uint16_t)n_v16);
        kernel->set_regs(10, (uint16_t)n_tail);
        kernel->set_regs(11, (uint16_t)n_v16_tail);
        kernel->set_regs(12, (uint16_t)c_per_thd);
        kernel->set_regs(13, (uint16_t)c_per_wrp);
        kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
        kernel->set_regs(15, (uint16_t)c_v256_per_wrp);
        kernel->set_regs(16, (uint16_t)n_per_thd);
        kernel->set_regs(17, (uint16_t)n_per_wrp);
        kernel->set_regs(18, (uint16_t)n_v16_per_wrp);
        kernel->set_regs(19, (uint16_t)b);
    } else {
        // get_tailb_config (e.g. 120 / 210): output v16 tail on b
        const int64_t b_v16_per_wrp =
            std::min(b_v16, (int64_t)MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);
        const int64_t b_v16_tail = b_v16 + 1;
        const int64_t b_per_thd = b_v16_per_wrp * V16_NUM;
        const int64_t b_per_wrp = b_v16_per_wrp * V16_NUM;
        gx = CeilDiv(b_v16, b_v16_per_wrp);
        gy = CeilDiv(c_v16, c_v16_per_wrp);
        gz = n;
        kernel->set_regs(8,  (uint16_t)b);
        kernel->set_regs(9,  (uint16_t)b_v16);
        kernel->set_regs(10, (uint16_t)b_tail);
        kernel->set_regs(11, (uint16_t)b_v16_tail);
        kernel->set_regs(12, (uint16_t)c_per_thd);
        kernel->set_regs(13, (uint16_t)c_per_wrp);
        kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
        kernel->set_regs(15, (uint16_t)c_v256_per_wrp);
        kernel->set_regs(16, (uint16_t)b_per_thd);
        kernel->set_regs(17, (uint16_t)b_per_wrp);
        kernel->set_regs(18, (uint16_t)b_v16_per_wrp);
        kernel->set_regs(19, (uint16_t)n);
    }
    kernel->set_regs(64, (uint16_t)gx);
    kernel->set_regs(65, (uint16_t)gy);
    kernel->set_regs(66, (uint16_t)gz);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {(uint16_t)gx, (uint16_t)gy, (uint16_t)gz}, cores);
}

// SPM-resident 4D 0213 permute: [q,b,n,c] -> [q,n,b,c]. This is the SPM
// SPM form of the 0213 transpose contract. Keep the
// production surface deliberately narrow: G0.5 patch layout only needs 0213.
void rpu_launch_permute4d_0213_spm_kernel(
    uint32_t input_spm_addr, uint32_t output_spm_addr,
    int64_t q, int64_t b, int64_t n, int64_t c, int num_cores) {
    TORCH_CHECK(q > 0 && b > 0 && n > 0 && c > 0,
                "permute4d_0213_spm: dimensions must be positive");
    TORCH_CHECK(c % V16_NUM == 0,
                "permute4d_0213_spm: c must be 16-aligned, got ", c);
    constexpr int64_t kU16Max = 65535;
    TORCH_CHECK(q <= kU16Max && b <= kU16Max && n <= kU16Max &&
                    c <= kU16Max && b * q <= kU16Max,
                "permute4d_0213_spm: dimensions exceed u16 register/grid range: q=",
                q, " b=", b, " n=", n, " c=", c);
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "permute4d_0213_spm: num_cores must be in [1,8], got ", num_cores);

    Kernel_t* kernel =
        RpuKernelGraph::active().get_kernel_reset("transpose_qnbc_c16");
    TORCH_CHECK(kernel != nullptr,
                "permute4d_0213_spm: failed to get transpose_qnbc_c16 kernel");

    const int64_t c_v16 = CeilDiv(c, V16_NUM);
    const int64_t n_v16 = CeilDiv(n, V16_NUM);
    const int64_t c_v16_per_wrp =
        std::min((int64_t)MAX_FP16_VLM_V16_PER_THD, c_v16);
    const int64_t n_v16_per_wrp =
        std::min(n_v16,
                 (int64_t)MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);
    const int64_t c_v256_per_wrp = c_v16_per_wrp / V16_NUM;
    const int64_t c_per_thd = c_v16_per_wrp * V16_NUM;
    const int64_t c_per_wrp = c_per_thd;
    const int64_t n_per_thd = n_v16_per_wrp * V16_NUM;
    const int64_t n_per_wrp = n_per_thd;
    const int64_t gx = CeilDiv(n_v16, n_v16_per_wrp);
    const int64_t gy = CeilDiv(c_v16, c_v16_per_wrp);
    const int64_t gz = b * q;

    kernel->set_regs(0, (uint16_t)(input_spm_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(input_spm_addr >> 16));
    kernel->set_regs(2, (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(output_spm_addr >> 16));
    kernel->set_regs(4, (uint16_t)1);  // data_type=1 (fp16)
    kernel->set_regs(6, (uint16_t)c);
    kernel->set_regs(7, (uint16_t)c);  // c_in_tail: contiguous, unpadded
    kernel->set_regs(8, (uint16_t)c);  // c_out_tail
    kernel->set_regs(9, (uint16_t)n);
    kernel->set_regs(10, (uint16_t)n_v16);
    kernel->set_regs(12, (uint16_t)c_per_thd);
    kernel->set_regs(13, (uint16_t)c_per_wrp);
    kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
    kernel->set_regs(15, (uint16_t)c_v256_per_wrp);
    kernel->set_regs(16, (uint16_t)n_per_thd);
    kernel->set_regs(17, (uint16_t)n_per_wrp);
    kernel->set_regs(18, (uint16_t)n_v16_per_wrp);
    kernel->set_regs(19, (uint16_t)b);
    kernel->set_regs(20, (uint16_t)q);
    kernel->set_regs(64, (uint16_t)gx);
    kernel->set_regs(65, (uint16_t)gy);
    kernel->set_regs(66, (uint16_t)gz);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel,
                      {(uint16_t)gx, (uint16_t)gy, (uint16_t)gz}, cores);
}

// rpu_sdpa_dispatch.cpp — Unified SDPA SPM kernel dispatch by SdpaKernelType

#include "rpu_ops.h"

void rpu_launch_sdpa_spm_dispatch(
    SdpaKernelType kernel,
    const at::Tensor &key, const at::Tensor &value,
    int attn_mask_type, std::optional<double> scale,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off, uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t kv_seq_len, int num_cores, int virtual_num_cores,
    int64_t cache_batch_offset_elems)
{
    switch (kernel) {
        case SdpaKernelType::FLASH_ATTN_SPM:
            rpu_launch_sdpa_spm_unified_kernel_v2(
                key, value, attn_mask_type, scale,
                q_off, output_off, sdpa_tmp_off, sdpa_mask_off,
                seq_q, num_q_heads, num_kv_heads, head_dim, kv_seq_len,
                num_cores, virtual_num_cores, cache_batch_offset_elems);
            break;

        case SdpaKernelType::FLASH_ATTN_SPM_VCTXLEN:
            // Batch decode is wired for FLASH_ATTN_SPM only (the CausalDecoderModel
            // default). vctxlen has no batched caller yet; fail loud rather than
            // silently reading batch slot 0 for every sequence.
            TORCH_CHECK(cache_batch_offset_elems == 0,
                        "rpu_launch_sdpa_spm_dispatch: batch decode "
                        "(cache_batch_offset_elems != 0) is not supported for "
                        "FLASH_ATTN_SPM_VCTXLEN");
            rpu_launch_sdpa_spm_vctxlen_kernel(
                key, value, attn_mask_type, scale,
                q_off, output_off, sdpa_tmp_off, sdpa_mask_off,
                seq_q, num_q_heads, num_kv_heads, head_dim, kv_seq_len,
                num_cores, virtual_num_cores);
            break;

        default:
            TORCH_CHECK(false, "rpu_launch_sdpa_spm_dispatch: unhandled kernel type ", (int)kernel);
    }
}

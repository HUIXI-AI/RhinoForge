// rpu_concat.cpp - Concat operation for RPU backend (memcpy version)
#include "rpu_ops.h"
#include <string.h>  // for memcpy
#include <vector>

using namespace at;

// ============================================================================
// rpu_cat - 原生 memcpy 实现
// 直接在 DDR 上进行内存拷贝，避免 CPU allocator 分配和设备转换开销
// ============================================================================
at::Tensor rpu_cat(const c10::IListRef<at::Tensor>& tensors, int64_t dim) {
    TORCH_CHECK(!tensors.empty(), "cat: expected a non-empty list of tensors");

    // 收集输入 tensor 并确保 contiguous
    std::vector<at::Tensor> inputs;
    inputs.reserve(tensors.size());
    c10::Device target_device = c10::kCPU;

    for (const auto& t : tensors) {
        if (t.numel() == 0) continue;  // 跳过空 tensor
        if (target_device == c10::kCPU &&
            t.device().type() == c10::DeviceType::PrivateUse1) {
            target_device = t.device();
        }
        inputs.push_back(t.contiguous());
    }

    if (inputs.empty()) {
        // 所有输入都是空的，返回空 tensor
        return at::empty({0}, tensors.front().options());
    }

    const auto& first = inputs[0];
    int64_t ndim = first.dim();

    // 处理标量 tensor (0 维)
    if (ndim == 0) {
        TORCH_CHECK(false, "cat: zero-dimensional tensors cannot be concatenated");
    }

    // 处理负数 dim
    if (dim < 0) {
        dim = dim + ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
                "cat: dimension ", dim, " out of range for tensor with ",
                ndim, " dimensions");

    // 计算输出形状
    std::vector<int64_t> output_shape = first.sizes().vec();
    int64_t cat_dim_size = first.size(dim);

    for (size_t i = 1; i < inputs.size(); i++) {
        const auto& t = inputs[i];
        TORCH_CHECK(t.dim() == ndim,
                    "cat: all tensors must have the same number of dimensions, "
                    "got ", ndim, " and ", t.dim());
        TORCH_CHECK(t.scalar_type() == first.scalar_type(),
                    "cat: all tensors must have the same dtype");

        // 检查非 cat 维度是否匹配
        for (int64_t d = 0; d < ndim; d++) {
            if (d != dim) {
                TORCH_CHECK(t.size(d) == first.size(d),
                            "cat: sizes must match except in dimension ", dim,
                            ", got ", first.size(d), " and ", t.size(d),
                            " in dimension ", d);
            }
        }
        cat_dim_size += t.size(dim);
    }
    output_shape[dim] = cat_dim_size;

    // 在 RPU 上分配输出
    auto output = at::empty(output_shape, first.options().device(target_device));

    // 计算 memcpy 参数
    // outer_size: dim 之前所有维度的乘积
    // inner_size: dim 之后所有维度的乘积
    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; d++) {
        outer_size *= output_shape[d];
    }

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; d++) {
        inner_size *= output_shape[d];
    }

    size_t elem_size = first.element_size();
    // 输出在 cat 维度上每个元素占用的字节数
    int64_t out_dim_stride = inner_size * elem_size;
    // 输出在 outer 维度上的步长 (bytes)
    int64_t out_outer_stride = output_shape[dim] * out_dim_stride;

    char* out_ptr = static_cast<char*>(output.data_ptr());
    int64_t cat_offset = 0;  // 当前在 cat 维度上的偏移

    // Host memcpy crosses the CPU<->RPU coherency boundary.  The regular
    // rpu_ddr_flush knob is intentionally disabled for intra-RPU traffic, so
    // it cannot guard these reads/writes.  Flush only the live tensor ranges.
    for (const auto& inp : inputs) {
        if (inp.device().type() == c10::DeviceType::PrivateUse1) {
            rpu_ddr_flush_force_sized(inp.data_ptr(), inp.nbytes());
        }
    }

    for (const auto& inp : inputs) {
        const char* inp_ptr = static_cast<const char*>(inp.data_ptr());
        int64_t inp_cat_size = inp.size(dim);
        // 输入在 outer 维度上的步长 (bytes)
        int64_t inp_outer_stride = inp_cat_size * out_dim_stride;
        // 每次拷贝的字节数
        int64_t copy_size = inp_cat_size * out_dim_stride;

        for (int64_t o = 0; o < outer_size; o++) {
            char* dst = out_ptr + o * out_outer_stride + cat_offset * out_dim_stride;
            const char* src = inp_ptr + o * inp_outer_stride;
            memcpy(dst, src, copy_size);
        }
        cat_offset += inp_cat_size;
    }

    if (target_device.type() == c10::DeviceType::PrivateUse1) {
        // Publish the host-written output before the next RPU DMA reads it.
        rpu_ddr_flush_force_sized(out_ptr, output.nbytes());
    }

    return output;
}

// ============================================================================
// rpu_cat_out - 带输出参数的版本 (用于 inplace 场景)
// ============================================================================
at::Tensor& rpu_cat_out(const c10::IListRef<at::Tensor>& tensors,
                         int64_t dim,
                         at::Tensor& out) {
    auto result = rpu_cat(tensors, dim);
    out.resize_(result.sizes());
    out.copy_(result);
    return out;
}

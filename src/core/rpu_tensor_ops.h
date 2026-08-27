// Core PrivateUse1 tensor/storage bridge declared for rpu_tensor_ops.inc.
#pragma once

#include <ATen/ATen.h>

#include <optional>
#include <vector>

// Called exactly once by the pybind composition root before KernelCache init.
// The registered allocator/device guard intentionally have process lifetime;
// PyTorch does not provide safe unregister APIs for either registry.
void initialize_rpu_tensor_runtime();
bool rpu_device_nodes_accessible();

std::vector<int64_t> compute_contiguous_strides(c10::IntArrayRef sizes);

at::Tensor rpu_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory);
at::Tensor rpu_empty(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format);

at::Tensor rpu_view(const at::Tensor& self, at::IntArrayRef new_sizes);
at::Tensor rpu_as_strided(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);
at::Tensor rpu_transpose(
    const at::Tensor& self, int64_t dim0, int64_t dim1);
at::Tensor rpu_expand(
    const at::Tensor& self, at::IntArrayRef size, bool implicit);
at::Tensor rpu_unsqueeze(const at::Tensor& self, int64_t dim);

at::Tensor rpu_arange(
    const c10::Scalar& end,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory);
at::Tensor rpu_arange_start(
    const c10::Scalar& start,
    const c10::Scalar& end,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory);
at::Tensor rpu_arange_start_step(
    const c10::Scalar& start,
    const c10::Scalar& end,
    const c10::Scalar& step,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory);

at::Tensor& rpu_copy_(
    at::Tensor& self, const at::Tensor& src, bool non_blocking);
at::Tensor rpu_contiguous(
    const at::Tensor& self, c10::MemoryFormat memory_format);
at::Tensor rpu_to_copy(
    const at::Tensor& self,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> memory_format);
at::Tensor rpu__copy_from_and_resize(
    const at::Tensor& self, const at::Tensor& src);

at::Tensor rpu_silu(const at::Tensor& self);
at::Tensor rpu_sqrt(const at::Tensor& self);
at::Tensor rpu_exp(const at::Tensor& self);
at::Tensor rpu_log(const at::Tensor& self);
at::Tensor rpu_sin(const at::Tensor& self);
at::Tensor rpu_cos(const at::Tensor& self);
at::Tensor rpu_rsqrt(const at::Tensor& self);
at::Tensor rpu_tanh(const at::Tensor& self);
at::Tensor& rpu_normal_(
    at::Tensor& self,
    double mean,
    double std,
    c10::optional<at::Generator> generator);
at::Tensor rpu_matmul(
    const at::Tensor& tensor1, const at::Tensor& tensor2);
at::Tensor rpu_bmm_fallback(
    const at::Tensor& self, const at::Tensor& mat2);
at::Tensor rpu_softmax(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype);
at::Tensor rpu_bmm(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor rpu_layernorm(
    const at::Tensor& input,
    c10::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    double eps,
    bool cudnn_enable);
at::Tensor rpu_dropout(
    const at::Tensor& input, double p, bool training);
std::vector<at::Tensor> rpu_broadcast(
    c10::ArrayRef<at::Tensor> tensors);

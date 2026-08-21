// PrivateUse1 allocator, device guard, views, copies, and basic tensor fragment.
#include "rpu_tensor_ops.h"

#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/transformers/attention.h>

#include "rpu_ops.h"
#include "rpu_caching_allocator.h"
#include "rpu_spm_allocator.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace at;

// Implemented by graph_spm_reset_guard.cpp. Keep graph runtime details out of
// this storage bridge fragment.
bool graph_aware_rpu_to_cpu_zerocopy_sync();

// ------------------------ RPUAllocator ------------------------
// 支持切换 direct allocator 和 caching allocator
extern bool g_rpu_caching_allocator_enabled;

bool rpu_device_nodes_accessible() {
  return ::access("/dev/rpu", R_OK | W_OK) == 0 &&
         ::access("/dev/mem", R_OK | W_OK) == 0;
}

struct RPUAllocator final : public c10::Allocator {
  at::DataPtr allocate(size_t nbytes) override {
    // Handle empty tensor allocation: return nullptr DataPtr
    if (nbytes == 0) {
      return {nullptr, nullptr, nullptr, at::Device(c10::DeviceType::PrivateUse1, 0)};
    }

    TORCH_CHECK(
        rpu_device_nodes_accessible(),
        "RPU device is unavailable to this process: require read/write access "
        "to /dev/rpu and /dev/mem");

    if (g_rpu_caching_allocator_enabled) {
      // 使用 caching allocator
      return rpu::RPUCachingAllocatorAdapter::get().allocate(nbytes);
    } else {
      // 使用 direct allocator
      void *ptr = rhino_lkn::RpuDdrAlloc(nbytes);
      if (!ptr)
        throw std::bad_alloc();
      return {ptr, ptr, &rhino_lkn::RpuDdrFree, at::Device(c10::DeviceType::PrivateUse1, 0)};
    }
  }
  void copy_data(void *dest, const void *src,
                 std::size_t count) const override {
    std::memcpy(dest, src, count);
  }
};

// ------------------------ DeviceGuardImpl ------------------------
class RPUDeviceGuardImpl : public c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }
  c10::Device exchangeDevice(c10::Device d) const override { return d; }
  c10::Device getDevice() const override {
    return {c10::DeviceType::PrivateUse1, 0};
  }
  void setDevice(c10::Device d) const override {}
  void uncheckedSetDevice(c10::Device d) const noexcept override {}
  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return s;
  }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
};

void initialize_rpu_tensor_runtime() {
  static std::once_flag registration_once;
  std::call_once(registration_once, [] {
    // PyTorch registries keep raw pointers for process lifetime and expose no
    // safe unregister operation. Deliberately leak these two tiny objects so
    // shared-library/static destruction order cannot invalidate the pointers.
    auto* allocator = new RPUAllocator();
    c10::SetAllocator(c10::DeviceType::PrivateUse1, allocator);

    auto* device_guard = new RPUDeviceGuardImpl();
    c10::impl::registerDeviceGuard(
        c10::DeviceType::PrivateUse1, device_guard);
  });
}

// ------------------------ Helpers ------------------------
std::vector<int64_t> compute_contiguous_strides(c10::IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (int i = (int)sizes.size() - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= sizes[i];
  }
  return strides;
}

at::Tensor make_tensor_from_data_ptr(c10::DataPtr &&dp, at::IntArrayRef sizes,
                                     at::IntArrayRef strides,
                                     c10::TensorOptions options) {
  TORCH_CHECK(options.has_dtype(), "make_tensor_from_data_ptr requires dtype");
  // only float support in minimal demo
  TORCH_CHECK(options.dtype() == at::kFloat,
              "only float supported in minimal demo");

  // compute byte size
  int64_t numel = 1;
  for (auto s : sizes)
    numel *= s;
  size_t byte_size =
      numel * c10::elementSize(c10::typeMetaToScalarType(options.dtype()));

  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), static_cast<int64_t>(byte_size),
      std::move(dp),
      /* allocator */ nullptr,
      /* resizable */ false);

  // Create TensorImpl that shares this storage_impl and has PrivateUse1
  // dispatch key
  auto tensor_impl = c10::make_intrusive<at::TensorImpl>(
      c10::Storage(storage_impl),
      c10::DispatchKeySet{c10::DispatchKey::PrivateUse1}, options.dtype());

  tensor_impl->set_sizes_and_strides(sizes, strides);
  return at::Tensor(std::move(tensor_impl));
}

// // ------------------------ empty_strided ------------------------
at::Tensor rpu_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                             std::optional<c10::ScalarType> dtype_opt,
                             std::optional<c10::Layout> layout_opt,
                             std::optional<c10::Device> device_opt,
                             std::optional<bool> pin_memory_opt) {
  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, GetAllocator(DeviceType::PrivateUse1), privateuse1_ks,
      dtype);
}

// ------------------------ empty ------------------------
at::Tensor rpu_empty(c10::IntArrayRef size,
                     std::optional<c10::ScalarType> dtype_opt,
                     std::optional<c10::Layout> layout_opt,
                     std::optional<c10::Device> device_opt,
                     std::optional<bool> pin_memory_opt,
                     std::optional<c10::MemoryFormat> memory_format_opt) {
  // Compute contiguous strides
  auto strides = compute_contiguous_strides(size);
  return rpu_empty_strided(size, strides, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// ------------------------ view (zero-copy) ------------------------
at::Tensor rpu_view(const at::Tensor &self, at::IntArrayRef new_sizes) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_view: self must be on rpu");
  // validate numel matches
  int64_t old_numel = self.numel();
  int64_t new_numel = 1;
  for (auto s : new_sizes)
    new_numel *= s;
  TORCH_CHECK(old_numel == new_numel, "rpu_view: numel mismatch");

  // compute new strides as contiguous for simplicity
  std::vector<int64_t> new_strides = compute_contiguous_strides(new_sizes);

  auto storage = self.storage();
  auto new_tensor_impl = c10::make_intrusive<at::TensorImpl>(
      c10::Storage(std::move(storage)),
      c10::DispatchKeySet{c10::DispatchKey::PrivateUse1}, self.dtype());

  // preserve storage offset
  new_tensor_impl->set_storage_offset(self.storage_offset());
  new_tensor_impl->set_sizes_and_strides(new_sizes, new_strides);

  return at::Tensor(std::move(new_tensor_impl));
}

at::Tensor rpu_as_strided(const at::Tensor &self, at::IntArrayRef size,
                          at::IntArrayRef stride,
                          c10::optional<int64_t> storage_offset) {
  // as_strided 是视图操作，直接在 RPU 上创建新的视图
  auto storage = self.storage();
  auto new_tensor_impl = c10::make_intrusive<at::TensorImpl>(
      c10::Storage(std::move(storage)),
      c10::DispatchKeySet{c10::DispatchKey::PrivateUse1}, self.dtype());

  new_tensor_impl->set_storage_offset(storage_offset.value_or(self.storage_offset()));
  new_tensor_impl->set_sizes_and_strides(size, stride);

  return at::Tensor(std::move(new_tensor_impl));
}

// at::Tensor rpu_transpose(const at::Tensor &self, int64_t dim0, int64_t dim1)
// {
//   auto transposed = self.cpu().transpose(dim0, dim1).contiguous();
//   return transposed.to(self.device());
// }

at::Tensor rpu_transpose(const at::Tensor &self, int64_t dim0, int64_t dim1) {
  auto sizes = self.sizes().vec();
  std::swap(sizes[dim0], sizes[dim1]);

  // 对 contiguous tensor 来说，只需交换 stride
  auto strides = self.strides().vec();
  std::swap(strides[dim0], strides[dim1]);

  // return at::as_strided(self, sizes, strides, self.storage_offset());

  // // This creates a Tensor that aliases the same storage and
  // // properly registers aliasing with autograd.
  // return self.alias_with_sizes_and_strides(sizes, strides);

  auto storage = self.storage();
  auto new_tensor_impl = c10::make_intrusive<at::TensorImpl>(
      c10::Storage(std::move(storage)),
      c10::DispatchKeySet{c10::DispatchKey::PrivateUse1}, self.dtype());
  new_tensor_impl->set_storage_offset(self.storage_offset());
  new_tensor_impl->set_sizes_and_strides(sizes, strides);

  return at::Tensor(std::move(new_tensor_impl));
}

// ------------------------ expand (view operation) ------------------------
at::Tensor rpu_expand(const at::Tensor &self, at::IntArrayRef size, bool implicit) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_expand: self must be on rpu");

  auto self_sizes = self.sizes();
  auto self_strides = self.strides();
  int64_t self_dim = self.dim();
  int64_t target_dim = size.size();

  // Validate: target_dim must be >= self_dim
  TORCH_CHECK(target_dim >= self_dim,
              "rpu_expand: the number of sizes provided (", target_dim,
              ") must be greater or equal to the number of dimensions in the tensor (",
              self_dim, ")");

  // Compute new sizes and strides
  std::vector<int64_t> new_sizes(target_dim);
  std::vector<int64_t> new_strides(target_dim);

  // Fill from the back (align to the right)
  for (int64_t i = target_dim - 1; i >= 0; --i) {
    int64_t offset = self_dim - (target_dim - i);

    if (offset < 0) {
      // New dimension added at the front
      new_sizes[i] = size[i];
      new_strides[i] = 0;  // Broadcast stride
    } else {
      // Existing dimension
      int64_t self_size = self_sizes[offset];
      int64_t target_size = size[i];

      if (target_size == -1) {
        // Keep original size
        new_sizes[i] = self_size;
        new_strides[i] = self_strides[offset];
      } else if (self_size == 1) {
        // Expand dimension of size 1
        new_sizes[i] = target_size;
        new_strides[i] = 0;  // Broadcast: stride becomes 0
      } else if (self_size == target_size) {
        // Size matches, keep stride
        new_sizes[i] = target_size;
        new_strides[i] = self_strides[offset];
      } else {
        TORCH_CHECK(false,
                    "rpu_expand: the expanded size of the tensor (", target_size,
                    ") must match the existing size (", self_size,
                    ") at non-singleton dimension ", i);
      }
    }
  }

  // Create new tensor view with same storage but different sizes/strides
  auto storage = self.storage();
  auto new_tensor_impl = c10::make_intrusive<at::TensorImpl>(
      c10::Storage(std::move(storage)),
      c10::DispatchKeySet{c10::DispatchKey::PrivateUse1}, self.dtype());
  new_tensor_impl->set_storage_offset(self.storage_offset());
  new_tensor_impl->set_sizes_and_strides(new_sizes, new_strides);

  return at::Tensor(std::move(new_tensor_impl));
}

// ------------------------ unsqueeze (view operation) ------------------------
at::Tensor rpu_unsqueeze(const at::Tensor &self, int64_t dim) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_unsqueeze: self must be on rpu");

  int64_t ndim = self.dim();
  // Normalize dim: allow negative indexing
  if (dim < 0) {
    dim = dim + ndim + 1;
  }

  TORCH_CHECK(dim >= 0 && dim <= ndim,
              "rpu_unsqueeze: dimension out of range (expected to be in range of [",
              -ndim - 1, ", ", ndim, "], but got ", dim, ")");

  // Compute new sizes and strides
  auto old_sizes = self.sizes();
  auto old_strides = self.strides();

  std::vector<int64_t> new_sizes;
  std::vector<int64_t> new_strides;

  new_sizes.reserve(ndim + 1);
  new_strides.reserve(ndim + 1);

  for (int64_t i = 0; i < dim; ++i) {
    new_sizes.push_back(old_sizes[i]);
    new_strides.push_back(old_strides[i]);
  }

  // Insert size=1 dimension
  new_sizes.push_back(1);
  // Stride for size-1 dimension can be anything, use 1
  new_strides.push_back(ndim > 0 ? old_strides[std::min(dim, ndim - 1)] : 1);

  for (int64_t i = dim; i < ndim; ++i) {
    new_sizes.push_back(old_sizes[i]);
    new_strides.push_back(old_strides[i]);
  }

  // Create new tensor view with same storage but different sizes/strides
  auto storage = self.storage();
  auto new_tensor_impl = c10::make_intrusive<at::TensorImpl>(
      c10::Storage(std::move(storage)),
      c10::DispatchKeySet{c10::DispatchKey::PrivateUse1}, self.dtype());
  new_tensor_impl->set_storage_offset(self.storage_offset());
  new_tensor_impl->set_sizes_and_strides(new_sizes, new_strides);

  return at::Tensor(std::move(new_tensor_impl));
}

// ------------------------ arange (creation op, fallback to CPU) ------------------------
at::Tensor rpu_arange(const c10::Scalar &end,
                      c10::optional<c10::ScalarType> dtype,
                      c10::optional<c10::Layout> layout,
                      c10::optional<c10::Device> device,
                      c10::optional<bool> pin_memory) {
  // Generate on CPU then move to RPU
  // Note: we force device to CPU to avoid infinite recursion
  auto cpu_result = at::arange(end,
                                dtype.value_or(at::kLong),
                                layout.value_or(at::kStrided),
                                at::kCPU,
                                pin_memory.value_or(false));
  return cpu_result.to(c10::Device(c10::DeviceType::PrivateUse1, 0));
}

at::Tensor rpu_arange_start(const c10::Scalar &start,
                            const c10::Scalar &end,
                            c10::optional<c10::ScalarType> dtype,
                            c10::optional<c10::Layout> layout,
                            c10::optional<c10::Device> device,
                            c10::optional<bool> pin_memory) {
  // Generate on CPU then move to RPU
  auto cpu_result = at::arange(start, end,
                                dtype.value_or(at::kLong),
                                layout.value_or(at::kStrided),
                                at::kCPU,
                                pin_memory.value_or(false));
  return cpu_result.to(c10::Device(c10::DeviceType::PrivateUse1, 0));
}

at::Tensor rpu_arange_start_step(const c10::Scalar &start,
                                 const c10::Scalar &end,
                                 const c10::Scalar &step,
                                 c10::optional<c10::ScalarType> dtype,
                                 c10::optional<c10::Layout> layout,
                                 c10::optional<c10::Device> device,
                                 c10::optional<bool> pin_memory) {
  // Generate on CPU then move to RPU
  auto cpu_result = at::arange(start, end, step,
                                dtype.value_or(at::kLong),
                                layout.value_or(at::kStrided),
                                at::kCPU,
                                pin_memory.value_or(false));
  return cpu_result.to(c10::Device(c10::DeviceType::PrivateUse1, 0));
}

// ======================== Zero-Copy Device Conversion ========================
// 由于 CPU 和 RPU 共享 DDR，可以在不复制数据的情况下转换设备类型
// 这些函数创建一个新的 tensor，使用相同的存储但不同的 device 类型

// RPU tensor -> CPU tensor (零拷贝)
// 注意: 返回的 CPU tensor 与原 RPU tensor 共享存储，修改一个会影响另一个
// 通过持有原始 Storage 的副本来保证生命周期安全（Storage 内部使用引用计数）
static at::Tensor rpu_to_cpu_zerocopy_impl(const at::Tensor &rpu_tensor,
                                           bool flush,
                                           bool graph_sync) {
  TORCH_CHECK(rpu_tensor.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_to_cpu_zerocopy: input must be on RPU device");
  // Defensive sync — 内部直接调本函数的 op (e.g.
  // rpu_linear CPU fallback / rpu_eltwise_binary CPU fallback) 期望立即拿到
  // 有效 DDR data。RECORDING + pending nodes 时 sync_point() 把已录前缀 oneshot
  // 落地。注意:PyTorch `tensor.to('cpu')` 的 Composite 分解走 CPU side memcpy
  // 完全不经过本函数;那条主路径靠 spm_alloc_reset_temporary 的 sync hook 覆盖
  // (Pi0.5 adapter 子系统出口模式)。本钩子是兜底,捕获 op-internal RPU→CPU。
  if (graph_sync) {
    graph_aware_rpu_to_cpu_zerocopy_sync();
  }

  // 获取原始存储
  auto original_storage = rpu_tensor.storage();
  void* data_ptr = original_storage.data_ptr().get();

  // Cache 一致性: RPU(DDR) → CPU 读之前,必须 flush,确保 CPU 读到的是 RPU
  // kernel 最新写入的数据。集中在边界处理,op 内部不再各自 flush。
  if (flush) {
    rpu_ddr_flush_force_sized(data_ptr, original_storage.nbytes());
  }

  // 复制 Storage 对象到堆上（这会自动增加 StorageImpl 的引用计数）
  // 当 deleter 删除它时，引用计数会减少
  auto* storage_holder = new c10::Storage(original_storage);

  // 创建 deleter: 当零拷贝 tensor 被释放时，删除持有的 Storage 副本
  auto release_storage = [](void* ctx) {
    if (ctx) {
      delete static_cast<c10::Storage*>(ctx);
    }
  };

  // 创建一个 CPU storage，共享同一块内存
  auto cpu_storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      original_storage.nbytes(),
      c10::DataPtr(data_ptr, storage_holder, release_storage, at::Device(at::kCPU)),
      nullptr,  // allocator = nullptr
      false     // resizable = false
  );

  // 创建 CPU TensorImpl
  auto tensor_impl = c10::make_intrusive<at::TensorImpl>(
      std::move(cpu_storage),
      c10::DispatchKeySet{c10::DispatchKey::CPU},
      rpu_tensor.dtype());

  tensor_impl->set_storage_offset(rpu_tensor.storage_offset());
  tensor_impl->set_sizes_and_strides(rpu_tensor.sizes(), rpu_tensor.strides());

  return at::Tensor(std::move(tensor_impl));
}

at::Tensor rpu_to_cpu_zerocopy(const at::Tensor &rpu_tensor, bool flush) {
  return rpu_to_cpu_zerocopy_impl(rpu_tensor, flush, /*graph_sync=*/true);
}

at::Tensor rpu_to_cpu_zerocopy_no_graph_sync(
    const at::Tensor &rpu_tensor, bool flush) {
  return rpu_to_cpu_zerocopy_impl(rpu_tensor, flush, /*graph_sync=*/false);
}

// CPU tensor -> RPU tensor (零拷贝)
// 注意: 仅适用于通过 RpuDdrAlloc 分配的内存（即原本就在共享 DDR 上的数据）
// 通过持有原始 Storage 的副本来保证生命周期安全（Storage 内部使用引用计数）
at::Tensor cpu_to_rpu_zerocopy(const at::Tensor &cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(),
              "cpu_to_rpu_zerocopy: input must be on CPU device");

  // 获取原始存储
  auto original_storage = cpu_tensor.storage();
  void* data_ptr = original_storage.data_ptr().get();
  rpu_ddr_flush_force_sized(data_ptr, original_storage.nbytes());

  // 复制 Storage 对象到堆上（这会自动增加 StorageImpl 的引用计数）
  auto* storage_holder = new c10::Storage(original_storage);

  // 创建 deleter
  auto release_storage = [](void* ctx) {
    if (ctx) {
      delete static_cast<c10::Storage*>(ctx);
    }
  };

  // 创建一个 RPU storage，共享同一块内存
  auto rpu_storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      original_storage.nbytes(),
      c10::DataPtr(data_ptr, storage_holder, release_storage, at::Device(c10::DeviceType::PrivateUse1, 0)),
      nullptr,
      false
  );

  // 创建 RPU TensorImpl
  auto tensor_impl = c10::make_intrusive<at::TensorImpl>(
      std::move(rpu_storage),
      c10::DispatchKeySet{c10::DispatchKey::PrivateUse1},
      cpu_tensor.dtype());

  tensor_impl->set_storage_offset(cpu_tensor.storage_offset());
  tensor_impl->set_sizes_and_strides(cpu_tensor.sizes(), cpu_tensor.strides());

  return at::Tensor(std::move(tensor_impl));
}

// 便捷宏：用于 CPU fallback 场景
// 将 RPU tensors 零拷贝转换为 CPU，执行 CPU 操作，再将结果零拷贝转回 RPU
#define RPU_TO_CPU_ZEROCOPY(tensor) rpu_to_cpu_zerocopy(tensor)
#define CPU_TO_RPU_ZEROCOPY(tensor) cpu_to_rpu_zerocopy(tensor)

// ------------------------ copy_ (CPU <-> RPU, RPU <-> RPU)
// ------------------------

// Helper: compute element offset given multi-index and strides (element units)
static inline int64_t offset_from_index(const int64_t *idx,
                                        const int64_t *strides, int64_t dim) {
  int64_t off = 0;
  for (int64_t d = 0; d < dim; ++d)
    off += idx[d] * strides[d];
  return off;
}

// Generic elementwise copy for arbitrary strides on same device.
// Assumes src.numel() == dst.numel() and same sizes (but strides may differ).
template <typename scalar_t>
static void elementwise_copy_same_device(const Tensor &dst, const Tensor &src) {

  const int64_t dim = src.dim();
  const auto sizes = src.sizes();
  const auto src_strides = src.strides();
  const auto dst_strides = dst.strides();

  const scalar_t *src_ptr = src.data_ptr<scalar_t>();
  scalar_t *dst_ptr = dst.data_ptr<scalar_t>();

  const int64_t total = src.numel();
  if (total == 0)
    return;

  // Multi-index initialized to zeros.
  std::vector<int64_t> idx(dim, 0);

  for (int64_t i = 0; i < total; ++i) {
    // compute offsets (element offsets)
    int64_t s_off = 0;
    int64_t d_off = 0;
    for (int64_t d = 0; d < dim; ++d) {
      s_off += idx[d] * src_strides[d];
      d_off += idx[d] * dst_strides[d];
    }
    dst_ptr[d_off] = src_ptr[s_off];

    // increment multi-index
    for (int64_t d = dim - 1; d >= 0; --d) {
      idx[d] += 1;
      if (idx[d] < sizes[d])
        break;
      idx[d] = 0;
    }
  }
}


// 优化：检查是否可以用块拷贝代替逐元素拷贝
// 对于最后几个维度是连续的情况，可以用 memcpy 拷贝连续块
template <typename scalar_t>
static void optimized_strided_copy(const Tensor &dst, const Tensor &src) {
  const int64_t dim = src.dim();
  const auto sizes = src.sizes();
  const auto src_strides = src.strides();
  const auto dst_strides = dst.strides();

  const scalar_t *src_ptr = src.data_ptr<scalar_t>();
  scalar_t *dst_ptr = dst.data_ptr<scalar_t>();

  const int64_t total = src.numel();
  if (total == 0) return;

  // 找出从最后一个维度开始，有多少个维度是连续的
  // 连续的定义：stride[d] == sizes[d+1] * stride[d+1]
  int64_t contig_dims = 0;
  int64_t contig_size = 1;

  for (int64_t d = dim - 1; d >= 0; --d) {
    // 检查 src 和 dst 在这个维度是否都是连续的
    bool src_contig = (src_strides[d] == contig_size);
    bool dst_contig = (dst_strides[d] == contig_size);

    if (src_contig && dst_contig) {
      contig_dims++;
      contig_size *= sizes[d];
    } else {
      break;
    }
  }

  // 如果最后 contig_dims 个维度是连续的，可以用 memcpy 拷贝
  if (contig_dims > 0 && contig_size > 1) {
    // 外层循环的维度数
    int64_t outer_dims = dim - contig_dims;
    int64_t outer_total = total / contig_size;
    size_t copy_bytes = contig_size * sizeof(scalar_t);

    if (outer_dims == 0) {
      // 整个 tensor 都是连续的，直接 memcpy
      std::memcpy(dst_ptr, src_ptr, copy_bytes);
      return;
    }

    // 获取外层维度的 sizes 和 strides
    std::vector<int64_t> outer_sizes(sizes.begin(), sizes.begin() + outer_dims);
    std::vector<int64_t> outer_src_strides(src_strides.begin(), src_strides.begin() + outer_dims);
    std::vector<int64_t> outer_dst_strides(dst_strides.begin(), dst_strides.begin() + outer_dims);

    std::vector<int64_t> idx(outer_dims, 0);

    for (int64_t i = 0; i < outer_total; ++i) {
      // 计算外层偏移
      int64_t s_off = 0, d_off = 0;
      for (int64_t d = 0; d < outer_dims; ++d) {
        s_off += idx[d] * outer_src_strides[d];
        d_off += idx[d] * outer_dst_strides[d];
      }

      // 拷贝连续块
      std::memcpy(dst_ptr + d_off, src_ptr + s_off, copy_bytes);

      // 递增索引
      for (int64_t d = outer_dims - 1; d >= 0; --d) {
        idx[d] += 1;
        if (idx[d] < outer_sizes[d]) break;
        idx[d] = 0;
      }
    }
    return;
  }

  // Fallback: 逐元素拷贝
  elementwise_copy_same_device<scalar_t>(dst, src);
}

// Wrapper that dispatches by dtype
static void rpu_elementwise_copy_impl(const Tensor &dst, const Tensor &src) {
  TORCH_CHECK(dst.numel() == src.numel(),
              "rpu_elementwise_copy: numel mismatch");
  TORCH_CHECK(dst.sizes() == src.sizes(),
              "rpu_elementwise_copy: sizes mismatch (use as_strided/reshape as "
              "needed)");

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
      src.scalar_type(), "rpu_elementwise_copy",
      [&] { optimized_strided_copy<scalar_t>(dst, src); });
}

// Helper: Cross-device elementwise copy (RPU -> CPU or CPU -> RPU)
// Avoids creating temporary contiguous tensor on source device
template <typename SrcT, typename DstT>
static void cross_device_elementwise_copy(const Tensor &dst, const Tensor &src) {
  const int64_t dim = src.dim();
  const auto sizes = src.sizes();
  const auto src_strides = src.strides();
  const auto dst_strides = dst.strides();

  const SrcT *src_ptr = src.data_ptr<SrcT>();
  DstT *dst_ptr = dst.data_ptr<DstT>();

  const int64_t total = src.numel();
  if (total == 0) return;

  std::vector<int64_t> idx(dim, 0);

  for (int64_t i = 0; i < total; ++i) {
    int64_t s_off = 0, d_off = 0;
    for (int64_t d = 0; d < dim; ++d) {
      s_off += idx[d] * src_strides[d];
      d_off += idx[d] * dst_strides[d];
    }
    dst_ptr[d_off] = static_cast<DstT>(src_ptr[s_off]);

    // Increment multi-index
    for (int64_t d = dim - 1; d >= 0; --d) {
      idx[d] += 1;
      if (idx[d] < sizes[d]) break;
      idx[d] = 0;
    }
  }
}

// Unified copy implementation that supports:
// 1. Cross-device copy (CPU <-> RPU, RPU <-> RPU)
// 2. Cross-dtype copy (via CPU intermediate conversion)
// 3. Contiguous handling (optimized to avoid temporary allocations)
at::Tensor &rpu_copy_impl(at::Tensor &self, const at::Tensor &src,
                          bool non_blocking) {
  // Empty tensor - nothing to do
  if (src.numel() == 0) {
    return self;
  }

  // Check sizes match
  TORCH_CHECK(self.sizes() == src.sizes(),
              "rpu_copy: size mismatch, dst=", self.sizes(), " src=", src.sizes());

  // Early exit: same underlying data
  if (self.is_same(src)) {
    return self;
  }

  const bool same_dtype = (self.dtype() == src.dtype());
  const bool same_device = (self.device() == src.device());

  // Fast path: Same device + Same dtype
  if (same_device && same_dtype) {
    const bool is_rpu = (src.device().type() == c10::DeviceType::PrivateUse1);

    if (is_rpu) {
      // RPU -> RPU
      if (self.is_contiguous() && src.is_contiguous()) {
        rpu_ddr_flush_force_sized(src.data_ptr(), src.nbytes());
        std::memcpy(self.data_ptr(), src.data_ptr(), src.nbytes());
        rpu_ddr_flush_force_sized(self.data_ptr(), self.nbytes());
      } else {
        rpu_elementwise_copy_impl(self, src);
        const auto storage = self.storage();
        rpu_ddr_flush_force_sized(
            storage.data_ptr().get(), storage.nbytes());
      }
    } else {
      // CPU -> CPU (fallback to native)
      if (self.is_contiguous() && src.is_contiguous()) {
        std::memcpy(self.data_ptr(), src.data_ptr(), src.nbytes());
      } else {
        self.copy_(src, non_blocking);
      }
    }
    return self;
  }

  const bool is_src_rpu = (src.device().type() == c10::DeviceType::PrivateUse1);
  const bool is_dst_rpu = (self.device().type() == c10::DeviceType::PrivateUse1);
  const bool is_src_cpu = src.device().is_cpu();
  const bool is_dst_cpu = self.device().is_cpu();

  // Case 2: Cross-dtype (must go through CPU for conversion)
  if (!same_dtype) {
    Tensor src_cpu;

    if (is_src_cpu) {
      // Already on CPU
      src_cpu = src.is_contiguous() ? src : src.contiguous();
    } else {
      // RPU -> CPU: Optimize for non-contiguous case
      src_cpu = at::empty(src.sizes(),
                          at::TensorOptions()
                              .dtype(src.dtype())
                              .device(at::kCPU));

      if (src.is_contiguous()) {
        // Fast path: direct memcpy
        rpu_ddr_flush_force_sized(src.data_ptr(), src.nbytes());
        std::memcpy(src_cpu.data_ptr(), src.data_ptr(), src.nbytes());
      } else {
        // Optimized: cross-device elementwise copy (avoids RPU temp allocation)
        const auto storage = src.storage();
        rpu_ddr_flush_force_sized(
            storage.data_ptr().get(), storage.nbytes());
        AT_DISPATCH_ALL_TYPES_AND(
            at::ScalarType::Half,
            src.scalar_type(), "cross_device_copy_rpu_to_cpu",
            [&] {
              cross_device_elementwise_copy<scalar_t, scalar_t>(src_cpu, src);
            });
      }
    }

    // Convert dtype on CPU
    Tensor converted_cpu = src_cpu.to(self.dtype());

    // Copy to destination
    if (is_dst_cpu) {
      if (self.is_contiguous()) {
        std::memcpy(self.data_ptr(), converted_cpu.data_ptr(), converted_cpu.nbytes());
      } else {
        self.copy_(converted_cpu);
      }
    } else {
      // CPU -> RPU
      TORCH_CHECK(self.is_contiguous(),
                  "rpu_copy: dst must be contiguous for CPU->RPU dtype conversion");
      std::memcpy(self.data_ptr(), converted_cpu.data_ptr(), converted_cpu.nbytes());
      rpu_ddr_flush_force_sized(self.data_ptr(), self.nbytes());
    }
    return self;
  }

  // Case 3: Cross-device + Same dtype
  TORCH_CHECK(self.is_contiguous(),
              "rpu_copy: dst must be contiguous for cross-device copy");

  if (is_src_rpu && is_dst_cpu) {
    if (src.is_contiguous()) {
      rpu_ddr_flush_force_sized(src.data_ptr(), src.nbytes());
    } else {
      const auto storage = src.storage();
      rpu_ddr_flush_force_sized(
          storage.data_ptr().get(), storage.nbytes());
    }
  }

  if (src.is_contiguous()) {
    // Fast path: both contiguous
    std::memcpy(self.data_ptr(), src.data_ptr(), src.nbytes());
  } else {
    // Optimized: cross-device elementwise copy
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half,
        src.scalar_type(), "cross_device_copy",
        [&] {
          cross_device_elementwise_copy<scalar_t, scalar_t>(self, src);
        });
  }

  if (is_src_cpu && is_dst_rpu) {
    rpu_ddr_flush_force_sized(self.data_ptr(), self.nbytes());
  }

  return self;
}

// rpu_copy_: in-place copy src -> self. Signature matches aten::copy_
Tensor &rpu_copy_(Tensor &self, const Tensor &src, bool non_blocking = false) {
  // If src and self are same storage and same tensor, nothing to do.
  if (&self == &src)
    return self;

  // Use unified copy implementation (handles all cases)
  return rpu_copy_impl(self, src, non_blocking);
}

// contiguous implementation for rpu
Tensor rpu_contiguous(const Tensor &self, c10::MemoryFormat memory_format) {
  // 如果已经是 contiguous（按照给定 memory_format），直接返回 self
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  // allocate new contiguous tensor on same device/dtype/options
  Tensor out =
      at::empty(self.sizes(), self.options().memory_format(memory_format));
  // copy data src -> out using copy_
  out.copy_(self);
  return out;
}

  // Tensor rpu_contiguous(const Tensor &self, c10::MemoryFormat memory_format) {
  //   if (self.is_contiguous(memory_format)) {
  //     return self;
  //   }

  //   // 预分配连续的 RPU tensor
  //   Tensor out = at::empty(self.sizes(), self.options().memory_format(memory_format));

  //   // 零拷贝：让 CPU 来做 strided copy（更高效）
  //   auto self_cpu = rpu_to_cpu_zerocopy(self);
  //   auto out_cpu = rpu_to_cpu_zerocopy(out);
  //   out_cpu.copy_(self_cpu);

  //   return out;
  // }

// ------------------------ _to_copy (dtype conversion support) ------------------------
// 仅对 RPU -> CPU 启用零拷贝（RPU 内存一定在共享 DDR 上）
// CPU -> RPU 不能零拷贝，因为普通 CPU tensor 的内存可能不在共享 DDR 上
Tensor rpu_to_copy(const Tensor &self,
                   c10::optional<c10::ScalarType> dtype,
                   c10::optional<c10::Layout> layout,
                   c10::optional<c10::Device> device,
                   c10::optional<bool> pin_memory,
                   bool non_blocking,
                   c10::optional<c10::MemoryFormat> memory_format) {

  auto target_dtype = dtype.value_or(self.scalar_type());
  auto target_device = device.value_or(self.device());
  auto target_layout = layout.value_or(self.layout());

  const bool is_src_rpu = (self.device().type() == c10::DeviceType::PrivateUse1);
  const bool is_dst_cpu = target_device.is_cpu();
  const bool same_dtype = (target_dtype == self.scalar_type());

  // Zero-copy optimization: 仅 RPU -> CPU 且 dtype 相同
  // RPU 内存一定在共享 DDR 上，所以这是安全的
  //
  // Graph RECORDING 期 zero-copy 不安全 — 此刻 RPU kernels 还在
  // nodes_ 没跑,直接返回 DDR view 让 CPU 读到 stale 数据。Pi0.5 denoise loop
  // 内 adapter 触发本路径 → action_out_proj CPU fp32 读
  // stale fp16 → Euler `x_t += dt*v_t` 累积放大成 0.6 量级偏差。
  // 调 graph_aware_rpu_to_cpu_zerocopy_sync() 在 RECORDING + pending nodes
  // 时 sync_point() 把已录前缀 oneshot 落地,kernels 真正写完 DDR 再 zerocopy。
  if (same_dtype && self.is_contiguous() && is_src_rpu && is_dst_cpu) {
    graph_aware_rpu_to_cpu_zerocopy_sync();
    return rpu_to_cpu_zerocopy(self);
  }

  // Create result tensor with target dtype/device/layout
  Tensor result;
  if (target_device.type() == c10::DeviceType::PrivateUse1) {
    // Target is RPU
    result = rpu_empty_strided(self.sizes(),
                              compute_contiguous_strides(self.sizes()),
                              target_dtype,
                              target_layout,
                              target_device,
                              pin_memory.value_or(false));
  } else {
    // Target is CPU or other device
    result = at::empty(self.sizes(),
                      at::TensorOptions()
                          .dtype(target_dtype)
                          .layout(target_layout)
                          .device(target_device)
                          .pinned_memory(pin_memory.value_or(false)));
  }

  // Use unified copy (handles cross-device + cross-dtype + contiguous)
  rpu_copy_impl(result, self, non_blocking);

  return result;
}

// ------------------------ _copy_from_and_resize ------------------------
at::Tensor rpu__copy_from_and_resize(const at::Tensor &self,
                                     const at::Tensor &src) {
  auto tmp = rpu_empty_strided(src.sizes(), src.strides(), src.scalar_type(),
                               src.layout(), src.device(), false);

  // copy src -> tmp (handles CPU->RPU or RPU->RPU)
  return rpu_copy_(tmp, src, /*non_blocking=*/false);
}

// Note: add/mul/pow operations moved to rpu_eltwise_binary.cpp

// ------------------------ silu ------------------------
at::Tensor rpu_silu(const at::Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_silu: self must be on RPU");

  // Only Half is supported by RPU kernel - 零拷贝优化
  if (self.scalar_type() != at::kHalf) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::silu_out(cpu_result, cpu_self);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  }

  auto result = rpu_empty_strided(self.sizes(), self.strides(), self.scalar_type(),
                                  self.layout(), self.device(), false);

  rpu_launch_eltwise_unary_kernel(self.contiguous(), result, ValuOpType::SILU);

  return result;
}

// ------------------------ sqrt ------------------------
at::Tensor rpu_sqrt(const at::Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_sqrt: self must be on RPU");

  if (self.scalar_type() != at::kHalf) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::sqrt_out(cpu_result, cpu_self);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  }

  auto result = rpu_empty_strided(self.sizes(), self.strides(), self.scalar_type(),
                                  self.layout(), self.device(), false);

  rpu_launch_eltwise_unary_kernel(self.contiguous(), result, ValuOpType::SQRT);

  return result;
}

// ------------------------ exp ------------------------
at::Tensor rpu_exp(const at::Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_exp: self must be on RPU");

  if (self.scalar_type() != at::kHalf) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::exp_out(cpu_result, cpu_self);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  }

  auto result = rpu_empty_strided(self.sizes(), self.strides(), self.scalar_type(),
                                  self.layout(), self.device(), false);

  rpu_launch_eltwise_unary_kernel(self.contiguous(), result, ValuOpType::EXP);

  return result;
}

// ------------------------ log ------------------------
at::Tensor rpu_log(const at::Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_log: self must be on RPU");

  if (self.scalar_type() != at::kHalf) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::log_out(cpu_result, cpu_self);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  }

  auto result = rpu_empty_strided(self.sizes(), self.strides(), self.scalar_type(),
                                  self.layout(), self.device(), false);

  rpu_launch_eltwise_unary_kernel(self.contiguous(), result, ValuOpType::LOG);

  return result;
}

// ------------------------ sin ------------------------
at::Tensor rpu_sin(const at::Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_sin: self must be on RPU");

  if (self.scalar_type() != at::kHalf) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::sin_out(cpu_result, cpu_self);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  }

  auto input = self.contiguous();
  auto result = rpu_empty_strided(input.sizes(), input.strides(), input.scalar_type(),
                                  input.layout(), input.device(), false);

  rpu_launch_eltwise_unary_kernel(input, result, ValuOpType::SIN);

  return result;
}

// ------------------------ cos ------------------------
at::Tensor rpu_cos(const at::Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_cos: self must be on RPU");

  if (self.scalar_type() != at::kHalf) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::cos_out(cpu_result, cpu_self);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  }

  auto input = self.contiguous();
  auto result = rpu_empty_strided(input.sizes(), input.strides(), input.scalar_type(),
                                  input.layout(), input.device(), false);

  rpu_launch_eltwise_unary_kernel(input, result, ValuOpType::COS);

  return result;
}

// ------------------------ rsqrt ------------------------
at::Tensor rpu_rsqrt(const at::Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_rsqrt: self must be on RPU");

  // Only Half is supported by RPU kernel - 零拷贝优化
  if (self.scalar_type() != at::kHalf) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::rsqrt_out(cpu_result, cpu_self);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  }

  auto input = self.contiguous();
  auto result = rpu_empty_strided(input.sizes(), input.strides(), input.scalar_type(),
                                  input.layout(), input.device(), false);

  rpu_launch_eltwise_unary_kernel(input, result, ValuOpType::SQRT_RECIPROCAL);

  return result;
}

at::Tensor rpu_tanh(const Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "self not on RPU");
  auto result =
      rpu_empty_strided(self.sizes(), self.strides(), self.scalar_type(),
                        self.layout(), self.device(), false);

  rpu_launch_eltwise_unary_kernel(self, result, ValuOpType::TANH);

  return result;
}

// 零拷贝优化
Tensor &rpu_normal_(Tensor &self, double mean, double std,
                    optional<Generator> gen) {
  auto cpu_tensor = rpu_to_cpu_zerocopy(self);
  cpu_tensor.normal_(mean, std, gen);
  rpu_ddr_flush_force(self.data_ptr());
  // 无需 copy_ 回来，因为共享内存
  return self;
}

// 零拷贝优化
Tensor rpu_matmul(const Tensor &tensor1, const Tensor &tensor2) {
  auto cpu_tensor1 = rpu_to_cpu_zerocopy(tensor1);
  auto cpu_tensor2 = rpu_to_cpu_zerocopy(tensor2);
  // matmul 输出形状可能变化，先计算 CPU 结果
  auto cpu_result = at::matmul(cpu_tensor1, cpu_tensor2);
  // 分配 RPU 输出并复制
  auto result = rpu_empty_strided(cpu_result.sizes(), cpu_result.strides(),
                                  cpu_result.scalar_type(), cpu_result.layout(),
                                  tensor1.device(), false);
  auto cpu_result_view = rpu_to_cpu_zerocopy(result);
  cpu_result_view.copy_(cpu_result);
  rpu_ddr_flush_force(result.data_ptr());
  return result;
}

// 零拷贝优化
at::Tensor rpu_bmm_fallback(const at::Tensor &self, const at::Tensor &mat2) {
  auto a_cpu = rpu_to_cpu_zerocopy(self);
  auto b_cpu = rpu_to_cpu_zerocopy(mat2);
  // 预分配输出: [B, M, N] = [B, M, K] @ [B, K, N]
  auto result = at::empty({self.size(0), self.size(1), mat2.size(2)}, self.options());
  auto c_cpu = rpu_to_cpu_zerocopy(result);
  at::bmm_out(c_cpu, a_cpu, b_cpu);
  rpu_ddr_flush_force(result.data_ptr());
  return result;
}

at::Tensor rpu_softmax(const at::Tensor &self, int64_t dim,
                       c10::optional<at::ScalarType> dtype_opt) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_softmax: tensor must be on rpu");

  // The native kernel is a narrow [1,N,C] FP16 fast path. It ignores B, uses
  // 16-element DMA blocks, and allocates both input and output in core-0 SPM;
  // route every shape outside that proven envelope through the CPU fallback.
  const bool is_last_dim = self.dim() > 0
      && (dim == self.dim() - 1 || dim == -1);
  const bool keeps_fp16 = !dtype_opt.has_value()
      || dtype_opt.value() == at::kHalf;
  const bool shape_supported = self.dim() == 3
      && self.size(0) == 1
      && self.size(1) > 0
      && self.size(1) <= UINT16_MAX
      && self.size(2) > 0
      && self.size(2) % 16 == 0
      && self.size(2) <= 2048;
  const uint64_t spm_bytes = shape_supported
      ? static_cast<uint64_t>(self.size(1))
            * static_cast<uint64_t>(self.size(2)) * sizeof(c10::Half) * 2
      : 0;

  if (!is_last_dim || !keeps_fp16 || !shape_supported
      || self.scalar_type() != at::kHalf
      || spm_bytes > SpmAllocator::SPM_PLANNING_BUDGET) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_result = at::softmax(cpu_self, dim, dtype_opt);
    return cpu_result.to(self.device());
  }

  Tensor a_contig = self.contiguous();
  auto batch = a_contig.size(0);
  auto n = a_contig.size(1);
  auto c = a_contig.size(2);
  auto result = at::empty({batch, n, c}, self.options());

  rpu_launch_softmax_c16_kernel(a_contig, result);
  return result;
}

at::Tensor rpu_bmm(const at::Tensor &self, const at::Tensor &mat2) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "self must be on rpu");
  TORCH_CHECK(mat2.device().type() == c10::DeviceType::PrivateUse1,
              "mat2 must be on rpu");
  TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3,
              "rpu_bmm: bmm expects 3D tensors");
  TORCH_CHECK(self.options().dtype() == at::kHalf,
              "rpu_bmm: only half supported");

  Tensor a_contig = self.contiguous();
  Tensor b_contig = mat2.contiguous();

  auto batch = a_contig.size(0);
  auto m = a_contig.size(1);
  auto k = a_contig.size(2);
  auto n = b_contig.size(2);
  TORCH_CHECK(k == b_contig.size(1), "matmul dim mismatch");

  auto result = at::empty({batch, m, n}, self.options());
  c10::Half *out = result.data_ptr<c10::Half>();

  rpu_launch_bmm_kernel(a_contig, b_contig, result);

  return result;
}

// Unified layout SDPA: accepts [batch, seq_len, num_heads, head_dim] format (4D unified)
// Different from standard PyTorch [batch, num_heads, seq_len, head_dim]
// Only difference: dimensions 1 and 2 are swapped (seq_len and heads)
//
// This entry point is CPU-only because its device implementation does not meet
// the required full-sequence numerical and determinism contract. Fused decoders
// use rpu_launch_sdpa_spm_unified_kernel_v2
// (KernelId::SDPA_FLASH_ATTN_SPM), which is a separate path.
// The CPU implementation honours attn_mask, dropout_p and enable_gqa.
namespace {

// Shared CPU body. `unified` says how to READ the caller's tensors; it is the
// entry point's contract, never a guess about the shapes (see below).
Tensor sdpa_on_cpu(
    const Tensor &query, const Tensor &key, const Tensor &value,
    const optional<Tensor> &attn_mask, double dropout_p, bool is_causal,
    std::optional<double> scale, bool enable_gqa, bool unified
) {
  auto to_cpu_f32 = [](const Tensor &t) {
    return rpu_to_cpu_zerocopy(t).to(at::kFloat);
  };

  c10::optional<at::Tensor> mask_cpu;
  if (attn_mask.has_value() && attn_mask->defined()) {
    mask_cpu = to_cpu_f32(*attn_mask);
  }

  if (!unified) {
    auto out_cpu = at::scaled_dot_product_attention(
        to_cpu_f32(query), to_cpu_f32(key), to_cpu_f32(value), mask_cpu,
        dropout_p, is_causal, scale, enable_gqa);
    return out_cpu.to(query.scalar_type()).to(query.device());
  }

  // Unified [batch, seq, heads, dim] -> PyTorch [batch, heads, seq, dim].
  auto out_cpu = at::scaled_dot_product_attention(
      to_cpu_f32(query).transpose(1, 2).contiguous(),
      to_cpu_f32(key).transpose(1, 2).contiguous(),
      to_cpu_f32(value).transpose(1, 2).contiguous(),
      mask_cpu, dropout_p, is_causal, scale, enable_gqa);

  return out_cpu.transpose(1, 2).contiguous()
      .to(query.scalar_type()).to(query.device());
}

}  // namespace

// aten::scaled_dot_product_attention — PyTorch layout [batch, heads, seq, dim],
// which is what the ATen schema means, unconditionally.
//
// Layout is fixed by the ATen schema and must not be inferred from tensor
// dimensions. The unified-layout entry point below owns a separate contract.
Tensor rpu_scaled_dot_product_attention(
    const Tensor &query, const Tensor &key, const Tensor &value,
    const optional<Tensor> &attn_mask, double dropout_p, bool is_causal,
    std::optional<double> scale,
    bool enable_gqa
) {
  return sdpa_on_cpu(query, key, value, attn_mask, dropout_p, is_causal, scale,
                     enable_gqa, /*unified=*/false);
}

Tensor rpu_scaled_dot_product_attention_unified(
    const Tensor &query, const Tensor &key, const Tensor &value,
    const optional<Tensor> &attn_mask, double dropout_p, bool is_causal,
    std::optional<double> scale,
    bool enable_gqa
) {
  TORCH_CHECK(query.dim() == 4,
              "rpu_scaled_dot_product_attention_unified expects 4D input "
              "[batch, seq_len, heads, head_dim], got ", query.dim(), "D");

  // Unified layout, unconditionally — it is this op's name and its contract.
  return sdpa_on_cpu(query, key, value, attn_mask, dropout_p, is_causal, scale,
                     enable_gqa, /*unified=*/true);
}

Tensor rpu_layernorm(const at::Tensor &input, c10::IntArrayRef normalized_shape,
                     const c10::optional<at::Tensor> &weight, // gamma
                     const c10::optional<at::Tensor> &bias,   // beta
                     double eps, bool cudnn_enable = false) {
  // auto rpu_w = weight.value_or(at::ones(normalized_shape, input.options()));

  // auto rpu_b = bias.value_or(at::zeros(normalized_shape, input.options()));

  auto rpu_w = weight.value();

  auto rpu_b = bias.value();

  auto result =
      rpu_empty_strided(input.sizes(), input.strides(), input.scalar_type(),
                        input.layout(), input.device(), false);
  rpu_launch_layernorm_kernel(input, result, normalized_shape, rpu_w, rpu_b,
                              eps);
  return result;
}

// 零拷贝优化
Tensor rpu_dropout(const Tensor &input, double p, bool training) {
  if (input.device().type() != at::DeviceType::PrivateUse1) {
    return at::native::dropout(input, p, training);
  }

  // 如果不是训练模式或 p=0，直接返回输入的副本
  if (!training || p == 0) {
    return input.clone();
  }

  auto cpu_in = rpu_to_cpu_zerocopy(input);
  // dropout 会创建新的 tensor，无法预分配
  auto cpu_out = at::native::dropout(cpu_in, p, training);
  return cpu_out.to(input.device());
}

std::vector<at::Tensor> rpu_broadcast(c10::ArrayRef<at::Tensor> tensors) {
    if (tensors.size() < 2) {
        throw std::runtime_error("Need at least 2 tensors to broadcast");
    }

    // 1️⃣ 计算最终 broadcast shape
    std::vector<int64_t> out_shape(tensors[0].sizes().begin(), tensors[0].sizes().end());
    for (size_t i = 1; i < tensors.size(); ++i) {
        const auto& sizes = tensors[i].sizes();
        size_t ndim = std::max(out_shape.size(), sizes.size());
        std::vector<int64_t> new_shape(ndim);

        for (size_t d = 0; d < ndim; ++d) {
            int64_t dim1 = (d < ndim - out_shape.size()) ? 1 : out_shape[d - (ndim - out_shape.size())];
            int64_t dim2 = (d < ndim - sizes.size()) ? 1 : sizes[d - (ndim - sizes.size())];
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                throw std::runtime_error("Shapes cannot be broadcasted at dim " + std::to_string(d));
            new_shape[d] = std::max(dim1, dim2);
        }
        out_shape = std::move(new_shape);
    }

    // 2️⃣ 对每个 tensor 做 expand（view，无拷贝）
    std::vector<at::Tensor> result;
    result.reserve(tensors.size());
    for (const auto& t : tensors) {
        // 如果已经 match out_shape，就直接复用 tensor
        if (t.sizes() == out_shape) {
            result.push_back(t);
        } else {
            // 调用 rpu_expand 或 at::Tensor::expand
            result.push_back(t.expand(out_shape));
        }
    }

    return result;
}

// Tensor rpu_linear(const Tensor &input, const Tensor &weight,
//                   const c10::optional<Tensor> &bias_opt) {
//   // // fallback 到 CPU
//   // auto cpu_input = input.to(at::kCPU);
//   // auto cpu_weight = weight.to(at::kCPU);
//   // Tensor cpu_bias;
//   // if (bias_opt.has_value()) {
//   //   cpu_bias = bias_opt->to(at::kCPU);
//   // }

//   // Tensor out_cpu;
//   // if (bias_opt.has_value()) {
//   //   out_cpu = at::linear(cpu_input, cpu_weight, cpu_bias);
//   // } else {
//   //   out_cpu = at::linear(cpu_input, cpu_weight, Tensor());
//   // }

//   // // 返回到原设备
//   // return out_cpu.to(input.device());

//   TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
//               "input must be on rpu");
//   TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1,
//               "weight must be on rpu");
//   TORCH_CHECK(input.options().dtype() == at::kHalf,
//               "rpu_linear: only half supported");
//   std::cout << input.sizes() << std::endl;
//   std::cout << weight.sizes() << std::endl;
//   Tensor input_contig = input.contiguous();
//   // auto weightT = rpu_transpose(weight, 1, 0);
//   // // Tensor weight_contig = weightT.contiguous().to(at::kCPU).unsqueeze(0).to(input.device());
//   // Tensor weight_contig = weightT.contiguous();

//   int64_t m = 1;
//   for (size_t i = 0; i + 1 < input_contig.dim(); ++i) {
//     m *= input_contig.size(i);
//   }
//   auto n = weight.size(0);
//   auto k = weight.size(1);

//   bool has_bias = false;
//   Tensor bias;
//   if (bias_opt.has_value()) {
//     has_bias = true;
//     bias = bias_opt.value();
//   }

//   auto result = at::empty({m, n}, input.options());
//   rpu_launch_linear_kernel(input, weight, result, bias, has_bias);
//   return result;
//   // auto bmm_res = at::empty({1, input.size(1), weight.size(0)}, input.options());

//   // bmm_res = rpu_bmm(input_contig, weight_contig);
//   // std::cout << bmm_res.to(at::kCPU)
//   //         .reshape({input.size(1), weight.size(0)})
//   //         .to(input.device()).sizes() << std::endl;
//   // std::cout << bias.to(at::kCPU).reshape({1, weight.size(0)}).to(input.device()).sizes() << std::endl;
//   // auto result =
//   //     bmm_res.to(at::kCPU)
//   //         .reshape({input.size(1), weight.size(0)})
//   //          +
//   //     broadcast_to(bias.to(at::kCPU).reshape({1, weight.size(0)}), {input.size(1), weight.size(0)});
//   // result = result.to(input.device());
//   // const c10::Scalar alpha = 1.0;
//   // auto result = at::empty({bias.size(0)}, input.options());
//   // auto bias_broadcast = at::
//   // result = rpu_add(bmm_res, bias, alpha);
//   // bool has_bias = false;
//   // if (bias_opt.has_value()) {
//   //   has_bias = true;
//   //   bias = bias_opt.value();
//   // }

//   // int64_t m = 1;
//   // for (size_t i = 0; i + 1 < input_contig.dim(); ++i) {
//   //   m *= input_contig.size(i);
//   // }
//   // auto m = input.size(0);
//   // auto n = weight.size(0) * weight.size(3);
//   // auto result = at::empty({m, n}, input.options());
//   // rpu_launch_linear_kernel(input, weight, result, bias, has_bias);
//   // return result;
// }

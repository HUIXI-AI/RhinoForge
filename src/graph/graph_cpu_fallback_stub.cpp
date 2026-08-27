// graph_cpu_fallback_stub.cpp — graph-aware zero-copy CPU fallback.
// graph_runtime_capture.cpp / graph_runtime_execute.cpp 调用
// run_cpu_fallback_zerocopy。逻辑分三段:
//   Step 1: 把 stack 上的 RPU tensor / TensorList / OptionalTensorList 改成 CPU
//           zero-copy view (共用 RPU DDR storage);view 进 cpu_view_lifeline +
//           ZeroCopyHolderGuard 镜像登记,防 .out variant redispatch 期间
//           storage_holder 被提前 deleter 触发 use-after-free
//           防止 .out variant redispatch 期间提前释放 storage_holder。
//   Step 2: op.redispatchBoxed(CPU dispatch_key, stack) — 在 CPU 上跑真 op。
//   Step 3:
//     alias=true (Tier1 .out variant): return slot 替换回原 RPU .out= input tensor
//                (CPU op 已通过 zero-copy view 写穿到原 RPU storage)。
//     alias=false (Tier2 functional): return CPU tensor `.to(target_device)` 拷回 RPU。
//
// RECORDING/REPLAYING 期 dispatcher 调到的 CPU fallback op 会走 HostCallback
// 节点流程,不再触发 sync_point() → graph 内部支持 CPU fallback。

#include "graph/graph_infra.h"
#include "graph/graph_runtime.h"
#include "rpu_helpers.h"   // rpu_to_cpu_zerocopy

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <vector>

void run_cpu_fallback_zerocopy(const c10::OperatorHandle& op,
                                torch::jit::Stack* stack,
                                bool alias_output_args) {
    const auto& schema = op.schema();
    const auto num_arguments = schema.arguments().size();
    const auto stack_start = stack->size() - num_arguments;

    // 保存原始 RPU tensor (alias=true 时回填 return slot 用) + 原 device
    // (alias=false 时 .to() 拷回用)。
    std::vector<at::Tensor> rpu_inputs(num_arguments);
    std::vector<c10::optional<c10::Device>> original_devices(num_arguments);
    std::vector<void*> rpu_backed_writes;

    // zero-copy CPU view 必须持有到 redispatch + return 处理结束,否则
    // boxed dispatcher pop arg 阶段会提前触发 view storage_holder deleter,而
    // .out variant 的 CPU op 又把同一 tensor push 回 stack 当 return → stack
    // IValue 引用已 delete 的 storage_holder = use-after-free。
    std::vector<at::Tensor> cpu_view_lifeline;
    cpu_view_lifeline.reserve(num_arguments);

    // Owner-tree 镜像: guard ctor 记当前 size,dtor 退栈截回。嵌套
    // fallback 安全。RpuKernelGraph::active() 在无活动 graph scope 时返回
    // fallback_graph_,镜像登记仍走 thread-local 实例,guard 退栈清理。
    ZeroCopyHolderGuard zc_guard(RpuKernelGraph::active());
    auto register_zc_mirror = [&](const at::Tensor& view) {
        RpuKernelGraph::active().register_zero_copy_holder(view);
    };
    auto track_write_arg = [&](size_t arg_idx, const at::Tensor& tensor) {
        if (is_mutable_output_arg(schema, arg_idx)) {
            rpu_backed_writes.push_back(tensor.data_ptr());
        }
    };

    // Step 1: RPU tensor / TensorList / OptionalTensorList → CPU view
    for (size_t i = 0; i < num_arguments; ++i) {
        auto& ivalue = (*stack)[stack_start + i];
        if (ivalue.isTensor()) {
            auto tensor = ivalue.toTensor();
            if (tensor.defined() &&
                tensor.device().type() == c10::DeviceType::PrivateUse1) {
                rpu_inputs[i] = tensor;
                original_devices[i] = tensor.device();
                track_write_arg(i, tensor);
                auto view = rpu_to_cpu_zerocopy_no_graph_sync(tensor);
                cpu_view_lifeline.push_back(view);
                register_zc_mirror(view);
                (*stack)[stack_start + i] = std::move(view);
            }
        } else if (ivalue.isTensorList()) {
            auto tensor_list = ivalue.toTensorList();
            std::vector<at::Tensor> cpu_list;
            bool has_rpu = false;
            for (size_t j = 0; j < tensor_list.size(); ++j) {
                at::Tensor t = tensor_list[j];
                if (t.defined() &&
                    t.device().type() == c10::DeviceType::PrivateUse1) {
                    has_rpu = true;
                    track_write_arg(i, t);
                    auto view = rpu_to_cpu_zerocopy_no_graph_sync(t);
                    cpu_view_lifeline.push_back(view);
                    register_zc_mirror(view);
                    cpu_list.push_back(std::move(view));
                } else {
                    cpu_list.push_back(t);
                }
            }
            if (has_rpu) {
                original_devices[i] =
                    c10::Device(c10::DeviceType::PrivateUse1, 0);
                (*stack)[stack_start + i] = c10::List<at::Tensor>(cpu_list);
            }
        } else if (ivalue.isOptionalTensorList()) {
            auto opt_list = ivalue.toOptionalTensorList();
            std::vector<c10::optional<at::Tensor>> cpu_list;
            bool has_rpu = false;
            for (size_t j = 0; j < opt_list.size(); ++j) {
                c10::optional<at::Tensor> opt_t = opt_list[j];
                if (opt_t.has_value() && opt_t->defined() &&
                    opt_t->device().type() == c10::DeviceType::PrivateUse1) {
                    has_rpu = true;
                    track_write_arg(i, *opt_t);
                    auto view = rpu_to_cpu_zerocopy_no_graph_sync(*opt_t);
                    cpu_view_lifeline.push_back(view);
                    register_zc_mirror(view);
                    cpu_list.push_back(std::move(view));
                } else {
                    cpu_list.push_back(opt_t);
                }
            }
            if (has_rpu) {
                original_devices[i] =
                    c10::Device(c10::DeviceType::PrivateUse1, 0);
                (*stack)[stack_start + i] =
                    c10::List<c10::optional<at::Tensor>>(cpu_list);
            }
        }
    }

    // Step 2: redispatch 到 CPU 实现
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);

    // Write-back flush. The CPU op writes `.out=` results through zero-copy
    // views into cached RPU-backed host DDR. Flush only schema-marked write
    // arguments so the next device read observes those writes; read-only
    // fallbacks never enter this loop.
    for (void* ptr : rpu_backed_writes) {
        rpu_ddr_flush_force(ptr);
    }

    const auto num_returns = schema.returns().size();
    const auto ret_start = stack->size() - num_returns;

    // Step 3: return 处理 — alias=true (Tier1 .out variant) 跟 alias=false (Tier2
    // functional) 走两条路。
    if (alias_output_args) {
        // Tier1 .out variant: returns 与 output_arg_indices schema 顺序一一对应。
        // 把 stack 上每个 return slot 替换回原 RPU .out= input arg —— CPU op 已
        // 经通过 zero-copy view 写穿到该 RPU storage,直接 return 原 tensor 即
        // 跨 replay 稳定 dev_addr。如果对 zero-copy CPU view 做 .to(rpu),
        // storage allocator=nullptr 会导致未定义行为。
        auto out_indices = collect_output_arg_indices(schema);
        TORCH_CHECK(out_indices.size() == num_returns,
                    "run_cpu_fallback_zerocopy(alias=true): #out_args=",
                    out_indices.size(), " != #returns=", num_returns,
                    " op=", schema.name());
        for (size_t r = 0; r < num_returns; ++r) {
            size_t arg_idx = out_indices[r];
            TORCH_CHECK(arg_idx < num_arguments &&
                            rpu_inputs[arg_idx].defined(),
                        "run_cpu_fallback_zerocopy(alias=true): out arg ",
                        arg_idx, " was not RPU tensor at input; op=",
                        schema.name());
            (*stack)[ret_start + r] = rpu_inputs[arg_idx];
        }
    } else {
        // Tier2 functional: CPU op 内部自己 alloc fresh CPU tensor (allocator
        // 非 nullptr),.to(target_device) 拷回 RPU 安全。
        c10::optional<c10::Device> target_device;
        for (const auto& dev : original_devices) {
            if (dev.has_value()) {
                target_device = dev;
                break;
            }
        }
        if (target_device.has_value()) {
            for (size_t i = 0; i < num_returns; ++i) {
                auto& ret = (*stack)[ret_start + i];
                if (ret.isTensor()) {
                    auto tensor = ret.toTensor();
                    if (tensor.defined() && tensor.device().is_cpu()) {
                        (*stack)[ret_start + i] = tensor.to(*target_device);
                    }
                } else if (ret.isTensorList()) {
                    auto tensor_list = ret.toTensorList();
                    std::vector<at::Tensor> rpu_list;
                    for (size_t j = 0; j < tensor_list.size(); ++j) {
                        at::Tensor t = tensor_list[j];
                        if (t.defined() && t.device().is_cpu()) {
                            rpu_list.push_back(t.to(*target_device));
                        } else {
                            rpu_list.push_back(t);
                        }
                    }
                    (*stack)[ret_start + i] = c10::List<at::Tensor>(rpu_list);
                }
            }
        }
    }
}

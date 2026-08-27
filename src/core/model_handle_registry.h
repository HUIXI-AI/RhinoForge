#pragma once
// =============================================================================
// ModelHandleRegistry<T> — Thread-safe handle-based instance registry
//
// 每个 Python 侧的 model 对应一个独立的 C++ 实例, 通过 int64_t handle 索引。
// 同一个模板类可被不同 model 类型 (Qwen3Model / GemmaModel / ...) 实例化,
// 每种 T 产生独立的 Meyers singleton → 天然隔离。
//
// 用法:
//   using Qwen3Registry = ModelHandleRegistry<Qwen3Model>;
//   int64_t h = Qwen3Registry::create();
//   auto* m = Qwen3Registry::get(h, "rpu_qwen3");
//   Qwen3Registry::destroy(h, "rpu_qwen3_destroy");
//
// ── Thread-safety contract ────────────────────────────────────────────────
// create/destroy/get 内部用 mutex 保护 map 操作。
// get() 查表后释放锁并返回裸指针, 意味着调用方必须保证:
//   **create / destroy / set_* / forward 不能与 destroy 并发**
//   (即同一线程串行, 或 Python 场景下 GIL 保证单 op 执行)
//
// 典型 Python 推理场景 (GIL + 单模型串行推理) 下 raw pointer 是安全的。
//
// 若未来需并发 destroy (GC 线程 vs 推理线程), 必须改为 shared_ptr 存储 +
// get() 返回 shared_ptr 的强引用, 让 destroy 仅 erase map 槽但实例存活到
// forward 结束。
//
// ── Handle identity ───────────────────────────────────────────────────────
// 每个 T 的 next_handle_ 单调递增且 handle 不复用,因此 destroy+create 不会
// 把旧 Python handle 指向新实例。GraphCache admission/lifecycle 由 Python
// adapter 独立持有,不由 registry 或 FusedModelBase destructor 管理。
// =============================================================================

#include <ATen/ATen.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

template <typename T>
class ModelHandleRegistry {
public:
    // 创建新实例, 返回其 handle。
    static int64_t create() {
        auto& r = instance();
        std::lock_guard<std::mutex> lock(r.mutex_);
        int64_t handle = r.next_handle_++;
        r.instances_[handle] = std::make_unique<T>();
        return handle;
    }

    // 按 handle 销毁实例并触发 ~T()。adapter 负责先释放自己的 GraphCache。
    // ctx: 调用方 tag, 出现在错误消息中 (例如 "rpu_qwen3_destroy")。
    static void destroy(int64_t handle, const char* ctx = "ModelHandleRegistry") {
        auto& r = instance();
        std::lock_guard<std::mutex> lock(r.mutex_);
        auto it = r.instances_.find(handle);
        TORCH_CHECK(it != r.instances_.end(),
                    ctx, ": handle ", handle, " not found");
        r.instances_.erase(it);
    }

    // 按 handle 取裸指针。调用方保证不与 destroy 并发。
    // ctx: 调用方 tag, 出现在错误消息中 (例如 "rpu_qwen3")。
    static T* get(int64_t handle, const char* ctx = "ModelHandleRegistry") {
        auto& r = instance();
        std::lock_guard<std::mutex> lock(r.mutex_);
        auto it = r.instances_.find(handle);
        TORCH_CHECK(it != r.instances_.end(),
                    ctx, ": invalid handle ", handle,
                    ". Handle must be obtained from create() and "
                    "not yet destroyed.");
        return it->second.get();
    }

private:
    ModelHandleRegistry() = default;
    ModelHandleRegistry(const ModelHandleRegistry&) = delete;
    ModelHandleRegistry& operator=(const ModelHandleRegistry&) = delete;

    static ModelHandleRegistry& instance() {
        static ModelHandleRegistry registry;
        return registry;
    }

    std::unordered_map<int64_t, std::unique_ptr<T>> instances_;
    std::mutex mutex_;
    int64_t next_handle_ = 1;
};

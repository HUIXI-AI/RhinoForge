#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <utility>

// Process-wide policy for PrivateUse1 tensor storage.  The choice is cold:
// either an explicit declaration or the first non-empty tensor allocation
// freezes it for the rest of the process.  Keeping this state independent of
// model adapters prevents a live workload from changing how another thread's
// storage is allocated.
class RpuTensorAllocatorPolicy final {
 public:
  enum class ClaimResult : uint8_t {
    kApplied,
    kIdempotent,
    kConflict,
  };

  struct Snapshot {
    bool frozen;
    bool caching;
  };

  ClaimResult claim(bool caching) noexcept {
    const State desired = caching ? State::kCaching : State::kDirect;
    State expected = State::kUnclaimed;
    if (state_.compare_exchange_strong(
            expected, desired, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      return ClaimResult::kApplied;
    }
    return expected == desired ? ClaimResult::kIdempotent
                               : ClaimResult::kConflict;
  }

  bool freeze_for_allocation() noexcept {
    State expected = State::kUnclaimed;
    if (state_.compare_exchange_strong(
            expected, State::kDirect, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      return false;
    }
    return expected == State::kCaching;
  }

  Snapshot snapshot() const noexcept {
    const State state = state_.load(std::memory_order_acquire);
    return {state != State::kUnclaimed, state == State::kCaching};
  }

 private:
  enum class State : uint8_t {
    kUnclaimed,
    kDirect,
    kCaching,
  };

  std::atomic<State> state_{State::kUnclaimed};
};

// Process-wide lifetime gate for PyTorch-owned DDR storage.  A non-empty
// allocation holds this gate through the complete direct or caching allocator
// transaction.  begin_shutdown() therefore waits for every in-flight
// transaction before permanently rejecting new allocations.
//
// Direct tensor deleters use the same gate only around the SDK free.  They
// remove their logical/segment registry entries before entering it, so a late
// deleter still retires local identity after shutdown while leaving physical
// teardown to RpuDdrShutdown.  Caching allocator teardown intentionally does
// not use this gate: rpu_shutdown owns and drains its idle segments after the
// gate has closed.
class RpuTensorDdrLifecycle final {
 public:
  template <typename Operation>
  decltype(auto) run_allocation(Operation&& operation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_started_) {
      throw std::runtime_error("RPU tensor DDR allocator is shut down");
    }
    return std::forward<Operation>(operation)();
  }

  template <typename Operation>
  bool run_free_if_live(Operation&& operation) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_started_) {
        return false;
      }
      std::forward<Operation>(operation)();
      return true;
    } catch (...) {
      // Tensor deleters must not throw during Python/C++ object destruction.
      return false;
    }
  }

  void begin_shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_started_ = true;
  }

  bool shutdown_started() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_started_;
  }

 private:
  mutable std::mutex mutex_;
  bool shutdown_started_{false};
};

inline RpuTensorDdrLifecycle& rpu_tensor_ddr_lifecycle() {
  // PyTorch retains the allocator and its deleter for process lifetime.  Keep
  // the gate alive through static destruction so late DataPtr cleanup cannot
  // access an already-destroyed mutex.
  static auto* const lifecycle = new RpuTensorDdrLifecycle();
  return *lifecycle;
}

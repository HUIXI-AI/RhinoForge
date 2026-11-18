// A small private pool for host work that must not hold the GIL or pay a thread
// start per request.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace rpu_workers {

// Work that any number of threads may run at once.
class Job {
 public:
  virtual ~Job() = default;
  virtual void drain() = 0;
};

// Deliberately leaked: parked threads outlive static destruction at exit.
class Pool {
 public:
  static Pool& instance() {
    static Pool* pool = new Pool();
    return *pool;
  }

  int size() const { return size_; }

  // At most ``helpers`` workers pick up the job; a late one finds it exhausted.
  void start(std::shared_ptr<Job> job, int helpers) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_.push_back({std::move(job), std::min(helpers, size_)});
    }
    // Woken workers wake the next one, so the request thread pays one wake.
    wake_.notify_one();
  }

 private:
  struct Entry {
    std::shared_ptr<Job> job;
    int helpers;
  };

  Pool() : size_(std::min(12, std::max(1, static_cast<int>(std::thread::hardware_concurrency())))) {
    for (int index = 0; index < size_; ++index) std::thread([this] { run(); }).detach();
  }

  void run() {
    for (;;) {
      std::shared_ptr<Job> job;
      bool more;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [this] { return !pending_.empty(); });
        Entry& front = pending_.front();
        job = front.job;
        if (--front.helpers <= 0) pending_.pop_front();
        more = !pending_.empty();
      }
      if (more) wake_.notify_one();
      job->drain();
    }
  }

  const int size_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Entry> pending_;
};

namespace detail {

class Steps final : public Job {
 public:
  Steps(size_t count, const std::function<void(size_t)>& step)
      : count_(count), step_(step), left_(count) {}

  // Past the last index nothing touches ``step``, which only the caller owns.
  void drain() override {
    for (size_t index; (index = next_.fetch_add(1, std::memory_order_relaxed)) < count_;) {
      try {
        step_(index);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_) failure_ = std::current_exception();
      }
      if (left_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        done_.notify_all();
      }
    }
  }

  void join() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return finished_; });
    if (failure_) std::rethrow_exception(failure_);
  }

 private:
  const size_t count_;
  const std::function<void(size_t)>& step_;
  std::atomic<size_t> next_{0};
  std::atomic<size_t> left_;
  std::mutex mutex_;
  std::condition_variable done_;
  bool finished_ = false;
  std::exception_ptr failure_;
};

}  // namespace detail

// step(0..count-1) on the caller and up to ``threads - 1`` workers. Every step has
// finished on return, and the first failure is rethrown here.
inline void parallel_steps(size_t count, int threads, const std::function<void(size_t)>& step) {
  if (count == 0) return;
  auto steps = std::make_shared<detail::Steps>(count, step);
  if (threads > 1 && count > 1) {
    Pool::instance().start(steps, std::min<size_t>(threads - 1, count - 1));
  }
  steps->drain();
  steps->join();
}

}  // namespace rpu_workers

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_allocator_policy_is_process_cold_and_race_free(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")

    source = tmp_path / "allocator_policy.cpp"
    binary = tmp_path / "allocator_policy"
    source.write_text(
        r'''
#include "src/core/rpu_allocator_policy.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

int main() {
  using Result = RpuTensorAllocatorPolicy::ClaimResult;

  RpuTensorAllocatorPolicy explicit_caching;
  assert(!explicit_caching.snapshot().frozen);
  assert(!explicit_caching.snapshot().caching);
  assert(explicit_caching.claim(true) == Result::kApplied);
  assert(explicit_caching.claim(true) == Result::kIdempotent);
  assert(explicit_caching.claim(false) == Result::kConflict);
  assert(explicit_caching.freeze_for_allocation());

  RpuTensorAllocatorPolicy implicit_direct;
  assert(!implicit_direct.freeze_for_allocation());
  assert(implicit_direct.snapshot().frozen);
  assert(implicit_direct.claim(false) == Result::kIdempotent);
  assert(implicit_direct.claim(true) == Result::kConflict);

  for (int iteration = 0; iteration < 1000; ++iteration) {
    RpuTensorAllocatorPolicy raced;
    std::atomic<bool> start{false};
    Result claim_result = Result::kConflict;
    bool allocation_uses_caching = false;
    std::thread claimant([&] {
      while (!start.load(std::memory_order_acquire)) {}
      claim_result = raced.claim(true);
    });
    std::thread allocator([&] {
      while (!start.load(std::memory_order_acquire)) {}
      allocation_uses_caching = raced.freeze_for_allocation();
    });
    start.store(true, std::memory_order_release);
    claimant.join();
    allocator.join();
    const auto final = raced.snapshot();
    assert(final.frozen);
    if (claim_result == Result::kConflict) {
      assert(!final.caching);
      assert(!allocation_uses_caching);
    } else {
      assert(final.caching);
      assert(allocation_uses_caching);
    }
  }

  RpuTensorDdrLifecycle lifecycle;
  int free_calls = 0;
  assert(lifecycle.run_free_if_live([&] { ++free_calls; }));
  assert(free_calls == 1);

  std::mutex control_mutex;
  std::condition_variable control;
  bool allocation_entered = false;
  bool release_allocation = false;
  int allocation_result = 0;
  std::thread in_flight_allocation([&] {
    allocation_result = lifecycle.run_allocation([&] {
      std::unique_lock<std::mutex> lock(control_mutex);
      allocation_entered = true;
      control.notify_all();
      control.wait(lock, [&] { return release_allocation; });
      return 73;
    });
  });
  {
    std::unique_lock<std::mutex> lock(control_mutex);
    control.wait(lock, [&] { return allocation_entered; });
  }

  std::atomic<bool> shutdown_call_started{false};
  std::atomic<bool> shutdown_returned{false};
  std::thread shutdown([&] {
    shutdown_call_started.store(true, std::memory_order_release);
    lifecycle.begin_shutdown();
    shutdown_returned.store(true, std::memory_order_release);
  });
  while (!shutdown_call_started.load(std::memory_order_acquire)) {}
  const auto wait_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  while (!shutdown_returned.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < wait_deadline) {
    std::this_thread::yield();
  }
  assert(!shutdown_returned.load(std::memory_order_acquire));

  {
    std::lock_guard<std::mutex> lock(control_mutex);
    release_allocation = true;
  }
  control.notify_all();
  in_flight_allocation.join();
  shutdown.join();
  assert(allocation_result == 73);
  assert(shutdown_returned.load(std::memory_order_acquire));
  assert(lifecycle.shutdown_started());

  int post_shutdown_allocation_calls = 0;
  bool allocation_rejected = false;
  try {
    (void)lifecycle.run_allocation([&] {
      ++post_shutdown_allocation_calls;
      return 0;
    });
  } catch (const std::runtime_error& error) {
    allocation_rejected =
        std::string(error.what()).find("shut down") != std::string::npos;
  }
  assert(allocation_rejected);
  assert(post_shutdown_allocation_calls == 0);
  assert(!lifecycle.run_free_if_live([&] { ++free_calls; }));
  assert(free_calls == 1);
}
'''
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-pthread",
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(binary)], check=True, timeout=30)

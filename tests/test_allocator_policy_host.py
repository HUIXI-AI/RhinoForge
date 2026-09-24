"""Compile the production sizing/counter bodies with tiny host-only data doubles.

No Torch headers, Launch library, native extension import or device allocation.
The separate explicit board tests cover real mapping ownership and DMA values.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(source, signature):
    start = source.index(signature)
    return source[start:source.index("\n}", start) + 2]


@pytest.fixture(scope="module")
def allocator_host_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    header = (ROOT / "src/core/rpu_caching_allocator.h").read_text()
    allocator = (ROOT / "src/core/rpu_caching_allocator.cpp").read_text()
    backend = (ROOT / "src/core/rpu_backend.cpp").read_text()
    constants = "\n".join(
        re.search(rf"constexpr size_t {name} = \d+;", header).group(0)
        for name in ("kMinBlockSize", "kSmallSize", "kSmallBuffer", "kRoundLarge", "kNumSizeClasses")
    )
    allocator_bodies = "\n".join(_function(allocator, signature) for signature in (
        "size_t RPUCachingAllocator::round_size(",
        "size_t RPUCachingAllocator::get_allocation_size(",
        "int64_t RPUCachingAllocator::getCachedSegmentCount(",
    ))
    stats_bodies = "\n".join(_function(backend, signature) for signature in (
        "pybind11::dict rpu_stat_to_dict(",
        "pybind11::dict rpu_get_memory_stats_py(",
    ))
    source = r'''
#include <algorithm>
#include <any>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
namespace pybind11 { using dict = std::map<std::string, std::any>; }
namespace rpu {
CONSTANTS
struct Block { bool split; bool is_split() const { return split; } };
struct BlockPool { std::set<Block*> blocks; };
class RPUCachingAllocator {
public:
    size_t round_size(size_t);
    size_t get_allocation_size(size_t);
    int64_t getCachedSegmentCount() const;
    size_t getLargestAvailableBlock() const { return 0; }
    size_t getTotalAllocatedMemory() const { return 0; }
    size_t getTotalCachedMemory() const { return 0; }
    mutable std::recursive_mutex mutex_;
    size_t size_class_counts_[kNumSizeClasses]{};
    BlockPool small_blocks_, large_blocks_;
};
ALLOCATOR_BODIES
RPUCachingAllocator allocator;
struct Stat { int64_t current{}, peak{}, allocated{}, freed{}; };
struct DeviceStats {
    Stat allocation, reserved_bytes, allocated_bytes, active_bytes, segment, inactive_split_bytes;
    int64_t num_device_alloc{}, num_device_free{}, num_ooms{}, num_alloc_retries{};
};
DeviceStats snapshot;
DeviceStats get_memory_stats() { return snapshot; }
RPUCachingAllocator& get_caching_allocator() { return allocator; }
}
struct AllocatorPolicySnapshot { bool frozen; bool caching; };
struct AllocatorPolicyDouble {
    AllocatorPolicySnapshot value{true, true};
    AllocatorPolicySnapshot snapshot() const { return value; }
} g_rpu_tensor_allocator_policy;
STATS_BODIES
int main(int argc, char** argv) {
    assert(argc == 2);
    auto& allocator = rpu::allocator;
    constexpr size_t MiB = 1024 * 1024;
    if (std::string(argv[1]) == "policy") {
        const size_t sizes[] = {1, MiB - 1, MiB, MiB + 2, 2 * MiB, 2 * MiB + 2, 10 * MiB, 10 * MiB + 2};
        const size_t expected[] = {2 * MiB, 2 * MiB, 2 * MiB, 2 * MiB, 2 * MiB, 4 * MiB, 10 * MiB, 12 * MiB};
        for (size_t i = 0; i < std::size(sizes); ++i) {
            const size_t rounded = allocator.round_size(sizes[i]);
            assert(rounded >= sizes[i] && rounded % 32 == 0);
            assert(allocator.get_allocation_size(rounded) == expected[i]);
        }
        const size_t medium = allocator.round_size(524289 * sizeof(uint16_t));
        assert(allocator.get_allocation_size(medium) == 2 * MiB);
        return 0;
    }
    assert(std::string(argv[1]) == "stats");
    rpu::Block small_idle{false}, small_split{true}, large_idle{false}, large_split{true};
    allocator.size_class_counts_[0] = 2;
    allocator.small_blocks_.blocks = {&small_idle, &small_split};
    allocator.large_blocks_.blocks = {&large_idle, &large_split};
    rpu::snapshot.segment = {7, 9, 12, 5};
    rpu::snapshot.allocation = {5000, 5000, 5000, 0};
    auto stats = rpu_get_memory_stats_py();
    assert(std::any_cast<bool>(stats.at("caching_allocator_enabled")));
    assert(std::any_cast<bool>(stats.at("caching_allocator_policy_frozen")));
    assert(std::any_cast<int64_t>(stats.at("cached_idle_mapping")) == 4);
    const auto mapping = std::any_cast<pybind11::dict>(stats.at("caching_allocator_mapping"));
    const auto segment = std::any_cast<pybind11::dict>(stats.at("segment"));
    for (const char* key : {"current", "peak", "allocated", "freed"})
        assert(std::any_cast<int64_t>(mapping.at(key)) == std::any_cast<int64_t>(segment.at(key)));
    assert(std::any_cast<int64_t>(mapping.at("current")) == 7);
    const auto allocations = std::any_cast<pybind11::dict>(stats.at("allocation"));
    assert(std::any_cast<int64_t>(allocations.at("current")) == 5000);
    g_rpu_tensor_allocator_policy.value = {true, false};
    std::fill(std::begin(allocator.size_class_counts_), std::end(allocator.size_class_counts_), 0);
    allocator.small_blocks_.blocks.clear();
    allocator.large_blocks_.blocks.clear();
    stats = rpu_get_memory_stats_py();
    assert(!std::any_cast<bool>(stats.at("caching_allocator_enabled")));
    assert(std::any_cast<int64_t>(stats.at("cached_idle_mapping")) == 0);
}
'''.replace("CONSTANTS", constants).replace("ALLOCATOR_BODIES", allocator_bodies).replace("STATS_BODIES", stats_bodies)
    directory = tmp_path_factory.mktemp("allocator-policy-host")
    cpp, binary = directory / "allocator.cpp", directory / "allocator"
    cpp.write_text(source)
    subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)],
                   check=True, capture_output=True, text=True, timeout=30)
    return binary


def test_medium_reservation_and_alignment_boundaries(allocator_host_binary):
    subprocess.run([str(allocator_host_binary), "policy"], check=True, timeout=5)


def test_public_mapping_stats_exclude_split_fragments(allocator_host_binary):
    subprocess.run([str(allocator_host_binary), "stats"], check=True, timeout=5)

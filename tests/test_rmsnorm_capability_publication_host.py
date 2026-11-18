"""Board-free lifecycle tests for RMSNorm payload capability snapshots."""

from __future__ import annotations

from pathlib import Path
from pi_wall_host_catalog import with_actual_linear_contracts
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
KERNEL_CACHE_HEADER = (ROOT / "src/core/rpu_kernel_cache.h").read_text(
    encoding="utf-8"
)
KERNEL_CACHE_SOURCE = (ROOT / "src/core/rpu_kernel_cache.inc").read_text(
    encoding="utf-8"
)
QWEN25_SOURCE = (ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp").read_text(
    encoding="utf-8"
)
QWEN_SOURCE = (ROOT / "src/fused/rpu_qwen3_model.h").read_text(
    encoding="utf-8"
)


def _block(source: str, marker: str) -> str:
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def _compile_and_run(tmp_path: Path, name: str, program: str) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = tmp_path / f"{name}.cpp"
    executable = tmp_path / name
    source.write_text(with_actual_linear_contracts(program), encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-pthread",
            "-I", str(ROOT / "src/core"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_kernel_payload_availability_is_atomic_and_withdrawn_before_clear(
    tmp_path: Path,
) -> None:
    availability = _block(
        KERNEL_CACHE_HEADER, "class KernelPayloadAvailability final"
    )
    cache_clear = _block(KERNEL_CACHE_HEADER, "void clear()")
    initialize = _block(KERNEL_CACHE_SOURCE, "void KernelCache::initialize()")

    assert "std::shared_ptr<const Snapshot> state_" in availability
    assert "std::atomic_load_explicit(&state_, std::memory_order_acquire)" in availability
    assert "std::atomic_store_explicit(&state_, next, std::memory_order_release)" in availability
    assert "std::shared_ptr<const Snapshot>{}" in availability
    assert cache_clear.index("payload_availability_.withdraw()") < cache_clear.index(
        "kernels_.clear()"
    )
    assert initialize.index("populate_fast_cache();") < initialize.index(
        "payload_availability_.publish("
    )
    assert initialize.index("payload_availability_.publish(") < initialize.index(
        "initialized_ = true;"
    )
    publication = initialize.split("payload_availability_.publish(", 1)[1].split(");", 1)[0]
    assert "pi05_extended_payloads" in publication
    assert "if (fast_cache_[i]) pi05_extended_payloads.set(i - extended_begin);" in initialize

    program = (
        r'''
#include <atomic>
#include <bitset>
#include <memory>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
'''
        + _block(KERNEL_CACHE_HEADER, "enum class KernelId") + ";\n"
        + r'''
'''
        + availability
        + ";\n"
        + r'''
int main() {
    KernelPayloadAvailability availability;
    bool rejected = false;
    try { (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V16); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    availability.publish(true, false);
    assert(!availability.has_loaded(KernelId::PI_WALL_QKV_W8A16_ACC16_M464N64K128_256X128X128));
    KernelPayloadAvailability::ExtendedPayloads wall;
    wall.set(static_cast<size_t>(KernelId::PI_WALL_QKV_W8A16_ACC16_M464N64K128_256X128X128) - KernelPayloadAvailability::kExtendedBegin);
    availability.publish(true, false, false, false, false, false, wall);
    assert(availability.has_loaded(KernelId::PI_WALL_QKV_W8A16_ACC16_M464N64K128_256X128X128));
    assert(availability.has_loaded(KernelId::RMS_NORM_SPM_V16));
    assert(!availability.has_loaded(KernelId::RMS_NORM_SPM_V32));
    assert(!availability.has_loaded(KernelId::RMS_NORM_SPM_NEWTON_V32));
    availability.publish(true, false);
    assert(!availability.has_loaded(KernelId::PI_WALL_QKV_W8A16_ACC16_M464N64K128_256X128X128));
    assert(availability.has_loaded(KernelId::RMS_NORM_SPM_V16));
    assert(!availability.has_loaded(KernelId::RMS_NORM_SPM_V32));
    assert(!availability.has_loaded(KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152));
    assert(!availability.has_loaded(KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152));
    assert(!availability.has_loaded(KernelId::PI_WALL_PI_GEGLU_A8_ONLINE_I32_C272_M48N32K2048));
    availability.publish(true, false, false, true, false);
    assert(availability.has_loaded(KernelId::RMS_NORM_SPM_V16));
    assert(!availability.has_loaded(KernelId::RMS_NORM_SPM_V32));
    assert(availability.has_loaded(KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152));
    assert(!availability.has_loaded(KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152));
    availability.publish(false, true, false, false, true);
    assert(!availability.has_loaded(KernelId::RMS_NORM_SPM_V16));
    assert(availability.has_loaded(KernelId::RMS_NORM_SPM_V32));
    assert(!availability.has_loaded(KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152));
    assert(availability.has_loaded(KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152));
    assert(!availability.has_loaded(KernelId::PI_WALL_PI_GEGLU_A8_ONLINE_I32_C272_M48N32K2048));
    KernelPayloadAvailability::ExtendedPayloads a8;
    a8.set(static_cast<size_t>(KernelId::PI_WALL_PI_GEGLU_A8_ONLINE_I32_C272_M48N32K2048));
    availability.publish(false, true, false, false, true, false, a8);
    assert(availability.has_loaded(KernelId::PI_WALL_PI_GEGLU_A8_ONLINE_I32_C272_M48N32K2048));
    assert(availability.has_loaded(KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152));
    availability.publish(false, true, false, false, true, false);
    assert(!availability.has_loaded(KernelId::PI_WALL_PI_GEGLU_A8_ONLINE_I32_C272_M48N32K2048));
    assert(!availability.has_loaded(KernelId::PI_WALL_WALL_ROW_REUSE_W8A16_ACC16_M208N80K256));
    KernelPayloadAvailability::ExtendedPayloads row_reuse;
    row_reuse.set(static_cast<size_t>(KernelId::PI_WALL_WALL_ROW_REUSE_W8A16_ACC16_M208N80K256));
    availability.publish(false, false, false, false, false, false, row_reuse);
    assert(availability.has_loaded(KernelId::PI_WALL_WALL_ROW_REUSE_W8A16_ACC16_M208N80K256));
    assert(!availability.has_loaded(KernelId::RMS_NORM_SPM_V16));
    assert(!availability.has_loaded(KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152));
    availability.publish(false, false, false, true);
    assert(!availability.has_loaded(KernelId::PI_WALL_WALL_ROW_REUSE_W8A16_ACC16_M208N80K256));
    assert(availability.has_loaded(KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152));
    constexpr auto first = static_cast<size_t>(KernelId::PL_AUTOTILE_BEGIN);
    constexpr auto base = KernelPayloadAvailability::kExtendedBegin;
    constexpr auto last = static_cast<size_t>(KernelId::_COUNT) - 1;
    static_assert(last > 255, "exercise IDs beyond the old uint8 boundary");
    KernelPayloadAvailability::ExtendedPayloads payloads;
    static_assert(last - first >= 64, "exercise capability publication beyond one word");
    payloads.set(last - base);
    availability.publish(false, false, false, false, false, false, payloads);
    assert(availability.has_loaded(static_cast<KernelId>(last)));
    assert(!availability.has_loaded(static_cast<KernelId>(first)));
    payloads.reset();payloads.set(first - base);
    availability.publish(false, false, false, false, false, false, payloads);
    assert(!availability.has_loaded(static_cast<KernelId>(last)));
    assert(availability.has_loaded(static_cast<KernelId>(first)));
    for (size_t bit : {size_t{63}, size_t{64}, last - first}) {
        payloads.reset();payloads.set(first + bit - base);
        availability.publish(false, false, false, false, false, false, payloads);
        for (size_t other = 0; other <= last - first; ++other)
            assert(availability.has_loaded(static_cast<KernelId>(first + other)) == (other == bit));
    }
    rejected = false;
    try { (void)availability.has_loaded(static_cast<KernelId>(65535)); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    std::atomic<bool> start{false};
    std::atomic<bool> done{false};
    std::atomic<int> ready{0};
    std::atomic<int> reads{0};
    std::vector<std::thread> readers;
    for (int thread = 0; thread < 4; ++thread) {
        readers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {}
            (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V16);
            (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V32);
            reads.fetch_add(1, std::memory_order_relaxed);
            ready.fetch_add(1, std::memory_order_release);
            while (!done.load(std::memory_order_acquire)) {
                try {
                    (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V16);
                    (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V32);
                    (void)availability.has_loaded(KernelId::PI_WALL_WALL_ROW_REUSE_W8A16_ACC16_M208N80K256);
                    (void)availability.has_loaded(KernelId::PI_WALL_QKV_W8A16_ACC16_M464N64K128_256X128X128);
                    (void)availability.has_loaded(KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152);
                    (void)availability.has_loaded(KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152);
                    (void)availability.has_loaded(KernelId::PI_WALL_PI_GEGLU_A8_ONLINE_I32_C272_M48N32K2048);
                    (void)availability.has_loaded(static_cast<KernelId>(first + 63));
                    (void)availability.has_loaded(static_cast<KernelId>(first + 64));
                    (void)availability.has_loaded(static_cast<KernelId>(last));
                    reads.fetch_add(1, std::memory_order_relaxed);
                } catch (const std::runtime_error&) {
                    // Withdrawal is the fail-closed state during teardown.
                }
            }
        });
    }
    std::thread publisher([&] {
        while (!start.load(std::memory_order_acquire)) {}
        while (ready.load(std::memory_order_acquire) != 4) {}
        for (int i = 0; i < 100000; ++i) {
            availability.withdraw();
            KernelPayloadAvailability::ExtendedPayloads current;
            current.set(first + ((i & 1) ? 63 : 64) - base);
            current.set(static_cast<size_t>(KernelId::PI_WALL_QKV_W8A16_ACC16_M464N64K128_256X128X128) - base, (i & 64) != 0);
            current.set(last - base, (i & 2) != 0);
            availability.publish((i & 1) != 0, (i & 2) != 0,
                                 (i & 4) != 0, (i & 8) != 0, (i & 16) != 0,
                                 (i & 32) != 0, current);
        }
        done.store(true, std::memory_order_release);
    });
    start.store(true, std::memory_order_release);
    publisher.join();
    for (auto& reader : readers) reader.join();
    assert(reads.load(std::memory_order_relaxed) > 0);

    availability.withdraw();
    rejected = false;
    try { (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V32); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    for (auto id : {KernelId::PI_WALL_WALL_ROW_REUSE_W8A16_ACC16_M208N80K256,
                    KernelId::PI_WALL_QKV_W8A16_ACC16_M464N64K128_256X128X128,
                    KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152,
                    KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152,
                    KernelId::PI_WALL_PI_GEGLU_A8_ONLINE_I32_C272_M48N32K2048,
                    KernelId::PI_WALL_PI_GEGLU_A8_PAIRED_I32_C272_M48N32K2048}) {
        rejected = false;
        try { (void)availability.has_loaded(id); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
}
'''
    )
    _compile_and_run(tmp_path, "payload_publication", program)


def test_qwen25vl_and_wall_route_setters_commit_only_after_snapshot(
    tmp_path: Path,
) -> None:
    qwen25_setter = _block(QWEN25_SOURCE, "void set_execution_routes(")
    wall_setter = _block(QWEN_SOURCE, "void set_wall_prefill_routes(")
    program = (
        r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
enum RpuRmsNormCapability : uint32_t {
    RPU_RMSNORM_CAP_BASE = 1u << 0,
    RPU_RMSNORM_CAP_V16 = 1u << 1,
    RPU_RMSNORM_CAP_V32 = 1u << 2,
};
constexpr uint32_t RPU_RMSNORM_CAP_ALL =
    RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V16 | RPU_RMSNORM_CAP_V32;
bool rpu_rmsnorm_capability_valid(uint32_t capability) {
    return (capability & RPU_RMSNORM_CAP_BASE) != 0 &&
        (capability & ~RPU_RMSNORM_CAP_ALL) == 0;
}
struct RpuRmsNormSpmContract { uint32_t capability; };
uint32_t loaded_capability = RPU_RMSNORM_CAP_BASE;
bool snapshot_rejected = false;
RpuRmsNormSpmContract rpu_snapshot_rmsnorm_spm_contract(uint32_t requested) {
    if (snapshot_rejected) throw std::runtime_error("snapshot rejected");
    return {RPU_RMSNORM_CAP_BASE |
            (requested & loaded_capability & ~RPU_RMSNORM_CAP_BASE)};
}

class VisionModel {
public:
'''
        + qwen25_setter
        + r'''
    int64_t num_layers() const { return 0; }
    bool linear_acc32_ = false;
    bool has_rope_ = false;
    bool rope_spm_enabled_ = false;
    bool fused_merger_enabled_ = false;
    bool mixed_precision_enabled_ = false;
    bool layer_group_enabled_ = false;
    bool exact_mimc_enabled_ = false;
    uint32_t rmsnorm_capability_ = RPU_RMSNORM_CAP_BASE;
};

class WallModel {
public:
'''
        + wall_setter
        + r'''
    int64_t num_layers() const { return 0; }
    bool wall_prefill_mixed_requested_ = false;
    bool wall_prefill_staged_requested_ = false;
    bool wall_prefill_owner_direct_requested_ = false;
    uint32_t wall_rmsnorm_capability_ = RPU_RMSNORM_CAP_BASE;
};

int main() {
    constexpr uint32_t fixed32 =
        RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V32;

    VisionModel vision;
    bool rejected = false;
    try { vision.set_execution_routes(true, true, true, false, true, fixed32); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    assert(!vision.rope_spm_enabled_ && !vision.fused_merger_enabled_ &&
           !vision.mixed_precision_enabled_ && !vision.layer_group_enabled_ &&
           !vision.exact_mimc_enabled_ &&
           vision.rmsnorm_capability_ == RPU_RMSNORM_CAP_BASE);

    snapshot_rejected = true;
    rejected = false;
    try { vision.set_execution_routes(true, false, false, true, false,
                                      RPU_RMSNORM_CAP_BASE); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    assert(!vision.rope_spm_enabled_ && !vision.layer_group_enabled_);
    snapshot_rejected = false;
    loaded_capability = fixed32;
    vision.set_execution_routes(true, true, true, false, true, fixed32);
    assert(vision.rope_spm_enabled_ && vision.fused_merger_enabled_ &&
           vision.mixed_precision_enabled_ && !vision.layer_group_enabled_ &&
           vision.exact_mimc_enabled_ &&
           vision.rmsnorm_capability_ == fixed32);

    loaded_capability = RPU_RMSNORM_CAP_BASE;
    WallModel wall;
    rejected = false;
    try { wall.set_wall_prefill_routes(true, true, fixed32); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    assert(!wall.wall_prefill_mixed_requested_ &&
           !wall.wall_prefill_staged_requested_ &&
           wall.wall_rmsnorm_capability_ == RPU_RMSNORM_CAP_BASE);

    snapshot_rejected = true;
    rejected = false;
    try { wall.set_wall_prefill_routes(true, true, RPU_RMSNORM_CAP_BASE); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    assert(!wall.wall_prefill_mixed_requested_ &&
           !wall.wall_prefill_staged_requested_);
    snapshot_rejected = false;
    loaded_capability = fixed32;
    wall.set_wall_prefill_routes(true, true, fixed32);
    assert(wall.wall_prefill_mixed_requested_ &&
           wall.wall_prefill_staged_requested_ &&
           wall.wall_rmsnorm_capability_ == fixed32);
}
'''
    )
    _compile_and_run(tmp_path, "route_setter_transaction", program)

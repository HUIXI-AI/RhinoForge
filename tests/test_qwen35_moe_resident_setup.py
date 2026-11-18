"""Host regression for resident installation's required-payload probe."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _braced(source, start, opening=None):
    opening = source.index("{", start) if opening is None else opening
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.mark.parametrize("weight_mode", [5, 6])
def test_resident_install_checks_required_main_and_extended_cache_entries(tmp_path, weight_mode):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/fused/rpu_qwen3_5_moe_model.cpp").read_text()
    setter = _braced(source, source.index("void Qwen3_5MoeModel::set_resident_prefill("))
    match = re.search(r"for\s*\(const auto id\s*:\s*\{(.*?)\}\)\s*\{", setter, re.S)
    assert match is not None
    selected = re.sub(r"weight_mode == 5 \? (KernelId::\w+)\s*: (KernelId::\w+)",
                      lambda m: m[1] if weight_mode == 5 else m[2], match.group(1))
    payloads = re.findall(r"KernelId::([A-Z0-9_]+)", selected)
    expected = {
        "GPL_INDEXED_W4_M256N32K256", "GPL_SCATTER_W4_M256N128K64",
        "MOE_KVSORT_FP16", "MOE_BINCOUNT_SCALAR", "MOE_TOPK_PROB_NORM_FP16",
        "CAST_UINT16_INT32", "GATHER_FP16",
    }
    if weight_mode == 5:
        expected = {name.replace("_W4_", "_W8_") for name in expected}
    assert len(payloads) == len(expected) and set(payloads) == expected
    probe = _braced(setter, match.start(), match.end() - 1)
    # Missing dependencies must fail before allocating the two route tables.
    assert setter.index(probe) < setter.index("resident_route_ordinals_ = at::empty")
    header = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    enum = _braced(header, header.index("enum class KernelId")) + ";"
    availability = _braced(
        header, header.index("class KernelPayloadAvailability final")
    ) + ";"
    getter = _braced(header, header.index("inline ::rhino_lkn::Kernel_t* get(KernelId id)"))
    harness = r"""
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"
#define TORCH_CHECK(c, ...) do {if (!(c)) throw std::runtime_error("rejected");} while (0)
""" + enum + "\n" + availability + r"""
namespace rhino_lkn {struct Kernel_t {};}
struct KernelCache {
    static KernelCache& instance() {static KernelCache cache;return cache;}
    rhino_lkn::Kernel_t* fast_cache_[static_cast<size_t>(KernelId::_COUNT)]{};
    KernelPayloadAvailability payload;
    bool has_loaded(KernelId id) const {return payload.has_loaded(id);}
""" + getter + r"""
};
void installation_probe() {
""" + f"constexpr int weight_mode={weight_mode};\n" + probe + r"""
}
int main() {
    auto& cache = KernelCache::instance();
    rhino_lkn::Kernel_t kernel;
    const std::array<KernelId, 7> required{
""" + ",".join("KernelId::" + name for name in payloads) + r"""
    };
    KernelPayloadAvailability::ExtendedPayloads extended;
    extended.set();
    cache.payload.publish(false, false, false, false, false, false, extended);
    for (const auto id : required)
        cache.fast_cache_[static_cast<size_t>(id)] = &kernel;
    // The new publication covers main and extension IDs. Installation still
    // verifies the actual cache entries rather than accepting capability bits.
    assert(cache.has_loaded(KernelId::MOE_BINCOUNT_SCALAR));
    installation_probe();
    // Independently missing required main or extension entries must reject.
    // Optional capability publication intentionally remains populated.
    for (const auto id : required) {
        auto& entry = cache.fast_cache_[static_cast<size_t>(id)];
        entry = nullptr;
        bool rejected = false;
        try {installation_probe();}
        catch (const std::runtime_error&) {rejected = true;}
        assert(rejected);
        entry = &kernel;
    }
    installation_probe();
    // Conversely, an unpublished optional bit cannot hide an actual required
    // entry from the initialized resident installation path.
    cache.payload.publish(false, false);
    assert(!cache.has_loaded(KernelId::MOE_BINCOUNT_SCALAR));
    installation_probe();
}
"""
    cpp, executable = tmp_path / "resident_setup.cpp", tmp_path / "resident_setup"
    cpp.write_text(harness)
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
         "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"),
         str(cpp), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_resident_setup_follows_completed_runtime_and_weight_installation():
    backend = (ROOT / "src/core/rpu_backend.cpp").read_text()
    startup = _braced(backend, backend.index("static void initialize_rpu_backend_runtime()"))
    assert startup.index("KernelCache::instance().initialize();") < startup.index(
        "g_rpu_backend_lifecycle = RpuBackendLifecycleState::kReady;"
    )
    cache = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
    initialize = _braced(cache, cache.index("void KernelCache::initialize()"))
    assert initialize.index("populate_fast_cache();") < initialize.index("initialized_ = true;")
    pybind = (ROOT / "src/core/rpu_pybind.inc").read_text()
    module = pybind[pybind.index("PYBIND11_MODULE(rpu_backend, m)"):]
    assert module.index("initialize_rpu_backend_runtime();") < module.index("m.def(")
    adapter = (ROOT / "python/rpu_backend/adapters/qwen3_5_moe/text.py").read_text()
    assert adapter.index("torch.ops.rpu.qwen3_5_moe_set_weights(") < adapter.index(
        "torch.ops.rpu.qwen3_5_moe_set_resident_uint4_n32k32_weights("
    ) < adapter.index("torch.ops.rpu.qwen3_5_moe_set_resident_prefill(handle, True)")



def test_native_resident_and_acc32_cold_setters_are_order_independent(tmp_path):
    from test_qwen35_moe_core_native import _definition, _run_cpp

    source = (ROOT / "src/fused/rpu_qwen3_5_moe_model.cpp").read_text()
    methods = "\n".join(_definition(source, "void Qwen3_5MoeModel::" + name + "(")
                        for name in ("set_linear_acc32", "set_resident_prefill"))
    # The new ordinary GroupedPL ABI admits UINT4 ACC32; the indexed resident
    # schedule still requires ACC16, regardless of cold setter order.
    weights = _definition(source, "void Qwen3_5MoeModel::set_weights(")
    assert "TORCH_CHECK(!routed_uint4 || !linear_acc32_," not in weights
    _run_cpp(tmp_path, r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c, ...) do {if (!(c)) throw std::runtime_error("rejected");} while (0)
namespace c10 {using Half = uint16_t;}
namespace at {
struct Tensor {
    std::shared_ptr<std::vector<uint16_t>> storage;
    std::vector<int64_t> dimensions;
    bool contiguous=true;
    Tensor() = default;
    explicit Tensor(int64_t n):storage(std::make_shared<std::vector<uint16_t>>(n)),dimensions{n} {}
    Tensor(int64_t rows,int64_t cols):storage(std::make_shared<std::vector<uint16_t>>()),dimensions{rows,cols} {}
    bool defined() const {return bool(storage);}
    bool is_contiguous() const {return contiguous;}
    int dim() const {return dimensions.size();}
    int64_t size(int axis) const {return dimensions.at(axis);}
    int options() const {return 0;}
    template<class T> T* data_ptr() {return reinterpret_cast<T*>(storage->data());}
};
Tensor empty(std::initializer_list<int64_t> shape, int) {return Tensor(*shape.begin());}
}
enum class KernelId {GPL_INDEXED_W8_M256N32K256,GPL_INDEXED_W4_M256N32K256,
    GPL_SCATTER_W8_M256N128K64,GPL_SCATTER_W4_M256N128K64,MOE_KVSORT_FP16,
    MOE_BINCOUNT_SCALAR,MOE_TOPK_PROB_NORM_FP16,CAST_UINT16_INT32,GATHER_FP16};
struct KernelCache {
    static KernelCache& instance() {static KernelCache c;return c;}
    void* get(KernelId) {return this;}
};
constexpr int kRoutedUint4Mode=6, kResidentMaxRoutes=32767;
int64_t Align(int64_t n,int64_t a) {return (n+a-1)/a*a;}
void rpu_ddr_flush_force_sized(void*,int64_t) {}
struct Qwen3_5MoeModel {
    struct Layer {
        int routed_weight_mode=5;
        at::Tensor routed_gate_scale{256,512},routed_up_scale{256,512},routed_down_scale{256,2048};
    };
    struct Config {int num_experts=256,top_k=8,routed_intermediate=512,shared_intermediate=512;} moe_;
    std::vector<Layer> moe_layer_weights_ = std::vector<Layer>(40);
    at::Tensor expert_ids_{512},resident_route_ordinals_,resident_route_tokens_;
    bool resident_uint4_n32k32_installed_=false,resident_prefill_=false;
    bool linear_acc32_=false,retained_prefill_graph_=false;
    int resolved=0, invalidations=0,cores=8;
    int64_t envelope=0,routed_width=512;
    int num_cores() const {return cores;}
    int64_t physical_routed_intermediate() const {return routed_width;}
    int hidden_size() const {return 2048;}
    int num_layers() const {return 40;}
    int get_last_resolved_chunk_size() const {return resolved;}
    void set_chunk_envelope(int64_t,int64_t value) {envelope=value;}
    void invalidate_model_state() {++invalidations;}
    void set_linear_acc32(bool);
    void set_resident_prefill(bool);
};
""" + methods + r"""
template<class F> void rejects(F fn) {
    bool rejected=false;
    try {fn();} catch(const std::runtime_error&) {rejected=true;}
    assert(rejected);
}
int main() {
    // FP8: both setter orders reject the unsupported combination atomically.
    Qwen3_5MoeModel resident;
    resident.set_resident_prefill(true);
    const auto owner=resident.resident_route_ordinals_.storage;
    const auto generation=resident.invalidations;
    rejects([&] {resident.set_linear_acc32(true);});
    assert(resident.resident_prefill_ && !resident.linear_acc32_);
    assert(resident.invalidations==generation && resident.envelope==8192);
    assert(resident.resident_route_ordinals_.storage==owner);
    resident.set_linear_acc32(false);
    resident.set_resident_prefill(false);
    resident.set_linear_acc32(true);
    assert(!resident.resident_prefill_ && resident.linear_acc32_);
    assert(!resident.resident_route_ordinals_.defined());
    const auto after_disable=resident.invalidations;
    rejects([&] {resident.set_resident_prefill(true);});
    assert(!resident.resident_prefill_ && resident.linear_acc32_);
    assert(resident.invalidations==after_disable && !resident.resident_route_ordinals_.defined());
    resident.set_linear_acc32(false);
    resident.set_resident_prefill(true);
    assert(resident.resident_prefill_ && !resident.linear_acc32_);
    // Ordinary UINT4 supports both orders and both accumulations. The
    // resident GIL schedule must still reject ACC32 without mutating owners.
    Qwen3_5MoeModel cold;
    cold.moe_layer_weights_.clear();
    cold.set_linear_acc32(true);
    cold.moe_layer_weights_.resize(40);
    for(auto& layer:cold.moe_layer_weights_) layer.routed_weight_mode=6;
    cold.set_linear_acc32(false);
    cold.set_linear_acc32(true);
    assert(cold.linear_acc32_ && cold.invalidations==3);
    cold.resident_uint4_n32k32_installed_=true;
    rejects([&] {cold.set_resident_prefill(true);});
    assert(!cold.resident_prefill_ && cold.linear_acc32_);
    cold.set_linear_acc32(false);
    cold.set_resident_prefill(true);
    const auto uint4_generation=cold.invalidations;
    const auto uint4_owner=cold.resident_route_ordinals_.storage;
    rejects([&] {cold.set_linear_acc32(true);});
    assert(cold.resident_prefill_ && !cold.linear_acc32_);
    assert(cold.invalidations==uint4_generation &&
           cold.resident_route_ordinals_.storage==uint4_owner);
    cold.resolved=64;
    rejects([&] {cold.set_linear_acc32(false);});
    // TP4/8's aligned owners support both modes without replacement.
    for(int cores:{4,8}) {
        Qwen3_5MoeModel aligned;aligned.cores=cores;
        const auto scale=aligned.moe_layer_weights_.front().routed_gate_scale.storage;
        aligned.set_linear_acc32(true);
        aligned.set_linear_acc32(false);
        assert(!aligned.linear_acc32_ && aligned.invalidations==2);
        assert(aligned.moe_layer_weights_.front().routed_gate_scale.storage==scale);
    }
    // New-ABI TP6 FP8 owners always use the padded 640-column expert stride,
    // so accumulation changes reuse every owner without re-uploading weights.
    for(bool acc32:{false,true}) {
        Qwen3_5MoeModel strided;strided.cores=6;strided.routed_width=576;
        strided.linear_acc32_=acc32;
        for(auto& layer:strided.moe_layer_weights_) {
            layer.routed_gate_scale=at::Tensor(256,640);
            layer.routed_up_scale=at::Tensor(256,640);
        }
        const auto scale=strided.moe_layer_weights_.back().routed_up_scale.storage;
        strided.set_linear_acc32(acc32);
        strided.set_linear_acc32(!acc32);
        assert(strided.linear_acc32_!=acc32 && strided.invalidations==2);
        assert(strided.moe_layer_weights_.back().routed_up_scale.storage==scale);
        // An old unpadded owner is rejected in either mode, before mutation.
        strided.moe_layer_weights_.back().routed_up_scale=at::Tensor(256,576);
        rejects([&] {strided.set_linear_acc32(acc32);});
        assert(strided.linear_acc32_!=acc32 && strided.invalidations==2);
    }
    // Validate all layers and all projection roles, not only the first owner.
    for(int role:{0,1,2}) {
        Qwen3_5MoeModel bad;
        auto& last=bad.moe_layer_weights_.back();
        auto* scale=role==0?&last.routed_gate_scale:
                    role==1?&last.routed_up_scale:&last.routed_down_scale;
        scale->dimensions[1]-=1;
        rejects([&] {bad.set_linear_acc32(true);});
        assert(!bad.linear_acc32_ && bad.invalidations==0);
    }
}
""")


def test_adapter_binds_precision_before_native_scale_stride_validation():
    import ast

    source = (ROOT / "python/rpu_backend/adapters/qwen3_5_moe/text.py").read_text()
    tree = ast.parse(source)
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("qwen3_5_moe_set_execution_cores",
                                  "qwen3_5_moe_set_linear_acc32",
                                  "qwen3_5_moe_set_weights",
                                  "qwen3_5_moe_set_resident_prefill"):
                calls.setdefault(node.func.attr, []).append(node)
    assert all(len(value) == 1 for value in calls.values())
    ordered = [calls[name][0] for name in (
        "qwen3_5_moe_set_execution_cores", "qwen3_5_moe_set_linear_acc32",
        "qwen3_5_moe_set_weights", "qwen3_5_moe_set_resident_prefill")]
    assert [node.lineno for node in ordered] == sorted(node.lineno for node in ordered)
    assert ast.unparse(ordered[1].args[1]) == "pending_state.linear_acc32"

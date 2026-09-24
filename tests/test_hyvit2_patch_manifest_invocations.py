"""Host-only regression for HYViT2's three-image patch manifest identities."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(
    os.environ.get(
        "RHINOFORGE_TEST_ROOT",
        Path(__file__).resolve().parents[1],
    )
)
SOURCE = ROOT / "src/fused/rpu_hyvit2_vision_model.cpp"


def _balanced_block(source: str, marker: str, *, trailing_semicolon: bool = False) -> str:
    begin = source.index(marker)
    opening = (
        source.index(") {", begin) + 2
        if "const auto append" in marker else source.index("{", begin)
    )
    depth = 0
    for cursor in range(opening, len(source)):
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
            if depth == 0:
                end = cursor + 1
                if trailing_semicolon:
                    assert source[end] == ";"
                    end += 1
                return source[begin:end]
    raise AssertionError(f"unterminated block: {marker}")


def test_three_image_external_patch_routes_use_invocation_not_flags(tmp_path: Path) -> None:
    """Execute the production append block for the packed S588/s240 profile."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host-native manifest test")

    source = SOURCE.read_text(encoding="utf-8")
    append_lambda = _balanced_block(
        source,
        "        const auto append = [&](FmbRouteFamily family, int64_t site_id,",
        trailing_semicolon=True,
    )
    patch_prologue = _balanced_block(
        source,
        "        if (planning_external_patch_prologue_) {",
    )
    patch_constants = "\n".join(
        re.findall(
            r"constexpr int64_t HYVIT2_PATCH_(?:INPUT_DMA|LINEAR|ALL_GATHER|POSITION_DMA|OUTPUT_DMA)_SITE\s*=\s*[^;]+;",
            source,
        )
    )
    assert patch_constants.count("constexpr int64_t") == 5
    patch_route_enum = _balanced_block(
        source,
        "enum class HyViT2PatchMutableDmaRoute : int64_t",
        trailing_semicolon=True,
    )

    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
#define DWIDTH 2

enum class FmbRouteFamily : int64_t {
    ATTENTION = 1, LINEAR = 2, ALL_REDUCE = 3, ROPE = 4, KV_INSERT = 5,
    MUTABLE_DMA = 6, NORMALIZATION = 7, ACTIVATION = 8,
    GRAPH_SCHEDULE = 9, COLLECTIVE = 10,
};
enum class FmbLinearRouteSelector : int64_t { AUTO_TILE = 1 };
__PATCH_ROUTE_ENUM__
__PATCH_CONSTANTS__

using RpuAllGatherSchedule = int64_t;
RpuAllGatherSchedule rpu_resolve_all_gather_schedule(int64_t, int64_t) {
    return 1;
}

struct FmbExecutionSpan {
    int64_t offset;
    int64_t len;
    int64_t group_id;
};
struct FmbThreeStageChunkPlan {
    std::vector<FmbExecutionSpan> spans;
};
struct FmbRouteManifestEntry {
    int64_t site_id;
    FmbRouteFamily family;
    int64_t selector;
    int64_t flags;
    std::vector<int64_t> arguments;
    int64_t invocation;
};
struct FmbPhysicalExecutionManifest {
    std::vector<FmbRouteManifestEntry> routes;
};

struct Harness {
    bool planning_external_patch_prologue_ = true;
    bool has_patch_emb_ = true;
    int64_t image_batch_count_ = 3;
    int64_t pe_kh_ = 16;
    int64_t pe_kw_ = 16;
    int64_t pe_cin_orig_ = 3;
    int64_t pe_cin_padded_ = 16;
    int64_t pe_cout_ = 1152;
    int64_t pe_strideh_ = 16;
    int64_t pe_stridew_ = 16;
    int64_t patch_embed_cores_ = 8;

    FmbPhysicalExecutionManifest build(const FmbThreeStageChunkPlan& plan) const {
        FmbPhysicalExecutionManifest manifest;
__APPEND_LAMBDA__
__PATCH_PROLOGUE__
        std::sort(
            manifest.routes.begin(), manifest.routes.end(),
            [](const FmbRouteManifestEntry& lhs,
               const FmbRouteManifestEntry& rhs) {
                return std::make_tuple(
                           static_cast<int64_t>(lhs.family), lhs.site_id,
                           lhs.invocation) <
                       std::make_tuple(
                           static_cast<int64_t>(rhs.family), rhs.site_id,
                           rhs.invocation);
            });
        return manifest;
    }
};

int main() {
    const FmbThreeStageChunkPlan plan{{
        {0, 196, 0}, {196, 196, 1}, {392, 196, 2},
    }};
    const auto manifest = Harness{}.build(plan);
    assert(manifest.routes.size() == 15);

    const std::vector<int64_t> sites{
        HYVIT2_PATCH_INPUT_DMA_SITE,
        HYVIT2_PATCH_LINEAR_SITE,
        HYVIT2_PATCH_ALL_GATHER_SITE,
        HYVIT2_PATCH_POSITION_DMA_SITE,
        HYVIT2_PATCH_OUTPUT_DMA_SITE,
    };
    std::set<std::tuple<int64_t, int64_t, int64_t>> identities;
    for (const auto& route : manifest.routes) {
        assert(route.flags == 0);
        assert(identities.emplace(
            static_cast<int64_t>(route.family), route.site_id,
            route.invocation).second);
    }
    for (const int64_t site : sites) {
        std::vector<int64_t> invocations;
        for (const auto& route : manifest.routes) {
            if (route.site_id == site) invocations.push_back(route.invocation);
        }
        assert((invocations == std::vector<int64_t>{0, 1, 2}));
    }

    std::vector<int64_t> output_offsets;
    for (const auto& route : manifest.routes) {
        if (route.site_id == HYVIT2_PATCH_OUTPUT_DMA_SITE) {
            output_offsets.push_back(route.arguments.at(0));
        }
    }
    assert((output_offsets == std::vector<int64_t>{0, 451584, 903168}));
}
'''
    for key, value in {
        "PATCH_ROUTE_ENUM": patch_route_enum,
        "PATCH_CONSTANTS": patch_constants,
        "APPEND_LAMBDA": append_lambda,
        "PATCH_PROLOGUE": patch_prologue,
    }.items():
        program = program.replace(f"__{key}__", value)

    cpp = tmp_path / "hyvit2-manifest-invocations.cpp"
    binary = tmp_path / "hyvit2-manifest-invocations"
    cpp.write_text(program, encoding="utf-8")
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr


def test_patch_geometry_preserves_batch_semantics_and_rejects_bad_inputs(tmp_path):
    """Execute the native preflight and setter with allocation-free metadata."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host-native geometry test")
    source = SOURCE.read_text(encoding="utf-8")
    preflight = _balanced_block(source, "    int64_t validate_patch_inputs(")
    setter = _balanced_block(source, "    void set_patch_emb_params(")
    program = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
#define DWIDTH 2
constexpr int64_t kMaxPackedImages = 8;
namespace at {
constexpr int kHalf = 1, kPrivateUse1 = 2;
struct Tensor {
    std::vector<int64_t> shape;
    int dtype = kHalf, device_type = kPrivateUse1;
    bool contiguous = true;
    bool defined() const { return !shape.empty(); }
    int64_t dim() const { return shape.size(); }
    int64_t size(int i) const { return shape.at(i); }
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    int scalar_type() const { return dtype; }
    struct Device { int value; int type() const { return value; } };
    Device device() const { return {device_type}; }
    bool is_contiguous() const { return contiguous; }
};
using TensorList = std::vector<Tensor>;
}
struct Model {
    bool has_patch_emb_ = false;
    int64_t patch_kernel_size_ = 0, patch_stride_ = 0;
    int64_t pe_kh_ = 0, pe_kw_ = 0, pe_strideh_ = 0, pe_stridew_ = 0;
    int64_t pe_cin_orig_ = 3, pe_cin_padded_ = 0, pe_cout_ = 0;
    int patch_embed_cores_ = 8;
    at::Tensor patch_emb_weight_, patch_emb_pos_emb_;
__SETTER__
__PREFLIGHT__
};
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
int main() {
    Model model;
    const at::Tensor weight{{1152, 16 * 16 * 16}};
    const at::Tensor positions{{1, 196, 1152}};
    const at::Tensor image{{1, 3, 224, 224}};
    model.set_patch_emb_params(weight, positions, 16, 16);
    // Single tensor B=N and N list entries must pack the same number of rows.
    assert(model.validate_patch_inputs({{{3, 3, 224, 224}}}, true) == 588);
    assert(model.validate_patch_inputs({image, image, image}, false) == 588);
    assert(model.validate_patch_inputs({{{8, 3, 224, 224}}}, true) == 1568);
    rejects([&] { model.validate_patch_inputs({{{2, 3, 224, 224}}}, false); });
    rejects([&] { model.validate_patch_inputs({image, {{1, 3, 240, 240}}}, false); });
    rejects([&] { model.validate_patch_inputs({{{1, 1, 224, 224}}}, true); });
    rejects([&] { model.validate_patch_inputs({{{9, 3, 224, 224}}}, true); });
    rejects([&] { model.validate_patch_inputs(std::vector<at::Tensor>(9, image), false); });
    rejects([&] { model.validate_patch_inputs({}, false); });
    rejects([&] { model.validate_patch_inputs({{{1, 3, 224, 225}}}, true); });
    rejects([&] { model.validate_patch_inputs({{{1, 3, 16, 0}}}, true); });
    // Width is the permute stride hazard; a bottom height tail is valid.
    assert(model.validate_patch_inputs({{{1, 3, 225, 224}}}, true) == 196);
    auto invalid = image;
    invalid.dtype = 0;
    rejects([&] { model.validate_patch_inputs({invalid}, true); });
    invalid = image;
    invalid.device_type = 0;
    rejects([&] { model.validate_patch_inputs({invalid}, true); });
    invalid = image;
    invalid.contiguous = false;
    rejects([&] { model.validate_patch_inputs({invalid}, true); });
    model.set_patch_emb_params(weight, {{1, 195, 1152}}, 16, 16);
    rejects([&] { model.validate_patch_inputs({image}, true); });
    model.set_patch_emb_params(weight, positions, 16, 16);
    // A failed setter must leave the previous valid configuration intact.
    rejects([&] { model.set_patch_emb_params(weight, positions, 0, 16); });
    assert(model.pe_kh_ == 16 && model.has_patch_emb_);
    rejects([&] { model.set_patch_emb_params({{1152, 16 * 16 * 4}}, positions, 16, 16); });
    assert(model.pe_cin_padded_ == 16);
    // Single-core im2col supports width tails and overlapping patches.
    model.patch_embed_cores_ = 1;
    assert(model.validate_patch_inputs({{{1, 3, 224, 225}}}, true) == 196);
    model.set_patch_emb_params(weight, {{1, 729, 1152}}, 16, 8);
    assert(model.validate_patch_inputs({image}, true) == 729);
    model.patch_embed_cores_ = 8;
    rejects([&] { model.validate_patch_inputs({image}, true); });
}
'''.replace("__SETTER__", setter).replace("__PREFLIGHT__", preflight)
    cpp = tmp_path / "hyvit2-patch-geometry.cpp"
    binary = tmp_path / "hyvit2-patch-geometry"
    cpp.write_text(program, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)

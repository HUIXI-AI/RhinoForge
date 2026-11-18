"""Board-free contracts for LingBot2 MoE weight-boundary validation."""

from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp").read_text()


def _section(start: str, end: str) -> str:
    begin = SOURCE.index(start)
    return SOURCE[begin : SOURCE.index(end, begin)]


def test_unpacked_weights_reject_shape_extent_or_layout_drift():
    setter = _section(
        "void LingbotV2MoeExpertModel::set_moe_weights(",
        "// declare_buffers",
    )
    compact = " ".join(setter.split())

    for token in (
        "t.is_contiguous()",
        "t.sizes() == expected_shape",
        "t.numel() == expected_numel",
        "check_exact(router_gate_w[i], {num_experts, h}",
        "check_exact(router_bias[i], {num_experts}",
        "check_exact(expert_gate_w[i], {routed_inter, h}",
        "check_exact(expert_up_w[i], {routed_inter, h}",
        "check_exact(expert_down_w[i], {h, routed_inter}",
    ):
        assert token in compact

    assert setter.index("check_exact(router_gate_w[i]") < setter.index(
        "router_gate_.assign"
    )
    assert setter.index("check_exact(expert_down_w[i]") < setter.index(
        "expert_down_.assign"
    )


def test_packed_weights_reject_noncontiguous_views_before_retention():
    setter = _section(
        "void LingbotV2MoeExpertModel::set_packed_expert_weights(",
        "// build_layer_subgraph",
    )

    assert "t.is_contiguous()" in setter
    assert "must be contiguous fp16/int8" in setter
    assert setter.index("t.is_contiguous()") < setter.index("packed_gate_.assign")


def test_sparse_scale_install_matches_real_packer_and_rejects_compact_tail(tmp_path):
    """H800 is an ABI boundary probe, not an additional supported model profile."""
    import torch
    from rpu_backend.adapters.lingbot_vla_v2.convert import pack_sparse_expert_projection

    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    setter = _definition(SOURCE, "void LingbotV2MoeExpertModel::set_sparse_expert_weights(")
    validate = _definition(setter, "auto validate =") + ";"
    cases = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for hidden in (768, 800):
            weight, scale = pack_sparse_expert_projection(
                torch.ones((32, hidden, 512), dtype=torch.float16),
                partition="row", num_cores=8, quantized=True,
            )
            assert weight.dtype == torch.int8 and scale.dtype == torch.float16
            assert weight.is_contiguous() and scale.is_contiguous()
            assert scale.shape == (32, 768 if hidden == 768 else 896)
            assert bool((scale[:, :hidden] > 0).all())
            assert not bool(scale[:, hidden:].any())
            cases.append(
                f"Case{{{hidden}, {{{weight.size(0)},{weight.size(1)},at::kChar}}, "
                f"{{{scale.size(0)},{scale.size(1)},at::kHalf}}}}"
            )
    finally:
        torch.set_num_threads(threads)
    probe = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define Align(x,a) (((x)+(a)-1)/(a)*(a))
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
namespace at {
constexpr int kHalf=1, kChar=2, kPrivateUse1=3, kCPU=4;
struct Device { int value; int type() const {return value;} };
struct Tensor {
    int64_t rows, columns;
    int dtype, location=kPrivateUse1;
    bool contiguous=true, present=true;
    bool defined() const {return present;}
    int scalar_type() const {return dtype;}
    Device device() const {return {location};}
    bool is_contiguous() const {return contiguous;}
    int dim() const {return 2;}
    int64_t size(int axis) const {return axis==0?rows:columns;}
};
using TensorList=std::vector<Tensor>;
}
struct Case {int64_t hidden; at::Tensor weight, scale;};
void check(const Case& value, at::TensorList scales) {
    const int64_t nl=1, e=32;
    const auto dtype=value.weight.scalar_type();
    const bool quantized=dtype==at::kChar;
    __VALIDATE__
    validate({value.weight}, scales, value.hidden, 512);
}
void rejects(const Case& value, at::TensorList scales) {
    bool rejected=false;
    try {check(value, scales);} catch (const std::runtime_error&) {rejected=true;}
    assert(rejected);
}
int main() {
    for (const auto& value:std::vector<Case>{__CASES__}) {
        check(value, {value.scale});
        auto scale=value.scale;
        scale.columns+=128; rejects(value, {scale});
        scale=value.scale; scale.columns-=128; rejects(value, {scale});
        // Identical bytes with different expert/row axes must still be rejected.
        scale=value.scale; scale.rows/=2; scale.columns*=2; rejects(value, {scale});
        scale=value.scale; scale.dtype=at::kChar; rejects(value, {scale});
        scale=value.scale; scale.location=at::kCPU; rejects(value, {scale});
        scale=value.scale; scale.contiguous=false; rejects(value, {scale});
        scale=value.scale; scale.present=false; rejects(value, {scale});
        rejects(value, {});
        if (value.hidden==800) {
            scale=value.scale; scale.columns=800; rejects(value, {scale});
        }
        auto fp16=value; fp16.weight.dtype=at::kHalf;
        check(fp16, {});
        rejects(fp16, {value.scale});
    }
}
'''.replace("__VALIDATE__", validate).replace("__CASES__", ",".join(cases))
    cpp, binary = tmp_path / "sparse_scale.cpp", tmp_path / "sparse_scale"
    cpp.write_text(probe, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)

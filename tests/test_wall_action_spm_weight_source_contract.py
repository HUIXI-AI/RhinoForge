from pi_wall_host_catalog import assert_current_catalog, CATALOG_INCLUDE
from test_siglip_attention_dry_live_contract import _function
"""Board-free contract for exact Wall Action core-local SPM weights."""

import ast
from numbers import Integral
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ACTION = (ROOT / "python/rpu_backend/adapters/wall_oss/action.py").read_text()
WEIGHTS = (ROOT / "python/rpu_backend/runtime/weights.py").read_text()
CACHE_H = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
CACHE_INC = ((ROOT / "src/core/rpu_kernel_cache.inc").read_text() + (ROOT / "src/core/rpu_kernel_cache.h").read_text())
DECLS = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
LINEAR = (ROOT / "src/ops/rpu_linear.cpp").read_text()
DECODER = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
ACTION_MODEL = (ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()

QKV_ID = "WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64_SPM_WEIGHT"
GATE_UP_ID = "WALL_ACTION_GATE_UP_W8A16_MULTIBASE_M592N32_SPM_WEIGHT"
ROW_ID = "WALL_ACTION_LINEAR_RING_POSTNORM_W8A16_M20N1024K256_SPM_WEIGHT"
QKV_NAME = (
    "qkv_parallel_w8a16_multibase_partial_mrope_m20n64k128_spm_weight"
)
GATE_UP_NAME = "gate_up_parallel_w8a16_multibase_spm_weight_m592n32k128"
ROW_NAME = (
    "wall_action_w8a16_n128_linear_ring_postnorm_"
    "spm_weight_m20n1024k256"
)


def _section(start: str, end: str | None = None) -> str:
    begin = LINEAR.index(start)
    return LINEAR[begin:] if end is None else LINEAR[begin : LINEAR.index(end, begin)]


def _load_pack_helpers():
    action_tree = ast.parse(ACTION)
    action_nodes = [
        node for node in action_tree.body
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name)
                and target.id.startswith("_ACTION_")
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
        ) or (
            isinstance(node, ast.FunctionDef)
            and node.name in {
                "_compact_w8_weight_per_core",
                "_pack_action_resident_weight_bundle",
                "_pack_action_fp16_resident_weight_bundle",
            }
        )
    ]
    namespace = {"torch": torch, "_MLP_CORES": 8}
    exec(compile(ast.Module(action_nodes, []), "action.py", "exec"), namespace)

    weights_tree = ast.parse(WEIGHTS)
    weight_nodes = [
        node for node in weights_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_positive_int",
            "_validate_swizzle_tensor",
            "tp_col_swizzle_mc_weight",
            "tp_row_swizzle_mc_weight",
        }
    ]
    canonical = {"torch": torch, "Integral": Integral, "NUM_CORES": 8}
    exec(compile(ast.Module(weight_nodes, []), "weights.py", "exec"), canonical)
    return namespace, canonical


@pytest.mark.parametrize("dwidth", [1, 2], ids=["w8a16", "fp16"])
def test_action_resident_pack_matches_canonical_tp8_swizzle_bytes(dwidth):
    pack, canonical = _load_pack_helpers()
    layout = pack["_ACTION_WEIGHT_LAYOUT"]
    dtype = torch.int8 if dwidth == 1 else torch.float16
    weights = [
        ((torch.arange(n * k, dtype=torch.int64) % 257) - 128)
        .to(dtype).reshape(n, k)
        for _, _, (n, k), _ in layout
    ]
    pack_bundle = pack[
        "_pack_action_resident_weight_bundle" if dwidth == 1
        else "_pack_action_fp16_resident_weight_bundle"
    ]
    bundle = pack_bundle(*weights)
    resident_layer_count = 4 if dwidth == 1 else 2
    assert pack["_ACTION_WEIGHT_BYTES_PER_CORE"] == resident_layer_count * bundle.size(1)

    for weight, (_, offset, _, partition) in zip(weights, layout):
        n, k = weight.shape
        local_n = n // 8 if partition == 1 else n
        local_k = k // 8 if partition == 0 else k
        cell_k = 32 // dwidth
        swizzle = canonical[
            "tp_col_swizzle_mc_weight"
            if partition == 1 else "tp_row_swizzle_mc_weight"
        ](weight, 8, dwidth)
        expected = (
            swizzle.view(local_n // 16, local_k // cell_k, 8, 16, cell_k)
            .permute(2, 0, 1, 3, 4).contiguous().view(torch.int8)
            .reshape(8, -1)
        )
        begin = offset * dwidth
        actual = bundle[:, begin:begin + expected.size(1)]
        assert torch.equal(actual, expected)

    with pytest.raises(ValueError, match="expected (7|seven)"):
        pack_bundle(*weights[:-1])
    with pytest.raises(ValueError, match="resident q weight|invalid FP16 resident q"):
        pack_bundle(weights[0][:-1], *weights[1:])
    if dwidth == 1:
        with pytest.raises(ValueError, match="contiguous CPU int8"):
            pack["_compact_w8_weight_per_core"](weights[0].float(), 1)


def test_four_regular_layers_share_one_core_major_preload():
    assert "L in resident_layers" in ACTION
    assert "torch.cat(resident_layer_bundles, dim=1)" in ACTION
    assert "WALL_ACTION_FIRST_RESIDENT_LAYER = 1" in DECODER
    assert "WALL_ACTION_RESIDENT_LAYER_COUNT = 4" in DECODER
    assert "WALL_ACTION_WEIGHT_BYTES_PER_LAYER = 0x180000" in DECODER
    assert "layer_idx - WALL_ACTION_FIRST_RESIDENT_LAYER" in DECODER
    for name in ("Q", "K", "V", "O", "GATE", "UP", "DOWN"):
        assert f"wall_action_weight_base + WALL_ACTION_{name}_OFFSET" in DECODER
    assert "A(WALL_ACTION_WEIGHT_BYTES_PER_CORE), 0, 9" in ACTION_MODEL
    assert "core_stride_bytes=*/WALL_ACTION_WEIGHT_BYTES_PER_CORE" in ACTION_MODEL
    assert "WALL_ACTION_WEIGHT_BYTES_PER_CORE" in ACTION_MODEL[
        ACTION_MODEL.index("void WallOssActionStepModel::set_weight_residency("):
        ACTION_MODEL.index("// ──", ACTION_MODEL.index("void WallOssActionStepModel::set_weight_residency("))
    ]


def test_spm_weight_kernels_are_optional_expansion_cache_entries():
    assert_current_catalog("SharedRawSpmWeight", "ActionRingPostnormSpmWeight")
    assert '"wall_action_qkv_w8_compact_partial_mrope_v2"' in CACHE_H
    assert '"wall_action_qkv_w8_compact_partial_mrope_v2"' in CACHE_INC


def test_m32_resident_launchers_keep_cold_and_unsupported_payload_guards(tmp_path):
    """Exercise the production launcher admission before any device/DDR access."""
    from test_rmsnorm_capability_publication_host import _compile_and_run

    def admission(name, end):
        start = LINEAR.index(f"void {name}(")
        begin = LINEAR.index("  const bool fp16 =", start)
        return LINEAR[begin:LINEAR.index(end, begin)]

    gate = admission("rpu_launch_wall_action_gate_up_w8a16_multibase_spm_kernel",
                     "  auto check_weight =")
    qkv = admission("rpu_launch_wall_action_qkv_w8a16_multibase_spm_kernel",
                    "  auto check_weight =")
    row = admission("rpu_launch_wall_action_linear_ring_postnorm_spm_kernel",
                    "  if (use_spm_weight) {")
    code = f'#include "{ROOT / "src/core/rpu_wall_sorting_config.h"}"\n' + r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while(0)
constexpr int NUM_CORES=8;
namespace c10 { struct Half { uint16_t bits; }; }
namespace at {
enum ScalarType { kHalf,kChar };
struct Tensor { ScalarType dtype; ScalarType scalar_type() const { return dtype; } };
}
void gate_guard(int64_t rows,const WallSortingConfig& sorting,bool fp16_weights,
                uint32_t gate_weight_spm_addr,uint32_t up_weight_spm_addr) {
    at::Tensor gate_weight{fp16_weights?at::kHalf:at::kChar};
''' + gate + r'''
}
void qkv_guard(int64_t rows,const WallSortingConfig& sorting,bool fp16_weights,
               uint32_t q_weight_spm_addr,uint32_t k_weight_spm_addr,uint32_t v_weight_spm_addr,
               bool kv_owner_fanout=false) {
    at::Tensor q_weight{fp16_weights?at::kHalf:at::kChar};
''' + qkv + r'''
}
void row_guard(int64_t rows,const WallSortingConfig& sorting,bool fp16_weights,
               uint32_t weight_spm_addr) {
    at::Tensor weight{fp16_weights?at::kHalf:at::kChar};
''' + row + r'''
}
template<class F> void rejects(F run) {
    bool rejected=false;try{run();}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
}
int main() {
    for(bool fp16:{false,true}) for(int64_t rows:{20,32}) {
        WallSortingConfig sorting;sorting.fp16_transfer_mask=fp16?4:0;
        sorting.silu_mul=rows==32;
        for(bool resident:{false,true}) {
            const uint32_t q=resident?0x100000:0,k=resident?0x140000:0,v=resident?0x160000:0;
            gate_guard(rows,sorting,fp16,q,k);
            qkv_guard(rows,sorting,fp16,q,k,v);
            row_guard(rows,sorting,fp16,q);
        }
        // Existing M20 activation-reuse remains selectable. Its M32 layout
        // has not been admitted; reject it before selecting that fixed blob.
        sorting.action_activation_reuse=true;
        if(rows==20) gate_guard(rows,sorting,fp16,0x100000,0x140000);
        else rejects([&]{gate_guard(rows,sorting,fp16,0x100000,0x140000);});
        sorting.action_activation_reuse=false;
        rejects([&]{gate_guard(rows,sorting,fp16,0x100000,0);});
        rejects([&]{qkv_guard(rows,sorting,fp16,0x100000,0x140000,0);});
        if(rows==32) {
            rejects([&]{qkv_guard(rows,sorting,fp16,0,0,0,true);});
            sorting.silu_mul=false;
            rejects([&]{gate_guard(rows,sorting,fp16,0x100000,0x140000);});
            rejects([&]{qkv_guard(rows,sorting,fp16,0x100000,0x140000,0x160000);});
            rejects([&]{row_guard(rows,sorting,fp16,0x100000);});
        }
        if(fp16) {
            sorting.silu_mul=true;sorting.fp16_transfer_mask=0;
            rejects([&]{gate_guard(rows,sorting,true,0x100000,0x140000);});
            rejects([&]{qkv_guard(rows,sorting,true,0x100000,0x140000,0x160000);});
            rejects([&]{row_guard(rows,sorting,true,0x100000);});
        }
    }
}
'''
    _compile_and_run(tmp_path, "wall_action_m32_resident_guards", code)


def test_gate_up_spm_weights_are_raw_addresses_and_only_scales_use_ddr():
    launcher=_function(LINEAR,"void rpu_launch_wall_action_gate_up_w8a16_multibase_spm_kernel(")
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 4
    assert launcher.count("GraphDdrRegisterRole::PerChannelScale") == 2
    assert launcher.count("GraphDdrRegisterRole::Int8ModelWeight") == 2
    for contract in ("if (!use_spm_weight)", "set_u32_reg_pair(*kernel, 2, gate_weight_spm_addr);",
     "set_u32_reg_pair(*kernel, 32, up_weight_spm_addr);", "Operation::SharedRawSpmWeight",
     "Operation::SharedRaw", "set_regs(36", "{selected.grid_x, selected.grid_y, 1}"):
        assert contract in launcher


def test_qkv_spm_weights_preserve_partial_mrope_registers():
    launcher=_function(LINEAR,"void rpu_launch_wall_action_qkv_w8a16_multibase_spm_kernel(")
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 8
    assert launcher.count("GraphDdrRegisterRole::PerChannelScale") == 3
    assert launcher.count("GraphDdrRegisterRole::SemanticInput") == 2
    assert launcher.count("GraphDdrRegisterRole::Int8ModelWeight") == 3
    assert "if (!use_spm_weight)" in launcher
    for register,value in ((2,"q_weight_spm_addr"),(32,"k_weight_spm_addr"),(40,"v_weight_spm_addr")):
        assert f"set_u32({register}, {value});" in launcher
    for slot in ("48, 49","50, 51"):
        assert "DevAddrShift8LoHi, "+slot in launcher
    assert "supports only fused partial M-RoPE" in launcher
    assert "reg = 128; reg <= 255" in launcher
    assert "queue->set_broadcast_mode(true);" in launcher
    assert "rpu_wall_action_qkv::grid(kernel_id)" in launcher


def test_row_spm_weight_preserves_ring_abi_and_only_scale_uses_ddr():
    launcher=_function(LINEAR,"void rpu_launch_wall_action_linear_ring_postnorm_spm_kernel(")
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 2
    assert launcher.count("GraphDdrRegisterRole::PerChannelScale") == 1
    assert launcher.count("GraphDdrRegisterRole::Int8ModelWeight") == 1
    for contract in ("if (!use_spm_weight)", "set_u32_reg_pair(*kernel, 2, weight_spm_addr);",
     "Operation::ActionRingPostnormSpmWeight", "constexpr uint32_t kScm = 4096;",
     "queue->set_broadcast_mode(true);", "{selected.grid_x, selected.grid_y, 1}"):
        assert contract in launcher
    for slot in (32,33,34,36,38,40,42,44,45):
        assert f"({slot}," in launcher or f", {slot}," in launcher

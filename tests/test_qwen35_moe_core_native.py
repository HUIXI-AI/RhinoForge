"""Board-free native contracts for Qwen3.5-35B-A3B TP4/TP6/TP8 ownership."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "src/fused/rpu_qwen3_5_moe_model.cpp"
INCLUDE_ROOT = (
    ROOT / "src"
    if (ROOT / "src/core/execution_topology.h").is_file()
    else ROOT.parent / "source/src"
)


def test_planner_cache_identity_is_registered_for_python_admission():
    """The first prefill queries this dispatcher ABI before executing kernels."""
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text(
        encoding="utf-8"
    )
    assert re.search(
        r'm\.def\("qwen3_5_moe_planner_cache_identity\(int handle\) -> int\[\]",'
        r'\s*TORCH_FN\(rpu_qwen3_5_moe_planner_cache_identity\)\);',
        registrations,
    ), "MoE planner cache identity must be exported through torch.ops.rpu"


def _masked(source: str) -> str:
    return re.sub(
        r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
        lambda match: " " * len(match[0]),
        source,
    )


def _definition(source: str, marker: str) -> str:
    masked = _masked(source)
    start = source.index(marker)
    line_start = source.rfind("\n", 0, start) + 1
    brace = masked.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[line_start:end]


def _scope(source: str, marker: str) -> str:
    return _definition(source, marker)


def _call_arguments(source: str, call: str) -> list[list[str]]:
    masked = _masked(source)
    found = []
    for match in re.finditer(rf"\b{re.escape(call)}\s*\(", masked):
        begin = masked.index("(", match.start())
        depth, end = 1, begin + 1
        while depth:
            depth += (masked[end] == "(") - (masked[end] == ")")
            end += 1
        raw = source[begin + 1 : end - 1]
        raw_masked = masked[begin + 1 : end - 1]
        args, arg_start, nested = [], 0, 0
        for index, char in enumerate(raw_masked):
            if char in "([{<":
                nested += 1
            elif char in ")]}>":
                nested -= 1
            elif char == "," and nested == 0:
                args.append(raw[arg_start:index].strip())
                arg_start = index + 1
        args.append(raw[arg_start:].strip())
        found.append(args)
    return found


def _run_cpp(tmp_path: Path, source: str) -> None:
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    cpp = tmp_path / "probe.cpp"
    exe = tmp_path / "probe"
    cpp.write_text(source, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", "-O0", "-I", str(INCLUDE_ROOT),
         str(cpp), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


def test_real_cold_topology_methods_admit_fixed_tp4_tp6_tp8(tmp_path):
    source = NATIVE.read_text(encoding="utf-8")
    methods = "\n\n".join(
        _definition(source, marker)
        for marker in (
            "void Qwen3_5MoeModel::set_execution_cores(",
            "std::vector<int64_t> Qwen3_5MoeModel::execution_topology() const",
            "std::vector<int64_t> Qwen3_5MoeModel::execution_topology_v2() const",
            "DecoderExecutionTopology Qwen3_5MoeModel::resolve_model_execution_topology(",
            "std::vector<int64_t> Qwen3_5MoeModel::cold_topology_arguments() const",
        )
    )
    probe = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "core/execution_topology.h"
#define NUM_CORES 8
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
namespace at {
struct Tensor { bool present=false; bool defined() const { return present; } };
}
namespace v3 {
struct FusedModelBase {
    int owner=8, attention=8, mlp=8;
    int64_t layers=0;
    virtual DecoderExecutionTopology resolve_model_execution_topology(
        int64_t, int64_t nkv, int64_t, int64_t, int64_t) const {
        return {8, std::min(8, static_cast<int>(nkv)), 8};
    }
    void set_execution_core_count(int value) { owner=value; }
    int num_cores() const { return owner; }
    int attn_tp() const { return attention; }
    int mlp_tp() const { return mlp; }
    int64_t num_layers() const { return layers; }
};
struct Qwen3_5MoeModel : FusedModelBase {
    std::vector<int> layer_weights_, moe_layer_weights_;
    at::Tensor expert_ids_, token_ids_, router_bridge_;
    int gdn_tp() const { return num_cores() == 6 ? 4 : num_cores(); }
    int router_tp() const { return num_cores() == 6 ? 4 : num_cores(); }
    int routing_tp() const { return num_cores(); }
    int routed_expert_tp() const { return num_cores(); }
    int shared_expert_tp() const { return num_cores(); }
    int lm_head_tp() const { return num_cores() == 6 ? 4 : num_cores(); }
    void set_execution_cores(int64_t);
    std::vector<int64_t> execution_topology() const;
    std::vector<int64_t> execution_topology_v2() const;
    DecoderExecutionTopology resolve_model_execution_topology(
        int64_t, int64_t, int64_t, int64_t, int64_t) const override;
    std::vector<int64_t> cold_topology_arguments() const;
};
__METHODS__
template <class F> void rejects(F call) {
    bool rejected=false;
    try { call(); } catch (const std::runtime_error&) { rejected=true; }
      catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
}
int run_probe() {
    for (int bad : {0,1,2,3,5,7,9}) {
        Qwen3_5MoeModel model;
        rejects([&] { model.set_execution_cores(bad); });
        assert(model.num_cores() == 8);
    }
    for (int owner : {4,6,8}) {
        Qwen3_5MoeModel model;
        model.set_execution_cores(owner);
        const int effective_kv = owner == 8 ? 8 : 4;
        const auto resolved = model.resolve_model_execution_topology(
            16, effective_kv, 256, 2048, 512);
        const int narrow = owner == 6 ? 4 : owner;
        assert(resolved.num_cores == owner);
        assert(resolved.attention_tp == narrow);
        assert(resolved.mlp_tp == owner);
        assert(resolved.physical_kv_cores == 8);
        model.attention=resolved.attention_tp;
        model.mlp=resolved.mlp_tp;
        model.layers=40;
        model.layer_weights_.resize(40);
        const auto topology=model.execution_topology();
        assert((topology == std::vector<int64_t>{owner,narrow,owner,narrow,8}));
        const auto topology_v2=model.execution_topology_v2();
        assert((topology_v2 == std::vector<int64_t>{
            2,owner,narrow,narrow,narrow,owner,owner,owner,narrow,8}));
        const auto cold=model.cold_topology_arguments();
        if (owner == 6) assert(cold == topology_v2);
        else assert((cold == std::vector<int64_t>{
            1,owner,owner,owner,owner,owner,8}));
        rejects([&] { model.set_execution_cores(owner); });
    }
    Qwen3_5MoeModel tensor_hot;
    tensor_hot.expert_ids_.present=true;
    rejects([&] { tensor_hot.set_execution_cores(4); });
    Qwen3_5MoeModel bridge_hot;
    bridge_hot.router_bridge_.present=true;
    rejects([&] { bridge_hot.set_execution_cores(6); });
    for (int axis=0; axis<5; ++axis) {
        Qwen3_5MoeModel model;
        model.set_execution_cores(6);
        int64_t args[5]={16,4,256,2048,512};
        ++args[axis];
        rejects([&] { (void)model.resolve_model_execution_topology(
            args[0],args[1],args[2],args[3],args[4]); });
    }
    return 0;
}
} // namespace v3
int main() { return v3::run_probe(); }
'''.replace("__METHODS__", methods)
    _run_cpp(tmp_path, probe)


def test_exact_install_binds_logical_and_physical_dimensions_to_cold_owner():
    source = NATIVE.read_text(encoding="utf-8")
    install = _scope(source, "void Qwen3_5MoeModel::set_weights(")
    admission = install[: install.index("const auto check_list")]
    assert "const int64_t effective_kv_heads = num_cores() == 8 ? 8 : 4;" in admission
    assert "num_cores() == 4 || num_cores() == 6 || num_cores() == 8" in admission
    assert "num_kv_heads == effective_kv_heads" in admission
    assert "routed_intermediate == 512 && shared_intermediate == 512" in admission
    assert "num_cores() == 6 ? 576 : routed_intermediate" in admission
    assert "num_cores() == 6 ? 576 : shared_intermediate" in admission
    assert "tensor_parallel == num_cores()" in admission
    assert "routed_weight_mode == 5" in admission
    assert "{num_experts, hidden_size * physical_routed_intermediate}" in install
    assert "{physical_shared_intermediate, hidden_size}" in install
    assert "{hidden_size, physical_shared_intermediate}" in install
    assert "{16 * shared_expert_tp(), hidden_size}" in install
    assert install.index("at::Tensor next_router_bridge") < install.index(
        "set_chunk_envelope("
    )
    assert install.index("router_bridge_ = next_router_bridge") > install.index(
        "has_qk_norm_ ="
    )
    assert install.index("set_model_params(") < install.index("set_num_layers(N)")
    assert install.rstrip().endswith(
        "invalidate_model_state();   // D-503: MUST be the last statement.\n}"
    )


def test_owner_sensitive_launches_bind_the_mixed_domains_explicitly():
    source = NATIVE.read_text(encoding="utf-8")
    scopes = {
        "layer": _scope(source, "void Qwen3_5MoeModel::build_layer_subgraph("),
        "attention": _scope(source, "void Qwen3_5MoeModel::build_full_attention("),
        "gdn": _scope(source, "void Qwen3_5MoeModel::build_gdn("),
        "tail": _scope(source, "void Qwen3_5MoeModel::emit_mlp_and_output("),
        "router": _scope(source, "void Qwen3_5MoeModel::emit_router_and_shuffle("),
        "shared": _scope(source, "void Qwen3_5MoeModel::emit_shared_expert("),
        "routed": _scope(source, "void Qwen3_5MoeModel::emit_routed_experts("),
        "merge": _scope(source, "void Qwen3_5MoeModel::emit_moe_merge_and_residual("),
        "buffers": _scope(source, "std::vector<BufferDecl> Qwen3_5MoeModel::declare_buffers("),
    }
    assert "const int     nc = NUM_CORES" not in source
    assert "static_cast<int>(moe_.tensor_parallel)" not in source
    assert "rpu_launch_eltwise_mul_spm_kernel" not in scopes["attention"]

    for scope_name in ("layer", "attention", "gdn", "tail"):
        for args in _call_arguments(scopes[scope_name], "rpu_launch_rmsnorm_spm_kernel"):
            assert len(args) == 8, (scope_name, args)

    attention_binary = _call_arguments(
        scopes["attention"], "rpu_launch_eltwise_binary_spm_kernel"
    )
    assert len(attention_binary) == 1 and len(attention_binary[0]) == 7
    assert attention_binary[0][-1] == "auxiliary_cores"
    prepare = _call_arguments(
        scopes["attention"], "rpu_prepare_ring_all_reduce_input"
    )
    assert len(prepare) == 1 and prepare[0][-2:] == ["tp", "num_cores()"]

    expected_arity = {
        "rpu_launch_eltwise_binary_scalar_spm_kernel": 6,
        "rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel": 9,
        "rpu_launch_eltwise_binary_1xC_NxC_spm_kernel": 9,
        "rpu_launch_eltwise_binary_Bx1xC_BxNxC_spm_kernel": 10,
    }
    for call, arity in expected_arity.items():
        calls = _call_arguments(scopes["gdn"], call)
        assert calls, call
        assert all(len(args) == arity and args[-1] == "nc" for args in calls)
    gdn_prepare = _call_arguments(
        scopes["gdn"], "rpu_prepare_ring_all_reduce_input"
    )
    assert len(gdn_prepare) == 1
    assert gdn_prepare[0][-2:] == ["nc", "num_cores()"]

    assert "const int router_nc = router_tp();" in scopes["router"]
    assert "const int route_nc = routing_tp();" in scopes["router"]
    assert "rpu_launch_spm_copy_ddr_dma(" in scopes["router"]
    assert "rpu_launch_ddr_broadcast_spm_dma(" in scopes["router"]
    assert 'addr(0, "moe_route_logits")' in scopes["router"]
    assert "const int64_t si = physical_shared_intermediate();" in scopes["shared"]
    assert "const int nc = shared_expert_tp();" in scopes["shared"]
    assert "physical_routed_intermediate() / routed_expert_tp()" in scopes["routed"]
    assert "const int nc = routed_expert_tp();" in scopes["routed"]
    assert "const int nc = routing_tp();" in scopes["merge"]

    broadcasts = _call_arguments(source, "rpu_launch_ddr_broadcast_spm_dma")
    assert broadcasts
    assert all(len(args) in (4, 5) for args in broadcasts)
    assert all(
        "num_cores()" in args[-1] or args[-1] in {"nc", "tp", "route_nc"}
        for args in broadcasts
    )


def test_complete_manifest_and_identity_publish_full_mixed_topology():
    source = NATIVE.read_text(encoding="utf-8")
    build = _scope(source, "void Qwen3_5MoeModel::build_layer_subgraph(")
    manifest = _scope(
        source, "Qwen3_5MoeModel::physical_manifest_for_candidate("
    ) + _scope(source, "void Qwen3_5MoeModel::append_mlp_manifest_routes(")
    layout = _scope(source, "int64_t Qwen3_5MoeModel::subclass_layout_hash(")
    identity = _scope(
        source, "std::vector<int64_t> Qwen3_5MoeModel::kvinsert_cost_weight_identity("
    )
    assert "QWEN35_MOE_CORE_TOPOLOGY_SITE" in build
    assert "num_cores() != 8 && chunk.idx == 0" in build
    assert "cold_topology_arguments()" in build
    assert "exact_profile_arguments()" in build
    assert "QWEN35_MOE_CORE_TOPOLOGY_SITE" in manifest
    assert "QWEN35_MOE_ROUTER_BRIDGE_STORE_SITE" in manifest
    assert "QWEN35_MOE_ROUTER_BRIDGE_LOAD_SITE" in manifest
    assert "QWEN35_MOE_GDN_PREPARE_ALL_REDUCE_SITE" in manifest
    assert "physical_routed_intermediate() / routed_expert_tp()" in manifest
    assert "text_ring_route(chunk.len, hidden_size())" in manifest
    assert "/*compute_row_multiplier=*/1, num_cores(), mlp_tp()" in manifest
    assert "cold_topology_arguments()" in layout
    assert "physical_routed_intermediate()" in layout
    assert "physical_shared_intermediate()" in layout
    assert "identity.insert(identity.end(), topology.begin(), topology.end())" in identity
    assert "append_kvinsert_cost_tensor_identity(identity, router_bridge_)" in identity
    sdpa = _call_arguments(
        _scope(source, "void Qwen3_5MoeModel::build_full_attention("),
        "rpu_launch_sdpa_spm_dispatch",
    )
    assert len(sdpa) == 1
    assert sdpa[0][-2:] == ["tp", "NUM_CORES"]

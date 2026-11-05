"""Board-free contracts for the Qwen3.5 35B E4M3 native integration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT
    if (ROOT / "src/graph/graph_runtime.h").is_file()
    else ROOT.parent / "source"
)


def read(relative: str) -> str:
    draft = ROOT / relative
    return (draft if draft.is_file() else SOURCE_ROOT / relative).read_text(
        encoding="utf-8"
    )


def test_prefill_rope_dispatch_preserves_public_tensor_inputs(tmp_path, monkeypatch):
    """Real wrapper/Dispatcher route; full setter behavior has its own fixture."""
    import re
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    from test_siglip_attention_dry_live_contract import _function, _run_native

    source = read("src/fused/rpu_qwen3_5_moe_model.cpp")
    dispatch = read("src/core/rpu_dispatch_registrations.inc")
    wrapper = _function(source, "void rpu_qwen3_5_moe_set_prefill_rope(")
    schema = re.search(r'  m.def\("qwen3_5_moe_set_prefill_rope\(.*?;', dispatch).group()
    registrations = []
    for key in ("CPU", "PrivateUse1"):
        block = _function(dispatch, f"TORCH_LIBRARY_IMPL(rpu, {key}, m)")
        lines = re.findall(r'  m.impl\("qwen3_5_moe_set_prefill_rope".*?;', block, re.S)
        assert len(lines) == 1
        registrations.append(f"TORCH_LIBRARY_IMPL(moe_rope_host,{key},m){{\n"
                             + lines[0] + "\n}")
    cpp = r'''
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/InferenceMode.h>
#include <torch/library.h>
#include <cassert>
struct Model {
 at::Tensor cos,sin;int calls=0;
 void set_prefill_rope(const at::Tensor& c,const at::Tensor& s){cos=c;sin=s;++calls;}
};
struct Qwen3_5MoeRegistry {
 static Model* get(int64_t handle,const char*){
  TORCH_CHECK(handle==71,"invalid handle");static Model model;return &model;
 }
};
__WRAPPER__
TORCH_LIBRARY(moe_rope_host,m){__SCHEMA__}
__REGISTRATIONS__
int main(){
 auto op=c10::Dispatcher::singleton().findSchemaOrThrow("moe_rope_host::qwen3_5_moe_set_prefill_rope","")
  .typed<void(int64_t,const at::Tensor&,const at::Tensor&)>();
 assert(op.hasKernelForDispatchKey(c10::DispatchKey::CPU));
 assert(op.hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1));
 auto* model=Qwen3_5MoeRegistry::get(71,"");
 for(bool inference:{false,true}){
  c10::InferenceMode mode(inference);
  auto c=at::zeros({66,32},at::kHalf).narrow(0,1,64);
  auto s=at::ones_like(c);auto calls=model->calls;
  const auto version=inference?-1:c._version();
  for(int value:{7,-3,7}){
   c.data_ptr<c10::Half>()[0]=c10::Half(value);op.call(71,c,s);
   assert(model->cos.is_same(c)&&model->sin.is_same(s));
   assert(float(model->cos.data_ptr<c10::Half>()[0])==value);
   assert((inference?-1:c._version())==version);
  }
  assert(model->calls==calls+3);
  bool failed=false;try{op.call(99,c,s);}catch(const c10::Error&){failed=true;}
  assert(failed&&model->calls==calls+3);
 }
}
'''
    cpp = (cpp.replace("__WRAPPER__", wrapper).replace("__SCHEMA__", schema)
           .replace("__REGISTRATIONS__", "\n".join(registrations)))
    _run_native(tmp_path, cpp, torch)


def test_fp8_ddr_role_is_append_only_and_typed():
    runtime = read("src/graph/graph_runtime.h")
    census = read("src/graph/graph_register_census.cpp")
    dump = read("src/graph/graph_runtime_dump.cpp")
    assert runtime.index("PerChannelScale = 6") < runtime.index(
        "Fp8ModelWeight = 7"
    )
    assert "case GraphDdrRegisterRole::Fp8ModelWeight:" in census
    assert "return dtype == at::kByte;" in census
    assert "Fp8ModelWeight" not in dump
    assert "safe_node_summary" in dump


def test_ordinary_linear_keeps_full_byte_e4m3_separate_from_packed_4bit():
    linear = read("src/ops/rpu_linear.cpp")
    full_byte = linear.index("const bool fp8_e4m3_weight")
    packed_byte = linear.index("if (weight.scalar_type() == at::kByte)", full_byte)
    assert "weight.size(0) == N && weight.size(1) == K" in linear[
        full_byte:packed_byte
    ]
    assert "GraphDdrRegisterRole::Fp8ModelWeight" in linear[
        full_byte:packed_byte
    ]
    assert "GraphDdrRegisterRole::PerChannelScale" in linear[
        full_byte:packed_byte
    ]
    assert "static_cast<uint16_t>(is_fp8_e4m3 ? 5 : 1)" in linear[
        full_byte:packed_byte
    ]
    packed = linear[packed_byte:]
    assert "weight.size(1) == K / 2" in packed
    assert "scale.size(0) == K / 16 && scale.size(1) == N" in packed


def test_grouped_mode5_uses_typed_weight_and_scale_registers():
    grouped = read("src/ops/rpu_grouped_parallel_linear.cpp")
    assert "only mode 1(int8), 2(fp16), or 5(fp8_e4m3)" in grouped
    assert "GraphDdrRegisterRole::Fp8ModelWeight" in grouped
    assert "GraphDdrRegisterRole::PerChannelScale" in grouped
    assert "stage_kernel_ddr_registers(" in grouped
    assert "2, 3" in grouped
    assert "20, 21" in grouped
    staged = grouped.index("stage_kernel_ddr_registers(")
    required_kernel = grouped.index("if (!kernel) kernel = GET_KERNEL(kernel_id)", staged)
    assert staged < required_kernel < grouped.index("register_writer.write(*kernel)", staged)
    launcher = grouped[grouped.index("static void launch_grouped_parallel_linear_spm("):grouped.index("void rpu_launch_grouped_parallel_linear_spm_kernel(")]
    assert "RpuGetDevAddr" not in launcher


def test_routing_kernel_ids_names_declarations_and_sources_are_complete():
    cache = read("src/core/rpu_kernel_cache.h")
    cache_impl = read("src/core/rpu_kernel_cache.inc")
    decls = read("src/core/rpu_kernel_decls.h")
    cmake = read("src/CMakeLists.txt")
    ids_and_names = {
        "GROUPED_PARALLEL_LINEAR_ACC16": "grouped_parallel_linear_acc16",
        "GROUPED_PARALLEL_LINEAR_8BIT_ACC16":
            "grouped_parallel_linear_w8a16_acc16",
        "MOE_TOPK_SCALAR_FP16": "topk_scalar_fp16",
        "MOE_TOPK_BY_KVSORT_FP16": "topk_by_kvsort_fp16",
        "MOE_BINCOUNT_SCALAR": "bincount_scalar",
        "MOE_TOPK_PROB_NORM_FP16": "llm_topk_prob_norm",
        "CAST_UINT16_INT32": "unary_cast_uint16_int32",
        "GATHER_SCALAR_FP16": "gather_scalar",
        "GATHER_FP16": "gather",
        "MOE_GATHER_MUL_V2_FP16": "llm_gather_mul_v2",
        "SCATTERND_ADD_INT16_FP16": "scatternd_int16_reduction",
    }
    optional_kernel_body = cache_impl[
        cache_impl.index("static bool is_optional_kernel") :
        cache_impl.index("const std::unordered_set<std::string>& ",
                         cache_impl.index("static bool is_optional_kernel"))
    ]
    for kernel_id, symbol in ids_and_names.items():
        assert kernel_id in cache
        assert cache.count(f'"{symbol}"') == 1
        assert cache_impl.count(
            f'{{"{symbol}", RHINO_OP_LIB_PATH}}'
        ) == 1
        assert f'"{symbol}"' not in optional_kernel_body

    launchers = (
        "rpu_launch_softmax_c16_spm_kernel",
        "rpu_launch_grouped_parallel_linear_spm_kernel",
        "rpu_launch_moe_topk_scalar_fp16_spm_kernel",
        "rpu_launch_moe_topk_by_kvsort_fp16_spm_kernel",
        "rpu_launch_moe_bincount_scalar_spm_kernel",
        "rpu_launch_moe_topk_prob_norm_fp16_spm_kernel",
        "rpu_launch_cast_uint16_int32_spm_kernel",
        "rpu_launch_gather_fp16_spm_kernel",
        "rpu_launch_moe_gather_mul_v2_fp16_spm_kernel",
        "rpu_launch_scatternd_add_u16_fp16_spm_kernel",
    )
    for launcher in launchers:
        assert launcher in decls

    for source in (
        "rpu_gather.cpp",
        "rpu_grouped_parallel_linear.cpp",
        "rpu_moe_gather_mul.cpp",
        "rpu_moe_index_shuffle.cpp",
        "rpu_moe_topk_prob_norm.cpp",
        "rpu_scatternd_add.cpp",
    ):
        assert f'ops/{source}"' in cmake
    assert 'fused/rpu_qwen3_5_moe_model.cpp"' in cmake


def test_dispatch_registration_exposes_only_stable_moe_abi():
    registrations = read("src/core/rpu_dispatch_registrations.inc")
    required = (
        "qwen3_5_moe_planner_cache_identity",
        "qwen3_5_moe_bind_kvinsert_costs",
        "qwen3_5_moe_kvinsert_cost_catalog_sha256",
        "qwen3_5_moe_kvinsert_exact_candidate",
        "qwen3_5_moe_kvinsert_cost_domain",
        "qwen3_5_moe_create",
        "qwen3_5_moe_set_execution_cores",
        "qwen3_5_moe_get_execution_topology",
        "qwen3_5_moe_get_execution_topology_v2",
        "qwen3_5_moe_destroy",
        "qwen3_5_moe_set_weights",
        "qwen3_5_moe_set_prefill_rope",
        "qwen3_5_moe_set_valid_prefill_len",
        "qwen3_5_moe_set_mrope_position_delta",
        "qwen3_5_moe_set_chunk_size_cap",
        "qwen3_5_moe_set_prefill_chunk_size",
        "qwen3_5_moe_set_linear_acc32",
        "qwen3_5_moe_set_retained_prefill_graph",
        "qwen3_5_moe_resolve_prefill_chunk_size",
        "qwen3_5_moe_resolve_prefill_stage_domain",
        "qwen3_5_moe_resolve_decode_stage_descriptor",
        "qwen3_5_moe_forward",
    )
    assert len(required) == 22
    for name in required:
        assert f'"{name}' in registrations
    for forbidden in (
        "qwen3_5_moe_action",
        "qwen3_5_moe_spm_z2",
        "qwen3_5_moe_debug",
        "qwen3_5_moe_reduced",
    ):
        assert forbidden not in registrations
    assert (
        "int planned_chunk_size=0, int[] planned_stage_descriptor=[]"
        in registrations
    )
    assert (
        "int execution_len, int logical_len, "
        "int planning_chunk_size_override=-1"
        in registrations
    )


def test_installed_native_owner_binds_its_prefill_admission_before_weights():
    native = read("src/fused/rpu_qwen3_5_moe_model.cpp")
    install = native.split("void Qwen3_5MoeModel::set_weights(", 1)[1]
    install = install.split("c10::optional<at::Tensor>", 1)[0]
    # Even a rejected installation invalidates the previous CPU publication.
    assert install.index("prefill_rope_cpu_published_ = false;") < install.index("TORCH_CHECK(")
    bind = install.index("set_chunk_envelope(")
    assert install.index("RPU_MODEL_REJECT:EXACT_PROFILE") < bind
    assert install.index("cos/sin shapes must match") < bind
    assert install.index("same max_chunk as expert_ids") < bind
    assert bind < install.index("layer_is_full_.assign(")
    assert "physical_kv_capacity," in install
    assert "cos.size(0) + total_padding + 15" in install
    assert "std::min<int64_t>(4096 / top_k, expert_ids.size(0))" in install
    assert install.index("invalidate_model_state();") > bind
    valid = native.split("bool Qwen3_5MoeModel::subclass_chunk_size_valid(", 1)[1]
    valid = valid.split("int64_t Qwen3_5MoeModel::subclass_chunk_size_cap(", 1)[0]
    assert "sdpa_is_valid_chunk_size" in valid
    assert "cs * moe_.top_k > 4096" in valid
    assert "chunk_within_envelope(cs)" in valid

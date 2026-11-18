"""Loaded operator roles and native runtime identity remain consistent."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


EXPANSION_REQUIRED = {
    "binary_sameshape_ddr",
    "binary_scalar_ddr",
    "binary_scalar_power_ddr",
    "binary_Nx1_NxC256_batch_ddr",
    "binary_Nx1_NxC256_batch_bopa_ddr",
    "binary_Nx1_NxC_v16_batch_ddr",
    "binary_Nx1_NxC_v16_batch_bopa_ddr",
    "binary_1xC_NxC_v256_batch_ddr",
    "binary_1xC_NxC_v256_batch_bopa_ddr",
    "binary_1xC_NxC_v16_batch_ddr",
    "binary_1xC_NxC_v16_batch_bopa_ddr",
    "unary_ddr",
    "layer_norm_bf16",
    "reduce_mean_last_dim_ddr",
    "llama_rms_norm_bf16_ddr",
    "llama_rope_ddr",
    "memset_spm_multi_core_v2",
}


def test_native_receipt_tracks_only_successfully_loaded_canonical_oplibs():
    header = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    source = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()

    assert "std::vector<std::string> loaded_oplib_paths() const" in header
    assert "std::mutex loaded_oplib_paths_mutex_" in header
    assert "std::set<std::string> loaded_oplib_paths_" in header
    assert source.count("create_with_binary_file(") == 1
    body = source[source.index("CachedKernel KernelCache::load_kernel_from_oplib") :]
    assert body.index("oplib_snapshot(path)") < body.index("create_with_binary_file(")
    assert "snapshot->path(), kernel_name.c_str()" in body
    assert "canonical_payload_path(path)" not in body
    assert body.index("rc == 0") < body.index("loaded_oplib_paths_.insert")
    assert body.index("std::make_unique<::rhino_lkn::Kernel_t>") < body.index(
        "loaded_oplib_paths_.insert"
    )


def test_required_preloads_are_bound_to_one_oplib_role():
    source = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
    start = source.index(
        "static const std::vector<std::pair<std::string, std::string>> KERNEL_LIST"
    )
    preload = source[start : source.index("\n};", start)]

    for name in EXPANSION_REQUIRED:
        assert f'{{"{name}", RHINO_EXPANSION_OP_LIB_PATH}}' in preload
        assert f'{{"{name}", RHINO_OP_LIB_PATH}}' not in preload

    assert "main_path != expansion_path" in source
    assert "kMainRoleCanaries" in source
    assert "lib_path == KERNEL_LIB_PATH_EXPANSION()" in source
    assert "else if (kernel_manifest_names(KERNEL_LIB_PATH_EXPANSION())" not in source
    dynamic = source[source.index("KernelCache::get_kernel(const std::string&") :]
    assert "FROZEN_KERNEL_LIB_PATH()" in dynamic
    assert "FROZEN_KERNEL_LIB_PATH_EXPANSION()" in dynamic
    assert "else if (kernel_manifest_names(expansion_path)" in dynamic
    assert "RHINO_EXPANSION_OP_LIB_PATH" not in dynamic


def test_kernel_ids_keep_their_name_table_and_relative_order():
    header = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    enum_start = header.index("enum class KernelId")
    enum_body = header[enum_start : header.index("_COUNT", enum_start)]
    names_start = header.index("static constexpr const char* KERNEL_ID_NAMES[]")
    names_body = header[names_start : header.index("};", names_start)]

    enum_tail = (
        "WALL_RING_POSTNORM_PAIR_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320X2N2048",
        "FLA_CONV1D_NOINPLACE",
        "QWEN3_5_RMS_NORM_GATED",
        "QWEN3_5_RANK1_FMA",
        "QWEN3_5_MUL_REDUCE_ROWS",
    )
    name_tail = (
        "wall_ring_postnorm_pair_fp16_partial_bf16_residual_local_inout_m320x2n2048",
        "llm_fla_conv1d_noinplace_w16a16_acc16",
        "qwen3_5_rms_norm_gated",
        "qwen3_5_rank1_fma",
        "qwen3_5_mul_reduce_rows",
    )
    enum_body = enum_body.split("PL_AUTOTILE_BEGIN")[0]
    names_body = names_body.split("#define")[0]
    uncommented = re.sub(r"//[^\n]*|/\*.*?\*/", "", enum_body, flags=re.S)
    enum_ids = re.findall(r"\b[A-Z][A-Z0-9_]*\b", uncommented)
    kernel_names = re.findall(r'"([^"]+)"', names_body)
    assert len(enum_ids) == len(kernel_names)
    assert len(set(enum_ids)) == len(enum_ids)
    # Internal KernelId ordinals can move when native families are integrated;
    # every use is rebuilt together. The enum/name correspondence cannot move.
    start = enum_ids.index(enum_tail[0])
    end = start + len(enum_tail)
    assert tuple(enum_ids[start:end]) == enum_tail
    assert tuple(kernel_names[start:end]) == name_tail
    assert tuple(enum_ids[end:end + 2]) == ("LLAMA_GEMV", "LLAMA_GEMV_WINT8")
    assert tuple(kernel_names[end:end + 2]) == ("llama_gemv", "llama_gemv_wint8")


def test_payload_path_binding_reports_actual_extension_sdk_and_oplibs():
    binding = (ROOT / "src/core/rpu_pybind.inc").read_text()
    assert 'm.def("get_runtime_payload_paths"' in binding
    assert 'dlsym(RTLD_DEFAULT, "RpuInit")' in binding
    assert "static_cast<const void*>(&KERNEL_LIST)" in binding
    assert 'result["extension"]' in binding
    assert 'result["sdk"]' in binding
    assert 'result["launch"]' in binding
    assert 'result["oplibs"] = KernelCache::instance().loaded_oplib_paths()' in binding
    assert 'oplib_roles["main"] = FROZEN_KERNEL_LIB_PATH()' in binding
    assert 'oplib_roles["expansion"] = FROZEN_KERNEL_LIB_PATH_EXPANSION()' in binding
    assert 'result["oplib_roles"]' in binding
    assert 'KernelCache::instance().loaded_oplib_bytes()' in binding
    assert 'sha256(py::bytes(bytes))' in binding
    assert 'result["oplib_payloads"]' in binding

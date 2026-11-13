"""Actual cold-profile wrappers/install closures on CPU, never hardware proof."""
import ast
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest
import torch

from test_siglip_cold_profile import _body

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    ("qwen3vl_vision", "Qwen3VLVisionRegistry", "rpu_qwen3vl_vision_model.cpp", 8192),
    ("qwen25vl_vision", "Qwen25VLVisionRegistry", "rpu_qwen25vl_vision_model.cpp", 8192),
    ("dinov3_vision", "DINOv3VisionRegistry", "rpu_dinov3_vision_model.cpp", 8192),
    ("hyvit2", "HYViT2Registry", "rpu_hyvit2_vision_model.cpp", 8192),
)


def _member_method(source, signature):
    start = source.index(signature)
    body = source.index("{", start)
    depth, end = 1, body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.mark.parametrize("family,registry,filename,upper", FAMILIES)
def test_native_profile_intersects_actual_cache_with_original_geometry(
    tmp_path, family, registry, filename, upper
):
    # Reuse the closed CPU ABI doubles, not a fake implementation of the setter.
    tree = ast.parse((ROOT / "tests/test_siglip_cold_profile.py").read_text())
    fixture = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "cpp" for t in node.targets))
    cpp = ast.literal_eval(fixture.value).split("int main() {", 1)[0]
    cpp = cpp.replace("SigLIPRegistry", registry).replace("rpu_siglip_", "rpu_" + family + "_")
    native = (ROOT / "src/fused" / filename).read_text()
    assert native.count("set_chunk_envelope(") == 2  # cold op + unchanged FMB setter only
    constants = re.findall(r"constexpr int64_t \w*VISION_(?:MAX_KEEPALIVE_SEQ|NUM_SPECIAL_TOKENS) = \d+;", native)
    extra_method = ""
    for candidate in ("void set_execution_envelope(", "void set_cache_envelope("):
        if candidate in native:
            extra_method = _member_method(native, candidate)
            break
    at_stub = r'''
namespace at {
constexpr int kShort = 1, kPrivateUse1 = 2;
struct TensorOptions {
    TensorOptions dtype(int) const { return *this; }
    TensorOptions device(int) const { return *this; }
};
struct Tensor {
    int64_t rows = 0;
    bool present = false;
    bool defined() const { return present; }
    int64_t size(int) const { return rows; }
};
inline Tensor zeros(std::initializer_list<int64_t> shape, TensorOptions) {
    return {*shape.begin(), true};
}
}  // namespace at
'''
    cpp = cpp.replace(
        "struct FusedModelBase",
        "\n".join(constants) + at_stub + "\nstruct FusedModelBase",
        1,
    )
    cpp = cpp.replace(
        "    void set_chunk_envelope(int64_t max_kv_len, int64_t chunk);",
        "    at::Tensor position_idx_keepalive_;\n"
        "    int64_t max_keepalive_seq_ = 0;\n"
        "    void set_chunk_envelope(int64_t max_kv_len, int64_t chunk);\n"
        + extra_method,
        1,
    )
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    for marker, source, signature in (
        ("__SETTER__", base, "void FusedModelBase::set_chunk_envelope("),
        ("__INVALIDATE__", base, "void FusedModelBase::invalidate_planner_cache("),
        ("__PROFILE__", base, "std::vector<int64_t> FusedModelBase::installed_profile_identity()"),
        ("__WRAPPER__", native, "void rpu_" + family + "_set_chunk_envelope("),
    ):
        cpp = cpp.replace(marker, _body(source, signature))
    cpp = "#include <algorithm>\n" + cpp + r'''
int main() {
    auto set=[](int64_t capacity, int64_t chunk) { __CALL__(42,capacity,chunk); };
    set(768,0);
    assert(model.installed_profile_identity()[8]==768);
    if (model.position_idx_keepalive_.defined()) {
        reject([&] { set(8192,0); });
        model.position_idx_keepalive_ = {};
    }
    model.impl.kv_cost_candidates_={1}; model.impl.kv_cost_planning_context_=3;
    model.impl.kv_cost_layout_scope_=5; model.impl.kv_cost_snapshot_reason_="READY";
    set(8192,0);
    auto profile=model.installed_profile_identity();
    assert(profile[8]==__UPPER__ && profile[9]==0);
    assert(model.impl.kv_cost_candidates_.empty() && !model.impl.kv_cost_planning_context_);
    assert(model.impl.kv_cost_layout_scope_==0);
    assert(model.impl.kv_cost_snapshot_reason_=="NATIVE_STAGE_ORACLE_REQUIRED");
    reject([&] { set(0,0); }); reject([&] { set(-1,0); });
    if (model.position_idx_keepalive_.defined()) model.position_idx_keepalive_ = {};
    reject([&] { set(768,15); }); reject([&] { __CALL__(43,768,0); });
    assert(model.installed_profile_identity()==profile);
    model.last=16; reject([&] { set(768,0); });
    assert(model.installed_profile_identity()==profile);
    for (auto word : profile) std::cout << word << ' ';
}
'''.replace("__CALL__", "rpu_" + family + "_set_chunk_envelope").replace("__UPPER__", str(upper))
    source, binary = tmp_path / "profile.cpp", tmp_path / "profile"
    source.write_text(cpp)
    built = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
        str(source), "-o", str(binary)], text=True, capture_output=True, timeout=30)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(binary)], text=True, capture_output=True, timeout=5)
    assert ran.returncode == 0, ran.stderr
    assert f'm.def("{family}_set_chunk_envelope", TORCH_FN(rpu_{family}_set_chunk_envelope));' in (
        ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    assert f"void rpu_{family}_set_chunk_envelope(" in (ROOT / "src/core/rpu_kernel_decls.h").read_text()


@pytest.mark.parametrize("packed,w8", [(False, False), (True, False), (True, True)])
def test_real_hyvit_bind_uses_installed_packed_cache_and_cleans_failure(monkeypatch, packed, w8):
    from rpu_backend.adapters.hy_vla import runtime
    tree = ast.parse((ROOT / "python/rpu_backend/adapters/hy_vla/runtime.py").read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HyVlaRunner")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_vit")
    bind = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == "bind")
    value = object()
    cfg = SimpleNamespace(vit_heads=8, vit_head_dim_padded=80, vit_hidden=640,
        vit_inter_padded=2560, proj_dim=1024, vit_eps=1e-5)
    weights = SimpleNamespace(vit={name: [value] for name in (
        "qw kw vw ow f1w f2w n1w n1b n2w n2b qb kb vb ob f1b f2b qws kws vws ows f1ws f2ws").split()},
        vit_proj1_w=value, vit_proj1_b=value, patch_embed_cores=8, vit_patch_gemm_w=value, vit_pos_fused=value)
    weights.vit["qws"] = [value] if w8 else []
    cache = SimpleNamespace(max_seq_len=196 * (3 if packed else 1))
    runner = SimpleNamespace(_native_cold=SimpleNamespace(vit=SimpleNamespace(
        fast_replay=False, fast_replay_preload=False, mask_once=False, kvpad16=False, q_inplace=False)),
        _fused_merger=False, _proj1_in_merger=False)
    namespace = dict(torch=torch, c=cfg, w=weights, cache=cache, self=runner)
    exec(compile(ast.Module(body=[bind], type_ignores=[]), "<actual-hyvit-bind>", "exec"), namespace)
    events = []
    def weights_op(label):
        return lambda *args: events.append(label)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_set_runtime_config", weights_op("config"), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_set_weights", weights_op("fp16"), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_set_weights_w8a16", weights_op("w8"), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_model_set_patch_emb", weights_op("patch"), raising=False)
    def envelope(handle, capacity, chunk):
        assert events[-1] == ("w8" if w8 else "fp16")
        events.append((handle, capacity, chunk))
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_set_chunk_envelope", envelope, raising=False)
    namespace["bind"](42)
    assert events == ["config", "w8" if w8 else "fp16", (42, cache.max_seq_len, 0), "patch"]
    def fail(*args):
        raise RuntimeError("HY envelope refused")
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_set_chunk_envelope", fail, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_all", lambda: events.append("reset"), raising=False)
    sub = runtime._Sub("vit", lambda: 43, lambda h: events.append(("destroy", h)), namespace["bind"], None)
    with pytest.raises(RuntimeError, match="HY envelope refused"):
        sub.open()
    assert sub.handle is None and sub.gc is None and sub._failed
    assert events.count(("destroy", 43)) == 1


def test_wall_vision_builder_threads_actual_cache_capacity_to_pending_helper():
    source = (ROOT / "python/rpu_backend/adapters/wall_oss/vision.py").read_text()
    call, = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_configure_qwen25vl_vision_handle"]
    capacity, = [kw.value for kw in call.keywords if kw.arg == "max_seq_len"]
    assert ast.unparse(capacity) == "cache.max_seq_len"

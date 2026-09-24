"""Run native M-RoPE refresh predicates against real Tensor version semantics."""
import ctypes
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native_refresh(tmp_path_factory):
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    cpp = """
#include <cstdint>
struct Counter {
    bool tracked;
    int64_t version;
    bool enabled() const { return tracked; }
    int64_t current_version() const { return version; }
};
struct Tensor {
    Counter counter;
    uintptr_t address;
    const Tensor* unsafeGetTensorImpl() const { return this; }
    const Counter& version_counter() const { return counter; }
    const void* data_ptr() const { return reinterpret_cast<const void*>(address); }
};
"""
    for name, tensor in (("pos", "pos"), ("cos", "cil"), ("sin", "sil")):
        start = source.index(f"const auto& {name}_version_counter =")
        end = source.index("if (qwen3vl_2b_w8_profile_", start)
        # Compile the real version extraction, content-change predicate, and
        # ordinary branch condition. Only the Tensor/SDK boundary is doubled.
        declarations = source[start:end]
        guard_end = source.index(")) {", end)
        guard = source[end:guard_end]
        ordinary = re.search(r": \((.*)", guard, re.S).group(1)
        cpp += f"""
extern "C" bool refresh_{name}(bool tracked, int64_t version, uintptr_t address,
    int64_t previous_version, uintptr_t previous_address, bool ka_geom_same) {{
    const Tensor {tensor}{{{{tracked, version}}, address}};
    const int64_t ka_last_{name}_src_version_ = previous_version;
    const void* ka_last_{name}_src_ = reinterpret_cast<const void*>(previous_address);
    {declarations}
    return {ordinary};
}}
"""
    folder = tmp_path_factory.mktemp("mrope-version-native")
    path = folder / "refresh.cpp"
    path.write_text(cpp)
    library = folder / "refresh.so"
    compiler = shutil.which("g++")
    assert compiler, "host native contract tests require g++"
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", str(path),
                    "-o", str(library)], check=True, capture_output=True, text=True)
    module = ctypes.CDLL(str(library))
    for name in ("pos", "cos", "sin"):
        function = getattr(module, f"refresh_{name}")
        function.argtypes = [ctypes.c_bool, ctypes.c_int64, ctypes.c_size_t,
                             ctypes.c_int64, ctypes.c_size_t, ctypes.c_bool]
        function.restype = ctypes.c_bool
    return module


@pytest.mark.parametrize("name", ["pos", "cos", "sin"])
@pytest.mark.parametrize("inference", [False, True])
def test_same_geometry_slot_refreshes_changed_content(
    native_refresh, name, inference,
):
    refresh = getattr(native_refresh, f"refresh_{name}")
    with torch.inference_mode(inference):
        dtype = torch.int32 if name == "pos" else torch.float16
        source = torch.full((320, 3), 41, dtype=dtype)
        keepalive = torch.empty_like(source)
        pointer = source.data_ptr()
        previous_version, previous_address = -1, 0
        copies = 0

        def copy_if_needed():
            nonlocal previous_version, previous_address, copies
            version = -1 if torch.is_inference(source) else source._version
            needed = refresh(not torch.is_inference(source), version,
                             source.data_ptr(), previous_version,
                             previous_address, True)
            if needed:
                keepalive.copy_(source)
                previous_version, previous_address = version, source.data_ptr()
                copies += 1
            return needed

        assert copy_if_needed()
        assert torch.equal(keepalive, source)
        # Tracked unchanged sources retain the cache hit. Untracked inference
        # storage cannot prove a hit even when its current bytes happen to match.
        assert copy_if_needed() is inference
        assert copies == (2 if inference else 1)

        # Dual P313 and photo P314 both execute E320 in the same stable slot.
        # A changed M-RoPE payload must refresh before any intervening decode.
        source.fill_(33)
        assert source.data_ptr() == pointer
        assert not torch.equal(keepalive, source)
        assert copy_if_needed()
        assert torch.equal(keepalive, source)
        assert copy_if_needed() is inference

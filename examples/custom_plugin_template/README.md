# Out-of-Tree Plugin & Custom Operator Template / 树外二次开发插件模板

This directory provides a minimal, self-contained template for extending RhinoForge
out-of-tree **without forking or modifying RhinoForge's source repository**.
It covers both layers of secondary development:

1. **Python Model Adapter Plugin (`rpu_backend.plugins`)**: Register support for a
   custom Hugging Face architecture string via standard Python packaging
   entry points ([pyproject.toml.example](pyproject.toml.example) and
   [my_rhino_plugin/__init__.py](my_rhino_plugin/__init__.py)).
2. **Optional C++ Custom Operator Library (`TORCH_LIBRARY_FRAGMENT`)**: Build a
   standalone shared library ([csrc/CMakeLists.txt.example](csrc/CMakeLists.txt.example) and
   [csrc/custom_ops_fragment.cpp.example](csrc/custom_ops_fragment.cpp.example)) that
   attaches new operators into `torch.ops.rpu.*` at runtime via
   `torch.ops.load_library(...)`.

---

## Layout

| File | Purpose |
|---|---|
| [pyproject.toml.example](pyproject.toml.example) | Out-of-tree Python package metadata declaring `[project.entry-points."rpu_backend.plugins"]` |
| [my_rhino_plugin/__init__.py](my_rhino_plugin/__init__.py) | Plugin `register(api)` hook + example adapter class + lazy native `.so` loader |
| [csrc/CMakeLists.txt.example](csrc/CMakeLists.txt.example) | Standalone CMake build recipe compiling against installed PyTorch (and optional `rhino_launch`) |
| [csrc/custom_ops_fragment.cpp.example](csrc/custom_ops_fragment.cpp.example) | C++17 `TORCH_LIBRARY_FRAGMENT(rpu, m)` + `PrivateUse1` / `CPU` dispatch example |

---

## 1. Python Adapter Entry-Point Workflow

Copy this template into your own repository, rename `pyproject.toml.example` to
`pyproject.toml`, and install into the same Python 3.12 environment as `rhinoforge`:

```bash
cp -r examples/custom_plugin_template /tmp/my-rhino-plugin
cd /tmp/my-rhino-plugin
mv pyproject.toml.example pyproject.toml
pip install --no-build-isolation -e .
```

At runtime, RhinoForge discovers `[project.entry-points."rpu_backend.plugins"]`
lazily when an unknown architecture is requested, or explicitly when calling
`discover_plugins()`:

```python
from rpu_backend.runtime.registry import discover_plugins, get_adapter

discover_plugins()
adapter_cls = get_adapter("MyCustomForCausalLM")
```

### Registry Safety Rules

- **No Built-In Shadowing (`REG-03`)**: External plugins cannot override built-in
  architectures declared in RhinoForge; attempting to register an existing built-in
  architecture raises `RuntimeError`.
- **Idempotent Same-Class Re-Registration**: Re-registering the exact same
  `(architecture, cls)` pair is a safe no-op; registering a conflicting class for
  an already bound architecture raises `RuntimeError`.
- See [API Reference](../../docs/api_reference.md) and
  [Model Porting Guide](../../docs/model_porting.md) for adapter lifecycle contracts.

---

## 2. Optional Out-of-Tree C++ Operator Extension

When a custom model requires host/device operator glue beyond Python composition:

1. Copy [csrc/CMakeLists.txt.example](csrc/CMakeLists.txt.example) and
   [csrc/custom_ops_fragment.cpp.example](csrc/custom_ops_fragment.cpp.example) to
   `csrc/CMakeLists.txt` and `csrc/custom_ops_fragment.cpp`.
2. Build against the exact Torch 2.10 / Rhino Launch ABI pinned in
   [runtime-dependencies.json](../../release/runtime-dependencies.json)
   (check `op_abi_epoch` and `launch_kernel.abi_version` before deploying binary
   wheels across releases):

```bash
cmake -S csrc -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

3. Load the resulting `libmy_rhino_custom_ops.so` from your adapter using
   `MyCustomAdapter.load_native_ops(path)` (which wraps `torch.ops.load_library`),
   making `torch.ops.rpu.custom_scale_bias` callable without rebuilding RhinoForge.

---

## 中文说明

本目录提供一套完整的**树外（Out-of-Tree）二次开发脚手架**，便于客户及外部模型团队在**不 fork、不修改 RhinoForge 仓库源码**的前提下扩展新模型架构与自定义算子：

1. **Python 模型适配器插件（`rpu_backend.plugins`）**：在独立分发包的 `pyproject.toml` 中声明 `[project.entry-points."rpu_backend.plugins"]`（见 [pyproject.toml.example](pyproject.toml.example) 与 [my_rhino_plugin/__init__.py](my_rhino_plugin/__init__.py)）。RhinoForge 在遇到未知 Hugging Face 架构名时自动按需加载插件，也可通过 `rpu_backend.runtime.registry.discover_plugins()` 显式触发发现。内置架构受快照保护（`REG-03`），外部插件不可覆盖内置模型绑定。详见 [API 参考](../../docs/api_reference.zh.md) 与 [模型移植指南](../../docs/model_porting.zh.md)。
2. **可选 C++ 自定义算子动态库（`TORCH_LIBRARY_FRAGMENT`）**：当自定义模型需要扩展底层算子或自定义调度封装时，可参考 [csrc/CMakeLists.txt.example](csrc/CMakeLists.txt.example) 与 [csrc/custom_ops_fragment.cpp.example](csrc/custom_ops_fragment.cpp.example) 独立编译 `.so`，通过 `TORCH_LIBRARY_FRAGMENT(rpu, m)` 与 `TORCH_LIBRARY_IMPL(rpu, PrivateUse1, m)` 注册算子，并在 Python 插件内调用 `torch.ops.load_library(...)` 加载。发布二进制插件时请核对 [runtime-dependencies.json](../../release/runtime-dependencies.json) 中的 `op_abi_epoch` 与 `launch_kernel.abi_version`。

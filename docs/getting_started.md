# Getting started

This guide starts from a clean shell on a preconfigured RPU board environment
and ends with a model example launched from a TOML configuration. The first
RhinoForge release is built from source; it is not published on PyPI.

## 1. Check the host tools

The first release supports Python 3.12 and the PyTorch version declared in
`pyproject.toml`. The build also needs scikit-build-core 0.12.x, CMake 3.18 or
newer, and a C++17 compiler.

```bash
python --version
cmake --version
python -c "import torch; print(torch.__version__)"
python -m pip install "scikit-build-core>=0.12,<0.13"
```

The board environment itself must already be configured before installing
RhinoForge.

## 2. Install Rhino Launch

Follow [Restricted runtime assets](runtime_assets.md) to select one authorized
compatibility set from its HTTPS or SFTP release index, verify its signed BOM
and status, and verify the Rhino Launch 1.0.0 binary development package. A
plain HTTP index may be used only for an explicitly labelled, network-restricted
internal engineering preview. The package must provide its headers, shared
library, CMake package files, install manifest, and separate license files.
RhinoForge does not fetch or copy this package.

For a system installation, no extra build setting is needed. For a non-system
installation, point CMake at the package prefix:

```bash
export RHINO_LAUNCH_PREFIX=/opt/rhino-launch
export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
```

The shared library must also be visible to the system loader. Prefer an
administrator-managed loader configuration. For a local installation, the
board administrator may instead expose the package's `lib` directory through
the platform's standard loader-path mechanism.

## 3. Install the combined operator asset

From the same verified compatibility set, obtain the release-matched
`rhinoOpLib_current.ref` and its SHA256 checksum file. Keep the asset outside
the source checkout and Python environment. The complete download, signature,
status, checksum, and versioned-install procedure is in
[Restricted runtime assets](runtime_assets.md#download-and-verify).

```bash
export RHINOFORGE_DOWNLOAD_DIR="/path/to/downloaded/compatibility-set"
export RHINOFORGE_RUNTIME_SET="COMPATIBILITY_SET_ID"
export RHINOFORGE_RUNTIME_DIR="$HOME/.local/share/rhinoforge/runtime/$RHINOFORGE_RUNTIME_SET"
install -d "$RHINOFORGE_RUNTIME_DIR"
cd "$RHINOFORGE_DOWNLOAD_DIR"
sha256sum -c rhinoOpLib_current.ref.sha256
install -m 0644 rhinoOpLib_current.ref "$RHINOFORGE_RUNTIME_DIR/"
export RPU_KERNEL_LIB_PATH="$RHINOFORGE_RUNTIME_DIR/rhinoOpLib_current.ref"
```

`RPU_KERNEL_LIB_PATH` is the only operator-asset path used by RhinoForge. Do not
split the asset or configure additional operator-library paths. A combined
asset is versioned separately from the source; use only the values from the
same verified compatibility set. Do not place this path or any download token
in a model TOML file.

## 4. Build and install RhinoForge

The `rhinoforge` distribution and the legacy `rpu_backend` distribution both
install the `rpu_backend` import package and must not coexist. Use a fresh
virtual environment, or uninstall the legacy distribution before continuing:

```bash
python -m pip uninstall rpu_backend
```

From the root of a RhinoForge source checkout:

```bash
python -m pip install -e . --no-build-isolation
```

The editable build is convenient for development. A regular source install uses
the same build path:

```bash
python -m pip install . --no-build-isolation
```

Rhino Launch remains an external shared-library dependency; neither install
mode places it in the RhinoForge package.

## 5. Verify the installation

Keep `RPU_KERNEL_LIB_PATH` set and run:

```bash
python examples/verify_install.py
```

A zero exit status confirms that the Python package, `torch.rpu` registration,
and configured runtime prerequisites are visible. This check does not download
or load a model, and it does not replace the earlier signature, status, and
checksum verification.

## 6. Download a model checkpoint

Choose a runnable profile from [Model support](model_support.md), then use the
source identifier listed in [Model assets](model_assets.md). A published
release manifest pins the exact revision and hashes used for release
validation; until those fields are complete, a local run is not release
validation. Model terms and access requirements are separate from the
RhinoForge license.

For a model distributed through the Hugging Face Hub, the standard download
shape is:

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
mkdir -p "$RPU_MODEL_CACHE"
hf download SOURCE_ID \
  --revision REVISION \
  --local-dir "$RPU_MODEL_CACHE/LOCAL_MODEL_DIRECTORY"
```

Replace `SOURCE_ID`, `REVISION`, and `LOCAL_MODEL_DIRECTORY` with the values for
the selected release row. Use the model owner's documented downloader for
assets that are not distributed through the Hub. Do not run a checkpoint whose
revision or hash differs from the release asset manifest.

You may also pass an explicit checkpoint directory to an example instead of
setting `RPU_MODEL_CACHE`.

## 7. Optional offline quantization

Quantization is profile-specific. Do not assume that a quantized checkpoint
inherits the status of its FP16 source. The planned Qwen3 quantized release row
is the exact 14B W8A16 profile with an untied INT8 `lm_head` and FP16
embeddings:

```bash
python -m rpu_backend.quant.convert_qwen3 \
  --src "$RPU_MODEL_CACHE/Qwen3-14B" \
  --dst "$RPU_MODEL_CACHE/qwen3_lmhead_int8_ckpts/qwen3-14b-w8a16-lmhead-int8" \
  --quant-lm-head
```

The destination must not already exist. Qwen3 W8A16 outputs for other sizes are
Source-only until an exact release row promotes them. Other conversion entry
points and their profile limits are listed in [Model assets](model_assets.md)
and [Quantization](quantization.md).

## 8. Configure an example

Each example reads one TOML file. Copy the checked-in profile rather than
editing it in place, then set the checkpoint path and input fields documented by
that example:

```bash
cp examples/configs/qwen3_0_6b.toml qwen3.local.toml
```

Keep the precision and execution settings from the release profile until you
have validated a separate configuration. A registry alias only resolves a local
path; it does not expand a profile's support status.

For process-level runtime parameters and optional profiling, start from the
complete runner configuration instead:

```bash
cp examples/configs/qwen3_0_6b_full.toml qwen3.full.local.toml
python examples/run_model.py --config qwen3.full.local.toml --check-config
```

The complete schema and the entry point for each model are documented in
[Model execution and profiling](model_testing.md).

## 9. Run end-to-end inference

```bash
python examples/causal_lm.py --config qwen3.local.toml
```

The full runner invokes the same model script after applying its validated
runtime configuration:

```bash
python examples/run_model.py --config qwen3.full.local.toml
```

The process should complete with a model result and a zero exit status. The
model-specific examples under `examples/` use the same `--config` form. VLA
examples require the optional `vla` dependencies, and vision examples require
the optional `vision` dependencies:

```bash
python -m pip install -e '.[vision]' --no-build-isolation
python -m pip install -e '.[vla]' --no-build-isolation
```

## Common setup errors

| Symptom | Action |
|---|---|
| CMake cannot find `rhino_launch` | Install the 1.0.0 development package or add its prefix to `CMAKE_PREFIX_PATH` |
| The Rhino Launch shared library cannot be loaded | Ask the board administrator to configure the system loader for the installed package |
| A BOM signature, compatibility ID, checksum, or signed status differs | Stop and obtain one complete, active compatibility set from the authorized release index |
| `RPU_KERNEL_LIB_PATH` is unset or unreadable | Point it at the single downloaded `rhinoOpLib_current.ref` |
| A required operator is missing | Obtain the combined asset version matched to this RhinoForge release and Rhino Launch package |
| A checkpoint hash or revision differs | Re-download the exact model asset recorded for the release |
| Model preflight reports an unsupported profile | Select an exact runnable profile from `docs/model_support.md` |

If installation succeeds but a model cannot run, capture the complete error,
the RhinoForge version, compatibility-set and BOM IDs, Rhino Launch version,
operator-asset SHA256, checkpoint revision, and TOML SHA256 when reporting the
issue. Do not attach restricted assets or download credentials.

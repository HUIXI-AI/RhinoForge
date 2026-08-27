# Getting started

[简体中文](getting_started.zh.md) | English

This is the customer Quick Start for RhinoForge v1.0.0. It starts with the
RhinoForge v1.0.0 source release and the approved
`RhinoForge-runtime-v1.0.0-r4.tar.gz` delivery, and ends with Qwen3-0.6B
inference from the checked-in TOML. The board SDK/runtime and device access
must already be configured; models are not included.

Run all commands in one shell so the exported runtime paths remain active.

## 1. Check the board environment

Use the platform-provided Python 3.12. Step 4 installs PyTorch 2.10.0 in a
fresh virtual environment. The build also needs a C++17 compiler.

```bash
python3.12 --version
c++ --version
```

Obtain the RhinoForge source from the public v1.0.0 release. Runtime
compatibility is defined by that release version.

## 2. Receive and verify the runtime bundle

An internal delivery owner normally verifies and transfers the archive and its
adjacent checksum file to an approved customer. Put both files in the same
directory:

```text
RhinoForge-runtime-v1.0.0-r4.tar.gz
RhinoForge-runtime-v1.0.0-r4.tar.gz.sha256
```

If your distributor instead gave you an authorized download URL, set
`RHINOFORGE_RUNTIME_BUNDLE_URL` to that versioned directory and download the
same pair:

```bash
set -e
export RHINOFORGE_RUNTIME_BUNDLE_URL="<distributor-provided-runtime-v1.0.0-r4-URL>"
export RHINOFORGE_RUNTIME_ARCHIVE="RhinoForge-runtime-v1.0.0-r4.tar.gz"
curl -fSLO "${RHINOFORGE_RUNTIME_BUNDLE_URL%/}/$RHINOFORGE_RUNTIME_ARCHIVE"
curl -fSLO "${RHINOFORGE_RUNTIME_BUNDLE_URL%/}/$RHINOFORGE_RUNTIME_ARCHIVE.sha256"
```

From the directory containing the received files, verify the outer archive,
extract it, read the distribution terms, and enter the extracted directory:

```bash
set -e
export RHINOFORGE_RUNTIME_ARCHIVE="RhinoForge-runtime-v1.0.0-r4.tar.gz"
sha256sum -c "$RHINOFORGE_RUNTIME_ARCHIVE.sha256"
tar -xzf "$RHINOFORGE_RUNTIME_ARCHIVE"
cd RhinoForge-runtime-v1.0.0
cat DISTRIBUTION_TERMS.txt
```

Fail closed unless the bundle is active, download-enabled, compatible with the
RhinoForge v1.0.0 release, and covers all three runtime payloads. Derive every
payload filename from `RELEASE.txt`; do not hardcode or rename the opaque
operator files:

```bash
set -e
grep -qx 'status=active' RELEASE.txt
grep -qx 'download_enabled=true' RELEASE.txt
grep -qx 'compatible_rhinoforge_release=v1.0.0' RELEASE.txt
grep -qx 'checksum_scope=all_runtime_payloads' RELEASE.txt

export RHINOFORGE_RUNTIME_SET="$(sed -n 's/^runtime_set=//p' RELEASE.txt)"
export RHINOFORGE_LAUNCH_PACKAGE="$(sed -n 's/^launch_package_file=//p' RELEASE.txt)"
export RHINOFORGE_OPERATOR_ASSET="$(sed -n 's/^operator_asset_file=//p' RELEASE.txt)"
export RHINOFORGE_OPERATOR_KERNEL_MANIFEST="$(sed -n 's/^operator_kernel_manifest_file=//p' RELEASE.txt)"

case "$RHINOFORGE_RUNTIME_SET" in ""|*/*) echo "invalid runtime-set ID"; exit 1;; esac
case "$RHINOFORGE_LAUNCH_PACKAGE" in ""|*/*) echo "invalid Launch filename"; exit 1;; *.tar.gz) :;; *) echo "invalid Launch filename"; exit 1;; esac
case "$RHINOFORGE_OPERATOR_ASSET" in ""|*/*) echo "invalid operator filename"; exit 1;; *.ref) :;; *) echo "invalid operator filename"; exit 1;; esac
test "$RHINOFORGE_OPERATOR_KERNEL_MANIFEST" = "$RHINOFORGE_OPERATOR_ASSET.kernels"
test -f "$RHINOFORGE_LAUNCH_PACKAGE"
test -f "$RHINOFORGE_OPERATOR_ASSET"
test -f "$RHINOFORGE_OPERATOR_KERNEL_MANIFEST"
sha256sum -c SHA256SUMS
export RHINOFORGE_BUNDLE_DIR="$PWD"
```

The last checksum command must report success for the Launch package, the
opaque operator asset, and its adjacent `.kernels` manifest. See
[Restricted runtime assets](runtime_assets.md) for the distribution and
revocation contract.

## 3. Install the runtime assets without root

Install Rhino Launch into a versioned user-owned prefix. Its directory name is
derived from the package name in `RELEASE.txt`:

```bash
export RHINO_LAUNCH_ROOT="$HOME/.local/opt/${RHINOFORGE_LAUNCH_PACKAGE%.tar.gz}"
mkdir -p "$HOME/.local/opt"
tar -xzf "$RHINOFORGE_LAUNCH_PACKAGE" -C "$HOME/.local/opt"
test -f "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"

export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Install the operator asset and its manifest together, outside the source tree
and Python environment:

```bash
export RHINOFORGE_RUNTIME_DIR="$HOME/.local/share/rhinoforge/runtime/$RHINOFORGE_RUNTIME_SET"
install -d "$RHINOFORGE_RUNTIME_DIR"
install -m 0644 "$RHINOFORGE_BUNDLE_DIR/$RHINOFORGE_OPERATOR_ASSET" "$RHINOFORGE_RUNTIME_DIR/"
install -m 0644 "$RHINOFORGE_BUNDLE_DIR/$RHINOFORGE_OPERATOR_KERNEL_MANIFEST" "$RHINOFORGE_RUNTIME_DIR/"

export RPU_KERNEL_LIB_PATH="$RHINOFORGE_RUNTIME_DIR/$RHINOFORGE_OPERATOR_ASSET"
test -r "$RPU_KERNEL_LIB_PATH"
test -r "$RPU_KERNEL_LIB_PATH.kernels"
```

Keep the asset opaque and unchanged. Do not split it, inspect its contents,
generate a replacement manifest, or put its path or download credentials in a
model TOML.

## 4. Install RhinoForge in a fresh environment

Set `RHINOFORGE_SOURCE` to the RhinoForge v1.0.0 source directory. Use a truly
isolated virtual environment to avoid collision with the legacy `rpu_backend`
distribution; do not use `--system-site-packages`.

```bash
export RHINOFORGE_SOURCE="/absolute/path/to/RhinoForge"
cd "$RHINOFORGE_SOURCE"
python3.12 -m venv .venv
. .venv/bin/activate

python -m pip install "torch==2.10.0" "scikit-build-core>=0.12,<0.13" "cmake==4.1.3" "ninja==1.13.0"
python -m pip install . --no-build-isolation
python -m pip check
```

The second command is a regular, non-editable install. Do not add
`--no-deps`: the default install must resolve all runtime dependencies,
including Accelerate. Editable installs are for source development, not this
customer path.

Verify the installed release and the model-loading dependencies:

```bash
python - <<'PY'
from importlib.metadata import version
from packaging.version import Version

assert version("rhinoforge") == "1.0.0"
assert Version(version("torch")).base_version == "2.10.0"
assert version("transformers") == "5.5.0"
assert Version("1.1.0") <= Version(version("accelerate")) < Version("2")
print("RhinoForge Python dependencies: OK")
PY
```

## 5. Verify the installation and RPU

First check the import and version path without running a device copy, then run
the RPU copy check:

```bash
python examples/verify_install.py --check-config
python examples/verify_install.py
```

Both commands must exit with status zero. If the second command reports a
device-permission error, ask the board administrator to grant access to the RPU
device; do not reinstall the runtime assets with root.

## 6. Prepare Qwen3-0.6B

The v1.0.0 public model ledger pins this Quick Start profile to
`Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Download it into the directory
expected by the checked-in registry alias:

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
mkdir -p "$RPU_MODEL_CACHE"
hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-0.6B"
test -f "$RPU_MODEL_CACHE/Qwen3-0.6B/config.json"
```

If a distributor provides an already verified copy of that exact checkpoint,
place it at the same final directory instead. Model licenses and access terms
are separate from the RhinoForge and Runtime Bundle terms.

## 7. Check the TOML configuration

Copy the complete checked-in profile. Keep its `qwen3-0.6b` alias; the alias
resolves through `RPU_MODEL_CACHE`, so no absolute checkpoint edit is needed.

```bash
cp examples/configs/qwen3_0_6b.toml qwen3.local.toml
python examples/causal_lm.py --config qwen3.local.toml --check-config
python examples/run_model.py --config qwen3.local.toml --check-config
```

Both configuration checks must exit with status zero. Runtime asset paths and
credentials belong in the shell environment, never in this TOML.

## 8. Run Qwen3-0.6B inference

```bash
python examples/run_model.py --config qwen3.local.toml
```

The command should print a completion and exit with status zero. One live
RPU-resident model or policy is permitted per process on the fused CausalLM
path.

## Common setup errors

| Symptom | Action |
|---|---|
| Outer or inner checksum fails | Stop; obtain the matching archive and checksum pair again. Do not mix runtime sets |
| Bundle metadata does not name v1.0.0 or is not active | Stop; ask the distributor for the active bundle compatible with the RhinoForge v1.0.0 release |
| CMake cannot find `rhino_launch` | Re-export `CMAKE_PREFIX_PATH` to the Launch prefix derived above |
| The Launch shared library cannot be loaded | Re-export its `lib` directory in `LD_LIBRARY_PATH` |
| `RPU_KERNEL_LIB_PATH` or the adjacent manifest is unreadable | Reinstall both unchanged files named by `RELEASE.txt` from the same bundle |
| `pip check` or the version check reports missing Accelerate | Re-run the regular project install without `--no-deps` |
| `torch.rpu` is unavailable | Confirm board setup and device permissions with the board administrator |
| The model alias cannot resolve | Keep `RPU_MODEL_CACHE/Qwen3-0.6B` complete and keep the TOML alias unchanged |
| Model preflight rejects the profile | Use the exact checkpoint revision and checked-in v1.0.0 TOML |

For other supported models and controls, continue with [Model support](model_support.md),
[Model assets](model_assets.md), [Runtime configuration](runtime_config.md),
and [Model execution and profiling](model_testing.md).

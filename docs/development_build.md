# Build the current development source

This guide covers editable installs and wheel builds for the current source.
For a first model run, use [Getting started](getting_started.md). Build
requirements follow the current `pyproject.toml` and matching external runtime
assets. A successful build establishes neither model accuracy nor performance.

## Select matching external runtime assets

Use a C++17 toolchain, Python 3.12 and the platform's configured board SDK. Obtain
the following matching assets from the runtime provider before building:

- A sanitized **Release** Rhino Launch installation with its headers, libraries
  and `lib/cmake/rhino_launch` package. Current CMake requires
  `graph-arena-pool-v2`, `register-state-token`, sanitized Chrome hwperf support and no development Program
  API. The package version or library filename alone does not prove compatibility.
- The combined opaque operator asset selected for this source revision, with its
  adjacent `.kernels` sidecar and provider-supplied integrity values. The current
  backend consumes one combined asset containing the required public-model kernels.
- The matching board SDK/runtime and its required shared libraries. Keep the
  build-time and run-time Launch installation consistent.

Launch and operator assets remain external to this repository and wheel. Do not
substitute an older incompatible runtime, copy its payload into the package,
merge operator payloads locally, or generate a replacement sidecar.
If a matching asset is unavailable, obtain it from the runtime provider.

Set paths to the selected source, a new environment and a dedicated build
directory. Keep this environment separate from any installed `rpu_backend` or
other RhinoForge checkout; do not use `--system-site-packages`.

```bash
export RHINOFORGE_SOURCE="/absolute/path/to/current/RhinoForge"
export RHINOFORGE_DEV_ENV="/absolute/path/to/new/rhinoforge-dev-env"
export RHINOFORGE_DEV_BUILD="/absolute/path/to/new/rhinoforge-dev-build"
export RHINO_LAUNCH_ROOT="/absolute/path/to/matching/launch-install"
export RPU_KERNEL_LIB_PATH="/absolute/path/to/matching/combined-operators.ref"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

test -r "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"
test -r "$RPU_KERNEL_LIB_PATH"
test -r "$RPU_KERNEL_LIB_PATH.kernels"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Verify the supplied asset digests before use. The existence checks above only
check paths. Remove conflicting Launch directories from an inherited library
search path; retain the platform SDK paths required by the selected runtime.

## Install from source

The current [pyproject.toml](../pyproject.toml) is the dependency authority:
Python 3.12, PyTorch 2.10.0 and Transformers **5.3.0**. Its optional `vision` extra
adds TorchVision 0.25.0; the `vla` extra also selects LeRobot 0.5.1 and Diffusers
0.33.1. Install the extra needed by the intended entry point.

```bash
python3.12 -m venv "$RHINOFORGE_DEV_ENV"
. "$RHINOFORGE_DEV_ENV/bin/activate"
cd "$RHINOFORGE_SOURCE"

python -m pip install "torch==2.10.0" "scikit-build-core>=0.12,<0.13" "cmake==4.1.3" "ninja==1.13.0"
python -m pip install -e . --no-build-isolation \
  -Cbuild-dir="$RHINOFORGE_DEV_BUILD" \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
python -m pip check
```

For a vision or VLA environment, replace `-e .` with `-e '.[vision]'` or
`-e '.[vla]'` in that install command. Keep dependency resolution enabled; the
initial source install must not use `--no-deps`. Two build workers bound compiler
memory use; adjust this limit to the host's available resources.

Inspect the dependency versions and selected Python source without loading the
native extension or issuing a device operation:

```bash
python - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec

assert version("transformers") == "5.3.0"
assert version("torch").split("+")[0] == "2.10.0"
print("rhinoforge:", version("rhinoforge"))
print("Python source:", find_spec("rpu_backend").origin)
PY
```

An editable install follows Python edits in this checkout. Native changes still
require rebuilding and reinstalling with the same selected Launch installation.

## Build a reviewable wheel

After installing and checking the dependencies above, build into a new artifact
directory. This command reuses the dedicated build directory and does not replace
the installed backend:

```bash
python -m pip wheel . --no-build-isolation --no-deps \
  --wheel-dir /absolute/path/to/new/wheel-artifacts \
  -Cbuild-dir="$RHINOFORGE_DEV_BUILD" \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
```

Record the source commit and local diff, resolved dependencies, wheel/native
digests, and selected Launch/operator/SDK identities with the build. Install the
wheel in another fresh environment and check package/native origins and
dependencies before device validation. Rebuild the wheel after Python changes
as well as native changes: an unchanged `rpu_backend.so` digest or `1.0.0` package
version does not establish that an older wheel contains the current Python code.

# External runtime assets

[简体中文](runtime_assets.zh.md) | English

RhinoForge source and wheels do not contain Rhino Launch binaries, the external
main and expansion operator assets, the board SDK, or model weights. The
[public dependency environment bundle](environment_bundle.md) also excludes all
of these components and RhinoForge itself. Obtain RhinoForge from GitHub, the
OS/driver and complete SDK from the platform provider, and the matching Launch
and operator assets through their separate delivery channel. Follow each
package's distribution terms and integrity instructions.

This `1.3.0.dev0` development line depends on archived Rhino Launch **1.2.0**,
commit `3dd1b410f5f46373997fef3dfbf9b67eecec6a84`, with CMake version `1.2.0` and
ABI version `1`, and advances `op_abi_epoch` to **`2`**. Obtain the matching
`RhinoForge-runtime-v1.3.0.dev0.tar.gz` and its `.tar.gz.sha256` from the provider.
Its nested Launch package directory remains `rhino-launch-kernel-v1.2.0-linux-aarch64`;
its main and DDR-expansion assets match `op_abi_epoch = 2` in
[runtime-dependencies.json](../release/runtime-dependencies.json) (`RhinoForge-runtime-v1.2.0.tar.gz`
remains fixed for `v1.2.0` / `op_abi_epoch = 1`).
Each operator asset requires its adjacent `.ref.kernels` file.
Historical candidates can have the same bundle filename: verify the checksum,
`RELEASE.txt` and Launch commit. A directory named `v1.1.0-2-gbdc0dae` contains
an older Launch package whose CMake version is `1.0.0`, and does not satisfy this dependency.
See the [release policy](release_policy.md).

Select the bundle by the dependency manifest in the checkout being built. A
provider's development download page or a bundle README may describe an older
source revision: `1.3.0.dev0` does not imply a Launch `1.3.0.dev0` dependency.
For this source line, use the stable 1.2.0 identities above. Request corrected
provider metadata when it conflicts; do not bypass CMake or asset identity checks.

## Request runtime assets

Submit an [asset request](https://github.com/HUIXI-AI/RhinoForge/issues/new)
with the source version or commit, board platform, and required Launch, main and
DDR-expansion components. Keep restricted files, credentials and machine details
out of the public issue. Asset access follows the provider's distribution terms.

## Required components

- **Rhino Launch 1.2.0**, built as a sanitized Release package, with headers,
  shared libraries and `lib/cmake/rhino_launch/rhino_launchConfig.cmake`.
  Current CMake checks `graph-arena-pool-v2`, `register-state-token`, Chrome
  hwperf support and the absence of the development Program API.
- **Two operator assets**, built from the main and DDR-expansion branches respectively, matching this source, each with an adjacent
  `<asset>.kernels` manifest. It must contain the kernels required by the public
  models and precision profiles being used.
- **The matching board OS, driver and complete SDK**, including its user-space
  libraries, supplied and configured according to the platform's installation
  instructions. The recommended SDK baseline is listed below.

Keep build-time and run-time Launch paths consistent.
Do not mix an old operator payload with a new sidecar or merge payloads locally.

## Configure paths

Download `RhinoForge-runtime-v1.3.0.dev0.tar.gz` and its `.sha256` from the provider's
delivery location into the same download directory. The following
commands verify both layers and unpack the nested Launch archive; the outer
archive alone does not create an installed Launch prefix. Use new directories
outside the source checkout and Python environment, and stop if any check fails:

```bash
export RHINOFORGE_DOWNLOADS="/absolute/path/to/runtime-downloads"
export RHINOFORGE_RUNTIME="/absolute/path/to/new/runtime"
cd "$RHINOFORGE_DOWNLOADS"
sha256sum -c RhinoForge-runtime-v1.3.0.dev0.tar.gz.sha256
mkdir -p "$RHINOFORGE_RUNTIME"
tar -xzf RhinoForge-runtime-v1.3.0.dev0.tar.gz -C "$RHINOFORGE_RUNTIME"
export RHINOFORGE_BUNDLE="$RHINOFORGE_RUNTIME/RhinoForge-runtime-v1.3.0.dev0"
cd "$RHINOFORGE_BUNDLE"
sha256sum -c SHA256SUMS
cat RELEASE.txt
tar -xzf rhino-launch-kernel-v1.2.0-linux-aarch64.tar.gz -C "$RHINOFORGE_RUNTIME"

export RHINO_LAUNCH_ROOT="$RHINOFORGE_RUNTIME/rhino-launch-kernel-v1.2.0-linux-aarch64"
export RPU_KERNEL_LIB_PATH="$RHINOFORGE_BUNDLE/rhinoOpLib_current.ref"
export RPU_KERNEL_LIB_PATH_EXPANSION="$RHINOFORGE_BUNDLE/rhinoExpansionOpLib_current.ref"
export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -r "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"
test -r "$RPU_KERNEL_LIB_PATH"
test -r "$RPU_KERNEL_LIB_PATH.kernels"
test -r "$RPU_KERNEL_LIB_PATH_EXPANSION"
test -r "$RPU_KERNEL_LIB_PATH_EXPANSION.kernels"
ls "$RHINO_LAUNCH_ROOT/lib/"librhino_launch.so*
```

The resulting directories are:

```text
runtime/
  RhinoForge-runtime-v1.3.0.dev0/
    RELEASE.txt, SHA256SUMS
    rhino-launch-kernel-v1.2.0-linux-aarch64.tar.gz
    rhinoOpLib_current.ref, rhinoOpLib_current.ref.kernels
    rhinoExpansionOpLib_current.ref, rhinoExpansionOpLib_current.ref.kernels
  rhino-launch-kernel-v1.2.0-linux-aarch64/
    include/
    lib/                       # shared libraries and cmake/rhino_launch/
```

Compare `RELEASE.txt` with [the source dependency lock](../release/runtime-dependencies.json):
Launch commit/version/ABI and both operator commits and SHA256 values must match.
The supplied checksum validates receipt of a particular bundle; the dependency
lock establishes whether that bundle belongs to this source. Remove conflicting
Launch paths from the inherited environment while preserving required SDK paths.
Credentials and machine paths belong in the shell environment, outside model
TOMLs. Continue with the [unified installation and development guide](getting_started.md).

## Board SDK and device access

**Recommended board SDK: `EA_R1SDK_2026_08F-2156007a`.** This is the baseline used
for our board deployment and RhinoVLA v3 RTC delay `0–6` validation:

| Component | Validated baseline |
|---|---|
| Board platform | R1, Linux `aarch64` |
| Board SDK | `EA_R1SDK_2026_08F-2156007a` |
| OS / system C library | Ubuntu 22.04 / glibc 2.35 |
| Host compiler | GCC 11.4 |
| Application Python | Python 3.12.15 |

This recommendation is a tested combination, not a universal minimum SDK
version or a claim that newer SDKs are compatible. The platform provider must
confirm the matching OS/kernel/driver combination and any alternative SDK.
The board SDK and **Rhino Launch 1.2.0** are different components; the Launch
version and operator identities remain pinned by
[runtime-dependencies.json](../release/runtime-dependencies.json).

The runtime package and public dependency environment bundle do **not** supply
the board SDK, its user-space libraries, or the kernel driver. In particular,
`libHxMem.so`, `libHxil.so` and `librpu_api.so` come from the platform SDK.
Obtain the complete SDK/driver package, installation commands, loader setup and
access/locking policy from the platform provider. This repository has no
universal SDK download or driver-install command. On an unprovisioned board,
complete that platform step before building RhinoForge. Record the actual
SDK/driver versions used; retaining a provisioned board baseline is an isolated
application install, not a clean SDK/driver installation.

After activating the Python environment, apply the provider's loader setup.
If it requires `LD_LIBRARY_PATH`, use the actual SDK library directory (or
directories) supplied by the platform; the following example assumes all three
libraries are in one directory. Keep the current application environment's `lib`
first, followed by Launch and SDK paths. This matters when the SDK uses `/lib`
or `/usr/lib`, which also contain system libraries that could otherwise override
the environment's media and C++ runtime libraries:

```bash
export RHINOFORGE_SDK_LIB_DIR="/absolute/path/to/platform-sdk/lib"
test -r "$RHINOFORGE_SDK_LIB_DIR/libHxMem.so"
test -r "$RHINOFORGE_SDK_LIB_DIR/libHxil.so"
test -r "$RHINOFORGE_SDK_LIB_DIR/librpu_api.so"
export LD_LIBRARY_PATH="${RHINOFORGE_ENV:+$RHINOFORGE_ENV/lib:}$RHINO_LAUNCH_ROOT/lib:$RHINOFORGE_SDK_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Inspect library resolution and device nodes:

```bash
uname -smr
ldd "$RHINO_LAUNCH_ROOT/lib/librhino_launch.so"
ls -l /dev/rpu* /dev/mem
```

Every `ldd` dependency must resolve to the selected Launch or platform SDK/system
libraries; resolve `not found` with the provider's SDK loader configuration. Use
the actual library filename if the package has no unversioned `.so` symlink.
Device existence alone does not establish access. The process needs `O_RDWR`
access to at least one `/dev/rpu*` node and to `/dev/mem`. Under the same account
that will run inference, check both without reading or writing device contents:

```bash
python3 - <<'PY'
import glob, os
candidates = sorted(glob.glob("/dev/rpu*"))
if not candidates:
    raise SystemExit("No /dev/rpu* node: check the platform driver")
for path in candidates:
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    except OSError as error:
        print(f"{path}: {error}")
    else:
        os.close(fd)
        print(f"O_RDWR available: {path}")
        break
else:
    raise SystemExit("No accessible RPU node")
fd = os.open("/dev/mem", os.O_RDWR | os.O_CLOEXEC)
os.close(fd)
print("O_RDWR available: /dev/mem")
PY
```

If access is denied, have the platform administrator configure its supported
access policy, or use its approved privileged launcher. Do not globally relax
`/dev/mem` permissions. After correcting access, start a fresh Python process.
An ordinary CPU-only import can succeed with the native backend unavailable;
`--check-config` also does not test the device. See the
[idle-device and privileged-run examples](getting_started.md#3-run-inference-on-an-idle-device).

## Operator roles and replacement

The main and DDR-expansion roles are fixed. [runtime-dependencies.json](../release/runtime-dependencies.json) records their source commits and SHA256 identities. Source and installed-package entries use the same roles across models.

`RPU_KERNEL_LIB_PATH` and `RPU_KERNEL_LIB_PATH_EXPANSION` may point to identical installed copies. Entries reject hash mismatches, swapped roles and conflicting inherited paths. GR00T configurations reject `example.expansion_ref`. Start a new process after replacing either asset.

Update the complete main and expansion assets together with their adjacent
manifests and the source/size/SHA256 records. A partial build or a locally merged
payload is not a replacement for either role. New operator implementations
belong to the operator repository; this repository owns model routing and host
wrappers. To reproduce an older source, use that release's archived runtime set
and original checksums together. Do not point the current source at an older
payload while retaining current dependency metadata.

## Operator manifest

Keep the `.kernels` file adjacent to its asset. Its first line is
`rhinoforge-kernels-v1`, its second is `asset-size=<bytes>`, and the remaining lines
are unique ASCII kernel names in sorted order. The runtime validates this format
and the asset size before resolving kernels. This check catches missing metadata
and size mismatches; it does not verify the operator implementation or replace
the provider's integrity checks.

If a required kernel is missing, obtain a compatible public-model asset from the
provider. Adding a name to the sidecar does not add its implementation to the
payload. Operator source and payload contents remain outside this repository.

## Common setup errors

| Symptom | Action |
|---|---|
| CMake cannot find `rhino_launch` | Set the Launch prefix or pass `-Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch"` |
| CMake reports a Launch version mismatch | Select the archived stable Launch package pinned by the dependency manifest and update both build and runtime paths; do not rewrite Launch version files or remove the EXACT check |
| CMake rejects a required capability | Obtain a matching sanitized Release Launch package |
| Launch cannot be loaded | Check its `lib` path and the board SDK library paths |
| Asset or sidecar is missing, or a kernel is absent | Obtain the matching asset and unchanged sidecar together |
| Device node missing or `Permission denied` / `Operation not permitted` | Check driver setup and `O_RDWR` access to `/dev/rpu*` and `/dev/mem` under the inference account; restart Python after fixing access |
| Older code reports `PyTorch is not linked with support for rpu devices` | Check preceding native-loader diagnostics and device access first; this can be a skipped backend registration, not a PyTorch build problem |

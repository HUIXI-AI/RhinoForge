# External runtime assets

[简体中文](runtime_assets.zh.md) | English

RhinoForge source and wheels do not contain Rhino Launch binaries, the combined
opaque operator asset, the board SDK, or model weights. Obtain a runtime package
compatible with the source revision you are installing from the runtime provider.
Follow that package's distribution terms and integrity instructions. Historical
runtime receipts do not establish compatibility with the current source.

For the RhinoForge v1.1.0 internal prerelease, request
`RhinoForge-runtime-v1.1.0.tar.gz` and its `.tar.gz.sha256` file from the provider.
The archive extracts to `RhinoForge-runtime-v1.1.0/` and contains
`rhinoOpLib_rhinoforge_v1.1.0.ref` with its adjacent `.ref.kernels` manifest.
Rhino Launch's own version remains **1.0.0**; the bundle version does not rename
or change that API version. Use the exact nested Launch package named by
`RELEASE.txt`. This prerelease does not imply a published GitHub release/tag.

## Required components

- **Rhino Launch 1.0.0**, built as a sanitized Release package, with headers,
  shared libraries and `lib/cmake/rhino_launch/rhino_launchConfig.cmake`.
  Current CMake checks `graph-arena-pool-v2`, `register-state-token`, Chrome
  hwperf support and the absence of the development Program API.
- **One combined operator asset** matching this source, plus its adjacent
  `<asset>.kernels` manifest. It must contain the kernels required by the public
  models and precision profiles being used.
- **The matching board SDK/runtime**, configured according to the platform's
  installation instructions.

A package version or shared-library filename alone does not establish the
required capabilities. Keep build-time and run-time Launch paths consistent.
Do not mix an old operator payload with a new sidecar or merge payloads locally.

## Configure paths

Verify the provider's checksums when receiving the package, then extract it to a
stable directory outside the source checkout and Python environment. Set paths
to the installed Launch prefix and the unchanged operator asset:

```bash
export RHINO_LAUNCH_ROOT="/absolute/path/to/matching/launch-install"
export RPU_KERNEL_LIB_PATH="/absolute/path/to/RhinoForge-runtime-v1.1.0/rhinoOpLib_rhinoforge_v1.1.0.ref"
export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -r "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"
test -r "$RPU_KERNEL_LIB_PATH"
test -r "$RPU_KERNEL_LIB_PATH.kernels"
```

Replace the sample paths with the provider's actual filenames. Remove conflicting
Launch paths from the inherited environment while preserving required SDK paths.
Credentials and machine paths belong in the shell environment, outside model
TOMLs. Continue with [Getting started](getting_started.md) or the
[source build guide](development_build.md).

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
| CMake rejects a required capability | Obtain a matching sanitized Release Launch package |
| Launch cannot be loaded | Check its `lib` path and the board SDK library paths |
| Asset or sidecar is missing, or a kernel is absent | Obtain the matching asset and unchanged sidecar together |
| RPU is unavailable | Check device permissions and the platform runtime setup |

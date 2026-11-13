# Release assets

This directory contains the operator-manifest generator and provenance manifests
for public model sources and weights. Runtime binaries, opaque operator payloads,
and model checkpoints are distributed separately under their own terms.

RhinoForge `1.3.0.dev0` depends on archived Rhino Launch `1.2.0` at
`3dd1b410f5f46373997fef3dfbf9b67eecec6a84`, with CMake version `1.2.0` and ABI
version `1`. Use the Launch package and two operator assets from the stable
`RhinoForge-runtime-v1.2.0.tar.gz` distribution, as pinned in
[runtime-dependencies.json](runtime-dependencies.json). Keep each operator payload
with its adjacent `.ref.kernels` file. Advancing RhinoForge's development version
does not change the Launch dependency version.
See the [release policy](../docs/release_policy.md).

The recommended board SDK is **`EA_R1SDK_2026_08F-2156007a`**, validated on R1
Linux `aarch64`, Ubuntu 22.04/glibc 2.35, GCC 11.4 and Python 3.12.15. It is a
tested baseline, not a minimum-version promise. The platform provider supplies
the matching OS/driver and complete SDK, including its user-space libraries.
The board SDK is separate from Rhino Launch 1.2.0. See the
[SDK setup and validation scope](../docs/runtime_assets.md#board-sdk-and-device-access).

The [dependency environment bundle design](../docs/environment_bundle.md)
([中文](../docs/environment_bundle.zh.md)) defines a public Conda bundle for
Python, third-party libraries and tools. It excludes RhinoForge source and
installed code/native libraries, all board SDK components, Launch, operator
assets and model weights. Users obtain RhinoForge from GitHub and build it in
that environment; Launch and operator assets retain their separate delivery
terms.

Environment versions are independent of RhinoForge source versions. The design
binds each supported source release or commit to a tested environment ID and
manifest, with download location and SHA256 recorded when an artifact is
published. Source-only changes can reuse a compatible environment. Requirement or
compatibility changes need revalidation and an updated source binding; publish a
new environment version when its actual contents must change.
GitHub Releases distributes the reviewed dependency-only bundle. The current
bound environment is [`deps-2026.10.1` (preview)](https://github.com/HUIXI-AI/RhinoForge/releases/tag/deps-2026.10.1),
whose archive URL and SHA-256 digests are recorded in
[environment.json](environment.json). Legacy restricted full-bundle tooling and
archives remain separate and must not be tagged or uploaded as a public
dependency environment release.

Keep model evaluation data and reports outside source releases, package artifacts and release notes.

For version changes and installation steps, see the [changelog](../CHANGELOG.md)
and [upgrade guide](../docs/upgrading_1_2.md). Request separately distributed
assets through the [runtime asset guide](../docs/runtime_assets.md#request-runtime-assets).

For current examples, see [Examples](../examples/README.md); for
installation, see [runtime assets](../docs/runtime_assets.md). Use the
[proportionate validation policy](../docs/validation_policy.md) for code changes.

`generate_kernel_manifest.py` is a distribution utility. Its output lists public
host-side kernel references intersected with a provider-supplied kernel inventory;
it neither modifies nor verifies the implementation inside an opaque asset.
Changing a sidecar cannot add missing kernels to the payload.

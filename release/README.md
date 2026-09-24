# Release assets

This directory contains the operator-manifest generator and provenance manifests
for public model sources and weights. Runtime binaries, opaque operator payloads,
and model checkpoints are distributed separately under their own terms.

The RhinoForge v1.1.0 internal prerelease uses the provider-delivered
`RhinoForge-runtime-v1.1.0.tar.gz`, extracted under `RhinoForge-runtime-v1.1.0/`,
with `rhinoOpLib_rhinoforge_v1.1.0.ref` and its adjacent `.ref.kernels` manifest.
Rhino Launch remains version 1.0.0; `RELEASE.txt` selects the matching sanitized
Release package. Verify the outer checksum and inner `SHA256SUMS` before using
the [provider paths](../docs/runtime_assets.md#configure-paths). These filenames
do not imply that a GitHub release/tag has been published.

Internal board measurements, benchmark tables and performance comparisons are
not included in source releases, package artifacts or release notes. Keep local
measurement reports outside the repository. Model configuration guides describe
supported options and tuning practices without publishing measured results.

Obsolete runtime receipts, model qualification matrices, and candidate validation
closures have been removed. They described earlier source and asset combinations
and do not establish correctness or performance for the current checkout.

For current examples, see [Examples](../examples/README.md); for
installation, see [runtime assets](../docs/runtime_assets.md). Use the
[proportionate validation policy](../docs/validation_policy.md) for code changes.

`generate_kernel_manifest.py` is a distribution utility. Its output lists public
host-side kernel references intersected with a provider-supplied kernel inventory;
it neither modifies nor verifies the implementation inside an opaque asset.
Changing a sidecar cannot add missing kernels to the payload.

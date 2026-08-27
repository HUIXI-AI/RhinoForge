# Model assets

[简体中文](model_assets.zh.md) | English

RhinoForge does not redistribute model checkpoints. Obtain each checkpoint from
the model owner or through an authorized asset channel, accept its separate
terms, and verify its public repository and revision before use.

This page separates profile correctness from asset provenance. The v1.0.0
ledger fixes every public upstream identity used by this source release;
derived local artifacts remain outside that ledger and do not change the
status in [Model support](model_support.md).

## Required record for a runnable profile

Every runnable profile must publish all of these fields together:

- model family and exact profile;
- source identifier and immutable revision, when the model owner provides one;
- model license and access conditions;
- source or quantized checkpoint format;
- conversion command and converter version, when conversion is required;
- the matching example TOML;
- the compatible public RhinoForge release tag; and
- the required Rhino Launch and combined operator-asset release versions.

Do not combine model or runtime-asset versions from different supported
profiles.

The machine-readable v1.0.0 upstream ledger is
[`release/public-models-v1.0.0.json`](../release/public-models-v1.0.0.json). It
pins only official public repositories and immutable revisions. It deliberately
does not name customer assets or claim that a derived quantized directory is
identical to its public source checkpoint.

## Release identity boundary

The JSON ledger above is the v1.0.0 identity contract for every public
upstream integration in this repository. Each record includes the public
repository, immutable revision, access condition, declared license status, and
checkpoint format; an implementation repository and revision are included when
the adapter depends on model-owned code.

The ledger does not turn a Source-only or Limited profile into Supported, does
not grant access to gated assets, and does not identify locally derived W8/W4
outputs. A derived checkpoint remains outside the public identity ledger unless
a public release lists it explicitly.

Compatibility is defined by the public RhinoForge tag and the required Rhino
Launch and operator-asset release versions. It does not depend on an internal
source commit or tree, a receipt, or runtime linker/source mappings.

## Public upstream sources

The JSON ledger above is authoritative for repository, revision, access,
license declaration, and checkpoint format. Its Wall entry binds the public
`wall-oss-0.5` checkpoint to the Apache-2.0 `wall-x` implementation, but the
checkpoint repository declares no weight license. Its G0.5 entry binds the
public Galaxea implementation and gated `g05-base` assets; those assets remain
non-commercial under the G0.5 Community License.

SigLIP used by Pi0.5 is part of the pinned Pi0.5 bundle rather than a separate
RhinoForge checkpoint identity. W8/W4 outputs are derived assets and are not
covered by a runnable release claim unless a public release lists them.

## Download pattern

For a release row that names a Hugging Face source and immutable revision:

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
hf download SOURCE_ID \
  --revision REVISION \
  --local-dir "$RPU_MODEL_CACHE/LOCAL_MODEL_DIRECTORY"
```

Use the source identifier, revision, subfolder (when present), and access terms
from the same release row. Access to a gated model must be granted by the model
owner.

The exact Hy-Embodied profile downloads directly into its registry path:

```bash
hf download tencent/Hy-Embodied-0.5-VLA-UMI \
  --revision 3f53d1f8d2bc587c523cfdc9f1041ceee42c2524 \
  --local-dir "$RPU_MODEL_CACHE/Hy-Embodied-0.5-VLA-UMI"
```

## Existing offline converters

Run converters only for a profile that explicitly names the corresponding
output format.

```bash
# Qwen3 14B local-conversion recipe
python -m rpu_backend.quant.convert_qwen3 \
  --src SOURCE_DIR --dst OUTPUT_DIR --quant-lm-head

# Pi0.5 W8A16
python -m rpu_backend.quant.convert_pi05 --src SOURCE_DIR --dst OUTPUT_DIR

# Pi0.5 packed W4 evaluation profile
python -m rpu_backend.quant.convert_pi05 \
  --src SOURCE_DIR --dst OUTPUT_DIR --fake-w4 --real-w4

# Wall-OSS W8A16
python -m rpu_backend.quant.convert_wall_oss_w8a16 \
  --src SOURCE_DIR --dst OUTPUT_DIR

# Wall-OSS group-wise W4 evaluation profile
python -m rpu_backend.quant.convert_wall_oss \
  --src SOURCE_DIR --dst OUTPUT_DIR --group-size 32
```

Converters create a new destination and reject an existing output directory.
After conversion, point the matching TOML at that output and retain the source
model and converter versions for reproducibility.

## Runtime assets are separate

Model assets do not replace the two restricted runtime prerequisites:

- The Rhino Launch binary development-package release required by the public
  RhinoForge tag.
- The combined operator-asset release selected through `RPU_KERNEL_LIB_PATH`,
  plus its adjacent kernel manifest.

Neither prerequisite is stored in this repository, a source archive, or a
Python package. Compatibility is determined by the public RhinoForge tag and
the two runtime-asset release versions documented for that tag; no internal
commit, linker mapping, or source-build identity is required. Obtain both
through the authorized distribution channel and verify their supplied
checksums. Follow [Restricted runtime assets](runtime_assets.md) for download,
installation, and revocation handling.

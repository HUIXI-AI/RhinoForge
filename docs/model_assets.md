# Model assets

[简体中文](model_assets.zh.md) | English

RhinoForge does not redistribute model checkpoints. Obtain each checkpoint from
the model owner or through an authorized asset channel, accept its separate
terms, and verify its public repository and revision before use.

Keep provenance records for both original and locally converted checkpoints.
Runtime admission depends on the checkpoint format, precision and input scope
described by the corresponding configuration and API. Source identity alone
does not establish support; see [Examples](model_support.md).

## Model provenance and precision

Record the public checkpoint source, revision, license and actual precision.
Keep conversion commands, converter versions and quantization metadata for
derived checkpoints so the loader can distinguish AWQ, INT4, NVFP4 and W8.
Model weights and request data are supplied separately from source and wheels.

Examples are archived by model under [`examples/configs/`](../examples/configs/).
See the [example index](../examples/README.md) for entry points and configuration
guides. Current API admission checks define the runnable envelope.

## Public upstream sources

Use [Getting started](getting_started.md) for the pinned Qwen3-0.6B download.
The [Qwen3](../examples/configs/qwen3/README.md),
[Qwen3-VL](../examples/configs/qwen3_vl/README.md) and
[Pi0.5](../examples/configs/pi05/README.md) configuration guides describe the
checkpoint formats, conversion options and input constraints for their examples.

The public Wall-OSS base checkpoint is
[`x-square-robot/wall-oss-0.5`](https://huggingface.co/x-square-robot/wall-oss-0.5).
Use its processor, normalizer and action definitions as described in the
[Wall-OSS example](../examples/README.md#公开-wall-oss-与-rhinovla). For G0.5, the
[configuration](../examples/configs/g05/base/fp16.toml) records the source,
revision, subfolder and implementation. Consult the model owner's access and
license terms and [third-party notices](../THIRD_PARTY_NOTICES.md); RhinoForge's
source license does not grant rights to these weights.

Keep the complete checkpoint, including its configuration, tokenizer and
processor files. Pi0.5's SigLIP weights belong to the policy checkpoint.
Locally converted weights must match the selected configuration's format and
do not inherit the original checkpoint's task-quality conclusions.

## Download pattern

For a Hugging Face checkpoint, select the repository and immutable revision
that match the intended configuration:

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
hf download SOURCE_ID \
  --revision REVISION \
  --local-dir "$RPU_MODEL_CACHE/LOCAL_MODEL_DIRECTORY"
```

Use any source, revision and subfolder pinned by the selected configuration or
guide. When an example specifies only a local alias, obtain the checkpoint from
its owner and record the actual repository and revision. Aliases map local cache
paths; they do not download weights or pin upstream revisions. Access to a gated
model must be granted by the model owner.

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

# Pi0.5 mixed W4A16-G32-KV8 evaluation profile (not pure W4)
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

Model assets do not replace the external runtime prerequisites:

- A matching sanitized Release Rhino Launch package with headers and libraries.
- The matching combined operator asset selected through `RPU_KERNEL_LIB_PATH`,
  plus its adjacent `.kernels` manifest.
- The matching board SDK/runtime configured according to the platform's
  installation instructions.

These components are not stored in this repository, a source archive, or a
Python package. Obtain assets compatible with the source revision from the
runtime provider and verify its supplied checksums. Package versions alone do
not establish compatibility; keep build-time and run-time Launch paths
consistent. Follow [External runtime assets](runtime_assets.md) for required
capabilities, path configuration and operator-manifest checks.

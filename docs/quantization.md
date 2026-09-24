# Quantization

[简体中文](quantization.zh.md) | English

RhinoForge provides offline checkpoint converters for selected model families.
Quantization support is profile-specific: a converted checkpoint does not
inherit the support status of its FP16 source or of another model size. Check
[Examples](model_support.md) and the release's
[Model assets](model_assets.md) before using a converted checkpoint.

All converters read a source checkpoint and create a new destination directory.
They refuse to overwrite an existing destination and clean up their temporary
directory on failure. Keep the original checkpoint immutable and record hashes
for both source and converted files.

## Qwen3 W8A16

The Qwen3 converter replaces the seven decoder projection weights with signed
INT8 values and one FP16 scale per output channel. Other floating tensors remain
FP16 by default. In v1.0.0 every generated W8A16 checkpoint is Source-only
because no public immutable derived checkpoint identity or hash is bound. The
14B local-conversion recipe uses an untied INT8 `lm_head` and FP16 embeddings:

```bash
python -m rpu_backend.quant.convert_qwen3 \
  --src /path/to/Qwen3-14B \
  --dst /path/to/qwen3-14b-w8a16-lmhead-int8 \
  --quant-lm-head
```

Options:

- `--skip-modules NAME [NAME ...]` excludes matching decoder projections;
- `--quant-lm-head` writes an untied, quantized `lm_head`; and
- `--quant-embed-tokens` also quantizes the embedding and requires
  `--quant-lm-head`. This embedding path remains Source-only.

All Qwen3 W8A16 converter outputs remain Source-only in v1.0.0.

Use the same options when running the checkpoint verifier:

```bash
python -m rpu_backend.quant.verify_qwen3_w8a16 \
  --src /path/to/Qwen3-14B \
  --dst /path/to/qwen3-14b-w8a16-lmhead-int8 \
  --quant-lm-head
```

The verifier checks the declared method, expected INT8/scale tensor pairs, and
a representative sample of dequantized weights. Its default sample limit is 21
and its default minimum weight cosine is `0.999`; these are converter checks,
not model-level acceptance criteria. Use `--max-samples` and `--min-cosine` to
apply the release procedure's values, then run end-to-end validation.

## Qwen3 explicit W4/G32 installation

The public CausalLM loader accepts `quantization="w4a16"` for the exact dense
0.6B, 1.7B, 4B, 8B, 14B and 32B source geometries. It quantizes the seven
decoder projections once during installation; `quantization="w4a16_lm_head"`
also quantizes the head. The source must contain original floating weights,
not AWQ/GPTQ or offline W8 tensors. Embeddings, norms and activations stay FP16.
These recipes require TP8 and batch one; cold `prefill.linear_acc32=false/true`
selects ACC16/ACC32. They do not extend ordinary FP16 admission or certify
model quality. See the [Qwen3 templates](../examples/configs/qwen3/README.md).

## Qwen3-VL

Use `python -m rpu_backend.quant.convert_qwen3_vl` for offline 2B/4B/8B conversion.
`--bits 8` converts public FP16 weights to W8A16. For the `awq_w4.toml`
examples, use `--bits 4` with symmetric G32 compressed-tensors AWQ input:
Text AWQ is preserved, the LM head uses RTN W4, and Vision uses W8.
The converter also accepts the exact unquantized public 2B/4B/8B FP16 profile
with `--bits 4`; that produces RTN W4 text/head weights, not AWQ weights.
The two recipes have distinct metadata. Incompatible input formats are rejected
before conversion.

```sh
python -m rpu_backend.quant.convert_qwen3_vl --bits 8 --src SOURCE_DIR --dst W8_DIR
python -m rpu_backend.quant.convert_qwen3_vl --bits 4 --src AWQ_DIR --dst W4_DIR
```

Load the converted directory without a precision environment override. The
original AWQ directory and converted runtime directory are distinct artifacts.

## Pi0.5

The default Pi0.5 conversion is W8A16 for the VLM decoder and action-expert
decoder projections. Vision encoder, AdaRMS dense layers, action projection,
and processor sidecars remain FP16 or unchanged.

Pi0.5 W8A16 and mixed W4A16-G32-KV8 outputs are Source-only in v1.0.0: the
release binds no public immutable derived checkpoint identity or hash for them.
Their exact same-quantization oracle and checkpoint-owned task evidence also
remain pending.

```bash
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source \
  --dst /path/to/pi05-W8A16
```

The converter also exposes two W4 evaluation formats:

```bash
# Signed four-bit values stored in INT8 tensors; numerical probe only.
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source \
  --dst /path/to/pi05-fake-W4 \
  --fake-w4

# Runtime mixed W4A16-G32-KV8 evaluation format (not pure W4).
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source \
  --dst /path/to/pi05-mixed-W4A16-G32-KV8 \
  --fake-w4 --real-w4
```

The runtime profile is mixed W4A16-G32-KV8, not pure W4. Only the declared
Gemma VLM/action-expert Linear projection weights are quantized:
q/o/gate/up/down use symmetric W4 group-size-32 weights along K, while K/V
projection weights remain W8. Activations and the KV cache remain FP16. The
checkpoint stores logical FP16 scales as `[K/32, N]`; loading stripes them into
the packed pgrp ABI. SigLIP, AdaRMS dense, action projection, and processor
sidecars remain FP16 or unchanged.

`--keep-int8` accepts a comma-separated list of projection names for a
controlled mixed-precision W4 experiment. `--real-w4` is the legacy CLI name
for the mixed profile and automatically keeps the K/V projection weights at W8.
Both W4 modes are Source-only evaluation paths and require their own immutable
asset identity, exact same-quantization oracle, and checkpoint-owned task
evidence before promotion.

The converter deliberately omits a stale remapped checkpoint so the Pi0.5
loader can regenerate it from the new quantized tensors.

### Optimized Action NVFP4 format

```sh
python -m rpu_backend.quant.convert_pi05 --src SOURCE_DIR --dst NVFP4_DIR --action-w4 --action-w4-format nvfp4
```

`--action-w4` defaults to NVFP4 with `striped_v2` ABI and block16. Action
Q/O/Gate/Up/Down use NVFP4; K/V and VLM remain W8. The converter writes
`rpu_quant_config.json`. Use this format with
`optimized_profile.precision=w8_action_nvfp4`; the two- or three-camera
`w8_prefill_a8_action_nvfp4` variant additionally selects Prefill GateUp A8 at
installation. It is not interchangeable with the legacy `--real-w4` INT4
G32/KV8 format described above.

## Wall-OSS

The W8A16 converter quantizes every floating two-dimensional weight except
modules matched by `--skip-modules`; embeddings are skipped by default.

```bash
python -m rpu_backend.quant.convert_wall_oss_w8a16 \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W8A16
```

The W4 converter supports per-channel scales by default and group-wise scales
when `--group-size` is positive. A positive group size must divide the input
dimension of every converted W4 layer:

```bash
# Per-channel W4.
python -m rpu_backend.quant.convert_wall_oss \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W4A16

# Group-wise W4; 32 is the existing evaluation setting.
python -m rpu_backend.quant.convert_wall_oss \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W4A16-g32 \
  --group-size 32
```

Do not pass a non-empty `--keep-int8` to the current Wall-OSS W4 converter. The
runtime does not consume that mixed-precision metadata, so the converter rejects
the option. Wall-OSS W4 remains Source-only.

The W4A16 runtime consumes only the packed group representation. For an input
width `K` and group size 32, existing helpers repeat each one-dimensional
per-output-channel scale across `K / 32` groups, then produce the
controller-striped layout. Do not add a model-specific packing or tile
override. FP16, W8A16, and W4A16 Linear paths all use the shared generated auto-tiler described in
[Generated Linear auto-tiling](../knowledge/concepts/linear-autotiling.md).

## Qwen3.5 Dense image-text examples

The [Dense VL configurations](../examples/configs/qwen3_5/README.md) select
checkpoint-declared text projection quantization with an FP16 vision tower.
Use `rpu_backend.quant.convert_qwen3_5` to produce the complete conditional
generation checkpoint and `RPUModelForConditionalGeneration` to load it;
ordinary Hugging Face loading can lose integer storage and scale metadata.
Embedding, LM head, norms and GDN convolution remain floating point.
The example checks the declared precision before processor or weight loading.

Quantized image evaluation requires both
`QWEN3_5_QUANT_ALLOW_UNCERTIFIED=1` and
`QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1`. These are separate admission controls;
neither declares numerical or task-quality certification. See the configuration
index for the exact sizes, quantization methods and core counts. MoE mixed-E4M3
and legacy text bundles keep their separate formats.

## RhinoVLA v3 expert examples

The [v3 W8 configurations](../examples/configs/rhinovla/v3/README.md) pass
`expert_w8a16` and `full_w8a16` to a caller-owned, trusted runtime factory.
`w8a16.toml` quantizes the action expert's Q/K/V/O and gate/up/down projections.
`full_expert_w8a16.toml` also selects its AdaRMS condition projections.
Both use INT8 weights with FP16 scales and activations; neither template selects
vision, prefix-language or action-IO quantization. The runner verifies actual
installed expert storage rather than accepting a precision label from the
factory. Checkpoint composition and preprocessing remain owned by that factory.

## Source-only helpers

Hy-Embodied W8/W4 are runtime-derived evaluation paths. v1.0.0 binds no public
immutable derived checkpoint identity or hash for them, so they are
Source-only; the public FP16/W16 profile does not transfer its status.

`rpu_backend.quant.convert_qwen3_w4a16` is an in-memory evaluation helper for
Qwen3 0.6B. It is not a general offline `--src`/`--dst` converter and does not
create a distributable checkpoint. Treat it as Source-only unless an exact
release profile says otherwise.

The public `rpu_backend.quant` package exports
`quantize_linear_per_channel` and `dequantize_linear_per_channel` for converter
authors. INT4 packing helpers are implementation-facing and their availability
does not imply model support.

## Metadata and loading

Qwen3 records its W8A16 declaration in `config.json` under `quant_config`.
Pi0.5 and Wall-OSS write `rpu_quant_config.json` beside the checkpoint. Loaders
consume this metadata and apply model-specific checks to the method, tensors,
and profile. Do not hand-edit the metadata or rename scale tensors.

Conversion does not encrypt model weights or change their license. Follow the
source model's terms and the distribution policy recorded in
[Model assets](model_assets.md).

## Validation checklist

Before publishing or selecting a quantized profile:

1. pin and hash every source checkpoint file;
2. run the converter in a new destination and preserve its complete log;
3. verify the output manifest, metadata, tensor names, dtypes, shapes, and file
   hashes;
4. compare sampled or complete dequantized weights with the FP16 source;
5. load the checkpoint through its public RhinoForge entry point and confirm
   fail-fast rejection of mismatched profiles;
6. compare RPU with the exact quantized same-dtype reference for implementation
   parity, then compare retained task quality against the frozen FP16 or CPU
   FP32 anchor under [Model validation policy](validation_policy.md);
7. verify warmup and graph lifecycle behavior; and
8. run the public end-to-end TOML example in a clean process.

Record quantized results separately from the floating-point source, with the
exact checkpoint and execution configuration. An [example](model_support.md)
or a passing weight-cosine check alone is not an end-to-end support result.

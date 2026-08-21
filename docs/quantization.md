# Quantization

RhinoForge provides offline checkpoint converters for selected model families.
Quantization support is profile-specific: a converted checkpoint does not
inherit the support status of its FP16 source or of another model size. Check
[Model support](model_support.md) and the release's
[Model assets](model_assets.md) before using a converted checkpoint.

All converters read a source checkpoint and create a new destination directory.
They refuse to overwrite an existing destination and clean up their temporary
directory on failure. Keep the original checkpoint immutable and record hashes
for both source and converted files.

## Qwen3 W8A16

The Qwen3 converter replaces the seven decoder projection weights with signed
INT8 values and one FP16 scale per output channel. Other floating tensors remain
FP16 by default. The only planned runnable release row is Qwen3 14B with an
untied INT8 `lm_head` and FP16 embeddings:

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
  `--quant-lm-head`. This embedding path is experimental.

Other Qwen3 W8A16 converter outputs are Source-only until an exact model,
checkpoint, configuration, and hardware gate are promoted in the release
manifest.

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

## Pi0.5

The default Pi0.5 conversion is W8A16 for the VLM decoder and action-expert
decoder projections. Vision encoder, AdaRMS dense layers, action projection,
and processor sidecars remain FP16 or unchanged.

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

# Runtime W4 evaluation format.
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source \
  --dst /path/to/pi05-W4A16 \
  --fake-w4 --real-w4
```

`--keep-int8` accepts a comma-separated list of projection names for a
controlled mixed-precision W4 experiment. Real W4 automatically keeps the key
and value projections at INT8. Both W4 modes are Experimental and require their
own numerical and end-to-end evidence.

The converter deliberately omits a stale remapped checkpoint so the Pi0.5
loader can regenerate it from the new quantized tensors.

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
the option. Wall-OSS W4 remains Experimental.

## Source-only helpers

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

Record quantized results as a separate row in
[Model support](model_support.md). A passing weight-cosine check alone is not an
end-to-end support result.

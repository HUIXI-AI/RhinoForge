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
FP16 by default. The 14B local-conversion recipe uses an untied INT8 `lm_head`
and FP16 embeddings:

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

Use the same options when running the checkpoint verifier:

```bash
python -m rpu_backend.quant.verify_qwen3_w8a16 \
  --src /path/to/Qwen3-14B \
  --dst /path/to/qwen3-14b-w8a16-lmhead-int8 \
  --quant-lm-head
```

The verifier checks the declared method, expected INT8/scale tensor pairs, and
representative dequantized weights. These checks cover conversion consistency;
they do not certify a model or device.

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

Use `python -m rpu_backend.quant.convert_qwen3_vl` for offline 2B/4B/8B and
exact LightNav-0 conversion. For LightNav-0, `--bits 8` quantizes only the
Text7 projections and keeps the action head and Vision in FP16. Other public
models retain their existing W8A16 recipes. For the `awq_w4.toml`
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

Select the matching [Pi0.5 profile](../examples/configs/pi05/README.md) and
preserve the generated quantization metadata with the checkpoint.

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

# Runtime mixed W4A16-KV8 evaluation format (not pure W4).
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source \
  --dst /path/to/pi05-mixed-W4A16-KV8 \
  --fake-w4 --real-w4
```

The runtime format uses per-channel signed INT4 for q/o/gate/up/down and W8
for K/V projections. Checkpoints store weights in INT8 tensors with FP16 scales
of shape `[N]`; loading converts them to the runtime packed layout. Activations,
KV cache, SigLIP, AdaRMS dense, and action projection remain FP16. Processor
sidecars are unchanged.

`--keep-int8` accepts a comma-separated list of projection names for a
controlled mixed-precision W4 experiment. `--real-w4` is the legacy CLI name
for the mixed profile and automatically keeps the K/V projection weights at W8.
Keep the fake-W4 diagnostic tensors separate from the runtime mixed-W4 format.

The converter deliberately omits a stale remapped checkpoint so the Pi0.5
loader can regenerate it from the new quantized tensors.

### Optimized Action NVFP4 format

```sh
python -m rpu_backend.quant.convert_pi05 --src SOURCE_DIR --dst NVFP4_DIR --action-w4 --action-w4-format nvfp4
```

`--action-w4` defaults to NVFP4 with `striped_v2` ABI and block16. Action
Q/O/Gate/Up/Down use NVFP4; K/V and VLM remain W8. The converter writes
`rpu_quant_config.json`. Select a two- or three-camera
[`w8a16_action_nvfp4` template](../examples/configs/pi05/README.md).
The `w8a16_prefill_w8a8_action_nvfp4` variant also selects Prefill GateUp A8
at installation. Keep the checkpoint recipe paired with the selected template;
`--real-w4` produces the separate per-channel INT4/KV8 format described above.

The same Action-only converter also accepts these explicit formats:

| `--action-w4-format` | Action weight and scale storage |
|---|---|
| `nvfp4` (default) | E2M1 codes, K16 FP8 scales and a separate FP32 tensor scale; `striped_v2` ABI |
| `int4_g128` | Signed INT4 codes stored as INT8 `[N,K]`, independent FP16 scales `[K/128,N]` |
| `legacy_int4` | Per-output-channel signed INT4; channel scales are repeated when packed |

Pass `--action-w4` and the selected format when converting original floating
weights to a new directory. All three mark `int4_components=["expert"]` and
keep Action K/V and all VLM projections at W8. These are distinct recipes;
renaming a directory or editing its metadata does not convert weights.

The current optimized templates fix the camera count, text bucket and
accumulation mode before installation. `input.text_tokens` selects a physical
bucket; it does not tokenize, truncate or pad the input. Supply matching
`[1,T]` integer tokens and a nonempty, right-padded boolean mask, then prepare
Graphs with that batch. Two-camera input has two real image slots. Changing
camera count or text bucket requires a new policy. The FP16 template requires
floating weights without quantization metadata and disables dynamic Vision
and AdaRMS quantization.

## Wall-OSS

The W8A16 converter quantizes every floating two-dimensional weight except
modules matched by `--skip-modules`; embeddings are skipped by default.

```bash
python -m rpu_backend.quant.convert_wall_oss_w8a16 \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W8A16
```

The W4 converter defaults to group-wise scales with `--group-size 32`.
Use `--group-size 0` for legacy per-channel scales. A positive group size must
divide the input dimension of every converted W4 layer:

```bash
# Per-channel W4.
python -m rpu_backend.quant.convert_wall_oss \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W4A16 \
  --group-size 0

# Group-wise W4 (default).
python -m rpu_backend.quant.convert_wall_oss \
  --src /path/to/wall-oss-source \
  --dst /path/to/wall-oss-W4A16-g32 \
  --group-size 32
```

Do not pass a non-empty `--keep-int8` to the current Wall-OSS W4 converter. The
runtime does not consume that mixed-precision metadata, so the converter rejects
the option. Both decoder experts, including expert-0 Down, use INT4.

The W4A16 runtime consumes only the packed group representation. For an input
width `K` and group size 32, existing helpers repeat each one-dimensional
per-output-channel scale across `K / 32` groups, then produce the
controller-striped layout. Do not add a model-specific packing or tile
override. FP16, W8A16, and W4A16 Linear paths all use the shared generated auto-tiler described in
[Generated Linear auto-tiling](../knowledge/concepts/linear-autotiling.md).

## Qwen3.5 Dense

Convert with `python -m rpu_backend.quant.convert_qwen3_5 --src SOURCE_DIR --dst OUTPUT_DIR`.
Choose `--method w8a16` or `--method w4a16_pgrp` for uniform text projection
quantization, or one of the mixed `--profile` recipes:

| Profile | Text projections |
|---|---|
| `w4_main_w8_gdn` | Main projections W4/G32; GDN QKV/Z/out W8; GDN B/A FP16 |
| `w8_main_w8_gdn` | Main and GDN QKV/Z/out W8; GDN B/A FP16 |
| `w4_mlp_only` | MLP Gate/Up/Down W4/G32; Attention and GDN FP16 |
| `w4_mlp_w8_attn_gdn` | MLP Gate/Up/Down W4/G32; Attention and GDN W8 |

The mixed recipes use `quant_config.format_version=2` in `config.json`.
The last two recipes include MLP projections in every layer and require
`--group-size 32`. Vision, embeddings, LM head, norms and GDN convolution
remain FP16. Floating source values are cast directly to FP16; existing
converted checkpoints must be regenerated to remove an earlier BF16
intermediate cast.

```sh
python -m rpu_backend.quant.convert_qwen3_5 --src SOURCE_DIR --dst MIXED_DIR --profile w4_mlp_w8_attn_gdn
```

Load converted checkpoints through `RPUModelForConditionalGeneration` so
integer weights and scales retain their declared storage. Select the matching
[text or image configuration](../examples/configs/qwen3_5/README.md); the runner
checks quantization metadata before loading weights. Quantized image profiles
require `QWEN3_5_QUANT_ALLOW_UNCERTIFIED=1` and
`QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1` as shown in their templates.

The separate QuantTrio AWQ G128 source profile keeps layer-zero MLP,
Attention, GDN, Vision and LM head in FP16; later MLP projections retain
AWQ weights. Load its original checkpoint with the matching AWQ template;
these RTN converter recipes do not produce that format.

## Qwen3.5 MoE mixed E4M3

Generate this checkpoint from the official unquantized `Qwen/Qwen3.5-35B-A3B`
with the standalone CPU converter. Run from the repository root with an existing
source checkpoint and a new destination directory:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5_moe_fp8.py \
  --src SOURCE_DIR --dst OUTPUT_DIR --expert-chunk 1 --threads 8
```

The [complete download and run recipe](../examples/configs/qwen3_5/README.md#cmd-moe-mixed)
uses `$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-rpu-fp8-mixed` for the output. Conversion
requires PyTorch, safetensors and Transformers with `Qwen3_5MoeConfig`; it needs
no calibration data or device runtime.

The converter quantizes 210 routed-expert, full-attention Q/K/V/O and GDN
QKV/Z/out tensors to raw UINT8 E4M3 codes with per-output FP16 scales. Other
floating tensors, including router, shared expert, head, embeddings, norms and
Vision, are stored as FP16. It retains configuration, tokenizer, processor and
shard metadata. Official block-FP8 `weight_scale_inv`, GPTQ and RTN checkpoints
are not valid sources. Text/VL, core-count and ACC templates share this output;
use each template's existing admission rules. Newly converted checkpoints do
not inherit the identity or measurements of historical derived assets.

## Qwen3.5 MoE routed INT4

The 35B-A3B RTN and official GPTQ profiles quantize routed Gate/Up/Down experts
to UINT4/G128. Attention, GDN, router, shared expert, embeddings, norms and LM
head use FP16. Both profiles require batch one, eight cores and Text ACC16;
`prefill.linear_acc32=true` is rejected. Vision has its own accumulation flag.

| Source | Required checkpoint declaration |
|---|---|
| RTN | Exact `rtn_metadata`, indexed safetensors, completed `derivation_manifest.json`; no calibration |
| Official GPTQ | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`, symmetric four-bit G128, no act-order, routed-only exclusions |

RTN stores zero-minus-one words `0x77777777`; official GPTQ stores zero8 words
`0x88888888`. Their loaders validate these distinct conventions before using
the shared UINT4 runtime. Original integer codes and FP16 scales are preserved
for GPTQ. A present but invalid `rtn_metadata` field cannot fall back to GPTQ.
Neither entry accepts arbitrary GPTQ metadata or the mixed-E4M3 format.

For RTN, promote source groups to FP32 and compute
`s=max(abs(group))*(2/15)` (use `2/15` for an all-zero group), then
`q=clamp(round_even(w/s)+8,0,15)`. Cast the scale to FP16 only after selecting
codes; preserve scales that underflow to zero. Dequantization is `(q-8)*s`.
The source stores INT32 `qweight[K/8,N]`, FP16 `scales[K/128,N]`, and
`g_idx[k]=k//128`. RTN and GPTQ retain their separate zero-point conventions.
At TP8, Gate/Up retain G128; Down repeats each original G128 scale into two
physical G64 groups without changing codes or dequantized values. Packing
happens during installation and must match the release's operator pair.

The text aliases are `qwen3_5-35b-a3b-rtn-w4a16` and
`qwen3_5-35b-a3b-gptq-w4a16` under `RPU_MODEL_CACHE`; `--model` selects an explicit
checkpoint. GPTQ requires the stock indexed checkpoint and no RTN derivation
metadata. Incomplete RTN staging directories are rejected. Input IDs are CPU
`int64` tensors with shape `[1,T]`; create `Qwen3_5MoeCache.from_model` after
installation with capacity for the complete request.

Use the normal conditional-generation loader and `Qwen3_5MoeCache`. Installation
streams per-layer expert sources without materializing all experts as FP16.
The indexed reader validates shard paths, headers, shapes, dtypes and inventory;
verify full file hashes when preparing the checkpoint. The runtime checkpoint
identity records metadata and file state, not a full payload checksum.

```sh
python -m examples.run --config examples/configs/qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128.toml --model RTN_DIR
python -m examples.run --config examples/configs/qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128.toml --model GPTQ_DIR
```

The corresponding `vl/35b_a3b/` templates add images. Supply `input_ids`,
`pixel_values`, `image_grid_thw` and `mm_token_type_ids` together at cache
position zero, within the installed Vision KV and RoPE bounds. Video is
unsupported. Install once and call `cache.reset()` between requests. Prefill
uses bounded one-shot execution; decode retains a replay graph. Logits are
returned as CPU FP32 after FP16 computation.

## RhinoVLA v3 expert examples

The [v3 configurations](../examples/configs/rhinovla/v3/README.md) use the built-in
`rpu_backend.adapters.rhinovla.factory:v3_runtime_factory` with the pinned v3
source, checkpoint and prepared CPU inputs. `w8a16_action.toml` selects action
Q/K/V/O and Gate/Up/Down; `w8a16_action_adarms.toml` also selects AdaRMS condition
projections. Both use INT8 weights and FP16 scales/activations. Vision, prefix
language and action IO keep their selected floating format. A custom trusted
factory may implement the same input and lifecycle contract; the runner checks
installed expert storage rather than relying on its precision label.

`w8a16_pipeline.toml` additionally quantizes Vision, Text, action IO and final
AdaRMS condition projections. The HIGH/vector QK variant keeps that weight recipe
and selects the documented nonlinear and normalization options. These four W8
recipes and FP16 each have an `rtc_*.toml` counterpart: RTC preserves the same
weight scope and prepares per-delay conditioning with the selected projection
weights and scales. Use the built-in v3 factory and supply an action prefix plus
`delay=0..6`; the [RTC guide](../examples/configs/rhinovla/v3/README.md#rtc)
describes input bounds and verification status. No new quantized checkpoint
directory is required for RTC.

## Runtime conversion helpers

Hy-Embodied W8/W4 are runtime-derived paths with no offline export CLI. Select
the input and conversion recipe in the [Hy-Embodied guide](../examples/configs/hy_embodied/README.md).

`rpu_backend.quant.convert_qwen3_w4a16` is an in-memory evaluation helper for
Qwen3 0.6B. It is not a general offline `--src`/`--dst` converter and does not
create a distributable checkpoint. Treat it as Source-only unless an exact
release profile says otherwise.

The public `rpu_backend.quant` package exports
`quantize_linear_per_channel` and `dequantize_linear_per_channel` for converter
authors. INT4 packing helpers are implementation-facing; use the
[model profiles](model_support.md) to select supported configurations.

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
6. compare device with the exact quantized same-dtype reference for implementation
   parity, then compare retained task quality against the frozen FP16 or CPU
   FP32 anchor under [Model validation policy](validation_policy.md);
7. verify warmup and graph lifecycle behavior; and
8. run the public end-to-end TOML example in a clean process.

Record quantized results separately from the floating-point source, with the
exact checkpoint and execution configuration. An [example](model_support.md)
or a passing weight-cosine check alone is not an end-to-end support result.

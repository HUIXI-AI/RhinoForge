# LightNav-0 R1 Run Guide

[简体中文](lightnav_0_r1.zh.md) | English

This port targets the public `LightOriginsHQ/LightNav-0` model and pins:

- RhinoForge branch: `feat/lightnav-0-r1-cache`
- LightNav-0 revision: `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`
- Upstream LightNav source commit: `3015508b70fb6e30bbd66701e94c9b916dcb8df8`
- R1 runtime: v1.2.0

Each request produces five tokens: two pointing tokens and three RVQ action
tokens.

## Current delivery configuration

| Item | Configuration |
|---|---|
| Input window | Up to nine history frames plus one current frame |
| Vision cache | Two frames per tubelet; steady full-window result is 4 hits and 1 miss |
| Prefetch | `prefetch_latest=False` to avoid recomputing a duplicate tubelet |
| Text precision | W8A16 for layers 0–3 Attention O projections; FP16 otherwise |
| Vision precision | FP16 |
| Retained at FP16 | Remaining Text, Vision, action output head, embedding, and norms |
| Output head | Five-position action grammar, physically padded to 1,408 rows per position |
| Decode | Standard autoregressive generation of five tokens |
| Previous-request token draft | Disabled |
| Graphs | Safely rebound Prefill and Decode Graphs reused after the full window is established |

The Vision cache stores the pooler output and three DeepStack features for each
tubelet. Historical hits are reused, and only the current missing tubelet runs
the ViT. Vision embeddings and M-RoPE change with every request, so the text
path still executes a complete Prefill. Graph reuse only reduces capture and
kernel-dispatch overhead; it does not reuse text results from the previous
request.

W4 is not part of the current delivery. Mixed W4/W8 experiments were faster
but changed action tokens and failed the accuracy gate.

## Environment

The GitHub branch contains RhinoForge source only. It does not include the
runtime, a prebuilt `rpu_backend.so`, upstream LightNav source, or model weights.

1. Follow the [unified installation guide](getting_started.md) to create a
   Python 3.12 environment and build RhinoForge from the current checkout. Do
   not reuse an `rpu_backend.so` built from another commit.
2. Follow [runtime assets](runtime_assets.md#configure-paths) to configure the
   official v1.2.0 runtime and set `RHINO_LAUNCH_ROOT`,
   `RPU_KERNEL_LIB_PATH`, `RPU_KERNEL_LIB_PATH_EXPANSION`, and
   `LD_LIBRARY_PATH`. The runtime must match
   [`release/runtime-dependencies.json`](../release/runtime-dependencies.json).
3. Obtain the upstream LightNav source and complete checkpoint revisions pinned
   above. Do not run `pip install -e .` in the LightNav checkout; the runner
   loads it through `--lightnav-source`.

The model directory must contain the weights, `eval_config.json`,
processor/tokenizer files, and `action_tokenizer/`.

## W8A16 conversion

Create the output-parity checkpoint from the original FP16 checkpoint:

```bash
PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --bits 8 \
  --lightnav-parity \
  --src /path/to/LightNav-0 \
  --dst /path/to/LightNav-0-parity-w8a16
```

This profile quantizes only the Attention O projections in Text layers 0–3.
The remaining Text weights, Vision, action output head, embedding, and norms
stay FP16. This scope matches full FP16 for all 175 tokens in the 35-frame set
below. Full-Text W8 and Vision W8 change outputs and are not parity profiles.

## Run an ordered image sequence

The input JSON must contain a nonempty `prompt` (or `instruction`) and an
ordered `images` array. Only `task="tracking"` is supported. Image paths are
relative to the JSON by default or to `--images-dir` when provided.

Validate paths and input first:

```bash
python examples/lightnav_inference.py \
  --input /path/to/dataset.json \
  --images-dir /path/to/images \
  --checkpoint /path/to/LightNav-0-parity-w8a16 \
  --lightnav-source /path/to/LightNav-0-source \
  --history-frames 9 \
  --constrained-head \
  --output /path/to/inference_result.json \
  --check-config
```

After validation, remove `--check-config` and run the same command on R1. Do not
add `--history-draft`; the current performance result uses standard five-token
autoregressive Decode.

The output JSON records five tokens, input-token count, Vision cache hits and
misses, and stage timings for every frame. `total` includes preparation, Vision,
and text generation. `frame_total` also includes Graph clearing and outer
request processing. Both exclude model loading and the initial decoding and
resizing of all images.

`--constrained-head` selects the LightNav five-position action grammar and is a
condition of the performance result below. Applications that use full-vocabulary
decoding must validate output and performance separately.

## Current performance

Test conditions:

- the ordered 35-frame customer sequence;
- nine history frames plus one current frame per request, with 347 prompt
  tokens at the full window;
- a steady 4/1 Vision cache result, with tubelet prefetch and previous-request
  token drafts disabled;
- W8A16 Attention O projections in Text layers 0–3, with the rest of the model
  in FP16, and the five-position grammar head;
- medians over the 25 steady requests where `frame_index >= 10`;
- model loading, initial image processing, and window filling excluded from the
  steady result.

| Run | Prepare | Vision | Prefill | Decode 5 | Total | Frame total | Output equality |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full FP16 | 17.736 ms | 75.981 ms | 139.914 ms | 210.631 ms | **444.629 ms** | **447.406 ms** | Reference |
| Parity W8/FP16 mix, cold run 1 | 17.977 ms | 74.080 ms | 146.035 ms | 220.407 ms | **459.340 ms** | **461.777 ms** | 35/35 frames, 175/175 tokens |
| Parity W8/FP16 mix, cold run 2 | 18.120 ms | 74.464 ms | 146.212 ms | 220.383 ms | **459.264 ms** | **462.291 ms** | 35/35 frames, 175/175 tokens |

Stage medians are computed independently and may not add up exactly to the
median Total. The parity profile cannot use the all-W8 specialized schedule and
is not faster than FP16; it exists to preserve outputs. Do not quote the old
approximately 248 ms all-W8 result as a parity result. That comparison used a
Text-W8 + Vision-FP16 baseline rather than full FP16. Full W8 matches only
165/175 tokens against full FP16 on this set.

The complete summary is in
[`benchmark_results/lightnav_quant_parity_20261118/summary.json`](../benchmark_results/lightnav_quant_parity_20261118/summary.json).

## Correctness and limits

The runner compares the pooler and DeepStack features from cached and recomputed
paths with an FP16 maximum absolute tolerance of `1e-3`. If the check fails, do
not loosen the tolerance. First verify that the RhinoForge commit,
`rpu_backend.so`, runtime operator assets, and model checkpoint match.

Equality across these 35 frames and 175 tokens proves token-level parity only
for this offline sample. It does not replace broader closed-loop navigation
success-rate evaluation.

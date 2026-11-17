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
| Text precision | W8A16 |
| Vision precision | LightNav-calibrated W8A16 |
| Retained at FP16 | Action output head, embedding, and norms |
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

Create the current Text + Vision W8A16 checkpoint from the original FP16
checkpoint:

```bash
PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --bits 8 \
  --lightnav-vision-w8 \
  --src /path/to/LightNav-0 \
  --dst /path/to/LightNav-0-text-vision-w8a16
```

This profile uses W8A16 for Text7 projections and installs per-channel Vision
quantization. The LightNav Vision encoder uses a calibrated 1.04 range scale;
patch embedding and merger use the standard 1.0 scale. The action output head,
embedding, and norms remain FP16. The converter accepts only the exact
LightNav-0 configuration.

## Run an ordered image sequence

The input JSON must contain a nonempty `prompt` (or `instruction`) and an
ordered `images` array. Only `task="tracking"` is supported. Image paths are
relative to the JSON by default or to `--images-dir` when provided.

Validate paths and input first:

```bash
python examples/lightnav_inference.py \
  --input /path/to/dataset.json \
  --images-dir /path/to/images \
  --checkpoint /path/to/LightNav-0-text-vision-w8a16 \
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
- Text W8A16, calibrated Vision W8A16, and the five-position grammar head;
- medians over the 25 steady requests where `frame_index >= 10`;
- model loading, initial image processing, and window filling excluded from the
  steady result.

| Run | Prepare | Vision | Prefill | Decode 5 | Total | Frame total | Output equality |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial production run | 17.310 ms | 26.879 ms | 89.169 ms | 114.535 ms | **248.388 ms** | **252.037 ms** | 35/35 frames, 175/175 tokens |
| Independent cold reload | — | 26.828 ms | 88.980 ms | 114.494 ms | **247.856 ms** | **251.554 ms** | 35/35 frames, 175/175 tokens |

Stage medians are computed independently and may not add up exactly to the
median Total. The current delivery result is approximately 248 ms of steady
request latency on this offline customer sequence, including about 114.5 ms of
normal Decode, with all 175 tokens matching the FP16 reference.

The fastest mixed W4/W8 experiment reached 220.200 ms but matched only 130/175
tokens. The more conservative Gate/Up W4 profile reached 236.765 ms and matched
only 144/175 tokens. Neither is a delivery result and neither should be quoted
as current supported performance.

The complete raw data is in
[`benchmark_results/lightnav_9_history_20261112/summary.json`](../benchmark_results/lightnav_9_history_20261112/summary.json).

## Correctness and limits

The runner compares the pooler and DeepStack features from cached and recomputed
paths with an FP16 maximum absolute tolerance of `1e-3`. If the check fails, do
not loosen the tolerance. First verify that the RhinoForge commit,
`rpu_backend.so`, runtime operator assets, and model checkpoint match.

Equality across these 35 frames and 175 tokens proves token-level parity only
for this offline sample. It does not replace broader closed-loop navigation
success-rate evaluation.

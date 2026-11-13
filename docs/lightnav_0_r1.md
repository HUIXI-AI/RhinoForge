# LightNav-0 runtime and cache

[简体中文](lightnav_0_r1.zh.md) | English

This extension targets public checkpoint `LightOriginsHQ/LightNav-0` at
Hugging Face revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64` and upstream
LightNav commit `3015508b70fb6e30bbd66701e94c9b916dcb8df8`. Every request
produces exactly five tokens: two pointing tokens and three RVQ action tokens.

RhinoForge reuses its Qwen3-VL 4B text, vision, and DeepStack paths. With the
constrained head enabled, the five fixed output positions retain only the
corresponding `apos/opos/L0/L1/L2` candidates plus EOS/STOP. Each physical head
is padded to 1,408 rows. Prefill uses the first head, and the four decode calls
rebind the following head addresses in one checked-replay Graph. Admission
remains restricted to the exact LightNav profile.

The vision cache uses upstream two-frame tubelet keys and defaults to 1,024
entries, following the upstream Thor long-session recommendation. Each miss runs the ViT
once and stores its pooler output plus three DeepStack levels on CPU. Post-ViT
spatial pooling keeps the unpooled values and caches each tubelet by SlowFast
factor. New historical-factor results are computed on a CPU worker while the
current request runs prefill/decode, so the next sliding window can reassemble
them without recomputing the whole segment.

A real eight-history-plus-current sliding caller can pass
`prefetch_latest=True` to `_run_request`. The current ViT batch also computes
the duplicated-latest-frame boundary tubelet needed by the next request, moving
steady-state accounting from 3/2 to 4/1. `vision_prefetches` reports that work.
This moves one miss earlier; the ViT still evaluates two tubelets per steady
request.

For one physical prompt shape, prefill Graph replay refreshes stable prompt,
visual-embedding, and M-RoPE slots in place. A compatible replacement KV cache
rebinds both KV-insert write registers and attention read registers according to
the resolved chunk/layer traversal. Incompatible layout, dtype, capacity, or
device metadata rejects fast replay. Growing windows should clear old text
Graphs through the first full eight-frame window, then reuse only the fixed
shape so multiple prefill shapes do not exhaust the Graph arena.

## Sequential optimization for nine history frames plus current

The benchmark uses the customer-provided 35-frame sequence, R1 v1.2.0, Text
W8A16, and no previous-request token draft. Values are medians over the 25
steady requests from frame index 10. Each request has nine history frames plus
one current frame, 347 prompt tokens, and a stable 4/1 tubelet cache hit/miss.

| Stage | Prepare | Vision | Prefill | Decode 5 | Total | Frame total |
|---|---:|---:|---:|---:|---:|---:|
| Before | 20.879 ms | 87.947 ms | 89.066 ms | 114.759 ms | 313.817 ms | 318.510 ms |
| Tubelet pooling cache | 17.637 ms | 63.384 ms | 89.119 ms | 114.758 ms | 285.661 ms | 289.438 ms |
| Plus position grammar heads | 17.448 ms | 63.451 ms | 88.486 ms | 114.396 ms | 284.663 ms | 288.472 ms |

Tubelet-level pooling removes repeated whole-segment work after a window shift,
reducing Vision by 24.563 ms and end-to-end latency by 28.156 ms. The five
position heads reduce another roughly 1.0 ms. All 35 frames remain token-exact
after both changes. Fixed-address replay, rebuild replay, and alternating KV
addresses were also checked for token equality.

A fixed four-step decode Graph was not added. Tokens two through five depend on
the previous token, while the current runtime lacks a device-side
`top1 -> original token ID -> embedding -> next decoder` chain. Host selection
and upload cost about 1.1 ms total; decoder model execution costs about 113.3 ms.

Current-frame `pool_spatial=2` reduced Total to 277.802 ms but matched only 1/35
frames, so it remains rejected. Uncalibrated Vision W8A16 reduced Total to
247.730 ms but matched 29/35 frames and 164/175 tokens. Calibrating the encoder
range to 1.04 while retaining the standard patch/merger range matched all 35/35
frames and 175/175 tokens, with 26.879 ms Vision, 248.388 ms Total, and 252.037 ms
frame total. This reduces Vision by 57.6% and Total by 12.7% versus Vision FP16.
This proves token equality on the customer set, not navigation-task
accuracy on a broader evaluation. An independent cold reload repeated 35/35
frames and 175/175 tokens.

On top of that calibrated Vision W8A16 configuration, the decode device trace
attributed 44.8% to Gate/Up, 39.2% to the QKV and Down GEMVs combined, and only
2.4% to attention. The existing projection/ring/norm kernel is therefore reused
to combine Down, AllReduce, the residual update, and the next RMSNorm for layers
3 through 34. Layers 0 through 2 keep the original path because DeepStack is
injected after the MLP, and the final layer also stays unchanged. All 175/175
tokens across the 35 frames remain identical. The steady-state mean
`decode_5` time changes from 114.625 ms to 114.285 ms and Total from 248.851 ms
to 248.617 ms. A single-decode device trace changes from 27.134 ms to 27.079 ms;
this is a verified small gain, while Gate/Up and weight traffic remain the main
bottlenecks.

Three W4A16/G32 profiles were then measured to reduce weight traffic further.
All-MLP W4 reached 84.437/91.843/220.200 ms Prefill/Decode/Total, but matched
only 130/175 tokens and 11/35 frames. Gate/Up W4 reached
89.448/101.874/236.765 ms and matched 144/175 tokens and 19/35 frames. The final
profile quantized only Up while retaining Attention, Gate, and Down at W8A16
and the action head at FP16. Its early accuracy gate still matched only 8/45
tokens and 0/9 frames on the first nine identical input windows, so a full
timing run was not performed. All three W4 profiles are excluded from the
delivery code; the token-exact 175/175 W8A16 weight path remains the production
configuration. The lower W4 latency does not justify changing the hierarchical
action tokens.

The same production build was cold-loaded twice on a second R1 with the same
35-frame input. Both runs matched each other for 35/35 frames and 175/175
tokens. Steady-state medians were 47.655/48.283 ms Vision,
91.024/91.196 ms Prefill, 114.988/115.036 ms Decode, and 276.413/276.618 ms
Total. Prefill and Decode stayed within about 1.4 ms of the original R1 result.
Most of the roughly 27.8 ms Total increase came from Vision, which was about
20.8 ms slower. The stable repeat indicates a machine-side Vision performance
difference rather than a regression in the W8 text-weight path.

See
`benchmark_results/lightnav_9_history_20261112/summary.json` for the full result.

## Run

Prepare RhinoForge, the complete checkpoint directory, and a separate checkout
of the upstream LightNav source. Then validate the request without loading the
model:

```bash
python examples/lightnav_benchmark.py \
  --checkpoint /path/to/LightNav-0 \
  --lightnav-source /path/to/LightNav-0-source \
  --output benchmark_results/lightnav_0_r1.json \
  --check-config
```

Remove `--check-config` on R1. The default uses eight past frames plus one
current frame, for nine actual input frames; `--history-frames` can still
override it. Pass `--history-frames 9` to reproduce the nine-history-plus-current
case above. Both scenarios run five times and report medians, while
every request generates five tokens. Deterministic synthetic 256×448 frames make
the report a latency and cache comparison rather than a navigation-accuracy test.
Each tier warms the full-window and current-tubelet Rhino Graphs outside the
measurement and retains one stable KV-cache address. Its Graphs are released
before the next tier.

When the application uses the fixed five-position LightNav action grammar, pass
`--constrained-head` to install five position-specific 1,408-row projections
and fuse them into single-token decode. This opt-in changes the default
full-vocabulary decoding domain and requires separate navigation-quality
evaluation.

## Thor approach and 8-bit weights

The [upstream Thor guide](https://github.com/lightorigins/LightNav-0/blob/main/docs/JETSON_THOR.md)
quantizes only the language model and keeps Vision in BF16; it reports that
full-model FP8 changed visual decisions and made the ViT slower. R1 does not
provide the same CUDA FP8/vLLM kernels, so this port keeps the same boundary:
the 36-layer Text7 projections use offline W8A16 while the action head,
embeddings, norms, and Vision stay FP16. Admission remains limited to the exact
LightNav-0 profile.

```bash
PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --bits 8 \
  --src /path/to/LightNav-0 \
  --dst /path/to/LightNav-0-text-w8a16
```

The weight file falls from 9.70 GB to 6.06 GB, a 37.5% reduction. It does not
halve because Vision, the action head, and other non-Linear weights remain FP16.

Vision W8A16 uses the calibrated LightNav profile:

```bash
PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --bits 8 --lightnav-vision-w8 \
  --src /path/to/LightNav-0 \
  --dst /path/to/LightNav-0-text-vision-w8a16
```

This profile quantizes Vision per channel during installation and applies the
LightNav calibration above. It matched all 175 tokens on the customer sequence;
Vision FP16 remains the conservative fallback until broader navigation-accuracy
evaluation is complete.

Five fixed output tokens are still autoregressive: tokens two through five each
depend on the previous token. The delivered path uses one shared checked-replay
decode Graph and rebinds the matching 1,408-row head address at each step.

R1 v1.2.0 used eight history frames plus one current frame, deterministic
256x448 input, the 4,369-token action allowlist, and five measured requests per
cache scenario. No previous-request token draft was used:

| Cache scenario | Precision | Prepare | Vision | Prefill | Decode 5 | Total |
|---|---|---:|---:|---:|---:|---:|
| All miss (0/5) | FP16 | 44.3 ms | 316.4 ms | 140.3 ms | 213.6 ms | 716.7 ms |
| All miss (0/5) | Text W8A16 | 24.1 ms | 292.5 ms | 89.8 ms | 116.0 ms | 525.3 ms |
| All history hit (4/1) | FP16 | 22.0 ms | 75.4 ms | 138.6 ms | 209.7 ms | 446.9 ms |
| All history hit (4/1) | Text W8A16 | 15.9 ms | 72.1 ms | 89.5 ms | 116.2 ms | 294.9 ms |

For the all-history-hit case, Text W8A16 reduces prefill by 35.4%, decode by
44.6%, and end-to-end latency by 34.0%. The all-miss end-to-end reduction is
26.7%. All 20 timed requests at both precisions produced
`[153716, 153743, 151675, 151961, 152205]`. This establishes consistency only
for the deterministic benchmark input and does not replace navigation task
evaluation. Cached and recomputed visual features differed by at most
`2.44e-4`.

## Prefill and decode device optimization

This path applies two established Pi0.5 principles: replay a fixed device Graph
and keep the stage-final projection inside the same device boundary. Prefill uses
owner-checked fast replay. Decode fuses the current 1,408-row grammar head into
the decoder Graph and uses checked replay to patch its weight, current position,
and KV addresses. It does not reuse tokens from an earlier request.

Run the serial five-token path without `--history-draft`:

```bash
python examples/lightnav_benchmark.py \
  --checkpoint /path/to/LightNav-0 \
  --lightnav-source /path/to/LightNav-0-source \
  --history-frames 12,24,36,48,64 \
  --repeats 5 \
  --constrained-head \
  --output benchmark_results/lightnav_0_r1_serial.json
```

R1 v1.2.0 medians for the all-history-cached scenario are:

| History | Prepare | Vision | Prefill | Decode 5 | Total | Previous total | Reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 19.1 ms | 71.5 ms | 147.1 ms | 209.9 ms | 448.0 ms | 460.1 ms | 2.6% |
| 24 | 23.3 ms | 73.4 ms | 216.6 ms | 211.2 ms | 524.1 ms | 535.4 ms | 2.1% |
| 36 | 25.1 ms | 75.6 ms | 292.9 ms | 215.1 ms | 609.5 ms | 610.9 ms | 0.2% |
| 48 | 26.3 ms | 70.0 ms | 289.5 ms | 212.5 ms | 597.4 ms | 618.4 ms | 3.4% |
| 64 | 27.1 ms | 72.6 ms | 305.9 ms | 213.0 ms | 618.3 ms | 635.6 ms | 2.7% |

Decode improves by 2.1%-3.3%. Prefill improves by 2.4%-2.8% for most lengths;
the 36-frame all-history-cached result changes by 0.1%. The timed requests
produce exactly the same token sequences as the previous implementation. More
history still produces more pooled visual tokens, so prefill continues to grow
with input length.

For consecutive navigation requests, pass `--history-draft`. The benchmark uses
the previous five tokens as a draft, appends its first four tokens to the current
prompt, and verifies all five positions in one prefill. A full match skips serial
decode; any mismatch falls back to the original five-token greedy path. The first
request remains serial and no MTP weights are required.

Run the complete benchmark with:

```bash
python examples/lightnav_benchmark.py \
  --checkpoint /path/to/LightNav-0 \
  --lightnav-source /path/to/LightNav-0-source \
  --history-frames 12,24,36,48,64 \
  --repeats 5 \
  --constrained-head \
  --history-draft \
  --output benchmark_results/lightnav_0_r1_history_draft.json
```

## History-draft verification

The previous output is only a draft. Adjacent navigation observations and
actions are often correlated, but the current request still verifies the draft
with its own visual input, prompt, and model execution. A mismatch falls back to
standard decoding, so draft quality affects latency rather than greedy output.

For prompt `P` and draft `[d1, d2, d3, d4, d5]`, next-token alignment is:

| Input position | Logits predict |
|---|---|
| Last token of `P` | `d1` |
| `d1` | `d2` |
| `d2` | `d3` |
| `d3` | `d4` |
| `d4` | `d5` |

The verification input is therefore `P + [d1, d2, d3, d4]`. One prefill keeps
the final five logits rows and compares their greedy tokens with the draft. `d5`
is not appended because its logits would predict an unused sixth token.

The first request runs standard prefill plus four single-token decode calls.
Later requests verify all five positions in one prefill. A full match returns the
draft with zero `decode_5_ms`; a mismatch clears the verification Graph, rewinds
the KV cache, and runs standard serial decoding. The final output becomes the
next request's draft. This is temporal speculative decoding with full target-model
verification and requires no MTP head.

### R1 v1.2.0 measurements

The test used deterministic 256×448 synthetic input, the 4,369-token action
allowlist, five measured requests per cache scenario, and excluded Rhino Graph
warmup. With all history tubelets cached, median latency was:

| History | Prepare | Vision | Prefill/verify | Decode | Total | Baseline total | Reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 23.8 ms | 72.2 ms | 150.7 ms | 0 ms | 247.1 ms | 458.4 ms | 46.1% |
| 24 | 19.3 ms | 70.6 ms | 221.4 ms | 0 ms | 311.4 ms | 526.6 ms | 40.9% |
| 36 | 23.4 ms | 74.1 ms | 292.2 ms | 0 ms | 390.0 ms | 605.8 ms | 35.6% |
| 48 | 28.4 ms | 71.3 ms | 300.9 ms | 0 ms | 402.2 ms | 615.5 ms | 34.7% |
| 64 | 29.8 ms | 75.2 ms | 314.7 ms | 0 ms | 420.1 ms | 631.7 ms | 33.5% |

All-miss optimized totals were 598.0/968.8/1311.2/1415.9/1584.0 ms, reductions
of 26.3%/18.2%/11.6%/13.8%/11.9%. All 50 measured requests accepted their drafts.
That acceptance rate comes from repeated deterministic input and represents the
optimization path and latency ceiling; real navigation requires separate
acceptance-rate and task-accuracy evaluation.

`all_miss_median` uses a new empty vision cache for every request.
`all_history_hit_median` preloads every selected tubelet that excludes the
current frame, so only the single tubelet containing the current frame runs the
ViT. Cached and fully recomputed pooler plus three DeepStack levels must match
within `1e-3` FP16 absolute tolerance. The report records tubelet hits/misses and
preparation, vision, prefill, five-token decode, and end-to-end latency. The
current SlowFast settings yield about 288/456/568/624/708 pooled visual tokens.
Runtime admission still enforces the Qwen3-VL 4B P4176 limit.

`draft_attempts` and `draft_accepts` report verification and full five-token
acceptance counts. With `--history-draft`, `prefill_ms` includes verification and
an accepted request reports zero `decode_5_ms`; compare `total_ms` for the actual
end-to-end gain.

## Execution notes

The processor patchifies only the current uncached tubelet. Tubelet-level
pooling reuses pooled history while an empty cache runs the full-window vision
path. Prefill remains complete because changed visual tokens and M-RoPE
positions invalidate the old KV prefix. Five-token decode is autoregressive
and uses the fused decoder and device argmax by default. History-draft mode
instead verifies all five positions in one prefill.

The constrained heads follow the five-position LightNav grammar. Their narrower
token domains can change token selection; select them to match the application's
decoding contract. History-draft gains depend on acceptance across consecutive requests;
this checkpoint has no multi-token prediction heads or weights.

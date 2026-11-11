# LightNav-0 runtime and cache

[简体中文](lightnav_0_r1.zh.md) | English

This extension targets public checkpoint `LightOriginsHQ/LightNav-0` at
Hugging Face revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64` and upstream
LightNav commit `3015508b70fb6e30bbd66701e94c9b916dcb8df8`. Every request
produces exactly five tokens: two pointing tokens and three RVQ action tokens.

RhinoForge reuses its Qwen3-VL 4B text, vision, and DeepStack paths. The full
156048-row vocabulary cannot use the fused column-partitioned head, whose output
dimension must be 128-aligned. With the constrained head enabled, its 4,369
candidates are padded to 4,480 rows and fused into the device decode Graph.
Admission remains restricted to the exact LightNav profile.

The vision cache uses upstream two-frame tubelet keys and defaults to 1,024
entries, following the upstream Thor long-session recommendation. Each miss runs the ViT
once and stores its pooler output plus three DeepStack levels on CPU. Post-ViT
spatial pooling keeps the unpooled values and also caches an exact segment result
by tubelet sequence and SlowFast factor. A changed sequence or factor therefore
recomputes pooling. Sliding windows use a full feature scatter because token
positions move between requests.

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
override it. Both scenarios run five times and report medians, while
every request generates five tokens. Deterministic synthetic 256×448 frames make
the report a latency and cache comparison rather than a navigation-accuracy test.
Each tier warms the full-window and current-tubelet Rhino Graphs outside the
measurement and retains one stable KV-cache address. Its Graphs are released
before the next tier.

When the application already uses upstream `VLN_EVAL_TRAJ_TOP1`, pass
`--constrained-head` to install its 4,369-token action-allowlist projection,
padded to 4,480 rows for column-parallel Linear and fused into single-token
decode. This opt-in changes the default full-vocabulary decoding domain and
requires separate navigation-quality evaluation.

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

Five fixed output tokens are still autoregressive: tokens two through five each
depend on the previous token. On R1 v1.2.0, retaining the second
position-specific decode Graph already exhausted the Graph arena, so four such
Graphs cannot stay resident. Clearing and rebuilding for every position would
add latency. The delivered path therefore keeps one shared checked-replay
decode Graph and no separate fixed-position option.

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
owner-checked fast replay. Decode fuses the 4,480-row candidate head into the
decoder Graph and uses checked replay to patch the current position and KV
addresses. It does not reuse tokens from an earlier request.

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

The processor patchifies only the current uncached tubelet. Exact segment
pooling reuses pooled history while an empty cache runs the full-window vision
path. Prefill remains complete because changed visual tokens and M-RoPE
positions invalidate the old KV prefix. Five-token decode is autoregressive
and uses the fused decoder and device argmax by default. History-draft mode
instead verifies all five positions in one prefill.

The constrained head follows the upstream action allowlist. Its narrower token
domain can change token selection; select it to match the application's decoding
contract. History-draft gains depend on acceptance across consecutive requests;
this checkpoint has no multi-token prediction heads or weights.

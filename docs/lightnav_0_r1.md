# LightNav-0 R1 port and benchmark

[简体中文](lightnav_0_r1.zh.md) | English

This extension targets public checkpoint `LightOriginsHQ/LightNav-0` at
Hugging Face revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64` and upstream
LightNav commit `3015508b70fb6e30bbd66701e94c9b916dcb8df8`. Every request
produces exactly five tokens: two pointing tokens and three RVQ action tokens.

RhinoForge reuses its Qwen3-VL 4B text, vision, and DeepStack paths. The
156048-row vocabulary cannot use the fused column-partitioned head, whose output
dimension must be 128-aligned. The decoder therefore returns hidden states and
an eager TP8 row-partitioned head computes logits. Admission remains restricted
to the exact LightNav profile.

The vision cache uses upstream two-frame tubelet keys. Each miss runs the ViT
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

Remove `--check-config` on R1. The values 12/24/36/48/64 are past-history
counts; one current frame is added, so the actual inputs contain
13/25/37/49/65 frames. Both scenarios run five times and report medians, while
every request generates five tokens. Deterministic synthetic 256×448 frames make
the report a latency and cache comparison rather than a navigation-accuracy test.
Each tier warms the full-window and current-tubelet RPU Graphs outside the
measurement and retains one stable KV-cache address. Its Graphs are released
before the next tier.

`all_miss_median` uses a new empty vision cache for every request.
`all_history_hit_median` preloads every selected tubelet that excludes the
current frame, so only the single tubelet containing the current frame runs the
ViT. Cached and fully recomputed pooler plus three DeepStack levels must match
within `1e-3` FP16 absolute tolerance. The report records tubelet hits/misses and
preparation, vision, prefill, five-token decode, and end-to-end latency. The
current SlowFast settings yield about 288/456/568/624/708 pooled visual tokens.
Runtime admission still enforces the Qwen3-VL 4B P4176 limit.

## Stage analysis and optimization

The processor already patchifies only the current miss tubelet, leaving cached
prepare latency at 21–28 ms. The 64-frame profile showed about 77 ms for the
current-tubelet ViT and 87 ms for repeated CPU pooling/concatenation. Exact
segment pooling caching cuts pooling to about 3 ms and keeps cached vision at
78–82 ms. Prefill remains complete because changing visual tokens and M-RoPE
positions invalidate a reused KV prefix. Five-token decode remains autoregressive
and already uses the fused RPU decoder and device argmax.

| History | Before (ms) | After (ms) | Further reduction |
|---:|---:|---:|---:|
| 12 | 532.2 | 500.1 | 6.0% |
| 24 | 616.5 | 572.4 | 7.1% |
| 36 | 718.7 | 647.3 | 9.9% |
| 48 | 738.1 | 652.2 | 11.6% |
| 64 | 762.7 | 668.9 | 12.3% |

## R1 results (2026-10-12)

Both scenarios exclude Graph construction and report the median of five runs.

| History + current | Input tokens | all miss (ms) | all history hit (ms) | tubelet hit/miss | E2E reduction |
|---:|---:|---:|---:|---:|---:|
| 12 + 1 | 414 | 886.8 | 500.1 | 6 / 1 | 43.6% |
| 24 + 1 | 630 | 1283.4 | 572.4 | 12 / 1 | 55.4% |
| 36 + 1 | 790 | 1639.9 | 647.3 | 17 / 1 | 60.5% |
| 48 + 1 | 864 | 1744.6 | 652.2 | 19 / 1 | 62.6% |
| 64 + 1 | 975 | 1922.8 | 668.9 | 22 / 1 | 65.2% |

Cached-history versus fully recomputed vision features differ by at most
`0.00048828125`, passing the `1e-3` FP16 tolerance. The complete raw report is
`benchmark_results/lightnav_0_r1_20261012.json`.

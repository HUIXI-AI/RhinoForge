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
spatial pooling runs from the unpooled cached values with the current SlowFast
segment factor. Sliding windows use a full feature scatter because token
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

## R1 results (2026-10-12)

Both scenarios exclude Graph construction and report the median of five runs.

| History + current | Input tokens | all miss (ms) | all history hit (ms) | tubelet hit/miss | E2E reduction |
|---:|---:|---:|---:|---:|---:|
| 12 + 1 | 414 | 897.6 | 532.2 | 6 / 1 | 40.7% |
| 24 + 1 | 630 | 1288.8 | 616.5 | 12 / 1 | 52.2% |
| 36 + 1 | 790 | 1643.7 | 718.7 | 17 / 1 | 56.3% |
| 48 + 1 | 864 | 1763.3 | 738.1 | 19 / 1 | 58.1% |
| 64 + 1 | 975 | 1934.2 | 762.7 | 22 / 1 | 60.6% |

Cached-history versus fully recomputed vision features differ by at most
`0.00048828125`, passing the `1e-3` FP16 tolerance. The complete raw report is
`benchmark_results/lightnav_0_r1_20261012.json`.

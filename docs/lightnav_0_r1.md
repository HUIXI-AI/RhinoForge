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

Remove `--check-config` on R1. The script fixes history lengths to
12/24/36/48/64 frames, performs six sliding-window warmups and five measured
runs, and generates five tokens. It uses deterministic synthetic 256×448 frames,
so the report measures inference and cache performance rather than navigation
accuracy. Six warmups cover the SlowFast burst pairing cycle at 48/64 frames so
the steady request has only the newest tubelet miss.
After each history tier, the script clears that tier's RPU Graphs before warming
the next shape; one stable KV-cache address is retained throughout each tier.

For each history length, an unmeasured request builds the RPU Graph, then `cold`
is measured with an empty vision cache. `cached_repeat` is the all-hit upper
bound and must reproduce the pooler and three DeepStack feature levels exactly;
`steady_median` summarizes the five
measured sliding requests. `sliding_fresh` recomputes the first measured window
with an empty cache; its vision features must match within `1e-3` FP16 absolute
tolerance. The report records tubelet
hits/misses and preparation, vision, prefill, five-token decode, and end-to-end
latency. The current SlowFast settings yield about
252/420/568/624/708 pooled visual tokens for the five history lengths. Runtime
admission still enforces the Qwen3-VL 4B P4176 limit on the complete request.

## R1 results (2026-10-10)

These results use deterministic synthetic frames, six warmups, and five measured
runs. `cold` has a warm RPU Graph and an empty vision cache; `steady` is the
sliding-window median.

| History frames | Input tokens | cold (ms) | all-hit (ms) | steady (ms) | steady hit/miss | cold reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 378 | 854.0 | 443.3 | 517.0 | 5 / 1 | 39.5% |
| 24 | 594 | 1262.5 | 546.2 | 607.1 | 11 / 1 | 51.9% |
| 36 | 790 | 1603.6 | 634.0 | 710.1 | 17 / 1 | 55.7% |
| 48 | 864 | 1782.2 | 654.1 | 733.4 | 19 / 1 | 58.8% |
| 64 | 975 | 1932.1 | 680.4 | 770.7 | 22 / 1 | 60.1% |

Same-window cached features are bit-identical. Sliding cached versus fresh
features differ by at most `0.00048828125`, no more than one FP16 rounding step.
The complete raw report is `benchmark_results/lightnav_0_r1_20261010.json`.

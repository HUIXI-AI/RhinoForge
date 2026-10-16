# LightNav-0 runtime and cache

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
Each tier warms the full-window and current-tubelet Rhino Graphs outside the
measurement and retains one stable KV-cache address. Its Graphs are released
before the next tier.

When the application already uses upstream `VLN_EVAL_TRAJ_TOP1`, pass
`--constrained-head` to install its 4,369-token action-allowlist projection,
padded to 4,480 rows for column-parallel Linear. This opt-in changes the default
full-vocabulary decoding domain and requires separate navigation-quality evaluation.

For consecutive navigation requests, pass `--history-draft`. The benchmark uses
the previous five tokens as a draft, appends its first four tokens to the current
prompt, and verifies all five positions in one prefill. A full match skips serial
decode; any mismatch falls back to the original five-token greedy path. The first
request remains serial and no MTP weights are required.

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

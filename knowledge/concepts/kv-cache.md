# KV cache

`RPUCache` is the Hugging Face-compatible cache used by fused causal attention.
It owns per-layer K/V tensors, capacity, and the logical sequence position. [Cache API](../../docs/api_reference.md#rpucache)

## Construction and ownership

- `RPUCache.from_model` accepts exactly one sizing mode: prompt plus
  `max_new_tokens`, or an explicit `max_seq_len`. [Cache implementation](../../python/rpu_backend/api/cache.py)
- Prompt-based sizing reserves enough physical rows for the model's admitted
  prefill padding while keeping the returned cache position logical. [Cache implementation](../../python/rpu_backend/api/cache.py)
- Board-free sizing checks may use `device="cpu"`; normal inference allocates on
  `rpu`. The cache accepts FP16 and validates its layout constraints before
  allocation. [Cache implementation](../../python/rpu_backend/api/cache.py)
- `position` is the source of truth for the cached length. Updates reject
  overflow; `reset()` clears data, while `reset_to_position()` rewinds the
  logical suffix for overwrite without clearing the preserved prefix. [Cache implementation](../../python/rpu_backend/api/cache.py)
- The Hugging Face `update()` method is a compatibility no-op because the fused
  attention path owns cache insertion. [Cache implementation](../../python/rpu_backend/api/cache.py)
- Multiple cache slots do not imply general batch support. `from_model`
  currently admits `batch_size > 1` only for the exact Qwen3 model type. [Cache implementation](../../python/rpu_backend/api/cache.py)

Create the cache after model configuration is final, size it for the complete
prefill-plus-decode envelope, and let the adapter advance or rewind its logical
position. [CausalLM API](../../docs/api_reference.md#causal-language-models)

Allocation has no fixed 8K ceiling: aligned storage must fit the allocator's
available memory, and allocation errors propagate without truncating capacity.
For Qwen3.5 MoE, install the model and create its cache with the same explicit
`max_seq_len`. Installation sizes rotary tables and the physical prefill
envelope, including padding, from that capacity. Changing it requires a fresh
model/cache. Keep allocation separate from execution admission: model context,
rotary bounds, SPM feasibility and operator register limits remain mandatory.
In particular, MoE SDPA execution K/V length is at most 1,048,560 rows including
padding; larger allocated storage does not authorize that execution. Other
adapters retain their model-specific prefill/position limits. [MoE adapter](../../python/rpu_backend/adapters/qwen3_5_moe/text.py)

## Wall-OSS KV head replication

The admitted Wall-OSS attention profile has 16 query heads and two real KV
heads. `RPU_WALL_OSS_ATTN_TP8` defaults on: installation repeats each KV head's
weights, bias and quantization scales four times, yielding eight effective KV
heads. Adjacent replicas remain associated with the same real head; each core
handles two query heads and its matching KV head. Q and output projections
retain their logical head ordering.

Create caches from the installed model's effective topology. An old two-head
cache cannot become an eight-head cache by changing metadata. The choice is
cold and applies to both language and action owners. See the
[language loader](../../python/rpu_backend/adapters/wall_oss/llm.py) and
[action loader](../../python/rpu_backend/adapters/wall_oss/action.py).

## Prefix replacement

HALO action-prefix replacement validates every layer and prepares the complete
new K/V bank before publishing it. A failed validation or upload preserves the
previous bank, position, and prefix epoch. A successful replacement advances
the epoch used by the action Graph signature. [HALO runner](../../python/rpu_backend/adapters/halo/runtime.py)

## Sources

- [API reference: `RPUCache`](../../docs/api_reference.md#rpucache)
- [Public cache implementation](../../python/rpu_backend/api/cache.py)
- [Runtime decoder cache checks](../../python/rpu_backend/runtime/decoder.py)

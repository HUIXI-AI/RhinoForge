# KV cache

`RPUCache` is the Hugging Face-compatible cache used by fused causal attention.
It owns per-layer K/V tensors, capacity, and the logical sequence position.
[Cache API](../../docs/api_reference.md#rpucache)

## Construction and ownership

- `RPUCache.from_model` accepts exactly one sizing mode: prompt plus
  `max_new_tokens`, or an explicit `max_seq_len`.
  [Cache implementation](../../python/rpu_backend/api/cache.py)
- Prompt-based sizing reserves enough physical rows for the model's admitted
  prefill padding while keeping the returned cache position logical.
  [Cache implementation](../../python/rpu_backend/api/cache.py)
- Board-free sizing checks may use `device="cpu"`; normal inference allocates on
  `rpu`. The cache accepts FP16 and validates its layout constraints before
  allocation. [Cache implementation](../../python/rpu_backend/api/cache.py)
- `position` is the source of truth for the cached length. Updates reject
  overflow; `reset()` clears data, while `reset_to_position()` rewinds the
  logical suffix for overwrite without clearing the preserved prefix.
  [Cache implementation](../../python/rpu_backend/api/cache.py)
- The Hugging Face `update()` method is a compatibility no-op because the fused
  attention path owns cache insertion.
  [Cache implementation](../../python/rpu_backend/api/cache.py)
- Multiple cache slots do not imply general batch support. `from_model`
  currently admits `batch_size > 1` only for the exact Qwen3 model type.
  [Cache implementation](../../python/rpu_backend/api/cache.py)

Create the cache after model configuration is final, size it for the complete
prefill-plus-decode envelope, and let the adapter advance or rewind its logical
position. [CausalLM API](../../docs/api_reference.md#causal-language-models)

## Sources

- [API reference: `RPUCache`](../../docs/api_reference.md#rpucache)
- [Public cache implementation](../../python/rpu_backend/api/cache.py)
- [Runtime decoder cache checks](../../python/rpu_backend/runtime/decoder.py)

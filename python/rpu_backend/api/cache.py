"""RPU KV cache and model-aware cache sizing.

``RPUCache.from_model`` accepts either prompt-plus-generation sizing or an
explicit maximum length. ``device='cpu'`` permits board-free sizing checks;
normal inference uses the default ``device='rpu'``.
"""

import os
from numbers import Integral

import torch

# Import HuggingFace Cache base class for compatibility
try:
    from transformers.cache_utils import Cache as HFCache
except ImportError:
    # Fallback for older transformers versions
    HFCache = object

from rpu_backend.runtime.weights import NUM_CORES


# Distinguish an omitted ``max_new_tokens`` from explicit ``None``.
_REQUIRED = object()
_PREFILL_PADDING_BUDGET_ENV = "RPU_CAUSAL_PREFILL_PADDING_BUDGET"
_PREFILL_PADDING_BUDGET_DEFAULT = 64


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(
            f"RPUCache {name} must be a positive integer, got {value!r}"
        )
    return int(value)


def _prefill_padding_room(model, prefill_len: int) -> int:
    """Physical rows the installed prefill policy may use temporarily."""
    stage = getattr(model, "_rpu_execution", {}).get("prefill", {})
    padding_rows = stage.get("padding_rows", "auto")
    if isinstance(padding_rows, int) and not isinstance(padding_rows, bool):
        return padding_rows
    alignment = getattr(model, "_rpu_prefill_execution_alignment", 1)
    if (
        isinstance(alignment, bool)
        or not isinstance(alignment, Integral)
        or alignment < 1
    ):
        raise ValueError(
            "RPUCache model._rpu_prefill_execution_alignment must be a "
            f"positive integer, got {alignment!r}"
        )
    mandatory_rows = (-int(prefill_len)) % int(alignment)
    raw_budget = stage.get(
        "padding_budget",
        os.environ.get(
            _PREFILL_PADDING_BUDGET_ENV,
            _PREFILL_PADDING_BUDGET_DEFAULT,
        ),
    )
    try:
        budget = int(raw_budget)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_PREFILL_PADDING_BUDGET_ENV} must be an integer"
        ) from exc
    if budget < 0:
        raise ValueError(
            f"{_PREFILL_PADDING_BUDGET_ENV} must be non-negative"
        )
    return mandatory_rows + budget


class RPUCache(HFCache):
    """
    KV Cache for RPU fused attention in 7D swizzle format.

    K-cache shape: [batch, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
    V-cache shape: [batch, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk]

    This class inherits from transformers.cache_utils.Cache for compatibility
    with HuggingFace models (if available).
    """

    # Mark this as a valid cache type for HuggingFace
    is_sliding_window = False

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "rpu",
        dtype: torch.dtype = torch.float16,
        attn_tp: int = NUM_CORES,
    ):
        num_layers = _positive_int("num_layers", num_layers)
        batch_size = _positive_int("batch_size", batch_size)
        max_seq_len = _positive_int("max_seq_len", max_seq_len)
        num_kv_heads = _positive_int("num_kv_heads", num_kv_heads)
        head_dim = _positive_int("head_dim", head_dim)
        attn_tp = _positive_int("attn_tp", attn_tp)
        # batch_size > 1 only allocates independent KV slots. The Qwen3 adapter
        # owns the explicit capability gate that allows those slots to be used
        # by batched prefill/decode; other architectures remain batch==1.
        if dtype is not torch.float16:
            raise ValueError(
                f"RPUCache dtype must be torch.float16, got {dtype}"
            )
        try:
            resolved_device = torch.device(device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"RPUCache invalid device {device!r}") from exc
        if (
            resolved_device.type not in {"cpu", "rpu"}
            or resolved_device.index not in {None, 0}
        ):
            raise ValueError(
                "RPUCache device must be cpu, cpu:0, rpu, or rpu:0; "
                f"got {device!r}"
            )
        if head_dim % 16 != 0:
            raise ValueError(
                f"RPUCache head_dim must be divisible by 16, got {head_dim}"
            )
        if attn_tp > NUM_CORES:
            raise ValueError(
                f"RPUCache attn_tp must be <= {NUM_CORES}, got {attn_tp}"
            )
        if (NUM_CORES * num_kv_heads) % attn_tp != 0:
            raise ValueError(
                "RPUCache layout requires "
                f"({NUM_CORES} * {num_kv_heads}) % "
                f"attn_tp={attn_tp} == 0"
            )

        # Skip parent __init__ - we manage our own state
        # Note: HFCache.__init__ requires layers/layer_class_to_replicate which we don't use
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Chunk sizes for swizzle format
        self.sKeyChunk = 16
        self.sValChunk = 16
        self.headDimChunk = 16
        # nKVHeadChunk × nKVHeadVx must equal NUM_CORES × nhkv / attn_tp
        # to match the cache-insert operator's DDR stride contract.
        effective_kv_slots = NUM_CORES * num_kv_heads // attn_tp
        self.nKVHeadChunk = min(NUM_CORES, effective_kv_slots)

        # Derived dimensions
        self.nKVHeadVx = _ceil_div(effective_kv_slots, self.nKVHeadChunk)
        self.headDimVx = head_dim // self.headDimChunk

        self.sKeyVx = _ceil_div(max_seq_len, self.sKeyChunk)
        self.sValVx = _ceil_div(max_seq_len, self.sValChunk)

        # Allocate K/V caches for each layer
        # K-cache: [batch, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
        # V-cache: [batch, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk]
        self.k_caches = []
        self.v_caches = []

        for _ in range(num_layers):
            k_cache = torch.zeros(
                (batch_size, self.sKeyVx, self.nKVHeadVx, self.headDimVx,
                 self.nKVHeadChunk, self.sKeyChunk, self.headDimChunk),
                dtype=dtype, device=resolved_device
            )
            v_cache = torch.zeros(
                (batch_size, self.sValVx, self.nKVHeadVx, self.headDimVx,
                 self.nKVHeadChunk, self.headDimChunk, self.sValChunk),
                dtype=dtype, device=resolved_device
            )
            self.k_caches.append(k_cache)
            self.v_caches.append(v_cache)

        # Track current position (sequence length written so far)
        self.position = 0

        # Track per-layer seen tokens for transformers compatibility
        self._seen_tokens = 0

        # For HuggingFace Cache compatibility - provide empty layers list
        # The actual layer data is in k_caches and v_caches
        self.layers = []

    def get_cache(self, layer_idx: int):
        """Get K/V cache for a specific layer."""
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def update_position(self, seq_len: int):
        """Update position after processing seq_len tokens."""
        if isinstance(seq_len, bool) or not isinstance(seq_len, Integral):
            raise ValueError(
                f"RPUCache seq_len must be a non-negative integer, got {seq_len!r}"
            )
        seq_len = int(seq_len)
        if seq_len < 0 or self.position + seq_len > self.max_seq_len:
            raise ValueError(
                "RPUCache position overflow: "
                f"position={self.position}, seq_len={seq_len}, "
                f"max_seq_len={self.max_seq_len}"
            )
        self.position += seq_len
        self._seen_tokens = self.position

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length (total tokens cached)."""
        return self.position

    def get_max_length(self) -> int:
        """Get the maximum cache length."""
        return self.max_seq_len

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Get the usable length considering the new sequence."""
        return self.position

    def reset(self):
        """Reset cache to empty state."""
        self.position = 0
        self._seen_tokens = 0
        for k_cache, v_cache in zip(self.k_caches, self.v_caches):
            k_cache.zero_()
            v_cache.zero_()

    def reset_to_position(self, pos: int):
        """Reset cache position without clearing data.

        Used for Pi0.5 shared cache: prefix KV at [0, pos) is preserved,
        suffix region [pos, ...) will be overwritten by the next forward pass.
        No zero-fill needed because insert kernels overwrite before SDPA reads.
        """
        if isinstance(pos, bool) or not isinstance(pos, Integral):
            raise ValueError(
                f"RPUCache position must be an integer in range, got {pos!r}"
            )
        pos = int(pos)
        if not 0 <= pos <= self.max_seq_len:
            raise ValueError(
                f"RPUCache position must be in [0, {self.max_seq_len}], got {pos}"
            )
        self.position = pos
        self._seen_tokens = pos

    # =========================================================================
    # Transformers Cache Interface Methods
    # =========================================================================

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        """
        Get mask sizes for attention mask creation.
        Required by transformers masking_utils.py.

        Returns:
            Tuple of (kv_length, kv_offset)
            - kv_length: Total length of KV cache
            - kv_offset: Offset for the current position
        """
        kv_length = self.position
        kv_offset = 0
        return kv_length, kv_offset

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        Update cache with new key/value states.
        This is a no-op for RPUCache since fused attention handles cache update internally.

        For compatibility, returns the input key/value states unchanged.
        """
        # Note: In fused attention mode, the cache update is handled by
        # rpu_launch_insert_kcache_spm_unified and rpu_launch_insert_vcache_spm_unified
        # within torch.ops.rpu.fused_qkv_attention.
        # This method is only for fallback compatibility.
        return key_states, value_states

    def __len__(self) -> int:
        """Return number of layers in cache."""
        return self.num_layers

    def __getitem__(self, layer_idx: int):
        """Get cache for a specific layer (for compatibility)."""
        return self.get_cache(layer_idx)

    def __iter__(self):
        """Iterate over layer caches."""
        for i in range(self.num_layers):
            yield self.get_cache(i)

    @property
    def seen_tokens(self) -> int:
        """Return total number of tokens seen (for transformers compatibility)."""
        return self._seen_tokens

    def __repr__(self) -> str:
        """Custom repr for RPUCache."""
        return (f"RPUCache(num_layers={self.num_layers}, batch_size={self.batch_size}, "
                f"max_seq_len={self.max_seq_len}, position={self.position})")

    @classmethod
    def from_model(
        cls,
        model,
        *,
        input_ids=None,
        max_new_tokens=_REQUIRED,
        max_seq_len: int | None = None,
        batch_size: int = 1,
        device: str = "rpu",  # "cpu" supports board-free sizing checks
    ) -> "RPUCache":
        """Build a cache sized from a model and one of two input forms.

        Two valid call forms (strict XOR):

        1. ``RPUCache.from_model(model, input_ids=ids, max_new_tokens=N)``
           sized to cover both ``ids.shape[1] + N`` logical rows and the
           temporary physical rows allowed by the model's prefill-padding
           policy. Padded rows are rewound before decode and do not add to N.
        2. ``RPUCache.from_model(model, max_seq_len=L)``
           sized to L explicitly (use when ``input_ids`` is not yet known,
           e.g., server-side preallocation).

        Raises ``ValueError`` on:
        - both / neither slot specified (XOR violation),
        - ``input_ids=`` without ``max_new_tokens=``,
        - ``max_new_tokens < 0``,
        - computed ``max_seq_len <= 0``.

        ``device='cpu'`` is for board-free factory sizing verification.
        Production callers
        omit it (default ``'rpu'``).

        ``batch_size > 1`` is available only when
        ``model.config.model_type == 'qwen3'``. Every row must contain the same
        number of real tokens: padding must not be used to manufacture a common
        prompt length. On forward, an omitted attention mask asserts that all
        rows are real tokens; a supplied mask must contain only ones.
        """
        if (input_ids is None) == (max_seq_len is None):
            raise ValueError(
                "RPUCache.from_model: pass exactly one of `input_ids=` "
                "(with `max_new_tokens=`) or `max_seq_len=`. Got "
                f"input_ids={'set' if input_ids is not None else 'None'}, "
                f"max_seq_len={max_seq_len!r}."
            )
        config = model.config
        batch_size = _positive_int("batch_size", batch_size)
        if batch_size > 1 and getattr(config, "model_type", None) != "qwen3":
            raise NotImplementedError(
                "RPUCache.from_model batch_size > 1 is supported only for "
                "model.config.model_type == 'qwen3'"
            )
        head_dim = getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        if max_seq_len is None:
            if max_new_tokens is _REQUIRED:
                raise ValueError(
                    "RPUCache.from_model: `max_new_tokens=` is required "
                    "when `input_ids=` is passed; otherwise the cache is "
                    "sized to the prefill prompt only and any "
                    "`model.generate(...)` call will overflow."
                )
            if (
                isinstance(max_new_tokens, bool)
                or not isinstance(max_new_tokens, Integral)
                or max_new_tokens < 0
            ):
                raise ValueError(
                    f"RPUCache.from_model: max_new_tokens="
                    f"{max_new_tokens!r} must be a non-negative integer."
                )
            shape = getattr(input_ids, "shape", None)
            if shape is None or len(shape) != 2 or int(shape[0]) != batch_size:
                raise ValueError(
                    "RPUCache.from_model: input_ids must have shape "
                    f"[{batch_size}, seq_len], got {shape}"
                )
            prefill_len = int(shape[1])
            # Physical padding is temporary: forward slices back to the
            # logical rows and rewinds cache.position before decode. Reserve
            # the larger of those two independent needs, rather than adding
            # them (which would over-allocate) or letting decode capacity
            # silently truncate a configured joint padding/chunk search.
            padding_room = _prefill_padding_room(model, prefill_len)
            max_seq_len = prefill_len + max(
                int(max_new_tokens), padding_room
            )
        else:
            if isinstance(max_seq_len, bool) or not isinstance(
                max_seq_len, Integral
            ):
                raise ValueError(
                    "RPUCache.from_model: max_seq_len must be a positive "
                    f"integer, got {max_seq_len!r}."
                )
            max_seq_len = int(max_seq_len)
        if max_seq_len <= 0:
            raise ValueError(
                f"RPUCache.from_model: computed max_seq_len="
                f"{max_seq_len} must be > 0."
            )
        return cls(
            num_layers=config.num_hidden_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            device=device,
            dtype=torch.float16,
        )

    def to_dynamic_cache(self, device="cpu"):
        """
        Convert RPUCache (7D swizzled) to HuggingFace DynamicCache format.

        Unswizzles K/V cache tensors from RPU 7D format back to standard
        [batch, num_kv_heads, seq_len, head_dim] and wraps in DynamicCache.

        Args:
            device: Target device for the output cache (default "cpu")

        Returns:
            DynamicCache with K/V tensors in standard layout
        """
        from transformers.cache_utils import DynamicCache

        seq_len = self.position
        if seq_len == 0:
            return DynamicCache()

        dynamic_cache = DynamicCache()

        for layer_idx in range(self.num_layers):
            k_7d = self.k_caches[layer_idx]  # [B, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
            v_7d = self.v_caches[layer_idx]  # [B, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk]

            B = k_7d.shape[0]

            # Move to CPU for reshape
            k_cpu = k_7d.to(device)
            v_cpu = v_7d.to(device)

            # K unswizzle: [B, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
            #   → [B, nKVHeadChunk, nKVHeadVx, sKeyVx, sKeyChunk, headDimVx, headDimChunk]
            #   → [B, effective_kv_slots, max_seq, head_dim]
            effective_kv_slots = self.nKVHeadChunk * self.nKVHeadVx
            k_out = k_cpu.permute(0, 4, 2, 1, 5, 3, 6).contiguous()
            k_out = k_out.reshape(B, effective_kv_slots, self.sKeyVx * self.sKeyChunk, self.head_dim)
            k_out = k_out[:, :self.num_kv_heads, :seq_len, :]  # trim to actual heads and seq_len

            # V unswizzle: [B, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk]
            #   → [B, nKVHeadChunk, nKVHeadVx, sValVx, sValChunk, headDimVx, headDimChunk]
            #   → [B, effective_kv_slots, max_seq, head_dim]
            v_out = v_cpu.permute(0, 4, 2, 1, 6, 3, 5).contiguous()
            v_out = v_out.reshape(B, effective_kv_slots, self.sValVx * self.sValChunk, self.head_dim)
            v_out = v_out[:, :self.num_kv_heads, :seq_len, :]

            # DynamicCache.update expects [batch, num_heads, seq_len, head_dim]
            dynamic_cache.update(k_out, v_out, layer_idx)

        return dynamic_cache

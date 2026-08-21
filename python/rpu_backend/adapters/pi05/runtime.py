"""Pi0.5 handle ownership, denoise execution, and lifecycle helpers."""
from __future__ import annotations
import os
import threading
import weakref
from typing import Any

import torch

from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.log import _LOG
from rpu_backend.api.errors import RPUBackendError


# Per-instance handle map.
_PI05_HANDLES: "weakref.WeakKeyDictionary[Any, int]" = weakref.WeakKeyDictionary()
_PI05_PREPARED_GRAPH_PROFILES: "weakref.WeakKeyDictionary[Any, dict]" = (
    weakref.WeakKeyDictionary()
)


def _denoise_graph_enabled() -> bool:
    return rpu_env_bool("RPU_PI05_DENOISE_GRAPH", default=True)


def _euler_fp16_enabled() -> bool:
    """When set, round the Euler accumulator to fp16 each step.

    The default uses a CPU fp32 accumulator. The setting is read per call so
    callers can compare both precision modes in one process.
    """
    return rpu_env_bool("RPU_PI05_EULER_FP16")


def _denoise_unroll_enabled() -> bool:
    """In-graph N-step denoise unroll (1 BUILD + 1 REPLAY, on-device fp16 Euler)
    via pi05_denoise_loop_forward. Default off; enable explicitly."""
    return rpu_env_bool("RPU_PI05_DENOISE_UNROLL")


def _prefix_pad16_enabled() -> bool:
    """Pad the prefix up to a multiple of 16 (see _pad_prefix_to_16). Default on;
    it is a no-op whenever the prefix is already aligned, which every shipped
    profile is (3*256 + a 32-token prompt)."""
    return rpu_env_bool("RPU_PI05_PREFIX_PAD16", default=True)


def _kvinsert_pad16_enabled() -> bool:
    """Mirror of the C++ kvinsert_pad16_enabled() (rpu_pi05_denoise_step_model.cpp):
    enabled iff RPU_PI05_KVINSERT_PAD16's first char is in {1,t,T}. This flag changes
    the denoise graph's k/v buffer size + KV-insert length, so it MUST be part of the
    GraphSignature key — otherwise a same-process A/B (toggling the env between
    forwards) replays the stale graph (pad graph after disable, or vice-versa)."""
    return rpu_env_bool(
        "RPU_PI05_KVINSERT_PAD16",
        cpp_mirror="src/fused/rpu_pi05_denoise_step_model.cpp",
    )


def _pi05_graph_runtime_profile(self, num_steps: int) -> dict:
    """Return the runtime knobs that determine Pi0.5 graph admission."""
    from rpu_backend.adapters.pi05.adarms import _adarms_graph_enabled
    from rpu_backend.adapters.pi05.gemma import _gemma_graph_enabled
    from rpu_backend.adapters.siglip import _siglip_graph_enabled

    vlm = self.paligemma_with_expert.paligemma.model.language_model
    execution = getattr(self, "_rpu_execution", {})
    return {
        "num_steps": int(num_steps),
        "vlm_chunk_size": int(getattr(vlm, "_rpu_chunk_size", 0)),
        "siglip_batch_n": int(getattr(self, "_rpu_siglip_batch_n", 1)),
        "siglip_graph": bool(_siglip_graph_enabled()),
        "gemma_graph": bool(_gemma_graph_enabled()),
        "adarms_graph": bool(_adarms_graph_enabled()),
        "fused_denoise": getattr(
            self, "_rpu_fused_denoise_handle", None
        ) is not None,
        "denoise_graph": bool(_denoise_graph_enabled()),
        "denoise_unroll": bool(_denoise_unroll_enabled()),
        "kvinsert_pad16": bool(_kvinsert_pad16_enabled()),
        "execution": tuple(
            (stage, tuple(sorted(fields.items())))
            for stage, fields in sorted(execution.items())
        ),
    }


def _validate_prepared_graph_profile(self, num_steps: int) -> None:
    """Reject runtime-profile drift before any frozen GraphCache is entered."""
    prepared = _PI05_PREPARED_GRAPH_PROFILES.get(self)
    if prepared is None:
        return
    expected = prepared["runtime"]
    actual = _pi05_graph_runtime_profile(self, num_steps)
    changed = {
        key: (expected[key], actual[key])
        for key in expected
        if expected[key] != actual[key]
    }
    if changed:
        raise RuntimeError(
            "Pi0.5 request/runtime differs from the prepared READY profile "
            f"(expected, actual): {changed}. Online graph BUILD is disabled."
        )


# Allowed ``sample_actions`` kwargs; unknown values fail instead of being dropped.
_RPU_ALLOWED_SAMPLE_ACTIONS_KWARGS = {"num_steps"}


# ===========================================================================
# Prefix 4D attention-mask LRU-1 cache and helpers.
# ===========================================================================

# Atomic tuple snapshot — readers grab `snap = _PREFIX_MASK_SNAPSHOT`
# locally before comparing key. Writers serialize via the lock.
_PREFIX_MASK_SNAPSHOT: "tuple | None" = None  # (key, mask4d, pos_ids) or None
_PREFIX_MASK_LOCK = threading.Lock()
_PREFIX_MASK_CACHE_HITS: int = 0
_PREFIX_MASK_CACHE_MISSES: int = 0

# Once-per-device sentinel to avoid repeated warnings.
_HASH_MASK_NONCPU_WARNED: set = set()


def _prefix_mask_cache_enabled() -> bool:
    """Read the live per-call cache switch instead of freezing it at import."""
    return rpu_env_bool("RPU_PI05_PREFIX_MASK_CACHE", default=True)


def _hash_mask(t: "torch.Tensor") -> bytes:
    """Hash a mask tensor to bytes for cache-key construction.

    The production path keeps prefix masks on CPU. If a caller supplies an RPU
    tensor, this `.cpu()` synchronizes the device. A once-per-device warning
    makes this visible.
    """
    if t.device.type != 'cpu' and str(t.device) not in _HASH_MASK_NONCPU_WARNED:
        import warnings
        warnings.warn(
            f"_hash_mask: tensor on {t.device}, .cpu() sync per cache lookup",
            RuntimeWarning, stacklevel=2)
        _HASH_MASK_NONCPU_WARNED.add(str(t.device))
    return t.contiguous().cpu().numpy().tobytes()


def _make_cache_key(prefix_pad_masks, prefix_att_masks):
    """Eight-field key (four fields for each of two masks).

    Includes shape/dtype/device + content bytes for BOTH masks to avoid
    collisions across episodes / dtype changes / device migrations.
    """
    return (
        # pad
        tuple(prefix_pad_masks.shape),
        prefix_pad_masks.dtype,
        str(prefix_pad_masks.device),
        _hash_mask(prefix_pad_masks),
        # att
        tuple(prefix_att_masks.shape),
        prefix_att_masks.dtype,
        str(prefix_att_masks.device),
        _hash_mask(prefix_att_masks),
    )


def _cached_build_prefix_mask_4d(prefix_pad_masks, prefix_att_masks, prepare_4d_fn):
    """Build (mask_4d, position_ids) with LRU-1 cache + fast-path guard.

    Contract:
    - The helper accepts prefix_att_masks and fast-paths only when attention is
      truly all-zero. Otherwise it calls upstream make_att_2d_masks.
    - The tuple snapshot is atomic under the GIL; writes use a lock.
    - The environment opt-out is evaluated per call.
    """
    global _PREFIX_MASK_CACHE_HITS, _PREFIX_MASK_CACHE_MISSES, _PREFIX_MASK_SNAPSHOT

    if not _prefix_mask_cache_enabled():
        # Honest passthrough — bit-identical to lerobot upstream behavior.
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        return (prepare_4d_fn(att_2d),
                torch.cumsum(prefix_pad_masks, dim=1) - 1)

    key = _make_cache_key(prefix_pad_masks, prefix_att_masks)
    snap = _PREFIX_MASK_SNAPSHOT  # GIL-atomic read of tuple/None
    if snap is not None and snap[0] == key:
        _PREFIX_MASK_CACHE_HITS += 1
        return snap[1], snap[2]

    _PREFIX_MASK_CACHE_MISSES += 1
    # Fast-path guard: att must be truly all-zero for outer-product shortcut.
    if torch.count_nonzero(prefix_att_masks) == 0:
        att_2d = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
    else:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    mask4d = prepare_4d_fn(att_2d)
    pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    with _PREFIX_MASK_LOCK:
        _PREFIX_MASK_SNAPSHOT = (key, mask4d, pos_ids)  # atomic re-assign

    return mask4d, pos_ids


def _reset_prefix_mask_cache():
    """Reset cache state for isolated callers and test fixtures."""
    global _PREFIX_MASK_SNAPSHOT, _PREFIX_MASK_CACHE_HITS, _PREFIX_MASK_CACHE_MISSES
    with _PREFIX_MASK_LOCK:
        _PREFIX_MASK_SNAPSHOT = None
        _PREFIX_MASK_CACHE_HITS = 0
        _PREFIX_MASK_CACHE_MISSES = 0


# =============================================================================
# Debug probe: dump named tensors when RPU_PI05_PROBE_DIR is set. It is a no-op
# when unset and fails silently on errors
# to avoid perturbing exec.
# =============================================================================
def _pi05_probe_save(name: str, tensor: Any) -> None:
    probe_dir = os.environ.get("RPU_PI05_PROBE_DIR")
    if not probe_dir or tensor is None:
        return
    try:
        os.makedirs(probe_dir, exist_ok=True)
        t = tensor
        if hasattr(t, "device") and t.device.type == "rpu":
            t = t.cpu()
        if hasattr(t, "dtype") and t.dtype in (torch.float16, torch.bfloat16):
            t = t.float()
        torch.save(t, os.path.join(probe_dir, f"{name}.pt"))
    except Exception as exc:  # probes must never break the forward
        _LOG.warning("Pi05 probe dump %s failed: %s", name, exc)


def _load_pi05_probe_tensor(
    path: str,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    """Load one local debug override without enabling arbitrary pickle."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} override must contain a tensor, got "
            f"{type(value).__name__}"
        )
    shape = tuple(value.shape)
    if shape != expected_shape:
        raise ValueError(
            f"{name} override must have shape {expected_shape}, got {shape}"
        )
    if value.is_complex():
        raise TypeError(f"{name} override must be real-valued")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} override must contain only finite values")
    return value


# =============================================================================
# Handle destroyer.
# =============================================================================
def _pi05_destroy_handle_and_unmap(handle: int, model_ref: Any) -> None:
    """Remove the weak-map entry and destroy the native handle."""
    _PI05_HANDLES.pop(model_ref, None)
    _PI05_PREPARED_GRAPH_PROFILES.pop(model_ref, None)
    try:
        torch.ops.rpu.pi05_destroy(handle)
    except Exception:
        pass   # Swallow — interpreter shutdown may have destroyed the op.


# =============================================================================
# Top-level helpers (lifted from install_sample_actions_patch nested closures).
# These had no meaningful captures — they only call other module-scope helpers.
# =============================================================================

def _resolve_rpu_device(self):
    """Resolve the RPU device from VLM layer 0."""
    cached = getattr(self, "_rpu_runtime_device", None)
    if cached is not None:
        return cached

    vlm = self.paligemma_with_expert.paligemma.model.language_model
    # Read device from the VLM's first decoder layer (not
    # next(vlm.parameters()) which may return embed_tokens — that stays
    # on CPU by design.
    rpu_device = next(vlm.layers[0].parameters()).device
    if rpu_device.type != 'rpu':
        raise RPUBackendError(
            f"Pi05 RPU path: vlm is not on RPU (device={rpu_device}). "
            "adapter.to_rpu() did not install the VLM on RPU. Reload the policy.")
    return rpu_device


def _sample_noise(self, bsize, noise):
    """Sample noise or validate the caller-provided override."""
    actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
    if noise is None:
        seed = int(getattr(self, '_rpu_noise_seed', 0))
        gen = torch.Generator(device='cpu').manual_seed(seed)
        x_t = torch.randn(actions_shape, generator=gen, dtype=torch.float32, device='cpu')
    else:
        # Caller-provided noise: move to CPU fp32 for the Python loop.
        x_t = noise.to(device='cpu', dtype=torch.float32)

    # Probe: bisection harness may override noise via env var to force
    # identical inputs across reference and library runs.
    _override = os.environ.get("RPU_PI05_LOAD_NOISE")
    if _override and os.path.exists(_override):
        x_t = _load_pi05_probe_tensor(
            _override,
            name="RPU_PI05_LOAD_NOISE",
            expected_shape=actions_shape,
        ).to(dtype=torch.float32)
    _pi05_probe_save("x_t_init", x_t)
    return x_t


def _prepare_prefill_siglip_inputs(
    self, images, img_masks, rpu_device
):
    """Upload SigLIP inputs without splitting a marked multi-camera slab.

    The batch-preprocessing path returns CPU dim-0 views of one camera slab.
    Keep those views intact until `_siglip_forward_images`, which performs one
    packed upload. Dynamic/reference inputs retain the established per-image
    upload. Masks stay on CPU: they only feed prefix-mask construction, and the
    fused Gemma path normalizes its attention mask on CPU.
    """
    if images is None:
        return images, img_masks

    batch_n = getattr(self, "_rpu_siglip_batch_n", 1)
    from rpu_backend.adapters.siglip import _shared_siglip_image_slab

    shared_batch = (
        batch_n > 1
        and len(images) == batch_n
        and _shared_siglip_image_slab(images) is not None
    )
    if not shared_batch:
        images = [image.to(rpu_device) for image in images]
    return images, img_masks



def _pad_prefix_rows(
    prefix_embs, prefix_pad_masks, prefix_att_masks, padding_rows: int
):
    """Append an exact number of masked, zero-valued physical prefix rows."""
    padding_rows = int(padding_rows)
    if padding_rows < 0:
        raise ValueError(f"padding_rows must be non-negative, got {padding_rows}")
    if padding_rows == 0:
        return prefix_embs, prefix_pad_masks, prefix_att_masks
    b = prefix_pad_masks.shape[0]
    embs = torch.cat([
        prefix_embs,
        prefix_embs.new_zeros(b, padding_rows, prefix_embs.shape[2]),
    ], dim=1)
    pads = torch.cat([
        prefix_pad_masks,
        prefix_pad_masks.new_zeros(b, padding_rows),
    ], dim=1)
    atts = torch.cat([
        prefix_att_masks,
        prefix_att_masks.new_zeros(b, padding_rows),
    ], dim=1)
    return embs, pads, atts


def _pad_prefix_to_16(prefix_embs, prefix_pad_masks, prefix_att_masks):
    """Round the prefix up to a multiple of 16 with TRAILING pad rows.

    The denoise KV-insert path requires a 16-aligned insert position, which is
    the physical prefix length. Padding is semantics-preserving because RoPE
    uses logical positions, KV uses physical rows, and the added rows are
    masked out of both prefix and suffix attention.
    """
    n = int(prefix_pad_masks.shape[1])
    pad = (-n) % 16
    return _pad_prefix_rows(
        prefix_embs, prefix_pad_masks, prefix_att_masks, pad
    )


def _plan_pi05_prefix_execution(self, logical_len: int) -> tuple[int, int]:
    """Jointly choose Pi0.5's physical prefix length and Gemma chunk size.

    The shared planner owns SPM/kernel feasibility. Pi0.5 uses a 16-aligned
    physical prefix as the tie-break among equal-launch plans. Padding remains
    trailing and masked, and the RoPE
    index/cache-row split makes it semantics-preserving.
    """
    vlm = self.paligemma_with_expert.paligemma.model.language_model
    handle = vlm._rpu_vlm_decoder_handle
    requested_chunk = int(getattr(vlm, "_rpu_chunk_size", 0))
    prefill = getattr(self, "_rpu_execution", {}).get("prefill", {})
    has_padding_policy = bool(requested_chunk) or any(
        field in prefill for field in ("padding_rows", "padding_budget")
    )

    def resolve(execution_len: int) -> int:
        return int(torch.ops.rpu.gemma_resolve_prefill_chunk_size(
            handle, int(execution_len), requested_chunk
        ))

    if not has_padding_policy:
        execution_len = (
            ((logical_len + 15) // 16) * 16
            if _prefix_pad16_enabled()
            else logical_len
        )
        return execution_len, resolve(execution_len)

    padding_rows = prefill.get("padding_rows", "auto")
    padding_budget = int(prefill.get("padding_budget", 15))
    # Leave the configured action horizon addressable after the physical prefix.
    physical_limit = (
        int(getattr(vlm, "_rpu_max_seq_len", 2048))
        - int(self.config.chunk_size)
    )

    def score(execution_len, chunk_size, num_chunks, padding, tail_deficit):
        return (
            num_chunks,
            int(execution_len % 16 != 0),
            padding,
            tail_deficit,
        )

    return plan_bounded_prefill_execution(
        logical_len,
        physical_limit,
        padding_budget,
        resolve,
        alignment=1,
        padding_rows=padding_rows,
        exact_chunk_size=(requested_chunk or None),
        candidate_score=score,
    )


def _prefill_prefix_embs(self, images, img_masks, tokens, masks, rpu_device):
    """Run VLM prefix embedding and cache-backed prefill."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from lerobot.policies.common.vla_utils import prepare_attention_masks_4d

    # VLM embed_tokens lives on CPU, so tokens must stay on CPU; they would
    # otherwise hit a dtype/device mismatch inside torch.embedding. masks
    # stay on CPU for lerobot mask ops. Images go to RPU only when SigLIP
    # is on RPU — lerobot's patched `embed_image` then passes them to the
    # RPU vision_tower.
    _siglip_param = next(
        self.paligemma_with_expert.paligemma.model.vision_tower.parameters(),
        None,
    )
    _siglip_on_rpu = (
        _siglip_param is not None and _siglip_param.device.type == 'rpu'
    )
    if _siglip_on_rpu and images is not None:
        images, img_masks = _prepare_prefill_siglip_inputs(
            self, images, img_masks, rpu_device
        )

    # --- VLM prefill matching the reference sample-actions flow ---
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, tokens, masks)
    if images is not None:
        vision_tower = (
            self.paligemma_with_expert.paligemma.model.vision_tower
        )
        vision_inner = getattr(vision_tower, "vision_model", vision_tower)
        vision_handle = getattr(vision_inner, "_rpu_vision_handle", None)
        if vision_handle is not None:
            vision_rows = len(images) * 256
            plans = vars(self).setdefault("_rpu_last_execution_plan", {})
            plans["vision"] = {
                "stage": "vision",
                "logical_len": vision_rows,
                "execution_len": vision_rows,
                "chunk_size": int(
                    torch.ops.rpu.siglip_get_resolved_chunk_size(vision_handle)
                ),
                "padding_rows": 0,
                "position": 0,
            }
    # Debug: optionally replace prefix_embs with a configured probe tensor.
    _pe_override = os.environ.get("RPU_PI05_LOAD_PREFIX_EMBS")
    if _pe_override and os.path.exists(_pe_override):
        prefix_embs = _load_pi05_probe_tensor(
            _pe_override,
            name="RPU_PI05_LOAD_PREFIX_EMBS",
            expected_shape=tuple(prefix_embs.shape),
        ).to(dtype=prefix_embs.dtype, device=prefix_embs.device)
    logical_prefix_len = int(prefix_pad_masks.shape[1])
    execution_prefix_len, planned_chunk_size = _plan_pi05_prefix_execution(
        self, logical_prefix_len
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = _pad_prefix_rows(
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        execution_prefix_len - logical_prefix_len,
    )
    _pi05_probe_save("prefix_embs", prefix_embs)
    _pi05_probe_save("prefix_pad_masks", prefix_pad_masks)
    _pi05_probe_save("prefix_att_masks", prefix_att_masks)
    # A cache hit returns the last (mask4d, pos_ids) verbatim;
    # cache miss with all-zero att uses pad-outer-product fast-path; otherwise
    # calls upstream make_att_2d_masks.
    prefix_att_2d_masks_4d, prefix_position_ids = _cached_build_prefix_mask_4d(
        prefix_pad_masks, prefix_att_masks, prepare_attention_masks_4d)
    _pi05_probe_save("prefix_att_2d_masks_4d", prefix_att_2d_masks_4d)
    _pi05_probe_save("prefix_position_ids", prefix_position_ids)
    self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    vlm = self.paligemma_with_expert.paligemma.model.language_model
    vlm._rpu_planned_prefill_chunk_size = int(planned_chunk_size)
    try:
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
    finally:
        vars(vlm).pop("_rpu_planned_prefill_chunk_size", None)
    resolved_chunk_size = int(torch.ops.rpu.gemma_get_resolved_chunk_size(
        vlm._rpu_vlm_decoder_handle
    ))
    if resolved_chunk_size != planned_chunk_size:
        raise RuntimeError(
            "Pi0.5 prefill dry/forward chunk mismatch: "
            f"planned={planned_chunk_size}, resolved={resolved_chunk_size}, "
            f"logical_len={logical_prefix_len}, "
            f"execution_len={execution_prefix_len}"
        )
    plans = vars(self).setdefault("_rpu_last_execution_plan", {})
    plans["prefill"] = {
        "stage": "prefill",
        "logical_len": logical_prefix_len,
        "execution_len": execution_prefix_len,
        "chunk_size": resolved_chunk_size,
        "padding_rows": execution_prefix_len - logical_prefix_len,
        "position": 0,
    }
    prefix_len = int(prefix_pad_masks.shape[1])
    past_key_values._prefix_len = prefix_len
    # Probe KV cache state after prefill for comparison with a reference run.
    if os.environ.get("RPU_PI05_PROBE_DIR"):
        _pi05_probe_save("kv_post_prefill_L0_k", past_key_values.k_caches[0])
        _pi05_probe_save("kv_post_prefill_L0_v", past_key_values.v_caches[0])
        _pi05_probe_save("kv_post_prefill_L8_k", past_key_values.k_caches[8])
        _pi05_probe_save("kv_post_prefill_L8_v", past_key_values.v_caches[8])
        _pi05_probe_save("kv_post_prefill_L17_k", past_key_values.k_caches[17])
        _pi05_probe_save("kv_post_prefill_L17_v", past_key_values.v_caches[17])

    return past_key_values, prefix_pad_masks, prefix_len


def _apply_kv_cache(self, past_key_values):
    """Bind the shared VLM KV cache to the action expert."""
    # Set expert._rpu_cache
    # to the VLM's RPUCache so `rpu_adarms_model_forward` sees it during the
    # denoise loop and skips its own RPUCache auto-creation.
    expert = self.paligemma_with_expert.gemma_expert.model
    expert._rpu_cache = past_key_values
    return expert


# =============================================================================
# Top-level denoise helpers.
# =============================================================================

def _build_attention_mask_4d(self, prefix_pad_masks, prefix_len, bsize,
                              chunk_size, dtype, device):
    """Build the [B, 1, S, S] 4D attention mask used by the denoise step.

    ``prefix_len`` is constant across denoise steps in one sample-actions call,
    so the mask is invariant and can be precomputed once.
    """
    cache_enabled = hasattr(self, "_rpu_denoise_mask_cache")
    cache_key = None
    if cache_enabled:
        cache_key = (
            tuple(prefix_pad_masks.shape),
            str(prefix_pad_masks.dtype),
            str(prefix_pad_masks.device),
            _hash_mask(prefix_pad_masks),
            int(prefix_len),
            int(bsize),
            int(chunk_size),
            str(dtype),
            str(device),
        )
        snapshot = self._rpu_denoise_mask_cache
        if snapshot is not None and snapshot[0] == cache_key:
            return snapshot[1]

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from lerobot.policies.common.vla_utils import prepare_attention_masks_4d
    suffix_pad_masks = torch.ones(bsize, chunk_size, dtype=torch.bool, device=device)
    suffix_att_masks = torch.tensor(
        [1] + [0] * (chunk_size - 1), dtype=dtype, device=device,
    )[None, :].expand(bsize, chunk_size)
    suffix_len = suffix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
        bsize, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
    if cache_enabled:
        self._rpu_denoise_mask_cache = (cache_key, full_att_2d_masks_4d)
    return full_att_2d_masks_4d


def _kv_prefix_full_hash(past_key_values, prefix_len: int) -> bytes:
    """Compute blake2b over all-layer K/V prefix slices.

    Walks the 7-D cache layout documented by ``RPUCache``:
      K[batch, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
      V[batch, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk]

    For prefix of length P with sKeyChunk = sValChunk = 16:
      - vx_full = P // 16
      - tail    = P % 16
      Full vx-blocks: cache[:, :vx_full, ..., :, :]  → contiguous
      Tail slice:     cache[:, vx_full:vx_full+1, ..., :, :tail] (K) /
                                                       :, :, :tail (V)
    """
    import hashlib
    h = hashlib.blake2b()
    s_chunk_k = past_key_values.sKeyChunk
    s_chunk_v = past_key_values.sValChunk
    for kc in past_key_values.k_caches:
        vx_full = prefix_len // s_chunk_k
        tail    = prefix_len %  s_chunk_k
        if vx_full > 0:
            h.update(kc[:, :vx_full].contiguous().cpu().numpy().tobytes())
        if tail > 0:
            h.update(kc[:, vx_full:vx_full + 1, :, :, :, :tail, :]
                       .contiguous().cpu().numpy().tobytes())
    for vc in past_key_values.v_caches:
        vx_full = prefix_len // s_chunk_v
        tail    = prefix_len %  s_chunk_v
        if vx_full > 0:
            h.update(vc[:, :vx_full].contiguous().cpu().numpy().tobytes())
        if tail > 0:
            # V last dim is sValChunk (dim 6), not dim 5 as in K.
            h.update(vc[:, vx_full:vx_full + 1, :, :, :, :, :tail]
                       .contiguous().cpu().numpy().tobytes())
    return h.digest()


def _run_denoise_python_baseline(self, x_t, prefix_pad_masks, past_key_values,
                                  num_steps, prefix_len, bsize):
    """CPU/RPU AdaRMS-per-step denoise path with an FP32 Euler accumulator."""
    from lerobot.policies.pi05.modeling_pi05 import (
        make_att_2d_masks, create_sinusoidal_pos_embedding)
    from lerobot.policies.common.vla_utils import prepare_attention_masks_4d
    import torch.nn.functional as F

    dt = -1.0 / num_steps
    cfg = self.config
    chunk_size = cfg.chunk_size
    device = x_t.device  # CPU for Pi0.5 denoise

    # _attn_implementation is invariant across denoise steps.
    self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
    # The Action Expert always returns chunk_size tokens.
    take_len = chunk_size

    # Precompute the deterministic time-position embeddings for all steps.
    sin_dim = self.action_in_proj.out_features
    sin_lut: list[torch.Tensor] = []
    for s in range(num_steps):
        tv = 1.0 + s * dt
        t_step = torch.tensor(tv, dtype=torch.float32, device=device).expand(bsize)
        emb = create_sinusoidal_pos_embedding(
            t_step, sin_dim, cfg.min_period, cfg.max_period, device=device)
        sin_lut.append(emb.type(t_step.dtype))

    # Suffix masks are shape-only and shared across denoise steps. Materialize
    # the attention mask after action_emb establishes its dtype.
    suffix_pad_masks = torch.ones(
        bsize, chunk_size, dtype=torch.bool, device=device)
    suffix_att_masks: torch.Tensor | None = None

    # Reuse one mask tensor so the pointer-keyed native cache remains valid.
    full_att_2d_masks_4d: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None

    for step in range(num_steps):
        # Reset before each step so suffix writes preserve prefix KV rows.
        past_key_values.reset_to_position(prefix_len)

        # Module calls retain the action/time projection dtype hooks.
        action_emb = self.action_in_proj(x_t)        # fp16 via hooks
        time_emb_pre = sin_lut[step]                 # fp32 (from LUT)
        x = self.time_mlp_in(time_emb_pre)           # fp16 via hooks
        x = F.silu(x)
        x = self.time_mlp_out(x)                     # fp16
        time_emb = F.silu(x)                         # fp16
        suffix_embs = action_emb
        adarms_cond = time_emb
        if suffix_att_masks is None:
            # First step only: materialize using action_emb's dtype.
            # Attention-mask dtype follows the embedding dtype.
            suffix_att_masks = torch.tensor(
                [1] + [0] * (chunk_size - 1),
                dtype=suffix_embs.dtype, device=suffix_embs.device,
            )[None, :].expand(bsize, chunk_size)
        if step == 0:
            _pi05_probe_save("suffix_embs_step0", suffix_embs)
            _pi05_probe_save("adarms_cond_step0", adarms_cond)
            _pi05_probe_save("suffix_pad_masks_step0", suffix_pad_masks)
            _pi05_probe_save("suffix_att_masks_step0", suffix_att_masks)

        # Construct the mask once at step 0 (data_ptr stable across
        # 5 forwards → C++ AdaRMSModel mask cache hits on steps 1-4).
        if step == 0:
            suffix_len = suffix_pad_masks.shape[1]
            # Preserve the reference denoise-step mask construction.
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                bsize, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
            _pi05_probe_save("att_mask_4d_step0", full_att_2d_masks_4d)
            _pi05_probe_save("position_ids_step0", position_ids)

        # Expert forward via paligemma_with_expert — routes through
        # `rpu_adarms_model_forward`. That function:
        #   - auto-routes CPU inputs to RPU
        #   - invokes torch.ops.rpu.adarms_forward (fused AdaRMS C++ op)
        #   - applies final PiGemmaRMSNorm(output, adarms_cond)
        #   - returns `BaseModelOutputWithPast(last_hidden_state=...)` on CPU
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        if step == 0:
            _pi05_probe_save("suffix_out_step0_full", suffix_out)

        # Preserve the reference output projection semantics:
        #   - take last chunk_size tokens (take_len precomputed above)
        #   - cast to FP32 because action_out_proj runs in FP32 via its hook
        #   - action_out_proj on CPU fp32 -> v_t is fp16 (post-hook cast)
        suffix_out = suffix_out[:, -take_len:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        if step == 0:
            _pi05_probe_save("suffix_out_step0_sliced", suffix_out)
        v_t = self.action_out_proj(suffix_out)
        if step == 0:
            _pi05_probe_save("v_t_step0", v_t)

        # Python type promotion keeps the Euler accumulator in FP32:
        # `x_t + dt * v_t` to fp32 because x_t is fp32 and (dt*v_t) is fp32
        # (Python float * fp16 -> fp32).
        x_t = x_t + dt * v_t
        if step == 0:
            _pi05_probe_save("x_t_after_step0", x_t)

    return x_t


def _run_denoise_fused(self, x_t_cpu_fp32, prefix_pad_masks, past_key_values,
                        num_steps, prefix_len, bsize):
    """Run the fused denoise loop.

    Mirrors _run_denoise_python_baseline at the boundary (x_t CPU fp32 in/out);
    inside the loop x_t lives in RPU FP16 while the Euler accumulator stays in
    CPU FP32.
    """
    import torch.nn.functional as F
    from rpu_backend.adapters.pi05.weights import _precompute_adarms_cond_all
    from rpu_backend.graph import GraphSignature

    handle = getattr(self, "_rpu_fused_denoise_handle", None)
    if handle is None:
        raise RuntimeError("_run_denoise_fused requires fused handle")
    cache = getattr(self, "_rpu_fused_denoise_graph_cache", None)
    if cache is None:
        raise RuntimeError("_run_denoise_fused requires fused graph cache")
    use_graph = _denoise_graph_enabled()

    # In-graph unroll uses device-side FP16 Euler and requires graph mode.
    if _denoise_unroll_enabled() and use_graph:
        return _run_denoise_loop(self, x_t_cpu_fp32, prefix_pad_masks,
                                 past_key_values, num_steps, prefix_len, bsize)

    cfg = self.config
    cs   = cfg.chunk_size
    mad  = cfg.max_action_dim
    dt   = -1.0 / num_steps

    # Bound check (echoes the C++ TORCH_CHECK; fail fast in Python).
    k0 = past_key_values.k_caches[0]
    cache_max_seq = k0.size(1) * k0.size(5)  # sKeyVx * sKeyChunk
    if prefix_len + cs > cache_max_seq:
        raise RuntimeError(
            f"_run_denoise_fused: prefix_len({prefix_len}) + chunk_size({cs}) "
            f"exceeds k_cache.max_seq_len={cache_max_seq}.")

    adarms_cond_all_rpu = _precompute_adarms_cond_all(self, num_steps)

    x_t_rpu = torch.empty(bsize, cs, mad, dtype=torch.float16, device='rpu')
    v_t_buf = torch.empty(bsize, cs, mad, dtype=torch.float16, device='rpu')
    x_t_rpu.copy_(x_t_cpu_fp32.to(torch.float16))

    mask_4d = _build_attention_mask_4d(
        self, prefix_pad_masks, prefix_len, bsize, cs,
        dtype=torch.float16, device='rpu')

    expert_model = self.paligemma_with_expert.gemma_expert.model
    ec = expert_model.config
    effective_num_kv_heads = int(getattr(
        expert_model, "_rpu_effective_num_kv_heads", ec.num_key_value_heads
    ))
    rope_pos = _set_denoise_rope_position(handle, prefix_pad_masks)
    sig = GraphSignature(
        op_id="pi05_denoise_step",
        shapes=[bsize, cs, mad],
        dyn_dims=[len(past_key_values.k_caches),
                  ec.num_attention_heads, effective_num_kv_heads,
                  ec.head_dim, prefix_len, num_steps,
                  self.action_in_proj.out_features,
                  int(_kvinsert_pad16_enabled()),  # pad16 changes buffer/insert len
                  rope_pos],                       # baked at BUILD
        dtypes=[torch.float16],
    )

    for step in range(num_steps):
        cond_step = adarms_cond_all_rpu.select(0, step).contiguous()
        if use_graph:
            with cache.capture(sig):
                torch.ops.rpu.pi05_denoise_step_forward(
                    handle, x_t_rpu,
                    past_key_values.k_caches, past_key_values.v_caches,
                    cond_step, mask_4d, v_t_buf, prefix_len)
        else:
            torch.ops.rpu.pi05_denoise_step_forward(
                handle, x_t_rpu,
                past_key_values.k_caches, past_key_values.v_caches,
                cond_step, mask_4d, v_t_buf, prefix_len)
        # Keep the Euler accumulator in CPU FP32.
        x_t_cpu_fp32 = x_t_cpu_fp32 + dt * v_t_buf.to('cpu', torch.float32)
        if _euler_fp16_enabled():  # A/B: model an on-device fp16 Euler
            x_t_cpu_fp32 = x_t_cpu_fp32.to(torch.float16).to(torch.float32)
        x_t_rpu.copy_(x_t_cpu_fp32.to(torch.float16))  # auto-flushes
    plans = vars(self).setdefault("_rpu_last_execution_plan", {})
    plans["action"] = {
        "stage": "action",
        "logical_len": int(cs),
        "execution_len": int(cs),
        "chunk_size": int(
            torch.ops.rpu.pi05_denoise_step_get_resolved_chunk_size(handle)
        ),
        "padding_rows": 0,
        "position": int(prefix_len),
    }
    return x_t_cpu_fp32



def _set_denoise_rope_position(handle, prefix_pad_masks) -> int:
    """Give the fused denoise expert the suffix's LOGICAL RoPE start.

    `prefix_len` (= prefix_pad_masks.shape[1]) is the PHYSICAL cache row the
    suffix K/V is written at, and the fused expert used it for RoPE too. The
    reference derives the suffix positions from `sum(prefix_pad_masks)` — the
    LOGICAL count — so the two agree only while the prefix has no pad rows.
    The suffix itself is never padded (suffix_pad_masks is all ones), so a
    scalar start is enough here; the PREFIX needs a gathered table instead
    (adapters/pi05/gemma.py).

    Returns the value so callers can put it in the graph signature — it is
    baked into the kernel at BUILD.
    """
    pos = int(prefix_pad_masks[0].sum())
    torch.ops.rpu.pi05_denoise_step_set_rope_position(handle, pos)
    return pos

def _run_denoise_loop(self, x_t_cpu_fp32, prefix_pad_masks, past_key_values,
                      num_steps, prefix_len, bsize):
    """In-graph N-step unroll: ONE pi05_denoise_loop_forward call (on-device
    fp16 Euler; 1 BUILD + 1 REPLAY). Same boundary contract as
    _run_denoise_fused (x_t CPU fp32 in, final action CPU fp32 out). The cond
    stack [num_steps, h_ada] is passed whole; C++ indexes it per body_iter."""
    from rpu_backend.adapters.pi05.weights import _precompute_adarms_cond_all
    from rpu_backend.graph import GraphSignature

    handle = getattr(self, "_rpu_fused_denoise_handle", None)
    if handle is None:
        raise RuntimeError("_run_denoise_loop requires fused handle")
    cache = getattr(self, "_rpu_fused_denoise_graph_cache", None)
    if cache is None:
        raise RuntimeError("_run_denoise_loop requires fused graph cache")

    cfg = self.config
    cs   = cfg.chunk_size
    mad  = cfg.max_action_dim
    dt   = -1.0 / num_steps

    k0 = past_key_values.k_caches[0]
    cache_max_seq = k0.size(1) * k0.size(5)
    if prefix_len + cs > cache_max_seq:
        raise RuntimeError(
            f"_run_denoise_loop: prefix_len({prefix_len}) + chunk_size({cs}) "
            f"exceeds k_cache.max_seq_len={cache_max_seq}.")

    cond_all = _precompute_adarms_cond_all(self, num_steps)  # [num_steps, h_ada]
    x0 = torch.empty(bsize, cs, mad, dtype=torch.float16, device='rpu')
    x0.copy_(x_t_cpu_fp32.to(torch.float16))
    x_out = torch.empty(bsize, cs, mad, dtype=torch.float16, device='rpu')

    mask_4d = _build_attention_mask_4d(
        self, prefix_pad_masks, prefix_len, bsize, cs,
        dtype=torch.float16, device='rpu')

    expert_model = self.paligemma_with_expert.gemma_expert.model
    ec = expert_model.config
    effective_num_kv_heads = int(getattr(
        expert_model, "_rpu_effective_num_kv_heads", ec.num_key_value_heads
    ))
    rope_pos = _set_denoise_rope_position(handle, prefix_pad_masks)
    sig = GraphSignature(
        op_id="pi05_denoise_loop",
        shapes=[bsize, cs, mad],
        dyn_dims=[len(past_key_values.k_caches),
                  ec.num_attention_heads, effective_num_kv_heads,
                  ec.head_dim, prefix_len, num_steps,
                  self.action_in_proj.out_features,
                  int(_kvinsert_pad16_enabled()),  # pad16 changes buffer/insert len
                  rope_pos],                       # baked at BUILD
        dtypes=[torch.float16],
    )
    with cache.capture(sig):
        torch.ops.rpu.pi05_denoise_loop_forward(
            handle, x0,
            past_key_values.k_caches, past_key_values.v_caches,
            cond_all, mask_4d, x_out, dt, prefix_len, num_steps)
    plans = vars(self).setdefault("_rpu_last_execution_plan", {})
    plans["action"] = {
        "stage": "action",
        "logical_len": int(cs),
        "execution_len": int(cs),
        "chunk_size": int(
            torch.ops.rpu.pi05_denoise_step_get_resolved_chunk_size(handle)
        ),
        "padding_rows": 0,
        "position": int(prefix_len),
    }
    return x_out.to('cpu', torch.float32)


# =============================================================================
# ``sample_actions`` class patch.
# =============================================================================

def install_sample_actions_patch() -> None:
    """Class-level idempotent patch of `PI05Pytorch.sample_actions`. Installed
    ONCE per process (sentinel-guarded). The patched body:
      - Uses the installed LeRobot signature.
      - Looks up pi05_handle from `_PI05_HANDLES[self]` at call time.
      - Precomputes ONLY adarms_cond_list (safe — no x_t dep) + attention_mask + prefix K/V.
        Passes initial noise to pi05_forward.
      - Does not precompute a position_ids tensor.
      - Validates kwargs against the allowlist; unknown kwargs raise.
      - Routes `num_steps` through to pi05_forward.
      - If RTC is enabled, falls back to the original CPU path.

    Helper 1 (_resolve_handle_and_validate) stays nested: closes over
    `_orig_sample_actions` + `_RPU_ALLOWED_SAMPLE_ACTIONS_KWARGS`.
    _run_denoise stays nested: dispatcher that references top-level
    _run_denoise_python_baseline and _run_denoise_fused.
    """
    try:
        import lerobot  # noqa: F401 — probe only
    except ImportError as e:
        raise ImportError("Pi0.5 requires lerobot. Install with: pip install lerobot\nSee docs/api_reference.md#policy-apis for Pi0.5 setup.") from e
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    if getattr(PI05Pytorch, "_rpu_patched_sample_actions", False):
        return

    _orig_sample_actions = PI05Pytorch.sample_actions

    # -------------------------------------------------------------------
    # Validate the handle, RTC fallback, kwarg allowlist, and batch size. Returns a sentinel tuple
    # `(proceed, fallback_kwargs)` — if proceed=False, caller returns
    # `_orig_sample_actions(..., **fallback_kwargs)` immediately.
    # STAYS nested: closes over `_orig_sample_actions`.
    # -------------------------------------------------------------------
    def _resolve_handle_and_validate(self, images, img_masks, tokens, masks,
                                      noise, num_steps, kwargs):
        handle = _PI05_HANDLES.get(self)
        if handle is None:
            # Not RPU-swizzled for this instance -> CPU original fallback.
            return (False, _orig_sample_actions(
                self, images, img_masks, tokens, masks,
                noise=noise, num_steps=num_steps, **kwargs))

        # RTC uses the original CPU path.
        if hasattr(self, "_rtc_enabled") and self._rtc_enabled():
            return (False, _orig_sample_actions(
                self, images, img_masks, tokens, masks,
                noise=noise, num_steps=num_steps, **kwargs))

        # Fail on unexpected kwargs instead of silently dropping them.
        unexpected = set(kwargs) - _RPU_ALLOWED_SAMPLE_ACTIONS_KWARGS
        if unexpected:
            raise RPUBackendError(
                f"Pi05Policy RPU-path sample_actions received unsupported kwargs: "
                f"{sorted(unexpected)}. Either extend "
                f"_RPU_ALLOWED_SAMPLE_ACTIONS_KWARGS in runtime.py or disable the "
                f"RPU path for this inference call.")

        bsize = tokens.shape[0]
        if bsize != 1:
            raise RPUBackendError(
                f"Pi05 RPU path requires batch_size=1, got {bsize}.")

        return (True, None)

    # -------------------------------------------------------------------
    # Denoise dispatcher.
    # STAYS nested: sits alongside _patched_sample_actions; references
    # top-level _run_denoise_python_baseline and _run_denoise_fused.
    # -------------------------------------------------------------------
    def _run_denoise(self, x_t, prefix_pad_masks, past_key_values, num_steps,
                     prefix_len, bsize):
        """Dispatcher: fused (default) -> baseline + safety-net on first call."""
        import torch.nn.functional as F

        handle = getattr(self, "_rpu_fused_denoise_handle", None)
        if handle is None:
            return _run_denoise_python_baseline(
                self, x_t, prefix_pad_masks, past_key_values,
                num_steps, prefix_len, bsize)

        if getattr(self, "_rpu_fuse_validated", False):
            return _run_denoise_fused(
                self, x_t, prefix_pad_masks, past_key_values,
                num_steps, prefix_len, bsize)

        # Validate the fused path against the baseline on its first call.
        prefix_hash_before = _kv_prefix_full_hash(past_key_values, prefix_len)

        final_baseline = _run_denoise_python_baseline(
            self, x_t.clone(), prefix_pad_masks, past_key_values,
            num_steps, prefix_len, bsize)
        prefix_hash_after_baseline = _kv_prefix_full_hash(past_key_values, prefix_len)
        if prefix_hash_before != prefix_hash_after_baseline:
            raise RuntimeError(
                "[Pi05Fused] safety net: baseline path corrupted prefix.")

        final_fused = _run_denoise_fused(
            self, x_t.clone(), prefix_pad_masks, past_key_values,
            num_steps, prefix_len, bsize)
        prefix_hash_after_fused = _kv_prefix_full_hash(past_key_values, prefix_len)
        if prefix_hash_before != prefix_hash_after_fused:
            raise RuntimeError(
                "[Pi05Fused] zero-copy violated: fused wrote into prefix KV. "
                "Disable via RPU_PI05_FUSED_DENOISE=0 and file a bug.")

        cos = F.cosine_similarity(
            final_fused.flatten().float(),
            final_baseline.flatten().float(), dim=0).item()
        if cos < 0.9995:
            raise RuntimeError(
                f"[Pi05Fused] safety net mismatch: cos_sim={cos:.4f} < 0.9995. "
                f"Disable via RPU_PI05_FUSED_DENOISE=0 and file a bug.")

        self._rpu_fuse_validated = True
        return final_fused

    # -------------------------------------------------------------------
    # OUTER wrapper — matches lerobot signature; calls helpers 1..6 in order.
    # -------------------------------------------------------------------
    @torch.no_grad()
    def _patched_sample_actions(self, images, img_masks, tokens, masks,
                                 noise=None, num_steps=None, **kwargs):
        """Match the installed LeRobot ``sample_actions`` signature."""
        # Helper 1: resolve handle + RTC + allowlist + bsize guard
        proceed, fallback = _resolve_handle_and_validate(
            self, images, img_masks, tokens, masks, noise, num_steps, kwargs)
        if not proceed:
            return fallback

        if num_steps is None:
            num_steps = self.config.num_inference_steps
        _validate_prepared_graph_profile(self, num_steps)

        # The fused denoise loop is the previous prediction's final subsystem
        # and intentionally keeps its graph-owned SPM plan alive. Reset the
        # global temporary allocator at the next top-level boundary, before
        # entering any per-subsystem capture. This gives ordinary consecutive
        # policy calls the same deterministic boundary.
        torch.ops.rpu.spm_alloc_reset_temporary()

        bsize = tokens.shape[0]

        # Python denoise loop matching `_patched_denoise_step` semantics.

        # Helper 2: resolve RPU device (top-level)
        rpu_device = _resolve_rpu_device(self)

        # Helper 3: noise sampling (top-level)
        x_t = _sample_noise(self, bsize, noise)

        # Helper 4: VLM prefill (top-level)
        past_key_values, prefix_pad_masks, prefix_len = _prefill_prefix_embs(
            self, images, img_masks, tokens, masks, rpu_device)

        # Helper 5: apply shared KV cache to expert (top-level)
        _apply_kv_cache(self, past_key_values)

        # Helper 6: denoise dispatcher
        x_t = _run_denoise(self, x_t, prefix_pad_masks, past_key_values,
                           num_steps, prefix_len, bsize)

        return x_t

    PI05Pytorch.sample_actions = _patched_sample_actions
    PI05Pytorch._rpu_patched_sample_actions = True

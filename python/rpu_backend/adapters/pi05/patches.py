"""Lazy, idempotent class-level patches used by ``Pi05Adapter.to_rpu``.

Class-level sentinels (``cls._rpu_X_patched``) prevent repeated installation.
"""
from __future__ import annotations
import torch
from rpu_backend.runtime import rpu_env_bool


# Exact Python mirror of SpmAllocator::SPM_PLANNING_BUDGET.
_SPM_USABLE_BYTES = 8 * 1024 * 1024 - 1024
_SPM_PLANNING_BUDGET_BYTES = _SPM_USABLE_BYTES * 99 // 100


# =============================================================================
# Class-level patch helpers (idempotent and sentinel-guarded).
# =============================================================================

def _install_pi_gemma_rmsnorm_class_patch() -> None:
    """CLASS-level idempotent patch of PiGemmaRMSNorm to route compute CPU-fp32."""
    try:
        from lerobot.policies.pi_gemma import PiGemmaRMSNorm
    except ImportError:
        return

    if getattr(PiGemmaRMSNorm, "_rpu_cpu_safe_patched", False):
        return

    _orig_forward = PiGemmaRMSNorm.forward

    def _cpu_safe_forward(self, x, cond=None):
        device = x.device
        x_cpu = x.to('cpu') if x.device.type != 'cpu' else x
        cond_cpu = cond.to('cpu') if (cond is not None and cond.device.type != 'cpu') else cond
        normed, gate = _orig_forward(self, x_cpu, cond=cond_cpu)
        if device.type != 'cpu':
            normed = normed.to(device)
            if gate is not None:
                gate = gate.to(device)
        return normed, gate

    PiGemmaRMSNorm.forward = _cpu_safe_forward
    PiGemmaRMSNorm._rpu_cpu_safe_patched = True


def _install_gemma_rotary_class_patch() -> None:
    """CLASS-level idempotent patch of GemmaRotaryEmbedding to output kernel-format cos/sin.

    The fused Gemma/AdaRMS kernels consume cos/sin in kernel format
    `[seq_len, head_dim//2]`; the unpatched transformers Gemma rotary returns
    `[batch, seq_len, head_dim]` with `cat(freqs, freqs)` duplicated halves.
    Without this patch, `patch_gemma_model_for_rpu_all_layers_once` reads
    the wrong format when it calls `rotary(dummy, dummy_pos)` on RPU and passes
    wrong cos/sin to `gemma_set_weights`, producing incorrect RoPE angles.

    The patch must be installed before the fused model consumes the RoPE tables.
    """
    try:
        from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
    except ImportError:
        return
    from rpu_backend.runtime.decoder import _install_rotary_class_swap
    if getattr(GemmaRotaryEmbedding, "_rpu_kernel_fmt_patched", False):
        return
    _install_rotary_class_swap(GemmaRotaryEmbedding)
    GemmaRotaryEmbedding._rpu_kernel_fmt_patched = True


def _pi05_runtime_device(policy) -> torch.device:
    """Return the adapter-pinned device without walking the model tree."""
    device = getattr(policy.model, "_rpu_runtime_device", None)
    if device is None:
        device = next(policy.parameters()).device
    return torch.device(device)


def _batched_preprocess_images(
    policy,
    batch,
    device: torch.device,
    *,
    fp16_output: bool | None = None,
):
    """Batch equal-shape camera preprocessing into one tensor operation stream.

    Returns ``None`` when the input is outside the narrow fast-path contract.
    Camera ordering, resize/pad math, normalization, and output layout mirror
    LeRobot's ``PI05Policy._preprocess_images`` exactly. Pi0.5 inference is
    batch-1, so each camera becomes one dim-0 slice of the shared slab.
    """
    image_keys = list(policy.config.image_features)
    if len(image_keys) < 2 or any(key not in batch for key in image_keys):
        return None

    source = [batch[key] for key in image_keys]
    first = source[0]
    if (
        not all(isinstance(img, torch.Tensor) for img in source)
        or first.ndim != 4
        or first.shape[0] != 1
        or any(
            img.ndim != 4
            or img.shape != first.shape
            or img.dtype != first.dtype
            or img.device != first.device
            for img in source[1:]
        )
    ):
        return None

    channels_first = first.shape[1] == 3
    channels_last = first.shape[-1] <= 4
    if not channels_first and not channels_last:
        return None

    # These CPU-only operations become scheduling-bound with excessive
    # intra-op parallelism. Scope the limit to this block and always restore
    # the caller's setting.
    ambient_threads = torch.get_num_threads()
    scoped_threads = (
        min(ambient_threads, 8)
        if first.device.type == "cpu"
        else ambient_threads
    )
    if scoped_threads != ambient_threads:
        torch.set_num_threads(scoped_threads)
    try:
        # Concatenate on the source device (normally CPU) before the single
        # upload. One [N,3,H,W] allocation also lets SigLIP consume cheap views.
        images = torch.cat(source, dim=0)
        if images.device != device:
            images = images.to(device)
        if images.dtype != torch.float32:
            images = images.to(torch.float32)

        if channels_first:
            images = images.permute(0, 2, 3, 1)

        if tuple(images.shape[1:3]) != tuple(
            policy.config.image_resolution
        ):
            from lerobot.policies.pi05.modeling_pi05 import (
                resize_with_pad_torch,
            )
            images = resize_with_pad_torch(
                images, *policy.config.image_resolution
            )

        images.mul_(2.0).sub_(1.0)
        if channels_first:
            images = images.permute(0, 3, 1, 2)
        if fp16_output is None:
            fp16_output = device.type != "cpu"
        if fp16_output and images.dtype != torch.float16:
            images = images.half()

        per_camera = [
            images.narrow(0, camera_idx, 1)
            for camera_idx in range(len(image_keys))
        ]
        # Preserve the shared slab identity for the packed SigLIP upload.
        for camera_idx, image in enumerate(per_camera):
            image._pi05_batch_slab = images
            image._pi05_batch_index = camera_idx
        mask = torch.ones(1, dtype=torch.bool, device=device)
        return per_camera, [mask] * len(image_keys)
    finally:
        if scoped_threads != ambient_threads:
            torch.set_num_threads(ambient_threads)


def _preprocess_outputs_equal(candidate, reference) -> bool:
    """Strict one-time parity check for the batched preprocessing fast path."""
    candidate_images, candidate_masks = candidate
    reference_images, reference_masks = reference
    if (
        len(candidate_images) != len(reference_images)
        or len(candidate_masks) != len(reference_masks)
    ):
        return False

    for actual, expected in zip(
        candidate_images, reference_images, strict=True
    ):
        if (
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or actual.device != expected.device
            or not torch.equal(actual.cpu(), expected.cpu())
        ):
            return False
    for actual, expected in zip(
        candidate_masks, reference_masks, strict=True
    ):
        if (
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or actual.device != expected.device
            or not torch.equal(actual.cpu(), expected.cpu())
        ):
            return False
    return True


def _install_preprocess_images_class_patch(policy) -> None:
    """Batch eligible camera preprocessing and keep RPU images fp16.

    The inference fast path combines equal-shape, fully-present cameras into one
    slab. Its first eligible call is compared strictly against LeRobot's
    per-camera implementation. A mismatch or runtime failure disables the fast
    path for that policy instance; training and dynamic/missing-camera inputs
    always retain the reference path.
    """
    cls = type(policy)
    if getattr(cls, "_rpu_preprocess_images_patched", False):
        return
    _orig_preprocess = cls._preprocess_images

    def _reference_preprocess_images(self, batch):
        images, img_masks = _orig_preprocess(self, batch)
        device = _pi05_runtime_device(self)
        if device.type != 'cpu':
            images = [img.half() if img.is_floating_point() and img.dtype != torch.float16 else img
                      for img in images]
        return images, img_masks

    def _fp16_preprocess_images(self, batch):
        if (
            self.training
            or not getattr(self, "_pi05_batch_preproc_enabled", True)
            or getattr(self, "_pi05_batch_preproc_ok", None) is False
        ):
            return _reference_preprocess_images(self, batch)

        runtime_device = _pi05_runtime_device(self)
        preprocess_device = getattr(
            self, "_pi05_preprocess_device", None
        )
        if preprocess_device is None:
            try:
                preprocess_device = next(self.parameters()).device
            except StopIteration:
                preprocess_device = runtime_device
            preprocess_device = torch.device(preprocess_device)
            self._pi05_preprocess_device = preprocess_device
        try:
            candidate = _batched_preprocess_images(
                self,
                batch,
                preprocess_device,
                fp16_output=runtime_device.type != "cpu",
            )
        except Exception as exc:
            self._pi05_batch_preproc_ok = False
            import warnings
            warnings.warn(
                "pi05 batched image preprocessing failed; falling back for "
                f"this policy instance ({type(exc).__name__}: {exc}).",
                RuntimeWarning,
            )
            return _reference_preprocess_images(self, batch)

        if candidate is None:
            return _reference_preprocess_images(self, batch)

        if getattr(self, "_pi05_batch_preproc_ok", None) is None:
            reference = _reference_preprocess_images(self, batch)
            if not _preprocess_outputs_equal(candidate, reference):
                self._pi05_batch_preproc_ok = False
                import warnings
                warnings.warn(
                    "pi05 batched image preprocessing differs from LeRobot; "
                    "falling back for this policy instance.",
                    RuntimeWarning,
                )
                return reference
            self._pi05_batch_preproc_ok = True
        return candidate

    cls._preprocess_images = _fp16_preprocess_images
    cls._rpu_preprocess_images_patched = True


def _install_eval_fastpath_class_patch(policy) -> None:
    """Make repeated ``eval()`` calls O(1) for an already-eval RPU policy.

    LeRobot calls ``self.eval()`` on every ``predict_action_chunk``. Once the
    adapter has recursively put the policy in eval mode, repeating that tree
    walk cannot change inference state. A later explicit ``train()`` flips
    ``self.training`` back to True, so the next eval still delegates to the
    original recursive implementation.
    """
    cls = type(policy)
    if getattr(cls, "_rpu_eval_fastpath_patched", False):
        return
    _orig_eval = cls.eval

    def _rpu_eval_fastpath(self):
        if getattr(self, "_rpu_swizzled", False) and not self.training:
            return self
        return _orig_eval(self)

    cls.eval = _rpu_eval_fastpath
    cls._rpu_eval_fastpath_patched = True


# Pi0.5 is fully prefix-LM, so the patched path emits zero attention masks.

# Captured at module-import time. If/when other patches start modifying
# embed_prefix, lift this capture into _install_embed_prefix_class_patch body.
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch as _PI05Pytorch
_orig_embed_prefix = _PI05Pytorch.embed_prefix


def _language_weight_identity(model):
    """Return a cheap cache identity for the CPU token-embedding weight."""
    try:
        weight = (
            model.paligemma_with_expert.paligemma.model.language_model
            .embed_tokens.weight
        )
    except AttributeError:
        return None
    return id(weight), int(weight._version)


def _assemble_packed_prefix_ondevice(self, packed_emb, tokens):
    """Assemble image + language prefix on RPU with an LRU-1 text cache.

    `assemble_inputs_embeds` normally gathers from a full vocabulary. Here the
    already-scaled language rows form a tiny per-prompt vocabulary: image rows
    temporarily gather row 0 and are then overwritten by `packed_emb`; language
    rows gather 0..L-1. This reuses Wall-OSS's device-only gather/scatter/DMA
    path without keeping Pi0.5's ~1 GiB embedding table resident on RPU.

    Returns ``(None, lang_emb)`` when the narrow batch-1 fp16 contract is not
    met, so the caller can preserve the established `torch.cat` path.
    """
    lang_emb = None
    if (
        packed_emb.device.type != "rpu"
        or packed_emb.dtype != torch.float16
        or packed_emb.ndim != 3
        or packed_emb.shape[0] != 1
        or tokens.device.type != "cpu"
        or tokens.ndim != 2
        or tokens.shape[0] != 1
    ):
        lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
        return None, lang_emb

    image_rows = int(packed_emb.shape[1])
    weight_identity = _language_weight_identity(self)
    cache = getattr(self, "_pi05_language_prefix_cache", None)
    cache_hit = (
        cache is not None
        and cache["image_rows"] == image_rows
        and cache["weight_identity"] == weight_identity
        and cache["tokens"].shape == tokens.shape
        and cache["tokens"].dtype == tokens.dtype
        and torch.equal(cache["tokens"], tokens)
    )
    if cache_hit:
        lang_emb = cache["lang_cpu"]
        lang_rpu = cache["lang_rpu"]
        ids_rpu = cache["ids_rpu"]
    else:
        lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
        if (
            lang_emb.device.type != "cpu"
            or lang_emb.dtype != torch.float16
            or lang_emb.ndim != 3
            or lang_emb.shape[0] != 1
            or lang_emb.shape[2] != packed_emb.shape[2]
        ):
            return None, lang_emb

        lang_rows = int(lang_emb.shape[1])
        assembly_bytes = (
            (image_rows + lang_rows) * int(lang_emb.shape[2]) * 2
        )
        # Python mirror of floor(0.99 * SPM_USABLE) = 8,303,708 bytes/core.
        if lang_rows == 0 or assembly_bytes > _SPM_PLANNING_BUDGET_BYTES:
            return None, lang_emb
        lang_rpu = lang_emb.squeeze(0).contiguous().to("rpu")
        ids = torch.empty(
            image_rows + lang_rows, dtype=torch.int32, device="cpu"
        )
        ids[:image_rows].zero_()
        ids[image_rows:] = torch.arange(lang_rows, dtype=torch.int32)
        ids_rpu = ids.to("rpu").contiguous()
        self._pi05_language_prefix_cache = {
            "image_rows": image_rows,
            "weight_identity": weight_identity,
            "tokens": tokens.detach().clone(),
            "lang_cpu": lang_emb,
            "lang_rpu": lang_rpu,
            "ids_rpu": ids_rpu,
        }

    assembled = torch.ops.rpu.assemble_inputs_embeds(
        lang_rpu,
        ids_rpu,
        packed_emb.squeeze(0),
        0,
    )
    return assembled.unsqueeze(0), lang_emb


def _patched_embed_prefix(self, images, img_masks, tokens, masks):
    """Inference ``embed_prefix`` for Pi0.5.

    Pi0.5 is fully prefix-LM, so the final attention mask is equivalent to
    ``torch.zeros(B, total, dtype=bool)``.

    Training delegates to the reference path to preserve checkpointing semantics.
    """
    if self.training:
        return _orig_embed_prefix(self, images, img_masks, tokens, masks)

    embs = []
    pad_masks = []
    total_prefix_len = 0
    packed_emb = None

    # Pack configured camera images into one encoder forward. Minibatch SDPA
    # preserves per-image isolation, and output rows remain grouped by image.
    _batch_n = getattr(self, '_rpu_siglip_batch_n', 1)
    if _batch_n > 1 and len(images) == _batch_n:
        # The multi op owns patch embedding and returns [1, N*256, 2048].
        from rpu_backend.adapters.siglip import _siglip_forward_images
        _vt = self.paligemma_with_expert.paligemma.model.vision_tower
        _vision_model = _vt.vision_model if hasattr(_vt, 'vision_model') else _vt
        packed_emb = _siglip_forward_images(_vision_model, list(images))  # [1, N*256, 2048]
        bsize = packed_emb.shape[0]
        per_img_embs = packed_emb.shape[1] // _batch_n
        # ABSENT CAMERAS ARE DROPPED, NOT MASKED. Upstream keeps a
        # missing camera as an all-(-1) image with a ZERO img_mask, so the
        # prefix gets a 256-row hole and lerobot's positions become
        # `cumsum(pad_masks) - 1`, renumbering everything after the hole. The
        # fused Gemma prefill takes `position_ids` in its signature and NEVER
        # PASSES IT to the op (adapters/pi05/gemma.py: only attention_mask and
        # is_causal reach torch.ops.rpu.gemma_forward), so RoPE there is always
        # 0..L-1. With every camera present cumsum-1 is arange; with one absent,
        # later text tokens would otherwise use the wrong positions.
        #
        # Dropping the rows is exact, not a workaround: those tokens are masked
        # out of every query (att_2d = pad ⊗ pad), their own outputs are
        # discarded, and removing them is precisely what makes the surviving
        # positions contiguous — i.e. it reproduces the numbering RoPE here
        # already assumes. The prefix also gets shorter, so it costs less.
        keep = [i for i, m in enumerate(img_masks) if bool(m.any())]
        if len(keep) < _batch_n:
            if keep == list(range(len(keep))):
                packed_emb = packed_emb[:, :len(keep) * per_img_embs, :]
            else:
                # Non-prefix subset: reindex host-side, the same round-trip the
                # assembly fallback further down already uses.
                packed_emb = torch.cat(
                    [packed_emb[:, i * per_img_embs:(i + 1) * per_img_embs, :].cpu()
                     for i in keep], dim=1).to(packed_emb.device)
            img_masks = [img_masks[i] for i in keep]
        total_prefix_len += packed_emb.shape[1]
        for img_mask in img_masks:
            pad_masks.append(torch.ones(bsize, per_img_embs, dtype=torch.bool,
                                        device=img_mask.device))
    else:
        for img, img_mask in zip(images, img_masks, strict=True):
            # eval mode: _apply_checkpoint is a no-op direct-call wrapper; skip it.
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            total_prefix_len += num_img_embs

    assembled_embs = None
    lang_emb = None
    assembly_enabled = (
        packed_emb is not None
        and getattr(self, "_pi05_prefix_assembly_enabled", True)
        and getattr(self, "_pi05_prefix_assembly_ok", None) is not False
    )
    if assembly_enabled:
        try:
            assembled_embs, lang_emb = _assemble_packed_prefix_ondevice(
                self, packed_emb, tokens
            )
        except Exception as exc:
            self._pi05_prefix_assembly_ok = False
            self._pi05_language_prefix_cache = None
            import warnings
            warnings.warn(
                "pi05 device prefix assembly failed; using torch.cat for "
                f"this model instance ({type(exc).__name__}: {exc}).",
                RuntimeWarning,
            )

    if lang_emb is None:
        lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)

    if assembled_embs is not None:
        if getattr(self, "_pi05_prefix_assembly_ok", None) is None:
            # The optimized path deliberately keeps the packed image rows on
            # RPU while the stable language cache stays on CPU.  Build the
            # one-time parity oracle on CPU; a mixed-device torch.cat would
            # fail before the assembly path could certify itself.
            reference_cpu = torch.cat(
                [packed_emb.cpu(), lang_emb.cpu()], dim=1
            )
            assembly_equal = (
                assembled_embs.shape == reference_cpu.shape
                and assembled_embs.dtype == reference_cpu.dtype
                and torch.equal(
                    assembled_embs.cpu(), reference_cpu
                )
            )
            if not assembly_equal:
                self._pi05_prefix_assembly_ok = False
                self._pi05_language_prefix_cache = None
                import warnings
                warnings.warn(
                    "pi05 device prefix assembly differs from torch.cat; "
                    "using the reference path for this model instance.",
                    RuntimeWarning,
                )
                assembled_embs = reference_cpu.to(packed_emb.device)
            else:
                self._pi05_prefix_assembly_ok = True
        embs = assembled_embs
    else:
        if packed_emb is not None:
            # The optimized packed image stays on RPU while the reference
            # language embedding is intentionally cached on CPU. Assembly
            # failures must still take a valid reference path.
            embs = torch.cat(
                [packed_emb.cpu(), lang_emb.cpu()], dim=1
            ).to(packed_emb.device)
        else:
            embs.append(lang_emb)
            embs = torch.cat(embs, dim=1)

    pad_masks.append(masks)
    total_prefix_len += lang_emb.shape[1]

    pad_masks = torch.cat(pad_masks, dim=1)
    bsize = pad_masks.shape[0]
    # All prefix-LM → att_masks 全 0；跳过 list 拼 + torch.tensor() 转换
    att_masks = torch.zeros(bsize, total_prefix_len, dtype=torch.bool,
                            device=pad_masks.device)
    return embs, pad_masks, att_masks


def _install_embed_prefix_class_patch() -> None:
    """Install patched embed_prefix at class level. Idempotent.

    Env opt-out: RPU_PI05_EMBED_PREFIX_PATCH=0 skips installation
    (install-time semantics — requires fresh process to flip).
    """
    if not rpu_env_bool("RPU_PI05_EMBED_PREFIX_PATCH", default=True):
        return  # opt out
    if getattr(_PI05Pytorch, "_rpu_patched_embed_prefix", False):
        return  # idempotent
    _PI05Pytorch.embed_prefix = _patched_embed_prefix
    _PI05Pytorch._rpu_patched_embed_prefix = True


def install_class_patches(policy) -> None:
    """Apply all Pi0.5 class-level patches.

    Called by Pi05Adapter.to_rpu() Step A. Idempotent across calls (each patch
    has its own sentinel guard).
    """
    _install_pi_gemma_rmsnorm_class_patch()
    _install_gemma_rotary_class_patch()
    _install_preprocess_images_class_patch(policy)
    _install_eval_fastpath_class_patch(policy)
    _install_embed_prefix_class_patch()

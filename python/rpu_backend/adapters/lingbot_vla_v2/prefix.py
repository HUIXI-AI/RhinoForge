"""LingBot-VLA V2 prefix composition (``embed_prefix``), implemented on CPU.

This module implements the model's prefix layout, special-token embedding,
position-id construction, query-segment spans, and suffix-to-future-video
blocking contract.

⚠️ IMPORT CONTRACT: pure torch, NO module-level `import rpu_backend`, no relative imports,
   so this file remains importable on an RPU-less host. Same rule as ``convert.py``.

Prefix token layout produced for the AUTHORITATIVE YAML (kuavo_v2_depth):

    [<vision_start> <image>*num_patch <vision_end>] * n_images
    ++ lang_tokens
    ++ <eos>*8    (current_depth align queries)
    ++ <eos>*8    (future_depth  align queries)

The `<eos>` ids are FAKE — they exist only so `get_rope_index` treats the align queries as text
(type 0 → sequential positions). Their EMBEDDINGS are overwritten with the align tokens, never
the eos embedding. Likewise `<image>` ids are placeholders whose embeddings are overwritten by
the ViT patch embeddings; `<vision_start>` / `<vision_end>` keep their real token embeddings
(special tokens use the language-token embedding table).
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
import torch.nn.functional as F

# HF Qwen3-VL `get_rope_index` modality encoding: text=0, image=1, video=2.
# vision_start/vision_end are text (type 0); only the
# `<image>` placeholder run is type 1, which also makes it the deepstack visual mask.
MM_TYPE_TEXT = 0
MM_TYPE_IMAGE = 1


@dataclass(frozen=True)
class SpecialTokenIds:
    """Resolved from the Qwen3-VL config (NOT hardcoded)."""
    vision_start: int
    vision_end: int
    image: int
    eos: int

    @staticmethod
    def from_config(cfg) -> "SpecialTokenIds":
        return SpecialTokenIds(
            vision_start=int(cfg.vision_start_token_id),
            vision_end=int(cfg.vision_end_token_id),
            image=int(cfg.image_token_id),
            eos=int(cfg.text_config.eos_token_id),
        )


@dataclass(frozen=True)
class AlignConfig:
    """Resolved ``align_params`` flags.

    Defaults match the supported ``kuavo_v2_depth`` training profile. Each field
    is annotated with the configuration key it mirrors.
    """
    use_depth_align: bool = True                      # align_params present + mode == "query"
    align_type: str = "query"                         # align_params.mode
    num_task_tokens: int = 8                          # align_params.num_task_tokens
    num_backbone_tokens: int = 256                    # align_params.depth.num_backbone_tokens
    use_future_depth: bool = True                     # align_params.depth.use_future_depth
    use_future_video: bool = True                     # align_params.use_future_video
    use_future_video_patch: bool = True               # align_params.video.use_patch_loss
    use_current_video_patch: bool = True              # align_params.video.use_current_patch_loss
    use_current_shared_task_proj: bool = True         # align_params.video.use_current_shared_task_proj
    use_future_video_cls: bool = False                # align_params.video.use_cls_loss
    future_video_share_future_depth_query: bool = True   # align_params.video.share_future_depth_query
    use_shared_future_task_proj: bool = True          # align_params.video.use_shared_future_task_proj
    block_future_depth_to_action: bool = True         # align_params.depth.block_future_depth_to_action
    block_suffix_to_future_video: bool = True         # align_params.video.block_suffix_to_future_video
    qwen3vl_use_vision_boundaries: bool = True        # config.qwen3vl_use_vision_boundaries
    vlm_causal: bool = True                           # config.vlm_causal

    def __post_init__(self):
        # Validate dependent alignment options.
        if self.use_depth_align and self.align_type != "query":
            raise ValueError(f"Only query depth alignment is supported, got {self.align_type!r}.")
        if self.num_task_tokens <= 0 or self.num_backbone_tokens % self.num_task_tokens != 0:
            raise ValueError(
                f"num_backbone_tokens ({self.num_backbone_tokens}) must be divisible by "
                f"num_task_tokens ({self.num_task_tokens}).")
        if self.use_current_video_patch and not self.use_future_video_patch:
            raise ValueError("video.use_current_patch_loss=True requires video.use_patch_loss=True.")
        if self.use_current_shared_task_proj and not self.use_current_video_patch:
            raise ValueError(
                "video.use_current_shared_task_proj=True requires video.use_current_patch_loss=True.")
        if self.use_shared_future_task_proj and not self.use_future_video_patch:
            raise ValueError(
                "video.use_shared_future_task_proj=True requires video.use_patch_loss=True.")
        if self.use_shared_future_task_proj and not self.future_video_share_future_depth_query:
            raise ValueError(
                "video.use_shared_future_task_proj=True requires video.share_future_depth_query=True.")
        if self.future_video_share_future_depth_query and not self.use_future_depth:
            raise ValueError(
                "video.share_future_depth_query=True requires depth.use_future_depth=True.")


# =============================================================================
# Segment order / spans (utils.py mirrors)
# =============================================================================
def prefix_query_segments(cfg: AlignConfig) -> tuple[str, ...]:
    """Return the prefix segment order after the image block.

    Current task queries precede future task queries; future_depth stays LAST so the
    suffix-to-future-depth tail blocking can keep using the tail span.

    For the authoritative YAML this evaluates to ("language", "current_depth", "future_depth"):
    future_video contributes NO own segment because share_future_depth_query=True.
    """
    segments = ["language"]
    if not cfg.use_depth_align:
        return tuple(segments)
    segments.append("current_depth")
    if cfg.use_future_video:
        if cfg.use_future_video_cls:
            segments.append("future_video_cls")
        if cfg.use_future_video_patch and not cfg.future_video_share_future_depth_query:
            segments.append("future_video")
    if cfg.use_future_depth:
        segments.append("future_depth")
    return tuple(segments)


def segment_token_counts(cfg: AlignConfig) -> dict[str, int]:
    return {"current_depth": cfg.num_task_tokens, "future_video_cls": 1,
            "future_video": cfg.num_task_tokens, "future_depth": cfg.num_task_tokens}


def prefix_query_token_spans(prefix_len: int, cfg: AlignConfig) -> dict[str, tuple[int, int]]:
    """Return ``[start, end)`` spans of the non-language query segments."""
    counts = segment_token_counts(cfg)
    ordered = prefix_query_segments(cfg)
    query_segments = [n for n in ordered if n != "language"]
    cursor = prefix_len - sum(counts[n] for n in query_segments)
    spans = {}
    for name in query_segments:
        spans[name] = (cursor, cursor + counts[name])
        cursor += counts[name]
    return spans


def num_query_tokens(cfg: AlignConfig) -> int:
    counts = segment_token_counts(cfg)
    return sum(counts[n] for n in prefix_query_segments(cfg) if n != "language")


def fv_col_span(prefix_len: int, num_task_tokens: int, use_cls: bool, use_patch: bool):
    """Return the tail query block used for future-depth blocking."""
    fv_len = (1 if use_cls else 0) + (num_task_tokens if use_patch else 0)
    return prefix_len - fv_len, prefix_len


def future_video_query_span(prefix_len: int, cfg: AlignConfig) -> tuple[int, int]:
    """Return the future-video query span.

    With share_future_depth_query=True the future-video segment owns NO tokens ⇒ start == end
    ⇒ _block_suffix_to_future_video_ is a NO-OP for the authoritative YAML.
    """
    if not cfg.use_future_video:
        return prefix_len, prefix_len
    future_depth_count = cfg.num_task_tokens if cfg.use_future_depth else 0
    own = 1 if cfg.use_future_video_cls else 0
    if cfg.use_future_video_patch and not cfg.future_video_share_future_depth_query:
        own += cfg.num_task_tokens
    end = prefix_len - future_depth_count
    return end - own, end


# =============================================================================
# Align query tokens
# =============================================================================
def get_align_tokens(tokens: torch.Tensor, num_task_tokens: int) -> torch.Tensor:
    """Reduce ``[num_backbone_tokens, D]`` to ``[num_task_tokens, D]``.

    ORDER IS LOAD-BEARING: `view(num_task_tokens, N // num_task_tokens, D).mean(dim=1)`, i.e.
    task token i = mean of backbone rows [i*stride, (i+1)*stride). NOT view(N//ntt, ntt, D).
    """
    n, d = tokens.shape
    if n % num_task_tokens != 0:
        raise ValueError(f"get_align_tokens: {n} backbone tokens not divisible by {num_task_tokens}")
    return tokens.view(num_task_tokens, n // num_task_tokens, d).mean(dim=1)


def compute_align_tokens(align_W: dict, cfg: AlignConfig) -> dict[str, torch.Tensor]:
    """Build ``current_task`` and ``future_task`` query embeddings in CPU FP32.

    current_task = shared_task_proj(cat[ align(depth_align_embs), align(current_video_align_embs) ])
    future_task  = shared_task_proj(cat[ align(future_depth_align_embs), align(future_video_align_embs) ])
    Both projections are gated by the YAML flags exactly as the official does.
    """
    out: dict[str, torch.Tensor] = {}
    if not cfg.use_depth_align:
        return out

    current_task = get_align_tokens(align_W["model.depth_align_embs"], cfg.num_task_tokens)
    if cfg.use_future_video and cfg.use_current_video_patch and cfg.use_current_shared_task_proj:
        current_video_task = get_align_tokens(
            align_W["model.current_video_align_embs"], cfg.num_task_tokens)
        current_task = F.linear(
            torch.cat([current_task, current_video_task], dim=-1),
            align_W["model.current_shared_task_proj.weight"],
            align_W["model.current_shared_task_proj.bias"])
    out["current_depth"] = current_task

    if cfg.use_future_depth:
        future_task = get_align_tokens(align_W["model.future_depth_align_embs"], cfg.num_task_tokens)
        if (cfg.use_future_video and cfg.use_future_video_patch
                and cfg.future_video_share_future_depth_query and cfg.use_shared_future_task_proj):
            future_video_task = get_align_tokens(
                align_W["model.future_video_align_embs"], cfg.num_task_tokens)
            future_task = F.linear(
                torch.cat([future_task, future_video_task], dim=-1),
                align_W["model.future_shared_task_proj.weight"],
                align_W["model.future_shared_task_proj.bias"])
        out["future_depth"] = future_task

    # Reject an invalid shared-query configuration rather than dropping it.
    if (not cfg.use_future_depth and cfg.use_future_video
            and cfg.future_video_share_future_depth_query):
        raise ValueError("share_future_depth_query=True requires depth.use_future_depth=True.")

    if "future_video" in prefix_query_segments(cfg):
        out["future_video"] = get_align_tokens(
            align_W["model.future_video_align_embs"], cfg.num_task_tokens)
    if "future_video_cls" in prefix_query_segments(cfg):
        out["future_video_cls"] = align_W["model.future_video_cls_align_emb.weight"]
    return out


# =============================================================================
# Prefix ID layout
# =============================================================================
@dataclass(frozen=True)
class PrefixLayout:
    input_ids: torch.Tensor          # [1, S] long
    mm_token_type_ids: torch.Tensor  # [1, S] long — 1 ONLY on <image> placeholders
    pad_masks: torch.Tensor          # [1, S] bool
    visual_pos_masks: torch.Tensor   # [1, S] bool — deepstack scatter target (== mm type 1)
    spans: dict                      # segment -> (start, end)
    S: int
    image_token_len: int             # per image, incl. vision_start/end when enabled


@dataclass(frozen=True)
class PrefixExecutionLayout:
    """A REAL prefix plus adapter-owned tail padding for one execution bucket.

    Padding is appended *after* every real image/language/align token.  This is
    load-bearing: inserting padding inside the language span would let the later
    align-query rows attend those pads and would no longer match the official
    unpadded prefix.  ``pad_masks`` marks only the real rows; the action suffix
    must use the same mask when it attends the prefix.
    """

    real: PrefixLayout
    input_ids: torch.Tensor
    mm_token_type_ids: torch.Tensor
    pad_masks: torch.Tensor
    visual_pos_masks: torch.Tensor
    execution_len: int

    @property
    def real_len(self) -> int:
        return int(self.real.S)


@dataclass(frozen=True)
class PrefixExecutionPlan:
    """Finite adapter-level REAL-prefix to execution-shape mapping."""

    real_len: int
    execution_len: int
    chunk_size: int

    @property
    def key(self) -> tuple[int, int]:
        return self.execution_len, self.chunk_size


def plan_prefix_execution(
    real_len: int,
    *,
    max_seq: int,
    suffix_len: int,
    chunk_size: int,
    padding_multiple: int = 1,
    fixed_execution_len: int | None = None,
) -> PrefixExecutionPlan:
    """Return one bounded execution profile without changing REAL semantics.

    This mirrors Wall-OSS' adapter-level planning split: the REAL processor
    length remains semantic, while the execution length is a finite Graph shape.
    Generic LingBot2 may align the tail; production Z2 passes
    ``fixed_execution_len=225`` and therefore rejects, rather than truncates, a
    REAL prefix that does not fit that sealed physical profile.

    ``chunk_size`` is the already-selected per-instance decoder policy.  This
    helper does not second-guess the generic C++ exact planner when it is zero.
    It is carried in the execution key so changing that policy cannot collide
    with a prepared Graph envelope.
    """

    values = {
        "real_len": real_len,
        "max_seq": max_seq,
        "suffix_len": suffix_len,
        "chunk_size": chunk_size,
        "padding_multiple": padding_multiple,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    real_len = int(real_len)
    max_seq = int(max_seq)
    suffix_len = int(suffix_len)
    chunk_size = int(chunk_size)
    padding_multiple = int(padding_multiple)
    if real_len <= 0:
        raise ValueError(f"real_len must be positive, got {real_len}")
    if max_seq <= 0 or suffix_len <= 0:
        raise ValueError(
            f"max_seq and suffix_len must be positive, got {max_seq}/{suffix_len}"
        )
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be >= 0, got {chunk_size}")
    if padding_multiple <= 0:
        raise ValueError(
            f"padding_multiple must be positive, got {padding_multiple}"
        )

    if fixed_execution_len is not None:
        if isinstance(fixed_execution_len, bool) or not isinstance(
            fixed_execution_len, Integral
        ):
            raise ValueError(
                "fixed_execution_len must be an integer or None, got "
                f"{fixed_execution_len!r}"
            )
        execution_len = int(fixed_execution_len)
        if execution_len < real_len:
            raise ValueError(
                "REAL prefix exceeds the fixed execution profile: "
                f"real_len={real_len}, execution_len={execution_len}"
            )
    else:
        execution_len = (
            (real_len + padding_multiple - 1) // padding_multiple
        ) * padding_multiple

    if execution_len + suffix_len > max_seq:
        raise ValueError(
            "prefix execution plus suffix exceeds KV capacity: "
            f"execution_len={execution_len}, suffix_len={suffix_len}, "
            f"max_seq={max_seq}"
        )
    return PrefixExecutionPlan(real_len, execution_len, chunk_size)


def pad_prefix_layout_for_execution(
    layout: PrefixLayout,
    execution_len: int,
    *,
    pad_token_id: int,
) -> PrefixExecutionLayout:
    """Append masked token rows to ``layout`` while preserving all REAL spans."""

    if isinstance(execution_len, bool) or not isinstance(execution_len, Integral):
        raise ValueError(
            f"execution_len must be an integer, got {execution_len!r}"
        )
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, Integral):
        raise ValueError(f"pad_token_id must be an integer, got {pad_token_id!r}")
    execution_len = int(execution_len)
    pad_rows = execution_len - int(layout.S)
    if pad_rows < 0:
        raise ValueError(
            f"execution_len={execution_len} is smaller than real_len={layout.S}"
        )
    if pad_rows == 0:
        return PrefixExecutionLayout(
            real=layout,
            input_ids=layout.input_ids,
            mm_token_type_ids=layout.mm_token_type_ids,
            pad_masks=layout.pad_masks,
            visual_pos_masks=layout.visual_pos_masks,
            execution_len=execution_len,
        )

    token_pad = torch.full(
        (1, pad_rows), int(pad_token_id), dtype=layout.input_ids.dtype
    )
    text_type_pad = torch.full(
        (1, pad_rows), MM_TYPE_TEXT, dtype=layout.mm_token_type_ids.dtype
    )
    false_pad = torch.zeros((1, pad_rows), dtype=torch.bool)
    return PrefixExecutionLayout(
        real=layout,
        input_ids=torch.cat((layout.input_ids, token_pad), dim=1),
        mm_token_type_ids=torch.cat(
            (layout.mm_token_type_ids, text_type_pad), dim=1
        ),
        pad_masks=torch.cat((layout.pad_masks, false_pad), dim=1),
        visual_pos_masks=torch.cat(
            (layout.visual_pos_masks, false_pad.clone()), dim=1
        ),
        execution_len=execution_len,
    )


def build_prefix_layout(n_images: int, num_patch: int, lang_ids: torch.Tensor,
                        cfg: AlignConfig, ids: SpecialTokenIds) -> PrefixLayout:
    """Build the prefix ``input_ids`` and masks used by ``embed_prefix``.

    `n_images` is the count of VALID images (img_masks already applied), `num_patch` the
    post-merger token count per image, `lang_ids` the [1, L] masked language tokens.
    """
    if lang_ids.dim() != 2 or lang_ids.shape[0] != 1:
        raise ValueError(f"build_prefix_layout: lang_ids must be [1, L], got {tuple(lang_ids.shape)}")
    L = lang_ids.shape[1]

    # ---- image block: <vision_start> <image>*num_patch <vision_end> ----
    if cfg.qwen3vl_use_vision_boundaries:
        image_token_len = num_patch + 2
        one = torch.full((image_token_len,), ids.image, dtype=torch.long)
        one[0] = ids.vision_start
        one[-1] = ids.vision_end
        vis_one = torch.zeros(image_token_len, dtype=torch.bool)
        vis_one[1:1 + num_patch] = True                    # image_visual_masks[:, :, 1:1+num_patch]
    else:
        image_token_len = num_patch
        one = torch.full((image_token_len,), ids.image, dtype=torch.long)
        vis_one = torch.ones(image_token_len, dtype=torch.bool)

    img_ids = one.repeat(n_images).view(1, -1)
    img_vis = vis_one.repeat(n_images).view(1, -1)

    parts_ids = [img_ids]
    parts_vis = [img_vis]
    spans: dict[str, tuple[int, int]] = {}
    cursor = n_images * image_token_len
    spans["image"] = (0, cursor)

    # ---- segments after the image block in model order ----
    n_task = cfg.num_task_tokens
    for seg in prefix_query_segments(cfg):
        if seg == "language":
            parts_ids.append(lang_ids.to(torch.long))
            parts_vis.append(torch.zeros(1, L, dtype=torch.bool))
            spans["language"] = (cursor, cursor + L)
            cursor += L
        else:
            count = 1 if seg == "future_video_cls" else n_task
            # Fake IDs use text_config.eos_token_id and serve only as placeholders.
            parts_ids.append(torch.full((1, count), ids.eos, dtype=torch.long))
            parts_vis.append(torch.zeros(1, count, dtype=torch.bool))
            spans[seg] = (cursor, cursor + count)
            cursor += count

    input_ids = torch.cat(parts_ids, dim=1)
    visual_pos_masks = torch.cat(parts_vis, dim=1)
    S = input_ids.shape[1]
    if cursor != S:
        raise RuntimeError(f"prefix cursor {cursor} != S {S}")

    # mm_token_type_ids: 1 ONLY on the <image> placeholder runs (vision_start/end are TEXT).
    mm = torch.where(visual_pos_masks, MM_TYPE_IMAGE, MM_TYPE_TEXT).to(torch.long)
    # pad_masks: all-ones — img_masks/lang_masks are applied by DROPPING tokens upstream, so the
    # assembled prefix has no padding. (The official carries padding through; our obs contract
    # filters first, which is equivalent for batch_size 1.)
    pad_masks = torch.ones(1, S, dtype=torch.bool)

    # Cross-check against the independent span calculator.
    ref_spans = prefix_query_token_spans(S, cfg)
    for name, span in ref_spans.items():
        if spans[name] != span:
            raise AssertionError(
                f"prefix layout span mismatch for {name!r}: built {spans[name]} != "
                f"prefix_query_token_spans {span}")
    return PrefixLayout(input_ids=input_ids, mm_token_type_ids=mm, pad_masks=pad_masks,
                        visual_pos_masks=visual_pos_masks, spans=spans, S=S,
                        image_token_len=image_token_len)


def suffix_position_start(prefix_position_ids: torch.Tensor, pad_masks: torch.Tensor) -> int:
    """``p0`` — the suffix's RoPE position base.

        valid_prefix_pos = prefix_position_ids.masked_fill(~prefix_pad_masks.unsqueeze(0), 0)
        prefix_offsets   = valid_prefix_pos.amax(dim=(0, 2)) + 1
        suffix_1d        = prefix_offsets[:, None] + cumsum(suffix_pad_masks) - 1

    With an unpadded prefix and an all-ones suffix pad mask, suffix position k = p0 + k where
    p0 = max(prefix positions) + 1. This is NOT S whenever the prefix holds images — M-RoPE
    advances only max(h, w) // spatial_merge_size per image, not one per visual token.
    """
    valid = prefix_position_ids.masked_fill(~pad_masks.unsqueeze(0), 0)
    return int(valid.amax(dim=(0, 2)).item()) + 1


def apply_suffix_prefix_blocking_(prefix_2d: torch.Tensor, prefix_len: int, cfg: AlignConfig):
    """Mask the configured suffix→prefix edges.

    `prefix_2d` is bool [suffix_len, prefix_len], True == visible. Modified IN PLACE.

      1. block_future_depth_to_action  -> block_suffix_to_fv_(..., num_task_tokens) zeroes the
         TAIL [prefix_len - 8, prefix_len) block == the future_depth queries (utils.py
         fv_col_span with use_cls=False, use_patch=True).
      2. block_suffix_to_future_video  -> _block_suffix_to_future_video_ zeroes the future-video
         OWN span, which is EMPTY under share_future_depth_query=True ⇒ no-op here. Implemented
         anyway so a config that turns sharing off stays correct.
    """
    if cfg.block_future_depth_to_action and cfg.use_future_depth:
        s, e = fv_col_span(prefix_len, cfg.num_task_tokens,
                           use_cls=False, use_patch=True)
        if e > s:
            prefix_2d[:, s:e] = False
    if cfg.block_suffix_to_future_video:
        s, e = future_video_query_span(prefix_len, cfg)
        if e > s:
            prefix_2d[:, s:e] = False
    return prefix_2d

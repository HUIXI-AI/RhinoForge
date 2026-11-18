"""CPU position inputs shared by spatial-merge vision encoders."""
from collections.abc import Mapping


def canonical_vision_type(config, legacy_type):
    """Recognize HF's vision-only config rename without widening RoPE semantics."""
    def value(name, default=None):
        return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)

    name = value("model_type")
    rope = value("rope_parameters")
    if name not in (legacy_type, legacy_type + "_vision"):
        return None
    if rope is not None and (not isinstance(rope, Mapping)
                            or dict(rope) != {"rope_type": "axial", "rope_theta": 10000.0}):
        return None
    head_dim = value("head_dim")
    if head_dim is not None:
        hidden, heads = value("hidden_size"), value("num_heads")
        if type(head_dim) is not int or not heads or head_dim != hidden // heads:
            return None
    return legacy_type


def interpolate_vision_positions(model, grid_thw):
    """Use the installed HF interpolation contract and return the embedding dtype."""
    legacy = getattr(type(model), "fast_pos_embed_interpolate", None)
    if legacy is not None:
        return legacy(model, grid_thw)

    from transformers.vision_utils import get_vision_interpolation_indices_and_weights

    indices, weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=model.num_grid_per_side,
        mode=model.interpolation_mode,
        align_corners=model.interpolation_align_corners,
        spatial_merge_size=model.config.spatial_merge_size,
    )
    # New HF sums in FP32, then casts when adding to the patch embedding.
    return (model.pos_embed(indices) * weights[:, :, None]).sum(1).to(model.pos_embed.weight.dtype)

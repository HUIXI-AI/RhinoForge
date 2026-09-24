"""G0.5 RPU adapter facade."""
from .runtime import (
    _g05_policy_runtime_ready,
    patch_g05_action_expert_for_rpu,
    patch_g05_data_processor_for_rpu,
    patch_g05_inferencer_for_rpu,
    patch_g05_policy_for_rpu,
    patch_g05_vision_for_rpu,
    patch_g05_vlm_for_rpu,
)

__all__ = [
    "patch_g05_action_expert_for_rpu",
    "patch_g05_data_processor_for_rpu",
    "patch_g05_inferencer_for_rpu",
    "patch_g05_policy_for_rpu",
    "patch_g05_vision_for_rpu",
    "patch_g05_vlm_for_rpu",
]

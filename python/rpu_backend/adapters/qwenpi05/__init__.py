"""QwenPI05 (RhinoVLA) action-expert RPU adapter.

Two RPU execution paths (both consume a Qwen3-VL prefix KV computed in PyTorch):
  - op-by-op (``convert.py``): each Linear/RMSNorm/SDPA/RoPE dispatches independently
    to RPU kernels. Entry: ``convert_qwenpi05_action_model_for_rpu`` →
    ``run_qwenpi05_action_on_rpu`` / ``predict_action_rpu`` (10-step Euler).
  - fused (``fused.py``): patches ``expert.forward`` to route through the C++
    ``qwenpi05_forward`` all-layers-once model. Entry: ``patch_qwenpi05_for_rpu_fused``.
"""
from rpu_backend.adapters.qwenpi05.convert import (
    convert_qwenpi05_action_model_for_rpu,
    run_qwenpi05_action_on_rpu,
    predict_action_rpu,
)
from rpu_backend.adapters.qwenpi05.fused import patch_qwenpi05_for_rpu_fused

__all__ = [
    "convert_qwenpi05_action_model_for_rpu",
    "run_qwenpi05_action_on_rpu",
    "predict_action_rpu",
    "patch_qwenpi05_for_rpu_fused",
]

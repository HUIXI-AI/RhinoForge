"""GR00T-N1.7-3B 的独立 RPU VLA adapter。

组合 Qwen3-VL backbone、视觉语言 encoder 与图内 denoise action expert。"""
from rpu_backend.adapters.gr00t.runtime import Gr00tN1d7VLA, build_gr00t_vla

__all__ = ["Gr00tN1d7VLA", "build_gr00t_vla"]

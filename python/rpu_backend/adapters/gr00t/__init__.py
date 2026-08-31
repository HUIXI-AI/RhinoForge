"""GR00T-N1.7 DROID VLA RPU adapter (standalone runtime).

Base-zero-shot and finetuned DROID are separate exact Source-only profiles.
"""
from rpu_backend.adapters.gr00t.runtime import Gr00tN1d7VLA, build_gr00t_vla

__all__ = ["Gr00tN1d7VLA", "build_gr00t_vla"]

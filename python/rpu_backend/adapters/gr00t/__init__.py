"""GR00T-N1.7-3B VLA RPU adapter (standalone runtime — gr00t package not importable in RPU env).

Full image→action execution on RPU.
"""
from rpu_backend.adapters.gr00t.runtime import Gr00tN1d7VLA, build_gr00t_vla

__all__ = ["Gr00tN1d7VLA", "build_gr00t_vla"]

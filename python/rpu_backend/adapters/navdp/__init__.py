"""Source-only InternVLA-N1 NavDP RPU evaluator.

NavDP is embedded in the public ``InternRobotics/InternVLA-N1-w-NavDP``
checkpoint rather than exposed as a separate HF AutoModel.

`build_navdp` reads the 16-layer TransformerDecoder weights directly from the
official sharded checkpoint, creates the fused handle, and pushes them through
`torch.ops.rpu.navdp_set_weights`. The fused C++ path is numerically implemented;
standalone evidence remains component-scoped, while the exact full-chain profile
gates final action, performance, Graph lifecycle, and kernel census separately.
"""

from __future__ import annotations

from rpu_backend.adapters.navdp.runtime import NavdpRuntime, build_navdp

__all__ = ["NavdpRuntime", "build_navdp"]

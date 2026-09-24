"""Controlled InternVLA-N1 NavDP RPU evaluator (standalone builder).

NavDP is a standalone diffusion-policy checkpoint (not an HF AutoModel), so it
follows the GR00T standalone-build pattern, NOT the `register_adapter` HF path.

`build_navdp` loads the 16-layer TransformerDecoder weights from the extracted
`navdp_head.safetensors`, creates the fused handle, and pushes them through
`torch.ops.rpu.navdp_set_weights`. The fused C++ path is numerically implemented;
standalone evidence remains component-scoped, while the exact full-chain profile
gates final action, performance, Graph lifecycle, and kernel census separately.
"""

from __future__ import annotations

from rpu_backend.adapters.navdp.runtime import NavdpRuntime, build_navdp

__all__ = ["NavdpRuntime", "build_navdp"]

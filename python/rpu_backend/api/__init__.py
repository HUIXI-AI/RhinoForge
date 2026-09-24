"""v5.0 secondary public surface — `rpu_backend.api`.

Per ADR §2.2 + REQ SKEL-01a (v5-01a-1) + REQ SKEL-01b (v5-01a-2). End-state exports:
  - RPUCache (owning module: api.cache; CUT-06 from_model factory)
  - Pi05Policy (owning module: api.policy; SKEL-03 + ADR §10 #16; lazy-imports
    loader/adapter inside method bodies per ADR §6.2)
  - PaliGemmaPolicy (owning module: api.paligemma; unsupported compatibility
    tombstone that fails before adapter/model loading)
  - RhinoVLAPolicy (owning module: api.rhinovla; wraps an explicit external
    runtime factory without importing the model repository at package import)
  - API_VERSION (v5 public-contract level; distinct from wheel release version)
  - 8 error classes (owning module: api.errors; SKEL-04)

Pi05Policy was added in v5-01a-2 (this commit). Top-level
``from rpu_backend import Pi05Policy`` STILL raises ImportError — the
canonical surface is ``from rpu_backend.api import Pi05Policy`` (CUT-04
from-import gate; SKEL-05b: top-level __init__.py 0 hits on Pi05Policy).

Explicit per-class form (D-03 / D-04 / codex G2-A): NOT ``from .errors
import *``. Explicit names keep the public surface auditable + grep-friendly.
"""
from .cache import RPUCache
from .qwen3_5_cache import Qwen3_5Cache
from .qwen3_5_moe_cache import Qwen3_5MoeCache
from .policy import Pi05Policy
from .paligemma import PaliGemmaPolicy
from .rhinovla import RhinoVLAPolicy
from .wall_oss import WallOssActionOutput, WallOssPolicy
from .lingbot2 import Lingbot2ActionOutput, Lingbot2Policy
from .hy_embodied import HyEmbodiedActionOutput, HyEmbodiedPolicy
from .conditional_generation import RPUModelForConditionalGeneration
from .generation import greedy_token_ids
from ._execution import reconfigure_rpu_execution, rpu_execution_stats
from rpu_backend._version import API_VERSION

# ⚠️ `hy_action_decode`（HyVlaActionDecoder）**故意不在这里导入**：它在模块级
# `import scipy`（vendor 的 6D 旋转↔四元数全用 scipy），放进来就会让
# `import rpu_backend` 硬依赖 scipy。需要它的人显式
# `from rpu_backend.api.hy_action_decode import HyVlaActionDecoder`；
# `HyEmbodiedPolicy` 也只在 `.to('rpu')` 里按需惰性导入。
from .errors import (
    RPUBackendError,
    RPUConfigError,
    PlannerRejectError,
    UnsupportedModelError,
    RPUUnsupportedDtypeError,
    RPUSingleHandleError,
    SPMExhaustionError,
    WeightShapeMismatchError,
)

__all__ = [
    "API_VERSION",
    "RPUCache",
    "Qwen3_5Cache",
    "Qwen3_5MoeCache",
    "Pi05Policy",
    "PaliGemmaPolicy",
    "RhinoVLAPolicy",
    "WallOssActionOutput",
    "WallOssPolicy",
    "Lingbot2ActionOutput",
    "Lingbot2Policy",
    "HyEmbodiedActionOutput",
    "HyEmbodiedPolicy",
    "RPUModelForConditionalGeneration",
    "greedy_token_ids",
    "reconfigure_rpu_execution",
    "rpu_execution_stats",
    "RPUBackendError",
    "RPUConfigError",
    "PlannerRejectError",
    "UnsupportedModelError",
    "RPUUnsupportedDtypeError",
    "RPUSingleHandleError",
    "SPMExhaustionError",
    "WeightShapeMismatchError",
]

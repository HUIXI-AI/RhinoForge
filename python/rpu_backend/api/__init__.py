"""Secondary public API for caches, model loaders, policies, and errors.

Policy classes live under ``rpu_backend.api`` rather than the four-name
top-level package. Explicit imports keep this public surface auditable.
"""
from .cache import RPUCache
from .qwen3_5_cache import Qwen3_5Cache
from .policy import Pi05Policy
from .rhinovla import RhinoVLAPolicy
from .wall_oss import WallOssActionOutput, WallOssPolicy
from .lingbot2 import Lingbot2ActionOutput, Lingbot2Policy
from .hy_embodied import HyEmbodiedActionOutput, HyEmbodiedPolicy
from .conditional_generation import RPUModelForConditionalGeneration
from rpu_backend._version import API_VERSION

# ⚠️ `hy_action_decode`（HyVlaActionDecoder）**故意不在这里导入**：它在模块级
# `import scipy`（vendor 的 6D 旋转↔四元数全用 scipy），放进来就会让
# `import rpu_backend` 硬依赖 scipy。需要它的人显式
# `from rpu_backend.api.hy_action_decode import HyVlaActionDecoder`；
# `HyEmbodiedPolicy` 也只在 `.to('rpu')` 里按需惰性导入。
from .errors import (
    RPUBackendError,
    RPUConfigError,
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
    "Pi05Policy",
    "RhinoVLAPolicy",
    "WallOssActionOutput",
    "WallOssPolicy",
    "Lingbot2ActionOutput",
    "Lingbot2Policy",
    "HyEmbodiedActionOutput",
    "HyEmbodiedPolicy",
    "RPUModelForConditionalGeneration",
    "RPUBackendError",
    "RPUConfigError",
    "UnsupportedModelError",
    "RPUUnsupportedDtypeError",
    "RPUSingleHandleError",
    "SPMExhaustionError",
    "WeightShapeMismatchError",
]

"""Hy-Embodied-0.5-VLA 的公共 Policy 接口。

规范用法：

    from rpu_backend.api import HyEmbodiedPolicy

    policy = HyEmbodiedPolicy.from_checkpoint(
        "<MODEL_ROOT>/Hy-Embodied-0.5-VLA-UMI",
        dtype="w8a16-expert-vlmv",      # 推荐档
        norm_stats_path="<...>/norm_stats.pkl",   # **必须显式给**，否则没有物理动作
        trust_norm_stats_pickle=True,
        norm_stats_sha256="<64-digit SHA-256>",
    ).to("rpu")

    out = policy.infer(
        images=[top_head, hand_left, hand_right],   # 三相机，顺序见 camera_names
        instruction="pick up the bottle",
        ee_pose=current_dual_arm_ee_pose_16_wxyz,
    )
    out.actions              # (49, 16) 双臂位姿（配置的坐标约定，未认证真机）
    out.actions_normalized   # [50, 20] 模型原始归一化输出

形状选择的依据（都来自这个 ckpt 的 `config.json` 与既有 RPU 实现）：

* 没有离线 prepare 产物、也没有预量化 ckpt：三档精度都从**同一份 fp32/bf16
  权重**在装载时量化，所以入口是 `from_checkpoint(...)`，`.to('rpu')` 触发构建
  （包括权重 layout 转换与 RPU DDR 安装）。
* 模型每步吐 **32** 个 action 维，**只有前 20 维是真的**；[20:32] 从未对任何
  target 训练过。`actions_normalized` 是裁好的 `[50, 20]`；`actions_full` 保留
  未裁的网格，只应该用于调试。
* 10 步 Euler 固化在图里，**不是**每次调用可传的参数。
* `prefix_len`（S）在构造时固定 —— 它决定 SPM 布局与图签名，换值要重建。

⚠️ **单实例单进程**：RPU 一次只服务一个 handle 集合。Hy-VLA 的原生
env 开关会在首次物化时固化，所以 `.to("rpu")` 一旦开始装载权重，该 Python
进程就不能再装第二个 RPU 模型/policy。`close()` 会回收 handle、图和本 policy
改过的全局设置，但不会使进程可复用；换模型必须重启进程。并发服务时，
使用"模型常驻 + 请求串行"的结构。
"""
from __future__ import annotations

import dataclasses
import hashlib
import operator
import os
import pickle
import re
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from rpu_backend.api._runtime_env import (
    LKN_CAPACITY_ENV,
    normalize_runtime_env,
)

# 模型吐 32 维，只有前 20 维是真的（双臂各 [平移3 + 6D旋转6 + 夹爪1]）。
VALID_ACTION_DIMS = 20

# 模型**归一化网格**的步数（ckpt `config.json` 的 `n_action_steps`）。解码时第 0 行
# 是"当前位姿"这一行，由 `hy_action_decode.drop_first` 丢弃，所以物理轨迹
# 只有 `MODEL_ACTION_STEPS - 1` 步。两者别混用：`actions_normalized` 是 50 行，
# `actions` 是 49 行。
MODEL_ACTION_STEPS = 50

# 公共 dtype profile → `RPU_HY_VLA_W8A16` 的逐塔选择。
_PROFILE_ENV: dict[str, str] = {
    "fp16":              "0",                  # 三座全 fp16（W16A16）
    "w8a16":             "all",
    "w8a16-expert":      "expert",
    "w8a16-expert-vlmv": "expert,vlm_vision",  # ⭐ 推荐：速度/精度的拐点
    "w8a16-vlm":         "vlm",
    "w8a16-vit":         "vit",
    "w8a16-no-vlm":      "vit,expert",         # 已被 expert-vlmv 严格支配，勿新用
}

_RUNTIME_ENV_ALLOWLIST = frozenset({
    "RPU_ALLREDUCE_CHUNK_V2",
    "RPU_ALLREDUCE_FUSED",
    "RPU_ALLREDUCE_PARTIAL_TWOSTAGE",
    "RPU_ALLREDUCE_TWOSTAGE",
    "RPU_ALLREDUCE_V2_CAP",
    "RPU_FASTREPLAY_SKIP_SYNC",
    "RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN",
    "RPU_HY_VLA_ACTION_MLP_MC",
    "RPU_HY_VLA_ATTN_TP8",
    "RPU_HY_VLA_CACHING_ALLOC",
    "RPU_HY_VLA_DENOISE_UNROLL",
    "RPU_HY_VLA_FAST_REPLAY",
    "RPU_HY_VLA_FAST_REPLAY_PRELOAD",
    "RPU_HY_VLA_FUSED_MERGER",
    "RPU_HY_VLA_KVPAD16",
    "RPU_HY_VLA_MASK_ONCE",
    "RPU_HY_VLA_MERGER_IN_GRAPH",
    "RPU_HY_VLA_MOT_NORM_NOMERGE",
    "RPU_HY_VLA_PARTIAL_ROPE",
    "RPU_HY_VLA_PATCH_EMBED_IN_GRAPH",
    "RPU_HY_VLA_PATCH_EMBED_MC",
    "RPU_HY_VLA_PERSIST_HANDLES",
    "RPU_HY_VLA_PREFIX_TEMPLATE",
    "RPU_HY_VLA_PROJ1_IN_MERGER",
    "RPU_HY_VLA_Q_INPLACE",
    "RPU_HY_VLA_RMSNORM_PAD16",
    "RPU_HY_VLA_SILU_MUL",
    "RPU_HY_VLA_VIT_PACKED",
    "RPU_HY_VLA_W4A16",
    "RPU_HY_VLA_W8A16",
    "RPU_KVINSERT_HYBRID_V16",
    "RPU_KVINSERT_V16_ANY_TP",
    "RPU_LINEAR_TILE_GY1",
    "RPU_LINEAR_TILE_M384W",
    "RPU_LINEAR_TILE_M64",
    "RPU_LINEAR_TILE_W8_M64",
    "RPU_LINEAR_TILE_W8_M64N",
    "RPU_RMSNORM_VWARP",
    "RPU_SKIP_IDLE_RECORD_FUNCTION",
})


def _load_trusted_norm_stats_pickle(
    path: str | os.PathLike[str],
    *,
    trust_norm_stats_pickle: bool,
    norm_stats_sha256: str | None,
) -> Mapping[str, Any]:
    """Verify the exact pickle bytes before deserializing those same bytes."""
    if not isinstance(trust_norm_stats_pickle, bool):
        raise TypeError("trust_norm_stats_pickle must be bool")
    if not trust_norm_stats_pickle:
        raise PermissionError(
            "norm_stats.pkl uses executable pickle deserialization; pass "
            "trust_norm_stats_pickle=True only for a reviewed asset."
        )
    if (
        not isinstance(norm_stats_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", norm_stats_sha256) is None
    ):
        raise ValueError("norm_stats_sha256 must be an exact 64-digit SHA-256 hex digest")

    import pathlib

    payload = pathlib.Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    expected = norm_stats_sha256.lower()
    if actual != expected:
        raise ValueError(
            f"norm_stats.pkl SHA-256 mismatch: expected {expected}, got {actual}"
        )
    value = pickle.loads(payload)
    if not isinstance(value, Mapping):
        raise TypeError(
            "norm_stats.pkl must deserialize to a mapping, got "
            f"{type(value).__name__}"
        )
    return value


def _remember_environment(
    snapshot: dict[str, str | None], key: str
) -> None:
    """Remember one policy-owned env key before its first mutation."""
    if key not in snapshot:
        snapshot[key] = os.environ.get(key)


def _restore_environment(snapshot: Mapping[str, str | None] | None) -> None:
    if not snapshot:
        return
    for key, old_value in snapshot.items():
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def _finalize_runner(
    runner: Any,
    env_snapshot: Mapping[str, str | None],
    previous_threads: int,
) -> bool:
    """Best-effort owner cleanup without retaining the policy itself."""
    cleanup_ok = True
    try:
        runner.close()
    except BaseException:
        cleanup_ok = False
    try:
        _restore_environment(env_snapshot)
    except BaseException:
        cleanup_ok = False
    try:
        torch.set_num_threads(previous_threads)
    except BaseException:
        cleanup_ok = False
    return cleanup_ok


def _check_prefix_envelope(n_valid: Any, prefix_len: int) -> int:
    if isinstance(n_valid, bool):
        raise ValueError("obs['n_valid'] 必须是正整数。")
    try:
        value = operator.index(n_valid)
    except TypeError as exc:
        raise ValueError("obs['n_valid'] 必须是正整数。") from exc
    if value <= 0:
        raise ValueError("obs['n_valid'] 必须是正整数。")
    needed = _ceil16(value)
    if needed > prefix_len:
        raise ValueError(
            f"这条 observation 有 {value} 个有效 prefix 行 ⇒ 需要 "
            f"prefix_len ≥ {needed}，但 policy 建的是 {prefix_len}。"
            "请用更大的 prefix_len 重建（指令太长）。")
    return value


def _validate_runner_observation(
    obs: Mapping[str, Any], *, n_cameras: int, prefix_len: int
) -> dict[str, Any]:
    """Validate the fixed Hy-VLA runner envelope before any device call."""
    values = dict(obs)
    if "n_valid" not in values:
        raise ValueError(
            "obs= 必须带 n_valid；它用于防止有效 prefix 超过 "
            "policy.prefix_len 后被静默截断。")
    claimed_valid = _check_prefix_envelope(values.pop("n_valid"), prefix_len)

    required = (
        "images", "lang_tokens", "prefill_mask", "prefill_pos",
        "modality_mask", "denoise_mask", "suffix_pos", "noise",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"obs= 缺少 runner 入参: {missing}。")

    images = values["images"]
    if not isinstance(images, Sequence) or len(images) != n_cameras:
        raise ValueError(f"obs['images'] 必须是 {n_cameras} 张图。")
    for index, image in enumerate(images):
        if not torch.is_tensor(image) or tuple(image.shape) != (1, 3, 224, 224):
            raise ValueError(
                f"obs['images'][{index}] 必须是 tensor [1,3,224,224]。")

    def require_shape(name: str, expected: str, valid) -> None:
        value = values[name]
        if not torch.is_tensor(value) or not valid(tuple(value.shape)):
            got = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            raise ValueError(f"obs[{name!r}] 形状必须是 {expected}，得到 {got}。")

    require_shape("lang_tokens", "[64]", lambda s: s == (64,))
    require_shape(
        "prefill_mask", f"[>={prefix_len},>={prefix_len}]",
        lambda s: len(s) == 2 and s[0] >= prefix_len and s[1] >= prefix_len)
    valid_rows = values["prefill_mask"].bool().any(dim=-1)
    mask_valid = int(valid_rows.sum().item())
    valid_is_prefix = bool(valid_rows[:claimed_valid].all().item()) and not bool(
        valid_rows[claimed_valid:].any().item()
    )
    if mask_valid != claimed_valid or not valid_is_prefix:
        raise ValueError(
            "obs['n_valid'] 必须与 prefill_mask 的连续有效前缀一致；"
            f"声明 {claimed_valid} 行，mask 推导为 {mask_valid} 行。"
        )
    require_shape(
        "prefill_pos", f"[>={prefix_len}]",
        lambda s: len(s) == 1 and s[0] >= prefix_len)
    require_shape(
        "modality_mask", f"[>={prefix_len}]",
        lambda s: len(s) == 1 and s[0] >= prefix_len)
    require_shape(
        "denoise_mask", f"[51,>={prefix_len + 51}]",
        lambda s: len(s) == 2 and s[0] == 51 and s[1] >= prefix_len + 51)
    require_shape("suffix_pos", "[51]", lambda s: s == (51,))
    require_shape("noise", "[1,50,32]", lambda s: s == (1, 50, 32))

    state_keys = [name for name in ("state", "state_emb") if values.get(name) is not None]
    if len(state_keys) != 1:
        raise ValueError("obs= 必须在 state / state_emb 中恰好给一个。")
    if state_keys[0] == "state":
        require_shape("state", "[1,32]", lambda s: s == (1, 32))
    else:
        require_shape("state_emb", "[1,1,1024]", lambda s: s == (1, 1, 1024))
    return values


def _ceil16(n: int) -> int:
    return ((int(n) + 15) // 16) * 16


@dataclasses.dataclass(frozen=True)
class HyEmbodiedActionOutput:
    """一次 action-chunk 推理的结果。

    `actions` 是**已解码成物理量纲**的双臂 EE 位姿 `(H-1, 16)`（wxyz），
    口径是**配置的坐标约定下的解码轨迹（未认证真机）**。只有在构造
    policy 时给了 `norm_stats_path`、且本次调用传了 `ee_pose` 时才有值；否则读它
    会抛异常：模型吐的是**归一化的相对增量**，不能直接作为物理动作下发。

    `actions_normalized` 是 `[50, 20]` 的归一化有效维，
    `actions_full` 是未裁的 `[50, 32]`。
    """

    actions_normalized: torch.Tensor
    actions_full: torch.Tensor
    latency_ms: float
    phase_ms: dict[str, float]
    actions_physical: Any = None          # np.ndarray (H-1, 16)，未配解码时为 None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def actions(self):
        """物理双臂 EE 位姿 `(H-1, 16)`。未配置解码时**抛异常**，不返回归一化值。"""
        if self.actions_physical is None:
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "Hy-VLA 的 action 是**归一化的 RT-相对增量**，不能直接下发给机器人。"
                "要拿物理动作，请 ① 用 HyEmbodiedPolicy.from_checkpoint(..., "
                "norm_stats_path=<ckpt>/norm_stats.pkl) 构造，② 调用 infer(..., "
                "ee_pose=<当前双臂 EE 位姿 16 维 wxyz>)。"
                "确实想看归一化网格就读 .actions_normalized / .actions_full。")
        return self.actions_physical

    @property
    def physical_action_available(self) -> bool:
        """**这一次调用**有没有产出物理动作（`.actions` 读得到值）。

        ⚠️ 刻意**不**与 `HyEmbodiedPolicy.action_decode_configured` 同名 —— 那个是
        policy 级的"解码链装上了没有"，这个是本次调用级的"这次真的解出来了没有"。
        两者可以不一致：解码链配好了、但这次没传 `ee_pose`（或走的是 `obs=` 那条
        复现入口）就没有起始位姿，本次仍然没有物理动作。

        ⚠️ 它也**不是** "robot_ready"。为真只说明"输出是物理量纲"，不说明"量纲对得上
        你的机器人"：坐标系、末端位姿原点与夹爪量纲属于机器人业务契约，须由机器人
        团队书面确认，任何数值对拍都替代不了。
        """
        return self.actions_physical is not None


class HyEmbodiedPolicy:
    """Hy-Embodied-0.5-VLA（HYViT2 ViT + HunYuanVL-MoT VLM + flow-matching expert）。

    ckpt、相机数、精度档、prefix 长度在构造时固定；图像、指令、本体感受每次调用可变。
    """

    def __init__(self) -> None:
        raise RuntimeError(
            "HyEmbodiedPolicy() 不是公共构造函数。请用 "
            "HyEmbodiedPolicy.from_checkpoint(...).to('rpu')。")

    # ── 构造 ────────────────────────────────────────────────────────────────
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str | os.PathLike[str],
        *,
        dtype: str = "fp16",
        prefix_len: int = 240,
        n_cameras: int = 3,
        norm_stats_path: str | os.PathLike[str] | None = None,
        trust_norm_stats_pickle: bool = False,
        norm_stats_sha256: str | None = None,
        umi_coord_frame: bool = True,
        umi_gripper_space: bool = False,
        runtime_env: Mapping[str, str] | None = None,
        threads: int = 12,
    ) -> "HyEmbodiedPolicy":
        """绑定一个 Hy-VLA ckpt 目录。

        `ckpt_dir` 里要有 `model.safetensors`、tokenizer 三件套
        （`tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json`），
        以及（想要物理动作时）`norm_stats.pkl`。**同一个目录服务全部精度档** ——
        量化在装载时做，没有单独的量化 ckpt 要发。

        Args:
            dtype: 见 `_PROFILE_ENV`。`fp16` 是精度参考；`w8a16-expert-vlmv`
                是推荐部署档。
            prefix_len: S，`{16,32,…,240}`。240 覆盖 `tokenizer_max_length=64`
                下的最坏 prompt，是默认值；调小只为省时间，且必须 ≥
                `ceil16(本条 observation 的有效行数)`。
            norm_stats_path: `norm_stats.pkl`。**必须显式给，不会自动探测**；
                不给就拿不到物理动作（`out.actions` 会抛，只能读
                `.actions_normalized`）。通常就是 `<ckpt_dir>/norm_stats.pkl`。
            trust_norm_stats_pickle / norm_stats_sha256: `norm_stats.pkl` 是可执行的
                pickle 格式；必须显式信任并提供该文件的精确 SHA-256 才会读取。
            umi_coord_frame / umi_gripper_space: 坐标帧与夹爪空间约定。
                ⚠️ **必须与机器人团队确认**，它们决定
                下发动作落在哪个世界帧。
            threads: `torch.set_num_threads`。⚠️ 生产口径是 12；板上默认 24，
                不设就会量错延迟。
        """
        dtype = (dtype or "fp16").strip().lower()
        if dtype not in _PROFILE_ENV:
            raise ValueError(f"dtype 必须是 {sorted(_PROFILE_ENV)} 之一，得到 {dtype!r}")
        if prefix_len % 16 or not 16 <= prefix_len <= 240:
            raise ValueError(f"prefix_len 必须是 16 的倍数且在 [16,240]，得到 {prefix_len}")
        if not isinstance(trust_norm_stats_pickle, bool):
            raise TypeError("trust_norm_stats_pickle must be bool")
        if norm_stats_path is None:
            if trust_norm_stats_pickle or norm_stats_sha256 is not None:
                raise ValueError(
                    "trust_norm_stats_pickle/norm_stats_sha256 require norm_stats_path"
                )
        else:
            if not trust_norm_stats_pickle:
                raise PermissionError(
                    "norm_stats_path requires trust_norm_stats_pickle=True and an exact "
                    "norm_stats_sha256"
                )
            if (
                not isinstance(norm_stats_sha256, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", norm_stats_sha256) is None
            ):
                raise ValueError(
                    "norm_stats_sha256 must be an exact 64-digit SHA-256 hex digest"
                )

        import pathlib

        d = pathlib.Path(ckpt_dir)
        inst = cls.__new__(cls)
        inst._ckpt_dir = d
        inst._ckpt = d / "model.safetensors"
        inst._dtype = dtype
        inst._prefix_len = int(prefix_len)
        # ⚠️ **UMI profile 固定 3 相机**，不接受任意 n_cameras。
        #    整条链上有三处把 3 写死：`preprocess.CAMERA_ORDER` 恰好 3 项、
        #    ckpt `config.json` 的 `image_features` 是这三个键、ViT packed 路径的
        #    KV cache 按相机数定长。传别的值不会在这里报错，只会在更深处
        #    产出错位的 prefix 布局或对不上的 KV 长度 —— 那时已经很难查了。
        #    支持其他相机数需要对应的 checkpoint 与数值验证，不是只改参数。
        from rpu_backend.adapters.hy_vla.preprocess import CAMERA_ORDER

        if int(n_cameras) != len(CAMERA_ORDER):
            raise ValueError(
                f"Hy-Embodied-0.5-VLA-UMI 是 {len(CAMERA_ORDER)} 相机 profile"
                f"（{', '.join(CAMERA_ORDER)}），不支持 n_cameras={n_cameras}。")
        inst._n_cameras = int(n_cameras)
        # ⚠️ **不自动探测 `norm_stats.pkl`**：物理动作解码必须由调用方**显式**打开，
        # 避免目录内容静默改变是否产出物理动作。
        #    不给这个参数时 `out.actions` 继续抛异常（读 `.actions_normalized`）。
        inst._norm_stats_path = norm_stats_path
        inst._trust_norm_stats_pickle = trust_norm_stats_pickle
        inst._norm_stats_sha256 = norm_stats_sha256
        inst._umi_coord_frame = bool(umi_coord_frame)
        inst._umi_gripper_space = bool(umi_gripper_space)
        inst._runtime_env = normalize_runtime_env(
            runtime_env,
            owner="HyEmbodiedPolicy",
            allowed=_RUNTIME_ENV_ALLOWLIST,
            preimport_only=LKN_CAPACITY_ENV,
        )
        inst._threads = int(threads)
        inst._runner = None
        inst._runner_finalizer = None
        inst._tok = None
        inst._decoder = None
        inst._rpu_ready = False
        inst._rpu_build_started = False
        inst._rpu_build_lock = threading.Lock()
        inst._closed = False
        inst._env_snapshot = None
        inst._previous_threads = None
        inst._prepare_ms: dict[str, float] = {}
        return inst

    # ── 自描述（供服务健康检查使用） ─────────────────────────────────
    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def prefix_len(self) -> int:
        return self._prefix_len

    @property
    def action_dim(self) -> int:
        """每步真实机器人自由度数（模型吐 32，只有前 20 维是真的）。"""
        return VALID_ACTION_DIMS

    @property
    def model_horizon(self) -> int:
        """模型**归一化网格**的步数 —— `actions_normalized` / `actions_full` 的行数。"""
        return MODEL_ACTION_STEPS

    @property
    def action_horizon(self) -> int:
        """一次推理产出的**物理**动作步数 —— `out.actions` 的行数。

        比 `model_horizon` 少 1：解码时第 0 行是"当前位姿"并被丢弃。
        ⚠️ 这两个数不能混用：按 50 索引 `actions` 会越界，归一化网格仍有 50 行。
        """
        return MODEL_ACTION_STEPS - 1

    @property
    def camera_names(self) -> list[str]:
        """相机顺序 —— `infer(images=...)` 必须按这个顺序给图。"""
        from rpu_backend.adapters.hy_vla.preprocess import CAMERA_ORDER

        return list(CAMERA_ORDER[: self._n_cameras])

    @property
    def action_decode_configured(self) -> bool:
        """是否配好了 action 解码（配好了 `out.actions` 才有物理值）。

        ⚠️ 这**不是** "robot_ready"。它只回答"`norm_stats.pkl` 装上了没有"，
        不回答"解出来的位姿能不能下发给你的机器人"——后者取决于
        `umi_coord_frame` / `umi_gripper_space` / EE 帧与原点 / 夹爪量纲这组
        **机器人业务契约**，本仓库没有、也无法用数值验证确认它。
        """
        return self._decoder is not None

    # ── 上设备 ──────────────────────────────────────────────────────────────
    def to(self, device: Any) -> "HyEmbodiedPolicy":
        target = str(device)
        if not target.startswith("rpu"):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError("HyEmbodiedPolicy 只支持 .to('rpu')。")
        if self._closed:
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError("HyEmbodiedPolicy 已 close；请构建新 policy。")
        if self._rpu_ready:
            finalizer = self._runner_finalizer
            if (
                self._runner is not None
                and finalizer is not None
                and finalizer.alive
            ):
                return self
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "HyEmbodiedPolicy 被标记为 ready，但 runner ownership 已不完整。"
                "请重启 Python 进程后重新构建。")
        required_files = (
            self._ckpt,
            self._ckpt_dir / "tokenizer.json",
            self._ckpt_dir / "tokenizer_config.json",
            self._ckpt_dir / "special_tokens_map.json",
        )
        missing_files = [path.name for path in required_files if not path.is_file()]
        if missing_files:
            from rpu_backend.api.errors import RPUConfigError

            raise RPUConfigError(
                f"Hy-VLA ckpt 不完整（{self._ckpt_dir}），缺少: "
                f"{', '.join(missing_files)}")

        if not self._rpu_build_lock.acquire(blocking=False):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "HyEmbodiedPolicy.to('rpu') 已在另一线程中构建该 policy。")
        try:
            if self._closed:
                from rpu_backend.api.errors import RPUBackendError

                raise RPUBackendError("HyEmbodiedPolicy 已 close；请构建新 policy。")
            if self._rpu_build_started:
                from rpu_backend.api.errors import RPUBackendError

                raise RPUBackendError(
                    "HyEmbodiedPolicy.to('rpu') 先前在权重/RPU 物化开始后失败。"
                    "部分 swizzle/handle 状态不能安全复用；请重启 Python 进程。")

            # Host-only preflight stays before process ownership. A malformed
            # decoder file must not consume the only usable RPU lifetime.
            decoder = None
            if self._norm_stats_path is not None:
                from rpu_backend.api.hy_action_decode import HyVlaActionDecoder

                norm_stats = _load_trusted_norm_stats_pickle(
                    self._norm_stats_path,
                    trust_norm_stats_pickle=self._trust_norm_stats_pickle,
                    norm_stats_sha256=self._norm_stats_sha256,
                )
                decoder = HyVlaActionDecoder.from_norm_stats_mapping(
                    norm_stats,
                    source=str(self._norm_stats_path),
                    umi_coord_frame=self._umi_coord_frame,
                    umi_gripper_space=self._umi_gripper_space)

            import rpu_backend  # noqa: F401  （注册 PrivateUse1 后端）
            from rpu_backend.adapters.hy_vla import build_hy_vla
            from rpu_backend.api.causal_lm import (
                _claim_live_instance,
                _release_live_instance,
            )

            # Claim before changing process-global env/thread state. A second
            # policy must fail without perturbing the live owner's profile.
            _claim_live_instance(self)
            env_snapshot: dict[str, str | None] = {}
            previous_threads = torch.get_num_threads()
            pending_runner = None
            pending_finalizer = None
            materialization_started = False
            try:
                # dtype 派生的 env 先写死（fp16 也显式写 "0"），再让
                # runtime_env 覆盖；后者按契约优先级最高。
                _remember_environment(env_snapshot, "RPU_HY_VLA_W8A16")
                os.environ["RPU_HY_VLA_W8A16"] = _PROFILE_ENV[self._dtype]
                for key, value in self._runtime_env.items():
                    _remember_environment(env_snapshot, key)
                    os.environ[key] = value
                torch.set_num_threads(self._threads)

                self._rpu_build_started = True
                materialization_started = True
                t0 = time.perf_counter()
                pending_runner = build_hy_vla(
                    self._ckpt,
                    prefix_len=self._prefix_len,
                    _env_snapshot=env_snapshot,
                    _owner=self,
                )
                prepare_ms = {
                    "build_ms": (time.perf_counter() - t0) * 1000.0
                }
                pending_finalizer = weakref.finalize(
                    self,
                    _finalize_runner,
                    pending_runner,
                    env_snapshot,
                    previous_threads,
                )

                self._runner = pending_runner
                self._runner_finalizer = pending_finalizer
                self._env_snapshot = env_snapshot
                self._previous_threads = previous_threads
                self._prepare_ms = prepare_ms
                self._decoder = decoder
                self._rpu_ready = True
                return self
            except BaseException as exc:
                cleanup_ok = False
                if pending_finalizer is not None and pending_finalizer.alive:
                    cleanup_ok = pending_finalizer() is True
                elif pending_runner is not None:
                    cleanup_ok = _finalize_runner(
                        pending_runner,
                        env_snapshot,
                        previous_threads,
                    )
                else:
                    cleanup_ok = True
                    try:
                        _restore_environment(env_snapshot)
                    except BaseException:
                        cleanup_ok = False
                    try:
                        torch.set_num_threads(previous_threads)
                    except BaseException:
                        cleanup_ok = False

                self._runner = None
                self._runner_finalizer = None
                self._env_snapshot = None
                self._previous_threads = None
                self._prepare_ms = {}
                self._decoder = None
                self._rpu_ready = False
                if not materialization_started:
                    self._rpu_build_started = False
                    _release_live_instance(self)
                elif not cleanup_ok:
                    raise RuntimeError(
                        "Hy-VLA 构建失败，且原生/全局状态清理未完成；"
                        "请重启进程后再加载 policy。") from exc
                raise
        finally:
            self._rpu_build_lock.release()

    def close(self) -> None:
        """Retire native resources; loading another RPU owner still needs restart."""
        with self._rpu_build_lock:
            if self._closed:
                return
            if self._rpu_build_started and not self._rpu_ready:
                from rpu_backend.api.errors import RPUBackendError

                raise RPUBackendError(
                    "HyEmbodiedPolicy 曾在 RPU 物化中失败，无法证明已安全回收；"
                    "请删除对象并重启进程。")

            finalizer = self._runner_finalizer
            if finalizer is not None and finalizer.alive:
                if finalizer() is not True:
                    raise RuntimeError(
                        "Hy-VLA runner 清理失败；process-wide RPU slot 仍保留。"
                        "请重启进程后再加载 policy。")
            elif self._runner is not None:
                self._runner.close()

            self._runner = None
            self._runner_finalizer = None
            self._env_snapshot = None
            self._previous_threads = None
            self._prepare_ms = {}
            self._decoder = None
            self._rpu_ready = False
            self._closed = True

    def _tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(str(self._ckpt_dir),
                                                      trust_remote_code=False)
        return self._tok

    # ── 推理 ────────────────────────────────────────────────────────────────
    def infer(
        self,
        *,
        images: Sequence[Any] | None = None,
        instruction: str | None = None,
        ee_pose: Any = None,
        state: Any = None,
        noise: torch.Tensor | None = None,
        noise_seed: int | None = None,
        obs: Mapping[str, Any] | None = None,
    ) -> HyEmbodiedActionOutput:
        """跑一次 action-chunk 推理。

        两种入口二选一：
        * **原始观测** —— `images` + `instruction`（+ `ee_pose`）：由 policy 自己
          做预处理（缩放补边、分词、mask、position_ids、noise）。
        * **`obs=`** —— 一份已经组好的入参字典，用于复现预处理后的输入。

        Args:
            images: 每相机一帧，顺序见 `camera_names`。收 CHW/HWC、torch/numpy/PIL、
                uint8（自动 `/255`）或已归一到 [0,1] 的浮点；**浮点越界会报错，
                不会被猜着自动缩放**。逐条规则见
                `adapters.hy_vla.preprocess._as_float01_nchw`。通道序必须是 **RGB**。
            ee_pose: 当前双臂 EE 位姿，16 维、四元数 **wxyz**。给了它才会
                ① 编码成模型的 state 入参，② 解码出物理动作。
            state: 直接给**已归一化**的 state（跳过 ①）。与 `ee_pose` 互斥。
            obs: 已组好的 runner 张量字典，必须额外带正整数
                `n_valid`。调用前会验证 `ceil16(n_valid) <= prefix_len`
                以及所有固定张量包络，不允许静默截断 prefix。
            noise_seed: 给了就用独立 Generator，不动全局 RNG，逐位可复现。
        """
        if not self._rpu_ready:
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError("请先 .to('rpu') 再 infer()。")
        if ee_pose is not None and state is not None:
            raise ValueError("ee_pose 与 state 二选一：前者是原始位姿，后者是已归一化的 state。")
        # ⚠️ `obs=` 与原始观测入口**互斥**，不是"可以混着传、多余的忽略掉"。
        #    混用时静默丢参数是最难查的一类错：`infer(obs=…, images=…)` 会拿 obs 跑、
        #    把新图悄悄扔掉；`infer(obs=…, ee_pose=…)` 更糟 —— 推理用的是 obs 里那个
        #    state，解码却用外面传的位姿，两者可以完全对不上，而结果看起来很正常。
        if obs is not None:
            mixed = [n for n, v in (("images", images), ("instruction", instruction),
                                    ("ee_pose", ee_pose), ("state", state),
                                    ("noise", noise), ("noise_seed", noise_seed))
                     if v is not None]
            if mixed:
                raise ValueError(
                    f"obs= 与原始观测入口互斥，但同时传了 {mixed}。"
                    "obs= 是「逐位复现一条录下来的输入」，它自带 state/noise/图像张量；"
                    "再传这些只会被忽略或与 obs 冲突。"
                    "⇒ 要复现就只给 obs=；要跑新观测就给 images=/instruction=（+ee_pose=）。"
                    "⚠️ obs= 这条路**不产出物理动作**（没有可信的初始位姿），"
                    "`out.actions` 会抛异常，请读 `.actions_normalized`。")

        from rpu_backend.adapters.hy_vla import preprocess as P

        if obs is None:
            if images is None or instruction is None:
                raise ValueError("要么给 obs=…，要么给 images=… 和 instruction=…")
            if len(images) != self._n_cameras:
                raise ValueError(f"需要 {self._n_cameras} 张图（n_cameras），"
                                 f"得到 {len(images)}")
            norm_state = None
            if ee_pose is not None:
                if self._decoder is None:
                    raise ValueError(
                        "传了 ee_pose 但没有 norm_stats —— 无法把位姿编码成模型的 "
                        "state。请在 from_checkpoint 里给 norm_stats_path。")
                norm_state = torch.as_tensor(
                    self._decoder.encode_state(ee_pose), dtype=torch.float32)
            elif state is not None:
                norm_state = torch.as_tensor(state, dtype=torch.float32)

            built = P.build_observation(
                images, instruction, self._tokenizer(),
                state=norm_state, noise=noise, noise_seed=noise_seed)
            obs = built
        obs = _validate_runner_observation(
            obs, n_cameras=self._n_cameras, prefix_len=self._prefix_len)

        t0 = time.perf_counter()
        with torch.no_grad():
            raw = self._runner.get_action(**obs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        full = raw.detach().float().cpu()
        if full.dim() == 3:
            full = full[0]
        if not bool(torch.isfinite(full).all()):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError("Hy-VLA 输出了非有限值（NaN/Inf）—— 拒绝返回。")

        norm20 = full[:, :VALID_ACTION_DIMS].contiguous()
        physical = None
        if self._decoder is not None and ee_pose is not None:
            physical = self._decoder.decode(norm20.numpy(), ee_pose)
            # 解码链走的是 host 侧 numpy/scipy（矩阵求逆、四元数归一化）——
            # 上游有限不代表这里有限：起始位姿退化、四元数模长为 0 都会产出 NaN。
            # 这是**流向机器人的最后一道值**，必须在这里挡住，不能让它出函数。
            import numpy as _np

            # 对着**对外宣称的**契约校验，而不是对着解码链自己的中间值 —— 这样
            # `action_horizon` 与实际返回的行数不一致时会在这里就炸，不会让调用方
            # 拿着 50 去索引一个 49 行的轨迹。
            exp = self.action_horizon
            if physical.ndim != 2 or physical.shape[1] != 16:
                from rpu_backend.api.errors import RPUBackendError

                raise RPUBackendError(
                    f"action 解码产出了意外形状 {physical.shape}，期望 (H-1, 16)。")
            if physical.shape[0] != exp:
                from rpu_backend.api.errors import RPUBackendError

                raise RPUBackendError(
                    f"action 解码产出 {physical.shape[0]} 步，期望 {exp} 步"
                    f"（模型 {norm20.shape[0]} 步，解码丢掉第 0 行）。")
            if not bool(_np.isfinite(physical).all()):
                from rpu_backend.api.errors import RPUBackendError

                raise RPUBackendError(
                    "action 解码产出了非有限值（NaN/Inf）—— 拒绝返回。"
                    "最常见成因是 ee_pose 的四元数模长为 0 或位姿退化；"
                    "请检查传入的 ee_pose（16 维、wxyz）。")

        return HyEmbodiedActionOutput(
            actions_normalized=norm20,
            actions_full=full,
            latency_ms=latency_ms,
            phase_ms=dict(self._prepare_ms),
            actions_physical=physical,
            extra={
                "dtype": self._dtype,
                "prefix_len": self._prefix_len,
                "valid_action_dims": VALID_ACTION_DIMS,
                "model_horizon": self.model_horizon,
                "action_horizon": self.action_horizon,
                # **本次调用**有没有物理动作。policy 级的"解码链装上了没有"是
                # `policy.action_decode_configured`，两者可以不一致（配好了但这次
                # 没传 ee_pose）。
                "physical_action_available": physical is not None,
                # ⚠️ 把坐标约定连同**它的出处**一起带出来。配上 norm_stats 只说明
                #    解码链能算，不说明这套约定与你的机器人一致 —— 这两个值目前
                #    一律来自 vendor 默认，**没有任何一方确认过**。
                "umi_coord_frame": self._umi_coord_frame,
                "umi_gripper_space": self._umi_gripper_space,
                "coord_convention_source": "vendor-default (unconfirmed)",
            },
        )

    def predict_action_chunk(self, **kwargs: Any):
        """便捷封装：只要解码后的物理量纲动作 `(H-1, 16)`。

        ⚠️ 口径是**配置的坐标约定下的解码轨迹（未认证真机）** ——
        上机前必须由机器人团队确认坐标系与夹爪量纲，首次上机需值守急停。
        """
        return self.infer(**kwargs).actions

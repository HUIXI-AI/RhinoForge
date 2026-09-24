"""Hy-Embodied-0.5-VLA 的 action 解码 —— 归一化网格 → 物理双臂 EE 位姿。

存在的理由
    模型吐的是**归一化的 RT-相对增量**，不是可以直接下发的关节/位姿指令。
    vendor 的评测封装（`robodojo_eval/policy_wrapper.py::_infer_chunk_wxyz`）在
    交给机器人之前一定会走完下面这条链；RPU 交付此前只把归一化网格原样返回，
    调用方一旦直接下发就是错的。同类交付上已经出过事故：LingBot-VLA-V2 的
    `api/action_denorm.py` 记录了"漏掉反归一化 ⇒ 肩关节偏 1.84 rad ≈ 106°"。

    ⇒ 本模块补上这一步，并由 `HyEmbodiedActionOutput.actions` 强制：**没配解码
    器就抛异常**，绝不把归一化值伪装成物理动作。

解码链（逐字对应 vendor）
    1. `rel = actions[:, :20] * act_std + act_mean`
       ⚠️ `act_std/act_mean` 形状是 **(50, 20)** —— **逐时间步**的统计量，不是
          一条 20 维向量。当成 1-D 广播会把整个 chunk 算错。
    2. 初始 EE 位姿 wxyz → xyzw
    3. （UMI ckpt）`convert_frame_robo_to_umi`
    4. `relative_to_dual_arm_poses(rel, initial_xyzw)`：6D 旋转 → 矩阵 →
       `T_i = T_0 @ ΔT_i` → 位置 + 四元数
    5. （UMI ckpt）`convert_frame_umi_to_robo`
    6. xyzw → wxyz
    7. **丢掉第 0 行** —— 第 0 行的增量≈0，对应"当前位姿"，vendor 下发的是
       `actions_wxyz[1 : n+1]`

⚠️⚠️ **口径：只能声明"与 vendor 代码一致"，不能声明"可下发真机"。**

交付 golden 里**只有归一化的** `p7.action_valid20` —— vendor 没有留下解码后的参考
轨迹，也没有留下当时的 `observation.state`。⇒ **不存在"物理动作 vs golden"这种对拍**，
凡是这么说的都是把两件事混了。

已经做到的：在隔离进程里跑 **vendor 官方 `_decode_actions` 源码**产出 oracle，再由
另一个进程加载本模块比对，解码轨迹 `max_abs_diff 0.000e+00`（逐位相同）。
⚠️ 那一轮是在**重放到主干之前**的基点上跑的，按本轮纪律记作**历史参考**；
本模块自身在重放中未发生逻辑改动（仅措辞），但**重放后尚未重跑** ⇒ 标 NOT RUN。

即使重跑通过，它也**只**支持"与 vendor 代码一致"。真机接入前仍必须：
  ① 与机器人团队确认 `umi_coord_frame` / `umi_gripper_space` / EE 位姿帧与原点 /
     夹爪量纲（见 DEPLOYMENT.md 的"未决契约"一节）—— 这是**业务契约**，
     任何数值对拍都替代不了；
  ② 首次上机必须有人在场并握住急停。
"""
from __future__ import annotations

import dataclasses
import pickle

import numpy as np

# vendor `robotwin_eval/transforms.py` 与 `hy_vla/utils/transform_utils.py` 的
# 每一步旋转都用 scipy，这里照用同一个库：oracle 也是跑 vendor 那份 scipy 代码得到的，
# 再换一套四元数实现只会多引入一层无法归因的差异。
try:
    from scipy.spatial.transform import Rotation as _R
except ImportError as exc:                                  # pragma: no cover
    raise ImportError(
        "Hy-VLA 的 action 解码需要 scipy（vendor 的 transforms.py 用它做 "
        "6D 旋转 ↔ 四元数）。`pip install scipy`，或见 requirements.txt。"
    ) from exc

# UMI ↔ RoboTwin 的世界/局部坐标变换（vendor transform_utils.py）：
#   世界 W = [[0,1,0],[-1,0,0],[0,0,1]]，局部列轮换 P = [[0,0,1],[1,0,0],[0,1,0]]
#   p_umi = W @ p_native ，R_umi = W @ R_native @ P
_W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
_P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

# 双臂 16 维布局（xyzw）：[lxyz3, lquat4, lgrip1, rxyz3, rquat4, rgrip1]
_L_POS, _L_QUAT, _L_GRIP = slice(0, 3), slice(3, 7), 7
_R_POS, _R_QUAT, _R_GRIP = slice(8, 11), slice(11, 15), 15


def _rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """`(N,6)` Gram-Schmidt 6D 表示 → `(N,3,3)` 旋转矩阵（vendor 同名函数）。"""
    a1, a2 = d6[:, :3], d6[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=1)


def _relative_matrices_to_poses(rel: np.ndarray, start_xyzw: np.ndarray) -> np.ndarray:
    """单臂：`(N,9)` = [平移3, 6D旋转6] + 起始位姿 → `(N,7)` PosQuat(xyzw)。"""
    n = rel.shape[0]
    t0 = np.eye(4)
    t0[:3, :3] = _R.from_quat(start_xyzw[3:]).as_matrix()
    t0[:3, 3] = start_xyzw[:3]

    dt = np.eye(4).reshape(1, 4, 4).repeat(n, axis=0)
    dt[:, :3, :3] = _rotation_6d_to_matrix(rel[:, 3:])
    dt[:, :3, 3] = rel[:, :3]

    ti = t0 @ dt
    return np.concatenate([ti[:, :3, 3], _R.from_matrix(ti[:, :3, :3]).as_quat()], axis=1)


def relative_to_dual_arm_poses(rel20: np.ndarray, start_dual_xyzw: np.ndarray) -> np.ndarray:
    """`(N,20)` RT-相对输出 → `(N,16)` 双臂 PosQuat(xyzw)（vendor 同名函数）。

    `rel20` 的分段：[左平移3, 左6D6, 左夹爪1, 右平移3, 右6D6, 右夹爪1]。
    夹爪**不参与位姿合成**，原样透传。
    """
    left = _relative_matrices_to_poses(rel20[:, 0:9], start_dual_xyzw[0:7])
    right = _relative_matrices_to_poses(rel20[:, 10:19], start_dual_xyzw[8:15])
    return np.concatenate([left, rel20[:, 9:10], right, rel20[:, 19:20]], axis=1)


def _convert_frame(qpos: np.ndarray, *, to_umi: bool, convert_gripper: bool) -> np.ndarray:
    """UMI ↔ RoboTwin 世界/局部帧互转（vendor `convert_frame_{robo_to_umi,umi_to_robo}`）。

    夹爪约定（`convert_gripper=True` 时）：RoboTwin 开=1/闭=0（归一化）
    ↔ UMI 开=0/闭=90（mm）。
    """
    out = np.asarray(qpos, dtype=np.float64).copy()
    if out.shape[0] == 0:
        return out
    w, p = (_W, _P) if to_umi else (_W.T, _P.T)
    qw, qp = _R.from_matrix(w), _R.from_matrix(p)

    for pos, quat, grip in ((_L_POS, _L_QUAT, _L_GRIP), (_R_POS, _R_QUAT, _R_GRIP)):
        out[:, pos] = out[:, pos] @ w.T
        out[:, quat] = (qw * _R.from_quat(out[:, quat]) * qp).as_quat()
        if convert_gripper:
            out[:, grip] = ((1.0 - out[:, grip]) * 90.0 if to_umi
                            else 1.0 - out[:, grip] / 90.0)
    return out


def _quat_wxyz_to_xyzw(a: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64).copy()
    out[..., 3:7] = np.asarray(a)[..., [4, 5, 6, 3]]
    out[..., 11:15] = np.asarray(a)[..., [12, 13, 14, 11]]
    return out


def _quat_xyzw_to_wxyz(a: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64).copy()
    out[..., 3:7] = np.asarray(a)[..., [6, 3, 4, 5]]
    out[..., 11:15] = np.asarray(a)[..., [14, 11, 12, 13]]
    return out


def _pos_quat_to_rot6d(pq16_xyzw: np.ndarray) -> np.ndarray:
    """`(N,16)` PosQuat(xyzw) → `(N,20)` PosRot6d
    （vendor `convert_PosQuat2PosRotationMatrix_batch`）。

    6D 旋转 = 旋转矩阵的**前两行**（不是前两列），与训练时的约定一致。
    """
    out = np.zeros((pq16_xyzw.shape[0], 20), dtype=np.float64)
    for src_pos, src_quat, src_grip, dst in (
            (_L_POS, _L_QUAT, _L_GRIP, 0), (_R_POS, _R_QUAT, _R_GRIP, 10)):
        m = _R.from_quat(pq16_xyzw[:, src_quat]).as_matrix()
        out[:, dst:dst + 3] = pq16_xyzw[:, src_pos]
        out[:, dst + 3:dst + 6] = m[:, 0, :]
        out[:, dst + 6:dst + 9] = m[:, 1, :]
        out[:, dst + 9] = pq16_xyzw[:, src_grip]
    return out


@dataclasses.dataclass(frozen=True)
class HyVlaActionDecoder:
    """这个 ckpt 的**归一化契约**：raw EE 位姿 → 模型 state，以及模型 action → 物理位姿。

    两个方向放在一个对象里，是因为它们共用同一份 `norm_stats.pkl` 和同一组坐标
    帧开关 —— 分开配就一定会有人只配一半，然后得到一条方向自洽、数值全错的轨迹。

    ⚠️ 每一个约定都来自配置，**不硬编码任何机器人**：统计量来自
    `norm_stats.pkl`，坐标帧/夹爪空间由构造参数给出。默认值取的是 vendor
    `HyVLARoboDojoPolicyWrapper.__init__` 的默认值（`umi_coord_frame=True`,
    `umi_gripper_space=False`），**它们对你的机器人未必成立**。
    """

    act_mean: np.ndarray                 # (H, 20) 逐时间步
    act_std: np.ndarray                  # (H, 20)
    qpos_mean: np.ndarray | None = None  # (20,) —— 只有 encode_state 用
    qpos_std: np.ndarray | None = None   # (20,)
    umi_coord_frame: bool = True
    umi_gripper_space: bool = False
    drop_first: bool = True              # vendor 下发 actions[1:]

    def __post_init__(self):
        if self.act_mean.shape != self.act_std.shape:
            raise ValueError(f"act_mean{self.act_mean.shape} 与 "
                             f"act_std{self.act_std.shape} 形状不一致")
        if self.act_mean.ndim != 2 or self.act_mean.shape[1] != 20:
            raise ValueError(f"统计量应为 (H, 20)，得到 {self.act_mean.shape}")
        if self.umi_gripper_space and not self.umi_coord_frame:
            raise ValueError("umi_gripper_space=True 需要 umi_coord_frame=True"
                             "（vendor 的同一条断言）")

    @classmethod
    def from_norm_stats(cls, path, **kwargs) -> "HyVlaActionDecoder":
        """从 ckpt 目录里的 `norm_stats.pkl` 构造。

        需要 `action_mean` / `action_std`（`(50,20)`）；`qpos_mean` / `qpos_std`
        （`(20,)`）有则一并读入，供 `encode_state` 用。若该 pickle 还带
        `action_mean_abs` / `action_std_abs`（`act_type=..._with_absolute` 训练的
        ckpt），说明它是**另一种**解码分支 —— 本类未实现，显式拒绝而不是按纯
        relative 静默算错。
        """
        with open(path, "rb") as f:
            info = pickle.load(f)
        for k in ("action_mean", "action_std"):
            if k not in info:
                raise KeyError(f"{path} 缺 {k!r}（有: {sorted(info)}）")
        if info.get("action_mean_abs") is not None and info.get("action_std_abs") is not None:
            raise NotImplementedError(
                f"{path} 带 action_*_abs ⇒ 该 ckpt 是 "
                f"`relative_chunk_ee_RT_with_absolute` 训练的，解码要走 vendor "
                f"`_decode_actions` 的前/后半段混合分支，本类只实现了纯 relative。")
        arr = lambda k: (None if info.get(k) is None
                         else np.asarray(info[k], dtype=np.float64))
        return cls(act_mean=np.asarray(info["action_mean"], dtype=np.float64),
                   act_std=np.asarray(info["action_std"], dtype=np.float64),
                   qpos_mean=arr("qpos_mean"), qpos_std=arr("qpos_std"), **kwargs)

    @property
    def horizon(self) -> int:
        return int(self.act_mean.shape[0])

    def encode_state(self, ee_pose16_wxyz) -> np.ndarray:
        """`(16,)` 当前双臂 EE 位姿（wxyz）→ `(1,20)` 归一化 state（模型入参）。

        逐字对应 vendor `_convert_pose_robo_dojo`：wxyz→xyzw → （UMI ckpt）帧转换
        → 四元数转 6D 旋转 → `(x - qpos_mean) / (qpos_std + 1e-8)`。

        ⚠️ vendor 另有一处 `transforms.convert_pose` 除的是 `qpos_std`（无 1e-8）。
        本 ckpt 的 `qpos_std` 最小 0.0712，两者相对差 ~1.4e-7，可忽略；这里取
        real-robot 那条（robodojo）的写法。
        """
        if self.qpos_mean is None or self.qpos_std is None:
            raise ValueError("norm_stats 里没有 qpos_mean/qpos_std，无法编码 state")
        e = np.asarray(ee_pose16_wxyz, dtype=np.float64).reshape(-1)[:16]
        e = _quat_wxyz_to_xyzw(e)[None]
        if self.umi_coord_frame:
            e = _convert_frame(e, to_umi=True, convert_gripper=self.umi_gripper_space)
        return (_pos_quat_to_rot6d(e) - self.qpos_mean) / (self.qpos_std + 1e-8)

    def decode(self, actions_norm, initial_ee_pose_wxyz) -> np.ndarray:
        """`(H,20)`（或 `(1,H,·)` / `(H,32)`）归一化 action + `(16,)` 当前 EE 位姿
        → `(H-1, 16)` 物理位姿（wxyz，RoboTwin 世界帧）。

        Args:
            actions_norm: 模型输出。多余的动作维（20 之后是补零占位）会被截掉。
            initial_ee_pose_wxyz: **当前**双臂 EE 位姿，16 维、四元数 wxyz、
                与训练时同一个世界帧。相对增量就是相对它合成的 —— 传错这一项，
                整条轨迹平移/旋转整体偏掉，而且**不会报错**。
        """
        a = np.asarray(actions_norm, dtype=np.float64)
        if a.ndim == 3:
            if a.shape[0] != 1:
                raise ValueError(f"只支持 batch=1，得到 {a.shape}")
            a = a[0]
        if a.ndim != 2:
            raise ValueError(f"action 应为 (H, D)，得到 {a.shape}")
        h = a.shape[0]
        if h > self.horizon:
            raise ValueError(f"action 有 {h} 步，超过统计量的 {self.horizon} 步")
        if a.shape[1] < 20:
            raise ValueError(f"action 只有 {a.shape[1]} 维，需要至少 20")

        start = np.asarray(initial_ee_pose_wxyz, dtype=np.float64).reshape(-1)
        if start.shape[0] < 16:
            raise ValueError(f"initial_ee_pose 需要 16 维，得到 {start.shape[0]}")
        start_xyzw = _quat_wxyz_to_xyzw(start[:16])
        if self.umi_coord_frame:
            start_xyzw = _convert_frame(start_xyzw[None], to_umi=True,
                                        convert_gripper=self.umi_gripper_space)[0]

        rel = a[:, :20] * self.act_std[:h] + self.act_mean[:h]
        poses = relative_to_dual_arm_poses(rel, start_xyzw)
        if self.umi_coord_frame:
            poses = _convert_frame(poses, to_umi=False,
                                   convert_gripper=self.umi_gripper_space)
        poses = _quat_xyzw_to_wxyz(poses)
        return poses[1:] if self.drop_first else poses

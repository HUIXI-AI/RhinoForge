"""Leju kuavo_server adapter body for openpi pi05 on RPU.

``kuavo_server/adapters/openpi_pi05_rpu.py`` (next to this module, copied into the
customer's kuavo_server package) registers the adapter and builds this class.
The class turns one kuavo_server observation into the ``OpenpiPi05Server``
request and the action chunk back into robot actions. It mirrors
``kuavo_server.adapters.openpi`` (LeTools-Learning eb8687c) for:

- observation keys and the head-camera fallback for a missing wrist camera;
- image coercion (``_as_hwc_uint8``);
- ``which_arm`` state slicing (``_adapt_state_dim``) and action slicing
  (``_convert_action``);
- the ``execution_horizon`` cut and the pending-step cache of ``select_action``.

This module imports no kuavo_server code, so host tests run without it.
"""
from __future__ import annotations

from pathlib import Path

ADAPTER_NAME = "openpi_pi05_rpu"
HEAD, LEFT, RIGHT = ("observation.images.head_cam_h", "observation.images.wrist_cam_l",
                     "observation.images.wrist_cam_r")
STATE = "observation.state"


def to_numpy(value):
    import numpy as np

    try:
        import torch
    except ModuleNotFoundError:  # kuavo_server clients may send numpy only
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value if isinstance(value, np.ndarray) else np.asarray(value)


def _unit_float_tensor_as_uint8(image):
    """A float32 torch frame in [0, 1] (kuavo_deploy's ``to_tensor``) as uint8, or None.

    The arithmetic of the numpy path in ``as_hwc_uint8`` (float32 product with 255,
    clip to [0, 255], truncation), run by torch on the contiguous source with its
    intra-op threads, then one permute of the uint8 result. The upper clip is a no-op
    there: max <= 1 bounds every rounded product by 255. Anything else returns None
    and takes the numpy path.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not isinstance(image, torch.Tensor) or image.dtype != torch.float32 or image.numel() == 0:
        return None
    tensor = image.detach().cpu()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    low, high = torch.aminmax(tensor)  # a NaN propagates into both
    if not bool(torch.isfinite(low) & torch.isfinite(high)):
        raise ValueError("camera frame contains NaN/Inf")
    if high > 1.0:
        return None
    scaled = tensor.mul(255.0)
    if low < 0:
        scaled.clamp_(min=0)
    result = scaled.to(torch.uint8)
    if result.ndim == 3 and result.shape[0] in (1, 3) and result.shape[-1] not in (1, 3):
        result = result.permute(1, 2, 0)
    return result.contiguous().numpy()


def as_hwc_uint8(image):
    """kuavo_server openpi ``_as_hwc_uint8``; a non-finite float frame is refused.

    kuavo_deploy frames take ``_unit_float_tensor_as_uint8``, which returns the same
    bytes; on the board (16 torch threads) an 848x480 frame takes 0.6 instead of 12.2 ms.
    """
    import numpy as np

    array = _unit_float_tensor_as_uint8(image)
    if array is None:
        array = to_numpy(image)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                if not np.isfinite(array).all():
                    raise ValueError("camera frame contains NaN/Inf")
                if array.max(initial=0) <= 1.0:
                    array = (array * 255.0).clip(0, 255)
            array = array.astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Image must be HWC with 3 channels, got shape={array.shape}")
    return np.ascontiguousarray(array)


class ArmLayout:
    """Robot state/action <-> model state/action, as kuavo_server openpi slices them.

    The robot sends ``[left arm, left eef, right arm, right eef, lower body...]``.
    A one-arm model sees ``[selected arm, its eef, lower body...]``.
    """

    def __init__(self, *, which_arm: str, state_dim: int, action_dim: int,
                 arm_joint_dim: int = 7, end_effector_dim: int = 1):
        if which_arm not in ("left", "right", "both"):
            raise ValueError("which_arm must be left, right or both")
        if arm_joint_dim <= 0 or end_effector_dim < 0:
            raise ValueError("arm_joint_dim must be positive and end_effector_dim must be non-negative")
        self.which_arm = which_arm
        self.state_dim, self.action_dim = state_dim, action_dim
        self.per_arm = arm_joint_dim + end_effector_dim
        self.arm_joint_dim, self.end_effector_dim = arm_joint_dim, end_effector_dim

    def adapt_state(self, state):
        """``_adapt_state_dim``: pass through at the model width, else slice one arm + lower body."""
        import numpy as np

        state = to_numpy(state).astype(np.float64).reshape(-1)
        if state.shape[0] == self.state_dim:
            return state
        full_arm = 2 * self.per_arm
        lower = state.shape[0] - full_arm
        if self.which_arm in ("left", "right") and lower >= 0 and self.state_dim - self.per_arm == lower:
            offset = 0 if self.which_arm == "left" else self.per_arm
            return np.concatenate([state[offset:offset + self.per_arm], state[full_arm:]], axis=0)
        raise ValueError(f"Unsupported state shape {state.shape} for expected_state_dim={self.state_dim} "
                         f"under which_arm={self.which_arm}. Check that the deploy layout and the "
                         "checkpoint use the same physical state layout.")

    def convert_action(self, action):
        """``_convert_action``: one model action step -> the robot command."""
        import numpy as np

        action = to_numpy(action).reshape(-1).astype(np.float64)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"the policy returned {action.shape[0]} action dims, expected {self.action_dim}")
        if self.which_arm == "both":
            return action
        full_arm = 2 * self.per_arm
        if action.shape[0] >= full_arm:
            offset = 0 if self.which_arm == "left" else self.per_arm
            return np.concatenate([action[offset:offset + self.per_arm], action[full_arm:]], axis=0)
        if action.shape[0] >= self.per_arm:
            return action
        raise ValueError(f"Unsupported action shape {action.shape} for which_arm={self.which_arm}")


class KuavoOpenpiPi05:
    """kuavo_server adapter body around one ``OpenpiPi05Server`` (or a test double)."""

    name = ADAPTER_NAME

    def __init__(self, server, *, which_arm: str = "both", arm_joint_dim: int = 7,
                 end_effector_dim: int = 1, execution_horizon: int = 0, config_path=None):
        horizon = 50 if execution_horizon in (0, None) else execution_horizon
        if type(horizon) is not int or not 1 <= horizon <= 50:
            raise ValueError("execution_horizon must be 0 (all 50 steps) or 1..50")
        self.server = server
        self.execution_horizon = horizon
        self.config_path = None if config_path is None else str(config_path)
        self.layout = ArmLayout(which_arm=which_arm, state_dim=server.state_dim,
                                action_dim=server.action_dim, arm_joint_dim=arm_joint_dim,
                                end_effector_dim=end_effector_dim)
        self._pending: list = []
        self.last_info: dict = {}

    @classmethod
    def from_config(cls, config_path, *, prompts=(), nonfinite="hold", noise="per_request",
                    seed=None, which_arm="both", arm_joint_dim=7, end_effector_dim=1,
                    execution_horizon=0, log=print):
        from examples.runners.pi05.serving import OpenpiPi05Server, load_example_config

        server = OpenpiPi05Server(load_example_config(config_path), instructions=list(prompts) or None,
                                  nonfinite=nonfinite, noise=noise, seed=seed, log=log)
        try:
            return cls(server, which_arm=which_arm, arm_joint_dim=arm_joint_dim,
                       end_effector_dim=end_effector_dim, execution_horizon=execution_horizon,
                       config_path=Path(config_path).expanduser())
        except BaseException:
            server.close()
            raise

    def _request(self, obs: dict):
        if not isinstance(obs, dict) or HEAD not in obs or STATE not in obs:
            raise ValueError(f"observation needs {HEAD} and {STATE}")
        frames = [as_hwc_uint8(obs[HEAD]), as_hwc_uint8(obs.get(LEFT, obs[HEAD])),
                  as_hwc_uint8(obs.get(RIGHT, obs[HEAD]))]
        state = self.layout.adapt_state(obs[STATE])
        prompt = str(obs.get("prompt", "") or "").strip() or None
        return frames, state, prompt

    def predict_chunk(self, obs: dict):
        """[execution_horizon, robot action dim] float64 robot actions."""
        import numpy as np

        frames, state, prompt = self._request(obs)
        actions, info = self.server.infer(frames, state, prompt)
        self.last_info = info
        chunk = actions[: self.execution_horizon]
        return np.stack([self.layout.convert_action(step) for step in chunk], axis=0)

    def metadata(self) -> dict:
        meta = {"status": "ok", "adapter": self.name, "config": self.config_path,
                "which_arm": self.layout.which_arm, "arm_joint_dim": self.layout.arm_joint_dim,
                "end_effector_dim": self.layout.end_effector_dim,
                "execution_horizon": self.execution_horizon}
        meta.update(self.server.metadata())
        return meta

    def reset(self) -> dict:
        self._pending.clear()
        self.server.reset()
        return {"status": "ok", "message": "adapter state cleared"}

    def select_action(self, obs: dict):
        import numpy as np

        if self._pending:
            return self._pending.pop(0)
        chunk = self.predict_chunk(obs)
        self._pending = [np.asarray(step) for step in chunk[1:]]
        return np.asarray(chunk[0])

    def select_action_chunk(self, obs: dict):
        self._pending.clear()
        return self.predict_chunk(obs)

    def close(self) -> None:
        self.server.close()

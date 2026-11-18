"""openpi pi05 live serving and the kuavo_server adapter (host doubles, no RPU)."""
import ast
from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from examples.runners.pi05 import kuavo
from examples.runners.pi05 import openpi_inputs as openpi
from examples.runners.pi05 import optimized as runner
from examples.runners.pi05 import serving

ROOT = Path(__file__).resolve().parents[1]
DROP_IN = ROOT / "examples/runners/pi05/kuavo_server/adapters/openpi_pi05_rpu.py"
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class Tokenizer:
    """PaliGemma stand-in: BOS, then one id per character (a state bin costs its digit count)."""

    def encode(self, text, add_bos=False):
        return ([2] if add_bos else []) + [3 + ord(char) % 997 for char in text]


# ---------------------------------------------------------------- reachable lengths


def test_reachable_token_counts_cover_every_state_digit_width():
    tokenizer = Tokenizer()
    counts = openpi.reachable_token_counts(tokenizer, "pick", 2)
    base = len(tokenizer.encode(openpi.format_prompt("pick", [0, 0]), add_bos=True))
    # Per dim: "0".."9" +0, "10".."99" and "-1" +1, "100".."255" +2.
    assert counts == list(range(base, base + 5))
    for bins in ([-1, 255], [7, 42], [255, 255]):
        assert len(tokenizer.encode(openpi.format_prompt("pick", bins), add_bos=True)) in counts
    with pytest.raises(ValueError, match="text_tokens=10 is too small"):
        openpi.reachable_token_counts(tokenizer, "pick", 2, text_tokens=10)


def test_reachable_token_counts_refuse_a_non_additive_tokenizer():
    class Merging:
        def encode(self, text, add_bos=False):  # pairs of characters share a token
            return [2] + [0] * ((len(text) + 1) // 2)

    with pytest.raises(RuntimeError, match="not per-dimension additive"):
        openpi.reachable_token_counts(Merging(), "pick", 3)


# ---------------------------------------------------------------- kuavo_server mapping


def test_image_coercion_matches_kuavo_server_openpi():
    chw = np.full((1, 3, 4, 5), 0.5, dtype=np.float32)
    image = kuavo.as_hwc_uint8(chw)
    assert image.shape == (4, 5, 3) and image.dtype == np.uint8 and int(image[0, 0, 0]) == 127
    assert kuavo.as_hwc_uint8(torch.full((4, 5, 3), 9, dtype=torch.uint8)).shape == (4, 5, 3)
    bad = chw.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        kuavo.as_hwc_uint8(bad)
    with pytest.raises(ValueError, match="3 channels"):
        kuavo.as_hwc_uint8(np.zeros((4, 5), dtype=np.uint8))


def test_float_tensor_frames_take_the_torch_path_with_the_same_bytes(monkeypatch):
    # kuavo_deploy sends to_tensor frames: float32 torch [1, 3, H, W] in [0, 1]. The numpy
    # input of the same values is the reference path.
    levels = torch.arange(256, dtype=torch.uint8).repeat(3, 2, 1).float().div(255)  # to_tensor of every level
    rng = np.random.default_rng(0)
    below_one = np.nextafter(np.float32(1), np.float32(0))
    cases = [
        levels[None],
        torch.from_numpy(rng.uniform(-0.3, 1.0, (1, 3, 6, 7)).astype(np.float32)),  # negatives clip to 0
        torch.full((1, 3, 2, 4), float(below_one)),
        torch.tensor([-0.0, 1e-45, 1e-38, 0.5, 1.0, -1e-30]).reshape(1, 3, 1, 2).repeat(1, 1, 2, 2),
        torch.from_numpy(rng.uniform(0, 1, (5, 4, 3)).astype(np.float32)),  # already HWC
        torch.from_numpy(rng.uniform(0, 1, (1, 3, 4, 10)).astype(np.float32))[..., ::2],  # strided
    ]
    seen = []
    original = kuavo._unit_float_tensor_as_uint8
    monkeypatch.setattr(kuavo, "_unit_float_tensor_as_uint8",
                        lambda image: seen.append(original(image)) or seen[-1])
    for case in cases:
        expected = kuavo.as_hwc_uint8(case.numpy())
        actual = kuavo.as_hwc_uint8(case)
        assert seen[-1] is not None and actual.flags.c_contiguous
        assert actual.dtype == np.uint8 and np.array_equal(actual, expected)
    # 0..255 float frames keep the numpy cast; NaN / Inf are refused on both paths
    assert np.array_equal(kuavo.as_hwc_uint8(torch.full((1, 3, 2, 2), 200.7)), np.full((2, 2, 3), 200, np.uint8))
    assert seen[-1] is None
    for value in (float("nan"), float("inf"), float("-inf")):
        bad = torch.full((1, 3, 2, 2), 0.5)
        bad[0, 1, 1, 0] = value
        with pytest.raises(ValueError, match="NaN/Inf"):
            kuavo.as_hwc_uint8(bad)


def test_arm_layout_slices_state_and_actions_like_kuavo_server_openpi():
    robot = np.arange(20, dtype=np.float64)  # [L arm 7, L eef, R arm 7, R eef, lower body 4]
    right = kuavo.ArmLayout(which_arm="right", state_dim=12, action_dim=20)
    assert right.adapt_state(robot).tolist() == list(range(8, 20))
    assert right.convert_action(robot).tolist() == list(range(8, 20))
    left = kuavo.ArmLayout(which_arm="left", state_dim=8, action_dim=8)
    assert left.adapt_state(robot[:16]).tolist() == list(range(8))
    assert left.convert_action(robot[:8]).tolist() == list(range(8))
    both = kuavo.ArmLayout(which_arm="both", state_dim=20, action_dim=20)
    assert both.adapt_state(robot).tolist() == both.convert_action(robot).tolist() == list(range(20))
    with pytest.raises(ValueError, match="Unsupported state shape"):
        both.adapt_state(robot[:16])
    with pytest.raises(ValueError, match="Unsupported state shape"):
        left.adapt_state(robot)  # 4 lower-body values, but the model has none
    with pytest.raises(ValueError, match="returned 7 action dims"):
        left.convert_action(robot[:7])


class _Server:
    state_dim = action_dim = 16

    def __init__(self):
        self.requests, self.resets = [], 0

    def infer(self, frames, state, prompt):
        self.requests.append((frames, state, prompt))
        steps = np.arange(50, dtype=np.float32)[:, None] + np.zeros((1, 16), dtype=np.float32)
        return steps, {"valid_text_tokens": 99}

    def metadata(self):
        return {"precision": "w8a16", "counters": {"requests": len(self.requests)}}

    def reset(self):
        self.resets += 1

    def close(self):
        pass


def _obs(**extra):
    head = np.full((480, 848, 3), 10, dtype=np.uint8)
    obs = {kuavo.HEAD: head, kuavo.STATE: torch.arange(16, dtype=torch.float32), **extra}
    return obs


def test_adapter_body_caches_steps_cuts_the_horizon_and_falls_back_to_the_head_camera():
    server = _Server()
    body = kuavo.KuavoOpenpiPi05(server, which_arm="both", execution_horizon=3)
    first = body.select_action(_obs(prompt=" "))
    frames, state, prompt = server.requests[0]
    assert prompt is None and len(frames) == 3 and all(frame is not None for frame in frames)
    assert all(np.array_equal(frame, frames[0]) for frame in frames)  # both wrists fall back
    assert state.dtype == np.float64 and state.tolist() == list(range(16))
    assert first.tolist() == [0.0] * 16
    assert body.select_action(_obs())[0] == 1.0 and body.select_action(_obs())[0] == 2.0
    assert len(server.requests) == 1
    assert body.select_action(_obs())[0] == 0.0 and len(server.requests) == 2  # horizon 3 used up
    body.select_action(_obs())
    chunk = body.select_action_chunk(_obs(prompt="flip the parcel"))
    assert chunk.shape == (3, 16) and not body._pending and server.requests[-1][2] == "flip the parcel"
    assert body.reset()["status"] == "ok" and server.resets == 1
    meta = body.metadata()
    assert meta["adapter"] == "openpi_pi05_rpu" and meta["precision"] == "w8a16" and meta["execution_horizon"] == 3
    with pytest.raises(ValueError, match="execution_horizon"):
        kuavo.KuavoOpenpiPi05(server, execution_horizon=51)
    with pytest.raises(ValueError, match="observation needs"):
        body.select_action_chunk({kuavo.HEAD: _obs()[kuavo.HEAD]})


# ---------------------------------------------------------------- the serving class


def _base_config(action):
    return {"max_action_dim": 32, "image_resolution": [224, 224], "chunk_size": 50,
            "input_features": {**{f"observation.images.{key}": {"type": "VISUAL", "shape": [3, 224, 224]}
                                  for key in CAMERAS},
                               "observation.state": {"type": "STATE", "shape": [32]}},
            "output_features": {"action": {"type": "ACTION", "shape": [action]}}}


def _config(tmp_path, *, action=3):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text(json.dumps(_base_config(action)))
    frames = []
    for index in range(3):
        path = tmp_path / f"cam{index}.npy"
        np.save(path, np.full((480, 848, 3), 60 * (index + 1), dtype=np.uint8))
        frames.append(str(path))
    stats = {"norm_stats": {
        "state": {"q01": [0.0, -1.0, 0.0], "q99": [1.0, 1.0, 100.0]},
        "actions": {"q01": [-0.1, -0.2, 0.0], "q99": [0.1, 0.2, 100.0]}}}
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    return {"pi05": {"precision": "fp16"}, "cpu": {"threads": 1},
            "input": {"checkpoint": str(checkpoint), "text_tokens": 128, "action_dim": action, "cameras": 3,
                      "images": frames, "frame_size": [848, 480], "instruction": "toml instruction",
                      "state": [0.25, -0.5, 40.0],
                      "norm_stats": str(tmp_path / "norm_stats.json"),
                      "tokenizer": str(tmp_path / "tokenizer.model"), "delta_action_mask": [True, True, False]},
            "run": {"warmup": 1, "runs": 1, "seed": 3}}


@pytest.fixture
def served(monkeypatch):
    """Stub the backend and the cold selectors; the policy double returns ``outputs.pop(0)``."""
    calls = {"prepare": [], "predict": [], "closed": 0, "outputs": [], "fail_prepare": False, "loaded": []}

    class GraphRuntimePolicy:
        @classmethod
        def from_environment(cls, **options):
            return SimpleNamespace(**options)

    class Policy:
        _lerobot_policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, image_features={}))
        graph_arena_count = staticmethod(lambda counts=1: 4 + counts)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["loaded"].append(kwargs)
            return cls()

        def to(self, device):
            pass

        def prepare_graphs(self, batch, **kwargs):
            calls["prepare"].append((batch, kwargs))
            if calls["fail_prepare"]:
                raise RuntimeError("prepare failed")
            return {"phase": "READY", "graph_counts": {"denoise": len(kwargs["language_token_counts"])},
                    "language_token_counts": list(kwargs["language_token_counts"])}

        def predict_action_chunk(self, batch, *, num_steps, noise):
            calls["predict"].append((batch, noise.clone()))
            return calls["outputs"].pop(0) if calls["outputs"] else torch.zeros(1, 50, 3)

        def close(self):
            calls["closed"] += 1

    stubs = {"rpu_backend": ModuleType("rpu_backend"), "rpu_backend.api": SimpleNamespace(Pi05Policy=Policy),
             "rpu_backend.graph": SimpleNamespace(GraphRuntimePolicy=GraphRuntimePolicy),
             "rpu_backend.model_registry": SimpleNamespace(model_path=lambda alias: Path(alias))}
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(openpi, "load_tokenizer", lambda path: Tokenizer())
    monkeypatch.setattr(runner, "bind_cold_environment", lambda env: None)
    monkeypatch.setattr(runner, "profile_libraries", lambda: ({}, {}))
    monkeypatch.setattr(runner, "runtime_payloads", lambda libraries: {})
    monkeypatch.setattr(runner, "configure_text_profile", lambda *args: {})
    monkeypatch.setattr(runner, "validate_camera_batch", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    monkeypatch.setitem(runner._CPU, "configure_cpu_threads", lambda threads: {"requested_threads": threads})
    return calls


def _frames(width=848):
    return [np.full((480, width, 3), 30 * index, dtype=np.uint8) for index in range(3)]


def test_server_prepares_every_reachable_length_and_returns_physical_actions(served, tmp_path):
    logs = []
    server = serving.OpenpiPi05Server(_config(tmp_path), instructions=["flip it", "flip it", "stack"],
                                      log=logs.append)
    tokenizer = Tokenizer()
    expected = sorted(set(openpi.reachable_token_counts(tokenizer, "flip it", 3))
                      | set(openpi.reachable_token_counts(tokenizer, "stack", 3)))
    batch, kwargs = served["prepare"][0]
    assert kwargs["language_token_counts"] == tuple(expected) and kwargs["num_steps"] == 10
    # The cold arena plan holds one denoise graph per prepared length.
    assert served["loaded"][0]["graph_runtime_policy"].graph_arena_count == 4 + len(expected)
    assert server.instructions == ("flip it", "stack")
    # The warmup request carries the first instruction, so its length is prepared.
    assert int(batch["observation.language.attention_mask"].sum()) in expected
    assert server.metadata()["prepared"]["graph_counts"] == {"denoise": len(expected)}
    assert "READY" in logs[0]

    served["outputs"] += [torch.zeros(1, 50, 3), torch.zeros(1, 50, 3)]
    state = [0.5, 0.0, 10.0]
    actions, info = server.infer(_frames(), state)
    assert actions.dtype == np.float32 and actions.shape == (50, 3) and not info["guarded"]
    # Unnormalize(0) = midpoint, plus the state on the two delta dims.
    assert np.allclose(actions[0], [0.5 + 0.0, 0.0 + 0.0, 50.0], atol=1e-5)
    second, _ = server.infer(_frames(), state, "stack")
    first_noise, second_noise = served["predict"][0][1], served["predict"][1][1]
    assert first_noise.shape == (1, 50, 32) and not torch.equal(first_noise, second_noise)
    server.reset()
    server.infer(_frames(), state)
    assert torch.equal(served["predict"][2][1], first_noise)  # reset restarts the noise sequence
    server.close()
    assert served["closed"] == 1


def test_server_guards_nonfinite_outputs_and_refuses_bad_requests(served, tmp_path):
    logs = []
    server = serving.OpenpiPi05Server(_config(tmp_path), log=logs.append)
    state = [0.5, 0.0, 10.0]
    nan = torch.zeros(1, 50, 3)
    nan[0, 7, 1] = float("nan")
    served["outputs"].append(nan)
    actions, info = server.infer(_frames(), state)
    assert info["guarded"] and np.array_equal(actions, np.tile(np.float32(state), (50, 1)))
    assert server.counters["nonfinite_outputs"] == 1 and "hold chunk" in logs[-1]
    # A finite normalized output that overflows physically is also held.
    served["outputs"].append(torch.full((1, 50, 3), 3e38))
    huge_stats = server.stats["actions"]
    server.stats["actions"] = {"q01": [0.0] * 3, "q99": [1e300] * 3}
    assert server.infer(_frames(), state)[1]["guarded"]
    server.stats["actions"] = huge_stats

    predicted = len(served["predict"])
    for bad, message in (([0.5, float("inf"), 1.0], "NaN/Inf"), ([0.5, 0.0], "expects 3")):
        with pytest.raises(ValueError, match=message):
            server.infer(_frames(), bad)
    with pytest.raises(ValueError, match="valid language tokens"):
        server.infer(_frames(), state, "a much longer instruction than any prepared one")
    with pytest.raises(ValueError, match="uint8"):
        server.infer([frame.astype(np.float32) for frame in _frames()], state)
    # The server takes the camera's own size only (a 640x480 simulation frame to an 848x480 server).
    with pytest.raises(ValueError, match=r"\[480, 848, 3\] camera frames"):
        server.infer(_frames(640), state)
    assert len(served["predict"]) == predicted and server.counters["refused"] == 5

    server.nonfinite = "error"
    served["outputs"].append(nan)
    with pytest.raises(serving.NonFiniteActionError):
        server.infer(_frames(), state)
    assert server.counters["nonfinite_outputs"] == 3


def test_server_refusals_at_startup(served, tmp_path):
    with pytest.raises(ValueError, match="hold chunk repeats the state"):
        serving.OpenpiPi05Server(_config(tmp_path, action=4))
    assert not served["prepare"]
    serving.OpenpiPi05Server(_config(tmp_path, action=4), nonfinite="error").close()
    served["fail_prepare"] = True
    closed = served["closed"]
    with pytest.raises(RuntimeError, match="prepare failed"):
        serving.OpenpiPi05Server(_config(tmp_path))
    assert served["closed"] == closed + 1
    config = _config(tmp_path)
    config["input"]["noise"] = "/noise.pt"
    with pytest.raises(ValueError, match="own noise"):
        serving.OpenpiPi05Server(config)
    with pytest.raises(ValueError, match="nonfinite"):
        serving.OpenpiPi05Server(_config(tmp_path), nonfinite="ignore")


# ---------------------------------------------------------------- the drop-in adapter file


def _stub_kuavo_server(tmp_path):
    """A minimal kuavo_server package with the customer's adapter contract."""
    package = tmp_path / "kuavo_server"
    (package / "adapters").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "runtime.py").write_text(
        "_ADAPTER_REGISTRY = {}\n"
        "def register_adapter(cls):\n"
        "    _ADAPTER_REGISTRY[cls.name] = cls\n"
        "    return cls\n")
    (package / "adapters/__init__.py").write_text("")
    (package / "adapters/base.py").write_text(
        "from abc import ABC, abstractmethod\n"
        "class ModelServerAdapter(ABC):\n"
        "    name = ''\n"
        "    @classmethod\n"
        "    def add_cli_args(cls, parser):\n"
        "        pass\n"
        "    @classmethod\n"
        "    @abstractmethod\n"
        "    def from_args(cls, args):\n"
        "        raise NotImplementedError\n"
        "    @abstractmethod\n"
        "    def select_action(self, obs):\n"
        "        raise NotImplementedError\n")
    shutil.copy(DROP_IN, package / "adapters/openpi_pi05_rpu.py")


@pytest.fixture
def drop_in(tmp_path, monkeypatch):
    _stub_kuavo_server(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [name for name in sys.modules if name.split(".")[0] == "kuavo_server"]:
        monkeypatch.delitem(sys.modules, name)
    import importlib

    module = importlib.import_module("kuavo_server.adapters.openpi_pi05_rpu")
    yield module
    for name in [name for name in sys.modules if name.split(".")[0] == "kuavo_server"]:
        sys.modules.pop(name, None)


def test_drop_in_registers_parses_and_delegates(drop_in, tmp_path, monkeypatch):
    registry = sys.modules["kuavo_server.runtime"]._ADAPTER_REGISTRY
    adapter_cls = registry["openpi_pi05_rpu"]
    # serve.py's base arguments plus the adapter's, as build_full_parser composes them.
    parser = ArgumentParser()
    for flag in ("--adapter", "--host", "--api_token"):
        parser.add_argument(flag)
    parser.add_argument("--port", type=int)
    adapter_cls.add_cli_args(parser)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("stack the boxes\n\n  sort parcels \n")
    args = parser.parse_args(["--adapter", "openpi_pi05_rpu", "--rpu_backend_root", str(ROOT),
                              "--rpu_config", "/cfg.toml", "--prompt", "flip the parcel",
                              "--prompts_file", str(prompts), "--which_arm", "right",
                              "--execution_horizon", "8", "--nonfinite", "error", "--seed", "4"])
    seen = {}

    def from_config(config, **kwargs):
        seen.update(config=config, **kwargs)
        return SimpleNamespace(metadata=lambda: {"adapter": "openpi_pi05_rpu"},
                               reset=lambda: {"status": "ok"}, select_action=lambda obs: ("step", obs),
                               select_action_chunk=lambda obs: ("chunk", obs))

    monkeypatch.setattr(kuavo.KuavoOpenpiPi05, "from_config", staticmethod(from_config))
    adapter = adapter_cls.from_args(args)
    assert seen == {"config": "/cfg.toml", "prompts": ["flip the parcel", "stack the boxes", "sort parcels"],
                    "nonfinite": "error", "noise": "per_request", "seed": 4, "which_arm": "right",
                    "arm_joint_dim": 7, "end_effector_dim": 1, "execution_horizon": 8}
    assert adapter.select_action({"x": 1}) == ("step", {"x": 1})
    assert adapter.select_action_chunk({}) == ("chunk", {})
    assert adapter.reset() == {"status": "ok"} and adapter.metadata()["adapter"] == "openpi_pi05_rpu"
    monkeypatch.delenv("RPU_BACKEND_ROOT", raising=False)
    args.rpu_backend_root = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="rpu_backend_root"):
        adapter_cls.from_args(args)


def test_drop_in_imports_only_the_standard_library_and_kuavo_server():
    # The RPU process binds CPU threads and cold selectors before torch loads.
    tree = ast.parse(DROP_IN.read_text(encoding="utf-8"))
    absolute = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    absolute |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module}
    assert absolute <= {"__future__", "os", "sys", "argparse", "pathlib", "typing", "examples"}

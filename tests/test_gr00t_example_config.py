"""Public GR00T config/input boundaries, without initializing the device."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest
import torch
from _example_toml import load_example_toml


ROOT = Path(__file__).resolve().parents[1]
CFG = importlib.import_module("examples.runners.gr00t.config")


def config(precision="fp16", size=224, text=32, cameras=3):
    path = ROOT / f"examples/configs/gr00t/n1_7_droid/{size}/{precision}.toml"
    result = load_example_toml(path)
    result["input"]["text_tokens"] = text
    result["input"]["cameras"] = cameras
    return result


class Tokenizer:
    all_special_ids = [151644, 151645, 151652, 151653, 151655]

    def encode(self, value, **kwargs):
        return {"<|im_start|>user\n": [151644, 872, 198],
                "<|vision_start|>": [151652], "<|vision_end|>": [151653],
                "<|im_end|>\n<|im_start|>assistant\n": [151645, 198, 151644, 77091, 198]}[value]

    def convert_tokens_to_ids(self, value):
        assert value == "<|image_pad|>"
        return 151655


def observation(size=224, text=32, cameras=3):
    edge = size // 16
    image = [151652] + [151655] * (edge // 2) ** 2 + [151653]
    ids = torch.tensor([[151644, 872, 198] + image * cameras + [1234] * text
                        + [151645, 198, 151644, 77091, 198]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                mm_token_type_ids=(ids == 151655).long(),
                image_grid_thw=torch.tensor([[1, edge, edge]] * cameras),
                pixel_values=torch.zeros(cameras * edge * edge, 1536),
                state=torch.zeros(1, 1, 132), embodiment_id=torch.tensor([20]))


@pytest.mark.parametrize("size", [224, 256])
@pytest.mark.parametrize("text", [32, 64, 128])
def test_actual_input_geometry_and_instruction_length(size, text):
    cfg = config(size=size, text=text)
    value = observation(size, text)
    CFG.validate_observation(cfg, value, Tokenizer())
    cfg["input"]["text_tokens"] = 64 if text == 32 else 32
    with pytest.raises(ValueError, match="instruction tokens"):
        CFG.validate_observation(cfg, value, Tokenizer())


@pytest.mark.parametrize("precision", ["fp16", "w8a16"])
@pytest.mark.parametrize("size", [224, 256])
@pytest.mark.parametrize("cameras", [1, 2, 3])
@pytest.mark.parametrize("text", [32, 64, 128, 160, 192, 224, 256])
def test_camera_and_long_instruction_profiles(precision, size, cameras, text):
    cfg = config(precision, size=size, text=text, cameras=cameras)
    assert CFG.validate_config(cfg)
    value = observation(size, text, cameras)
    CFG.validate_observation(cfg, value, Tokenizer())
    # Only camera/text geometry changes: retain the existing cold optimization
    # policy for both public precisions.
    assert CFG.runtime_environment(cfg) == CFG.runtime_environment(config(precision, size=256))


@pytest.mark.parametrize("size,cameras,text", [
    (256, 2, 129), (256, 2, 288), (256, 2, True),
    (256, 4, 128), (256, True, 128), (224, 0, 128), (224, 4, 256),
    (224, 2, 129), (224, 3, 288), (224, True, 128),
])
def test_geometry_outside_requested_profiles_is_rejected(size, cameras, text):
    with pytest.raises(ValueError, match="GR00T input"):
        CFG.validate_config(config(size=size, cameras=cameras, text=text))


@pytest.mark.parametrize("field", ["image_grid_thw", "pixel_values", "input_ids"])
@pytest.mark.parametrize("size", [224, 256])
def test_camera_count_is_checked_against_each_observation_field(field, size):
    cfg = config(size=size, text=256, cameras=2)
    value = observation(size, 256, 2)
    value[field] = observation(size, 256, 3)[field]
    if field == "input_ids":
        value["attention_mask"] = torch.ones_like(value[field])
    with pytest.raises(ValueError, match="GR00T observation"):
        CFG.validate_observation(cfg, value, Tokenizer())


@pytest.mark.parametrize("size,cameras,text,capacity", [
    (224, 2, 128, 256), (224, 2, 256, 368), (224, 3, 256, 432),
    (256, 2, 256, 400), (256, 3, 256, 464),
])
def test_runner_sizes_capacity_from_real_multimodal_prefix(monkeypatch, size, cameras, text, capacity):
    from examples.runners.gr00t import runner as public_runner

    cfg = config(size=size, cameras=cameras, text=text)
    value = observation(size, text, cameras)
    installs = []
    runtime = SimpleNamespace(get_action=lambda *args, **kwargs: None)

    def build(*args, **kwargs):
        installs.append(kwargs)
        return runtime

    monkeypatch.setitem(sys.modules, "rpu_backend.adapters.gr00t",
                        SimpleNamespace(build_gr00t_vla=build))
    monkeypatch.setitem(sys.modules, "rpu_backend.graph", SimpleNamespace(
        GraphRuntimePolicy=SimpleNamespace(from_environment=lambda **kwargs:
            SimpleNamespace(prepare_arenas=lambda: True))))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: Tokenizer())))
    monkeypatch.setattr(torch, "rpu", SimpleNamespace(
        set_ddr_flush=lambda value: None, set_ddr_flush_force=lambda value: None), raising=False)
    monkeypatch.setattr(public_runner, "_load_mapping", lambda *args: value)
    monkeypatch.setattr(public_runner, "_model_path", lambda *args, **kwargs: Path("/qwen"))
    monkeypatch.setattr(public_runner, "_checkpoint", lambda *args: Path("/gr00t"))
    caller = SimpleNamespace(source_config=cfg)
    assert callable(public_runner.prepare(caller))
    assert caller.owner is runtime
    assert installs[0]["max_prefix_len"] == capacity
    cfg["input"]["max_prefix_len"] = (value["input_ids"].shape[-1] - 1) // 16 * 16
    with pytest.raises(ValueError, match="must cover the observation"):
        public_runner.prepare(caller)
    assert len(installs) == 1


@pytest.mark.parametrize("field,change", [
    ("image_grid_thw", lambda x: torch.tensor([[1, 16, 16]] * 3)),
    ("pixel_values", lambda x: x[:196]),
    ("pixel_values", lambda x: x.fill_(float("nan"))),
    ("attention_mask", lambda x: x.zero_()),
    ("mm_token_type_ids", lambda x: x.zero_()),
    ("embodiment_id", lambda x: torch.tensor([21])),
    ("input_ids", lambda x: x.float()),
])
def test_mislabelled_or_invalid_observations_are_rejected(field, change):
    value = observation()
    value[field] = change(value[field])
    with pytest.raises(ValueError, match="GR00T observation"):
        CFG.validate_observation(config(), value, Tokenizer())


@pytest.mark.parametrize("precision", ["fp16", "w8a16"])
def test_precision_selects_cold_environment_and_conflicts_do_not_partially_apply(monkeypatch, precision):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    cfg = config(precision)
    env = CFG.runtime_environment(cfg)
    for key in env:
        monkeypatch.delenv(key, raising=False)
    assert not any(name.startswith("RPU_KERNEL_LIB_PATH") for name in env)
    assert env["RPU_GR00T_FF1_RESIDENT_W8"] == ("1" if precision == "w8a16" else "0")
    monkeypatch.setenv("RPU_GR00T_CROSS_KV_REUSE", "0")
    before = dict(os.environ)
    with pytest.raises(ValueError, match="RPU_GR00T_CROSS_KV_REUSE"):
        CFG.configure_environment(cfg)
    assert dict(os.environ) == before
    monkeypatch.delenv("RPU_GR00T_CROSS_KV_REUSE")
    assert CFG.configure_environment(cfg) == env
    assert all(os.environ[key] == value for key, value in env.items())


def test_config_checks_are_device_free_and_reject_incomplete_geometry(tmp_path):
    code = """
import sys
sys.path.insert(0, 'examples')
from examples.common.config import load_config
from pathlib import Path
for path in Path('examples/configs/gr00t').rglob('*.toml'):
    load_config(path)
assert 'torch' not in sys.modules and 'rpu_backend' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    cfg = config()
    del cfg["input"]["text_tokens"]
    with pytest.raises(ValueError, match="text_tokens"):
        CFG.validate_config(cfg)
    assert CFG.runtime_environment({"example": {}, "input": {"observation": "legacy.pt"}}) == {}


def test_public_runner_sets_optimizations_before_backend_import(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    cfg = config("w8a16")
    for key in CFG.runtime_environment(cfg):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from examples.common import measurement as _runner
    import builtins
    original = builtins.__import__

    def intercept(name, *args, **kwargs):
        if name == "rpu_backend":
            assert os.environ["RPU_GR00T_PRECOMPUTE_ADALN"] == "1"
            assert os.environ["RPU_GR00T_FF1_RESIDENT_W8"] == "1"
            assert os.environ["RPU_KERNEL_LIB_PATH_EXPANSION"] == "/runtime/rhinoExpansionOpLib_current.ref"
            raise RuntimeError("cold environment checked before backend import")
        return original(name, *args, **kwargs)

    # Runtime installation owns both roles; model configuration preserves them.
    monkeypatch.setenv("RPU_KERNEL_LIB_PATH_EXPANSION", "/runtime/rhinoExpansionOpLib_current.ref")
    # Avoid changing this test worker's thread count.
    cfg.pop("cpu")
    monkeypatch.setattr(builtins, "__import__", intercept)
    with pytest.raises(RuntimeError, match="cold environment checked"):
        _runner.run(cfg)


def test_model_specific_ref_selector_is_rejected_before_backend_import(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from examples.common.config import load_config, ConfigError
    source = ROOT / "examples/configs/gr00t/n1_7_droid/224/fp16.toml"
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(source.read_text().replace("[model]", '[model]\nexpansion_ref = "/old/gr00t.ref"'))
    with pytest.raises(ConfigError, match="expansion_ref is retired"):
        load_config(legacy)
    cfg = config()
    cfg["example"]["expansion_ref"] = "/old/gr00t.ref"
    with pytest.raises(ValueError, match="expansion_ref is retired"):
        CFG.configure_environment(cfg)

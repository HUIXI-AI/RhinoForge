"""HyVLA's shared example entry, real CPU input envelope and retained timing loop."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "examples/configs/hy_vla"
CONFIG_FILENAMES = {
    "fp16": "fp16", "w8a16": "w8a16", "w8a16_nvfp4": "w8a16_action_nvfp4",
    "w4a16": "w4a16_rtn_g32_vision_w8a16", "w4a16_vision": "w4a16_rtn_g32",
}


def load(relative):
    return importlib.import_module(relative[:-3].replace("/", "."))


def template(name, cameras=3):
    """A HyVLA template (3 cameras); cameras=1/2 apply preset 1cam_t32/2cam_t32 of examples/inputs/vla.toml."""
    return load("examples/common/config.py").load_config(
        CONFIGS / f"{name}.toml", input_preset=None if cameras == 3 else f"{cameras}cam_t32")


def test_configs_route_through_shared_entry_without_inheriting_certification(monkeypatch, capsys):
    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    seen = []
    monkeypatch.setattr(runner, "run_example_config", lambda config: seen.append(config) or 0)
    monkeypatch.setattr(importlib.import_module("examples.common.cli"), "_hyvla_runner", lambda: runner)
    paths = sorted(CONFIGS.rglob("*.toml"))
    expected = {(cameras, precision) for cameras in (2, 3) for precision in runner.PRECISIONS}
    assert len(paths) == len(runner.PRECISIONS)
    for path, cameras in ((path, cameras) for path in paths for cameras in (2, 3)):
        config = template(path.stem, cameras)
        args = runner.example_arguments(config)
        assert args.cpu_threads == 12 and args.cameras == cameras
        assert path == CONFIGS / f"{CONFIG_FILENAMES[args.precision]}.toml"
        assert args.output_dir == Path(f"./hyvla_results/{args.precision}")
        assert args.prefix_len == runner.prefix_rows(args.cameras, args.text_tokens)
        assert args.source == "synthetic" and args.batch is None
        assert common.example_metadata(config) is None
        env = runner.profile_environment(args)
        assert env["RPU_HY_VLA_W4A16"] == {"w4a16": "vlm,expert",
                                           "w4a16_vision": "vit,vlm,expert"}.get(args.precision, "0")
        assert env["RPU_HY_VLA_EXPERT_RESIDENT_FP8_SCALE_LAYERS"] == "0"
        assert args.rmsnorm_vwarp == env["RPU_RMSNORM_VWARP"] == "auto"
        if args.precision == "w8a16_nvfp4":
            assert env["RPU_HY_VLA_W8A16"] == "vit,vlm"
            assert env["RPU_HY_VLA_EXPERT_NVFP4"] == "1"
        elif args.precision == "w4a16":
            assert env["RPU_HY_VLA_W8A16"] == "vit"
            assert env["RPU_HY_VLA_EXPERT_GATEUP_RESIDENT"] == "0"
            assert env["RPU_HY_VLA_EXPERT_NVFP4"] == "0"
        elif args.precision == "w4a16_vision":
            assert env["RPU_HY_VLA_W8A16"] == "0"
            assert env["RPU_HY_VLA_EXPERT_GATEUP_RESIDENT"] == "0"
            assert env["RPU_HY_VLA_VISION_KV_SPM_ONLY"] == "0"
        else:
            assert env["RPU_HY_VLA_EXPERT_NVFP4"] == "0"
        monkeypatch.setattr(sys, "argv", ["hy_vla.py", "--config", str(path), "--input", f"{cameras}cam_t32"])
        assert importlib.import_module("examples.common.cli").main() == 0
    assert len(seen) == len(expected)
    assert {(item["input"]["cameras"], item["hy_vla"]["precision"]) for item in seen} == expected
    monkeypatch.setattr(sys, "argv", ["hy_vla.py", "--config", str(path), "--check-config"])
    assert importlib.import_module("examples.common.cli").main() == 0
    assert "syntax/input envelope only" in capsys.readouterr().out
    # The original fixture remains explicitly selectable with its own metadata.
    assert common.example_metadata(common.load_config(ROOT / "examples/configs/hy_embodied/umi/fp16.toml")) is not None


@pytest.mark.parametrize('fields, threads', [
    ({'cpu': {'threads': 3}}, 3),
    ({'run': {'cpu_threads': 4}}, 4),
    ({'run': {'torch_num_threads': 8}}, 8),
    ({'cpu': {'threads': 12}, 'run': {'cpu_threads': 12, 'torch_num_threads': 12}}, 12),
])
def test_public_cpu_setting_changes_only_thread_choice(fields, threads):
    common = load('examples/common/config.py')
    runner = load('examples/runners/hy_vla/optimized.py')
    config = common.load_config(CONFIGS / 'w8a16_action_nvfp4.toml')
    for section, values in fields.items():
        config.setdefault(section, {}).update(values)
    before = deepcopy(config)
    args = runner.example_arguments(config)
    assert args.cpu_threads == threads
    assert args.precision == 'w8a16_nvfp4' and args.cameras == 3
    assert args.prefix_len == runner.prefix_rows(3, config['input']['text_tokens'])
    assert config == before


@pytest.mark.parametrize('fields', [
    {'cpu': {'threads': 0}}, {'cpu': {'threads': True}}, {'cpu': {'threads': 3, 'affinity': '0-2'}},
    {'cpu': {'threads': 3}, 'run': {'cpu_threads': 12}},
])
def test_invalid_cpu_setting_fails_before_cold_binding(fields, monkeypatch):
    common = load('examples/common/config.py')
    runner = load('examples/runners/hy_vla/optimized.py')
    config = common.load_config(CONFIGS / 'fp16.toml')
    for section, values in fields.items():
        config.setdefault(section, {}).update(values)
    monkeypatch.setattr(runner, 'bind_libraries', lambda args: pytest.fail('CPU admission reached cold binding'))
    with pytest.raises(ValueError):
        runner.run_example_config(config)


@pytest.mark.parametrize("section,key,value", [
    ("input", "text_tokens", 96), ("input", "text_tokens", 0), ("input", "text_tokens", True),
    ("input", "cameras", 0), ("input", "cameras", 4),
    ("input", "prefix_len", 128), ("run", "warmup", 2),
    ("input", "num_steps", 5), ("hy_vla", "precision", "nvfp4"),
    ("run", "kernel_count", 42), ("libraries", "main", "/main.ref"),
])
def test_invalid_geometry_and_incomplete_artifacts_fail_before_load(section, key, value):
    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    config = common.load_config(CONFIGS / "w8a16_action_nvfp4.toml")
    if section == "libraries":
        # Artifact paths are caller-filled; the three roles remain atomic.
        config.pop("libraries", None)
    config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        runner.example_arguments(config)


@pytest.mark.parametrize("precision", ["fp16", "w8a16", "w8a16_nvfp4", "w8a16_action_nvfp4"])
@pytest.mark.parametrize("cameras", [1, 2, 3])
def test_prefix_admission_matches_resident_action_precision(precision, cameras):
    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    config = template("w8a16_action_nvfp4", cameras)
    config["hy_vla"]["precision"] = precision
    maximum = {1: 128, 2: 192, 3: 240}[cameras]
    for text in (1, 7, 16, 32, 48, 64):
        config["input"]["text_tokens"] = text
        # The action expert's admitted floor is 128 rows; one camera runs at it.
        minimum = max(128, ((2 + cameras * 58 + text + 15) // 16) * 16)
        for prefix in ("auto", minimum, maximum):
            config["input"]["prefix_len"] = prefix
            args = runner.example_arguments(config)
            assert args.prefix_len == (minimum if prefix == "auto" else prefix)
            assert args.precision == runner.canonical_precision(precision)
        for prefix in (minimum - 16, maximum + 16, minimum + 1):
            config["input"]["prefix_len"] = prefix
            with pytest.raises(ValueError, match="covering"):
                runner.example_arguments(config)


def test_shared_core_table_maps_to_real_public_hyvla_execution_scopes():
    from rpu_backend.api._execution import normalize_rpu_execution
    from rpu_backend.adapters.hy_vla.runtime import (
        HY_VLA_EXECUTION_COMPONENTS, resolve_hy_vla_execution,
        _validate_hy_vla_execution_geometry,
    )
    from rpu_backend.adapters.hy_vla.weights import HyVlaConfig

    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    for path in sorted(CONFIGS.rglob("*.toml")):
        original = common.load_config(path)
        for explicit in (False, True):
            config = deepcopy(original)
            if explicit:
                config["rpu_execution"] = {"model": {"num_cores": 8}}
            else:
                config.pop("rpu_execution", None)
            args = runner.example_arguments(config)
            normalized = normalize_rpu_execution(args.rpu_execution,
                entry_point="HyEmbodiedPolicy.from_checkpoint",
                supported_components=HY_VLA_EXECUTION_COMPONENTS)
            root, children = resolve_hy_vla_execution(normalized, entry_point="config runner")
            assert root == {stage: {"chunk_size": "auto"} for stage in ("vision", "prefill", "action")}
            assert children == {
                "vision_encoder": {"vision": {"chunk_size": "auto"}},
                "language_model": {"prefill": {"chunk_size": "auto"}},
                "action_expert": {"action": {"chunk_size": "auto"}},
            }
            _validate_hy_vla_execution_geometry(children, cfg=HyVlaConfig(num_cameras=args.cameras),
                prefix_len=args.prefix_len, packed_vision=True, entry_point="config runner")
            env = runner.profile_environment(args)
            assert env["RPU_HY_VLA_ATTN_TP8"] == env["RPU_HY_VLA_ACTION_MLP_MC"] == "1"
            # 4/6 cores: the VLM takes the budget, the other children four lanes.
            for cores in (4, 6):
                config["rpu_execution"] = {"model": {"num_cores": cores}}
                args = runner.example_arguments(config)
                _root, children = resolve_hy_vla_execution(args.rpu_execution, entry_point="config runner")
                assert {component: child["model"]["num_cores"] for component, child in children.items()} == {
                    "vision_encoder": 4, "language_model": cores, "action_expert": 4}
                assert runner.profile_environment(args)["RPU_HY_VLA_ATTN_TP8"] == "0"


@pytest.mark.parametrize("layers",[4,8])
def test_resident_fp8_scale_config_binds_exact_cold_envelope(layers):
    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    config = common.load_config(CONFIGS / "w8a16_action_nvfp4.toml")
    config["hy_vla"]["resident_fp8_scale_layers"] = layers
    args = runner.example_arguments(config)
    env = runner.profile_environment(args)
    assert env["RPU_HY_VLA_EXPERT_RESIDENT_FP8_SCALE_LAYERS"] == str(layers)
    assert env["RPU_HY_VLA_EXPERT_GATEUP_RESIDENT_LAYERS"] == "24"
    assert env["RPU_HY_VLA_EXPERT_O_RESIDENT_LAYERS"] == "0"
    for value in (True, "4", "8", 1, 3, 5, 7, 16):
        invalid = deepcopy(config)
        invalid["hy_vla"]["resident_fp8_scale_layers"] = value
        with pytest.raises(ValueError, match="resident_fp8_scale_layers"):
            runner.example_arguments(invalid)
    for precision in ("fp16", "w8a16"):
        invalid = deepcopy(config)
        invalid["hy_vla"]["precision"] = precision
        with pytest.raises(ValueError, match="requires Action NVFP4"):
            runner.example_arguments(invalid)
    config["input"]["text_tokens"] = 16
    config["input"]["prefix_len"] = "auto"
    assert runner.example_arguments(config).prefix_len == 192
    config["input"]["prefix_len"] = 240
    assert runner.example_arguments(config).resident_fp8_scale_layers == layers


def test_rmsnorm_config_routes_to_real_cold_plan_and_rejects_process_conflicts(monkeypatch):
    from rpu_backend.adapters.hy_vla import runtime
    from test_hyvla_lazy_weights import _clear_hyvla_cold_environment

    _clear_hyvla_cold_environment(monkeypatch)
    monkeypatch.delenv("RPU_CHUNK_FORCE_UNSAFE", raising=False)
    monkeypatch.delitem(sys.modules, "rpu_backend", raising=False)
    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    # Successful cold binding mutates environ directly; keep every write local
    # to this test, including keys that were absent before the call.
    monkeypatch.setattr(runner.os, "environ", dict(runner.os.environ))
    seen = []
    monkeypatch.setattr(runner, "run_example_config", lambda config: seen.append(config) or 0)
    monkeypatch.setattr(importlib.import_module("examples.common.cli"), "_hyvla_runner", lambda: runner)
    expert = None
    for text in (16, 48):
        path = CONFIGS / "w8a16_action_nvfp4.toml"
        for mode, capability in (("auto", 7), ("16", 3)):
            config = common.load_config(path)
            config["input"]["text_tokens"] = text
            config["hy_vla"]["rmsnorm_vwarp"] = mode
            config["hy_vla"]["resident_fp8_scale_layers"] = 8
            args = runner.example_arguments(config)
            assert args.prefix_len == 176 + text
            env = runner.profile_environment(args)
            assert env["RPU_RMSNORM_VWARP"] == mode
            assert env["RPU_HY_VLA_EXPERT_RESIDENT_FP8_SCALE_LAYERS"] == "8"
            monkeypatch.setenv("RPU_RMSNORM_VWARP", "16" if mode == "auto" else "auto")
            with pytest.raises(ValueError, match="conflicting cold environment:.*RPU_RMSNORM_VWARP"):
                runner.bind_cold_environment(env)
            monkeypatch.setenv("RPU_RMSNORM_VWARP", mode)
            runner.bind_cold_environment(env)
            plan = runtime.prepare_hy_vla_cold_plan(env)
            assert plan.native.vlm.rmsnorm_capability == capability
            assert dict(plan.default_environment)["RPU_RMSNORM_VWARP"] == mode
            if expert is None:
                expert = plan.native.expert
            assert plan.native.expert == expert
            # The common entry passes this config to the same model runner.
            with monkeypatch.context() as routed:
                routed.setattr(importlib.import_module("examples.common.cli"), "load_config", lambda _, **__: config)
                routed.setattr(sys, "argv", ["hy_vla.py", "--config", str(path)])
                assert importlib.import_module("examples.common.cli").main() == 0
            assert seen[-1]["hy_vla"]["rmsnorm_vwarp"] == mode
    for value in (16, 32, True, None, "32", "0", "AUTO", "off", " 16"):
        config["hy_vla"]["rmsnorm_vwarp"] = value
        with pytest.raises(ValueError, match="rmsnorm_vwarp"):
            runner.example_arguments(config)


class TinyTokenizer:
    """A deterministic whitespace tokenizer for exercising production masks."""
    pad_token_id = 0
    def __call__(self, texts, *, max_length, **kwargs):
        words = texts[0].replace("<｜hy_Assistant｜>", " <｜hy_Assistant｜>").split()
        ids = [sum(map(ord, word)) for word in words][:max_length]
        return {"input_ids": torch.tensor([ids + [0] * (max_length-len(ids))]),
                "attention_mask": torch.tensor([[1]*len(ids)+[0]*(max_length-len(ids))])}


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("text", [1, 7, 16, 32, 48, 64])
def test_synthetic_input_uses_real_tokenizer_masks_and_config_geometry(cameras, text):
    from rpu_backend.adapters.hy_vla.preprocess import prepare_language
    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    config = template("w8a16_action_nvfp4", cameras)
    config["input"]["text_tokens"] = text
    args = runner.example_arguments(config)
    tokenizer = TinyTokenizer()
    batch = runner.make_synthetic_batch(args, tokenizer)
    obs = runner.validate_observation(batch, args)
    runner.validate_text_tokens(batch, tokenizer, args)
    short, _ = prepare_language(args.instruction, tokenizer, max_length=text)
    assert torch.equal(obs["lang_tokens"][:text], short[0])
    assert torch.equal(obs["lang_tokens"][text:], torch.full((64-text,), tokenizer.pad_token_id))
    assert obs["n_valid"] == 2 + cameras * 58 + text
    full_rows = 192 if cameras == 2 else 240
    assert obs["prefill_mask"].shape == (full_rows, full_rows)
    assert obs["denoise_mask"].shape == (51, full_rows + 51)
    assert not obs["prefill_mask"][obs["n_valid"]:].any()
    assert not obs["denoise_mask"][:, obs["n_valid"]:full_rows].any()
    assert len(obs["images"]) == cameras
    assert batch["source"]["actual_token_ids"] == short[0].tolist()
    assert batch["source"]["prompt_truncated"]
    assert batch["source"]["cpu_reference"] == "not_generated"
    # Saved generated inputs replay using their actual tokenization bound.
    replay = deepcopy(args)
    replay.source = "batch"
    runner.validate_text_tokens(batch, tokenizer, replay)
    corrupt = deepcopy(batch)
    corrupt["source"]["tokenizer_max_length"] = 65
    with pytest.raises(ValueError, match="tokenizer_max_length"):
        runner.validate_text_tokens(corrupt, tokenizer, replay)


def test_synthetic_seed_preserves_common_sensors_and_content_identity(tmp_path):
    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    config = common.load_config(CONFIGS / "fp16.toml")
    args = runner.example_arguments(config)
    tokenizer = TinyTokenizer()
    original = runner.make_synthetic_batch(args, tokenizer)
    repeat = runner.make_synthetic_batch(args, tokenizer)
    identity = runner.observation_digest(original)
    assert runner.observation_digest(repeat) == identity
    path = tmp_path / "input.pt"
    torch.save(original, path)
    assert runner.observation_digest(torch.load(path, weights_only=True)) == identity
    args.cameras, args.text_tokens = 2, 7
    other = runner.make_synthetic_batch(args, tokenizer)
    for key in ("state", "noise"):
        assert torch.equal(original["observation"][key], other["observation"][key])
    assert all(torch.equal(a, b) for a, b in zip(original["observation"]["images"], other["observation"]["images"]))
    assert runner.observation_digest(other) != identity
    args.instruction = "short"
    with pytest.raises(ValueError, match="at least text_tokens"):
        runner.make_synthetic_batch(args, tokenizer)


@pytest.mark.parametrize("updates", [
    {"source": "unknown"}, {"source": "batch"}, {"source": "synthetic", "batch": "/input.pt"},
    {"source": "batch", "batch": "/input.pt", "instruction": "override"}, {"instruction": ""},
])
def test_input_sources_reject_ambiguous_or_missing_input(updates):
    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    config = common.load_config(CONFIGS / "fp16.toml")
    config["input"].update(updates)
    with pytest.raises(ValueError):
        runner.example_arguments(config)


def test_legacy_batch_source_and_precision_alias_remain_accepted():
    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    config = common.load_config(CONFIGS / "w8a16_action_nvfp4.toml")
    config["input"].pop("source")
    config["input"]["batch"] = "/historical/3cam_t32.pt"
    config["hy_vla"]["precision"] = "w8a16_action_nvfp4"
    args = runner.example_arguments(config)
    assert args.source == "batch" and args.batch == Path("/historical/3cam_t32.pt")
    assert args.precision == "w8a16_nvfp4"
    assert runner.profile_environment(args)["RPU_HY_VLA_EXPERT_NVFP4"] == "1"


def test_prepared_text_preserves_sensor_inputs_and_validates_real_masks_and_tokens():
    from rpu_backend.adapters.hy_vla.preprocess import build_observation
    runner = load("examples/runners/hy_vla/optimized.py")
    prepare = load("scripts/prepare_hyvla_text_batches.py")
    tokenizer = TinyTokenizer()
    source = build_observation([torch.zeros(1, 3, 8, 8)]*3, "original", tokenizer,
        state=torch.arange(20), noise=torch.ones(1, 50, 32))
    batch = prepare.make_text_batch(source, tokenizer,
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen", 16)
    args = SimpleNamespace(prefix_len=192, text_tokens=16)
    obs = runner.validate_observation(batch, args)
    runner.validate_text_tokens(batch, tokenizer)
    assert all(obs[key] is source[key] for key in ("images", "state", "noise"))
    assert obs["lang_tokens"].shape == (64,) and obs["n_valid"] == 192
    # A larger physical prefix uses the identical input and still passes.
    runner.validate_observation(batch, SimpleNamespace(prefix_len=240, text_tokens=16))
    with pytest.raises(ValueError):
        runner.validate_observation(batch, SimpleNamespace(prefix_len=240, text_tokens=32))
    corrupt = deepcopy(batch)
    corrupt["observation"]["suffix_pos"][0] += 1
    with pytest.raises(ValueError, match="all positions"):
        runner.validate_observation(corrupt, args)
    corrupt = deepcopy(batch)
    corrupt["instruction"] = "different actual text"
    with pytest.raises(ValueError, match="checkpoint tokenizer"):
        runner.validate_text_tokens(corrupt, tokenizer)
    corrupt = deepcopy(batch)
    corrupt["observation"]["noise"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner.validate_observation(corrupt, args)


def fake_policy(*, recapture=False):
    class Cache:
        frozen = False
        entry = None
        def step(self):
            if self.entry is None:
                assert not self.frozen
                self.entry = SimpleNamespace(signature="owned", replay_count=0, recapture_count=0, kernel_count=7)
            else:
                self.entry.replay_count += 1
                self.entry.recapture_count += int(recapture)
        def freeze(self):
            self.frozen = True
        def snapshot(self):
            return [self.entry]
        def cache_invariant_ok(self):
            return True
    runner = SimpleNamespace(_psubs={key: SimpleNamespace(handle=index+1, gc=Cache())
        for index, key in enumerate(("vit", "vlm", "expert"))}, last_rpu_execution_plan={"consumed": True})
    runner._run_vit = runner._psubs["vit"].gc.step
    runner._run_vlm_prefill = runner._psubs["vlm"].gc.step
    runner._run_denoise = runner._psubs["expert"].gc.step
    policy = SimpleNamespace(_runner=runner, calls=[])
    def infer(*, obs):
        policy.calls.append(obs["noise"].clone())
        runner._run_vit()
        runner._run_vlm_prefill()
        runner._run_denoise()
        return SimpleNamespace(actions_normalized=obs["noise"][0, :, :20]*2)
    policy.infer = infer
    return policy


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("precision", ["fp16", "w8a16", "w8a16_nvfp4"])
def test_config_run_generates_and_replays_identical_input(monkeypatch, tmp_path, capsys, request, cameras, precision):
    import rpu_backend.api
    common = load("examples/common/config.py")
    runner = load("examples/runners/hy_vla/optimized.py")
    config = template(CONFIG_FILENAMES[precision], cameras)
    previous_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    monkeypatch.setenv('MKL_NUM_THREADS', '8')
    config['cpu'] = {'threads': 3}
    config["input"].update(checkpoint=str(tmp_path), text_tokens=7)
    config["run"].update(output_dir=str(tmp_path / "generated"), runs=1)
    policies = []
    def from_checkpoint(checkpoint, **kwargs):
        assert checkpoint == tmp_path
        assert kwargs["n_cameras"] == cameras
        assert kwargs["prefix_len"] == (128 if cameras == 2 else 192)
        assert kwargs["dtype"] == ("fp16" if precision == "fp16" else "w8a16")
        assert kwargs['threads'] == torch.get_num_threads() == 3
        assert runner.os.environ['OMP_NUM_THREADS'] == runner.os.environ['MKL_NUM_THREADS'] == '3'
        policy = fake_policy()
        policy._runner._native_cold = SimpleNamespace()
        policy._runner.w = SimpleNamespace(expert_nvfp4=None)
        policy._tokenizer = TinyTokenizer
        policy.to = lambda device: policy
        policy.close = lambda: setattr(policy, "closed", True)
        policies.append(policy)
        return policy
    monkeypatch.setattr(rpu_backend.api, "HyEmbodiedPolicy", SimpleNamespace(from_checkpoint=from_checkpoint))
    monkeypatch.setattr(runner, "bind_libraries", lambda args: ({}, {}))
    monkeypatch.setattr(runner, "bind_cold_environment", lambda env: None)
    monkeypatch.setattr(runner, "runtime_payloads", lambda libraries: {})
    monkeypatch.setattr(runner, "asdict", lambda value: {})
    assert runner.run_example_config(config) == 0
    generated = json.loads((tmp_path / "generated/report.json").read_text())
    assert generated["execution"]["status"] == "COMPLETED"
    assert generated["validation"]["reference"]["status"] == "NOT_RUN"
    assert generated["timing"][0]["samples"] == [row["total_ms"] for row in generated["samples"]]
    assert generated["cameras"] == cameras and generated["valid_text_tokens"] == 7
    assert generated["precision"] == precision
    assert generated['cpu_runtime']['requested_threads'] == generated['cpu_runtime']['torch_num_threads'] == 3
    assert generated["cpu_reference"] == "not_run" and not generated["formal_certification"]
    config["input"].update(source="batch", batch=str(tmp_path / "generated/input_batch.pt"),
        reference=str(tmp_path / "generated/reference.pt"))
    config["run"]["output_dir"] = str(tmp_path / "replayed")
    assert runner.run_example_config(config) == 0
    replayed = json.loads((tmp_path / "replayed/report.json").read_text())
    assert replayed["batch_sha256"] == generated["batch_sha256"]
    assert replayed["batch_identity"] == generated["batch_identity"] == "observation_content_sha256"
    assert replayed["validation"]["reference"] == {"status": "MEASURED", "report_fields": ["reference"]}
    assert replayed["validation"]["task"]["status"] == "NOT_EVALUATED"
    assert replayed["reference"]["metrics"]["byte_equal"]
    assert all(policy.closed for policy in policies)
    capsys.readouterr()


def test_measurement_freezes_real_owner_protocol_and_excludes_settling_and_aba():
    runner = load("examples/runners/hy_vla/optimized.py")
    policy = fake_policy()
    output, report = runner.measure_policy(policy, {"noise": torch.ones(1, 50, 32)}, SimpleNamespace(warmup=3, runs=2))
    assert len(policy.calls) == 7 and len(report["samples"]) == 2
    assert all(sub.gc.frozen for sub in policy._runner._psubs.values())
    assert report["aba"]["a_restore"]["byte_equal"]
    assert report["aba"]["a_vs_b"]["max_abs"] == 4
    assert torch.equal(output, torch.full((50, 20), 2.))
    assert set(report["summary"]) == {"total_ms", "vision_ms", "prefill_ms", "denoise_ms"}
    assert report["execution_receipts"] == {"consumed": True}
    assert "_run_vit" in vars(policy._runner)  # Observer restored the original bound callable.
    assert policy._runner._run_vit == policy._runner._psubs["vit"].gc.step


def test_metrics_distinguish_byte_identity_from_equal_floating_values():
    runner = load("examples/runners/hy_vla/optimized.py")
    actual = torch.arange(1000, dtype=torch.float16).reshape(20, 50).T
    assert not actual.is_contiguous()
    assert runner.metrics(actual, actual.contiguous())["byte_equal"]
    for reference in (actual.float(), actual.clone()):
        if reference.dtype == actual.dtype:
            reference[0, 0] = -0.0
        result = runner.metrics(actual, reference)
        assert result["mse"] == result["max_abs"] == 0
        assert not result["byte_equal"]


def test_recapture_aborts_before_timed_samples_and_restores_observers():
    runner = load("examples/runners/hy_vla/optimized.py")
    policy = fake_policy(recapture=True)
    with pytest.raises(RuntimeError, match="without recapture"):
        runner.measure_policy(policy, {"noise": torch.ones(1, 50, 32)}, SimpleNamespace(warmup=3, runs=2))
    assert len(policy.calls) == 3 and not any(sub.gc.frozen for sub in policy._runner._psubs.values())
    assert policy._runner._run_vit == policy._runner._psubs["vit"].gc.step


@pytest.mark.parametrize("fault", ["stale", "restore"])
def test_aba_rejects_stale_or_nonrestoring_outputs(fault):
    runner = load("examples/runners/hy_vla/optimized.py")
    policy = fake_policy()
    original = policy.infer
    def infer(**kwargs):
        result = original(**kwargs)
        if fault == "stale":
            result.actions_normalized = torch.full((50, 20), 2.)
        elif len(policy.calls) == 7:
            result.actions_normalized += 1
        return result
    policy.infer = infer
    with pytest.raises(RuntimeError, match="A/B/A"):
        runner.measure_policy(policy, {"noise": torch.ones(1, 50, 32)}, SimpleNamespace(warmup=3, runs=2))


def test_loaded_native_and_refs_must_match_selected_artifacts(monkeypatch, tmp_path):
    runner = load("examples/runners/hy_vla/optimized.py")
    native = tmp_path / "native.so"
    native.write_bytes(b"native")
    libraries = {"RPU_BACKEND_SO": {"path": str(native), "sha256": runner.digest(native)},
        "RPU_KERNEL_LIB_PATH": {"path": "/main.ref", "sha256": "main"},
        "RPU_KERNEL_LIB_PATH_EXPANSION": {"path": "/hyvla.ref", "sha256": "hyvla"}}
    actual = {"extension": str(native), "oplib_roles": {"main": "/main.ref", "expansion": "/hyvla.ref"},
        "oplib_payloads": {"/main.ref": {"sha256": "main"}, "/hyvla.ref": {"sha256": "hyvla"}}}
    backend = ModuleType("rpu_backend")
    backend._cpp_ext = SimpleNamespace(get_runtime_payload_paths=lambda: actual)
    monkeypatch.setitem(sys.modules, "rpu_backend", backend)
    assert runner.runtime_payloads(libraries) is actual
    actual["oplib_payloads"]["/hyvla.ref"]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="loaded expansion"):
        runner.runtime_payloads(libraries)
    actual["oplib_payloads"]["/hyvla.ref"]["sha256"] = "hyvla"
    native.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="loaded native"):
        runner.runtime_payloads(libraries)


def test_cold_config_rejects_conflict_and_unsafe_override(monkeypatch):
    runner = load("examples/runners/hy_vla/optimized.py")
    monkeypatch.delitem(sys.modules, "rpu_backend", raising=False)
    monkeypatch.setenv("RPU_HY_VLA_EXPERT_NVFP4", "0")
    with pytest.raises(ValueError, match="RPU_HY_VLA_EXPERT_NVFP4"):
        runner.bind_cold_environment({"RPU_HY_VLA_EXPERT_NVFP4": "1"})
    monkeypatch.setenv("RPU_CHUNK_FORCE_UNSAFE", "1")
    with pytest.raises(ValueError, match="RPU_CHUNK_FORCE_UNSAFE"):
        runner.bind_cold_environment({})


@pytest.mark.parametrize("failure", ["install", "timed_replay", "aba"])
def test_execution_and_cleanup_failure_preserve_primary_error_and_report(monkeypatch, tmp_path, failure):
    runner = load("examples/runners/hy_vla/optimized.py")
    common = load("examples/common/config.py")
    config = common.load_config(CONFIGS / "fp16.toml")
    config["input"].update(checkpoint=str(tmp_path), text_tokens=7)
    config["run"].update(output_dir=str(tmp_path / "failed"), runs=3)
    primary = RuntimeError(f"{failure} failed")
    calls = []

    def install(device):
        if failure == "install":
            raise primary
        return policy

    def close():
        calls.append("close")
        raise RuntimeError("retirement failed")

    policy = fake_policy()
    policy._tokenizer, policy.to, policy.close = TinyTokenizer, install, close
    policy._runner._native_cold = SimpleNamespace()
    original_infer = policy.infer

    def infer(**kwargs):
        completed_calls = len(policy.calls)
        if ((failure == "timed_replay" and completed_calls == config["run"]["warmup"] + 1)
                or (failure == "aba" and completed_calls == config["run"]["warmup"] + config["run"]["runs"])):
            raise primary
        return original_infer(**kwargs)

    policy.infer = infer
    api = ModuleType("rpu_backend.api")
    api.HyEmbodiedPolicy = SimpleNamespace(from_checkpoint=lambda *a, **kw: policy)
    monkeypatch.setitem(sys.modules, "rpu_backend.api", api)
    monkeypatch.setattr(runner, "bind_libraries", lambda args: ({}, {}))
    monkeypatch.setattr(runner, "bind_cold_environment", lambda env: None)
    monkeypatch.setattr(runner, "runtime_payloads", lambda libraries: {})
    monkeypatch.setattr(runner, "asdict", lambda value: {})
    monkeypatch.setitem(runner._CPU, "configure_cpu_threads", lambda threads: {})
    with pytest.raises(RuntimeError, match=f"{failure} failed") as raised:
        runner.run_example_config(config)
    assert raised.value is primary and calls == ["close"]
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["execution"]["status"] == "FAILED"
    assert report["error"] == f"RuntimeError: {failure} failed"
    assert report["cleanup_error"] == "RuntimeError: retirement failed"
    assert report["validation"]["runtime"]["status"] == "NOT_COMPLETED"
    assert report["validation"]["runtime"]["checks"] == []
    assert report["validation"]["reference"]["status"] == "NOT_RUN"
    assert report["correctness"] == "not_run"
    if failure == "install":
        assert report.get("samples", []) == [] and report["timing"] == []
    else:
        expected_samples = 1 if failure == "timed_replay" else config["run"]["runs"]
        assert len(report["samples"]) == expected_samples
        assert len(report["settling_graphs"]) == config["run"]["warmup"]
        assert set(report["graphs_before"]) == {"vit", "vlm", "expert"}
        total, = [row for row in report["timing"] if row["metric"] == "total_latency"]
        assert total["samples"] == [row["total_ms"] for row in report["samples"]]
        assert total["sample_count"] == expected_samples
        assert {row["metric"] for row in report["timing"]} == {
            "total_latency", "vision_latency", "prefill_latency", "denoise_latency"}
        assert policy._runner._run_vit == policy._runner._psubs["vit"].gc.step
        if failure == "aba":
            assert set(report["graphs_after"]) == {"vit", "vlm", "expert"}

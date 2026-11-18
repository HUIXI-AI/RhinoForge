import hashlib
"""Real LingBot example orchestration with mocked model/device boundaries."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from _config_tables import config_cells
import torch

ROOT = Path(__file__).resolve().parents[1]


def load():
    import importlib
    return importlib.import_module("examples.runners.lingbot2.optimized")


def config(tmp_path, precision="w8a16"):
    return {"lingbot2": {"precision": precision}, "input": {"checkpoint": "/checkpoint",
        "training_config": "/train.yaml", "case": "/publiccase"},
        "run": {"warmup": 2, "runs": 5, "seed": 0, "output_dir": str(tmp_path)}, "cpu": {"threads": 1}}


@pytest.mark.parametrize("section,key,value", [
    ("lingbot2", "precision", "w4a16"), ("lingbot2", "instruction_residency", True),
    ("lingbot2", "top_k", 2), ("input", "case", ""), ("input", "num_steps", 5),
    ("run", "warmup", 0), ("run", "runs", True), ("run", "output_dir", ""),
    ("run", "hwperf", 1), ("cpu", "threads", 0)])
def test_config_rejects_semantic_and_ambiguous_controls(tmp_path, section, key, value):
    runner = load(); value_config = config(tmp_path); value_config[section][key] = value
    with pytest.raises(ValueError): runner.validate_example_config(value_config)


def test_validation_stays_cpu_safe_and_does_not_relabel_z2(tmp_path, monkeypatch):
    runner = load(); value = config(tmp_path)
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("rpu_backend"):
            raise AssertionError("configuration validation imported native-dependent module")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    runner.validate_example_config(value)
    assert runner.registered_example(value) is None
    value["rpu_execution"] = {"model": {"num_cores": 4}}
    runner.validate_example_config(value)
    value["rpu_execution"] = {"model": {"num_cores": 5}}
    with pytest.raises(ValueError, match="4, 6 or 8"): runner.validate_example_config(value)


def test_precision_cold_flags_preserve_fp16_and_keep_experiments_off():
    runner = load()
    for precision in ("fp16", "w8a16"):
        env = runner.profile_environment(precision)
        assert "RPU_L2_BUFONLY" not in env and "RPU_L2_ADDR" not in env
        for name in ("EXPERT", "PREFILL", "BASE", "VISION"):
            assert env[f"RPU_LINGBOT2_{name}_W8A16"] == str(int(precision == "w8a16"))
        assert env["RPU_WALL_OSS_FAST_REPLAY"] == str(int(precision == "fp16"))
        assert not any("SNAPSHOT" in name for name in env)
        for name in ("PREFILL_KV_PAIR", "PREFILL_QK_NORM_ROPE", "MULTIVIEW_SPM_Z2"):
            assert env["RPU_LINGBOT2_" + name] == "0"
        assert env["RPU_LINGBOT2_FP16_TOP4"] == env["RPU_LINGBOT2_DENOISE_UNROLL"] == "1"


def test_cold_environment_rejects_stale_debug_and_preloaded_backend(monkeypatch):
    runner = load()
    monkeypatch.delitem(sys.modules, "rpu_backend", raising=False)
    monkeypatch.setattr(runner.os, "environ", {"RPU_LINGBOT2_UNKNOWN_EXPERIMENT": "1"})
    with pytest.raises(ValueError, match="unsupported cold"): runner.bind_cold_environment({})
    monkeypatch.setitem(sys.modules, "rpu_backend", ModuleType("rpu_backend"))
    with pytest.raises(ValueError, match="fresh process"): runner.bind_cold_environment({})


class FakeCache:
    def __init__(self):
        self.entry = SimpleNamespace(kernel_count=1, recapture_count=0, replay_count=4,
            segment_count=1, prepared_segment_hit_total=4, prepared_segment_miss_total=0,
            hw_batch_submit_total=4, graph_lifetime_id=17, build_generation=1)
    def is_frozen(self): return True
    def cache_invariant_ok(self): return True
    def size(self): return 1
    def snapshot(self): return [self.entry]


class FakePolicy:
    def __init__(self, fail=False):
        self._visual = SimpleNamespace(_rpu_vision_graph_cache=FakeCache())
        self._vgc, self._dgc = FakeCache(), FakeCache()
        self._last_execution_plan = {"test": "actual consumed planner report"}
        self.calls = 0; self.closed = 0; self.fail = fail
    def _request_prefix_geometry(self, *a): return {"real_len": 217}
    def prepare_graphs(self, obs, **kwargs):
        self.prepared = kwargs
        return {"phase": "READY", "ready_replay_validated": True, "num_steps": 10}
    def _build_prefix(self): pass
    def _vlm_fill(self): pass
    def _denoise_unroll_run(self): pass
    def get_action(self, obs):
        self.calls += 1
        self._build_prefix(); self._vlm_fill(); self._denoise_unroll_run()
        if self.fail and self.calls == 3: raise RuntimeError("device failure")
        for cache in (self._visual._rpu_vision_graph_cache, self._vgc, self._dgc):
            cache.entry.replay_count += 1; cache.entry.prepared_segment_hit_total += 1
            cache.entry.hw_batch_submit_total += 1
        return torch.ones(1, 50, 55)
    def close(self): self.closed += 1


def mock_runtime(monkeypatch, runner, policy):
    calls = []
    def build(path, **kwargs): calls.append((path, kwargs)); return policy
    monkeypatch.setattr(runner, "profile_libraries", lambda: ({}, {}))
    monkeypatch.setattr(runner, "bind_cold_environment", lambda env: None)
    monkeypatch.setattr(runner.cpu, "configure_cpu_threads", lambda n: {"torch_num_threads": n})
    monkeypatch.setattr(runner, "source_identity", lambda: {"git_head": "test"})
    monkeypatch.setattr(runner, "runtime_payloads", lambda *a: {"launch": "/sdk.so", "extension": "/native.so"})
    monkeypatch.setattr(runner, "installed_precision_evidence", lambda *a: {"mock": "owner metadata"})
    obs = dict(images=torch.zeros(3, 1), img_masks=torch.ones(3, dtype=torch.bool),
        lang_tokens=torch.ones(1, 72, dtype=torch.long), lang_masks=torch.ones(1, 72, dtype=torch.bool),
        image_grid_thw=torch.ones(3, 3), state=torch.zeros(1, 55), noise=torch.ones(1, 50, 55))
    # Intentionally distant CPU reference: comparison must be reported, not
    # subjected to a stale uncalibrated cosine threshold.
    golden = {"normalized_actions": -torch.ones(1, 50, 55)}
    from examples.runners.lingbot2 import inputs
    monkeypatch.setattr(inputs, "load_inputs", lambda values, **kwargs: (obs, golden,
        {"checkpoint": "/checkpoint", "training_config": "/train.yaml"},
        {"block_future_depth_to_action": False, "block_suffix_to_future_video": False}))
    graph = ModuleType("rpu_backend.graph"); graph.GraphRuntimePolicy = SimpleNamespace(from_environment=lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules, "rpu_backend.graph", graph)
    monkeypatch.setattr(runner, "load_runtime", lambda: (SimpleNamespace(build_lingbot_vla_v2=build), SimpleNamespace))
    monkeypatch.setattr(torch, "rpu", SimpleNamespace(get_ddr_flush=lambda: False,
        get_ddr_flush_force=lambda: True), raising=False)
    return calls


@pytest.mark.parametrize("precision", ["fp16", "w8a16"])
def test_real_flow_counts_clean_calls_and_separates_diagnostics(tmp_path, monkeypatch, precision):
    runner = load(); policy = FakePolicy(); calls = mock_runtime(monkeypatch, runner, policy)
    value = config(tmp_path, precision); value["run"].update(profile=True, hwperf=True)
    def diagnostics(owner, infer, first, cfg, out, report):
        assert owner.calls == 7 and len(report["samples_ms"]) == 5
        assert torch.equal(infer(), first); assert torch.equal(infer(), first)
        report["diagnostic_calls"] = 2
    monkeypatch.setattr(runner, "_diagnostics", diagnostics)
    assert runner.run_example_config(value) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert policy.closed == 1 and policy.calls == 9 and policy.prepared == {"prefix_length_range": (217, 217)}
    kwargs = calls[0][1]
    assert kwargs["graph_runtime_policy"].batch_snapshot_mode == "auto"
    assert kwargs["training_config_path"] == "/train.yaml"
    assert kwargs["align_cfg"].block_future_depth_to_action is False
    assert kwargs["align_cfg"].block_suffix_to_future_video is False
    assert report["execution"]["status"] == "COMPLETED"
    assert len(report["samples_ms"]) == report["timing"][0]["sample_count"] == 5
    assert set(report["samples_ms"][0]) == {"total_ms", "vision_ms", "prefill_ms", "action_ms"}
    assert report["graphs_after"]["action"][0]["replays"] == 11
    assert report["runtime_calls"] == {"prepare": 5, "warmup": 2, "measured": 5, "diagnostic": 2, "total": 14}
    assert report["cpu_reference"]["metrics"]["cosine"] == pytest.approx(-1)
    assert report["validation"]["reference"]["status"] == "MEASURED"
    assert torch.equal(torch.load(tmp_path / "actions.pt", weights_only=True), torch.ones(1, 50, 55))



@pytest.mark.parametrize(("cores", "cpu", "threads"), [
    (8, None, 1), (6, None, 12), (4, None, 12), (6, 3, 3), (8, 12, 12)])
def test_omitted_cpu_threads_follow_the_core_budget(tmp_path, monkeypatch, cores, cpu, threads):
    # Below eight cores the Vision patch embed and mergers run on the CPU, so the
    # omitted default is 12 there; an explicit [cpu].threads always wins.
    runner = load(); policy = FakePolicy(); mock_runtime(monkeypatch, runner, policy)
    value = config(tmp_path); value["rpu_execution"] = {"model": {"num_cores": cores}}
    value["cpu"] = {} if cpu is None else {"threads": cpu}
    runner.validate_example_config(value)
    assert runner.run_example_config(value) == 0
    assert json.loads((tmp_path / "report.json").read_text())["cpu_runtime"] == {"torch_num_threads": threads}


def test_failure_closes_owner_restores_stage_methods_and_records_error(tmp_path, monkeypatch):
    runner = load(); policy = FakePolicy(fail=True); mock_runtime(monkeypatch, runner, policy)
    original = policy._build_prefix
    with pytest.raises(RuntimeError, match="device failure"):
        runner.run_example_config(config(tmp_path))
    assert policy.closed == 1 and policy._build_prefix == original
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["execution"]["status"] == "FAILED" and "device failure" in report["execution"]["error"]
    assert report["artifacts"]["outputs"] == []


def test_graph_replay_accepts_queue_fallback_but_rejects_recapture():
    runner = load(); policy = FakePolicy(); before = runner.graph_stats(policy)
    policy.get_action({}); policy._dgc.entry.prepared_segment_miss_total += 1
    runner.check_replay(before, runner.graph_stats(policy))
    policy._dgc.entry.build_generation += 1
    with pytest.raises(RuntimeError, match="clean replay"): runner.check_replay(before, runner.graph_stats(policy))


def test_paths_expand_variables_only_when_executing(tmp_path, monkeypatch):
    runner = load(); monkeypatch.setenv("LINGBOT2_TEST_CASE", str(tmp_path))
    assert runner.expanded_path("${LINGBOT2_TEST_CASE}") == tmp_path.resolve()
    value = config(tmp_path); value["input"]["case"] = "${LINGBOT2_UNDEFINED_CASE}"
    runner.validate_example_config(value)
    monkeypatch.delenv("LINGBOT2_UNDEFINED_CASE", raising=False)
    with pytest.raises(ValueError, match="unresolved environment variable"):
        runner.expanded_path(value["input"]["case"])


def test_success_does_not_publish_stale_profile_artifacts(tmp_path, monkeypatch):
    runner = load(); policy = FakePolicy(); mock_runtime(monkeypatch, runner, policy)
    (tmp_path / "profile.trace.json").write_text("old trace")
    (tmp_path / "profile.txt").write_text("old summary")
    runner.run_example_config(config(tmp_path))
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["artifacts"]["outputs"] == [str(tmp_path / "actions.pt")]
    assert report["runtime_calls"]["total"] == 12


def test_hwperf_rejects_old_directory_before_model_load(tmp_path, monkeypatch):
    runner = load(); policy = FakePolicy(); calls = mock_runtime(monkeypatch, runner, policy)
    (tmp_path / "hwperf").mkdir(); (tmp_path / "hwperf/old.json").write_text("old")
    value = config(tmp_path); value["run"]["hwperf"] = True
    with pytest.raises(ValueError, match="nonempty"): runner.run_example_config(value)
    assert calls == [] and policy.calls == 0


def test_source_identity_without_git_keeps_actual_source_hashes(monkeypatch):
    runner = load()
    def missing_git(*args, **kwargs):
        raise runner.subprocess.CalledProcessError(128, args[0], stderr="not a git repository")
    monkeypatch.setattr(runner.subprocess, "run", missing_git)
    result = runner.source_identity()
    assert result["git_head"] is None and result["tracked_dirty"] is None
    assert result["git_status_unavailable"] == "CalledProcessError"
    expected = ("examples/runners/lingbot2/optimized.py", "examples/runners/lingbot2/inputs.py",
                "python/rpu_backend/adapters/lingbot_vla_v2/runtime.py",
                "python/rpu_backend/adapters/lingbot_vla_v2/prefix.py")
    assert result["source_sha256"] == {name: runner.libraries.file_sha256(ROOT / name) for name in expected}


def test_text_matrix_templates_cpu_validation_and_unified_entry(monkeypatch, capsys):
    import subprocess
    from examples.common import config as common, cli
    paths = sorted((ROOT / "examples/configs/lingbot/v2").glob("*.toml"))
    assert [path.name for path in paths] == ["fp16.toml", "w8a16.toml"]
    # The templates hold 3 cameras; 2 cameras is preset 2cam_t32 with a 2-camera observation.
    two_cameras = ["--input", "2cam_t32", "--set", "input.observation=/path/to/2cam_t32.pt"]
    targets = {(cameras, tokens, precision) for cameras in (2, 3)
               for tokens in (32,) for precision in ("fp16", "w8a16")}
    seen = []
    configs = {}
    monkeypatch.setattr(load(), "run_example_config", lambda value: seen.append(value) or 0)
    for path in paths:
        configs[path] = common.load_config(path)
        for extra in ([], two_cameras):
            assert cli.main(["--config", str(path), *extra, "--check-config"]) == 0
            assert cli.main(["--config", str(path), *extra]) == 0
    assert {(value["input"]["cameras"], value["input"]["expected_language_tokens"],
             value["lingbot2"]["precision"]) for value in seen} == targets
    assert len(seen) == 4
    for value in seen:
        assert common.config_target(value) == "lingbot2" and common.example_metadata(value) is None
    from scripts import check_examples
    index = (ROOT / "examples/configs/lingbot/README.md").read_text(encoding="utf-8")
    for path, value in configs.items():
        relative = path.relative_to(check_examples.CONFIGS / "lingbot").as_posix()
        cells = config_cells(index, relative)
        assert cells["精度"] == value["lingbot2"]["precision"]
        assert cells["cameras 默认值"] == str(value["input"]["cameras"])
        assert cells["expected_language_tokens 默认值"] == "32"
        assert cells["max_language_tokens 默认值"] == "72"
    assert "CONTROLLED_EVALUATION" not in index
    # An isolated interpreter catches accidental module-level Torch/native imports.
    code = '''import builtins, sys
original = builtins.__import__
def guard(name,*args,**kwargs):
    if name == 'torch' or name.startswith('rpu_backend'): raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
from examples.common.cli import main
assert main(['--config',sys.argv[1],'--check-config'])==0
'''
    result = subprocess.run([sys.executable, "-c", code, str(paths[0])], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_library_binding_requires_exact_two_manifest_payloads(monkeypatch, tmp_path):
    runner = load()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "release").mkdir()
    artifacts, selected = {}, {}
    for key, role, alias, content in (
        ("RPU_KERNEL_LIB_PATH", "rhino_ops_main", "rhinoOpLib_current.ref", b"main payload"),
        ("RPU_KERNEL_LIB_PATH_EXPANSION", "rhino_ops_expansion", "rhinoExpansionOpLib_current.ref", b"expansion payload"),
    ):
        payload = tmp_path / alias
        payload.write_bytes(content)
        artifacts[role] = {"path": alias, "sha256": hashlib.sha256(content).hexdigest()}
        selected[key] = str(payload)
        monkeypatch.setenv(key, str(payload))
    (tmp_path / "release/runtime-dependencies.json").write_text(json.dumps({"artifacts": artifacts}))
    env, records = runner.profile_libraries()
    assert env == selected
    for key, role in (("RPU_KERNEL_LIB_PATH", "rhino_ops_main"), ("RPU_KERNEL_LIB_PATH_EXPANSION", "rhino_ops_expansion")):
        assert records[key] == {"path": env[key], "sha256": artifacts[role]["sha256"]}
    (tmp_path / "rhinoExpansionOpLib_current.ref").write_bytes(b"changed payload")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.profile_libraries()


def observation(cameras=3, tokens=3):
    return dict(images=torch.zeros(cameras, 256, 1536), image_grid_thw=torch.tensor([[1,16,16]] * cameras),
        img_masks=torch.ones(cameras, dtype=torch.bool), lang_tokens=torch.ones(1,72,dtype=torch.long),
        lang_masks=torch.arange(72).reshape(1,72) < tokens, state=torch.zeros(1,55), noise=torch.ones(1,50,55))


@pytest.mark.parametrize("cameras,tokens", [(2,16), (3,3), (3,64)])
def test_observation_checks_actual_counts_without_altering_tokens(cameras, tokens):
    from examples.runners.lingbot2.inputs import validate_observation
    obs = observation(cameras, tokens); original = obs["lang_tokens"].clone()
    metadata = validate_observation(obs, {"cameras": cameras, "expected_language_tokens": tokens})
    assert metadata["language_tokens"] == tokens and torch.equal(original, obs["lang_tokens"])
    with pytest.raises(ValueError, match="counts differ"):
        validate_observation(obs, {"expected_language_tokens": tokens + 1})
    obs["lang_tokens"] = obs["lang_tokens"].float()
    with pytest.raises(ValueError, match="Int64/Boolean"): validate_observation(obs, {})


def test_training_alignment_follows_official_defaults_and_explicit_masks():
    from examples.runners.lingbot2.inputs import training_alignment
    train = {"align_params": {"mode": "query", "num_task_tokens": 8,
        "use_future_video": True, "depth": {"num_backbone_tokens": 256, "use_future_depth": True},
        "video": {"use_current_patch_loss": True}}}
    result = training_alignment(train, {})
    assert result["block_future_depth_to_action"] is result["block_suffix_to_future_video"] is False
    assert result["use_future_video_patch"] is result["use_current_shared_task_proj"] is True
    train["align_params"]["depth"]["block_future_depth_to_action"] = True
    train["align_params"]["video"]["block_suffix_to_future_video"] = True
    result = training_alignment(train, {"qwen3vl_use_vision_boundaries": False})
    assert result["block_future_depth_to_action"] is result["block_suffix_to_future_video"] is True
    assert result["qwen3vl_use_vision_boundaries"] is False
    train["token_top_k"] = 2
    with pytest.raises(ValueError, match="token_top_k"): training_alignment(train, {})


def test_actual_case_masks_steps_hash_and_tensor_observation(tmp_path):
    import yaml
    from examples.runners.lingbot2 import inputs
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir(); (checkpoint / "config.json").write_text("{}")
    training = tmp_path / "train.yaml"
    training.write_text(yaml.safe_dump({"train": {"align_params": {"mode": "query", "use_future_video": True,
        "depth": {"use_future_depth": True}, "video": {}}}}))
    directory = tmp_path / "case"; directory.mkdir()
    obs = observation(); torch.save(obs, directory / "input.pt")
    torch.save({"normalized_actions": torch.zeros(1,50,55)}, directory / "golden.pt")
    case = {"model_id": "lingbot-vla-v2-6b", "call": {"denoise_steps": 10,
        "output_stage": "full55-before-action-postprocessing", "policy_postprocessor": False,
        "profile_mask_defaults": {"block_future_depth_to_action": False, "block_suffix_to_future_video": False}}}
    for key in ("input", "golden"):
        path = directory / (key + ".pt")
        case[key] = {"path": path.name, "sha256": inputs.file_sha256(path)}
    metadata_path = directory / "case.json"; metadata_path.write_text(json.dumps(case))
    settings = {"checkpoint": str(checkpoint), "training_config": str(training), "case": str(directory)}
    actual, golden, assets, align = inputs.load_inputs(settings, seed=0)
    assert torch.equal(actual["images"], obs["images"]) and assets["observation"]["language_tokens"] == 3
    assert golden["normalized_actions"].shape == (1,50,55) and not align["block_suffix_to_future_video"]
    case["call"]["denoise_steps"] = 4; metadata_path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="ten native steps"): inputs.load_inputs(settings, seed=0)
    case["call"]["denoise_steps"] = 10
    case["call"]["profile_mask_defaults"]["block_suffix_to_future_video"] = True
    metadata_path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="training alignment masks"): inputs.load_inputs(settings, seed=0)
    case["call"]["profile_mask_defaults"]["block_suffix_to_future_video"] = False
    case["input"]["sha256"] = "0"*64; metadata_path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="checksum"): inputs.load_inputs(settings, seed=0)
    settings.pop("case"); settings["observation"] = str(directory / "input.pt")
    actual, golden, assets, _ = inputs.load_inputs(settings, seed=0)
    assert golden is None and assets["input_mode"] == "observation" and torch.equal(actual["noise"], obs["noise"])


def test_raw_mode_schema_and_seeded_cpu_input(monkeypatch, tmp_path):
    from examples.runners.lingbot2 import inputs
    value = config(tmp_path); value["input"].pop("case")
    value["input"].update(images=["one.png", "two.png"], instruction="Move the red object.", state=[0.0]*16)
    load().validate_example_config(value)
    changed = deepcopy(value); changed["input"]["case"] = "ambiguous"
    with pytest.raises(ValueError, match="exactly one"): load().validate_example_config(changed)
    changed = deepcopy(value); changed["input"]["state"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"): load().validate_example_config(changed)
    # Exercise exact preprocessing orchestration with CPU processors; no backend.
    import numpy as np
    from PIL import Image
    images = []
    for index in range(2):
        path = tmp_path / f"image{index}.png"; Image.fromarray(np.full((8,8,3), index, dtype=np.uint8)).save(path); images.append(str(path))
    value["input"]["images"] = images
    processor = SimpleNamespace(
        image_processor=lambda images, return_tensors: {"pixel_values": torch.zeros(512,1536), "image_grid_thw": torch.tensor([[1,16,16]]*2)},
        tokenizer=lambda text, **kwargs: {"input_ids": torch.ones(1,72,dtype=torch.long), "attention_mask": torch.arange(72).reshape(1,72)<16})
    transformers = ModuleType("transformers"); transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda *a, **kw: processor)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    transforms = ModuleType("torchvision.transforms.v2"); transforms.Resize = lambda size: lambda tensor: tensor
    monkeypatch.setitem(sys.modules, "torchvision.transforms.v2", transforms)
    a = inputs.raw_observation(value["input"], tmp_path, 4)
    b = inputs.raw_observation(value["input"], tmp_path, 4)
    assert inputs.validate_observation(a, {"cameras": 2, "expected_language_tokens": 16})["language_tokens"] == 16
    assert torch.equal(a["noise"], b["noise"]) and a["state"].shape == (1,55)
    assert a["state"][0,16:].eq(0).all()


def test_policy_off_and_variable_geometry_forwarded_without_reference(tmp_path, monkeypatch):
    from examples.runners.lingbot2 import inputs
    runner = load(); policy = FakePolicy(); calls = mock_runtime(monkeypatch, runner, policy)
    monkeypatch.setattr(policy, "_request_prefix_geometry", lambda *a: {"real_len": 164, "language_len": 16})
    monkeypatch.setattr(inputs, "load_inputs", lambda *a, **kw: (observation(2,16), None,
        {"checkpoint": "/checkpoint", "training_config": "/train.yaml"},
        {"block_future_depth_to_action": True, "block_suffix_to_future_video": True}))
    value = config(tmp_path); value["lingbot2"]["instruction_residency"] = "off"
    runner.run_example_config(value)
    assert policy.prepared == {"prefix_length_range": (164,164)}
    assert calls[0][1]["graph_runtime_policy"].batch_snapshot_mode == "off"
    assert calls[0][1]["align_cfg"].block_suffix_to_future_video is True
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["validation"]["reference"]["status"] == "NOT_RUN" and report["runtime_calls"]["total"] == 12


@pytest.mark.parametrize("cap", [0, 161, True])
def test_language_cap_rejects_invalid_limits(tmp_path, cap):
    value = config(tmp_path); value["input"]["max_language_tokens"] = cap
    with pytest.raises(ValueError, match="max_language_tokens"):
        load().validate_example_config(value)


def test_long_instruction_preserves_all_tokens_and_requires_explicit_cap(tmp_path):
    from examples.runners.lingbot2 import inputs
    calls = []
    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        assert kwargs["truncation"] is False
        width = max(160, kwargs["max_length"])
        return {"input_ids": torch.arange(width).reshape(1, width),
                "attention_mask": (torch.arange(width) < 160).reshape(1, width)}
    instruction = "A complete long instruction provided by the user."
    with pytest.raises(ValueError, match="160 valid tokens"):
        inputs.tokenize_instruction(tokenizer, instruction)
    encoded = inputs.tokenize_instruction(tokenizer, instruction, 160)
    obs = observation(); obs["lang_tokens"] = encoded["input_ids"]
    obs["lang_masks"] = encoded["attention_mask"]
    original = obs["lang_tokens"].clone()
    with pytest.raises(ValueError, match="language cap"):
        inputs.validate_observation(obs, {})
    settings = {"max_language_tokens": 160, "expected_language_tokens": 160}
    info = inputs.validate_observation(obs, settings)
    assert info["language_tokens"] == info["language_storage_tokens"] == info["max_language_tokens"] == 160
    assert torch.equal(obs["lang_tokens"], original)
    value = config(tmp_path); value["input"].update(settings)
    load().validate_example_config(value)
    assert all(text == instruction for text, _ in calls)
    assert calls[0][1]["max_length"] == 72  # Existing default remains unchanged.

"""`examples/run.py --cores/--set/--emit-toml`: caller overrides resolve like the TOML would."""
from __future__ import annotations

import collections
import contextlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import export_examples  # noqa: E402
from examples.common.cli import main  # noqa: E402
from examples.common.config import ConfigError, load_config  # noqa: E402
from examples.common.config import _FIXED_EIGHT as FIXED_EIGHT  # noqa: E402
from examples.common.overrides import (  # noqa: E402
    caller_overrides, input_family, parse_assignment, preset_files, recipe_path, toml_text,
)

QWEN = ROOT / "examples/configs/qwen3/text/4b/fp16.toml"
HYVLA = ROOT / "examples/configs/hy_vla/fp16.toml"
WALL_RTC = ROOT / "examples/configs/wall_oss/rtc/fp16.toml"
WALL_LEGACY = ROOT / "examples/configs/wall_oss/legacy/fp16.toml"


def test_assignment_values_are_toml_literals_and_bare_words_stay_strings():
    assert parse_assignment("rpu_execution.prefill.chunk_size=128") == (
        ("rpu_execution", "prefill", "chunk_size"), 128)
    assert parse_assignment("rpu_execution.prefill.chunk_size=auto")[1] == "auto"
    assert parse_assignment("run.hwperf=true")[1] is True
    assert parse_assignment('input.prompt="a = b"')[1] == "a = b"
    for text in ("chunk_size=128", "rpu_execution.prefill.chunk_size", ".x=1"):
        with pytest.raises(ConfigError, match="section.key=value"):
            parse_assignment(text)


def test_cores_and_set_cannot_name_the_same_field_twice():
    assert caller_overrides(["input.decode_steps=32"], 6) == {
        "rpu_execution": {"model": {"num_cores": 6}}, "input": {"decode_steps": 32}}
    with pytest.raises(ConfigError, match="set twice"):
        caller_overrides(["rpu_execution.model.num_cores=4"], 6)
    with pytest.raises(ConfigError, match="conflicts"):
        caller_overrides(["input.decode_steps=32", "input.decode_steps.x=1"])


def test_cores_override_is_checked_like_the_toml_value(tmp_path, capsys):
    bound = tmp_path / "bound.toml"
    assert main(["--config", str(QWEN), "--check-config", "--cores", "6",
                 "--set", "rpu_execution.prefill.chunk_size=128", "--emit-toml", str(bound)]) == 0
    resolved = load_config(bound)
    assert resolved["rpu_execution"]["model"]["num_cores"] == 6
    assert resolved["rpu_execution"]["prefill"]["chunk_size"] == 128
    assert main(["--config", str(bound), "--check-config"]) == 0  # The bound TOML runs as is.
    assert main(["--config", str(HYVLA), "--check-config", "--cores", "6"]) == 0
    assert main(["--config", str(WALL_RTC), "--check-config", "--cores", "6"]) == 0
    capsys.readouterr()
    # An eight-core-only entry refuses 6 with the reason the catalog shows.
    with pytest.raises(SystemExit):
        main(["--config", str(WALL_LEGACY), "--check-config", "--cores", "6"])
    assert FIXED_EIGHT["wall_oss_legacy"] in capsys.readouterr().err


def test_recipe_selects_a_template_of_the_model_directory(tmp_path):
    directory = QWEN.parent
    target = directory / "w8a16_text_head.toml"
    assert recipe_path(QWEN) == QWEN
    assert recipe_path(QWEN, "w8a16_text_head") == recipe_path(directory, "w8a16_text_head") == target

    flags = ["--check-config", "--cores", "6", "--input", "p128_d32"]
    emitted = {}
    for name, selection in (("direct", ["--config", str(target)]),
                            ("sibling", ["--config", str(QWEN), "--recipe", "w8a16_text_head"]),
                            ("directory", ["--config", str(directory), "--recipe", "w8a16_text_head"])):
        path = tmp_path / f"{name}.toml"
        assert main([*selection, *flags, "--emit-toml", str(path)]) == 0
        emitted[name] = path.read_text(encoding="utf-8")
    assert emitted["direct"] == emitted["sibling"] == emitted["directory"]
    assert load_config(tmp_path / "directory.toml")["example"]["dtype"] == "w8a16"


@pytest.mark.parametrize("selection, message", [
    (["--config", str(QWEN), "--recipe", "w8a16"], "--recipe w8a16: .*fp16, fp16_instruct, nvfp4_text, w4a16_text_head_rtn_g32"),
    (["--config", str(QWEN), "--recipe", "../8b/fp16"], "--recipe ../8b/fp16: "),
    (["--config", str(QWEN.parent)], "is a directory; choose a template with --recipe: fp16, "),
    (["--config", str(ROOT / "examples/configs/qwen3/text"), "--recipe", "fp16"], "has no templates")])
def test_recipe_errors_list_what_the_directory_offers(selection, message, capsys):
    with pytest.raises(SystemExit):
        main([*selection, "--check-config"])
    assert re.search(message, capsys.readouterr().err)


@pytest.mark.parametrize("assignment, message", [
    ("example.dtype=w8a16", "model identity"),
    ("libraries.native=/tmp/x.so", "bound libraries"),
    ("input.checkpoint=/tmp/x", "model assets"),
    ("rpu_execution.prefill.chunk_size=100", "multiple of 16"),
    ("rpu_execution.vision.linear_acc32=true", "existing stage scope"),
])
def test_identity_assets_and_unlisted_execution_fields_are_refused(assignment, message):
    with pytest.raises(ConfigError, match=message):
        load_config(QWEN, overrides=caller_overrides([assignment]))


def test_model_scope_accumulation_is_refused_before_a_runner_that_checks_it_late():
    # G0.5 --check-config validates only its input envelope, so the override rule must refuse it.
    with pytest.raises(ConfigError, match="overridden"):
        load_config(ROOT / "examples/configs/g05/fp16.toml",
                    overrides=caller_overrides(["rpu_execution.model.linear_acc32=true"]))


def _templates():
    """Every run.py template as (id, TOML path, loaded config); RhinoVLA v3 fragments use validate_v3.py."""
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.toml"
        for row in export_examples.export_catalog()["entries"]:
            source.write_text(toml_text(row["config"]))
            config = load_config(source)
            if "env" not in config:
                yield row["id"], source, config


def test_every_template_reloads_unchanged_from_its_emitted_toml(tmp_path):
    """--emit-toml writes the loaded configuration; reloading it must not change it."""
    emitted = tmp_path / "emitted.toml"
    for config_id, _, loaded in _templates():
        emitted.write_text(toml_text(loaded))
        assert load_config(emitted) == loaded, config_id


def _check(source, *argv):
    """`run.py --check-config`: the emitted configuration when accepted, else the error text."""
    stderr = io.StringIO()
    with tempfile.TemporaryDirectory() as directory:
        emitted = Path(directory) / "emitted.toml"
        try:
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                main(["--config", str(source), "--check-config", "--emit-toml", str(emitted), *argv])
        except SystemExit:
            return stderr.getvalue()
        return load_config(emitted)


# Input envelopes the runners report for a preset: (family, preset name pattern) -> reason.
OUTSIDE_ENVELOPE = {
    ("hy_vla", r"_t128$"): "integer from 1 to 64",
    ("lingbot2", r"_t128$"): r"must be 1\.\.72, the language cap",
    ("pi05", r"^1cam_"): "one camera needs paired-prefill projection payloads",
}


def test_every_preset_applies_to_its_families_or_reports_why():
    """Each template of a listed family takes every preset through the runner's own check, or says why not."""
    files = preset_files()
    families = [family for _, table in files for family in table["applies_to"]]
    assert files and len(families) == len(set(families))  # One preset file per family.
    outcomes = collections.Counter()
    for config_id, source, config in _templates():
        family, declared = input_family(config), config.get("input", {})
        table = next((table for _, table in files if family in table["applies_to"]), None)
        if table is None:
            continue
        fixture = table.get("fixtures", {}).get(family)
        for name, fields in table["preset"].items():
            result = _check(source, "--input", name)
            if any(key not in declared for key in table.get("requires", [])):
                # Such a layout (a legacy fixture) would record the preset without reading it.
                assert "do not apply to this input layout" in result, (config_id, name)
                outcomes["fixed layout"] += 1
                continue
            if config_id.startswith("qwen3_5/text/35b_a3b/"):
                assert "forward_argmax" in result, (config_id, name)
                continue
            if config_id.startswith("qwen3/text/14b/") and fields["prefill_tokens"] > 192:
                assert "上限是 192" in result, (config_id, name)
                outcomes["known limit"] += 1
                continue
            if config_id.startswith("pi05/kuavo/") and fields.get("cameras") != 3 and not name.startswith("1cam_"):
                assert "raw openpi inputs use exactly three cameras" in result, (config_id, name)
                outcomes["known limit"] += 1
                continue
            if fixture in declared and any(declared[key] != value for key, value in fields.items()):
                assert f"--set input.{fixture}=<path>" in result, (config_id, name)
                result = _check(source, "--input", name, "--set", f"input.{fixture}=/data/other-shape.pt")
                outcomes["needs its file"] += 1
            reason = next((reason for (owner, pattern), reason in OUTSIDE_ENVELOPE.items()
                           if owner == family and re.search(pattern, name)), None)
            if reason is not None:
                assert re.search(reason, result), (config_id, name, result)
                outcomes["outside envelope"] += 1
                continue
            assert isinstance(result, dict), (config_id, name, result)
            assert {key: result["input"][key] for key in fields} == fields, (config_id, name)
            outcomes["accepted"] += 1
    # Text 3 x 60 and VLA 6 x 30 templates; image-text 3 x 32, Vision 3 x 3, ASR 3 x 8, TTS 4 x 9.
    assert outcomes["accepted"] >= 3 * 60 + 6 * 30 + 3 * 32 + 3 * 3 + 3 * 8 + 4 * 9, outcomes
    assert min(outcomes.values()) > 0 and len(outcomes) == 5, outcomes


def test_a_known_length_limit_is_refused_by_the_configuration_check():
    fourteen = ROOT / "examples/configs/qwen3/text/14b/fp16.toml"
    assert "上限是 192" in _check(fourteen, "--input", "p2048_d32")
    assert _check(fourteen, "--input", "p128_d128")["input"]["prefill_tokens"] == 128
    # An explicit length inside the limit overrides the preset's.
    assert _check(fourteen, "--input", "p2048_d32", "--set", "input.prefill_tokens=192")["input"]["prefill_tokens"] == 192
    # The limit restates the adapter's table, which needs Torch to import.
    assert '("qwen3", 40, 5120): ChunkEnvelope(' in (ROOT / "python/rpu_backend/adapters/qwen3.py").read_text()


def test_every_grid_cell_is_one_command():
    """Each (config, cores, input) of input_grid.json gets the state the documented command reports."""
    grid_path = ROOT / "examples/configs/input_grid.json"
    if not grid_path.is_file():
        pytest.skip("examples/configs/input_grid.json is internal-only")
    grid = json.loads(grid_path.read_text())["configs"]
    states = collections.Counter()
    for config_id, source, config in _templates():
        if "env" in config:  # RhinoVLA v3 fragments have their own entry.
            continue
        for name, cells in grid[config_id.removesuffix(".toml") + ".toml"]["inputs"].items():
            for core, cell in cells.items():
                argv = ["--cores", core] + ([] if name == "template" else ["--input", name])
                if "needs_file" in cell:
                    argv += ["--set", f"input.{cell['needs_file']}=/path/to/{name}.pt"]
                result = _check(source, *argv)
                states[cell["state"]] += 1
                if cell["state"] == "refused":
                    assert cell["reason"] in result, (config_id, name, core, result)
                else:
                    assert isinstance(result, dict), (config_id, name, core, result)
    assert min(states["refused"], states["untested"], states["supported"]) > 0, states


def test_merged_camera_templates_take_their_old_input_as_a_preset():
    """A template merged into another TOML differed only in input; config_names.json names the preset for it."""
    configs = ROOT / "examples/configs"
    merged = json.loads((configs / "config_names.json").read_text(encoding="utf-8"))["merged"]
    assert merged
    for old, target in merged.items():
        assert not (ROOT / old).exists(), old
        joined = ROOT / target["config"]
        config = load_config(joined)
        if target["input"] is None:
            continue
        table = next(table for _, table in preset_files() if input_family(config) in table["applies_to"])
        fixture = table.get("fixtures", {}).get(input_family(config))
        assigned = [f'input.{fixture}="/path/to/{target["input"]}.pt"'] if fixture in config["input"] else []
        reached = load_config(joined, overrides=caller_overrides(assigned), input_preset=target["input"])["input"]
        fields = table["preset"][target["input"]]
        assert {key: reached[key] for key in fields} == fields, old
        assert reached != config["input"], old
    # The 3-camera T32 preset is each merged TOML's own input, file included.
    for name in ("g05/fp16", "hy_vla/w8a16", "pi05/h50/fp16", "lingbot/v2/w8a16", "gr00t/n1_7_droid/256/fp16"):
        config = load_config(configs / f"{name}.toml")
        assert load_config(configs / f"{name}.toml", input_preset="3cam_t32")["input"] == config["input"], name
    # A raised language cap admits the longer LingBot v2 instruction.
    assert isinstance(_check(configs / "lingbot/v2/fp16.toml", "--input", "2cam_t128",
                             "--set", "input.max_language_tokens=128", "--set", "input.observation=/data/t128.pt"), dict)


def test_preset_names_canonical_set_precedence_and_families_without_presets():
    config = load_config(QWEN, input_preset="canonical")["input"]
    assert (config["prefill_tokens"], config["decode_steps"], config["stop_on_eos"]) == (1024, 128, False)
    assigned = caller_overrides(["input.decode_steps=8"], 6)
    config = load_config(QWEN, overrides=assigned, input_preset="p4096_d32")
    assert (config["input"]["prefill_tokens"], config["input"]["decode_steps"]) == (4096, 8)
    assert config["rpu_execution"]["model"]["num_cores"] == 6
    with pytest.raises(ConfigError, match="defines p128_d32.*canonical is p1024_d128"):
        load_config(QWEN, input_preset="p9")
    config = load_config(HYVLA, input_preset="canonical")["input"]
    assert (config["cameras"], config["text_tokens"]) == (2, 32)
    configs = ROOT / "examples/configs"
    config = load_config(configs / "qwen3_vl/vl/4b/w8a16_text_head_vision.toml", input_preset="canonical")["input"]
    assert (config["image_size"], config["prefill_tokens"], config["decode_steps"]) == (896, 1024, 32)
    assert load_config(configs / "qwen3_asr/0_6b/fp16.toml", input_preset="canonical")["input"]["audio_seconds"] == 6
    # DINOv3's processor fixes the image at 224 x 224, so it has no size presets.
    with pytest.raises(ConfigError, match="dinov3 has no input presets"):
        load_config(configs / "dinov3/vit_b/fp16.toml", input_preset="canonical")


def test_image_size_reaches_both_image_paths_as_the_same_square(tmp_path):
    """Qwen3-VL (runner and CPU producer) and Qwen3.5 resize a preset image identically."""
    from PIL import Image
    from examples.common.generation import _image_inputs
    from examples.common.inputs import _controlled_images

    path = tmp_path / "frame.png"
    Image.new("RGB", (640, 480), (40, 90, 160)).save(path)
    for preset, size in (("i448_p256_d32", 448), ("i1024_p1280_d32", 1024)):
        inputs = dict(load_config(ROOT / "examples/configs/qwen3_vl/vl/2b/fp16.toml", input_preset=preset)["input"],
                      images=[str(path)])
        (vl,), (controlled,) = _image_inputs(inputs), _controlled_images(inputs)
        assert vl.size == (size, size) and vl.tobytes() == controlled.tobytes()
    assert _image_inputs({"images": [str(path)]})[0].size == (640, 480)  # Omitted: the file's own size.

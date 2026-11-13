"""Catalog discovery and generated configuration navigation contracts."""
from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def test_cli_dispatch_preserves_user_config_options(monkeypatch, tmp_path):
    from examples.common import cli, logging

    config = tmp_path / "user.toml"
    config.write_text((EXAMPLES / "configs/qwen3/text/0_6b/fp16.toml").read_text().replace(
        "runs = 3", "runs = 7"))
    seen = []
    monkeypatch.setattr(logging, "logging_session", lambda *a, **k: nullcontext())
    monkeypatch.setattr(cli, "_dispatch", lambda config, args: seen.append((config, args)) or 17)
    assert cli.main(["--config", str(config), "--profile"]) == 17
    parsed, args = seen.pop()
    assert parsed["run"]["runs"] == 7 and parsed["run"]["profile"] is True
    assert args.config == config
    assert parsed["example"]["registry_alias"] == "qwen3-0.6b"


def test_cli_symlink_keeps_prompt_relative_to_real_config(monkeypatch, tmp_path):
    from examples.common import cli

    actual = tmp_path / "inputs/config.toml"
    actual.parent.mkdir()
    actual.write_text((EXAMPLES / "configs/qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_p4096_d4096.toml").read_text())
    alias = tmp_path / "config.toml"
    alias.symlink_to(actual)
    seen = []
    monkeypatch.setattr(cli, "_dispatch", lambda config, args: seen.append(config) or 0)
    assert cli.main(["--config", str(alias), "--check-config"]) == 0
    assert seen[0]["input"]["prompt_file"] == str(actual.parent / "prefill_realtext.txt")


@pytest.fixture
def example_catalog(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "check_examples_test", ROOT / "scripts/check_examples.py")
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    examples = tmp_path / "examples"
    shutil.copytree(EXAMPLES / "configs", examples / "configs")
    for entry in EXAMPLES.glob("*.py"):
        if not entry.name.startswith("_"):
            shutil.copy2(entry, examples / entry.name)
    (tmp_path / "docs").mkdir()
    for name in ("api_reference.md", "api_reference.zh.md"):
        shutil.copy2(ROOT / "docs" / name, tmp_path / "docs" / name)
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "EXAMPLES", examples)
    monkeypatch.setattr(checker, "CONFIGS", examples / "configs")
    return checker


def test_catalog_groups_nested_input_variants_without_executing_entries(example_catalog):
    checker = example_catalog
    original = checker.CONFIGS / "qwen3/text/0_6b/fp16.toml"
    variant = checker.CONFIGS / "qwen3/inputs/another_prompt.toml"
    variant.parent.mkdir()
    variant.write_text(original.read_text().replace(
        'prompt = "Explain what a compiler does."', 'prompt = "Explain DMA."'))
    # Discovery is static even if importing/executing an entry would fail.
    entry = checker.EXAMPLES / "run.py"
    entry.write_text('raise RuntimeError("entry must not execute")\n' + entry.read_text())
    catalog = checker.catalog()
    profile_id = "qwen3-0.6b.fp16.text"
    assert {item[1] for item in catalog["api"][profile_id]} == {original, variant}
    public_paths = {item[1] for section in catalog.values()
                    for group in section.values() for item in group}
    assert checker.CONFIGS / "pi05/2cam/fp16.toml" not in public_paths
    assert checker.CONFIGS / "wall_oss/sorting/fp16.toml" not in public_paths
    assert checker.CONFIGS / "rhinovla/v3/controlled/fp16.toml" not in public_paths
    for language in ("en", "zh"):
        rows = [line for line in checker.render(language, catalog).splitlines()
                if f"`{profile_id}`" in line]
        assert len(rows) == 1
        assert original.relative_to(checker.ROOT).as_posix() in rows[0]
        assert variant.relative_to(checker.ROOT).as_posix() in rows[0]


def test_catalog_discovers_renamed_configs_even_when_stems_repeat(example_catalog):
    checker = example_catalog
    original = checker.CONFIGS / "qwen3/text/0_6b/fp16.toml"
    duplicate = checker.CONFIGS / "qwen3/another/fp16.toml"
    duplicate.parent.mkdir()
    shutil.copy2(original, duplicate)
    renamed = original.with_name("renamed.toml")
    original.rename(renamed)
    assert {item[1] for item in checker.catalog()["api"]["qwen3-0.6b.fp16.text"]} == {
        renamed, duplicate}


def test_catalog_requires_the_unified_runtime_entry(example_catalog):
    checker = example_catalog
    (checker.EXAMPLES / "run.py").unlink()
    with pytest.raises(RuntimeError, match="missing unified example entry"):
        checker.catalog()


def test_config_index_includes_unregistered_templates_without_support_inheritance(example_catalog):
    checker = example_catalog
    configs = checker.load_configs()
    index = checker.render_config_index(configs)
    for path in configs:
        relative = path.relative_to(checker.CONFIGS).as_posix()
        assert index.count(f"| [{relative}]({relative}) |") == (
            1 if checker.documented_config(path) else 0
        )
    rows = index.splitlines()
    def row(relative):
        return next(line for line in rows if line.startswith(f"| [{relative}]"))
    assert "未登记；模型专用模板" in row("wall_oss/base/3cam/224/fp16.toml")
    assert "run.py" in row("hy_vla/3cam/fp16.toml")
    assert "12（运行器默认）" in row("hy_vla/3cam/fp16.toml")
    fragment = row("rhinovla/v3/controlled/fp16.toml")
    assert "独立验证片段" in fragment and "validate_rhinovla_v3.py" in fragment and "`--config`" in fragment
    assert "SUPPORTED" not in fragment
    assert "调用者 factory 模板" in row("rhinovla/v3/w8a16_action.toml")
    assert "固定 P/D 复现" in row("qwen3_5/text/0_8b/fp16_p64_d4096.toml")
    assert "SUPPORTED_LIMITED" in row("hy_embodied/umi/fp16.toml")


@pytest.mark.parametrize("relative", [
    "qwen3/text/4b/fp16.toml", "pi05/2cam/fp16.toml",
    "wall_oss/sorting/fp16.toml", "wall_oss/base/3cam/224/fp16.toml", "rhinovla/v3/controlled/fp16.toml",
])
def test_catalog_validates_deep_public_and_custom_configs(example_catalog, relative):
    checker = example_catalog
    original = checker.CONFIGS / relative
    invalid = checker.CONFIGS / "nested/inputs/invalid.toml"
    invalid.parent.mkdir(parents=True)
    invalid.write_text(original.read_text() + "\n[unknown_settings]\nvalue = true\n")
    original.unlink()
    with pytest.raises(RuntimeError, match=r"invalid example config nested/inputs/invalid\.toml"):
        checker.catalog()


def test_catalog_still_requires_each_controlled_profile(example_catalog):
    checker = example_catalog
    profile_id = "qwen3_5-0.8b.w4a16.text-controlled"
    for path in checker.CONFIGS.rglob("*.toml"):
        if f'profile_id = "{profile_id}"' in path.read_text():
            path.unlink()
    with pytest.raises(RuntimeError, match="required profile example drift") as error:
        checker.catalog()
    assert profile_id in str(error.value)


def test_model_matrix_and_examples_do_not_drift():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_examples.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "public adapter profiles" in result.stdout
    assert "VLA profiles" in result.stdout

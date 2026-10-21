#!/usr/bin/env python3
"""Generate or verify the public example/model matrix without device imports."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
CONFIGS = EXAMPLES / "configs"
DOCS = {
    "en": ROOT / "docs" / "model_matrix.md",
    "zh": ROOT / "docs" / "model_matrix.zh.md",
}
CONFIG_INDEX = CONFIGS / "CONFIGURATIONS.md"
# The unoptimized TP8 INT4 example was explicitly withdrawn from the public
# catalog. Its registered asset, support metadata and execution path remain.
RETIRED_PUBLIC_CONFIG_PROFILES = {
    "pi05-libero-finetuned-w4real-kvint8.w4a16-kvint8.vla",
}


def _load_common():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "rpu_examples_common", EXAMPLES / "common/config.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load examples/common/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_common()
CatalogEntry = tuple[Path, Path, dict, dict]
Catalog = dict[str, dict[str, list[CatalogEntry]]]


def _runtime_entry() -> Path:
    """Resolve the user entry without importing or executing model code."""
    path = EXAMPLES / "run.py"
    if not path.is_file():
        raise RuntimeError("missing unified example entry: examples/run.py")
    return path


def load_configs() -> dict[Path, dict]:
    """Validate the whole inventory, including templates without a profile."""
    configs = {}
    for path in sorted(CONFIGS.rglob("*.toml")):
        try:
            configs[path] = COMMON.load_config(path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"invalid example config {path.relative_to(CONFIGS)}: {exc}"
            ) from exc
    return configs


def catalog(configs: dict[Path, dict] | None = None) -> Catalog:
    profiles = COMMON.example_profiles()
    api_entries: dict[str, list[CatalogEntry]] = {}
    vla_entries: dict[str, list[CatalogEntry]] = {}
    if configs is None:
        configs = load_configs()
    entry_path = _runtime_entry()
    # Explicit development examples do not inherit a certified planner alias.
    sorting_configs = [config for config in configs.values()
                       if "wall_oss_sorting" in config]
    precisions = {config["wall_oss_sorting"]["precision"]
                  for config in sorting_configs}
    if len(sorting_configs) != 3 or precisions != {"fp16", "w8a16", "w8a16_action_w4a16"}:
        raise RuntimeError("parcel-sorting precision example drift")
    for config_path, config in configs.items():
        example = COMMON.example_metadata(config)
        if example is None:
            # Extra Pi geometries/precisions, sorting and Rhino v3 runtime
            # fragments have no inherited public profile.
            continue
        profile_id = example["profile_id"]
        profile = profiles[profile_id]
        target = COMMON.profile_runner_target(profile)
        if COMMON.config_target(config) != target:
            raise RuntimeError(f"profile {profile_id} config/runner target mismatch")
        entry = (entry_path, config_path, config, profile)
        architecture = example.get("architecture")
        destination = api_entries if architecture is not None else vla_entries
        destination.setdefault(profile_id, []).append(entry)

    documented = set(COMMON.ARCH_FAMILIES)
    represented = {
        entries[0][2]["example"]["architecture"] for entries in api_entries.values()
    }
    if represented != documented:
        raise RuntimeError(
            "public architecture example drift: "
            f"missing={sorted(documented - represented)}, "
            f"extra={sorted(represented - documented)}"
        )
    eligible_profiles = {
        profile_id
        for profile_id, profile in profiles.items()
        if COMMON.required_example(profile) and profile_id not in RETIRED_PUBLIC_CONFIG_PROFILES
    }
    configured_profiles = set(api_entries) | set(vla_entries)
    if configured_profiles != eligible_profiles:
        raise RuntimeError(
            "required profile example drift: "
            f"missing={sorted(eligible_profiles - configured_profiles)}, "
            f"extra={sorted(configured_profiles - eligible_profiles)}"
        )
    return {"api": api_entries, "vla": vla_entries}


def _support(profile: dict) -> str:
    if profile.get("example_only"):
        return "Caller factory template; not evaluated"
    support = profile["support"]
    return f"`{support['status']}`<br>`{support['reason_code']}`"


def render(
    language: str,
    entries: Catalog,
) -> str:
    represented = {
        group[0][2]["example"]["architecture"] for group in entries["api"].values()
    }
    if set(COMMON.ARCH_FAMILIES) != represented:
        raise RuntimeError("architecture configuration matrix drift")

    if language == "zh":
        title = "# 模型与示例矩阵"
        intro = (
            "由 `python scripts/check_examples.py --write` 生成。配置与登记状态来自 "
            "`examples/configs/profiles.json`；同一 profile 的 TOML 汇总在一行。"
            "使用 `python examples/run.py --config <TOML>` 运行，按配置准备权重和输入。"
        )
        api_scope_title = "## API 范围"
        api_scope_header = "| HF 架构 | 模型族 |"
        api_title = "## 可运行 API profiles"
        api_header = (
            "| HF 架构 | Profile / 精度 | Planner 支持状态 | 共享入口 | TOML / alias |"
        )
        vla_title = "## VLA workload 示例"
        vla_header = "| Family | Profile / 精度 | Planner 支持状态 | 共享入口 | TOML / alias |"
        switch = "[English](model_matrix.md) | 简体中文"
    else:
        title = "# Model and example matrix"
        intro = (
            "Generated by `python scripts/check_examples.py --write` from "
            "`examples/configs/profiles.json`. Each row groups the configurations for "
            "one profile. Prepare its weights and inputs, then run "
            "`python examples/run.py --config <TOML>`."
        )
        api_scope_title = "## API scope"
        api_scope_header = "| HF architecture | Family |"
        api_title = "## Runnable API profiles"
        api_header = (
            "| HF architecture | Profile / precision | Planner support | Shared entry | "
            "TOML / alias |"
        )
        vla_title = "## VLA workload examples"
        vla_header = "| Family | Profile / precision | Planner support | Shared entry | TOML / alias |"
        switch = "English | [简体中文](model_matrix.zh.md)"

    inventory_link = (
        "完整配置及用途见[配置索引](../examples/configs/CONFIGURATIONS.md)；本页仅列登记 profile。"
        if language == "zh" else
        "See the [complete configuration index](../examples/configs/CONFIGURATIONS.md) for all templates and their purposes; this page lists registered profiles."
    )
    lines = [title, "", switch, "", intro, "", inventory_link, "", api_scope_title, "",
             api_scope_header, "|---|---|"]
    for architecture, family in sorted(COMMON.ARCH_FAMILIES.items()):
        lines.append(f"| {architecture} | {family} |")
    lines += ["", api_title, "", api_header, "|---|---|---|---|---|"]
    profile_order = {
        profile_id: index
        for index, profile_id in enumerate(COMMON.example_profiles())
    }
    for _profile_id, group in sorted(
        entries["api"].items(), key=lambda item: (
            item[1][0][2]["example"]["architecture"], profile_order[item[0]]
        )
    ):
        entry_path, _config_path, config, profile = group[0]
        architecture = config["example"]["architecture"]
        entry = entry_path.relative_to(ROOT).as_posix()
        lines.append(
            f"| {architecture} | `{profile['id']}` / `{profile['precision']}` | "
            f"{_support(profile)} | [`{entry}`](../{entry}) | "
            f"{_config_links(group)} |"
        )
    lines += ["", vla_title, "", vla_header, "|---|---|---|---|---|"]
    for _profile_id, group in sorted(
        entries["vla"].items(), key=lambda item: (
            item[1][0][3]["family"], profile_order[item[0]]
        )
    ):
        entry_path, _config_path, _config, profile = group[0]
        entry = entry_path.relative_to(ROOT).as_posix()
        lines.append(
            f"| `{profile['family']}` | `{profile['id']}` / `{profile['precision']}` | "
            f"{_support(profile)} | [`{entry}`](../{entry}) | "
            f"{_config_links(group)} |"
        )
    lines += [""]
    return "\n".join(lines)


def _config_links(entries: list[CatalogEntry]) -> str:
    links = []
    for _entry_path, config_path, config, _profile in entries:
        relative = config_path.relative_to(ROOT).as_posix()
        alias = COMMON.example_metadata(config)["registry_alias"]
        links.append(f"[`{relative}`](../{relative}) / `{alias}`")
    return "<br>".join(links)


def render_config_index(configs: dict[Path, dict]) -> str:
    """Navigation describes configuration facts; it never grants support."""
    profiles = COMMON.example_profiles()
    entry_path = _runtime_entry()
    lines = ["# 完整配置索引", "",
             "由 `python scripts/check_examples.py --write` 生成，覆盖所有 TOML。"
             "登记状态来自各配置的 profile。配置差异、权重来源和路径绑定见模型指南。", "",
             "[选型与使用](README.md) · [登记支持矩阵](../../docs/model_matrix.zh.md) · "
             "[权重准备](WEIGHTS.md) · [通用参数](PARAMETERS.md)", "",
             "CPU 列为请求值，`继承` 表示按运行器或环境设置。"
             "固定 P/D 模板指定输入长度、生成长度及计时方式。", ""]
    family = None
    for path, config in configs.items():
        relative = path.relative_to(CONFIGS)
        if relative.parts[0] != family:
            if family is not None:
                lines.append("")
            family = relative.parts[0]
            lines += [f"## {family}", "",
                      f"[配置说明]({family}/README.md) · [权重准备]({family}/README.md#quantization)", "",
                      "| 配置 | 入口 / 用途 | 精度 / 输入默认 | CPU 请求 | 登记支持 |",
                      "|---|---|---|---|---|"]
        metadata = COMMON.example_metadata(config)
        custom = next((key for key in (
            "g05", "pi05", "hy_vla", "wall_oss", "wall_oss_sorting", "lingbot2",
            "qwen3_asr", "qwen3_tts", "whisper_asr",
        ) if key in config), None)
        fragment = "env" in config
        target = custom or (metadata or {}).get("target")
        if metadata is not None:
            profile = profiles[metadata["profile_id"]]
            target = COMMON.profile_runner_target(profile)
            support = "调用者 factory 模板" if profile.get("example_only") else _support(profile)
        else:
            support = "未登记；独立验证片段" if fragment else "未登记；模型专用模板"
        if fragment:
            entry = "[validate_rhinovla_v3.py](../../scripts/validate_rhinovla_v3.py) `--config`"
        elif COMMON.config_target(config) == target and target is not None:
            link = Path(os.path.relpath(entry_path, CONFIGS)).as_posix()
            entry = f"[{entry_path.name}]({link})"
        else:
            raise RuntimeError(f"config has no public entry: {relative}")
        inputs = config.get("input", {})
        if fragment:
            purpose = "独立验证片段"
        elif "legacy" in relative.parts or "legacy" in relative.stem:
            purpose = "历史复现"
        elif "generic" in relative.parts:
            purpose = "通用执行"
        elif inputs.get("prefill_tokens") is not None:
            purpose = "固定 P/D 复现"
        elif "controlled" in relative.parts:
            purpose = "受控验证"
        elif metadata and profiles[metadata["profile_id"]].get("example_only"):
            purpose = "调用者 factory"
        else:
            purpose = "运行模板"
        section = config.get(custom, {}) if custom else config.get("example", {})
        precision = section.get("precision", section.get("weight_dtype", section.get("dtype", "—")))
        dimensions = []
        for key, label in (("cameras", "相机"), ("image_size", "图像"),
                           ("text_tokens", "T"), ("expected_language_tokens", "有效T"), ("text_lengths", "T列表"),
                           ("prefill_tokens", "P"), ("decode_steps", "D")):
            if key in inputs:
                dimensions.append(f"{label}={inputs[key]}")
        cores = config.get("rpu_execution", {}).get("model", {}).get("num_cores")
        if cores is not None:
            dimensions.append(f"TP={cores}")
        cpu = config.get("cpu", {}).get("threads")
        if cpu is None:
            cpu = config.get("run", {}).get("torch_num_threads", config.get("run", {}).get("cpu_threads"))
        if cpu is None and (custom in {"g05", "pi05", "wall_oss"} or target == "g05"):
            cpu = "8（运行器默认）"
        if cpu is None and custom == "lingbot2":
            cpu = "1（运行器默认）"
        if cpu is None and custom == "hy_vla":
            cpu = "12（运行器默认）"
        if cpu is None and fragment:
            cpu = "12（验证 CLI 默认，可覆盖）"
        detail = "；".join([f"`{precision}`", *dimensions])
        lines.append(f"| [{relative.as_posix()}]({relative.as_posix()}) | {entry}<br>{purpose} | "
                     f"{detail} | {cpu if cpu is not None else '继承'} | {support} |")
    lines += ["", f"合计 **{len(configs)}** 份配置。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite model matrices and the full configuration index")
    args = parser.parse_args()
    try:
        configs = load_configs()
        entries = catalog(configs)
        expected = {path: render(language, entries) for language, path in DOCS.items()}
        expected[CONFIG_INDEX] = render_config_index(configs)
        if args.write:
            for path, content in expected.items():
                path.write_text(content, encoding="utf-8")
        else:
            stale = [
                str(path)
                for path, content in expected.items()
                if not path.is_file()
                or path.read_text(encoding="utf-8") != content
            ]
            if stale:
                raise RuntimeError(f"generated model matrix is stale: {', '.join(stale)}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"PASS: {len(entries['api'])} public adapter profiles and "
        f"{len(entries['vla'])} VLA profiles; EN/ZH matrices and {len(configs)} configurations are in sync"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

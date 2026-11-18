"""Caller overrides of one example configuration, and its TOML serializer.

`examples/run.py --cores/--set` and the Model Zoo bridge
(`scripts/run_zoo_example.py`) share these rules, so a matrix cell reproduced
from the command line and one run by the harness resolve to the same
configuration. `--recipe NAME` selects a template of the same model directory.
`--input NAME` reads a named [input] preset from `examples/inputs/` and applies
it as input overrides. Overrides apply to the
normalized layout before `load_config` validates it; the profile stays the
support authority.
Import-light: no Torch, no backend.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import tomllib

from examples.common.config import MODEL_ASSET_KEYS, ConfigError, config_target
from examples.common.config import _CORE_BUDGETS as CORE_BUDGETS

# Tables a caller may change from the command line. Identity tables
# ([example] and the dedicated runner tables) and bound libraries are not
# here: changing them changes which profile or build is being run.
CALLER_TABLES = ("input", "run", "cpu", "rpu_execution", "g05")
# `--input NAME` presets: one file per input family, outside configs/ so
# catalog scans do not read them as configurations.
INPUT_PRESETS = Path(__file__).resolve().parents[1] / "inputs"


def recipe_path(config, recipe=None):
    """`--recipe NAME`: template NAME.toml next to --config, or inside it when --config is a model directory."""
    directory = config if config.is_dir() else config.parent
    if recipe is None and not config.is_dir():
        return config
    names = sorted(path.stem for path in directory.glob("*.toml"))
    if recipe is None:
        raise ConfigError(f"--config {directory} is a directory; choose a template with --recipe: {', '.join(names)}")
    if recipe not in names:
        raise ConfigError(f"--recipe {recipe}: {directory} has {', '.join(names) or 'no templates'}")
    return directory / f"{recipe}.toml"


def parse_assignment(text):
    """`section.key=value` -> ((section, key), value); VALUE is a TOML literal, bare words are strings."""
    key, separator, raw = text.partition("=")
    path = tuple(part.strip() for part in key.split("."))
    if not separator or len(path) < 2 or not all(path):
        raise ConfigError(f"--set {text!r}: expected section.key=value, "
                          "for example rpu_execution.prefill.chunk_size=128")
    try:
        value = tomllib.loads(f"value = {raw.strip()}")["value"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return path, value


def caller_overrides(assignments=(), cores=None):
    """Nest `--set` assignments (and `--cores`) into override tables; a key set twice is an error."""
    items = [(("rpu_execution", "model", "num_cores"), cores)] if cores is not None else []
    items += [parse_assignment(text) for text in assignments]
    overrides = {}
    for path, value in items:
        node = overrides
        for depth, part in enumerate(path[:-1]):
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"{'.'.join(path)} conflicts with {'.'.join(path[:depth + 1])}")
        if path[-1] in node:
            raise ConfigError(f"{'.'.join(path)} is set twice (--cores sets rpu_execution.model.num_cores)")
        node[path[-1]] = value
    return overrides


def input_family(config):
    """The preset family of a normalized configuration: its runner target, plus the Qwen3-VL component."""
    target, component = config_target(config), config.get("input", {}).get("component")
    return f"{target}_{component}" if target and component else target


@lru_cache(maxsize=None)
def preset_files():
    """Every preset file as (path, table), checked for the fields `--input` reads."""
    files = []
    for path in sorted(INPUT_PRESETS.glob("*.toml")):
        with path.open("rb") as stream:
            table = tomllib.load(stream)
        presets, families = table.get("preset"), table.get("applies_to")
        requires, fixtures = table.get("requires", []), table.get("fixtures", {})
        limits = table.get("limit", [])
        if (not isinstance(presets, dict) or not presets or "canonical" in presets
                or not all(isinstance(value, dict) and value for value in presets.values())
                or table.get("canonical") not in presets
                or not isinstance(families, list) or not families
                or not all(isinstance(name, str) and name for name in families)
                or not isinstance(requires, list) or not all(isinstance(key, str) and key for key in requires)
                or not isinstance(fixtures, dict) or not set(fixtures) <= set(families)
                or not all(isinstance(key, str) and key for key in fixtures.values())
                or not all(set(limit) == {"alias_prefix", "max_prefill_tokens", "reason"} for limit in limits)):
            raise ConfigError(f"{path.name}: an input preset file needs applies_to, canonical and "
                              "nonempty [preset.<name>] tables; requires lists [input] keys, "
                              "fixtures maps a listed family to its input-file key and each "
                              "[[limit]] has alias_prefix, max_prefill_tokens and reason")
        files.append((path, table))
    return files


def family_presets(config):
    """The preset file of this configuration's family as (path, table), or None when it has none."""
    family = input_family(config)
    matches = [(path, table) for path, table in preset_files() if family in table["applies_to"]]
    if len(matches) > 1:
        raise ConfigError(f"--input: {family} is listed by {', '.join(path.name for path, _ in matches)}")
    return matches[0] if matches else None


def preset_fixture(config, table, fields):
    """The [input] file key FIELDS would contradict: an input file holds the shape the TOML declares."""
    fixture, declared = table.get("fixtures", {}).get(input_family(config)), config.get("input", {})
    return fixture if fixture in declared and any(value != declared.get(key) for key, value in fields.items()) else None


def preset_input(config, name, assigned=None):
    """`--input NAME` -> [input] overrides; `--set input.*` fields (ASSIGNED) take precedence."""
    family, declared, assigned = input_family(config), config.get("input", {}), assigned or {}
    located = family_presets(config)
    if located is None:
        known = sorted({family for _, table in preset_files() for family in table["applies_to"]})
        raise ConfigError(f"--input: {family or 'this configuration'} has no input presets "
                          f"(presets exist for {', '.join(known)}); "
                          "set its fields with --set input.<key>=<value>")
    path, table = located
    where = path.relative_to(INPUT_PRESETS.parent).as_posix()
    # A layout without these fields would record the preset without reading it.
    missing = [key for key in table.get("requires", []) if key not in declared]
    if missing:
        raise ConfigError(f"--input: this {family} configuration has no [input] {', '.join(missing)}; "
                          f"the presets of {where} set those fields, so they do not apply to this input layout")
    chosen = table["canonical"] if name == "canonical" else name
    if chosen not in table["preset"]:
        raise ConfigError(f"--input {name!r}: {where} defines "
                          f"{', '.join(table['preset'])}; canonical is {table['canonical']}")
    fields = {**table["preset"][chosen], **assigned}
    alias = config.get("example", {}).get("registry_alias") or ""
    for limit in table.get("limit", []):
        if alias.startswith(limit["alias_prefix"]) and fields.get("prefill_tokens", 0) > limit["max_prefill_tokens"]:
            raise ConfigError(f"--input {chosen}: {limit['reason']}")
    resolved = {key: fields[key] for key in table["preset"][chosen]}
    fixture = preset_fixture(config, table, resolved)
    if fixture is not None and fixture not in assigned:
        shape = ", ".join(f"{key}={declared.get(key)}" for key, value in resolved.items() if value != declared.get(key))
        raise ConfigError(f"--input {chosen}: [input].{fixture} holds this TOML's {shape}; "
                          f"give a file of the preset's shape with --set input.{fixture}=<path>")
    return fields


def execution_overrides(config, overrides):
    """Record public cold choices; the actual config loader validates the profile."""
    def apply(original, changed, prefix=()):
        if not isinstance(changed, dict):
            raise ValueError('execution overrides must be tables')
        for key, value in changed.items():
            if isinstance(value, dict):
                if not prefix and key == 'model' and key not in original:
                    original[key] = {}
                if key not in original or not isinstance(original[key], dict):
                    raise ValueError('execution override must use an existing stage scope')
                apply(original[key], value, (*prefix, key))
            elif key == 'padding_rows' and value is None and prefix and prefix[-1] == 'prefill':
                if key not in original:
                    raise ValueError('prefill padding_rows deletion requires an existing template pin')
                del original[key]
            elif key == 'num_cores' and prefix == ('model',):
                if type(value) is not int or value not in CORE_BUDGETS:
                    raise ValueError('model.num_cores override must be 4, 6 or 8')
                original[key] = value
            elif key == 'chunk_size' and prefix == ('prefill',):
                if key not in original or not (value == 'auto' or
                        type(value) is int and value > 0 and value % 16 == 0):
                    raise ValueError('prefill chunk_size override requires an existing field and auto or a positive multiple of 16')
                original[key] = value
            elif key != 'linear_acc32' or type(value) is not bool or prefix == ('model',):
                # Accumulation is a stage choice; the model scope carries only num_cores.
                raise ValueError('only public core, prefill chunk and accumulation choices may be overridden')
            else:
                original[key] = value
    apply(config.setdefault('rpu_execution', {}), overrides)


def g05_overrides(config, value):
    if ('g05' not in config or set(value) - {'linear_accumulation', 'math_precision'}
            or value.get('linear_accumulation', 'acc16') not in {'acc16', 'acc32'}
            or value.get('math_precision', 'base') not in {'base', 'high'}):
        raise ValueError('G05 override only accepts supported accumulation and math precision')


def apply_overrides(config, overrides):
    """Apply command-line overrides to a normalized configuration in place."""
    unknown = sorted(set(overrides) - set(CALLER_TABLES))
    if unknown:
        raise ConfigError(f"--set changes only {', '.join(CALLER_TABLES)}; {unknown} would change "
                          "the model identity or bound libraries, so choose another TOML")
    for table, value in overrides.items():
        if not isinstance(value, dict):
            raise ConfigError(f"--set {table} needs a field: {table}.<key>=<value>")
        if table == "rpu_execution":
            execution_overrides(config, value)
            continue
        if table == "g05":
            g05_overrides(config, value)
        if table == "input":
            assets = sorted(name for name in value if name in MODEL_ASSET_KEYS
                            or "checkpoint" in name or name.endswith("registry_alias"))
            if assets:
                raise ConfigError(f"input {assets} select model assets; edit them in the TOML")
        destination = config.setdefault(table, {})
        if not isinstance(destination, dict):
            raise ConfigError(f"[{table}] must be a table")
        _merge(destination, value)


def _merge(destination, changed):
    for key, value in changed.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _merge(destination[key], value)
        else:
            destination[key] = value


def toml_text(config):
    def atom(value):
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if type(value) is int:
            return str(value)
        if isinstance(value, float) and math.isfinite(value):
            return repr(value)
        if isinstance(value, list):
            return '[' + ', '.join(atom(x) for x in value) + ']'
        raise ValueError('config contains an unsupported TOML value')
    lines = []
    def table(value, prefix=()):
        if prefix:
            lines.append('[' + '.'.join(json.dumps(k) for k in prefix) + ']')
        for key, item in value.items():
            if not isinstance(item, dict):
                lines.append(json.dumps(key) + ' = ' + atom(item))
        lines.append('')
        for key, item in value.items():
            if isinstance(item, dict):
                table(item, (*prefix, key))
    table(config)
    return '\n'.join(lines)


def config_sha256(config):
    """SHA-256 of the canonical JSON form; report.json records the same value as provenance.config_sha256."""
    data = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()

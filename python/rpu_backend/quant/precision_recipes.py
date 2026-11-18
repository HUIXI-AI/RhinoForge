"""Precision recipes: the one registered name a TOML ``precision`` takes.

``manifests/precision_recipes.v1.json`` expands each name into per-module
rules (a default spec plus component / role / layer rules, the most specific
one winning), the runner's registered selector, the profiles that realize it,
the profile limits it implies, the cold switches it sets and the converted
checkpoints it loads. Each implementation owner admits a recipe by the specs it
resolves to (``route_key``), never by its name. Converters record the recipe they produced in
``rpu_precision_recipe.json`` next to the weights. Stdlib only, so the example
loader, the catalog tools and the converters read it without importing Torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import lru_cache
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[3] / "manifests" / "precision_recipes.v1.json"
SCHEMA = "rpu.precision_recipes.v1"
FORMATS = {"fp16": "FP16", "w8a16": "W8A16", "w8a8": "W8A8", "w4a16": "W4A16",
           "nvfp4": "NVFP4", "fp8_e4m3": "FP8 E4M3", "cpu_fp32": "CPU FP32"}
METHODS = {"rtn": "RTN", "awq": "AWQ", "activation_awq": "activation AWQ", "gptq": "GPTQ"}
# Where a switch lands in the normalized config: a [model.opt_in] env or an [input] key.
SWITCH_TABLES = ("opt_in", "input")
# The [input] keys that name a weight directory a recipe may bind to a converted recipe.
CHECKPOINT_SLOTS = ("checkpoint", "fp16_checkpoint", "w8_checkpoint", "action_w4_checkpoint")
RECORD = "rpu_precision_recipe.json"
RECORD_SCHEMA = "rpu.precision_recipe_record.v1"


class RecipeError(ValueError):
    """A precision name or registry entry that cannot be resolved."""


@dataclass(frozen=True)
class Spec:
    format: str
    group: int | str | None = None
    method: str | None = None

    def text(self) -> str:
        words = [FORMATS[self.format]]
        if self.group == "channel":
            words.append("逐通道")
        elif self.group is not None:
            words.append(f"G{self.group}")
        if self.method is not None:
            words.append(METHODS[self.method])
        return " ".join(words)

    def key(self) -> str:
        """The spelling owner route tables use: format, then channel / g<group>, then method."""
        group = None if self.group is None else "channel" if self.group == "channel" else f"g{self.group}"
        return "-".join(part for part in (self.format, group, self.method) if part is not None)


@dataclass(frozen=True)
class Rule:
    component: str
    roles: tuple[str, ...]
    layers: tuple[int, int] | None
    spec: Spec

    @property
    def rank(self) -> int:
        return bool(self.roles) + bool(self.layers)


@dataclass(frozen=True)
class Recipe:
    family: str
    name: str
    label: str
    selector: str
    default: Spec
    rules: tuple[Rule, ...]
    profiles: tuple[str, ...]
    limits: dict
    switches: tuple[dict, ...]
    checkpoints: dict

    def spec(self, component: str, role: str | None = None, layer: int | None = None) -> Spec:
        return resolve(self, component, role, layer)


@dataclass(frozen=True)
class Family:
    name: str
    targets: tuple[str, ...]
    components: dict
    recipes: dict


def _spec(value, where) -> Spec:
    if isinstance(value, str):
        value = {"format": value}
    if not isinstance(value, dict) or set(value) - {"format", "group", "method"}:
        raise RecipeError(f"{where}: a spec is a format name or {{format, group, method}}")
    spec = Spec(value.get("format"), value.get("group"), value.get("method"))
    if spec.format not in FORMATS:
        raise RecipeError(f"{where}: unknown format {spec.format!r}")
    if spec.group is not None and spec.group != "channel" and (
            type(spec.group) is not int or spec.group <= 0):
        raise RecipeError(f"{where}: group is a positive integer or 'channel'")
    if spec.method is not None and spec.method not in METHODS:
        raise RecipeError(f"{where}: unknown method {spec.method!r}")
    if (spec.group is not None or spec.method is not None) and spec.format not in ("w8a16", "w4a16"):
        raise RecipeError(f"{where}: group/method apply to integer weight formats only")
    return spec


def _rule(value, components, where) -> Rule:
    if not isinstance(value, dict) or set(value) - {"component", "roles", "layers", "spec"}:
        raise RecipeError(f"{where}: a rule has component, optional roles/layers and spec")
    component = value.get("component")
    if component not in components:
        raise RecipeError(f"{where}: unknown component {component!r}")
    roles = value.get("roles", [])
    known = components[component].get("roles", [])
    if not isinstance(roles, list) or len(set(roles)) != len(roles) or set(roles) - set(known):
        raise RecipeError(f"{where}: roles must be distinct roles of {component} {known}")
    layers = value.get("layers")
    if layers is not None:
        if (not isinstance(layers, list) or len(layers) != 2
                or any(type(item) is not int or item < 0 for item in layers) or layers[0] > layers[1]):
            raise RecipeError(f"{where}: layers is an inclusive [first, last] range")
        layers = tuple(layers)
    return Rule(component, tuple(roles), layers, _spec(value.get("spec"), where))


def _overlap(a: Rule, b: Rule) -> bool:
    roles = not a.roles or not b.roles or bool(set(a.roles) & set(b.roles))
    layers = a.layers is None or b.layers is None or (a.layers[0] <= b.layers[1] and b.layers[0] <= a.layers[1])
    return a.component == b.component and a.rank == b.rank and roles and layers


def _switches(value, where) -> tuple[dict, ...]:
    if not isinstance(value, list):
        raise RecipeError(f"{where}: switches is a list")
    rows = []
    for index, row in enumerate(value):
        if (not isinstance(row, dict) or set(row) - {"profiles", *SWITCH_TABLES}
                or any(not isinstance(row.get(table, {}), dict) for table in SWITCH_TABLES)):
            raise RecipeError(f"{where}.switches[{index}]: profiles plus opt_in/input tables")
        rows.append(row)
    return tuple(rows)


def _recipe(family, name, value, components) -> Recipe:
    where = f"{family}.{name}"
    allowed = {"label", "selector", "default", "rules", "profiles", "limits", "switches", "checkpoints"}
    if not isinstance(value, dict) or set(value) - allowed or not {"label", "selector"} <= set(value):
        raise RecipeError(f"{where}: a recipe has label, selector and optional {sorted(allowed)}")
    rules = tuple(_rule(rule, components, f"{where}.rules[{index}]")
                  for index, rule in enumerate(value.get("rules", [])))
    for index, rule in enumerate(rules):
        for other in rules[:index]:
            if _overlap(rule, other):
                raise RecipeError(f"{where}: rules for {rule.component} overlap at the same specificity")
    profiles = value.get("profiles", [])
    if not isinstance(profiles, list) or any(not isinstance(item, str) or not item for item in profiles):
        raise RecipeError(f"{where}: profiles is a list of profile ID patterns")
    limits = value.get("limits", {})
    if not isinstance(limits, dict):
        raise RecipeError(f"{where}: limits is a table")
    checkpoints = value.get("checkpoints", {})
    if (not isinstance(checkpoints, dict) or set(checkpoints) - set(CHECKPOINT_SLOTS)
            or any(not isinstance(ref, str) or ref.count("/") > 1 for ref in checkpoints.values())):
        raise RecipeError(f"{where}: checkpoints maps {list(CHECKPOINT_SLOTS)} to '<recipe>' or '<family>/<recipe>'")
    checkpoints = {slot: tuple(ref.split("/")) if "/" in ref else (family, ref) for slot, ref in checkpoints.items()}
    return Recipe(family, name, value["label"], value["selector"], _spec(value.get("default", "fp16"), where),
                  rules, tuple(profiles), dict(limits), _switches(value.get("switches", []), where), checkpoints)


def parse(data: dict) -> dict[str, Family]:
    """Validate a registry document and return its families by name."""
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or not isinstance(data.get("families"), dict):
        raise RecipeError(f"precision recipe registry must declare schema {SCHEMA} and families")
    families, owners = {}, {}
    for name, value in data["families"].items():
        if (not isinstance(value, dict) or set(value) != {"targets", "components", "recipes"}
                or not isinstance(value["targets"], list) or not value["targets"]
                or not isinstance(value["components"], dict) or not isinstance(value["recipes"], dict)
                or not value["recipes"]):
            raise RecipeError(f"family {name}: requires targets, components and recipes")
        for target in value["targets"]:
            if target in owners:
                raise RecipeError(f"runner target {target} belongs to {owners[target]} and {name}")
            owners[target] = name
        for component, entry in value["components"].items():
            if not isinstance(entry, dict) or set(entry) - {"label", "roles"} or not entry.get("label"):
                raise RecipeError(f"family {name}: component {component} needs a label and optional roles")
        recipes = {recipe: _recipe(name, recipe, entry, value["components"])
                   for recipe, entry in value["recipes"].items()}
        families[name] = Family(name, tuple(value["targets"]), value["components"], recipes)
    for family in families.values():
        for item in family.recipes.values():
            for slot, (owner, ref) in item.checkpoints.items():
                if ref not in getattr(families.get(owner), "recipes", {}):
                    raise RecipeError(f"{family.name}.{item.name}: {slot} names unregistered recipe {owner}/{ref}")
    return families


@lru_cache(maxsize=4)
def load(path: Path = REGISTRY) -> dict[str, Family]:
    return parse(json.loads(Path(path).read_text(encoding="utf-8")))


def family_for(target: str, path: Path = REGISTRY) -> Family:
    for family in load(path).values():
        if target in family.targets:
            return family
    raise RecipeError(f"runner target {target!r} has no registered precision recipes")


def recipe(target: str, name, path: Path = REGISTRY) -> Recipe:
    """The recipe a TOML ``precision`` names for this runner target."""
    family = family_for(target, path)
    if name not in family.recipes:
        raise RecipeError(f"precision must name a registered {family.name} recipe "
                          f"{sorted(family.recipes)}, got {name!r}")
    return family.recipes[name]


def by_selector(target: str, selector, path: Path = REGISTRY) -> Recipe:
    """The one recipe whose runner selector a normalized config carries."""
    family = family_for(target, path)
    found = [item for item in family.recipes.values() if item.selector == selector]
    if len(found) != 1:
        raise RecipeError(f"{family.name} runner precision {selector!r} names {len(found)} recipes, not one")
    return found[0]


def for_profile(target: str, profile_id: str, path: Path = REGISTRY) -> Recipe:
    """The one recipe that binds a profile; legacy layouts reach their recipe this way too."""
    family = family_for(target, path)
    found = [item for item in family.recipes.values() if item.profiles and binds(item, profile_id)]
    if len(found) != 1:
        raise RecipeError(f"profile {profile_id} is bound by {len(found)} {family.name} recipes, not one")
    return found[0]


def binds(item: Recipe, profile_id: str | None) -> bool:
    """Profile-less entries take any recipe of their family; a profile must match a recipe pattern."""
    if profile_id is None:
        return True
    return any(fnmatchcase(profile_id, pattern) for pattern in item.profiles)


def switches(item: Recipe, profile_id: str | None) -> dict:
    """The cold switches this recipe sets for the given profile (first matching row)."""
    for row in item.switches:
        patterns = row.get("profiles")
        if patterns is None or (profile_id is not None
                                and any(fnmatchcase(profile_id, pattern) for pattern in patterns)):
            return {table: dict(row.get(table, {})) for table in SWITCH_TABLES}
    return {table: {} for table in SWITCH_TABLES}


def resolve(item: Recipe, component: str, role: str | None = None, layer: int | None = None) -> Spec:
    """The spec of one module: the most specific matching rule, else the default."""
    matches = [rule for rule in item.rules if rule.component == component
               and (not rule.roles or role in rule.roles)
               and (rule.layers is None or (layer is not None and rule.layers[0] <= layer <= rule.layers[1]))]
    if not matches:
        return item.default
    best = max(rule.rank for rule in matches)
    specs = {rule.spec for rule in matches if rule.rank == best}
    if len(specs) != 1:
        raise RecipeError(f"{item.family}.{item.name}: {component} {role} layer {layer} matches conflicting rules")
    return specs.pop()


def route_key(item: Recipe, components: dict, probes) -> tuple[str, ...]:
    """The spec keys of PROBES, each ``(component[, role[, layer]])``, for an owner's route table.

    The probes must observe the whole recipe: every component and role of the
    family without a layer, and every layer-ranged rule at one of its layers.
    A rule the owner does not look at would otherwise run unnoticed.
    """
    probes = [tuple(probe) for probe in probes]
    modules = [(component,) for component in components]
    modules += [(component, role) for component, entry in components.items() for role in entry.get("roles", [])]
    missing = [module for module in modules if module not in probes]
    for rule in item.rules:
        if rule.layers and not any(len(probe) == 3 and probe[0] == rule.component
                                   and (not rule.roles or probe[1] in rule.roles)
                                   and rule.layers[0] <= probe[2] <= rule.layers[1] for probe in probes):
            missing.append((rule.component, *rule.roles, rule.layers))
    if missing:
        raise RecipeError(f"{item.family}.{item.name}: the owner's route probes do not observe {missing}")
    return tuple(resolve(item, *probe).key() for probe in probes)


def describe(item: Recipe, components: dict) -> str:
    """Catalog quant-scope text: the rules in registry order, grouped by spec; '—' when none."""
    if not item.rules:
        return "—" if item.default.format == "fp16" else f"全部投影 {item.default.text()}"
    groups: list[tuple[Spec, list[str]]] = []
    for rule in item.rules:
        if rule.rank:
            continue
        for spec, names in groups:
            if spec == rule.spec:
                names.append(rule.component)
                break
        else:
            groups.append((rule.spec, [rule.component]))
    based = {name for _, names in groups for name in names}
    standalone = [rule for rule in item.rules if rule.rank and rule.component not in based]

    def where(rule: Rule, named: bool) -> str:
        words = [components[rule.component]["label"]] if named else []
        if rule.layers:
            first, last = rule.layers
            words.append(f"第 {first} 层" if first == last else f"第 {first}–{last} 层")
        if rule.roles:
            words.append("/".join(rule.roles))
        return " ".join(words)

    parts = []
    for spec, names in groups:
        text = " / ".join(components[name]["label"] for name in names) + " " + spec.text()
        exceptions = [f"{where(rule, len(names) > 1)} {rule.spec.text()}" for rule in item.rules
                      if rule.rank and rule.component in names]
        parts.append(text + (f"（{'，'.join(exceptions)}）" if exceptions else ""))
    parts.extend(f"{where(rule, True)} {rule.spec.text()}" for rule in standalone)
    return "；".join(parts)


def checkpoint_recipe(item: Recipe, slot: str = "checkpoint") -> tuple[str, str] | None:
    """The (family, recipe) whose converted weights SLOT holds; None when the recipe binds none.

    A registered ``checkpoints`` entry wins. Otherwise the main checkpoint is the
    dense source for a recipe that quantizes on install and its own conversion
    for any other recipe; the other slots bind only when registered.
    """
    if slot in item.checkpoints:
        return item.checkpoints[slot]
    if slot != "checkpoint":
        return None
    return (item.family, "fp16") if item.limits.get("on_install_quantization") else (item.family, item.name)


def produced(family: str, name: str | None, requested: str | None = None) -> str | None:
    """The recipe a converter's arguments produce, after checking it is the REQUESTED one.

    NAME is None for a mix no recipe registers; such weights get no record.
    Converters call this before writing anything, so a wrong --recipe fails fast.
    """
    if requested is not None and requested != name:
        raise RecipeError(f"these converter arguments produce "
                          f"{f'{family}/{name}' if name else 'no registered recipe'}, not {family}/{requested}")
    return name


def write_record(directory, family: str, name: str | None) -> None:
    """Record in DIRECTORY that a converter produced FAMILY/NAME; nothing for an unregistered mix."""
    if name is not None:
        (Path(directory) / RECORD).write_text(
            json.dumps({"schema": RECORD_SCHEMA, "family": family, "recipe": name}, indent=2) + "\n",
            encoding="utf-8")


def read_record(directory) -> tuple[str, str] | None:
    """The (family, recipe) a converted checkpoint records; None for weights without a record."""
    path = Path(directory) / RECORD
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RecipeError(f"{path}: not JSON: {exc}") from exc
    if (not isinstance(value, dict) or set(value) != {"schema", "family", "recipe"}
            or value["schema"] != RECORD_SCHEMA
            or any(not isinstance(value[key], str) or not value[key] for key in ("family", "recipe"))):
        raise RecipeError(f"{path}: a record holds schema {RECORD_SCHEMA}, family and recipe")
    return value["family"], value["recipe"]


def check_record(item: Recipe, directory, slot: str = "checkpoint") -> None:
    """Reject weights in DIRECTORY recorded for another recipe than the one ITEM loads from SLOT."""
    recorded = read_record(directory)
    expected = checkpoint_recipe(item, slot)
    if recorded is not None and expected is not None and recorded != expected:
        raise RecipeError(f"{slot} {directory} was converted for precision recipe {'/'.join(recorded)}; "
                          f"{item.family}/{item.name} loads {'/'.join(expected)}")


def stem_recipe(family: Family, stem: str) -> str | None:
    """The longest registered recipe name a file stem equals or starts with (``<name>_<variant>``)."""
    names = [name for name in family.recipes if stem == name or stem.startswith(name + "_")]
    return max(names, key=len) if names else None

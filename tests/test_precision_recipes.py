"""Precision recipe registry: one TOML precision name, expanded and checked against its profile."""
from __future__ import annotations

import copy
from fnmatch import fnmatchcase
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RECIPES = _load("precision_recipes_under_test", ROOT / "python/rpu_backend/quant/precision_recipes.py")
COMMON = _load("examples_common_precision_recipes", ROOT / "examples/common/config.py")
W4_RTN32 = {"format": "w4a16", "group": 32, "method": "rtn"}


def _toy(**recipes) -> dict:
    return {"schema": RECIPES.SCHEMA, "families": {"toy": {
        "targets": ["toy"],
        "components": {"text": {"label": "文本层", "roles": ["QKV", "O", "Down"]}, "head": {"label": "输出头"}},
        "recipes": recipes or {"fp16": {"label": "fp16", "selector": "fp16"}}}}}


def _one(entry: dict):
    return RECIPES.parse(_toy(item=entry))["toy"].recipes["item"]


def test_the_most_specific_rule_wins_and_the_scope_text_follows_registry_order():
    item = _one({"label": "w4a16", "selector": "w4a16", "rules": [
        {"component": "text", "spec": W4_RTN32},
        {"component": "text", "roles": ["Down"], "spec": "w8a16"},
        {"component": "text", "roles": ["Down"], "layers": [0, 1], "spec": "fp16"},
        {"component": "head", "spec": "w8a16"}]})
    assert RECIPES.resolve(item, "text", "QKV", 7) == RECIPES.Spec("w4a16", 32, "rtn")
    assert RECIPES.resolve(item, "text", "Down", 7).format == "w8a16"
    assert RECIPES.resolve(item, "text", "Down", 1).format == "fp16"
    # A layer-ranged rule never applies when the caller names no layer.
    assert RECIPES.resolve(item, "text", "Down").format == "w8a16"
    assert RECIPES.resolve(item, "head").format == "w8a16"
    assert RECIPES.describe(item, _toy()["families"]["toy"]["components"]) == (
        "文本层 W4A16 G32 RTN（Down W8A16，第 0–1 层 Down FP16）；输出头 W8A16")
    assert RECIPES.describe(_one({"label": "fp16", "selector": "fp16"}), {}) == "—"


@pytest.mark.parametrize("change, message", [
    ({"rules": [{"component": "text", "spec": "int3"}]}, "unknown format"),
    ({"default": {"format": "fp16", "group": 32}}, "integer weight formats only"),
    ({"rules": [{"component": "text", "spec": {"format": "w4a16", "group": 0}}]}, "positive integer"),
    ({"rules": [{"component": "vision", "spec": "w8a16"}]}, "unknown component"),
    ({"rules": [{"component": "text", "roles": ["Gate"], "spec": "w8a16"}]}, "distinct roles"),
    ({"rules": [{"component": "text", "layers": [3, 1], "spec": "w8a16"}]}, "inclusive"),
    ({"rules": [{"component": "text", "spec": "w8a16"}, {"component": "text", "spec": W4_RTN32}]},
     "overlap at the same specificity"),
    ({"rules": [{"component": "text", "roles": ["O", "Down"], "spec": "w8a16"},
                {"component": "text", "roles": ["Down"], "spec": "fp16"}]}, "overlap at the same specificity"),
    ({"switches": [{"opt_in": "1"}]}, "profiles plus opt_in/input tables"),
    ({"quantization": {"text": "w8a16"}}, "a recipe has label, selector"),
    ({"checkpoints": {"weights": "fp16"}}, "checkpoints maps"),
    ({"checkpoints": {"checkpoint": "toy/fp16/w8"}}, "checkpoints maps"),
    ({"checkpoints": {"checkpoint": "missing"}}, "names unregistered recipe toy/missing"),
])
def test_a_registry_entry_off_the_grammar_is_rejected(change, message):
    with pytest.raises(RECIPES.RecipeError, match=message):
        _one({"label": "w8a16", "selector": "w8a16", **change})


def test_a_runner_target_belongs_to_one_family_and_names_resolve_per_target(tmp_path):
    data = _toy()
    data["families"]["other"] = copy.deepcopy(data["families"]["toy"])
    with pytest.raises(RECIPES.RecipeError, match="belongs to toy and other"):
        RECIPES.parse(data)
    with pytest.raises(RECIPES.RecipeError, match="must declare schema"):
        RECIPES.parse({"families": {}})
    path = tmp_path / "recipes.json"
    path.write_text(json.dumps(_toy()), encoding="utf-8")
    assert RECIPES.recipe("toy", "fp16", path).selector == "fp16"
    with pytest.raises(RECIPES.RecipeError, match="registered toy recipe"):
        RECIPES.recipe("toy", "w8a16", path)
    with pytest.raises(RECIPES.RecipeError, match="no registered precision recipes"):
        RECIPES.recipe("qwen3", "fp16", path)


def test_profiles_bind_a_recipe_and_the_first_matching_switch_row_applies():
    item = _one({"label": "w8a16", "selector": "w8a16", "profiles": ["toy-*.w8a16.*"], "switches": [
        {"profiles": ["*.source"], "opt_in": {"TOY_W8A16": "1"}},
        {"opt_in": {"TOY_W8A16": "0"}, "input": {"w8a16": True}}]})
    assert RECIPES.binds(item, None) and RECIPES.binds(item, "toy-1b.w8a16.source")
    assert not RECIPES.binds(item, "toy-1b.fp16.source")
    # A recipe that registers no profile is for profile-less entries only.
    assert not RECIPES.binds(_one({"label": "fp16", "selector": "fp16"}), "toy-1b.fp16.source")
    assert RECIPES.switches(item, "toy-1b.w8a16.source") == {"opt_in": {"TOY_W8A16": "1"}, "input": {}}
    assert RECIPES.switches(item, "toy-1b.w8a16.factory") == {"opt_in": {"TOY_W8A16": "0"}, "input": {"w8a16": True}}
    assert RECIPES.switches(item, None)["opt_in"] == {"TOY_W8A16": "0"}


def test_a_file_stem_is_the_longest_recipe_name_it_equals_or_extends():
    family = RECIPES.parse(_toy(**{name: {"label": name, "selector": name}
                                   for name in ("fp16", "w8a16", "w8a16_action")}))["toy"]
    assert RECIPES.stem_recipe(family, "w8a16_action_high_fused") == "w8a16_action"
    assert RECIPES.stem_recipe(family, "w8a16_text") == "w8a16"
    assert RECIPES.stem_recipe(family, "fp16") == "fp16"
    assert RECIPES.stem_recipe(family, "w8a16x") is None


def test_every_published_recipe_pattern_binds_a_registered_profile():
    profiles = COMMON.example_profiles()
    for family in RECIPES.load(COMMON.PRECISION_RECIPES).values():
        for name, item in family.recipes.items():
            for pattern in item.profiles:
                assert any(fnmatchcase(profile, pattern) for profile in profiles), (family.name, name, pattern)


def test_a_catalog_group_holds_one_recipe_and_one_scope():
    # Group IDs are ASCII slugs; the Chinese scope text alone would collapse distinct scopes into one group.
    catalog_path = ROOT / "examples/configs/catalog.json"
    if not catalog_path.is_file():
        pytest.skip("examples/configs/catalog.json is internal-only")
    entries = json.loads(catalog_path.read_text(encoding="utf-8"))["entries"]
    groups: dict[str, set] = {}
    for entry in entries:
        groups.setdefault(entry["group_id"], set()).add((entry["recipe"], entry["quant_scope"]))
    assert {group: rows for group, rows in groups.items() if len(rows) > 1} == {}


@pytest.mark.parametrize("name, profile_id, edit, message", [
    ("nvfp4_text", "qwen3-0.6b.nvfp4.text",
     lambda entry: entry["limits"].update(on_install_quantization="w4a16"), "disagree on on_install_quantization"),
    ("w4a16_text_rtn_g32", "qwen3-0.6b.w4a16-g32.text",
     lambda entry: entry["rules"][0]["spec"].update(group=128), "group_size"),
    ("w4a16_text_rtn_g32", "qwen3-0.6b.w4a16-g32.text",
     lambda entry: entry.update(profiles=["qwen3-*.w4a16-g128.text"]), "not registered for profile"),
])
def test_a_recipe_that_drifts_from_its_profile_fails_at_load(monkeypatch, tmp_path, name, profile_id, edit, message):
    data = json.loads(COMMON.PRECISION_RECIPES.read_text(encoding="utf-8"))
    edit(data["families"]["qwen3"]["recipes"][name])
    path = tmp_path / "precision_recipes.v1.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert COMMON.precision_recipe("qwen3", name, profile_id).name == name
    monkeypatch.setattr(COMMON, "PRECISION_RECIPES", path)
    with pytest.raises(COMMON.ConfigError, match=message):
        COMMON.precision_recipe("qwen3", name, profile_id)


def test_a_runner_selector_or_a_bound_profile_names_one_recipe(tmp_path):
    data = _toy(fp16={"label": "fp16", "selector": "fp16", "profiles": ["toy-*.fp16"]},
                w8a16={"label": "w8a16", "selector": "w8a16", "profiles": ["toy-1b.w8a16"]},
                w8a16_head={"label": "w8a16", "selector": "w8a16", "profiles": ["toy-*.w8a16"]})
    path = tmp_path / "recipes.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert RECIPES.by_selector("toy", "fp16", path).name == "fp16"
    for selector, count in (("w8a16", 2), ("int4", 0)):
        with pytest.raises(RECIPES.RecipeError, match=f"names {count} recipes"):
            RECIPES.by_selector("toy", selector, path)
    assert RECIPES.for_profile("toy", "toy-2b.w8a16", path).name == "w8a16_head"
    for profile_id, count in (("toy-1b.w8a16", 2), ("toy-1b.int4", 0)):
        with pytest.raises(RECIPES.RecipeError, match=f"bound by {count} toy recipes"):
            RECIPES.for_profile("toy", profile_id, path)


def test_the_cold_install_quantization_comes_from_the_recipe(monkeypatch, tmp_path):
    profile_id = "qwen3-0.6b.nvfp4.text"
    assert COMMON.install_quantization("qwen3", profile_id) == "nvfp4"
    assert COMMON.install_quantization("qwen3", "qwen3-0.6b.fp16.text") is None
    data = json.loads(COMMON.PRECISION_RECIPES.read_text(encoding="utf-8"))
    data["families"]["qwen3"]["recipes"]["nvfp4_text"]["limits"]["on_install_quantization"] = "w4a16"
    path = tmp_path / "precision_recipes.v1.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(COMMON, "PRECISION_RECIPES", path)
    # The runner installs what the recipe says; loading the TOML still refuses the disagreement.
    assert COMMON.install_quantization("qwen3", profile_id) == "w4a16"
    with pytest.raises(COMMON.ConfigError, match="disagree on on_install_quantization"):
        COMMON.precision_recipe("qwen3", "nvfp4_text", profile_id)


def test_a_conversion_records_its_recipe_and_a_malformed_record_is_rejected(tmp_path):
    assert RECIPES.read_record(tmp_path) is None
    RECIPES.write_record(tmp_path, "toy", None)
    assert RECIPES.read_record(tmp_path) is None
    RECIPES.write_record(tmp_path, "toy", "w8")
    assert RECIPES.read_record(tmp_path) == ("toy", "w8")
    record = tmp_path / RECIPES.RECORD
    for text, message in (("{", "not JSON"),
                          (json.dumps({"schema": "other", "family": "toy", "recipe": "w8"}), "a record holds"),
                          (json.dumps({"schema": RECIPES.RECORD_SCHEMA, "family": "toy"}), "a record holds")):
        record.write_text(text)
        with pytest.raises(RECIPES.RecipeError, match=message):
            RECIPES.read_record(tmp_path)
    assert RECIPES.produced("toy", "w8", "w8") == "w8" and RECIPES.produced("toy", None) is None
    with pytest.raises(RECIPES.RecipeError, match="produce no registered recipe, not toy/w8"):
        RECIPES.produced("toy", None, "w8")


def test_each_weight_slot_binds_the_conversion_its_recipe_loads(tmp_path):
    recipes = RECIPES.parse(_toy(
        fp16={"label": "fp16", "selector": "fp16"},
        w8={"label": "w8", "selector": "w8"},
        install={"label": "w4", "selector": "w4", "limits": {"on_install_quantization": "w4a16"}},
        shared={"label": "w8", "selector": "shared", "checkpoints": {"checkpoint": "w8", "w8_checkpoint": "toy/w8"}},
    ))["toy"].recipes
    assert RECIPES.checkpoint_recipe(recipes["w8"]) == ("toy", "w8")
    assert RECIPES.checkpoint_recipe(recipes["install"]) == ("toy", "fp16")
    assert RECIPES.checkpoint_recipe(recipes["shared"], "w8_checkpoint") == ("toy", "w8")
    assert RECIPES.checkpoint_recipe(recipes["w8"], "w8_checkpoint") is None
    RECIPES.write_record(tmp_path, "toy", "w8")
    RECIPES.check_record(recipes["shared"], tmp_path)
    RECIPES.check_record(recipes["w8"], tmp_path, "w8_checkpoint")
    with pytest.raises(RECIPES.RecipeError, match="converted for precision recipe toy/w8; toy/install loads toy/fp16"):
        RECIPES.check_record(recipes["install"], tmp_path)


def test_an_owner_route_key_must_observe_every_rule_of_the_recipe():
    item = _one({"label": "w4a16", "selector": "w4a16", "rules": [
        {"component": "text", "spec": W4_RTN32},
        {"component": "text", "roles": ["Down"], "spec": "w8a16"},
        {"component": "head", "spec": "w8a16"}]})
    components = _toy()["families"]["toy"]["components"]
    probes = (("text",), ("text", "QKV"), ("text", "O"), ("text", "Down"), ("head",))
    assert RECIPES.route_key(item, components, probes) == ("w4a16-g32-rtn", "w4a16-g32-rtn", "w4a16-g32-rtn", "w8a16", "w8a16")
    assert RECIPES.Spec("w4a16", "channel").key() == "w4a16-channel" and RECIPES.Spec("nvfp4").key() == "nvfp4"
    # An owner that does not look at Down (or at a layer-ranged rule) would run it unnoticed.
    with pytest.raises(RECIPES.RecipeError, match=r"do not observe \[\('text', 'Down'\)\]"):
        RECIPES.route_key(item, components, probes[:3] + probes[4:])
    layered = _one({"label": "w8a16", "selector": "w8a16", "rules": [
        {"component": "text", "spec": "w8a16"}, {"component": "head", "spec": "w8a16"},
        {"component": "text", "roles": ["Down"], "layers": [2, 3], "spec": "fp16"}]})
    with pytest.raises(RECIPES.RecipeError, match=r"\('text', 'Down', \(2, 3\)\)"):
        RECIPES.route_key(layered, components, probes)
    assert RECIPES.route_key(layered, components, (*probes, ("text", "Down", 3)))[-1] == "fp16"


# Each runner owner's route table: the target whose recipes it admits, whether it serves
# table entries (every recipe) or profile entries (recipes that bind a profile), and the
# recipes it refuses on purpose.
def _owner_routes():
    from examples.runners.g05 import optimized as g05
    from examples.runners.gr00t import config as gr00t
    from examples.runners.hy_vla import optimized as hy_vla, runner as hy_policy
    from examples.runners.lingbot2 import optimized as lingbot2, runner as lingbot2_policy
    from examples.runners.pi05 import optimized as pi05
    from examples.runners.qwen3_audio import runner as audio
    from examples.runners.wall_oss import base, rtc, runner as wall_policy, sorting
    return [("pi05", pi05._ROUTE_PROBES, pi05._RECIPE_ROUTES, "selector", ()),
            # The base table runs no W4; the W4 recipes are the rc9 profiles' policy route.
            ("wall_oss", base.ROUTE_PROBES, base.ROUTES, "selector", ("w4a16_channel", "w4a16_g32")),
            ("wall_oss", wall_policy._ROUTE_PROBES, wall_policy._ROUTES, "profile", ()),
            ("wall_oss_sorting", sorting.ROUTE_PROBES, sorting.ROUTES, "selector", ()),
            ("wall_oss_rtc", rtc._ROUTE_PROBES, rtc._ROUTES, "profile", ()),
            ("g05", g05._ROUTE_PROBES, g05._ROUTES, "selector", ()),
            ("hy_vla", hy_vla.ROUTE_PROBES, hy_vla.ROUTES, "selector", ()),
            ("hy_vla", hy_policy._ROUTE_PROBES, hy_policy._ROUTES, "profile", ()),
            ("lingbot2", lingbot2.ROUTE_PROBES, lingbot2.ROUTES, "selector", ()),
            ("lingbot2", lingbot2_policy._ROUTE_PROBES, lingbot2_policy._ROUTES, "profile", ()),
            ("gr00t", gr00t.ROUTE_PROBES, gr00t.ROUTES, "profile", ()),
            *((target, audio.ROUTE_PROBES[target], audio.ROUTES[target], "selector", ())
              for target in ("qwen3_asr", "qwen3_tts"))]


def test_every_recipe_an_owner_admits_resolves_to_one_of_its_routes():
    families = RECIPES.load(COMMON.PRECISION_RECIPES)
    for target, probes, routes, entries, refused in _owner_routes():
        family = RECIPES.family_for(target, COMMON.PRECISION_RECIPES)
        admitted = [item for item in families[family.name].recipes.values() if entries == "selector" or item.profiles]
        keys = {item.name: RECIPES.route_key(item, family.components, probes) for item in admitted}
        assert {name for name, key in keys.items() if key not in routes} == set(refused), target
        # Every route is reachable from a registered recipe, so a table cannot keep a dead mode.
        assert set(keys.values()) >= set(routes), (target, set(routes) - set(keys.values()))
        for name in refused:
            with pytest.raises(COMMON.ConfigError, match=f"no route for precision recipe {family.name}/{name}"):
                COMMON.owner_route(target, target, probes, routes, selector=family.recipes[name].selector)


@pytest.mark.parametrize("target, selector, edit", [
    ("wall_oss_sorting", "w8a16_action_w4a16", lambda rules: rules[-1].update(spec={"format": "w4a16", "group": 128})),
    ("g05", "w8a16", lambda rules: rules.pop(2)),
    ("hy_vla", "w8a16_nvfp4", lambda rules: rules[0].update(spec="fp16")),
    ("qwen3_asr", "nvfp4", lambda rules: rules[1].update(spec="nvfp4")),
])
def test_a_recipe_whose_specs_drift_from_every_route_is_refused(monkeypatch, tmp_path, target, selector, edit):
    data = json.loads(COMMON.PRECISION_RECIPES.read_text(encoding="utf-8"))
    family = RECIPES.family_for(target, COMMON.PRECISION_RECIPES)
    name = RECIPES.by_selector(target, selector, COMMON.PRECISION_RECIPES).name
    edit(data["families"][family.name]["recipes"][name]["rules"])
    path = tmp_path / "precision_recipes.v1.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    owner = next((owner, probes, routes) for owner, probes, routes, *_ in _owner_routes() if owner == target)
    COMMON.owner_route(target, *owner, selector=selector)
    monkeypatch.setattr(COMMON, "PRECISION_RECIPES", path)
    with pytest.raises(COMMON.ConfigError, match=f"has no route for precision recipe {family.name}/{name}"):
        COMMON.owner_route(target, *owner, selector=selector)


@pytest.mark.parametrize("target, slot, selector, recorded, matches", [
    ("pi05", "checkpoint", "w8a16_prefill_w8a8_action_nvfp4", "pi05/w8a16_action_nvfp4", True),
    ("pi05", "checkpoint", "w8a16_prefill_w8a8_action_nvfp4", "pi05/w8a16", False),
    ("pi05", "checkpoint", "w8a16_vision_w4a16_action_nvfp4", "pi05/w8a16_action_nvfp4", True),
    ("wall_oss", "w8_checkpoint", "w8a16_action_nvfp4", "wall_oss/w8a16", True),
    ("wall_oss_rtc", "checkpoint", "w8a16", "wall_oss/w8a16", True),
    ("wall_oss_sorting", "action_w4_checkpoint", "w8a16_action_w4a16", "wall_oss/w4a16_g32", True),
    ("wall_oss_sorting", "action_w4_checkpoint", "w8a16_action_w4a16", "wall_oss/w4a16_channel", False),
])
def test_published_entries_accept_only_the_conversion_they_load(tmp_path, target, slot, selector, recorded, matches):
    RECIPES.write_record(tmp_path, *recorded.split("/"))
    if matches:
        COMMON.check_checkpoint_recipe(target, tmp_path, slot=slot, selector=selector)
    else:
        with pytest.raises(COMMON.ConfigError, match=f"was converted for precision recipe {recorded}"):
            COMMON.check_checkpoint_recipe(target, tmp_path, slot=slot, selector=selector)


def test_unrecorded_weights_keep_only_their_structural_checks(tmp_path):
    # No record: the target is not even resolved, so existing weights load as before.
    COMMON.check_checkpoint_recipe("no-such-runner", tmp_path)
    (tmp_path / RECIPES.RECORD).write_text("{")
    with pytest.raises(COMMON.ConfigError, match="not JSON"):
        COMMON.check_checkpoint_recipe("pi05", tmp_path, selector="w8a16")


def test_every_converter_records_a_registered_recipe_that_its_own_arguments_produce():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import importlib

    families = RECIPES.load()

    def module(name):
        return importlib.import_module(f"rpu_backend.quant.{name}")

    for name in ("convert_qwen3_5_awq", "convert_qwen3_5_moe_fp8", "convert_qwen3_5_moe_int8",
                 "convert_qwen3_5_moe_rtn", "convert_wall_oss_w8a16"):
        converter = module(name)
        assert converter.RECIPE in families[converter.FAMILY].recipes, name
    for name in ("convert_qwen3_5", "convert_pi05", "convert_wall_oss"):
        converter = module(name)
        for recipe, arguments in converter.RECIPES.items():
            assert recipe in families[converter.FAMILY].recipes, (name, recipe)
            assert converter.produced_recipe(**arguments) == recipe, (name, recipe)
    qwen3 = module("convert_qwen3")
    for recipe, (family, arguments) in qwen3.RECIPES.items():
        assert recipe in families[family].recipes and qwen3.produced_recipe(family, **arguments) == (family, recipe)
    vl = module("convert_qwen3_vl")
    assert set(vl.RECIPES) <= set(families[vl.FAMILY].recipes)
    assert {vl.produced_recipe(8), *(vl.produced_recipe(4, awq_source=awq) for awq in (False, True))} == set(vl.RECIPES)
    # Mixes no recipe registers get no record.
    assert module("convert_qwen3_5").produced_recipe(profile="w4_main_w8_gdn") is None
    assert module("convert_pi05").produced_recipe(bits=4) is None
    assert module("convert_wall_oss").produced_recipe() is None
    assert qwen3.produced_recipe("qwen3", ["lm_head"]) is None

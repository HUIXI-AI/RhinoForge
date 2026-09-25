"""Cold arithmetic and installed full-pipeline precision checks without a device."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python/rpu_backend"


@pytest.fixture
def cold(monkeypatch):
    for name in ("rpu_backend", "rpu_backend.api", "rpu_backend.adapters",
                 "rpu_backend.adapters.rhinovla", "rpu_backend.runtime"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    planner = ModuleType("rpu_backend.runtime.execution_planner")
    planner.GRAPH_COMPOSITE_CHILD = "COMPOSITE_CHILD"
    monkeypatch.setitem(sys.modules, planner.__name__, planner)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, PYTHON / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module
    tree = ast.parse((PYTHON / "runtime/control.py").read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name == "rpu_env_bool" or
             isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in
             {"_ENV_BOOL_TRUE", "_ENV_BOOL_FALSE"} for target in node.targets)]
    control = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "control.py", "exec"), control)
    sys.modules["rpu_backend.runtime"].rpu_env_bool = control["rpu_env_bool"]
    load("rpu_backend.api._execution", "api/_execution.py")
    runtime = load("rpu_backend.adapters.rhinovla.runtime", "adapters/rhinovla/runtime.py")
    for fields in runtime._PIPELINE_COLD_ENV.values():
        for env in fields.values():
            monkeypatch.delenv(env, raising=False)
    return runtime


def test_acc32_is_explicit_child_overridable_and_cold_bound(cold):
    snapshot, effective = cold.RhinoVLAPipelineColdConfig.resolve({
        "prefill": {"linear_acc32": True},
        "components": {"language_model": {"prefill": {"linear_acc32": False}},
                       "vision_encoder": {"vision": {"linear_acc32": True}}},
    })
    assert snapshot.stages["prefill"]["linear_acc32"] is False
    assert snapshot.stages["vision"]["linear_acc32"] is True
    snapshot.validate_bound(effective)
    snapshot.validate_bound({"vision": {"chunk_size": 64}})
    with pytest.raises(ValueError, match="cold-bound"):
        snapshot.validate_bound({"vision": {"linear_acc32": False}})
    with pytest.raises(ValueError, match="cold-bound"):
        snapshot.validate_bound({"prefill": {"linear_acc32": True}})
    with pytest.raises(ValueError, match="bind a pipeline cold snapshot"):
        cold._validate_pipeline_cold(NS(), {"prefill": {"linear_acc32": False}})
    default, _ = cold.RhinoVLAPipelineColdConfig.resolve({})
    assert default.stages["prefill"]["linear_acc32"] is False
    assert default.stages["vision"]["linear_acc32"] is False


@pytest.mark.parametrize("stage,value,error", [
    ("prefill", 1, TypeError), ("vision", "true", TypeError),
    ("action", True, ValueError),
])
def test_acc32_rejects_unsupported_or_non_boolean_fields(cold, stage, value, error):
    with pytest.raises(error):
        cold.RhinoVLAPipelineColdConfig.resolve({stage: {"linear_acc32": value}})


@pytest.fixture
def precision():
    path = PYTHON / "adapters/rhinovla/precision.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             or isinstance(node, ast.Assign)]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return NS(**scope)


@pytest.mark.parametrize("prefill,vision", [(False, False), (True, False), (False, True), (True, True)])
def test_arithmetic_binds_actual_native_owners(precision, monkeypatch, prefill, vision):
    calls = []
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_set_linear_acc32",
                        lambda *args: calls.append(("vision", *args)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_linear_acc32",
                        lambda *args: calls.append(("prefill", *args)), raising=False)
    text, visual = NS(_rpu_decoder_handle=7), NS(_rpu_vision_handle=9)
    precision.bind_pipeline_linear_accumulation(text, visual, prefill=prefill, vision=vision)
    assert calls == [("vision", 9, vision), ("prefill", 7, prefill)]
    assert visual._rpu_vision_linear_acc32 is vision
    with pytest.raises(TypeError):
        precision.bind_pipeline_linear_accumulation(text, visual, prefill=1, vision=vision)
    assert len(calls) == 2


class ResidentTensor(torch.Tensor):
    """Only device residency is mocked; shape, dtype and layout are real tensors."""
    @property
    def device(self):
        return NS(type="rpu", index=0)


def resident(shape, dtype):
    return torch.ones(shape, dtype=dtype).as_subclass(ResidentTensor)


@pytest.fixture
def installed():
    def weight():
        return resident((2, 2), torch.int8)
    def scale():
        return resident((2,), torch.float16)
    projections = {name: NS(weight=weight(), weight_scale=scale(), out_features=2)
                   for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    text = NS(layers=[NS(get_submodule=projections.__getitem__)], config=NS(num_hidden_layers=1))
    merger = NS(_rpu_merger_fc1_w_rpu=weight(), _rpu_merger_fc2_fused_w_rpu=weight(),
                _rpu_merger_fc1_scale_rpu=scale(), _rpu_merger_fc2_scale_rpu=scale())
    visual = NS(_rpu_vision_w8a16=True, _rpu_vision_fused_merger_w8a16=True,
                blocks=[object()], _rpu_vision_projection_weights=[[weight()] for _ in range(6)],
                _rpu_vision_projection_scales=[[scale()] for _ in range(6)],
                _rpu_vision_patch_embed_w_rpu=weight(), _rpu_vision_patch_embed_scale=scale(),
                merger=merger, deepstack_merger_list=[merger]*3)
    loop = {name: weight() for name in
            ("action_in_w", "state_w", "state_mask_w", "action_mask_w", "action_out_w")}
    loop["io_scales"] = [scale() for _ in range(6)]
    loop["adarms_cold_owners"] = ([weight() for _ in range(19)],
                                  [scale() for _ in range(19)], [scale() for _ in range(19)])
    return NS(text_model=text, vision_model=visual, _denoise_loop_meta=loop,
              er=NS(layers=[object()]*18, _rpu_full_w8a16=True))


def test_full_inventory_checks_all_component_storage(precision, installed):
    result = precision.full_w8_inventory(installed)
    assert {key: item["physical_matrices"] for key, item in result.items()} == {
        "text_prefix": 7, "vision_blocks": 6, "patch_projection": 1,
        "vision_mergers": 8, "action_io": 5, "adarms_cold": 19,
    }
    assert result["adarms_cold"]["logical_projections"] == 37


@pytest.mark.parametrize("failure", ["text_layer", "text_dtype", "vision_missing", "vision_dtype",
                                     "patch_cpu", "merger_scale", "io_scale", "expert_only",
                                     "final_bias"])
def test_full_inventory_rejects_partial_or_mislabeled_precision(precision, installed, failure):
    if failure == "text_layer":
        installed.text_model.config.num_hidden_layers = 2
    elif failure == "text_dtype":
        installed.text_model.layers[0].get_submodule("self_attn.q_proj").weight = resident((2, 2), torch.float16)
    elif failure == "vision_missing":
        installed.vision_model._rpu_vision_projection_weights[0].clear()
    elif failure == "vision_dtype":
        installed.vision_model._rpu_vision_projection_weights[0][0] = resident((2, 2), torch.float16)
    elif failure == "patch_cpu":
        installed.vision_model._rpu_vision_patch_embed_w_rpu = torch.ones(2, 2, dtype=torch.int8)
    elif failure == "merger_scale":
        installed.vision_model.merger._rpu_merger_fc1_scale_rpu = resident((1,), torch.float16)
    elif failure == "io_scale":
        installed._denoise_loop_meta["io_scales"].pop()
    elif failure == "expert_only":
        for group in installed._denoise_loop_meta["adarms_cold_owners"]:
            group.pop()
    elif failure == "final_bias":
        installed._denoise_loop_meta["adarms_cold_owners"][2][-1] = resident((2,), torch.float32)
    with pytest.raises(ValueError):
        precision.full_w8_inventory(installed)

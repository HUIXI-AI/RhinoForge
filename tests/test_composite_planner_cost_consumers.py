"""CPU spies for real planner call sites; no device or cost/performance claims."""
import ast
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "pi05/runtime.py", "pi05/adarms.py", "qwenpi05/fused.py", "gr00t/runtime.py",
    "lingbot_vla/runtime.py", "lingbot_vla_v2/runtime.py", "hy_vla/runtime.py",
    "internvla_n1/execution.py", "internvla_n1/backbone.py",
    "internvla_n1/nextdit.py", "internvla_n1/rgbd_rpu.py", "navdp/runtime.py",
)


@pytest.mark.parametrize("relative", FILES)
def test_every_production_shared_planner_binds_owner_component_and_native(relative):
    path = ROOT / "python/rpu_backend/adapters" / relative
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id in {
                 "plan_bounded_prefill_execution", "plan_native_component_execution"}]
    assert calls
    for call in calls:
        keys = {item.arg: item.value for item in call.keywords}
        for key in ("execution_owner", "execution_component", "execution_native"):
            assert key in keys, (relative, call.lineno, key)
            assert not (isinstance(keys[key], ast.Constant) and keys[key].value is None)
        if call.func.id == "plan_bounded_prefill_execution":
            assert "execution_stage" in keys


def _cases():
    common = dict(logical_len=31, execution_len=31, kv_len=111,
                  use_attention_mask=True, is_causal=False)
    owner = NS(_rpu_legacy_kvinsert_pad16_request=False)
    yield "pi05.runtime", "_plan_pi05_denoise_action_execution", owner, (), dict(
        handle=77, logical_len=32, position=80, kv_len=112, cache_capacity=128,
        loop_mode=False, num_steps=1), owner, "action_expert", "action", "pi05_denoise_step"
    owner = NS(_rpu_action_handle=77)
    yield "pi05.adarms", "_plan_adarms_action_execution", owner, (), dict(
        common, position=80), owner, "action_expert", "action", "adarms"
    owner = NS(_rpu_qwenpi05_handle=77,
               _fmb_execution_generation=0,
               _fmb_execution_component_config={"action": {"chunk_size": "auto"}})
    yield "qwenpi05.fused", "_plan_qwenpi05_action_execution", owner, (), dict(
        common, prefix_len=80), owner, "action_expert", "action", "qwenpi05"
    owner = NS(_text_execution={}, _vl_execution={}, _action_execution={},
               _text=NS(_rpu_decoder_handle=77, _rpu_text_graph_cache=object()),
               _a_handle=77, _d_handle=77, _a_gc=object(), _d_gc=object(),
               _bb_cache=NS(max_seq_len=512), _component_generations={
                   "text_backbone": 0, "vl_encoder": 0, "action_dit": 0})
    for method, component, stage, prefix, actual in (
        ("_prefill_execution_plan", "text_backbone", "prefill", "causal_decoder", owner._text),
        ("_vl_execution_plan", "vl_encoder", "prefill", "gr00t_vl_encoder", owner),
        ("_action_execution_plan", "action_dit", "action", "gr00t_dit", owner),
    ):
        yield "gr00t.runtime", "Gr00tN1d7VLA." + method, owner, (80,), {}, actual, component, stage, prefix
    owner = NS(_prefill_chunk_size=0, _max_seq=512, _vlm=77, _exp=77,
               _du_handle=77, _denoise_unroll=False, _vlm_replay=True,
               _expert_replay=True, _vgc=object(), _egc=object(), _dgc=object())
    yield "lingbot_vla.runtime", "LingbotVlaPolicy._plan_prefill_execution", owner, (80,), {}, owner, "language_model", "prefill", "causal_decoder"
    for unroll in (False, True):
        current = NS(**(vars(owner) | {"_denoise_unroll": unroll}))
        yield "lingbot_vla.runtime", "LingbotVlaPolicy._plan_action_execution", current, (80,), {}, current, "action_expert", "action", "lingbot_denoise" if unroll else "causal_decoder"
    owner = NS(_multiview_spm_z2=False, _prepared_graph_profile=None,
               _text_execution={}, _action_execution={}, _chunk_size=0,
               _max_seq=8192, _text=NS(_rpu_decoder_handle=77), _exp=77,
               _vlm_replay=True, _expert_replay=True, _denoise_unroll=False,
               _vgc=object(), _egc=object(), _dgc=object(),
               _component_generations={
                   "vision_encoder": 0, "language_model": 0, "action_expert": 0})
    yield "lingbot_vla_v2.runtime", "LingbotVlaV2Policy._prefix_execution_plan", owner, (80,), {}, owner._text, "language_model", "prefill", "causal_decoder"
    yield "lingbot_vla_v2.runtime", "LingbotVlaV2Policy._action_execution_plan", owner, (80, 0), {}, owner, "action_expert", "action", "lingbot_v2_moe"
    owner = NS(_component_config=lambda *_: {"chunk_size": "auto"},
               _component_generation=lambda *_: 0)
    for component, stage, prefix in (
        ("vision_encoder", "vision", "hyvit2"),
        ("language_model", "prefill", "hyvla_vlm"),
        ("action_expert", "action", "hyvla_expert"),
    ):
        yield "hy_vla.runtime", "HyVlaRunner._plan_execution", owner, (), dict(
            component=component, stage=stage, logical_len=32, position=0,
            sub=NS(gc=object(), handle=77), resolve_stage_domain=lambda _: (),
            envelope={}, spans=(), kv_route="DDR_REQUIRED", route_reason="test",
        ), owner, component, stage, prefix
    owner = NS(_rpu_execution={}, _handle=77, _fmb_execution_generation=0,
               _graph_cache=object())
    yield "internvla_n1.nextdit", "NextDiTRPURuntime._execution_plan", owner, (), dict(
        action_rows=32, cross_seq=80, use_action_encoder=False,
        populate_cross_cache=False, action_batch=1,
    ), owner, "nextdit_action", "action", "internvla_nextdit"


class _ReachedPlanner(BaseException):
    pass


@pytest.mark.parametrize(
    "module_name,method_name,owner,args,kwargs,actual,component,stage,prefix", list(_cases())
)
def test_real_producer_passes_exact_cost_consumer(
    monkeypatch, module_name, method_name, owner, args, kwargs, actual, component, stage, prefix,
):
    module = importlib.import_module("rpu_backend.adapters." + module_name)
    method = module
    for name in method_name.split("."):
        method = getattr(method, name)

    def spy(*_args, **options):
        assert options["execution_owner"] is actual
        assert options["execution_component"] == component
        assert options["execution_stage"] == stage
        assert options["execution_native"] == (prefix, 77)
        raise _ReachedPlanner

    monkeypatch.setattr(module, "plan_bounded_prefill_execution", spy)
    # Each producer reads its captured graph owner before invoking the planner.
    for attr in ("_vlm_replay", "_expert_replay", "_use_graph", "_denoise_unroll"):
        if not hasattr(owner, attr):
            setattr(owner, attr, False)
    for attr in ("_graph_cache", "_vl_graph", "_action_graph", "_vgc", "_egc", "_dgc", "_agc", "_a_gc", "_d_gc", "_vl_gc", "_dit_gc"):
        if not hasattr(owner, attr):
            setattr(owner, attr, None)
    if hasattr(owner, "_text") and not hasattr(owner._text, "_rpu_text_graph_cache"):
        owner._text._rpu_text_graph_cache = None
    with pytest.raises(_ReachedPlanner):
        method(owner, *args, **kwargs)

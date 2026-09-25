"""Exercise optional persistent prefix construction across request contexts."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch


def _prepare():
    path = Path(__file__).parents[1] / "python/rpu_backend/adapters/rhinovla/pipeline.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "RhinoVLAOnRPU")
    node = next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "_prepare_prefix_inputs")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["_prepare_prefix_inputs"]


class Resident(torch.Tensor):
    @property
    def device(self):
        return NS(type="rpu")


def test_optional_prefix_slots_survive_inference_to_no_grad(monkeypatch):
    real_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return real_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    ids = torch.tensor([[1, 2, 3, 4]])
    base = torch.arange(12, dtype=torch.float16).reshape(1, 4, 3)
    cache = {
        "semantic_inputs": {"input_ids": ids.clone()},
        "inputs_embeds_base_rpu": base,
        "deepstack_zero_rpu": torch.zeros(4, 3, dtype=torch.float16),
        "visual_runs": [(1, 2)],
        "position_ids_rpu": torch.arange(4),
        "attention_mask_rpu": torch.ones(1, 4),
        "prefix_len": 4, "prefix_mask": torch.ones(1, 4),
        "prefix_rope_deltas": torch.zeros(1),
        "visual_pos_mask": torch.tensor([False, True, True, False]),
    }
    owner = NS(_prefix_static_cache=cache, qin={"input_ids": ids},
               _prefix_prep_rpu_enabled=True, _prefix_merger_scatter=False,
               _prefix_no_clone=True)
    visual = NS(pooler_output=torch.full((2, 3), 20.0).as_subclass(Resident),
                deepstack_features=[torch.full((2, 3), 30.0)])
    prepare = _prepare()
    with torch.inference_mode():
        first = prepare(owner, visual)
    prefix, dense = first["inputs_embeds"], first["deepstack_dense"][0]
    assert not prefix.is_inference() and not dense.is_inference()
    saved = prefix.clone()
    visual.pooler_output.add_(10)
    visual.deepstack_features[0].add_(10)
    with torch.no_grad():
        second = prepare(owner, visual)
    assert second["inputs_embeds"] is prefix
    assert second["deepstack_dense"][0] is dense
    assert torch.equal(prefix[0, 1:3], torch.full((2, 3), 30, dtype=torch.float16))
    assert torch.equal(dense[1:3], torch.full((2, 3), 40, dtype=torch.float16))
    assert torch.equal(prefix[:, [0, 3]], base[:, [0, 3]])
    assert torch.count_nonzero(dense[[0, 3]]) == 0
    assert torch.equal(saved[0, 1:3], torch.full((2, 3), 20, dtype=torch.float16))

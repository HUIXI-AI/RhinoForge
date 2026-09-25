"""Run the actual backbone input glue with CPU device doubles, stopping at planning."""
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.gr00t import runtime


def test_backbone_memo_survives_request_context_and_input_mutation(monkeypatch):
    real_to, real_zeros = torch.Tensor.to, torch.zeros

    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return real_to(tensor, *args, **kwargs)

    def cpu_zeros(*args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    def rope(*args, **kwargs):
        return torch.arange(4).reshape(1, 1, 4).expand(3, 1, 4), None

    # Public delivery normalizes the upstream helper's signature; backend main
    # still calls the model method directly. Neither route is under test here.
    if hasattr(runtime, "_get_gr00t_rope_index"):
        monkeypatch.setattr(runtime, "_get_gr00t_rope_index", rope)
    embedding = torch.nn.Embedding(16, 3)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(48).reshape(16, 3))
    visual = NS(pooler_output=torch.full((2, 3), 20.0),
                deepstack_features=[torch.full((2, 3), 30.0)])

    class Vision:
        _rpu_last_execution_plan = {'component': runtime._VISION_COMPONENT,
                                   'generation': 0, 'dry_forward_agreement': True}
        def __call__(self, *args, **kwargs):
            return visual

    class AtPlanning(Exception):
        pass

    def stop_before_native(*args):
        raise AtPlanning

    owner = NS(_qwen_z1_plan_hash=None,
               _bb=NS(config=NS(image_token_id=15), model=NS(get_rope_index=rope),
                      get_input_embeddings=lambda: embedding),
               _cfg=NS(text_config=NS(hidden_size=3)), _text=NS(), _vision=Vision(),
               _component_generations={runtime._VISION_COMPONENT: 0},
               _rpu_last_execution_plan={}, _emb_text_rpu=None, _emb_key=None,
               _rope_ids=None, _partial_mrope=False,
               _prefill_execution_plan=stop_before_native)
    inputs = {'input_ids': torch.tensor([[1, 15, 15, 2]]),
              'attention_mask': torch.ones(1, 4), 'pixel_values': torch.ones(1),
              'image_grid_thw': torch.tensor([[1, 1, 2]])}
    with torch.inference_mode(), pytest.raises(AtPlanning):
        runtime.Gr00tN1d7VLA._backbone(owner, inputs)
    first, deep = owner._emb_text_rpu, owner._ds_dense[0]
    visual.pooler_output.add_(10); visual.deepstack_features[0].add_(10)
    with torch.no_grad(), pytest.raises(AtPlanning):
        runtime.Gr00tN1d7VLA._backbone(owner, inputs)
    assert owner._emb_text_rpu is first and owner._ds_dense[0] is deep
    assert not first.is_inference() and not deep.is_inference()
    assert torch.equal(first[0, 1:3], visual.pooler_output.half())
    assert torch.equal(deep[1:3], visual.deepstack_features[0].half())
    assert torch.count_nonzero(deep[[0, 3]]) == 0
    inputs['input_ids'][0, 0] = 3
    with torch.no_grad(), pytest.raises(AtPlanning):
        runtime.Gr00tN1d7VLA._backbone(owner, inputs)
    assert owner._emb_text_rpu is not first
    assert torch.equal(owner._emb_text_rpu[0, 0], embedding.weight[3].half())
    assert torch.equal(first[0, 0], embedding.weight[1].half())
    assert torch.equal(owner._emb_text_rpu[0, 1:3], visual.pooler_output.half())

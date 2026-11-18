from __future__ import annotations
import ast
from pathlib import Path
import pytest
import torch
from rpu_backend.adapters.wall_oss import llm

def test_wall_oss_llm_has_no_optimized_away_assertions():
    tree = ast.parse(Path(llm.__file__).read_text(encoding='utf-8'))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]

@pytest.mark.parametrize('nkv', [0, -1, True, 2.5])
def test_attention_parallelism_rejects_invalid_kv_head_count(monkeypatch, nkv):
    monkeypatch.setenv('RPU_WALL_OSS_ATTN_TP8', '1')
    with pytest.raises(ValueError, match='positive integer'):
        llm._eff_attn_tp(nkv)

def test_attention_parallelism_rejects_non_divisible_tp8(monkeypatch):
    monkeypatch.setenv('RPU_WALL_OSS_ATTN_TP8', '1')
    with pytest.raises(ValueError, match='8 % nkv\\(3\\)'):
        llm._eff_attn_tp(3)
    assert llm._eff_attn_tp(2) == 8

def test_llm_forwards_reject_bad_inputs_before_rpu_access():
    model = llm.WallOssLLM.__new__(llm.WallOssLLM)
    with pytest.raises(TypeError, match='input_ids must be a torch.Tensor'):
        model.forward([])
    with pytest.raises(ValueError, match='\\[1, seq\\]'):
        model.forward(torch.zeros(2, 3, dtype=torch.int64))
    with pytest.raises(TypeError, match='inputs_embeds must be a torch.Tensor'):
        model.forward_embeds([], torch.zeros(2, 3))
    with pytest.raises(ValueError, match='\\[1, seq, H\\]'):
        model.forward_embeds(torch.zeros(2, 3), torch.zeros(3, 3))

def test_prefill_rejects_bad_position_shape_before_planning_or_rpu_access():
    model = llm.WallOssLLM.__new__(llm.WallOssLLM)
    with pytest.raises(ValueError, match='\\[seq, 3\\]'):
        model._run_prefill(torch.zeros(1, 4, 8), torch.zeros(4, 2), 'test')

def test_direct_forward_pads_once_then_restores_logical_output_and_cache():

    class Cache:
        position = 0

        def reset_to_position(self, position):
            self.position = position
    model = llm.WallOssLLM.__new__(llm.WallOssLLM)
    model._embed_w = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    model.cache = Cache()
    model.prefill_execution_plan = lambda logical_len, *, reserve_rows=0: (4, 4)
    seen = {}

    def run_prefill(inputs_embeds, position_ids, op_id, *, logical_len=None, reserve_rows=0, fixed_bucket=False):
        seen.update(inputs_embeds=inputs_embeds, position_ids=position_ids, op_id=op_id, logical_len=logical_len, reserve_rows=reserve_rows, fixed_bucket=fixed_bucket)
        model.cache.position = inputs_embeds.size(1)
        return torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    model._run_prefill = run_prefill
    output = model.forward(torch.tensor([[1, 2, 3]]), reserve_rows=2)
    assert output.shape == (1, 3, 4)
    assert model.cache.position == 3
    assert seen['inputs_embeds'].shape == (1, 4, 4)
    assert torch.equal(seen['inputs_embeds'][:, 3], torch.zeros(1, 4))
    assert seen['position_ids'].tolist() == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 2]]
    assert seen['logical_len'] == 3
    assert seen['reserve_rows'] == 2
    assert seen['fixed_bucket'] is False
    # The VLA owns its padded embeddings and needs the physical cache tail for
    # its 2D action mask, so forward_embeds deliberately does not rewind it.
    physical = model.forward_embeds(torch.zeros(1, 4, 4), torch.zeros(4, 3, dtype=torch.int32), logical_len=3)
    assert physical.shape == (1, 4, 4)
    assert model.cache.position == 4

@pytest.mark.parametrize('flags,match', [({'w8a16': 'false'}, 'must be bool'), ({'w8a16': True, 'w4a16': True}, 'pick one')])
def test_llm_builder_rejects_ambiguous_precision_before_checkpoint_access(flags, match):
    with pytest.raises(ValueError, match=match):
        llm.build_wall_oss_llm(**flags)

from __future__ import annotations
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.adapters.wall_oss import action

def test_wall_oss_action_has_no_optimized_away_assertions():
    tree = ast.parse(Path(action.__file__).read_text(encoding='utf-8'))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]

@pytest.mark.parametrize('noise,match', [([], 'torch.Tensor'), (torch.zeros(2, 3), '\\[1, H, action_dim\\]'), (torch.zeros(2, 4, 3), '\\[1, H, action_dim\\]'), (torch.zeros(1, 4, 2), 'action_dim=3')])
def test_action_noise_shape_contract_is_explicit(noise, match):
    with pytest.raises((TypeError, ValueError), match=match):
        action._action_horizon(noise, 3)

@pytest.mark.parametrize('position_ids,match', [([], 'torch.Tensor'), (torch.zeros(4, 2), '\\[H, 3\\]'), (torch.zeros(3, 3), '\\[H, 3\\]')])
def test_action_position_shape_contract_is_explicit(position_ids, match):
    with pytest.raises((TypeError, ValueError), match=match):
        action._validate_action_positions(position_ids, 4)

def test_predict_rejects_bad_inputs_before_prefill_or_rpu_access():
    model = action.WallOssAction.__new__(action.WallOssAction)
    model._fused = False
    model.action_dim = 3
    with pytest.raises(TypeError, match='input_ids must be a torch.Tensor'):
        model.predict([], torch.zeros(1, 4, 3), torch.ones(1, 4, 3))
    calls = []
    model.llm = SimpleNamespace(forward=lambda value, *, reserve_rows=0: calls.append((value, reserve_rows)), cache=SimpleNamespace(position=3, max_seq_len=64))
    with pytest.raises(RuntimeError, match='cache position'):
        model.predict(torch.zeros(1, 4, dtype=torch.int64), torch.zeros(1, 2, 3), torch.ones(1, 2, 3))
    assert len(calls) == 1
    assert calls[0][1] == 2

def test_predict_rejects_capacity_before_prefill():
    model = action.WallOssAction.__new__(action.WallOssAction)
    model._fused = False
    model._execution_chunk_size = 'auto'
    model.action_dim = 3
    calls = []
    model.llm = SimpleNamespace(forward=lambda value, *, reserve_rows=0: calls.append((value, reserve_rows)), cache=SimpleNamespace(position=0, max_seq_len=8))
    with pytest.raises(RuntimeError, match='required=10.*max_seq_len=8'):
        model.predict(torch.zeros(1, 6, dtype=torch.int64), torch.zeros(1, 4, 3), torch.ones(1, 4, 3))
    assert calls == []

@pytest.mark.parametrize('flags,match', [({'w8a16': 'false'}, 'must be bool'), ({'w8a16': True, 'w4a16': True}, 'pick one')])
def test_action_builder_rejects_ambiguous_precision_before_checkpoint_access(flags, match):
    with pytest.raises(ValueError, match=match):
        action.build_wall_oss_action(**flags)

def test_exact_action_chunk_rejects_horizon_before_prefill():
    model = action.WallOssAction.__new__(action.WallOssAction)
    model._fused = False
    model._execution_chunk_size = 32
    model.action_dim = 3
    calls = []
    model.llm = SimpleNamespace(forward=lambda value, *, reserve_rows=0: calls.append((value, reserve_rows)), cache=SimpleNamespace(position=0, max_seq_len=64))
    with pytest.raises(ValueError, match='requested=32, horizon=16.*required=16'):
        model.predict(torch.zeros(1, 4, dtype=torch.int64), torch.zeros(1, 16, 3), torch.ones(1, 16, 3))
    assert calls == []
    model.validate_execution_horizon(32)
    model._execution_chunk_size = 16
    model.validate_execution_horizon(10)
    model._execution_chunk_size = 'auto'
    model.validate_execution_horizon(10)

@pytest.mark.parametrize('field', ['padding_rows', 'padding_budget'])
def test_action_builder_rejects_padding_before_checkpoint_access(field):
    with pytest.raises(ValueError, match='not supported'):
        action.build_wall_oss_action(execution_config={'action': {field: 0}})

def test_action_builder_rejects_shared_cache_depth_before_weights(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps({'num_hidden_layers': 2}), encoding='utf-8')
    monkeypatch.setattr(action, '_resolve_ckpt', lambda path, quantized: str(tmp_path))
    with pytest.raises(ValueError, match='shared-cache.*num_layers=1.*2'):
        action.build_wall_oss_action(ckpt_dir=str(tmp_path), llm=SimpleNamespace(num_layers=1))

def test_fused_action_keeps_exact_native_chunk_override():
    root = Path(__file__).resolve().parents[4]
    source = (root / 'src/fused/rpu_wall_oss_action_step_model.cpp').read_text(encoding='utf-8')
    assert 'configured_chunk_size_' in source
    assert 'ceil16(horizon)' in source
    assert 'set_chunk_envelope(/*max_kv_len=*/cos.size(0), /*chunk=*/chunk_size)' in source
    assert source.count('set_chunk_size_override(configured_chunk_size_)') >= 3
    assert 'set_chunk_size_override(0);' not in source

def test_kv_owner_has_distinct_graph_identity():
    model = action.WallOssAction.__new__(action.WallOssAction)
    model.hidden_size = 1024
    model.num_layers = 36
    model._precision_mode = 1
    model._kv_owner_fanout = False
    assert model._denoise_graph_signature('action', 20, 640, 6).op_id == 'action'
    model._kv_owner_fanout = True
    assert model._denoise_graph_signature('action', 20, 640, 6).op_id == 'action_kv_owner'

@pytest.mark.parametrize('feature', ['WEIGHT_RESIDENCY', 'KV_OWNER_FANOUT'])
@pytest.mark.parametrize('mask,fused,precision,admitted', [('0', True, {}, False), ('3', True, {}, False), ('4', True, {}, True), ('7', False, {}, False), ('7', True, {'w4a16': True}, False), ('7', True, {'nvfp4a16': True}, False), ('0', True, {'w8a16': True}, True)])
def test_action_fp16_resident_routes_require_explicit_precision_opt_in(monkeypatch, feature, mask, fused, precision, admitted):
    monkeypatch.setenv('RPU_WALL_OSS_FP16_TRANSFER_MASK', mask)
    for name in ('WEIGHT_RESIDENCY', 'KV_OWNER_FANOUT'):
        monkeypatch.setenv('RPU_WALL_OSS_ACTION_' + name, str(int(name == feature)))

    def checkpoint_access(*_):
        raise RuntimeError('admitted before checkpoint access')
    monkeypatch.setattr(action, '_resolve_ckpt', checkpoint_access)
    error = RuntimeError if admitted else ValueError
    match = 'admitted before checkpoint' if admitted else 'requires fused=True'
    with pytest.raises(error, match=match):
        action.build_wall_oss_action('/must-not-be-read', fused=fused, **precision)

from __future__ import annotations
import ast
import hashlib
import importlib
import inspect
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import types
import pytest
import torch
from rpu_backend.adapters.internvla_n1._policy import NUMERIC_BLOCKED_ENV, S2_SDPA_BF16_ENV, require_controlled_environment, verify_asset_hashes, verify_asset_manifest
from rpu_backend.adapters.internvla_n1.backbone import build_qwen25vl_backbone
from rpu_backend.adapters.internvla_n1.nextdit import build_nextdit_runtime
from rpu_backend.adapters.internvla_n1.goal_glue import build_goal_glue
from rpu_backend.adapters.internvla_n1.vision import build_internvla_vision
from rpu_backend.adapters.internvla_n1.rgbd_rpu import FormerRPU, RGBDTowerRPU, _verify_dinov2_source_tree, build_rgbd_encoder_rpu
from rpu_backend.adapters.navdp.runtime import build_navdp
from rpu_backend.runtime import UnsupportedModelError

def test_former_rpu_is_registered_module():
    assert issubclass(FormerRPU, torch.nn.Module)
    parent = torch.nn.Module()
    parent.former_net = torch.nn.Identity()
    replacement = FormerRPU.__new__(FormerRPU)
    torch.nn.Module.__init__(replacement)
    parent.former_net = replacement
    assert parent._modules['former_net'] is replacement

@pytest.fixture(autouse=True)
def _controlled_environment_is_clean(monkeypatch):
    monkeypatch.delenv(NUMERIC_BLOCKED_ENV, raising=False)
    monkeypatch.delenv(S2_SDPA_BF16_ENV, raising=False)

@pytest.mark.parametrize('value', [None, '', '0', 'true', 'True', '01', ' 1', '1 '])
def test_every_rpu_entry_requires_exact_opt_in_before_io_or_handle(monkeypatch, value):
    monkeypatch.delenv(NUMERIC_BLOCKED_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(NUMERIC_BLOCKED_ENV, value)
    before = os.environ.copy()
    entries = (lambda : build_qwen25vl_backbone('/must-not-be-read'), lambda : build_internvla_vision('/must-not-be-read'), lambda : build_navdp('/must-not-be-read'), lambda : build_rgbd_encoder_rpu('/must-not-be-read', '/must-not-import'), lambda : FormerRPU('/must-not-be-read'), lambda : RGBDTowerRPU(SimpleNamespace()), lambda : build_nextdit_runtime(SimpleNamespace()))
    for entry in entries:
        with pytest.raises(UnsupportedModelError, match='numeric-blocked'):
            entry()
    assert os.environ == before

def test_import_does_not_mutate_environment(monkeypatch):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    before = os.environ.copy()
    module = importlib.import_module('rpu_backend.adapters.internvla_n1')
    importlib.reload(module)
    assert os.environ == before

def test_numeric_blocked_opt_in_is_literal_and_side_effect_free(monkeypatch):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    before = os.environ.copy()
    require_controlled_environment()
    assert os.environ == before

@pytest.mark.parametrize('value', ['', 'true', '01', ' 1', '1 '])
def test_numeric_blocked_rejects_nonliteral_opt_in(monkeypatch, value):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, value)
    with pytest.raises(UnsupportedModelError, match=NUMERIC_BLOCKED_ENV):
        require_controlled_environment()

@pytest.mark.parametrize('value', ['1', 'true', 'on', '01', ' 1'])
def test_retired_s2_bf16_knob_fails_before_asset_io(monkeypatch, value):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    monkeypatch.setenv(S2_SDPA_BF16_ENV, value)
    pattern = 'current latest' if value == '1' else "literal '0' or '1'"
    with pytest.raises(UnsupportedModelError, match=pattern):
        build_navdp('/must-not-be-read', asset_manifest={'unused': '0' * 64})

def test_manifest_is_required_and_hash_checked(monkeypatch, tmp_path):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    asset = tmp_path / 'checkpoint.safetensors'
    asset.write_bytes(b'known checkpoint bytes')
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    with pytest.raises(UnsupportedModelError, match='asset_manifest'):
        verify_asset_manifest((asset,), None)
    with pytest.raises(UnsupportedModelError, match='malformed'):
        verify_asset_manifest((asset,), {asset: 'not-a-sha256'})
    with pytest.raises(UnsupportedModelError, match='mismatch'):
        verify_asset_manifest((asset,), {asset: '0' * 64})
    verify_asset_manifest((asset,), {asset: digest})

def test_cpu_asset_hash_verification_does_not_require_rpu_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv(NUMERIC_BLOCKED_ENV, raising=False)
    asset = tmp_path / 'cpu-golden-input.pt'
    asset.write_bytes(b'exact cpu asset')
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    verify_asset_hashes((asset,), {asset: digest})

@pytest.mark.parametrize('name', ['RPU_INTERNVLA_NEXTDIT_FOLD_NORM_MOD', 'RPU_INTERNVLA_NEXTDIT_STEP_PREP_RPU', 'RPU_INTERNVLA_NEXTDIT_FUSED_QKQ'])
def test_nextdit_env_is_parsed_before_geometry_or_rpu_materialization(monkeypatch, tmp_path, name):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    monkeypatch.setenv(name, 'invalid')
    asset = tmp_path / 'asset.bin'
    asset.write_bytes(b'verified')
    manifest = {asset: hashlib.sha256(asset.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match=name):
        build_nextdit_runtime(SimpleNamespace(), asset_manifest=manifest, asset_paths=(asset,))

@pytest.mark.parametrize('name', ['RPU_INTERNVLA_DINO_FINAL_NORM_RPU', 'RPU_INTERNVLA_DINO_NORM_FOLD'])
def test_rgbd_env_is_parsed_before_geometry_or_handle(monkeypatch, tmp_path, name):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    monkeypatch.setenv(name, 'invalid')
    asset = tmp_path / 'asset.bin'
    asset.write_bytes(b'verified')
    manifest = {asset: hashlib.sha256(asset.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match=name):
        RGBDTowerRPU(SimpleNamespace(), asset_manifest=manifest, asset_paths=(asset,))

def test_backbone_host_cache_env_is_parsed_before_checkpoint_io(monkeypatch):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    monkeypatch.setenv('RPU_INTERNVLA_HOST_CACHE', 'invalid')
    with pytest.raises(ValueError, match='RPU_INTERNVLA_HOST_CACHE'):
        build_qwen25vl_backbone('/must-not-be-read', asset_manifest={'unused': '0' * 64})

def test_backbone_preserves_int8_weights_before_w8_swizzle(monkeypatch, tmp_path):
    from rpu_backend.adapters.internvla_n1 import backbone
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'hidden_size': 3584, 'num_attention_heads': 28, 'num_key_value_heads': 4, 'head_dim': 128, 'intermediate_size': 18944, 'num_hidden_layers': 28, 'rms_norm_eps': 1e-06, 'rope_theta': 1000000.0, 'rope_scaling': {'mrope_section': [16, 24, 24]}}))
    checkpoint = tmp_path / 'model.safetensors'
    checkpoint.write_bytes(b'manifest-bound fake store')
    manifest = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (config, checkpoint)}

    class FakeStore:

        def __init__(self, _root, mmap=False):
            self.weight_map = {'first': checkpoint.name}

        def get_tensor(self, _key):
            return torch.zeros((1, 1), dtype=torch.int8)

    class StopAfterFirstWeight(Exception):
        pass
    seen = []

    def capture_dtype(weight, _cores, dwidth=2):
        seen.append((weight.dtype, dwidth))
        raise StopAfterFirstWeight
    monkeypatch.setattr(backbone, '_SafeTensorStore', FakeStore)
    monkeypatch.setattr(backbone, '_validate_checkpoint_profile', lambda _store: None)
    monkeypatch.setattr(backbone, 'tp_col_swizzle_mc_weight', capture_dtype)
    with pytest.raises(StopAfterFirstWeight):
        backbone.build_qwen25vl_backbone(tmp_path, asset_manifest=manifest)
    assert seen == [(torch.int8, 1)]

def test_rgbd_dynamic_source_tree_is_manifest_bound_before_import(monkeypatch, tmp_path):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    source_root = tmp_path / 'rgbd_src'
    layers = source_root / 'dinov2_layers'
    layers.mkdir(parents=True)
    sources = (source_root / 'dinov2.py', layers / '__init__.py', layers / 'attention.py')
    for (index, path) in enumerate(sources):
        path.write_text(f'SOURCE_MARKER = {index}\n')
    manifest = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    assert _verify_dinov2_source_tree(source_root, manifest) == source_root.resolve()
    with pytest.raises(UnsupportedModelError, match='incomplete or malformed'):
        _verify_dinov2_source_tree(source_root, {path: digest for (path, digest) in manifest.items() if path != sources[-1]})
    escaped = tmp_path / 'escaped.py'
    escaped.write_text('UNVERIFIED = True\n')
    (layers / 'escape.py').symlink_to(escaped)
    with pytest.raises(UnsupportedModelError, match='must not contain symlinks'):
        _verify_dinov2_source_tree(source_root, manifest)

def test_rgbd_dynamic_source_tree_rejects_unverified_import_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    source_root = tmp_path / 'rgbd_src_artifacts'
    layers = source_root / 'dinov2_layers'
    layers.mkdir(parents=True)
    sources = (source_root / 'dinov2.py', layers / '__init__.py')
    for path in sources:
        path.write_text('SOURCE_MARKER = True\n')
    manifest = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    bytecode = layers / '__pycache__' / 'attention.cpython-312.pyc'
    bytecode.parent.mkdir()
    bytecode.write_bytes(b'unverified bytecode')
    with pytest.raises(UnsupportedModelError, match='unverified import artifacts'):
        _verify_dinov2_source_tree(source_root, manifest)

@pytest.mark.parametrize('same_path', [False, True])
def test_rgbd_encoder_rejects_preloaded_same_name_module(monkeypatch, tmp_path, same_path):
    from rpu_backend.adapters.internvla_n1.rgbd import RGBDEncoder
    source_root = tmp_path / 'rgbd_src_preloaded'
    layers = source_root / 'dinov2_layers'
    layers.mkdir(parents=True)
    (source_root / 'dinov2.py').write_text('class DINOv2: pass\n')
    (layers / '__init__.py').write_text('')
    module_name = f'{source_root.name}.dinov2'
    attacker = types.ModuleType(module_name)
    attacker.__file__ = str(source_root / 'dinov2.py' if same_path else tmp_path / 'attacker.py')
    monkeypatch.setitem(sys.modules, module_name, attacker)
    with pytest.raises(ImportError, match='already loaded.*fresh process'):
        RGBDEncoder(source_root)

def test_rgbd_dynamic_import_does_not_create_unverified_bytecode(monkeypatch, tmp_path):
    from rpu_backend.adapters.internvla_n1.rgbd import _load_dinov2
    source_root = tmp_path / 'rgbd_src_no_bytecode'
    layers = source_root / 'dinov2_layers'
    layers.mkdir(parents=True)
    (source_root / 'dinov2.py').write_text('class DINOv2: pass\n')
    (layers / '__init__.py').write_text('')
    before = sys.dont_write_bytecode
    assert _load_dinov2(source_root).__name__ == 'DINOv2'
    assert sys.dont_write_bytecode is before
    assert not tuple(source_root.rglob('*.pyc'))

def test_exact_opt_in_still_requires_manifest_before_every_rpu_entry(monkeypatch):
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    entries = (lambda : build_qwen25vl_backbone('/must-not-be-read'), lambda : build_internvla_vision('/must-not-be-read'), lambda : build_navdp('/must-not-be-read'), lambda : build_rgbd_encoder_rpu('/must-not-be-read', '/must-not-import'), lambda : FormerRPU('/must-not-be-read'), lambda : RGBDTowerRPU(SimpleNamespace()), lambda : build_nextdit_runtime(SimpleNamespace()))
    for entry in entries:
        with pytest.raises(UnsupportedModelError, match='asset_manifest'):
            entry()

def test_unknown_and_partial_rpu_profiles_fail_before_swizzle_or_handle(monkeypatch, tmp_path):
    import safetensors.torch
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    checkpoint = tmp_path / 'model.safetensors'
    safetensors.torch.save_file({'partial': torch.ones(1)}, str(checkpoint))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'hidden_size': 3584, 'num_attention_heads': 28, 'num_key_value_heads': 4, 'head_dim': 128, 'intermediate_size': 18944, 'num_hidden_layers': 28, 'rms_norm_eps': 1e-06, 'rope_theta': 1000000.0, 'rope_scaling': {'mrope_section': [16, 24, 24]}}))
    manifest = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (config, checkpoint)}
    with pytest.raises(ValueError, match='fixed 28-layer W8A16 profile'):
        build_qwen25vl_backbone(tmp_path, asset_manifest=manifest)
    cfg = json.loads(config.read_text())
    del cfg['head_dim']
    config.write_text(json.dumps(cfg))
    manifest[config] = hashlib.sha256(config.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='fixed 28-layer W8A16 profile') as exc:
        build_qwen25vl_backbone(tmp_path, asset_manifest=manifest)
    assert 'head_dim=None' not in str(exc.value)
    assert 'missing model.norm.weight' in str(exc.value)
    with pytest.raises(ValueError, match='positive integer'):
        build_qwen25vl_backbone(tmp_path, max_seq_len=0, asset_manifest=manifest)
    nav_checkpoint = tmp_path / 'navdp.safetensors'
    nav_checkpoint.write_bytes(b'manifest-verified mock')
    nav_manifest = {nav_checkpoint: hashlib.sha256(nav_checkpoint.read_bytes()).hexdigest()}
    monkeypatch.setattr(safetensors.torch, 'load_file', lambda _path: {})
    with pytest.raises(ValueError, match='fixed 16-layer profile'):
        build_navdp(nav_checkpoint, asset_manifest=nav_manifest)
    with pytest.raises(ValueError, match='Former checkpoint profile mismatch'):
        FormerRPU(nav_checkpoint, asset_manifest=nav_manifest)
    with pytest.raises(ValueError, match='requires 12 blocks'):
        RGBDTowerRPU(SimpleNamespace(blocks=[]), asset_manifest=nav_manifest, asset_paths=(nav_checkpoint,))

def test_goal_glue_rejects_rpu_before_load_file(monkeypatch):
    called = False

    def fail_if_called(_path):
        nonlocal called
        called = True
        raise AssertionError('load_file must not run')
    import safetensors.torch
    monkeypatch.setattr(safetensors.torch, 'load_file', fail_if_called)
    with pytest.raises(ValueError, match="only device='cpu'"):
        build_goal_glue('/must-not-be-read', device='rpu')
    from rpu_backend.adapters.internvla_n1.rgbd import build_rgbd_encoder
    with pytest.raises(ValueError, match='build_rgbd_encoder_rpu'):
        build_rgbd_encoder('/must-not-be-read', '/must-not-import', device='rpu')
    assert not called

def test_goal_glue_cpu_is_fp32_and_strict(tmp_path):
    from safetensors.torch import save_file
    from rpu_backend.adapters.internvla_n1.goal_glue import GoalGlue
    checkpoint = tmp_path / 'goal.safetensors'
    model = GoalGlue()
    state = {f'vlm_embed_mlp.{key}': value for (key, value) in model.vlm_embed_mlp.state_dict().items()}
    state.update({f'goal_compressor.{key}': value for (key, value) in model.goal_compressor.state_dict().items()})
    state['goal_compressor.positional_encoding.pe'] = torch.zeros(1000, 384)
    save_file(state, str(checkpoint))
    loaded = build_goal_glue(checkpoint)
    assert all((value.dtype == torch.float32 for value in loaded.parameters()))
    state['goal_compressor.not_a_source_key'] = torch.zeros(1)
    save_file(state, str(checkpoint))
    with pytest.raises(RuntimeError, match='Unexpected key'):
        build_goal_glue(checkpoint)
    del state['goal_compressor.not_a_source_key']
    state['goal_compressor.positional_encoding.pe'] = torch.zeros(999, 384)
    save_file(state, str(checkpoint))
    with pytest.raises(ValueError, match='must have shape'):
        build_goal_glue(checkpoint)
    state['goal_compressor.positional_encoding.pe'] = torch.zeros(1000, 384)
    del state['goal_compressor.cross_attention.out_proj.bias']
    save_file(state, str(checkpoint))
    with pytest.raises(RuntimeError, match='Missing key'):
        build_goal_glue(checkpoint)

def test_rgbd_cpu_is_fp32_and_strict(monkeypatch):
    import safetensors.torch
    import rpu_backend.adapters.internvla_n1.rgbd as rgbd

    class FakeRGBD(torch.nn.Module):

        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
    monkeypatch.setattr(rgbd, 'RGBDEncoder', FakeRGBD)
    monkeypatch.setattr(safetensors.torch, 'load_file', lambda _path: {'rgbd_encoder.linear.weight': torch.ones(2, 2), 'rgbd_encoder.linear.bias': torch.zeros(2)})
    loaded = rgbd.build_rgbd_encoder('unused', 'unused')
    assert all((value.dtype == torch.float32 for value in loaded.parameters()))
    monkeypatch.setattr(safetensors.torch, 'load_file', lambda _path: {'rgbd_encoder.linear.weight': torch.ones(2, 2)})
    with pytest.raises(RuntimeError, match='Missing key'):
        rgbd.build_rgbd_encoder('unused', 'unused')

def test_rgbd_preprocess_constants_do_not_weaken_strict_weights(monkeypatch, tmp_path):
    import rpu_backend.adapters.internvla_n1.rgbd as rgbd
    monkeypatch.setattr(rgbd, '_load_dinov2', lambda _src: lambda _name: torch.nn.Linear(1, 1))
    monkeypatch.setattr(rgbd, '_assert_dinov2_import_source', lambda _src: None)
    monkeypatch.setattr(rgbd.nn, 'TransformerDecoderLayer', lambda *_args, **_kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(rgbd.nn, 'TransformerDecoder', lambda layer, _count: layer)
    encoder = rgbd.RGBDEncoder(tmp_path, input_dtype=torch.float32)
    state = encoder.state_dict()
    assert 'preprocess_mean' not in state
    assert 'preprocess_std' not in state
    assert {'preprocess_mean', 'preprocess_std'}.issubset(encoder._non_persistent_buffers_set)
    encoder.load_state_dict(state, strict=True)
    missing = dict(state)
    del missing['project_layer.bias']
    with pytest.raises(RuntimeError) as error:
        encoder.load_state_dict(missing, strict=True)
    assert '"project_layer.bias"' in str(error.value)
    unexpected = dict(state)
    unexpected['not_a_model_key'] = torch.ones(1)
    with pytest.raises(RuntimeError) as error:
        encoder.load_state_dict(unexpected, strict=True)
    assert '"not_a_model_key"' in str(error.value)
    encoder.double()
    assert encoder.preprocess_mean.dtype == torch.float64
    assert encoder.preprocess_std.dtype == torch.float64

def test_source_has_no_production_debug_or_import_env_switches():
    root = Path(__file__).resolve().parents[4]
    adapter_root = root / 'python/rpu_backend/adapters'
    paths = tuple(sorted((adapter_root / 'internvla_n1').glob('*.py'))) + (adapter_root / 'navdp/runtime.py',)
    sources = {path: path.read_text(encoding='utf-8') for path in paths}
    source = '\n'.join(sources.values())

    def is_os_environ(node):
        return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and (node.value.id == 'os') and (node.attr == 'environ')
    writes = []
    for (path, text) in sources.items():
        for node in ast.walk(ast.parse(text, filename=str(path))):
            direct_write = isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)) and is_os_environ(node.value) or (is_os_environ(node) and isinstance(node.ctx, (ast.Store, ast.Del)))
            mutating_call = isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (is_os_environ(node.func.value) and node.func.attr in {'clear', 'pop', 'popitem', 'setdefault', 'update', '__delitem__', '__setitem__'} or (isinstance(node.func.value, ast.Name) and node.func.value.id == 'os' and (node.func.attr in {'putenv', 'unsetenv'})))
            if direct_write or mutating_call:
                writes.append(f'{path}:{node.lineno}')
    assert writes == []
    assert 'RPU_NAVDP_TAP' not in source
    assert 'RPU_NAVDP_OTAP' not in source
    assert 'RPU_NAVDP_XKV_NORM' not in source

def test_controlled_context_and_vision_builder_are_exact(monkeypatch, tmp_path):
    import rpu_backend.adapters.internvla_n1.vision as internvla_vision
    assert inspect.signature(build_qwen25vl_backbone).parameters['max_seq_len'].default == 4096
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'vision_config': {'depth': 32, 'hidden_size': 1280, 'num_heads': 16, 'intermediate_size': 3420, 'spatial_merge_size': 2, 'out_hidden_size': 3584, 'window_size': 112, 'patch_size': 14, 'fullatt_block_indexes': [7, 15, 23, 31]}}))
    checkpoint = tmp_path / 'model.safetensors'
    checkpoint.write_bytes(b'manifest-verified vision checkpoint')
    manifest = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (config, checkpoint)}
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    captured = {}
    monkeypatch.setattr(internvla_vision, 'build_wall_oss_vision', lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs) or SimpleNamespace())
    build_internvla_vision(tmp_path, asset_manifest=manifest)
    assert captured['kwargs'] == {'window': True, 'max_hw': 32, 'max_seq_len': 1024, 'w8a16': False, 'fp16_ckpt_dir': str(tmp_path.resolve()), 'execution_config': {'vision': {'chunk_size': 'auto'}}, 'per_window_sdpa': True}

def test_backbone_installs_requested_capacity_and_measured_chunk():
    import rpu_backend.adapters.internvla_n1.backbone as backbone
    assert backbone._CERTIFIED_CHUNK_SIZE == 176
    source = inspect.getsource(backbone.build_qwen25vl_backbone)
    weights = source.index('set_weights(handle, *args)')
    envelope = source.index('causal_decoder_set_chunk_envelope')
    auto = source.index('causal_decoder_set_chunk_size_override(handle, 0)')
    publish = source.index('Qwen25VLBackbone.__init__(pending')
    assert weights < envelope < auto < publish

@pytest.mark.parametrize(('logical_len', 'execution_len', 'resolved_chunk'), [(1904, 1936, 176), (157, 157, 80)])
def test_backbone_prefill_preserves_logical_rows_and_native_plan(monkeypatch, logical_len, execution_len, resolved_chunk):
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_planner_cache_identity', lambda handle: (handle, 1), raising=False)
    import rpu_backend.adapters.internvla_n1.backbone as backbone
    calls = []
    captured = {}
    schedules = []
    count = (execution_len + resolved_chunk - 1) // resolved_chunk
    for _stage in range(3):
        for index in range(count):
            offset = index * resolved_chunk
            length = min(resolved_chunk, execution_len - offset)
            schedules.extend((index, offset, length, offset + length))
    descriptor = [1, 0, resolved_chunk, resolved_chunk, resolved_chunk, 0, 1, 0, 0, 0, 0, count, count, count, 1, *schedules, 0, execution_len, -1]
    descriptor[1] = len(descriptor)
    domain = [1, 1, len(descriptor), *descriptor]

    def resolve(handle, rows, position, _causal, _mask_kv_len, rope_mode, graph_lifecycle, native_logical_len):
        assert (rope_mode, graph_lifecycle) == (1, 4)
        assert native_logical_len == logical_len
        calls.append(('resolve', handle, rows, position))
        return domain if rows == execution_len else [1, 0]

    def forward(handle, embeds, _k, _v, _mask, position, _causal, position_ids, _deepstack, _cos, _sin, _input_chunk, _qkv_chunk, _unsafe, planned_chunk, planned_descriptor):
        calls.append(('forward', handle, embeds.shape[1], position))
        assert planned_chunk == 0
        assert tuple(planned_descriptor) == tuple(descriptor)
        captured['embeds'] = embeds
        captured['position_ids'] = position_ids
        return embeds.clone()
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_resolve_prefill_stage_domain', resolve, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_forward', forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_get_resolved_chunk_size', lambda handle: resolved_chunk, raising=False)
    monkeypatch.setattr(backbone, '_to_rpu_half', lambda tensor: tensor.half())

    class Cache:
        k_caches = []
        v_caches = []
        position = 0
        max_seq_len = 2048

        def reset_to_position(self, position):
            self.position = position

        def update_position(self, rows):
            self.position += rows

    class GraphCache:

        def prepare_plan(self, key, prepare):
            return prepare()

        def is_frozen(self):
            return False

        def capture(self, signature):
            captured['signature'] = signature
            return nullcontext()
    model = backbone.Qwen25VLBackbone.__new__(backbone.Qwen25VLBackbone)
    model._handle = 17
    model._head_dim = 128
    model._rope_theta = 1000000.0
    model._mrope_section = [16, 24, 24]
    model._partial_mrope = False
    model.cache = Cache()
    model._graph_cache = GraphCache()
    model.hidden_size = 2
    model.num_layers = 28
    model._prefill_rope_cache = {}
    model._prefill_position_cache = {}
    model._prefill_input_cache = {}
    model._host_cache = False
    model._prefill_position_ids = lambda value, _rows: value.to(torch.int32)
    embeds = torch.ones((1, logical_len, 2), dtype=torch.float32)
    positions = torch.ones((logical_len, 3), dtype=torch.int32)
    output = model.forward_embeds(embeds, positions)
    assert output.shape == (1, logical_len, 2)
    assert captured['embeds'].shape == (1, execution_len, 2)
    assert captured['position_ids'].shape == (execution_len, 3)
    if execution_len > logical_len:
        assert torch.count_nonzero(captured['embeds'][:, logical_len:]).item() == 0
        assert torch.count_nonzero(captured['position_ids'][logical_len:]).item() == 0
    assert model.cache.position == logical_len
    receipt = model._rpu_last_execution_plan
    assert receipt['logical_len'] == logical_len
    assert receipt['execution_len'] == execution_len
    assert receipt['chunk_size'] == resolved_chunk
    assert receipt['padding_rows'] == execution_len - logical_len
    assert receipt['component'] == 'system2_backbone'
    assert receipt['stage'] == 'prefill'
    assert receipt['generation'] == 0
    assert receipt['descriptor_authority'] is True
    assert calls[0] == ('resolve', 17, execution_len, 0)

def test_controlled_vision_rejects_wall_oss_2048_merger(monkeypatch, tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'vision_config': {'depth': 32, 'hidden_size': 1280, 'num_heads': 16, 'intermediate_size': 3420, 'spatial_merge_size': 2, 'out_hidden_size': 2048, 'window_size': 112, 'patch_size': 14, 'fullatt_block_indexes': [7, 15, 23, 31]}}))
    checkpoint = tmp_path / 'model.safetensors'
    checkpoint.write_bytes(b'wrong-width vision checkpoint')
    manifest = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (config, checkpoint)}
    monkeypatch.setenv(NUMERIC_BLOCKED_ENV, '1')
    with pytest.raises(ValueError, match='out_hidden_size=2048.*expected 3584'):
        build_internvla_vision(tmp_path, asset_manifest=manifest)

def test_per_window_cpp_contract_is_strict_and_mask_free():
    root = Path(__file__).resolve().parents[4]
    cpp = (root / 'src/fused/rpu_qwen25vl_vision_model.cpp').read_text(encoding='utf-8')
    schema = (root / 'src/core/rpu_dispatch_registrations.inc').read_text(encoding='utf-8')
    for required in ('!per_window_sdpa_enabled_ || has_window_boundaries', '!has_window_boundaries || per_window_sdpa_enabled_ ||', 'is_exact_mimc()', 'boundaries[0] == 0', 'num_patches_in / image_batch_count_', 'boundaries[cu.numel() - 1] == boundary_seq', 'boundaries[i] > boundaries[i - 1]', 'boundaries[i] % 16 == 0', '!z1_dispatch && image_batch_count_ == 1', '!mixed_precision_ && is_generic() &&', 'schedule_ == VisionSchedule::SISC', 'must not receive a dense window mask', 'else if (!is_full && per_window_sdpa_enabled_)', 'k_cache.narrow(\n                    1, window_begin / 16, window_len / 16)', 'h = detail::layout_mix(h, per_window_sdpa_enabled_ ? 1 : 0)', 'Z1 rejects per-window SDPA handles'):
        assert required in cpp
    assert 'Tensor? cu_window_seqlens=None' in schema

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch
from rpu_backend.adapters.wall_oss import runtime as wall_runtime
from rpu_backend.adapters.wall_oss import checkpoint_validation
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'examples/configs/wall_oss/sorting'
PRECISIONS = ('fp16', 'w8a16', 'w8a16_action_w4a16')
CONFIG_FILENAMES = {'fp16': 'fp16', 'w8a16': 'w8a16', 'w8a16_action_w4a16': 'w8a16_action_w4a16_g32'}

def load(relative):
    return importlib.import_module(relative[:-3].replace('/', '.'))

@pytest.fixture
def runner(tmp_path, monkeypatch):
    runner = load('examples/runners/wall_oss/sorting.py')
    (tmp_path / 'release').mkdir(exist_ok=True)
    artifacts = {}
    for role, key, filename in [('rhino_ops_main', 'RPU_KERNEL_LIB_PATH', 'rhinoOpLib_current.ref'), ('rhino_ops_expansion', 'RPU_KERNEL_LIB_PATH_EXPANSION', 'rhinoExpansionOpLib_current.ref')]:
        payload = tmp_path / filename
        payload.write_bytes(role.encode())
        artifacts[role] = {'path': filename, 'sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}
        monkeypatch.setenv(key, str(payload))
    (tmp_path / 'release/runtime-dependencies.json').write_text(json.dumps({'artifacts': artifacts}))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    return runner

def config_for(precision='fp16'):
    return load('examples/common/config.py').load_config(CONFIGS / f'{CONFIG_FILENAMES[precision]}.toml')

def batch_for(delay=0):
    active = torch.tensor([True] * 20 + [False] * 6)
    batch = {'images': torch.zeros(3, 252, 448, 3, dtype=torch.uint8), 'instruction': 'Move this parcel to its target.', 'state': torch.arange(26, dtype=torch.float32), 'state_mask': active, 'noise': torch.ones(1, 20, 26), 'dof_mask': active.clone()}
    if delay:
        batch['prefix_actions'] = torch.arange(520, dtype=torch.float32).reshape(20, 26)
    return batch

@pytest.mark.parametrize('precision', PRECISIONS)
def test_shipped_precision_config_dispatches_separate_sorting_entry(precision, monkeypatch):
    common = load('examples/common/config.py')
    selected = common._wall_sorting_runner()
    calls = []
    monkeypatch.setattr(selected, 'run_example_config', lambda config: calls.append(config) or 0)
    monkeypatch.setattr(importlib.import_module('examples.common.cli'), '_wall_sorting_runner', lambda : selected)
    path = CONFIGS / f'{CONFIG_FILENAMES[precision]}.toml'
    monkeypatch.setattr(sys, 'argv', ['run.py', '--config', str(path)])
    assert importlib.import_module('examples.common.cli').main() == 0
    assert len(calls) == 1
    assert calls[0]['wall_oss_sorting'] == {'precision': precision}

def test_check_config_runs_without_torch_or_backend_imports():
    code = "\nimport importlib.abc\nfrom pathlib import Path\nimport runpy\nimport sys\nclass NoRuntime(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname.split('.')[0] in {'torch', 'rpu_backend'}:\n            raise AssertionError('check-config imported ' + fullname)\nsys.meta_path.insert(0, NoRuntime())\nroot = Path(sys.argv[1])\nsys.path.insert(0, str(root / 'examples'))\nsys.argv = ['run.py', '--config', str(root / 'examples/configs/wall_oss/sorting/fp16.toml'), '--check-config']\nrunpy.run_path(str(root / 'examples/run.py'), run_name='__main__')\n"
    result = subprocess.run([sys.executable, '-S', '-c', code, str(ROOT)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'syntax only' in result.stdout

@pytest.mark.parametrize('target,flag,error', [('wall_oss_sorting', '--profile', 'do not enable tracing'), ('wall_oss_sorting', '--hwperf', 'do not enable tracing')])
def test_tracing_flags_fail_before_running(target, flag, error, monkeypatch, capsys):
    common = load('examples/common/config.py')
    path = CONFIGS / 'fp16.toml'
    monkeypatch.setattr(sys, 'argv', ['example.py', '--config', str(path), flag])
    with pytest.raises(SystemExit) as caught:
        importlib.import_module('examples.common.cli').main()
    assert caught.value.code == 2
    assert error in capsys.readouterr().err

@pytest.mark.parametrize('precision,section,key,value', [('fp16', 'wall_oss_sorting', 'evaluation_only', False), ('fp16', 'wall_oss_sorting', 'evaluation_only', True), ('fp16', 'wall_oss_sorting', 'precision', 'w8a8'), ('fp16', 'wall_oss_sorting', 'num_cores', 4), ('fp16', 'input', 'w8_checkpoint', '/unused/w8'), ('w8a16', 'input', 'w8_checkpoint', ''), ('w8a16', 'input', 'action_w4_checkpoint', '/unused/w4'), ('w8a16_action_w4a16', 'input', 'action_w4_checkpoint', None), ('fp16', 'input', 'inference_delay', 2), ('fp16', 'input', 'inference_delay', True), ('fp16', 'input', 'batch', ' '), ('fp16', 'run', 'warmup', 0), ('fp16', 'run', 'runs', True), ('fp16', 'run', 'output_dir', ''), ('fp16', 'run', 'profile', True)])
def test_config_rejects_incompatible_assets_and_controls(runner, precision, section, key, value):
    config = config_for(precision)
    config[section][key] = value
    with pytest.raises(ValueError):
        runner.validate_example_config(config)

@pytest.mark.parametrize('delay', [0, 1, 3, 10])
def test_observation_preserves_input_and_expands_dof_mask(runner, delay):
    batch = batch_for(delay)
    batch['images'][1, 0, 0] = torch.tensor([23, 42, 71], dtype=torch.uint8)
    common = runner.observation(batch, delay)
    assert [image.size for image in common['image']] == [(448, 252)] * 3
    assert common['image'][1].getpixel((0, 0)) == (23, 42, 71)
    assert common['dof_mask'].shape == (1, 20, 26)
    assert common['dof_mask'].is_contiguous()
    assert torch.equal(common['dof_mask'].bool(), batch['dof_mask'].expand(1, 20, 26))
    assert torch.equal(common['proprioception'], batch['state'])
    assert common['num_steps'] == 6 and common['use_rtc']
    assert common['inference_delay'] == delay
    if delay:
        assert torch.equal(common['rtc_prefix_actions'], batch['prefix_actions'])
    else:
        assert common['rtc_prefix_actions'] is None

@pytest.mark.parametrize('field,value,error', [('images', torch.zeros(2, 252, 448, 3, dtype=torch.uint8), 'images'), ('images', torch.zeros(3, 252, 448, 3), 'dtype'), ('state', torch.zeros(1, 26), 'state'), ('state', torch.full((26,), float('nan')), 'nonfinite'), ('noise', torch.zeros(20, 26), 'noise'), ('dof_mask', torch.ones(26, dtype=torch.bool), 'must be'), ('state_mask', torch.ones(26), 'dtype'), ('instruction', ' ', 'instruction')])
def test_observation_rejects_invalid_input_before_model_creation(runner, field, value, error):
    batch = batch_for()
    batch[field] = value
    with pytest.raises(ValueError, match=error):
        runner.observation(batch, 0)

def test_rtc_prefix_is_required_only_for_nonzero_delay(runner):
    with pytest.raises(ValueError, match='requires batch prefix_actions'):
        runner.observation(batch_for(), 3)
    with pytest.raises(ValueError, match='requires nonzero'):
        runner.observation(batch_for(3), 0)

@pytest.mark.parametrize('precision,mode', [('fp16', 'fp16'), ('w8a16', 'w8'), ('w8a16_action_w4a16', 'action-w4')])
def test_cold_configuration_reuses_validation_selectors(runner, precision, mode, monkeypatch):
    monkeypatch.delenv('RPU_BACKEND_SO', raising=False)
    expected = runner.validation_runner().exact_environment(runner.ROOT, mode)
    expected.pop('RPU_BACKEND_SO')
    (env, records) = runner.cold_environment(precision)
    assert 'RPU_BACKEND_SO' not in env  # The installed wheel selects its native.
    assert {key: env[key] for key in expected if not key.startswith('RPU_KERNEL_LIB_PATH')} == {key: value for (key, value) in expected.items() if not key.startswith('RPU_KERNEL_LIB_PATH')}
    assert env['RPU_LINEAR_ACC32'] == '0'
    assert set(records) == {'RPU_KERNEL_LIB_PATH', 'RPU_KERNEL_LIB_PATH_EXPANSION'}
    for (key, record) in records.items():
        assert env[key] == record['path'] and Path(record['path']).is_file()

def test_existing_checkout_build_is_selected(runner, monkeypatch):
    monkeypatch.delenv('RPU_BACKEND_SO', raising=False)
    native = runner.ROOT / 'build/rpu_backend/rpu_backend.so'
    native.parent.mkdir(parents=True)
    native.touch()
    env, _ = runner.cold_environment('fp16')
    assert env['RPU_BACKEND_SO'] == str(native.resolve())

def test_cold_configuration_follows_current_aliases_and_manifest(runner, tmp_path, monkeypatch):
    validation = runner.validation_runner()
    source = tmp_path / 'src'
    source.mkdir()
    (tmp_path / 'release').mkdir(exist_ok=True)
    artifacts = {}
    for (role, alias, content) in (('rhino_ops_main', 'rhinoOpLib_current.ref', b'new main payload'), ('rhino_ops_expansion', 'rhinoExpansionOpLib_current.ref', b'new expansion payload')):
        (source / alias).write_bytes(content)
        artifacts[role] = {'path': alias, 'sha256': hashlib.sha256(content).hexdigest()}
        monkeypatch.setenv('RPU_KERNEL_LIB_PATH' if role == 'rhino_ops_main' else 'RPU_KERNEL_LIB_PATH_EXPANSION', str(source / alias))
    (tmp_path / 'release/runtime-dependencies.json').write_text(json.dumps({'artifacts': artifacts}))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'validation_runner', lambda : validation)
    (env, records) = runner.cold_environment('fp16')
    for (key, role) in (('RPU_KERNEL_LIB_PATH', 'rhino_ops_main'), ('RPU_KERNEL_LIB_PATH_EXPANSION', 'rhino_ops_expansion')):
        assert env[key] == str(source / artifacts[role]['path'])
        assert records[key] == {'path': env[key], 'sha256': artifacts[role]['sha256']}
    (source / 'rhinoOpLib_current.ref').write_bytes(b'changed payload')
    with pytest.raises(ValueError, match='checksum mismatch'):
        runner.cold_environment('fp16')

@pytest.mark.parametrize('key,value', [('RPU_WALL_OSS_FP16_TRANSFER_MASK', '0'), ('RPU_WALL_OSS_UNKNOWN_OVERRIDE', '1'), ('RPU_CHUNK_FORCE_UNSAFE', '1'), ('WALL_OSS_PROFILE_PREFIX', '/tmp/profile')])
def test_cold_conflicts_are_atomic(runner, key, value, monkeypatch):
    (env, _) = runner.cold_environment('fp16')
    isolated = {name: existing for (name, existing) in os.environ.items() if name not in env and (not name.startswith('RPU_WALL_OSS_')) and (name not in {'RPU_CHUNK_FORCE_UNSAFE', 'WALL_OSS_PROFILE_PREFIX', 'RPU_RMSNORM_VWARP'})}
    isolated[key] = value
    monkeypatch.setattr(os, 'environ', isolated)
    monkeypatch.delitem(sys.modules, 'rpu_backend', raising=False)
    before = dict(os.environ)
    with pytest.raises(ValueError, match=key):
        runner.bind_cold_environment(env)
    assert dict(os.environ) == before

def test_already_loaded_backend_rejects_cold_binding(runner, monkeypatch):
    monkeypatch.setitem(sys.modules, 'rpu_backend', ModuleType('rpu_backend'))
    before = dict(os.environ)
    with pytest.raises(ValueError, match='fresh process'):
        runner.bind_cold_environment({'RPU_LINEAR_ACC32': '0'})
    assert dict(os.environ) == before

@pytest.mark.parametrize('spelling', ['relative', 'symlink'])
def test_selected_native_build_accepts_equivalent_path_spelling(runner, spelling, tmp_path, monkeypatch):
    native = tmp_path / 'native.so'
    native.touch()
    alias = tmp_path / 'selected.so'
    alias.symlink_to(native)
    monkeypatch.chdir(tmp_path)
    selected = 'native.so' if spelling == 'relative' else str(alias)
    (env, _) = runner.cold_environment('fp16')
    isolated = {key: value for (key, value) in os.environ.items() if key not in env and (not key.startswith('RPU_WALL_OSS_')) and (key not in {'RPU_CHUNK_FORCE_UNSAFE', 'WALL_OSS_PROFILE_PREFIX', 'RPU_RMSNORM_VWARP'})}
    isolated.update({key: value for key, value in env.items() if key.startswith('RPU_KERNEL_LIB_PATH')})
    isolated['RPU_BACKEND_SO'] = selected
    monkeypatch.setattr(os, 'environ', isolated)
    monkeypatch.delitem(sys.modules, 'rpu_backend', raising=False)
    (env, _) = runner.cold_environment('fp16')
    runner.bind_cold_environment(env)
    assert env['RPU_BACKEND_SO'] == str(native)
    assert os.environ['RPU_BACKEND_SO'] == str(native)

@pytest.mark.parametrize('precision', PRECISIONS)
def test_checkpoint_pairing_routes_w4_only_to_action(runner, precision, tmp_path, monkeypatch):
    config = config_for(precision)
    inputs = config['input']
    for key in ('fp16_checkpoint', 'w8_checkpoint', 'action_w4_checkpoint'):
        if key in inputs:
            folder = tmp_path / key
            folder.mkdir()
            (folder / 'config.json').write_text('{}')
            inputs[key] = str(folder)
    checks = []
    monkeypatch.setattr(checkpoint_validation, 'validate_w8_nvfp4_checkpoint_pair', lambda quant, source: checks.append(('w8', quant, source)))
    monkeypatch.setattr(checkpoint_validation, 'validate_rtc_w4_checkpoint_pair', lambda quant, source: checks.append(('w4', quant, source)))
    (checkpoint, kwargs) = runner.checkpoint_arguments(inputs, precision)
    assert checkpoint == inputs.get('w8_checkpoint', inputs['fp16_checkpoint'])
    assert kwargs['w8a16'] == (precision != 'fp16')
    assert kwargs['action_w4a16_ckpt_dir'] == inputs.get('action_w4_checkpoint')
    (_root, text, vision, action) = wall_runtime.resolve_wall_oss_execution_components(kwargs['rpu_execution'], entry_point='parcel-sorting example checkpoint')
    assert text['prefill']['chunk_size'] == 'auto'
    assert vision['vision']['chunk_size'] == 'auto'
    assert action['action']['chunk_size'] == 'auto'
    assert [kind for (kind, _, _) in checks] == ([] if precision == 'fp16' else ['w8'] if precision == 'w8a16' else ['w8', 'w4'])
    assert all((source == Path(inputs['fp16_checkpoint']) for (_, _, source) in checks))

@pytest.mark.parametrize('metadata_location', ['sidecar', 'embedded'])
def test_fp16_source_rejects_quantized_checkpoint_metadata(runner, metadata_location, tmp_path):
    source_config = {}
    quant = {'method': 'w8a16'}
    if metadata_location == 'sidecar':
        (tmp_path / 'rpu_quant_config.json').write_text(json.dumps(quant))
    else:
        source_config['quant_config'] = quant
    (tmp_path / 'config.json').write_text(json.dumps(source_config))
    with pytest.raises(ValueError, match='must be the unquantized'):
        runner.checkpoint_arguments({'fp16_checkpoint': str(tmp_path)}, 'fp16')

@pytest.mark.parametrize('failure,warmup,delay', [(failure, warmup, 3) for failure in (None, 'arenas', 'predict', 'replay', 'close') for warmup in (1, 2)] + [(None, 2, 0)])
def test_full_run_excludes_warmup_and_closes_before_reporting(runner, failure, warmup, delay, tmp_path, monkeypatch):
    config = config_for()
    config['cpu'] = {'threads': 3}
    threads = [8]
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    monkeypatch.setenv('MKL_NUM_THREADS', '8')
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: threads.__setitem__(0, count))
    monkeypatch.setattr(torch, 'get_num_threads', lambda : threads[0])
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    config['input'].update(fp16_checkpoint=str(checkpoint), batch=str(tmp_path / 'batch.pt'), inference_delay=delay)
    config['run'].update(warmup=warmup, runs=3, output_dir=str(tmp_path / 'result'))
    batch = batch_for(delay)
    expected_actions = batch['prefix_actions'] if delay else torch.zeros(20, 26)
    torch.save(batch, config['input']['batch'])
    calls = []

    class Cache:
        phase = 'READY'

        def __init__(self, count):
            (self.count, self.replays) = (count, 0)

        def snapshot(self):
            return [SimpleNamespace(replay_count=self.replays, recapture_count=0)]

        def size(self):
            return self.count

        def cache_invariant_ok(self):
            return True

    class Model:

        def __init__(self):
            self.vision = SimpleNamespace(_graph_cache=Cache(1))
            self.llm = SimpleNamespace(_graph_cache=Cache(1))
            self.action = SimpleNamespace(_graph_cache=Cache(12))

        def prepare_graphs(self, **kwargs):
            calls.append('prepare')
            assert kwargs['prefix_length_range'] == (599, 640)
            assert kwargs['noise'].shape == kwargs['dof_mask'].shape == (1, 20, 26)
            return {'phase': 'READY', 'ready_replay_validated': True, 'graph_counts': {'vision': 1, 'prefill': 1, 'denoise': 12}}

        def predict(self, **kwargs):
            calls.append('predict')
            if failure == 'predict':
                raise RuntimeError('predict failed')
            assert kwargs['debug'] is False and kwargs['inference_delay'] == delay
            for component in (self.vision, self.llm, self.action):
                component._graph_cache.replays += 1
            action = expected_actions.unsqueeze(0).clone()
            if failure == 'replay' and calls.count('predict') > 1:
                action[0, 19, 0] += 1
            return {'action': action, 'prefix_len': 640, 'real_prefix_len': 639}

        def close(self):
            calls.append('close')
            assert not (tmp_path / 'result/report.json').exists()
            if failure == 'close':
                raise RuntimeError('close failed')

    def build(checkpoint, **kwargs):
        assert calls == ['environment', 'arenas']
        assert torch.get_num_threads() == 3
        assert os.environ['OMP_NUM_THREADS'] == os.environ['MKL_NUM_THREADS'] == '3'
        wall_runtime.resolve_wall_oss_execution_components(kwargs['rpu_execution'], entry_point='parcel-sorting example build')
        calls.append('build')
        return Model()
    (backend, wall) = (ModuleType('rpu_backend'), ModuleType('rpu_backend.adapters.wall_oss'))
    graph = ModuleType('rpu_backend.graph')

    class ArenaPolicy:

        @classmethod
        def from_environment(cls, *, graph_arena_count):
            assert graph_arena_count == 15
            return cls()

        def prepare_arenas(self):
            calls.append('arenas')
            return failure != 'arenas'
    graph.GraphRuntimePolicy = ArenaPolicy
    backend.__file__ = str(runner.ROOT / 'python/rpu_backend/__init__.py')
    backend._cpp_ext = SimpleNamespace(get_runtime_payload_paths=lambda : {'oplib_roles': {'main': '/main.ref', 'expansion': '/sorting.ref'}, 'oplib_payloads': {'/main.ref': {'sha256': 'main'}, '/sorting.ref': {'sha256': 'sorting'}}})
    wall.build_wall_oss_vla = build
    monkeypatch.setitem(sys.modules, 'rpu_backend', backend)
    monkeypatch.setitem(sys.modules, 'rpu_backend.adapters.wall_oss', wall)
    monkeypatch.setitem(sys.modules, 'rpu_backend.graph', graph)
    monkeypatch.setattr(torch, 'rpu', SimpleNamespace(set_caching_allocator=lambda value: None, set_ddr_flush=lambda value: None, set_ddr_flush_force=lambda value: None), raising=False)
    monkeypatch.setattr(runner, 'cold_environment', lambda precision, cores: ({}, {'RPU_KERNEL_LIB_PATH': {'path': '/main.ref', 'sha256': 'main'}, 'RPU_KERNEL_LIB_PATH_EXPANSION': {'path': '/sorting.ref', 'sha256': 'sorting'}}))
    monkeypatch.setattr(runner, 'bind_cold_environment', lambda env: calls.append('environment'))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    class EditableFinder:
        __module__ = '_rpu_backend_editable'

        def find_spec(self, fullname, path=None, target=None):
            return None
    sys.meta_path.append(EditableFinder())
    original_paths, original_finders = list(sys.path), list(sys.meta_path)
    if failure:
        with pytest.raises(RuntimeError, match={'arenas': 'could not prepare fifteen', 'predict': 'predict failed', 'replay': 'changed actions', 'close': 'close failed'}[failure]):
            runner.run_example_config(config)
        report = json.loads((tmp_path / 'result/report.json').read_text())
        assert report['execution']['status'] == 'FAILED'
        assert report['validation']['runtime']['status'] == 'NOT_COMPLETED'
        assert report['validation']['reference']['status'] == 'NOT_RUN'
        assert not (tmp_path / 'result/actions.pt').exists()
    else:
        assert runner.run_example_config(config) == 0
        report = json.loads((tmp_path / 'result/report.json').read_text())
        assert report['execution']['status'] == 'COMPLETED'
        assert report['timing'][0]['samples'] == report['e2e']['samples_ms']
        assert report['timing'][0]['unit'] == 'ms'
        assert report['validation']['task']['status'] == 'NOT_EVALUATED'
        assert len(report['e2e']['samples_ms']) == 3
        assert report['cpu_runtime']['torch_num_threads'] == 3
        assert report['cpu_runtime']['OMP_NUM_THREADS'] == '3'
        assert report['warmup'] == warmup and report['runs'] == 3
        assert report['precision'] == 'fp16' and report['formal_certification'] is False
        assert report['evaluation_only'] is True
        assert report['fixed_input_replay_equal'] and report['inference_delay'] == delay
        assert report['graph_arena_count'] == 15
        assert ('supplied RTC prefix preserved' in report['validation']['runtime']['checks']) == bool(delay)
        assert calls.count('predict') == warmup + 3
        actions = torch.load(tmp_path / 'result/actions.pt', weights_only=True)
        assert torch.equal(actions, expected_actions.unsqueeze(0))
    if failure == 'arenas':
        assert calls == ['environment', 'arenas']
    else:
        assert calls[:4] == ['environment', 'arenas', 'build', 'prepare'] and calls[-1] == 'close'
    assert sys.path == original_paths
    assert sys.meta_path == original_finders

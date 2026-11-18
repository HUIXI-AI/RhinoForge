"""Wall base config admission and exact measured-route dispatch, without device."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tomllib

import pytest
from _example_toml import load_example_toml


ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return importlib.import_module(relative[:-3].replace("/", "."))


def config_for(precision='w8a16_action_nvfp4', size=256):
    config = {
        'wall_oss': {'precision': precision},
        'input': {'fp16_checkpoint': '/missing/fp16', 'batch': '/missing/observation.pt',
                  'image_size': [size, size], 'cameras': 3, 'text_tokens': 32,
                  'action_horizon': 32, 'num_steps': 10},
        'run': {'warmup': 2, 'runs': 5, 'cpu_threads': 8, 'seed': 0,
                'output_dir': '/missing/result'},
    }
    if precision != 'fp16':
        config['input']['w8_checkpoint'] = '/missing/w8'
    return config


def write_config(path):
    path.write_text('''[wall_oss]
precision = "w8a16_action_nvfp4"
[input]
fp16_checkpoint = "/missing/fp16"
w8_checkpoint = "/missing/w8"
batch = "/missing/batch.pt"
image_size = [256, 256]
cameras = 3
text_tokens = 32
action_horizon = 32
num_steps = 10
[run]
output_dir = "/missing/output"
[runtime]
native = "/missing/backend.so"
main_ref = "/missing/main.ref"
expansion_ref = "/missing/expansion.ref"
[rpu_execution.action]
chunk_size = 32
''')
    return path


@pytest.mark.parametrize('precision', ('fp16', 'w8a16', 'w8a16_action_nvfp4'))
@pytest.mark.parametrize('size', (224, 256))
def test_config_resolves_exact_route_without_private_runtime_dependencies(precision, size):
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for(precision, size)
    argv = runner.example_argv(config)
    value = lambda key: argv[argv.index(key) + 1]
    assert value('--mode') == 'candidate' and value('--components') == 'small'
    assert value('--precision') == ('fp16' if precision == 'fp16' else 'w8')
    assert value('--cpu-threads') == '8' and value('--warmup') == '2' and value('--runs') == '5'
    assert value('--text-tokens') == '32' and value('--text-length-scope') == 'instruction'
    assert value('--cameras') == '3'
    hybrid = precision == 'w8a16_action_nvfp4'
    assert ('--action-nvfp4' in argv) == hybrid
    assert ('--action-residency' in argv) == (not hybrid)
    assert ('--vision-window' in argv) == (size == 224 and not hybrid)
    assert ('--vision-pair' in argv) == (size == 256)
    assert '--expansion-ref' not in argv and '--main-ref' not in argv
    assert ('--w8' in argv) == (precision != 'fp16')


def test_runtime_override_and_execution_reach_one_engine_without_config_mutation(monkeypatch):
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for()
    config['input']['text_length_scope'] = 'non_image'
    config['input']['text_tokens'] = 173
    config['run']['seed'] = 42
    config['run']['cpu_threads'] = 4
    config['run']['cpu_thread_sweep'] = [8, 12, 16]
    config['runtime'] = {'native': '/custom/backend.so', 'main_ref': '/custom/main.ref',
                         'expansion_ref': '/custom/expansion.ref'}
    config['rpu_execution'] = {'prefill': {'chunk_size': 384, 'padding_budget': 16},
                               'vision': {'chunk_size': 'auto'}, 'action': {'chunk_size': 32}}
    before = deepcopy(config)
    parsed, ran = [], []

    def arguments(argv):
        parsed.append(argv)
        return SimpleNamespace()

    def run(args):
        ran.append(args)
        assert args.rpu_execution == before['rpu_execution']
        assert args.source_config == before and args.seed == 42
        args.source_config['run']['runs'] = 99
        return 7

    monkeypatch.setattr(runner, 'validation_runner',
                        lambda: SimpleNamespace(arguments=arguments, main=run))
    assert runner.run_example_config(config) == 7
    assert len(parsed) == len(ran) == 1 and config == before
    argv = parsed[0]
    for flag, expected in (('--so', '/custom/backend.so'), ('--main-ref', '/custom/main.ref'),
                            ('--expansion-ref', '/custom/expansion.ref'),
                            ('--cpu-threads', '4'),
                            ('--text-tokens', '173'), ('--text-length-scope', 'non_image')):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == expected
    start = argv.index('--cpu-thread-sweep') + 1
    assert argv[start:start + 3] == ['8', '12', '16']


def test_cpu_sweep_budget_and_default_thread_setting():
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for()
    del config['run']['cpu_threads']
    config['run'].update(runs=10, cpu_thread_sweep=[4, 12])
    argv = runner.example_argv(config)
    assert argv[argv.index('--cpu-threads') + 1] == '8'
    config['run']['cpu_thread_sweep'].append(16)
    with pytest.raises(ValueError, match='at most 30'):
        runner.example_argv(config)


@pytest.mark.parametrize('precision', ('fp16', 'w8a16', 'w8a16_action_nvfp4'))
@pytest.mark.parametrize('acc32', (False, True))
def test_accumulation_is_per_stage_and_action_residency_is_acc16_only(precision, acc32):
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for(precision)
    config['rpu_execution'] = {
        'vision': {'linear_acc32': acc32},
        'prefill': {'linear_acc32': not acc32},
        'action': {'linear_acc32': acc32},
    }
    assert runner.execution_config(config['rpu_execution']) == config['rpu_execution']
    argv = runner.example_argv(config)
    assert ('--action-residency' in argv) == (not acc32 and precision != 'w8a16_action_nvfp4')


def test_same_config_selects_matching_text_fixture_without_mutating_inputs():
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for()
    del config['input']['batch']
    config['input']['batch_template'] = '/fixtures/{cameras}cam{image_size}_t{text_tokens}.pt'
    for tokens in (32, 64, 96, 128, 160):
        config['input']['text_tokens'] = tokens
        before = deepcopy(config)
        argv = runner.example_argv(config)
        assert argv[argv.index('--batch') + 1] == f'/fixtures/3cam256_t{tokens}.pt'
        assert argv[argv.index('--text-tokens') + 1] == str(tokens)
        assert before == config


@pytest.mark.parametrize('template', ('/batch.pt', '/{text_tokens.__class__}.pt',
    '/{text_tokens!r}.pt', '/{text_tokens:03d}.pt', '/{missing}.pt', '/{text_tokens.pt'))
def test_batch_template_rejects_unknown_fields_and_format_operations(template):
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for()
    del config['input']['batch']
    config['input']['batch_template'] = template
    with pytest.raises(ValueError):
        runner.validate_example_config(config)


def test_batch_template_conflicts_with_literal_batch():
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for()
    config['input']['batch_template'] = '/{text_tokens}.pt'
    with pytest.raises(ValueError, match='exactly one'):
        runner.validate_example_config(config)


@pytest.mark.parametrize('precision', ('fp16', 'w8a16', 'w8a16_action_nvfp4'))
@pytest.mark.parametrize('enabled', (False, True))
def test_explicit_224_window_override_changes_only_window_flag(precision, enabled):
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for(precision, 224)
    original = runner.example_argv(config)
    config['wall_oss']['vision_window'] = enabled
    selected = runner.example_argv(config)
    assert ('--vision-window' in selected) == enabled
    assert [value for value in selected if value != '--vision-window'] == [
        value for value in original if value != '--vision-window']


def test_explicit_disabled_window_keeps_256_pair():
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for(size=256)
    original = runner.example_argv(config)
    config['wall_oss']['vision_window'] = False
    assert runner.example_argv(config) == original
    config['wall_oss']['vision_window'] = True
    with pytest.raises(ValueError, match='requires 224x224'):
        runner.validate_example_config(config)


@pytest.mark.parametrize('precision', ('fp16', 'w8a16', 'w8a16_action_nvfp4'))
@pytest.mark.parametrize('size', (224, 256))
def test_two_camera_dispatch_and_pair_override(precision, size):
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for(precision, size)
    config['input']['cameras'] = 2
    argv = runner.example_argv(config)
    assert argv[argv.index('--cameras') + 1] == '2'
    assert ('--vision-window' in argv) == (size == 224 and precision != 'w8a16_action_nvfp4')
    assert ('--vision-pair' in argv) == (size == 256)
    config['wall_oss']['vision_pair'] = False
    assert runner.example_argv(config) == [v for v in argv if v != '--vision-pair']
    config['wall_oss']['vision_pair'] = True
    if size == 256:
        assert runner.example_argv(config) == argv
    else:
        with pytest.raises(ValueError, match='requires 256x256'):
            runner.example_argv(config)
        config['wall_oss']['vision_pair'] = False
        config['wall_oss']['vision_window'] = True
        assert '--vision-window' in runner.example_argv(config)


@pytest.mark.parametrize('precision', ('fp16', 'w8a16', 'w8a16_action_nvfp4'))
@pytest.mark.parametrize('size', (224, 256))
def test_row_reuse_override_maps_to_existing_cold_component(precision, size):
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for(precision, size)
    config['wall_oss']['vision_row_reuse'] = False
    argv = runner.example_argv(config)
    assert argv[argv.index('--components') + 1] == 'small-prefill'
    assert '--vision-window' not in argv
    assert ('--vision-pair' in argv) == (size == 256)
    if size == 224:
        config['wall_oss']['vision_window'] = True
        with pytest.raises(ValueError, match='requires vision_row_reuse'):
            runner.example_argv(config)


@pytest.mark.parametrize('section,key,value', [
    ('wall_oss', 'precision', 'nvfp4a16'),
    ('wall_oss', 'linear_acc32', False),
    ('wall_oss', 'vision_window', 1),
    ('wall_oss', 'vision_window', 'true'),
    ('wall_oss', 'vision_window', None),
    ('wall_oss', 'vision_pair', 1),
    ('wall_oss', 'vision_pair', 'false'),
    ('wall_oss', 'vision_pair', None),
    ('wall_oss', 'vision_row_reuse', 0),
    ('wall_oss', 'vision_row_reuse', 'true'),
    ('wall_oss', 'vision_row_reuse', None),
    ('input', 'fp16_checkpoint', ''),
    ('input', 'w8_checkpoint', None),
    ('input', 'batch', 1),
    ('input', 'image_size', [252, 252]),
    ('input', 'image_size', [224, 256]),
    ('input', 'cameras', 0),
    ('input', 'cameras', 4),
    ('input', 'cameras', True),
    ('input', 'text_tokens', True),
    ('input', 'text_tokens', 0),
    ('input', 'text_length_scope', 'characters'),
    ('input', 'action_horizon', 10),
    ('input', 'num_steps', 6),
    ('run', 'warmup', 0),
    ('run', 'runs', True),
    ('run', 'runs', 31),
    ('run', 'cpu_threads', 0),
    ('run', 'cpu_threads', True),
    ('run', 'cpu_thread_sweep', [8]),
    ('run', 'cpu_thread_sweep', [4, 4]),
    ('run', 'cpu_thread_sweep', [True]),
    ('run', 'cpu_thread_sweep', [3]),
    ('run', 'cpu_thread_sweep', '4,12'),
    ('run', 'seed', -1),
    ('run', 'output_dir', ''),
    ('run', 'profile', True),
    ('runtime', 'main_ref', ''),
    ('runtime', 'env', {}),
])
def test_invalid_config_fails_before_engine_import(section, key, value, monkeypatch):
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for()
    config.setdefault(section, {})[key] = value
    monkeypatch.setattr(runner, 'validation_runner', lambda: pytest.fail('engine imported before admission'))
    with pytest.raises((TypeError, ValueError)):
        runner.run_example_config(config)


@pytest.mark.parametrize('execution', [
    {'model': {'num_cores': 5}},
    {'components': {'language_model': {'prefill': {'chunk_size': 384}}}},
    {'prefill': {'linear_acc32': 1}},
    {'vision': {'linear_acc32': 'true'}},
    {'action': {'linear_acc32': None}},
    {'vision': {'padding_rows': 16}},
    {'action': {'padding_budget': 0}},
    {'action': {'chunk_size': 16}},
])
def test_public_execution_parser_rejects_unsupported_controls(execution):
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for()
    config['rpu_execution'] = execution
    with pytest.raises((TypeError, ValueError)):
        runner.validate_example_config(config)


def test_common_dispatch_check_only_and_no_profile_inheritance(tmp_path, monkeypatch, capsys):
    import json
    common = load('examples/common/config.py')
    path = write_config(tmp_path / 'wall.toml')
    # Actual CLI logging creates its diagnostics directory before the stubbed
    # engine runs; weight/input paths intentionally remain unavailable.
    path.write_text(path.read_text().replace('"/missing/output"', json.dumps(str(tmp_path / 'results'))))
    config = common.load_config(path)
    assert common.example_metadata(config) is None
    legacy = common.load_config(ROOT / 'examples/configs/wall_oss/legacy/fp16.toml')
    assert common.example_metadata(legacy)['target'] == 'wall_oss'
    monkeypatch.setattr(sys, 'argv', ['wall_oss.py', '--config', str(path), '--check-config'])
    assert importlib.import_module("examples.common.cli").main() == 0
    assert 'syntax only' in capsys.readouterr().out
    calls = []
    runner = common._wall_base_runner()
    monkeypatch.setattr(runner, 'run_example_config', lambda cfg: calls.append(cfg) or 0)
    monkeypatch.setattr(importlib.import_module("examples.common.cli"), '_wall_base_runner', lambda: runner)
    monkeypatch.setattr(sys, 'argv', ['wall_oss.py', '--config', str(path)])
    assert importlib.import_module("examples.common.cli").main() == 0 and calls == [config]
    for flag in ('--hwperf', '--profile'):
        monkeypatch.setattr(sys, 'argv', ['wall_oss.py', '--config', str(path), flag])
        with pytest.raises(SystemExit) as caught:
            importlib.import_module("examples.common.cli").main()
        assert caught.value.code == 2
    assert len(calls) == 1


def test_check_config_has_no_torch_backend_engine_or_asset_dependency(tmp_path):
    path = write_config(tmp_path / 'wall.toml')
    code = '''
import importlib.abc
from pathlib import Path
import runpy
import sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'rpu_backend'}:
            raise AssertionError('check-config imported ' + fullname)
sys.meta_path.insert(0, NoRuntime())
root, config = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root / 'examples'))
sys.argv = ['wall_oss.py', '--config', config, '--check-config']
runpy.run_path(str(root / 'examples/run.py'), run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-S', '-c', code, str(ROOT), str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Wall-OSS; syntax only' in result.stdout
    assert not (tmp_path / 'result').exists()


@pytest.mark.parametrize('cameras', (2, 3))
@pytest.mark.parametrize('size', (224, 256))
@pytest.mark.parametrize('precision', ('fp16', 'w8a16', 'w8a16_action_nvfp4'))
def test_public_templates_admit_same_file_text_and_accumulation_changes(cameras, size, precision):
    """Public templates retain their input/weight routes when cold fields change."""
    runner = load('examples/runners/wall_oss/base.py')
    path = ROOT / f'examples/configs/wall_oss/base/{cameras}cam/{size}/{precision}.toml'
    config = load_example_toml(path)
    assert config['input']['text_tokens'] == 32
    for tokens in (32, 64, 96, 128, 160):
        config['input']['text_tokens'] = tokens
        if tokens != 32:
            # T32's exact prompt padding is not a valid default for another input.
            config['rpu_execution']['prefill'].pop('padding_rows', None)
            config['rpu_execution']['prefill'].pop('padding_budget', None)
        for acc32 in (False, True):
            for stage in ('vision', 'prefill', 'action'):
                config['rpu_execution'][stage]['linear_acc32'] = acc32
            before = deepcopy(config)
            argv = runner.example_argv(config)
            value = lambda flag: argv[argv.index(flag) + 1]
            assert value('--batch') == f'/path/to/{cameras}cam{size}_t{tokens}.pt'
            assert value('--text-tokens') == str(tokens)
            assert value('--cameras') == str(cameras)
            assert argv[argv.index('--image-size') + 1:argv.index('--image-size') + 3] == [str(size)] * 2
            assert value('--precision') == ('fp16' if precision == 'fp16' else 'w8')
            assert ('--w8' in argv) == (precision != 'fp16')
            assert ('--action-nvfp4' in argv) == (precision == 'w8a16_action_nvfp4')
            assert ('--action-residency' in argv) == (not acc32 and precision != 'w8a16_action_nvfp4')
            assert ('--vision-window' in argv) == config['wall_oss']['vision_window']
            assert ('--vision-pair' in argv) == config['wall_oss']['vision_pair']
            execution = runner.execution_config(config['rpu_execution'])
            assert all(execution[stage]['linear_acc32'] == acc32
                       for stage in ('vision', 'prefill', 'action'))
            assert config == before


def test_instruction_residency_is_optional_and_reaches_engine():
    runner = load('examples/runners/wall_oss/base.py')
    config = config_for('w8a16')
    assert '--instruction-residency' not in runner.example_argv(config)
    for mode in ('auto', 'off'):
        config['wall_oss']['instruction_residency'] = mode
        argv = runner.example_argv(config)
        assert argv[argv.index('--instruction-residency') + 1] == mode
    for invalid in (True, None, 'force', 1):
        config['wall_oss']['instruction_residency'] = invalid
        with pytest.raises(ValueError, match='instruction_residency'):
            runner.example_argv(config)

"""Public G05 examples dispatch actual cold precision without registry certification."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
from _example_toml import load_example_toml

ROOT = Path(__file__).resolve().parents[1]
EX = importlib.import_module("examples.runners.g05.optimized")
CONFIG_FILENAMES = {
    "fp16": "fp16", "w8a16": "w8a16", "w8a16_nvfp4": "w8a16_action_nvfp4",
}


@pytest.fixture
def preserve_cpu_state(monkeypatch):
    import torch
    previous = torch.get_num_threads()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, "11"))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_g05_example_precision_mapping_and_real_entry_checks():
    expected = {
        'fp16': ('fp16', 'acc32', None),
        'fp16_acc16': ('fp16', 'acc16', None),
        'w8a16_acc32': ('w8a16', 'acc32', None),
        'w8a16': ('w8a16', 'acc16', None),
        'w8a16_nvfp4': ('w8a16', 'acc16', 'nvfp4a16'),
        'w8a16_nvfp4_acc32': ('w8a16', 'acc32', 'nvfp4a16'),
    }
    assert set(EX._PRECISIONS) == set(expected)
    for precision in EX._PRECISIONS:
        opts = EX.precision_options(precision)
        assert (opts['weight_dtype'], opts['linear_accumulation'],
                opts.get('action_weight_dtype')) == expected[precision]
    paths = sorted((ROOT/'examples/configs/g05/2cam').glob('*.toml')) + sorted(
        (ROOT/'examples/configs/g05/3cam').glob('*.toml'))
    assert len(paths) == 6
    for path in paths:
        cfg = load_example_toml(path)
        EX.validate_example_config(cfg)
        assert EX.registered_example(cfg) is None
        assert cfg['g05']['precision'] in ('fp16', 'w8a16', 'w8a16_nvfp4')
        assert cfg['g05']['math_precision'] == 'base'
        assert cfg['input']['text_tokens'] == 32 and 'text_lengths' not in cfg['input']
        assert (cfg['run']['seed'], cfg['run']['warmup'], cfg['run']['runs'],
                cfg['input']['num_steps']) == (7, 2, 5, 10)
        result = subprocess.run([sys.executable, str(ROOT/'examples/run.py'), '--config', str(path), '--check-config'],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    cfg = load_example_toml((ROOT/'examples/configs/g05/3cam/w8a16_action_nvfp4.toml'))
    for section, key, value in [('g05','precision','w4a16'),('input','cameras',2),
                               ('input','num_steps',9),('run','runs',True),
                               ('input','text_tokens',True),('input','text_tokens',257),
                               ('input','source','tokens'),('input','source_step',True),('input','source_step',2),
                               ('input','instruction','')]:
        invalid = copy.deepcopy(cfg); invalid[section][key] = value
        with pytest.raises(ValueError): EX.validate_example_config(invalid)


def test_legacy_precision_and_batch_remain_compatible():
    assert EX.precision_options('w8a16_action_nvfp4') == EX.precision_options('w8a16_nvfp4')
    for alias, canonical in EX.PRECISION_ALIASES.items():
        assert EX.precision_options(alias) == EX.precision_options(canonical)
    for name in ('fp16', 'w8a16', 'w8a16_action_nvfp4'):
        path = ROOT/f'examples/configs/g05/3cam/{name}.toml'
        assert path.is_file()
        EX.validate_example_config(load_example_toml(path))
    cfg = load_example_toml((ROOT/'examples/configs/g05/3cam/w8a16_action_nvfp4.toml'))
    cfg['g05']['precision'] = 'w8a16_action_nvfp4'
    cfg.pop('libraries')
    cfg['input'] = {key: value for key, value in cfg['input'].items()
                    if key not in ('source', 'source_golden', 'source_step', 'text_tokens', 'text_lengths')}
    cfg['input']['batch'] = '/path/to/original.pt'
    EX.validate_example_config(cfg)
    cfg['input']['instruction'] = 'silently replace saved text'
    with pytest.raises(ValueError, match='saved batch'):
        EX.validate_example_config(cfg)


@pytest.mark.parametrize('precision', ['fp16', 'fp16_acc16', 'fp16_acc32'])
def test_batched_vision_requires_w8(precision):
    cfg = load_example_toml((ROOT/'examples/configs/g05/3cam/w8a16.toml'))
    cfg['g05'].pop('linear_accumulation')
    cfg['g05']['precision'] = precision
    cfg['g05']['vision_camera_batch'] = 3
    with pytest.raises(ValueError, match='W8A16 Vision'):
        EX.validate_example_config(cfg)


@pytest.mark.parametrize('cameras', [2, 3])
@pytest.mark.parametrize('precision', ['w8a16', 'w8a16_nvfp4'])
@pytest.mark.parametrize('accumulation', ['acc16', 'acc32'])
def test_grouped_vision_preserves_weight_and_accumulation_for_matching_cameras(cameras, precision, accumulation):
    cfg = load_example_toml((ROOT/f'examples/configs/g05/{cameras}cam/{CONFIG_FILENAMES[precision]}.toml'))
    cfg['g05'].update(vision_camera_batch=cameras, linear_accumulation=accumulation)
    EX.validate_example_config(cfg)
    actual = EX.precision_options(cfg['g05']['precision'], cfg['g05']['linear_accumulation'])
    assert actual['weight_dtype'] == 'w8a16' and actual['linear_accumulation'] == accumulation
    assert actual.get('action_weight_dtype') == ('nvfp4a16' if precision == 'w8a16_nvfp4' else None)
    cfg['g05']['vision_camera_batch'] = 5 - cameras
    with pytest.raises(ValueError, match='selected camera count'):
        EX.validate_example_config(cfg)
    cfg['g05']['vision_camera_batch'] = True
    with pytest.raises(ValueError, match='auto, 1, 2 or 3'):
        EX.validate_example_config(cfg)


@pytest.mark.parametrize('camera_batch', [1, 2, 3])
def test_report_reads_actual_direct_vision_plan_and_owns_snapshot(camera_batch):
    from types import SimpleNamespace
    actual = {'selected': {'physical_metadata': {'camera_batch_count': camera_batch}}}
    policy = SimpleNamespace(model=SimpleNamespace(
        vision_tower=SimpleNamespace(_rpu_last_execution_plan={'stale': True},
                                     _rpu_vision_last_a6_plan=actual),
        vlm=SimpleNamespace(_rpu_last_execution_plan={'chunk_size': 256}),
        action_expert=SimpleNamespace(_rpu_last_execution_plan={'chunk_size': 32})))
    saved = EX.execution_receipts(policy)
    assert saved['vision'] == actual and saved['text']['chunk_size'] == 256
    actual['selected']['physical_metadata']['camera_batch_count'] = 99
    assert saved['vision']['selected']['physical_metadata']['camera_batch_count'] == camera_batch


def test_public_entry_requires_explicit_config():
    result = subprocess.run([sys.executable, str(ROOT/'examples/run.py'), '--check-config'], capture_output=True, text=True)
    assert result.returncode == 2
    assert '--config' in result.stderr


@pytest.mark.parametrize('math_precision', ['base', 'high'])
@pytest.mark.parametrize('cameras,explicit_count', [(3, True), (2, True), (2, False)])
def test_new_example_consumes_mixed_precision_in_existing_release_loader(
        monkeypatch, tmp_path, math_precision, cameras, explicit_count):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import Mock
    from test_controlled_example_runners import module
    monkeypatch.syspath_prepend(str(ROOT/'examples'))
    cfg = load_example_toml((ROOT/'examples/configs/g05/3cam/w8a16_action_nvfp4.toml'))
    cfg['example'] = {'target': 'g05'}
    cfg['g05']['vision_camera_batch'] = 1
    cfg['g05']['math_precision'] = math_precision
    cfg['input']['camera_names'] = cfg['input']['camera_names'][:cameras]
    if explicit_count:
        cfg['input']['cameras'] = cameras
    else:
        cfg['input'].pop('cameras')
    image = SimpleNamespace(ndim=5, shape=(1, 1, 3, 224, 224))
    batch = {'samples': [{'sample': 1}], 'pixel_values': {str(i): image for i in range(cameras)}}
    monkeypatch.setattr(importlib.import_module('examples.runners.g05.runner'), '_load_mapping', lambda *a: batch)
    monkeypatch.setattr(importlib.import_module('examples.runners.g05.runner'), '_checkpoint', lambda _: tmp_path/'checkpoint')
    policy = Mock(_rpu_g05_fm_only_customer_release=True, num_input_images=3)
    policy.eval.return_value = policy
    policy.predict_action.return_value = {'action': 'result'}
    def install(actual, **kwargs):
        assert actual.num_input_images == cameras
    patch = Mock(side_effect=install)
    camera_batch = Mock()
    camera_profile = Mock(side_effect=lambda actual, processor, **kwargs:
                          setattr(actual, 'num_input_images', len(kwargs['camera_names'])))
    module(monkeypatch, 'rpu_backend.adapters.g05', patch_g05_policy_for_rpu=patch,
           configure_g05_vision_camera_batch=camera_batch,
           configure_g05_camera_profile=camera_profile)
    @contextmanager
    def transaction(root): yield lambda: None
    module(monkeypatch, 'rpu_backend.adapters.g05.release_replay', _checkpoint_manifest=lambda p: None,
           _release_import_transaction=transaction,
           _resolve_release_paths=lambda checkpoint, root: (tmp_path, checkpoint))
    processor = object()
    module(monkeypatch, 'vla_rtc.loader', __file__=str(tmp_path/'vla_rtc/loader.py'),
           load_for_eval=lambda *a, **kw: (policy, processor))
    runner = SimpleNamespace(source_config=cfg, owner=None,
                             _g05_patch_options=EX.precision_options(cfg['g05']['precision'], math_precision=math_precision))
    infer = importlib.import_module("examples.runners.g05.runner")._g05(runner)
    assert infer() == 'result'
    assert patch.call_args.kwargs == dict(rpu_execution=cfg['rpu_execution'],
        weight_dtype='w8a16', action_weight_dtype='nvfp4a16', linear_accumulation='acc16', math_precision=math_precision)
    assert runner.owner is policy
    camera_batch.assert_called_once_with(policy, 1)
    if cameras == 2:
        camera_profile.assert_called_once_with(
            policy, processor, camera_names=tuple(cfg['input']['camera_names']))
    else:
        camera_profile.assert_not_called()


@pytest.mark.parametrize('cameras,camera_batch', [(2, 1), (2, 2), (3, 1), (3, 3)])
def test_stable_graph_checks_actual_owners_and_replay_geometry(cameras, camera_batch):
    before = {name: {'size': 1, 'invariant': True, 'entries': [dict(
        signature=name, graph_lifetime_id=7, build_generation=1, kernel_count=20,
        segment_count=1, dma_count=2, barrier_count=3, replay_count=4, recapture_count=0)]}
        for name in ('vision', 'text', 'action')}
    after = copy.deepcopy(before)
    for name in before:
        after[name]['entries'][0]['replay_count'] += 6 * (cameras // camera_batch if name == 'vision' else 1)
    EX.require_replay(before, after, 6, camera_batch, cameras)
    for field, value in [('graph_lifetime_id', 8), ('build_generation', 2), ('recapture_count', 1),
                         ('kernel_count', 0), ('replay_count', 99)]:
        changed = copy.deepcopy(after)
        changed['text']['entries'][0][field] = value
        with pytest.raises(RuntimeError):
            EX.require_replay(before, changed, 6, camera_batch, cameras)


@pytest.mark.parametrize('reuse_output,close_error,lengths', [
    (False, False, None), (True, False, None), (False, True, None), (True, True, None),
    (False, False, [32, 64])])
def test_public_measurement_keeps_runtime_checks_and_timing_scopes_separate(
        monkeypatch, tmp_path, preserve_cpu_state, reuse_output, close_error, lengths):
    import json
    import torch
    from types import SimpleNamespace
    from test_controlled_example_runners import module

    monkeypatch.syspath_prepend(str(ROOT/'examples'))
    from examples.runners.g05 import runner as _controlled_runners
    from examples.common import measurement as _runner
    cfg = load_example_toml((ROOT/'examples/configs/g05/3cam/w8a16_action_nvfp4.toml'))
    cfg['input'].pop('text_lengths', None)
    cfg['input']['text_tokens'] = 52
    if lengths:
        cfg['input'].pop('text_tokens')
        cfg['input']['text_lengths'] = lengths
    cfg['g05']['vision_camera_batch'] = 1
    cfg['cpu']['threads'] = 3
    cfg['run'].update(output_dir=str(tmp_path), runs=2)
    monkeypatch.setattr(_runner, '_provenance', lambda _: {'test': True})
    events = []
    monkeypatch.setattr(EX, 'library_helpers', lambda: dict(
        validate_libraries=lambda _: None,
        bind_libraries=lambda _: ({}, {}),
        bind_cold_environment=lambda _: events.append('cold'),
        runtime_payloads=lambda _: events.append('identity') or {}))
    module(monkeypatch, 'rpu_backend')
    module(monkeypatch, 'rpu_backend.runtime.performance', _profile_ctx=None)

    class Cache:
        def __init__(self):
            self.entry = SimpleNamespace(signature='s', graph_lifetime_id=1, build_generation=1,
                replay_count=0, recapture_count=0, kernel_count=10, segment_count=1,
                dma_count=1, barrier_count=1, register_census_present=0,
                register_census_build_committed=0)
        def size(self): return 1
        def snapshot(self): return [self.entry]
        def cache_invariant_ok(self): return True
        def freeze(self): events.append('freeze')

    def close():
        events.append('close')
        if close_error:
            raise RuntimeError('retirement failed')
    shared = torch.zeros(1, 32, 27)
    def prepare(runner):
        assert events == ['cold', 'identity'] or events[-2:] == ['close', 'identity']
        assert torch.get_num_threads() == 3
        assert os.environ['OMP_NUM_THREADS'] == os.environ['MKL_NUM_THREADS'] == '3'
        assert runner._g05_patch_options['action_weight_dtype'] == 'nvfp4a16'
        caches = {name: Cache() for name in ('vision', 'text', 'action')}
        model = SimpleNamespace(
            vision_tower=SimpleNamespace(_rpu_vision_graph_cache=caches['vision']),
            vlm=SimpleNamespace(_rpu_qwen3_5=SimpleNamespace(prefill_graph_cache=caches['text']), forward=lambda: None),
            action_expert=SimpleNamespace(_rpu_qwen3_5=SimpleNamespace(action_graph_cache=caches['action'])),
            _forward_embed=lambda: None, inference_fm=lambda: None)
        runner.owner = SimpleNamespace(model=model, close=close)
        runner._g05_input_metadata = {'content_sha256': 'input',
                                     'text_tokens': runner.source_config['input']['text_tokens']}
        def infer():
            for name, cache in caches.items():
                cache.entry.replay_count += 3 if name == 'vision' else 1
            model._forward_embed(); model.vlm.forward(); model.inference_fm()
            output = torch.randn(1, 32, 27)
            if reuse_output:
                shared.copy_(output)
                return shared
            return output
        return infer
    monkeypatch.setattr(_controlled_runners, '_g05', prepare)
    # This fixture exercises policy measurements with in-process torch doubles;
    # the separate subprocess regression covers multi-row SDK lifetime isolation.
    monkeypatch.setattr(EX, '_run_row_process',
                        lambda row: EX._run_single_example_config(row, libraries={}))
    if reuse_output:
        with pytest.raises(RuntimeError, match='overwrote'):
            EX.run_example_config(cfg)
    elif close_error:
        with pytest.raises(RuntimeError, match='retirement failed'):
            EX.run_example_config(cfg)
    else:
        assert EX.run_example_config(cfg) == 0
    report = json.loads((tmp_path/'report.json').read_text())
    assert report['result_schema_version'] == 1
    assert report['execution']['status'] == ('FAILED' if reuse_output or close_error else 'COMPLETED')
    assert report['validation']['reference']['status'] == 'NOT_RUN'
    assert report['validation']['task']['status'] == 'NOT_EVALUATED'
    assert report['cpu_runtime']['requested_threads'] == 3
    assert report['cpu_runtime']['torch_num_threads'] == 3
    assert report['cpu_runtime']['OMP_NUM_THREADS'] == report['cpu_runtime']['MKL_NUM_THREADS'] == '3'
    if lengths:
        assert report['status'] == 'COMPLETE'
        assert list(report['rows']) == list(map(str, lengths))
        assert events.count('cold') == 1 and events.count('close') == len(lengths)
        for length in lengths:
            row = report['rows'][str(length)]
            assert row == json.loads((tmp_path/f'text_{length}/report.json').read_text())
            assert row['cpu_runtime'] == report['cpu_runtime']
            assert row['input']['text_tokens'] == length
            assert row['correctness'] == 'runtime_checks_only'
            assert row['graphs_final']['action']['entries'][0]['replay_count'] == 8
        report = report['rows'][str(lengths[-1])]
    assert events[-1] == 'close'
    if close_error:
        assert report['status'] == 'FAILED'
        assert report['cleanup_error'] == 'RuntimeError: retirement failed'
    assert len(report['samples']) == 2
    assert all(set(row) == {'total_ms'} for row in report['samples'])
    if reuse_output:
        assert report['status'] == 'FAILED'
        assert report['correctness'] == 'not_run'
    else:
        assert report['status'] == ('FAILED' if close_error else 'COMPLETE')
        assert report['correctness'] == 'runtime_checks_only'
        assert report['cpu_reference'] == 'not_run'
        assert report['aba']['previous_outputs_unchanged']
        assert len(report['phase_samples']) == 2
        assert report['graphs_final']['action']['entries'][0]['replay_count'] == 8


@pytest.mark.parametrize('precision', [*EX._PRECISIONS, *EX.PRECISION_ALIASES])
def test_explicit_accumulation_overrides_legacy_default_without_changing_weight_format(precision):
    original = EX.precision_options(precision)
    for accumulation in ('acc16', 'acc32'):
        actual = EX.precision_options(precision, accumulation)
        assert actual == dict(original, linear_accumulation=accumulation)
    assert EX.precision_options(precision) == original
    with pytest.raises(ValueError, match='linear_accumulation'):
        EX.precision_options(precision, 'fp32')


@pytest.mark.parametrize('cameras', [2, 3])
@pytest.mark.parametrize('precision', ['fp16', 'w8a16', 'w8a16_nvfp4'])
@pytest.mark.parametrize('accumulation', ['acc16', 'acc32'])
@pytest.mark.parametrize('math_precision', ['base', 'high'])
def test_math_precision_is_independent_of_weights_accumulation_and_lengths(cameras, precision, accumulation, math_precision):
    cfg = load_example_toml((ROOT/f'examples/configs/g05/{cameras}cam/{CONFIG_FILENAMES[precision]}.toml'))
    cfg['g05'].update(linear_accumulation=accumulation, math_precision=math_precision)
    EX.validate_example_config(cfg)
    opts = EX.precision_options(precision, accumulation, math_precision)
    assert opts == dict(EX.precision_options(precision, accumulation), math_precision=math_precision)
    for value in (None, True, 'HIGH', 'fp32'):
        cfg['g05']['math_precision'] = value
        with pytest.raises(ValueError, match='math_precision'):
            EX.validate_example_config(cfg)


@pytest.mark.parametrize('lengths', [[], [32, 32], [True], [31], [257], '32,64'])
def test_text_length_lists_reject_ambiguous_or_invalid_inputs(lengths):
    cfg = load_example_toml((ROOT/'examples/configs/g05/2cam/fp16.toml'))
    cfg['input'].pop('text_tokens')  # check list validation, not the exclusivity rule
    cfg['input']['text_lengths'] = lengths
    with pytest.raises(ValueError, match='text_lengths'):
        EX.validate_example_config(cfg)


def test_text_tokens_and_lengths_are_exclusive_and_saved_batch_has_one_context():
    cfg = load_example_toml((ROOT/'examples/configs/g05/2cam/fp16.toml'))
    cfg['input']['text_lengths'] = [32, 52, 64, 128, 192, 256]
    with pytest.raises(ValueError, match='mutually exclusive'):
        EX.validate_example_config(cfg)
    cfg['input'].pop('text_tokens')
    cfg['input']['instruction'] = 'one instruction'
    with pytest.raises(ValueError, match='multiple actual'):
        EX.validate_example_config(cfg)
    cfg['input'].pop('instruction')
    cfg['input'].pop('source_golden')
    cfg['input'].pop('source_step')
    cfg['input'].update(source='batch', batch='/path/to/saved.pt')
    with pytest.raises(ValueError, match='saved batch has one actual'):
        EX.validate_example_config(cfg)


@pytest.mark.parametrize('failure_mode', ['none', 'reported', 'abrupt'])
def test_multiple_actual_lengths_use_fresh_processes_and_preserve_failed_rows(
        monkeypatch, tmp_path, preserve_cpu_state, failure_mode):
    import json
    cfg = load_example_toml((ROOT/'examples/configs/g05/2cam/w8a16_action_nvfp4.toml'))
    cfg['g05']['linear_accumulation'] = 'acc32'
    cfg['input'].pop('text_tokens')
    cfg['input']['text_lengths'] = [32, 64, 128]
    cfg['run']['output_dir'] = str(tmp_path)
    events = []
    records = {'native': {'path': 'bound-native', 'sha256': 'same'}}
    def bind_cold(_):
        events.append('cold')
        monkeypatch.setenv('G05_TEST_COLD_BINDING', 'inherited')
    monkeypatch.setattr(EX, 'library_helpers', lambda: dict(
        validate_libraries=lambda _: None,
        bind_libraries=lambda _: ({'cold': 'binding'}, records),
        bind_cold_environment=bind_cold))
    # Real short-lived interpreters replace only the unavailable RPU policy.
    # Exercise JSON transport, inherited cold settings, exit and report handling.
    child = """
import json, os, sys
from pathlib import Path
row = json.load(sys.stdin)
path = Path(row['run']['output_dir'])
length = row['input']['text_tokens']
(path/'seen.json').write_text(json.dumps(dict(config=row, pid=os.getpid(),
    cold=os.environ['G05_TEST_COLD_BINDING'], threads=os.environ['OMP_NUM_THREADS'])))
if length == 64 and sys.argv[1] == 'abrupt':
    os._exit(17)
failed = length == 64 and sys.argv[1] == 'reported'
report = dict(status='FAILED' if failed else 'COMPLETE', input=dict(text_tokens=length))
if failed:
    report['error'] = 'original child failure'
(path/'report.json').write_text(json.dumps(report))
print('child row', length)
raise SystemExit(3 if failed else 0)
"""
    actual_run = subprocess.run
    def run_child(command, **kwargs):
        assert '_run_single_example_config' in command[-1]
        return actual_run([sys.executable, '-B', '-c', child, failure_mode], **kwargs)
    monkeypatch.setattr(EX.subprocess, 'run', run_child)
    # A stale success must not be inherited by an SDK abort before final report.
    stale = tmp_path / 'text_64'
    stale.mkdir()
    (stale/'report.json').write_text(json.dumps({'status': 'COMPLETE', 'stale': True}))
    if failure_mode != 'none':
        with pytest.raises(RuntimeError, match='G05 T64'):
            EX.run_example_config(cfg)
    else:
        assert EX.run_example_config(cfg) == 0
    report = json.loads((tmp_path/'report.json').read_text())
    expected = [32, 64] if failure_mode != 'none' else [32, 64, 128]
    assert list(report['rows']) == list(map(str, expected))
    assert report['status'] == ('FAILED' if failure_mode != 'none' else 'COMPLETE')
    assert events == ['cold']
    pids = set()
    for length in expected:
        seen = json.loads((tmp_path/f'text_{length}'/'seen.json').read_text())
        pids.add(seen['pid'])
        assert seen['cold'] == 'inherited' and seen['threads'] == '8'
        row = seen['config']
        assert 'text_lengths' not in row['input']
        assert row['input']['text_tokens'] == length
        assert row['input']['num_steps'] == 10
        assert row['g05']['linear_accumulation'] == 'acc32'
        assert (row['run']['warmup'], row['run']['runs'], row['run']['seed']) == (2, 5, 7)
        assert row['libraries'] == cfg['libraries']
        assert EX.input_camera_names(row['input']) == ('head_rgb', 'left_wrist_rgb')
    assert len(pids) == len(expected) and os.getpid() not in pids
    final = report['rows']['64']
    assert 'stale' not in final
    assert final['process_exit_code'] == {'none': 0, 'reported': 3, 'abrupt': 17}[failure_mode]
    if failure_mode != 'none':
        assert final['status'] == 'FAILED'
        assert not (tmp_path/'text_128').exists()
        if failure_mode == 'reported':
            assert final['error'] == 'original child failure'
        else:
            assert final['execution']['status'] == 'FAILED'
            assert final['artifacts']['outputs'] == []
    assert cfg['input']['text_lengths'] == [32, 64, 128]
    assert 'text_tokens' not in cfg['input']


def test_auto_schedule_selects_only_exact_measured_camera_mode_and_text(monkeypatch, tmp_path):
    import json
    cfg = load_example_toml((ROOT/'examples/configs/g05/2cam/w8a16_action_nvfp4.toml'))
    cfg['input'].pop('text_lengths', None)
    cfg['input']['text_tokens'] = 64
    cfg['g05']['linear_accumulation'] = 'acc32'
    profile = {'cameras': 2, 'camera_names': ['head_rgb', 'left_wrist_rgb'],
               'precision': 'w8a16_action_nvfp4_acc32', 'text_length': 64,
               'vision_camera_batch': 1, 'text_chunk': {'requested': 192}, 'selected_build': 'measured'}
    path = tmp_path/'selected.json'
    path.write_text(json.dumps({'schema_version': 2, 'profiles': [profile]}))
    resolved, selection = EX.resolve_row_config(cfg, path)
    assert selection['matched']
    assert selection['key']['precision'] == 'w8a16_action_nvfp4_acc32'
    assert resolved['g05']['vision_camera_batch'] == 1
    assert resolved['rpu_execution']['components']['language_model']['prefill']['chunk_size'] == 192
    assert resolved['libraries'] == cfg['libraries']
    assert cfg['g05']['vision_camera_batch'] == 'auto'
    assert cfg['rpu_execution']['components']['language_model']['prefill']['chunk_size'] == 'auto'
    high = copy.deepcopy(cfg)
    high['g05']['math_precision'] = 'high'
    high_resolved, high_selection = EX.resolve_row_config(high, path)
    assert high_resolved == dict(resolved, g05=dict(resolved['g05'], math_precision='high'))
    assert high_selection['key'] == selection['key']
    assert not high_selection['matched'] and high_selection['borrowed_base_schedule']
    assert high_selection['requested_math_precision'] == 'high'
    assert high_selection['schedule_source_math_precision'] == 'base'
    assert 'selected_build' not in high_selection
    assert high_selection['reason'] == 'HIGH borrows the BASE schedule; this catalog contains no measured HIGH optimum'
    for field, value in (('text_length', 128), ('precision', 'w8a16_action_nvfp4_acc16'), ('cameras', 3)):
        path.write_text(json.dumps({'schema_version': 2, 'profiles': [dict(profile, **{field: value})]}))
        resolved, selection = EX.resolve_row_config(cfg, path)
        assert not selection['matched']
        assert resolved['g05']['vision_camera_batch'] == 1
        assert resolved['rpu_execution']['components']['language_model']['prefill']['chunk_size'] == 'auto'
    path.write_text(json.dumps({'schema_version': 1, 'profiles': [profile]}))
    assert not EX.resolve_row_config(cfg, path)[1]['matched']


def test_explicit_schedules_win_and_mismatched_camera_groups_are_rejected(tmp_path):
    import json
    cfg = load_example_toml((ROOT/'examples/configs/g05/3cam/w8a16.toml'))
    cfg['input'].pop('text_lengths', None)
    cfg['input']['text_tokens'] = 64
    cfg['g05']['vision_camera_batch'] = 1
    cfg['rpu_execution']['components']['language_model']['prefill']['chunk_size'] = 128
    path = tmp_path/'selected.json'
    path.write_text(json.dumps({'schema_version': 2, 'profiles': [{
        'cameras': 3, 'camera_names': ['head_rgb', 'left_wrist_rgb', 'right_wrist_rgb'],
        'precision': 'w8a16_acc16', 'text_length': 64, 'vision_camera_batch': 3,
        'text_chunk': {'requested': 256}}]}))
    resolved, selection = EX.resolve_row_config(cfg, path)
    assert selection['matched'] and selection['applied_parameters'] == {}
    assert resolved['g05']['vision_camera_batch'] == 1
    assert resolved['rpu_execution']['components']['language_model']['prefill']['chunk_size'] == 128
    cfg['g05'].update(linear_accumulation='acc32', vision_camera_batch=3)
    EX.validate_example_config(cfg)
    cfg['input']['cameras'] = 2
    cfg['input']['camera_names'] = ['head_rgb', 'left_wrist_rgb']
    with pytest.raises(ValueError, match='selected camera count'):
        EX.validate_example_config(cfg)
    cfg['g05'].update(linear_accumulation='acc16', vision_camera_batch=2)
    EX.validate_example_config(cfg)
    cfg['g05']['precision'] = 'fp16'
    with pytest.raises(ValueError, match='W8A16 Vision'):
        EX.validate_example_config(cfg)


@pytest.mark.parametrize('cameras', [2, 3])
@pytest.mark.parametrize('precision', ['w8a16', 'w8a16_nvfp4'])
@pytest.mark.parametrize('accumulation', ['acc16', 'acc32'])
def test_auto_group_selection_requires_exact_mode_and_retains_explicit_library_binding(tmp_path, cameras, precision, accumulation):
    import json

    cfg = load_example_toml((ROOT/f'examples/configs/g05/{cameras}cam/{CONFIG_FILENAMES[precision]}.toml'))
    cfg['input'].pop('text_lengths', None)
    cfg['input']['text_tokens'] = 128
    cfg['g05']['linear_accumulation'] = accumulation
    base = 'w8a16_action_nvfp4' if precision == 'w8a16_nvfp4' else precision
    profile = {'cameras': cameras, 'camera_names': cfg['input']['camera_names'],
               'precision': base + '_' + accumulation, 'text_length': 128,
               'vision_camera_batch': cameras, 'text_chunk': {'requested': 192},
               'selected_build': 'HOST_FIXTURE_ONLY_NOT_MEASURED'}
    path = tmp_path/'selected.json'
    path.write_text(json.dumps({'schema_version': 2, 'profiles': [profile]}))
    resolved, selection = EX.resolve_row_config(cfg, path)
    assert selection['matched'] and resolved['g05']['vision_camera_batch'] == cameras
    assert resolved['g05']['linear_accumulation'] == accumulation
    assert resolved['rpu_execution']['components']['language_model']['prefill']['chunk_size'] == 192
    assert resolved['libraries'] == cfg['libraries']
    assert cfg['g05']['vision_camera_batch'] == 'auto'
    path.write_text(json.dumps({'schema_version': 2, 'profiles': [dict(profile,
        precision=base + ('_acc32' if accumulation == 'acc16' else '_acc16'))]}))
    resolved, selection = EX.resolve_row_config(cfg, path)
    assert not selection['matched'] and resolved['g05']['vision_camera_batch'] == 1
    assert resolved['rpu_execution']['components']['language_model']['prefill']['chunk_size'] == 'auto'
    assert resolved['libraries'] == cfg['libraries']
    path.write_text(json.dumps({'schema_version': 2, 'profiles': [dict(profile, vision_camera_batch=5 - cameras)]}))
    with pytest.raises(ValueError, match='selected camera count'):
        EX.resolve_row_config(cfg, path)

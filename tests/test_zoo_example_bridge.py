from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_examples
import run_zoo_example


@pytest.fixture(scope='module')
def catalog():
    return export_examples.export_catalog()



def test_all_templates_roundtrip_to_actual_validator(catalog, tmp_path):
    for row in catalog['entries']:
        text = run_zoo_example.toml_text(row['config'])
        assert tomllib.loads(text) == row['config']
        path = tmp_path / 'config.toml'
        path.write_text(text)
        export_examples.COMMON.load_config(path)
    assert catalog['duplicates'] == [['qwen3_tts/0_6b/base', 'qwen3_tts/0_6b/base_rpu_codec']]


def test_catalog_rejects_runtime_release_identity_drift(catalog, monkeypatch):
    release_file = ROOT / 'release/runtime-dependencies.json'
    original_sha = export_examples.file_sha
    assert catalog['source']['files']['VERSION'] == original_sha(ROOT / 'VERSION')
    assert catalog['source']['files']['release/version.cmake'] == original_sha(ROOT / 'release/version.cmake')
    assert catalog['source']['files'][release_file.relative_to(ROOT).as_posix()] == original_sha(release_file)
    monkeypatch.setattr(export_examples, 'file_sha',
                        lambda path: '0' * 64 if path == release_file else original_sha(path))
    with pytest.raises(ValueError, match='source changed: release/runtime-dependencies.json'):
        export_examples.verify_source(catalog)



def test_run_bundle_rejects_stale_source_and_changed_template(catalog):
    row = next(r for r in catalog['entries'] if r['id'] == 'qwen3/text/1_7b/fp16')
    bundle = {'schema': 'rpu-model-zoo/example-run-bundle/v1', 'source': catalog['source'],
              'configs': [copy.deepcopy(row)], 'registry_paths': catalog['registry_paths']}
    _, config = run_zoo_example.resolve(bundle, row['id'])
    assert config == row['config']
    bundle['configs'][0]['config']['rpu_execution']['prefill']['chunk_size'] = 16
    with pytest.raises(ValueError, match='identity'):
        run_zoo_example.resolve(bundle, row['id'])
    bundle['configs'][0] = copy.deepcopy(row)
    bundle['source'] = dict(catalog['source'], content_sha256='0'*64)
    with pytest.raises(ValueError, match='source'):
        run_zoo_example.resolve(bundle, row['id'])



def prepared_bundle(catalog, overrides):
    row = copy.deepcopy(next(r for r in catalog['entries']
                             if r['id'] == 'qwen3_vl/vl/4b/w8a16_text_head_vision'))
    row['overrides'] = {'input': overrides}
    return {'schema': 'rpu-model-zoo/example-run-bundle/v1',
            'source': catalog['source'], 'configs': [row],
            'registry_paths': catalog['registry_paths']}


def test_prepared_input_bundle_removes_explicit_defaults_without_changing_template(catalog, tmp_path):
    path = tmp_path / 'prepared.safetensors'
    path.write_bytes(b'caller file identity checked separately from tensor validation')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    overrides = {'prepared_input_safetensors': str(path), 'prepared_input_sha256': digest,
                 'prompt': None, 'images': None, 'prefill_tokens': None, 'decode_steps': 4}
    bundle = prepared_bundle(catalog, overrides)
    original = copy.deepcopy(bundle)
    _, config = run_zoo_example.resolve(bundle, bundle['configs'][0]['id'])
    assert config['input'] == {'component': 'multimodal', 'prepared_input_safetensors': str(path),
                               'prepared_input_sha256': digest, 'decode_steps': 4, 'stop_on_eos': False}
    assert bundle == original
    assert run_zoo_example.physical_input_files(config['input']) == [
        {'field': 'prepared_input_safetensors', 'bytes': path.stat().st_size, 'sha256': digest}]
    result = run_zoo_example.run(bundle, bundle['configs'][0]['id'],
                                output=tmp_path / 'unused', check=True)
    assert result['status'] == 'configuration-valid'
    assert result['hardware'] == 'not-run'
    assert bundle == original


@pytest.mark.parametrize('overrides', [
    {'prompt': None},
    {'prepared_input_safetensors': '/caller/input.safetensors', 'prompt': None},
    {'prepared_input_sha256': 'a' * 64, 'images': None},
    {'prepared_input_safetensors': '/caller/input.safetensors', 'prepared_input_sha256': 'bad',
     'prefill_tokens': None},
    {'prepared_input_safetensors': '/caller/input.safetensors', 'prepared_input_sha256': 'a' * 64,
     'decode_steps': None},
    {'prepared_input_safetensors': '/caller/input.safetensors', 'prepared_input_sha256': 'a' * 64,
     'prompt_token_ids': None},
])
def test_input_null_requires_paired_prepared_input_and_only_three_default_fields(catalog, overrides):
    bundle = prepared_bundle(catalog, overrides)
    with pytest.raises(ValueError, match='null input overrides'):
        run_zoo_example.resolve(bundle, bundle['configs'][0]['id'])


def test_prepared_bundle_retains_reader_exclusive_input_contract(catalog, tmp_path):
    overrides = {'prepared_input_safetensors': '/caller/input.safetensors',
                 'prepared_input_sha256': 'a' * 64, 'images': None, 'prefill_tokens': None,
                 'prompt': 'Still a competing prompt.'}
    bundle = prepared_bundle(catalog, overrides)
    with pytest.raises(ValueError, match='cannot also specify prompt/images'):
        run_zoo_example.run(bundle, bundle['configs'][0]['id'],
                            output=tmp_path / 'unused', check=True)


def test_paired_prepared_null_deletion_rejects_other_model_and_component(catalog):
    for config_id in ('qwen3/text/1_7b/fp16', 'qwen3_vl/text/4b/w8a16_text_head_vision'):
        row = copy.deepcopy(next(r for r in catalog['entries'] if r['id'] == config_id))
        row['overrides'] = {'input': {'prepared_input_safetensors': '/caller/input.safetensors',
                                      'prepared_input_sha256': 'a' * 64, 'prompt': None}}
        bundle = {'schema': 'rpu-model-zoo/example-run-bundle/v1', 'source': catalog['source'],
                  'configs': [row], 'registry_paths': catalog['registry_paths']}
        with pytest.raises(ValueError, match='null input overrides'):
            run_zoo_example.resolve(bundle, config_id)


def test_raw_image_identity_requires_files_but_frozen_input_does_not_read_images(tmp_path):
    import hashlib
    missing = tmp_path / 'missing-image.png'
    with pytest.raises(ValueError, match='images\\[0\\].*physical input'):
        run_zoo_example.physical_input_files({'images': [str(missing)]})
    for empty in ('', None):
        with pytest.raises(ValueError, match='frozen_input.*physical input'):
            run_zoo_example.physical_input_files({'images': [str(missing)], 'frozen_input': empty})
    with pytest.raises(ValueError, match='frozen_input.*physical input'):
        run_zoo_example.physical_input_files({'frozen_input': str(missing)})
    frozen = tmp_path / 'input.safetensors'
    frozen.write_bytes(b'exact preprocessed tensor bytes')
    assert run_zoo_example.physical_input_files({'images': [str(missing)], 'frozen_input': str(frozen)}) == [
        {'field': 'frozen_input', 'bytes': frozen.stat().st_size,
         'sha256': hashlib.sha256(frozen.read_bytes()).hexdigest()}]


def test_observation_binds_exact_request_config_report_and_tensor(tmp_path):
    observation = tmp_path / 'observation'
    observation.mkdir()
    request = {'effective_config_sha256': 'a' * 64, 'observer': {'sha256': 'b' * 64}}
    request_path = tmp_path / 'zoo-request.json'
    request_path.write_text(json.dumps(request))
    raw_path = tmp_path / 'report.json'
    raw = {'wall_ms_runs': [1., 2.], 'config': {'run': {'runs': 2}}}
    raw_path.write_text(json.dumps(raw))
    tensor = observation / 'outputs.safetensors'
    tensor.write_bytes(b'caller-owned output snapshot')
    data = {
        'request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
        'effective_config_sha256': request['effective_config_sha256'],
        'raw_report_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'observer_sha256': request['observer']['sha256'],
        'artifacts': [{'path': tensor.name, 'bytes': tensor.stat().st_size,
                       'sha256': hashlib.sha256(tensor.read_bytes()).hexdigest()}],
        'audit': {'output_sha256_runs': ['c' * 64, 'c' * 64]},
    }
    path = observation / 'observation.json'
    path.write_text(json.dumps(data))
    merged, _ = run_zoo_example.observed_report(tmp_path, request, raw_path, raw)
    assert merged['wall_ms_runs'] == raw['wall_ms_runs']
    assert merged['config'] == raw['config']
    for field in ('request_sha256', 'effective_config_sha256', 'raw_report_sha256', 'observer_sha256'):
        bad = dict(data, **{field: 'd' * 64})
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='bound'):
            run_zoo_example.observed_report(tmp_path, request, raw_path, raw)
    path.write_text(json.dumps(dict(data, audit={'wall_ms_runs': [0.1, 0.1]})))
    with pytest.raises(ValueError, match='untimed'):
        run_zoo_example.observed_report(tmp_path, request, raw_path, raw)
    path.write_text(json.dumps(data))
    tensor.write_bytes(b'changed output snapshot')
    with pytest.raises(ValueError, match='artifact identity'):
        run_zoo_example.observed_report(tmp_path, request, raw_path, raw)


def test_graph_evidence_rejects_empty_and_recaptured_cache():
    with pytest.raises(ValueError, match='Graph observations'):
        run_zoo_example.graph_evidence({'graph_before_measurement': {}, 'graph_measured_runs': [{}]}, 1)
    before = {'owner/cache': {'size': 1, 'invariant_ok': True, 'entries': [{
        'graph_lifetime_id': 1, 'build_generation': 1, 'recapture_count': 0,
        'replay_count': 2, 'kernel_count': 1, 'child_graph_count': 0, 'non_replayable_reason': '',
    }]}}
    after = copy.deepcopy(before)
    after['owner/cache']['entries'][0]['replay_count'] = 3
    raw = {'graph_before_measurement': before, 'graph_measured_runs': [after],
           'graph_observation': {'scope': 'post-timer snapshots', 'retained_replay_observed': True}}
    assert run_zoo_example.graph_evidence(raw, 1)['retained_replay_observed']
    after['owner/cache']['entries'][0]['recapture_count'] = 1
    with pytest.raises(ValueError, match='recaptured'):
        run_zoo_example.graph_evidence(raw, 1)


def test_accumulation_override_preserves_schedule_and_rejects_new_scope():
    original = {'rpu_execution': {'prefill': {'chunk_size': 'auto', 'linear_acc32': False}}}
    changed = copy.deepcopy(original)
    run_zoo_example.execution_overrides(changed, {'prefill': {'linear_acc32': True}})
    assert changed['rpu_execution']['prefill'] == {'chunk_size': 'auto', 'linear_acc32': True}
    assert original['rpu_execution']['prefill']['linear_acc32'] is False
    for override in ({'prefill': {'chunk_size': 15}}, {'prefill': {'linear_acc32': 1}},
                     {'vision': {'linear_acc32': True}}):
        with pytest.raises(ValueError, match='override|overridden'):
            run_zoo_example.execution_overrides(copy.deepcopy(original), override)


def test_public_cold_choices_and_input_dependent_padding_use_actual_validator(catalog, tmp_path):
    row = next(r for r in catalog['entries'] if r['id'] == 'qwen3/text/1_7b/fp16')
    config = copy.deepcopy(row['config'])
    run_zoo_example.execution_overrides(config, {
        'model': {'num_cores': 4}, 'prefill': {'chunk_size': 32}})
    path = tmp_path / 'config.toml'
    path.write_text(run_zoo_example.toml_text(config))
    validated = export_examples.COMMON.load_config(path)
    assert validated['rpu_execution']['model']['num_cores'] == 4
    assert validated['rpu_execution']['prefill']['chunk_size'] == 32
    for override in ({'model': {'num_cores': True}}, {'model': {'num_cores': 5}},
                     {'prefill': {'chunk_size': True}}, {'prefill': {'chunk_size': 0}}):
        with pytest.raises(ValueError, match='override'):
            run_zoo_example.execution_overrides(copy.deepcopy(row['config']), override)
    wall = next(r for r in catalog['entries'] if r['id'] == 'wall_oss/base/2cam/256/fp16')
    changed = copy.deepcopy(wall['config'])
    assert 'padding_rows' in changed['rpu_execution']['prefill']
    changed['input']['text_tokens'] = 64
    run_zoo_example.execution_overrides(changed, {'prefill': {'padding_rows': None}})
    path.write_text(run_zoo_example.toml_text(changed))
    assert 'padding_rows' not in export_examples.COMMON.load_config(path)['rpu_execution']['prefill']
    with pytest.raises(ValueError, match='existing template pin'):
        run_zoo_example.execution_overrides(changed, {'prefill': {'padding_rows': None}})


def test_source_manifest_covers_native_includes(catalog):
    includes = {str(p.relative_to(ROOT)) for p in (ROOT / 'src').rglob('*.inc')}
    assert includes and includes <= set(catalog['source']['files'])


def test_legacy_g05_release_is_bound_to_checkpoint_parent(catalog, tmp_path):
    row = copy.deepcopy(next(r for r in catalog['entries'] if r['id'] == 'g05/legacy/fp16'))
    root = tmp_path / 'release' / 'checkpoint'
    root.mkdir(parents=True)
    row.update(asset_root=str(root), release_root=str(root.parent),
               overrides={'input': {'checkpoint': str(root), 'release_root': str(root.parent)}})
    bundle = {'schema': 'rpu-model-zoo/example-run-bundle/v1', 'source': catalog['source'],
              'configs': [row], 'registry_paths': catalog['registry_paths']}
    _, config = run_zoo_example.resolve(bundle, row['id'])
    assert config['input']['release_root'] == str(root.parent)
    row['overrides']['input']['release_root'] = str(tmp_path)
    with pytest.raises(ValueError, match='bound complete G05 release parent'):
        run_zoo_example.resolve(bundle, row['id'])


def test_sorting_primary_and_multiple_auxiliary_weights_are_bound(catalog, tmp_path):
    config = copy.deepcopy(next(r for r in catalog['entries']
                               if r['id'] == 'wall_oss/sorting/w8a16_action_w4a16_g32')['config'])
    assert run_zoo_example.primary_checkpoint(config) == 'w8_checkpoint'
    auxiliary = []
    for key, text in [('fp16_checkpoint', b'original FP16'), ('action_w4_checkpoint', b'packed action W4')]:
        root = tmp_path / key
        root.mkdir()
        file = root / 'model.safetensors'
        file.write_bytes(text)
        config['input'][key] = str(root)
        auxiliary.append({'model_id': key, 'variant_id': 'host-fixture', 'asset_root': str(root),
                          'asset_manifest_sha256': 'a' * 64, 'asset_files': [{
                              'path': file.name, 'size': len(text), 'sha256': hashlib.sha256(text).hexdigest()}]})
    row = {'auxiliary_assets': auxiliary}
    verified = run_zoo_example.verify_auxiliary_assets(row, config, {}, None, None, 'w8_checkpoint')
    assert [v['model_id'] for v in verified] == ['fp16_checkpoint', 'action_w4_checkpoint']
    with pytest.raises(ValueError, match='ambiguous'):
        run_zoo_example.verify_auxiliary_assets({'auxiliary_assets': [auxiliary[0], auxiliary[0]]},
                                                config, {}, None, None, 'w8_checkpoint')
    config['input']['action_w4_checkpoint'] = str(tmp_path / 'unbound')
    with pytest.raises(ValueError, match='unused|differs'):
        run_zoo_example.verify_auxiliary_assets(row, config, {}, None, None, 'w8_checkpoint')


def test_wall_rtc_w8_requires_exact_fp16_auxiliary(catalog, tmp_path):
    config = copy.deepcopy(next(r for r in catalog['entries']
                               if r['id'] == 'wall_oss/rtc/w8a16')['config'])
    assert export_examples.COMMON.config_target(config) == 'wall_oss_rtc'
    primary = run_zoo_example.primary_checkpoint(config)
    assert primary != 'fp16_checkpoint'
    root = tmp_path / 'fp16'
    root.mkdir()
    file = root / 'model.safetensors'
    payload = b'exact RTC FP16 parent fixture'
    file.write_bytes(payload)
    config['input']['fp16_checkpoint'] = str(root)
    auxiliary = {'model_id': 'wall-oss-0.5', 'variant_id': 'upstream',
                 'asset_root': str(root), 'asset_manifest_sha256': 'a' * 64,
                 'asset_files': [{'path': file.name, 'size': len(payload),
                                  'sha256': hashlib.sha256(payload).hexdigest()}]}
    row = {'auxiliary_assets': [auxiliary]}
    verified = run_zoo_example.verify_auxiliary_assets(row, config, {}, None, None, primary)
    assert verified[0]['weight_verification'][0]['sha256'] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match='separately bound'):
        run_zoo_example.verify_auxiliary_assets({}, config, {}, None, None, primary)
    unbound = copy.deepcopy(config)
    unbound['input']['fp16_checkpoint'] = str(tmp_path / 'different-parent')
    with pytest.raises(ValueError, match='unused|differs'):
        run_zoo_example.verify_auxiliary_assets(row, unbound, {}, None, None, primary)
    with pytest.raises(ValueError, match='ambiguous'):
        run_zoo_example.verify_auxiliary_assets(
            {'auxiliary_assets': [auxiliary, auxiliary]}, config, {}, None, None, primary)
    fp16 = copy.deepcopy(next(r for r in catalog['entries']
                             if r['id'] == 'wall_oss/rtc/fp16')['config'])
    with pytest.raises(ValueError, match='does not require'):
        run_zoo_example.verify_auxiliary_assets(row, fp16, {}, None, None,
                                                run_zoo_example.primary_checkpoint(fp16))
    unknown = copy.deepcopy(config)
    unknown['example']['target'] = 'unsupported_auxiliary_target'
    with pytest.raises(ValueError, match='separately bound'):
        run_zoo_example.verify_auxiliary_assets(row, unknown, {}, None, None, primary)


def test_gr00t_bound_vision_auxiliary_is_admitted(catalog):
    row = copy.deepcopy(next(r for r in catalog['entries'] if r['id'] == 'gr00t/n1_7_droid/224/fp16'))
    row['auxiliary_assets'] = [{'registry_alias': 'qwen3-vl-2b'}]
    bundle = {'schema': 'rpu-model-zoo/example-run-bundle/v1', 'source': catalog['source'],
              'configs': [row], 'registry_paths': catalog['registry_paths']}
    _, config = run_zoo_example.resolve(bundle, row['id'])
    assert config['input']['qwen3vl_registry_alias'] == 'qwen3-vl-2b'


@pytest.mark.parametrize('config_id', ['wall_oss/legacy/w4a16_channel', 'wall_oss/legacy/w4a16_g32'])
def test_legacy_wall_w4_binds_implicit_w8_vision(catalog, tmp_path, config_id):
    row = copy.deepcopy(next(r for r in catalog['entries'] if r['id'] == config_id))
    auxiliary = []
    cache = tmp_path / 'cache'
    cache.mkdir()
    for alias in ('wall-oss-0.5', 'wall-oss-0.5-w8a16'):
        root = tmp_path / alias
        root.mkdir()
        payload = root / 'config.json'
        payload.write_bytes(alias.encode())
        consumed = cache / catalog['registry_paths'][alias]
        consumed.parent.mkdir(parents=True, exist_ok=True)
        consumed.symlink_to(root, target_is_directory=True)
        auxiliary.append({'model_id': 'wall-oss-0.5', 'variant_id': alias,
                          'registry_alias': alias, 'asset_root': str(root),
                          'asset_manifest_sha256': 'a' * 64,
                          'asset_files': [{'path': payload.name, 'size': payload.stat().st_size,
                                           'sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}]})
    row['auxiliary_assets'] = auxiliary
    bundle = {'schema': 'rpu-model-zoo/example-run-bundle/v1', 'source': catalog['source'],
              'configs': [row], 'registry_paths': catalog['registry_paths']}
    _, config = run_zoo_example.resolve(bundle, config_id)
    verified = run_zoo_example.verify_auxiliary_assets(
        row, config, catalog['registry_paths'], None, cache, 'checkpoint')
    assert [item['registry_alias'] for item in verified] == ['wall-oss-0.5', 'wall-oss-0.5-w8a16']
    with pytest.raises(ValueError, match='differs'):
        run_zoo_example.verify_auxiliary_assets(
            {'auxiliary_assets': auxiliary[:1]}, config, catalog['registry_paths'], None, cache, 'checkpoint')
    ordinary = copy.deepcopy(config)
    ordinary['example']['dtype'] = 'w8a16'
    with pytest.raises(ValueError, match='unused'):
        run_zoo_example.verify_auxiliary_assets(
            row, ordinary, catalog['registry_paths'], None, cache, 'checkpoint')
    (tmp_path / 'wall-oss-0.5-w8a16/config.json').write_bytes(b'changed vision payload')
    with pytest.raises(ValueError, match='size|digest'):
        run_zoo_example.verify_auxiliary_assets(
            row, config, catalog['registry_paths'], None, cache, 'checkpoint')


@pytest.mark.parametrize('samples', [[200., 201.], [200., 0.]])
def test_sorting_report_uses_actual_e2e_samples_and_rejects_invalid_timing(tmp_path, samples):
    entry = {'graph_lifetime_id': 1, 'build_generation': 1, 'recapture_count': 0,
             'replay_count': 2, 'kernel_count': 1, 'child_graph_count': 0,
             'non_replayable_reason': ''}
    before = {'owner/cache': {'size': 1, 'invariant_ok': True, 'entries': [entry]}}
    after = [copy.deepcopy(before), copy.deepcopy(before)]
    for index, graph in enumerate(after):
        graph['owner/cache']['entries'][0]['replay_count'] = 3 + index
    request = {'backend_source_sha256': 'a' * 64, 'asset_manifest_sha256': 'c' * 64,
               'config': {'input': {}, 'run': {'warmup': 2, 'runs': 2}}}
    (tmp_path / 'zoo-request.json').write_text(json.dumps(request))
    raw = {'e2e': {'samples_ms': samples}, 'execution': {'status': 'COMPLETED'},
           'validation': {'runtime': {'status': 'CHECKED'}},
           'stable_output': True, 'stable_graphs': True, 'output_sha256_runs': ['b' * 64] * 2,
           'graph_before_measurement': before, 'graph_measured_runs': after,
           'graph_observation': {'scope': 'post-timer snapshots', 'retained_replay_observed': True},
           'stability_contract': 'integer predictions stable',
           'environment': {'backend_source_sha256': 'a' * 64,
                           'native_libraries': [{'sha256': 'c' * 64}],
                           'python_modules': {'examples': {'sha256': 'd' * 64}}}}
    (tmp_path / 'report.json').write_text(json.dumps(raw))
    row = {'id': 'wall_oss/sorting/fp16', 'case_id': 'e' * 64,
           'config_sha256': 'f' * 64}
    if samples[-1] == 0.:
        with pytest.raises(ValueError, match='qualified'):
            run_zoo_example.performance_record(tmp_path, row, request)
    else:
        path = Path(run_zoo_example.performance_record(tmp_path, row, request))
        record = json.loads(path.read_text())
        report = json.loads((path.parent / record['artifacts']['report']['path']).read_text())
        assert report['request_ms_samples'] == samples
        assert report['raw_example_report'] == raw


@pytest.fixture
def partial_g05_run(tmp_path):
    from examples.common.results import add_result_fields

    output = tmp_path / 'run'
    root = output / 'example'
    directory = root / 'text_32'
    directory.mkdir(parents=True)
    for length in (32, 52):
        (tmp_path / f'batch_{length}.pt').write_bytes(f'caller batch T{length}'.encode())
    observer = tmp_path / 'observer.py'
    observer.write_text('# caller observation fixture\n')
    request = {'backend_source_sha256': 'a' * 64, 'asset_manifest_sha256': 'c' * 64,
               'observer': {'entry_point': str(observer), 'sha256': export_examples.file_sha(observer)},
               'config': {'g05': {}, 'input': {'text_lengths': [32, 52], 'cameras': 2,
                           'image_size': [224, 224],
                           'batch_template': str(tmp_path / 'batch_{text_tokens}.pt')},
                          'run': {'warmup': 2, 'runs': 2, 'output_dir': str(root)}}}
    request.update(run_zoo_example.capture_input_verification(request['config']))
    request_path = output / 'zoo-request.json'
    request_path.write_text(json.dumps(request))
    child_config = copy.deepcopy(request['config'])
    child_config['input'].pop('text_lengths')
    child_config['input']['text_tokens'] = 32
    child_config['run']['output_dir'] = str(directory)
    raw = {'status': 'COMPLETE', 'requested_config': child_config, 'wall_ms_runs': [200., 201.]}
    add_result_fields(raw, directory, completed=True, runtime_checks=(
        'finite action output shape', 'Graph REPLAY/no recapture'))
    report = directory / 'report.json'
    report.write_text(json.dumps(raw))
    (root / 'report.json').write_text(json.dumps({'status': 'FAILED', 'rows': {'32': raw}}))
    entry = {'graph_lifetime_id': 1, 'build_generation': 1, 'recapture_count': 0,
             'replay_count': 2, 'kernel_count': 1, 'child_graph_count': 0,
             'non_replayable_reason': ''}
    before = {'owner/cache': {'size': 1, 'invariant_ok': True, 'entries': [entry]}}
    after = [copy.deepcopy(before), copy.deepcopy(before)]
    for index, graph in enumerate(after):
        graph['owner/cache']['entries'][0]['replay_count'] = 3 + index
    observation_dir = output / 'observation' / 'text_32'
    observation_dir.mkdir(parents=True)
    tensor = observation_dir / 'outputs.safetensors'
    tensor.write_bytes(b'caller-owned output snapshot')
    observation = {'request_sha256': export_examples.file_sha(request_path),
                   'effective_config_sha256': export_examples.identity(
                       export_examples.semantic_config(child_config)),
                   'raw_report_sha256': export_examples.file_sha(report),
                   'observer_sha256': request['observer']['sha256'],
                   'artifacts': [{'path': tensor.name, 'bytes': tensor.stat().st_size,
                                  'sha256': export_examples.file_sha(tensor)}],
                   'audit': {'stable_output': True, 'stable_graphs': True,
                             'output_sha256_runs': ['b' * 64] * 2,
                             'graph_before_measurement': before, 'graph_measured_runs': after,
                             'graph_observation': {'scope': 'post-timer snapshots',
                                                   'retained_replay_observed': True},
                             'stability_contract': 'integer predictions stable',
                             'environment': {'backend_source_sha256': 'a' * 64,
                                             'native_libraries': [{'sha256': 'c' * 64}],
                                             'python_modules': {'examples': {'sha256': 'd' * 64}}}}}
    observation_path = observation_dir / 'observation.json'
    observation_path.write_text(json.dumps(observation))
    row = {'id': 'g05/2cam/w8a16', 'case_id': 'e' * 64, 'config_sha256': 'f' * 64}
    return output, row, request, report, observation_path


@pytest.mark.parametrize('schema', ['modern', 'legacy'])
def test_partial_g05_preserves_completed_condition_with_qualified_evidence(partial_g05_run, schema):
    output, row, request, report, observation_path = partial_g05_run
    raw = json.loads(report.read_text())
    if schema == 'legacy':
        raw.pop('execution')
        raw.pop('validation')
        raw['runtime_checks'] = 'passed'
        report.write_text(json.dumps(raw))
        observation = json.loads(observation_path.read_text())
        observation['raw_report_sha256'] = export_examples.file_sha(report)
        observation_path.write_text(json.dumps(observation))
    records = run_zoo_example.performance_records(output, row, request, allow_partial=True)
    assert len(records) == 1
    path = Path(records[0])
    record = json.loads(path.read_text())
    assert record['record']['status'] == 'measured'
    recorded = json.loads((path.parent / record['artifacts']['report']['path']).read_text())
    assert recorded['raw_example_report'] == raw
    child = json.loads((path.parent / record['artifacts']['request']['path']).read_text())
    assert child['config']['input']['text_tokens'] == 32
    assert child['input_verification'] == request['condition_input_verification']['32']
    assert not (output / 'example' / 'text_52' / 'performance').exists()


@pytest.mark.parametrize('change', [
    {'status': 'FAILED'},
    {'execution': {'status': 'FAILED'}, 'runtime_checks': 'passed'},
    {'validation': {'runtime': {'status': 'NOT_EVALUATED'}}, 'runtime_checks': 'passed'},
    {'execution': {}, 'runtime_checks': 'passed'},
    {'execution': None, 'runtime_checks': 'passed'},
    {'validation': None, 'runtime_checks': 'passed'},
    {'execution': {'status': 'COMPLETED'}, 'validation': {}, 'runtime_checks': 'passed'},
])
def test_partial_g05_does_not_promote_failed_or_unchecked_conditions(partial_g05_run, change):
    output, row, request, report, _ = partial_g05_run
    raw = json.loads(report.read_text())
    raw.update(change)
    report.write_text(json.dumps(raw))
    assert run_zoo_example.performance_records(output, row, request, allow_partial=True) == []
    assert not (report.parent / 'performance').exists()


@pytest.mark.parametrize('corruption,match', [
    ('observer', 'bound'), ('graph', 'recaptured'), ('input', 'input files changed'),
    ('requested_input', 'row input differs'),
])
def test_partial_g05_keeps_observer_graph_and_input_qualification(partial_g05_run, corruption, match):
    output, row, request, report, observation_path = partial_g05_run
    observation = json.loads(observation_path.read_text())
    if corruption == 'observer':
        observation['request_sha256'] = '0' * 64
    elif corruption == 'graph':
        observation['audit']['graph_measured_runs'][0]['owner/cache']['entries'][0]['recapture_count'] = 1
    elif corruption == 'input':
        Path(request['config']['input']['batch_template'].format(text_tokens=32)).write_bytes(b'changed batch')
    else:
        raw = json.loads(report.read_text())
        raw['requested_config']['input']['cameras'] = 3
        report.write_text(json.dumps(raw))
        observation['raw_report_sha256'] = export_examples.file_sha(report)
    observation_path.write_text(json.dumps(observation))
    with pytest.raises(ValueError, match=match):
        run_zoo_example.performance_records(output, row, request, allow_partial=True)
    assert not (report.parent / 'performance' / 'record.json').exists()


@pytest.mark.parametrize('profile_id, variant', [
    ('qwen3_5-2b.w4-mlp-only-g32.text-controlled', 'w4_mlp_only_g32'),
    ('qwen3_5-2b.w4-mlp-w8-attn-gdn-g32.text-controlled', 'w4_mlp_w8_attn_gdn_g32'),
    ('qwen3_5-2b.quanttrio-awq-g128.text-controlled', 'quanttrio_awq_g128'),
    ('qwen3_5-4b.quanttrio-awq-g128.text-controlled', 'quanttrio_awq_g128'),
    ('qwen3_5-9b.quanttrio-awq-g128.text-controlled', 'quanttrio_awq_g128'),
])
def test_controlled_quantized_profiles_keep_recipe_identity(catalog, profile_id, variant):
    profile = export_examples.COMMON.example_profiles()[profile_id]
    assert profile['variant'] == variant
    rows = [row for row in catalog['entries'] if row['profile_id'] == profile_id]
    assert rows
    for row in rows:
        assert row['support']['reason_code'] == 'QWEN35_QUANT_IMPLEMENTED_UNVERIFIED_HW'
        assert row['support']['status'] == 'CONTROLLED_EVALUATION'
        bundle = {'schema': 'rpu-model-zoo/example-run-bundle/v1',
                  'source': catalog['source'], 'configs': [copy.deepcopy(row)],
                  'registry_paths': catalog['registry_paths']}
        _, config = run_zoo_example.resolve(bundle, row['id'])
        assert export_examples.COMMON.example_metadata(config)['profile_id'] == profile_id
        assert export_examples.COMMON.config_target(config) == 'qwen3_5'

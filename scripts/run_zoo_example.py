#!/usr/bin/env python3
"""Validate a Zoo selection and execute its actual examples entry point."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

import export_examples
# export_examples puts the repository root on sys.path; the command line
# (examples/run.py --cores/--set) applies the same override rules.
from examples.common.overrides import execution_overrides, g05_overrides, toml_text

ROOT = export_examples.ROOT
SHA256 = re.compile(r'[0-9a-f]{64}')


def primary_checkpoint(config):
    for name in ('wall_oss', 'wall_oss_sorting'):
        if name in config:
            return 'fp16_checkpoint' if config[name]['precision'] == 'fp16' else 'w8_checkpoint'
    return 'checkpoint'


def auxiliary_checkpoint_keys(config):
    target = export_examples.examples.COMMON.config_target(config)
    if target == 'gr00t':
        return ('qwen3vl_registry_alias',)
    if target == 'wall_oss_sorting':
        return ('fp16_checkpoint', 'fp16_registry_alias', 'action_w4_checkpoint')
    return ('fp16_checkpoint', 'fp16_registry_alias')


def auxiliary_checkpoints(config, primary=None):
    """Include the checkpoints selected implicitly by the public runtime."""
    inputs = config.get('input', {})
    primary = primary or primary_checkpoint(config)
    required = {key: inputs[key] for key in auxiliary_checkpoint_keys(config)
                if key != primary and inputs.get(key)}
    if (export_examples.examples.COMMON.config_target(config) == 'wall_oss'
            and 'wall_oss' not in config
            and config.get('example', {}).get('dtype') in {'w4a16', 'w4a16-pgrp'}):
        # Legacy W4 quantizes the decoder; build_wall_oss_vla loads its
        # unchanged INT8 vision tower through this fixed registry alias.
        required['vision_w8_registry_alias'] = 'wall-oss-0.5-w8a16'
    return required


def bound_asset_root(item, registry_paths, registry=None, cache_root=None):
    """Resolve the exact Zoo payload, without replacing template aliases."""
    if item.get('asset_root'):
        path = Path(item['asset_root']).expanduser()
        if not path.is_absolute():
            raise ValueError('bound asset_root must be an absolute physical path')
        return path.resolve()
    alias = item.get('asset_alias') or item.get('registry_alias')
    if not alias:
        return None
    relative = Path(registry_paths[alias])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unsafe model registry path')
    if cache_root is not None:
        return cache_root / relative
    if registry is None:
        return None
    return registry.model_path(alias)


def asset_matches(value, item, registry_paths, *, registry=None, cache_root=None):
    alias = item.get('asset_alias') or item.get('registry_alias')
    root = bound_asset_root(item, registry_paths, registry, cache_root)
    if alias is not None and value == alias:
        if item.get('asset_root') and (registry is not None or cache_root is not None):
            consumed = (cache_root / registry_paths[alias] if cache_root is not None
                        else registry.model_path(alias))
            return consumed.resolve() == root.resolve()
        return True
    return (isinstance(value, str) and root is not None
                             and Path(value).expanduser().resolve() == root.resolve())


def input_summary(config):
    inputs = config.get('input', {})
    if 'cameras' in inputs:
        lengths = inputs.get('text_lengths')
        text = ','.join(map(str, lengths)) if lengths else inputs.get('text_tokens', 32)
        return f"{inputs['cameras']}图 · T{text} · Euler{inputs.get('num_steps', 10)}"
    if inputs.get('component') == 'vision':
        return '视觉编码 · 输入尺寸见配置'
    if config.get('example', {}).get('target') == 'rhinovla':
        return f"冻结图文前缀 · Euler{inputs.get('num_steps', 10)}"
    return f"P{inputs.get('prefill_tokens', '自动')} · G{inputs.get('decode_steps', 4)}"


def physical_input_files(inputs):
    """Bind consumed caller files, including every image in a raw image request."""
    if 'frozen_input' in inputs and (not isinstance(inputs['frozen_input'], str) or not inputs['frozen_input']):
        raise ValueError('frozen_input must name a physical input file')
    result = []
    for key, value in inputs.items():
        required = key == 'frozen_input'
        if key == 'batch_template':
            value = value.format(cameras=inputs['cameras'], image_size=inputs['image_size'][0],
                                 text_tokens=inputs['text_tokens'])
            required = True
        if key == 'images':
            if 'frozen_input' in inputs:
                continue
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise ValueError('images must contain physical file paths')
            paths = [(f'images[{i}]', v) for i, v in enumerate(value)]
            required = True
        else:
            paths = [(key, value)]
        for field, candidate in paths:
            if not isinstance(candidate, str) or 'checkpoint' in key:
                continue
            path = Path(candidate).expanduser()
            if not path.is_file():
                if required:
                    raise ValueError(f'{field} has no physical input file')
                continue
            with path.open('rb') as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            result.append({'field': field, 'bytes': path.stat().st_size, 'sha256': sha})
    return result


def capture_input_verification(config):
    """Capture each consumed condition before invoking the actual example."""
    inputs = config.get('input', {})
    lengths = inputs.get('text_lengths')
    if 'g05' in config and lengths is not None:
        captured = {}
        for length in lengths:
            condition = copy.deepcopy(inputs)
            condition.pop('text_lengths')
            condition['text_tokens'] = length
            captured[str(length)] = physical_input_files(condition)
        return {'condition_input_verification': captured}
    return {'input_verification': physical_input_files(inputs)}


def verify_payload(weights, files):
    verification = []
    seen = set()
    for item in files:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in seen:
            raise ValueError('unsafe checkpoint payload path')
        seen.add(relative.as_posix())
        source = weights / relative
        if source.stat().st_size != item['size']:
            raise ValueError(f'checkpoint size differs: {relative}')
        with source.open('rb') as stream:
            actual_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual_digest != item['sha256']:
            raise ValueError(f'checkpoint digest differs: {relative}')
        verification.append({'file': item['path'], 'sha256': actual_digest, 'size': item['size']})
    return verification


def verify_auxiliary_assets(row, config, registry_paths, registry, cache_root, primary):
    target = export_examples.examples.COMMON.config_target(config)
    requirements = auxiliary_checkpoints(config, primary)
    auxiliary = row.get('auxiliary_assets', [])
    if not isinstance(auxiliary, list):
        raise ValueError('auxiliary asset bindings must be a list')
    if not requirements:
        if auxiliary:
            raise ValueError('example does not require an auxiliary checkpoint')
        return []
    if target not in {'wall_oss', 'wall_oss_rtc', 'wall_oss_sorting', 'gr00t'} or not auxiliary:
        raise ValueError('example requires a separately bound auxiliary checkpoint')
    results = []
    matched = set()
    for item in auxiliary:
        results.append(verify_auxiliary_item(item, requirements, registry_paths, registry, cache_root, matched))
    if matched != set(requirements):
        raise ValueError('examples auxiliary checkpoint differs from the verified registry asset')
    return results


def verify_auxiliary_item(item, requirements, registry_paths, registry, cache_root, matched):
    required = {'model_id', 'variant_id', 'asset_manifest_sha256', 'asset_files'}
    if (not required <= set(item) or set(item) - required - {
            'registry_alias', 'registry_path', 'asset_root', 'existing_nas_alias'}
            or not item['model_id'] or not item['variant_id'] or not item['asset_files']
            or not isinstance(item['asset_manifest_sha256'], str)
            or len(item['asset_manifest_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in item['asset_manifest_sha256'])):
        raise ValueError('auxiliary checkpoint is not bound to a complete Zoo asset')
    alias = item.get('registry_alias')
    if not item.get('asset_root') and registry_paths.get(alias) != item.get('registry_path'):
        raise ValueError('auxiliary asset registry path differs')
    weights = bound_asset_root(item, registry_paths, registry, cache_root)
    if weights is None:
        raise ValueError('auxiliary asset has no bound physical payload')
    matching = {key for key, value in requirements.items()
                if asset_matches(value, item, registry_paths, registry=registry, cache_root=cache_root)}
    if not matching or matching & matched:
        raise ValueError('auxiliary asset is unused or ambiguous for the requested checkpoints')
    matched.update(matching)
    return ({key: item[key] for key in
             ('model_id', 'variant_id', 'registry_alias', 'asset_root', 'asset_manifest_sha256') if key in item}
            | {'weight_verification': verify_payload(weights, item['asset_files'])})


def observed_report(output, request, raw_report, raw):
    """Read this run's task observer, never arbitrary replacement report fields."""
    observer = request.get('observer')
    if observer is None:
        return raw, None
    if request.get('parent_request_sha256'):
        length = request['config']['input']['text_tokens']
        path = output.parents[1] / 'observation' / f'text_{length}' / 'observation.json'
    else:
        path = output / 'observation' / 'observation.json'
    data = json.loads(path.read_text())
    bindings = {'request_sha256': request.get('parent_request_sha256') or hashlib.sha256(
                    (output / 'zoo-request.json').read_bytes()).hexdigest(),
                'effective_config_sha256': request['effective_config_sha256'],
                'raw_report_sha256': hashlib.sha256(raw_report.read_bytes()).hexdigest(),
                'observer_sha256': observer['sha256']}
    if any(data.get(key) != value for key, value in bindings.items()):
        raise ValueError('observer is not bound to this request/config/raw report')
    artifacts = data.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError('observer needs actual tensor/noise/Graph artifact identities')
    names = set()
    for item in artifacts:
        relative = Path(item['path'])
        source = path.parent / relative
        if (relative.is_absolute() or '..' in relative.parts or relative.as_posix() in names
                or not source.resolve().is_relative_to(path.parent.resolve())
                or not source.is_file() or source.is_symlink()
                or source.stat().st_size != item['bytes']
                or hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']):
            raise ValueError('observer artifact identity differs')
        names.add(relative.as_posix())
    allowed = {'environment', 'output_sha256_runs', 'floating_payload_sha256_runs',
               'floating_diagnostics_runs', 'output_tensors', 'graph_before_measurement',
               'graph_measured_runs', 'graph_observation', 'execution_receipts_runs',
               'stability_contract', 'stable_output', 'stable_graphs'}
    audit = data.get('audit')
    if not isinstance(audit, dict) or set(audit) - allowed:
        raise ValueError('observer may only supply untimed output/Graph/environment observations')
    return dict(raw, **audit), data


def graph_evidence(raw, runs, *, config_id=''):
    """Check the observed lifecycle; empty caches are not retained replay proof."""
    previous = raw.get('graph_before_measurement', {})
    measured = raw.get('graph_measured_runs', [])
    proof = raw.get('graph_observation', {})
    if not isinstance(proof, dict) or not proof.get('scope') or len(measured) != runs or not previous:
        raise ValueError('per-request Graph observations are required')
    active_runs, empty_runs = [], []
    for current in measured:
        if previous.keys() != current.keys():
            raise ValueError('Graph cache owners changed during measurement')
        active = False
        for key, old in previous.items():
            new = current[key]
            cache_active = False
            if (old.get('invariant_ok') is not True or new.get('invariant_ok') is not True
                    or old.get('size') != new.get('size')
                    or len(old.get('entries', [])) != len(new.get('entries', []))):
                raise ValueError('Graph cache size or invariants changed during measurement')
            for before, after in zip(old['entries'], new['entries']):
                if (before['graph_lifetime_id'] != after['graph_lifetime_id']
                        or before['build_generation'] != after['build_generation']
                        or before['recapture_count'] or after['recapture_count']
                        or before['non_replayable_reason'] or after['non_replayable_reason']
                        or after['replay_count'] < before['replay_count']
                        or after['kernel_count'] + after['child_graph_count'] <= 0):
                    raise ValueError('Graph identity changed or recaptured during measurement')
                cache_active |= after['replay_count'] > before['replay_count']
            setup_only = (config_id.startswith('pi05/') and key.endswith('/_rpu_adarms_graph_cache')
                          and any(name.endswith('/_rpu_fused_denoise_graph_cache') and cache['entries']
                                  for name, cache in current.items()))
            if old['entries'] and not cache_active and not setup_only:
                raise ValueError('measured Graph component did not replay')
            active |= cache_active
        active_runs.append(active)
        empty_runs.append(all(not value['entries'] for value in current.values())
                          and all(not value['entries'] for value in previous.values()))
        previous = current
    if all(active_runs):
        if proof.get('retained_replay_observed') is not True:
            raise ValueError('retained replay observation is missing')
    else:
        # Some supported profiles intentionally own bounded/one-shot graphs.
        # Require their actual per-call receipt and lifetime observations rather
        # than manufacturing retained entries for empty caches.
        receipts = raw.get('execution_receipts_runs', [])
        if (proof.get('bounded_lifecycle_observed') is not True or len(receipts) != runs
                or not all(empty_runs)
                or not proof.get('lifecycle_modes')
                or any(not item for item in receipts)
                or not set(proof['lifecycle_modes']) <= {
                    'RETAINED_CACHE', 'BOUNDED_ONESHOT', 'NATIVE_COMPOSITE1', 'COMPOSITE_CHILD'}):
            raise ValueError('bounded lifecycle needs actual per-request execution/lifetime evidence')
        def modes(value):
            if isinstance(value, dict):
                return ({value['graph_mode']} if 'graph_mode' in value else set()).union(
                    *(modes(child) for child in value.values()))
            if isinstance(value, list):
                return set().union(*(modes(child) for child in value))
            return set()
        expected = set(proof['lifecycle_modes'])
        if ('BOUNDED_ONESHOT' not in expected and 'NATIVE_COMPOSITE1' not in expected
                or any(modes(receipt) != expected for receipt in receipts)):
            raise ValueError('observed execution receipts do not prove the bounded lifecycle modes')
    return proof


def _runtime_checked(report):
    """Modern execution/validation fields take precedence over legacy labels."""
    if 'execution' in report or 'validation' in report:
        execution, validation = report.get('execution'), report.get('validation')
        runtime = validation.get('runtime') if isinstance(validation, dict) else None
        return (isinstance(execution, dict) and execution.get('status') == 'COMPLETED'
                and isinstance(runtime, dict) and runtime.get('status') == 'CHECKED')
    return report.get('runtime_checks') == 'passed'


def performance_record(output, row, request, *, raw_report=None):
    """Project the actual example report into the Zoo's existing result format."""
    reports = [raw_report] if raw_report is not None else list(output.glob('**/report.json'))
    if len(reports) != 1:
        raise ValueError('performance needs exactly one raw example report')
    original = json.loads(reports[0].read_text())
    raw, observation = observed_report(output, request, reports[0], original)
    samples = raw.get('wall_ms_runs')
    if samples is None:
        samples = raw.get('latency', {}).get('samples_ms')
    if samples is None:
        samples = raw.get('e2e', {}).get('samples_ms')
    if samples is None:
        samples = [item['total_ms'] for item in raw.get('samples', raw.get('samples_ms', []))]
    hashes = raw.get('output_sha256_runs', [])
    proof = graph_evidence(raw, request['config']['run']['runs'], config_id=row['id'])
    trace = reports[0].parent / 'profile.trace.json'
    if (not _runtime_checked(original)
            or original.get('error') or original.get('cleanup_error')
            or raw.get('stable_output') is not True or raw.get('stable_graphs') is not True
            or len(samples) != request['config']['run']['runs']
            or len(hashes) != len(samples) or len(set(hashes)) != 1
            or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in hashes)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples)
            or raw.get('environment', {}).get('backend_source_sha256') != request['backend_source_sha256']):
        raise ValueError('example lacks qualified output, Graph replay or environment evidence')
    if not raw['environment'].get('native_libraries') or not raw['environment'].get('python_modules'):
        raise ValueError('actual loaded Python/native identities are required')
    directory = output / 'performance'
    directory.mkdir()
    # Keep output locations out of the comparison scope, while retaining all
    # inference semantics and physical input content identities.
    functional = copy.deepcopy(request['config'])
    functional.pop('rpu_execution', None)
    functional.get('example', {}).pop('profile_id', None)
    for key in ('output_dir', 'warmup', 'runs', 'profile', 'hwperf', 'audit_replay'):
        functional.get('run', {}).pop(key, None)
    physical_files = physical_input_files(request['config'].get('input', {}))
    if request.get('input_verification', physical_files) != physical_files:
        raise ValueError('caller input files changed during inference')
    policy = {'warmup': request['config']['run']['warmup'], 'runs': request['config']['run']['runs'],
              'statistic': 'median', 'timing_scope': raw.get('timing_scope', 'example infer wall time; output audit excluded'),
              'graph_observation_scope': proof['scope'], 'hardware_tracing_during_measurement': False}
    request_sha = hashlib.sha256((output / 'zoo-request.json').read_bytes()).hexdigest()
    input_evidence = {'functional_config': functional, 'physical_files': physical_files}
    if row.get('auxiliary_assets'):
        expected = [{key: item[key] for key in
            ('model_id', 'variant_id', 'registry_alias', 'asset_root', 'asset_manifest_sha256') if key in item}
            | {'weight_verification': [{'file': entry['path'], 'size': entry['size'], 'sha256': entry['sha256']}
                                       for entry in item['asset_files']]}
            for item in row['auxiliary_assets']]
        if (request.get('auxiliary_assets') != row['auxiliary_assets']
                or request.get('auxiliary_verification') != expected):
            raise ValueError('performance needs the bound auxiliary asset verification')
        input_evidence['auxiliary_verification'] = request['auxiliary_verification']
    elif request.get('auxiliary_assets') or request.get('auxiliary_verification'):
        raise ValueError('unbound auxiliary performance evidence')
    artifacts = {'request': request, 'input': input_evidence,
                 'environment': raw['environment'], 'measurement_policy': policy,
                 'replay': {'request_ms_samples': samples, 'output_sha256_runs': hashes,
                            'runtime_checks': 'passed', 'stable_replay': True, 'graph_replay_evidence': proof,
                            'graph_before_measurement': raw['graph_before_measurement'],
                            'graph_measured_runs': raw['graph_measured_runs'],
                            'stability_contract': raw['stability_contract'],
                            'execution_receipts_runs': raw.get('execution_receipts_runs', []),
                            'floating_payload_sha256_runs': raw.get('floating_payload_sha256_runs', []),
                            'floating_diagnostics_runs': raw.get('floating_diagnostics_runs', [])},
                 'report': {'config_id': row['id'], 'backend_source_sha256': request['backend_source_sha256'],
                            'request_sha256': request_sha, 'request_ms_samples': samples,
                            'runtime_checks': 'passed', 'raw_example_report': original}}
    if observation is not None:
        artifacts['observation'] = observation
    refs = {}
    for role, value in artifacts.items():
        path = directory / (role + '.json')
        if role == 'observation':
            path = directory / 'observation' / 'observation.json'
            path.parent.mkdir()
            source = (output.parents[1] / 'observation' / f"text_{request['config']['input']['text_tokens']}"
                      if request.get('parent_request_sha256') else output / 'observation')
            for item in value['artifacts']:
                target = path.parent / item['path']
                if target.name in {'observation.json', 'raw-report.json', 'observer.py'}:
                    raise ValueError('observer artifact uses a reserved provenance filename')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / item['path'], target)
            shutil.copyfile(reports[0], path.parent / 'raw-report.json')
            shutil.copyfile(request['observer']['entry_point'], path.parent / 'observer.py')
        if role == 'request':
            path.write_bytes((output / 'zoo-request.json').read_bytes())
        else:
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        refs[role] = {'path': path.relative_to(directory).as_posix(),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if trace.is_file():
        shutil.copyfile(trace, directory / 'trace.json')
        refs['trace'] = {'path': 'trace.json', 'sha256': hashlib.sha256(trace.read_bytes()).hexdigest()}
    record = {'case_id': row['case_id'], 'config_id': row['id'], 'config_sha256': row['config_sha256'],
              'input_summary': input_summary(request['config']),
              'backend_source_sha256': request['backend_source_sha256'],
              'asset_manifest_sha256': request['asset_manifest_sha256'],
              'status': 'measured', 'runtime_checks': 'passed', 'stable_replay': True,
              'request_ms_p50': statistics.median(samples), 'admission_status': None,
              **{key + '_sha256': refs[key]['sha256'] for key in ('input', 'environment', 'measurement_policy')}}
    (directory / 'record.json').write_text(json.dumps(
        {'schema': 'rpu-model-zoo/example-performance-record/v1', 'record': record, 'artifacts': refs},
        ensure_ascii=False, indent=2) + '\n')
    return str(directory / 'record.json')


def performance_records(output, row, request, *, allow_partial=False):
    lengths = request['config'].get('input', {}).get('text_lengths')
    if lengths is None:
        if allow_partial:
            raise ValueError('partial collection requires independently completed G05 text conditions')
        return [performance_record(output, row, request)]
    if 'g05' not in request['config']:
        raise ValueError('multi-condition reports require a registered G05 example')
    root = Path(request['config']['run']['output_dir'])
    aggregate = json.loads((root / 'report.json').read_text())
    completed = set(aggregate.get('rows', {}))
    expected = {str(n) for n in lengths}
    if (not completed <= expected or (not allow_partial and
            (aggregate.get('status') != 'COMPLETE' or completed != expected))):
        raise ValueError('G05 aggregate does not cover every requested text length')
    captured = request.get('condition_input_verification')
    if 'condition_input_verification' in request and (not isinstance(captured, dict) or set(captured) != expected
            or any(not isinstance(proof, list) for proof in captured.values())):
        raise ValueError('G05 pre-execution input verification does not cover every condition')
    records = []
    for length in lengths:
        directory = root / f'text_{length}'
        report = directory / 'report.json'
        if allow_partial and not report.is_file():
            continue
        raw = json.loads(report.read_text())
        if allow_partial and (raw.get('status') != 'COMPLETE' or not _runtime_checked(raw)):
            continue
        inputs = raw.get('requested_config', {}).get('input', {})
        expected_inputs = dict(request['config']['input'])
        expected_inputs.pop('text_lengths')
        expected_inputs['text_tokens'] = length
        if inputs != expected_inputs:
            raise ValueError('G05 row input differs from the requested condition')
        child = copy.deepcopy(request)
        child.pop('condition_input_verification', None)
        if captured is not None:
            child['input_verification'] = captured[str(length)]
        child['config']['input'] = expected_inputs
        child['config']['run']['output_dir'] = str(directory)
        child['effective_config_sha256'] = export_examples.identity(export_examples.semantic_config(child['config']))
        child['parent_request_sha256'] = hashlib.sha256((output / 'zoo-request.json').read_bytes()).hexdigest()
        (directory / 'zoo-request.json').write_text(json.dumps(child, ensure_ascii=False, indent=2) + '\n')
        records.append(performance_record(directory, row, child, raw_report=report))
    return records


def verify_bound_libraries(config):
    table = config.get('libraries')
    if table is None:
        return None
    from examples.common import cold_libraries
    cold_libraries.validate_libraries(table)
    native = os.environ.get('RPU_BACKEND_SO')
    if not native or Path(table['native']).expanduser().resolve() != Path(native).expanduser().resolve():
        raise ValueError('libraries.native must equal the supervisor-bound RPU_BACKEND_SO')
    # The existing verifier admits only the manifest's two operator role hashes.
    _, verified = cold_libraries.bind_libraries(config)
    return verified


def resolve(bundle, config_id):
    if bundle.get('schema') != 'rpu-model-zoo/example-run-bundle/v1':
        raise ValueError('unknown Model Zoo example run bundle')
    actual = export_examples.export_catalog(commit=bundle.get('source', {}).get('commit'))
    if bundle.get('source') != actual['source']:
        raise ValueError('backend source differs from selected examples; re-import and resolve')
    if bundle.get('registry_paths') != actual['registry_paths']:
        raise ValueError('model registry differs from selected examples')
    matches = [r for r in bundle['configs'] if r['id'] == config_id]
    if len(matches) != 1:
        raise ValueError('select exactly one config in this bundle')
    row = matches[0]
    current = next((r for r in actual['entries'] if r['id'] == config_id), None)
    if current is None or any(row.get(k) != current[k] for k in current):
        raise ValueError('selected example config identity differs')
    if row['entry_kind'] != 'example':
        raise ValueError('validation fragments are not public examples')
    config = copy.deepcopy(current['config'])
    primary = primary_checkpoint(config)
    if row.get('auxiliary_assets') and not auxiliary_checkpoints(config, primary):
        raise ValueError('example does not require an auxiliary checkpoint')
    overrides = row.get('overrides', {})
    if set(overrides) - {'input', 'run', 'cpu', 'libraries', 'rpu_execution', 'g05'}:
        raise ValueError('only caller inputs, timing, bound libraries and public cold execution choices may vary')
    for key, value in overrides.items():
        if not isinstance(value, dict):
            raise ValueError('example overrides must be tables')
        if key == 'rpu_execution':
            execution_overrides(config, value)
            continue
        if key == 'g05':
            g05_overrides(config, value)
        if key == 'libraries' and (key not in config or set(value) != {'native', 'main', 'expansion'}):
            raise ValueError('libraries override must bind the native and both formal REF roles together')
        if key == 'input':
            prepared = (export_examples.examples.COMMON.config_target(config) == 'qwen3_vl'
                        and config.get('input', {}).get('component') == 'multimodal'
                        and isinstance(value.get('prepared_input_safetensors'), str)
                        and bool(value['prepared_input_safetensors'])
                        and isinstance(value.get('prepared_input_sha256'), str)
                        and SHA256.fullmatch(value['prepared_input_sha256']))
            for name, candidate in value.items():
                if candidate is None:
                    if not prepared or name not in {'prompt', 'images', 'prefill_tokens'}:
                        raise ValueError('null input overrides only remove prompt/images/prefill_tokens for paired Qwen3-VL prepared input')
                    continue
                if name == 'release_root':
                    root = bound_asset_root(row, bundle['registry_paths'])
                    if (export_examples.examples.COMMON.config_target(config) != 'g05'
                            or root is None or not row.get('release_root')
                            or Path(candidate).expanduser().resolve() != root.parent
                            or Path(row['release_root']).expanduser().resolve() != root.parent):
                        raise ValueError('release_root must be the bound complete G05 release parent')
                    continue
                if name == 'runtime_factory' or name.endswith('registry_alias'):
                    raise ValueError('checkpoint/factory overrides would change the bound asset')
                if 'checkpoint' in name:
                    item = row if name == primary else next((item for item in row.get('auxiliary_assets', [])
                        if asset_matches(candidate, item, bundle['registry_paths'])), {})
                    if (not item.get('asset_root')
                            or not asset_matches(candidate, item, bundle['registry_paths'])):
                        raise ValueError('checkpoint override differs from the bound asset')
        destination = config.setdefault(key, {})
        for name, candidate in value.items():
            if key == 'input' and candidate is None:
                destination.pop(name, None)
            else:
                destination[name] = candidate
    return row, config


def run(bundle, config_id, *, output, cache_root=None, check=False, performance=False,
        observer_entry=None):
    row, config = resolve(bundle, config_id)
    if performance:
        if not row.get('case_id'):
            raise ValueError('resolve a fresh Zoo selection with a case identity before measuring')
        target = export_examples.examples.COMMON.config_target(config)
        warmup = 3 if target == 'hy_vla' else 2
        config.setdefault('run', {}).update(warmup=warmup, runs=5)
        if target not in {'wall_oss', 'wall_oss_sorting'}:
            config['run'].update(profile=False, hwperf=False)
        if 'audit_replay' in config['run']:
            raise ValueError('audit_replay is not a supported public example option')
        if config.get('runner', {}).get('hw_profile', {}).get('enabled'):
            raise ValueError('hardware tracing must be disabled during performance measurement')
    if not check:
        if output.exists() and any(output.iterdir()):
            raise ValueError('run output must be a new or empty directory')
        config.setdefault('run', {})['output_dir'] = str((output / 'example').resolve())
    with tempfile.TemporaryDirectory(prefix='zoo-example-') as temp:
        path = Path(temp) / 'config.toml'
        path.write_text(toml_text(config))
        # Same loader used by examples --check-config, before device imports.
        export_examples.examples.COMMON.load_config(path)
        if check:
            return {'config_id': config_id, 'config_sha256': export_examples.identity(config),
                    'status': 'configuration-valid', 'hardware': 'not-run'}
        alias = row['asset_alias']
        if ((not alias and not row.get('asset_root'))
                or not row.get('asset_manifest_sha256') or not row.get('asset_files')):
            raise ValueError('checkpoint is not bound to a Zoo asset; bind and verify it before execution')
        registry = export_examples.examples.COMMON._load_file('_zoo_registry', ROOT / 'python/rpu_backend/model_registry.py')
        library_verification = verify_bound_libraries(config)
        weights = bound_asset_root(row, bundle['registry_paths'], registry, cache_root)
        primary = primary_checkpoint(config)
        checkpoint = config.get('input', {}).get(primary)
        if checkpoint and not asset_matches(checkpoint, row, bundle['registry_paths'],
                                           registry=registry, cache_root=cache_root):
            raise ValueError('examples checkpoint differs from the verified registry asset')
        auxiliary_verification = verify_auxiliary_assets(row, config, bundle['registry_paths'], registry,
                                                         cache_root, primary)
        verification = verify_payload(weights, row['asset_files'])
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        if cache_root:
            env['RPU_MODEL_CACHE'] = str(cache_root.resolve())
        output.mkdir(parents=True, exist_ok=True)
        request = {'schema': 'rpu-model-zoo/example-execution-request/v1',
                   'config_id': config_id, 'template_config_sha256': row['config_sha256'],
                   'effective_config_sha256': export_examples.identity(export_examples.semantic_config(config)),
                   'backend_source_sha256': bundle['source']['content_sha256'],
                   'model_id': row['model_id'], 'variant_id': row['variant_id'],
                   'asset_manifest_sha256': row['asset_manifest_sha256'],
                   'weight_verification': verification,
                   'entry_point': row['entry_point'], 'config': config}
        if row.get('overrides'):
            request['overrides'] = copy.deepcopy(row['overrides'])
        if row.get('asset_root'):
            request['asset_root'] = str(weights.resolve())
        if library_verification is not None:
            request['library_verification'] = library_verification
        if observer_entry is not None:
            observer_entry = observer_entry.expanduser().resolve()
            if not observer_entry.is_file():
                raise ValueError('observer_entry must name the actual task observer file')
            request['observer'] = {'entry_point': str(observer_entry),
                                   'sha256': hashlib.sha256(observer_entry.read_bytes()).hexdigest()}
        if auxiliary_verification:
            request['auxiliary_assets'] = copy.deepcopy(row['auxiliary_assets'])
            request['auxiliary_verification'] = auxiliary_verification
        if performance:
            request.update(capture_input_verification(config))
        (output / 'zoo-request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
        actual_config = output / 'config.toml'
        actual_config.write_text(path.read_text())
        public = ROOT / row['entry_point']
        if public != ROOT / 'examples/run.py':
            raise ValueError('Model Zoo examples must use the unified public dispatcher')
        command = [sys.executable, '-B', str(public), '--config', str(actual_config.resolve())]
        if observer_entry is not None:
            command = [sys.executable, '-B', str(observer_entry),
                       str((output / 'observation').resolve()), str(public),
                       '--config', str(actual_config.resolve())]
            env['RPU_ZOO_REQUEST_PATH'] = str((output / 'zoo-request.json').resolve())
        with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
            result = subprocess.run(command, cwd=ROOT, env=env,
                                    stdout=stdout, stderr=stderr)
        status = {'schema': 'rpu-model-zoo/example-execution-status/v1',
                  'request_sha256': hashlib.sha256((output / 'zoo-request.json').read_bytes()).hexdigest(),
                  'returncode': result.returncode,
                  'status': 'executed' if result.returncode == 0 else 'error',
                  'performance_status': 'requires-matching-replay-and-environment-evidence'}
        if performance:
            try:
                if result.returncode and not ("g05" in config and config.get("input", {}).get("text_lengths")):
                    raise ValueError(f'example execution failed (returncode {result.returncode}); see stderr.log')
                records = performance_records(output, row, request, allow_partial=result.returncode != 0)
                if not records:
                    raise ValueError('no completed condition has qualified evidence')
                status['performance_records'] = records
                if len(records) == 1:
                    status['performance_record'] = records[0]
                status['performance_status'] = ('measured-with-replay-evidence' if result.returncode == 0
                                                else 'measured-partially-with-replay-evidence')
            except (OSError, ValueError, KeyError) as exc:
                status.update(performance_status='unqualified', performance_reason=str(exc))
        (output / 'zoo-status.json').write_text(json.dumps(status, indent=2) + '\n')
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--config-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--observer-entry', type=Path,
                        help='Task observer: receives OUTPUT/observation, public entry, then --config')
    parser.add_argument('--performance', action='store_true',
                        help='Collect W2/R5 (HyVLA W3/R5); qualify only actual untimed observation evidence')
    args = parser.parse_args()
    try:
        result = run(json.loads(args.request.read_text()), args.config_id, output=args.output,
                     cache_root=args.cache_root, check=args.check_config, performance=args.performance,
                     observer_entry=args.observer_entry)
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result['status'] == 'error' else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Config adapter for Wall-OSS base two- and three-camera execution paths."""
from __future__ import annotations

from examples.common import cpu as _cpu_helpers

import argparse
from copy import deepcopy
from pathlib import Path
from string import Formatter
import tomllib


ROOT = Path(__file__).resolve().parents[3]
_CPU = vars(_cpu_helpers)
# The modules the Wall-OSS owners tell apart, and the base engine's (--precision,
# Action NVFP4) for each mix it implements. Hybrid quantizes Action from the FP16 source.
ROUTE_PROBES = (('vision',), ('prefill',), ('prefill', 'Down'), ('action',), ('action_preprocessor',))
ROUTES = {
    ('fp16',) * 5: ('fp16', False),
    ('w8a16',) * 5: ('w8', False),
    ('w8a16', 'w8a16', 'w8a16', 'nvfp4', 'fp16'): ('w8', True),
}
CPU_THREADS = (1, 2, 4, 8, 12, 16, 24)
EXECUTION_FIELDS = {
    'prefill': ('chunk_size', 'padding_rows', 'padding_budget', 'linear_acc32'),
    'vision': ('chunk_size', 'linear_acc32'),
    'action': ('chunk_size', 'linear_acc32'),
    'model': ('num_cores',),
}


def precision_route(precision):
    """The engine precision and Action NVFP4 switch that realize a registered Wall-OSS base recipe."""
    from examples.common.config import owner_route
    return owner_route('Wall-OSS base', 'wall_oss', ROUTE_PROBES, ROUTES, selector=precision)


def execution_config(value):
    # Use the public parser without importing torch or initializing the backend.
    from examples.common.config import _backend_leaf
    normalized = _backend_leaf('api/_execution.py').normalize_rpu_execution(
        value, entry_point='Wall-OSS base example', supported=EXECUTION_FIELDS)
    if normalized.get('action', {}).get('chunk_size', 'auto') not in ('auto', 32):
        raise ValueError('Wall-OSS base action.chunk_size must be auto or 32 for H32')
    return {stage: dict(fields) for stage, fields in normalized.items()}


def core_count(config):
    return execution_config(config.get('rpu_execution')).get('model', {}).get('num_cores', 8)


def _path(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a nonempty path')


def _integer(value, field, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bound = f'{minimum}..{maximum}' if maximum is not None else f'>= {minimum}'
        raise ValueError(f'{field} must be an integer {bound}')


def batch_path(inputs):
    """Select an existing input fixture without rewriting its instruction."""
    if ('batch' in inputs) == ('batch_template' in inputs):
        raise ValueError('[input] requires exactly one of batch or batch_template')
    if 'batch' in inputs:
        _path(inputs['batch'], '[input].batch')
        return inputs['batch']
    template = inputs['batch_template']
    _path(template, '[input].batch_template')
    values = {'cameras': inputs.get('cameras', 3),
              'image_size': inputs['image_size'][0], 'text_tokens': inputs['text_tokens']}
    fields = []
    for _, field, spec, conversion in Formatter().parse(template):
        if field is not None:
            if field not in values or spec or conversion:
                raise ValueError('batch_template only supports {cameras}, {image_size}, {text_tokens}')
            fields.append(field)
    if 'text_tokens' not in fields:
        raise ValueError('batch_template must contain {text_tokens}')
    return template.format(**values)


def validate_example_config(config: dict) -> None:
    required = {'wall_oss', 'input', 'run'}
    allowed = {
        'wall_oss': {'precision', 'vision_window', 'vision_pair', 'vision_row_reuse', 'instruction_residency'},
        'input': {'fp16_checkpoint', 'w8_checkpoint', 'batch', 'batch_template', 'image_size', 'cameras',
                  'text_tokens', 'text_length_scope', 'action_horizon', 'num_steps'},
        'run': {'warmup', 'runs', 'cpu_threads', 'torch_num_threads', 'cpu_thread_sweep', 'seed', 'output_dir'},
        'cpu': {'threads'},
        'runtime': {'native', 'main_ref', 'expansion_ref'},
        'rpu_execution': None,
        'logging': None,
    }
    if not isinstance(config, dict) or not required <= set(config) or set(config) - set(allowed):
        raise ValueError('Wall-OSS base requires [wall_oss], [input], [run], '
                         'and optional [cpu]/[runtime]/[rpu_execution]/[logging]')
    from examples.common.config import validate_logging_config
    validate_logging_config(config)
    for name, values in config.items():
        if not isinstance(values, dict) or (allowed[name] is not None and set(values) - allowed[name]):
            raise ValueError(f'[{name}] contains unsupported keys or is not a table')
    precision = config['wall_oss'].get('precision')
    engine, _ = precision_route(precision)
    mode = config['wall_oss'].get('instruction_residency', 'auto')
    if mode not in ('auto', 'off'):
        raise ValueError('[wall_oss].instruction_residency must be auto or off')
    inputs, run = config['input'], config['run']
    for key in ('fp16_checkpoint',):
        _path(inputs.get(key), f'[input].{key}')
    if engine != 'fp16' or 'w8_checkpoint' in inputs:
        _path(inputs.get('w8_checkpoint'), '[input].w8_checkpoint')
        from examples.common.config import check_checkpoint_recipe
        check_checkpoint_recipe('wall_oss', Path(inputs['w8_checkpoint']).expanduser(),
                                slot='w8_checkpoint', selector=precision)
    shape = inputs.get('image_size')
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(size) is not int for size in shape)
            or tuple(shape) not in ((224, 224), (256, 256))):
        raise ValueError('[input].image_size must be [224, 224] or [256, 256]')
    cameras = inputs.get('cameras', 3)
    if type(cameras) is not int or cameras not in (1, 2, 3):
        raise ValueError('[input].cameras must be 1, 2 or 3')
    row_reuse = config['wall_oss'].get('vision_row_reuse', True)
    if type(row_reuse) is not bool:
        raise ValueError('[wall_oss].vision_row_reuse must be boolean')
    if 'vision_window' in config['wall_oss']:
        window = config['wall_oss']['vision_window']
        if type(window) is not bool:
            raise ValueError('[wall_oss].vision_window must be boolean')
        if window and shape != [224, 224]:
            raise ValueError('[wall_oss].vision_window=true requires 224x224 input')
        if window and not row_reuse:
            raise ValueError('[wall_oss].vision_window=true requires vision_row_reuse=true')
    if 'vision_pair' in config['wall_oss']:
        pair = config['wall_oss']['vision_pair']
        if type(pair) is not bool:
            raise ValueError('[wall_oss].vision_pair must be boolean')
        if pair and shape != [256, 256]:
            raise ValueError('[wall_oss].vision_pair=true requires 256x256 input')
    for key, expected in (('action_horizon', 32), ('num_steps', 10)):
        value = inputs.get(key, expected)
        if type(value) is not int or value != expected:
            raise ValueError(f'[input].{key} must be {expected}')
    _integer(inputs.get('text_tokens'), '[input].text_tokens', 1)
    batch_path(inputs)
    if inputs.get('text_length_scope', 'instruction') not in ('instruction', 'non_image'):
        raise ValueError('[input].text_length_scope must be instruction or non_image')
    _integer(run.get('warmup', 2), '[run].warmup', 1, 10)
    _integer(run.get('runs', 5), '[run].runs', 1, 30)
    _integer(run.get('seed', 0), '[run].seed', 0)
    threads = _CPU['resolve_cpu_threads'](config, default=8)
    sweep = run.get('cpu_thread_sweep', [])
    if not isinstance(sweep, list) or any(type(n) is not int or n not in CPU_THREADS for n in sweep):
        raise ValueError(f'[run].cpu_thread_sweep must be a list of counts from {CPU_THREADS}')
    if len(set([threads, *sweep])) != 1 + len(sweep):
        raise ValueError('[run] CPU thread counts must be distinct')
    if run.get('runs', 5) * (1 + len(sweep)) > 30:
        raise ValueError('[run] permits at most 30 measured calls across CPU thread counts')
    _path(run.get('output_dir'), '[run].output_dir')
    for key, value in config.get('runtime', {}).items():
        _path(value, f'[runtime].{key}')
    execution_config(config.get('rpu_execution'))
    cores = core_count(config)
    if cores != 8 and 'instruction_residency' in config['wall_oss']:
        raise ValueError(f'[wall_oss].instruction_residency is eight-core only, got model.num_cores={cores}')
    linear_acc32 = [stage for stage in ('prefill', 'vision', 'action')
                    if config.get('rpu_execution', {}).get(stage, {}).get('linear_acc32')]
    if cores != 8 and linear_acc32:
        raise ValueError(f"linear_acc32 is eight-core only, got model.num_cores={cores} with {', '.join(linear_acc32)}")


def example_argv(config: dict) -> list[str]:
    """Resolve public defaults; asset existence remains the engine's check."""
    validate_example_config(config)
    precision = config['wall_oss']['precision']
    inputs, run, runtime = config['input'], config['run'], config.get('runtime', {})
    size = inputs['image_size'][0]
    cameras = inputs.get('cameras', 3)
    row_reuse = config['wall_oss'].get('vision_row_reuse', True)
    engine, hybrid = precision_route(precision)
    reduced = core_count(config) != 8
    argv = ['--mode', 'candidate', '--precision', engine,
            '--components', 'small' if row_reuse else 'small-prefill',
            '--fp16', str(Path(inputs['fp16_checkpoint']).expanduser()),
            '--batch', str(Path(batch_path(inputs)).expanduser()), '--image-size', str(size), str(size),
            '--cameras', str(cameras),
            '--text-tokens', str(inputs['text_tokens']),
            '--text-length-scope', inputs.get('text_length_scope', 'instruction'),
            '--cpu-threads', str(_CPU['resolve_cpu_threads'](config, default=8)),
            '--warmup', str(run.get('warmup', 2)),
            '--runs', str(run.get('runs', 5)),
            '--output-dir', str(Path(run['output_dir']).expanduser())]
    if 'instruction_residency' in config['wall_oss']:
        argv += ['--instruction-residency', config['wall_oss']['instruction_residency']]
    if run.get('cpu_thread_sweep'):
        argv += ['--cpu-thread-sweep', *(str(n) for n in run['cpu_thread_sweep'])]
    if engine != 'fp16':
        argv += ['--w8', str(Path(inputs['w8_checkpoint']).expanduser())]
    # Pair and residency have eight-core owners only; a reduced budget runs one
    # camera per forward with the ordinary Action.
    if not reduced and config['wall_oss'].get('vision_pair', size == 256):
        argv.append('--vision-pair')
    if hybrid:
        # R39/R43 Action NVFP4 measurements used neither residency nor the
        # 224 native-window opt-in. Keep those defaults; resolve an explicit
        # window override below without changing Action residency.
        argv.append('--action-nvfp4')
    elif not reduced and not execution_config(config.get('rpu_execution')).get('action', {}).get('linear_acc32', False):
        argv.append('--action-residency')
    window = config['wall_oss'].get('vision_window', size == 224 and not hybrid and row_reuse)
    if window:
        argv.append('--vision-window')
    for key, flag in (('native', '--so'), ('main_ref', '--main-ref'),
                      ('expansion_ref', '--expansion-ref')):
        if key in runtime:
            argv += [flag, str(Path(runtime[key]).expanduser())]
    return argv


def validation_runner():
    from importlib import import_module
    return import_module("examples.runners.wall_oss.base_engine")


def run_example_config(config: dict) -> int:
    argv = example_argv(config)
    engine = validation_runner()
    args = engine.arguments(argv)
    args.rpu_execution = execution_config(config.get('rpu_execution'))
    args.source_config = deepcopy(config)
    args.seed = config['run'].get('seed', 0)
    return engine.main(args)


def main(argv=None) -> int:
    from examples.common.logging import configuration_checked, logging_session
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args(argv)
    try:
        with args.config.expanduser().open('rb') as stream:
            config = tomllib.load(stream)
        validate_example_config(config)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.check_config:
        configuration_checked(f'configuration OK: {args.config} (Wall-OSS base; syntax only)')
        return 0
    with logging_session(config):
        return run_example_config(config)


if __name__ == '__main__':
    raise SystemExit(main())

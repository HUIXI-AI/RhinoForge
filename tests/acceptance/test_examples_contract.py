from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import re
import pytest
ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'examples'

def _load_common():
    spec = importlib.util.spec_from_file_location('examples_common_test', EXAMPLES / 'common/config.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def example_catalog(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('check_examples_test', ROOT / 'scripts/check_examples.py')
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    examples = tmp_path / 'examples'
    shutil.copytree(EXAMPLES / 'configs', examples / 'configs')
    for entry in EXAMPLES.glob('*.py'):
        if not entry.name.startswith('_'):
            shutil.copy2(entry, examples / entry.name)
    (tmp_path / 'docs').mkdir()
    for name in ('api_reference.md', 'api_reference.zh.md'):
        shutil.copy2(ROOT / 'docs' / name, tmp_path / 'docs' / name)
    monkeypatch.setattr(checker, 'ROOT', tmp_path)
    monkeypatch.setattr(checker, 'EXAMPLES', examples)
    monkeypatch.setattr(checker, 'CONFIGS', examples / 'configs')
    return checker

def test_wall_base_default_keeps_legacy_profiles_separate(example_catalog):
    checker = example_catalog
    default = checker.CONFIGS / 'wall_oss/base/3cam/224/fp16.toml'
    assert default == checker.CONFIGS / 'wall_oss/base/3cam/224/fp16.toml'
    assert checker.COMMON.example_metadata(checker.COMMON.load_config(default)) is None
    catalog = checker.catalog()
    legacy = catalog['vla']['wall-oss-0.5.fp16.vla']
    assert {row[1] for row in legacy} == {checker.CONFIGS / 'wall_oss/legacy/fp16.toml'}
    assert legacy[0][3]['support']['status'] == 'SUPPORTED_LIMITED'

def test_g05_custom_default_keeps_registered_legacy_profile_separate(example_catalog):
    checker = example_catalog
    default = checker.CONFIGS / 'g05/3cam/fp16.toml'
    assert default == checker.CONFIGS / 'g05/3cam/fp16.toml'
    config = checker.COMMON.load_config(default)
    assert checker.COMMON.example_metadata(config) is None
    rows = checker.catalog()['vla']['g05-base.fp16.k1-rtc-fm']
    assert {row[1] for row in rows} == {checker.CONFIGS / 'g05/legacy/fp16.toml'}
    legacy = rows[0][2]
    assert legacy['example']['profile_id'] == 'g05-base.fp16.k1-rtc-fm'
    assert rows[0][3]['support']['status'] == 'CONTROLLED_EVALUATION'

def test_integration_configs_remain_runnable_without_public_documentation(example_catalog):
    checker = example_catalog
    configs = checker.load_configs()
    catalog = checker.catalog(configs)
    private_paths = [path for path in configs if not checker.documented_config(path)]
    assert len(private_paths) == 12
    assert len(configs) > len(private_paths)
    rendered = checker.render_config_index(configs)
    for language in ('en', 'zh'):
        rendered += checker.render(language, catalog)
    for path in private_paths:
        assert path.is_file()
        assert path.relative_to(checker.CONFIGS).as_posix() not in rendered
        metadata = checker.COMMON.example_metadata(configs[path])
        if metadata is not None:
            assert metadata['profile_id'] in catalog['vla']
            assert metadata['profile_id'] not in rendered
    assert 'wall_oss/base/3cam/224/fp16.toml' in rendered
    assert 'wall_oss/legacy/fp16.toml' in rendered

def test_hyvla_optimized_default_keeps_registered_fixture_separate(example_catalog):
    checker = example_catalog
    default = checker.CONFIGS / 'hy_vla/3cam/fp16.toml'
    assert default == checker.CONFIGS / 'hy_vla/3cam/fp16.toml'
    assert checker.COMMON.example_metadata(checker.COMMON.load_config(default)) is None
    rows = checker.catalog()['vla']['hy-embodied-0.5-vla-umi.fp16.vla']
    assert {row[1] for row in rows} == {checker.CONFIGS / 'hy_embodied/umi/fp16.toml'}
    assert rows[0][3]['support']['status'] == 'SUPPORTED_LIMITED'

@pytest.mark.parametrize('setting', ['[env]\nRPU_OTHER_SWITCH = "1"', '[env]\nRPU_RHINOVLA_HIGH_PRECISION = true', '[rpu_execution.action]\nunknown_switch = true'])
def test_rhinovla_runtime_fragments_validate_syntax_without_profile_inheritance(tmp_path, setting):
    common = _load_common()
    path = tmp_path / 'fragment.toml'
    section = setting.split('\n', 1)[0]
    other = '[rpu_execution.action]\nfast_replay = true' if section == '[env]' else '[env]\nRPU_RHINOVLA_HIGH_PRECISION = "1"'
    path.write_text(setting + '\n' + other + '\n')
    with pytest.raises(common.ConfigError):
        common.load_config(path)

def test_rhinovla_v3_fragment_requires_its_own_runtime_entry(monkeypatch, capsys):
    common = _load_common()
    path = EXAMPLES / 'configs/rhinovla/v3/controlled/fp16.toml'
    assert common.example_metadata(common.load_config(path)) is None
    monkeypatch.setattr(sys, 'argv', ['run.py', '--config', str(path)])
    with pytest.raises(SystemExit) as caught:
        from examples.common.cli import main
        main()
    assert caught.value.code == 2
    assert 'scripts/validate_rhinovla_v3.py run' in capsys.readouterr().err

@pytest.mark.parametrize('failure', ['planner_collision', 'template_collision', 'support_claim'])
def test_factory_catalog_cannot_override_planner_admission(tmp_path, monkeypatch, failure):
    common = _load_common()
    data = json.loads(common.EXAMPLE_REGISTRY.read_text())
    data['profiles'] = [row for row in data['profiles'] if row.get('example_only')]
    if failure == 'planner_collision':
        data['profiles'][0]['id'] = next(iter(common.planner_profiles()))
    elif failure == 'template_collision':
        data['profiles'].append(data['profiles'][0])
    else:
        data['profiles'][0]['support'] = {'status': 'SUPPORTED'}
    path = tmp_path / 'profiles.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(common, 'EXAMPLE_REGISTRY', path)
    with pytest.raises(common.ConfigError, match='duplicate|cannot declare planner support'):
        common.example_profiles()

def test_v3_factory_templates_are_runnable_without_inheriting_v1_entry(monkeypatch, capsys):
    common = _load_common()
    for filename in ('w8a16_action.toml', 'w8a16_action_adarms.toml', 'w8a16_pipeline.toml', 'w8a16_pipeline_high_fused_vector_qk.toml'):
        path = EXAMPLES / 'configs/rhinovla/v3' / filename
        monkeypatch.setattr(sys, 'argv', ['run.py', '--config', str(path), '--check-config'])
        from examples.common.cli import main
        assert main() == 0
    assert 'configuration OK' in capsys.readouterr().out

def test_example_navigation_links_resolve():
    for path in (EXAMPLES / 'README.md', EXAMPLES / 'README.zh.md', ROOT / 'docs/README.md'):
        for target in re.findall('\\]\\(([^)]+)\\)', path.read_text()):
            if '://' not in target and (not target.startswith('#')):
                assert (path.parent / target.split('#', 1)[0]).exists(), (path, target)

@pytest.mark.parametrize('setting', ['enabled = "true"', 'max_dumps = 0', 'output_dir = "/tmp/override"', 'max_dumps = true'])
def test_hw_profile_rejects_bad_values_and_arbitrary_paths(tmp_path, setting):
    common = _load_common()
    path = tmp_path / 'bad.toml'
    path.write_text((EXAMPLES / 'configs/qwen3/text/0_6b/fp16.toml').read_text() + '\n[runner.hw_profile]\n' + setting + '\n')
    with pytest.raises(common.ConfigError, match='hw_profile'):
        common.load_config(path)

@pytest.mark.parametrize('setting, message', [('[run]\nwarmup_decode_steps = 5', 'must not exceed'), ('[input]\nprompt_token_ids = [true, 2]', 'non-negative integers'), ('[input]\nprefill_fill_token_id = 248046', 'requires input.prefill_tokens'), ('[input]\nstop_on_eos = "false"', 'must be boolean')])
def test_fixed_length_benchmark_rejects_ambiguous_or_out_of_capacity_controls(tmp_path, setting, message):
    common = _load_common()
    source = (EXAMPLES / 'configs/qwen3_5/text/0_8b/fp16.toml').read_text()
    (table, value) = setting.split('\n', 1)
    path = tmp_path / 'benchmark.toml'
    path.write_text(source.replace(table, table + '\n' + value))
    with pytest.raises(common.ConfigError, match=message):
        common.load_config(path)

def test_cold_boolean_syntax_uses_shared_normalizer_without_model_import():
    common = _load_common()
    common._validate_execution({'components': {'vision_encoder': {'vision': {'fast_replay': True}}}})
    with pytest.raises(TypeError, match='must be bool'):
        common._validate_execution({'vision': {'fast_replay': 'true'}})

def test_release_example_config_uses_installed_leaf_not_protected_source(monkeypatch, tmp_path):
    common = _load_common()
    (bundle, installed) = (tmp_path / 'bundle', tmp_path / 'installed')
    bundle.mkdir()
    installed.mkdir()
    (bundle / 'examples').mkdir()
    (bundle / 'examples/common').mkdir()
    shutil.copy2(EXAMPLES / 'common/cpu.py', bundle / 'examples/common/cpu.py')
    monkeypatch.setattr(common, 'ROOT', bundle)
    monkeypatch.setattr(common, 'MODEL_REGISTRY', bundle / 'python/rpu_backend/model_registry.py')
    common.registry_aliases.cache_clear()
    files = ['rpu_backend/model_registry.py', 'rpu_backend/api/_execution.py']
    for relative in files:
        path = installed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'python' / relative, path)
    from types import SimpleNamespace
    dist = SimpleNamespace(files=files, locate_file=lambda path: installed / path)
    monkeypatch.setattr(common.metadata, 'distribution', lambda name: dist)
    config = common.load_config(EXAMPLES / 'configs/qwen3/text/0_6b/fp16.toml')
    assert config['example']['registry_alias'] == 'qwen3-0.6b'
    assert not (bundle / 'python').exists()
    with pytest.raises(ValueError, match='unknown field'):
        common._validate_execution({'prefill': {'unknown': 1}})
    common._backend_leaf.cache_clear()
    dist.files = []
    with pytest.raises(common.ConfigError, match='exactly one'):
        common._validate_execution({})

@pytest.fixture
def released_model_examples(tmp_path):
    import hashlib
    from importlib.machinery import EXTENSION_SUFFIXES
    (bundle, installed) = (tmp_path / 'release', tmp_path / 'site-packages')
    shutil.copytree(EXAMPLES, bundle / 'examples', ignore=shutil.ignore_patterns('__pycache__'))
    (bundle / 'release').mkdir()
    package = installed / 'rpu_backend'
    (package / 'api').mkdir(parents=True)
    (package / '__init__.py').write_text("raise AssertionError('backend was imported')\n")
    shutil.copy2(ROOT / 'python/rpu_backend/api/_execution.py', package / 'api/_execution.py')
    adapter = package / 'adapters/lingbot_vla_v2'
    adapter.mkdir(parents=True)
    (adapter / ('runtime' + EXTENSION_SUFFIXES[0])).write_bytes(b'compiled-runtime-identity')
    (adapter / 'prefix.py').write_text("raise AssertionError('prefix was imported')\n")
    artifacts = {}
    for (role, filename, payload) in (('rhino_ops_main', 'rhinoOpLib_current.ref', b'main-payload'), ('rhino_ops_expansion', 'rhinoExpansionOpLib_current.ref', b'expansion-payload')):
        (package / filename).write_bytes(payload)
        artifacts[role] = {'path': filename, 'sha256': hashlib.sha256(payload).hexdigest()}
    (bundle / 'release/runtime-dependencies.json').write_text(json.dumps({'artifacts': artifacts}))
    info = installed / 'rpu_backend-5.0.0.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: rpu_backend\nVersion: 5.0.0\n')
    (info / 'RECORD').write_text(''.join((f'{path.relative_to(installed).as_posix()},,\n' for path in package.rglob('*') if path.is_file())))
    (installed / 'sitecustomize.py').write_text("import builtins\n_import = builtins.__import__\ndef guarded(name, *args, **kwargs):\n    if name == 'torch' or name.startswith('torch.') or name == 'rpu_backend' or name.startswith('rpu_backend.'):\n        raise AssertionError('device import during cold resolution: ' + name)\n    return _import(name, *args, **kwargs)\nbuiltins.__import__ = guarded\n")
    env = {key: value for (key, value) in os.environ.items() if not key.startswith(('RPU_', 'LKN_'))}
    env.update(PYTHONPATH=str(installed), PYTHONDONTWRITEBYTECODE='1', RPU_KERNEL_LIB_PATH=str(package / 'rhinoOpLib_current.ref'), RPU_KERNEL_LIB_PATH_EXPANSION=str(package / 'rhinoExpansionOpLib_current.ref'))
    assert not (bundle / 'python').exists() and (not (bundle / 'src').exists())
    return (bundle, installed, env)

@pytest.mark.parametrize('relative', ('lingbot/v2/3cam_t160/w8a16.toml', 'pi05/2cam/w8a16.toml', 'wall_oss/base/2cam/256/w8a16.toml'))
def test_release_model_check_config_without_backend_sources(released_model_examples, relative):
    (bundle, _, env) = released_model_examples
    path = bundle / 'examples/configs' / relative
    result = subprocess.run([sys.executable, str(bundle / 'examples/run.py'), '--config', str(path), '--check-config'], cwd=bundle, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'configuration OK' in result.stdout

def test_release_lingbot_cold_dependencies_and_identity(released_model_examples):
    (bundle, installed, env) = released_model_examples
    code = "\nfrom examples.runners.lingbot2 import optimized as runner\nfrom examples.runners.pi05 import optimized as pi05\nenv, records = runner.profile_libraries()\nassert len(records) == 2\nassert pi05.profile_libraries()[0] == env\nrunner.bind_cold_environment(env)\nidentity = runner.source_identity()\nassert len(identity['source_sha256']) == 2\ninstalled = identity['installed_backend_artifacts']\nassert installed['adapters/lingbot_vla_v2/runtime.py']['kind'] == 'compiled_extension'\nassert installed['adapters/lingbot_vla_v2/prefix.py']['kind'] == 'python_source'\nassert all(len(item['sha256']) == 64 for item in installed.values())\nassert not identity.get('unavailable_backend_sources')\nprint('installed cold dependencies OK')\n"
    result = subprocess.run([sys.executable, '-c', code], cwd=bundle, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'installed cold dependencies OK' in result.stdout

def test_release_wall_execution_resolves_installed_refs_and_preserves_override_priority(released_model_examples):
    (bundle, _, env) = released_model_examples
    code = "\nimport os\nfrom pathlib import Path\nimport shutil\nfrom examples.common.libraries import current_libraries\nfrom examples.runners.wall_oss import base_engine as runner\nargs = runner.arguments(['--mode', 'candidate', '--precision', 'fp16',\n    '--fp16', '.', '--output-dir', 'outputs'])\ndefaults, _ = current_libraries(Path.cwd())\nactual = runner.cold_environment(args)\nassert all(actual[key] == value for key, value in defaults.items())\ncopy = Path('explicit-main.ref')\nshutil.copy2(defaults['RPU_KERNEL_LIB_PATH'], copy)\nargs.main_ref = copy\nos.environ['RPU_KERNEL_LIB_PATH'] = '/ignored-by-explicit-argument.ref'\nactual = runner.cold_environment(args)\nassert actual['RPU_KERNEL_LIB_PATH'] == str(copy.resolve())\nassert actual['RPU_KERNEL_LIB_PATH_EXPANSION'] == defaults['RPU_KERNEL_LIB_PATH_EXPANSION']\nprint('Wall installed defaults and argument priority OK')\n"
    result = subprocess.run([sys.executable, '-c', code], cwd=bundle, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Wall installed defaults and argument priority OK' in result.stdout

@pytest.mark.parametrize('override', ('argument', 'environment'))
def test_release_wall_execution_rejects_conflicting_payload_before_environment_mutation(released_model_examples, override):
    (bundle, _, env) = released_model_examples
    code = "\nimport os\nfrom pathlib import Path\nfrom examples.runners.wall_oss import base_engine as runner\nargs = runner.arguments(['--mode', 'candidate', '--precision', 'fp16',\n    '--fp16', '.', '--output-dir', 'outputs'])\nbad = Path('wrong-main.ref'); bad.write_bytes(b'unpublished-payload')\nif OVERRIDE == 'argument': args.main_ref = bad\nelse: os.environ['RPU_KERNEL_LIB_PATH'] = str(bad.resolve())\nos.environ['RPU_WALL_OSS_TEST_SENTINEL'] = 'unchanged'\nbefore = dict(os.environ)\ntry:\n    runner.cold_environment(args)\nexcept ValueError as error:\n    assert 'checksum mismatch' in str(error), error\nelse:\n    raise AssertionError('mismatched payload was accepted')\nassert dict(os.environ) == before\nprint('Wall invalid override rejected before effects')\n".replace('OVERRIDE', repr(override))
    result = subprocess.run([sys.executable, '-c', code], cwd=bundle, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Wall invalid override rejected before effects' in result.stdout

def test_release_lingbot_rejects_invalid_libraries(released_model_examples):
    (bundle, installed, env) = released_model_examples
    package = installed / 'rpu_backend'
    (package / 'rhinoOpLib_current.ref').write_bytes(b'wrong bytes')
    code = '\nfrom examples.runners.lingbot2 import optimized as runner\nenv, records = runner.profile_libraries()\nrunner.bind_cold_environment(env)\n'
    result = subprocess.run([sys.executable, '-c', code], cwd=bundle, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'checksum mismatch' in result.stderr, result.stderr

def test_check_config_runs_before_torch_or_rpu_backend_import(tmp_path):
    external_examples = tmp_path / 'examples'
    external_examples.mkdir()
    (external_examples / '__init__.py').write_text("raise RuntimeError('external examples package was selected')\n")
    (tmp_path / 'sitecustomize.py').write_text("import builtins\n_real_import = builtins.__import__\ndef _guard(name, *args, **kwargs):\n    if name == 'torch' or name.startswith('torch.') or name == 'rpu_backend' or name.startswith('rpu_backend.'):\n        raise RuntimeError(f'forbidden device import: {name}')\n    return _real_import(name, *args, **kwargs)\nbuiltins.__import__ = _guard\n", encoding='utf-8')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(tmp_path), str(ROOT)))
    for config in sorted((EXAMPLES / 'configs').rglob('*.toml')):
        import tomllib
        if 'env' in tomllib.loads(config.read_text()):
            continue
        result = subprocess.run([sys.executable, str(EXAMPLES / 'run.py'), '--config', str(config), '--check-config'], cwd=tmp_path, env=env, text=True, capture_output=True, check=False)
        assert result.returncode == 0, f'{config}: {result.stderr}'
        assert 'configuration OK' in result.stdout
    missing = subprocess.run([sys.executable, str(EXAMPLES / 'run.py')], cwd=tmp_path, env=env, text=True, capture_output=True, check=False)
    assert missing.returncode == 2
    assert '--config' in missing.stderr
    assert 'forbidden device import' not in missing.stderr

def test_example_modes_use_public_inference_and_exact_profile_metadata(monkeypatch):
    common = _load_common()
    profiles = common.example_profiles()
    for path in (EXAMPLES / 'configs').rglob('*.toml'):
        config = common.load_config(path)
        example = common.example_metadata(config)
        if example is None:
            continue
        assert common.required_example(profiles[example['profile_id']])
        assert 'input' in config
        assert 'rpu_execution' in config or 'pi05' in config
        assert 'config_dir' not in example
    from examples.common import cli, measurement
    received = []
    monkeypatch.setattr(measurement, 'run', lambda config: received.append(config) or 17)
    assert cli.main(['--config', str(EXAMPLES / 'configs/qwen3/text/0_6b/fp16.toml')]) == 17
    assert received[0]['example']['profile_id'] == 'qwen3-0.6b.fp16.text'

def test_required_profiles_cannot_hide_behind_eligible_false():
    common = _load_common()
    for status in ('SUPPORTED', 'SUPPORTED_LIMITED', 'CONTROLLED_EVALUATION', 'IMPLEMENTED_UNVERIFIED_HW', 'EXPERIMENTAL'):
        assert common.required_example({'support': {'status': status}, 'example': {'eligible': False}})
    for status in ('UNSUPPORTED', 'ASSET_ONLY'):
        assert not common.required_example({'support': {'status': status}, 'example': {'eligible': True}})

def test_internal_controlled_harness_does_not_hide_public_profiles():
    from copy import deepcopy
    common = _load_common()
    profile = {'support': {'status': 'CONTROLLED_EVALUATION'}, 'hw_status': 'IMPLEMENTED_UNVERIFIED_HW', 'input_envelope': {'kind': 'EXACT'}, 'example': {'reason_code': 'INTERNAL_CONTROLLED_EVALUATION', 'eligible': False, 'runner_target': None}, 'controlled_gates': [{'kind': 'env', 'name': 'RPU_TEST_CONTROLLED', 'value': '1'}]}
    assert not common.required_example(profile)
    for (group, key, value) in (('support', 'status', 'SUPPORTED'), ('input_envelope', 'kind', 'BOUNDED'), ('example', 'eligible', True)):
        bad = deepcopy(profile)
        bad[group][key] = value
        assert common.required_example(bad)
    for (key, value) in (('controlled_gates', []), ('hw_status', 'VERIFIED')):
        bad = deepcopy(profile)
        bad[key] = value
        assert common.required_example(bad)

def test_registry_alias_is_validated_but_not_promoted_to_support(tmp_path):
    source = (EXAMPLES / 'configs/qwen3/text/0_6b/fp16.toml').read_text(encoding='utf-8')
    bad_config = tmp_path / 'qwen3.toml'
    bad_config.write_text(source.replace('registry_alias = "qwen3-0.6b"', 'registry_alias = "qwen3-1.7b"'), encoding='utf-8')
    result = subprocess.run([sys.executable, str(EXAMPLES / 'run.py'), '--config', str(bad_config), '--check-config'], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert 'does not match profile' in result.stderr

def test_profile_metadata_is_the_matrix_support_authority(tmp_path, monkeypatch, example_catalog):
    common = _load_common()
    profiles = common.example_profiles()
    for config_path in (EXAMPLES / 'configs').rglob('*.toml'):
        config = common.load_config(config_path)
        example = common.example_metadata(config)
        if example is None:
            continue
        profile = profiles[example['profile_id']]
        for matrix in ('model_matrix.md', 'model_matrix.zh.md'):
            text = (ROOT / 'docs' / matrix).read_text(encoding='utf-8')
            rows = [line for line in text.splitlines() if f"`{profile['id']}`" in line]
            copied_path = example_catalog.CONFIGS / config_path.relative_to(EXAMPLES / 'configs')
            if not example_catalog.documented_config(copied_path):
                assert not rows, (profile['id'], matrix)
                continue
            assert len(rows) == 1, (profile['id'], matrix)
            row = rows[0]
            if profile.get('example_only'):
                assert 'Caller factory template; not evaluated' in row
                assert profile['id'] not in common.planner_profiles()
            else:
                assert f"`{profile['support']['status']}`" in row
                assert f"`{profile['support']['reason_code']}`" in row
            assert config_path.relative_to(ROOT).as_posix() in row
    source = (EXAMPLES / 'configs/qwen3/text/0_6b/fp16.toml').read_text(encoding='utf-8')
    from copy import deepcopy
    asset_only = deepcopy(profiles['qwen3-0.6b.fp16.text'])
    asset_only['id'] = 'synthetic.asset-only'
    asset_only['support']['status'] = 'ASSET_ONLY'
    asset_only['example']['eligible'] = False
    monkeypatch.setattr(common, 'example_profiles', lambda: profiles | {'synthetic.asset-only': asset_only})
    ineligible = tmp_path / 'ineligible.toml'
    ineligible.write_text(source.replace('profile_id = "qwen3-0.6b.fp16.text"', 'profile_id = "synthetic.asset-only"'))
    with pytest.raises(common.ConfigError, match='is not example.eligible=true'):
        common.load_config(ineligible)
    ineligible.write_text(source.replace('target = "qwen3"', 'target = "pi05"'))
    with pytest.raises(common.ConfigError, match='Pi0.5 uses \\[pi05\\]'):
        common.load_config(ineligible)

def test_qwen3vl_32b_controlled_api_example_keeps_exact_admission(tmp_path):
    common = _load_common()
    path = EXAMPLES / 'configs/qwen3_vl/vl/32b/w8a16_text.toml'
    config = common.load_config(path)
    profile = common.planner_profiles()[config['example']['profile_id']]
    assert profile['id'] == 'qwen3-vl-32b.w8a16.multimodal-controlled'
    assert profile['support']['status'] == 'CONTROLLED_EVALUATION'
    assert profile['hw_status'] == 'IMPLEMENTED_UNVERIFIED_HW'
    assert config['example']['registry_alias'] == 'qwen3-vl-32b-w8a16'
    assert config['example']['opt_in'] == {'QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED': '1'}
    source = path.read_text()
    for (old, new, reason) in (('QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED = "1"', '', 'exact opt_in'), ('[example]', '[example]\nevaluation_only = false', 'unsupported keys'), ('[example]', '[example]\nevaluation_only = true', 'unsupported keys'), ('registry_alias = "qwen3-vl-32b-w8a16"', 'registry_alias = "qwen3-vl-8b"', 'does not match profile')):
        bad = tmp_path / 'invalid.toml'
        bad.write_text(source.replace(old, new))
        with pytest.raises(common.ConfigError, match=reason):
            common.load_config(bad)

@pytest.mark.parametrize('cores', [4, 6, 8])
@pytest.mark.parametrize('file', ['qwen3/text/0_6b/fp16.toml', 'qwen3/text/1_7b/fp16.toml', 'qwen3/text/4b/fp16.toml', 'qwen3/text/8b/fp16.toml', 'qwen3_vl/vl/2b/fp16.toml', 'qwen3_vl/vl/4b/fp16.toml', 'qwen3_vl/vl/8b/fp16.toml', 'qwen3_5/vl/2b/fp16.toml', 'qwen3_5/vl/4b/fp16.toml', 'qwen3_5/vl/9b/fp16.toml', 'qwen3_5/text/2b/fp16.toml', 'qwen3_5/text/4b/fp16.toml', 'qwen3_5/text/9b/fp16.toml', 'qwen3_5/text/0_8b/fp16.toml', 'qwen3_5/vl/0_8b/fp16.toml'])
def test_qwen3_example_accepts_public_core_budget_without_importing_backend(tmp_path, cores, file):
    common = _load_common()
    path = tmp_path / 'cores.toml'
    path.write_text((EXAMPLES / 'configs' / file).read_text() + f'\n[rpu_execution.model]\nnum_cores = {cores}\n')
    config = common.load_config(path)
    assert config['rpu_execution']['model'] == {'num_cores': cores}

@pytest.mark.parametrize('file', ['qwen3_vl/text/2b/fp16.toml', 'qwen3_vl/vision/2b/fp16.toml', 'qwen3_vl/text/4b/fp16.toml', 'qwen3_vl/vision/4b/fp16.toml', 'qwen3_vl/text/8b/fp16.toml', 'qwen3_vl/vision/8b/fp16.toml', 'qwen3_vl/vl/32b/w8a16_text.toml'])
def test_qwen3vl_component_and_controlled_examples_accept_explicit_default_core_budget(tmp_path, file):
    common = _load_common()
    path = tmp_path / 'default-cores.toml'
    path.write_text((EXAMPLES / 'configs' / file).read_text() + '\n[rpu_execution.model]\nnum_cores = 8\n')
    config = common.load_config(path)
    assert config['rpu_execution']['model'] == {'num_cores': 8}

@pytest.mark.parametrize('file, setting', [('qwen3/text/0_6b/fp16.toml', 'num_cores = 5'), ('qwen3/text/0_6b/fp16.toml', 'attn_tp = 4'), ('qwen3/text/1_7b/fp16.toml', 'num_cores = true'), ('qwen3/text/0_6b/w8a16_text_head.toml', 'num_cores = 4'), ('llama/3_2_1b/fp16.toml', 'num_cores = 8'), ('qwen3_5/text/0_8b/w8a16_text.toml', 'num_cores = 6')] + [(file, f'num_cores = {cores}') for file in ('qwen3_vl/text/2b/fp16.toml', 'qwen3_vl/vision/2b/fp16.toml', 'qwen3_vl/text/4b/fp16.toml', 'qwen3_vl/vision/4b/fp16.toml', 'qwen3_vl/text/8b/fp16.toml', 'qwen3_vl/vision/8b/fp16.toml', 'qwen3_vl/vl/32b/w8a16_text.toml') for cores in (4, 6)])
def test_example_core_budget_rejects_unsupported_profile_or_role_knob(tmp_path, file, setting):
    common = _load_common()
    path = tmp_path / 'bad-cores.toml'
    path.write_text((EXAMPLES / 'configs' / file).read_text() + f'\n[rpu_execution.model]\n{setting}\n')
    with pytest.raises(common.ConfigError, match='num_cores|unknown field|unknown stage'):
        common.load_config(path)

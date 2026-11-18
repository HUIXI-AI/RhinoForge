import importlib.util
import hashlib
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
import pytest
ROOT = Path(__file__).resolve().parents[1]
VALIDATION = importlib.import_module('examples.runners.wall_oss.sorting_engine')

@pytest.fixture
def published_libraries(tmp_path):
    root = tmp_path / 'checkout'
    (root / 'src').mkdir(parents=True)
    (root / 'release').mkdir()
    (paths, artifacts) = ({}, {})
    for (role, filename) in (('main', 'rhinoOpLib_current.ref'), ('expansion', 'rhinoExpansionOpLib_current.ref')):
        path = root / 'src' / filename
        path.write_bytes(role.encode())
        paths[role] = path
        artifacts['rhino_ops_' + role] = {'path': filename, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / 'release/runtime-dependencies.json').write_text(json.dumps({'artifacts': artifacts}))
    paths['native'] = tmp_path / 'rpu_backend.so'
    paths['native'].write_bytes(b'native')
    return (root, paths)

@pytest.fixture(params=['base', 'sorting'])
def cold_caller(request):
    if request.param == "sorting":
        return lambda root, paths: VALIDATION.configure_exact_all_on(root, SimpleNamespace(
            mode="fp16", so=paths["native"], main_ref=paths["main"], expansion=paths["expansion"],
            num_cores=8))
    engine = importlib.import_module("examples.runners.wall_oss.base_engine")
    return lambda root, paths: engine.cold_environment(SimpleNamespace(
        mode="candidate", precision="fp16", components="action", vision_pair=False,
        action_residency=False, vision_window=False, so=paths["native"],
        main_ref=paths["main"], expansion_ref=paths["expansion"]), root)


def test_runner_rejects_inherited_unsafe_override_before_profile_mutation(monkeypatch):
    monkeypatch.setenv('RPU_CHUNK_FORCE_UNSAFE', '1')
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match='rejects RPU_CHUNK_FORCE_UNSAFE'):
        VALIDATION.configure_exact_all_on(ROOT, None)
    assert dict(os.environ) == before

@pytest.mark.parametrize('published', [True, False])
def test_wall_base_inherited_overrides_follow_published_roles(monkeypatch, tmp_path, published_libraries, published):
    (root, paths) = published_libraries
    copied = tmp_path / 'installed-expansion.ref'
    copied.write_bytes(paths['expansion'].read_bytes() if published else b'old-round')
    inherited = {'RPU_KERNEL_LIB_PATH': str(paths['main']), 'RPU_KERNEL_LIB_PATH_EXPANSION': str(copied)}
    monkeypatch.setattr(os, 'environ', dict(inherited))
    engine = vars(importlib.import_module('examples.runners.wall_oss.base_engine'))
    args = SimpleNamespace(mode='candidate', precision='fp16', components='action', vision_pair=False, action_residency=False, vision_window=False, so=paths['native'], main_ref=None, expansion_ref=None)
    if published:
        configured = engine['cold_environment'](args, root)
        assert configured['RPU_KERNEL_LIB_PATH_EXPANSION'] == str(copied)
    else:
        with pytest.raises(ValueError, match='checksum mismatch'):
            engine['cold_environment'](args, root)
        assert dict(os.environ) == inherited

def test_runner_accepts_disabled_unsafe_override(monkeypatch, tmp_path, published_libraries):
    monkeypatch.setattr(os, 'environ', {'RPU_CHUNK_FORCE_UNSAFE': '0'})
    (root, paths) = published_libraries
    args = VALIDATION.parse_args(['--mode', 'fp16', '--output-dir', str(tmp_path / 'results'), '--fp16', str(tmp_path / 'assets'), '--golden', str(tmp_path / 'golden.pt'), '--so', str(paths['native']), '--main-ref', str(paths['main']), '--expansion', str(paths['expansion']), '--backend-root', str(root)])
    configured = VALIDATION.configure_exact_all_on(root, args)
    assert configured['RPU_WALL_OSS_FP16_TRANSFER_MASK'] == '7'
    assert os.environ['RPU_CHUNK_FORCE_UNSAFE'] == '0'

def test_cold_callers_accept_published_installed_copies(monkeypatch, tmp_path, published_libraries, cold_caller):
    (root, paths) = published_libraries
    monkeypatch.setattr(os, 'environ', {})
    for role in ('main', 'expansion'):
        copy = tmp_path / (role + '.ref')
        copy.write_bytes(paths[role].read_bytes())
        paths[role].unlink()
        paths[role] = copy
    configured = cold_caller(root, paths)
    assert configured['RPU_KERNEL_LIB_PATH'] == str(paths['main'])
    assert configured['RPU_KERNEL_LIB_PATH_EXPANSION'] == str(paths['expansion'])
    assert os.environ['RPU_KERNEL_LIB_PATH'] == str(paths['main'])

@pytest.mark.parametrize('fault', ['main', 'expansion', 'swapped', 'missing_manifest'])
def test_cold_callers_reject_unpublished_roles_before_environment_mutation(monkeypatch, tmp_path, published_libraries, cold_caller, fault):
    (root, paths) = published_libraries
    monkeypatch.setattr(os, 'environ', {'RPU_WALL_OSS_INHERITED': 'retain', 'RPU_RMSNORM_VWARP': 'retain'})
    if fault == 'missing_manifest':
        (root / 'release/runtime-dependencies.json').unlink()
    elif fault == 'swapped':
        (paths['main'], paths['expansion']) = (paths['expansion'], paths['main'])
    else:
        old = tmp_path / 'old-round.ref'
        old.write_bytes(b'old-round')
        paths[fault] = old
    before = dict(os.environ)
    error = FileNotFoundError if fault == 'missing_manifest' else ValueError
    with pytest.raises(error, match='runtime-dependencies' if fault == 'missing_manifest' else 'checksum mismatch'):
        cold_caller(root, paths)
    assert dict(os.environ) == before

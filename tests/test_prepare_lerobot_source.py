"""Admission and source integrity for the private LeRobot compatibility copy."""
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from scripts import prepare_lerobot_source as prepare


def test_private_install_rejects_shared_base_and_foreign_package(tmp_path, monkeypatch):
    site = tmp_path / 'private/lib/site-packages'
    site.mkdir(parents=True)
    monkeypatch.setattr(prepare.sys, 'prefix', str(tmp_path / 'private'))
    monkeypatch.setattr(prepare.sys, 'base_prefix', str(tmp_path / 'base'))
    monkeypatch.setattr(prepare.sysconfig, 'get_path', lambda name: str(site))
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '5.19.0')
    monkeypatch.setattr(importlib.metadata, 'distribution', lambda name: SimpleNamespace(
        version='0.5.1', locate_file=lambda name: site))
    assert prepare.private_site(site) == site
    with pytest.raises(ValueError, match='private virtual environment'):
        prepare.private_site(tmp_path / 'base')
    monkeypatch.setattr(prepare.sys, 'prefix', prepare.sys.base_prefix)
    with pytest.raises(ValueError, match='private virtual environment'):
        prepare.private_site(site)
    monkeypatch.setattr(prepare.sys, 'prefix', str(tmp_path / 'private'))
    monkeypatch.setattr(importlib.metadata, 'distribution', lambda name: SimpleNamespace(
        version='0.5.1', locate_file=lambda name: tmp_path / 'base'))
    with pytest.raises(ValueError, match='official lerobot'):
        prepare.private_site(site)


@pytest.mark.parametrize('fault', [None, 'extra', 'modified', 'missing', 'symlink'])
def test_installed_package_rejects_unknown_source_before_replacement(tmp_path, fault):
    wheel = tmp_path / 'source.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('lerobot/__init__.py', 'version = "0.5.1"\n')
        archive.writestr('lerobot/model.py', 'value = 1\n')
    package = tmp_path / 'installed/lerobot'
    package.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(package.parent)
        if fault == 'extra':
            (package / 'unknown.py').write_text('value = 2\n')
        elif fault == 'modified':
            (package / 'model.py').write_text('value = 2\n')
        elif fault == 'missing':
            (package / 'model.py').unlink()
        elif fault == 'symlink':
            (package / 'model.py').unlink()
            (package / 'model.py').symlink_to(package / '__init__.py')
        if fault is None:
            prepare.verify_package(package, archive)
        else:
            with pytest.raises(ValueError, match='LeRobot'):
                prepare.verify_package(package, archive)

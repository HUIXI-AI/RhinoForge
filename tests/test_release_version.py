"""Release metadata must not accept a mismatched stage or runtime package."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_version", ROOT / "scripts/check_release_version.py")
version_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_check)


@pytest.fixture
def source(tmp_path):
    (tmp_path / "release").mkdir()
    for name in ("VERSION", "pyproject.toml", "release/runtime-dependencies.json", "release/version.cmake"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    return tmp_path


def set_version(source, version):
    (source / "VERSION").write_text(version + "\n")
    pyproject = source / "pyproject.toml"
    classifier = "3 - Alpha" if ".dev" in version else "4 - Beta" if "rc" in version else "5 - Production/Stable"
    pyproject.write_text(re.sub(r'Development Status :: [^\"]+', "Development Status :: " + classifier, pyproject.read_text()))
    path = source / "release/runtime-dependencies.json"
    dependencies = json.loads(path.read_text())
    launch = dependencies["launch_kernel"]
    launch["release_version"] = version
    launch["version"] = launch["cmake_version"] = version_check.VERSION_PATTERN.fullmatch(version)["base"]
    path.write_text(json.dumps(dependencies))


@pytest.mark.parametrize("version,branch,tag", [
    ("1.2.0rc1", "release/1.2", "v1.2.0rc1"),
    ("1.2.0", "release/1.2", "v1.2.0"),
    ("1.2.1", "release/1.2", "v1.2.1"),
    ("1.3.0.dev0", "main", ""),
])
def test_supported_versions_match_cmake(source, version, branch, tag):
    set_version(source, version)
    result = version_check.check(source, branch=branch, tag=tag)
    driver = source / "check.cmake"
    driver.write_text('include("${CMAKE_CURRENT_LIST_DIR}/release/version.cmake")\n'
                      'file(WRITE "${CMAKE_CURRENT_LIST_DIR}/result.txt" '
                      '"${RHINOFORGE_VERSION}|${RHINOFORGE_CMAKE_VERSION}|${RHINOFORGE_VERSION_STAGE}")\n')
    subprocess.run(["cmake", "-P", str(driver)], check=True)
    assert (source / "result.txt").read_text() == f'{version}|{result["cmake_version"]}|{result["stage"] if result["stage"] != "stable" else ""}'


@pytest.mark.parametrize("version,branch,tag", [
    ("1.2.0", "main", ""),
    ("1.3.0.dev0", "release/1.3", ""),
    ("1.2.0rc1", "release/1.3", ""),
    ("1.2.0rc1", "", "v1.2.0"),
    ("1.3.0.dev0", "", "v1.3.0.dev0"),
])
def test_rejects_wrong_branch_or_tag(source, version, branch, tag):
    set_version(source, version)
    with pytest.raises(ValueError):
        version_check.check(source, branch=branch, tag=tag)


def test_rejects_inconsistent_launch_versions(source):
    set_version(source, "1.2.0rc1")
    path = source / "release/runtime-dependencies.json"
    dependencies = json.loads(path.read_text())
    dependencies["launch_kernel"]["release_version"] = "2.0.0"
    path.write_text(json.dumps(dependencies))
    with pytest.raises(ValueError, match="Launch release and CMake"):
        version_check.check(source)
    failed = subprocess.run(["cmake", "-P", str(source / "release/version.cmake")], capture_output=True)
    assert failed.returncode != 0


def test_existing_build_reconfigures_after_version_or_lock_changes(source):
    set_version(source, "1.2.0rc1")
    (source / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.19)\n'
        'include(release/version.cmake)\n'
        'project(VersionCheck VERSION ${RHINOFORGE_CMAKE_VERSION} LANGUAGES NONE)\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/version.txt" "${RHINOFORGE_VERSION}")\n')
    build = source / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True)
    # Makefile generators may compare timestamps at one-second resolution.
    time.sleep(1.05)
    set_version(source, "1.2.0rc2")
    subprocess.run(["cmake", "--build", str(build)], check=True)
    assert (build / "version.txt").read_text() == "1.2.0rc2"
    time.sleep(1.05)
    path = source / "release/runtime-dependencies.json"
    dependencies = json.loads(path.read_text())
    dependencies["launch_kernel"]["release_version"] = "2.0.0"
    path.write_text(json.dumps(dependencies))
    failed = subprocess.run(["cmake", "--build", str(build)], capture_output=True)
    assert failed.returncode != 0


def test_source_and_installed_package_resolve_their_own_version(tmp_path):
    source_version = subprocess.check_output(
        [sys.executable, "-c", "import runpy; print(runpy.run_path('python/rpu_backend/_version.py')['DISTRIBUTION_VERSION'])"],
        cwd=ROOT, text=True).strip()
    assert source_version == (ROOT / "VERSION").read_text().strip()
    package = tmp_path / "site-packages/rpu_backend"
    package.mkdir(parents=True)
    shutil.copyfile(ROOT / "python/rpu_backend/_version.py", package / "_version.py")
    metadata = package.parent / "rhinoforge-1.2.0rc1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: rhinoforge\nVersion: 1.2.0rc1\n")
    installed_version = subprocess.check_output([
        sys.executable, "-I", "-c",
        "import runpy, sys; sys.path.insert(0, sys.argv[1]); "
        "print(runpy.run_path(sys.argv[2])['DISTRIBUTION_VERSION'])",
        str(package.parent), str(package / "_version.py"),
    ], text=True).strip()
    assert installed_version == "1.2.0rc1"


def test_main_can_pin_the_delivered_runtime120(source):
    path = source / "release/runtime-dependencies.json"
    dependencies = json.loads(path.read_text())
    dependencies["runtime_bundle"] = {"version": "1.2.0"}
    dependencies["launch_kernel"].update(version="1.0.0", cmake_version="1.0.0",
                                         release_version=None)
    path.write_text(json.dumps(dependencies))
    result = version_check.check(source, branch="main")
    assert result["version"] == "1.3.0.dev0"
    assert result["runtime_bundle_version"] == "1.2.0"
    subprocess.run(["cmake", "-P", str(source / "release/version.cmake")], check=True)


@pytest.mark.parametrize("fault", [None, "library", "header", "missing", "empty"])
def test_pinned_launch_package_rejects_mixed_or_missing_files(source, fault):
    prefix = source / "launch"
    files = {"include/rhino_launch_kernel.h": b"header", "lib/librhino_launch.so.1.0.0": b"library"}
    for name, value in files.items():
        target = prefix / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    package = prefix / "lib/cmake/rhino_launch"
    package.mkdir(parents=True)
    path = source / "release/runtime-dependencies.json"
    dependencies = json.loads(path.read_text())
    dependencies["launch_kernel"]["files"] = {
        name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    if fault == "empty":
        dependencies["launch_kernel"]["files"] = {}
    path.write_text(json.dumps(dependencies))
    if fault in ("header", "library"):
        target = prefix / ("include/rhino_launch_kernel.h" if fault == "header" else "lib/librhino_launch.so.1.0.0")
        target.write_bytes(b"different package")
    if fault == "missing":
        (prefix / "include/rhino_launch_kernel.h").unlink()
    driver = source / "package.cmake"
    driver.write_text('include("${CMAKE_CURRENT_LIST_DIR}/release/version.cmake")\n'
                      f'set(rhino_launch_DIR "{package}")\n'
                      'rhinoforge_check_launch_package()\n')
    result = subprocess.run(["cmake", "-P", str(driver)], capture_output=True, text=True)
    if fault is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and "Launch package" in result.stderr


@pytest.mark.parametrize("field,value", [("files", {}), ("files", {"../outside": "a" * 64}),
                                         ("files", {"include/version.h": "not-a-digest"}),
                                         ("version", "1.3.0")])
def test_rejects_invalid_launch_lock(source, field, value):
    path = source / "release/runtime-dependencies.json"
    dependencies = json.loads(path.read_text())
    dependencies["launch_kernel"][field] = value
    path.write_text(json.dumps(dependencies))
    with pytest.raises(ValueError, match="Launch"):
        version_check.check(source)

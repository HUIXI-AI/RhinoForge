import gc
import pytest
_ALLOWED_SKIPS = {'test_responsibility_boundaries.py::test_components_no_outward_imports': 'Retired component directory is absent.'}

def _skip_key(nodeid: str) -> str:
    parts = nodeid.split('::')
    return f"{parts[0].split('/')[-1]}::{parts[-1].split('[')[0]}"

def pytest_configure(config):
    config.addinivalue_line('markers', 'hardware: requires Rhino RPU (skipped in CI / non-hardware runs)')
    config._acceptance_skipped = set()
    config._acceptance_ran = set()

def pytest_runtest_logreport(report):
    if not report.nodeid.startswith("python/rpu_backend/tests/acceptance/"):
        return
    config = _CONFIG[0]
    if config is None:
        return
    if report.skipped and report.when in ('setup', 'call'):
        config._acceptance_skipped.add(_skip_key(report.nodeid))
    elif report.when == 'call' and report.passed:
        config._acceptance_ran.add(_skip_key(report.nodeid))
_CONFIG = [None]

def pytest_sessionstart(session):
    _CONFIG[0] = session.config

def pytest_sessionfinish(session, exitstatus):
    config = session.config
    skipped = getattr(config, '_acceptance_skipped', set())
    ran = getattr(config, '_acceptance_ran', set())
    problems = []
    for key in sorted(skipped - set(_ALLOWED_SKIPS)):
        problems.append(f'  UNLISTED SKIP    {key}\n      A skip is green. Either make it run, or add it to _ALLOWED_SKIPS in this conftest WITH the reason it may not.')
    for key in sorted(set(_ALLOWED_SKIPS) & ran):
        problems.append(f'  STALE EXEMPTION  {key}\n      Listed as an allowed skip but it RAN. Delete the entry, so the next silent skip is visible again.')
    if problems:
        print('\n[acceptance skip allowlist] FAILED:\n' + '\n'.join(problems))
        session.exitstatus = 1

@pytest.fixture(autouse=True)
def _reset_live_ref():
    import rpu_backend.api.causal_lm as _cl
    from rpu_backend.api import _execution
    _cl._LIVE_REF = None
    _cl._LIVE_TERMINAL_REASON = None
    _execution._UNSAFE_PROCESS_REASON = None
    gc.collect()
    yield
    _cl._LIVE_REF = None
    _cl._LIVE_TERMINAL_REASON = None
    _execution._UNSAFE_PROCESS_REASON = None
    gc.collect()

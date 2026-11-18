"""Public CLI logging with fake inference; no Torch, SDK, weights or device."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, source, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT), TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    return subprocess.run([sys.executable, "-c", textwrap.dedent(source), str(tmp_path), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)


_CLI = r'''
import json, logging, os, sys
from pathlib import Path
from types import SimpleNamespace
from examples.common import cli, results
root = Path(sys.argv[1])
level, summary = sys.argv[2], sys.argv[3] == "true"
config = root / "case.toml"
config.write_text('[lingbot2]\nprecision="w8a16"\n[input]\n'
    'checkpoint="/models/lingbot"\ntraining_config="/inputs/train.yaml"\n'
    'case="/inputs/case"\ncameras=3\nexpected_language_tokens=3\n'
    '[run]\nwarmup=2\nruns=5\noutput_dir=' + json.dumps(str(root / "results")) + '\n'
    '[logging]\nlevel=' + json.dumps(level) + '\nsummary=' + str(summary).lower() + '\n')
def run(config):
    assert "torch" not in sys.modules and "rpu_backend" not in sys.modules
    expected = {"ERROR": ("1", "4"), "INFO": ("3", "2"), "DEBUG": ("4", "0")}[level]
    assert (os.environ["RPU_LOG_LEVEL"], os.environ["LKN_LOG_LEVEL"]) == expected
    assert config["run"]["profile"] is True
    print("python print detail")
    logging.getLogger("third.party").warning("CPU fallback detail")
    logging.getLogger("examples.fake").debug("Graph debug detail")
    os.write(1, b"RpuInit 177: Rpu api version: fake SDK\n")
    os.write(2, b"[WARN] native warning detail\n")
    os.write(2, b"[RpuKernelGraph] WARN: native Graph fallback detail\n")
    os.write(2, b"[TRACE] native trace detail\n")
    logging.getLogger("examples.fake").error("recoverable error visible")
    report = {"model": "lingbot2"}
    with results.record_result(report, root / "results",
            timing=lambda: results.latency_timing([20., 30.], "fake inference; CPU actions included")):
        pass
    # A repeated request to publish this same report must not duplicate it.
    from examples.common.logging import emit_performance
    emit_performance(config, report)
    return 0
cli._lingbot2_runner = lambda: SimpleNamespace(validate_example_config=lambda _: None,
                                              run_example_config=run)
assert cli.main(["--config", str(config), "--profile"]) == 0
os.write(1, b"AFTER_SESSION\n")
'''


@pytest.mark.parametrize("level", ["ERROR", "INFO", "DEBUG"])
def test_cli_level_config_and_single_performance_summary(tmp_path, level):
    process = _run(tmp_path, _CLI, level, "true")
    assert process.returncode == 0, process.stderr
    output = process.stdout + process.stderr
    assert "[MODEL] model=lingbot2; precision=w8a16" in output
    assert '"profile": true' in output  # CLI override already applied at summary time.
    assert output.count("[PERF] VLA E2E: median=25.000 ms") == 1
    assert "report=" in output and "AFTER_SESSION" in output
    assert "recoverable error visible" in output
    for detail in ("python print detail", "CPU fallback detail", "fake SDK", "native warning detail", "native Graph fallback detail"):
        assert (detail in output) == (level != "ERROR")
    assert ("Graph debug detail" in output) == (level == "DEBUG")
    assert ("native trace detail" in output) == (level == "DEBUG")
    assert len(list((tmp_path / "results").glob("native-*.log"))) == 2
    assert (tmp_path / "results/report.json").is_file()


def test_summary_switch_does_not_hide_errors(tmp_path):
    process = _run(tmp_path, _CLI, "ERROR", "false")
    assert process.returncode == 0, process.stderr
    output = process.stdout + process.stderr
    assert "[MODEL]" not in output and "[PERF]" not in output
    assert "recoverable error visible" in output


@pytest.mark.parametrize("level", ["ERROR", "INFO"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_model_root_handler_can_write_to_captured_stream(tmp_path, level, stream):
    process = _run(tmp_path, r'''
        import logging, os, sys, warnings
        from examples.common.logging import logging_session
        level, stream = sys.argv[2:]
        root = logging.getLogger()
        original = logging.StreamHandler(sys.stderr)
        root.handlers, root.level = [original], logging.WARNING
        with logging_session({"run": {"output_dir": sys.argv[1]},
                              "logging": {"level": level, "summary": False}}):
            # Model imports may replace the root handler after capture starts.
            handler = logging.StreamHandler(getattr(sys, stream))
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            root.handlers, root.level = [handler], logging.DEBUG
            print("model initialization detail")
            root.error("model error visible")
            warnings.warn("model warning detail")
            os.write(2, b"[ERROR] native error visible\n")
        assert root.handlers == [original] and root.level == logging.WARNING
        root.error("logging restored")
    ''', level, stream)
    assert process.returncode == 0, process.stderr
    output = process.stdout + process.stderr
    for message in ("model error visible", "native error visible", "logging restored"):
        assert output.count(message) == 1
    assert ("model initialization detail" in output) == (level == "INFO")
    assert ("model warning detail" in output) == (level == "INFO")


def test_state_restored_and_primary_exception_preserved(tmp_path):
    process = _run(tmp_path, r'''
        import logging, os, sys, warnings
        from pathlib import Path
        from examples.common.logging import logging_session
        root = logging.getLogger()
        handler = logging.StreamHandler()
        root.handlers, root.level = [handler], logging.WARNING
        logging.captureWarnings(True)
        old_warning = warnings.showwarning
        old_streams = sys.stdout, sys.stderr
        os.environ.update(RPU_LOG_LEVEL="5", LKN_LOG_LEVEL="0")
        config = {"run": {"output_dir": sys.argv[1]}, "logging": {"summary": False}}
        try:
            with logging_session(config):
                print("hidden")
                raise RuntimeError("original model failure")
        except RuntimeError as error:
            assert str(error) == "original model failure"
        assert (sys.stdout, sys.stderr) == old_streams
        assert warnings.showwarning is old_warning
        assert root.handlers == [handler] and root.level == logging.WARNING
        assert os.environ["RPU_LOG_LEVEL"] == "5" and os.environ["LKN_LOG_LEVEL"] == "0"
        warnings.warn("warning restored")
        os.write(1, b"restored\n")
    ''')
    assert process.returncode == 0, process.stderr
    assert process.stdout == "restored\n"
    assert "warning restored" in process.stderr


def test_start_failure_retains_original_io_error(tmp_path):
    process = _run(tmp_path, r'''
        import os, sys
        from pathlib import Path
        from examples.common.logging import logging_session
        path = Path(sys.argv[1]) / "existing-file"
        path.write_text("unchanged")
        try:
            with logging_session({"run": {"output_dir": str(path)}}):
                raise AssertionError("must not reach inference")
        except FileExistsError:
            pass
        else:
            raise AssertionError("primary IO error lost")
        assert path.read_text() == "unchanged"
        os.write(1, b"console restored\n")
    ''')
    assert process.returncode == 0, process.stderr
    assert process.stdout == "console restored\n"


def test_script_capture_covers_backend_atexit(tmp_path):
    process = _run(tmp_path, r'''
        import atexit, os, sys
        from examples.common.logging import logging_session
        config = {"run": {"output_dir": sys.argv[1]}, "logging": {"summary": False}}
        with logging_session(config, keep_until_exit=True):
            # Same registration order as the backend imported inside the run.
            atexit.register(lambda: os.write(1, b"heap name: hx_mem, close fd: 7\n"))
            atexit.register(lambda: os.write(2, b"[ERROR] cleanup failure visible\n"))
    ''')
    assert process.returncode == 0, process.stderr
    assert "heap name" not in process.stdout + process.stderr
    assert "cleanup failure visible" in process.stderr
    raw = "".join(path.read_text() for path in tmp_path.glob("native-*.log"))
    assert "heap name" in raw and "cleanup failure visible" in raw


def test_python_atexit_exception_is_visible_at_default_level(tmp_path):
    process = _run(tmp_path, r'''
        import atexit, sys
        from examples.common.logging import logging_session
        def cleanup():
            raise RuntimeError("backend teardown failed")
        with logging_session({"run": {"output_dir": sys.argv[1]}}, keep_until_exit=True):
            atexit.register(cleanup)
    ''')
    assert "Exception ignored in atexit callback" in process.stderr
    assert "Traceback (most recent call last)" in process.stderr
    assert "RuntimeError: backend teardown failed" in process.stderr


def test_stream_flush_error_still_restores_native_descriptors(tmp_path):
    process = _run(tmp_path, r'''
        import os, sys
        from examples.common.logging import logging_session
        def bad_flush():
            raise OSError("Python stream flush failed")
        try:
            with logging_session({"run": {"output_dir": sys.argv[1]}, "logging": {"summary": False}}):
                sys.stdout.flush = bad_flush
                raise LookupError("original failure")
        except LookupError as error:
            assert str(error) == "original failure"
            assert any("Python stream flush failed" in note for note in error.__notes__)
        os.write(1, b"console restored after failed flush\n")
    ''')
    assert process.returncode == 0, process.stderr
    assert process.stdout == "console restored after failed flush\n"


@pytest.mark.parametrize("family,relative", [
    ("wall_oss", "wall_oss/base/3cam/224/w8a16.toml"),
    ("wall_oss_sorting", "wall_oss/sorting/w8a16.toml"),
])
def test_cli_preserves_wall_run_schema(tmp_path, family, relative):
    process = _run(tmp_path, r'''
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        from examples.common import cli, config as common
        family, relative = sys.argv[2:]
        path = "examples/configs/" + relative
        config = common.load_config(Path(path))
        config["run"]["output_dir"] = sys.argv[1]
        actual = common._wall_base_runner() if family == "wall_oss" else common._wall_sorting_runner()
        def run(value):
            actual.validate_example_config(value)
            assert "hwperf" not in value["run"] and "profile" not in value["run"]
            return 0
        cli.load_config = lambda _, **__: config
        setattr(cli, "_wall_base_runner" if family == "wall_oss" else "_wall_sorting_runner",
                lambda: SimpleNamespace(run_example_config=run))
        assert cli.main(["--config", path]) == 0
    ''', family, relative)
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize("family", ["base", "sorting"])
@pytest.mark.parametrize("existing", ["", "actions.pt", "native-old.stdout.log"])
def test_wall_cli_reaches_engine_with_only_current_session_logs(tmp_path, family, existing):
    process = _run(tmp_path, r'''
        import sys
        from pathlib import Path
        from examples.common import cli, config as common
        from examples.runners.wall_oss import base_engine, sorting
        root = Path(sys.argv[1])
        family, existing = sys.argv[2:]
        relative = "base/2cam/224/fp16.toml" if family == "base" else "sorting/fp16.toml"
        path = Path("examples/configs/wall_oss") / relative
        config = common.load_config(path)
        output = root / "results"
        config["run"]["output_dir"] = str(output)
        config["logging"] = {"summary": False}
        if existing:
            output.mkdir()
            (output / existing).write_text("user content")
        if family == "base":
            batch = root / "batch.pt"
            batch.touch()
            config["input"].pop("batch_template", None)
            config["input"].update(fp16_checkpoint=str(root), batch=str(batch))
        class EngineReached(Exception):
            pass
        def reached(*args):
            assert len(list(output.glob("native-*.log"))) == 2
            raise EngineReached
        # Keep the real CLI, config validation and output-directory preflight.
        base_engine.main = reached
        sorting.cold_environment = reached
        cli.load_config = lambda _, **__: config
        try:
            cli.main(["--config", str(path)])
        except EngineReached:
            assert not existing
        except SystemExit as error:
            assert existing and error.code == 2
            assert (output / existing).read_text() == "user content"
        else:
            raise AssertionError("engine preflight was skipped")
    ''', family, existing)
    assert process.returncode == 0, process.stderr
    if existing:
        assert "output directory must be empty" in process.stderr


def test_output_directory_rejects_other_sessions_and_log_aliases(tmp_path):
    process = _run(tmp_path, r'''
        import sys
        from pathlib import Path
        from examples.common.logging import logging_session, prepare_output_directory
        root = Path(sys.argv[1])
        output = root / "results"
        assert prepare_output_directory(output) == output
        with logging_session({"run": {"output_dir": str(output)}, "logging": {"summary": False}}) as session:
            assert prepare_output_directory(output) == output
            alias = output / "user-alias.log"
            alias.symlink_to(session.native.paths["stdout"])
            try:
                prepare_output_directory(output)
            except FileExistsError:
                pass
            else:
                raise AssertionError("a user alias is not a session-owned file")
            alias.unlink()
        for active in (False, True):
            def reject():
                try:
                    prepare_output_directory(output)
                except FileExistsError:
                    return
                raise AssertionError("prior-session logs must not be accepted")
            if active:
                with logging_session({"run": {"output_dir": str(root / "other")}, "logging": {"summary": False}}):
                    reject()
            else:
                reject()
    ''')
    assert process.returncode == 0, process.stderr


def test_argparse_error_visible_at_default_level(tmp_path):
    process = _run(tmp_path, r'''
        import argparse, sys
        from examples.common.logging import logging_session
        with logging_session({"run": {"output_dir": sys.argv[1]}, "logging": {"summary": False}}):
            parser = argparse.ArgumentParser(prog="example")
            parser.add_argument("--required", required=True)
            parser.parse_args([])
    ''')
    assert process.returncode == 2
    assert "usage: example" in process.stderr
    assert "example: error: the following arguments are required: --required" in process.stderr


def test_unmarked_stderr_retains_default_filtering(tmp_path):
    process = _run(tmp_path, r'''
        import sys
        from examples.common.logging import logging_session
        with logging_session({"run": {"output_dir": sys.argv[1]}, "logging": {"summary": False}}):
            print("The image processor now uses the fast implementation.", file=sys.stderr)
            print("[WARNING] processor option changed", file=sys.stderr)
            print("[ERROR] model input failed", file=sys.stderr)
    ''')
    assert process.returncode == 0, process.stderr
    assert "processor" not in process.stderr
    assert "[ERROR] model input failed" in process.stderr


@pytest.mark.parametrize("level", ["ERROR", "INFO"])
def test_g05_spawned_row_uses_same_logging_policy(tmp_path, monkeypatch, level):
    from examples.runners.g05 import optimized

    actual_run = subprocess.run
    fake_model = '''
from examples.runners.g05 import optimized as runner
def fake(config):
    import logging, os
    from pathlib import Path
    from examples.common import results
    assert os.environ['RPU_LOG_LEVEL'] == ('1' if config['logging']['level'] == 'ERROR' else '3')
    print('child print detail')
    logging.getLogger('examples.child').info('child logger detail')
    report = {'status': 'COMPLETE', 'model': 'g05', 'config': config}
    results.add_result_fields(report, config['run']['output_dir'], completed=True,
        timing=results.latency_timing([1., 2.], 'fake child inference'))
    results.write_report(report, Path(config['run']['output_dir']))
    return 0
runner._run_single_example_config = fake
'''

    def run_child(command, **kwargs):
        # Execute the real child logging/bootstrap command, replacing only the
        # model function; still exercise JSON transport and normal process exit.
        return actual_run([*command[:-1], fake_model + command[-1]], cwd=ROOT,
                          timeout=20, **kwargs)

    monkeypatch.setattr(optimized.subprocess, "run", run_child)
    config = {"g05": {"precision": "w8a16"}, "input": {"text_tokens": 32},
              "run": {"output_dir": str(tmp_path)}, "logging": {"level": level}}
    optimized._run_row_process(config)
    output = (tmp_path / "stdout.log").read_text()
    assert "[MODEL] model=g05; precision=w8a16" in output
    assert "[PERF] VLA E2E: median=1.500 ms" in output
    for detail in ("child print detail", "child logger detail"):
        assert (detail in output) == (level == "INFO")

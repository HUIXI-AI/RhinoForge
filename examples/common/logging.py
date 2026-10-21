"""CLI-scoped diagnostics and model/performance summaries, without device imports.

Diagnostics obey [logging].level. Model/performance summaries have their own
switch so ERROR can be the quiet default without hiding benchmark results.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import warnings


_ACTIVE = None
_LEVELS = {
    "DEBUG": (logging.DEBUG, "4", "0"),
    "INFO": (logging.INFO, "3", "2"),
    "WARNING": (logging.WARNING, "2", "3"),
    "ERROR": (logging.ERROR, "1", "4"),
    "CRITICAL": (logging.CRITICAL, "1", "5"),
}
_TAG = re.compile(r"(?:\[\s*(DEBUG|TRACE|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\s*\]|"
                  r"(?:^|\]\s*)(DEBUG|TRACE|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL):)")


def _line_level(text):
    match = _TAG.search(text)
    if match:
        name = next(value for value in match.groups() if value)
        return {"TRACE": logging.DEBUG, "WARN": logging.WARNING,
                "FATAL": logging.CRITICAL}.get(name, getattr(logging, name, logging.INFO))
    if "CPU_FALLBACK" in text or "MANUAL_FALLBACK" in text:
        return logging.WARNING
    return logging.INFO


class _PrintLog(io.TextIOBase):
    """Adapt Python/third-party print calls; no monkeypatch of builtins.print."""

    def __init__(self, original, *, stderr=False):
        self.original = original
        self.pending = ""
        self.stderr = stderr
        self.traceback = False

    @property
    def encoding(self):
        return getattr(self.original, "encoding", None) or "utf-8"

    @property
    def errors(self):
        return "replace"

    @property
    def buffer(self):
        # Binary writers are handled by the native descriptor capture.
        return self.original.buffer

    def fileno(self):
        return self.original.fileno()

    def isatty(self):
        return False

    def writable(self):
        return True

    def write(self, value):
        if not isinstance(value, str):
            raise TypeError("write() argument must be str")
        self.pending += value.replace("\r", "\n")
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._emit(line)
        return len(value)

    def _emit(self, line):
        if self.stderr and line.startswith(("Traceback (most recent call last):",
                                             "Exception ignored in", "Fatal Python error:")):
            self.traceback = True
        if line.strip():
            argument_error = self.stderr and (line.startswith("usage: ") or ": error: " in line)
            level = logging.ERROR if self.traceback or argument_error else _line_level(line)
            logging.getLogger("examples.captured").log(level, "%s", line)

    def flush(self):
        if self.pending:
            line, self.pending = self.pending, ""
            self._emit(line)


def _summary_line(channel, message):
    logger = logging.getLogger("examples.summary")
    if _ACTIVE is not None:
        if _ACTIVE.options["summary"]:
            logger.info("%s", message, extra={"channel": channel})
        return
    # --check-config reports its result without installing a runtime session.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(channel)s] %(message)s"))
    record = logger.makeRecord(logger.name, logging.INFO, "", 0, message, (), None,
                               extra={"channel": channel})
    handler.handle(record)


def configuration_checked(message):
    _summary_line("CONFIG", message)


def prepare_output_directory(path):
    """Require an empty directory, apart from this session's native logs."""
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    owned = set()
    if _ACTIVE is not None and _ACTIVE.capturing and not _ACTIVE.closed:
        owned = set(_ACTIVE.native.paths.values())
    for entry in directory.resolve().iterdir():
        if entry not in owned or entry.is_symlink() or not entry.is_file():
            raise FileExistsError(f"output directory must be empty: {directory}")
    return directory


def emit_performance(config, report):
    """Publish a completed report once; callers retain the original JSON fields."""
    if (_ACTIVE is None or not _ACTIVE.options["summary"]
            or report.get("execution", {}).get("status") != "COMPLETED"
            or not report.get("timing")
            or any(item is report for item in _ACTIVE.reported)):
        return
    from examples.common.performance import summary
    _ACTIVE.reported.append(report)
    for line in summary(config or _ACTIVE.config, report):
        _summary_line("PERF", line)
    path = report.get("artifacts", {}).get("report")
    if path:
        _summary_line("PERF", f"report={path}")


def _model_summary(config):
    example = config.get("example", {})
    special = next((name for name in ("lingbot2", "pi05", "g05", "hy_vla",
                    "wall_oss", "wall_oss_sorting", "qwen3_asr", "qwen3_tts",
                    "whisper_asr") if name in config), None)
    profile = config.get(special, {}) if special else example
    model = special or example.get("profile_id", example.get("target", "model"))
    precision = profile.get("precision", example.get("dtype", "profile default"))
    _summary_line("MODEL", f"model={model}; precision={precision}")
    inputs = config.get("input", {})
    geometry = {key: value for key, value in inputs.items() if key in {
        "component", "cameras", "expected_language_tokens", "text_tokens",
        "prefill_tokens", "decode_steps", "image_size", "resolution", "batch_size",
        "horizon", "num_steps", "stop_on_eos", "source"}}
    if "images" in inputs:
        geometry["images"] = len(inputs["images"])
    selected = {"input": geometry, "model_options": profile,
                "rpu_execution": config.get("rpu_execution", {}),
                "cpu": config.get("cpu", {}), "run": config.get("run", {})}
    _summary_line("MODEL", json.dumps(selected, ensure_ascii=False, sort_keys=True, default=str))
    checkpoint = config.get("_model_override") or inputs.get("checkpoint", example.get("registry_alias"))
    if checkpoint:
        _summary_line("MODEL", f"checkpoint={checkpoint}")


class _Session:
    def __init__(self, config):
        from examples.common.config import validate_logging_config
        self.config = config
        self.options = validate_logging_config(config)
        self.reported = []
        self.closed = False
        self.native = None
        self.capturing = False
        self.streams = []
        self.handlers = []
        self.logger_states = []
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_showwarning = warnings.showwarning
        self.old_env = {name: os.environ.get(name) for name in ("RPU_LOG_LEVEL", "LKN_LOG_LEVEL")}
        self.old_native_level = None

    def start(self):
        global _ACTIVE
        from examples.common.native_logging import NativeCapture
        level, rpu, lkn = _LEVELS[self.options["level"]]
        os.environ.update(RPU_LOG_LEVEL=rpu, LKN_LOG_LEVEL=lkn)
        # Existing imports are supported for Python/native backend levels. The
        # SDK reads its environment only at startup; normal CLI starts cold.
        native = sys.modules.get("rpu_backend._cpp_ext")
        if native is not None and hasattr(native, "rpu_get_debug_level"):
            self.old_native_level = native.rpu_get_debug_level()
            native.rpu_set_debug_level(int(rpu))
        for fd in (2, 1):
            self.streams.append(os.fdopen(os.dup(fd), "w", encoding="utf-8", errors="replace", buffering=1))
        diagnostic = logging.StreamHandler(self.streams[0])
        diagnostic.setLevel(level)
        diagnostic.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        summary_handler = logging.StreamHandler(self.streams[1])
        summary_handler.setFormatter(logging.Formatter("[%(channel)s] %(message)s"))
        self.handlers = [diagnostic, summary_handler]
        # Keep pre-existing handlers/levels intact outside this CLI invocation.
        loggers = [logging.getLogger(), *[item for item in logging.root.manager.loggerDict.values()
                   if isinstance(item, logging.Logger)]]
        for logger in loggers:
            self.logger_states.append((logger, logger.level, logger.handlers[:], logger.propagate, logger.disabled))
            logger.handlers = []
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            logger.disabled = False
        root = logging.getLogger()
        root.handlers = [diagnostic]
        root.setLevel(level)
        # Model imports can replace root handlers with captured stdout/stderr.
        # Send capture records directly to the saved descriptor to avoid a loop.
        for name in ("examples.captured", "examples.native", "py.warnings"):
            logger = logging.getLogger(name)
            logger.handlers = [diagnostic]
            logger.setLevel(level)
            logger.propagate = False
        logger = logging.getLogger("examples.summary")
        logger.handlers = [summary_handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: (
            logging.getLogger("py.warnings").warning("%s", warnings.formatwarning(
                message, category, filename, lineno, line).rstrip()))
        directory = Path(self.config.get("run", {}).get("output_dir", "perf_results/examples")).expanduser()
        self.native = NativeCapture(directory)
        self.native.start()
        self.capturing = True
        sys.stdout, sys.stderr = _PrintLog(self.old_stdout), _PrintLog(self.old_stderr, stderr=True)
        _ACTIVE = self
        # Registered before any model import, so backend atexit shutdown runs
        # first when the script keeps the session alive through process exit.
        atexit.register(self.close)
        _model_summary(self.config)
        _summary_line("MODEL", "native_logs=" + ", ".join(str(path) for path in self.native.paths.values()))

    def close(self):
        global _ACTIVE
        if self.closed:
            return
        self.closed = True
        failures = []

        def attempt(function):
            try:
                function()
            except BaseException as error:
                failures.append(error)

        try:
            for stream in (sys.stdout, sys.stderr):
                if isinstance(stream, _PrintLog):
                    attempt(stream.flush)
            sys.stdout, sys.stderr = self.old_stdout, self.old_stderr
            if self.capturing:
                # A failing Python logger/flush must never skip fd restoration.
                attempt(self.native.close)
                attempt(lambda: self.native.replay(
                    lambda level, line: logging.getLogger("examples.native").log(level, "%s", line)
                    if line.strip() else None))
        finally:
            sys.stdout, sys.stderr = self.old_stdout, self.old_stderr
            _ACTIVE = None
            for logger, level, handlers, propagate, disabled in self.logger_states:
                logger.setLevel(level)
                logger.handlers, logger.propagate, logger.disabled = handlers, propagate, disabled
            # New named loggers created by this invocation must not retain our
            # closed console streams after a reusable main() returns.
            original_loggers = {item[0] for item in self.logger_states}
            for item in logging.root.manager.loggerDict.values():
                if isinstance(item, logging.Logger):
                    if item not in original_loggers:
                        item.handlers = []
                        item.setLevel(logging.NOTSET)
                        item.propagate = True
                    else:
                        item.handlers = [handler for handler in item.handlers if handler not in self.handlers]
            warnings.showwarning = self.old_showwarning
            for name, value in self.old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            native = sys.modules.get("rpu_backend._cpp_ext")
            if self.old_native_level is not None and native is not None:
                attempt(lambda: native.rpu_set_debug_level(self.old_native_level))
            for handler in self.handlers:
                attempt(handler.close)
            for stream in self.streams:
                attempt(stream.close)
            atexit.unregister(self.close)
        if failures:
            first = failures[0]
            for error in failures[1:]:
                if hasattr(first, "add_note"):
                    first.add_note(f"additional logging cleanup error: {error}")
            raise first


@contextmanager
def logging_session(config, *, keep_until_exit=False):
    """Own logging for one run; script mode also covers backend atexit output."""
    if _ACTIVE is not None:
        raise RuntimeError("an example logging session is already active")
    session = _Session(config)
    try:
        session.start()
        yield session
    except BaseException as error:
        try:
            session.close()
        except BaseException as cleanup:
            if hasattr(error, "add_note"):
                error.add_note(f"logging cleanup failed: {cleanup}")
        raise
    else:
        if not keep_until_exit:
            session.close()

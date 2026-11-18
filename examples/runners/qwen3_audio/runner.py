"""Bounded speech examples; configuration checks do not import Torch."""
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import statistics
import time

from examples.common.cpu import configure_cpu_threads, resolve_cpu_threads
from examples.common.libraries import file_sha256
from examples.common.results import close_with_report, latency_timing, record_result


def audio_identity(filename):
    import soundfile as sf
    path = Path(filename)
    if not path.is_file():
        return {"status": "UNAVAILABLE", "path": str(path)}
    info = sf.info(path)
    digest = file_sha256(path)
    return {"path": str(path), "sha256": digest, "sample_rate": info.samplerate,
            "channels": info.channels, "frames": info.frames, "seconds": info.duration}


def validate_example_config(config):
    targets = set(config) & {"qwen3_asr", "qwen3_tts"}
    if len(targets) != 1:
        raise ValueError("select exactly one of [qwen3_asr] or [qwen3_tts]")
    target = next(iter(targets))
    tts = target == "qwen3_tts"
    if not {target, "input", "run"} <= set(config) or set(config) - {
            target, "input", "run", "cpu", "logging", "rpu_execution"}:
        raise ValueError("speech requires model, input and run tables, optional cpu/logging/rpu_execution")
    model, inputs, run = config[target], config["input"], config["run"]
    allowed = {"checkpoint", "source", "codec_backend", "precision"} if tts else {"checkpoint", "precision"}
    for table, keys, label in ((model, allowed, target),
            (inputs, {"text", "reference_audio", "language", "max_new_tokens"} if tts else
             {"audio", "max_new_tokens"}, "input"),
            (run, {"warmup", "runs", "seed", "output_dir", "profile", "hwperf"}, "run")):
        if not isinstance(table, dict) or set(table) - keys:
            raise ValueError(f"[{label}] contains unsupported fields or is not a table")
    required = [(model, "checkpoint"), (run, "output_dir")]
    required += [(model, "source"), (inputs, "text"), (inputs, "reference_audio")] if tts else [(inputs, "audio")]
    for table, key in required:
        if not isinstance(table.get(key), str) or not table[key].strip():
            raise ValueError(f"{key} requires non-empty text")
    if tts and (len(inputs["text"]) > 1024 or inputs.get("language", "Chinese") not in ("Chinese", "English")):
        raise ValueError("TTS requires 1–1024 characters and Chinese or English")
    if tts and model.get("codec_backend", "rpu") not in ("cpu", "rpu"):
        raise ValueError("codec_backend must be cpu or rpu")
    from examples.common.config import _backend_leaf
    _backend_leaf("api/_qwen3_audio_precision.py").normalize_audio_precision(
        model.get("precision", "fp16"), config.get("rpu_execution"), not tts)
    for value, label, low, high in (
            (inputs.get("max_new_tokens", 256 if tts else 64), "max_new_tokens", 2 if tts else 1, 256 if tts else 64),
            (run.get("warmup", 2), "warmup", 0, None),
            (run.get("runs", 5), "runs", 1, None),
            (run.get("seed", 0), "seed", 0, None)):
        if type(value) is not int or value < low or (high is not None and value > high):
            raise ValueError(f"invalid {label}")
    if type(run.get("profile", False)) is not bool or run.get("hwperf", False) is not False:
        raise ValueError("speech accepts boolean profile; hwperf is unavailable")
    resolve_cpu_threads(config)


def resolve_paths(config, parent):
    target = "qwen3_tts" if "qwen3_tts" in config else "qwen3_asr"
    for table, keys in ((config[target], ("checkpoint", "source")),
                        (config["input"], ("audio", "reference_audio")),
                        (config["run"], ("output_dir",))):
        for key in keys:
            if key in table:
                table[key] = str((parent / Path(table[key]).expanduser()).resolve())


def run_example_config(config):
    validate_example_config(config)
    from examples.common.logging import prepare_output_directory
    from examples.common.libraries import current_libraries

    target = "qwen3_tts" if "qwen3_tts" in config else "qwen3_asr"
    tts = target == "qwen3_tts"
    options, inputs, run = config[target], config["input"], config["run"]
    directory = prepare_output_directory(run["output_dir"])
    report = {"config": config, "wall_ms_runs": [], "cpu_ms_runs": [], "rtf_runs": [],
              "warmup_wall_ms_runs": [], "execution_stats": [], "formal_certification": False}
    with record_result(report, directory,
            timing=lambda: latency_timing(report["wall_ms_runs"],
                "resident speech request including CPU output; load, warmup and output file I/O excluded"),
            runtime_checks=("public speech API and Graph invariants",),
            output_names=("output.wav", "transcription.json", "profile.txt", "profile.trace.json")):
        _, report["libraries"] = current_libraries(Path(__file__).resolve().parents[3])
        report["cpu_runtime"] = configure_cpu_threads(resolve_cpu_threads(config, default=16 if tts else 4))
        import numpy as np
        import soundfile as sf
        import torch
        from rpu_backend.api import Qwen3ASRModel, Qwen3TTSModel

        limit = inputs.get("max_new_tokens", 256 if tts else 64)
        if tts:
            from rpu_backend.api.qwen3_tts import _read_reference, _validate_request
            _validate_request(inputs["text"], inputs.get("language", "Chinese"), limit)
            _read_reference(inputs["reference_audio"])
        else:
            from rpu_backend.api.qwen3_asr import _validate_audio
            info = sf.info(inputs["audio"])
            if info.channels != 1 or info.samplerate != 16000 or not 8000 <= info.frames <= 128000:
                raise ValueError("ASR input must be mono 16 kHz audio, 0.5–8 seconds")
            audio, rate = sf.read(inputs["audio"], dtype="float32")
            _validate_audio(audio, rate, limit)
        report["input_identity"] = audio_identity(inputs["reference_audio" if tts else "audio"])
        if tts:
            report["input_identity"]["text_sha256"] = hashlib.sha256(inputs["text"].encode()).hexdigest()
            report["input_identity"]["text_characters"] = len(inputs["text"])
        owner = None
        try:
            start = time.perf_counter()
            if tts:
                owner = Qwen3TTSModel.from_pretrained(options["checkpoint"], source=options["source"],
                    codec_backend=options.get("codec_backend", "rpu"),
                    precision=options.get("precision", "fp16"), rpu_execution=config.get("rpu_execution"))
                def infer():
                    return owner.synthesize(inputs["text"], reference_audio=inputs["reference_audio"],
                        language=inputs.get("language", "Chinese"), max_new_tokens=limit)
            else:
                owner = Qwen3ASRModel.from_pretrained(options["checkpoint"],
                    precision=options.get("precision", "fp16"), rpu_execution=config.get("rpu_execution"))
                def infer():
                    return owner.transcribe(audio, sample_rate=rate, max_new_tokens=limit)
            report["load_seconds"] = time.perf_counter() - start
            warmup, runs = run.get("warmup", 2), run.get("runs", 5)
            report.update(warmup=warmup, runs=runs)
            for index in range(warmup + runs):
                torch.manual_seed(run.get("seed", 0))
                start, cpu_start = time.perf_counter(), time.process_time()
                output = infer()
                elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu_start
                stats = owner.execution_stats()
                components = stats["components"].values() if tts else (stats,)
                if not all(part["graph_invariant_ok"] for part in components) or (
                        not tts and not stats["audio_graph_invariant_ok"]):
                    raise RuntimeError("speech Graph invariant failed")
                if index < warmup:
                    report["warmup_wall_ms_runs"].append(elapsed * 1000)
                    continue
                duration = len(output[0]) / output[1] if tts else len(audio) / rate
                report["wall_ms_runs"].append(elapsed * 1000)
                report["cpu_ms_runs"].append(cpu_elapsed * 1000)
                report["rtf_runs"].append(elapsed / duration)
                report.setdefault("audio_seconds_runs", []).append(duration)
                if isinstance(stats.get("stage_ms"), dict):
                    report.setdefault("stage_ms_runs", []).append(stats["stage_ms"])
                report["execution_stats"].append(stats)
            if tts:
                wave, sample_rate = output
                if sample_rate != 24000 or wave.ndim != 1 or not np.isfinite(wave).all() or not np.any(wave):
                    raise ValueError("invalid TTS waveform")
                sf.write(directory / "output.wav", wave, sample_rate, subtype="FLOAT")
            else:
                (directory / "transcription.json").write_text(json.dumps(asdict(output), ensure_ascii=False, indent=2) + "\n")
                logging.getLogger("examples.output").info("%s", output.text)
            if run.get("profile", False):
                torch.manual_seed(run.get("seed", 0))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                    infer()
                profiler.export_chrome_trace(str(directory / "profile.trace.json"))
                (directory / "profile.txt").write_text(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))
            report["median_ms"] = statistics.median(report["wall_ms_runs"])
            report["median_cpu_ms"] = statistics.median(report["cpu_ms_runs"])
            report["median_rtf"] = statistics.median(report["rtf_runs"])
        finally:
            close_with_report(owner, report)
    return 0

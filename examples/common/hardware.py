"""Read the board identity and driver core-allocation settings a report ran under.

Every value is read back from the running system. A value that cannot be read
is recorded UNAVAILABLE with the reason, never filled from a requested or
default value. The driver exposes no per-task readback of the physical cores it
granted, so the core sets are derived only for a driver build and allocation
tuple whose placement was checked on hardware; any other tuple records
UNAVAILABLE instead of assuming cores 0..N-1.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

DRIVER = Path("/sys/module/rpu_drv")
BOOT_SHM = Path("/sys/boot_shm")
# Driver settings that decide which physical cores a task is granted.
ALLOCATION_PARAMETERS = ("core_binning", "core_mask", "core_pair_order", "resv_pin", "alloc_low",
                         "xcu_reserve", "class_big_uid", "class_small_uid", "class_uid_only", "uid_mask")
# Board fields named here only; the rest of boot_shm holds fuse and key material.
BOARD_FIELDS = ("board_main_name", "board_module_name", "plat_type", "rtl_version", "rpuVoltage",
                "ddr_frequency", "ddr_rankNum", "ddr_dramtype", "ddr_interleave", "ddr_chipBitMap")

# Checked on 2026-11-02 from the driver's own task-grant log (debug_level 0xf,
# "core - 0x.." per grant): Qwen3-0.6B FP16 runs at 4, 6 and 8 cores on an 8-core
# board with the device held exclusively, 60 grants. A grant is not one kernel: a
# 6-core run's attention and MLP share one 6-core grant, attention on 4 of its
# cores, and its output head gets 4-core grants.
VERIFIED_PLACEMENTS = (
    {"driver": {"srcversion": "F25632D3A60C21E0B7B60A0", "core_binning": 255, "core_mask": 0,
                "core_pair_order": 1, "resv_pin": 2, "alloc_low": 2, "xcu_reserve": 2,
                "class_big_uid": 0, "class_small_uid": 0, "class_uid_only": 0, "uid_mask": 0},
     "task_cores": {4: [0, 1, 4, 5], 6: [0, 1, 2, 4, 5, 6], 8: [0, 1, 2, 3, 4, 5, 6, 7]},
     "checked": "2026-11-02: driver task-grant log, Qwen3-0.6B FP16 at 4/6/8 cores, device held exclusively"},
)


def _unavailable(reason):
    return {"status": "UNAVAILABLE", "reason": reason}


def _read(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        return exc


def _integer(text):
    try:
        return int(text, 0)
    except (TypeError, ValueError):
        return text


def _chip():
    tool = shutil.which("chip_info")
    if tool is None:
        return _unavailable("chip_info not installed")
    try:
        result = subprocess.run([tool, "-c"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable(f"chip_info -c: {type(exc).__name__}")
    lines = [line.strip() for line in result.stdout.splitlines()]
    if "SOC info:" in lines:
        following = lines[lines.index("SOC info:") + 1:]
        if following and following[0]:
            return {"soc": following[0], "source": "chip_info -c"}
    detail = (result.stderr.strip().splitlines() or [f"exit {result.returncode}"])[0][:120]
    return _unavailable(f"chip_info -c printed no SOC line ({detail}); it needs root")


def _driver():
    if not DRIVER.is_dir():
        return _unavailable(f"{DRIVER} absent: rpu_drv not loaded")
    driver = {"source": str(DRIVER)}
    for name in ("version", "srcversion"):
        value = _read(DRIVER / name)
        driver[name] = value if isinstance(value, str) else None
    parameters, missing = {}, []
    for name in ALLOCATION_PARAMETERS:
        value = _read(DRIVER / "parameters" / name)
        if isinstance(value, str):
            parameters[name] = _integer(value)
        else:
            missing.append(name)
    driver["parameters"] = parameters
    if missing:
        driver["missing_parameters"] = missing
    return driver


def _board():
    if not BOOT_SHM.is_dir():
        return _unavailable(f"{BOOT_SHM} absent")
    board = {"source": str(BOOT_SHM)}
    for name in BOARD_FIELDS:
        value = _read(BOOT_SHM / name)
        if isinstance(value, str):
            board[name] = value
    return board


def core_sets(driver):
    """Physical cores per task size for a verified driver tuple, else UNAVAILABLE."""
    if driver.get("status") == "UNAVAILABLE":
        return _unavailable("driver settings unreadable")
    observed = dict(driver.get("parameters", {}), srcversion=driver.get("srcversion"))
    for placement in VERIFIED_PLACEMENTS:
        if all(observed.get(key) == value for key, value in placement["driver"].items()):
            return {"status": "DERIVED", "task_cores": {str(cores): list(value) for cores, value
                                                        in placement["task_cores"].items()},
                    "basis": "driver build and allocation settings equal a tuple whose task grants were "
                             "read from the driver log; " + placement["checked"],
                    "condition": "no other process holds RPU tasks during the run"}
    return _unavailable("no hardware-checked placement for this driver build and allocation settings; "
                        "the driver has no per-task core readback")


def snapshot():
    """Collect once when the report is finalized, outside every timer."""
    driver = _driver()
    return {"scope": "board identity and driver core-allocation settings read when the report is finalized; "
                     "core sets are derived per granted task size, not read back per task or per kernel",
            "effective_uid": os.geteuid() if hasattr(os, "geteuid") else None,
            "chip": _chip(), "driver": driver, "board": _board(), "core_sets": core_sets(driver),
            "missing_fields_policy": "unavailable; never filled from requested auto/default values"}

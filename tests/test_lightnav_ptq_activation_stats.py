"""Board-free checks for customer PTQ calibration sampling."""

from pathlib import Path
import importlib.util

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples/lightnav_collect_ptq_activation_stats.py"
SPEC = importlib.util.spec_from_file_location("lightnav_collect_ptq_activation_stats", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def test_calibration_uses_one_deterministic_middle_frame_per_episode():
    assert COLLECTOR.calibration_frame(1) == 0
    assert COLLECTOR.calibration_frame(9) == 4
    assert COLLECTOR.calibration_frame(10) == 5
    assert COLLECTOR.calibration_frame(300) == 150
    with pytest.raises(ValueError, match="at least one"):
        COLLECTOR.calibration_frame(0)

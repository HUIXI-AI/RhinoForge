"""report.json hardware identity: read back from sysfs, never assumed."""
from pathlib import Path
import stat

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIED = {"core_binning": "255", "core_mask": "0", "core_pair_order": "1", "resv_pin": "2",
            "alloc_low": "2", "xcu_reserve": "2", "class_big_uid": "0", "class_small_uid": "0",
            "class_uid_only": "0", "uid_mask": "0"}


@pytest.fixture
def hardware(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT))
    from examples.common import hardware
    monkeypatch.setattr(hardware, "DRIVER", tmp_path / "rpu_drv")
    monkeypatch.setattr(hardware, "BOOT_SHM", tmp_path / "boot_shm")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    return hardware


def board(tmp_path, parameters=VERIFIED, srcversion="F25632D3A60C21E0B7B60A0", chip_info="  R1"):
    driver = tmp_path / "rpu_drv"
    (driver / "parameters").mkdir(parents=True)
    (driver / "version").write_text("1.0.1\n")
    (driver / "srcversion").write_text(srcversion + "\n")
    for name, value in parameters.items():
        (driver / "parameters" / name).write_text(value + "\n")
    shm = tmp_path / "boot_shm"
    shm.mkdir()
    (shm / "ddr_frequency").write_text("0x320 \n")
    (shm / "board_main_name").write_text("r1_main\n")
    (shm / "fuse_secret_example").write_text("0x1 0x2\n")
    if chip_info is not None:
        tool = tmp_path / "bin" / "chip_info"
        tool.parent.mkdir()
        tool.write_text(f"#!/bin/sh\necho 'SOC info:'\necho '{chip_info}'\n")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)


def test_verified_driver_tuple_derives_the_checked_core_sets(hardware, tmp_path):
    board(tmp_path)
    record = hardware.snapshot()
    assert record["chip"] == {"soc": "R1", "source": "chip_info -c"}
    assert record["driver"]["parameters"]["core_pair_order"] == 1
    assert record["board"]["ddr_frequency"] == "0x320"
    assert "fuse_secret_example" not in record["board"]  # fuse/key material is never copied
    sets = record["core_sets"]
    assert sets["status"] == "DERIVED"
    assert sets["task_cores"] == {"4": [0, 1, 4, 5], "6": [0, 1, 2, 4, 5, 6], "8": list(range(8))}


@pytest.mark.parametrize("change", [{"core_pair_order": "0"}, {"core_mask": "192"}, {"srcversion": "OTHER"}])
def test_any_other_driver_tuple_records_unavailable_not_cores_0_to_n(hardware, tmp_path, change):
    parameters = dict(VERIFIED)
    srcversion = change.pop("srcversion", "F25632D3A60C21E0B7B60A0")
    parameters.update(change)
    board(tmp_path, parameters, srcversion)
    sets = hardware.snapshot()["core_sets"]
    assert sets["status"] == "UNAVAILABLE" and "task_cores" not in sets


def test_missing_driver_and_tool_are_unavailable_with_reasons(hardware):
    record = hardware.snapshot()
    assert record["driver"]["status"] == record["core_sets"]["status"] == "UNAVAILABLE"
    assert record["chip"] == {"status": "UNAVAILABLE", "reason": "chip_info not installed"}


def test_chip_info_without_root_is_unavailable(hardware, tmp_path):
    board(tmp_path, chip_info=None)
    tool = tmp_path / "bin" / "chip_info"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\necho 'shm_open: Permission denied' >&2\nexit 1\n")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    chip = hardware.snapshot()["chip"]
    assert chip["status"] == "UNAVAILABLE" and "Permission denied" in chip["reason"]


def test_shared_report_writer_records_hardware_once(hardware, tmp_path):
    from examples.common import results
    report = results.add_result_fields({}, tmp_path, completed=True)
    assert report["hardware"]["core_sets"]["status"] == "UNAVAILABLE"
    board(tmp_path)
    again = results.add_result_fields(report, tmp_path, completed=True)
    assert again["hardware"]["core_sets"]["status"] == "UNAVAILABLE"  # first reading is kept

"""Unit tests for convert_pi05 real-W4 (method=w4a16) tagging. Standalone import."""
import sys
from pathlib import Path

_QUANT = Path(__file__).resolve().parents[2] / "quant"
sys.path.insert(0, str(_QUANT))
import convert_pi05 as _cp  # noqa: E402

KEEP = ("k_proj.weight", "v_proj.weight")


def test_quant_config_real_w4_method_tag():
    cfg = _cp._quant_config(bits=4, int8_keep_suffixes=KEEP, real_w4=True)
    assert cfg["method"] == "w4a16"
    assert cfg["storage"] == "int8"          # on-disk is still int8 [N,K]
    assert cfg["value_bits"] == 4
    assert cfg["kernel"] == "w4a16_pgrp"
    assert cfg["mixed_int8_suffixes"] == list(KEEP)


def test_quant_config_fake_w4_unchanged():
    cfg = _cp._quant_config(bits=4, int8_keep_suffixes=KEEP, real_w4=False)
    assert cfg["method"] == "w4a16_fake_int8"
    assert cfg["kernel"] == "w8a16"


def test_real_w4_requires_bits4():
    import pytest
    with pytest.raises(ValueError):
        _cp._quant_config(bits=8, int8_keep_suffixes=(), real_w4=True)


def test_real_w4_injects_kv_keep_int8():
    cfg = _cp._quant_config(bits=4, int8_keep_suffixes=(), real_w4=True)
    assert set(cfg["mixed_int8_suffixes"]) >= {"k_proj.weight", "v_proj.weight"}


def test_real_w4_convert_stats_method_tag():
    assert _cp._method_for_stats(bits=4, real_w4=True) == "w4a16"
    assert _cp._method_for_stats(bits=4, real_w4=False) == "w4a16_fake_int8"

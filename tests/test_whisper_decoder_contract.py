"""Source contract for the dedicated Whisper decoder layer owner."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_whisper_decoder_registers_one_owned_layer_path():
    source = (ROOT / "src/fused/rpu_whisper_decoder.cpp").read_text()
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()

    assert "class WhisperDecoderModel" in source
    assert "ModelHandleRegistry<WhisperDecoderModel>" in source
    for name in (
        "whisper_decoder_create",
        "whisper_decoder_destroy",
        "whisper_decoder_set_weights",
        "whisper_decoder_reset",
        "whisper_decoder_prefill",
        "whisper_decoder_decode",
    ):
        assert f"rpu_{name}" in declarations
        assert f'"{name}' in registrations


def test_whisper_decoder_keeps_three_pre_norm_residual_blocks():
    source = (ROOT / "src/fused/rpu_whisper_decoder.cpp").read_text()

    for norm in ("layer.n1_w", "layer.n2_w", "layer.n3_w"):
        assert norm in source
    assert source.count("layer_norm(") >= 4  # three call sites plus the helper
    assert source.count("rpu_launch_all_reduce_sum_residual_kernel(") == 3
    assert "GeluMode::ERF" in source
    assert "WHISPER_DECODER_CROSS_TOKENS = 1500" in source


def test_whisper_decode_position_is_checked_and_uses_fixed_v2_topology():
    source = (ROOT / "src/fused/rpu_whisper_decoder.cpp").read_text()

    assert "position == expected_position_" in source
    assert "position + seq <= WHISPER_DECODER_TEXT_TOKENS" in source
    assert "KvInsertRoute::V2" in source
    assert "kv_len = position + seq" in source

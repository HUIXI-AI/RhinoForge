"""Host checks for the dedicated Whisper encoder native seam."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_whisper_encoder_has_separate_owner_and_public_ops():
    source = (ROOT / "src/fused/rpu_dinov3_vision_model.cpp").read_text()
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()

    assert "class WhisperEncoderModel" in source
    assert "ModelHandleRegistry<v3::WhisperEncoderModel>" in source
    for name in (
        "whisper_encoder_create",
        "whisper_encoder_destroy",
        "whisper_encoder_set_weights",
        "whisper_encoder_forward",
    ):
        assert f"rpu_{name}" in declarations
        assert f'"{name}' in registrations


def test_whisper_attention_chunks_queries_but_keeps_full_kv():
    source = (ROOT / "src/fused/rpu_dinov3_vision_model.cpp").read_text()
    assert "WHISPER_ENCODER_QUERY_CHUNK = 128" in source
    assert "query_offset += WHISPER_ENCODER_QUERY_CHUNK" in source
    assert "const int64_t query_len = std::min(" in source
    assert "query_len, nq, nq, hd," in source
    assert "seq_len, NUM_CORES, NUM_CORES" in source


def test_whisper_encoder_owns_fixed_conv_frontend_and_position_add():
    source = (ROOT / "src/fused/rpu_dinov3_vision_model.cpp").read_text()
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()

    assert "WHISPER_MEL_FRAMES = 3000" in source
    assert "WHISPER_AUDIO_CTX = 1500" in source
    assert "whisper_encoder_set_frontend" in declarations
    assert '"whisper_encoder_set_frontend' in registrations
    assert source.count("rpu_launch_conv2d_spm_kernel(") >= 2
    assert "GeluMode::ERF" in source
    assert "rpu_launch_slice_spm_kernel(" in source
    assert "position_embedding_" in source

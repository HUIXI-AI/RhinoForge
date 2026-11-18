"""Host regression for head boundaries in both no-QK-norm schedules."""
import math
from pathlib import Path
import re

import pytest


SOURCE = (Path(__file__).resolve().parents[1] /
          "src/fused/rpu_qwen3_model.h").read_text()


def _rope(values, heads, width, position):
    result = []
    for head in range(heads):
        row = values[head * width:(head + 1) * width]
        half = width // 2
        for index in range(width):
            phase = position / (10000 ** (2 * (index % half) / width))
            rotated = -row[index + half] if index < half else row[index - half]
            result.append(row[index] * math.cos(phase) + rotated * math.sin(phase))
    return result


# (per-head route, nkv, tp): the TTS codec (16 KV heads) at TP8/TP4, reduced
# Llama-3.2-1B (8 KV heads) at TP4, and the unchanged one-head-per-core TP8 row.
_CASES = [(True, 16, 8), (True, 16, 4), (True, 8, 4), (False, 8, 8)]


@pytest.mark.parametrize("schedule", ["kv_first", "sequential"])
@pytest.mark.parametrize("per_head, kv_heads, tp", _CASES)
@pytest.mark.parametrize("position", [1, 7])
def test_no_qknorm_key_rope_rotates_each_local_head(schedule, per_head, kv_heads, tp, position):
    # Exercise the arguments emitted by each real source branch. In particular,
    # two heads must not become one doubled-width head (including its RoPE table).
    calls = re.findall(r'emit_qk_rope\("k", "k",\s*([^,]+),\s*([^,]+), seq_len,', SOURCE)
    assert len(calls) == 2
    head_expr, width_expr = calls[0 if schedule == "kv_first" else 1]
    batch = 1 if schedule == "kv_first" else 2
    env = dict(nkv=kv_heads, tp=tp, hd=64, bs=batch,
               local_kv_dim=kv_heads * 64 // tp)
    def evaluate(expression):
        if "?" in expression:
            condition, branches = expression.split("?")
            assert condition.strip() == "per_head_key_rope()"
            expression = branches.split(":")[0 if per_head else 1]
        return int(eval(expression, {"__builtins__": {}}, env))

    heads, width = evaluate(head_expr), evaluate(width_expr)
    per_core_heads = batch * kv_heads // tp
    values = [math.sin(index * 0.37) for index in range(per_core_heads * 64)]
    actual = _rope(values, heads, width, position)
    expected = _rope(values, per_core_heads, 64, position)  # every D64 head
    assert width == 64 and width // 2 == 32  # the checkpoint table has 32 columns
    assert actual == pytest.approx(expected, abs=1e-12)


def test_doubled_width_key_rope_would_be_wrong_for_two_local_heads():
    # Negative control for reduced Llama TP4: one 128-wide row is not two heads.
    values = [math.sin(index * 0.37) for index in range(2 * 64)]
    assert _rope(values, 1, 128, 7) != pytest.approx(_rope(values, 2, 64, 7), abs=1e-6)


def test_per_head_key_rope_is_bound_to_codec_or_reduced_llama_only():
    accessor = SOURCE.split("bool per_head_key_rope() const {", 1)[1].split("}", 1)[0]
    assert accessor.strip() == "return tts_codec_mode_ || reduced_llama_profile_;"
    assert "bool reduced_llama_profile_ = false;" in SOURCE
    assert "reduced_llama_profile_ = llama_text_profile;" in SOURCE
    profile = SOURCE.split("const bool llama_text_profile =", 1)[1].split(";", 1)[0]
    # The profile is the exact geometry; its FP16 and on-install W8/W4/NVFP4
    # payloads are admitted separately, so it carries no precision clause.
    for clause in ("num_cores() != 8", "configured_vocab_size_ == 128256",
                   "!qk_norm_present", "llama_fp16_profile_num_layers("):
        assert clause in profile
    assert "has_scale" not in profile and "has_nvfp4" not in profile


def test_codec_mode_is_explicit_and_default_off():
    assert "bool tts_codec_mode_ = false;" in SOURCE
    setter = SOURCE.split("void set_tts_codec_mode()", 1)[1].split("\n    }", 1)[0]
    assert "num_layers() == 8" in setter and "hidden_size() == 512" in setter
    assert "!has_qk_norm_" in setter
    assert setter.index("TORCH_CHECK") < setter.index("tts_codec_mode_ = true")
    assert "invalidate_model_state();" in setter

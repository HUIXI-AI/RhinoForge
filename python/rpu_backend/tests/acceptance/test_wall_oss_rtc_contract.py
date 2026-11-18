"""Board-free structural contracts for the Wall-OSS v728 RTC extension."""
from pathlib import Path

import pytest

from rpu_backend.adapters.wall_oss.runtime import (
    build_wall_oss_vla,
    _validate_rtc_training_metadata,
)


ROOT = Path(__file__).resolve().parents[4]


def test_direct_rtc_builder_needs_no_opt_in_but_keeps_acc32_limit(monkeypatch):
    monkeypatch.delenv("RPU_WALL_OSS_RTC_ALLOW_NUMERIC_BLOCKED", raising=False)
    # RTC is supported without an opt-in; the hard ACC16-only limit remains.
    with pytest.raises(ValueError, match="does not support linear_acc32=True"):
        build_wall_oss_vla(rtc=True, rpu_execution={"prefill": {"linear_acc32": True}})


def test_rtc_hard_pin_is_inside_every_unrolled_euler_body():
    source = (ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()
    pre = source[source.index("void WallOssActionStepModel::emit_pre_layers_body()"):
                 source.index("void WallOssActionStepModel::emit_post_layers_body()")]
    post = source[source.index("void WallOssActionStepModel::emit_post_layers_body()"):
                  source.index("at::Tensor WallOssActionStepModel::step_forward(")]

    assert "rtc_prefix_src_base_" in pre
    assert "rtc_delay_ * k_comb_pad_" in pre
    assert "rpu_launch_fill_spm_kernel" in post
    assert post.count("rtc_prefix_src_base_") == 1
    assert post.index("rpu_launch_fill_spm_kernel") < post.index("ValuOpType::ADD, dt_")
    assert post.index("ValuOpType::ADD, dt_") < post.index("rtc_prefix_src_base_")


def test_rtc_prefix_is_mutable_keepalive_and_delay_is_in_graph_signature():
    cpp = (ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()
    action = (ROOT / "python/rpu_backend/adapters/wall_oss/action.py").read_text()
    schema = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()

    assert "rtc_prefix_ref_ = prefix" in cpp
    assert "rpu_ddr_flush_force_sized(prefix.data_ptr(), prefix.nbytes())" in cpp
    assert "wall_oss_action_rtc_denoise_loop_v2" in action
    assert "int(row_constant), rtc_delay, self._rtc_rope_gap" in action
    assert "Tensor? rtc_prefix=None, int rtc_delay=0" in schema


def test_masked_dof_blend_precedes_prefix_zero_and_euler():
    cpp = (ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()
    action = (ROOT / "python/rpu_backend/adapters/wall_oss/action.py").read_text()
    schema = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    post = cpp[cpp.index("void WallOssActionStepModel::emit_post_layers_body()"):
               cpp.index("at::Tensor WallOssActionStepModel::step_forward(")]

    mul = post.index("ValuOpType::MUL")
    override = post.index("velocity_override_src_base_")
    prefix_zero = post.index("rpu_launch_fill_spm_kernel")
    euler = post.index("ValuOpType::ADD, dt_")
    assert mul < override < prefix_zero < euler
    assert "rtc_delay_ > 0 && bit > 0" in post
    assert "_masked_dof_velocity_inputs" in action
    assert "Tensor? velocity_mask=None, Tensor? velocity_override=None" in schema


def test_rtc_prepare_checks_mutable_values_and_restores_physical_prefix():
    runtime = (
        ROOT / "python/rpu_backend/adapters/wall_oss/runtime.py"
    ).read_text()

    assert 'out_a["x_norm"], out_b["x_norm"]' in runtime
    assert 'out_a["x_norm"], out_a2["x_norm"]' in runtime
    denorm = runtime.index('action = (out["x_norm"] + 1) / 2 * dl + mn')
    restore = runtime.index(
        "action[:, :rtc_delay] = prefix_physical[:, :rtc_delay]")
    assert denorm < restore


def test_rtc_d0_has_an_explicit_request_path_and_graph_signature():
    runtime = (
        ROOT / "python/rpu_backend/adapters/wall_oss/runtime.py"
    ).read_text(encoding="utf-8")
    action = (
        ROOT / "python/rpu_backend/adapters/wall_oss/action.py"
    ).read_text(encoding="utf-8")
    facade = (
        ROOT / "python/rpu_backend/api/wall_oss.py"
    ).read_text(encoding="utf-8")

    assert "tuple(range(0, self._rtc_max_delay + 1))" in runtime
    assert "len(profiles) * (1 + len(rtc_delays))" in runtime
    assert "use_rtc=True" in runtime
    assert "ready_rtc_outs[delay]" in runtime
    assert 'os.environ["RPU_WALL_OSS_PREFILL_MIXED_PRECISION"] = "1"' in runtime
    assert '"wall_oss_action_rtc_denoise_loop_v2"' in action
    assert "if rtc_request:" in action
    assert "use_rtc: bool = False" in facade


def test_rtc_training_metadata_is_required_and_checked(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        """action_tokenizer_type: x2robot_tokenizer_v3_1_delta
per_step_normalization: true
disable_normalizer_clamp: true
use_state_string_representation: true
rtc_train:
  enabled: true
  max_delay: 10
use_per_token_tau: true
rtc_rope_gap: 10
data:
  action_horizon_flow: 20
  obs_action_keys:
  - follow_left_ee_cartesian_pos
  - follow_left_ee_rotation_6D
  - follow_left_gripper
  - follow_right_ee_cartesian_pos
  - follow_right_ee_rotation_6D
  - follow_right_gripper
  - velocity_decomposed
  - height
  - head_actions
  predict_action_keys:
  - master_left_ee_cartesian_pos_relative
  - master_left_ee_rotation_6D_relative
  - master_left_gripper
  - master_right_ee_cartesian_pos_relative
  - master_right_ee_rotation_6D_relative
  - master_right_gripper
  - velocity_decomposed
  - height
  - head_actions
  state_bins: 512
  use_state_string_representation: true
""",
        encoding="utf-8",
    )
    _validate_rtc_training_metadata(str(tmp_path))

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "use_per_token_tau: true", "use_per_token_tau: false"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="use_per_token_tau=true"):
        _validate_rtc_training_metadata(str(tmp_path))

    config.write_text(
        config.read_text(encoding="utf-8")
        .replace("use_per_token_tau: false", "use_per_token_tau: true")
        .replace(
            "  - master_left_ee_cartesian_pos_relative\n"
            "  - master_left_ee_rotation_6D_relative",
            "  - master_left_ee_rotation_6D_relative\n"
            "  - master_left_ee_cartesian_pos_relative",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.predict_action_keys"):
        _validate_rtc_training_metadata(str(tmp_path))

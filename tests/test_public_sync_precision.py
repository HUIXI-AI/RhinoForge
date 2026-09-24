"""Cold public-profile boundaries added with the runtime refresh."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.pi05 import optimized


@pytest.mark.parametrize("name", [
    "_rpu_decoder_topology", "_rpu_vision_num_cores", "_rpu_qwen3vl_vision_num_cores",
])
def test_installed_topology_is_admitted_and_remains_immutable(name):
    from rpu_backend.runtime import RPUConfigError
    from rpu_backend.runtime.hw_attrs import (
        install_hw_attr_validator, validate_postinstall, validate_preinstall,
    )
    owner = torch.nn.Linear(2, 2)
    validate_preinstall(owner)
    setattr(owner, name, (8, 8) if name.endswith("topology") else 8)
    validate_postinstall(owner)
    install_hw_attr_validator(owner)
    with pytest.raises(RPUConfigError, match="monotonic"):
        setattr(owner, name, 4)
    with pytest.raises(RPUConfigError, match="Unknown hardware attribute"):
        owner._rpu_unregistered_topology = 4


def test_moe_owner_state_is_admitted_after_install():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator, validate_postinstall
    owner = torch.nn.Linear(2, 2)
    owner._rpu_qwen3_5_moe_text_install_started = True
    owner._rpu_qwen3_5_moe = SimpleNamespace(handle=1)
    validate_postinstall(owner)
    install_hw_attr_validator(owner)
    owner._rpu_qwen3_5_moe = None


@pytest.mark.parametrize("kwargs,pixels,message", [
    ({"_rpu_temporal_num_frames": 2}, (4, 8), "temporal_num_frames=1"),
    ({"_rpu_camera_batch_count": 2}, (4, 8), "camera_batch_count=1"),
    ({}, (2, 3, 32, 32), "flattened pixel patches"),
])
def test_qwen35_rejects_removed_packed_inputs_before_dispatch(kwargs, pixels, message):
    from rpu_backend.adapters.qwen3_5 import vision
    owner = SimpleNamespace(_rpu_vision_handle=1, _rpu_vision_spatial_merge_size=2,
                            _rpu_vision_num_layers=0, _rpu_vision_kv_cache=object())
    before = vars(owner).copy()
    with pytest.raises(ValueError, match=message):
        vision._rpu_vision_forward(owner, torch.empty(pixels), torch.tensor([[1, 2, 2]]), **kwargs)
    assert vars(owner) == before


def test_qwen35_dense_precision_tracks_only_cold_settings():
    from rpu_backend.adapters.qwen3_5 import vision
    assert vision._qwen35_vision_precision({}) == (False, False)
    assert vision._qwen35_vision_precision({"vision": {"linear_acc32": True}}) == (True, True)


def test_two_camera_a8_requires_matching_pair_and_action_precision(monkeypatch):
    profile = optimized.normalize_profile({"precision": "w8_prefill_a8_action_nvfp4", "num_cameras": 2})
    execution = {"action": {"linear_acc32": True}}
    assert optimized.execution_for_profile(profile, execution)["prefill"]["chunk_size"] == 272
    calls = []
    monkeypatch.setattr(torch.ops.rpu, "require_kernel_names", lambda names: calls.append(set(names)), raising=False)
    optimized.require_profile_assets(profile)
    optimized.require_profile_assets(profile, execution)
    optimized.require_profile_assets(profile, {"components": {"action_expert": {"action": {"linear_acc32": True}}}})
    a16, a32, component = calls
    assert "pi05_prefill_gate_up_geglu_w8a8_split_c272_m48n32k2048" in a16
    assert "pi05_owner_norm_compact_a8_m272n2048" in a16
    nvfp4 = {name for name in a16 if "nvfp4" in name}
    assert len(nvfp4) == 3 and all("acc16" in name for name in nvfp4)
    assert a32 == component == (a16 - nvfp4) | {name.replace("acc16", "acc32") for name in nvfp4 if name.startswith("parallel_linear_")}
    import os
    with optimized.profile_environment_scope(profile):
        assert os.environ["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] == "1"
        assert os.environ["RPU_PI05_DENOISE_NVFP4_GEGLU_M50"] == "0"
    with optimized.profile_environment_scope(profile, execution):
        assert os.environ["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] == "0"
    assert execution == {"action": {"linear_acc32": True}}
    with pytest.raises(ValueError, match="num_cores=8"):
        optimized.execution_for_profile(profile, {"model": {"num_cores": 4}})


def test_moe_direct_loader_rejects_remote_code_before_config_or_weight_io(monkeypatch):
    from rpu_backend.adapters.qwen3_5_moe import loader
    def forbidden(*args, **kwargs):
        pytest.fail("untrusted request reached checkpoint IO")
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="trust_remote_code=False"):
        loader.load_qwen3_5_moe_model("unread-checkpoint", trust_remote_code=True)


def test_wall_public_processor_expands_token_ids_and_falls_back_on_marker_mismatch(monkeypatch):
    from rpu_backend.adapters.wall_oss.runtime import WallOssVLA
    class Processor:
        image_token_id = 42
        image_processor = SimpleNamespace(merge_size=2)
        calls = 0
        def tokenizer(self, texts, **kwargs):
            assert texts == ["image request"]
            return {"input_ids": torch.tensor([[7, 42, 8]])}
        def __call__(self, **kwargs):
            self.calls += 1
            return dict(input_ids=torch.tensor([[99]]), pixel_values=torch.ones(1), image_grid_thw=torch.tensor([[1, 2, 4]]))
    owner = SimpleNamespace(_processor=Processor(), _fast_processor=True, _fast_proc_ok=True,
                            _image_preproc=lambda images: (torch.ones(1), torch.tensor([[1, 2, 4]])))
    ids, _, _ = WallOssVLA._process_inputs(owner, "image request", [])
    assert ids.tolist() == [[7, 42, 42, 8]] and owner._processor.calls == 0
    owner._image_preproc = lambda images: (torch.ones(1), torch.tensor([[1, 2, 4], [1, 2, 4]]))
    with pytest.warns(UserWarning, match="marker count mismatch"):
        ids, _, _ = WallOssVLA._process_inputs(owner, "image request", [])
    assert ids.tolist() == [[99]] and owner._fast_proc_ok is False and owner._processor.calls == 1

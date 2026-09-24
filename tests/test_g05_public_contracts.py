"""Board-free public G0.5 admission, lifecycle and continuous-action contracts."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import types
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER = _ROOT / "python/rpu_backend/adapters/g05/runtime.py"


def _load_adapter():
    __import__("rpu_backend.api._execution")
    action_name = "rpu_backend.adapters.g05.action"
    action_stub = types.ModuleType(action_name)
    action_stub._check_g05_action_profile = lambda _config: None
    action_stub._g05_action_runtime_ready = lambda _expert: False
    action_stub._rpu_g05_inference_fm = lambda *_args, **_kwargs: None
    action_stub.patch_g05_action_expert_for_rpu = lambda *_args, **_kwargs: None
    causal_name = "rpu_backend.api.causal_lm"
    causal_stub = types.ModuleType(causal_name)
    causal_stub._claim_live_instance = lambda _owner: None
    modules = __import__("sys").modules
    previous = {
        action_name: modules.get(action_name),
        causal_name: modules.get(causal_name),
    }
    modules[action_name] = action_stub
    modules[causal_name] = causal_stub
    spec = importlib.util.spec_from_file_location("g05_adapter_under_test", _ADAPTER)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                modules.pop(name, None)
            else:
                modules[name] = old_module
    return module


def _canonical_fm_helper():
    return types.SimpleNamespace(
        action_causal=False,
        horizon_steps=32,
        action_dim=27,
        num_inference_steps=10,
        time_convention="pi_convention",
        _sample_noise=lambda *_args, **_kwargs: None,
    )


def test_g05_vision_reuses_qwen35_for_single_frame(monkeypatch):
    module = _load_adapter()
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    calls = []
    install_order = []
    forward_calls = []
    vision_module = types.ModuleType("rpu_backend.adapters.qwen3_5.vision")

    def install(vision, *, vision_config):
        install_order.append("install")
        calls.append((vision, vision_config))
        def forward(hidden, grid, **kwargs):
            forward_calls.append(kwargs)
            return types.SimpleNamespace(
                last_hidden_state=(hidden, grid),
                pooler_output="pooler",
            )

        vision.forward = forward
        vision._fake_shared_runtime_live = True
        return 7

    vision_module.install_qwen3_5_vision_for_rpu = install
    vision_module._qwen3_5_vision_runtime_complete = (
        lambda owner, *, installed_forward=None: bool(
            getattr(owner, "_fake_shared_runtime_live", False)
            and installed_forward is owner._rpu_g05_vision_forward_impl
        )
    )
    monkeypatch.setitem(__import__("sys").modules, vision_module.__name__, vision_module)

    vision = types.SimpleNamespace(
        config=types.SimpleNamespace(
            depth=24,
            num_heads=16,
            hidden_size=1024,
            intermediate_size=4096,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            out_hidden_size=2048,
            hidden_act="gelu_pytorch_tanh",
            temporal_freq=0,
            spacetime_mode="factorized",
            token_drop_layer=None,
            temporal_pe_pretrain_frames=None,
            batch_all_cameras=False,
        ),
        forward=lambda *args: None,
    )
    vision.half = lambda: install_order.append("half") or vision
    model = types.SimpleNamespace(training=False, vision_tower=vision)
    assert module.patch_g05_vision_for_rpu(model) is model
    assert calls == [(vision, vision.config)]
    assert install_order == ["half", "install"]
    assert vision.forward("hidden", "grid") == (("hidden", "grid"), "pooler")
    assert vision.forward("hidden", "grid", num_frames=1, bsz=1) == (
        ("hidden", "grid"), "pooler"
    )
    assert forward_calls == [{}, {}]
    with pytest.raises(NotImplementedError, match="single-frame"):
        vision.forward("hidden", "grid", num_frames=2, bsz=1)
    with pytest.raises(NotImplementedError, match="bsz=1"):
        vision.forward("hidden", "grid", num_frames=1, bsz=2)
    assert forward_calls == [{}, {}]
    assert module.patch_g05_vision_for_rpu(model) is model
    assert len(calls) == 1


def test_g05_vision_publish_failure_rolls_back_wrapper_and_poison(monkeypatch):
    module = _load_adapter()
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    events = []
    vision_module = types.ModuleType("rpu_backend.adapters.qwen3_5.vision")

    def install(vision, *, vision_config):
        del vision_config
        events.append("install")
        vision.forward = lambda *_args, **_kwargs: "shared-forward"
        return 13

    vision_module.install_qwen3_5_vision_for_rpu = install
    vision_module._qwen3_5_vision_runtime_complete = lambda _vision: False
    vision_module._rollback_qwen3_5_vision_install = (
        lambda _vision: events.append("rollback")
    )
    monkeypatch.setitem(
        __import__("sys").modules, vision_module.__name__, vision_module
    )
    class InstallInterrupted(BaseException):
        pass

    class Model:
        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "_rpu_g05_vision_ready" and value is True:
                raise InstallInterrupted

    original_forward = lambda *_args, **_kwargs: "original-forward"
    vision = types.SimpleNamespace(
        config=types.SimpleNamespace(
            depth=24,
            num_heads=16,
            hidden_size=1024,
            intermediate_size=4096,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            out_hidden_size=2048,
            hidden_act="gelu_pytorch_tanh",
            temporal_freq=0,
            spacetime_mode="factorized",
            token_drop_layer=None,
            temporal_pe_pretrain_frames=None,
            batch_all_cameras=False,
        ),
        forward=original_forward,
    )
    vision.half = lambda: events.append("half") or vision
    model = Model()
    model.training = False
    model.vision_tower = vision

    with pytest.raises(InstallInterrupted):
        module.patch_g05_vision_for_rpu(model)

    assert events == [
        "half",
        "install",
        "rollback",
    ]
    assert vision.forward is original_forward
    assert not hasattr(vision, "_rpu_g05_vision_forward_impl")
    assert not hasattr(model, "_rpu_g05_vision_ready")
    assert model._rpu_g05_vision_install_started is True
    with pytest.raises(RuntimeError, match="already attempted"):
        module.patch_g05_vision_for_rpu(model)


def test_g05_policy_rejects_discrete_only_before_swizzle(monkeypatch):
    module = _load_adapter()
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=False,
        discrete_action=True,
        model=object(),
    )

    with pytest.raises(NotImplementedError, match="discrete_action=false"):
        module.patch_g05_policy_for_rpu(policy, max_seq_len=1024)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training": True}, "call policy.eval"),
        ({"continuous_action": False}, "continuous_action=true"),
        ({"discrete_action": True}, "discrete_action=false"),
        ({"predict_cot": True}, "predict_cot=false"),
    ],
)
def test_g05_policy_patch_rejects_unsupported_modes(monkeypatch, overrides, message):
    module = _load_adapter()
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )
    values = dict(
        training=False,
        continuous_action=True,
        discrete_action=False,
        predict_cot=False,
        model=object(),
    )
    values.update(overrides)

    with pytest.raises(NotImplementedError, match=message):
        module.patch_g05_policy_for_rpu(types.SimpleNamespace(**values))


@pytest.mark.parametrize("perf_override", [None, "0"])
def test_g05_policy_patch_connects_continuous_action(monkeypatch, perf_override):
    perf_flags = ("RPU_FASTREPLAY_SKIP_SYNC",)
    linear_acc32 = "RPU_LINEAR_ACC32"
    for name in perf_flags:
        if perf_override is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, perf_override)
    if perf_override is None:
        monkeypatch.delenv(linear_acc32, raising=False)
    else:
        monkeypatch.setenv(linear_acc32, perf_override)
    module = _load_adapter()
    assert [os.environ.get(name) for name in perf_flags] == [perf_override] * len(perf_flags)
    assert os.environ.get(linear_acc32) == perf_override
    torch = __import__("torch")
    calls = []
    encode_calls = []
    expert = types.SimpleNamespace(config=object(), training=False)

    class EMAVectorQuantize:
        def __init__(self):
            self.inited = torch.tensor(True)

    class ResidualVectorQuantize:
        training = True
        n_codebooks = 4
        quantizers = [EMAVectorQuantize() for _ in range(4)]

    class ActionCodecV2Model:
        rvq = ResidualVectorQuantize()

    class ActionCodecV2Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.model = ActionCodecV2Model()
            self.key_dims = {
                "left_control": 9,
                "left_gripper": 1,
                "right_control": 9,
                "right_gripper": 1,
                "lower_body": 7,
            }
            self._partitioner = types.SimpleNamespace(merge_preprocessor=None)

        def get_decode_metadata(self):
            return types.SimpleNamespace(nn_chunked_key_count=3)

    class VQActionTokenizer:
        action_tokenizer = ActionCodecV2Wrapper()

    class Processor:
        batchify_action = True
        action_tokenizer = VQActionTokenizer()

        def encode_inference(
            self, samples, device, mode="fm", training=False, **kwargs
        ):
            encode_calls.append((samples, device, mode, training, kwargs))
            return "encoded"

        def _prepare_segments(self, samples, control_flag):
            assert control_flag == "context_only"
            return [[types.SimpleNamespace(sample_key="command", processor_key="text")]]

    processor = Processor()

    class Model:
        pass

    model = Model()
    model.training = False
    model.vlm = types.SimpleNamespace(layers=object(), norm=object(), config=object())
    model.action_expert = expert
    model.cfg = types.SimpleNamespace(
        ae_vlm_condition_mode="cross_attn_only",
        position_ids_type="pi0fast",
    )
    model.fm_helper = _canonical_fm_helper()
    original_inference_fm = lambda *_args, **_kwargs: None
    model.inference_fm = original_inference_fm
    predict_calls = []

    def original_predict_action(batch):
        predict_calls.append(("original", batch))
        return batch

    def forward_inference(**kwargs):
        predict_calls.append(("rpu", kwargs))
        return {"action": "generated"}

    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        predict_cot=False,
        model=model,
        processor=processor,
        predict_action=original_predict_action,
        forward_inference=forward_inference,
    )
    monkeypatch.setattr(module, "_check_g05_action_profile", lambda config: None)
    monkeypatch.setattr(module, "_check_g05_vlm_profile", lambda config: None)
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda value, *, max_seq_len, _execution_config=None: calls.append(
            ("vlm", value, max_seq_len)
        ),
    )
    monkeypatch.setattr(
        module,
        "patch_g05_action_expert_for_rpu",
        lambda value, *, max_seq_len, _execution_config=None: calls.append(
            ("action", value, max_seq_len)
        ),
    )

    assert module.patch_g05_policy_for_rpu(policy, max_seq_len=1536) is policy
    expected_perf = ["1" if perf_override is None else perf_override]
    assert [os.environ.get(name) for name in perf_flags] == expected_perf
    assert os.environ.get(linear_acc32) == perf_override
    assert calls == [
        ("vlm", model, 1536),
        ("action", expert, 1536),
    ]
    assert policy._rpu_g05_policy_ready
    monkeypatch.setattr(
        module, "_g05_policy_runtime_ready", lambda owner: owner is policy
    )
    assert module.patch_g05_policy_for_rpu(policy, max_seq_len=2048) is policy
    assert len(calls) == 2
    assert model._rpu_g05_original_inference_fm is original_inference_fm
    assert model.inference_fm.__func__ is module._rpu_g05_inference_fm
    assert processor.encode_inference.__func__ is module._rpu_g05_encode_inference
    assert policy._rpu_g05_original_predict_action is original_predict_action
    assert policy.predict_action.__func__ is module._rpu_g05_predict_action
    batch = {
        "samples": ["sample"],
        "pixel_values": "pixels",
        "action": "input-action",
        "action_dim_is_pad": "padding",
        "prev_action": "previous-action",
        "inference_interval": 5,
        "inference_delay": 7,
    }
    assert policy.predict_action(batch) is batch
    assert batch["action"] == "generated"
    assert predict_calls == [
        (
            "rpu",
            {
                "samples": ["sample"],
                "pixel_values": "pixels",
                "actions": "input-action",
                "action_dim_is_pad": "padding",
                "prev_action": "previous-action",
                "inference_interval": 5,
                "inference_delay": 7,
            },
        )
    ]
    policy.training = True
    with pytest.raises(NotImplementedError, match="inference only"):
        policy.predict_action(batch)
    policy.training = False

    sample = {
        "template": "<command_text_!><EOC><action_action>",
        "action": {
            "value": torch.zeros(1),
            "action_op_mask": torch.ones(1, dtype=torch.bool),
            "action_dim_is_pad": torch.zeros(1, dtype=torch.bool),
        },
    }
    torch.manual_seed(123)
    torch.rand(3)
    torch.randint(1, 5, (3,))
    expected_rng = torch.get_rng_state()
    torch.manual_seed(123)

    assert processor.encode_inference(
        [sample], device=torch.device("cpu"), mode="ar", training=False
    ) == "encoded"
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert "action" not in encode_calls[-1][0][0]
    assert "action" in sample

    processor._prepare_segments = types.MethodType(
        lambda self, samples, control_flag: [
            [types.SimpleNamespace(sample_key="action", processor_key="action")]
        ],
        processor,
    )
    before = torch.get_rng_state().clone()
    processor.encode_inference(
        [sample], device=torch.device("cpu"), mode="ar", training=False
    )
    assert torch.equal(torch.get_rng_state(), before)
    assert "action" in encode_calls[-1][0][0]


@pytest.mark.parametrize(
    ("profile_ok", "position_ids_type", "error", "message"),
    [
        (False, "pi0fast", ValueError, "bad VLM profile"),
        (True, "default", NotImplementedError, "position_ids_type=pi0fast"),
    ],
)
def test_g05_policy_preflights_vlm_before_vision_mutation(
    monkeypatch, profile_ok, position_ids_type, error, message
):
    module = _load_adapter()
    model = types.SimpleNamespace(
        training=False,
        vlm=types.SimpleNamespace(layers=object(), norm=object(), config=object()),
        action_expert=types.SimpleNamespace(config=object(), training=False),
        cfg=types.SimpleNamespace(
            ae_vlm_condition_mode="cross_attn_only",
            position_ids_type=position_ids_type,
        ),
        fm_helper=_canonical_fm_helper(),
    )
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        predict_cot=False,
        model=model,
    )
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    monkeypatch.setattr(module, "_check_g05_action_profile", lambda config: None)

    def check_profile(config):
        if not profile_ok:
            raise ValueError("bad VLM profile")

    monkeypatch.setattr(module, "_check_g05_vlm_profile", check_profile)
    for name in (
        "patch_g05_vision_for_rpu",
        "patch_g05_vlm_for_rpu",
        "patch_g05_action_expert_for_rpu",
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"{_name} must not mutate before VLM preflight"
            ),
        )

    with pytest.raises(error, match=message):
        module.patch_g05_policy_for_rpu(policy)
    assert not hasattr(model, "_rpu_g05_vision_ready")
    assert not hasattr(model, "_rpu_g05_vlm_ready")


def test_g05_policy_preflights_proprio_before_vision_mutation(monkeypatch):
    module = _load_adapter()
    model = types.SimpleNamespace(
        training=False,
        vlm=types.SimpleNamespace(layers=object(), norm=object(), config=object()),
        action_expert=types.SimpleNamespace(config=object(), training=False),
        cfg=types.SimpleNamespace(
            ae_vlm_condition_mode="cross_attn_only",
            position_ids_type="pi0fast",
        ),
        fm_helper=_canonical_fm_helper(),
    )
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        predict_cot=False,
        model=model,
    )
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    monkeypatch.setattr(module, "_check_g05_action_profile", lambda config: None)
    monkeypatch.setattr(module, "_check_g05_vlm_profile", lambda config: None)
    for name in (
        "patch_g05_vision_for_rpu",
        "patch_g05_vlm_for_rpu",
        "patch_g05_action_expert_for_rpu",
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"{_name} must not mutate before proprio preflight"
            ),
        )

    with pytest.raises(TypeError, match="model.proprio_embedder.mlp"):
        module.patch_g05_policy_for_rpu(policy)
    assert not hasattr(model, "_rpu_g05_vision_ready")
    assert not hasattr(model, "_rpu_g05_vlm_ready")


def test_g05_policy_rejects_non_cpu_host_before_swizzle(monkeypatch):
    module = _load_adapter()
    torch = __import__("torch")
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        model=object(),
        parameters=lambda: iter(
            [torch.nn.Parameter(torch.empty(1, device="meta"))]
        ),
    )
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )

    with pytest.raises(NotImplementedError, match="CPU-hosted"):
        module.patch_g05_policy_for_rpu(policy)


@pytest.mark.parametrize("opt_in", [None, "0", "true", "01"])
def test_g05_vision_requires_controlled_numeric_opt_in(monkeypatch, opt_in):
    module = _load_adapter()
    if opt_in is None:
        monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    else:
        monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", opt_in)
    vision = types.SimpleNamespace(half=lambda: pytest.fail("must reject before casting"))
    model = types.SimpleNamespace(vision_tower=vision)
    monkeypatch.setattr(module, "_preflight_g05_vision", lambda owner: (vision, object(), (0,)))
    monkeypatch.setattr(module, "_claim_live_instance", lambda owner: pytest.fail("must reject before ownership claim"))
    with pytest.raises(NotImplementedError, match="numeric-blocked"):
        module.patch_g05_vision_for_rpu(model)
    assert not hasattr(model, "_rpu_g05_vision_install_started")
    assert not module._VISION_PATCH_LOCK.locked()


@pytest.mark.parametrize("frames", [2, 5])
def test_g05_host_packing_rejects_temporal_inputs(frames):
    import torch
    module = _load_adapter()
    vision = types.SimpleNamespace(
        patch_size=2, spatial_merge_size=1,
        config=types.SimpleNamespace(temporal_patch_size=2),
    )
    images = {"camera": torch.zeros(1, frames, 3, 4, 4)}
    with pytest.raises(NotImplementedError, match=r"\[1,1,C,H,W\]"):
        module._pack_g05_vision(images, vision)


def test_g05_vision_rejects_temporal_configuration_before_cast(monkeypatch):
    module = _load_adapter()
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    vision = types.SimpleNamespace(
        config=types.SimpleNamespace(
            depth=24, num_heads=16, hidden_size=1024, intermediate_size=4096,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
            in_channels=3, out_hidden_size=2048, hidden_act="gelu_pytorch_tanh",
            temporal_freq=2, spacetime_mode="factorized", token_drop_layer=None,
            temporal_pe_pretrain_frames=None, batch_all_cameras=False,
        ),
        half=lambda: pytest.fail("must reject before casting"),
    )
    model = types.SimpleNamespace(training=False, vision_tower=vision)
    with pytest.raises(NotImplementedError, match="single-frame profile"):
        module.patch_g05_vision_for_rpu(model)
    assert not hasattr(model, "_rpu_g05_vision_install_started")
    assert not module._VISION_PATCH_LOCK.locked()


def test_g05_data_processor_rejects_temporal_observations():
    module = _load_adapter()
    processor_type = type("GalaxeaCoTProcessor", (), {})
    processor_type.__module__ = "g05.data_processor.processor.galaxea_cot_processor"
    processor = processor_type()
    processor.is_train = False
    processor.num_obs_steps = 2
    with pytest.raises(NotImplementedError, match="single-frame"):
        module.patch_g05_data_processor_for_rpu(processor)
    assert not hasattr(processor, "_rpu_g05_data_processor_ready")

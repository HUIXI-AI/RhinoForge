"""Standalone/embedded NavDP lifecycle, with no device initialization."""
from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest
import torch

from rpu_backend.adapters.internvla_n1.execution import InternVLAExecutionController
from rpu_backend.adapters.navdp import runtime as navdp


class Cache:
    clears = 0

    def begin_warmup(self):
        pass

    def clear(self):
        self.clears += 1

    def cache_invariant_ok(self):
        return True


@pytest.fixture
def make_runtime(monkeypatch):
    destroyed = []
    monkeypatch.setattr(navdp.torch.ops.rpu, "navdp_destroy", destroyed.append, raising=False)
    owners = []

    def make(*, deferred=False, graph_cache=None):
        owner = navdp.NavdpRuntime(
            len(owners) + 1, [], {}, Cache() if graph_cache is None else graph_cache, SimpleNamespace(),
            denoise_unroll=False, batch_unroll=False,
            _defer_execution_session=deferred,
        )
        owners.append(owner)
        return owner

    yield make, destroyed
    for owner in owners:
        if owner._handle_finalizer.alive:
            owner._handle_finalizer()


def test_standalone_forward_binds_one_real_session_and_close_is_idempotent(make_runtime, monkeypatch):
    make, destroyed = make_runtime
    owner = make()
    session = owner._execution_session
    session.require_cold()
    monkeypatch.setattr(owner, "_predict_noise", lambda *_: "actual entry")
    assert owner.predict_noise(None, None, None, None) == "actual entry"
    with pytest.raises(RuntimeError, match="cold session"):
        session.require_cold()
    owner.destroy()
    owner.destroy()
    assert destroyed == [1]
    with pytest.raises(RuntimeError, match="closed"):
        owner.predict_noise(None, None, None, None)


def test_deferred_child_reuses_parent_and_rejects_unbound_execution(make_runtime, monkeypatch):
    make, _ = make_runtime
    child = make(deferred=True)
    monkeypatch.setattr(child, "_predict_noise", lambda *_: "parent entry")
    with pytest.raises(RuntimeError, match="bound to its parent"):
        child.predict_noise(None, None, None, None)
    parent = SimpleNamespace()
    controller = InternVLAExecutionController(parent, {"navdp_action": child})
    assert child._execution_session is controller.session
    assert child._standalone_execution_session is None
    assert child.predict_noise(None, None, None, None) == "parent entry"
    with pytest.raises(RuntimeError, match="cold session"):
        controller.session.require_cold()


def test_embedding_standalone_owner_rejects_before_child_config_mutation(make_runtime):
    make, _ = make_runtime
    child = make()
    before = dict(vars(child))
    with pytest.raises(ValueError, match="deferred session binding"):
        InternVLAExecutionController(SimpleNamespace(), {"navdp_action": child})
    assert vars(child) == before


def test_embedded_destroy_cannot_retire_a_live_parent_graph(make_runtime):
    make, destroyed = make_runtime
    child = make(deferred=True)
    controller = InternVLAExecutionController(SimpleNamespace(), {"navdp_action": child})
    with controller.execute():
        with pytest.raises(RuntimeError, match="through its parent"):
            child.destroy()
        assert child._handle == 1 and child._gc.clears == 0 and destroyed == []
    with pytest.raises(RuntimeError, match="through its parent"):
        child.destroy()
    controller.close()
    child.destroy()
    assert destroyed == [1] and child._handle is None


def test_standalone_hot_chunk_changes_generation_and_native_descriptor_input(make_runtime, monkeypatch):
    make, _ = make_runtime
    owner = make()
    monkeypatch.setattr(owner, "_predict_noise", lambda *_: None)
    owner.predict_noise(None, None, None, None)
    owner.reconfigure_rpu_execution({"components": {"navdp_action": {"action": {"chunk_size": 32}}}})
    assert owner._fmb_execution_generation == 1
    assert owner._gc.clears == 1
    observed = {}
    monkeypatch.setattr(navdp, "plan_native_component_execution",
                        lambda *args, **kwargs: observed.update(kwargs))
    navdp._plan_navdp_execution(
        owner._handle, 32, execution_owner=owner, generation=1, execution=owner._rpu_execution,
    )
    assert observed["execution"]["action"]["chunk_size"] == 32
    assert observed["execution_native"] == ("navdp", 1)
    assert observed["execution_owner"] is owner


def test_same_config_calibration_retires_actual_navdp_graph(make_runtime, monkeypatch):
    from rpu_backend.api._execution import record_planner_native
    from rpu_backend.runtime.execution_planner import PlannerCalibrationSelection, PlannerCostScope
    make, _ = make_runtime
    owner = make()
    session = owner._execution_session
    record_planner_native(owner, "action", "navdp_action", ("navdp", owner._handle))
    selection = PlannerCalibrationSelection(
        PlannerCostScope("a" * 64, "b" * 64, "c" * 64), "d" * 64, "e" * 64)
    old = session.config
    session._select_planner_calibration(owner, "action", selection,
                                        component="navdp_action", native_prefix="navdp")
    assert session.config == old and session.generation == 1
    assert owner._fmb_execution_generation == 1 and owner._gc.clears == 1


def test_standalone_commit_failure_compensates_and_can_retry(make_runtime, monkeypatch):
    make, _ = make_runtime
    owner = make()
    clear = owner._gc.clear

    def interrupted_clear():
        clear()
        if owner._gc.clears == 1:
            raise RuntimeError("graph cleanup interrupted")

    monkeypatch.setattr(owner._gc, "clear", interrupted_clear)
    before = owner._rpu_execution
    with pytest.raises(RuntimeError, match="cleanup interrupted"):
        owner.reconfigure_rpu_execution({"action": {"chunk_size": 32}})
    assert owner._rpu_execution == before
    assert owner._fmb_execution_generation == 0
    assert owner._execution_session.generation == 0
    assert owner._execution_session.stats()["state"] == "QUIESCENT"
    assert owner._gc.clears == 2
    owner.reconfigure_rpu_execution({"action": {"chunk_size": 32}})
    assert owner._fmb_execution_generation == 1
    assert owner._execution_session.generation == 1


@pytest.mark.parametrize("embedded", [False, True])
def test_descriptor_reconfigure_cleans_under_session_without_native_transaction(
    make_runtime, monkeypatch, embedded,
):
    from rpu_backend.api._execution import native_execution_reconfigure
    from test_graph_cache_freeze import _cache, _sig

    make, _ = make_runtime
    cache = _cache()
    cache._impl.cache_invariant_ok = lambda: True
    with cache.capture(_sig("ready")):
        pass
    cache.freeze()
    owner = make(deferred=embedded, graph_cache=cache)
    controller = (InternVLAExecutionController(SimpleNamespace(), {"navdp_action": owner})
                  if embedded else None)
    session = owner._execution_session
    state = {"active": False, "begins": 0, "clears": 0}

    def begin(attempt):
        state["active"] = True
        state["begins"] += 1
        return attempt

    def finish(token):
        state["active"] = False

    for name, fn in (("begin", begin), ("commit", finish), ("abort", finish),
                     ("abort_attempt", finish)):
        monkeypatch.setattr(torch.ops.rpu, "execution_reconfigure_" + name, fn,
                            raising=False)
    clear = cache._impl.clear

    def cleanup():
        # Model the native enter_cleanup contract, beneath the real GraphCache.
        if state["active"]:
            raise RuntimeError("cleanup reserved for owning quiesce callback")
        assert session._lock._is_owned()
        state["clears"] += 1
        clear()

    monkeypatch.setattr(cache._impl, "clear", cleanup)
    with pytest.raises(RuntimeError, match="owning quiesce"):
        with session._lock, native_execution_reconfigure():
            cache.clear()
    assert cache.size() == 1 and not state["active"]
    before_begins = state["begins"]
    reconfigure = (controller.reconfigure_rpu_execution if embedded
                   else owner.reconfigure_rpu_execution)
    reconfigure({"action": {"chunk_size": 32}})
    assert state == {"active": False, "begins": before_begins, "clears": 1}
    assert cache.size() == 0 and not cache.is_frozen()
    assert owner._fmb_execution_generation == session.generation == 1
    assert owner._rpu_execution["action"]["chunk_size"] == 32
    if controller is not None:
        controller.close()
    else:
        owner.destroy()


def test_latched_native_mode_rejects_before_upload_or_glue(make_runtime, monkeypatch):
    make, _ = make_runtime
    owner = make()
    calls = []
    def reject(handle, rows, steps):
        calls.append((handle, rows, steps))
        raise RuntimeError('latched execution mode')
    monkeypatch.setattr(navdp.torch.ops.rpu, 'navdp_validate_execution_admission', reject, raising=False)
    monkeypatch.setattr(navdp, '_hr', lambda *_: pytest.fail('uploaded before admission'))
    monkeypatch.setattr(owner, '_ensure_glue_installed', lambda: pytest.fail('prepared glue before admission'))
    with pytest.raises(RuntimeError, match='latched execution mode'):
        owner._decoder_rpu(SimpleNamespace(shape=(1, navdp.PREDICT_SIZE, navdp.HIDDEN)), None)
    with pytest.raises(RuntimeError, match='latched execution mode'):
        owner._predict_action_unroll(None, None, 1, 0, None)
    assert calls == [(owner._handle, navdp.PREDICT_SIZE, 0),
                     (owner._handle, navdp.PREDICT_SIZE, navdp.DDPM_STEPS)]


@pytest.fixture
def decoder_runtime(make_runtime, monkeypatch):
    """Exercise the public host glue with deferred device inputs and readback."""
    make, _ = make_runtime
    owner = make()
    generator = torch.Generator().manual_seed(37)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.1

    owner._glue = {
        "ie_w": rand(navdp.HIDDEN, navdp.ACTION_DIM),
        "ie_b": rand(navdp.HIDDEN),
        "cond_pos": rand(1, navdp.MEMORY_LEN, navdp.HIDDEN),
        "out_pos": rand(1, navdp.PREDICT_SIZE, navdp.HIDDEN),
        "ln_w": torch.ones(navdp.HIDDEN),
        "ln_b": torch.zeros(navdp.HIDDEN),
        "ah_w": rand(navdp.ACTION_DIM, navdp.HIDDEN),
        "ah_b": rand(navdp.ACTION_DIM),
    }
    owner._cache = SimpleNamespace(k_caches=[], v_caches=[])
    uploads, memories, readbacks, copies = [], [], [], []
    pending = []
    capture_open = False

    class DeviceInput:
        def __init__(self, value):
            self.value = value

        @property
        def shape(self):
            return self.value.shape

        def __getitem__(self, key):
            return type(self)(self.value[key])

        def view(self, *shape):
            return type(self)(self.value.view(*shape))

        def copy_(self, value):
            assert not capture_open, "CPU input copy occurred inside Graph capture"
            assert value.device.type == "cpu" and value.dtype == torch.float16
            assert value.is_contiguous() and self.value.is_contiguous()
            self.value.copy_(value)
            copies.append((tuple(value.shape), value.numel() * value.element_size(),
                           self.value.storage_offset(), self.value.untyped_storage().data_ptr()))
            return self

    class DeviceOutput(DeviceInput):

        def to(self, *, device, dtype):
            assert not capture_open, "readback preceded deferred Graph execution"
            assert device == "cpu", "readback allocated another device tensor"
            assert dtype == torch.float32
            readbacks.append(self.value.clone())
            return self.value.float()

    def upload(value):
        if isinstance(value, DeviceInput):
            return value
        result = DeviceInput(value.half().clone())
        # Avoid making the mock upload registry an extra lifetime owner.
        uploads.append((tuple(value.shape), weakref.ref(result)))
        return result

    @contextmanager
    def capture(signature):
        nonlocal capture_open
        assert not capture_open
        capture_open = True
        try:
            yield
            for target_ref, memory_ref, result in pending:
                target, memory = target_ref(), memory_ref()
                assert target is not None and memory is not None, "Graph lost a DMA input"
                memories.append((memory_ref, memory.value.clone()))
                result.value.copy_(target.value + memory.value.mean(dim=1, keepdim=True))
        finally:
            pending.clear()
            capture_open = False

    def forward(handle, target, memory, k_caches, v_caches, mask, descriptor,
                output=None, output_published=False):
        assert capture_open
        assert output_published == (output is not None)
        result = output if output is not None else DeviceOutput(torch.empty_like(target.value))
        pending.append((weakref.ref(target), weakref.ref(memory), result))
        return result

    original_empty = torch.empty

    def empty(*args, device=None, **kwargs):
        if device == "rpu":
            return DeviceOutput(original_empty(*args, **kwargs))
        return original_empty(*args, device=device, **kwargs)

    plan = SimpleNamespace(
        graph_key_words=lambda: (1,),
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(1,))),
    )
    monkeypatch.setattr(owner._gc, "capture", capture, raising=False)
    monkeypatch.setattr(navdp, "_hr", upload)
    monkeypatch.setattr(navdp.torch, "empty", empty)
    monkeypatch.setattr(navdp, "_plan_navdp_execution", lambda *a, **kw: (navdp.PREDICT_SIZE, plan))
    monkeypatch.setattr(navdp, "_publish_navdp_execution_receipt", lambda *a, **kw: None)
    monkeypatch.setattr(navdp.torch.ops.rpu, "navdp_validate_execution_admission", lambda *a: None, raising=False)
    monkeypatch.setattr(navdp.torch.ops.rpu, "navdp_forward", forward, raising=False)
    monkeypatch.setattr(navdp.torch.ops.rpu, "navdp_get_resolved_chunk_size", lambda *a: navdp.PREDICT_SIZE, raising=False)
    return owner, uploads, memories, readbacks, copies


@pytest.mark.parametrize("sample_num", (1, 4))
def test_predict_noise_batches_publication_with_fresh_outputs(decoder_runtime, sample_num):
    owner, uploads, memories, readbacks, copies = decoder_runtime
    generator = torch.Generator().manual_seed(11)
    actions = torch.randn(sample_num, navdp.PREDICT_SIZE, navdp.ACTION_DIM, generator=generator)
    goal = torch.randn(1, 1, navdp.HIDDEN, generator=generator)
    rgbd = torch.randn(1, navdp.MEMORY_LEN - 2, navdp.HIDDEN, generator=generator)

    first = owner.predict_noise(actions, torch.tensor([5.0]), goal, rgbd)
    saved = first.clone()
    original_goal = goal.clone()
    goal[:, :, 0].add_(3.0)
    second = owner.predict_noise(actions, torch.tensor([7.0]), goal, rgbd)
    restored = owner.predict_noise(actions, torch.tensor([5.0]), original_goal, rgbd)

    assert [shape for shape, _ in uploads] == [
        (navdp.MEMORY_LEN + sample_num * navdp.PREDICT_SIZE, navdp.HIDDEN)
    ] * 3
    assert len(readbacks) == 3
    assert copies == []
    assert all(value.shape == (sample_num, navdp.PREDICT_SIZE, navdp.HIDDEN)
               for value in readbacks)
    for index in range(3):
        group = memories[index * sample_num:(index + 1) * sample_num]
        assert all(ref is group[0][0] for ref, _ in group)
        assert all(torch.equal(value, group[0][1]) for _, value in group)
    assert not torch.equal(memories[0][1], memories[sample_num][1])
    assert first.shape == second.shape == (sample_num, navdp.PREDICT_SIZE, navdp.ACTION_DIM)
    assert first.device.type == second.device.type == "cpu"
    assert first.dtype == second.dtype == torch.float32
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert torch.equal(first, saved)
    assert torch.equal(first, restored)
    assert not torch.equal(first, second)
    if sample_num > 1:
        assert all(not torch.equal(readbacks[0][0], row) for row in readbacks[0][1:])


def test_direct_decoder_keeps_its_fresh_cpu_result(decoder_runtime):
    owner, uploads, memories, readbacks, copies = decoder_runtime
    target = torch.ones(1, navdp.PREDICT_SIZE, navdp.HIDDEN)
    memory = torch.ones(1, navdp.MEMORY_LEN, navdp.HIDDEN)
    first = owner._decoder_rpu(target, memory)
    second = owner._decoder_rpu(target * 3, memory)
    assert torch.equal(first, torch.full_like(first, 2))
    assert torch.equal(second, torch.full_like(second, 4))
    assert len(uploads) == 4 and len(readbacks) == 2


@pytest.fixture
def host_ddpm_loop(monkeypatch):
    # The unchanged scheduler is represented by a deterministic update here;
    # both the fresh-input control and the optimized path use these 20 steps.
    class Scheduler:
        def __init__(self, **kwargs):
            assert kwargs == dict(num_train_timesteps=20,
                                  beta_schedule="squaredcos_cap_v2",
                                  clip_sample=True, prediction_type="epsilon")

        def set_timesteps(self, steps):
            assert steps == 20
            self.timesteps = torch.arange(19, -1, -1)

        def step(self, *, model_output, timestep, sample, generator):
            return SimpleNamespace(prev_sample=sample - model_output * 0.05)

    for name in ("diffusers", "diffusers.schedulers", "diffusers.schedulers.scheduling_ddpm"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["diffusers.schedulers.scheduling_ddpm"].DDPMScheduler = Scheduler


@pytest.mark.parametrize("inference", (False, True))
def test_cpu_action_reuses_dynamic_prefix_and_refreshes_aliases_each_call(
        decoder_runtime, host_ddpm_loop, monkeypatch, inference):
    owner, uploads, memories, readbacks, copies = decoder_runtime
    generator = torch.Generator().manual_seed(12)
    goal = torch.randn(1, 1, navdp.HIDDEN, generator=generator)
    rgbd = torch.randn(1, navdp.MEMORY_LEN - 2, navdp.HIDDEN, generator=generator)
    initial = torch.randn(4, navdp.PREDICT_SIZE, navdp.ACTION_DIM, generator=generator)
    original_goal, original_rgbd = goal.clone(), rgbd.clone()
    original_initial = initial.clone()
    fresh_noise = owner._predict_noise

    with torch.inference_mode(inference):
        # Force the existing full-publication path for the same host arithmetic.
        with monkeypatch.context() as context:
            context.setattr(owner, "_predict_noise", lambda *a, **_kw: fresh_noise(*a))
            expected = owner.predict_action(goal, rgbd, init_noise=initial)
        expected_hidden = list(readbacks)
        uploads.clear()
        memories.clear()
        readbacks.clear()
        copies.clear()

        first = owner.predict_action(goal, rgbd, init_noise=initial)
        saved = first.clone()
        assert torch.equal(first, expected)
        assert all(torch.equal(a, b) for a, b in zip(readbacks, expected_hidden, strict=True))
        assert [shape for shape, _ in uploads] == [(162, navdp.HIDDEN)]
        assert [row[:3] for row in copies] == [((129, navdp.HIDDEN), 99072, 0)] * 19
        assert len({row[3] for row in copies}) == 1
        assert len(readbacks) == 20 and len(memories) == 80
        for step in range(20):
            group = memories[step * 4:(step + 1) * 4]
            assert all(torch.equal(value, group[0][1]) for _, value in group)
            assert torch.equal(group[0][1][:, 1:], memories[0][1][:, 1:])
            if step:
                assert not torch.equal(group[0][1][:, :1], memories[0][1][:, :1])
        # No action-local DeviceInput owner is retained by the runtime itself.
        assert all(ref() is None for _, ref in uploads)

        goal[..., 0].add_(0.25)
        rgbd[..., 1].sub_(0.2)
        second = owner.predict_action(goal, rgbd, init_noise=initial)
        assert torch.equal(first, saved) and not torch.equal(first, second)
        assert not torch.equal(memories[0][1][:, 1:], memories[80][1][:, 1:])
        goal.copy_(original_goal)
        rgbd.copy_(original_rgbd)
        restored = owner.predict_action(goal, rgbd, init_noise=initial)
        assert torch.equal(restored, expected) and torch.equal(first, saved)
        assert [shape for shape, _ in uploads] == [(162, navdp.HIDDEN)] * 3
        assert len(copies) == 57 and len(readbacks) == 60
        assert torch.equal(memories[0][1][:, 1:], memories[160][1][:, 1:])
        assert first.shape == (4, 32, 3) and first.dtype == torch.float32
        assert first.device.type == "cpu" and torch.isfinite(first).all()
        assert torch.equal(initial, original_initial)


@pytest.mark.parametrize("noncpu", ("goal", "rgbd", "initial"))
def test_action_noncpu_inputs_keep_full_publication_path(
        decoder_runtime, host_ddpm_loop, monkeypatch, noncpu):
    owner, uploads, memories, readbacks, copies = decoder_runtime
    values = dict(goal=torch.zeros(1, 1, navdp.HIDDEN),
                  rgbd=torch.zeros(1, navdp.MEMORY_LEN - 2, navdp.HIDDEN),
                  initial=torch.zeros(4, navdp.PREDICT_SIZE, navdp.ACTION_DIM))
    values[noncpu] = torch.empty(values[noncpu].shape, device="meta")
    banks = []

    def noise(actions, timestep, goal, rgbd, *, _action_input_bank=None):
        banks.append(_action_input_bank)
        return torch.zeros_like(actions)

    monkeypatch.setattr(owner, "_predict_noise", noise)
    owner.predict_action(values["goal"], values["rgbd"], init_noise=values["initial"])
    assert banks == [None] * 20
    assert uploads == memories == readbacks == copies == []


def test_predict_noise_rejects_latched_mode_before_shared_upload(
        decoder_runtime, host_ddpm_loop, monkeypatch):
    owner, uploads, memories, readbacks, copies = decoder_runtime

    def reject(*args):
        raise RuntimeError("latched execution mode")

    monkeypatch.setattr(navdp.torch.ops.rpu, "navdp_validate_execution_admission", reject)
    with pytest.raises(RuntimeError, match="latched execution mode"):
        owner.predict_noise(
            torch.zeros(4, navdp.PREDICT_SIZE, navdp.ACTION_DIM),
            torch.tensor([5.0]),
            torch.zeros(1, 1, navdp.HIDDEN),
            torch.zeros(1, navdp.MEMORY_LEN - 2, navdp.HIDDEN),
        )
    with pytest.raises(RuntimeError, match="latched execution mode"):
        owner.predict_action(torch.zeros(1, 1, navdp.HIDDEN),
                             torch.zeros(1, navdp.MEMORY_LEN - 2, navdp.HIDDEN))
    assert uploads == memories == readbacks == copies == []

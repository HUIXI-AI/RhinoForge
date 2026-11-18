"""CPU execution of the action seams; native DMA/coherence needs the board smoke."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.hy_vla import runtime as hy
from rpu_backend.adapters.wall_oss.action import WallOssAction


class _DeferredGraph:
    def __init__(self):
        self.pending = None

    @contextmanager
    def capture(self, _signature):
        yield
        self.pending()
        self.pending = None


@pytest.mark.parametrize("device_noise", [False, True])
@pytest.mark.parametrize("device_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_hy_unroll_refreshes_noise_and_state_without_changing_padding(
        monkeypatch, device_noise, device_state, dtype):
    cfg = SimpleNamespace(suffix_len=3, action_dim=4, expert_hidden=8,
                          num_steps=2, layers=1)
    graph = _DeferredGraph()
    expert = nullcontext()
    expert.handle, expert.gc = 7, graph
    x0 = torch.zeros(1, 3, 4, dtype=torch.float16)
    trajectory = torch.empty(2, 3, 4, dtype=torch.float16)
    owner = SimpleNamespace(
        w=SimpleNamespace(cfg=cfg, attn_tp=8), prefix_len=16, _unroll=True,
        _ux0=x0, _utraj=trajectory,
        _prepare_expert=lambda *args: (expert, (None, None, None)),
        _plan_action_execution=lambda sub: (
            SimpleNamespace(graph_key_words=lambda: ()), (), 16, {}),
        _component_generation=lambda component: 0,
        _record_execution_receipt=lambda **kwargs: None)
    cache = SimpleNamespace(k_caches=[], v_caches=[], reset_to_position=lambda pos: None)
    uploads = []

    def upload(value):
        uploads.append(value)
        return (value.tensor if isinstance(value, SimpleNamespace) else value).half()

    def native(_handle, noise, _k, _v, state, _cos, _sin, _mask, out, *args):
        assert noise is x0 and torch.count_nonzero(noise[:, :1]) == 0
        assert state.device.type == "cpu" and state.dtype == torch.float16
        assert args[-2] is True  # x0 is adapter-owned, including the cold zero row.
        if not device_noise and not device_state:
            assert args[-1].device.type == "cpu" and args[-1].is_contiguous()
            assert args[-1].dtype == torch.float16
            noise[:, 1:].copy_(args[-1])
        else:
            assert args[-1] is None
        expected = noise + state[0]
        graph.pending = lambda: out.copy_(expected.expand_as(out))

    monkeypatch.setattr(hy, "_rpu16", upload)
    monkeypatch.setattr(hy.rpu_backend.graph, "GraphSignature", lambda **kwargs: None)
    monkeypatch.setattr(torch.ops.rpu, "hyvla_expert_unroll_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvla_expert_get_resolved_chunk_size", lambda _: 16, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    base = torch.linspace(-1.003, 1.007, 16, dtype=dtype).reshape(1, 2, 8)
    outputs = []
    for offset in (0.0, 0.375, 0.0):
        # The CPU input is deliberately strided. Every live row must be refreshed.
        noise = (base + offset)[..., ::2]
        assert not noise.is_contiguous()
        state = torch.full((1, 1, 8), 0.123 + offset)
        argument = (SimpleNamespace(device=SimpleNamespace(type="rpu"), tensor=noise)
                    if device_noise else noise)
        state_argument = (
            SimpleNamespace(device=SimpleNamespace(type="rpu"),
                            reshape=lambda *shape: SimpleNamespace(tensor=state.reshape(*shape)))
            if device_state else state)
        result = hy.HyVlaRunner._run_denoise(
            owner, cache, 32, None, None, None, argument, state_argument)
        assert result.dtype == torch.float32 and result.device.type == "cpu"
        assert torch.equal(result, (noise.half() + state.flatten()[0].half()).float())
        assert torch.equal(x0[:, 1:], noise.half())
        assert torch.count_nonzero(x0[:, :1]) == 0
        outputs.append(result)
    assert torch.equal(outputs[0], outputs[2])
    assert not torch.equal(outputs[0], outputs[1])
    assert outputs[0].data_ptr() != outputs[2].data_ptr()
    # CPU noise/state go straight to their final owners; device inputs retain
    # the original conversion entry.
    assert len(uploads) == 3 * (device_noise + device_state)


@pytest.mark.parametrize("device_images", [(), (1,), (0, 1, 2)])
def test_hy_packed_images_publish_only_owned_cpu_uploads(monkeypatch, device_images):
    graph = _DeferredGraph()
    sub = nullcontext()
    sub.handle, sub.gc = 7, graph
    cfg = SimpleNamespace(vit_seq=4, vit_layers=1, vit_head_dim_padded=16,
                          vit_heads=1)
    captured, uploads = [], []
    cache = SimpleNamespace(k_caches=[None], v_caches=[None],
                            reset_to_position=lambda pos: None)
    owner = SimpleNamespace(
        w=SimpleNamespace(cfg=cfg), _vit_cache=cache, _vit_packed=True, _vision_lanes=8,
        _pe_in_graph=True, _merger_in_graph=False, _prefix_template=True,
        _prepare_vit=lambda n: sub, _vit_out_dim=lambda: 3,
        _plan_vision_execution=lambda *args: (
            SimpleNamespace(graph_key_words=lambda: ()), (), 16, {}),
        _component_generation=lambda component: 0,
        _record_execution_receipt=lambda **kwargs: None,
        _merger_rest=lambda values: values)

    def upload(value):
        uploads.append(value)
        return (value.tensor if isinstance(value, SimpleNamespace) else value).half()

    def native(_handle, images, _k, _v, _descriptor, inputs_published=False):
        assert inputs_published is (not device_images)
        assert all(im.is_contiguous() and im.shape == (1, 3, 2, 2) for im in images)
        out = torch.empty(1, 12, 3, dtype=torch.float16)

        def execute():
            # Input owners must remain live until capture exits. Camera order
            # and disjoint row ranges must survive the single packed upload.
            expected = torch.cat([im.reshape(1, 4, 3) for im in images])
            out.copy_(expected.reshape_as(out))
        graph.pending = execute
        captured.append(images)
        return out

    monkeypatch.setattr(hy, "_rpu16", upload)
    monkeypatch.setattr(hy.rpu_backend.graph, "GraphSignature", lambda **kwargs: None)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_forward_multi", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_get_resolved_chunk_size", lambda _: 16, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    saved = []
    for delta in (0.0, 0.5, 0.0):
        values = [torch.arange(24).reshape(1, 3, 2, 4)[..., ::2].float() + delta + i
                  for i in range(3)]
        images = [SimpleNamespace(device=SimpleNamespace(type="rpu"), tensor=value,
                                  ndim=value.ndim, shape=value.shape)
                  if i in device_images else value for i, value in enumerate(values)]
        result = hy.HyVlaRunner._run_vit(owner, images)
        assert all(torch.equal(got, want.half().reshape(4, 3))
                   for got, want in zip(result, values))
        if not device_images:
            assert len({im.untyped_storage().data_ptr() for im in captured[-1]}) == 1
        saved.append(result)
    assert all(torch.equal(a, b) for a, b in zip(saved[0], saved[2]))
    assert not torch.equal(saved[0][0], saved[1][0])
    assert len(uploads) == (9 if device_images else 3)


@pytest.mark.parametrize("action_dim", [26, 32])
def test_wall_unroll_reads_final_logical_rows_as_independent_cpu_float(
        monkeypatch, action_dim):
    graph = _DeferredGraph()
    h, steps, prefix = 3, 2, 16
    plan = SimpleNamespace(
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(1,))),
        graph_key_words=lambda: ())
    owner = SimpleNamespace(
        action_dim=action_dim, _k_comb_pad=32, _action_dim_pad=32, _handle=7,
        llm=SimpleNamespace(cache=SimpleNamespace(position=prefix, k_caches=[], v_caches=[])),
        _graph_cache=graph, validate_execution_horizon=lambda rows: None,
        _rtc_inputs=lambda pos, actions, delay, rows: (pos, None, None),
        _action_position_ids_rpu=lambda pos, key: pos,
        _partial_mrope_tables=lambda pos, key: (None, None),
        _action_mask_cached=lambda *args, **kwargs: None,
        _flow_times=lambda n: torch.linspace(1, 0, n + 1),
        _denoise_bias_inputs=lambda *args: (True, torch.ones(8), torch.ones(steps, 8)),
        _velocity_blend_inputs=lambda *args: (None, None),
        _plan_action_chunk=lambda *args, **kwargs: (16, plan),
        _denoise_graph_signature=lambda *args, **kwargs: None,
        _record_action_plan=lambda *args, **kwargs: None)
    original_to, original_empty = torch.Tensor.to, torch.empty

    def to_cpu(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    def empty_cpu(*args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    trajectories = []

    def native(_handle, x0, _k, _v, _bias, _time, _mask, _pos, x_traj, v_traj, *args):
        trajectories.append(x_traj)

        def execute():
            x_traj.fill_(77)  # Other steps and padded columns must not escape.
            x_traj[-1:, :, :action_dim].copy_(x0[:, :, :action_dim] + 0.25)
        graph.pending = execute

    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    monkeypatch.setattr(torch, "empty", empty_cpu)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "wall_oss_action_denoise_loop_forward", native, raising=False)
    noise = torch.linspace(-0.99, 1.01, h * action_dim).reshape(1, h, action_dim)
    positions = torch.arange(prefix, prefix + h, dtype=torch.int32)[:, None].expand(h, 3).contiguous()
    outputs = []
    for offset in (0.0, 0.5, 0.0):
        current = noise + offset
        result = WallOssAction._denoise_with_mask_loop(
            owner, prefix, current, torch.ones_like(noise), positions, num_steps=steps)
        output = result["x_norm"]
        assert output.dtype == torch.float32 and output.device.type == "cpu"
        assert tuple(output.shape) == (1, h, action_dim)
        assert torch.equal(output, (current.half() + 0.25).float())
        assert result["xs"] is result["vs"] is None
        trajectories[-1].fill_(-33)
        assert torch.equal(output, (current.half() + 0.25).float())
        outputs.append(output)
    assert torch.equal(outputs[0], outputs[2])
    assert not torch.equal(outputs[0], outputs[1])


def test_hy_native_cpu_action_batch_validates_before_writing(tmp_path):
    from pathlib import Path
    from test_siglip_attention_dry_live_contract import _function, _run_native

    source = (Path(__file__).resolve().parents[1] /
              "src/fused/rpu_hyvla_expert_model.cpp").read_text()
    body = _function(source, "void HyVlaExpertModel::unroll_forward(")
    guards = body[body.index("    TORCH_CHECK(state_emb.dim()"):
                  body.index("    TORCH_CHECK(attn_mask_4d.defined()")]
    writes = _function(body[body.index("    state_ref_ = state_emb;"):],
                       "    if (noise_cpu.has_value()) {")
    code = r'''
#include <ATen/ATen.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
bool g_rpu_ddr_flush_force_enabled = true;
int batch_calls = 0;
void* expected_noise = nullptr;
void* expected_state = nullptr;
size_t expected_noise_bytes = 16;
namespace rhino_lkn {
void RpuDdrFlushBatch(void** ptrs, size_t* sizes, size_t count) {
    assert(count == 2 && ptrs[0] == expected_noise && ptrs[1] == expected_state);
    assert(sizes[0] == expected_noise_bytes && sizes[1] == 16);
    if (sizes[0] == 24) assert(static_cast<c10::Half*>(ptrs[0])[0] == c10::Half(11));
    ++batch_calls;
}
}
void stage(at::Tensor& x0_rpu, at::Tensor& emb_stage_,
           const at::Tensor& state_emb, const std::optional<at::Tensor>& noise_cpu,
           bool inputs_published = true) {
    const int64_t cs = x0_rpu.size(1), np = x0_rpu.size(2), h = emb_stage_.size(2);
''' + guards + writes + r'''
}
int main() {
    auto opts = at::TensorOptions().dtype(at::kHalf);
    auto x0 = at::full({1, 3, 4}, -7, opts);
    auto emb = at::full({1, 3, 8}, -9, opts);
    auto noise = at::arange(8, opts).reshape({1, 2, 4});
    auto state = at::arange(8, opts) + 10;
    expected_noise = x0.data_ptr<c10::Half>() + 4;
    expected_state = emb.data_ptr();
    for (int delta : {0, 2, 0}) {
        auto n = noise + delta, s = state + delta;
        stage(x0, emb, s, n);
        assert(at::equal(x0.slice(1, 1), n));
        assert(at::equal(emb[0][0], s));
        assert(x0[0][0].eq(-7).all().item<bool>());
        assert(emb.slice(1, 1).eq(-9).all().item<bool>());
    }
    assert(batch_calls == 3);
    // A default public call also publishes an independently dirtied row 0.
    x0[0][0].fill_(11);
    expected_noise = x0.data_ptr();
    expected_noise_bytes = x0.nbytes();
    stage(x0, emb, state, noise, false);
    assert(batch_calls == 4 && x0[0][0].eq(11).all().item<bool>());
    auto reject = [&](const at::Tensor& n, const at::Tensor& s) {
        auto before_x = x0.clone(), before_e = emb.clone();
        bool caught = false;
        try { stage(x0, emb, s, n); } catch (const c10::Error&) { caught = true; }
        assert(caught && batch_calls == 4);
        assert(at::equal(before_x, x0) && at::equal(before_e, emb));
    };
    reject(noise.slice(1, 0, 1), state);
    reject(noise.to(at::kFloat), state);
    reject(at::ones({1, 2, 8}, opts).slice(2, 0, 8, 2), state);
    reject(at::empty({1, 2, 4}, opts.device(at::kMeta)), state);
    reject(noise, state.slice(0, 0, 7));
    reject(noise, state.to(at::kFloat));
    reject(noise, at::ones({16}, opts).slice(0, 0, 16, 2));
    reject(noise, at::empty({8}, opts.device(at::kMeta)));
    reject(x0.slice(1, 1), state);
    reject(noise, x0.view({-1}).slice(0, 0, 8));
    reject(noise, emb[0][0]);
}
'''
    _run_native(tmp_path, code, torch)


class _Fp16DeviceOutput:
    """Host stand-in that forbids widening the producer before CPU readback."""

    def __init__(self, tensor, graph, reads):
        assert tensor.dtype == torch.float16
        self.tensor, self.graph, self.reads = tensor, graph, reads

    def reshape(self, *shape):
        return type(self)(self.tensor.reshape(*shape), self.graph, self.reads)

    def __getitem__(self, index):
        return type(self)(self.tensor[index], self.graph, self.reads)

    def to(self, *, device, dtype):
        assert device == "cpu" and dtype == torch.float32
        assert self.graph.pending is None  # The native graph has completed.
        self.reads.append((tuple(self.tensor.shape), self.tensor.stride()))
        return self.tensor.to(device=device, dtype=dtype)

    def float(self):
        raise AssertionError("FP16 producer must be read directly into CPU FP32")


def _cpu_device_allocations(monkeypatch):
    original_to, original_empty = torch.Tensor.to, torch.empty

    def to_cpu(tensor, *args, **kwargs):
        args = tuple("cpu" if isinstance(arg, str) and arg == "rpu" else arg
                     for arg in args)
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    def empty_cpu(*args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    monkeypatch.setattr(torch, "empty", empty_cpu)


def _action_plan():
    return SimpleNamespace(
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(1,))),
        graph_key_words=lambda: ())


def test_gr00t_action_reads_padded_fp16_after_capture_and_retains_cpu_crop(monkeypatch):
    from rpu_backend.adapters.gr00t import runtime as gr

    for name, value in {"_AH": 3, "_AD": 4, "_AD_PAD": 16,
                        "_H": 8, "_NL": 1, "_ACTION_QUERY_ROWS": 4}.items():
        monkeypatch.setattr(gr, name, value)
    monkeypatch.setattr(gr, "_half_rpu", lambda value: value.half())
    monkeypatch.setattr(gr.rpu_backend.graph, "GraphSignature", lambda **kwargs: None)
    graph, reads, producers = _DeferredGraph(), [], []
    ids = torch.tensor([[1, 2, 3]])
    attn = torch.ones_like(ids)
    cache = SimpleNamespace(k_caches=[], v_caches=[])
    owner = SimpleNamespace(
        _masks=(torch.zeros(4, 3), torch.zeros(4, 3)), _mask_key=ids.clone(),
        _mask_attn_key=attn.clone(), _img_tok=2, _d_gc=graph, _d_handle=7,
        _d_cache=cache, _component_generations={gr._ACTION_COMPONENT: 0},
        _record_component_plan=lambda *args, **kwargs: None)

    def native(_handle, noise, state, *args):
        out = torch.empty_like(noise)
        producers.append(out)
        graph.pending = lambda: out.copy_(noise + state)
        return _Fp16DeviceOutput(out, graph, reads)

    monkeypatch.setattr(torch.ops.rpu, "gr00t_denoise_unroll_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "gr00t_denoise_get_resolved_chunk_size", lambda _: 16, raising=False)
    outputs = []
    for delta in (0.0, 0.375, 0.0):
        state = torch.full((1, 1, 4), delta)
        result = gr.Gr00tN1d7VLA._denoise(
            owner, None, {"input_ids": ids, "attention_mask": attn, "state": state},
            3, 0, execution_plan=(16, _action_plan()))
        expected = (torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(0)).half()
                    + state.half()).float()
        assert result.device.type == "cpu" and result.dtype == torch.float32
        assert result.shape == (1, 3, 4) and result.stride() == (48, 16, 1)
        assert torch.equal(result, expected)
        producers[-1].fill_(-99)
        assert torch.equal(result, expected)
        outputs.append(result)
    assert reads == [((1, 3, 16), (48, 16, 1))] * 3
    assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])
    assert outputs[0].data_ptr() != outputs[2].data_ptr()


def test_rgbd_former_reads_each_completed_fp16_output_in_batch_order(monkeypatch):
    from rpu_backend.adapters.internvla_n1 import rgbd_rpu as rgbd

    graph, reads, producers = _DeferredGraph(), [], []
    cache = SimpleNamespace(k_caches=[], v_caches=[])
    owner = SimpleNamespace(handle=7, NQ=3, H=4, ML=5, gc=graph, cache=cache,
                            _fmb_execution_generation=0, _fmb_execution_component_id="former",
                            _rpu_execution={}, _input_cache={})
    monkeypatch.setattr(rgbd, "_hr", lambda value: value.half())
    monkeypatch.setattr(rgbd, "plan_navdp_component", lambda *args, **kwargs: (16, _action_plan()))
    monkeypatch.setattr(rgbd, "publish_execution_receipt", lambda *args, **kwargs: None)
    monkeypatch.setattr(rgbd.rpu_backend.graph, "GraphSignature", lambda **kwargs: None)

    def native(_handle, q, memory, *args):
        out = torch.empty_like(q)
        producers.append(out)
        graph.pending = lambda: out.copy_(q + memory.mean(1, keepdim=True))
        return _Fp16DeviceOutput(out, graph, reads)

    monkeypatch.setattr(torch.ops.rpu, "navdp_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "navdp_get_resolved_chunk_size", lambda _: 16, raising=False)
    outputs = []
    q = torch.linspace(-1.01, 1.03, 48).reshape(2, 3, 8)[..., ::2]
    memory = torch.linspace(-0.81, 0.73, 80).reshape(2, 5, 8)[..., ::2]
    assert not q.is_contiguous() and not memory.is_contiguous()
    for delta in (0.0, 0.375, 0.0):
        result = rgbd.FormerRPU.forward(owner, q + delta, memory)
        expected = ((q + delta).half() + memory.half().mean(1, keepdim=True)).float()
        assert result.shape == (2, 3, 4) and result.is_contiguous()
        assert result.dtype == torch.float32 and result.device.type == "cpu"
        assert torch.equal(result, expected)
        for producer in producers:
            producer.fill_(-99)
        assert torch.equal(result, expected)
        outputs.append(result)
    assert len(reads) == 6 and all(shape == (1, 3, 4) for shape, _ in reads)
    assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])
    assert outputs[0].data_ptr() != outputs[2].data_ptr()


def test_hy_nonunroll_reads_before_temporary_reset_and_cpu_projection(monkeypatch):
    graph, reads, producers = _DeferredGraph(), [], []
    exp = nullcontext()
    exp.handle, exp.gc = 7, graph
    cfg = SimpleNamespace(suffix_len=3, action_dim=2, expert_hidden=4,
                          num_steps=2, layers=1, n_action=2)
    wt = torch.linspace(-0.7, 0.9, 8).reshape(4, 2)
    bias = torch.tensor([0.123, -0.234])
    owner = SimpleNamespace(
        w=SimpleNamespace(cfg=cfg, attn_tp=8), prefix_len=16, _unroll=False,
        _prepare_expert=lambda *args: (exp, (None, None, None)),
        _plan_action_execution=lambda sub: (_action_plan(), (1,), 16, {}),
        _component_generation=lambda component: 0,
        _record_execution_receipt=lambda **kwargs: None,
        _embed=lambda xt, k: torch.nn.functional.pad(xt, (0, 2)) + 0.13 * k,
        _out_proj_b=bias, _out_proj_wt=wt)
    cache = SimpleNamespace(k_caches=[], v_caches=[], reset_to_position=lambda pos: None)
    monkeypatch.setattr(hy, "_rpu16", lambda value: value.half())
    monkeypatch.setattr(hy.rpu_backend.graph, "GraphSignature", lambda **kwargs: None)

    def native(_handle, emb, *args):
        out = torch.empty_like(emb)
        producers.append(out)
        graph.pending = lambda: out.copy_(emb + 0.25)
        return _Fp16DeviceOutput(out, graph, reads)

    resets = []
    monkeypatch.setattr(torch.ops.rpu, "hyvla_expert_step_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvla_expert_get_resolved_chunk_size", lambda _: 16, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: resets.append(len(reads)), raising=False)
    outputs = []
    noise = torch.linspace(-1.01, 1.03, 8).reshape(1, 2, 4)[..., ::2]
    state = torch.zeros(1, 1, 4)
    for delta in (0.0, 0.375, 0.0):
        current = noise + delta
        result = hy.HyVlaRunner._run_denoise(owner, cache, 19, None, None, None, current, state)
        expected = current.clone()
        for k in range(2):
            emb = torch.cat([state, owner._embed(expected, k)], 1).half()
            so = (emb + 0.25).float()
            expected = expected - 0.5 * torch.addmm(bias, so[0, -2:], wt).reshape(1, 2, 2)
        assert result.shape == (1, 2, 2) and result.dtype == torch.float32
        assert result.device.type == "cpu" and torch.equal(result, expected)
        for producer in producers:
            producer.fill_(-99)
        assert torch.equal(result, expected)
        outputs.append(result)
    assert resets == list(range(1, 7)) and len(reads) == 6
    assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("batch_unroll", [False, True])
def test_navdp_unroll_reads_logical_strided_fp16_actions_and_retains_output(monkeypatch, batch_unroll):
    from rpu_backend.adapters.navdp import runtime as nav

    _cpu_device_allocations(monkeypatch)
    for name, value in {"PREDICT_SIZE": 3, "MEMORY_LEN": 4, "HIDDEN": 4,
                        "NUM_LAYERS": 1, "DDPM_STEPS": 2}.items():
        monkeypatch.setattr(nav, name, value)
    graph, reads, producers = _DeferredGraph(), [], []
    cache = SimpleNamespace(k_caches=[], v_caches=[])
    owner = SimpleNamespace(
        _batch_unroll=batch_unroll, _batch_n=2, _batch_cache=cache, _cache=cache,
        _gc=graph, _handle=7, _fmb_execution_generation=0, _rpu_execution={},
        _ensure_glue_installed=lambda: None,
        _time_emb=lambda time: torch.zeros(1, 4),
        _glue={"cond_pos": torch.zeros(1, 4, 4)})
    monkeypatch.setattr(nav, "_plan_navdp_execution", lambda *args, **kwargs: (16, _action_plan()))
    monkeypatch.setattr(nav, "_publish_navdp_execution_receipt", lambda *args, **kwargs: None)
    monkeypatch.setattr(nav.rpu_backend.graph, "GraphSignature", lambda **kwargs: None)
    monkeypatch.setattr(torch.ops.rpu, "navdp_validate_execution_admission", lambda *args: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "navdp_get_resolved_chunk_size", lambda _: 16, raising=False)

    def native(_handle, x0, _memory, _sigz, _coeff, _k, _v, out, _descriptor):
        producers.append(out)
        graph.pending = lambda: out.copy_(x0 + 0.25)
        return _Fp16DeviceOutput(out, graph, reads)

    monkeypatch.setattr(torch.ops.rpu, "navdp_denoise_loop_forward", native, raising=False)
    noise = torch.linspace(-1.01, 1.03, 36).reshape(2, 3, 6)[..., ::2]
    outputs = []
    for delta in (0.0, 0.375, 0.0):
        current = noise + delta
        result = nav.NavdpRuntime._predict_action_unroll(
            owner, torch.zeros(1, 1, 4), torch.zeros(1, 2, 4), 2, 0, current)
        expected = (current.half() + 0.25).float()
        assert result.shape == (2, 3, 3) and result.is_contiguous()
        assert result.device.type == "cpu" and result.dtype == torch.float32
        assert torch.equal(result, expected)
        for producer in producers:
            producer.fill_(-99)
        assert torch.equal(result, expected)
        outputs.append(result)
    expected_read = ((2, 3, 3), (48, 16, 1)) if batch_unroll else ((3, 3), (16, 1))
    assert reads == [expected_read] * (3 if batch_unroll else 6)
    assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])
    assert outputs[0].data_ptr() != outputs[2].data_ptr()

import importlib.util
from pathlib import Path
import sys
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
_RPU_TEST_LIBRARIES = []

def _define_schema(lib, schema):
    try:
        lib.define(schema)
    except RuntimeError as exc:
        message = str(exc)
        if not any((marker in message for marker in ('already defined', 'Duplicate registration'))):
            raise

def _load_fake_dispatch_with_test_schemas():
    lib = torch.library.Library('rpu', 'FRAGMENT')
    _RPU_TEST_LIBRARIES.append(lib)
    for schema in (
        "linear_w4a16(Tensor input, Tensor weight, Tensor scale, Tensor? bias=None, int logical_out_features=-1) -> Tensor",
        "rms_norm(Tensor input, int[] normalized_shape, Tensor weight, float eps=1e-5) -> Tensor",
        "apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, Tensor position_ids) -> (Tensor, Tensor)",
        "scaled_dot_product_attention_unified(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, float? scale=None, bool enable_gqa=False) -> Tensor",
        "causal_decoder_forward(int handle, Tensor hidden_states, Tensor[] k_caches, Tensor[] v_caches, Tensor? attention_mask, int position, bool is_causal, Tensor? position_ids=None, Tensor[]? deepstack_dense_visual_embeds=None, Tensor? rope_cos_il=None, Tensor? rope_sin_il=None, int cos_sin_offset=-1, int batch_slot=0, bool allow_batch_decode=False, int planned_chunk_size=0, int[] planned_stage_descriptor=[]) -> Tensor",
        "lm_head_logits_top1(Tensor logits) -> Tensor",
        "argmax_lastdim_host(Tensor logits) -> Tensor",
        "gemma_forward(int handle, Tensor hidden_states, Tensor[] k_caches, Tensor[] v_caches, Tensor attention_mask, int position, bool is_causal, int chunk_size=0, int[] planned_stage_descriptor=[]) -> Tensor",
        "gemma2_forward(int handle, Tensor hidden_states, Tensor[] k_caches, Tensor[] v_caches, Tensor attention_mask, int position, bool is_causal, int chunk_size=0) -> Tensor",
        "adarms_forward(int handle, Tensor hidden_states, Tensor cond, Tensor[] k_caches, Tensor[] v_caches, Tensor attention_mask, int position, bool is_causal, int[] planned_stage_descriptor=[]) -> Tensor",
        "siglip_forward(int handle, Tensor input, Tensor[] k_caches, Tensor[] v_caches, int[] planned_stage_descriptor=[]) -> Tensor",
        "siglip_patch_embed(int handle, Tensor input) -> Tensor",
        "siglip_patch_embed_multi(int handle, Tensor[] images) -> Tensor",
        "siglip_forward_packed(int handle, Tensor hidden, int image_batch_count, Tensor[] k_caches, Tensor[] v_caches, int[] planned_stage_descriptor=[]) -> Tensor",
        "siglip_forward_multi(int handle, Tensor[] images, Tensor[] k_caches, Tensor[] v_caches, int[] planned_stage_descriptor=[]) -> Tensor",
        "conv2d_mc_nhwc(Tensor input, Tensor weight, int[] stride=[], int[] padding=[], int[] dilation=[], int groups=1) -> Tensor",
        "halo_action_expert_step_forward(int handle, Tensor x_emb, Tensor[] k_caches, Tensor[] v_caches, Tensor cos, Tensor sin, Tensor attn_mask_4d, int prefix_len, int[] planned_stage_descriptor=[]) -> Tensor",
        "halo_image_flow_step_forward(int handle, Tensor x_emb, Tensor[] k_caches, Tensor[] v_caches, Tensor cos, Tensor sin, int prefix_len, int[] planned_stage_descriptor=[]) -> Tensor",
        "hyvit2_patch_embed(int handle, Tensor input) -> Tensor",
        "hyvit2_patch_embed_multi(int handle, Tensor[] images) -> Tensor",
        "hyvit2_forward(int handle, Tensor input, Tensor[] k_caches, Tensor[] v_caches) -> Tensor",
        "hyvit2_forward_packed(int handle, Tensor hidden, int image_batch_count, Tensor[] k_caches, Tensor[] v_caches) -> Tensor",
        "hyvit2_forward_multi(int handle, Tensor[] images, Tensor[] k_caches, Tensor[] v_caches) -> Tensor",
        "hyvla_merger_pool(Tensor nxc) -> Tensor",
        "hyvla_merger_combine(Tensor nxc, Tensor sc) -> Tensor",
        "hyvla_merger_fused(Tensor nxc, Tensor wa, Tensor b0, Tensor wb, Tensor w2, Tensor b2, Tensor wp2, Tensor bp2, Tensor? wp1=None, Tensor? bp1=None) -> Tensor",
        "hyvla_vlm_step_forward(int handle, Tensor x_emb, Tensor[] k_caches, Tensor[] v_caches, Tensor cos, Tensor sin, Tensor attn_mask_4d, int prefix_len) -> Tensor",
        "hyvla_vlm_prefill_kv_only(int handle, Tensor x_emb, Tensor[] k_caches, Tensor[] v_caches, Tensor cos, Tensor sin, Tensor attn_mask_4d, int prefix_len, int[] planned_stage_descriptor=[]) -> ()",
        "hyvla_expert_step_forward(int handle, Tensor x_emb, Tensor[] k_caches, Tensor[] v_caches, Tensor cos, Tensor sin, Tensor attn_mask_4d, int prefix_len) -> Tensor",
        "hyvla_expert_unroll_forward(int handle, Tensor x0_rpu, Tensor[] k_caches, Tensor[] v_caches, Tensor state_emb, Tensor cos, Tensor sin, Tensor attn_mask_4d, Tensor x_traj, float dt, int prefix_len, int num_steps) -> ()",
    ):
        _define_schema(lib, schema)
    if 'rpu_backend.graph.fake_dispatch' in sys.modules:
        return
    fake_dispatch_path = Path(__file__).resolve().parents[2] / 'graph' / 'fake_dispatch.py'
    spec = importlib.util.spec_from_file_location('_rpu_fake_dispatch_under_test', fake_dispatch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
_load_fake_dispatch_with_test_schemas()

def _assert_fake_shape(tensor, shape):
    assert isinstance(tensor, FakeTensor)
    assert tuple(tensor.shape) == shape
    assert tensor.dtype == torch.float16

def test_linear_w4a16_fake_dispatch_preserves_rows_and_scale_contract():
    with FakeTensorMode():
        hidden = torch.empty((1,17,256),dtype=torch.float16)
        packed = torch.empty((256,128),dtype=torch.uint8)
        scale = torch.empty((32,128),dtype=torch.float16)
        actual = torch.ops.rpu.linear_w4a16(hidden,packed,scale,logical_out_features=128)
        _assert_fake_shape(actual,(1,17,128))
        selected = torch.ops.rpu.linear_w4a16(hidden[:,-1:,:],packed,scale,logical_out_features=128)
        _assert_fake_shape(selected,(1,1,128))
        with pytest.raises(RuntimeError,match="scale"):
            torch.ops.rpu.linear_w4a16(hidden,packed,scale[:,:-1].contiguous(),logical_out_features=128)


def test_siglip_fake_dispatch_for_split_and_packed_paths():
    with FakeTensorMode():
        image_batch = torch.empty((3, 3, 224, 224), dtype=torch.float16)
        hidden = torch.ops.rpu.siglip_patch_embed(1, image_batch)
        _assert_fake_shape(hidden, (1, 3 * 256, 1152))
        packed = torch.ops.rpu.siglip_forward_packed(1, hidden, 3, [], [])
        _assert_fake_shape(packed, (1, 3 * 256, 2048))
        direct = torch.ops.rpu.siglip_forward(1, image_batch, [], [])
        _assert_fake_shape(direct, (1, 3 * 256, 2048))
        hidden_direct = torch.empty((1, 3 * 256, 1152), dtype=torch.float16)
        direct_3d = torch.ops.rpu.siglip_forward(1, hidden_direct, [], [])
        _assert_fake_shape(direct_3d, (1, 3 * 256, 2048))

def test_siglip_fake_dispatch_for_multi_image_path():
    with FakeTensorMode():
        images = [torch.empty((1, 3, 224, 224), dtype=torch.float16) for _ in range(3)]
        hidden = torch.ops.rpu.siglip_patch_embed_multi(1, images)
        _assert_fake_shape(hidden, (1, 3 * 256, 1152))
        output = torch.ops.rpu.siglip_forward_multi(1, images, [], [])
        _assert_fake_shape(output, (1, 3 * 256, 2048))

def test_causal_decoder_fake_dispatch_accepts_current_tail_args():
    with FakeTensorMode():
        hidden = torch.empty((1, 4, 1024), dtype=torch.float16)
        attention_mask = torch.empty((1, 1, 4, 4), dtype=torch.float16)
        position_ids = torch.empty((1, 4), dtype=torch.long)
        dense_visual = [torch.empty((1, 4, 1024), dtype=torch.float16)]
        out = torch.ops.rpu.causal_decoder_forward(1, hidden, [], [], attention_mask, 0, False, position_ids, dense_visual)
        _assert_fake_shape(out, (1, 4, 1024))

def test_halo_action_expert_fake_dispatch_returns_hidden_shape():
    with FakeTensorMode():
        x_emb = torch.empty((1, 18, 1536), dtype=torch.float16)
        cos = torch.empty((4724, 64), dtype=torch.float16)
        sin = torch.empty((4724, 64), dtype=torch.float16)
        attn_mask_4d = torch.empty((1, 1, 18, 4724), dtype=torch.float16)
        out = torch.ops.rpu.halo_action_expert_step_forward(1, x_emb, [], [], cos, sin, attn_mask_4d, 4706)
        _assert_fake_shape(out, (1, 18, 1536))

def test_halo_image_flow_fake_dispatch_returns_hidden_shape():
    with FakeTensorMode():
        x_emb = torch.empty((1, 1378, 1536), dtype=torch.float16)
        cos = torch.empty((10964, 64), dtype=torch.float16)
        sin = torch.empty((10964, 64), dtype=torch.float16)
        out = torch.ops.rpu.halo_image_flow_step_forward(1, x_emb, [], [], cos, sin, 9586)
        _assert_fake_shape(out, (1, 1378, 1536))

def test_hyvla_fake_dispatch_fails_fast_for_unsupported_compile_path():
    message = 'HyEmbodiedPolicy/Hy-VLA torch.compile is unsupported'
    with FakeTensorMode():
        hidden = torch.empty((1, 192, 1152), dtype=torch.float16)
        with pytest.raises(RuntimeError, match=message):
            torch.ops.rpu.hyvit2_forward_packed(1, hidden, 3, [], [])
        nxc = torch.empty((4, 147, 1152), dtype=torch.float16)
        weights = [torch.empty((1152, 1152), dtype=torch.float16) for _ in range(4)]
        biases = [torch.empty((1152,), dtype=torch.float16) for _ in range(3)]
        with pytest.raises(RuntimeError, match=message):
            torch.ops.rpu.hyvla_merger_fused(nxc, weights[0], biases[0], weights[1], weights[2], biases[1], weights[3], biases[2], None, None)
        x0 = torch.empty((1, 51, 32), dtype=torch.float16)
        state = torch.empty((1024,), dtype=torch.float16)
        cos = torch.empty((291, 64), dtype=torch.float16)
        sin = torch.empty_like(cos)
        mask = torch.empty((1, 1, 51, 291), dtype=torch.float16)
        with pytest.raises(RuntimeError, match=message):
            torch.ops.rpu.hyvla_vlm_prefill_kv_only(1, hidden, [], [], cos, sin, mask, 0, [])
        trajectory = torch.empty((11, 1, 51, 32), dtype=torch.float16)
        with pytest.raises(RuntimeError, match=message):
            torch.ops.rpu.hyvla_expert_unroll_forward(1, x0, [], [], state, cos, sin, mask, trajectory, -0.1, 240, 10)

def test_hyvla_tensor_ops_fall_through_autograd_before_python_fake_dispatch():
    root = Path(__file__).resolve().parents[4]
    registrations = (root / 'src' / 'core' / 'rpu_dispatch_registrations.inc').read_text(encoding='utf-8')
    autograd = registrations.split('TORCH_LIBRARY_IMPL(rpu, AutogradPrivateUse1, m) {', 1)[1].split('TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {', 1)[0]
    for op in ('hyvit2_patch_embed', 'hyvit2_patch_embed_multi', 'hyvit2_forward', 'hyvit2_forward_packed', 'hyvit2_forward_multi', 'hyvla_merger_pool', 'hyvla_merger_combine', 'hyvla_merger_fused'):
        registration = autograd.split(f'm.impl("{op}",', 1)[1].split(';', 1)[0]
        assert 'torch::CppFunction::makeFallthrough()' in registration

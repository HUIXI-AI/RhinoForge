"""Host regression controls for the private fused merger/prefix device seam."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.hy_vla import runtime as hy


def _runner():
    runner = object.__new__(hy.HyVlaRunner)
    cfg = NS(grid=7, proj_dim=2048, tok_bos=0, tok_user=1,
             tok_vision_start=2, tok_vision_split=3, tok_vision_end=4)
    tokens = torch.linspace(-2.01, 2.03, 64 * 2048).reshape(64, 2048)
    runner.w = NS(cfg=cfg, tok_emb=tokens)
    runner.prefix_len = 208
    runner._persist = frozenset()  # Children are doubles; no cold SPM barrier.
    runner._prefix_template = False
    runner._prefix_key = runner._prefix_val = None
    runner._device_prefix_key = runner._device_prefix_val = None
    return runner


@pytest.fixture
def cpu_device_seam(monkeypatch):
    uploads, assemblies = [], []
    original_to = torch.Tensor.to

    def upload(value):
        uploads.append(value.clone())
        return value.half().clone()

    def to(value, *args, **kwargs):
        if args and args[0] == "rpu":
            return value.clone()
        return original_to(value, *args, **kwargs)

    def assemble(skeleton, ids, parts, runs, prefix_is_embeddings=False):
        assert ids.dtype == torch.int32 and runs.dtype == torch.int64
        assert prefix_is_embeddings and skeleton.shape[0] == ids.numel()
        assert len(parts) in (14, 21) and all(part.is_contiguous() for part in parts)
        out = skeleton.clone()
        for part, (start, length) in zip(parts, runs.tolist()):
            assert part.shape == (length, 2048)
            out[start:start + length].copy_(part)
        assemblies.append((skeleton, ids, parts, runs, out))
        return out

    monkeypatch.setattr(hy, "_rpu16", upload)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.ops.rpu, "assemble_inputs_embeds_parts", assemble, raising=False)
    return uploads, assemblies


def test_device_prefix_matches_original_rows_and_retains_fresh_outputs(cpu_device_seam):
    uploads, assemblies = cpu_device_seam
    runner, lang = _runner(), torch.arange(5, 36)
    images = [torch.linspace(-1.003, 1.007, 49 * 2048).reshape(49, 2048).half() + i
              for i in range(3)]
    outputs = []
    for current_images, current_lang in (
        (images, lang), (list(reversed(images)), lang + 1), (images, lang),
    ):
        with torch.inference_mode():
            output = runner._assemble_device_prefix(current_images, current_lang)
        expected = runner.assemble_prefix(current_images, current_lang).half()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
        assert output.shape == (1, 208, 2048)
        assert not output[:, -1:].count_nonzero()  # One real pad row.
        outputs.append(output)
    assert torch.equal(outputs[0], outputs[2])
    assert not torch.equal(outputs[0], outputs[1])
    assert len({output.data_ptr() for output in outputs}) == 3
    assert len(uploads) == 3  # Only the A/B/A language skeleton changes.
    before = len(uploads)
    changed_images = [image + 0.125 for image in images]
    output = runner._assemble_device_prefix(changed_images, lang)
    assert len(uploads) == before
    assert torch.equal(output, runner.assemble_prefix(changed_images, lang).half())
    assert torch.equal(outputs[0], runner.assemble_prefix(images, lang).half())
    assert assemblies[0][2][0].untyped_storage().data_ptr() == images[0].untyped_storage().data_ptr()


@pytest.mark.parametrize("mutation", ["language", "version", "raw_alias", "owner", "special"])
@pytest.mark.parametrize("inference_owner", [False, True])
def test_device_prefix_invalidates_exact_static_owners_and_content(cpu_device_seam, mutation, inference_owner):
    uploads, _assemblies = cpu_device_seam
    runner, lang = _runner(), torch.arange(5, 37)
    images = [torch.full((49, 2048), float(i), dtype=torch.float16) for i in range(3)]
    if mutation == "raw_alias":
        runner.w.tok_emb[0, 0] = 0.0
    if inference_owner:
        # This is the actual public materialization context used by the board
        # entrypoint: lazy checkpoint FP32 tensors have no version counters.
        with torch.inference_mode():
            runner.w.tok_emb = runner.w.tok_emb.clone()
    first = runner._assemble_device_prefix(images, lang)
    first_snapshot = first.clone()
    first_owner = runner._device_prefix_val[2]
    if mutation == "language":
        lang[0] = 38
    elif mutation == "version":
        with torch.inference_mode():
            runner.w.tok_emb[0, 0] += 0.5
    elif mutation == "raw_alias":
        # A dirty CPU alias does not advance tok_emb._version. Signed-zero
        # bytes must still invalidate the published skeleton.
        version = None if inference_owner else runner.w.tok_emb._version
        with torch.inference_mode():
            runner.w.tok_emb.data[0, 0] = -0.0
        assert inference_owner or runner.w.tok_emb._version == version
    elif mutation == "owner":
        runner.w.tok_emb = runner.w.tok_emb.clone()
    else:
        runner.w.cfg.tok_bos = 39
    second = runner._assemble_device_prefix(images, lang)
    assert len(uploads) == 2
    assert runner._device_prefix_val[2] is not first_owner
    assert torch.equal(first.view(torch.int16), first_snapshot.view(torch.int16))
    expected = runner.assemble_prefix(images, lang).half()
    assert torch.equal(second.view(torch.int16), expected.view(torch.int16))
    runner._assemble_device_prefix(images, lang)
    assert len(uploads) == 2


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("return_device", [False, True])
def test_fused_vision_private_output_preserves_fresh_merger_without_cpu_readback(monkeypatch, return_device, cameras):
    sub = nullcontext()
    sub.handle, sub.gc = 7, NS(capture=lambda signature: nullcontext())
    cache = NS(k_caches=[None], v_caches=[None], reset_to_position=lambda position: None)
    owner = NS(w=NS(cfg=NS(vit_seq=4, vit_layers=1, vit_head_dim_padded=16, vit_heads=1)),
               _vit_cache=cache, _vit_packed=True, _pe_in_graph=True,
               _fused_merger=True, _merger_in_graph=True, _prefix_template=True,
               _vision_lanes=8, _prepare_vit=lambda count: sub, _vit_out_dim=lambda: 3,
               _plan_vision_execution=lambda *args: (NS(graph_key_words=lambda: ()), (), 16, {}),
               _component_generation=lambda component: 0,
               _record_execution_receipt=lambda **kwargs: None,
               _merger_sig=lambda count: "merger", _merger_rest=lambda values: values.clone())
    readbacks = []
    original_cpu = torch.Tensor.cpu

    def cpu(value, *args, **kwargs):
        readbacks.append(value)
        return original_cpu(value, *args, **kwargs)

    def vision(_handle, images, *_args):
        return torch.cat([image.reshape(1, 4, 3) for image in images]).reshape(1, cameras * 4, 3).clone()

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(hy, "_rpu16", lambda value: value.half())
    monkeypatch.setattr(hy.rpu_backend.graph, "GraphSignature", lambda **kwargs: None)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_forward_multi", vision, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_get_resolved_chunk_size", lambda handle: 16, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    images = [torch.arange(12).reshape(1, 3, 2, 2).float() + camera for camera in range(cameras)]
    first = hy.HyVlaRunner._run_vit(owner, images, return_device=return_device)
    saved = [image.clone() for image in first]
    second = hy.HyVlaRunner._run_vit(owner, [image + 1 for image in images], return_device=return_device)
    assert len(readbacks) == (0 if return_device else 2)
    assert all(torch.equal(image, expected) for image, expected in zip(first, saved))
    assert first[0].untyped_storage().data_ptr() != second[0].untyped_storage().data_ptr()


@pytest.mark.parametrize("fallback", [None, "length", "quantized", "nonfused", "host_merger", "template", "cameras", "inference_owner"])
def test_top_level_admits_only_private_measured_device_seam(monkeypatch, fallback):
    runner = _runner()
    runner._closed = False
    runner._vit_packed = runner._fused_merger = runner._merger_in_graph = runner._prefix_template = True
    runner.w.vit, runner.w.vlm_text, runner.w.expert = {"qws": []}, {"q_ws": []}, {"q_ws": []}
    runner.w.expert_nvfp4 = None
    runner._cache, runner._total = object(), 272
    images, lang = [object()] * 3, torch.arange(5, 37)
    if fallback == "length":
        runner.prefix_len = 240
    elif fallback == "quantized":
        runner.w.vit["qws"] = [object()]
    elif fallback == "nonfused":
        runner._fused_merger = False
    elif fallback == "host_merger":
        runner._merger_in_graph = False
    elif fallback == "template":
        runner._prefix_template = False
    elif fallback == "cameras":
        images = images[:2]
    elif fallback == "inference_owner":
        with torch.inference_mode():
            runner.w.tok_emb = runner.w.tok_emb.clone()
    calls = []
    runner._run_vit = lambda *args, **kwargs: calls.append(("vision", kwargs)) or images
    runner._assemble_device_prefix = lambda *args: calls.append(("device", {})) or object()
    runner.assemble_prefix = lambda *args: calls.append(("host", {})) or object()
    runner._run_vlm_prefill = lambda *args, **kwargs: calls.append(("prefill", kwargs))
    runner._run_denoise = lambda *args: args[-2].clone()
    noise = torch.randn(1, 50, 32)
    output = runner.get_action(images, lang, None, None, None, None, None,
                               noise, state_emb=torch.ones(1, 1, 1024))
    assert torch.equal(output, noise)
    assert calls == (
        [("vision", {"return_device": True}), ("device", {}),
         ("prefill", {"inputs_published": True})]
        if fallback in (None, "inference_owner") else
        [("vision", {}), ("host", {}), ("prefill", {})])


@pytest.mark.parametrize("boundary", [None, "T31", "T33", "hole", "mask_short",
                                     "mask_rank", "mask_device", "vision_w8",
                                     "P144", "P176", "one_camera"])
def test_two_camera_device_prefix_requires_exact_T32_right_padding(boundary):
    runner = _runner()
    runner.prefix_len = 160
    runner._closed = False
    runner._vit_packed = runner._fused_merger = runner._merger_in_graph = runner._prefix_template = True
    runner.w.vit, runner.w.vlm_text, runner.w.vlm_vision, runner.w.expert = (
        {"qws": []}, {"q_ws": []}, {"q_ws": []}, {"q_ws": []})
    runner.w.expert_nvfp4 = None
    runner._cache, runner._total = object(), 224
    images = [object()] * 2
    # Public language IDs always include 64 physical rows, independent of T.
    lang = torch.cat((torch.arange(5, 37), torch.zeros(32, dtype=torch.long)))
    mask = torch.zeros(192, 192, dtype=torch.bool)
    valid = 149 if boundary == "T31" else 151 if boundary == "T33" else 150
    mask[:valid, :valid] = True
    if boundary == "hole":
        mask[20] = False
        mask[150] = True  # Still 150 rows; an interior hole is not right padding.
    elif boundary == "mask_short":
        mask = mask[:150, :150]
    elif boundary == "mask_rank":
        mask = mask[:, 0]
    elif boundary == "mask_device":
        mask = torch.empty(192, 192, dtype=torch.bool, device="meta")
    elif boundary == "vision_w8":
        runner.w.vlm_vision["q_ws"] = [object()]
    elif boundary in ("P144", "P176"):
        runner.prefix_len = int(boundary[1:])
    elif boundary == "one_camera":
        images = images[:1]
    calls = []
    runner._run_vit = lambda *args, **kwargs: calls.append(("vision", kwargs)) or images
    runner._assemble_device_prefix = lambda *args: calls.append(("device", {})) or object()
    runner.assemble_prefix = lambda *args: calls.append(("host", {})) or object()
    runner._run_vlm_prefill = lambda *args, **kwargs: calls.append(("prefill", kwargs))
    runner._run_denoise = lambda *args: args[-2].clone()
    noise = torch.randn(1, 50, 32)
    result = runner.get_action(images, lang, mask, None, None, None, None,
                               noise, state_emb=torch.ones(1, 1, 1024))
    assert torch.equal(result, noise)
    assert calls == (
        [("vision", {"return_device": True}), ("device", {}),
         ("prefill", {"inputs_published": True})] if boundary is None else
        [("vision", {}), ("host", {}), ("prefill", {})])


@pytest.mark.parametrize("inference", [False, True])
def test_two_camera_actual_assembler_refreshes_raw_aliases_before_same_signature_prefill(
        cpu_device_seam, monkeypatch, inference):
    """Real Python caller/device program/prefill; CPU device and native-consumer doubles."""
    uploads, assemblies = cpu_device_seam

    class DeviceTensor(torch.Tensor):
        @property
        def device(self):
            return NS(type="rpu")

    active, signatures = [], []

    @contextmanager
    def capture(signature):
        signatures.append(signature)
        active.append(signature)
        try:
            yield
        finally:
            active.pop()

    sub = nullcontext()
    sub.handle, sub.gc = 7, NS(capture=capture)
    plan = NS(graph_key_words=lambda: (160,))
    prefilled, outputs, retained = [], [], []
    runner = _runner()
    runner.prefix_len = 160
    runner._closed = False
    runner._vit_packed = runner._fused_merger = runner._merger_in_graph = runner._prefix_template = True
    runner.w.vit, runner.w.vlm_text, runner.w.vlm_vision, runner.w.expert = (
        {"qws": []}, {"q_ws": []}, {"q_ws": []}, {"q_ws": []})
    runner.w.cfg.vlm_hidden, runner.w.cfg.layers, runner.w.attn_tp = 2048, 32, 8
    runner.w.expert_nvfp4 = None
    runner._native_cold = NS(vlm=NS(prefill_kv_only=False))
    runner._cache = NS(k_caches=[], v_caches=[], reset_to_position=lambda pos: None)
    runner._total = 224
    runner._prepare_vlm = lambda *args: (sub, (None, None, None))
    runner._plan_language_execution = lambda _sub: (plan, (1,), 160, {})
    runner._component_generation = lambda component: 0
    runner._record_execution_receipt = lambda **kwargs: None
    upload = hy._rpu16
    monkeypatch.setattr(hy, "_rpu16", lambda value: value if isinstance(value, DeviceTensor)
                        else upload(value).as_subclass(DeviceTensor))
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_args, **_kwargs:
                        pytest.fail("private merger/prefix unexpectedly read back to CPU"))
    monkeypatch.setattr(hy.rpu_backend.graph, "GraphSignature", lambda **fields: NS(**fields))
    monkeypatch.setattr(torch.ops.rpu, "hyvla_vlm_get_resolved_chunk_size", lambda handle: 160, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)

    def vision(images, *, return_device=False):
        assert return_device and not active
        # The unchanged fused merger boundary returns fresh [49,2048] owners.
        return [image.clone().as_subclass(DeviceTensor) for image in images]

    runner._run_vit = vision
    assemble = runner._assemble_device_prefix

    def device_prefix(*args):
        assert not active  # Immediate device program must precede VLM capture.
        value = assemble(*args).as_subclass(DeviceTensor)
        assert value.shape == (1, 160, 2048)
        return value

    runner._assemble_device_prefix = device_prefix

    def prefill(*args):
        assert active and args[-1] is True and args[1].device.type == "rpu"
        prefilled.append(args[1])

    monkeypatch.setattr(torch.ops.rpu, "hyvla_vlm_step_forward", prefill, raising=False)

    def denoise(*args):
        assert not active
        source = prefilled[-1].as_subclass(torch.Tensor)
        # CPU consumer double includes special, both camera and language rows.
        return source[0, [0, 3, 61, 118], :8].reshape(1, 1, 32).float().repeat(1, 50, 1)

    runner._run_denoise = denoise
    with torch.inference_mode(inference):
        runner.w.tok_emb = runner.w.tok_emb.clone()
        lang = torch.cat((torch.arange(5, 37), torch.zeros(32, dtype=torch.long)))
        images = [torch.full((49, 2048), float(camera), dtype=torch.float16) for camera in range(2)]
        lang_alias, token_alias = lang.numpy(), runner.w.tok_emb.numpy()
        image_alias = images[0].numpy()
        original_bos = float(token_alias[0, 0])
        versions = [None if torch.is_inference(t) else t._version
                    for t in (lang, runner.w.tok_emb, images[0])]
        mask = torch.zeros(192, 192, dtype=torch.bool)
        mask[:150, :150] = True
        for changed in (False, False, True, True, False, False):
            lang_alias[0] = 6 if changed else 5
            token_alias[0, 0] = original_bos + (0.5 if changed else 0)
            image_alias[:] = 0.125 if changed else 0
            assert [None if torch.is_inference(t) else t._version
                    for t in (lang, runner.w.tok_emb, images[0])] == versions
            result = runner.get_action(images, lang, mask, None, None, None, None,
                                       torch.zeros(1, 50, 32), state_emb=torch.ones(1, 1, 1024))
            # Rebuild the old host composition from current bytes, then apply its
            # existing P160 crop; its cached template is outside this oracle.
            runner._prefix_template = False
            try:
                expected = runner.assemble_prefix(images, lang)[:, :160].half()
            finally:
                runner._prefix_template = True
            assert torch.equal(prefilled[-1].view(torch.int16), expected.view(torch.int16))
            assert result.device.type == "cpu" and result.shape == (1, 50, 32)
            assert result.dtype == torch.float32 and torch.isfinite(result).all()
            outputs.append(result)
            retained.append((prefilled[-1], prefilled[-1].clone(), result, result.clone()))
            assert all(torch.equal(prefix, saved_prefix) and torch.equal(action, saved_action)
                       for prefix, saved_prefix, action, saved_action in retained)
    assert len(uploads) == 3  # Exact A/B/A static skeleton refresh, never per-image upload.
    assert len(assemblies) == 6 and all(len(entry[2]) == 14 for entry in assemblies)
    assert len({entry[-1].data_ptr() for entry in assemblies}) == 6
    assert len({value.data_ptr() for value in outputs}) == 6
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], outputs[4])
    assert not torch.equal(outputs[0], outputs[2]) and torch.equal(outputs[2], outputs[3])
    assert len(signatures) == 6 and all(sig == signatures[0] for sig in signatures)
    assert signatures[0].op_id == "hyvla_vlm_prefill" and signatures[0].shapes == [160, 2048]
    assert signatures[0].dyn_dims == [32, 0, 8, 0, 160] and not active


@pytest.mark.parametrize("device_input,published,expected", [(False, False, True), (True, False, False), (True, True, True)])
def test_prefill_keeps_public_dirty_device_inputs_conservative(monkeypatch, device_input, published, expected):
    sub = nullcontext()
    sub.handle, sub.gc = 7, NS(capture=lambda signature: nullcontext())
    plan = NS(graph_key_words=lambda: ())
    owner = NS(w=NS(cfg=NS(vlm_hidden=2048, layers=32), attn_tp=8), prefix_len=208,
               _native_cold=NS(vlm=NS(prefill_kv_only=False)),
               _prepare_vlm=lambda *args: (sub, (None, None, None)),
               _plan_language_execution=lambda sub: (plan, (1,), 208, {}),
               _component_generation=lambda component: 0,
               _record_execution_receipt=lambda **kwargs: None)
    tensor = torch.zeros(1, 208, 2048, dtype=torch.float16)

    class DeviceInput:
        device = NS(type="rpu")

        def __getitem__(self, index):
            return tensor[index]

    prefix = DeviceInput() if device_input else tensor
    calls = []
    cache = NS(k_caches=[], v_caches=[], reset_to_position=lambda position: None)
    monkeypatch.setattr(hy, "_rpu16", lambda value: value)
    monkeypatch.setattr(hy.rpu_backend.graph, "GraphSignature", lambda **kwargs: None)
    monkeypatch.setattr(torch.ops.rpu, "hyvla_vlm_step_forward", lambda *args: calls.append(args[-1]), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvla_vlm_get_resolved_chunk_size", lambda handle: 208, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    hy.HyVlaRunner._run_vlm_prefill(owner, cache, prefix, None, None, None,
                                   inputs_published=published)
    assert calls == [expected]


def _quantized_runner(cameras, precision):
    runner = _runner()
    runner.prefix_len = 160 if cameras == 2 else 208
    runner._closed = False
    runner._vit_packed = runner._fused_merger = runner._merger_in_graph = runner._prefix_template = True
    runner._pe_in_graph = True
    runner._proj1_in_merger = False
    runner.w.cfg.num_steps = 10
    runner.w.cfg.vlm_hidden, runner.w.cfg.layers, runner.w.attn_tp = 2048, 32, 8
    runner.w.cfg.vit_seq, runner.w.cfg.vit_layers = 196, 1
    runner.w.cfg.vit_head_dim_padded, runner.w.cfg.vit_heads = 80, 16
    dtype = torch.float16 if precision == "fp16" else torch.int8
    scale = [] if precision == "fp16" else [torch.ones(1, dtype=torch.float16)]
    runner.w.vit = {"qw": [torch.ones(1, dtype=dtype)], "qws": scale}
    runner.w.vlm_text = {"q_w": [torch.ones(1, dtype=dtype)], "q_ws": scale}
    runner.w.vlm_vision = {"q_w": [torch.ones(1, dtype=dtype)], "q_ws": scale}
    nv = precision == "w8a16_nvfp4"
    runner.w.expert = {"q_w": [torch.ones(1, dtype=torch.uint8 if nv else dtype)], "q_ws": scale}
    runner.w.expert_nvfp4 = NS(enabled=True) if nv else None
    runner.w.merger_rest = {name: torch.ones(1, dtype=torch.float16) for name in (
        "pooler.predictor.0.weight_a", "pooler.predictor.0.bias", "pooler.predictor.0.weight_b",
        "pooler.predictor.2.weight", "pooler.predictor.2.bias", "proj2.weight", "proj2.bias")}
    runner._cache = NS(k_caches=[], v_caches=[], reset_to_position=lambda _: None)
    runner._vit_cache = NS(k_caches=[None], v_caches=[None], reset_to_position=lambda _: None)
    runner._native_cold = NS(vlm=NS(prefill_kv_only=False))
    runner._total = runner.prefix_len + 64
    lang = torch.cat((torch.arange(5, 37), torch.zeros(32, dtype=torch.long)))
    mask = torch.zeros(runner.prefix_len + 32, runner.prefix_len + 32, dtype=torch.bool)
    valid = 2 + cameras * 58 + 32
    mask[:valid, :valid] = True
    return runner, lang, mask


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("precision", ["fp16", "w8a16", "w8a16_nvfp4"])
@pytest.mark.parametrize("inference", [False, True])
def test_quantized_prefix_actual_producer_assembler_consumer_raw_alias_aba(
        cpu_device_seam, monkeypatch, cameras, precision, inference):
    """Real Python four-stage seam, real CPU tensor owners; native/Graph are doubles."""
    import gc
    import weakref

    runner, lang, mask = _quantized_runner(cameras, precision)
    uploads, assemblies = cpu_device_seam
    active, signatures, produced, consumed, retained = [], [], [], [], []
    original_to = torch.Tensor.to

    class DeviceTensor(torch.Tensor):
        @property
        def device(self):
            return NS(type="rpu")

    def to(value, *args, **kwargs):
        if (args and args[0] == "rpu") or kwargs.get("device") == "rpu":
            return value.clone().as_subclass(DeviceTensor)
        return original_to(value, *args, **kwargs)

    @contextmanager
    def capture(sig):
        assert not active
        signatures.append(sig)
        active.append(sig)
        try:
            yield
        finally:
            # The real native consumer can consume inputs at capture exit.
            if sig.op_id == "hyvla_vlm_prefill":
                ref, expected = consumed[-1]
                gc.collect()
                assert ref() is not None and torch.equal(ref().as_subclass(torch.Tensor), expected)
            active.pop()

    sub = nullcontext()
    sub.handle, sub.gc = 7, NS(capture=capture)
    runner._prepare_vit = lambda _: sub
    runner._prepare_vlm = lambda *_: (sub, (None, None, None))
    runner._plan_vision_execution = lambda *_: (NS(graph_key_words=lambda: ()), (), 196, {})
    runner._plan_language_execution = lambda *_: (NS(graph_key_words=lambda: (runner.prefix_len,)), (), runner.prefix_len, {})
    runner._component_generation = lambda _: 0
    runner._record_execution_receipt = lambda **_: None
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(hy.rpu_backend.graph, "GraphSignature", lambda **kw: NS(**kw))
    upload = hy._rpu16
    monkeypatch.setattr(hy, "_rpu16", lambda t: t if isinstance(t, DeviceTensor) else upload(t).as_subclass(DeviceTensor))
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_a, **_k: pytest.fail("unexpected merger/prefix CPU readback"))
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_get_resolved_chunk_size", lambda _: 196, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvla_vlm_get_resolved_chunk_size", lambda _: runner.prefix_len, raising=False)

    def vit(_handle, images, *_args):
        assert active[-1].op_id == "hyvit2_vision_compute"
        assert all(im.dtype == torch.float16 and im.is_contiguous() for im in images)
        # Native producer double: per-camera signal with real fresh FP16 storage.
        values = [im.as_subclass(torch.Tensor).flatten()[:1].expand(196, 2048) for im in images]
        return torch.cat(values).reshape(1, cameras * 196, 2048).clone().as_subclass(DeviceTensor)

    def merger(value, *_weights):
        assert active[-1].op_id == "hyvla_merger_fused"
        assert value.dtype == torch.float16 and value.shape == (4, cameras * 49, 2048)
        result = value.reshape(cameras, 196, 2048)[:, :49].contiguous().clone()
        produced.append(weakref.ref(result))
        return result

    def prefill(*args):
        assert active[-1].op_id == "hyvla_vlm_prefill" and args[-1] is True
        prefix = args[1]
        assert prefix.device.type == "rpu" and prefix.dtype == torch.float16 and prefix.is_contiguous()
        consumed.append((weakref.ref(prefix), prefix.as_subclass(torch.Tensor).clone()))

    monkeypatch.setattr(torch.ops.rpu, "hyvit2_forward_multi", vit, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvla_merger_fused", merger, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvla_vlm_step_forward", prefill, raising=False)
    runner._run_denoise = lambda *_: consumed[-1][1][0, [0, 3, 61, 118], :8].reshape(1, 1, 32).float().repeat(1, 50, 1)
    assemble = runner._assemble_device_prefix

    def assemble_after_merger(*args):
        assert not active and len(args[0]) == cameras
        assert all(t.dtype == torch.float16 and t.is_contiguous() for t in args[0])
        return assemble(*args)

    runner._assemble_device_prefix = assemble_after_merger
    with torch.inference_mode(inference):
        runner.w.tok_emb = runner.w.tok_emb.clone()
        lang = lang.clone()
        images = [torch.full((1, 3, 224, 224), float(i), dtype=torch.float32) for i in range(cameras)]
        aliases = [lang.numpy(), runner.w.tok_emb.numpy(), images[0].numpy()]
        owners = [lang, runner.w.tok_emb, images[0]]
        versions = [None if torch.is_inference(t) else t._version for t in owners]
        bos = float(aliases[1][0, 0])
        for changed in (False, False, True, True, False, False):
            aliases[0][0] = 6 if changed else 5
            aliases[1][0, 0] = bos + (0.5 if changed else 0)
            aliases[2][...] = 0.125 if changed else 0
            assert [None if torch.is_inference(t) else t._version for t in owners] == versions
            action = runner.get_action(images, lang, mask, None, None, None, None,
                                       torch.zeros(1, 50, 32), state_emb=torch.ones(1, 1, 1024))
            feature_control = [im.flatten()[0].half().expand(49, 2048) for im in images]
            runner._prefix_template = False
            try:
                expected = runner.assemble_prefix(feature_control, lang)[:, :runner.prefix_len].half()
            finally:
                runner._prefix_template = True
            assert torch.equal(consumed[-1][1].view(torch.int16), expected.view(torch.int16))
            assert action.device.type == "cpu" and action.shape == (1, 50, 32) and torch.isfinite(action).all()
            retained.append((action, action.clone(), consumed[-1][1], expected.clone()))
            assert all(torch.equal(a, b) and torch.equal(c, d) for a, b, c, d in retained)
    assert len(assemblies) == 6 and all(len(a[2]) == cameras * 7 for a in assemblies)
    assert len({a[-1].data_ptr() for a in assemblies}) == 6
    assert [tuple(t.shape) for t in uploads].count((cameras, 3, 224, 224)) == 6
    assert [tuple(t.shape) for t in uploads].count((runner.prefix_len, 2048)) == 3
    assert len(uploads) == 9  # Six required pixel uploads plus skeleton A/B/A only.
    assert torch.equal(retained[0][0], retained[4][0]) and not torch.equal(retained[0][0], retained[2][0])
    prefill_sigs = [s for s in signatures if s.op_id == "hyvla_vlm_prefill"]
    assert len(prefill_sigs) == 6 and all(s == prefill_sigs[0] for s in prefill_sigs)
    assert len(signatures) == 18 and not active


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("precision", ["w8a16", "w8a16_nvfp4"])
@pytest.mark.parametrize("boundary", [None, "T31", "T33", "hole", "D9", "w4",
    "mixed", "host_merger", "nonpacked", "template", "grid", "width", "P240", "mask_device", "tokens_device"])
def test_quantized_device_prefix_keeps_other_routes_on_host(monkeypatch, cameras, precision, boundary):
    runner, lang, mask = _quantized_runner(cameras, precision)
    valid = 2 + cameras * 58 + 32
    if boundary in ("T31", "T33"):
        mask.zero_(); n = valid + (1 if boundary == "T33" else -1); mask[:n, :n] = True
    elif boundary == "hole":
        mask[20] = False; mask[valid] = True
    elif boundary == "D9": runner.w.cfg.num_steps = 9
    elif boundary == "w4": runner.w.vlm_text["q_w"] = [torch.ones(1, dtype=torch.uint8)]
    elif boundary == "mixed": runner.w.vlm_text["q_w"] = [torch.ones(1, dtype=torch.float16)]
    elif boundary == "host_merger": runner._merger_in_graph = False
    elif boundary == "nonpacked": runner._vit_packed = False
    elif boundary == "template": runner._prefix_template = False
    elif boundary == "grid": runner.w.cfg.grid = 8
    elif boundary == "width": runner.w.cfg.proj_dim = 1024
    elif boundary == "P240": runner.prefix_len = 240
    elif boundary == "mask_device": mask = torch.empty_like(mask, device="meta")
    elif boundary == "tokens_device": lang = torch.empty_like(lang, device="meta")
    calls = []
    runner._run_vit = lambda *a, **kw: calls.append(("vision", kw)) or a[0]
    runner._assemble_device_prefix = lambda *a: calls.append(("device", {})) or object()
    runner.assemble_prefix = lambda *a: calls.append(("host", {})) or object()
    runner._run_vlm_prefill = lambda *a, **kw: calls.append(("prefill", kw))
    runner._run_denoise = lambda *a: a[-2].clone()
    runner.get_action([object()] * cameras, lang, mask, None, None, None, None,
                      torch.zeros(1, 50, 32), state_emb=torch.ones(1, 1, 1024))
    assert calls == ([("vision", {"return_device": True}), ("device", {}), ("prefill", {"inputs_published": True})]
                     if boundary is None else [("vision", {}), ("host", {}), ("prefill", {})])

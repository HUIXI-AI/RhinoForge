"""Internal staged-load component fixture, not a Qwen3-VL-32B model loader.

The hardware leg uses exact projection geometries admitted by the existing
production config reader, but materializes only one synthetic projection at a
time. It invokes the real metadata reader, materializer, swizzler and transfer.
No 32B model/profile/support or FMB/Graph certificate is produced here.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import torch
from torch import nn
from safetensors.torch import save_file

from rpu_backend.quant import load as loader
from rpu_backend.tests.library.test_w8a16_loader_meta import _qwen3_vl_32b_w8_config


def projection_shapes(config):
    if not loader.is_qwen3_vl_32b_w8a16_config(config):
        raise ValueError("handoff fixture requires the production exact W8 config")
    text = config.text_config
    hidden, inter = int(text.hidden_size), int(text.intermediate_size)
    q = int(text.num_attention_heads) * int(text.head_dim)
    kv = int(text.num_key_value_heads) * int(text.head_dim)
    return {
        "self_attn.q_proj": (q, hidden, 1),
        "self_attn.k_proj": (kv, hidden, 1),
        "self_attn.v_proj": (kv, hidden, 1),
        "self_attn.o_proj": (hidden, q, 0),
        "mlp.gate_proj": (inter, hidden, 1),
        "mlp.up_proj": (inter, hidden, 1),
        "mlp.down_proj": (hidden, inter, 0),
    }


def metadata_fixture(directory: Path, projection: str, weight, scale):
    """Use the real reader on a one-projection meta-only component skeleton.

    The small host tests exercise bookkeeping on small tensors; the hardware
    caller below obtains its unmodified dimensions from projection_shapes.
    """
    branch, name = projection.split(".")
    n, k = weight.shape
    with torch.device("meta"):
        model = nn.Module()
        model.model = nn.Module()
        model.model.language_model = nn.Module()
        layer = nn.Module()
        setattr(layer, branch, nn.Module())
        getattr(layer, branch).add_module(name, nn.Linear(k, n, bias=False))
        model.model.language_model.layers = nn.ModuleList([layer])
        # A real non-decoder parameter proves the materializer's first phase.
        model.model.visual = nn.Module()
        model.model.visual.register_parameter(
            "sentinel", nn.Parameter(torch.empty(2), requires_grad=False))
    prefix = "model.language_model.layers.0." + projection
    shard = directory / "projection.safetensors"
    if shard.exists():
        raise FileExistsError(shard)
    save_file({prefix + ".weight": weight, prefix + ".weight_scale": scale,
               "model.visual.sentinel": torch.tensor([0.125, -0.25], device="cpu")}, str(shard))
    plan = loader._build_w8a16_imagetext_plan(model, directory, [shard])
    vars(model)[loader._W8A16_IMAGETEXT_STAGE_ATTR] = plan
    assert loader._validate_staged_w8a16_imagetext(model) is plan
    return model, plan, layer.get_submodule(projection)


def run_projection_handoff(directory: Path, projection: str):
    """Real device handoff and PASSTHROUGH numerical gate; hardware caller only."""
    from rpu_backend.api.causal_lm import _claim_live_instance, _release_live_instance
    from rpu_backend.runtime.weights import transform_linear_weight
    from rpu_backend.tests.conformance._numeric import require_same_semantic
    from rpu_backend.adapters.qwen3_vl import Qwen3VLAdapter

    config = _qwen3_vl_32b_w8_config()
    shapes = projection_shapes(config)
    n, k, partition = shapes[projection]
    generator = torch.Generator(device="cpu").manual_seed(
        0x51A6 + tuple(shapes).index(projection))
    weight = torch.randint(-127, 128, (n, k), generator=generator, dtype=torch.int8, device="cpu")
    scale = (torch.rand(n, generator=generator, device="cpu") * 0.002 + 0.001).half()
    model, plan, linear = metadata_fixture(directory, projection, weight, scale)
    assert all(parameter.is_meta for parameter in model.parameters())
    assert plan.model_id == id(model)
    # Exercise the same adapter identity guard, not a copied test condition or
    # a partially constructed whole-model to_rpu invocation.
    adapter = object.__new__(Qwen3VLAdapter)
    adapter.model = model
    adapter._rpu_w8a16_staged_plan = plan
    validated = adapter._validate_staged_checkpoint_plan(adapter._rpu_w8a16_staged_plan)
    assert validated is plan
    observation = {
        "projection": projection, "n": n, "k": k, "partition": partition, "num_cores": 8,
        "model_ids": [id(model), plan.model_id, id(adapter.model)],
        "plan_ids": [id(plan), id(vars(model)[loader._W8A16_IMAGETEXT_STAGE_ATTR]),
                     id(adapter._rpu_w8a16_staged_plan), id(validated)],
        "meta_counts": [sum(parameter.is_meta for parameter in model.parameters())],
    }

    _claim_live_instance(model)
    try:
        # Same irreversible boundary as Qwen3VLAdapter.to_rpu, after real claim.
        vars(model)["_rpu_swizzle_started"] = True
        loader._materialize_staged_w8a16_imagetext_for_rpu(model)
        assert loader._W8A16_IMAGETEXT_STAGE_ATTR not in vars(model)
        assert not any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers()))
        assert torch.equal(model.model.visual.sentinel,
                           torch.tensor([0.125, -0.25], device="cpu").half())
        assert linear.weight.device.type == linear.weight_scale.device.type == "rpu"
        assert linear.weight.dtype == torch.int8 and linear.weight_scale.dtype == torch.float16
        assert linear._rpu_linear_partition == partition
        assert linear._rpu_linear_num_cores == 8
        # Transform the untouched CPU reference, never the already-swizzled instance.
        expected = transform_linear_weight(weight, partition, num_cores=8)
        transferred_weight, transferred_scale = linear.weight.cpu(), linear.weight_scale.cpu()
        assert torch.equal(transferred_weight, expected)
        assert torch.equal(transferred_scale, scale)
        observation.update({
            "meta_counts": observation["meta_counts"] + [
                sum(tensor.is_meta for tensor in (*model.parameters(), *model.buffers()))],
            "stage_plan_after": vars(model).get(loader._W8A16_IMAGETEXT_STAGE_ATTR),
            "weight_device": linear.weight.device.type, "weight_dtype": str(linear.weight.dtype),
            "scale_device": linear.weight_scale.device.type, "scale_dtype": str(linear.weight_scale.dtype),
            "weight_sha256": _tensor_sha256(transferred_weight),
            "expected_weight_sha256": _tensor_sha256(expected),
            "scale_sha256": _tensor_sha256(transferred_scale),
            "expected_scale_sha256": _tensor_sha256(scale),
        })
        del expected, transferred_weight, transferred_scale

        metrics = []
        for m in (1, 16):
            x = (torch.rand((m, k), generator=generator, device="cpu") * 0.2 - 0.1).half()
            reference = x.float() @ (weight.float() * scale.float().unsqueeze(1)).t()
            first = None
            for ordinal, sign in enumerate((1, -1, 1)):
                actual_input = x * sign
                actual = torch.ops.rpu.test_w8a16_projection(
                    actual_input.to("rpu"), linear.weight, linear.weight_scale, partition, 8)
                assert actual.device.type == "cpu" and tuple(actual.shape) == (m, n)
                # Preserve existing W8 projection budgets, not full-model gates.
                row = require_same_semantic(
                    actual, reference * sign, cosine_min=0.999, relative_l2_max=0.002,
                    row_relative_l2_max=0.002,
                    label=f"staged-handoff {projection} M={m} invocation={ordinal}")
                if ordinal == 0:
                    first = actual.clone()
                elif ordinal == 2:
                    assert torch.equal(first, actual), "same transferred weight did not reproduce A"
                metrics.append({"m": m, "ordinal": ordinal, "input_sign": sign,
                                "input_sha256": _tensor_sha256(actual_input),
                                "output_sha256": _tensor_sha256(actual),
                                "reference_sha256": _tensor_sha256(reference * sign),
                                # Successful return passed both embedded native
                                # assertions. This is not a Python state getter.
                                "native_assertions": "test_w8a16_projection.entry_and_output_passthrough",
                                **row})
        return {**observation, "invocations": metrics}
    finally:
        _release_live_instance(model)


def _tensor_sha256(tensor):
    assert tensor.device.type == "cpu", "evidence hashes actual synchronized CPU bytes"
    return hashlib.sha256(memoryview(tensor.detach().contiguous().numpy())).hexdigest()

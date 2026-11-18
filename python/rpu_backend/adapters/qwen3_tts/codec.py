"""Convolution paths for the frozen 12 Hz speech decoder."""
import types

import torch


def _transpose_convolution(owner, hidden, output_size=None):
    if output_size is not None:
        raise ValueError("TTS codec uses the configured transposed-convolution output size")
    batch, _, count = hidden.shape
    stride, kernel = owner.stride[0], owner.kernel_size[0]
    projected = torch.matmul(hidden.transpose(1, 2), owner.weight.flatten(1))
    if kernel == stride:
        result = projected.view(batch, count, owner.out_channels, stride)
        result = result.transpose(1, 2).reshape(batch, owner.out_channels, count * stride)
    else:
        # The frozen decoder overlaps only two adjacent stride-sized blocks.
        chunks = projected.view(batch, count, owner.out_channels, 2, stride)
        result = hidden.new_empty((batch, owner.out_channels, count + 1, stride))
        result[:, :, 0] = chunks[:, 0, :, 0]
        result[:, :, -1] = chunks[:, -1, :, 1]
        torch.add(chunks[:, 1:, :, 0].transpose(1, 2),
                  chunks[:, :-1, :, 1].transpose(1, 2), out=result[:, :, 1:-1])
        result = result.flatten(2)
    if owner.bias is not None:
        result = result + owner.bias[None, :, None]
    return result


def _pointwise_convolution(owner, hidden):
    result = torch.matmul(owner.weight.squeeze(-1), hidden)
    if owner.bias is not None:
        result = result + owner.bias[None, :, None]
    return result


def install_codec_convolutions(decoder, restore):
    modules = [m for m in decoder.modules() if isinstance(m, torch.nn.ConvTranspose1d) or
               (isinstance(m, torch.nn.Conv1d) and m.kernel_size == (1,))]
    for module in modules:
        if module.weight.device.type != "cpu" or module.weight.dtype != torch.float32:
            raise ValueError("TTS codec convolutions require CPU FP32 weights")
        if module.padding_mode != "zeros":
            raise ValueError("TTS codec convolutions require zero padding")
        if module.groups != 1 or module.dilation != (1,) or module.padding != (0,):
            raise ValueError("TTS codec convolution geometry differs")
        if isinstance(module, torch.nn.ConvTranspose1d):
            if (module.output_padding != (0,) or
                    module.kernel_size[0] not in (module.stride[0], 2 * module.stride[0])):
                raise ValueError("TTS codec transposed-convolution geometry differs")
        elif module.stride != (1,):
            raise ValueError("TTS codec pointwise-convolution geometry differs")
    for module in modules:
        restore.append((module, module.forward))
        forward = _transpose_convolution if isinstance(module, torch.nn.ConvTranspose1d) else _pointwise_convolution
        module.forward = types.MethodType(forward, module)


def _check_residual_input(owner, hidden):
    if (hidden.device.type != "cpu" or hidden.dtype != torch.float32 or
            hidden.ndim != 3 or hidden.shape[:2] != (1, owner.in_channels)):
        raise ValueError("TTS RPU residual convolution requires batch-one CPU FP32 input")
    count = hidden.shape[-1] - 6 * owner.dilation[0]
    if count <= 0:
        raise ValueError("TTS RPU residual convolution input is shorter than its kernel")
    return count


def _residual_convolution(owner, hidden, weight, stats):
    """Host im2col, kept for a build whose .so predates the device operator."""
    dilation = owner.dilation[0]
    count = _check_residual_input(owner, hidden)
    # The patch matrix is 7x the activation, which costs 48 ms/request to
    # materialise and 73 ms to upload across the twelve layers. _residual_slab
    # below avoids both and is the route every layer takes when the operator
    # exists; this one survives only so a stale .so still computes the right
    # answer.
    patches = hidden.unfold(-1, 6 * dilation + 1, 1)[..., ::dilation]
    patches = patches.permute(0, 2, 1, 3).reshape(count, -1).half().contiguous().to("rpu")
    result = torch.ops.rpu.linear_with_accumulation(
        patches, weight, None, 1, stats.get("num_cores", 8), True).cpu().float()[:, :owner.out_channels]
    if owner.bias is not None:
        result = result + owner.bias
    stats["calls"] += 1
    return result.T.unsqueeze(0).contiguous()


def _residual_slab(owner, hidden, weight, stats):
    """Upload the compact [L, C] activation and let the operator do the im2col.

    The patch matrix never exists: the GEMM kernel's per-row input pitch is set
    to one channel row, so tap t of window m reads slab row m+t. `weight` must
    therefore be TAP-MAJOR (k = t*C + c) — `install_rpu_residual_convolutions`
    binds this forward only to the weights it prepared that way.

    Dilation is handled here rather than in the operator, because each residue
    class modulo the dilation is already a dilation-1 problem: window m = j + i*d
    reads rows m + t*d = j + (i + t)*d, which are CONSECUTIVE rows of the compact
    sub-slab `compact[j::d]`. Concatenating the d classes therefore gives a single
    slab that one undilated call consumes end to end: the six windows straddling
    each class boundary are garbage, but the row arithmetic never reads them,
    because class j owns `windows_j` of its `rows` rows and windows_j + 6 <= rows.
    So every layer is ONE call over the activation exactly once, never a 7x patch
    matrix and never one call per dilation.
    """
    count = _check_residual_input(owner, hidden)
    dilation = owner.dilation[0]
    channels = owner.out_channels
    # This transpose is the heaviest host step left, at 67 ms/request. Splitting
    # the cast from the transpose, fusing them, and transposing in fp32 first
    # were all measured: the first three are within 7% of each other and only
    # fp32-first is clearly worse, so the cost is inherent to turning [C, L] into
    # [L, C] and is not worth another rewrite. Removing it means carrying [L, C]
    # through the whole residual stack, which is a larger change than this one.
    compact = hidden[0].half().t().contiguous()
    rows = -(-compact.shape[0] // dilation)
    if dilation == 1:
        slab = compact
    else:
        # Zero-filled: the classes shorter than `rows` leave a tail row that no
        # window of theirs reads, but it is still multiplied by the weight.
        slab = compact.new_zeros((rows * dilation, channels))
        for offset in range(dilation):
            part = compact[offset::dilation]
            # Whole contiguous C-half rows at stride d*C*2, so this gather has a
            # contiguous inner dim where the per-class transpose did not.
            slab[offset * rows:offset * rows + part.shape[0]] = part
    raw = torch.ops.rpu.conv1d_slab_with_accumulation(
        slab.to("rpu"), weight, None, 7, 1,
        stats.get("num_cores", 8)).cpu().float()[:, :channels]
    if dilation == 1:
        result = raw
    else:
        result = hidden.new_empty((count, channels))
        for offset in range(dilation):
            windows = -(-(count - offset) // dilation)
            if windows <= 0:
                continue    # more residue classes than output rows
            result[offset::dilation] = raw[offset * rows:offset * rows + windows]
    if owner.bias is not None:
        result = result + owner.bias
    stats["calls"] += 1
    stats["slab_calls"] += 1
    return result.T.unsqueeze(0).contiguous()


def install_rpu_residual_convolutions(decoder, restore, stats, *, num_cores=8):
    """FP16/ACC32 Linear for the twelve frozen residual Conv1d layers."""
    from collections import Counter
    from rpu_backend.runtime.weights import transform_linear_weight
    from rpu_backend.runtime.topology import execution_core_count

    num_cores = execution_core_count({"model": {"num_cores": num_cores}})
    modules = [m for m in decoder.modules()
               if isinstance(m, torch.nn.Conv1d) and m.kernel_size == (7,) and
               m.in_channels == m.out_channels and m.in_channels in (768, 384, 192, 96)]
    expected = Counter((channels, dilation) for channels in (768, 384, 192, 96)
                       for dilation in (1, 3, 9))
    if Counter((m.in_channels, m.dilation[0]) for m in modules) != expected:
        raise ValueError("TTS RPU residual convolution geometry differs from the twelve-layer profile")
    for module in modules:
        if (module.out_channels != module.in_channels or module.groups != 1 or
                module.stride != (1,) or module.padding != (0,) or module.padding_mode != "zeros"):
            raise ValueError("TTS RPU residual convolution geometry differs")
        if (module.weight.device.type != "cpu" or module.weight.dtype != torch.float32 or
                (module.bias is not None and
                 (module.bias.device.type != "cpu" or module.bias.dtype != torch.float32))):
            raise ValueError("TTS RPU residual convolution requires CPU FP32 source weights")
    # The compact route needs the operator AND a tap-major weight, so the two
    # decisions are made together per layer: a build whose .so predates the
    # operator prepares every weight channel-major and keeps the host im2col,
    # which is slower but never reinterprets a weight it cannot consume.
    slab_op = getattr(torch.ops.rpu, "conv1d_slab_with_accumulation", None)
    prepared = []
    for module in modules:
        # Every dilation qualifies: _residual_slab concatenates the residue
        # classes into one slab, so the operator itself only ever sees tap
        # step 1 and a dilated layer still costs exactly one call.
        use_slab = slab_op is not None
        source = module.weight.detach()
        # tap-major (t, c) for the slab, channel-major (c, t) for the patch matrix
        weight = (source.permute(0, 2, 1) if use_slab else source).flatten(1).half()
        weight = torch.nn.functional.pad(weight, (0, 0, 0, (-module.out_channels) % (16 * num_cores)))
        prepared.append((use_slab, transform_linear_weight(weight, 1, num_cores).to("rpu")))
    for module, (use_slab, weight) in zip(modules, prepared):
        route = _residual_slab if use_slab else _residual_convolution

        def forward(owner, hidden, weight=weight, route=route):
            return route(owner, hidden, weight, stats)
        restore.append((module, module.forward))
        module.forward = types.MethodType(forward, module)
    stats["layers"] = len(modules)
    stats["slab_layers"] = sum(use_slab for use_slab, _ in prepared)
    stats["slab_calls"] = 0
    stats["num_cores"] = num_cores

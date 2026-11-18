"""Explicit fused Q+KV1 current-ABI BUILD/REPLAY conformance.

The native wrapper includes each core's scale preload. These are correctness
fixtures; they neither select a model route nor establish production speedup.
"""
from __future__ import annotations

import torch

CASES = ["pi_q_kv1_m1_l0", "pi_q_kv1_m50_l3", "pi_q_kv1_m64_l17"]
CONSUMER = "pi_q_kv1_m50_l3_production_cache"
CASES.append(CONSUMER)


def production_cache_expected(initial, compact, cos, sin, *, key):
    """Original PAD64 cache layout, independent dense CPU RoPE construction."""
    rows = compact.shape[1]
    if rows != 50:
        raise ValueError("production cache oracle requires M50")
    # Each core owns the two 16-lane slices of one rotate-half pair.
    dense = torch.cat((compact[:, :, :16].permute(1, 0, 2).reshape(rows, 128),
                       compact[:, :, 16:].permute(1, 0, 2).reshape(rows, 128)), dim=1)
    if key:
        lo, hi = dense[:, :128], dense[:, 128:]
        c, s = cos[531:581], sin[531:581]
        # ISA computes two FP16 products and their FP16 sum.
        dense = torch.cat(((lo * c).half() + (hi * -s).half(),
                           (hi * c).half() + (lo * s).half()), dim=1).half()
    # ACC16 starts at +0 and the actual producer/writer emits canonical +0.
    # Encode this in the oracle; never normalize the measured cache bytes.
    dense = torch.where(dense == 0, torch.zeros_like(dense), dense)
    padded = torch.zeros((64, 256), dtype=torch.float16)
    padded[:rows] = dense
    result = initial.clone().view(torch.float16).reshape(128, 16, 8, 16, 16)
    blocks = padded.reshape(4, 16, 16, 16).permute(0, 2, 1, 3)
    if not key:
        blocks = blocks.transpose(-1, -2)
    result[34:38] = blocks[:, :, None, :, :].expand(-1, -1, 8, -1, -1)
    return result.contiguous().view(torch.uint8).flatten()


def fixture(name, *, upload=None):
    from production_pi_db_candidates import nvfp4_fixture
    from production_linear_launchers import activation, raw
    from rpu_backend.runtime.weights import transform_linear_weight

    if name not in CASES:
        raise ValueError(f"unsupported Q+KV1 fixture: {name}")
    if name == CONSUMER:
        return consumer_fixture(upload=upload)
    rows, layer = (int(v) for v in name.removeprefix("pi_q_kv1_m").split("_l"))
    if upload is None:
        upload = lambda value: value.to("rpu")
    query = nvfp4_fixture("pi_nvfp4_q_db_sspm", rows=rows, upload=upload)
    owners = list(query[4])
    oracles = []
    nn = torch.arange(2048)
    for seed in (3, 22):
        index = (nn * 13 + seed) % 1024
        index[-1] = 1023
        sign = (1 - 2 * (nn % 2)).to(torch.int8)
        weight = torch.zeros((2048, 1024), dtype=torch.int8)
        weight[nn, index] = sign
        scale = ((nn % 3 + 2).float() / 16).half()
        owners.extend((upload(transform_linear_weight(weight, 1, 8)), upload(scale)))
        oracles.append((index, sign, scale))
    images, expected = [], []
    # Distinct layer values expose wrong tensor-scale selection, while powers
    # of two keep the independently computed sparse FP16 result exact.
    tensor_scale = (2.0 ** (torch.arange(64) % 3 - 3)).float()
    for arm in range(2):
        x = activation(8, rows, 1024, arm)
        result = [(query[7][arm][0].float() * (float(tensor_scale[layer]) / .25)).half()]
        for index, sign, scale in oracles:
            cores = []
            for core in range(8):
                # Pair owner chooses two n16 stripes inside each physical
                # 256-channel owner, not its first contiguous 32 channels.
                ids = core * 256 + torch.cat((torch.arange(core * 16, core * 16 + 16),
                                             torch.arange(core * 16 + 128, core * 16 + 144)))
                cores.append((x[core, :, index[ids]].float() * sign[ids].float() * scale[ids].float()).half())
            result.append(torch.stack(cores))
        image = [raw(x)] + [raw(torch.full_like(value, float("nan"))) for value in result]
        image += [query[5][arm][3], raw(tensor_scale.repeat(8, 1))]
        images.append(image)
        expected.append(result)
    return "pi_q_kv1_pair_sspm", rows, 8, layer, owners, images, [1, 2, 3], expected, True


def consumer_fixture(*, upload=None):
    if upload is None:
        upload = lambda value: value.to("rpu")
    value = list(fixture("pi_q_kv1_m50_l3", upload=upload))
    shape = (1, 128, 1, 16, 8, 16, 16)
    initial = [(torch.arange(256).byte() + shift).repeat(32768) for shift in (17, 83)]
    phase = (torch.arange(1024)[:, None] + torch.arange(128)[None, :]) % 4
    cos = torch.tensor([1, 0, -1, 0], dtype=torch.float16)[phase]
    sin = torch.tensor([0, 1, 0, -1], dtype=torch.float16)[phase]
    value[0] = "pi_q_kv1_pair_cache_sspm"
    value[4] += [upload(raw.view(torch.float16).reshape(shape)) for raw in initial]
    value[4] += [upload(cos), upload(sin)]
    value[7] = [result + [production_cache_expected(initial[0], result[1], cos, sin, key=True),
                          production_cache_expected(initial[1], result[2], cos, sin, key=False)]
                for result in value[7]]
    return tuple(value)


def main():
    import production_linear_launchers as common
    common.fixture = fixture
    common.CASES = CASES
    common.main()


if __name__ == "__main__":
    main()

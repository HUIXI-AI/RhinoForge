"""Separate same-semantics Pi candidate calls; does not change the frozen matrix.

Run through the original production-launcher BUILD/REPLAY/guard checks. The
reference cases call the current default launcher. All candidates remain
explicit test entries until same-board performance evidence selects a schedule.
"""
from __future__ import annotations

import torch

CASES = ["pi_kv_owner_reference", "pi_kv_owner_db"] + [
    f"pi_nvfp4_{projection}_{variant}"
    for projection in ("q", "o", "down", "geglu")
    for variant in ("reference", "db", "db_tscale", "db_sspm")
]


def nvfp4_fixture(name, *, upload=None, rows=50):
    from rpu_backend.quant.nvfp4_pack import (
        swizzle_pack_nvfp4_v2, repack_nvfp4_tile_scales, repack_nvfp4_spm_scales,
    )
    from production_linear_launchers import activation, raw

    if upload is None:
        upload = lambda value: value.to("rpu")
    projection, variant = name.removeprefix("pi_nvfp4_").split("_", 1)
    assert projection in ("q", "o", "down", "geglu")
    assert variant in ("reference", "db", "db_tscale", "db_sspm")
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 64:
        raise ValueError("Pi candidate fixtures require M1..64")
    if projection == "geglu" and rows != 50:
        raise ValueError("Pi GeGLU candidate fixture requires M50")
    cores = 8
    col = projection in ("q", "geglu")
    n = 4096 if projection == "geglu" else 2048 if col else 1024
    k = 1024 if col else 2048 if projection == "o" else 4096
    local_n, local_k = (n // cores, k) if col else (n, k // cores)
    nn = torch.arange(n)
    weights, scales, specs = [], [], []
    for seed in ((3, 41) if projection == "geglu" else (3,)):
        index = (nn * 13 + seed) % local_k
        index[-1] = local_k - 1
        sign = (1 - 2 * ((nn + seed) % 2)).half()
        codes = torch.zeros((n, k), dtype=torch.uint8)
        for core in range(1 if col else cores):
            codes[nn, index + core * local_k] = torch.where(sign > 0, 2, 10).byte()
        # Powers of two keep the sparse projection exact in FP16.
        logical = (2.0 ** ((torch.arange(k // 16)[:, None] + nn[None, :] + seed) % 3 - 1)).float()
        block = logical.to(torch.float8_e4m3fn).view(torch.uint8)
        weight, striped = swizzle_pack_nvfp4_v2(codes, block, int(col), cores)
        scale = (repack_nvfp4_spm_scales(striped, int(col)) if variant == "db_sspm" else
                 repack_nvfp4_tile_scales(striped, int(col)) if variant == "db_tscale" else striped)
        weights.append(upload(weight)); scales.append(upload(scale))
        specs.append((index, sign, logical))
    images, expected = [], []
    for arm in range(2):
        x = activation(cores, rows, local_k, arm)
        projected = []
        for index, sign, logical in specs:
            values = []
            for core in range(cores):
                ids = nn[core * local_n:(core + 1) * local_n] if col else nn
                groups = (index[ids] + (0 if col else core * local_k)) // 16
                factor = sign[ids] * logical[groups, ids].half() * 0.25
                values.append((x[core, :, index[ids]].float() * factor.float()).half())
            projected.append(torch.stack(values))
        result = projected[0]
        if projection == "geglu":
            result = (torch.nn.functional.gelu(result.float(), approximate="tanh").half() * projected[1]).half()
        poison = torch.full_like(result, float("nan"))
        table = torch.full((cores, 64), 0.25, dtype=torch.float32)
        image = [raw(x), raw(poison), raw(table)] + ([raw(table.clone())] if projection == "geglu" else [])
        if variant == "db_sspm":
            # Poison again before every BUILD/REPLAY. The recorded scale DMA
            # must restore each distinct per-core slab before compute.
            image.extend(torch.full((cores, scale.numel() // cores), 0x5a + arm,
                                    dtype=torch.uint8) for scale in scales)
        images.append(image)
        expected.append([result])
    profile = f"pi_nvfp4_{'geglu' if projection == 'geglu' else 'linear'}_{variant}"
    mode = {"q": 0, "o": 1, "down": 2, "geglu": 0}[projection]
    return profile, rows, cores, mode, weights + scales, images, [1], expected, projection != "geglu"


def main():
    import production_linear_launchers as common
    original = common.fixture

    def fixture(name):
        if name in ("pi_kv_owner_reference", "pi_kv_owner_db"):
            value = list(common.qkv_fixture("pi_kv_owner_w8"))
            if name == "pi_kv_owner_db": value[0] = "pi_kv_owner_db"
            return tuple(value)
        if name in CASES:
            return nvfp4_fixture(name)
        return original(name)

    common.fixture = fixture
    common.CASES = CASES
    common.main()


if __name__ == "__main__":
    main()

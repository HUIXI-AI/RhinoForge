"""Independent addresses and frozen rhino-ops bytes for the new weight codec."""
import hashlib

import numpy as np
import pytest
import torch

from rpu_backend.quant.grouped_uint4 import pack_grouped_uint4
from rpu_backend.quant.grouped_uint4_n32k32 import (
    pack_grouped_uint4_n32k32,
    unpack_grouped_uint4_n32k32,
)




def _source(partition, cores, nc, kc, experts=2):
    n, k = (nc * cores, kc) if partition == "col" else (nc, kc * cores)
    return torch.from_numpy(np.random.default_rng(42).integers(
        0, 16, (experts, n, k), dtype=np.uint8))


def _address_oracle(codes, partition, cores):
    """Place each logical nibble at its documented byte address, no reshape codec."""
    e_count, n_count, k_count = codes.shape
    nc = n_count // cores if partition == "col" else n_count
    kc = k_count if partition == "col" else k_count // cores
    e, n, k = np.indices(codes.shape, dtype=np.int64)
    core = n // nc if partition == "col" else k // kc
    nl, kl = n % nc, k % kc
    nb, kb = (nc + 31) // 32, kc // 32
    offset = (((((e * nb + nl // 32) * kb + kl // 32) * cores + core)
               * 16 + nl % 16) * 2 + (nl % 32) // 16) * 16 + kl % 16
    shift = ((kl % 32) // 16) * 4
    output = np.zeros(e_count * nb * kb * cores * 512, dtype=np.uint8)
    np.bitwise_or.at(output, offset.ravel(), (codes.numpy() << shift).ravel().astype(np.uint8))
    return torch.from_numpy(output.reshape(e_count, -1))


@pytest.mark.parametrize("cores", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("partition", ["col", "row"])
@pytest.mark.parametrize("nc,kc", [(16, 32), (32, 64), (48, 96), (80, 256)])
def test_pack_matches_independent_nibble_addresses_and_roundtrips(cores, partition, nc, kc):
    codes = _source(partition, cores, nc, kc)
    original = codes.clone()
    packed = pack_grouped_uint4_n32k32(codes, partition=partition, num_cores=cores)
    assert packed.dtype == torch.uint8 and packed.is_contiguous()
    assert tuple(packed.shape) == (2, cores * ((nc + 31) // 32) * 32 * kc // 2)
    assert torch.equal(packed, _address_oracle(codes, partition, cores))
    restored = unpack_grouped_uint4_n32k32(
        packed, n=codes.shape[1], k=codes.shape[2], partition=partition, num_cores=cores)
    assert torch.equal(restored, codes)
    assert torch.equal(codes, original)


# Produced with ParallelLinearGenerator.pack_w4_n32k32 from rhino-ops main
# 381d77f18876839a2d3793465903dbc08348fcf7, source SHA256
# 4b5cf4d3f86826b52a97942355e017d4316f61ac8cfa6147a82f35c8400177cb.
# Inputs are _source(..., experts=2). Production tests do not import rhino-ops.
_MAIN_BYTES = [
    ("col", 1, 16, 32, "b4d68b4c07f586a1556d59fd704a5318e108c737c4ecc17dc2e713abe5c19b03"),
    ("row", 1, 48, 96, "359978d9784e04d4484feb4963ecd107a5158f6a5c0215378f409b1ebc4ba29a"),
    ("col", 2, 48, 64, "eb6bd3ac357f56b61f95876036a231da9d5676d23b3987242e930ec2a35cdedc"),
    ("row", 2, 80, 128, "948a86561ec298ec4bb4c723a6458441ec5798f445c24f44f68b389fae97d2bd"),
    ("col", 4, 32, 96, "57e3b72e2a4e3b1995f04d38656a6bdfb40847143aec923cd31126615b1d3a91"),
    ("row", 4, 16, 64, "9c9cf22230f9967302ca88851a6130c68195346ee4adb73f33dcdbf67a7210b4"),
    ("col", 8, 48, 256, "ec642f560ba31a5834a7bfdbbdf8569b38f628de2cee819022a52846a5e653cb"),
    ("row", 8, 80, 192, "0ce2060806185c9a1e1cf4a61f1a6f835b7c04508d02ba8a0a3ab7d1ce65cefa"),
    ("col", 8, 64, 2048, "57ae02765f3de30be9048d2cb9becc27e25594e8e623143887387a6bf3766742"),
    ("row", 8, 2048, 64, "f34f0cc70aa1472908b43e99210468f8ebea4121072bf7cfe0e308169bd18f8b"),
]


@pytest.mark.parametrize("partition,cores,nc,kc,digest", _MAIN_BYTES)
def test_packed_bytes_match_pinned_main(partition, cores, nc, kc, digest):
    codes = _source(partition, cores, nc, kc)
    packed = pack_grouped_uint4_n32k32(codes, partition=partition, num_cores=cores)
    assert hashlib.sha256(packed.numpy().tobytes()).hexdigest() == digest




def test_noncontiguous_input_owners_and_outputs_do_not_alias():
    codes = _source("row", 4, 48, 128).transpose(1, 2).contiguous().transpose(1, 2)
    assert not codes.is_contiguous()
    packed = pack_grouped_uint4_n32k32(codes, partition="row", num_cores=4)
    byte_owner = torch.zeros((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8)
    byte_owner[:, ::2].copy_(packed)
    strided = byte_owner[:, ::2]
    assert not strided.is_contiguous()
    restored = unpack_grouped_uint4_n32k32(strided, n=48, k=512, partition="row", num_cores=4)
    assert torch.equal(restored, codes)
    restored.zero_()
    assert torch.equal(packed, _address_oracle(codes, "row", 4))


@pytest.mark.parametrize("shape,partition,cores,message", [
    ((1, 128, 64), "col", True, "num_cores"),
    ((1, 128, 64), "col", 0, "num_cores"),
    ((1, 128, 64), "col", 6, "N divisible"),
    ((1, 128, 64), "col", 9, "num_cores"),
    ((1, 128, 64), "bad", 8, "partition"),
    ((1, 127, 64), "col", 8, "N divisible"),
    ((1, 16, 255), "row", 8, "K divisible"),
    ((1, 120, 64), "col", 8, "localN16"),
    ((1, 128, 48), "col", 8, "localK32"),
    ((1, 16, 128), "row", 8, "localK32"),
])
def test_invalid_topology_and_shape_rejected(shape, partition, cores, message):
    with pytest.raises(ValueError, match=message):
        pack_grouped_uint4_n32k32(torch.zeros(shape, dtype=torch.uint8),
                                 partition=partition, num_cores=cores)


@pytest.mark.parametrize("codes", [
    torch.zeros((1, 16, 32), dtype=torch.int8),
    torch.zeros((16, 32), dtype=torch.uint8),
    torch.zeros((0, 16, 32), dtype=torch.uint8),
    torch.empty((1, 16, 32), dtype=torch.uint8, device="meta"),
])
def test_invalid_code_storage_rejected(codes):
    with pytest.raises(ValueError, match="nonempty CPU uint8 rank-3"):
        pack_grouped_uint4_n32k32(codes, partition="col", num_cores=1)


def test_out_of_range_codes_fail_without_modifying_input():
    codes = torch.zeros((1, 16, 32), dtype=torch.uint8)
    codes[0, 0, 0] = 16
    with pytest.raises(ValueError, match=r"\[0,15\]"):
        pack_grouped_uint4_n32k32(codes, partition="col", num_cores=1)
    assert codes[0, 0, 0] == 16


@pytest.mark.parametrize("function", [unpack_grouped_uint4_n32k32])
def test_packed_storage_extent_and_dimension_types_are_checked(function):
    with pytest.raises(ValueError, match="CPU uint8 rank-2"):
        function(torch.zeros((1, 512), dtype=torch.int8), n=16, k=64,
                 partition="col", num_cores=1)
    with pytest.raises(ValueError, match="expert byte count"):
        function(torch.zeros((1, 511), dtype=torch.uint8), n=16, k=64,
                 partition="col", num_cores=1)
    for n, k in ((True, 64), (16, 64.0), (0, 64), (16, -64)):
        with pytest.raises(ValueError, match="positive integer"):
            function(torch.zeros((1, 512), dtype=torch.uint8), n=n, k=k,
                     partition="col", num_cores=1)


def test_unpack_rejects_nonzero_padding():
    # For localN16, thread0's second byte16 vector is N16 padding.
    packed = pack_grouped_uint4_n32k32(torch.zeros((1, 16, 32), dtype=torch.uint8),
                                      partition="col", num_cores=1)
    packed[0, 16] = 1
    with pytest.raises(ValueError, match="N padding must be zero"):
        unpack_grouped_uint4_n32k32(packed, n=16, k=32, partition="col", num_cores=1)


@pytest.mark.parametrize("partition", ["col", "row"])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_public_grouped_and_dense_awq_share_published_layout(partition, cores):
    from rpu_backend.quant.awq_gemm import swizzle_pack_uint4_pgrp
    codes = _source(partition, cores, 32, 64, experts=1)
    expected = _address_oracle(codes, partition, cores)
    assert torch.equal(pack_grouped_uint4(codes, partition=partition, num_cores=cores), expected)
    assert torch.equal(swizzle_pack_uint4_pgrp(codes[0], int(partition == "col"), cores), expected.flatten())

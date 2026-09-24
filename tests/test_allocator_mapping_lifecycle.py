from __future__ import annotations

import gc
import os

import pytest
import rpu_backend  # noqa: F401 - installs the torch.rpu PrivateUse1 facade
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RHINOFORGE_RUN_BOARD_TESTS") != "1"
    or not torch.rpu.is_available(),
    reason="requires explicit RHINOFORGE_RUN_BOARD_TESTS=1 and an idle, locked RPU",
)


@pytest.fixture
def clean_allocator():
    torch.rpu.set_caching_allocator(True)
    torch.rpu.empty_cache()
    tensors = []
    try:
        yield tensors
    finally:
        # Clear the list even when pytest retains the failed test's traceback.
        # Otherwise a stats assertion can leave thousands of live mappings in
        # the next test and obscure its baseline reservation.
        tensors.clear()
        gc.collect()
        torch.rpu.empty_cache()

def test_many_logical_allocations_keep_hostddr_mappings_bounded(clean_allocator) -> None:
    """Exercise the real allocator/Launch mapping lifecycle on an RPU host."""
    tensors = clean_allocator
    tensors.extend(
        torch.empty(1, dtype=torch.uint8, device="rpu") for _ in range(5000)
    )
    live = torch.rpu.memory_stats()
    assert live["caching_allocator_enabled"] is True
    assert live["allocation"]["current"] >= 5000
    assert live["caching_allocator_mapping"]["current"] <= 4
    assert live["cached_idle_mapping"] == 0

    # Sample both sides of the former 4096-mapping failure boundary. Distinct
    # round trips prove the packed logical tensors do not alias.
    probes = (0, 1, 4095, 4096, 4999)
    for value, index in enumerate(probes, 1):
        tensors[index].copy_(torch.tensor([value], dtype=torch.uint8))
    assert [int(tensors[index].cpu().item()) for index in probes] == [
        1,
        2,
        3,
        4,
        5,
    ]

    tensors.clear()
    gc.collect()
    idle = torch.rpu.memory_stats()
    assert idle["allocation"]["current"] == 0
    assert (
        idle["cached_idle_mapping"]
        == idle["caching_allocator_mapping"]["current"]
    )

    torch.rpu.empty_cache()
    released = torch.rpu.memory_stats()
    assert released["caching_allocator_mapping"]["current"] == 0
    assert released["cached_idle_mapping"] == 0


def test_medium_allocations_do_not_reserve_twenty_megabytes_each(clean_allocator) -> None:
    before = torch.rpu.memory_stats()["reserved_bytes"]["current"]

    tensors = clean_allocator
    tensors.extend(
        torch.empty(524_289, dtype=torch.float16, device="rpu")
        for _ in range(5)
    )
    after = torch.rpu.memory_stats()["reserved_bytes"]["current"]
    assert after - before <= 5 * 2 * 1024 * 1024

    tensors.clear()
    gc.collect()
    torch.rpu.empty_cache()
    assert torch.rpu.memory_stats()["reserved_bytes"]["current"] <= before

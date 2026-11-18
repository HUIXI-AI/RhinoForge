# R1 DDR bandwidth limits

Multi-core DDR read bandwidth on R1 is governed by SMMU pair sharing, controller
striping, and the physical core set granted by the kernel driver rather than a
linear multiple of single-core throughput. Use this contract when reasoning
about reduced-core execution templates, decode weight-streaming floors, and
hardware provenance in reports. [Cold topology](../../python/rpu_backend/runtime/topology.py)

## Contract

- Physical DDR and KV-cache controller striping remains fixed at eight
  controllers (`PHYSICAL_CORE_COUNT = 8`) even when a model runs with a 4- or
  6-core compute budget. [Controller striping ABI](../../python/rpu_backend/runtime/topology.py)
- Launch queues address logical cores `0..N-1`; the kernel driver assigns the
  underlying physical cores per task grant. On an 8-core R1 board under the
  verified default pair-interleaved driver configuration (`core_pair_order=1`),
  two physical cores share each SMMU read path in stride-4 pairs (`{0,4}`,
  `{1,5}`, `{2,6}`, `{3,7}`): 4-core grants receive `{0,1,4,5}`, 6-core grants
  receive `{0,1,2,4,5,6}`, and 8-core grants receive `{0..7}`. [Verified driver placements](../../examples/common/hardware.py)
- Multi-core broadcast kernels synchronize across all participating cores, so
  effective weight-streaming read bandwidth is bounded by the slowest active
  SMMU-sharing core rather than `N * single_core_bw`. Reduced-core decoder
  templates (`(4,6,4)` and `(4,6,6)`) keep attention on 4 lanes and widen MLP
  (and supported output heads) to 6 lanes within the 6-core grant. [Planner core topology](../../docs/planner_core_topology.md)
- Large streaming DMA reads (`ddr2spm`, >= 8 MiB) approach the multi-core read
  plateau, whereas DMA writeback (`spm2ddr`), small transfers, and mixed
  read/write kernels sustain lower effective throughput; apply weight-streaming
  decode floors only to large read-dominated stages. [DMA wrappers](ddr-dma-wrappers.md)

## Provenance and stop conditions

- Never assume logical cores `0..N-1` map to physical cores `0..N-1`. Report
  tooling records `DERIVED` physical core sets only when the running driver
  `srcversion` and allocation parameters match a hardware-verified tuple on an
  exclusively held device, and records `UNAVAILABLE` otherwise. [Hardware readback](../../examples/common/hardware.py)
- Never extrapolate 4-, 6-, or 8-core bandwidth ceilings by multiplying a
  1-core measurement by core count.

## Sources

- [Cold model topology and reduced-core templates](../../python/rpu_backend/runtime/topology.py)
- [Planner core topology guide](../../docs/planner_core_topology.md)
- [Board and driver core-allocation readback](../../examples/common/hardware.py)
- [DDR/SPM DMA wrappers](ddr-dma-wrappers.md)
- [SPM allocation](spm-allocation.md)

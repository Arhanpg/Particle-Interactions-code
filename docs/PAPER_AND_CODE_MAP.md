# Paper and code walkthrough

This guide maps the terminology in Algis et al., arXiv:2406.16091v1, to the imported source. The source identifiers are kept exactly as written upstream.

## 1. Problem and data layout

The task is to compute interactions between particles (i) and (j) when they are distinct and their distance is below the cutoff radius. A regular 3-D grid partitions the simulation domain. With cell width at least equal to the cutoff, a particle's candidate neighbours are in its own cell or one of the 26 surrounding cells. A bounded number of particles per cell makes the cell-list strategy scale approximately linearly with particle count for the intended workload.

The implementation uses a structure-of-arrays (SoA) representation: x, y, z coordinates, values, and index information are stored in separate arrays.

## 2. Preprocessing pipeline

The benchmark driver is `cuda/testParticlesPaper.cu` or `hip/testParticlesPaper.hip`. The major stages are:

| Stage | Source function | Purpose |
|---|---|---|
| Random initialization | `InitRandStates`, `InitParticles` | Initialize random states and uniformly distributed particle positions. |
| Position mapping | `PosToCellCoord`, `CellCoordToIndex` | Convert a position to a cell coordinate and linearized cell index. |
| Cell population | `ComputeNbParticlePerCells` | Count particles in each cell using concurrent atomic updates. |
| Prefix offsets | `PrefixFullV2` in `prefixsum.hpp` | Convert counts into cumulative per-cell offsets. |
| Cell metadata | `InitNewCells` | Build cell descriptors from offsets and counts. |
| Particle reordering | `MoveParticlesToNewCells` | Place particles into contiguous cell ranges. |
| Interaction statistics | `ComputeNbInteractions` | Count interactions for benchmark reporting. |

For cell counts (c_0,c_1,ldots,c_{M-1}), the prefix offset array has (offset[0]=0) and (offset[k+1]=offset[k]+c_k). Cell (k) therefore owns the segment from `offset[k]` inclusive to `offset[k+1]` exclusive. This allows the kernel to enumerate one cell's particles through a contiguous section of the sorted arrays.

## 3. The interaction kernels

### Particle-parallel methods (paper Section 4.1 and 4.2)

- **Par-Part-NoLoop — `ComputeParticleInterationsParPartNoLoop`:** the global thread index identifies a target particle. Its thread finds the target cell and iterates over source particles in neighbouring cells. The launch should cover at least the number of particles.
- **Par-Part-Loop — `ComputeParticleInterationsParPartLoop`:** similar interaction logic, but a loop over targets permits a more flexible relationship between launch geometry and particle count.

These methods expose parallel work over target particles. Threads in a warp may visit different cells, so source accesses can be less regular.

### Cell-parallel methods (paper Section 4.3 and 4.4)

- **Par-Cell — `ComputeParticleInterationsParCell`:** a block is assigned to a target cell. Its threads process particles in that cell. Some threads can be idle when a cell has fewer particles than the block size.
- **Par-Cell-SM — `ComputeParticleInterationsParCellSm`:** follows cell-based ownership but cooperatively stages source particles into shared memory in chunks. Threads then reuse that staged tile for multiple interactions.

### Proposed shared-memory layouts (paper Section 5)

- **All-in-SM / V1 — `ComputeParticleInterationsV1`, `manager_v1::SphInteractionManager`:** a block owns a 3-D sub-box of target cells plus the required ghost/halo cells. It loads that region into shared memory so internal source cells can be reused by multiple target cells. The manager uses maximum particles per cell and device shared-memory capacity to choose a feasible sub-box. The layout involves global offsets, local offsets, particle-index mappings, and particle data. High shared-memory demand can lower occupancy.
- **X-pencil / V2 — `ComputeParticleInterationsV2`, `manager_v2::SphInteractionManager`:** target work is arranged along the X axis. The block stages an X pencil, computes its targets, then stages neighbouring Y/Z pencils one at a time. It trades repeated staging and synchronizations for a smaller shared-memory footprint than the full 3-D region.
- **X-pencil-reg / V3 — `ComputeParticleInterationsV3`, `manager_v3::SphInteractionManager`:** target particle values are kept in registers; source pencils are successively loaded into shared memory. The paper says this version was added to present an interesting programming pattern and does not report its performance results.

## 4. What the manager headers do

The three `sph_manager_v*.hpp` headers contain the launch-configuration and device-side manager logic for V1, V2, and V3. Their configuration logic queries device capabilities such as multiprocessor count, shared-memory capacity, and (where needed) maximum threads per block. It uses the maximum particles per cell and the desired geometric region to decide whether an approach can be launched and how much shared memory is required.

A configuration can be marked unavailable on a particular GPU/workload. When V1/V2/V3 is skipped, that can be a deliberate consequence of shared-memory, thread-count, or grid-parallelism constraints; it is not automatically a runtime failure.

## 5. Parallel execution and memory spaces

A kernel launch creates a grid of blocks. Blocks execute independently and may be scheduled on separate multiprocessors. Threads within one block cooperate on that block's target region.

- **Global memory:** stores the full particle arrays and cell metadata. It is large but has higher access latency.
- **Shared memory:** a block-local scratchpad. Threads stage source particles here and reuse them. All participating threads must respect the block's synchronization points around buffer writes and reads.
- **Registers:** private to a thread. V3 uses them to retain target-particle information while source tiles are changed in shared memory.

The algorithmic difference is therefore both the ownership of target work (particle, cell, sub-box, or pencil) and the reuse scope of source particles (global reads, chunked staging, entire sub-box staging, or pencil-by-pencil staging).

## 6. Prefix sum (paper Section 6)

The paper describes an in-place tree scan for shared memory. During the upward pass, values are accumulated at the tree's right-hand positions; during the downward pass, partial sums are propagated to obtain all prefix values. The stated design goal is to reduce memory accesses and avoid an extra shared-memory buffer. The global `PrefixFullV2` routine separately organizes the per-cell offsets needed for particle reordering.

## 7. Experimental interpretation

The manuscript's experimental devices are NVIDIA T600, NVIDIA A100, and AMD Instinct MI210. The workload uses single-precision arithmetic and uniformly distributed particles. The reported speedups are relative to Par-Part-NoLoop and only apply to the test configurations in the paper. In particular, the paper reports selected X-pencil gains and generally poor All-in-SM performance when shared-memory demand reduces occupancy. V3 source execution should not be confused with a reported published V3 result.

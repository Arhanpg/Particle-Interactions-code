# Complete Codebase Explanation — GPU Particle Interactions

> **Purpose of this document:** explain what every major source file and function does, how the CUDA/HIP work is distributed across threads and blocks, how the cell list is built, how shared memory and registers are used, and how the benchmark is run. It is written as a learning and presentation guide for explaining the implementation to a faculty member.

## 0. What this project is

This repository contains the CUDA and HIP benchmark implementation associated with:

**David Algis, Bérenger Bramas, Emmanuelle Darles, and Lilian Aveneau**, “Efficient GPU Implementation of Particle Interactions with Cutoff Radius and Few Particles per Cell,” arXiv:2406.16091v1, 23 June 2024.

- Research paper: https://arxiv.org/abs/2406.16091
- Original project identified by the paper: https://gitlab.inria.fr/bramas/gpu-particle-interaction
- Upstream source mirror used for this repository: https://github.com/berenger-eu/gpu-particle-interaction/tree/479a812b8cba168f911474f1b04e4c02665aa858

The implementation compares ways to compute local particle-particle interactions on a GPU. The central optimization question is:

**Can we load source-particle coordinates from slow global memory once, place them in fast block-local shared memory, and reuse them for several target particles without consuming so much shared memory that GPU occupancy collapses?**

The project contains simple particle-parallel and cell-parallel baselines, a cell-based shared-memory baseline, and three managed shared-memory variants: All-in-SM, X-pencil, and X-pencil-reg.

The source files have been imported from the upstream mirror. This document explains the existing code; it does not claim that the algorithms were independently rewritten or that new benchmark results were produced.

---

# Part I — The problem and the execution model

## 1. The particle interaction problem

Assume there are N particles in a three-dimensional simulation box. Each particle has a position:

- particle i: (xᵢ, yᵢ, zᵢ)
- particle j: (xⱼ, yⱼ, zⱼ)

The program computes a potential contribution from a target particle i and nearby source particles j. The self-pair i = j is excluded.

A naïve all-pairs implementation checks every other particle for every target particle. That gives approximately N × N candidate pairs and therefore O(N²) work.

The paper instead uses a regular grid of cells. The simulation domain is split into small three-dimensional regions. A particle is assigned to the cell containing its position. If each cell's width is at least the cutoff radius, a target particle only needs to search its own cell and the immediately neighbouring cells.

In three dimensions, that is at most:

- 1 current cell;
- 6 face-adjacent cells;
- 12 edge-adjacent cells;
- 8 corner-adjacent cells.

Total: 1 + 6 + 12 + 8 = 27 cells, commonly described as a 3 × 3 × 3 neighbourhood.

If the number of particles per cell is bounded, the number of candidates per target particle is also bounded. The cell-list approach can therefore make the work scale approximately linearly with N for the intended distribution.

### 1.1 How the grid is configured in this code

Inside executeSimulation:

- The box is a cube with BoxWidth = 1.0 in each dimension.
- CellWidth is set to the requested cutoff value in all three dimensions.
- GridDim is computed as BoxWidth / cutoff for each dimension.
- The benchmark passes cutoff = 1 / boxDiv.

For example, when boxDiv is 8:

- cell width = 1/8;
- grid dimensions = 8 × 8 × 8;
- number of cells = 8³ = 512.

The paper's tested grid divisions are 2, 4, 8, 16, and 32.

### 1.2 What “cutoff” means in the current source

The cell grid is used to restrict the candidate set to the target cell and its neighbouring cells. In the interaction routines, the Lennard-Jones function is then evaluated for every candidate in those cells except the target particle itself.

**Important source-level detail:** the current Lennard-Jones function does not contain an additional per-pair condition such as distance < cutoff. The cutoff locality is implemented through cell-neighbour traversal. Do not describe the current code as performing a separate exact distance cutoff test for every candidate pair unless you add and validate such a test yourself.

## 2. GPU execution basics used by this project

Understanding the following terms makes the source much easier to explain.

### 2.1 Host and device

- **Host:** the CPU. It allocates memory, configures the experiment, launches kernels, synchronizes, times work, and writes CSV files.
- **Device:** the GPU. It executes the functions marked with CUDA/HIP execution qualifiers such as __global__ and __device__.

The host does not normally execute the body of a GPU kernel. It launches the kernel with a grid and block configuration.

### 2.2 Grid, block, and thread

A kernel launch has a grid of thread blocks. Each block has multiple threads.

- **Grid:** the collection of blocks launched for one kernel.
- **Block:** a group of threads that can cooperate and synchronize with each other.
- **Thread:** one execution lane that performs its assigned work.

CUDA's standard global thread index in a one-dimensional launch is:

    globalThreadId = blockIdx.x * blockDim.x + threadIdx.x

The code calls this value uniqueIdx in several kernels.

For example, if blockDim.x = 128:

- block 0 handles global thread IDs 0–127;
- block 1 handles 128–255;
- block 2 handles 256–383.

The kernel can use that ID to assign a particle or a portion of a cell to a thread.

### 2.3 Global memory

Global memory is the device's large memory. The complete arrays of particle coordinates and cell metadata live here. It has much greater capacity than shared memory, but individual accesses are more expensive.

### 2.4 Shared memory

Shared memory is a small, fast memory region local to a thread block.

All threads in a block can read and write the same shared-memory arrays. This makes it useful when several target particles need the same source coordinates.

The shared-memory pattern is:

1. Threads cooperatively copy a tile from global memory to shared memory.
2. Threads synchronize so no one reads a partially filled tile.
3. Threads reuse the tile for calculations.
4. Threads synchronize again before the same shared buffer is overwritten by the next tile.

In CUDA, the block-wide barrier used in this source is __syncthreads(). HIP uses the corresponding block synchronization primitive through its HIP programming model.

### 2.5 Registers

Registers are private, very fast storage owned by an individual thread. Other threads cannot directly read another thread's registers.

The V3/X-pencil-reg strategy keeps each target particle's coordinates in the target thread's local variables/registers. It then stages source particles in shared memory. When the target pencil itself is needed as a source, its target threads copy their locally held coordinates into the shared source tile.

### 2.6 Atomics

An atomic operation performs an update to a shared memory location without allowing conflicting concurrent updates to corrupt the result.

This project uses:

- atomicAdd to count particles per cell;
- atomicMax to determine the maximum particles in any cell;
- atomicAdd to allocate a unique output slot within each cell during particle reordering;
- atomicAdd to accumulate the total candidate interaction count.

### 2.7 Occupancy

Occupancy describes how many warps/blocks can be active on a multiprocessor relative to the hardware limit.

Using more shared memory per block can reduce the number of blocks that fit concurrently on a multiprocessor. This is a key trade-off in the paper:

- too little reuse causes repeated global-memory traffic;
- too much shared-memory allocation can reduce occupancy and hide less latency.

---

# Part II — Repository and source-file map

## 3. What each file is responsible for

| File | Responsibility |
|---|---|
| README.md | Project overview, algorithm map, quick-start and citation. |
| explain.md | This full code walkthrough and presentation guide. |
| CITATION.cff | Machine-readable citation metadata for GitHub. |
| THIRD_PARTY_NOTICES.md | Upstream source provenance and licensing notes. |
| docs/PAPER_AND_CODE_MAP.md | Short paper-to-source reference. |
| docs/RUNNING_AND_BENCHMARKING.md | Build and benchmark instructions. |
| cuda/testParticlesPaper.cu | CUDA benchmark driver, data structures, preprocessing kernels, interaction kernels, and main(). |
| cuda/prefixsum.hpp | Global prefix-sum implementation and shared-memory scan helpers. |
| cuda/point3d.hpp | Point3D specializations based on CUDA vector types. |
| cuda/sph_manager_v1.hpp | V1 / All-in-SM shared-memory manager and its launch-configuration logic. |
| cuda/sph_manager_v2.hpp | V2 / X-pencil shared-memory manager and its launch-configuration logic. |
| cuda/sph_manager_v3.hpp | V3 / X-pencil-reg manager and its launch-configuration logic. |
| cuda/SpTimer.hpp | CPU wall-clock timer used around synchronized benchmark loops. |
| cuda/CMakeLists.txt | CUDA build configuration and executable creation. |
| hip/testParticlesPaper.hip | HIP version of the benchmark driver and kernels. |
| hip/prefixsum.hpp | HIP version of the prefix-sum implementation. |
| hip/point3d.hpp | Point3D specializations based on HIP vector types. |
| hip/sph_manager_v1.hpp | HIP V1 / All-in-SM manager. |
| hip/sph_manager_v2.hpp | HIP V2 / X-pencil manager. |
| hip/sph_manager_v3.hpp | HIP V3 / X-pencil-reg manager. |
| hip/SpTimer.hpp | Shared host timer header. |
| hip/CMakeLists.txt | HIP build configuration. |

The CUDA and HIP source files follow the same overall algorithmic structure. Their runtime APIs, headers, error types, and random-number generator types differ.

## 4. The full program flow

The program is easiest to understand as two large phases:

1. **Preprocessing:** create a compact cell-list representation of the particles.
2. **Interaction calculation:** run several alternative GPU kernels over that prepared representation and compare their execution times.

    CPU main()
       |
       v
    executeSimulation()
       |
       +--> Allocate particle arrays on GPU
       |
       +--> InitRandStates kernel
       |
       +--> InitParticles kernel
       |
       +--> ComputeNbParticlePerCells kernel
       |       |
       |       +--> atomicAdd per-cell counts
       |       +--> atomicMax maximum cell population
       |
       +--> PrefixFullV2()
       |
       +--> InitNewCells kernel
       |
       +--> MoveParticlesToNewCells kernel
       |       |
       |       +--> atomicAdd ticket per cell
       |       +--> write particle into its cell segment
       |
       +--> ComputeNbInteractions kernel
       |
       +--> Run and time interaction variants
       |       |
       |       +--> Par-Part-Loop
       |       +--> Par-Part-NoLoop
       |       +--> Par-Cell
       |       +--> Par-Cell-SM
       |       +--> V1 / All-in-SM
       |       +--> V2 / X-pencil
       |       +--> V3 / X-pencil-reg
       |
       +--> Return per-method metrics
       |
       v
    main() writes timestamped CSV

The key point to tell your professor is that **the expensive neighbour-search structure is constructed once per workload before the interaction kernels are timed**. The benchmark then compares different ways to traverse and reuse that prepared data.

---

# Part III — Data structures and mathematical helpers

## 5. ParticlesContainer: how particle data is stored

The ParticlesContainer structure in testParticlesPaper.cu stores pointers to separate arrays:

| Member | Meaning |
|---|---|
| x | X coordinate of each particle. |
| y | Y coordinate of each particle. |
| z | Z coordinate of each particle. |
| v | Accumulated potential/result for each particle. |
| index | Original particle index. |
| nbParticles | Total particle count. |

This is a **structure-of-arrays (SoA)** layout.

For example, particle 0's position is obtained from x[0], y[0], and z[0]. Particle 1's position is obtained from x[1], y[1], and z[1].

Why use SoA?

- A warp of threads can load the same coordinate component from consecutive particle indices.
- Coordinate arrays are contiguous.
- It avoids loading unrelated fields when a kernel only needs positions.
- Separate arrays make it straightforward to copy or stage only x, y, and z into shared memory.

The structure has two helper methods:

- swap exchanges the pointers and particle count with another container.
- copy copies every array (x, y, z, v, index) device-to-device using cudaMemcpy.

The GPU arrays are allocated by AllocateParticles. That function calls cudaMalloc for each array and records the allocated pointers in a host-side list. deletePtrs later iterates through that list and calls cudaFree.

Source: [ParticlesContainer and allocation helpers](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L65).

## 6. CellDescriptor: the cell-list representation

Each cell has only two pieces of metadata:

- nbParticleInCell: how many particles belong to the cell;
- offsetInContainer: the first particle index of that cell in the reordered particle arrays.

Suppose the cell populations are:

    Cell counts:  [2, 1, 3, 0, 2, 1]

The exclusive prefix offsets are:

    Offsets:      [0, 2, 3, 6, 6, 8, 9]

This means:

- cell 0 owns particle indices [0, 2);
- cell 1 owns [2, 3);
- cell 2 owns [3, 6);
- cell 3 owns [6, 6), so it is empty;
- cell 4 owns [6, 8);
- cell 5 owns [8, 9).

The pair (count, offset) lets a kernel jump directly to the particles of a cell without scanning all N particles.

The code builds this information in InitNewCells.

## 7. Point3D and grid indexing

point3d.hpp specializes Point3D for:

- float, using CUDA float3;
- double, using CUDA double3;
- int, using CUDA int3;
- uint32_t, using CUDA uint3.

It also provides int3ToDim3 to convert integer grid dimensions into the dim3 form expected by kernel launches.

The static assertions check that the float and double specializations have the expected packed sizes. This matters when a Point3D object is stored in a shared-memory array and the code assumes it occupies exactly three scalar values.

### 7.1 Position to cell coordinate

PosToCellCoord converts a physical coordinate to an integer cell coordinate:

    cellX = floor(positionX / cellWidthX)
    cellY = floor(positionY / cellWidthY)
    cellZ = floor(positionZ / cellWidthZ)

The source performs this conversion using integer casts after division. The overload accepting Point3D simply forwards its x, y, and z components.

### 7.2 Three-dimensional coordinate to one-dimensional cell index

The grid is stored as a linear array. The mapping in CellCoordToIndex is:

    linearIndex = (z * gridY + y) * gridX + x

The X coordinate changes fastest, then Y, then Z.

IndexToCoord performs the reverse operation by dividing/modulo operations.

This linearization is important because per-cell counts and prefix offsets are ordinary one-dimensional arrays even though the logical grid is three-dimensional.


### Additional utility and validation helpers

The benchmark driver also contains smaller functions that support the main path:

- **M_Min and M_Max:** device-side templated helpers that return the smaller or larger of two values. They are used when clamping neighbour-cell bounds to the valid grid.
- **M_Abs:** a device-side absolute-value helper. Similar absolute-value logic is used by the V3 manager when checking the distance between pencil coordinates.
- **PBC:** a helper that shifts a coordinate by one box length when it is outside the interval. It is defined in the driver, but the current interaction path does not call it; see the boundary-handling caveat in Part XII.
- **PrintParticles:** a debug kernel that uses a grid-stride loop to print particle indices and coordinates. It is not part of the normal timed benchmark path.
- **CheckEqualCpu:** copies the particle arrays from the device to host vectors and compares positions, original indices, and potential values. Potential differences are checked against a relative tolerance of 1e-5, with a special case for a zero reference.
- **CheckEqual:** performs similar checks in a GPU kernel. Each thread checks one or more particle indices using a grid-stride loop.
- **CUDA_ASSERT:** wraps CUDA runtime API calls. If a call returns an error, it prints the CUDA error string and the source file/line and terminates the process. Kernel launches are asynchronous, so a later synchronization or runtime call is needed to surface many execution errors.
- **AllocateParticles:** allocates the five device arrays in ParticlesContainer and records the pointers in a host-side list so they can be released later.
- **deletePtrs:** iterates over a pointer list, calls cudaFree for each pointer, and synchronizes the device.
- **ResultFrame:** groups one workload's particle count, candidate interaction count, loop count, grid division, per-method timing/throughput results, V2 launch statistics, and V3 launch statistics.
- **getFilename:** uses the host system clock and formats the result as results-YYYYMMDD-HHMMSS.csv. The CSV is therefore timestamped and written to the executable's current working directory.

These helpers are useful to understand the full program even though they are not new interaction algorithms.

## 8. Distance and Lennard-Jones potential

The distance helper calculates:

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    r = sqrt(dx*dx + dy*dy + dz*dz + softening)

The softening constant in this source is 1e-5 and is added to the squared-distance expression. Its purpose is to avoid a singular or numerically problematic distance when two randomly placed particles are extremely close.

The Lennard-Jones potential function then calculates:

    ratio = sigma / r
    ratio6 = ratio * ratio * ratio * ratio * ratio * ratio
    ratio12 = ratio6 * ratio6
    potential = 4 * epsilon * (ratio12 - ratio6)

The interaction kernels call it with epsilon = 1.0 and sigma = 1.0.

The function also has assertions checking that intermediate values are not zero where prohibited, NaN, or infinite.

**Presentation point:** this benchmark focuses on parallelizing the evaluation and data movement for the pair potential. The physical interaction expression is intentionally kept the same across the kernel variants so the implementations can be compared.

Source: [distance and Lennard-Jones functions](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L35).

---

# Part IV — Building the cell list on the GPU

## 9. Step 1: initialize random-number states

Function: InitRandStates  
Source: testParticlesPaper.cu, near line 101.

The benchmark allocates one CURAND state per launched thread position. The kernel computes:

    uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x
    totalThreads = gridDim.x * blockDim.x

Each thread initializes its assigned state entries using curand_init with the fixed seed 1234. The loop uses a grid-stride pattern:

    for (idx = uniqueIdx; idx < numberOfStates; idx += totalThreads)

A grid-stride loop means a thread handles its first index, then jumps by the total number of launched threads to handle another index if required.

Why this pattern?

- It supports input arrays larger than the number of launched threads.
- It spreads work among the GPU threads.
- It avoids requiring exactly one thread for every array element.

The fixed seed makes the random initialization reproducible for the same implementation and execution configuration.

## 10. Step 2: initialize the particles

Function: InitParticles  
Source: testParticlesPaper.cu, near line 178.

Each active thread obtains its CURAND state and generates x, y, and z values inside the simulation box.

For each particle, the kernel sets:

- x, y, z to random coordinates in the box;
- index to the original particle index;
- v to -1 as an initial placeholder.

The kernel uses the same grid-stride pattern. A thread may initialize more than one particle if the array is larger than the total number of launched threads.

The random states are copied back to their state array after the thread finishes its assigned particles.

## 11. Step 3: map every particle to a cell

Functions:

- PosToCellCoord
- CellCoordToIndex

For each particle, the physical position is divided by the cell width to get a cell coordinate. The coordinate is then linearized into a single cell index.

For example, for a 4 × 4 × 4 grid, a particle at cell coordinate (2, 1, 3) maps to:

    index = (3 * 4 + 1) * 4 + 2 = 54

This index selects the corresponding entry in particlePerCells.

## 12. Step 4: count particles per cell in parallel

Function: ComputeNbParticlePerCells  
Source: testParticlesPaper.cu, near line 663.

This is one of the most important preprocessing kernels.

### 12.1 What each thread does

Each GPU thread starts with:

- currentCellId = -1;
- currentCellAddition = 0;
- maxNbPartPerCell = 0.

Then it visits particle indices in a grid-stride loop.

For each particle:

1. Read x, y, and z.
2. Convert the position to a cell coordinate.
3. Convert that coordinate into a linear cell index.
4. Compare the new cell index to the cell currently being accumulated by this thread.

If the cell changes, the thread atomically adds its accumulated count to the global count array for the previous cell. It then switches to the new cell and resets its local count.

At the end of the loop, the thread flushes its final accumulated count.

### 12.2 Why does the kernel aggregate counts locally before using atomicAdd?

A simple implementation could perform one atomicAdd for every particle. This implementation first groups consecutive particles handled by the same thread when their computed cell ID is identical, then performs one atomicAdd for that local run.

The atomic operation is still required because multiple GPU threads may update the same cell. The local accumulation can reduce the number of atomic operations when a thread encounters consecutive particles in the same cell.

### 12.3 How is the maximum particles-per-cell value computed?

The kernel also calculates the maximum population of any cell.

- Each thread tracks the largest updated population it observes.
- Each block combines thread results into sharedMaxNbPartPerCell using atomicMax.
- Thread 0 of the block updates the global maximum with another atomicMax.

The host copies this maximum value back because V1, V2, and V3 use it when calculating safe shared-memory and thread configurations.

### 12.4 Why is the maximum important?

The manager cannot safely reserve enough storage based only on the average particles per cell. A cell can contain more particles than the average. The maximum count gives the managers a conservative upper bound for sizing the particle tiles.

Source: [ComputeNbParticlePerCells](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L663).

## 13. Step 5: compute the prefix sum

Function: PrefixFullV2  
Source: prefixsum.hpp.

The input array contains the population of every cell. The prefix sum converts these counts into starting offsets.

For counts:

    counts = [2, 1, 3, 0, 2, 1]

The exclusive offsets are:

    offsets = [0, 2, 3, 6, 6, 8, 9]

The final value is the total number of particles.

In executeSimulation, prefixParCell is allocated with NbCells + 1 entries and initialized to zero. The prefix routine is called with output pointer prefixParCell + 1. This preserves prefixParCell[0] = 0 and places the cumulative values after it.

The prefix array is subsequently used by InitNewCells and MoveParticlesToNewCells.

## 14. Step 6: build one descriptor for each cell

Function: InitNewCells  
Source: testParticlesPaper.cu, near line 760.

One global thread index selects one cell. A grid-stride loop permits threads to process more than one cell if required.

For cell k:

    count[k]  = prefix[k + 1] - prefix[k]
    offset[k] = prefix[k]

The kernel writes these values into CellDescriptor.

This stage does not move any particles. It only prepares the metadata that describes where each cell's particle segment will live.

## 15. Step 7: reorder particles into cell-contiguous storage

Function: MoveParticlesToNewCells  
Source: testParticlesPaper.cu, near line 773.

This kernel converts the original randomly ordered arrays into arrays grouped by cell.

Each thread:

1. Reads one particle's original position.
2. Computes its cell coordinate and cell index.
3. Calls atomicAdd on the counter/ticket for that cell.
4. Uses the returned old counter as the particle's unique local slot.
5. Adds the cell's starting offset to obtain the global destination index.
6. Writes x, y, z, and original index to that destination.

The key expression is:

    ticket = atomicAdd(ticketCounter[cellIdx], 1)
    destination = cell.offsetInContainer + ticket

### Why is atomicAdd necessary here?

Imagine four threads are trying to insert particles into the same cell at the same time. Without an atomic operation, they could all read the same counter value and overwrite the same destination slot.

atomicAdd gives every insertion a different ticket. Therefore, each particle gets a unique position in its cell's segment.

### Why reorder the data at all?

After reordering, all particles belonging to one cell are stored next to each other. A kernel can read:

    start = cell.offsetInContainer
    end   = start + cell.nbParticleInCell

and iterate from start to end without searching the entire particle array.

That is the core data-structure optimization used by all interaction kernels.

### Source-level detail about the result array

MoveParticlesToNewCells writes the coordinate arrays and original index array. It does not write the v array in this kernel. The surrounding code then copies the temporary container back using ParticlesContainer::copy, which copies v as well. The interaction kernels subsequently calculate and overwrite v for their target particles. Do not treat the temporary v contents as valid between reordering and the first interaction calculation.

## 16. Step 8: count the candidate interactions

Function: ComputeNbInteractions  
Source: testParticlesPaper.cu, near line 274.

This kernel calculates how many source-particle candidates the interaction kernels will visit.

For each target particle, it:

1. Finds the target cell.
2. Adds the number of other particles in the same cell: currentCell.nbParticleInCell - 1.
3. Adds the populations of the neighbouring cells.
4. Adds the thread's local total into the global interaction counter using atomicAdd.

This count is used to calculate interactions per second and the benchmark's GFLOP/s estimate.

It is a count derived from cell populations. It does not compute the Lennard-Jones potential and does not perform a per-pair distance test.

Because every target particle is counted against its possible source particles, the reported count is naturally a directed target-source interaction count: the work for target i with source j and target j with source i are counted as separate target calculations.

---

# Part V — The baseline interaction kernels

The preprocessing stages prepare the same sorted particle arrays and cell descriptors for all methods. The following kernels differ in how they assign target particles to threads and how they fetch source particles.

## 17. Par-Part-NoLoop: one thread per target particle

Function: ComputeParticleInterationsParPartNoLoop  
Source: testParticlesPaper.cu, near line 233.

### Mapping

The kernel computes:

    uniqueIdx = blockIdx.x * blockDim.x + threadIdx.x

This index is the target particle index.

If uniqueIdx is smaller than the particle count, the thread processes that one target particle. The code asserts that the total launched thread count is at least the number of particles.

### Work done by one thread

1. Load the target particle's x, y, and z.
2. Convert its position to a cell coordinate.
3. Loop over z-1 to z+1.
4. Loop over y-1 to y+1.
5. Loop over x-1 to x+1.
6. Clamp these neighbour coordinates to valid grid coordinates.
7. Convert each neighbour coordinate into a linear cell index.
8. Read the source cell's offset and particle count.
9. Loop over all source particles in that cell.
10. Skip the target particle itself.
11. Load each source position and calculate the Lennard-Jones contribution.
12. Store the accumulated potential in v[uniqueIdx].

### What does “NoLoop” actually mean?

It means that the kernel does not use a grid-stride loop to assign multiple target particles to a single thread. It is intended to assign one target particle to each active thread.

The thread still contains loops over neighbouring cells and source particles. Therefore, “NoLoop” does not mean that the whole kernel has no loops.

### Parallelism

Parallelism is across target particles. If there are N particles, the launch provides at least N threads. Each thread owns one output value v[i], so different threads write to different output positions.

### Strength and limitation

- Strength: simple mapping and many independent target-particle calculations.
- Limitation: different threads can belong to different cells, so their source accesses can be irregular. Source arrays are not necessarily accessed in a perfectly coalesced pattern.

## 18. Par-Part-Loop: grid-stride particle-parallel implementation

Function: ComputeParticleInterationsParPartLoop  
Source: testParticlesPaper.cu, near line 318.

This method uses the same general ownership model: threads own target particles.

The key difference is the grid-stride loop:

    for (idxPart = uniqueIdx;
         idxPart < numberOfParticles;
         idxPart += totalThreads)

A thread can handle multiple target particles if the particle count exceeds the number of launched threads.

For each target particle, it:

1. Loads the target position.
2. Finds the current cell.
3. Loops over particles in the current cell, excluding self.
4. Loops over the neighbouring cells other than the current cell.
5. Calculates potential contributions.
6. Stores the result for that target particle.

The current cell is handled separately from the other neighbour cells in this implementation. It is still a 27-cell neighbour search in total.

### Why keep both particle-parallel variants?

They let the experiment compare a direct one-thread-per-particle mapping with a flexible grid-stride mapping. With the same launch configuration used by the benchmark, the paper expects their performance to be close because the major work assignment is similar.

## 19. Par-Cell: one block owns a target cell

Function: ComputeParticleInterationsParCell  
Source: testParticlesPaper.cu, near line 382.

### Mapping

The block index selects a target cell:

    idxCell = blockIdx.x

The kernel uses a grid-stride loop over cells:

    for (idxCell = blockIdx.x;
         idxCell < numberOfCells;
         idxCell += gridDim.x)

In the benchmark, the launch uses one block per cell, so usually each block is assigned one cell.

### How threads cooperate

Inside the selected cell, each thread starts with its thread index and advances by the block size:

    for (idxPart = threadIdx.x;
         idxPart < currentCell.nbParticleInCell;
         idxPart += blockDim.x)

That thread calculates the interaction result for the corresponding target particle.

For each target particle, it scans:
- the other particles in its own cell;
- the particles in the neighbouring cells.

It stores the result into the target particle's v entry.

### Why can this method waste threads?

The block has a fixed number of threads, but each cell can have a different particle count.

For example, if a block has 128 threads and a cell contains only 10 particles, only 10 threads have target-particle work in that iteration. The remaining threads are inactive for that target-cell iteration.

The last warp can also be only partially occupied when the particle count is not a multiple of the warp size.

### Shared-memory status

Par-Cell itself does not stage the source positions in shared memory. It reads source coordinates from the particle arrays in global memory.

## 20. Par-Cell-SM: cell-based work with shared-memory tiles

Function: ComputeParticleInterationsParCellSm  
Source: testParticlesPaper.cu, near line 442.

This kernel keeps cell-based ownership but improves source-data reuse by staging particles into shared memory.

The code declares three static shared arrays:

- sharedMemX[sizeSm];
- sharedMemY[sizeSm];
- sharedMemZ[sizeSm].

The benchmark instantiates sizeSm = 512.

### 20.1 Target particles are processed in chunks

The block loops over the target cell's particles in chunks of blockDim.x.

Each thread obtains a target index within the current chunk:

    idxPart = idxPartIter + threadIdx.x

The Boolean isPartInBlock checks whether that target index exists. This is needed because the final target chunk may contain fewer particles than the block size.

### 20.2 The current cell is staged first

The block loads up to sizeSm source particles from the current cell into the three shared arrays.

The copy is cooperative:

    for (idxCopy = threadIdx.x;
         idxCopy < numberInThisChunk;
         idxCopy += blockDim.x)

After the copy, __syncthreads() ensures that the shared-memory tile is ready.

Every active target thread then loops over the tile and computes interactions, skipping itself when the source is the same particle.

A second __syncthreads() ensures that all readers have finished before the shared arrays are overwritten by the next chunk.

### 20.3 Neighbour cells are staged one at a time

The kernel next traverses the neighbouring cells. For each neighbour cell:

1. Split that cell's particles into sizeSm-sized chunks.
2. Cooperatively copy a source chunk into sharedMemX/Y/Z.
3. Synchronize.
4. Every active target thread evaluates its interaction with the staged sources.
5. Synchronize before reusing the buffer.

### 20.4 Why use chunks?

The number of particles in a cell can be larger than the shared-memory tile capacity. Chunking lets the same kernel process larger cells without allocating a shared buffer large enough for the entire simulation.

### 20.5 Main trade-off

This method reduces repeated global-memory reads for source particles within a block. However, it introduces shared-memory writes, barriers, and repeated staging loops. The paper found that the classical cell shared-memory method becomes more interesting for high particle populations per cell; it is not automatically faster for sparse cells.

---

# Part VI — The proposed shared-memory algorithms

The next three methods use manager classes. These managers decide which spatial region a block owns, how much shared memory it needs, and how local particle indices map back to the global particle array.

The manager files are important: the interaction kernel itself is a relatively small wrapper, while much of the spatial layout and data movement lives in the manager class.

## 21. V1 — All-in-SM / full-load approach

Main kernel: ComputeParticleInterationsV1  
Manager: manager_v1::SphInteractionManager  
Files:
- cuda/testParticlesPaper.cu, near line 578;
- cuda/sph_manager_v1.hpp.

### 21.1 Core idea

Instead of assigning one block to one cell, V1 assigns a block to a 3-D sub-box of target cells.

The block also loads the surrounding ghost/halo cells required by the target cells. The halo is needed because particles near the boundary of the block's target region still interact with particles in adjacent cells.

The block tries to load the entire local sub-box plus halo into shared memory. Then several target particles can reuse the same source particles without repeatedly loading them from global memory.

### 21.2 Why are ghost cells necessary?

Imagine the block owns a target region that is 2 × 2 × 2 cells.

A target particle in the leftmost cell may need source particles from one cell to its left. A target in the rightmost cell may need source particles to its right.

The manager therefore expands the local region by one cell in each direction wherever the global grid allows it.

In the interior of the grid, the loaded region is larger than the target region. The extra cells are source-only ghost cells.

### 21.3 How the launch configuration is chosen

The host-side getKernelConf method calls getSharedmemConf.

The configuration logic considers:
- total grid cells;
- multiprocessor count;
- maximum shared memory per block;
- memory needed per cell, based on maximum particles per cell;
- ghost cells;
- desired blocks and threads.

The manager requires enough space for at least a 3 × 3 × 3 region. That is the minimum region containing one target cell plus its 26 neighbours in the interior.

It estimates a 3-D region shape, tries to increase its dimensions while staying within the shared-memory and parallelism limits, then removes the two halo layers from each dimension to determine the active target-cell dimensions.

The returned configuration contains:
- grid dimensions;
- threads per block;
- dynamic shared-memory byte count;
- an enable flag.

If the device cannot support the required region, the method returns disabled and the benchmark records no timing for that configuration.

### 21.4 What happens inside the V1 kernel?

The wrapper allocates dynamic shared memory:

    extern __shared__ unsigned char bufferOfShared[]

It creates an SphInteractionManager using that buffer.

The kernel then executes three main manager stages.

#### Stage A — manager.init

The init method determines the target-cell range owned by this block from blockIdx and gridDim.

For each axis it calculates:
- the first target cell;
- the last target cell;
- the local lower boundary including a ghost cell where available;
- the local number of loaded cells.

It then calculates the total number of local cells.

If the target sub-box is empty, the block returns without interaction work.

The manager next constructs metadata arrays in shared memory.

These include:

1. globalCellOffset: global particle offsets for the loaded cells;
2. localCellOffset: the particle offsets within the compact local shared-memory region;
3. locToGlob: a mapping from the local particle index to the original global particle index.

#### Stage B — convert global offsets into local offsets

This is the complex part of V1.

The loaded cells are arranged as a 3-D local sub-box. The code identifies X-pencils inside the local sub-box and computes how many particles must be skipped between successive pencils in the global linearized particle array.

It uses the global cell offsets to construct an auxiliary offsetGlobalToLobal array.

That array is scanned using buildPrefixSMDevice. The result lets the manager subtract particles that belong to cells outside the local sub-box but are included in the global prefix offsets.

After subtraction, localCellOffset represents compact positions within the local shared-memory particle region.

The manager then fills locToGlob. For each local particle position, this mapping identifies its corresponding global particle position.

Why is this mapping needed?

The shared-memory layout is compact and local to the block, but final output values must be written to the original global particle index. locToGlob connects those two index spaces.

#### Stage C — manager.bufferBuildAndCopy

This method allocates shared-memory arrays for source coordinates:

- srcX;
- srcY;
- srcZ.

Threads cooperatively loop over the local particle indices. Each thread reads its local-to-global mapping, loads the corresponding global coordinates, and stores them in shared memory.

The optional lambda parameter allows extra particle fields to be copied. In this benchmark wrapper, the lambda is empty because only x, y, and z are required for the Lennard-Jones calculation.

The method ends with __syncthreads() so all shared-memory particle coordinates are ready before interaction computation begins.

#### Stage D — manager.computeInteraction

Each thread processes local target-particle indices in a block-stride loop.

For each candidate target:
1. Read xI from the shared srcX/srcY/srcZ arrays.
2. Call the initialization lambda. The wrapper sets vI = 0.
3. Convert the target position into a global cell coordinate.
4. Check whether this particle belongs to the active target region or a ghost cell.
5. Skip ghost-cell particles as targets. Ghost cells are loaded only as sources.
6. Convert the target cell to local coordinates.
7. Compute the local neighbour bounds, covering the target cell and adjacent cells.
8. Traverse source cells using localCellOffset.
9. Traverse source particles inside those cells.
10. Exclude the same local particle.
11. Call the interaction lambda to accumulate the Lennard-Jones potential.
12. Call the release lambda to store the result at the global particle index.

The wrapper's interaction lambda performs:

    vI += lennardJonesPotential(xI, xJ, 1.0, 1.0)

The release lambda writes vI into the global v array using the target's global index.

### 21.5 What makes V1 different?

The entire 3-D region is loaded before most of the interaction calculations begin. The same source coordinates can be reused by several target cells in the block.

### 21.6 Why can V1 be slow?

A 3-D target region plus halo can consume a large amount of shared memory. More shared memory per block can reduce the number of resident blocks per multiprocessor.

So V1 trades global-memory reuse for:
- shared-memory allocation;
- local index arrays;
- prefix-sum work to build local offsets;
- synchronization;
- potentially lower occupancy.

The paper reports that this approach was generally not competitive in its tested configurations because of the shared-memory footprint and resulting occupancy pressure.

---

## 22. V2 — X-pencil / pencil-load approach

Main kernel: ComputeParticleInterationsV2  
Manager: manager_v2::SphInteractionManager  
Files:
- cuda/testParticlesPaper.cu, near line 539;
- cuda/sph_manager_v2.hpp.

### 22.1 Core idea

V2 uses a one-dimensional sub-region along X called an X-pencil.

A block owns an X segment at a fixed Y and Z coordinate. The block loads that X-pencil, computes its target particles, then loads neighbouring X-pencils in adjacent Y and Z rows one at a time.

Instead of loading an entire 3-D sub-box of particles into shared memory at once, it stages one pencil at a time.

### 22.2 How the launch configuration is selected

manager_v2::getSharedmemConf considers:

- number of X-pencils, which is gridY × gridZ;
- available multiprocessors;
- X-grid length;
- maximum particles per cell;
- shared-memory capacity;
- the requested divSM setting;
- maximum thread count per block.

At least three cells must fit in shared memory for an interior X-pencil: one target cell and a ghost cell at each end.

The manager also constrains the maximum target pencil length by the thread limit. The code uses a maximum of 1024 threads in this manager.

The shared-memory budget is divided by divSM before calculating the number of cells that fit. The benchmark calls V2 with divSM = 1, 2, and 4. This varies the amount of shared memory the configuration is allowed to consume and therefore changes the pencil length and launch configuration.

### 22.3 How a V2 block is assigned

The grid dimensions are:

- X grid dimension: number of X-sub-pencils;
- Y grid dimension: number of simulation cells along Y;
- Z grid dimension: number of simulation cells along Z.

blockIdx.y and blockIdx.z identify the Y and Z coordinates of the pencil. blockIdx.x identifies which segment of the X axis this block handles.

The manager calculates:
- first target X cell;
- last target X cell;
- first loaded X cell, including left halo where available;
- last loaded X cell, including right halo where available.

### 22.4 First load: copy the X-pencil cell offsets

The manager copies the prefix offsets for all X cells in the loaded pencil into shared memory.

Those offsets allow the block to determine:
- how many particles are in the loaded pencil;
- which particle interval belongs to the active target cells;
- where the source particles for each target cell begin and end.

The code synchronizes after copying the offsets.

### 22.5 Second load: copy particle coordinates

The number of particles in the X-pencil is obtained by subtracting the first cell offset from the final cell offset.

Threads with indices smaller than the number of particles load their particle positions into a shared array of Point3D values.

The manager also exposes a copy lambda. The benchmark's copy lambda is empty, because no additional data fields are needed for this interaction.

After the load, __syncthreads() ensures the target/source tile is ready.

### 22.6 Assign target particles to threads

For threads that correspond to target particles, the manager:
1. reads the target position from the shared tile;
2. computes the target particle's global cell coordinate;
3. checks that it belongs to the current X-pencil's Y/Z row;
4. calculates its local X-cell index;
5. determines its neighbouring X-cell range.

The target's local and global particle indices are maintained separately.

### 22.7 Compute on the current pencil

The kernel first computes interactions within the currently loaded X-pencil.

A target particle only needs sources from:
- the target's own X cell;
- the immediately preceding X cell;
- the immediately following X cell.

This is the X portion of the 3-D neighbourhood.

### 22.8 Load adjacent Y/Z pencils one at a time

The manager then loops over neighbouring Z and Y pencil coordinates.

For each neighbouring pencil:

1. Load its cell offsets into the shared offset buffer.
2. Calculate the number of particles in that pencil.
3. Copy its particle coordinates from global memory into the shared source buffer.
4. Synchronize.
5. For target threads whose Y/Z coordinates are within one cell of the current source pencil, calculate interactions with the required X-neighbour cells.
6. Synchronize before the buffer is overwritten by the next source pencil.

The source pencil is therefore reused by all participating target threads in the block while it remains loaded.

### 22.9 Finish each target

The release lambda stores the accumulated potential into the target particle's global v entry.

### 22.10 Why can X-pencil be faster?

It reduces the amount of shared memory required at one time compared with loading the full 3-D sub-box. It also reuses loaded source data among several target particles.

However, it must stage neighbouring Y/Z pencils repeatedly and synchronize between staging phases. The balance between reuse, shared-memory consumption, block count, and synchronization determines whether it wins.

The paper reports selected speedups for X-pencil, but emphasizes that the gains depend on the hardware and workload. It is not universally faster.

---

## 23. V3 — X-pencil-reg / register-storage variant

Main kernel: ComputeParticleInterationsV3  
Manager: manager_v3::SphInteractionManager  
Files:
- cuda/testParticlesPaper.cu, near line 627;
- cuda/sph_manager_v3.hpp.

### 23.1 Core idea

V3 is related to X-pencil, but it changes where target particle information is stored.

- Target particle coordinates are first loaded into per-thread local variables/registers.
- Source particles are loaded into shared memory one pencil at a time.
- When the source pencil overlaps the target sub-box, the manager can reuse target coordinates already held in registers rather than loading those target particles again from global memory.

This makes the target region more flexible than a simple X-oriented target pencil.

### 23.2 Launch configuration

V3's getSharedmemConf uses:

- GPU multiprocessor count;
- shared-memory capacity;
- maximum thread count per block;
- maximum particles per cell;
- total cells;
- an estimate of the useful-target-to-loaded-cell ratio.

The code has an explicit T600 special case: it limits the maximum threads to 512 for a device whose reported GPU name contains T600. Otherwise it uses the device's maximum threads-per-block attribute.

The configuration requires enough cells to hold an X-pencil plus its halo. It chooses a three-dimensional target region that respects thread and memory limits, and tries to improve the ratio of useful target cells to loaded cells.

### 23.3 Build the target region and halo offsets

Each block determines its target sub-box in X, Y, and Z.

The manager expands this region by one cell in each direction where the global grid allows it. This produces the loaded region with ghost cells.

The manager then copies the required global cell prefix offsets into shared memory. It keeps an additional X offset per Y/Z pencil so it can calculate the particle intervals for individual pencils.

### 23.4 Place target particles in registers

The code computes how many maximum particles can fit in an X-pencil:

    MaxPartPerXPencil = maxParticlesPerCell * numberOfLoadedXCells

It then assigns threads to target X-pencils and particle slots within each pencil.

An active target thread:
1. determines its target particle's global array index;
2. loads x, y, and z into the local variable xI;
3. initializes its accumulator through FuncInit;
4. finds which X cell contains the target particle.

The manager uses a binary search over the cell prefix offsets to locate the target's X cell inside the pencil.

The target coordinates remain in the thread-local xI variable while the manager moves through source pencils.

### 23.5 Stage source pencils in shared memory

The manager allocates shared arrays for source coordinates:

- srcX;
- srcY;
- srcZ.

Then it loops through the relevant neighbouring Y/Z pencil positions.

There are two situations.

**Situation A — a pencil is outside the target sub-box.**

It is a ghost/source pencil. Its required particle positions are copied from global memory into shared memory.

**Situation B — a pencil overlaps the target sub-box.**

The manager can reuse information already held in registers for target particles. It copies the required boundary source cells from global memory where needed and copies the target particles from their threads' local xI values into the correct shared-memory positions.

This is the register-reuse feature that distinguishes V3.

### 23.6 Synchronize, then compute

After staging a source pencil, the manager synchronizes the block.

Each target thread checks whether that source pencil is within one Y and one Z cell of its target. If it is, the thread selects the relevant X-neighbour source interval and loops over the corresponding shared-memory source coordinates.

The thread excludes its own particle and invokes the interaction lambda for the rest.

After source processing, another synchronization protects the next shared-memory overwrite.

### 23.7 Store the result

The release lambda writes the accumulator to the target's global v location.

### 23.8 How V3 differs from V2

| Aspect | V2 / X-pencil | V3 / X-pencil-reg |
|---|---|---|
| Target layout | X-pencil-oriented target work. | Target cells can form a 3-D sub-box. |
| Target storage | Target pencil is staged in shared memory and read by its threads. | Target position is retained in each target thread's registers. |
| Source storage | Shared-memory pencil. | Shared-memory pencil. |
| Reuse | Reuse source pencil among target threads. | Reuse target coordinates from registers when loading overlapping source pencils. |
| Main trade-off | Repeated staging of neighbouring Y/Z pencils. | More complex indexing and register/shared-memory coordination. |

**Paper distinction:** the manuscript describes V3 as an additional programming pattern and states that its performance results are not presented in the paper. The benchmark source can produce V3 measurements, but those are additional local results, not published V3 results from the manuscript.

---

# Part VII — Prefix-sum implementation

## 24. Why prefix sum is required

The cell-count kernel gives one count per cell. The particle-reordering kernel needs to know the first storage position reserved for every cell.

Prefix sum converts counts into those offsets.

For an input array:

    [c0, c1, c2, c3, ...]

the exclusive offset is:

    offset[0] = 0
    offset[1] = c0
    offset[2] = c0 + c1
    offset[3] = c0 + c1 + c2

The output length is the input length plus one in the way this benchmark stores it.

## 25. What prefixsum.hpp implements

The main host entry point is PrefixFullV2.

It divides the input into windows. Each complete window is scanned by BuildPartialPrefix_AllSize using shared memory. An incomplete final window is handled with a partially filled shared-memory buffer.

Inside BuildPartialPrefix_AllSize:

1. Threads copy values into a shared prefixTree.
2. Unused entries in a partial window are filled with zero.
3. The upward pass accumulates tree values.
4. The downward pass propagates prefix results.
5. Threads copy the result into the output buffer.

For the complete-window path, the shared-memory capacity is a compile-time constant and the block has half as many threads as shared-memory entries. Each thread handles two positions during the initial copy.

If the input has more than one window, PrefixFullV2 recursively computes a prefix over the partial-window totals. ApplyPartialPrefixV3 then adds each preceding-window total to entries in later windows.

The implementation uses a CUDA stream so the kernels and asynchronous allocation/free operations can be ordered. PrefixFullV2 synchronizes the supplied stream before it returns.

## 26. The shared-memory scan used by V1

V1 needs a prefix scan while it is constructing local offsets from the global cell-offset array.

manager_v1.hpp contains buildPrefixSMDevice, an in-place tree-style scan for a shared-memory array. It has:

- an upward accumulation pass;
- a downward propagation pass;
- block barriers between levels.

This is conceptually related to the scan discussed in Section 6 of the paper. The purpose is to compute local offsets without creating a second full-sized shared-memory scan buffer.

There are two related pieces of code to distinguish:

- prefixsum.hpp contains the general global PrefixFullV2 pipeline and a shared scan helper.
- manager_v1.hpp contains buildPrefixSMDevice for the temporary local offset array created during V1 manager initialization.

Do not confuse the global cell-list prefix scan with the smaller local prefix scan inside the V1 shared-memory manager.

---

# Part VIII — Host-side benchmark and results

## 27. executeSimulation: one complete workload

Function: executeSimulation  
Source: testParticlesPaper.cu, near line 854.

This function receives:

- inNbParticles: number of particles in this workload;
- inNbLoops: number of timed interaction repetitions;
- inCutoff: cell width/cutoff used to build the grid;
- inCheckResult: whether optional output comparisons should run.

### 27.1 Establish launch defaults and geometry

The baseline launch uses 128 threads per block.

The block count is calculated as:

    max(ceil(numberOfParticles / 128), 1)

The function sets:
- origin = (0, 0, 0);
- BoxWidth = (1, 1, 1);
- CellWidth = (cutoff, cutoff, cutoff);
- GridDim = BoxWidth / cutoff;
- NbCells = GridDim.x * GridDim.y * GridDim.z.

### 27.2 Allocate and initialize

It calls AllocateParticles, initializes CURAND states, and launches InitParticles.

It then allocates the per-cell count array and computes the particle population of each cell.

### 27.3 Build offsets and reorder

It allocates prefixParCell, calls PrefixFullV2, creates CellDescriptor entries, allocates a temporary particle container, and launches MoveParticlesToNewCells.

After reordering, ParticlesContainer::copy copies the temporary arrays back into the main particle container.

### 27.4 Count candidate interactions

ComputeNbInteractions provides the candidate interaction count used in the benchmark metrics.

### 27.5 Run each interaction method

The function creates a result vector and times the interaction methods separately:

1. Par-Part-Loop
2. Par-Part-NoLoop
3. Par-Cell
4. Par-Cell-SM
5. V1 / All-in-SM
6. V2 / X-pencil for divSM values 1, 2, and 4
7. V3 / X-pencil-reg

For methods that need separate output arrays, the source allocates and copies a particle container before timing.

For V1, V2, and V3, the manager first calculates whether the configuration is enabled and returns the grid size, block size, and shared-memory size. The benchmark prints those values. If a manager disables a configuration, its result is recorded as zero instead of launching an invalid configuration.

### 27.6 Why synchronize before and after timing?

GPU kernel launches are asynchronous with respect to the CPU. If the CPU timer stopped immediately after a launch, it could measure only launch submission time rather than completed GPU work.

The benchmark synchronizes before starting the timed region and synchronizes again after the repeated launches. This ensures all launched work has finished before the elapsed wall-clock time is read.

SpTimer.hpp uses std::chrono::high_resolution_clock. It is a host wall-clock timer, not a CUDA event timer.

### 27.7 Metrics

The result structure stores:

- time: average elapsed time per interaction-kernel iteration;
- gflops: estimated GFLOP/s using the benchmark's configured FLOP count;
- interactionsPerSecond: counted candidate interactions divided by elapsed time.

The calculation is based on:

    timePerIteration = totalElapsed / numberOfLoops
    interactionRate = totalCandidateInteractions * numberOfLoops / totalElapsed
    estimatedGflops = totalCandidateInteractions * numberOfLoops
                      * flopsPerInteraction / totalElapsed / 1e9

The source currently sets NbFlopsPerInteraction = 18. The paper's experimental description states 21 FLOPs per interaction under its stated counting convention. This discrepancy is documented in the repository's paper-to-code map. For a research comparison, state which FLOP convention is being used; elapsed time and interaction throughput do not depend on that FLOP multiplier.

## 28. main(): the experiment sweep

The main function is near the end of testParticlesPaper.cu.

It sets:

- NumType = float;
- BoxWidth = 1.0;
- NbLoops = 200;
- MaxParticlesPerCell = 128;
- MaxBoxDiv = 32.

The outer loop tries boxDiv values:

    2, 4, 8, 16, 32

For each box division:

    cellWidth = BoxWidth / boxDiv
    numberOfCells = boxDiv * boxDiv * boxDiv

The inner loop starts at one particle per cell on average and multiplies particle count by 10 each time, while staying below the configured maximum population bound.

For each workload, main calls executeSimulation and stores its results in a ResultFrame.

After all workloads finish, the program creates a timestamped filename:

    results-YYYYMMDD-HHMMSS.csv

The CSV contains:
- number of particles;
- number of candidate interactions;
- number of loops;
- grid division;
- number of cells;
- average particles per cell;
- interactions per particle;
- timing, GFLOP/s, and interaction rate for each method;
- V2 block/thread/shared-memory configuration statistics;
- V3 block/thread/shared-memory configuration statistics.

The output is written relative to the process's current working directory.

### 28.1 The default run does not enable correctness checking

The checked-in main function calls executeSimulation with inCheckResult = false.

The optional checks exist in the source. When enabled, the benchmark compares the particle arrays and output potentials against the reference result using CheckEqualCpu and CheckEqual.

The position and original-index checks use exact equality. The potential comparison uses a relative-difference tolerance of 1e-5 (with a special case for a zero reference value).

Do not claim that a default run has executed these optional checks.

---

# Part IX — How to explain the parallelism clearly

## 29. The same work, different ownership

The interaction problem is the same for every kernel: for every target particle, accumulate contributions from source particles in neighbouring cells.

The methods differ in two design decisions.

### Decision 1: who owns a target particle?

- Par-Part: one thread owns a target particle.
- Par-Cell: a block owns a target cell and its threads divide the particles in that cell.
- V1: a block owns a 3-D target sub-box.
- V2: a block owns an X-pencil segment.
- V3: a block owns a 3-D target sub-box, while individual target threads retain their target data in registers.

### Decision 2: where are the source particles read from?

- Baseline kernels: source positions are read from global arrays.
- Par-Cell-SM: source positions are copied into shared-memory chunks cell by cell.
- V1: all particles in the block's loaded sub-box and halo are staged into shared memory.
- V2: source pencils are loaded into shared memory one at a time.
- V3: source pencils are loaded into shared memory, while target positions are retained in registers and reused when possible.

## 30. A simple example to explain shared-memory reuse

Suppose one block has 8 target particles and all 8 need to interact with the same 20 source particles.

### Without shared-memory staging

A straightforward implementation may load the same source coordinates repeatedly while each target thread loops over its source list.

Conceptually, the source coordinates are fetched from global memory for many target/source combinations.

### With shared-memory staging

1. Threads in the block load the 20 source particles into shared memory.
2. The block synchronizes.
3. Each of the 8 target threads reads the 20 source positions from the shared tile.
4. The block synchronizes before the tile is replaced.

The source data are loaded once for the tile and reused by several target threads.

The real kernels use larger, irregular cell/pencil regions, but this small example explains the reason for staging.

## 31. Why more shared memory is not always better

A shared-memory tile has a finite size. A larger tile may reduce the number of repeated global loads, but each block reserves more shared memory.

If a block uses a large fraction of the multiprocessor's available shared memory, fewer blocks may be resident at once.

This can reduce active warps and make it harder for the GPU to hide memory and instruction latency.

That is why:
- V1's full 3-D staging can be limited by occupancy;
- V2's X-pencil can fit in a smaller shared-memory budget;
- V3 combines shared-memory source tiles with registers for target data;
- Par-Cell-SM can be useful for high populations per cell, where staging a chunk is worthwhile.

## 32. Why synchronization is required

Consider a shared array being filled by many threads.

Without a barrier, one thread could start reading an entry while another thread has not finished writing it. That would create a race.

The source uses __syncthreads() at the important boundaries:

- after source coordinates are staged;
- after interaction threads finish reading a tile;
- before the same tile is overwritten for the next chunk/pencil.

The barrier is block-wide. All participating threads must reach the corresponding barrier consistently. It is not a synchronization mechanism between different blocks.

---

# Part X — CUDA and HIP versions

## 33. How the HIP code relates to CUDA

The HIP folder contains the corresponding benchmark implementation:

- testParticlesPaper.hip;
- prefixsum.hpp;
- point3d.hpp;
- sph_manager_v1.hpp;
- sph_manager_v2.hpp;
- sph_manager_v3.hpp;
- SpTimer.hpp;
- CMakeLists.txt.

The algorithmic structure and kernel names are intended to mirror the CUDA source.

The platform-specific differences include:
- CUDA runtime headers versus HIP runtime headers;
- cudaMalloc/cudaMemcpy/cudaDeviceSynchronize versus HIP runtime equivalents;
- CURAND state types versus HIPRAND state types;
- CUDA error types and error-string helpers versus HIP error types;
- CUDA vector types versus HIP vector types;
- compiler architecture options.

HIP is designed to provide a programming model that can target supported AMD GPUs. However, the presence of a HIP source file does not itself prove that it builds on every ROCm release or every AMD GPU. The compiler, runtime version, architecture flag, and CMake environment still need to match.

---

# Part XI — Build configuration and how to run

## 34. CUDA CMake file

File: cuda/CMakeLists.txt.

The CUDA project:
1. Requires CMake 3.20 or newer.
2. Enables C++ and CUDA languages.
3. Requires CUDA_SM to be supplied.
4. Uses CUDA_SM as CMAKE_CUDA_ARCHITECTURES.
5. Sets C++17.
6. Includes the CUDA runtime and library directories.
7. Links the executable with cudart and cublas.
8. Finds .cu files in the CUDA source directory.
9. Creates an executable named after each .cu file.

The benchmark executable is named testParticlesPaper.

Example build:

    cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release -DCUDA_SM=80
    cmake --build build/cuda --parallel
    ./build/cuda/testParticlesPaper

Replace 80 with a compute capability supported by the installed CUDA Toolkit and the target GPU.

## 35. HIP CMake file

File: hip/CMakeLists.txt.

The HIP project:
1. Starts a CMake project using C++.
2. Adds the ROCm installation path.
3. Includes /opt/rocm/include.
4. Defines __HIP_PLATFORM_AMD__.
5. Enables the HIP language.
6. Sets GPU_TARGET to gfx90a by default.
7. Passes --offload-arch=GPU_TARGET to the compiler.
8. Finds .hip files and creates an executable from each.

Example build:

    CXX=hipcc cmake -S hip -B build/hip -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx90a
    cmake --build build/hip --parallel
    ./build/hip/testParticlesPaper

The gfx90a example corresponds to the AMD Instinct MI210 used by the paper. Select the correct architecture identifier for your own GPU. The upstream CMake file assumes ROCm is installed under /opt/rocm; adjust the include and prefix paths if it is installed elsewhere.

---

# Part XII — Code-level caveats to understand before presenting

These observations come from reading the current source. They are important because they prevent overclaiming what the benchmark does.

## 36. PBC helper is defined but not called by the current neighbour traversal

testParticlesPaper.cu defines a helper named PBC. However, the current baseline neighbour loops use max(0, cell-1) and min(gridDim-1, cell+1) to clamp to valid grid coordinates.

The V1/V2/V3 managers similarly clip their halo ranges to the global grid boundaries.

A search of the current CUDA and HIP sources shows the PBC helper definition but no call to it in the interaction path. Therefore, do not claim that the current interaction traversal wraps opposite simulation faces using periodic boundary conditions. The code currently clips neighbour ranges at the grid boundary.

## 37. No separate distance-cutoff predicate in the interaction loops

The interaction kernels restrict source candidates through the cell neighbourhood. The Lennard-Jones function calculates the potential from the supplied positions but does not itself compare r against the cutoff.

When explaining the code, say that the grid limits the source candidates to nearby cells. Do not say that each candidate pair is additionally filtered by an exact r < cutoff condition in the current source.

## 38. The FLOP count differs between the source and the paper

The paper's experimental description states 21 FLOPs per interaction under its stated convention. The benchmark source sets NbFlopsPerInteraction = 18.

This changes the reported GFLOP/s estimate, not the measured elapsed time or the candidate interaction count.

For a rigorous report:
- preserve the source's original 18 value when reproducing its CSV;
- state that the paper describes 21 under its counting convention;
- recompute GFLOP/s consistently if you want a separate paper-convention metric.

## 39. Optional correctness checks are disabled in the default sweep

The main function calls executeSimulation with false for inCheckResult. The comparison code is present but does not run by default.

For a correctness-focused experiment, enable the check path deliberately, rebuild, and document the exact configuration used.

## 40. Two small device allocations are not freed in executeSimulation

In the current CUDA driver, cuMaxNbPartPerCell and cuNbInteractions are allocated with cudaMalloc and copied back to the host, but they are not added to the main cleanup list or explicitly cudaFree'd in the shown function.

They are small allocations, but executeSimulation is called repeatedly by the workload sweep. This is a source-level cleanup issue worth fixing before extending the benchmark to very large repeated experiment batches.

## 41. The reordered temporary v field

MoveParticlesToNewCells writes x, y, z, and index to the temporary particle container. It does not write v. ParticlesContainer::copy later copies v as well.

The interaction kernels then calculate and write v for their target particles. The temporary v value should not be treated as a meaningful result between reordering and the interaction stage.

## 42. V3 source support is not a published V3 result

V3 exists in the code and the benchmark can produce its timing columns. The paper states that its V3 performance results are not presented in the manuscript. Keep any new V3 measurements clearly labelled as your own runs.

---

# Part XIII — A suggested explanation to give your professor

## 43. Five-minute presentation sequence

You can explain the project in this order:

**1. Problem**

“We are computing particle-particle interactions. A naïve approach checks every particle against every other particle, which is O(N²). The paper uses a regular grid and a cutoff neighbourhood so each target particle only checks its own cell and nearby cells.”

**2. Data preparation**

“Initially the particles are randomly distributed. The GPU maps every particle to a cell, counts particles per cell using atomic operations, computes prefix offsets, creates cell descriptors, and reorders the particle arrays so each cell's particles are contiguous.”

**3. Baseline parallelism**

“In Par-Part, one GPU thread owns one target particle and scans its neighbour cells. In Par-Cell, one block owns a cell and its threads divide the target particles in that cell. These are two different ways to assign target work.”

**4. Shared-memory reuse**

“Par-Cell-SM loads source particles into shared memory in chunks. All-in-SM loads a whole 3-D sub-box and halo. X-pencil loads the X-direction target pencil and then neighbouring Y/Z pencils one by one. X-pencil-reg keeps target coordinates in registers and stages source pencils in shared memory.”

**5. Performance trade-off**

“Shared memory reduces repeated global-memory reads, but large shared-memory tiles can reduce occupancy. Therefore, more reuse does not always mean better performance. X-pencil can perform well in the workloads tested by the paper, while All-in-SM is often limited by its shared-memory footprint.”

**6. Measurement**

“The benchmark runs each method repeatedly, synchronizes the GPU before stopping its CPU wall-clock timer, computes time and throughput metrics, and writes the results to a CSV file.”

## 44. Questions your professor may ask

**Q: Why do we need prefix sum?**

A: We need to convert the number of particles in each cell into a starting offset in the reordered particle array. Prefix sum provides those offsets in parallel.

**Q: Why do we use atomicAdd in particle reordering?**

A: Multiple threads may insert particles into the same cell concurrently. atomicAdd provides a unique ticket to each insertion so threads do not write to the same slot.

**Q: Why use SoA instead of an array of particle structures?**

A: Coordinates are stored in separate contiguous arrays. This is convenient for loading one coordinate component for consecutive particles and for staging only the required position fields.

**Q: What is the difference between global memory and shared memory?**

A: Global memory contains the complete arrays and is accessible across the GPU. Shared memory is a smaller, block-local scratchpad used to reuse data among threads in the same block.

**Q: Why does shared memory need synchronization?**

A: Threads cooperatively fill the shared buffer. A barrier guarantees that the buffer is ready before any thread reads it, and another barrier prevents it from being overwritten while other threads are still consuming it.

**Q: Why is V1 not necessarily faster?**

A: It loads a large 3-D region, including halo cells, into shared memory. This can reduce global reads but consume so much shared memory that fewer blocks can be active on a multiprocessor.

**Q: Why can X-pencil be more efficient?**

A: It stages one X-oriented pencil and then the neighbouring Y/Z pencils one at a time. It can use less shared memory than loading the complete 3-D sub-box, while still reusing each loaded source pencil across multiple target threads.

**Q: What does V3 add?**

A: It retains target coordinates in per-thread registers and stages source pencils in shared memory. For an overlapping target pencil, it can copy target values from registers instead of fetching those target coordinates from global memory again.

**Q: What happens if a manager says its configuration is disabled?**

A: The shared-memory/thread configuration is not considered feasible for that device and workload. The benchmark skips that method for the workload and records a zero result rather than launching it with an invalid configuration.

**Q: Does this benchmark perform an exact distance cutoff test for every source pair?**

A: Not in the current interaction functions. The candidate list is limited by the cell-neighbour search, and the Lennard-Jones function evaluates each candidate except self. The code has no separate r < cutoff branch in those interaction loops.

**Q: Does the current code wrap interactions across opposite grid faces?**

A: The PBC helper is defined, but the current interaction loops clamp neighbour coordinates at grid boundaries. The PBC helper is not called in the interaction path.

**Q: Does the default benchmark check that all kernels produce matching results?**

A: No. The optional comparison functions exist, but main passes false for the correctness-check argument.

---

# Part XIV — Source navigation

The most useful places to open while explaining the code are listed below.

| Topic | Source |
|---|---|
| Particle storage and cell descriptor | [testParticlesPaper.cu, data structures](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L65) |
| Position-to-cell conversion | [testParticlesPaper.cu, cell mapping](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L114) |
| Par-Part-NoLoop | [testParticlesPaper.cu, particle-parallel kernel](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L233) |
| Par-Part-Loop | [testParticlesPaper.cu, grid-stride particle kernel](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L318) |
| Par-Cell | [testParticlesPaper.cu, cell-parallel kernel](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L382) |
| Par-Cell-SM | [testParticlesPaper.cu, chunked shared-memory kernel](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L442) |
| V2/X-pencil wrapper | [testParticlesPaper.cu, V2](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L539) |
| V1/All-in-SM wrapper | [testParticlesPaper.cu, V1](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L578) |
| V3/X-pencil-reg wrapper | [testParticlesPaper.cu, V3](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L627) |
| Particle counting | [testParticlesPaper.cu, ComputeNbParticlePerCells](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L663) |
| Cell descriptors | [testParticlesPaper.cu, InitNewCells](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L760) |
| Particle reordering | [testParticlesPaper.cu, MoveParticlesToNewCells](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L773) |
| V1 manager | [sph_manager_v1.hpp](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/sph_manager_v1.hpp) |
| V2 manager | [sph_manager_v2.hpp](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/sph_manager_v2.hpp) |
| V3 manager | [sph_manager_v3.hpp](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/sph_manager_v3.hpp) |
| Prefix sum | [prefixsum.hpp](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/prefixsum.hpp) |
| Full workload setup and timing | [testParticlesPaper.cu, executeSimulation](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L854) |
| Experiment sweep and CSV output | [testParticlesPaper.cu, main](https://github.com/Arhanpg/Particle-Interactions-code/blob/main/cuda/testParticlesPaper.cu#L1293) |

## Final takeaway

The most important concept in this project is not the Lennard-Jones formula itself. It is **how the same local neighbour-interaction calculation is mapped onto GPU threads and blocks, and how source-particle data are reused through global memory, shared memory, and registers**.

The preprocessing stage makes cell access efficient. The baseline kernels compare particle-based and cell-based ownership. The proposed V1, V2, and V3 managers experiment with increasingly structured source-data reuse. Their performance depends on the balance among global-memory traffic, shared-memory capacity, synchronization, register usage, thread count, and GPU occupancy.

That is the central story of the paper and the codebase.

# Efficient GPU Particle Interactions

This repository contains the CUDA and HIP benchmark implementation associated with:

**David Algis, Bérenger Bramas, Emmanuelle Darles, and Lilian Aveneau.** “Efficient GPU Implementation of Particle Interactions with Cutoff Radius and Few Particles per Cell.” arXiv:2406.16091v1, 23 June 2024.

- Paper: https://arxiv.org/abs/2406.16091
- Original project cited by the paper: https://gitlab.inria.fr/bramas/gpu-particle-interaction
- Public source mirror used for the import: https://github.com/berenger-eu/gpu-particle-interaction/tree/479a812b8cba168f911474f1b04e4c02665aa858

The source files are imported from the upstream author's public GitHub mirror at the exact revision above. Upstream source comments and license notices are retained. This repository adds navigation, explanations, and run instructions; it does not claim a new algorithm implementation or new benchmark results.

## Repository contents

```text
.
├── README.md
├── LICENSE                       # pre-existing repository file; does not automatically re-license imports
├── THIRD_PARTY_NOTICES.md
├── docs/
│   ├── PAPER_AND_CODE_MAP.md
│   └── RUNNING_AND_BENCHMARKING.md
├── cuda/
│   ├── CMakeLists.txt
│   ├── testParticlesPaper.cu
│   ├── prefixsum.hpp
│   ├── point3d.hpp
│   ├── SpTimer.hpp
│   └── sph_manager_v1.hpp
│   └── sph_manager_v2.hpp
│   └── sph_manager_v3.hpp
└── hip/
    ├── CMakeLists.txt
    ├── testParticlesPaper.hip
    ├── prefixsum.hpp
    ├── point3d.hpp
    ├── SpTimer.hpp
    └── sph_manager_v1.hpp
    └── sph_manager_v2.hpp
    └── sph_manager_v3.hpp
```

## What the benchmark computes

The simulation box is a 3-D grid of cells. Cell width is set to the cutoff distance, so a target particle only needs to examine its own cell and the 26 adjacent cells (with periodic boundary handling in the implementation). Particles are stored in a structure-of-arrays layout: coordinates and particle values are in separate arrays.

The interaction calculation uses a softened Lennard-Jones potential. The driver initializes uniformly distributed random particles, constructs per-cell data, runs the interaction-kernel variants, measures repeated executions, and writes timestamped CSV results.

## Implementations and paper mapping

| Source kernel symbol | Name in the paper | Work assignment / data reuse |
|---|---|---|
| `ComputeParticleInterationsParPartLoop` | Par-Part-Loop | Particle-parallel baseline with a target-particle loop. |
| `ComputeParticleInterationsParPartNoLoop` | Par-Part-NoLoop | One GPU thread per target particle; each thread traverses source particles in neighbouring cells. |
| `ComputeParticleInterationsParCell` | Par-Cell | One thread block per target cell; threads process target particles in that cell. |
| `ComputeParticleInterationsParCellSm` | Par-Cell-SM | Cell-based work; source particles are staged through shared memory in chunks. |
| `ComputeParticleInterationsV1` | All-in-SM / full load | Load a 3-D sub-box, including its halo cells, into shared memory for reuse. |
| `ComputeParticleInterationsV2` | X-pencil / pencil load | Load an X-oriented pencil; process it, then stage adjacent Y/Z pencils in turn. |
| `ComputeParticleInterationsV3` | X-pencil-reg | Keep target particle data in registers and stage source pencils in shared memory. |

The unusual spelling `Interations` is part of the upstream kernel identifiers and is intentionally preserved.

**Important distinction:** the paper describes X-pencil-reg as an additional programming variant and explicitly says its performance results are not presented in the manuscript. V3 is in the source benchmark, but its output must not be described as a published V3 result.

## How parallelism is implemented

### Preprocessing and cell-list construction

The benchmark performs the following GPU stages before evaluating interactions:

1. Initialize random states and particle positions.
2. Convert each particle position to a 3-D cell coordinate and linear cell index.
3. Count particles in each cell. Concurrent threads update the per-cell counts using atomic operations.
4. Compute prefix offsets from the per-cell counts.
5. Initialize cell descriptors containing each cell's particle count and starting offset.
6. Reorder particles into contiguous cell segments. The per-cell counters are reused as insertion tickets during this pass.

For cell populations (c_0, c_1, ldots, c_{M-1}), the offsets satisfy (offset[0]=0) and (offset[k+1]=offset[k]+c_k). Cell (k) then owns the half-open segment ([offset[k], offset[k+1])) in the reordered arrays. This makes neighbour traversal over a cell's particles contiguous.

### Thread/block decomposition

- **Particle-parallel kernels:** a thread owns one target particle and traverses the particles belonging to the target cell and its neighbour cells. This exposes parallelism over particles and avoids requiring one full block for every cell.
- **Cell-parallel kernels:** a block owns a target cell and its threads cooperate over particles in that cell. When a cell has fewer particles than the block's thread count, some lanes do no target work. The final warp can also be partially occupied.
- **Shared-memory variants:** threads in a block cooperatively stage a subset of source particles in block-local shared memory, synchronize, and reuse the staged values for several interactions.

### Shared memory, registers, and occupancy

Shared memory is a fast scratchpad shared by threads in one block. It can reduce repeated global-memory reads, but the amount allocated per block limits how many blocks can be resident on a multiprocessor (occupancy).

- **Par-Cell-SM** stages source particles cell by cell and in chunks.
- **All-in-SM (V1)** stages a complete 3-D active sub-box plus ghost/halo cells. This aims for substantial reuse, but its memory footprint can be large. Its manager computes a feasible block/sub-box configuration from the device's shared-memory and multiprocessor limits.
- **X-pencil (V2)** stages target/source data in an X-oriented pencil. It handles the target pencil, then loads the required neighbouring pencils in Y/Z one at a time. It uses less shared memory than the full 3-D load in many configurations, but repeats staging and synchronization.
- **X-pencil-reg (V3)** keeps target particles in thread registers while shared memory is used for source pencils.

The V1/V2/V3 manager headers calculate launch geometry and shared-memory requirements. The kernels use block synchronization between buffer staging and interaction computation. Registers are private to individual threads; shared memory is shared within a block; global memory holds the full particle and cell data.

### GPU prefix sum

`prefixsum.hpp` contains the prefix-sum machinery used during preprocessing. The paper's Section 6 describes an in-place tree-style scan for shared memory, with an upward accumulation phase followed by a downward propagation phase. The intended benefit is to reduce intermediate shared-memory storage and memory operations. The host-side `PrefixFullV2` routine also coordinates partial scans for the global per-cell offsets.

## Build and run

The repository has separate CUDA and HIP CMake projects. Build from the repository root and keep build outputs outside the source directories.

### CUDA

Requirements: NVIDIA GPU and driver, CUDA Toolkit with NVCC, CMake 3.20 or newer, and a host compiler supported by that Toolkit.

Check the toolchain:

```bash
nvcc --version
cmake --version
nvidia-smi
```

Configure and build (example architecture SM 80; replace it with a value supported by your GPU and CUDA Toolkit):

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release -DCUDA_SM=80
cmake --build build/cuda --parallel
```

Run:

```bash
./build/cuda/testParticlesPaper
```

The executable writes a file named `results-YYYYMMDD-HHMMSS.csv` to its **current working directory**. To keep output separate from source:

```bash
mkdir -p results
cd results
../build/cuda/testParticlesPaper
```

### HIP / ROCm

Requirements: a supported AMD GPU, installed ROCm stack and HIP compiler, and a CMake version/toolchain that supports HIP language compilation. The upstream CMake file assumes ROCm headers at `/opt/rocm` and defaults to `gfx90a` (the MI210 architecture used in the paper). Adjust the target and paths for your own installation.

Check:

```bash
hipcc --version
cmake --version
rocminfo
```

Configure and build (example target `gfx90a`):

```bash
CXX=hipcc cmake -S hip -B build/hip -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx90a
cmake --build build/hip --parallel
```

Run:

```bash
./build/hip/testParticlesPaper
```

### Benchmark workload and CSV

The checked-in upstream `main()` uses a fixed parameter sweep, including box divisions 2, 4, 8, 16, and 32, 200 interaction-loop repetitions, and a particle-count sweep capped by 128 particles per cell. It records per-kernel timing, GFLOP/s, interactions per second, and selected V2/V3 launch-configuration information.

The default driver passes `false` for the optional correctness-check argument to `executeSimulation`. Therefore, do not assume a default benchmark run performs the optional cross-implementation checks. Inspect the comparison routine and change the driver deliberately when enabling it.

For an experiment, record the GPU model, driver, CUDA/ROCm and compiler versions, architecture flags, build type, exact source commit, benchmark parameters, and system load. Preserve the raw CSV output. The paper's reported results are tied to its tested NVIDIA T600, NVIDIA A100, and AMD Instinct MI210 systems; they are not performance guarantees for other devices.

## Interpreting the paper

The paper's conclusion is conditional:

- X-pencil reports speedups up to approximately 2.5x on the NVIDIA T600 and 1.7x on the A100 in selected tested cases, with a peak around 1.65x on AMD MI210.
- All-in-SM was generally not competitive in the reported experiments; its shared-memory use constrained occupancy.
- Par-Cell-SM became more competitive at large particle counts per cell; the paper notes the classical cell-staging approach is useful mainly above roughly 100 particles per cell in the studied configuration.
- The benefit of reducing global-memory traffic decreases as arithmetic intensity increases.

Refer to the paper's figures and experimental setup before interpreting these numbers. The results depend on hardware, workload distribution, kernel configuration, and compiler.

## Documentation

- [Paper and code walkthrough](docs/PAPER_AND_CODE_MAP.md)
- [Build, run, and benchmark guide](docs/RUNNING_AND_BENCHMARKING.md)
- [Source provenance and third-party notices](THIRD_PARTY_NOTICES.md)

## Citation

Please cite the paper when using, updating, or benchmarking this code:

```bibtex
@article{algis2024efficientgpu,
  title   = {Efficient GPU Implementation of Particle Interactions with Cutoff Radius and Few Particles per Cell},
  author  = {Algis, David and Bramas, Berenger and Darles, Emmanuelle and Aveneau, Lilian},
  journal = {arXiv preprint arXiv:2406.16091},
  year    = {2024},
  url     = {https://arxiv.org/abs/2406.16091}
}
```

## Provenance and licensing

The CUDA/HIP implementation files are imported from the upstream author's public mirror at revision `479a812b8cba168f911474f1b04e4c02665aa858`. The paper points to the Inria GitLab project as the original open-source location. The imported file headers are preserved. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the important licensing limitation: the inspected GitHub mirror revision has no top-level license file, and the destination repository's existing LICENSE should not be assumed to cover the imported files.

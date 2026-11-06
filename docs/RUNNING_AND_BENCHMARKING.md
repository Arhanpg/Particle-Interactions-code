# Build, run, and benchmark

Commands below assume the shell's current directory is the repository root. The source uses separate CUDA and HIP CMake projects.

## CUDA

### Prerequisites

- NVIDIA GPU and a working NVIDIA driver.
- CUDA Toolkit with NVCC.
- CMake 3.20 or newer (the upstream CUDA CMakeLists requests 3.20).
- A host C++ compiler supported by the installed CUDA Toolkit.

Check the environment:

```bash
nvcc --version
cmake --version
nvidia-smi
```

### Configure and compile

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release -DCUDA_SM=80
cmake --build build/cuda --parallel
```

The `CUDA_SM` cache value is mandatory in the upstream CMake project. The value `80` is an example for compute capability 8.0; select a value supported by both the installed toolkit and target GPU. On Windows, use an x64 Visual Studio developer shell:

```powershell
cmake -S cuda -B build\cuda -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCUDA_SM=86
cmake --build build\cuda --config Release
```

The CUDA CMake project uses CMake's native CUDA support and requires the CUDA Toolkit, MSVC C++ build tools, and CMake 3.20 or newer.

### Run

```bash
./build/cuda/testParticlesPaper
```

The program writes a timestamped `results-YYYYMMDD-HHMMSS.csv` file to the process's current working directory, not necessarily the source folder. To store output under a dedicated directory:

```bash
mkdir -p results
cd results
../build/cuda/testParticlesPaper
```

On Windows, run from the repository root and collect the generated CSV:

```powershell
New-Item -ItemType Directory -Force results
.\build\cuda\testParticlesPaper.exe *> results\benchmark-console.md
Move-Item .\results-*.csv .\results\
```

## HIP / ROCm

### Prerequisites

- AMD GPU supported by the installed ROCm release.
- ROCm with HIP compiler (`hipcc`) and runtime headers.
- CMake with HIP language support.

Check:

```bash
hipcc --version
cmake --version
rocminfo
```

### Configure and compile

The upstream HIP CMakeLists assumes ROCm headers under `/opt/rocm`, sets `GPU_TARGET` to `gfx90a` by default, and enables HIP language. The paper's AMD Instinct MI210 uses `gfx90a`. For that target:

```bash
CXX=hipcc cmake -S hip -B build/hip -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx90a
cmake --build build/hip --parallel
```

Use your GPU's architecture identifier rather than assuming `gfx90a`. If ROCm is installed outside `/opt/rocm`, the upstream CMakeLists include path and prefix path may need adjustment.

### Run

```bash
./build/hip/testParticlesPaper
```

## What the benchmark runs

The upstream `main()` contains a fixed benchmark sweep. At the imported revision it sets the interaction repetition count to 200, sweeps box divisions 2, 4, 8, 16, and 32, and uses a particle-count sweep capped by a maximum of 128 particles per cell. The driver evaluates the baseline kernels, V1, V2 (with several shared-memory division factors), and V3 when the corresponding manager reports a valid configuration.

It writes one timestamped CSV containing workload metadata, elapsed time, GFLOP/s, interactions per second, and selected V2/V3 launch-configuration statistics.

**Correctness-check caveat:** the checked-in `main()` calls `executeSimulation(..., false)`. The optional output comparison path exists in the source but is not enabled by the default parameter sweep. Do not report the default run as having executed those checks.

## Reproducible experiment checklist

Record the following with each result:

- GPU model, memory, driver, and operating system.
- CUDA Toolkit or ROCm version, compiler version, and CMake version.
- `CUDA_SM` or `GPU_TARGET` and build type.
- Exact Git commit and any local source modifications.
- Workload parameters, loop count, and whether optional checks were enabled.
- Whether other processes were using the GPU.

Keep raw CSV files. Repeat runs and compare like-for-like configurations. The reported results in the paper were obtained on T600, A100, and MI210 hardware; timing on another GPU will not necessarily follow the same ranking.

## Troubleshooting

- **CUDA compiler not found:** check NVCC installation and `PATH`; confirm the host compiler is supported by the Toolkit.
- **Architecture rejected:** select a `CUDA_SM` supported by the installed Toolkit.
- **HIP compiler/header not found:** verify ROCm paths, `hipcc`, and the CMake HIP language support. Update the hard-coded `/opt/rocm` paths if necessary.
- **V1/V2/V3 disabled:** the manager may reject a configuration that exceeds shared-memory/thread limits or provides insufficient blocks for the device. This can be expected.
- **Long execution time:** the default driver runs a substantial sweep with 200 repetitions. For a smoke test, temporarily reduce sweep bounds and loop count in `main()`, rebuild, then restore the original parameters before collecting comparable results.
- **CSV not in the expected directory:** output is relative to the working directory from which the executable was launched.

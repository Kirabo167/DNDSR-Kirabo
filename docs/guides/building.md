# Building DNDSR {#building}

## Quick Start (Step-by-Step)

If you just want to build and run as fast as possible, follow these steps.
Detailed explanations for each step follow in the sections below.

**1. System dependencies**
```bash
sudo apt install build-essential cmake ninja-build openmpi-bin libopenmpi-dev doxygen
```

**2. Clone the repository**
```bash
git clone --recursive https://github.com/Kirabo167/DNDSR.git
cd DNDSR
```

**3. Create the Python environment and get external dependencies**
```bash
python3.12 -m venv venv
source venv/bin/activate

# Cantera build deps (must be installed before cfd_externals_build.py)
pip install -r external/cfd_externals/requirements.txt

# Header-only libs first
bash scripts/install_headeronly_deps.sh

# Then binary libs (needs header-onlys already extracted)
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
cd ../..

# Install remaining Python dependencies (h5py against the project's HDF5, etc.)
bash scripts/install_python_deps.sh
```

**4. Build solvers**
```bash
cmake --preset release-test
cmake --build build --target euler --parallel 4
```

The default build contains `euler`. Build `all_euler`, `all_acm`,
`all_ncfv_euler`, or `all_solvers` for the corresponding executable groups.
The `solvers`, `acm` and `ncfv_euler` build presets select these groups.
Choose `--parallel` for the available memory; independent targets do not
limit the number of concurrent compiler processes.

**5. Run a case**
```bash
# Run from build/ without changing the caller's repository-root directory
(cd build && ./app/euler.exe ../cases/euler/2D/euler_config_IV.json)
# or parallel:
(cd build && mpirun -np 4 ./app/euler.exe ../cases/euler/2D/euler_config_IV.json)
```

Maintained cases resolve their mesh and output paths from the `build/`
working directory. The IV example requires the pinned external mesh fixtures
described below.

**6. Install Python package (optional)**
```bash
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 \
  pip install -e . --no-build-isolation
```

**7. Run tests (optional)**

First fetch the pinned `cfd_meshes` fixtures described under
[Test meshes](v0.3.1_new_features_zh.md#72-测试网格).

```bash
# Solver executables are required by the configuration CTests
cmake --build --preset solvers --parallel 4
# C++ tests
cmake --build --preset tests --parallel 4
ctest --preset unit

# Python tests (never use missing or stale native modules)
cmake --build --preset python -j32
cmake --install build --component py
PYTHONPATH="$PWD/python" venv/bin/python -m pytest test/ -v
```

---

## Prerequisites

| Requirement     | Version       | Notes                                  |
|-----------------|---------------|----------------------------------------|
| C++ compiler    | GCC 9+ / Clang 8+ | Must support C++17                |
| MPI             | MPI-3         | OpenMPI or MPICH                       |
| CMake           | >= 3.21       |                                        |
| Python          | >= 3.10       | 3.12 is CI-tested; system Python recommended |
| Ninja           | any           | Optional but recommended for speed     |
| Doxygen         | any           | Required for Cantera CLib codegen      |

C++ libraries from the pinned header bundle: Eigen, Boost, CGAL,
nlohmann_json, fmt, and pybind11. Binary libraries built by the
`external/cfd_externals` submodule include HDF5, CGNS, Metis, ParMetis,
ZLIB, and Cantera. Optional system dependencies include the CUDA toolkit and
SuperLU_dist; Cantera use in DNDSR itself is controlled by
`DNDS_USE_CANTERA`.

## Building External Dependencies

DNDSR requires two sets of external dependencies: header-only libraries
shipped as a tarball, and binary libraries built from the `cfd_externals`
submodule.

### Header-only libraries (Eigen, Boost, CGAL, fmt, pybind11, ...)

Download the pinned release, verify its SHA-256 digest, and extract it into the
`external/` directory:

```bash
bash scripts/install_headeronly_deps.sh
```

After extraction, directories such as `external/eigen/`,
`external/boost/`, `external/CGAL/`, etc. should exist.

### Binary libraries (HDF5, CGNS, Metis, ParMetis, ZLIB, Cantera)

```bash
git submodule update --init --recursive --depth=1
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
cd ../..
```

This installs all binary libraries into `external/cfd_externals/install/`.

## CMake Module Architecture

The build system is split into focused modules under `cmake/`.  The
main `CMakeLists.txt` (~100 lines) orchestrates them in dependency
order:

| #  | Module                    | Purpose                                                  |
|----|---------------------------|----------------------------------------------------------|
| 1  | `DndsStdlibSetup.cmake`  | Detect the host libstdc++ or libc++ (system runtime; not bundled) |
| 2  | `DndsOptions.cmake`      | All user-facing cache options, commit recording, ccache  |
| 3  | `DndsCudaSetup.cmake`    | CUDA language enable, toolkit discovery, CCCL include path |
| 4  | `DndsCompilerFlags.cmake`| LTO, MPI discovery, platform flags, OpenMP               |
| 5  | `DndsExternalDeps.cmake` | find_library/find_path, pybind11/fmt/superlu subdirs, `dnds_external_deps` target |
| 6  | `cmakeCommonUtils.cmake`  | Helper functions: `add_fast_flags`, `dnds_add_lib`, `dnds_add_py_module` |
| 7  | `DndsTests.cmake`        | CTest registration for doctest C++ tests and pytest      |
| 8  | `DndsApps.cmake`         | Application executables and `ADD_EXE_APP` function       |
| 9  | `DndsDocs.cmake`         | Doxygen documentation target                             |
| 10 | `DndsTooling.cmake`      | compile_commands.json post-processing, automatic stub generation |

Between modules 6 and 7, the library subdirectories are added:
`src/DNDS`, `src/Geom`, `src/CFV`, `src/Euler`, `src/EulerP`, `src/ACM`,
`src/ACMVariable`, and `src/NCFV`.

## System Build Dependencies

Ensure MPI and basic build tools are available before proceeding:

```bash
# Debian/Ubuntu
sudo apt install build-essential cmake ninja-build openmpi-bin libopenmpi-dev doxygen

# RHEL/Fedora
sudo dnf install gcc-c++ cmake ninja-build openmpi-devel doxygen
```

## Python Virtual Environment

The Python package and tests need a virtual environment. Create it first,
then install Cantera build dependencies before building cfd_externals,
and the full Python environment after:

```bash
python3.12 -m venv venv
source venv/bin/activate

# First: install packages needed to build cfd_externals (Cantera)
pip install -r external/cfd_externals/requirements.txt

# Then build cfd_externals (HDF5, CGNS, Metis, ParMetis, Cantera)
# (see "Binary libraries" section above)

# Finally: install full Python environment (h5py, mpi4py, numpy, etc.)
bash scripts/install_python_deps.sh
```

> **Why not conda?** Conda Python binaries embed an `RPATH` pointing to
> conda's bundled libstdc++, which may be too old for the compiler used
> to build DNDSR.  Using the system Python avoids this entirely.

## Building C++ (Solvers and Libraries)

### Using CMake Presets

```bash
cmake --preset release-test        # Baseline CPU, Cantera/CUDA off
cmake --build --preset solvers --parallel 4 # Configuration-test entry points
cmake --build --preset tests --parallel 4   # Every C++ unit-test category
ctest --preset unit                # Run C++ tests (not pytest entries)
```

Available presets (defined in `CMakePresets.json`):

| Configure preset | Build type | Tests | Cantera | CUDA | Build directory |
|------------------|------------|-------|---------|------|-----------------|
| `default`        | Release    | OFF   | OFF     | OFF  | `build-default/` |
| `debug`          | Debug      | ON    | OFF     | OFF  | `build-debug/` |
| `release-test`   | Release    | ON    | OFF     | OFF  | `build/` |
| `reactive-test`  | Release    | ON    | ON      | OFF  | `build-reactive/` |
| `cuda`           | Release    | ON    | OFF     | ON   | `build-cuda/` |
| `ci`             | Release    | ON    | ON      | OFF  | `build-ci/` |

The feature switches are explicit in every preset. This prevents a cached
Cantera or CUDA setting from leaking into an unrelated build. Shared presets
also disable ccache for reproducibility; enable it in a local
`CMakeUserPresets.json` when desired. Useful build and test preset pairs are:

| Purpose | Build command | Test command |
|---------|---------------|--------------|
| All CPU solver entry points | `cmake --build --preset solvers` | `ctest --test-dir build -L solver_config` |
| ACM / NCFV Euler only | `cmake --build --preset acm` / `--preset ncfv_euler` | build the respective module tests |
| All baseline C++ tests | build `solvers`, then `cmake --build --preset tests` | `ctest --preset unit` |
| DNDS core only | `cmake --build --preset dnds-tests` | `ctest --preset dnds` |
| Full Cantera-enabled C++ matrix | build `all_solvers` in `build-reactive`, then `cmake --build --preset reactive` | `ctest --preset reactive` |
| Focused chemistry tests | same reactive build | `ctest --preset reactive-focused` |
| All pybind11 modules | `cmake --build --preset python` | `ctest --preset python` after install |
| CUDA modules/tests | `cmake --build --preset cuda` | `ctest --preset cuda` |

`ctest --preset all` includes both C++ and Python CTest entries. Build
`all_solvers` and `all_unit_tests`, then build and install the four pybind11
modules before using it. CUDA/debug/CI full-suite runs also need
`all_solvers` in their own build directories for configuration CTests. `CMakeUserPresets.json` may
be used for machine-specific compilers and paths and is intentionally ignored
by Git.

### Manual CMake Configuration

```bash
CC=mpicc CXX=mpicxx cmake -S . -B build -G Ninja \
  -DDNDS_BUILD_TESTS=ON -DDNDS_USE_CANTERA=OFF
cmake --build build --target euler --parallel 4           # Build a solver
cmake --build build -t dnds_unit_tests -j32 # Build C++ tests
ctest --test-dir build -R dnds_ --output-on-failure
```

Let CMake detect the system default compiler. Use `CC=mpicc CXX=mpicxx`
when unsure which MPI implementation CMake will find.

### Solver Targets

Each Euler model variant generates a separate executable:

| Target        | Model      | Mesh geometry | Velocity components |
|---------------|------------|---------------|---------------------|
| `euler`       | Navier-Stokes | 2D | 3 |
| `euler2D`     | Navier-Stokes | 2D | 2 |
| `euler3D`     | Navier-Stokes | 3D | 3 |
| `eulerSA`     | Spalart-Allmaras | 2D | 3 |
| `eulerSA3D`   | Spalart-Allmaras | 3D | 3 |
| `euler2EQ`    | k-omega 2-equation | 2D | 3 |
| `euler2EQ3D`  | k-omega 2-equation | 3D | 3 |
| `eulerEX`     | Multi-species reactive Navier-Stokes | 2D | 3 |
| `eulerEX3D`   | Multi-species reactive Navier-Stokes | 3D | 3 |

`eulerEX`, `eulerEX3D`, and `eulerState` still compile in the Cantera-free
compatibility build, but Cantera thermochemistry/kinetics and the
`canteraConstVolTrajectory` tool require `DNDS_USE_CANTERA=ON`. Reactive CFD is
currently CPU-only.

This fork also builds the following independent solver families:

| Target(s) | Model |
|-----------|-------|
| `ACM`, `acm2D`, `acm3D` | Constant-density artificial compressibility |
| `acmVariable2D`, `acmVariable3D` | Variable-density artificial compressibility |
| `ncfv_euler2D`, `ncfv_euler3D` | Two-/three-dimensional node-centred Euler flow |

The `ACM` compatibility target uses the 3D constant-density model. Source
module/namespace names remain `NCFV` / `DNDS::NCFV`; executable and case-family
names are `ncfv_euler`. The old unified `euler` dispatcher is removed.

```bash
cmake --build build --target acm2D ncfv_euler3D --parallel 4
cmake --build build --target all_solvers --parallel 4
```

### Configuration paths and checks

Launch maintained cases from the build directory. Case files are grouped by
family and dimension; schemas are stored beside the respective model's cases.
Mesh/output paths are relative to the process working directory, while
`$schema` is relative to the case file.

```bash
(cd build && ./app/acm2D.exe ../cases/acm/2D/acm2D.json --check-config)
(cd build && ./app/ncfv_euler3D.exe \
    ../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json)
python3 scripts/check_solver_cases.py --build-dir build \
    --reactive-build-dir build-reactive
python3 cases/validate_configs.py --quiet
```

`--check-config` validates typed parameters and executable selection without
mesh reads or solver result output. The audit needs all required executables;
reaction cases require Cantera. These tools invoke C++ programs or inspect
JSON and do not load DNDSR Python bindings.

Euler preserves an existing case-adjacent `<target>_default_config.json`;
if absent, it uses compiled defaults. Case JSON and `-k/-v` overrides are then
merged. ACM uses complete single-file cases and NCFV merges its typed defaults.
See @ref solver_config and @ref solver_split_zh for migration details and
the limits of the upstream comparison.

### CMake Cache Options

Key options (set via `-D<OPTION>=<VALUE>` or in a preset):

| Option                       | Default | Description                            |
|------------------------------|---------|----------------------------------------|
| `DNDS_BUILD_TESTS`          | OFF     | Build C++ unit tests (doctest)         |
| `DNDS_USE_CUDA`             | OFF     | Enable CUDA GPU support                |
| `DNDS_USE_CANTERA`          | ON      | Enable Cantera chemistry and reactive-flow tests |
| `DNDS_USE_OMP`              | ON      | Enable OpenMP                          |
| `DNDS_FAST_BUILD_FAST`      | ON      | Use -O3 -g0 on core library modules   |
| `DNDS_LTO`                  | OFF     | Link-time optimization                 |
| `DNDS_LTO_THIN`             | OFF     | Use -flto=thin (Clang)                 |
| `DNDS_PYBIND11_NO_LTO`     | ON      | Disable LTO for pybind11 modules only  |
| `DNDS_NATIVE_ARCH`          | OFF     | Use -march=native                      |
| `DNDS_UNSAFE_MATH_OPT`     | OFF     | Use -funsafe-math-optimizations        |
| `DNDS_USE_CCACHE`           | auto    | Use ccache (auto-detected, off for pip)|
| `DNDS_USE_RDYNAMIC`        | ON      | Use -rdynamic on POSIX                 |
| `DNDS_GENERATE_COMPILE_COMMANDS` | OFF | Generate compile_commands.json for clangd |
| `DNDS_USE_CLANG_TIDY`      | OFF     | Run clang-tidy during build            |
| `DNDS_RECORD_COMMIT`       | ON      | Record git commit hash at configure    |
| `DNDS_USE_PRECOMPILED_HEADER` | OFF  | Use precompiled headers                |
| `DNDS_EIGEN_USE_BLAS`      | OFF     | Use external BLAS in Eigen             |
| `DNDS_EIGEN_USE_LAPACK`    | OFF     | Use external LAPACK in Eigen           |

These are raw CMake defaults. The checked-in presets deliberately select a
clear feature matrix: `release-test` disables Cantera, `reactive-test` enables
it, and `cuda` enables CUDA without implying reactive GPU support.

## Building the Python Package

### Three Ways to Build the Python Package

#### 1. In-place build (no pip install)

Build the pybind11 `.so` modules via the main CMake build, then
install them (and auto-generate type stubs) into `python/DNDSR/`:

```bash
source venv/bin/activate
cmake --preset release-test
cmake --build --preset python -j32
cmake --install build --component py
```

Use the package by setting `PYTHONPATH`:

```bash
PYTHONPATH=python pytest test/
# or:
export PYTHONPATH=$PWD/python
python -c "from DNDSR import DNDS, Geom, CFV, EulerP"
```

After making C++ changes, rebuild and reinstall:

```bash
cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
cmake --install build --component py   # reinstalls .so AND regenerates stubs
```

#### 2. Editable install (development with pip)

Uses scikit-build-core. Builds into a wheel-tagged subdirectory under
`build_py/`, so different Python ABIs cannot share one CMake cache:

```bash
source venv/bin/activate
CMAKE_BUILD_PARALLEL_LEVEL=32 pip install -e . --no-build-isolation
```

This configures and builds all four pybind11 targets, installs `.so`
files into `python/DNDSR/<Module>/_ext/`, generates `.pyi` stubs, and
registers the package as editable in the venv.

After C++ changes, rerun the editable install command, or use the configured
`release-test` in-place build:

```bash
cmake --build --preset python -j32
cmake --install build --component py
```

#### 3. Full wheel install

```bash
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 \
  pip install . --no-build-isolation --verbose
```

Builds a local wheel with `.so` files, bundled shared libraries, and `.pyi`
stubs included. Build it only from a fully prepared checkout and use it with
the same MPI implementation/ABI as the build host. The separately distributed
header bundle and compiled external libraries mean the generated sdist is not
currently a self-contained installation artifact. Do not publish or
redistribute a wheel until the ParMETIS redistribution permission and all
third-party license/notice requirements have been reviewed and satisfied.
The repository-level `THIRD_PARTY_DEPENDENCIES.md` records the current binary
inventory and release checklist.

### Controlling Build Parallelism

`pyproject.toml` does not inject a fixed Ninja `-j` option. Set CMake's standard
parallelism variable so one unambiguous limit reaches the generated build:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e . --no-build-isolation
```

### Pybind11 Module Targets

| Target           | Module   | C++ Source               | Output .so location             |
|------------------|----------|--------------------------|---------------------------------|
| `dnds_pybind11`  | DNDS     | `src/DNDS/*_bind*.cpp`   | `python/DNDSR/DNDS/_ext/`      |
| `geom_pybind11`  | Geom     | `src/Geom/*_bind*.cpp`   | `python/DNDSR/Geom/_ext/`      |
| `cfv_pybind11`   | CFV      | `src/CFV/*_bind*.cpp`    | `python/DNDSR/CFV/_ext/`       |
| `eulerP_pybind11`| EulerP   | `src/EulerP/*_bind*.cpp` | `python/DNDSR/EulerP/_ext/`    |

Each pybind11 module links against a corresponding `*_shared` library
(`dnds_shared`, `geom_shared`, `cfv_shared`, `eulerP_shared`) which
contains the compiled C++ code.

## Type Stub Generation

Type stubs (`.pyi`) provide IDE autocompletion and type checking for
the pybind11 bindings.  Stubs are generated automatically during
install and are not tracked in git.

### How it works

`cmake/DndsTooling.cmake` registers an `install(CODE ...)` step on
the `py` component that runs `scripts/generate-stubs.sh` after all
`.so` files are installed.  This happens in both workflows:

- `cmake --install build --component py` (in-place build)
- `pip install -e . --no-build-isolation` (scikit-build-core editable install)

The script runs `pybind11-stubgen` for each submodule (DNDS, Geom,
CFV, EulerP), writes raw output to `stubs/`, and copies `.pyi` files
into `python/DNDSR/` for PEP 561 compliance.

### Manual stub regeneration

If you only changed Python code (no C++ binding changes), you can
regenerate stubs without rebuilding:

```bash
PYTHONPATH=python ./scripts/generate-stubs.sh
```

### Stubs in wheels

The `.pyi` files under `python/DNDSR/` are staged into wheel builds by the
CMake install step. They are generated while building the wheel and are not
copied from platform-specific artifacts left in the source distribution, so
wheel users get type hints without running stubgen themselves.

## Reactive Flow and Cantera

The `reactive-test` preset is the supported CPU validation configuration for
EulerEX chemistry. It assumes the pinned `cfd_externals` submodule has already
been built with Cantera. Its `reactive` build/test pair compiles and exercises
all C++ modules with Cantera enabled; use `reactive-focused` for only the eight
chemistry/reactive checks:

```bash
CC=mpicc CXX=mpicxx cmake --preset reactive-test
cmake --build build-reactive --target all_solvers --parallel 4
cmake --build --preset reactive --parallel 4
ctest --preset reactive
# Optional focused rerun:
ctest --preset reactive-focused
```

At runtime, point Cantera to additional mechanism/data directories when the
configuration does not use an explicit mechanism path:

```bash
export DNDS_MECH_PATH=/path/to/mechanisms
export CANTERA_DATA=/path/to/cantera/data
```

The standalone Python flame/reference scripts use the Python Cantera package,
which is separate from the C++ library built by `cfd_externals`. It is included
in the contributor `requirements.txt` and is also available through the
`chemistry` package extra.

## CUDA Support

### Enabling CUDA

```bash
cmake -S . -B build-cuda -G Ninja \
  -DDNDS_USE_CUDA=ON -DDNDS_USE_CANTERA=OFF
cmake --build build-cuda -t euler -j32
```

Or use the `cuda` preset:

```bash
cmake --preset cuda
cmake --build --preset cuda -j32
ctest --preset cuda
```

### CUDA 13.1 (CCCL 3.x) Compatibility

CUDA 13.1 moved thrust, cub, and libcudacxx headers into a `cccl/`
subdirectory under the CUDA toolkit include path.  `nvcc` adds this
path automatically, but the host C++ compiler (g++) does not.

`DndsCudaSetup.cmake` detects `${CUDAToolkit_INCLUDE_DIRS}/cccl` and
exposes it as `DNDS_CUDA_CCCL_INCLUDE_DIR`.  `DndsExternalDeps.cmake`
appends it to `DNDS_EXTERNAL_INCLUDES`, so `#include <thrust/...>`
works from both `.cu` and `.cpp` files.

This is backward compatible: on CUDA 12.x the `cccl/` path does not
exist, so nothing is added.

### CUDA-specific targets

GPU-accelerated versions of the test apps are built when
`DNDS_USE_CUDA=ON`:

- `cuda_test`, `array_cuda_Test`, `array_cuda_Bench`, `arrayDOF_test_cuda`
- `eulerP_pybind11` (Python bindings with GPU evaluator)

The CUDA preset does not enable reactive chemistry. EulerEX reactive CFD has
no CUDA implementation in v0.3.1; validate chemistry independently with the
`reactive-test` preset.

## Running Tests

### C++ Unit Tests

C++ tests use [doctest](https://github.com/doctest/doctest) and live under
`test/cpp/`. By default, MPI-aware tests are registered at np=1, 2, 4, and 8;
the NCFV I/O regression is intentionally limited to np=1 and 2. Override the
general matrix at configure time with `DNDS_TEST_NP_LIST`.

```bash
cmake --build --preset tests -j32
ctest --preset unit

# Run a single test executable directly
./build/test/cpp/dnds_test_array
mpirun -np 4 ./build/test/cpp/dnds_test_mpi
```

Aggregate build targets are `dnds_unit_tests`, `geom_unit_tests`,
`cfv_unit_tests`, `euler_unit_tests`, `acm_unit_tests`,
`acm_variable_unit_tests`, `solver_unit_tests`, and `ncfv_unit_tests`.
`all_unit_tests` depends on all of them. Cantera-enabled Euler tests are added
to `euler_unit_tests` by the `reactive-test` configuration.

Several Geom, CFV, and Euler regressions use external CGNS fixtures under
`data/mesh/`. The main repository intentionally ignores these files. Follow
the pinned `cfd_meshes` instructions in
[the v0.3.1 migration guide](v0.3.1_new_features_zh.md#72-测试网格) before
interpreting a CGNS-open failure as a solver regression.

### Python Tests

Python tests use pytest with pytest-timeout and live under `test/`; MPI runs
invoke `mpirun` explicitly. The root
`test/conftest.py` adds `python/` to `sys.path` so tests work with
both `PYTHONPATH=python` and `pip install -e .`.

```bash
cmake --build --preset python -j32
cmake --install build --component py
PYTHONPATH="$PWD/python" venv/bin/python -m pytest test/DNDS/test_basic.py -v

# With MPI
PYTHONPATH="$PWD/python" mpirun -np 4 venv/bin/python -m pytest test/DNDS/test_basic.py

# All tests
PYTHONPATH="$PWD/python" venv/bin/python -m pytest test/ -x --timeout=120
```

CTest also registers pytest suites when `DNDS_BUILD_TESTS=ON`:

```bash
ctest --preset python
```

## Build Mode Summary

| Mode                     | Command                                               | Build Dir   | Stubs         |
|--------------------------|-------------------------------------------------------|-------------|---------------|
| **Pure C++ build**       | `cmake --build build --target euler --parallel 4`                   | `build/`    | N/A           |
| **C++ unit tests**       | `cmake --build --preset tests -j32`                    | `build/`    | N/A           |
| **In-place Python**      | `cmake --install build --component py`                | `build/`    | Auto-generated|
| **Editable install**     | `pip install -e . --no-build-isolation`               | `build_py/<wheel-tag>/` | Auto-generated|
| **In-place C++ rebuild** | `cmake --build --preset python && cmake --install build --component py` | `build/` | Auto-generated|
| **Local wheel install**  | `pip install . --no-build-isolation`                  | `build_py/<wheel-tag>/` | Included in wheel|

## Developer Tooling

### compile_commands.json for clangd

clangd needs a `compile_commands.json` at the project root for C++
code intelligence.  CMake generates one in the build directory; the
`compdb` tool post-processes it to include header-only translation
units.

```bash
cmake -S . -B build -DDNDS_GENERATE_COMPILE_COMMANDS=ON -G Ninja
cmake --build build -j32
cmake --build build -t process-compile-commands
```

This creates `build/compile_commands_processed.json` and symlinks it
to `compile_commands.json` at the project root.

The build system uses a shipped, modified version of compdb at
`scripts/compdb/` (invoked as `PYTHONPATH=scripts python -m compdb`).
If the shipped version is not found, it falls back to a
system-installed `compdb` executable (`pip install compdb`).

### Unified Sphinx and Doxygen Documentation

```bash
pip install -r docs/sphinx/requirements.txt
# Reconfigure after installing Sphinx; documentation targets are discovered
# during CMake configure.
cmake --preset release-test
cmake --build build -t docs
```

The unified site goes to `build/docs/sphinx/`; its embedded raw Doxygen API is
also available at `build/docs/html/`.

### CMake Utility Targets

| Target                     | Description                                      |
|----------------------------|--------------------------------------------------|
| `process-compile-commands` | Post-process compile_commands.json for clangd    |
| `docs`                     | Build the unified Sphinx site with Doxygen API   |
| `dnds_unit_tests`          | Build DNDS core unit-test executables             |
| `all_unit_tests`           | Build every registered C++ unit-test category     |

### Shared Library Bundling

Both the in-place and pip install workflows copy CGNS, HDF5, Metis,
ParMetis, and zlib into the package's `DNDSR/_lib/dndsr_external/`
directory. The pybind11 modules use RPATH to find this controlled subset.

MPI and the C++ standard library (`libstdc++`/`libc++`) are intentionally not
bundled because they must remain consistent with the host runtime; loading a
second allocator/runtime can cause symbol conflicts and double frees. Cantera
is also an external runtime dependency for Cantera-enabled C++ builds. Use the
same compiler/MPI stack throughout and, when necessary, source the generated
`<build>/install/DNDSR/set_library_path.sh` before running executables.

### pyproject.toml Configuration

The Python package is built with
[scikit-build-core](https://scikit-build-core.readthedocs.io/).  Key
settings in `pyproject.toml`:

| Setting                | Value                   | Purpose                       |
|------------------------|-------------------------|-------------------------------|
| `build-dir`           | `build_py/{wheel_tag}`  | Separate cache for each Python ABI |
| `install.components` | `py`                    | Stage only Python package artifacts |
| `minimum-version`     | build-system requirement | Stable scikit-build-core defaults |
| `build.targets`       | 4 pybind11 targets      | Only build Python bindings    |
| `cmake.args`          | `["-G", "Ninja"]`       | Use Ninja generator           |
| `cmake.define.DNDS_USE_CANTERA` | `OFF` | Python bindings do not expose reactive Euler |
| `wheel.packages`      | `python/DNDSR`          | Package root                  |

When scikit-build-core configures CMake it sets `SKBUILD_PROJECT_NAME`,
which the build system uses to skip ccache and in-source symlink
creation (these are only relevant for the developer C++ build).
